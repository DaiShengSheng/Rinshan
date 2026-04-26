"""
OnlineBuffer — 自对弈数据 → 训练样本的实时转换缓冲区

把 Arena 产出的 GameRecord（mjai 事件流 + 终局分数）解析为
(Annotation, next_Annotation, reward, done) 四元组，
供 Stage 4 在线 IQL 训练直接消费。

奖励设计（与 GRP 思路一致，但不需要离线的 GRP 网络）：
  - 局内奖励 r_kyoku = delta_score / 1000  （每局结算时）
  - 终局奖励 r_final = final_rank_reward    （整场结束时，按顺位给稠密奖励）
  - 中间步奖励 r_step = 0                   （打牌/鸣牌本身不给奖励）
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from rinshan.data.annotation import Annotation
from rinshan.data.encoder import GameEncoder
from rinshan.engine.simulator import MjaiSimulator
from rinshan.self_play.arena import GameRecord


# 顺位奖励表（0=一位，3=四位）
_RANK_REWARD = [3.0, 1.0, -1.0, -3.0]


@dataclass
class Transition:
    """一条 (s, a, r, s', done) 训练样本"""
    encoded:      dict           # 当前状态编码
    next_encoded: Optional[dict] # 下一决策点编码（None = terminal）
    reward:       float
    done:         bool


class OnlineBuffer:
    """
    环形缓冲区：存储 Transition，支持随机采样 mini-batch。

    Args:
        capacity:       最多保留的 Transition 数量
        encoder:        GameEncoder 实例（复用）
        rank_scale:     顺位奖励的缩放系数
        score_scale:    局内分差奖励的缩放系数（分 / score_scale）
    """

    def __init__(
        self,
        capacity: int = 50_000,
        encoder: Optional[GameEncoder] = None,
        rank_scale: float = 1.0,
        score_scale: float = 1000.0,
    ):
        self.capacity    = capacity
        self.encoder     = encoder or GameEncoder()
        self.rank_scale  = rank_scale
        self.score_scale = score_scale
        self._buf: deque[Transition] = deque(maxlen=capacity)

    # ──────────────────────────────────────────────────────────
    # 写入接口
    # ──────────────────────────────────────────────────────────

    def ingest_record(self, record: GameRecord) -> int:
        """
        从一局 GameRecord 中解析所有决策点并写入缓冲区。
        返回写入的 Transition 数量。
        """
        sim = MjaiSimulator()
        n_written = 0

        # 计算顺位奖励：每位玩家一个标量
        rank_rewards = [
            _RANK_REWARD[rank] * self.rank_scale
            for rank in record.ranks
        ]

        # B10 fix: prev_by_seat 必须在每局（kyoku）结束时清零
        # 不能跨局配对，因为不同局的状态在语义上不连续
        prev_by_seat: dict[int, dict] = {}

        # 按局依次解析
        for k_idx, kyoku_events in enumerate(record.kyoku_logs):
            anns = sim.parse_game(kyoku_events, game_id=record.game_id)
            if not anns:
                continue

            # B10 fix: 每局开始时清空前驱状态，不跨局配对
            kyoku_prev: dict[int, dict] = {}

            # 找出本局中每个玩家的最后一次动作 index
            last_idx_by_seat = {}
            for i, ann in enumerate(anns):
                last_idx_by_seat[ann.player_id] = i

            for i, ann in enumerate(anns):
                seat = ann.player_id
                encoded = self.encoder.encode(ann)

                # 局内奖励：仅在本局该玩家的最后一次动作结算
                r = 0.0
                if i == last_idx_by_seat[seat]:
                    r += ann.round_delta_score / self.score_scale

                encoded["reward"] = torch.tensor(r, dtype=torch.float32)

                # 如果该玩家在本局有前驱状态，凑成一个 Transition
                if seat in kyoku_prev:
                    prev = kyoku_prev[seat]
                    self._buf.append(Transition(
                        encoded=prev,
                        next_encoded=encoded,
                        reward=prev["reward"].item(),
                        done=False,
                    ))
                    n_written += 1

                kyoku_prev[seat] = encoded
                prev_by_seat[seat] = encoded  # 更新全局最新状态（用于最终 done 处理）

        # 整场游戏结束，处理剩余的最后状态，打上 done=True 并给予顺位奖励
        for seat, prev in prev_by_seat.items():
            final_r = prev["reward"].item() + rank_rewards[seat]
            self._buf.append(Transition(
                encoded=prev,
                next_encoded=_zero_encoded_like(prev),
                reward=final_r,
                done=True,
            ))
            n_written += 1

        return n_written

    def ingest_records(self, records: list[GameRecord]) -> int:
        total = 0
        for r in records:
            total += self.ingest_record(r)
        return total

    # ──────────────────────────────────────────────────────────
    # 读取接口
    # ──────────────────────────────────────────────────────────

    def sample_batch(self, batch_size: int) -> Optional[dict]:
        """
        随机采样一个 mini-batch，返回已 stack 的 tensor dict。
        若缓冲区样本数不足 batch_size，返回 None。
        """
        if len(self._buf) < batch_size:
            return None

        samples = random.sample(list(self._buf), batch_size)
        return _collate_transitions(samples)

    def iter_batches(self, batch_size: int, n_batches: int) -> Iterator[dict]:
        """连续采样 n_batches 个 mini-batch（用于训练循环）"""
        for _ in range(n_batches):
            batch = self.sample_batch(batch_size)
            if batch is not None:
                yield batch

    @property
    def size(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"OnlineBuffer(size={self.size}/{self.capacity})"


# ──────────────────────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────────────────────

def _zero_encoded_like(ref: dict) -> dict:
    """生成与 ref 结构相同但全为零的 encoded dict（用于 terminal next state）"""
    result = {}
    for k, v in ref.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bool:
                result[k] = torch.ones_like(v)   # pad mask 全 True
            else:
                result[k] = torch.zeros_like(v)
        else:
            result[k] = v
    return result


def _collate_transitions(samples: list[Transition]) -> dict:
    """把一组 Transition 合并为 batched tensor dict"""
    from rinshan.data.dataset import collate_fn

    current_batch = [s.encoded for s in samples]
    next_batch    = [s.next_encoded for s in samples]
    rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
    dones   = torch.tensor([s.done   for s in samples], dtype=torch.bool)

    # 用现有的 collate_fn 处理 current 和 next
    current_collated = collate_fn(current_batch)
    next_collated    = collate_fn(next_batch)

    # 合并，next 字段加 next_ 前缀
    result = dict(current_collated)
    result["reward"] = rewards
    result["done"]   = dones

    for k, v in next_collated.items():
        if k in ("game_id", "player_id"):
            continue
        result[f"next_{k}"] = v

    # Stage 3/4 用的 next_tokens 等字段
    for field in ("tokens", "candidate_mask", "pad_mask",
                  "belief_tokens", "belief_pad_mask"):
        src_key = f"next_{field}"
        if src_key not in result and field in next_collated:
            result[src_key] = next_collated[field]

    return result
