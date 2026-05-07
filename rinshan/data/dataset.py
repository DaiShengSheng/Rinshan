"""
MjaiDataset — 从磁盘上的 .jsonl 标注文件加载训练数据

格式：每行一个 JSON 对象，对应一条 Annotation
文件由 scripts/parse_tenhou.py 生成

支持：
  - 多文件流式加载（内存友好）
  - 按数据质量分层采样
  - IQL 用的 (s, a, r, s', done) 配对
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Iterator

import torch
from torch.utils.data import IterableDataset

from rinshan.tile import Tile
from .annotation import Annotation, AuxTargets
from .encoder    import GameEncoder


# ─────────────────────────────────────────────
# JSON → Annotation 反序列化
# ─────────────────────────────────────────────

def _parse_tile_list(raw: list[str]) -> list[Tile]:
    return [Tile.from_mjai(s) for s in raw]


def _json_to_annotation(d: dict) -> Annotation:
    """将一行 JSON dict 还原为 Annotation 对象"""
    melds = []
    for seat_melds in d.get("melds", [[], [], [], []]):
        seat = []
        for m in seat_melds:
            seat.append((m["type"], _parse_tile_list(m["tiles"])))
        melds.append(seat)

    aux = None
    if "aux" in d and d["aux"] is not None:
        a = d["aux"]
        aux = AuxTargets(
            shanten      = a["shanten"],
            tenpai_prob  = float(a["tenpai_prob"]),
            deal_in_risk = a["deal_in_risk"],
            opp_tenpai   = a["opp_tenpai"],
            opp_wait_tiles = a.get("opp_wait_tiles", None),
        )

    opp_hands = None
    if "opponent_hands" in d and d["opponent_hands"] is not None:
        opp_hands = [_parse_tile_list(h) for h in d["opponent_hands"]]

    return Annotation(
        game_id            = d["game_id"],
        player_id          = d["player_id"],
        round_wind         = d["round_wind"],
        round_num          = d["round_num"],
        honba              = d["honba"],
        kyotaku            = d["kyotaku"],
        scores             = d["scores"],
        tiles_left         = d["tiles_left"],
        hand               = _parse_tile_list(d["hand"]),
        dora_indicators    = _parse_tile_list(d["dora_indicators"]),
        discards           = [_parse_tile_list(s) for s in d.get("discards", [[], [], [], []])],
        melds              = melds,
        riichi_declared    = d.get("riichi_declared", [False, False, False, False]),
        riichi_discard_tile= [
            Tile.from_mjai(t) if t is not None else None
            for t in d.get("riichi_discard_tile", [None]*4)
        ],
        riichi_junme       = d.get("riichi_junme", [-1]*4),
        riichi_furiten     = d.get("riichi_furiten", [False]*4),
        progression        = d["progression"],
        action_candidates  = d["action_candidates"],
        action_chosen      = d["action_chosen"],
        round_delta_score  = d.get("round_delta_score", 0),
        final_delta_score  = d.get("final_delta_score", 0),
        final_rank         = d.get("final_rank", 0),
        grp_reward         = float(d.get("grp_reward", 0.0)),
        hand_reward        = float(d.get("hand_reward", d.get("round_delta_score", 0)) / 1000.0),
        aux                = aux,
        opponent_hands     = opp_hands,
        is_done            = bool(d.get("is_done", False)),
    )


# ─────────────────────────────────────────────
# IterableDataset（流式，内存友好）
# ─────────────────────────────────────────────

class MjaiDataset(IterableDataset):
    """
    从 .jsonl 文件流式加载标注数据

    Args:
        file_paths    : .jsonl 文件路径列表
        encoder       : GameEncoder 实例
        shuffle_files : 是否在每个 epoch 开始时打乱文件顺序
        shuffle_buffer: 内存缓冲区大小，在缓冲区满后 shuffle 并 yield
        stage         : 1/2/3，影响需要包含的字段（Stage3 需要 next_state）
    """

    def __init__(
        self,
        file_paths: list[Path | str],
        encoder: Optional[GameEncoder] = None,
        shuffle_files:  bool = True,
        shuffle_buffer: int  = 2000,
        stage: int = 1,
    ):
        super().__init__()
        self.file_paths     = [Path(p) for p in file_paths]
        self.encoder        = encoder or GameEncoder()
        self.shuffle_files  = shuffle_files
        self.shuffle_buffer = shuffle_buffer
        self.stage          = stage

    def _iter_file(self, path: Path) -> Iterator[dict]:
        """逐行读取一个 .jsonl 文件，yield encoded dict"""
        with open(path, "r", encoding="utf-8") as f:
            # Stage 3 不能简单用“上一行”配 next state：
            # 同一文件里不同玩家/不同对局的决策会交错出现。
            # 必须按 (game_id, player_id) 维护各自的轨迹。
            #
            # GRP 2.0：当前离线 GRP 的粒度是“局结束后的游戏进程价值变化”，
            # 不能再把同一局的 delta 广播给局内所有 action。
            # 因此这里只在 GRP 状态真正变化（进入下一局/终局）时，把该局
            # 的 grp_reward 记到最后一个 action 上；局内其余 action reward=0。
            prev_by_key: dict[tuple[str, int], tuple[dict, tuple]] = {}

            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = _json_to_annotation(json.loads(line))
                    # Stage 2 使用 oracle 序列（含对手手牌）
                    if self.stage == 2:
                        encoded = self.encoder.encode_oracle(ann)
                    else:
                        encoded = self.encoder.encode(ann)
                except Exception as e:
                    # 跳过格式错误的行，记录但不中断
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Skip malformed line in {path}: {e}"
                    )
                    continue

                if self.stage != 3:
                    yield encoded
                    continue

                key = (encoded["game_id"], int(encoded["player_id"]))
                grp_state_key = (
                    int(ann.round_wind),
                    int(ann.round_num),
                    int(ann.honba),
                )
                prev_entry = prev_by_key.get(key)
                if prev_entry is not None:
                    prev_encoded, prev_grp_state_key = prev_entry
                    prev_encoded["next_tokens"] = encoded["tokens"]
                    prev_encoded["next_candidate_mask"] = encoded["candidate_mask"]
                    prev_encoded["next_pad_mask"] = encoded.get("pad_mask")
                    prev_encoded["next_belief_tokens"] = encoded.get("belief_tokens")
                    prev_encoded["next_belief_pad_mask"] = encoded.get("belief_pad_mask")
                    prev_encoded["done"] = torch.tensor(False, dtype=torch.bool)
                    prev_encoded["next_reward_game"] = encoded.get("reward_game")
                    prev_encoded["next_reward_hand"] = encoded.get("reward_hand")

                    # GRP 2.0：game reward 只在跨局/终局时结算；
                    # hand reward 保留为逐局局内 shaping（当前由 round_delta_score 代理）。
                    if grp_state_key == prev_grp_state_key:
                        prev_encoded["reward"] = torch.zeros_like(prev_encoded["reward"])
                        prev_encoded["reward_game"] = torch.zeros_like(prev_encoded["reward_game"])
                    else:
                        prev_encoded["reward"] = prev_encoded["reward_game"] + prev_encoded["reward_hand"]
                    yield prev_encoded

                prev_by_key[key] = (encoded, grp_state_key)

            # 处理轨迹末尾：仅对显式 done 的终止样本补一个 dummy next state。
            if self.stage == 3:
                for prev_encoded, _prev_grp_state_key in prev_by_key.values():
                    is_done = prev_encoded.get("is_done")
                    is_done = bool(is_done.item()) if isinstance(is_done, torch.Tensor) else bool(is_done)
                    if not is_done:
                        continue
                    prev_encoded["next_tokens"] = torch.zeros_like(prev_encoded["tokens"])
                    prev_encoded["next_candidate_mask"] = torch.zeros_like(prev_encoded["candidate_mask"])
                    prev_encoded["next_pad_mask"] = torch.ones_like(prev_encoded["pad_mask"])
                    prev_encoded["next_belief_tokens"] = torch.zeros_like(prev_encoded["belief_tokens"])
                    prev_encoded["next_belief_pad_mask"] = torch.ones_like(prev_encoded["belief_pad_mask"])
                    prev_encoded["next_reward_game"] = torch.zeros_like(prev_encoded["reward_game"])
                    prev_encoded["next_reward_hand"] = torch.zeros_like(prev_encoded["reward_hand"])
                    prev_encoded["done"] = torch.tensor(True, dtype=torch.bool)
                    prev_encoded["reward"] = prev_encoded["reward_game"] + prev_encoded["reward_hand"]
                    yield prev_encoded

    def __iter__(self) -> Iterator[dict]:
        files = list(self.file_paths)
        if self.shuffle_files:
            random.shuffle(files)

        # ── 多 worker 时按 worker id 切分文件，避免重复 ──────────
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            n  = worker_info.num_workers
            wid = worker_info.id
            files = files[wid::n]   # 每隔 n 取一个，均匀分配

        # ── 流式 shuffle buffer（逐条 yield，不整包 flush）────────
        buffer: list[dict] = []
        for path in files:
            for sample in self._iter_file(path):
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(sample)
                else:
                    # 随机替换并 yield 被替换的旧元素
                    idx = random.randrange(self.shuffle_buffer)
                    yield buffer[idx]
                    buffer[idx] = sample

        # 清空剩余缓冲
        random.shuffle(buffer)
        yield from buffer


# ─────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    将一个 batch 的 dict list 合并为一个 batch dict
    处理 None 值（actual_hands 等可选字段）
    """
    result: dict = {}
    keys = batch[0].keys()

    for k in keys:
        vals = [item[k] for item in batch]

        # 字符串字段（game_id 等）直接保留为列表
        if isinstance(vals[0], str):
            result[k] = vals
            continue
        # int 字段
        if isinstance(vals[0], int):
            result[k] = torch.tensor(vals, dtype=torch.long)
            continue
        # None 字段：所有都是 None 就跳过
        if all(v is None for v in vals):
            result[k] = None
            continue
        # 混合 None：填零 tensor
        if any(v is None for v in vals):
            # 找第一个非 None 的 shape
            ref = next(v for v in vals if v is not None)
            vals = [v if v is not None else torch.zeros_like(ref) for v in vals]

        # dict 字段（aux_targets）
        if isinstance(vals[0], dict):
            merged: dict = {}
            for sub_k in vals[0].keys():
                sub_vals = [v[sub_k] for v in vals]
                if isinstance(sub_vals[0], (int, float)):
                    merged[sub_k] = torch.tensor(sub_vals)
                elif isinstance(sub_vals[0], list):
                    merged[sub_k] = torch.tensor(sub_vals, dtype=torch.float32)
                elif isinstance(sub_vals[0], torch.Tensor):
                    merged[sub_k] = torch.stack(sub_vals)
                else:
                    # numpy.ndarray 或其他，转 tensor 后 stack
                    merged[sub_k] = torch.stack([
                        torch.as_tensor(v) for v in sub_vals
                    ])
            result[k] = merged
            continue

        # Tensor 字段
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
            continue

        result[k] = vals

    return result
