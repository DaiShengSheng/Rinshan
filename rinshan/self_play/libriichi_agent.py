"""
LibriichiBoostedAgent — 用 libriichi.PlayerState 加速候选动作生成

背景
----
Rinshan 的自对弈 Agent (RinshanAgent) 内部用纯 Python 的 _replay_events_to_state()
重建 GameState，然后再手动算候选动作（哪些牌能打/能立直/能荣和 etc.）。
这部分逻辑在 Python 里很慢，且容易和 libriichi 实现有微小规则差异。

本模块提供：
  LibriichiBoostedAgent
    - 继承 RinshanAgent，完全复用其模型推理逻辑
    - 用 libriichi.PlayerState 替换候选动作的生成部分
    - libriichi 在 Rust 实现了完整的天凤规则，比 Python 快 10~50x
    - 同时可从 PlayerState 精确读取向听数 / 振听 / 待牌，用于辅助任务

    对自对弈的提速来自：
      * 候选动作生成：Rust（PlayerState.last_cans）
      * 向听数计算：Rust（PlayerState.shanten）
      * 安全牌判断（未来 deal_in_risk）：Rust（PlayerState.waits）

依赖
----
    pip install libriichi  （本地已编译安装）
    或由 maturin build + pip install 得到

用法
----
    from rinshan.self_play.libriichi_agent import LibriichiBoostedAgent

    agent = LibriichiBoostedAgent(model, name="rinshan_lr", device="cuda")
    # 直接替换 RinshanAgent，Arena 接口完全兼容
"""
from __future__ import annotations

import json
from typing import Optional

try:
    from libriichi.state import PlayerState as LRPlayerState
    _LIBRIICHI_AVAILABLE = True
except ImportError:
    _LIBRIICHI_AVAILABLE = False
    LRPlayerState = None  # type: ignore

from rinshan.self_play.agent import RinshanAgent, BaseAgent
from rinshan.engine.action import (
    ActionType, encode_action, Action,
    chi_type_to_idx,
)
from rinshan.constants import (
    DISCARD_OFFSET, CHI_OFFSET, PON_OFFSET, DAIMINKAN_OFFSET,
    ANKAN_OFFSET, KAKAN_OFFSET,
    RIICHI_TOKEN, TSUMO_AGARI_TOKEN, RON_AGARI_TOKEN,
    PASS_TOKEN, NUM_TILE_TYPES,
)
from rinshan.tile import Tile


# ─────────────────────────────────────────────────────────────────────────────
# PlayerState 包装器
# ─────────────────────────────────────────────────────────────────────────────

class _LRStateTracker:
    """
    用 libriichi.PlayerState 跟踪单个玩家视角的局面。

    职责：
      - 接收 mjai 事件 dict，转为 JSON 字符串喂给 PlayerState.update()
      - 缓存最后一次 ActionCandidate，供 build_candidates() 读取
      - 在每局开始时重置状态
    """

    def __init__(self, seat: int):
        self.seat = seat
        self._ps: Optional[LRPlayerState] = None
        self._n_events = 0
        self._reset()

    def _reset(self) -> None:
        if _LIBRIICHI_AVAILABLE:
            self._ps = LRPlayerState(self.seat)
            # 喂 start_game
            self._ps.update(json.dumps({"type": "start_game", "id": self.seat}))
        self._n_events = 0

    def feed(self, events: list[dict]) -> None:
        """增量喂入新事件（比 n_events 之后的部分）"""
        if self._ps is None:
            return
        for ev in events[self._n_events:]:
            try:
                self._ps.update(json.dumps(ev))
            except Exception:
                # libriichi 对某些非标准事件会报错，忽略即可
                pass
        self._n_events = len(events)

    def feed_full(self, events: list[dict]) -> None:
        """全量重放（局初始化用）"""
        self._reset()
        self.feed(events)

    @property
    def ps(self) -> Optional[LRPlayerState]:
        return self._ps

    def shanten(self) -> int:
        if self._ps is None:
            return 8
        return int(self._ps.shanten)

    def waits(self) -> list[bool]:
        """34 维待牌布尔列表（听牌时有效）"""
        if self._ps is None:
            return [False] * 34
        return list(self._ps.waits)

    def build_candidates(self, pending: dict) -> list[int]:
        """
        从 libriichi.ActionCandidate 构建 Rinshan token 候选列表。

        返回 list[int]，每个元素是 Rinshan 动作空间的 token id。
        与 RinshanAgent 中的 _build_turn_candidates / _build_naki_candidates 等价，
        但完全依赖 Rust 计算，不依赖 Python GameState。
        """
        if self._ps is None:
            return []
        cans = self._ps.last_cans
        ptype = pending.get("type", "")
        candidates: list[int] = []

        if ptype == "turn_action":
            # ── 打牌候选 ──────────────────────────────
            if cans.can_discard:
                tehai = self._ps.tehai          # [u8; 34]，各牌张数
                akas  = self._ps.akas_in_hand   # [bool; 3]：5m/5p/5s 是否有赤

                for tile_id in range(34):
                    if tehai[tile_id] > 0:
                        candidates.append(DISCARD_OFFSET + tile_id)

                # 赤宝牌（同 tile_id 但 is_aka=True）额外加入候选
                aka_map = [(4, 34), (13, 35), (22, 36)]  # (tile_id, offset_idx)
                for tile_id, aka_offset_idx in aka_map:
                    suit_idx = [4, 13, 22].index(tile_id)
                    if akas[suit_idx] and tehai[tile_id] > 0:
                        candidates.append(DISCARD_OFFSET + aka_offset_idx)

            # ── 立直 ──────────────────────────────────
            if cans.can_riichi:
                candidates.append(RIICHI_TOKEN)

            # ── 自摸和 ────────────────────────────────
            if cans.can_tsumo_agari:
                candidates.append(TSUMO_AGARI_TOKEN)

            # ── 暗杠 ──────────────────────────────────
            if cans.can_ankan:
                for tile_id in self._ps.ankan_candidates:
                    candidates.append(ANKAN_OFFSET + int(tile_id))

            # ── 加杠 ──────────────────────────────────
            if cans.can_kakan:
                for tile_id in self._ps.kakan_candidates:
                    candidates.append(KAKAN_OFFSET + int(tile_id))

            # ── 九种九牌 ──────────────────────────────
            if cans.can_ryukyoku:
                from rinshan.constants import RYUKYOKU_TOKEN
                candidates.append(RYUKYOKU_TOKEN)

        elif ptype == "naki_or_pass":
            # ── 荣和 ──────────────────────────────────
            if cans.can_ron_agari:
                candidates.append(RON_AGARI_TOKEN)

            # ── 碰 ────────────────────────────────────
            if cans.can_pon:
                discard_tile_str = pending.get("tile", "")
                if discard_tile_str:
                    try:
                        t = Tile.from_mjai(discard_tile_str)
                        candidates.append(PON_OFFSET + t.tile_id)
                    except Exception:
                        pass

            # ── 吃 ────────────────────────────────────
            if cans.can_chi:
                discard_tile_str = pending.get("tile", "")
                if discard_tile_str:
                    try:
                        t = Tile.from_mjai(discard_tile_str)
                        suit = t.tile_id // 9
                        num  = t.tile_id % 9 + 1  # 1-based
                        tehai = self._ps.tehai
                        # 三种吃型：低(12x)/中(1x3)/高(x23)
                        # 吃型 form: 0=低(被吃牌最高), 1=中, 2=高(被吃牌最低)
                        if cans.can_chi_low and num >= 3:
                            low = num - 2
                            if (suit * 9 + low - 1) < 34 and (suit * 9 + low) < 34:
                                if tehai[suit * 9 + low - 1] > 0 and tehai[suit * 9 + low] > 0:
                                    candidates.append(CHI_OFFSET + chi_type_to_idx(suit, low, 2))
                        if cans.can_chi_mid and 2 <= num <= 8:
                            low = num - 1
                            if (suit * 9 + low - 1) < 34 and (suit * 9 + low + 1) < 34:
                                if tehai[suit * 9 + low - 1] > 0 and tehai[suit * 9 + low + 1] > 0:
                                    candidates.append(CHI_OFFSET + chi_type_to_idx(suit, low, 1))
                        if cans.can_chi_high and num <= 7:
                            low = num
                            if (suit * 9 + low) < 34 and (suit * 9 + low + 1) < 34:
                                if tehai[suit * 9 + low] > 0 and tehai[suit * 9 + low + 1] > 0:
                                    candidates.append(CHI_OFFSET + chi_type_to_idx(suit, low, 0))
                    except Exception:
                        pass

            # ── 大明杠 ────────────────────────────────
            if cans.can_daiminkan:
                discard_tile_str = pending.get("tile", "")
                if discard_tile_str:
                    try:
                        t = Tile.from_mjai(discard_tile_str)
                        candidates.append(DAIMINKAN_OFFSET + t.tile_id)
                    except Exception:
                        pass

            # ── PASS ──────────────────────────────────
            candidates.append(PASS_TOKEN)

        # 去重、保序
        seen: set[int] = set()
        result: list[int] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# LibriichiBoostedAgent
# ─────────────────────────────────────────────────────────────────────────────

class LibriichiBoostedAgent(RinshanAgent):
    """
    RinshanAgent + libriichi.PlayerState 加速版。

    候选动作生成、向听数计算均由 libriichi（Rust）完成。
    模型推理部分与 RinshanAgent 完全一致。

    当 libriichi 不可用时自动降级为标准 RinshanAgent。
    """

    def __init__(self, model, name: str = "rinshan_lr", **kwargs):
        super().__init__(model, name=name, **kwargs)

        if not _LIBRIICHI_AVAILABLE:
            import warnings
            warnings.warn(
                "libriichi not available, LibriichiBoostedAgent will "
                "fall back to standard RinshanAgent behavior.",
                RuntimeWarning,
                stacklevel=2,
            )
        # seat -> _LRStateTracker
        self._lr_trackers: dict[tuple, _LRStateTracker] = {}

    def _get_lr_tracker(self, seat: int, game_key: str,
                        player_events: list[dict]) -> _LRStateTracker:
        """按 (game_key, seat) 获取或创建 tracker，增量更新事件。"""
        cache_key = (game_key, seat)
        tracker = self._lr_trackers.get(cache_key)
        if tracker is None:
            tracker = _LRStateTracker(seat)
            self._lr_trackers[cache_key] = tracker

        # 如果事件变少了（新局），重置
        if len(player_events) < tracker._n_events:
            tracker.feed_full(player_events)
        else:
            tracker.feed(player_events)
        return tracker

    def react_batch_requests(
        self, requests: list[tuple[int, list[dict], dict]]
    ) -> list[dict]:
        """
        覆盖父类方法。

        libriichi 可用时：
          - 候选生成：_LRStateTracker.build_candidates()（Rust）
          - encoder + 模型推理 + token 解码：复用父类逻辑
          - _token_to_mjai 所需的 GameState 仍从父类缓存获取（chi 消耗牌解码需要）

        libriichi 不可用时：直接委托给父类。
        """
        if not _LIBRIICHI_AVAILABLE:
            return super().react_batch_requests(requests)

        from rinshan.data.dataset import collate_fn
        from rinshan.self_play.agent import _state_to_annotation, _token_to_mjai

        responses: list[dict | None] = [None] * len(requests)
        batch_indices:    list[int]       = []
        batch_encoded:    list[dict]      = []
        batch_candidates: list[list[int]] = []
        batch_pending:    list[dict]      = []
        batch_seats:      list[int]       = []
        batch_states:     list            = []

        for i, (seat, player_events, pending) in enumerate(requests):
            game_key = str(pending.get("_game_key", "default"))

            # ── 候选生成（Rust）──────────────────────────────
            tracker = self._get_lr_tracker(seat, game_key, player_events)
            candidates = tracker.build_candidates(pending)

            if not candidates:
                responses[i] = {"type": "pass", "actor": seat}
                continue

            # ── 编码仍需 Python GameState（token 序列） ──────
            state = self._get_cached_state(seat, player_events, pending)
            ann   = _state_to_annotation(state, seat, player_events, candidates)

            batch_indices.append(i)
            batch_encoded.append(self._encoder.encode(ann))
            batch_candidates.append(candidates)
            batch_pending.append(pending)
            batch_seats.append(seat)
            batch_states.append(state)

        # ── 批量推理 ──────────────────────────────────────────
        if batch_encoded:
            encoded    = collate_fn(batch_encoded)
            tokens     = encoded["tokens"].to(self.device)
            cand_mask  = encoded["candidate_mask"].to(self.device)
            pad_mask   = encoded["pad_mask"].to(self.device)
            b_tokens   = encoded["belief_tokens"].to(self.device)
            b_pad_mask = encoded["belief_pad_mask"].to(self.device)

            self.model.eval()
            with torch.inference_mode():
                action_idx, q_values = self.model.react(
                    tokens, cand_mask, pad_mask,
                    b_tokens, b_pad_mask,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    greedy=self.greedy,
                )

            for local_i, orig_i in enumerate(batch_indices):
                candidates    = batch_candidates[local_i]
                chosen_token  = candidates[action_idx[local_i].item()]
                responses[orig_i] = _token_to_mjai(
                    chosen_token,
                    batch_seats[local_i],
                    batch_states[local_i],
                    batch_pending[local_i],
                    can_tsumo=TSUMO_AGARI_TOKEN in candidates,
                    q_values=q_values[local_i],
                    candidates=candidates,
                )

        return [
            r if r is not None else {"type": "pass", "actor": requests[i][0]}
            for i, r in enumerate(responses)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数：检查 libriichi 是否可用
# ─────────────────────────────────────────────────────────────────────────────

def libriichi_available() -> bool:
    """返回 libriichi 是否已安装并可导入。"""
    return _LIBRIICHI_AVAILABLE
