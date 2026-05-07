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
import os
from typing import Optional

import torch

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


_VALID_DISCARD_TOKEN_MAP = {
    **{f"{n}m": DISCARD_OFFSET + (n - 1) for n in range(1, 10)},
    **{f"{n}p": DISCARD_OFFSET + (9 + n - 1) for n in range(1, 10)},
    **{f"{n}s": DISCARD_OFFSET + (18 + n - 1) for n in range(1, 10)},
    **{f"{n}z": DISCARD_OFFSET + (27 + n - 1) for n in range(1, 8)},
    "0m": DISCARD_OFFSET + 34,
    "0p": DISCARD_OFFSET + 35,
    "0s": DISCARD_OFFSET + 36,
}


def _replace_discard_candidates_with_valid_discards(candidates: list[int], pending: dict) -> list[int]:
    """
    用 Rust `valid_discards` 作为弃牌候选的权威来源。

    libriichi 的 PlayerState 通常很准，但一旦 Python/Rust 事件流出现轻微漂移，
    仅靠 `last_cans` 推出来的 discard token 仍可能和 Rust 当前真实手牌不一致，
    进而在 Rust 侧触发：
      "dahai XXX not in Rust hand, falling back to tsumogiri"

    为消除此类漂移：
    - 只要 pending 提供了 `valid_discards`，就无条件用它覆盖全部 discard 候选；
    - non-discard 候选（riichi/tsumo/ankan/...）保持原顺序；
    - discard 部分仍按 token 升序，尽量保持与训练时一致。
    """
    valid_discards = pending.get("valid_discards")
    if not valid_discards:
        return candidates

    rust_discards: list[int] = []
    seen_discards: set[int] = set()
    for pai in valid_discards:
        token = _VALID_DISCARD_TOKEN_MAP.get(pai)
        if token is not None and token not in seen_discards:
            seen_discards.add(token)
            rust_discards.append(token)

    if not rust_discards:
        return candidates

    rust_discards.sort()
    non_discards = [
        token for token in candidates
        if not (DISCARD_OFFSET <= token < DISCARD_OFFSET + 37)
    ]
    return rust_discards + non_discards


# ─────────────────────────────────────────────────────────────────────────────
# PlayerState 包装器
# ─────────────────────────────────────────────────────────────────────────────

# libriichi start_kyoku 要求 bakaze/jikaze 使用 "E"/"S"/"W"/"N"，
# 而 Rinshan 内部用 mjai 牌记法 "1z"~"4z"，需在喂入前转换。
_WIND_TILE_TO_STR: dict[str, str] = {"1z": "E", "2z": "S", "3z": "W", "4z": "N"}

# 喂给 libriichi 时需过滤掉的 Rinshan 私有字段
_LR_PRIV_KEYS: frozenset[str] = frozenset({
    "_game_key", "_rule_based_agari_guard",
    "_shanten", "_waits", "_tehai", "_candidates",
})


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
        new_events = events[self._n_events:]
        if not new_events:
            return
        dbg_path = os.environ.get("RINSHAN_ACTION_TRACE")  # 循环外取一次
        for ev in new_events:
            # 过滤私有字段（用模块级常量，避免每次构建）
            ev_clean = {k: v for k, v in ev.items() if k not in _LR_PRIV_KEYS}
            # start_kyoku: bakaze/jikaze 从 mjai 牌记法转为风向字符串
            if ev_clean.get("type") == "start_kyoku":
                bk = _WIND_TILE_TO_STR.get(ev_clean.get("bakaze"))
                jk = _WIND_TILE_TO_STR.get(ev_clean.get("jikaze"))
                if bk or jk:
                    ev_clean = dict(ev_clean)  # 只在需要时才拷贝
                    if bk:
                        ev_clean["bakaze"] = bk
                    if jk:
                        ev_clean["jikaze"] = jk
            try:
                self._ps.update(json.dumps(ev_clean))
            except Exception as exc:
                if dbg_path:
                    with open(dbg_path, "a", encoding="utf-8") as _f:
                        _f.write(json.dumps({
                            "phase": "lr_feed_error",
                            "seat": self.seat,
                            "event": ev_clean,
                            "error": str(exc),
                        }, ensure_ascii=False) + "\n")
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
        """按 (stable_game_key, seat) 获取或创建 tracker，增量更新事件。

        Rust arena 的 _game_key 格式为 "rust:gN:iK"，其中 iK 在同一局内
        可能随 iteration 变化（例如 reach 前后从 i0 变为 i2）。
        只取 ":i" 之前的稳定前缀，保证同一局所有 pending 共享同一 tracker。
        """
        _ri = game_key.find(":i")
        stable_key = game_key[:_ri] if _ri != -1 else game_key
        cache_key = (stable_key, seat)
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
        from rinshan.self_play.agent import _state_to_annotation, _token_to_mjai, _single_forced_response

        responses: list[dict | None] = [None] * len(requests)
        batch_indices:    list[int]       = []
        batch_encoded:    list[dict]      = []
        batch_candidates: list[list[int]] = []
        batch_pending:    list[dict]      = []
        batch_seats:      list[int]       = []
        batch_states:     list            = []

        for i, (seat, player_events, pending) in enumerate(requests):
            game_key = str(pending.get("_game_key", "default"))
            _ri = game_key.find(":i")
            stable_gk = game_key[:_ri] if _ri != -1 else game_key
            riichi_key = (stable_gk, seat)

            # ── 候选生成（Rust）──────────────────────────────
            tracker = self._get_lr_tracker(seat, game_key, player_events)
            candidates = tracker.build_candidates(pending)
            ptype = str(pending.get("type", ""))

            # turn_action 下 Rust pending 是动作合法性的权威来源。
            # libriichi 的 PlayerState.last_cans 已经很准，但这里仍用 pending
            # 做最后 reconcile，避免 Python 侧误产生 Rust 不接受的动作。
            if ptype == "turn_action":
                # ── 立直宣言后强制弃牌 ───────────────────────────────────
                # Rust 在收到 reach 后会再发一次 turn_action 要弃牌。
                # pending 的 can_riichi 此时可能仍为 True，所以用内部状态追踪。
                if riichi_key in self._pending_riichi_discard:
                    self._pending_riichi_discard.discard(riichi_key)
                    vd = pending.get("valid_discards")
                    if vd:
                        disc_cands: list[int] = []
                        seen_d: set[int] = set()
                        for _pai in vd:
                            tok = _VALID_DISCARD_TOKEN_MAP.get(_pai)
                            if tok is not None and tok not in seen_d:
                                seen_d.add(tok)
                                disc_cands.append(tok)
                        disc_cands.sort()
                        candidates = disc_cands
                    else:
                        candidates = [t for t in candidates
                                      if DISCARD_OFFSET <= t < DISCARD_OFFSET + 37]
                    if not candidates:
                        vd2 = pending.get("valid_discards")
                        if vd2:
                            responses[i] = {"type": "dahai", "actor": seat,
                                            "pai": vd2[0], "tsumogiri": False}
                        else:
                            responses[i] = {"type": "pass", "actor": seat}
                        continue
                    state = self._get_cached_state(seat, player_events, pending)
                    ann = _state_to_annotation(state, seat, player_events, candidates)
                    batch_indices.append(i)
                    batch_encoded.append(self._encoder.encode(ann))
                    batch_candidates.append(candidates)
                    batch_pending.append(pending)
                    batch_seats.append(seat)
                    batch_states.append(state)
                    continue

                candidates = _replace_discard_candidates_with_valid_discards(candidates, pending)

                if "can_tsumo" in pending:
                    if pending.get("can_tsumo", False):
                        if TSUMO_AGARI_TOKEN not in candidates:
                            candidates = [TSUMO_AGARI_TOKEN] + candidates
                    else:
                        candidates = [t for t in candidates if t != TSUMO_AGARI_TOKEN]
                if "can_riichi" in pending:
                    if pending.get("can_riichi", False):
                        if RIICHI_TOKEN not in candidates:
                            insert_at = 0
                            while insert_at < len(candidates) and (
                                DISCARD_OFFSET <= candidates[insert_at] < DISCARD_OFFSET + 37
                            ):
                                insert_at += 1
                            candidates = candidates[:insert_at] + [RIICHI_TOKEN] + candidates[insert_at:]
                    else:
                        candidates = [t for t in candidates if t != RIICHI_TOKEN]
                from rinshan.constants import RYUKYOKU_TOKEN
                if "can_ryukyoku" in pending:
                    if pending.get("can_ryukyoku", False):
                        if RYUKYOKU_TOKEN not in candidates:
                            candidates.append(RYUKYOKU_TOKEN)
                    else:
                        candidates = [t for t in candidates if t != RYUKYOKU_TOKEN]
                if "can_ankan" in pending and not pending.get("can_ankan", False):
                    candidates = [
                        t for t in candidates
                        if not (ANKAN_OFFSET <= t < ANKAN_OFFSET + NUM_TILE_TYPES)
                    ]
                if "can_kakan" in pending and not pending.get("can_kakan", False):
                    candidates = [
                        t for t in candidates
                        if not (KAKAN_OFFSET <= t < KAKAN_OFFSET + NUM_TILE_TYPES)
                    ]

            if self.enable_rule_based_agari_guard:
                pending["_rule_based_agari_guard"] = True

            # ── 编码仍需 Python GameState（token 序列） ──────
            state = self._get_cached_state(seat, player_events, pending)

            quick = _single_forced_response(
                seat,
                pending,
                candidates,
                state,
                force_one_candidate_only=bool(getattr(self, "enable_quick_eval", False)),
            )
            if quick is not None:
                responses[i] = quick
                # quick path 如果发出了 reach，同样需要记录强制弃牌
                if quick.get("type") == "reach":
                    self._pending_riichi_discard.add((stable_gk, seat))
                dbg_path = os.environ.get("RINSHAN_ACTION_TRACE")
                if dbg_path and quick.get("type") != "pass":
                    with open(dbg_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "phase": "quick",
                            "seat": seat,
                            "pending_type": ptype,
                            "pending": pending,
                            "response": quick,
                            "candidates": candidates,
                        }, ensure_ascii=False) + "\n")
                continue

            if not candidates:
                responses[i] = {"type": "pass", "actor": seat}
                continue

            ann = _state_to_annotation(state, seat, player_events, candidates)

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
            _amp_ctx = self._make_autocast_ctx()
            with torch.inference_mode(), _amp_ctx:
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
                # 如果模型选了 reach，记录该座位需要一次强制弃牌
                if responses[orig_i].get("type") == "reach":
                    _gk = str(batch_pending[local_i].get("_game_key", "default"))
                    _ri2 = _gk.find(":i")
                    _stable_gk = _gk[:_ri2] if _ri2 != -1 else _gk
                    self._pending_riichi_discard.add((_stable_gk, batch_seats[local_i]))
                dbg_path = os.environ.get("RINSHAN_ACTION_TRACE")
                if dbg_path and responses[orig_i].get("type") in {"reach", "hora", "chi", "pon", "daiminkan", "ankan", "kakan", "dahai", "pass"}:
                    with open(dbg_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "phase": "model",
                            "seat": batch_seats[local_i],
                            "pending_type": batch_pending[local_i].get("type"),
                            "pending": batch_pending[local_i],
                            "response": responses[orig_i],
                            "chosen_token": chosen_token,
                            "candidates": candidates,
                        }, ensure_ascii=False) + "\n")

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
