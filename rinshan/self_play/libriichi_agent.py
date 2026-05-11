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
    # 数牌（Rinshan 格式）
    **{f"{n}m": DISCARD_OFFSET + (n - 1) for n in range(1, 10)},
    **{f"{n}p": DISCARD_OFFSET + (9 + n - 1) for n in range(1, 10)},
    **{f"{n}s": DISCARD_OFFSET + (18 + n - 1) for n in range(1, 10)},
    # 字牌（Rinshan 格式 "1z~7z"）
    **{f"{n}z": DISCARD_OFFSET + (27 + n - 1) for n in range(1, 8)},
    # 字牌（libriichi 格式 "E/S/W/N/P/F/C"）
    "E": DISCARD_OFFSET + 27, "S": DISCARD_OFFSET + 28,
    "W": DISCARD_OFFSET + 29, "N": DISCARD_OFFSET + 30,
    "P": DISCARD_OFFSET + 31, "F": DISCARD_OFFSET + 32,
    "C": DISCARD_OFFSET + 33,
    # 赤宝牌（Rinshan 格式 "0x"）
    "0m": DISCARD_OFFSET + 34,
    "0p": DISCARD_OFFSET + 35,
    "0s": DISCARD_OFFSET + 36,
    # 赤宝牌（libriichi 格式 "5xr"）
    "5mr": DISCARD_OFFSET + 34,
    "5pr": DISCARD_OFFSET + 35,
    "5sr": DISCARD_OFFSET + 36,
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

# Rust libriichi tile_id(0-33) -> LR 格式字符串（与 MJAI_PAI_STRINGS 顺序一致）
_LR_PAI_STRINGS: list[str] = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "E","S","W","N","P","F","C",
]
# tile_id(deaka) -> 赤宝牌 LR 格式
_AKA_BY_TILE_ID: dict[int, str] = {4: "5mr", 13: "5pr", 22: "5sr"}
# akas_in_hand 下标
_AKA_IDX: dict[int, int] = {4: 0, 13: 1, 22: 2}
#
# libriichi PlayerState.update() 不接受以下 Rinshan/mjai 格式，喂入前必须转换：
#   1. 风牌/三元牌："1z"~"7z" → "E"/"S"/"W"/"N"/"P"/"F"/"C"
#   2. 赤宝牌：    "0m"/"0p"/"0s" → "5mr"/"5pr"/"5sr"
#
# 同一张表覆盖 bakaze/jikaze 字段（风向字符串）和所有 pai 牌字段。
_TILE_RINSHAN_TO_LR: dict[str, str] = {
    # 风牌
    "1z": "E", "2z": "S", "3z": "W", "4z": "N",
    # 三元牌
    "5z": "P", "6z": "F", "7z": "C",
    # 赤宝牌
    "0m": "5mr", "0p": "5pr", "0s": "5sr",
}
# 向后兼容：_WIND_TILE_TO_STR 仍被 feed() 中 bakaze/jikaze 分支引用
_WIND_TILE_TO_STR: dict[str, str] = {k: v for k, v in _TILE_RINSHAN_TO_LR.items()
                                     if k in ("1z", "2z", "3z", "4z")}

# libriichi → Rinshan：last_self_tsumo() 等接口返回值转回 Tile 构造参数
# 字牌直接走 Tile.from_mjai() 无法识别 libriichi 格式，需要此映射。
_TILE_LR_TO_RINSHAN: dict[str, tuple[int, bool]] = {
    # 赤宝牌
    "5mr": (4,  True),
    "5pr": (13, True),
    "5sr": (22, True),
    # 风牌
    "E": (27, False), "S": (28, False), "W": (29, False), "N": (30, False),
    # 三元牌
    "P": (31, False), "F": (32, False), "C": (33, False),
}
# 向后兼容别名
_AKA_RINSHAN_TO_LR = _TILE_RINSHAN_TO_LR
_AKA_LR_TO_RINSHAN = _TILE_LR_TO_RINSHAN


def _lr_tile_to_rinshan_args(lr_pai: str) -> tuple[int, bool]:
    """把 libriichi 牌字符串（含赤宝牌 5xr、字牌 E/S/W/N/P/F/C 格式）转为 Rinshan Tile 构造参数。"""
    if lr_pai in _TILE_LR_TO_RINSHAN:
        return _TILE_LR_TO_RINSHAN[lr_pai]
    t = Tile.from_mjai(lr_pai)  # 普通数牌，Rinshan 认识
    return t.tile_id, t.is_aka


def _convert_pai_fields_for_lr(ev: dict) -> dict:
    """
    把事件 dict 中所有牌字段从 Rinshan 记法转为 libriichi 记法。
    只在有需要转换的牌时才拷贝 dict（热路径优化）。

    覆盖的转换：
      字牌  "1z"~"7z" → "E"/"S"/"W"/"N"/"P"/"F"/"C"
      赤宝牌 "0m"/"0p"/"0s" → "5mr"/"5pr"/"5sr"

    处理的字段：
      pai, dora_marker, consumed（列表）, tehais（二维列表，start_kyoku 专用）
    """
    _CVT = _TILE_RINSHAN_TO_LR

    def _cvt(s: str) -> str:
        return _CVT.get(s, s)

    def _cvt_list(lst: list) -> list:
        return [_CVT.get(x, x) for x in lst]

    needs_copy = False
    for field in ("pai", "dora_marker"):
        if ev.get(field) in _CVT:
            needs_copy = True
            break
    if not needs_copy:
        if any(x in _CVT for x in ev.get("consumed", [])):
            needs_copy = True
    if not needs_copy and ev.get("type") == "start_kyoku":
        for hand in ev.get("tehais", []):
            if any(x in _CVT for x in hand):
                needs_copy = True
                break

    if not needs_copy:
        return ev

    ev = dict(ev)  # 浅拷贝
    if "pai" in ev:
        ev["pai"] = _cvt(ev["pai"])
    if "dora_marker" in ev:
        ev["dora_marker"] = _cvt(ev["dora_marker"])
    if "consumed" in ev:
        ev["consumed"] = _cvt_list(ev["consumed"])
    if ev.get("type") == "start_kyoku" and "tehais" in ev:
        ev["tehais"] = [_cvt_list(hand) for hand in ev["tehais"]]
    return ev


def _fix_naki_consumed(resp: dict, ps) -> dict:
    """
    将 chi/pon/daiminkan 的 consumed 字段规范化为 libriichi 格式。

    设计原则：
    - consumed 的 tile_id 由 Rust pending.can_chi/pon 保证合法（Rust 已验证手牌存在）
    - 赤宝牌处理：始终使用普通牌格式（不用 5xr），Rust validate_reaction 只检查 deaka 后
      的 tehai[tid]>0，普通牌格式永远安全。
      （tracker.akas_in_hand 可能与 Rust 不同步，用赤宝牌有误报风险）
    - 字牌格式从 Rinshan "xz" 转换为 LR "E/S/W/N/P/F/C"
    """
    rtype = resp.get("type")
    if rtype not in ("chi", "pon", "daiminkan"):
        return resp

    def _safe_str(tid: int) -> str:
        """返回 tid 对应的普通牌 LR 格式字符串（非赤宝牌）。"""
        return _LR_PAI_STRINGS[tid] if tid < 34 else "?"

    # ── 获取 pai 的 tile_id ──────────────────────────────
    pai_raw = resp.get("pai", "")
    pai_lr  = _TILE_RINSHAN_TO_LR.get(pai_raw, pai_raw)
    if pai_lr in _TILE_LR_TO_RINSHAN:
        pai_tid, _ = _TILE_LR_TO_RINSHAN[pai_lr]
    else:
        try:
            from rinshan.tile import Tile as _T
            pai_tid = _T.from_mjai(pai_lr).tile_id
        except Exception:
            return resp

    if rtype in ("pon", "daiminkan"):
        n_consumed = 2 if rtype == "pon" else 3
        new_consumed = [_safe_str(pai_tid)] * n_consumed
        if new_consumed != resp.get("consumed"):
            resp = dict(resp)
            resp["consumed"] = new_consumed
        return resp

    # chi: 将 consumed 各牌转为 LR 普通牌格式
    consumed_raw = resp.get("consumed", [])
    if len(consumed_raw) != 2:
        return resp

    new_consumed = []
    changed = False
    for c_raw in consumed_raw:
        c_lr = _TILE_RINSHAN_TO_LR.get(c_raw, c_raw)
        if c_lr in _TILE_LR_TO_RINSHAN:
            tid, _ = _TILE_LR_TO_RINSHAN[c_lr]
        else:
            try:
                from rinshan.tile import Tile as _T
                tid = _T.from_mjai(c_lr).tile_id
            except Exception:
                new_consumed.append(c_raw)
                continue
        new_c = _safe_str(tid)
        if new_c != c_raw:
            changed = True
        new_consumed.append(new_c)

    if changed:
        resp = dict(resp)
        resp["consumed"] = new_consumed
    return resp

    if new_consumed != list(consumed_raw):
        resp = dict(resp)
        resp["consumed"] = new_consumed
    return resp


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
            # 赤宝牌格式转换："0x" → "5xr"（libriichi 不接受 "0x" 格式）
            ev_clean = _convert_pai_fields_for_lr(ev_clean)
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

    def build_candidates(self, pending: dict, player_events: list | None = None) -> list[int]:
        """
        从 libriichi.ActionCandidate 构建 Rinshan token 候选列表。

        返回 list[int]，每个元素是 Rinshan 动作空间的 token id。
        与 RinshanAgent 中的 _build_turn_candidates / _build_naki_candidates 等价，
        但完全依赖 Rust 计算，不依赖 Python GameState。
        """
        if self._ps is None:
            import logging as _log3
            _log3.getLogger("rinshan_agent").warning(
                "build_candidates: _ps is None for seat=%d n_events=%d",
                self.seat, self._n_events,
            )
            return []
        cans = self._ps.last_cans
        ptype = pending.get("type", "")
        candidates: list[int] = []

        if ptype == "turn_action":
            # ── 打牌候选 ──────────────────────────────
            # 注意：在 can_kakan / can_ankan 为 True 时，Rust 的 last_cans 仍可能同时保持
            # can_discard=True；若直接从 tehai 枚举弃牌，会让模型在“应当先处理杠”的状态下
            # 继续看到普通弃牌候选。这里仍保留基础枚举，但后面会被 pending 的合法性做二次裁剪。
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
            # 优先使用 Rust pending.ankan_candidates（最权威，Rust 直接下发的合法牌列表）。
            # 若 pending 里有此字段（新版 Rust），直接用；否则退回 player_events 扫描。
            if pending.get("can_ankan", cans.can_ankan):
                ankan_from_pending: list = pending.get("ankan_candidates", [])
                if ankan_from_pending:
                    # pending 里的牌名是 Rinshan 格式（如 "3m", "0p"）
                    for pai_str in ankan_from_pending:
                        try:
                            t = Tile.from_mjai(pai_str)
                            candidates.append(ANKAN_OFFSET + t.tile_id)
                        except Exception:
                            pass
                else:
                    # 退回：从 player_events 末尾反向扫描 last_tsumo
                    ankan_tid_from_events: int | None = None
                    if player_events:
                        seat_id = self.seat
                        for ev in reversed(player_events):
                            if ev.get("type") == "tsumo" and ev.get("actor") == seat_id:
                                raw = ev.get("pai", "")
                                if raw and raw != "?":
                                    lr = _TILE_RINSHAN_TO_LR.get(raw, raw)
                                    if lr in _TILE_LR_TO_RINSHAN:
                                        ankan_tid_from_events, _ = _TILE_LR_TO_RINSHAN[lr]
                                    else:
                                        try:
                                            ankan_tid_from_events, _ = _lr_tile_to_rinshan_args(lr)
                                        except Exception:
                                            pass
                                break
                            elif ev.get("type") in ("dahai", "chi", "pon", "daiminkan",
                                                    "kakan", "ankan", "reach"):
                                if ev.get("actor") == seat_id:
                                    break
                    if ankan_tid_from_events is not None:
                        candidates.append(ANKAN_OFFSET + ankan_tid_from_events)
                    else:
                        for pai_str in self._ps.ankan_candidates():
                            tid, _ = _lr_tile_to_rinshan_args(pai_str)
                            candidates.append(ANKAN_OFFSET + tid)

            # ── 加杠 ──────────────────────────────────
            # 同样优先使用 Rust pending.kakan_candidates
            if pending.get("can_kakan", cans.can_kakan):
                kakan_from_pending: list = pending.get("kakan_candidates", [])
                if kakan_from_pending:
                    for pai_str in kakan_from_pending:
                        try:
                            t = Tile.from_mjai(pai_str)
                            candidates.append(KAKAN_OFFSET + t.tile_id)
                        except Exception:
                            pass
                else:
                    for pai_str in self._ps.kakan_candidates():
                        tid, _ = _lr_tile_to_rinshan_args(pai_str)
                        candidates.append(KAKAN_OFFSET + tid)

            # ── 九种九牌 ──────────────────────────────
            if cans.can_ryukyoku:
                from rinshan.constants import RYUKYOKU_TOKEN
                candidates.append(RYUKYOKU_TOKEN)

        elif ptype == "naki_or_pass":
            # naki_or_pass: Rust pending 是动作合法性的权威来源。
            # 直接使用 pending 里的 can_* 字段（而非 tracker.last_cans），
            # 避免 tracker 与 Rust 实际状态不同步导致非法动作。
            discard_tile_str = pending.get("tile", "")
            # tile 可能是 libriichi 格式（E/S/W/N/P/F/C 或 5xr）
            discard_tile_str_rinshan = {v: k for k, v in _TILE_RINSHAN_TO_LR.items()}.get(
                discard_tile_str, discard_tile_str
            )

            # ── 荣和 ──────────────────────────────────
            if pending.get("can_ron", cans.can_ron_agari):
                candidates.append(RON_AGARI_TOKEN)

            # ── 碰 ────────────────────────────────────
            if pending.get("can_pon", False):
                if discard_tile_str:
                    try:
                        t = Tile.from_mjai(discard_tile_str_rinshan)
                        candidates.append(PON_OFFSET + t.tile_id)
                    except Exception:
                        pass

            # ── 吃 ────────────────────────────────────
            can_chi_low  = pending.get("can_chi_low",  False)
            can_chi_mid  = pending.get("can_chi_mid",  False)
            can_chi_high = pending.get("can_chi_high", False)
            if (can_chi_low or can_chi_mid or can_chi_high) and discard_tile_str:
                try:
                    t = Tile.from_mjai(discard_tile_str_rinshan)
                    suit = t.tile_id // 9
                    num  = t.tile_id % 9 + 1  # 1-based
                    # Rust update.rs set_can_chi_from_tile 정의:
                    # can_chi_low:  tehai[tile+1]>0 && tehai[tile+2]>0  → consumed=[tile+1, tile+2]
                    # can_chi_mid:  tehai[tile-1]>0 && tehai[tile+1]>0  → consumed=[tile-1, tile+1]
                    # can_chi_high: tehai[tile-2]>0 && tehai[tile-1]>0  → consumed=[tile-2, tile-1]
                    # form=0: 吃対象=low+2 → consumed=[low,low+1]
                    # form=1: 吃対象=low+1 → consumed=[low,low+2]
                    # form=2: 吃対象=low   → consumed=[low+1,low+2]
                    if can_chi_low and num <= 7:   # consumed=[tile+1, tile+2], 吃対象=tile=low
                        low = num                  # low=tile_num, form=2
                        candidates.append(CHI_OFFSET + chi_type_to_idx(suit, low, 2))
                    if can_chi_mid and 2 <= num <= 8:  # consumed=[tile-1, tile+1], 吃対象=tile=low+1
                        low = num - 1                  # low=tile_num-1, form=1
                        candidates.append(CHI_OFFSET + chi_type_to_idx(suit, low, 1))
                    if can_chi_high and num >= 3:  # consumed=[tile-2, tile-1], 吃対象=tile=low+2
                        low = num - 2              # low=tile_num-2, form=0
                        candidates.append(CHI_OFFSET + chi_type_to_idx(suit, low, 0))
                except Exception:
                    pass

            # ── 大明杠 ────────────────────────────────
            if pending.get("can_daiminkan", False):
                if discard_tile_str:
                    try:
                        t = Tile.from_mjai(discard_tile_str_rinshan)
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

        国切换检测逻辑（修复版）：
        1. player_events 第一条是 start_kyoku → 必须全量重置（新局开始）
        2. len(player_events) < tracker._n_events → 日志被清空，全量重置
        3. 否则增量更新
        """
        _ri = game_key.find(":i")
        stable_key = game_key[:_ri] if _ri != -1 else game_key
        cache_key = (stable_key, seat)
        tracker = self._lr_trackers.get(cache_key)
        if tracker is None:
            tracker = _LRStateTracker(seat)
            self._lr_trackers[cache_key] = tracker

        # 检测新局开始：player_events[0] 为 start_kyoku 时强制全量重置
        # 这是最可靠的检测方式，不依赖事件数量比较（避免两者都为0时的漏检）
        is_new_kyoku = (
            bool(player_events)
            and player_events[0].get("type") == "start_kyoku"
            and tracker._n_events > 0  # 已有历史才需要重置
        ) or (
            len(player_events) < tracker._n_events  # 日志被清空（end_kyoku 后）
        )

        if is_new_kyoku:
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
        batch_trackers:   list            = []  # 为 ankan consumed 修正保留 tracker

        for i, (seat, player_events, pending) in enumerate(requests):
            game_key = str(pending.get("_game_key", "default"))
            _ri = game_key.find(":i")
            stable_gk = game_key[:_ri] if _ri != -1 else game_key
            riichi_key = (stable_gk, seat)

            # ── 候选生成（Rust）──────────────────────────────
            tracker = self._get_lr_tracker(seat, game_key, player_events)
            candidates = tracker.build_candidates(pending, player_events)
            ptype = str(pending.get("type", ""))

            # ── 注入 Rust 权威 last_self_tsumo，供 _token_to_mjai 判断 tsumogiri ──
            # 不依赖 tracker.ps（parallel_games>1 时 tracker 与 Rust 实际状态可能错位），
            # 直接反向扫描本轮 Rust 传入的 player_events 末尾找最后一次该 seat 的摸牌事件。
            # 这是最可靠的方式：player_events 由 Rust 本轮下发，末尾就是当前局面。
            if ptype == "turn_action":
                last_tsumo_pai: str | None = None
                for ev in reversed(player_events):
                    if ev.get("type") == "tsumo" and ev.get("actor") == seat:
                        raw_pai = ev.get("pai", "")
                        if raw_pai and raw_pai != "?":
                            # 转成 libriichi 格式（字牌/赤宝牌可能是 Rinshan 格式 1z/0s）
                            last_tsumo_pai = _TILE_RINSHAN_TO_LR.get(raw_pai, raw_pai)
                        break
                    elif ev.get("type") in ("dahai", "chi", "pon", "daiminkan",
                                            "kakan", "ankan", "reach"):
                        # 到了弃牌/副露/立直就说明摸牌已被消耗
                        if ev.get("actor") == seat:
                            break
                # 始终注入（包括 None），让 _token_to_mjai 统一走 rust 路径：
                # - last_tsumo_pai is not None → 正常 tsumogiri 判断
                # - last_tsumo_pai is None     → 强制 tsumogiri=False（吃/碰/杠/立直后无摸牌）
                pending = dict(pending)  # 浅拷贝，避免修改原始 pending
                pending["_rust_last_tsumo"] = last_tsumo_pai

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

                # 先把普通弃牌候选替换成 Rust 权威 valid_discards
                candidates = _replace_discard_candidates_with_valid_discards(candidates, pending)

                # 再严格按 pending 的 can_* 权威裁剪所有非 discard 动作，避免 stale can_*。
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
                if "can_ankan" in pending:
                    if not pending.get("can_ankan", False):
                        candidates = [
                            t for t in candidates
                            if not (ANKAN_OFFSET <= t < ANKAN_OFFSET + NUM_TILE_TYPES)
                        ]
                    else:
                        # 只保留 pending 明确允许的 ankan 牌
                        allowed = {
                            ANKAN_OFFSET + Tile.from_mjai(p).tile_id
                            for p in pending.get("ankan_candidates", [])
                        }
                        candidates = [
                            t for t in candidates
                            if not (ANKAN_OFFSET <= t < ANKAN_OFFSET + NUM_TILE_TYPES) or t in allowed
                        ]
                if "can_kakan" in pending:
                    if not pending.get("can_kakan", False):
                        candidates = [
                            t for t in candidates
                            if not (KAKAN_OFFSET <= t < KAKAN_OFFSET + NUM_TILE_TYPES)
                        ]
                    else:
                        # 只保留 pending 明确允许的 kakan 牌，防止 stale can_kakan / stale tracker
                        allowed = {
                            KAKAN_OFFSET + Tile.from_mjai(p).tile_id
                            for p in pending.get("kakan_candidates", [])
                        }
                        candidates = [
                            t for t in candidates
                            if not (KAKAN_OFFSET <= t < KAKAN_OFFSET + NUM_TILE_TYPES) or t in allowed
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
                # turn_action인데 candidates가 비면 트래커 드리프트가 원인.
                # pass를 반환하면 Rust가 Event::None으로 해석 → 연속 tsumo 발생.
                # valid_discards 또는 강제 tsumogiri로 안전하게 fallback.
                import logging as _log
                _log.getLogger("rinshan_agent").warning(
                    "empty candidates for seat=%d ptype=%s game_key=%s, "
                    "falling back to safe discard",
                    seat, ptype, pending.get("_game_key","?")
                )
                if ptype == "turn_action":
                    vd = pending.get("valid_discards")
                    if vd:
                        responses[i] = {
                            "type": "dahai", "actor": seat,
                            "pai": vd[0], "tsumogiri": False,
                        }
                    elif pending.get("can_tsumo"):
                        responses[i] = {"type": "hora", "actor": seat, "target": seat}
                    else:
                        # 최후 수단: last_self_tsumo 에서 tsumogiri
                        lt = pending.get("forced_pai")
                        if lt:
                            responses[i] = {
                                "type": "dahai", "actor": seat,
                                "pai": lt,
                                "tsumogiri": bool(pending.get("forced_tsumogiri", True)),
                            }
                        else:
                            responses[i] = {"type": "pass", "actor": seat}
                else:
                    responses[i] = {"type": "pass", "actor": seat}
                continue

            ann = _state_to_annotation(state, seat, player_events, candidates)

            batch_indices.append(i)
            batch_encoded.append(self._encoder.encode(ann))
            batch_candidates.append(candidates)
            batch_pending.append(pending)
            batch_seats.append(seat)
            batch_states.append(state)
            batch_trackers.append(tracker)

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
                # 模型选了 ankan：consumed 直接从 token 计算，不依赖 tracker
                # Rust build_candidates 里 can_ankan=true 已保证手牌有 4 张，
                # 与 chi/pon 相同原则：不用赤宝牌格式，只发普通牌，Rust 保证 deaka 合法
                if responses[orig_i].get("type") == "ankan":
                    tid = chosen_token - ANKAN_OFFSET   # deaka tile_id (0-33)
                    plain = _LR_PAI_STRINGS[tid] if tid < 34 else "?"
                    responses[orig_i] = {
                        "type": "ankan",
                        "actor": batch_seats[local_i],
                        "consumed": [plain] * 4,
                    }
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

        # ── 统一转换响应中的 pai 字段为 libriichi 格式 ──────────────────────
        # Rust libriichi 不接受 Rinshan/mjai 的 "0x"（赤宝牌）和 "xz"（字牌）格式，
        # 必须转换为 "5xr" 和 "E/S/W/N/P/F/C" 格式才能通过 Rust 校验。
        # 同时用 Rust PlayerState.tehai 修正 chi/pon/daiminkan 的 consumed 字段，
        # 避免 Python state 漂移导致手牌中不存在的牌出现在 consumed 里。
        final = []
        for i, r in enumerate(responses):
            if r is None:
                import logging as _log2
                seat_i, _evs_i, pend_i = requests[i]
                _log2.getLogger("rinshan_agent").warning(
                    "response[%d] is None (seat=%d ptype=%s), falling back to safe discard",
                    i, seat_i, pend_i.get("type","?"),
                )
                # 从 valid_discards 恢复，避免触发连续 tsumo
                vd = pend_i.get("valid_discards")
                if vd and pend_i.get("type") == "turn_action":
                    r = {"type": "dahai", "actor": seat_i, "pai": vd[0], "tsumogiri": False}
                else:
                    r = {"type": "pass", "actor": seat_i}
            # 修正 chi/pon/daiminkan consumed（用 Rust 权威手牌）
            if r.get("type") in ("chi", "pon", "daiminkan"):
                seat_i, player_events_i, pending_i = requests[i]
                game_key_i = str(pending_i.get("_game_key", "default"))
                tr_i = self._get_lr_tracker(seat_i, game_key_i, player_events_i)
                if tr_i.ps is not None:
                    r = _fix_naki_consumed(r, tr_i.ps)
            final.append(_convert_pai_fields_for_lr(r))
        return final


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数：检查 libriichi 是否可用
# ─────────────────────────────────────────────────────────────────────────────

def libriichi_available() -> bool:
    """返回 libriichi 是否已安装并可导入。"""
    return _LIBRIICHI_AVAILABLE
