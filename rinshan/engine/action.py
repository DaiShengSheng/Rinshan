"""
Action — 动作定义和 token 编解码

ActionType 枚举对应全部合法动作类型。
encode_action / decode_action 在 Action 对象 ↔ token id 之间转换。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional

from rinshan.tile import Tile
from rinshan.constants import (
    TILE_OFFSET, AKA_OFFSET, DISCARD_OFFSET,
    RIICHI_TOKEN, TSUMO_AGARI_TOKEN, RON_AGARI_TOKEN,
    RYUKYOKU_TOKEN, PASS_TOKEN,
    CHI_OFFSET, PON_OFFSET, DAIMINKAN_OFFSET,
    ANKAN_OFFSET, KAKAN_OFFSET,
    NUM_TILE_TYPES,
)


class ActionType(IntEnum):
    DISCARD    = 0   # 打牌
    RIICHI     = 1   # 立直宣言（之后跟打牌）
    TSUMO      = 2   # 自摸和
    RON        = 3   # 荣和
    CHI        = 4   # 吃
    PON        = 5   # 碰
    DAIMINKAN  = 6   # 大明杠
    ANKAN      = 7   # 暗杠
    KAKAN      = 8   # 加杠
    RYUKYOKU   = 9   # 九种九牌流局
    PASS       = 10  # 不动作


@dataclass
class Action:
    type: ActionType
    tile: Optional[Tile] = None      # 打牌/鸣牌时的牌
    consumed: Optional[list[Tile]] = None  # 鸣牌消耗的手牌
    actor: int = 0
    target: int = 0                  # 鸣牌来自的玩家

    def __repr__(self):
        if self.tile:
            return f"Action({self.type.name}, {self.tile})"
        return f"Action({self.type.name})"


# ─────────────────────────────────────────────
# CHI 编码辅助（90 种）
# ─────────────────────────────────────────────
# 吃有三种形式：低（12x）、中（1x3）、高（x89）
# 对每种花色 × 7 种起始位置 × 3 形式 → 总共 3 × 3 × 10 = 90 种
# 简化编码：chi_type = suit * 30 + offset（见下）

def chi_type_to_idx(suit: int, low_tile_num: int, form: int) -> int:
    """
    suit: 0m 1p 2s
    low_tile_num: 吃型最小的那张（1-7）
    form: 0=低(吃对象是 low+2), 1=中(吃对象是 low+1), 2=高(吃对象是 low)
    """
    return suit * 30 + (low_tile_num - 1) * 3 + form


def idx_to_chi_type(idx: int):
    suit    = idx // 30
    rem     = idx % 30
    low_num = rem // 3 + 1
    form    = rem % 3
    return suit, low_num, form


# ─────────────────────────────────────────────
# 编码：Action → token id
# ─────────────────────────────────────────────

def encode_action(action: Action) -> int:
    """将 Action 转为候选 token id"""
    match action.type:
        case ActionType.DISCARD:
            # 打牌：37 种（34 普通 + 3 赤）
            if action.tile and action.tile.is_aka:
                aka_idx = {4: 34, 13: 35, 22: 36}[action.tile.tile_id]
                return DISCARD_OFFSET + aka_idx
            return DISCARD_OFFSET + action.tile.tile_id

        case ActionType.RIICHI:
            return RIICHI_TOKEN

        case ActionType.TSUMO:
            return TSUMO_AGARI_TOKEN

        case ActionType.RON:
            return RON_AGARI_TOKEN

        case ActionType.CHI:
            # 从消耗牌推断 chi 类型
            assert action.consumed and len(action.consumed) == 2
            t1, t2 = sorted(action.consumed, key=lambda t: t.tile_id)
            target_num = action.tile.number if action.tile else 0
            suit = t1.tile_id // 9
            low  = min(t1.tile_id, t2.tile_id) % 9 + 1  # 1-based
            t_num= target_num
            if   t_num == low + 2: form = 0
            elif t_num == low + 1: form = 1
            else:                  form = 2
            return CHI_OFFSET + chi_type_to_idx(suit, low, form)

        case ActionType.PON:
            return PON_OFFSET + action.tile.tile_id if action.tile else PON_OFFSET

        case ActionType.DAIMINKAN:
            return DAIMINKAN_OFFSET + (action.tile.tile_id if action.tile else 0)

        case ActionType.ANKAN:
            return ANKAN_OFFSET + (action.tile.tile_id if action.tile else 0)

        case ActionType.KAKAN:
            return KAKAN_OFFSET + (action.tile.tile_id if action.tile else 0)

        case ActionType.RYUKYOKU:
            return RYUKYOKU_TOKEN

        case ActionType.PASS:
            return PASS_TOKEN

        case _:
            raise ValueError(f"Unknown action: {action}")


def decode_action(token_id: int, actor: int = 0) -> Action:
    """将 token id 还原为 Action 对象"""
    if DISCARD_OFFSET <= token_id < DISCARD_OFFSET + 37:
        idx = token_id - DISCARD_OFFSET
        if idx < 34:
            tile = Tile(idx)
        else:
            aka_map = {34: Tile(4, True), 35: Tile(13, True), 36: Tile(22, True)}
            tile = aka_map[idx]
        return Action(ActionType.DISCARD, tile=tile, actor=actor)

    if token_id == RIICHI_TOKEN:
        return Action(ActionType.RIICHI, actor=actor)
    if token_id == TSUMO_AGARI_TOKEN:
        return Action(ActionType.TSUMO, actor=actor)
    if token_id == RON_AGARI_TOKEN:
        return Action(ActionType.RON, actor=actor)
    if token_id == RYUKYOKU_TOKEN:
        return Action(ActionType.RYUKYOKU, actor=actor)
    if token_id == PASS_TOKEN:
        return Action(ActionType.PASS, actor=actor)

    if CHI_OFFSET <= token_id < CHI_OFFSET + 90:
        return Action(ActionType.CHI, actor=actor)

    if PON_OFFSET <= token_id < PON_OFFSET + NUM_TILE_TYPES:
        tile = Tile(token_id - PON_OFFSET)
        return Action(ActionType.PON, tile=tile, actor=actor)

    if DAIMINKAN_OFFSET <= token_id < DAIMINKAN_OFFSET + NUM_TILE_TYPES:
        tile = Tile(token_id - DAIMINKAN_OFFSET)
        return Action(ActionType.DAIMINKAN, tile=tile, actor=actor)

    if ANKAN_OFFSET <= token_id < ANKAN_OFFSET + NUM_TILE_TYPES:
        tile = Tile(token_id - ANKAN_OFFSET)
        return Action(ActionType.ANKAN, tile=tile, actor=actor)

    if KAKAN_OFFSET <= token_id < KAKAN_OFFSET + NUM_TILE_TYPES:
        tile = Tile(token_id - KAKAN_OFFSET)
        return Action(ActionType.KAKAN, tile=tile, actor=actor)

    raise ValueError(f"Cannot decode token_id={token_id}")
