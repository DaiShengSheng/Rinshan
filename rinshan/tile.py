"""
麻将牌的底层表示
兼容 mjai 协议的字符串格式（1m-9m, 1p-9p, 1s-9s, 1z-7z, 0m/0p/0s 赤宝牌）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# ── tile_id 编码规则 ──────────────────────────
# 0-8   : 1m-9m (man)
# 9-17  : 1p-9p (pin)
# 18-26 : 1s-9s (sou)
# 27    : 1z (東)
# 28    : 2z (南)
# 29    : 3z (西)
# 30    : 4z (北)
# 31    : 5z (白)
# 32    : 6z (發)
# 33    : 7z (中)
#
# 赤宝牌: tile_id 相同（4m=4, 4p=13, 4s=22），用 is_aka 区分

MJAI_TO_ID: dict[str, int] = {
    **{f"{i+1}m": i     for i in range(9)},
    **{f"{i+1}p": i + 9 for i in range(9)},
    **{f"{i+1}s": i +18 for i in range(9)},
    **{f"{i+1}z": i +27 for i in range(7)},
    "0m": 4,   # 赤5m
    "0p": 13,  # 赤5p
    "0s": 22,  # 赤5s
}

ID_TO_MJAI: list[str] = (
    [f"{i+1}m" for i in range(9)] +
    [f"{i+1}p" for i in range(9)] +
    [f"{i+1}s" for i in range(9)] +
    [f"{i+1}z" for i in range(7)]
)

AKA_MJAI = {"0m", "0p", "0s"}
AKA_IDS  = {4, 13, 22}   # deaka 后的 id


@dataclass(frozen=True, slots=True)
class Tile:
    tile_id: int   # 0-33, deaka
    is_aka: bool = False

    # ── 构造 ─────────────────────────────────

    @classmethod
    def from_mjai(cls, mjai_str: str) -> "Tile":
        """从 mjai 字符串构造，如 '5m', '0m', '3z'"""
        is_aka = mjai_str in AKA_MJAI
        tile_id = MJAI_TO_ID[mjai_str]
        return cls(tile_id=tile_id, is_aka=is_aka)

    @classmethod
    def from_id(cls, tile_id: int, is_aka: bool = False) -> "Tile":
        assert 0 <= tile_id < 34, f"Invalid tile_id: {tile_id}"
        if is_aka and tile_id not in AKA_IDS:
            raise ValueError(f"Tile {tile_id} cannot be aka")
        return cls(tile_id=tile_id, is_aka=is_aka)

    # ── 转换 ─────────────────────────────────

    def to_mjai(self) -> str:
        if self.is_aka:
            return {4: "0m", 13: "0p", 22: "0s"}[self.tile_id]
        return ID_TO_MJAI[self.tile_id]

    def deaka(self) -> "Tile":
        """去掉赤宝牌标记"""
        return Tile(tile_id=self.tile_id, is_aka=False)

    def akaize(self) -> "Tile":
        """变为赤宝牌（仅对 5m/5p/5s 有效）"""
        if self.tile_id not in AKA_IDS:
            raise ValueError(f"Tile {self.tile_id} cannot be akaized")
        return Tile(tile_id=self.tile_id, is_aka=True)

    # ── 属性 ─────────────────────────────────

    @property
    def suit(self) -> str:
        """返回花色: 'm' / 'p' / 's' / 'z'"""
        if self.tile_id < 9:  return 'm'
        if self.tile_id < 18: return 'p'
        if self.tile_id < 27: return 's'
        return 'z'

    @property
    def number(self) -> int:
        """返回数字 1-9（字牌 1-7）"""
        return self.tile_id % 9 + 1

    @property
    def is_honor(self) -> bool:
        return self.tile_id >= 27

    @property
    def is_terminal(self) -> bool:
        """幺九牌"""
        return self.is_honor or self.number in (1, 9)

    @property
    def is_yaochuhai(self) -> bool:
        return self.is_terminal  # 同义

    @property
    def next_tile(self) -> Optional["Tile"]:
        """下一张牌（用于向听计算，字牌/9无下一张）"""
        if self.is_honor or self.number == 9:
            return None
        return Tile(tile_id=self.tile_id + 1)

    @property
    def prev_tile(self) -> Optional["Tile"]:
        """上一张牌"""
        if self.is_honor or self.number == 1:
            return None
        return Tile(tile_id=self.tile_id - 1)

    def dora_from_indicator(self) -> "Tile":
        """
        从宝牌指示牌得到实际宝牌
        1z→2z→3z→4z→1z (风牌循环)
        5z→6z→7z→5z (三元牌循环)
        其他：数字+1，9→1
        """
        if self.tile_id == 30: return Tile(27)          # 北→東
        if self.tile_id == 33: return Tile(31)          # 中→白
        if self.is_honor:      return Tile(self.tile_id + 1)
        if self.number == 9:   return Tile(self.tile_id - 8)  # 9→1
        return Tile(self.tile_id + 1)

    # ── 魔法方法 ──────────────────────────────

    def __repr__(self) -> str:
        return f"Tile({self.to_mjai()})"

    def __str__(self) -> str:
        return self.to_mjai()

    def __lt__(self, other: "Tile") -> bool:
        return (self.tile_id, self.is_aka) < (other.tile_id, other.is_aka)


def tiles_from_mjai_list(mjai_list: list[str]) -> list[Tile]:
    return [Tile.from_mjai(s) for s in mjai_list]


def hand_to_counts(hand: list[Tile]) -> list[int]:
    """手牌 → 34 维计数向量（不计赤宝牌标记）"""
    counts = [0] * 34
    for t in hand:
        counts[t.tile_id] += 1
    return counts
