"""
向听数计算（纯 Python 实现）

计算三种和牌形式的向听数取最小值：
  - 通常形（4面子1雀头）
  - 七对子
  - 国士无双

后续可替换为 Rust 实现（libriichi 的 shanten 模块）。

参考：https://github.com/tomohxx/shanten-number-calculator
"""
from __future__ import annotations

INF = 100


from functools import lru_cache
from typing import Iterable


def _pack_counts(counts: Iterable[int]) -> bytes:
    """
    Pack 34 tile counts into bytes for hashing/caching.
    Each count is in [0, 4] in normal Riichi mahjong.
    """
    # Using bytes() is significantly faster than tuple(counts) for caching keys.
    return bytes(int(c) & 0xFF for c in counts)


@lru_cache(maxsize=1_000_000)
def _calc_shanten_cached(packed: bytes, meld_count: int) -> int:
    counts = list(packed)  # local mutable copy for the DFS
    normal = _normal_shanten(counts, meld_count)
    chiitoitsu = _chiitoitsu_shanten(counts, meld_count)
    kokushi = _kokushi_shanten(counts)
    return min(normal, chiitoitsu, kokushi)


def calc_shanten(counts: list[int], meld_count: int = 0) -> int:
    """
    计算向听数

    Args:
        counts:     34 维牌计数向量（deaka）
        meld_count: 已有副露数

    Returns:
        向听数（-1 = 已和）
    """
    # Important: counts is frequently mutated by callers during enumeration.
    # Never cache on the list object itself; cache on a packed snapshot.
    return _calc_shanten_cached(_pack_counts(counts), int(meld_count))


# ─────────────────────────────────────────────
# 通常形向听数
# ─────────────────────────────────────────────

def _normal_shanten(counts: list[int], meld_count: int) -> int:
    """
    标准 3n+2 形式向听数计算
    需要的面子数 = 4 - meld_count
    """
    mentsu_needed = 4 - meld_count
    # 总手牌张数
    total = sum(counts)
    # 3n+2: 手牌数应为 (4 - meld_count)*3 + 2 - meld_count*0 = 14 - 3*meld_count
    # 这里先做一个简化：用已知的 DFS 算法
    return _search_min(_pack_counts(counts), mentsu_needed, 0, False)


@lru_cache(maxsize=2_000_000)
def _search_min(packed: bytes, mentsu_need: int, mentsu_count: int, has_jantai: bool) -> int:
    """
    Memoized DFS for the normal hand shanten.

    This keeps the original semantics of the previous `_search` implementation,
    but avoids recomputing identical sub-states which occur frequently during
    enumeration (e.g., checking many candidate discards / waits).
    """
    counts = list(packed)

    # Current estimate (same as previous code): base from remaining mentsu + jantai
    shanten = (mentsu_need - mentsu_count) * 2 - 1
    if not has_jantai:
        shanten += 1
    shanten -= _count_taatsu(counts, mentsu_need - mentsu_count)

    if shanten <= -1:
        return shanten

    best = shanten

    # Try using each tile position.
    for i in range(34):
        c = counts[i]
        if c == 0:
            continue

        # Pair (jantai)
        if not has_jantai and c >= 2:
            counts[i] = c - 2
            best = min(best, _search_min(_pack_counts(counts), mentsu_need, mentsu_count, True))
            counts[i] = c
            if best <= -1:
                return best

        # Triplet (koutsu)
        if c >= 3 and mentsu_count < mentsu_need:
            counts[i] = c - 3
            best = min(best, _search_min(_pack_counts(counts), mentsu_need, mentsu_count + 1, has_jantai))
            counts[i] = c
            if best <= -1:
                return best

        # Sequence (shuntsu)
        if mentsu_count < mentsu_need and i < 27 and (i % 9) <= 6:
            if counts[i + 1] > 0 and counts[i + 2] > 0:
                counts[i] -= 1
                counts[i + 1] -= 1
                counts[i + 2] -= 1
                best = min(best, _search_min(_pack_counts(counts), mentsu_need, mentsu_count + 1, has_jantai))
                counts[i] += 1
                counts[i + 1] += 1
                counts[i + 2] += 1
                if best <= -1:
                    return best

    return best


def clear_shanten_caches() -> None:
    """Clear internal caches (useful for benchmarks / long self-play runs)."""
    _calc_shanten_cached.cache_clear()
    _search_min.cache_clear()


def _count_taatsu(counts: list[int], max_taatsu: int) -> int:
    """计算搭子数（不完整面子），最多 max_taatsu 个"""
    taatsu = 0
    for i in range(34):
        if counts[i] == 0:
            continue
        if i < 27 and i % 9 <= 7 and counts[i+1] > 0:
            taatsu += 1
        if i < 27 and i % 9 <= 6 and counts[i+2] > 0:
            taatsu += 1
    return min(taatsu, max_taatsu)


# ─────────────────────────────────────────────
# 七对子向听数
# ─────────────────────────────────────────────

def _chiitoitsu_shanten(counts: list[int], meld_count: int = 0) -> int:
    # B2 fix: chiitoitsu requires a closed hand (no melds)
    if meld_count > 0:
        return INF
    # pairs: number of tile types where we hold >= 2 (each type contributes at most 1 pair)
    pairs = sum(1 for c in counts if c >= 2)
    # kinds: number of distinct tile types held
    kinds = sum(1 for c in counts if c >= 1)
    # Need 7 unique pairs.
    # pairs_still_needed = 7 - pairs
    # kinds_still_needed = 7 - kinds  (each missing kind requires drawing a new type AND making a pair)
    # shanten = max(pairs_still_needed, kinds_still_needed) - 1
    return max(7 - pairs, 7 - kinds) - 1


# ─────────────────────────────────────────────
# 国士无双向听数
# ─────────────────────────────────────────────

_YAOCHUHAI_IDS = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]

def _kokushi_shanten(counts: list[int]) -> int:
    # B_kokushi fix: 正确公式为 13 - has_kinds - has_pair
    # 13幺九全有+对子 = shanten -1（已和）
    # 13幺九全有无对 = shanten 0（听牌，任意幺九牌）
    has_pair  = any(counts[i] >= 2 for i in _YAOCHUHAI_IDS)
    has_kinds = sum(1 for i in _YAOCHUHAI_IDS if counts[i] >= 1)
    return 13 - has_kinds - (1 if has_pair else 0)
