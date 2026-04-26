"""
HoraCalculator — 和牌得点计算器

把 Rinshan 内部的 mjai 格式 (Tile + melds) 翻译成 `mahjong` 包（pip install mahjong）
的 136 格式，计算役种、翻数、符数和得点。

mahjong 包 136 格式约定：
  man: tile_id(0-8)   → 136_idx = tile_id * 4         (赤5m = FIVE_RED_MAN = 16)
  pin: tile_id(9-17)  → 136_idx = (tile_id-9)*4 + 36  (赤5p = FIVE_RED_PIN = 52)
  sou: tile_id(18-26) → 136_idx = (tile_id-18)*4 + 72 (赤5s = FIVE_RED_SOU = 88)
  hon: tile_id(27-33) → 136_idx = (tile_id-27)*4 + 108

每种牌 4 张复本，赤宝牌占第一张（index = base）。

返回值 HoraResult：
  han          : 翻数（含役满时 = 役满倍数 * 13 的等效值，也可从 is_yakuman 判断）
  fu           : 符数
  ron          : 荣和支付点数
  tsumo_ko     : 自摸闲家支付点数
  tsumo_oya    : 自摸庄家支付（子家和了时有效）
  tsumo_total  : 自摸总收入（含庄家/闲家各家）
  is_yakuman   : 是否役满
  yaku_names   : 役种名称列表
  error        : 无役或计算失败的错误信息（None = 正常）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rinshan.tile import Tile


# ─────────────────────────────────────────────
# tile_id (0-33) → mahjong 136 base index
# ─────────────────────────────────────────────

def _tile_id_to_136(tile_id: int) -> int:
    """deaka tile_id → 136 格式基础 index（第一张普通牌）"""
    if tile_id < 9:
        return tile_id * 4
    if tile_id < 18:
        return (tile_id - 9) * 4 + 36
    if tile_id < 27:
        return (tile_id - 18) * 4 + 72
    return (tile_id - 27) * 4 + 108


# 赤宝牌的 136 index
_AKA_136 = {4: 16, 13: 52, 22: 88}   # 5m / 5p / 5s


def _tile_to_136(tile: Tile, used_counts: list[int]) -> int:
    """
    把一张 Tile 分配到具体的 136 复本 index（避免重复分配同一张牌）。
    used_counts: 长度 136 的已用计数数组，会就地更新。
    """
    base = _tile_id_to_136(tile.tile_id)

    if tile.is_aka:
        idx = _AKA_136[tile.tile_id]   # 赤宝牌固定占 base
        if used_counts[idx] > 0:
            # 极少见的同张赤（数据异常），fallback 到普通牌
            pass
        else:
            used_counts[idx] += 1
            return idx

    # 普通牌：从 base+0 到 base+3 依次分配（赤宝牌位跳过）
    aka_idx = _AKA_136.get(tile.tile_id)
    for offset in range(4):
        idx = base + offset
        if aka_idx is not None and idx == aka_idx:
            continue   # 赤宝牌位留给赤牌
        if used_counts[idx] == 0:
            used_counts[idx] += 1
            return idx

    # 理论上不应走到这里
    used_counts[base] += 1
    return base


def _tiles_to_136_list(tiles: list[Tile],
                       used_counts: list[int]) -> list[int]:
    return [_tile_to_136(t, used_counts) for t in tiles]


# ─────────────────────────────────────────────
# 副露 → mahjong Meld 对象
# ─────────────────────────────────────────────

def _meld_to_mahjong(meld_type: str, tiles: list[Tile],
                     used_counts: list[int]):
    """
    把 Rinshan 内部副露 (meld_type, tiles) 转换为 mahjong.Meld。

    Rinshan 副露格式：
      chi:        (meld_type="chi",        [pai, c1, c2])
      pon:        (meld_type="pon",        [pai, c1, c2])
      daiminkan:  (meld_type="daiminkan",  [pai, c1, c2, c3])
      kakan:      (meld_type="kakan",      [pai, c1, c2, c3, extra])
      ankan:      (meld_type="ankan",      [c1, c2, c3, c4])
    """
    from mahjong.meld import Meld

    tile136 = _tiles_to_136_list(tiles, used_counts)

    if meld_type == "chi":
        return Meld(meld_type="chi", tiles=tile136[:3], opened=True)
    if meld_type == "pon":
        return Meld(meld_type="pon", tiles=tile136[:3], opened=True)
    if meld_type in ("daiminkan", "kakan"):
        return Meld(meld_type="kan", tiles=tile136[:4], opened=True)
    if meld_type == "ankan":
        return Meld(meld_type="kan", tiles=tile136[:4], opened=False)

    raise ValueError(f"Unknown meld type: {meld_type}")


# ─────────────────────────────────────────────
# 返回值
# ─────────────────────────────────────────────

@dataclass
class HoraResult:
    han:         int   = 0
    fu:          int   = 0
    ron:         int   = 0
    tsumo_ko:    int   = 0
    tsumo_oya:   int   = 0
    is_yakuman:  bool  = False
    yaku_names:  list[str] = field(default_factory=list)
    error:       Optional[str] = None

    @property
    def tsumo_total(self) -> int:
        """自摸总收入（庄家：tsumo_ko*3，闲家：tsumo_ko*2+tsumo_oya）"""
        # 注意：调用方自己根据是否庄家来选择
        return self.tsumo_oya + self.tsumo_ko * 2   # 子家视角

    def tsumo_total_as(self, is_oya: bool) -> int:
        if is_oya:
            return self.tsumo_ko * 3
        return self.tsumo_ko * 2 + self.tsumo_oya


# ─────────────────────────────────────────────
# 主计算器
# ─────────────────────────────────────────────

class HoraCalculator:
    """
    和牌得点计算器（基于 `mahjong` 包）。

    线程安全（无可变状态），可在多局并行场景中复用同一实例。
    """

    def __init__(self) -> None:
        # 延迟导入，避免 mahjong 包未安装时整个模块加载失败
        from mahjong.hand_calculating.hand import HandCalculator
        from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
        self._impl    = HandCalculator()
        self._options = OptionalRules(has_open_tanyao=True, has_aka_dora=True)

    def calc(
        self,
        hand:         list[Tile],    # 手牌（不含和了牌）
        win_tile:     Tile,          # 和了牌
        melds:        list,          # [(meld_type, [Tile, ...]), ...]
        is_tsumo:     bool,
        is_riichi:    bool,
        is_ippatsu:   bool,
        is_rinshan:   bool,          # 岭上开花
        is_chankan:   bool,          # 抢杠
        is_haitei:    bool,          # 海底摸月
        is_houtei:    bool,          # 河底捞鱼
        is_tenhou:    bool,          # 天和
        is_chiihou:   bool,          # 地和
        dora_indicators:  list[Tile],
        ura_indicators:   list[Tile],
        player_wind:  int,           # 0=东 1=南 2=西 3=北（自风）
        round_wind:   int,           # 0=东 1=南（场风）
    ) -> HoraResult:
        """
        计算完整得点并返回 HoraResult。

        hand 应为 13 张（门前）或按副露减少后的剩余手牌，**不含** win_tile。
        win_tile 单独传入。
        """
        from mahjong.hand_calculating.hand_config import HandConfig

        # ── 用 used_counts 跟踪 136 复本分配 ──────────────
        used_counts = [0] * 136

        # 先处理副露（副露中的牌优先分配复本）
        meld_objs = []
        for mtype, tiles in melds:
            mobj = _meld_to_mahjong(mtype, tiles, used_counts)
            meld_objs.append(mobj)

        # 手牌 (13 张)
        hand_136 = _tiles_to_136_list(hand, used_counts)

        # 和了牌
        win_136 = _tile_to_136(win_tile, used_counts)

        # 手牌 + 和了牌 = 完整 14 张（门前）或 3n+2
        full_tiles = hand_136 + [win_136]

        # ── Dora 计算 ──────────────────────────────────────
        # mahjong 包通过 dora_indicators 参数处理宝牌（包括宝牌指示牌列表）
        dora_ind_136 = [_tile_id_to_136(t.tile_id) for t in dora_indicators]
        ura_ind_136  = [_tile_id_to_136(t.tile_id) for t in ura_indicators]

        # ── 场风/自风 → mahjong 包约定（27=东...30=北）──
        # mahjong 包用 tile_id 直接表示风牌：27=东 28=南 29=西 30=北
        round_wind_tile = 27 + round_wind   # 0→27, 1→28
        player_wind_tile= 27 + player_wind

        # ── HandConfig ─────────────────────────────────────
        config = HandConfig(
            is_tsumo    = is_tsumo,
            is_riichi   = is_riichi,
            is_ippatsu  = is_ippatsu,
            is_rinshan  = is_rinshan,
            is_chankan  = is_chankan,
            is_haitei   = is_haitei,
            is_houtei   = is_houtei,
            is_tenhou   = is_tenhou,
            is_chiihou  = is_chiihou,
            player_wind = player_wind_tile,
            round_wind  = round_wind_tile,
            options     = self._options,
        )

        # ── 调用 mahjong 包 ───────────────────────────────
        result = self._impl.estimate_hand_value(
            tiles            = full_tiles,
            win_tile         = win_136,
            melds            = meld_objs if meld_objs else None,
            config           = config,
            dora_indicators  = dora_ind_136 if dora_ind_136 else None,
            ura_dora_indicators = ura_ind_136 if ura_ind_136 else None,
        )

        if result.error is not None:
            return HoraResult(error=result.error)

        han = result.han
        fu  = result.fu or 30

        # B8 fix: is_oya 应由 player_wind==0（相对庄家位置）判断
        # player_wind=0 表示该玩家是当前局的庄家（东家），与 round_wind 无关
        is_oya = (player_wind == 0)
        point = _calc_point(is_oya, fu, han)

        yaku_names = [str(y) for y in (result.yaku or [])]
        is_yakuman = any("役満" in n or "yakuman" in n.lower()
                         for n in yaku_names) or (han >= 13)

        return HoraResult(
            han        = han,
            fu         = fu,
            ron        = point["ron"],
            tsumo_ko   = point["tsumo_ko"],
            tsumo_oya  = point["tsumo_oya"],
            is_yakuman = is_yakuman,
            yaku_names = yaku_names,
            error      = None,
        )


# ─────────────────────────────────────────────
# 得点查表（直接移植 mortal point.rs，避免依赖 mahjong 包的 cost 格式）
# ─────────────────────────────────────────────

def _calc_point(is_oya: bool, fu: int, han: int) -> dict:
    """
    根据是否庄家、符、翻计算得点。
    满贯以上直接按档次，无需精确 fu。
    返回 {"ron": int, "tsumo_ko": int, "tsumo_oya": int}
    """
    def _round100(x: int) -> int:
        return (x + 99) // 100 * 100

    if han == 0:
        return {"ron": 0, "tsumo_ko": 0, "tsumo_oya": 0}

    # 役满以上
    if han >= 13:
        count = han // 13
        if is_oya:
            return {"ron": 48000*count, "tsumo_ko": 16000*count, "tsumo_oya": 0}
        return {"ron": 32000*count, "tsumo_ko": 8000*count, "tsumo_oya": 16000*count}
    if han >= 11:
        if is_oya:
            return {"ron": 36000, "tsumo_ko": 12000, "tsumo_oya": 0}
        return {"ron": 24000, "tsumo_ko": 6000, "tsumo_oya": 12000}
    if han >= 8:
        if is_oya:
            return {"ron": 24000, "tsumo_ko": 8000, "tsumo_oya": 0}
        return {"ron": 16000, "tsumo_ko": 4000, "tsumo_oya": 8000}
    if han >= 6:
        if is_oya:
            return {"ron": 18000, "tsumo_ko": 6000, "tsumo_oya": 0}
        return {"ron": 12000, "tsumo_ko": 3000, "tsumo_oya": 6000}
    if han >= 5 or (han >= 4 and fu >= 30) or (han >= 3 and fu >= 70):
        # 满贯
        if is_oya:
            return {"ron": 12000, "tsumo_ko": 4000, "tsumo_oya": 0}
        return {"ron": 8000, "tsumo_ko": 2000, "tsumo_oya": 4000}

    # 通常计算：基本点 = fu * 2^(han+2)
    base = fu * (2 ** (han + 2))
    base = min(base, 2000)   # 满贯上限

    if is_oya:
        ron     = _round100(base * 6)
        tsumo_ko = _round100(base * 2)
        return {"ron": ron, "tsumo_ko": tsumo_ko, "tsumo_oya": 0}
    else:
        ron      = _round100(base * 4)
        tsumo_ko = _round100(base * 1)
        tsumo_oya= _round100(base * 2)
        return {"ron": ron, "tsumo_ko": tsumo_ko, "tsumo_oya": tsumo_oya}


# ─────────────────────────────────────────────
# 快速检查：是否有役（不做完整计算）
# ─────────────────────────────────────────────

def has_yaku(
    hand: list[Tile],
    win_tile: Tile,
    melds: list,
    is_tsumo: bool,
    is_riichi: bool,
    player_wind: int,
    round_wind: int,
) -> bool:
    """快速判断手牌是否有役（不做得点计算）"""
    calc = HoraCalculator()
    result = calc.calc(
        hand=hand, win_tile=win_tile, melds=melds,
        is_tsumo=is_tsumo, is_riichi=is_riichi,
        is_ippatsu=False, is_rinshan=False, is_chankan=False,
        is_haitei=False, is_houtei=False,
        is_tenhou=False, is_chiihou=False,
        dora_indicators=[], ura_indicators=[],
        player_wind=player_wind, round_wind=round_wind,
    )
    return result.error is None and result.han > 0
