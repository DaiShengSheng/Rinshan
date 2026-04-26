"""
GameState / PlayerView — 游戏状态表示

GameState: 完整全局状态（服务器视角，含所有玩家手牌）
PlayerView: 单玩家视角（隐藏对手手牌）

这两个类是 MjaiSimulator 的输出，同时也是 Annotation 的数据来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rinshan.tile import Tile, hand_to_counts


# ─────────────────────────────────────────────
# 合法动作候选
# ─────────────────────────────────────────────

@dataclass
class ActionCandidate:
    can_discard:    bool = False
    can_riichi:     bool = False
    can_tsumo:      bool = False
    can_ron:        bool = False
    can_chi_low:    bool = False
    can_chi_mid:    bool = False
    can_chi_high:   bool = False
    can_pon:        bool = False
    can_daiminkan:  bool = False
    can_ankan:      bool = False
    can_kakan:      bool = False
    can_ryukyoku:   bool = False
    can_pass:       bool = False

    # 鸣牌目标
    target_actor:   int  = -1
    discard_candidates: list[int] = field(default_factory=list)   # 合法打牌的 tile_id


# ─────────────────────────────────────────────
# 全局游戏状态（服务器视角）
# ─────────────────────────────────────────────

@dataclass
class GameState:
    # ── 场面信息 ─────────────────────────────
    round_wind:  int   = 0     # 0=东 1=南 2=西
    round_num:   int   = 1     # 1-4
    honba:       int   = 0
    kyotaku:     int   = 0
    dealer:      int   = 0     # 庄家座位

    scores:      list[int] = field(default_factory=lambda: [25000]*4)
    tiles_left:  int   = 70

    dora_indicators: list[Tile] = field(default_factory=list)

    # ── 四家状态 ─────────────────────────────
    # 注意：必须用 field(default_factory=...) 确保每个实例独立分配，
    # 不能用 [[]]*4（浅拷贝会导致四家共享同一 list 对象）
    hands:       list[list[Tile]] = field(default_factory=lambda: [[], [], [], []])
    discards:    list[list[Tile]] = field(default_factory=lambda: [[], [], [], []])
    melds:       list[list]       = field(default_factory=lambda: [[], [], [], []])

    riichi_declared: list[bool]  = field(default_factory=lambda: [False, False, False, False])
    riichi_accepted: list[bool]  = field(default_factory=lambda: [False, False, False, False])
    in_riichi:       list[bool]  = field(default_factory=lambda: [False, False, False, False])
    # 各家立直时的弃牌数（立直牌在 discards[seat][riichi_discard_idx[seat]] 处）
    # -1 表示未立直
    riichi_discard_idx: list[int] = field(default_factory=lambda: [-1, -1, -1, -1])
    # 振听（furiten）: True = cannot ron
    # Simplified: only tracks permanent furiten (player discarded a tile in their own waits).
    # Does not track temporary (same-round) furiten; that would require full wait calculation.
    furiten: list[bool] = field(default_factory=lambda: [False, False, False, False])

    # ── 当前动作玩家 ──────────────────────────
    current_player: int = 0
    last_discard:   Optional[Tile] = None
    last_draw:      Optional[Tile] = None

    # ── 进行 token 序列（公开事件，持续追加）─
    progression: list[int] = field(default_factory=list)

    def player_view(self, seat: int) -> "PlayerView":
        """生成指定座位的玩家视角（隐藏对手手牌）"""
        # 旋转：seat=0 → 自己
        def rot(lst: list, n: int):
            return lst[n:] + lst[:n]

        return PlayerView(
            player_id       = seat,
            round_wind      = self.round_wind,
            round_num       = self.round_num,
            honba           = self.honba,
            kyotaku         = self.kyotaku,
            scores          = rot(self.scores, seat),
            tiles_left      = self.tiles_left,
            dora_indicators = list(self.dora_indicators),
            hand            = list(self.hands[seat]),
            discards        = rot(self.discards, seat),
            melds           = rot(self.melds, seat),
            riichi_declared = rot(self.riichi_declared, seat),
            in_riichi       = rot(self.in_riichi, seat),
            last_discard    = self.last_discard,
            last_draw       = self.last_draw,
            progression     = list(self.progression),
        )

    def shanten(self, seat: int) -> int:
        """计算指定玩家的向听数（简化版，依赖 algo 模块）"""
        from rinshan.algo.shanten import calc_shanten
        counts = hand_to_counts(self.hands[seat])
        melds  = len(self.melds[seat])
        return calc_shanten(counts, melds)


# ─────────────────────────────────────────────
# 玩家视角（训练数据来源）
# ─────────────────────────────────────────────

@dataclass
class PlayerView:
    player_id:   int
    round_wind:  int
    round_num:   int
    honba:       int
    kyotaku:     int
    scores:      list[int]     # 旋转后，[0] = 己方
    tiles_left:  int

    dora_indicators: list[Tile]
    hand:            list[Tile]
    discards:        list[list[Tile]]
    melds:           list[list]
    riichi_declared: list[bool]
    in_riichi:       list[bool]

    last_discard:    Optional[Tile]
    last_draw:       Optional[Tile]
    progression:     list[int]   # 已编码的进行 token 序列
