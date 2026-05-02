"""
Annotation — 一个决策点的完整标注

每条 Annotation 对应一条训练样本，包含：
  - 游戏元信息（局数、分数、宝牌……）
  - 当前局面（手牌、副露、河牌历史）
  - 候选动作列表 + 玩家实际选择
  - 结果标注（局内分差、终局分差、排名）
  - 辅助任务标签（向听数、放铳风险……）

从 mjai 格式日志解析后填充；GRP 奖励在加载时计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rinshan.tile import Tile


# ─────────────────────────────────────────────
# 辅助任务标签
# ─────────────────────────────────────────────

@dataclass
class AuxTargets:
    shanten:      int            # -1(和) ~ 8，存储时 +1 → 0~9
    tenpai_prob:  float          # 0 / 1
    deal_in_risk: list[float]    # 34 维，每张牌的放铳危险度 0~1
    opp_tenpai:   list[int]      # 3 维，三个对手是否听牌 0/1
    # 三个对手各自的待张 (3, 34) binary；只有对手处于 tenpai 时有意义
    # None 表示本条样本没有该标签（数据源不支持或对手未听牌）
    opp_wait_tiles: Optional[list[list[int]]] = None  # [opp0, opp1, opp2] × 34 binary

    @property
    def shanten_label(self) -> int:
        """向听数 → 分类标签 (0~9)"""
        return max(0, min(9, self.shanten + 1))


# ─────────────────────────────────────────────
# 主数据结构
# ─────────────────────────────────────────────

@dataclass
class Annotation:
    # ── 游戏元信息 ───────────────────────────
    game_id:     str
    player_id:   int          # 0-3，本条标注对应的玩家视角
    round_wind:  int          # 0=东 1=南 2=西
    round_num:   int          # 1-4
    honba:       int          # 本场数
    kyotaku:     int          # 供托数（立直棒）
    scores:      list[int]    # 四家当前分数（绝对值，已旋转使 [0]=己方）
    tiles_left:  int          # 牌山剩余张数

    # ── 局面（公开可见部分）────────────────
    hand:              list[Tile]          # 己方手牌
    dora_indicators:   list[Tile]          # 宝牌指示牌
    discards:          list[list[Tile]]    # 四家河牌（旋转后 [0]=己方）
    melds:             list[list[tuple]]   # 四家副露，每个副露是 (type, tiles)
    riichi_declared:   list[bool]          # 四家是否立直宣言

    # ── 进行序列（历史事件，按时间顺序）────
    progression:   list[int]               # 已编码的进行 token 序列

    # ── 决策 ────────────────────────────────
    action_candidates: list[int]           # 候选动作 token 列表（最多 32 个）
    action_chosen:     int                 # 玩家实际执行的动作（在 candidates 中的 index）

    # ── 结果标注（由 GRP / 后处理填充）─────
    round_delta_score: int    = 0
    final_delta_score: int    = 0
    final_rank:        int    = 0          # 0-3
    grp_reward:        float  = 0.0        # GRP game value 变化（整场价值分支）
    hand_reward:       float  = 0.0        # 局内得失点 shaping（局内价值分支）

    # ── 立直上下文（公开信息，belief 用）─────────────────────────────
    # riichi_discard_tile[i]: 第 i 家立直宣言时打出的那张牌（旋转后 [0]=己方）
    #   None 表示该家未立直或数据不可用
    riichi_discard_tile: list[Optional[Tile]] = field(default_factory=lambda: [None]*4)
    # riichi_junme[i]: 第 i 家立直时已出了多少张牌（即宣言巡目，-1=未立直）
    riichi_junme:        list[int]            = field(default_factory=lambda: [-1]*4)
    # riichi_furiten[i]: 第 i 家是否在振听立直（公开可推断）
    riichi_furiten:      list[bool]           = field(default_factory=lambda: [False]*4)

    # ── 辅助任务标签 ──────────────────────
    aux: Optional[AuxTargets] = None

    # ── Oracle 数据（训练 Stage 2 用）──────
    # 对手真实手牌，推理时不存在，仅在有完整数据时填充
    opponent_hands: Optional[list[list[Tile]]] = None   # [opp0, opp1, opp2] 各自手牌

    # ── next state（IQL 用）───────────────
    next_idx:  Optional[int] = None
    is_done:   bool           = False
