"""
全局常量定义
覆盖天凤 / 雀魂四麻规则
"""

# ─────────────────────────────────────────────
# 牌面常量
# ─────────────────────────────────────────────

NUM_TILE_TYPES   = 34    # 0-8: 1-9m  9-17: 1-9p  18-26: 1-9s  27-33: 字牌
NUM_TILE_COPIES  = 4     # 每种牌 4 张

# 赤宝牌 tile_id（deaka 后的普通牌 id 相同，以 bool 标记是否 aka）
AKA_TILE_IDS = [4, 13, 22]   # 5m, 5p, 5s

# 字牌范围
HONOR_START = 27
HONOR_END   = 33   # inclusive

# ─────────────────────────────────────────────
# Token 词表设计
# ─────────────────────────────────────────────
# Token 空间分配（所有 token id 都在此命名）：
#
#   [0,   33]   普通牌 (deaka)         34 种
#   [34,  36]   赤宝牌 5m/5p/5s         3 种
#   [37,  70]   DISCARD_*              34 种  (打出某牌)
#   [71, 104]   TSUMO_*                34 种  (摸某牌)     —— 仅用于 Progression
#   [105,195]   CHI_*                  90 种  (吃，90种搭子)
#   [196,315]   PON_*                 120 种  (碰，34种牌×对手方向3+赤0/1)  — 简化为34×4=136…取120
#   [316,430]   DAIMINKAN_*           114 种
#   [431,464]   ANKAN_*                34 种
#   [465,498]   KAKAN_*                34 种
#   [499]       RIICHI
#   [500]       TSUMO_AGARI  (自摸)
#   [501]       RON_AGARI    (荣和)
#   [502]       RYUKYOKU     (流局宣告)
#   [503]       PASS         (pass / 无动作)
#   [504,507]   WIND_EAST/SOUTH/WEST/NORTH  (场风/自风标记用)
#   [508,511]   ROUND_1/2/3/4
#   [512]       GAME_START
#   [513]       ROUND_START
#   [514]       PAD          (padding)
#
# 总词表大小：515
#
# ── Progression 区（从 513 开始）──
#   [513, 660]  PROG_DISCARD（手切）   4 seats × 37 = 148
#   [661, 664]  PROG_DRAW              4 seats
#   [665, 668]  PROG_RIICHI            4 seats
#   [669, 758]  PROG_CHI               90 types
#   [759, 894]  PROG_PON               4 × 34 = 136
#   [895,1030]  PROG_DAIMINKAN         4 × 34 = 136
#  [1031,1166]  PROG_ANKAN             4 × 34 = 136
#  [1167,1302]  PROG_KAKAN             4 × 34 = 136
#  [1303,1336]  PROG_NEWDORA           34
#  [1337,1345]  HONBA_OFFSET           0-8
#  [1346,1350]  KYOTAKU_OFFSET         0-4
#  [1351,1359]  TILES_OFFSET           0-8
#  [1360,1507]  PROG_DISCARD_TSUMOGIRI（摸切）  4 seats × 37 = 148
#
# 总词表大小：1508

# 偏移量
TILE_OFFSET      = 0     # 普通牌 0-33
AKA_OFFSET       = 34    # 赤宝牌 34-36
DISCARD_OFFSET   = 37    # 打牌动作 37-70
TSUMO_OFFSET     = 71    # 摸牌事件 71-104  (progression 用)
CHI_OFFSET       = 105   # 吃 105-194
PON_OFFSET       = 195   # 碰 195-314  (34种 × ~4方向简化)
DAIMINKAN_OFFSET = 315   # 大明杠 315-428
ANKAN_OFFSET     = 429   # 暗杠 429-462
KAKAN_OFFSET     = 463   # 加杠 463-496

RIICHI_TOKEN     = 497
TSUMO_AGARI_TOKEN= 498
RON_AGARI_TOKEN  = 499
RYUKYOKU_TOKEN   = 500
PASS_TOKEN       = 501

WIND_OFFSET      = 502   # 502-505: East/South/West/North
ROUND_OFFSET     = 506   # 506-509: Round 1-4
GAME_START_TOKEN = 510
ROUND_START_TOKEN= 511
PAD_TOKEN        = 512

# ── Meta tokens for scalar game state (honba / kyotaku / tiles_left) ──
# These must NOT overlap with PAD_TOKEN, PROG_* tokens, or any action token.
# We allocate them AFTER the progression space ends (PROG_NEWDORA_BASE+34=1337).
HONBA_OFFSET     = 1337  # 1337-1345: honba 0-8 (bins)
KYOTAKU_OFFSET   = 1346  # 1346-1350: kyotaku 0-4
TILES_OFFSET     = 1351  # 1351-1359: tiles_left bins 0-8

# ─────────────────────────────────────────────
# 进行（Progression）Token 空间
# 从 513 开始，追加在动作空间之后
# ─────────────────────────────────────────────
# 编码规则：
#   打牌事件（手切）: PROG_DISCARD_BASE           + seat*37 + tile_token_idx
#   打牌事件（摸切）: PROG_DISCARD_TSUMOGIRI_BASE  + seat*37 + tile_token_idx
#   摸牌事件        : PROG_DRAW_BASE + seat
#   立直事件  : PROG_RIICHI_BASE + seat
#   吃         : PROG_CHI_BASE + chi_type (0-89)
#   碰         : PROG_PON_BASE + seat*34 + tile_id
#   大明杠     : PROG_DAIMINKAN_BASE + seat*34 + tile_id
#   暗杠       : PROG_ANKAN_BASE + seat*34 + tile_id
#   加杠       : PROG_KAKAN_BASE + seat*34 + tile_id
#   新宝牌翻开 : PROG_NEWDORA_BASE + tile_id

PROG_DISCARD_BASE   = 513    # 513 - 660  (4 seats × 37 tiles = 148)
PROG_DRAW_BASE      = 661    # 661 - 664  (4 seats)
PROG_RIICHI_BASE    = 665    # 665 - 668  (4 seats)
PROG_CHI_BASE       = 669    # 669 - 758  (90 types)
PROG_PON_BASE       = 759    # 759 - 894  (4 × 34 = 136)
PROG_DAIMINKAN_BASE = 895    # 895 - 1030 (4 × 34 = 136)
PROG_ANKAN_BASE     = 1031   # 1031 - 1166 (4 × 34 = 136)
PROG_KAKAN_BASE     = 1167   # 1167 - 1302 (4 × 34 = 136)
PROG_NEWDORA_BASE   = 1303   # 1303 - 1336 (34 tiles)

# PROG_DISCARD 摸切版：与手切版 token 空间完全对称，追加在词表末尾
# 避免插入中间导致所有 PROG_* offset 连锁偏移
PROG_DISCARD_TSUMOGIRI_BASE = 1360  # 1360-1507: 摸切打牌（4 seats × 37 = 148）

VOCAB_SIZE = 1508  # +148 摸切 PROG_DISCARD token

# ─────────────────────────────────────────────
# 序列长度常量
# ─────────────────────────────────────────────
MAX_GAME_META_LEN       =  16   # 场风/局数/本场/供托/四家分数(RBF) 等
MAX_DORA_LEN            =   5   # 最多5张宝牌指示牌
MAX_HAND_LEN            =  14   # 手牌最多14张
MAX_MELD_LEN            =  16   # 副露最多4组 × 4张
MAX_PROGRESSION_LEN     = 113   # 历史打牌/鸣牌序列（kanachan 实测值）
MAX_CANDIDATES_LEN      =  32   # 合法候选动作

MAX_SEQ_LEN = (
    MAX_GAME_META_LEN
    + MAX_DORA_LEN
    + MAX_HAND_LEN
    + MAX_MELD_LEN
    + MAX_PROGRESSION_LEN
    + MAX_CANDIDATES_LEN
    + 4   # [CLS] + 分隔符 × 3
)
# ≈ 200，保险起见设 256

# Oracle 序列长度（Stage 2 用：在主序列 HAND 之后插入三家对手手牌）
MAX_OPP_HAND_LEN   = MAX_HAND_LEN * 3        # 3 opponents × 14 = 42
MAX_ORACLE_SEQ_LEN = MAX_SEQ_LEN + MAX_OPP_HAND_LEN  # ≈ 242

# ─────────────────────────────────────────────
# 动作空间常量（Action Space）
# ─────────────────────────────────────────────
NUM_DISCARD_ACTIONS  =  37   # 34 普通 + 3 赤宝牌
NUM_CHI_ACTIONS      =  90
NUM_PON_ACTIONS      =  34
NUM_KAN_ACTIONS      =  34   # 暗杠/大明杠统一34种
NUM_KAKAN_ACTIONS    =  34
NUM_SPECIAL_ACTIONS  =   5   # riichi / tsumo / ron / ryukyoku / pass

ACTION_SPACE = (
    NUM_DISCARD_ACTIONS
    + NUM_CHI_ACTIONS
    + NUM_PON_ACTIONS
    + NUM_KAN_ACTIONS
    + NUM_KAKAN_ACTIONS
    + NUM_SPECIAL_ACTIONS
)
# = 234  (比 Mortal 的 46 大，因为细化了吃碰等)

# ─────────────────────────────────────────────
# Belief Network 常量
# ─────────────────────────────────────────────
BELIEF_DIM      = 256
BELIEF_LAYERS   =   4
BELIEF_HEADS    =   4

# 输出：34 种牌 × 3 个对手 = 102 维的概率矩阵
BELIEF_OUTPUT_DIM = NUM_TILE_TYPES * 3

# ─────────────────────────────────────────────
# 模型规模预设
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "nano": dict(
        dim=256, n_heads=4, n_kv_heads=2, n_layers=4,  ffn_dim=512,
    ),
    "base": dict(
        dim=768, n_heads=12, n_kv_heads=4, n_layers=12, ffn_dim=2048,
    ),
    "large": dict(
        dim=1536, n_heads=16, n_kv_heads=4, n_layers=24, ffn_dim=4096,
    ),
}

# ─────────────────────────────────────────────
# 训练超参默认值
# ─────────────────────────────────────────────
GAMMA            = 0.99    # 折扣因子
IQL_EXPECTILE    = 0.9     # Expectile τ，让 V(s) 追踪优秀动作
CQL_WEIGHT       = 0.5     # 保守约束权重
TARGET_EMA_TAU   = 0.005   # 目标网络 EMA 更新率
DISTILL_TEMP     = 2.0     # Oracle 蒸馏温度

# 推理温度（用于混合策略采样）
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P       = 0.9

# ─────────────────────────────────────────────
# 辅助任务权重
# ─────────────────────────────────────────────
AUX_WEIGHTS = {
    "shanten":      0.10,
    "tenpai_prob":  0.10,
    "deal_in_risk": 0.30,  # Label B（对手待张集合）; 由 calibrate_aux_weights.py 在3年天凤数据上统计推算
                           # baseline对齐=0.45, random-init对齐=0.18, 取几何均值√(0.45×0.18)≈0.28→保守取0.30
    "opp_tenpai":   0.10,
}

# ─────────────────────────────────────────────
# GRP 常量
# ─────────────────────────────────────────────
GRP_INPUT_DIM   = 7     # [局数, 本场, 供托, 四家分数]
GRP_HIDDEN_SIZE = 128
GRP_NUM_LAYERS  = 3
GRP_OUTPUT_DIM  = 24    # 4! 种排名组合

# 标准段位点换算（天凤凤凰桌）
RANK_PTS_TENHOU = [90.0, 45.0, 0.0, -135.0]
# 标准段位点换算（雀魂玉之间/王座）
RANK_PTS_JANTAMA = [80.0, 40.0, 0.0, -120.0]
