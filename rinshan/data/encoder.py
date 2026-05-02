"""
GameEncoder — 将 Annotation 转换为模型输入 tensor

输出：
  tokens         : (S,)   int — 主序列（META + DORA + HAND + MELD + PROG + CAND）
  candidate_mask : (32,)  bool
  pad_mask       : (S,)   bool  True=PAD
  belief_tokens  : (S',)  int   — 仅含公开信息（不含己方手牌），给 Belief Net 用
  belief_pad_mask: (S',)  bool

  以及可选的：
  actual_hands   : (34, 3) int  — 三个对手各牌张数（Oracle 训练用）
  aux_targets    : dict
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from rinshan.constants import (
    # 普通 token
    TILE_OFFSET, AKA_OFFSET, DISCARD_OFFSET,
    RIICHI_TOKEN, TSUMO_AGARI_TOKEN, RON_AGARI_TOKEN, RYUKYOKU_TOKEN, PASS_TOKEN,
    WIND_OFFSET, ROUND_OFFSET, GAME_START_TOKEN, ROUND_START_TOKEN, PAD_TOKEN,
    HONBA_OFFSET, KYOTAKU_OFFSET, TILES_OFFSET,
    # 立直上下文 token
    RIICHI_JUNME_OFFSET, RIICHI_FURITEN_OFFSET,
    # 进行 token
    PROG_DISCARD_BASE, PROG_DISCARD_TSUMOGIRI_BASE,
    PROG_DRAW_BASE, PROG_RIICHI_BASE,
    PROG_CHI_BASE, PROG_PON_BASE, PROG_DAIMINKAN_BASE,
    PROG_ANKAN_BASE, PROG_KAKAN_BASE, PROG_NEWDORA_BASE,
    # 序列长度
    MAX_GAME_META_LEN, MAX_DORA_LEN, MAX_HAND_LEN, MAX_MELD_LEN,
    MAX_PROGRESSION_LEN, MAX_CANDIDATES_LEN, MAX_SEQ_LEN,
    MAX_OPP_HAND_LEN, MAX_ORACLE_SEQ_LEN,
    NUM_TILE_TYPES, VOCAB_SIZE,
)
from rinshan.tile import Tile
from .annotation import Annotation

# ─────────────────────────────────────────────
# RBF 编码（处理分数等数值特征）
# ─────────────────────────────────────────────
# 分数范围大约 -60000 ~ +90000，用 8 个 RBF 中心
_SCORE_CENTERS = np.array([-30000, -10000, 0, 10000, 20000, 30000, 50000, 70000],
                           dtype=np.float32)
_SCORE_SIGMA   = 15000.0

def rbf_encode_score(score: int) -> list[float]:
    x = float(score)
    return [
        float(np.exp(-((x - c) ** 2) / (2 * _SCORE_SIGMA ** 2)))
        for c in _SCORE_CENTERS
    ]


# ─────────────────────────────────────────────
# タイル → token id
# ─────────────────────────────────────────────

def tile_to_token(tile: Tile) -> int:
    if tile.is_aka:
        aka_map = {4: 34, 13: 35, 22: 36}
        return aka_map[tile.tile_id]
    return TILE_OFFSET + tile.tile_id

def tile_to_discard_token(tile: Tile) -> int:
    """打牌动作 token"""
    base_id = tile_to_token(tile)
    # discard: 同样的 token 范围
    return DISCARD_OFFSET + (tile.tile_id if not tile.is_aka else {4:0, 13:1, 22:2}[tile.tile_id] + 34)


# ─────────────────────────────────────────────
# GameEncoder
# ─────────────────────────────────────────────

class GameEncoder:
    """
    Annotation → (tokens, candidate_mask, pad_mask, belief_tokens, ...)

    所有序列都 **右填充** PAD_TOKEN 到固定长度。
    """

    def encode(self, ann: Annotation) -> dict:
        tokens_list: list[int] = []

        # ─── 1. GAME META ──────────────────────────────────────────
        # ROUND_START token
        tokens_list.append(ROUND_START_TOKEN)
        # 场风
        tokens_list.append(WIND_OFFSET + ann.round_wind)
        # 局数 (0-based → token)
        tokens_list.append(ROUND_OFFSET + ann.round_num - 1)
        # 本场 & 供托: use dedicated offset ranges that don't overlap with any
        # other token (HONBA_OFFSET, KYOTAKU_OFFSET, TILES_OFFSET, all >= 1337).
        honba_tok   = HONBA_OFFSET   + min(ann.honba,   8)
        kyotaku_tok = KYOTAKU_OFFSET + min(ann.kyotaku, 4)
        tokens_list.append(honba_tok)
        tokens_list.append(kyotaku_tok)
        # 剩余张数（0-70 → 0-8 bins）— dedicated offset, no collision
        tiles_bin = min(ann.tiles_left // 9, 8)
        tokens_list.append(TILES_OFFSET + tiles_bin)
        # 暂不编码分数 RBF（需要数值 token 扩展，后续在 v2 加入）
        # 补 PAD 到 MAX_GAME_META_LEN
        while len(tokens_list) < MAX_GAME_META_LEN:
            tokens_list.append(PAD_TOKEN)

        # ─── 2. DORA ──────────────────────────────────────────────
        dora_start = len(tokens_list)
        for tile in ann.dora_indicators[:MAX_DORA_LEN]:
            tokens_list.append(tile_to_token(tile))
        while len(tokens_list) < dora_start + MAX_DORA_LEN:
            tokens_list.append(PAD_TOKEN)

        # ─── 3. HAND ──────────────────────────────────────────────
        hand_start = len(tokens_list)
        for tile in sorted(ann.hand)[:MAX_HAND_LEN]:
            tokens_list.append(tile_to_token(tile))
        while len(tokens_list) < hand_start + MAX_HAND_LEN:
            tokens_list.append(PAD_TOKEN)

        # ─── 4. MELDS ────────────────────────────────────────────
        meld_start = len(tokens_list)
        own_melds = ann.melds[0] if ann.melds else []
        for meld in own_melds[:4]:   # 最多 4 组副露
            mtype, tiles = meld[0], meld[1]
            for tile in tiles[:4]:
                tokens_list.append(tile_to_token(tile))
        while len(tokens_list) < meld_start + MAX_MELD_LEN:
            tokens_list.append(PAD_TOKEN)

        # ─── 5. PROGRESSION ──────────────────────────────────────
        prog_start = len(tokens_list)
        prog_tokens = ann.progression[-MAX_PROGRESSION_LEN:]  # 保留最近的
        tokens_list.extend(prog_tokens)
        while len(tokens_list) < prog_start + MAX_PROGRESSION_LEN:
            tokens_list.append(PAD_TOKEN)

        # ─── 6. CANDIDATES ────────────────────────────────────────
        cand_start = len(tokens_list)
        cands = ann.action_candidates[:MAX_CANDIDATES_LEN]
        tokens_list.extend(cands)
        n_real_cands = len(cands)
        while len(tokens_list) < cand_start + MAX_CANDIDATES_LEN:
            tokens_list.append(PAD_TOKEN)

        # ── pad_mask ──
        total_len = len(tokens_list)
        pad_mask = [tok == PAD_TOKEN for tok in tokens_list]

        # ── candidate_mask ──
        candidate_mask = [False] * MAX_CANDIDATES_LEN
        for i in range(n_real_cands):
            candidate_mask[i] = True

        # ─── Belief tokens（公开信息：不含己方手牌）──────────────
        belief_tokens_list: list[int] = []
        # META + DORA（直接复用上面的）
        belief_tokens_list.extend(tokens_list[:dora_start + MAX_DORA_LEN])
        # 四家副露（公开信息）
        for seat in range(4):
            seat_melds = ann.melds[seat] if seat < len(ann.melds) else []
            for meld in seat_melds[:4]:
                for tile in meld[1][:4]:
                    belief_tokens_list.append(tile_to_token(tile))
        # 四家立直状态（简单 0/1 flag）
        for rch in ann.riichi_declared:
            belief_tokens_list.append(RIICHI_TOKEN if rch else PAD_TOKEN)
        # 立直上下文：宣言牌 + 宣言巡目 + 振听（每家最多 3 个 token）
        riichi_discard_tiles = getattr(ann, 'riichi_discard_tile', [None]*4)
        riichi_junmes        = getattr(ann, 'riichi_junme',         [-1]*4)
        riichi_furitens      = getattr(ann, 'riichi_furiten',       [False]*4)
        for seat in range(4):
            # 宣言牌（已立直且有记录时）
            rdtile = riichi_discard_tiles[seat] if riichi_discard_tiles else None
            if rdtile is not None and ann.riichi_declared[seat]:
                belief_tokens_list.append(tile_to_token(rdtile))
            else:
                belief_tokens_list.append(PAD_TOKEN)
            # 宣言巡目分桶：-1=未立直 → PAD；0~8 bin → RIICHI_JUNME_OFFSET + seat*9 + bin
            junme = riichi_junmes[seat] if riichi_junmes else -1
            if junme >= 0 and ann.riichi_declared[seat]:
                jbin = min(junme // 2, 8)   # 每 2 巡一档，共 9 档
                belief_tokens_list.append(RIICHI_JUNME_OFFSET + seat * 9 + jbin)
            else:
                belief_tokens_list.append(PAD_TOKEN)
            # 振听 flag
            is_furiten = riichi_furitens[seat] if riichi_furitens else False
            if ann.riichi_declared[seat] and is_furiten:
                belief_tokens_list.append(RIICHI_FURITEN_OFFSET + seat)
            else:
                belief_tokens_list.append(PAD_TOKEN)
        # 进行序列（公开，和主序列共享）
        belief_tokens_list.extend(prog_tokens)
        # 截断 + 填充（+12 for 立直上下文 4家×3 token）
        max_belief_len = MAX_PROGRESSION_LEN + 20 + 12
        belief_tokens_list = belief_tokens_list[:max_belief_len]
        belief_pad_mask_list = [tok == PAD_TOKEN for tok in belief_tokens_list]
        while len(belief_tokens_list) < max_belief_len:
            belief_tokens_list.append(PAD_TOKEN)
            belief_pad_mask_list.append(True)

        # ─── actual_hands（Oracle 用）──────────────────────────────
        actual_hands = None
        if ann.opponent_hands is not None:
            actual_hands = np.zeros((NUM_TILE_TYPES, 3), dtype=np.int8)
            for opp_idx, hand in enumerate(ann.opponent_hands[:3]):
                for tile in hand:
                    actual_hands[tile.tile_id, opp_idx] += 1

        # ─── aux_targets ──────────────────────────────────────────
        aux_targets = {}
        if ann.aux is not None:
            a = ann.aux
            aux_targets = {
                "shanten":      a.shanten_label,
                "tenpai_prob":  float(a.tenpai_prob),
                "deal_in_risk": a.deal_in_risk,
                "opp_tenpai":   [float(x) for x in a.opp_tenpai],
            }
            # 待张标签：(34, 3) float binary，只有对手处于 tenpai 时有意义
            if a.opp_wait_tiles is not None:
                wait_arr = np.zeros((NUM_TILE_TYPES, 3), dtype=np.float32)
                for opp_idx, wait_list in enumerate(a.opp_wait_tiles[:3]):
                    for tid in wait_list:
                        if 0 <= tid < NUM_TILE_TYPES:
                            wait_arr[tid, opp_idx] = 1.0
                aux_targets["opp_wait_tiles"] = wait_arr
                # tenpai mask：(3,) float——只对 tenpai 对手计算 wait loss
                aux_targets["opp_tenpai_mask"] = np.array(
                    [float(x) for x in a.opp_tenpai], dtype=np.float32
                )

        return {
            "tokens":          torch.tensor(tokens_list,          dtype=torch.long),
            "pad_mask":        torch.tensor(pad_mask,             dtype=torch.bool),
            "candidate_mask":  torch.tensor(candidate_mask,       dtype=torch.bool),
            "belief_tokens":   torch.tensor(belief_tokens_list,   dtype=torch.long),
            "belief_pad_mask": torch.tensor(belief_pad_mask_list, dtype=torch.bool),
            "action_idx":      torch.tensor(ann.action_chosen,    dtype=torch.long),
            "reward":          torch.tensor(ann.grp_reward,       dtype=torch.float32),
            "reward_game":     torch.tensor(ann.grp_reward,       dtype=torch.float32),
            "reward_hand":     torch.tensor(ann.hand_reward,      dtype=torch.float32),
            "is_done":         torch.tensor(ann.is_done,          dtype=torch.bool),
            # Oracle
            "actual_hands":    torch.tensor(actual_hands, dtype=torch.float32)
                                if actual_hands is not None else None,
            # AUX
            "aux_targets":     aux_targets,
            # 元信息（不入模型，调试用）
            "game_id":         ann.game_id,
            "player_id":       ann.player_id,
        }

    def encode_oracle(self, ann: Annotation) -> dict:
        """
        Stage 2 专用：在标准序列的 HAND 段之后追加三家对手手牌 token。

        Oracle 模型是全知视角，拥有对手手牌信息，序列结构：
          [META][DORA][HAND(己方)][OPP_HAND_0][OPP_HAND_1][OPP_HAND_2]
          [MELD][PROGRESSION][CANDIDATES]

        序列长度上限 MAX_ORACLE_SEQ_LEN。
        如果 ann.opponent_hands 为 None（例如旧数据），
        退化为全 PAD（等同于标准 encode 加了一段空占位）。
        """
        # 先获取标准编码的中间产物（复用 encode 的构建逻辑）
        base = self.encode(ann)

        # 把标准 token 序列拆分，在 HAND 段后插入对手手牌
        # 段偏移（与 encode() 里的构建顺序一一对应，使用常量而非硬编码）
        hand_end = MAX_GAME_META_LEN + MAX_DORA_LEN + MAX_HAND_LEN   # HAND 段结束位置

        std_tokens = base["tokens"].tolist()
        prefix   = std_tokens[:hand_end]                    # META+DORA+HAND
        suffix   = std_tokens[hand_end:]                    # MELD+PROG+CAND

        # 构建三家对手手牌 token 段
        opp_tokens: list[int] = []
        if ann.opponent_hands is not None:
            for opp_hand in ann.opponent_hands[:3]:         # 最多 3 家
                for tile in sorted(opp_hand)[:MAX_HAND_LEN]:
                    opp_tokens.append(tile_to_token(tile))
        # 填充到固定长度
        while len(opp_tokens) < MAX_OPP_HAND_LEN:
            opp_tokens.append(PAD_TOKEN)

        oracle_tokens_list = prefix + opp_tokens + suffix

        # 截断 + 右填充到 MAX_ORACLE_SEQ_LEN
        oracle_tokens_list = oracle_tokens_list[:MAX_ORACLE_SEQ_LEN]
        while len(oracle_tokens_list) < MAX_ORACLE_SEQ_LEN:
            oracle_tokens_list.append(PAD_TOKEN)

        oracle_pad_mask = [tok == PAD_TOKEN for tok in oracle_tokens_list]

        return {
            **base,
            "oracle_tokens":   torch.tensor(oracle_tokens_list, dtype=torch.long),
            "oracle_pad_mask": torch.tensor(oracle_pad_mask,    dtype=torch.bool),
        }
