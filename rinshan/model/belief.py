"""
Belief Network — 从公开信息推断对手手牌概率分布

输入：公开可见的游戏状态（河牌、副露、宝牌、剩余张数等）
输出：(B, 34, 3) 概率矩阵，belief[b][tile_id][opp] = 该牌在对手 opp 手里的概率

设计原则：
  - 轻量（dim=256, layers=4），推理延迟控制在 20ms 以内
  - 融入硬约束（已打出/已知的牌概率强制为 0）
  - 训练时用真实手牌做 BCE 监督
  - 推理时用于生成 belief_vec 输入给 PolicyTransformer
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from rinshan.constants import (
    NUM_TILE_TYPES, VOCAB_SIZE, PAD_TOKEN,
    BELIEF_DIM, BELIEF_LAYERS, BELIEF_HEADS, BELIEF_OUTPUT_DIM, WAIT_OUTPUT_DIM,
    MAX_PROGRESSION_LEN,
)


class BeliefNetwork(nn.Module):
    """
    轻量 Transformer，专门用于不完全信息推断

    输入 token 序列：
      [河牌历史（4家）] [副露序列] [宝牌指示牌] [剩余牌数标量]
    最大长度约 120 tokens

    输出：
      cls_hidden: (B, BELIEF_DIM)  — 供 PolicyTransformer 使用的压缩向量
      belief_logits: (B, 34, 3)    — 三个对手各牌的 logit（训练时算 loss 用）
    """

    def __init__(
        self,
        dim: int = BELIEF_DIM,
        n_heads: int = BELIEF_HEADS,
        n_layers: int = BELIEF_LAYERS,
        dropout: float = 0.1,
        max_seq_len: int = MAX_PROGRESSION_LEN + 20 + 12,  # 145：+12 立直上下文
    ):
        super().__init__()
        self.dim = dim

        # CLS token（可学习）
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Token embedding（共享词表）
        self.token_embed  = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD_TOKEN)

        # Learned position embedding（序列不长，不用 RoPE）
        self.pos_embed = nn.Embedding(max_seq_len + 1, dim)  # +1 for CLS

        # Transformer Encoder（轻量，双向）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dim)

        # 手牌信念输出头：预测每张牌在三个对手手里的概率
        self.belief_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, BELIEF_OUTPUT_DIM),  # 34 * 3 = 102
        )

        # 待张预测头：预测三个对手各自的待张牌
        # 覆盖场景：立直/默听/副露听牌，统一作为概率分布学习
        # 训练时只对对手处于 tenpai 的位置计算 loss，非 tenpai 时自动 mask
        self.wait_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, WAIT_OUTPUT_DIM),    # 34 * 3 = 102
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,                      # (B, S) int, public-state token sequence
        pad_mask: Optional[torch.Tensor] = None,   # (B, S) bool, True = pad position
        known_absent: Optional[torch.Tensor] = None, # (B, 34, 3) bool, True = tile definitely absent
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          cls_vec:       (B, BELIEF_DIM)      — global summary
          belief_logits: (B, 34, 3)           — per-tile per-opponent logits（手牌预测）
          wait_logits:   (B, 34, 3)           — per-tile per-opponent logits（待张预测）
          memory:        (B, S+1, BELIEF_DIM) — full encoder hidden states for cross-attention
        """
        B, S = tokens.shape

        # Token embedding + positional encoding
        x = self.token_embed(tokens)   # (B, S, dim)
        pos_ids = torch.arange(1, S + 1, device=tokens.device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)

        # Prepend the CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        cls_pos = self.pos_embed(
            torch.zeros(B, 1, dtype=torch.long, device=tokens.device)
        )
        cls = cls + cls_pos
        x = torch.cat([cls, x], dim=1)   # (B, S+1, dim)

        # Extend pad_mask; the CLS position is never masked
        if pad_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
            src_key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)
        else:
            src_key_padding_mask = None

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)  # (B, S+1, BELIEF_DIM)

        # CLS token as global summary
        cls_vec = x[:, 0, :]   # (B, BELIEF_DIM)

        # Predict per-tile per-opponent probabilities
        belief_logits = self.belief_head(cls_vec)              # (B, 102)
        belief_logits = belief_logits.view(B, NUM_TILE_TYPES, 3)  # (B, 34, 3)

        # Apply hard constraints: force probability to zero for absent tiles
        if known_absent is not None:
            belief_logits = belief_logits.masked_fill(known_absent, float('-inf'))

        # 待张预测
        wait_logits = self.wait_head(cls_vec)              # (B, 102)
        wait_logits = wait_logits.view(B, NUM_TILE_TYPES, 3)  # (B, 34, 3)

        # 返回 logits（不再内部做 sigmoid）供 loss 用 BCEWithLogits 计算
        belief_logits_out = belief_logits  # (B, 34, 3)
        wait_logits_out   = wait_logits    # (B, 34, 3)

        # Return the full hidden-state sequence as cross-attention memory
        memory = x   # (B, S+1, BELIEF_DIM)

        return cls_vec, belief_logits_out, wait_logits_out, memory

    def compute_loss(
        self,
        belief_probs: torch.Tensor,   # (B, 34, 3)
        actual_hands: torch.Tensor,   # (B, 34, 3) binary，真实手牌 one-hot（0/1/2/3/4）
    ) -> torch.Tensor:
        """
        BCE loss：预测各牌是否在对手手里
        actual_hands 是 0/1 矩阵（有牌=1，无牌=0）
        注意：actual_hands 只在训练时（有完整数据）才能计算
        """
        # 把牌数 > 0 的位置视为 1（简化：是否有该牌，而不是精确张数）
        target = (actual_hands > 0).float()
        return F.binary_cross_entropy(belief_probs, target)
