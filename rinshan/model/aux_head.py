"""
辅助任务头 — 加速 encoder 表示学习

四个辅助任务：
  1. shanten:      当前向听数预测 (-1 ~ 8) → 10 分类
  2. tenpai_prob:  听牌概率预测 → 二分类
  3. deal_in_risk: 每张牌的放铳风险 → 34 维回归
  4. opp_tenpai:   三个对手是否听牌 → 3 个独立二分类

这些任务都有明确的监督信号，可以从标注数据直接获取
帮助 encoder 在 RL 阶段之前就学到有意义的中间表示
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rinshan.constants import NUM_TILE_TYPES, AUX_WEIGHTS


class AuxHeads(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = dim // 4

        # 1. 向听数：-1（已和）到 8（差 8 张），共 10 类
        self.shanten_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

        # 2. 听牌概率（二分类 logit）
        self.tenpai_prob_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # 3. 放铳风险：每张牌（34种）的危险度 logit
        self.deal_in_risk_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_TILE_TYPES),
        )

        # 4. 对手听牌：3 个对手各自的听牌 logit
        self.opp_tenpai_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, encode: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        encode: (B, S, dim) PolicyTransformer 输出
        使用序列均值作为全局表示（也可以用 [CLS]，但这里没有显式 CLS）

        Returns: dict of logits，每个 key 对应一个辅助任务
        """
        global_repr = encode.mean(dim=1)   # (B, dim)

        return {
            "shanten":      self.shanten_head(global_repr),       # (B, 10)
            "tenpai_prob":  self.tenpai_prob_head(global_repr).squeeze(-1),   # (B,)
            "deal_in_risk": self.deal_in_risk_head(global_repr),  # (B, 34)
            "opp_tenpai":   self.opp_tenpai_head(global_repr),    # (B, 3)
        }

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算辅助任务总损失

        targets 期望格式：
          "shanten":      (B,) int，值域 0-9（-1 映射到 0，0 映射到 1，…）
          "tenpai_prob":  (B,) float，0/1
          "deal_in_risk": (B, 34) float，0~1 危险度
          "opp_tenpai":   (B, 3) float，0/1
        """
        losses = {}

        if "shanten" in targets:
            losses["shanten"] = F.cross_entropy(
                preds["shanten"], targets["shanten"].long()
            )

        if "tenpai_prob" in targets:
            losses["tenpai_prob"] = F.binary_cross_entropy_with_logits(
                preds["tenpai_prob"], targets["tenpai_prob"].float()
            )

        if "deal_in_risk" in targets:
            losses["deal_in_risk"] = F.binary_cross_entropy_with_logits(
                preds["deal_in_risk"], targets["deal_in_risk"].float()
            )

        if "opp_tenpai" in targets:
            losses["opp_tenpai"] = F.binary_cross_entropy_with_logits(
                preds["opp_tenpai"], targets["opp_tenpai"].float()
            )

        # 加权求和
        total = sum(
            AUX_WEIGHTS.get(k, 0.1) * v
            for k, v in losses.items()
        )
        return total, losses
