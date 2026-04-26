"""
QV Head — Dueling Architecture

Q(s,a) = V(s) + A(s,a) - mean(A)

V(s): 全局状态价值，从序列全局池化计算
A(s,a): 每个候选动作的优势，从 candidate token 的 hidden state 计算

推理时输出混合策略（top-p sampling），而非 argmax
这在博弈论上更正确：保持策略不可预测性
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rinshan.constants import MAX_CANDIDATES_LEN, DEFAULT_TEMPERATURE, DEFAULT_TOP_P


class QVHead(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = dim // 2

        # V(s)：聚合全局信息 → 标量
        self.v_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # A(s,a)：每个 candidate token → 标量优势
        self.a_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # 最后一层初始化为 0，避免训练初期 Q 值爆炸
        nn.init.zeros_(self.v_net[-1].weight)
        nn.init.zeros_(self.v_net[-1].bias)
        nn.init.zeros_(self.a_net[-1].weight)
        nn.init.zeros_(self.a_net[-1].bias)

    def forward(
        self,
        encode: torch.Tensor,          # (B, S, dim) PolicyTransformer 输出
        candidate_mask: torch.Tensor,  # (B, MAX_CANDIDATES) bool，True = 合法动作
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          q: (B, MAX_CANDIDATES_LEN) 每个候选动作的 Q 值，非法动作为 -inf
          v: (B,) 当前状态的 V 值
        """
        B = encode.shape[0]

        # V(s)：对全部 token 取均值池化后过 MLP
        v = self.v_net(encode.mean(dim=1)).squeeze(-1)   # (B,)

        # A(s,a)：取序列最后 MAX_CANDIDATES 个位置（即 candidate tokens）
        cand_encode = encode[:, -MAX_CANDIDATES_LEN:, :]  # (B, 32, dim)
        a = self.a_net(cand_encode).squeeze(-1)            # (B, 32)

        # Dueling：Q = V + A - mean(A)（只对合法动作计算均值）
        masked_a = a.masked_fill(~candidate_mask, 0.0)
        a_sum  = masked_a.sum(dim=-1, keepdim=True)           # (B, 1)
        a_count = candidate_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        a_mean = a_sum / a_count                              # (B, 1)

        q = v.unsqueeze(-1) + a - a_mean                     # (B, 32)
        q = q.masked_fill(~candidate_mask, float('-inf'))    # 非法动作 -inf

        return q, v


# ─────────────────────────────────────────────
# 推理时动作采样（混合策略）
# ─────────────────────────────────────────────

def sample_action(
    q: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    greedy: bool = False,
) -> torch.Tensor:
    """
    从 Q 值分布采样动作

    Args:
        q: (B, num_candidates) Q 值，非法动作为 -inf
        temperature: 温度系数，越高越随机
        top_p: nucleus sampling 的累积概率阈值
        greedy: 为 True 时直接取 argmax（评估时用）

    Returns:
        actions: (B,) 每个样本选择的动作 index
    """
    if greedy or temperature <= 0:
        return q.argmax(dim=-1)

    # Temperature scaling
    logits = q / temperature

    # Top-p (nucleus) sampling
    probs = F.softmax(logits, dim=-1)                          # (B, N)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumprobs = sorted_probs.cumsum(dim=-1)

    # 把累积概率超过 top_p 的部分置 0
    # 注意：shifted 操作确保至少保留 top-1
    remove_mask = (cumprobs - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

    # 从过滤后的分布中采样
    sampled_sorted_idx = sorted_probs.multinomial(1).squeeze(-1)  # (B,)
    actions = sorted_idx.gather(1, sampled_sorted_idx.unsqueeze(-1)).squeeze(-1)
    return actions
