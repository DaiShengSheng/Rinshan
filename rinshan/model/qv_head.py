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

        # GRP 2.0：双 value 分解
        # - v_game / a_game：长期整场名次价值（GRP game value）
        # - v_hand / a_hand：局内得失点价值（hand / kyoku value）
        self.v_game_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.v_hand_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self.a_game_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.a_hand_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # 最后一层初始化为 0，避免训练初期 Q 值爆炸
        for head in (self.v_game_net, self.v_hand_net, self.a_game_net, self.a_hand_net):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        encode: torch.Tensor,          # (B, S, dim) PolicyTransformer 输出
        candidate_mask: torch.Tensor,  # (B, MAX_CANDIDATES) bool，True = 合法动作
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
          q:       (B, MAX_CANDIDATES_LEN) 总 Q 值，非法动作为 -inf
          v:       (B,) 总状态价值 = v_game + v_hand
          q_game:  (B, MAX_CANDIDATES_LEN) 整场价值分支
          q_hand:  (B, MAX_CANDIDATES_LEN) 局内价值分支
          v_game:  (B,) 整场状态价值
          v_hand:  (B,) 局内状态价值
        """

        def _dueling(v_branch: torch.Tensor, a_branch: torch.Tensor) -> torch.Tensor:
            masked_a = a_branch.masked_fill(~candidate_mask, 0.0)
            a_sum = masked_a.sum(dim=-1, keepdim=True)
            a_count = candidate_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
            a_mean = a_sum / a_count
            q_branch = v_branch.unsqueeze(-1) + a_branch - a_mean
            q_branch = q_branch.masked_fill(~candidate_mask, float('-inf'))
            return q_branch

        pooled = encode.mean(dim=1)
        v_game = self.v_game_net(pooled).squeeze(-1)
        v_hand = self.v_hand_net(pooled).squeeze(-1)

        cand_encode = encode[:, -MAX_CANDIDATES_LEN:, :]
        a_game = self.a_game_net(cand_encode).squeeze(-1)
        a_hand = self.a_hand_net(cand_encode).squeeze(-1)

        q_game = _dueling(v_game, a_game)
        q_hand = _dueling(v_hand, a_hand)
        q = q_game + q_hand
        v = v_game + v_hand

        return q, v, q_game, q_hand, v_game, v_hand


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
