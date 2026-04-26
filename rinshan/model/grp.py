"""
GRP — Game Result Predictor（局结果预测器）

从 Mortal 借来的核心设计，是整个奖励体系的基础。

功能：
  给定当前游戏进程（截止某局结束的分数序列），
  预测每位玩家最终排名的概率分布 P(player_i 得第 j 名)

这样可以把稀疏的终局奖励转化成每局（kyoku）结束时的
稠密奖励信号 ΔE[pts]，大幅提高 RL 的样本效率。

输入格式（每局一帧，7维）：
  [grand_kyoku, honba, kyotaku, score_p0, score_p1, score_p2, score_p3]
  grand_kyoku: E1=0, E2=1, ..., S4=7, W4=11
  分数以 10000 为单位归一化（25000点 → 2.5）

输出：
  logits: (B, 24) → 4! = 24 种排名组合的 logit
  通过 calc_matrix 转为 (B, 4, 4) 排名概率矩阵
"""
from __future__ import annotations

import math
from itertools import permutations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from rinshan.constants import (
    GRP_INPUT_DIM, GRP_HIDDEN_SIZE, GRP_NUM_LAYERS, GRP_OUTPUT_DIM,
    RANK_PTS_TENHOU, RANK_PTS_JANTAMA,
)


class GRP(nn.Module):
    """
    GRU-based Game Result Predictor

    使用 float64 保证分数计算精度（继承 Mortal 的做法）
    """

    def __init__(
        self,
        input_dim: int = GRP_INPUT_DIM,
        hidden_size: int = GRP_HIDDEN_SIZE,
        num_layers: int = GRP_NUM_LAYERS,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * num_layers, GRP_OUTPUT_DIM),  # 24
        )

        # 使用 float64 精度
        for m in self.modules():
            m.to(torch.float64)

        # 预计算所有 4! = 24 种排名排列
        # perms[i] = [r0, r1, r2, r3]，表示第 i 种排列下各玩家的排名
        perms = torch.tensor(list(permutations(range(4))), dtype=torch.long)
        perms_t = perms.transpose(0, 1)  # (4, 24)
        self.register_buffer("perms",   perms)    # (24, 4)
        self.register_buffer("perms_t", perms_t)  # (4, 24)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        inputs: List of Tensor，每个元素是一局游戏的帧序列
                每帧 shape (kyoku_count, 7)，长度可以不同

        返回 logits: (B, 24)
        """
        lengths = torch.tensor(
            [t.shape[0] for t in inputs], dtype=torch.int64
        )
        padded = pad_sequence(inputs, batch_first=True)  # (B, max_len, 7)
        packed = pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False
        )
        return self._forward_packed(packed)

    def _forward_packed(self, packed_inputs) -> torch.Tensor:
        _, hidden = self.rnn(packed_inputs)
        # hidden: (num_layers, B, hidden_size)
        hidden = hidden.transpose(0, 1).flatten(1)  # (B, num_layers * hidden_size)
        return self.fc(hidden)  # (B, 24)

    def calc_matrix(self, logits: torch.Tensor) -> torch.Tensor:
        """
        从 logits 计算排名概率矩阵

        Args:
            logits: (B, 24)
        Returns:
            matrix: (B, 4, 4)，matrix[b][player][rank] = 概率
        """
        B = logits.shape[0]
        probs = logits.softmax(dim=-1)   # (B, 24)
        matrix = torch.zeros(B, 4, 4, dtype=probs.dtype, device=probs.device)
        for player in range(4):
            for rank in range(4):
                # 找出所有 player 排名为 rank 的排列
                cond = (self.perms_t[player] == rank)   # (24,) bool
                matrix[:, player, rank] = probs[:, cond].sum(dim=-1)
        return matrix

    def get_label(self, rank_by_player: torch.Tensor) -> torch.Tensor:
        """
        将真实排名结果转换为 24 分类的 label

        Args:
            rank_by_player: (B, 4) int，rank_by_player[b][i] = 玩家 i 的排名(0-3)
        Returns:
            labels: (B,) int，对应 perms 中的索引
        """
        B = rank_by_player.shape[0]
        perms_exp = self.perms.unsqueeze(1).expand(-1, B, -1)  # (24, B, 4)
        matches = (perms_exp == rank_by_player.unsqueeze(0)).all(dim=-1)  # (24, B)
        labels = matches.nonzero()  # (B, 2)：[perm_idx, batch_idx]

        result = torch.zeros(B, dtype=torch.long, device=rank_by_player.device)
        result[labels[:, 1]] = labels[:, 0]
        return result

    def compute_loss(
        self,
        logits: torch.Tensor,
        rank_by_player: torch.Tensor,
    ) -> torch.Tensor:
        """交叉熵损失"""
        labels = self.get_label(rank_by_player)
        return F.cross_entropy(logits.float(), labels)


class RewardCalculator:
    """
    用训练好的 GRP 计算每个决策步骤的奖励

    奖励定义：
      reward[kyoku] = E[pts | game进程到kyoku+1局结束]
                    - E[pts | game进程到kyoku局结束]

    即每局结束时，期望得分的变化量作为该局的奖励信号
    """

    def __init__(
        self,
        grp: GRP,
        pts: Optional[list[float]] = None,
        platform: str = "tenhou",
    ):
        self.grp = grp.eval()
        if pts is not None:
            self.pts = torch.tensor(pts, dtype=torch.float64)
        elif platform == "tenhou":
            self.pts = torch.tensor(RANK_PTS_TENHOU, dtype=torch.float64)
        else:
            self.pts = torch.tensor(RANK_PTS_JANTAMA, dtype=torch.float64)

    @torch.no_grad()
    def calc_expected_pts(
        self,
        grp_feature_seq: torch.Tensor,  # (T, 7) 游戏进程到某时刻的帧序列
        player_id: int,
    ) -> torch.Tensor:
        """
        计算给定游戏进程下，该玩家的期望得分
        将所有前缀打包成一个 batch，一次 forward 搞定（原来是 T 次）

        Returns: (T,) 各时刻的期望得分
        """
        T = grp_feature_seq.shape[0]
        # 一次 batched forward：前缀长度 1, 2, ..., T
        seqs = [grp_feature_seq[:t] for t in range(1, T + 1)]
        logits = self.grp(seqs)                     # (T, 24)
        matrix = self.grp.calc_matrix(logits)       # (T, 4, 4)
        rank_probs = matrix[:, player_id, :]        # (T, 4)
        pts = self.pts.to(rank_probs.device)
        exp_pts = (rank_probs.double() * pts).sum(dim=-1)  # (T,)
        return exp_pts  # (T,)

    @torch.no_grad()
    def calc_delta_pts(
        self,
        player_id: int,
        grp_feature_seq: torch.Tensor,  # (T, 7)
        rank_by_player: torch.Tensor,   # (4,) 最终排名（用于计算最后一步真实奖励）
    ) -> torch.Tensor:
        """
        计算每局的奖励（期望得分变化）

        Returns: (T,) float，第 t 个元素 = 第 t 局结束时的奖励
        """
        exp_pts = self.calc_expected_pts(grp_feature_seq, player_id)

        # 最后一个奖励用真实排名
        final_rank = rank_by_player[player_id].item()
        final_pt = self.pts[final_rank].to(exp_pts.device)

        all_pts = torch.cat([exp_pts, final_pt.unsqueeze(0)])
        delta = all_pts[1:] - all_pts[:-1]  # (T,)
        return delta.cpu().numpy()
