"""
RinshanModel — 完整推理模型

将 BeliefNetwork + PolicyTransformer + QVHead + AuxHeads 组合在一起
训练时各阶段可以单独使用不同的组件组合
推理时只需要 RinshanModel.react()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .transformer import PolicyTransformer, TransformerConfig
from .belief     import BeliefNetwork
from .qv_head    import QVHead, sample_action
from .aux_head   import AuxHeads
from rinshan.constants import (
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_CANDIDATES_LEN
)


@dataclass
class RinshanOutput:
    q:            torch.Tensor             # (B, MAX_CANDIDATES)  总 Q 值
    v:            torch.Tensor             # (B,)  总状态价值
    q_game:       torch.Tensor             # (B, MAX_CANDIDATES)  整场价值分支 Q
    q_hand:       torch.Tensor             # (B, MAX_CANDIDATES)  局内价值分支 Q
    v_game:       torch.Tensor             # (B,) 整场状态价值
    v_hand:       torch.Tensor             # (B,) 局内状态价值
    belief_probs:  Optional[torch.Tensor]  # (B, 34, 3)  手牌信念概率（已 sigmoid，供推理）
    belief_logits: Optional[torch.Tensor]  # (B, 34, 3)  手牌信念 logits（用于 BCE loss）
    wait_logits:   Optional[torch.Tensor]  # (B, 34, 3)  待张预测 logits
    wait_probs:    Optional[torch.Tensor]  # (B, 34, 3)  待张概率（已 sigmoid，供推理）
    belief_vec:   Optional[torch.Tensor]   # (B, BELIEF_DIM)  信念向量
    aux_preds:    Optional[dict]           # 辅助任务预测
    encode:       torch.Tensor             # (B, S, dim)  Transformer 输出（供 loss 计算）


class RinshanModel(nn.Module):
    """
    完整的 Rinshan 模型

    Args:
        transformer_cfg: Transformer 配置（规模、dim、层数等）
        use_belief:  是否启用 Belief Network（默认 True）
        use_aux:     是否启用辅助任务头（默认 True，仅训练阶段有意义）
    """

    def __init__(
        self,
        transformer_cfg: Optional[TransformerConfig] = None,
        use_belief: bool = True,
        use_aux: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if transformer_cfg is None:
            transformer_cfg = TransformerConfig()
        self.cfg = transformer_cfg

        self.transformer = PolicyTransformer(transformer_cfg, gradient_checkpointing=gradient_checkpointing)
        self.qv_head     = QVHead(transformer_cfg.dim, transformer_cfg.dropout)
        self.belief_net  = BeliefNetwork() if use_belief else None
        self.aux_heads   = AuxHeads(transformer_cfg.dim, transformer_cfg.dropout) \
                           if use_aux else None

    def forward(
        self,
        tokens: torch.Tensor,                           # (B, S) main sequence
        candidate_mask: torch.Tensor,                   # (B, MAX_CANDIDATES) bool
        pad_mask: Optional[torch.Tensor] = None,        # (B, S) bool
        belief_tokens: Optional[torch.Tensor] = None,   # (B, S') public-info sequence for the Belief Net
        belief_pad_mask: Optional[torch.Tensor] = None,
        known_absent: Optional[torch.Tensor] = None,    # (B, 34, 3) hard absence constraints
        compute_aux: bool = False,
    ) -> RinshanOutput:

        # ── Belief Network ──────────────────────────────────────────────
        belief_vec    = None
        belief_logits = None
        wait_logits   = None
        belief_memory = None
        if self.belief_net is not None and belief_tokens is not None:
            belief_vec, belief_logits, wait_logits, belief_memory = self.belief_net(
                belief_tokens, belief_pad_mask, known_absent
            )

        # ── Policy Transformer (belief cross-attention inside each block) ─
        encode = self.transformer(tokens, belief_memory, pad_mask)

        # ── QV Head ────────────────────────────────────────────────────
        q, v, q_game, q_hand, v_game, v_hand = self.qv_head(encode, candidate_mask)

        # ── Aux Heads (computed on demand during training) ──────────────
        aux_preds = None
        if self.aux_heads is not None and compute_aux:
            aux_preds = self.aux_heads(encode)

        return RinshanOutput(
            q=q, v=v,
            q_game=q_game, q_hand=q_hand,
            v_game=v_game, v_hand=v_hand,
            belief_probs=torch.sigmoid(belief_logits) if belief_logits is not None else None,
            belief_logits=belief_logits,
            wait_logits=wait_logits,
            wait_probs=torch.sigmoid(wait_logits) if wait_logits is not None else None,
            belief_vec=belief_vec,
            aux_preds=aux_preds,
            encode=encode,
        )

    @torch.inference_mode()
    def react(
        self,
        tokens: torch.Tensor,
        candidate_mask: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        belief_tokens: Optional[torch.Tensor] = None,
        belief_pad_mask: Optional[torch.Tensor] = None,
        known_absent: Optional[torch.Tensor] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        推理接口：返回 (actions, q_values)

        actions: (B,) 每个样本选择的动作 index（相对于 candidate_mask）
        q_values: (B, MAX_CANDIDATES) 完整 Q 值，用于记录和分析
        """
        out = self.forward(
            tokens, candidate_mask, pad_mask,
            belief_tokens, belief_pad_mask, known_absent,
            compute_aux=False,
        )
        actions = sample_action(out.q, temperature, top_p, greedy)
        return actions, out.q

    def count_parameters(self) -> dict[str, int]:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        result = {"transformer": count(self.transformer), "qv_head": count(self.qv_head)}
        if self.belief_net:
            result["belief_net"] = count(self.belief_net)
        if self.aux_heads:
            result["aux_heads"] = count(self.aux_heads)
        result["total"] = sum(result.values())
        return result
