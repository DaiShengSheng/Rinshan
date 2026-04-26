"""
Policy Transformer — Llama 3 风格的 decoder-only Transformer
用于麻将决策的主干网络

关键设计选择：
  - RoPE 旋转位置编码（变长序列外推好）
  - RMSNorm（更稳定）
  - SwiGLU FFN（表达力强）
  - GQA 分组注意力（推理时 KV Cache 更小）
  - 双向注意力（决策时需要看完整局面，不需要 causal mask）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rinshan.constants import VOCAB_SIZE, MAX_SEQ_LEN, MODEL_CONFIGS


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

@dataclass
class TransformerConfig:
    dim: int        = 768
    n_heads: int    = 12
    n_kv_heads: int = 4      # GQA: kv heads < q heads
    n_layers: int   = 12
    ffn_dim: int    = 2048
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int= MAX_SEQ_LEN
    dropout: float  = 0.1
    rope_theta: float = 10000.0
    norm_eps: float   = 1e-5
    # Which layer indices receive a cross-attention sublayer for belief injection.
    # Default: every 4th layer starting from layer 3 (0-indexed).
    # None means "auto-compute from n_layers".
    cross_attn_layers: tuple = None   # e.g. (3, 7, 11) for base

    def __post_init__(self):
        if self.cross_attn_layers is None:
            # Place a cross-attn at the last layer of every group of 4.
            # nano (4 layers)  -> (3,)
            # base (12 layers) -> (3, 7, 11)
            # large (24 layers)-> (3, 7, 11, 15, 19, 23)
            self.cross_attn_layers = tuple(
                range(3, self.n_layers, 4)
            )

    @classmethod
    def from_preset(cls, name: str = "base") -> "TransformerConfig":
        assert name in MODEL_CONFIGS, f"Unknown preset: {name}"
        return cls(**MODEL_CONFIGS[name])

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """Number of Q heads per KV head."""
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads


# ─────────────────────────────────────────────
# Belief cross-attention sublayer
# ─────────────────────────────────────────────

def _cross_attn_heads(dim: int, n_heads: int) -> int:
    """
    为 BeliefCrossAttention 选一个合适的 head 数：
    约为 n_heads // 3，但必须能整除 dim。
    """
    target = max(1, n_heads // 3)
    # 从 target 往下找第一个能整除 dim 的正因数
    for h in range(target, 0, -1):
        if dim % h == 0:
            return h
    return 1  # fallback


class BeliefCrossAttention(nn.Module):
    """
    Single-head cross-attention: policy tokens (Q) attend to belief memory (K, V).

    Q comes from the policy hidden states (dim).
    K, V come from the BeliefNetwork hidden states (belief_dim).

    Uses standard multi-head attention (not GQA) because the memory sequence
    is short (~133 tokens) and each head seeing different belief aspects is valuable.
    """

    def __init__(self, dim: int, belief_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5

        # Q projects from policy dim; K/V project from belief dim
        self.q_proj = nn.Linear(dim,        dim,        bias=False)
        self.k_proj = nn.Linear(belief_dim, dim,        bias=False)
        self.v_proj = nn.Linear(belief_dim, dim,        bias=False)
        self.o_proj = nn.Linear(dim,        dim,        bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Pre-norm on both sides
        self.norm_q = RMSNorm(dim)
        self.norm_m = RMSNorm(belief_dim)

        self._init_weights()

    def _init_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(proj.weight, std=0.02)

    def forward(
        self,
        x:      torch.Tensor,   # (B, S, dim)         policy hidden states
        memory: torch.Tensor,   # (B, S_m, belief_dim) belief encoder output
    ) -> torch.Tensor:
        """Returns (B, S, dim), the cross-attention residual (to be added to x)."""
        B, S,  _ = x.shape
        B, Sm, _ = memory.shape

        # Pre-norm
        xn = self.norm_q(x)
        mn = self.norm_m(memory)

        q = self.q_proj(xn).view(B, S,  self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(mn).view(B, Sm, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mn).view(B, Sm, self.n_heads, self.head_dim).transpose(1, 2)
        # Note: no RoPE on cross-attn keys — the belief sequence has its own
        # positional encoding and the two position spaces should not be mixed.

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)   # (B, S, dim)


# ─────────────────────────────────────────────
# RoPE 旋转位置编码
# ─────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 的旋转频率"""
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)            # (seq, head_dim/2)
    cos = freqs.cos()                         # (seq, head_dim/2)
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:       (B, n_heads, seq, head_dim)
    cos/sin: (seq, head_dim/2)
    """
    B, H, S, D = x.shape
    x1, x2 = x[..., :D//2], x[..., D//2:]      # 各 (B, H, S, D/2)
    c = cos[:S].unsqueeze(0).unsqueeze(0)         # (1, 1, S, D/2)
    s = sin[:S].unsqueeze(0).unsqueeze(0)
    # 标准 RoPE 旋转公式
    x1_rot = x1 * c - x2 * s
    x2_rot = x1 * s + x2 * c
    return torch.cat([x1_rot, x2_rot], dim=-1)   # (B, H, S, D)


# ─────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ─────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU: FFN(x) = SiLU(W1·x) ⊙ (W2·x), 再经 W3 投影
    参数量比标准 FFN 多 50%，但效果更好
    """
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ─────────────────────────────────────────────
# GQA Attention（Grouped Query Attention）
# ─────────────────────────────────────────────

class GQAttention(nn.Module):
    """
    双向（非 causal）分组查询注意力
    n_kv_heads < n_heads 时节省 KV Cache
    """
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups   = cfg.n_kv_groups
        self.head_dim   = cfg.head_dim
        self.dropout    = cfg.dropout

        self.q_proj  = nn.Linear(cfg.dim, cfg.n_heads    * cfg.head_dim, bias=False)
        self.k_proj  = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj  = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj  = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,             # (B, S, dim)
        cos: torch.Tensor,           # (S, head_dim/2)
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,   # (B, 1, 1, S) or (B, 1, S, S) bool，True=mask out
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: 把 KV 重复 n_groups 次以匹配 Q 的 head 数
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, S, head_dim)
            v = v.repeat_interleave(self.n_groups, dim=1)

        sdpa_mask = None
        if attn_mask is not None:
            # attn_mask: (B,1,1,S) bool, True=mask out
            # 用张量运算替代 masked_fill，对 compile 友好
            neg_inf = torch.tensor(float('-inf'), dtype=q.dtype, device=q.device)
            sdpa_mask = torch.where(attn_mask, neg_inf, torch.zeros_like(neg_inf))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn      = GQAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn       = SwiGLUFFN(cfg.dim, cfg.ffn_dim, cfg.dropout)
        self.drop      = nn.Dropout(cfg.dropout)

        # Optional belief cross-attention (only on designated layers)
        self.has_cross_attn = (layer_idx in cfg.cross_attn_layers)
        if self.has_cross_attn:
            from rinshan.constants import BELIEF_DIM
            self.cross_attn = BeliefCrossAttention(
                dim=cfg.dim,
                belief_dim=BELIEF_DIM,
                n_heads=_cross_attn_heads(cfg.dim, cfg.n_heads),
                dropout=cfg.dropout,
            )
        else:
            self.cross_attn = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        belief_memory: Optional[torch.Tensor] = None,  # (B, S_m, BELIEF_DIM)
    ) -> torch.Tensor:
        # Self-attention with pre-norm residual
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, attn_mask))
        # Cross-attention on designated layers
        if self.has_cross_attn and belief_memory is not None:
            x = x + self.drop(self.cross_attn(x, belief_memory))
        # FFN
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


# ─────────────────────────────────────────────
# Policy Transformer（主模型）
# ─────────────────────────────────────────────

class PolicyTransformer(nn.Module):
    """
    麻将决策的主干 Transformer

    输入序列结构：
      [META tokens] [DORA tokens] [HAND tokens] [MELD tokens]
      [PROGRESSION tokens] [CANDIDATE tokens]
    + 可选的 belief context（拼接在序列头部）

    输出：每个 token 位置的 hidden state，shape (B, S, dim)
    后续由 QVHead / AuxHeads 使用最后 32 个 candidate token
    """

    def __init__(self, cfg: Optional[TransformerConfig] = None, gradient_checkpointing: bool = False):
        super().__init__()
        if cfg is None:
            cfg = TransformerConfig()
        self.cfg = cfg
        self.gradient_checkpointing = gradient_checkpointing

        # Token embedding（包含 PAD，padding_idx=PAD_TOKEN）
        from rinshan.constants import PAD_TOKEN
        self.token_embed = nn.Embedding(
            cfg.vocab_size, cfg.dim, padding_idx=PAD_TOKEN
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.drop = nn.Dropout(cfg.dropout)

        # 预计算 RoPE 频率，注册为 buffer（不参与梯度）
        cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        tokens: torch.Tensor,                           # (B, S) int
        belief_memory: Optional[torch.Tensor] = None,   # (B, S_m, BELIEF_DIM) full belief hidden states
        pad_mask: Optional[torch.Tensor] = None,        # (B, S) bool, True = pad
    ) -> torch.Tensor:
        """Returns (B, S, dim) hidden states."""
        B, S = tokens.shape

        # Token embedding
        x = self.drop(self.token_embed(tokens))   # (B, S, dim)

        # Padding mask -> attention mask (True = mask out this position)
        attn_mask = None
        if pad_mask is not None:
            # (B, 1, 1, S) broadcasts to (B, n_heads, S, S)
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        # Transformer layers — cross-attn layers will consume belief_memory
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # 不保存中间激活层，反向时重新计算，大幅节省显存
                x = checkpoint(
                    layer, x, cos, sin, attn_mask, belief_memory,
                    use_reentrant=False,
                )
            else:
                x = layer(x, cos, sin, attn_mask, belief_memory)

        return self.norm(x)   # (B, S, dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
