"""
训练损失函数

Stage 1: 行为克隆 — stage1_loss
Stage 2: Oracle 蒸馏 — distill_loss
Stage 3: 离线 IQL — iql_loss
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rinshan.constants import GAMMA, IQL_EXPECTILE, CQL_WEIGHT, DISTILL_TEMP


# ─────────────────────────────────────────────
# Stage 1：行为克隆损失
# ─────────────────────────────────────────────

def stage1_loss(
    q: torch.Tensor,              # (B, N) Q 值 logits
    action_idx: torch.Tensor,     # (B,) 人类选择的动作 index
    belief_logits: Optional[torch.Tensor] = None,  # (B, 34, 3) raw logits
    actual_hands: Optional[torch.Tensor] = None,  # (B, 34, 3) 真实手牌
    aux_preds: Optional[dict]     = None,
    aux_targets: Optional[dict]   = None,
    belief_weight: float = 0.3,
    aux_weight: float   = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    行为克隆（模仿学习）损失

    主损失：CrossEntropy(Q logits, 人类选择的动作)
    辅助：Belief BCE + AuxHead losses

    注意权重体系：
      belief loss 权重 = belief_weight（默认 0.3）
      aux task 权重   = aux_weight（默认 1.0）× AUX_WEIGHTS[task_name]
      例：shanten 实际权重 = 1.0 × 0.1 = 0.1
    """
    losses = {}

    # 主损失：模仿人类动作
    action_loss = F.cross_entropy(q, action_idx)
    losses["action"] = action_loss.item()

    total = action_loss

    # Belief 损失（BCEWithLogitsLoss：autocast 安全且数值更稳定）
    if belief_logits is not None and actual_hands is not None:
        target = (actual_hands > 0).float()
        b_loss = F.binary_cross_entropy_with_logits(belief_logits, target)
        losses["belief"] = b_loss.item()
        total = total + belief_weight * b_loss

    # 辅助任务损失
    if aux_preds is not None and aux_targets is not None:
        from rinshan.model.aux_head import AuxHeads
        # 调用 AuxHeads.compute_loss（静态方式）
        from rinshan.constants import AUX_WEIGHTS
        for k, pred in aux_preds.items():
            if k not in aux_targets:
                continue
            tgt = aux_targets[k]
            if k == "shanten":
                a_loss = F.cross_entropy(pred, tgt.long())
            else:
                a_loss = F.binary_cross_entropy_with_logits(pred, tgt.float())
            w = AUX_WEIGHTS.get(k, 0.1)
            losses[f"aux_{k}"] = a_loss.item()
            total = total + aux_weight * w * a_loss

    losses["total"] = total.item()
    return total, losses


# ─────────────────────────────────────────────
# Stage 2：Oracle 蒸馏损失
# ─────────────────────────────────────────────

def distill_loss(
    student_q: torch.Tensor,      # (B, N) 学生 Q 值
    oracle_q: torch.Tensor,       # (B, N) Oracle Q 值（no_grad）
    action_idx: torch.Tensor,     # (B,) 人类动作
    temperature: float = DISTILL_TEMP,
    distill_weight: float = 1.0,
    bc_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Oracle 蒸馏损失 = KL(student || oracle) + BC 交叉熵

    让学生网络（只看公开信息）逼近 Oracle（看全牌）的 Q 值分布
    """
    losses = {}

    # Oracle 软标签：把 -inf 的非法动作替换成极大负数再 softmax，
    # 避免 softmax 输出精确 0 导致 kl_div 出现 0*log(0)=nan
    oracle_q_safe = oracle_q.detach().float()
    oracle_q_safe = oracle_q_safe.masked_fill(
        oracle_q_safe == float('-inf'), -1e9
    )
    oracle_probs = F.softmax(oracle_q_safe / temperature, dim=-1)   # (B, N) 无零值

    # 学生同样做安全替换，避免 log(0)
    student_q_safe = student_q.float()
    student_q_safe = student_q_safe.masked_fill(
        student_q_safe == float('-inf'), -1e9
    )
    student_log_probs = F.log_softmax(student_q_safe / temperature, dim=-1)

    # KL 散度：学生分布 → Oracle 分布
    kl_loss = F.kl_div(student_log_probs, oracle_probs, reduction='batchmean')
    # 限幅：KL 无上界，少量 bad batch 可以把单步 loss 炸到几十甚至几百，
    # 导致权重被持续污染。10.0 约为正常收敛值的 20x，足以保留真实梯度信号。
    kl_loss = kl_loss.clamp(max=10.0)
    losses["kl"] = kl_loss.item()

    # 行为克隆（硬标签，防止蒸馏偏离真实分布太远）
    bc_loss = F.cross_entropy(student_q, action_idx)
    losses["bc"] = bc_loss.item()

    total = distill_weight * kl_loss + bc_weight * bc_loss
    losses["total"] = total.item()
    return total, losses


# ─────────────────────────────────────────────
# Stage 3：离线 IQL 损失
# ─────────────────────────────────────────────

def iql_loss(
    q: torch.Tensor,              # (B, N) 当前 Q 值
    v: torch.Tensor,              # (B,) 当前 V 值
    v_next: torch.Tensor,         # (B,) 下一状态 V 值（目标网络）
    action_idx: torch.Tensor,     # (B,) 执行的动作
    reward: torch.Tensor,         # (B,) GRP 2.0 shaped reward
    done: torch.Tensor,           # (B,) bool，是否终止
    q_target: torch.Tensor,       # (B,) Q target 值（来自当前状态的 target 网络）
    gamma: float = GAMMA,
    expectile: float = IQL_EXPECTILE,
    cql_weight: float = CQL_WEIGHT,
    v_weight: float = 1.0,
    offline: bool = True,
    reward_clip: float = 20.0,
    value_clip: float = 50.0,
    bc_weight: float = 0.0,
    adv_clip: float = 20.0,
    awr_temperature: float = 3.0,
    awr_max_weight: float = 20.0,
) -> tuple[torch.Tensor, dict]:
    """
    IQL（Implicit Q-Learning）损失 + GRP 2.0 anchored policy term

    三个基础部分：
    1. Q-Loss: Bellman 残差  E[(r + γV(s') - Q(s,a))²]
    2. V-Loss: Expectile 回归，让 V(s) 追踪 Q 的高分位数（τ=0.9）
    3. CQL:    保守约束（离线 RL 防过估计，在线时可以关掉）

    额外新增：
    4. AWR 风格 advantage-weighted BC anchor，防止策略脱离 Stage2 基线过快。
    """
    B = q.shape[0]
    losses = {}

    # 取执行动作的 Q 值
    idx = torch.arange(B, device=q.device)
    q_taken = q[idx, action_idx]   # (B,)

    # 数值保护：reward / value clip，避免 Bellman target 爆炸
    reward = torch.nan_to_num(reward.float(), nan=0.0, posinf=reward_clip, neginf=-reward_clip)
    reward = reward.clamp(-reward_clip, reward_clip)
    v = torch.nan_to_num(v.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
    v_next = torch.nan_to_num(v_next.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
    q_taken = torch.nan_to_num(q_taken.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
    q_target = torch.nan_to_num(q_target.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)

    # 1. Q-Loss: Bellman 残差
    v_next_masked = torch.where(done, torch.zeros_like(v_next), v_next)
    q_target_bellman = (reward + gamma * v_next_masked).detach().clamp(-value_clip, value_clip)
    q_loss = F.mse_loss(q_taken, q_target_bellman)
    losses["q_loss"] = float(q_loss.detach().cpu())

    # 2. V-Loss: Expectile 回归（应使用当前状态 target Q，而不是 next-state Q）
    adv = (q_target.detach() - v).clamp(-value_clip, value_clip)
    v_loss = torch.where(
        adv >= 0,
        expectile       * adv.pow(2),
        (1 - expectile) * adv.pow(2),
    ).mean()
    losses["v_loss"] = float(v_loss.detach().cpu())

    # 3. CQL：对非法动作的 -inf 做屏蔽，否则 logsumexp 直接变 nan/inf
    cql = torch.tensor(0.0, device=q.device)
    if offline and cql_weight > 0:
        q_safe = torch.nan_to_num(q.float(), nan=-1e9, neginf=-1e9, posinf=value_clip)
        cql = q_safe.logsumexp(dim=-1).mean() - q_taken.mean()
        cql = torch.nan_to_num(cql, nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        losses["cql_loss"] = float(cql.detach().cpu())

    # 4. AWR / BC anchor：只在 advantage 高时更强地模仿离线动作
    bc_loss = torch.tensor(0.0, device=q.device)
    if bc_weight > 0:
        q_safe = torch.nan_to_num(q.float(), nan=-1e9, neginf=-1e9, posinf=value_clip)
        log_probs = F.log_softmax(q_safe, dim=-1)
        taken_log_prob = log_probs[idx, action_idx]
        adv_for_policy = (q_target.detach() - v.detach()).clamp(-adv_clip, adv_clip)
        weights = torch.exp(adv_for_policy / max(awr_temperature, 1e-6)).clamp(max=awr_max_weight)
        bc_loss = -(weights * taken_log_prob).mean()
        losses["bc_loss"] = float(bc_loss.detach().cpu())
        losses["awr_weight_mean"] = float(weights.detach().mean().cpu())

    total = q_loss + v_weight * v_loss + cql_weight * cql + bc_weight * bc_loss
    total = torch.nan_to_num(total, nan=1e3, posinf=1e3, neginf=-1e3)
    losses["total"] = float(total.detach().cpu())
    return total, losses


# ─────────────────────────────────────────────
# 工具：期望 Q 目标（用于 V-Loss 中的 q_target）
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_q_target(
    target_model_output,   # RinshanOutput from target model
    action_idx: torch.Tensor,
) -> torch.Tensor:
    """
    从目标网络的输出中提取执行动作的 Q 值
    用于 IQL 的 V-Loss 中的 q_target
    """
    B = action_idx.shape[0]
    q_t = target_model_output.q   # (B, N)
    return q_t[torch.arange(B), action_idx]
