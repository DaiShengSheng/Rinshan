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

from rinshan.constants import GAMMA, IQL_EXPECTILE, CQL_WEIGHT, DISTILL_TEMP, BELIEF_WAIT_WEIGHT


# ─────────────────────────────────────────────
# 公共辅助：Belief + Wait loss（Stage1 和 Stage2 共用）
# ─────────────────────────────────────────────

def belief_and_wait_loss(
    belief_logits: Optional[torch.Tensor],   # (B, 34, 3)
    wait_logits:   Optional[torch.Tensor],   # (B, 34, 3)
    actual_hands:  Optional[torch.Tensor],   # (B, 34, 3)
    opp_wait_tiles: Optional[torch.Tensor],  # (B, 34, 3)
    opp_tenpai_mask: Optional[torch.Tensor], # (B, 3)
    belief_weight: float = 0.3,
    wait_weight: float   = BELIEF_WAIT_WEIGHT,
    belief_pos_weight: float = 1.0,          # 正样本权重，>1 时强调有牌的召回
) -> tuple[torch.Tensor, dict]:
    """
    独立的 Belief + Wait 损失，供 Stage1 和 Stage2 复用。
    返回 (total_loss_tensor, loss_dict)。
    total_loss_tensor 为 0 时表示没有可计算的 loss。
    """
    device = (belief_logits if belief_logits is not None else wait_logits).device
    total = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
    losses: dict = {}

    # 手牌 Belief BCE（支持 pos_weight 平衡正负样本）
    if belief_logits is not None and actual_hands is not None:
        target = (actual_hands > 0).float()
        if belief_pos_weight != 1.0:
            pw = torch.tensor(belief_pos_weight,
                              dtype=belief_logits.dtype,
                              device=belief_logits.device)
            b_loss = F.binary_cross_entropy_with_logits(
                belief_logits, target, pos_weight=pw
            )
        else:
            b_loss = F.binary_cross_entropy_with_logits(belief_logits, target)
        losses["belief"] = b_loss.item()
        total = total + belief_weight * b_loss

    # 待张 Wait BCE（只对 tenpai 对手计算）
    if wait_logits is not None and opp_wait_tiles is not None:
        if opp_tenpai_mask is not None:
            mask = opp_tenpai_mask.unsqueeze(1).expand_as(wait_logits)  # (B, 34, 3)
            if mask.any():
                w_loss = F.binary_cross_entropy_with_logits(
                    wait_logits[mask.bool()],
                    opp_wait_tiles[mask.bool()].float(),
                )
                losses["wait"] = w_loss.item()
                total = total + wait_weight * w_loss
        else:
            w_loss = F.binary_cross_entropy_with_logits(
                wait_logits, opp_wait_tiles.float()
            )
            losses["wait"] = w_loss.item()
            total = total + wait_weight * w_loss

    return total, losses


# ─────────────────────────────────────────────
# Stage 1：行为克隆损失
# ─────────────────────────────────────────────

def stage1_loss(
    q: torch.Tensor,              # (B, N) Q 值 logits
    action_idx: torch.Tensor,     # (B,) 人类选择的动作 index
    belief_logits: Optional[torch.Tensor] = None,
    wait_logits: Optional[torch.Tensor] = None,
    actual_hands: Optional[torch.Tensor] = None,
    opp_wait_tiles: Optional[torch.Tensor] = None,
    opp_tenpai_mask: Optional[torch.Tensor] = None,
    aux_preds: Optional[dict]     = None,
    aux_targets: Optional[dict]   = None,
    belief_weight: float = 0.3,
    wait_weight: float   = BELIEF_WAIT_WEIGHT,
    aux_weight: float    = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    行为克隆损失：BC 主损失 + Belief/Wait 辅助 + AuxHead 辅助
    """
    losses = {}

    # 主损失
    action_loss = F.cross_entropy(q, action_idx)
    losses["action"] = action_loss.item()
    total = action_loss

    # Belief + Wait（复用公共函数）
    has_belief = belief_logits is not None and actual_hands is not None
    has_wait   = wait_logits is not None and opp_wait_tiles is not None
    if has_belief or has_wait:
        bw_loss, bw_dict = belief_and_wait_loss(
            belief_logits=belief_logits if has_belief else None,
            wait_logits=wait_logits if has_wait else None,
            actual_hands=actual_hands,
            opp_wait_tiles=opp_wait_tiles,
            opp_tenpai_mask=opp_tenpai_mask,
            belief_weight=belief_weight,
            wait_weight=wait_weight,
        )
        total = total + bw_loss
        losses.update(bw_dict)

    # 辅助任务损失
    if aux_preds is not None and aux_targets is not None:
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
    q: torch.Tensor,              # (B, N) 当前总 Q 值
    v: torch.Tensor,              # (B,) 当前总 V 值
    v_next: torch.Tensor,         # (B,) 下一状态总 V 值（目标网络）
    action_idx: torch.Tensor,     # (B,) 执行的动作
    reward: torch.Tensor,         # (B,) 总 shaped reward
    done: torch.Tensor,           # (B,) bool，是否终止
    q_target: torch.Tensor,       # (B,) 总 Q target（来自当前状态的 target 网络）
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
    q_game: Optional[torch.Tensor] = None,
    v_game: Optional[torch.Tensor] = None,
    v_next_game: Optional[torch.Tensor] = None,
    reward_game: Optional[torch.Tensor] = None,
    q_target_game: Optional[torch.Tensor] = None,
    q_hand: Optional[torch.Tensor] = None,
    v_hand: Optional[torch.Tensor] = None,
    v_next_hand: Optional[torch.Tensor] = None,
    reward_hand: Optional[torch.Tensor] = None,
    q_target_hand: Optional[torch.Tensor] = None,
    game_expectile: Optional[float] = None,
    hand_expectile: Optional[float] = None,
    game_reward_weight: float = 1.0,
    hand_reward_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    IQL（Implicit Q-Learning）损失 + GRP 2.0 双 value 分解 + anchored policy term

    基础部分：
    1. 总 Q/V Bellman 与 expectile
    2. game 分支：负责长期整场名次价值
    3. hand 分支：负责局内得失点价值
    4. CQL：保守约束
    5. AWR 风格 advantage-weighted BC anchor，防止策略脱离 Stage2 基线过快。
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

    # 3.5 双 value 分支损失（GRP 2.0 完全体）
    branch_total = torch.tensor(0.0, device=q.device)
    if all(x is not None for x in (q_game, v_game, v_next_game, reward_game, q_target_game)):
        q_game_taken = q_game[idx, action_idx]
        reward_game = torch.nan_to_num(reward_game.float(), nan=0.0, posinf=reward_clip, neginf=-reward_clip).clamp(-reward_clip, reward_clip)
        v_game = torch.nan_to_num(v_game.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        v_next_game = torch.nan_to_num(v_next_game.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        q_game_taken = torch.nan_to_num(q_game_taken.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        q_target_game = torch.nan_to_num(q_target_game.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        q_game_bellman = (reward_game * game_reward_weight + gamma * torch.where(done, torch.zeros_like(v_next_game), v_next_game)).detach().clamp(-value_clip, value_clip)
        q_game_loss = F.mse_loss(q_game_taken, q_game_bellman)
        game_tau = expectile if game_expectile is None else game_expectile
        adv_game = (q_target_game.detach() - v_game).clamp(-value_clip, value_clip)
        v_game_loss = torch.where(
            adv_game >= 0,
            game_tau * adv_game.pow(2),
            (1 - game_tau) * adv_game.pow(2),
        ).mean()
        branch_total = branch_total + q_game_loss + v_weight * v_game_loss
        losses["q_game_loss"] = float(q_game_loss.detach().cpu())
        losses["v_game_loss"] = float(v_game_loss.detach().cpu())

    if all(x is not None for x in (q_hand, v_hand, v_next_hand, reward_hand, q_target_hand)):
        q_hand_taken = q_hand[idx, action_idx]
        reward_hand = torch.nan_to_num(reward_hand.float(), nan=0.0, posinf=reward_clip, neginf=-reward_clip).clamp(-reward_clip, reward_clip)
        v_hand = torch.nan_to_num(v_hand.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        v_next_hand = torch.nan_to_num(v_next_hand.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        q_hand_taken = torch.nan_to_num(q_hand_taken.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        q_target_hand = torch.nan_to_num(q_target_hand.float(), nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        q_hand_bellman = (reward_hand * hand_reward_weight + gamma * torch.where(done, torch.zeros_like(v_next_hand), v_next_hand)).detach().clamp(-value_clip, value_clip)
        q_hand_loss = F.mse_loss(q_hand_taken, q_hand_bellman)
        hand_tau = expectile if hand_expectile is None else hand_expectile
        adv_hand = (q_target_hand.detach() - v_hand).clamp(-value_clip, value_clip)
        v_hand_loss = torch.where(
            adv_hand >= 0,
            hand_tau * adv_hand.pow(2),
            (1 - hand_tau) * adv_hand.pow(2),
        ).mean()
        branch_total = branch_total + q_hand_loss + v_weight * v_hand_loss
        losses["q_hand_loss"] = float(q_hand_loss.detach().cpu())
        losses["v_hand_loss"] = float(v_hand_loss.detach().cpu())

    # 4. CQL：对非法动作的 -inf 做屏蔽，否则 logsumexp 直接变 nan/inf
    cql = torch.tensor(0.0, device=q.device)
    if offline and cql_weight > 0:
        q_safe = torch.nan_to_num(q.float(), nan=-1e9, neginf=-1e9, posinf=value_clip)
        cql = q_safe.logsumexp(dim=-1).mean() - q_taken.mean()
        cql = torch.nan_to_num(cql, nan=0.0, posinf=value_clip, neginf=-value_clip).clamp(-value_clip, value_clip)
        losses["cql_loss"] = float(cql.detach().cpu())

    # 5. AWR / BC anchor：只在 advantage 高时更强地模仿离线动作
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

    total = q_loss + v_weight * v_loss + branch_total + cql_weight * cql + bc_weight * bc_loss
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
