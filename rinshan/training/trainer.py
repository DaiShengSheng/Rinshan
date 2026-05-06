"""
Trainer — 统一训练器

支持四个训练阶段，通过 stage 参数切换：
  stage=1: 行为克隆
  stage=2: Oracle 蒸馏
  stage=3: 离线 IQL
  stage=4: 在线 MAPPO（TODO，后续实现）
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from rinshan.model import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.constants import TARGET_EMA_TAU
from .losses import stage1_loss, distill_loss, iql_loss, compute_q_target

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    # ── 通用 ──────────────────────────
    stage: int = 1                  # 训练阶段 1/2/3
    device: str = "cuda"
    dtype: str  = "float32"         # "float32" / "bfloat16"
    amp: bool   = True              # 自动混合精度
    compile: bool = False           # torch.compile 加速（需要 PyTorch >= 2.2）

    # ── 模型 ──────────────────────────
    model_preset: str = "base"      # "nano" / "base" / "large"

    # ── 优化器 ────────────────────────
    lr: float           = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float= 1.0
    grad_accum_steps: int = 1

    # ── LR Schedule ───────────────────
    warmup_steps: int   = 1000
    total_steps: int    = 100_000

    # ── 保存 ──────────────────────────
    save_dir: str = "checkpoints"
    save_every: int = 5000
    log_every:  int = 100

    # ── IQL 专用 ──────────────────────
    target_update_every: int = 100  # 每 N 步做一次目标网络 EMA 更新
    cql_weight: float = -1.0        # <0 表示使用 constants 默认值，>=0 则覆盖
    weights_only_save: bool = False  # True = 只存 model/target 权重，跳过 optimizer/scheduler
    # ── Belief Net 权重（Stage 2 可调）─────
    belief_weight: float = 1.0      # Belief BCE 损失相对蒸馏 loss 的权重
    belief_pos_weight: float = 2.4  # 正样本权重，补偿有牌(29%)/无牌(71%)不均衡

    bc_weight: float = 0.0          # Stage3 anchor：AWR/BC 正则权重
    reward_clip: float = 20.0       # Stage3 GRP 2.0 reward clip
    value_clip: float = 50.0        # Stage3 value/q clip
    adv_clip: float = 20.0          # Stage3 advantage clip for AWR
    awr_temperature: float = 3.0    # Stage3 AWR temperature
    awr_max_weight: float = 20.0    # Stage3 AWR max sample weight
    game_expectile: float = 0.95    # GRP 2.0: game branch expectile
    hand_expectile: float = 0.70    # GRP 2.0: hand branch expectile
    game_reward_weight: float = 1.0 # game branch reward scale
    hand_reward_weight: float = 1.0 # hand branch reward scale


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        transformer_cfg: Optional[TransformerConfig] = None,
    ):
        self.cfg = cfg
        self.step = 0
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ── 设备和精度 ────────────────
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype  = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        self.amp    = cfg.amp and self.device.type == "cuda"

        # ── 模型 ──────────────────────
        if transformer_cfg is None:
            transformer_cfg = TransformerConfig.from_preset(cfg.model_preset)
        # large 模型自动开启 gradient checkpointing，以层重计算换显存
        use_grad_ckpt = (cfg.model_preset == "large")
        if use_grad_ckpt:
            logger.info("Gradient checkpointing enabled (large model)")
        self.model = RinshanModel(
            transformer_cfg=transformer_cfg,
            use_belief=True,
            use_aux=(cfg.stage == 1),
            gradient_checkpointing=use_grad_ckpt,
        ).to(self.device)

        # ── Stage 1：冻结全部 V head（v_game_net / v_hand_net）──────────────
        # 原因：Dueling 架构中 q = v + a - mean(a)，CE loss 对 v 的梯度恒为零。
        # v 完全游离于监督之外，只受 weight_decay / bias 漂移驱动，会跑偏到 -数万。
        # Stage1 只需要学弃牌偏好（a_net），V 值交给 Stage2 用 GRP reward 从零学。
        # 冻结后 migrate_stage1_weights 的"v 归零"策略也能永久保持，无需每次手动处理。
        if cfg.stage in (1, 2):
            for name, param in self.model.named_parameters():
                if "qv_head.v_game_net" in name or "qv_head.v_hand_net" in name:
                    param.requires_grad_(False)
            if cfg.stage == 1:
                logger.info(
                    "Stage 1: v_game_net and v_hand_net frozen "
                    "(V-value learning deferred to Stage 2)"
                )
            else:
                logger.info(
                    "Stage 2: v_game_net and v_hand_net frozen "
                    "(distill_loss gradient on v is always zero; "
                    "V-value learning deferred to Stage 3 IQL)"
                )

        # 目标网络（Stage 3 用）
        self.target_model: Optional[RinshanModel] = None
        if cfg.stage == 3:
            # 注意：必须在 torch.compile 之前 deepcopy，避免 OptimizedModule 深拷贝失败
            self.target_model = copy.deepcopy(self.model)
            self.target_model.requires_grad_(False)

        # torch.compile
        # grad checkpointing 开启时不能 compile：
        #   - compile 整个模型：AOT autograd 穿透 checkpoint 边界，把 24 层内入一个大 kernel -> OOM
        #   - per-block compile：masked_fill data-dependent op 导致 fullgraph=True 持续重编译
        # 结论：有 grad_ckpt 时不用 compile；grad_ckpt 本身已等效于 -30% 成本
        if cfg.compile and self.device.type == "cuda" and not use_grad_ckpt:
            logger.info("Compiling model with torch.compile (mode=default)...")
            self.model = torch.compile(self.model, mode="default", fullgraph=False)
        elif cfg.compile and use_grad_ckpt:
            logger.info("torch.compile skipped (incompatible with gradient checkpointing, see comment)")

        # Oracle 模型（Stage 2 用，外部传入）
        self.oracle_model: Optional[RinshanModel] = None

        # ── 优化器 ────────────────────
        # 权重衰减只对 weight，不对 bias 和 norm
        decay_params    = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".weight") and "norm" not in name and "embed" not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params,    "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # LR Schedule: Linear warmup → Cosine decay
        warmup_scheduler   = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=cfg.warmup_steps
        )
        cosine_scheduler   = CosineAnnealingLR(
            self.optimizer, T_max=cfg.total_steps - cfg.warmup_steps, eta_min=cfg.lr * 0.1
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.warmup_steps],
        )

        # GradScaler: only needed for fp16; bf16 has sufficient dynamic range
        scaler_enabled = self.amp and (self.dtype == torch.float16)
        self.scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)

        # 参数统计
        param_counts = self.model.count_parameters()
        logger.info(f"Model parameters: {param_counts}")

    def _to_device(self, value):
        """安全地把可选 Tensor/字典移动到 device。"""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        return value

    def set_oracle_model(self, oracle: RinshanModel):
        """Stage 2 时传入预训练好的 Oracle 模型"""
        self.oracle_model = oracle.to(self.device).eval()
        self.oracle_model.requires_grad_(False)

    def _forward_and_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """根据当前 stage 计算前向传播和损失"""
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.amp,
            cache_enabled=False,   # 禁止 AMP kernel cache 占用额外显存
        ):
            # 通用前向传播
            out = self.model(
                tokens=self._to_device(batch["tokens"]),
                candidate_mask=self._to_device(batch["candidate_mask"]),
                pad_mask=self._to_device(batch.get("pad_mask")),
                belief_tokens=self._to_device(batch.get("belief_tokens")),
                belief_pad_mask=self._to_device(batch.get("belief_pad_mask")),
                compute_aux=(self.cfg.stage == 1),
            )

            action_idx = batch["action_idx"].to(self.device)
            reward = batch.get("reward")
            if reward is not None:
                reward = reward.to(self.device).float()
            reward_game = batch.get("reward_game")
            if reward_game is not None:
                reward_game = reward_game.to(self.device).float()
            reward_hand = batch.get("reward_hand")
            if reward_hand is not None:
                reward_hand = reward_hand.to(self.device).float()

            # ── Stage 1: 行为克隆 ────
            if self.cfg.stage == 1:
                aux_t = batch.get("aux_targets", {})
                return stage1_loss(
                    q=out.q,
                    action_idx=action_idx,
                    belief_logits=out.belief_logits,
                    wait_logits=out.wait_logits,
                    actual_hands=(self._to_device(batch.get("actual_hands")).float()
                                 if batch.get("actual_hands") is not None else None),
                    opp_wait_tiles=(self._to_device(aux_t["opp_wait_tiles"]).float()
                                    if "opp_wait_tiles" in aux_t else None),
                    opp_tenpai_mask=(self._to_device(aux_t["opp_tenpai_mask"]).float()
                                     if "opp_tenpai_mask" in aux_t else None),
                    aux_preds=out.aux_preds,
                    aux_targets={k: self._to_device(v)
                                 for k, v in aux_t.items()
                                 if k not in ("opp_wait_tiles", "opp_tenpai_mask")},
                    belief_weight=getattr(self.cfg, 'belief_weight', 1.0),
                    belief_pos_weight=getattr(self.cfg, 'belief_pos_weight', 2.4),
                )

            # ── Stage 2: Oracle 蒸馏 ──
            elif self.cfg.stage == 2:
                assert self.oracle_model is not None, "Oracle model not set for Stage 2"
                with torch.no_grad():
                    oracle_out = self.oracle_model(
                        tokens=self._to_device(batch["oracle_tokens"]),
                        candidate_mask=self._to_device(batch["candidate_mask"]),
                        pad_mask=self._to_device(batch.get("oracle_pad_mask")),
                    )
                total, losses = distill_loss(
                    student_q=out.q,
                    oracle_q=oracle_out.q,
                    action_idx=action_idx,
                )

                # ── Belief + Wait 辅助损失（Stage2 顺带训练，让 wait_head 收敛）──
                # belief_weight 降低，避免盖过蒸馏主损失
                from .losses import belief_and_wait_loss
                from rinshan.constants import BELIEF_WAIT_WEIGHT

                aux_t = batch.get("aux_targets", {})
                actual_hands  = batch.get("actual_hands")
                opp_wait      = aux_t.get("opp_wait_tiles")
                opp_mask      = aux_t.get("opp_tenpai_mask")
                has_belief    = out.belief_logits is not None and actual_hands is not None
                has_wait      = out.wait_logits is not None and opp_wait is not None

                if has_belief or has_wait:
                    # pos_weight=2.4：对手每张牌实际有牌频率≈29%，无牌≈71%
                    # 正负比 ≈ 0.71/0.29 ≈ 2.4，用 pos_weight 平衡 BCE
                    # belief_weight=1.0：让 Belief 梯度与蒸馏 loss 同量级
                    # wait_weight 在 Stage2 设小一点（yaml 里 wait_weight=0.1），
                    # 避免从零初始化的 wait_head 抢走 belief 的梯度
                    bw_loss, bw_dict = belief_and_wait_loss(
                        belief_logits  = out.belief_logits if has_belief else None,
                        wait_logits    = out.wait_logits   if has_wait   else None,
                        actual_hands   = self._to_device(actual_hands).float() if has_belief else None,
                        opp_wait_tiles = self._to_device(opp_wait).float()     if has_wait   else None,
                        opp_tenpai_mask= self._to_device(opp_mask).float() if opp_mask is not None else None,
                        belief_weight  = getattr(self.cfg, 'belief_weight', 1.0),
                        wait_weight    = getattr(self.cfg, 'wait_weight', BELIEF_WAIT_WEIGHT),
                        belief_pos_weight = getattr(self.cfg, 'belief_pos_weight', 2.4),
                    )
                    total = total + bw_loss
                    losses.update(bw_dict)

                losses["total"] = total.item()
                return total, losses

            # ── Stage 3: 离线 IQL ─────
            elif self.cfg.stage == 3:
                assert self.target_model is not None
                with torch.no_grad():
                    # next state target，用于 Bellman 目标 r + γV(s')
                    next_out = self.target_model(
                        tokens=self._to_device(batch["next_tokens"]),
                        candidate_mask=self._to_device(batch["next_candidate_mask"]),
                        pad_mask=self._to_device(batch.get("next_pad_mask")),
                        belief_tokens=self._to_device(batch.get("next_belief_tokens")),
                        belief_pad_mask=self._to_device(batch.get("next_belief_pad_mask")),
                    )
                    # current state target Q，用于 expectile V-loss
                    target_out = self.target_model(
                        tokens=self._to_device(batch["tokens"]),
                        candidate_mask=self._to_device(batch["candidate_mask"]),
                        pad_mask=self._to_device(batch.get("pad_mask")),
                        belief_tokens=self._to_device(batch.get("belief_tokens")),
                        belief_pad_mask=self._to_device(batch.get("belief_pad_mask")),
                    )
                    q_target_val = compute_q_target(target_out, action_idx)
                    q_target_game = target_out.q_game[torch.arange(action_idx.shape[0], device=action_idx.device), action_idx]
                    q_target_hand = target_out.q_hand[torch.arange(action_idx.shape[0], device=action_idx.device), action_idx]

                done = self._to_device(batch.get("done", batch.get("is_done"))).bool()

                # 在线场景允许通过 batch["_online_cql_weight"] 覆盖 CQL 权重
                cql_w = batch.get("_online_cql_weight", None)
                extra_kwargs = {}
                if cql_w is not None:
                    extra_kwargs["cql_weight"] = float(cql_w)
                    extra_kwargs["offline"] = False
                elif self.cfg.cql_weight >= 0:
                    extra_kwargs["cql_weight"] = self.cfg.cql_weight  # yaml 覆盖

                total, losses = iql_loss(
                    q=out.q,
                    v=out.v,
                    v_next=next_out.v,
                    action_idx=action_idx,
                    reward=reward,
                    done=done,
                    q_target=q_target_val,
                    bc_weight=self.cfg.bc_weight,
                    reward_clip=self.cfg.reward_clip,
                    value_clip=self.cfg.value_clip,
                    adv_clip=self.cfg.adv_clip,
                    awr_temperature=self.cfg.awr_temperature,
                    awr_max_weight=self.cfg.awr_max_weight,
                    q_game=out.q_game,
                    v_game=out.v_game,
                    v_next_game=next_out.v_game,
                    reward_game=reward_game,
                    q_target_game=q_target_game,
                    q_hand=out.q_hand,
                    v_hand=out.v_hand,
                    v_next_hand=next_out.v_hand,
                    reward_hand=reward_hand,
                    q_target_hand=q_target_hand,
                    game_expectile=self.cfg.game_expectile,
                    hand_expectile=self.cfg.hand_expectile,
                    game_reward_weight=self.cfg.game_reward_weight,
                    hand_reward_weight=self.cfg.hand_reward_weight,
                    **extra_kwargs,
                )

                # ── Stage3 Belief + Wait 辅助损失 ────────────────────────────
                # Stage2 用 wait_weight=0.1 预热了 wait_head，Stage3 放开到 0.5
                from .losses import belief_and_wait_loss
                from rinshan.constants import BELIEF_WAIT_WEIGHT
                aux_t        = batch.get("aux_targets", {})
                actual_hands = batch.get("actual_hands")
                opp_wait     = aux_t.get("opp_wait_tiles")
                opp_mask     = aux_t.get("opp_tenpai_mask")
                has_belief   = out.belief_logits is not None and actual_hands is not None
                has_wait     = out.wait_logits   is not None and opp_wait     is not None
                if has_belief or has_wait:
                    bw_loss, bw_dict = belief_and_wait_loss(
                        belief_logits   = out.belief_logits if has_belief else None,
                        wait_logits     = out.wait_logits   if has_wait   else None,
                        actual_hands    = self._to_device(actual_hands).float() if has_belief else None,
                        opp_wait_tiles  = self._to_device(opp_wait).float()     if has_wait   else None,
                        opp_tenpai_mask = self._to_device(opp_mask).float() if opp_mask is not None else None,
                        belief_weight   = getattr(self.cfg, 'belief_weight', 1.0),
                        wait_weight     = getattr(self.cfg, 'wait_weight', BELIEF_WAIT_WEIGHT),
                        belief_pos_weight = getattr(self.cfg, 'belief_pos_weight', 2.4),
                    )
                    total = total + bw_loss
                    losses.update(bw_dict)

                losses["total"] = total.item()
                return total, losses

            else:
                raise ValueError(f"Stage {self.cfg.stage} not implemented in Trainer")

    def train_step(self, batch: dict) -> dict:
        """执行一步训练，返回 loss 字典"""
        self.model.train()
        loss, loss_dict = self._forward_and_loss(batch)

        # 梯度累积
        loss_scaled = loss / self.cfg.grad_accum_steps
        self.scaler.scale(loss_scaled).backward()

        if (self.step + 1) % self.cfg.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                logger.warning("Non-finite grad norm detected; skipping optimizer step")
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

        self.step += 1

        # 目标网络 EMA 软更新（Stage 3）
        if self.cfg.stage == 3 and self.step % self.cfg.target_update_every == 0:
            self._update_target_network()

        # 定期日志
        if self.step % self.cfg.log_every == 0:
            lr = self.scheduler.get_last_lr()[0]
            if self.cfg.stage == 2:
                # Stage 2 专用：显示蒸馏核心分量 + belief/wait 辅助
                s2_keys = [("kl", "kl"), ("bc", "bc"), ("belief", "bel"),
                           ("wait", "wait"), ("total", "total")]
                parts = "  ".join(
                    f"{short}={loss_dict[k]:.4f}"
                    for k, short in s2_keys if k in loss_dict
                )
                logger.info(f"[step {self.step}] {parts}  lr={lr:.2e}")
            elif self.cfg.stage == 3:
                # Stage 3 专用：显示 IQL 主分量 + belief/wait 辅助
                s3_keys = [("q_loss", "q"), ("v_loss", "v"), ("bc", "bc"),
                           ("cql", "cql"), ("belief", "bel"), ("wait", "wait"),
                           ("total", "total")]
                parts3 = "  ".join(
                    f"{short}={loss_dict[k]:.4f}"
                    for k, short in s3_keys if k in loss_dict
                )
                logger.info(f"[step {self.step}] {parts3}  lr={lr:.2e}")
            else:
                # Stage 1 原有逻辑
                aux_keys = ["action", "belief", "aux_shanten", "aux_tenpai_prob",
                            "aux_deal_in_risk", "aux_opp_tenpai"]
                aux_parts = "  ".join(
                    f"{k.replace('aux_','')}={loss_dict[k]:.3f}"
                    for k in aux_keys if k in loss_dict
                )
                logger.info(
                    f"[step {self.step}] loss={loss_dict['total']:.4f}  lr={lr:.2e}"
                    + (f"  | {aux_parts}" if aux_parts else "")
                )

        # 定期保存
        if self.step % self.cfg.save_every == 0:
            self.save(self.save_dir / f"checkpoint_{self.step}.pt")

        return loss_dict

    def _update_target_network(self):
        """EMA 软更新目标网络"""
        tau = TARGET_EMA_TAU
        with torch.no_grad():
            for p_src, p_tgt in zip(
                self.model.parameters(), self.target_model.parameters()
            ):
                p_tgt.data.mul_(1 - tau).add_(tau * p_src.data)

    def save(self, path: Path):
        if self.cfg.weights_only_save:
            payload = {
                "step":         self.step,
                "stage":        self.cfg.stage,
                "model":        self.model.state_dict(),
                "target_model": self.target_model.state_dict()
                                if self.target_model else None,
            }
        else:
            payload = {
                "step":         self.step,
                "stage":        self.cfg.stage,
                "model":        self.model.state_dict(),
                "target_model": self.target_model.state_dict()
                                if self.target_model else None,
                "optimizer":    self.optimizer.state_dict(),
                "scheduler":    self.scheduler.state_dict(),
                "scaler":       self.scaler.state_dict(),
            }
        # 把训练 lr 存进去，重启时自动检测是否需要 reset
        payload["lr"] = self.cfg.lr
        torch.save(payload, path)
        logger.info(f"Saved checkpoint → {path}")

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"])
        if self.target_model and ckpt.get("target_model"):
            self.target_model.load_state_dict(ckpt["target_model"])
        ckpt_lr = ckpt.get("lr", None)
        if ckpt_lr is not None and abs(ckpt_lr - self.cfg.lr) > 1e-12:
            # lr 发生变化：只恢复权重，optimizer/scheduler 用新 lr 重建
            logger.info(
                f"lr changed ({ckpt_lr:.2e} → {self.cfg.lr:.2e}): "
                f"weights loaded, optimizer/scheduler reset"
            )
        else:
            # lr 一致：完整恢复，保证续训连续
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
        logger.info(f"Loaded checkpoint ← {path} (step {self.step}, lr={self.cfg.lr:.2e})")
