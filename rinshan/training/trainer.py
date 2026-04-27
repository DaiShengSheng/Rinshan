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

            # ── Stage 1: 行为克隆 ────
            if self.cfg.stage == 1:
                return stage1_loss(
                    q=out.q,
                    action_idx=action_idx,
                    belief_logits=out.belief_logits,
                    actual_hands=(self._to_device(batch.get("actual_hands")).float()
                                 if batch.get("actual_hands") is not None else None),
                    aux_preds=out.aux_preds,
                    aux_targets={k: self._to_device(v)
                                 for k, v in batch.get("aux_targets", {}).items()},
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
                return distill_loss(
                    student_q=out.q,
                    oracle_q=oracle_out.q,
                    action_idx=action_idx,
                )

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

                done = self._to_device(batch.get("done", batch.get("is_done"))).bool()

                # 在线场景允许通过 batch["_online_cql_weight"] 覆盖 CQL 权重
                cql_w = batch.get("_online_cql_weight", None)
                extra_kwargs = {}
                if cql_w is not None:
                    extra_kwargs["cql_weight"] = float(cql_w)
                    extra_kwargs["offline"] = False

                return iql_loss(
                    q=out.q,
                    v=out.v,
                    v_next=next_out.v,
                    action_idx=action_idx,
                    reward=reward,
                    done=done,
                    q_target=q_target_val,
                    **extra_kwargs,
                )

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
            logger.info(
                f"[step {self.step}] loss={loss_dict['total']:.4f}  lr={lr:.2e}"
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
        torch.save(
            {
                "step": self.step,
                "stage": self.cfg.stage,
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict()
                                if self.target_model else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            path,
        )
        logger.info(f"Saved checkpoint → {path}")

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"])
        if self.target_model and ckpt.get("target_model"):
            self.target_model.load_state_dict(ckpt["target_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        logger.info(f"Loaded checkpoint ← {path} (step {self.step})")
