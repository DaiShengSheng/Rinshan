"""
train_stage3.py — Stage 3: 离线 IQL 强化学习

在 Stage 1/2 的基础上，用 GRP 奖励信号做离线 RL 精调。
需要先跑完：
  1. train_grp.py      → 得到 GRP 模型
  2. fill_grp_rewards.py → 为标注数据填入 grp_reward 字段
  3. train_stage1.py   → 得到 Stage 1 初始化权重

Usage:
    python scripts/train_stage3.py configs/stage3_base.yaml \
        --stage1_ckpt checkpoints/stage1_base/best.pt
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.utils.config import load_config
from rinshan.data         import MjaiDataset, collate_fn
from rinshan.training     import Trainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stage3")


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_stage3.py <config.yaml> --stage1_ckpt <path>")
        sys.exit(1)

    cfg = load_config(sys.argv[1], sys.argv[2:])
    logger.info(f"Config: {cfg}")

    stage1_ckpt = cfg.get("stage1_ckpt", "")

    # ── 数据（Stage 3 需要 s,a,r,s' 对）────────
    data_dir = Path(cfg["data_dir"])
    all_files = sorted(data_dir.rglob("*.jsonl"))

    import random
    random.seed(42)
    random.shuffle(all_files)
    val_n = max(1, int(len(all_files) * cfg.get("val_ratio", 0.02)))
    train_files = all_files[val_n:]
    val_files   = all_files[:val_n]

    batch_size = cfg.get("batch_size", 64)
    train_ds = MjaiDataset(train_files, shuffle_files=True,
                           shuffle_buffer=cfg.get("shuffle_buffer", 2000),
                           stage=3)  # stage=3 会自动配对 (s, s')
    val_ds   = MjaiDataset(val_files, shuffle_files=False,
                           shuffle_buffer=500, stage=3)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2,
                              collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)

    # ── Trainer（Stage 3）─────────────────────
    trainer_cfg = TrainerConfig(
        stage            = 3,
        device           = cfg.get("device", "cuda"),
        dtype            = cfg.get("dtype", "float32"),
        amp              = cfg.get("amp", False),
        model_preset     = cfg.get("model_preset", "base"),
        lr               = float(cfg.get("lr", 1e-4)),
        weight_decay     = float(cfg.get("weight_decay", 0.01)),
        warmup_steps     = int(cfg.get("warmup_steps", 500)),
        total_steps      = int(cfg.get("total_steps", 100_000)),
        save_dir         = cfg.get("save_dir", "checkpoints/stage3"),
        save_every       = int(cfg.get("save_every", 5000)),
        log_every        = int(cfg.get("log_every", 100)),
        target_update_every = int(cfg.get("target_update_every", 100)),
    )
    trainer = Trainer(trainer_cfg)
    device  = trainer.device

    # 从 Stage 1/2 加载权重
    if stage1_ckpt and Path(stage1_ckpt).exists():
        logger.info(f"Loading Stage 1 weights from {stage1_ckpt}")
        s1_ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=True)
        trainer.model.load_state_dict(s1_ckpt["model"], strict=False)
        # 目标网络也初始化为相同权重
        trainer.target_model.load_state_dict(s1_ckpt["model"], strict=False)
        logger.info("Target network initialized with Stage 1 weights (strict=False)")
    else:
        logger.warning("No stage1_ckpt provided, training from scratch")

    ckpt_dir  = Path(trainer_cfg.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(cfg.get("total_steps", 100_000))
    val_every   = int(cfg.get("val_every", 2000))
    best_val_loss = float("inf")

    # 检查 grp_reward 是否存在
    # 取第一个样本验证
    _checked = False

    logger.info(f"Starting Stage 3 (IQL) for {total_steps} steps")

    step = 0
    for batch in train_loader:
        if step >= total_steps:
            break

        # 首次检查 reward 是否已填入
        if not _checked:
            rewards = batch.get("reward")
            if rewards is not None:
                avg_r = rewards.float().mean().item()
                if abs(avg_r) < 1e-6:
                    logger.warning(
                        "All grp_reward values are ~0. "
                        "Run fill_grp_rewards.py first for better training signal."
                    )
            _checked = True

        loss_dict = trainer.train_step(batch)
        step = trainer.step

        if step % val_every == 0:
            # 验证集 IQL 损失
            trainer.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for vb in val_loader:
                    _, ld = trainer._forward_and_loss(vb)
                    val_loss += ld["total"]
                    n_val += 1
                    if n_val >= 20:
                        break
            trainer.model.train()
            val_loss /= max(n_val, 1)
            logger.info(f"[val step={step}] val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save(ckpt_dir / "best.pt")

    logger.info("Stage 3 complete")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
