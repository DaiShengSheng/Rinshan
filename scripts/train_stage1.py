"""
train_stage1.py — Stage 1: 行为克隆训练

基于凤凰桌高段位玩家牌谱做监督学习，让模型学会"人类的打法"。

Usage:
    # 用默认配置快速验证（nano 模型）
    python scripts/train_stage1.py configs/stage1_nano.yaml

    # 正式训练（base 模型）
    python scripts/train_stage1.py configs/stage1_base.yaml

    # 覆盖配置项
    python scripts/train_stage1.py configs/stage1_base.yaml --lr 1e-4 --batch_size 64
"""
from __future__ import annotations

import logging
import os
import sys
import random
from pathlib import Path

# 必须在 import torch 之前设置，不然不生效
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader

# 把项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.utils.config   import load_config
from rinshan.model          import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.data           import MjaiDataset, collate_fn
from rinshan.training       import Trainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_stage1")


# ─────────────────────────────────────────────
# 验证集评估
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(trainer: Trainer, val_loader: DataLoader) -> dict:
    """跑一遍验证集，返回平均 loss 和 top-1 accuracy"""
    trainer.model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in val_loader:
        out = trainer.model(
            tokens          = batch["tokens"].to(trainer.device),
            candidate_mask  = batch["candidate_mask"].to(trainer.device),
            pad_mask        = batch["pad_mask"].to(trainer.device) if batch.get("pad_mask") is not None else None,
            belief_tokens   = batch["belief_tokens"].to(trainer.device) if batch.get("belief_tokens") is not None else None,
            belief_pad_mask = batch["belief_pad_mask"].to(trainer.device) if batch.get("belief_pad_mask") is not None else None,
            compute_aux=False,
        )
        from rinshan.training.losses import stage1_loss
        loss, _ = stage1_loss(
            q          = out.q,
            action_idx = batch["action_idx"].to(trainer.device),
        )
        total_loss += loss.item() * len(batch["action_idx"])

        # Top-1 accuracy（只看合法候选里的 argmax）
        q_valid = out.q.clone()
        q_valid[~batch["candidate_mask"].to(trainer.device)] = float("-inf")
        pred    = q_valid.argmax(dim=-1)
        target  = batch["action_idx"].to(trainer.device)
        total_correct += (pred == target).sum().item()
        total_samples += len(target)

    trainer.model.train()
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_acc":  total_correct / max(total_samples, 1),
    }


# ─────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────

def main():
    # ── 加载配置 ─────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python train_stage1.py <config.yaml> [--key value ...]")
        sys.exit(1)

    cfg = load_config(sys.argv[1], sys.argv[2:])
    logger.info(f"Config: {cfg}")

    # ── 构建数据集 ────────────────────────────
    data_dir = Path(cfg["data_dir"])
    all_files = sorted(data_dir.rglob("*.jsonl"))
    if not all_files:
        logger.error(f"No .jsonl files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(all_files)} annotation files")

    # 分割训练/验证
    random.seed(42)
    random.shuffle(all_files)
    val_n    = max(1, int(len(all_files) * cfg.get("val_ratio", 0.02)))
    val_files  = all_files[:val_n]
    train_files= all_files[val_n:]
    logger.info(f"Train files: {len(train_files)} | Val files: {len(val_files)}")

    train_ds = MjaiDataset(
        train_files,
        shuffle_files  = True,
        shuffle_buffer = cfg.get("shuffle_buffer", 2000),
        stage=1,
    )
    val_ds = MjaiDataset(
        val_files,
        shuffle_files  = False,
        shuffle_buffer = 500,
        stage=1,
    )

    batch_size  = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = cfg.get("pin_memory", True),
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = min(batch_size, 128),   # 验证不需要大 batch，固定上限防 OOM
        collate_fn  = collate_fn,
        num_workers = min(num_workers, 4),
        pin_memory  = cfg.get("pin_memory", True),
        persistent_workers = (num_workers > 0),
    )

    # ── 构建 Trainer ──────────────────────────
    trainer_cfg = TrainerConfig(
        stage            = 1,
        device           = cfg.get("device", "cuda"),
        dtype            = cfg.get("dtype", "float32"),
        amp              = cfg.get("amp", False),
        compile          = cfg.get("compile", False),
        model_preset     = cfg.get("model_preset", "base"),
        lr               = float(cfg.get("lr", 3e-4)),
        weight_decay     = float(cfg.get("weight_decay", 0.1)),
        max_grad_norm    = float(cfg.get("max_grad_norm", 1.0)),
        grad_accum_steps = int(cfg.get("grad_accum_steps", 1)),
        warmup_steps     = int(cfg.get("warmup_steps", 1000)),
        total_steps      = int(cfg.get("total_steps", 100_000)),
        save_dir         = cfg.get("save_dir", "checkpoints/stage1"),
        save_every       = int(cfg.get("save_every", 5000)),
        log_every        = int(cfg.get("log_every", 100)),
    )
    trainer = Trainer(trainer_cfg)

    # 尝试加载最新 checkpoint（断点续跑优先）
    ckpt_dir = Path(trainer_cfg.save_dir)
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if ckpts:
        trainer.load(ckpts[-1])
        logger.info(f"Resumed from {ckpts[-1]}")
    elif cfg.get("init_ckpt"):
        # 无断点时，从指定初始 checkpoint 热启动（只加载权重，optimizer 重建）
        init_path = Path(cfg["init_ckpt"])
        ckpt = torch.load(init_path, map_location=trainer.device, weights_only=True)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        # torch.compile 会把模型包成 OptimizedModule，真正的 nn.Module 在 _orig_mod 下
        raw_model = getattr(trainer.model, "_orig_mod", trainer.model)
        raw_model.load_state_dict(state, strict=True)
        logger.info(f"Warm-started from {init_path} (optimizer reset, step=0)")

    val_every   = int(cfg.get("val_every", 2000))
    total_steps = int(cfg.get("total_steps", 100_000))

    best_val_loss = float("inf")
    logger.info(f"Starting Stage 1 training for {total_steps} steps")
    logger.info(f"  Model: {trainer_cfg.model_preset} "
                f"({trainer.model.count_parameters()['total']:,} params)")

    # ── 训练主循环 ────────────────────────────
    step = trainer.step
    for batch in train_loader:
        if step >= total_steps:
            break

        loss_dict = trainer.train_step(batch)
        step = trainer.step

        # 梯度累积完成后主动释放显存碎片
        if step % trainer_cfg.grad_accum_steps == 0:
            torch.cuda.empty_cache()

        # 验证
        if step % val_every == 0:
            val_metrics = evaluate(trainer, val_loader)
            logger.info(
                f"[val step={step}] "
                f"loss={val_metrics['val_loss']:.4f}  "
                f"acc={val_metrics['val_acc']:.3f}"
            )
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                trainer.save(ckpt_dir / "best.pt")
                logger.info(f"  ↑ Best model saved (val_loss={best_val_loss:.4f})")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
