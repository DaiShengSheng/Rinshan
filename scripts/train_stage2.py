"""
train_stage2.py — Stage 2: Oracle 蒸馏

用"看全牌的 Oracle 模型"指导学生模型（只看公开信息）。
天凤牌谱包含完整的全流程信息（含对手手牌），因此可直接使用真 Oracle 蒸馏。

Oracle 模型以全信息序列（己方手牌 + 三家对手手牌）作为输入，
输出软标签 Q 分布指导学生模型，使学生在只能观测公开信息的条件下
尽量逼近全知视角下的决策分布。

Usage:
    python scripts/train_stage2.py configs/stage2_base.yaml --stage1_ckpt checkpoints/stage1_base/best.pt
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.utils.config    import load_config
from rinshan.model           import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.data            import MjaiDataset, collate_fn
from rinshan.training        import Trainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stage2")


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_stage2.py <config.yaml> --stage1_ckpt <path>")
        sys.exit(1)

    cfg = load_config(sys.argv[1], sys.argv[2:])
    logger.info(f"Config: {cfg}")

    stage1_ckpt = cfg.get("stage1_ckpt", "")
    if not stage1_ckpt:
        logger.error("Must specify stage1_ckpt in config or via --stage1_ckpt")
        sys.exit(1)

    # ── 数据 ─────────────────────────────────
    data_dir = Path(cfg["data_dir"])
    all_files = sorted(data_dir.rglob("*.jsonl"))

    import random
    random.seed(42)
    random.shuffle(all_files)
    val_n = max(1, int(len(all_files) * cfg.get("val_ratio", 0.02)))
    train_files = all_files[val_n:]
    val_files   = all_files[:val_n]

    train_ds = MjaiDataset(train_files, shuffle_files=True,
                           shuffle_buffer=cfg.get("shuffle_buffer", 2000), stage=2)
    val_ds   = MjaiDataset(val_files, shuffle_files=False,
                           shuffle_buffer=500, stage=2)

    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2,
                              collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)

    # ── Trainer（Stage 2）─────────────────────
    trainer_cfg = TrainerConfig(
        stage            = 2,
        device           = cfg.get("device", "cuda"),
        dtype            = cfg.get("dtype", "float32"),
        amp              = cfg.get("amp", False),
        compile          = cfg.get("compile", False),
        model_preset     = cfg.get("model_preset", "base"),
        lr               = float(cfg.get("lr", 1e-4)),
        weight_decay     = float(cfg.get("weight_decay", 0.01)),
        max_grad_norm    = float(cfg.get("max_grad_norm", 1.0)),
        grad_accum_steps = int(cfg.get("grad_accum_steps", 1)),
        warmup_steps     = int(cfg.get("warmup_steps", 500)),
        total_steps      = int(cfg.get("total_steps", 50_000)),
        save_dir         = cfg.get("save_dir", "checkpoints/stage2"),
        save_every       = int(cfg.get("save_every", 5000)),
        log_every        = int(cfg.get("log_every", 100)),
    )
    trainer = Trainer(trainer_cfg)
    device  = trainer.device

    # 从 Stage 1 加载权重（学生网络初始化）
    logger.info(f"Loading Stage 1 weights from {stage1_ckpt}")
    s1_ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=True)
    # 兼容两种键名：migrate 脚本用 "model"，旧格式可能是 "model_state_dict"
    raw_sd = s1_ckpt.get("model", s1_ckpt.get("model_state_dict", s1_ckpt))
    # torch.compile 保存的权重带 _orig_mod. 前缀，加载前剥掉
    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}
    missing, unexpected = trainer.model.load_state_dict(raw_sd, strict=False)
    logger.info(f"Student loaded — missing: {len(missing)}  unexpected: {len(unexpected)}")
    if missing:
        logger.warning(f"  missing keys: {missing[:8]}")

    # Oracle：全信息模型，接收含对手手牌的更长序列
    # 数据里 opponent_hands 已由 parse_tenhou 填充，使用真正的 Oracle 蒸馏
    logger.info("Oracle = full-information model (true oracle distillation)")
    from rinshan.constants import MAX_ORACLE_SEQ_LEN
    oracle_transformer_cfg = TransformerConfig.from_preset(cfg.get("model_preset", "base"))
    oracle_transformer_cfg.max_seq_len = MAX_ORACLE_SEQ_LEN   # Oracle 序列更长，含对手手牌
    oracle_model = RinshanModel(
        transformer_cfg=oracle_transformer_cfg,
        use_belief=False,  # Oracle 已看全牌，不需要 Belief Net
        use_aux=False,
    )
    # Oracle 加载相同的 Stage 1 权重（共享 encoder，RoPE 不含绝对位置 embed，长度 mismatch 安全）
    oracle_missing, oracle_unexpected = oracle_model.load_state_dict(raw_sd, strict=False)
    logger.info(f"Oracle loaded — missing: {len(oracle_missing)}  unexpected: {len(oracle_unexpected)}")
    trainer.set_oracle_model(oracle_model)

    ckpt_dir  = Path(trainer_cfg.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(cfg.get("total_steps", 50_000))
    val_every   = int(cfg.get("val_every", 1000))
    best_val_loss = float("inf")
    patience = int(cfg.get("patience", 5))   # 连续 N 次 val 不改善则停止
    patience_counter = 0

    # ── Resume from checkpoint ────────────────
    resume_ckpt = cfg.get("resume_ckpt", "")
    if not resume_ckpt:
        # 优先用 best.pt，没有再找编号最大的 checkpoint
        best_pt = ckpt_dir / "best.pt"
        if best_pt.exists():
            resume_ckpt = str(best_pt)
        else:
            # 按 step 数字排序，取最大编号（避免字符串排序 5000 > 40000 的问题）
            existing = sorted(
                ckpt_dir.glob("checkpoint_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1])
            )
            if existing:
                resume_ckpt = str(existing[-1])
    if resume_ckpt and Path(resume_ckpt).exists():
        logger.info(f"Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=True)
        # 剥 _orig_mod. 前缀（如有）
        sd = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        trainer.model.load_state_dict(sd, strict=False)
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.scheduler.load_state_dict(ckpt["scheduler"])
        trainer.step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed at step {trainer.step}, best_val_loss={best_val_loss:.4f}")

    logger.info(f"Starting Stage 2 (self-distillation) for {total_steps} steps")

    step = 0
    for batch in train_loader:
        if step >= total_steps:
            break

        loss_dict = trainer.train_step(batch)
        step = trainer.step

        if step % val_every == 0:
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
            logger.info(f"[val step={step}] val_loss={val_loss:.4f}  best={best_val_loss:.4f}  patience={patience_counter}/{patience}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 将 best_val_loss 一并存进 checkpoint
                torch.save(
                    {
                        "step": trainer.step,
                        "stage": 2,
                        "model": trainer.model.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "scheduler": trainer.scheduler.state_dict(),
                        "scaler": trainer.scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    ckpt_dir / "best.pt",
                )
                logger.info(f"New best saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at step {step} (no improvement for {patience} evals)")
                    break

    logger.info("Stage 2 complete")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
