"""
train_stage2.py — Stage 2: Oracle 蒸馏

用"看全牌的 Oracle 模型"指导学生模型（只看公开信息）。
注意：需要先有 Stage 1 训练好的模型，且数据要含 opponent_hands 字段。
天凤公开数据没有对手手牌，所以 Oracle 只能用于有全信息数据时（如自对弈或特殊格式）。

当数据里没有 opponent_hands 时，Stage 2 自动退化为：
  用 Stage 1 的模型本身作为 "Oracle"，做自蒸馏（knowledge distillation）
  实际上相当于用更高温度重训练，有一定正则化效果。

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
        stage        = 2,
        device       = cfg.get("device", "cuda"),
        dtype        = cfg.get("dtype", "float32"),
        amp          = cfg.get("amp", False),
        model_preset = cfg.get("model_preset", "base"),
        lr           = float(cfg.get("lr", 1e-4)),
        weight_decay = float(cfg.get("weight_decay", 0.01)),
        warmup_steps = int(cfg.get("warmup_steps", 500)),
        total_steps  = int(cfg.get("total_steps", 50_000)),
        save_dir     = cfg.get("save_dir", "checkpoints/stage2"),
        save_every   = int(cfg.get("save_every", 5000)),
        log_every    = int(cfg.get("log_every", 100)),
    )
    trainer = Trainer(trainer_cfg)
    device  = trainer.device

    # 从 Stage 1 加载权重（学生网络初始化）
    logger.info(f"Loading Stage 1 weights from {stage1_ckpt}")
    s1_ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=True)
    # torch.compile 保存的权重带 _orig_mod. 前缀，加载前剥掉
    raw_sd = s1_ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}
    trainer.model.load_state_dict(raw_sd, strict=False)  # aux_heads 在 stage2 不用，忽略多余 key

    # Oracle：加载相同架构但接收全信息序列（含对手手牌）的模型
    # 数据里有 opponent_hands，使用真正的 Oracle 蒸馏而非自蒸馏
    logger.info("Oracle = full-information model (true oracle distillation)")
    from rinshan.constants import MAX_ORACLE_SEQ_LEN
    oracle_transformer_cfg = TransformerConfig.from_preset(cfg.get("model_preset", "base"))
    oracle_transformer_cfg.max_seq_len = MAX_ORACLE_SEQ_LEN   # Oracle 序列更长，含对手手牌
    trainer.set_oracle_model(
        RinshanModel(
            transformer_cfg=oracle_transformer_cfg,
            use_belief=False,  # Oracle 已看全牌，不需要 Belief Net
            use_aux=False,
        )
    )
    # Oracle 加载相同的 Stage 1 权重（共享 encoder，只是输入序列更长）
    trainer.oracle_model.load_state_dict(raw_sd, strict=False)

    ckpt_dir  = Path(trainer_cfg.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(cfg.get("total_steps", 50_000))
    val_every   = int(cfg.get("val_every", 2000))
    best_val_loss = float("inf")

    logger.info(f"Starting Stage 2 (self-distillation) for {total_steps} steps")

    step = 0
    for batch in train_loader:
        if step >= total_steps:
            break

        # oracle_tokens 由 Dataset(stage=2) 通过 encode_oracle() 生成，直接使用
        # （不再用 tokens 覆盖，否则退化成自蒸馏）
        loss_dict = trainer.train_step(batch)
        step = trainer.step

        if step % val_every == 0:
            # 简单 val loss
            trainer.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for vb in val_loader:
                    # oracle_tokens 已由 Dataset(stage=2) 填充，直接使用
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

    logger.info("Stage 2 complete")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
