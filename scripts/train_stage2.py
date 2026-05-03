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


# ─────────────────────────────────────────────────────────────────────────────
# Oracle 诊断 + 自动 fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def _measure_oracle_kl(
    trainer,
    val_ds,
    batch_size: int,
    n_batches: int = 50,
) -> tuple[float, float]:
    """测量当前 Oracle-Student KL 散度和 Q 值绝对差，用于判断蒸馏信号强度。"""
    import torch.nn.functional as F
    trainer.model.eval()
    trainer.oracle_model.eval()
    diag_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn,
                             num_workers=0, pin_memory=False)
    kl_sum = q_gap_sum = 0.0
    n = 0
    with torch.no_grad():
        for db in diag_loader:
            s_out = trainer.model(
                tokens          = trainer._to_device(db["tokens"]),
                candidate_mask  = trainer._to_device(db["candidate_mask"]),
                pad_mask        = trainer._to_device(db.get("pad_mask")),
                belief_tokens   = trainer._to_device(db.get("belief_tokens")),
                belief_pad_mask = trainer._to_device(db.get("belief_pad_mask")),
            )
            o_out = trainer.oracle_model(
                tokens         = trainer._to_device(db["oracle_tokens"]),
                candidate_mask = trainer._to_device(db["candidate_mask"]),
                pad_mask       = trainer._to_device(db.get("oracle_pad_mask")),
            )
            sq = s_out.q.float().masked_fill(s_out.q == float('-inf'), -1e9)
            oq = o_out.q.float().masked_fill(o_out.q == float('-inf'), -1e9)
            kl_sum    += F.kl_div(F.log_softmax(sq / 2.0, dim=-1),
                                   F.softmax(oq / 2.0, dim=-1),
                                   reduction='batchmean').item()
            q_gap_sum += (oq - sq).abs().mean().item()
            n += 1
            if n >= n_batches:
                break
    trainer.model.train()
    return kl_sum / max(n, 1), q_gap_sum / max(n, 1)


def finetune_oracle(
    trainer,
    train_ds,
    val_ds,
    batch_size: int,
    device: torch.device,
    ckpt_dir: Path,
    finetune_steps: int = 3000,
    finetune_lr: float = 2e-5,
    num_workers: int = 4,
) -> None:
    """
    用 oracle_tokens 对 Oracle 做 BC fine-tune，使其真正利用对手手牌信息。
    fine-tune 完成后 Oracle 自动重新冻结，权重保存到 ckpt_dir/oracle_finetuned.pt。
    """
    import torch.nn.functional as F
    from torch.optim import AdamW

    oracle_ckpt = ckpt_dir / "oracle_finetuned.pt"
    if oracle_ckpt.exists():
        logger.info(f"[oracle_ft] Found existing fine-tuned Oracle at {oracle_ckpt}, loading directly.")
        sd = torch.load(oracle_ckpt, map_location=device, weights_only=True)["model"]
        trainer.oracle_model.load_state_dict(sd, strict=True)
        trainer.oracle_model.eval()
        trainer.oracle_model.requires_grad_(False)
        return

    logger.info(
        f"[oracle_ft] KL < 0.2 — starting Oracle BC fine-tune for {finetune_steps} steps "
        f"(lr={finetune_lr:.1e}) to strengthen distillation signal."
    )

    # 临时解冻 Oracle
    trainer.oracle_model.train()
    trainer.oracle_model.requires_grad_(True)

    optimizer = AdamW(
        [p for p in trainer.oracle_model.parameters() if p.requires_grad],
        lr=finetune_lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    ft_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn,
                           num_workers=num_workers, pin_memory=True)

    step = 0
    loss_acc = 0.0
    for batch in ft_loader:
        if step >= finetune_steps:
            break

        oracle_tokens  = trainer._to_device(batch["oracle_tokens"])
        candidate_mask = trainer._to_device(batch["candidate_mask"])
        oracle_pad     = trainer._to_device(batch.get("oracle_pad_mask"))
        action_idx     = batch["action_idx"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            out = trainer.oracle_model(
                tokens=oracle_tokens,
                candidate_mask=candidate_mask,
                pad_mask=oracle_pad,
            )
            # 纯 BC loss：让 Oracle 在全信息输入下模仿人类动作
            loss = F.cross_entropy(out.q.float().masked_fill(out.q == float('-inf'), -1e9),
                                   action_idx)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.oracle_model.parameters(), 1.0)
        optimizer.step()

        step += 1
        loss_acc += loss.item()
        if step % 500 == 0:
            logger.info(f"[oracle_ft step={step}/{finetune_steps}] bc_loss={loss_acc/500:.4f}")
            loss_acc = 0.0

    # 保存 fine-tuned Oracle
    torch.save({"model": trainer.oracle_model.state_dict()}, oracle_ckpt)
    logger.info(f"[oracle_ft] Fine-tune complete. Saved to {oracle_ckpt}")

    # 重新冻结
    trainer.oracle_model.eval()
    trainer.oracle_model.requires_grad_(False)


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
    patience = int(cfg.get("patience", 8))   # 连续 N 次 val 不改善则停止
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

    # ── Oracle 蒸馏信号诊断 + 自动 fine-tune ────────────────────────────
    kl_threshold = float(cfg.get("oracle_kl_threshold",  0.2))
    ft_steps     = int(cfg.get("oracle_finetune_steps",  3000))
    ft_lr        = float(cfg.get("oracle_finetune_lr",   2e-5))

    diag_kl, diag_gap = _measure_oracle_kl(trainer, val_ds, batch_size)
    logger.info(
        f"[diag] Oracle-Student KL={diag_kl:.4f}  Q_abs_gap={diag_gap:.4f}  "
        f"(threshold={kl_threshold})"
    )
    if diag_kl < kl_threshold:
        logger.warning(
            f"[diag] ⚠ KL={diag_kl:.4f} < {kl_threshold} — "
            "Oracle 未充分利用全信息，自动启动 BC fine-tune..."
        )
        finetune_oracle(
            trainer        = trainer,
            train_ds       = train_ds,
            val_ds         = val_ds,
            batch_size     = batch_size,
            device         = device,
            ckpt_dir       = ckpt_dir,
            finetune_steps = ft_steps,
            finetune_lr    = ft_lr,
            num_workers    = cfg.get("num_workers", 4),
        )
        # fine-tune 后重新测量，确认信号增强
        diag_kl_after, _ = _measure_oracle_kl(trainer, val_ds, batch_size)
        logger.info(
            f"[diag] Post-finetune KL={diag_kl_after:.4f}  "
            f"(+{diag_kl_after - diag_kl:+.4f} vs before)"
        )
        if diag_kl_after < kl_threshold:
            logger.warning(
                f"[diag] ⚠ KL 仍 < {kl_threshold}，蒸馏信号依然偏弱。"
                f"可尝试增大 oracle_finetune_steps（当前={ft_steps}）后重跑。"
            )
        else:
            logger.info("[diag] ✓ Oracle fine-tune 后信号充足，继续 Stage 2 蒸馏。")
    else:
        logger.info("[diag] ✓ KL 信号充足，Oracle 有效利用全信息，蒸馏可正常进行。")
    # ── end diag ─────────────────────────────────────────────────────────

    logger.info(f"Starting Stage 2 (self-distillation) for {total_steps} steps")

    step = 0
    for batch in train_loader:
        if step >= total_steps:
            break

        loss_dict = trainer.train_step(batch)
        step = trainer.step

        if step % val_every == 0:
            trainer.model.eval()
            val_kl = 0.0
            val_bc = 0.0
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for vb in val_loader:
                    _, ld = trainer._forward_and_loss(vb)
                    val_kl    += ld.get("kl", 0.0)
                    val_bc    += ld.get("bc", 0.0)
                    val_total += ld.get("total", 0.0)
                    n_val += 1
                    if n_val >= 200:
                        break
            trainer.model.train()
            n = max(n_val, 1)
            val_kl /= n; val_bc /= n; val_total /= n
            logger.info(
                f"[val step={step}] kl={val_kl:.4f}  bc={val_bc:.4f}  total={val_total:.4f}"
                f"  best_kl={best_val_loss:.4f}  patience={patience_counter}/{patience}"
            )
            if val_kl < best_val_loss:
                best_val_loss = val_kl
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
                logger.info(f"New best saved (val_kl={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at step {step} (no improvement for {patience} evals)")
                    break

    logger.info("Stage 2 complete")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
