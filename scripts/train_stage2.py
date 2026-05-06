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


def _strip_prefix(sd: dict) -> dict:
    """剥掉 torch.compile 保存的 _orig_mod. 前缀"""
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


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


# finetune_oracle 已移除。
# Oracle 必须通过 train_oracle.py 独立训练后，在 yaml 中以 oracle_ckpt 指定路径。
# 这样保证蒸馏信号来源可复现，不依赖启动时的自动修补。


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

    # Oracle：必须从 train_oracle.py 训练好的 checkpoint 加载
    # 不允许用 Stage1 权重直接初始化（那样 Oracle 不会利用对手手牌）
    oracle_ckpt_path = cfg.get("oracle_ckpt", "")
    if not oracle_ckpt_path or not Path(oracle_ckpt_path).exists():
        raise FileNotFoundError(
            f"oracle_ckpt not found: '{oracle_ckpt_path}'\n"
            "请先运行 train_oracle.py 训练 Oracle，再运行 Stage2。\n"
            "  python scripts/train_oracle.py configs/oracle_base.yaml"
        )
    logger.info(f"Oracle = {oracle_ckpt_path}")
    from rinshan.constants import MAX_ORACLE_SEQ_LEN
    oracle_transformer_cfg = TransformerConfig.from_preset(cfg.get("model_preset", "base"))
    oracle_transformer_cfg.max_seq_len = MAX_ORACLE_SEQ_LEN
    oracle_model = RinshanModel(
        transformer_cfg=oracle_transformer_cfg,
        use_belief=False,
        use_aux=False,
    )
    oracle_raw = torch.load(oracle_ckpt_path, map_location=device, weights_only=True)
    oracle_sd  = _strip_prefix(oracle_raw.get("model", oracle_raw))
    oracle_missing, oracle_unexpected = oracle_model.load_state_dict(oracle_sd, strict=False)
    logger.info(f"Oracle loaded — missing={len(oracle_missing)}  unexpected={len(oracle_unexpected)}")
    if oracle_missing:
        logger.warning(f"  Oracle missing keys: {oracle_missing[:4]}")
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
        # 优先用编号最大的 checkpoint 续跑（保证进度不丢失）
        # best.pt 只追踪最优模型，不用于 resume（否则重启会倒退回最优 step）
        existing = sorted(
            ckpt_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1])
        )
        if existing:
            resume_ckpt = str(existing[-1])
        else:
            # 没有任何编号 checkpoint，才退而求其次用 best.pt 冷启动
            best_pt = ckpt_dir / "best.pt"
            if best_pt.exists():
                resume_ckpt = str(best_pt)
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

    # ── Oracle KL 诊断（仅做信息展示，不自动 fine-tune）────────────────
    diag_kl, diag_gap = _measure_oracle_kl(trainer, val_ds, batch_size)
    logger.info(
        f"[diag] Oracle-Student KL={diag_kl:.4f}  Q_abs_gap={diag_gap:.4f}"
    )
    # _measure_oracle_kl 内部用 temperature=2.0 软化分布，正常收敛值约 0.005-0.02
    # 阈值 0.002：低于此值说明 Oracle 与 Student 几乎无差别，蒸馏信号不足
    if diag_kl < 0.002:
        logger.warning(
            f"[diag] ⚠ KL={diag_kl:.4f} 极低（temp=2.0 尺度下）。"
            " Oracle 与 Student 分布几乎相同，蒸馏信号不足。"
            " 请用 diagnose_oracle.py 确认 oracle_kl（原始尺度）> 0.25。"
        )
    else:
        logger.info(
            f"[diag] ✓ KL={diag_kl:.4f}（temp=2.0）信号充足，蒸馏可正常进行。"
            " （原始尺度 KL 可用 diagnose_oracle.py 查看）"
        )
    # ── end diag ─────────────────────────────────────────────────────────

    logger.info(f"Starting Stage 2 (self-distillation) for {total_steps} steps")

    # 移动平均窗口（用于训练 log 的趋势判断）
    _ema_alpha = 0.05           # 越小越平滑，约等于最近 20 步平均
    _ema:      dict[str, float] = {}
    _ema_prev: dict[str, float] = {}   # 上一个 log_every 周期的 EMA，用于算方向
    log_every = int(cfg.get("log_every", 100))

    def _update_ema(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, float):
                _ema[k] = v if k not in _ema else (1 - _ema_alpha) * _ema[k] + _ema_alpha * v

    def _trend(key: str) -> str:
        """返回 ↓ / ↑ / → 趋势符号（和上一周期 EMA 对比）"""
        if key not in _ema or key not in _ema_prev:
            return ""
        delta = _ema[key] - _ema_prev[key]
        if delta < -0.001:
            return "↓"
        if delta > 0.001:
            return "↑"
        return "→"

    step = 0
    for batch in train_loader:
        if step >= total_steps:
            break

        loss_dict = trainer.train_step(batch)
        step = trainer.step
        _update_ema(loss_dict)

        # EMA 摘要行：每 log_every 步追加一行趋势，紧跟 trainer 内部的瞬时 log
        if step % log_every == 0:
            ema_kl    = _ema.get("kl",    float("nan"))
            ema_bc    = _ema.get("bc",    float("nan"))
            ema_bel   = _ema.get("belief", float("nan"))
            ema_wait  = _ema.get("wait",   float("nan"))
            ema_score = ema_kl + 0.3 * ema_bc
            logger.info(
                f"[ema  {step}] "
                f"kl={ema_kl:.4f}{_trend('kl')}  "
                f"bc={ema_bc:.4f}{_trend('bc')}  "
                f"bel={ema_bel:.4f}{_trend('belief')}  "
                f"wait={ema_wait:.4f}{_trend('wait')}  "
                f"score(kl+0.3bc)={ema_score:.4f}"
                + ("" if "kl" in _ema_prev else "  (warming up)")
            )
            # 快照当前 EMA，供下一周期计算趋势
            _ema_prev.update(_ema)

        if step % val_every == 0:
            trainer.model.eval()
            val_kl = 0.0
            val_bc = 0.0
            val_total = 0.0
            n_val = 0
            val_bel = val_wait = 0.0
            # belief 准确率（每步算一次，用于判断 BeliefNet 是否收敛）
            import torch.nn.functional as F
            bel_tp = bel_total = 0
            with torch.no_grad():
                for vb in val_loader:
                    _, ld = trainer._forward_and_loss(vb)
                    val_kl    += ld.get("kl", 0.0)
                    val_bc    += ld.get("bc", 0.0)
                    val_bel   += ld.get("belief", 0.0)
                    val_wait  += ld.get("wait", 0.0)
                    val_total += ld.get("total", 0.0)
                    # 顺带算 belief 二分类准确率
                    s_out = trainer.model(
                        tokens=trainer._to_device(vb["tokens"]),
                        candidate_mask=trainer._to_device(vb["candidate_mask"]),
                        pad_mask=trainer._to_device(vb.get("pad_mask")),
                        belief_tokens=trainer._to_device(vb.get("belief_tokens")),
                        belief_pad_mask=trainer._to_device(vb.get("belief_pad_mask")),
                    )
                    if s_out.belief_logits is not None and vb.get("actual_hands") is not None:
                        ah = trainer._to_device(vb["actual_hands"]).float()
                        pred = (s_out.belief_probs > 0.5).float()
                        tgt  = (ah > 0).float()
                        bel_tp    += (pred * tgt).sum().item()
                        bel_total += tgt.sum().item()
                    n_val += 1
                    if n_val >= 200:
                        break
            trainer.model.train()
            n = max(n_val, 1)
            val_kl /= n; val_bc /= n; val_bel /= n; val_wait /= n; val_total /= n
            bel_recall = bel_tp / max(bel_total, 1)   # 有牌的位置预测对了多少
            val_score = val_kl + 0.3 * val_bc   # KL 主导，BC 作约束
            is_best = val_score < best_val_loss
            best_tag = " ← best" if is_best else ""
            logger.info(
                f"[val step={step}] kl={val_kl:.4f}  bc={val_bc:.4f}  "
                f"bel={val_bel:.4f}  wait={val_wait:.4f}  "
                f"bel_recall={bel_recall:.3f}  "
                f"score(kl+0.3bc)={val_score:.4f}  total={val_total:.4f}"
                f"  best={best_val_loss:.4f}  patience={patience_counter}/{patience}{best_tag}"
            )
            if is_best:
                best_val_loss = val_score
                patience_counter = 0
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
                logger.info(f"New best saved (score={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at step {step} (no improvement for {patience} evals)")
                    break

    logger.info("Stage 2 complete")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
