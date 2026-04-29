"""
train_stage3.py — Stage 3: 离线 IQL 强化学习（GRP 2.0）

在 Stage 1/2 的基础上，用 GRP 价值差分信号做离线 RL 精调。
GRP 2.0 的关键变化：
  1. 保留 GRP 作为 learned game-value estimator
  2. 不再把同一局的 grp_reward 广播到局内所有 action
  3. 只在 GRP game-state 真正变化（进入下一局 / 终局）时，
     将该局 delta-value 记到最后一个 action 上；其余 action reward=0
  4. 训练时加入 AWR 风格 BC anchor，防止策略快速偏离 Stage2 基线

需要先跑完：
  1. train_grp.py         → 得到 GRP 模型
  2. fill_grp_rewards.py  → 为标注数据填入 grp_reward 字段（按局 delta）
  3. train_stage1.py / train_stage2.py → 得到 Stage 初始化权重

Usage:
    python scripts/train_stage3.py configs/stage3_base.yaml \
        --stage1_ckpt checkpoints/stage1_base/best.pt
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.utils.config import load_config
from rinshan.data         import MjaiDataset, collate_fn
from rinshan.training     import Trainer, TrainerConfig


def _arena_gate(cfg: dict, ckpt_path: Path, step: int, save_dir: Path) -> tuple[bool, dict]:
    gate_games = int(cfg.get("arena_gate_games", 0))
    baseline_ckpt = cfg.get("arena_gate_baseline_ckpt", cfg.get("stage2_ckpt", ""))
    if gate_games <= 0 or not baseline_ckpt:
        return False, {}

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_self_play.py")),
        "--mode", "versus",
        "--ckpt", str(ckpt_path),
        "--ckpt2", str(baseline_ckpt),
        "--model_preset", str(cfg.get("model_preset", "base")),
        "--n_games", str(gate_games),
        "--parallel_games", str(cfg.get("arena_parallel_games", gate_games)),
        "--device", str(cfg.get("arena_device", cfg.get("device", "cuda"))),
        "--seed", str(int(cfg.get("arena_seed", 1234)) + int(step)),
        "--greedy",
        "--quiet",
    ]
    ckpt2_preset = cfg.get("arena_gate_baseline_preset")
    if ckpt2_preset:
        cmd.extend(["--ckpt2_preset", str(ckpt2_preset)])

    logger.info("Running arena gate: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(Path(__file__).parents[1]), capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("Arena gate failed (returncode=%s): %s", proc.returncode, proc.stderr.strip())
        return False, {"error": proc.stderr.strip(), "returncode": proc.returncode}

    # 直接在 stdout 中解析最后打印的 delta 行
    metrics = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "顺位差 Δ=" in line:
            try:
                delta_str = line.split("Δ=")[-1].split()[0]
                metrics["delta_rank"] = float(delta_str)
            except Exception:
                pass
        elif "Challenger  平均顺位" in line:
            parts = line.replace("Challenger  平均顺位", "").split()
            if parts:
                try:
                    metrics["challenger_avg_rank"] = float(parts[0])
                except Exception:
                    pass
    if "delta_rank" not in metrics:
        logger.warning("Arena gate parse failed; stdout tail:\n%s", "\n".join(proc.stdout.splitlines()[-20:]))
        return False, {"error": "parse_failed"}

    threshold = float(cfg.get("arena_gate_rank_delta_threshold", 0.0))
    passed = metrics["delta_rank"] <= threshold
    metrics["passed"] = passed
    metrics["threshold"] = threshold
    gate_log = save_dir / f"arena_gate_step{step}.log"
    gate_log.write_text(proc.stdout + "\n\nSTDERR:\n" + proc.stderr, encoding="utf-8")
    logger.info("Arena gate step=%s delta_rank=%.4f threshold=%.4f passed=%s", step, metrics["delta_rank"], threshold, passed)
    return passed, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stage3")


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_stage3.py <config.yaml> --stage1_ckpt <path>")
        sys.exit(1)

    cfg = load_config(sys.argv[1], sys.argv[2:])
    logger.info(f"Config: {cfg}")

    stage1_ckpt = cfg.get("stage1_ckpt", "")
    stage2_ckpt = cfg.get("stage2_ckpt", "")

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
    val_loader   = DataLoader(val_ds, batch_size=batch_size,   # 不扩大，Stage3 val 同样有3次forward
                              collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)

    # ── Trainer（Stage 3）─────────────────────
    trainer_cfg = TrainerConfig(
        stage            = 3,
        device           = cfg.get("device", "cuda"),
        dtype            = cfg.get("dtype", "float32"),
        amp              = cfg.get("amp", False),
        compile          = cfg.get("compile", False),
        model_preset     = cfg.get("model_preset", "base"),
        lr               = float(cfg.get("lr", 1e-4)),
        weight_decay     = float(cfg.get("weight_decay", 0.01)),
        max_grad_norm    = float(cfg.get("max_grad_norm", 1.0)),
        grad_accum_steps = int(cfg.get("grad_accum_steps", 1)),   # ← 修复：之前漏传，永远是1
        warmup_steps     = int(cfg.get("warmup_steps", 500)),
        total_steps      = int(cfg.get("total_steps", 100_000)),
        save_dir         = cfg.get("save_dir", "checkpoints/stage3"),
        save_every       = int(cfg.get("save_every", 5000)),
        log_every        = int(cfg.get("log_every", 100)),
        target_update_every = int(cfg.get("target_update_every", 100)),
        cql_weight          = float(cfg.get("cql_weight", -1.0)),
        weights_only_save   = bool(cfg.get("weights_only_save", False)),
        bc_weight           = float(cfg.get("bc_weight", 0.2)),
        reward_clip         = float(cfg.get("reward_clip", 20.0)),
        value_clip          = float(cfg.get("value_clip", 50.0)),
        adv_clip            = float(cfg.get("adv_clip", 20.0)),
        awr_temperature     = float(cfg.get("awr_temperature", 3.0)),
        awr_max_weight      = float(cfg.get("awr_max_weight", 20.0)),
        game_expectile      = float(cfg.get("game_expectile", 0.95)),
        hand_expectile      = float(cfg.get("hand_expectile", 0.70)),
        game_reward_weight  = float(cfg.get("game_reward_weight", 1.0)),
        hand_reward_weight  = float(cfg.get("hand_reward_weight", 1.0)),
    )
    trainer = Trainer(trainer_cfg)
    device  = trainer.device

    # ── 断点续训：优先恢复已有 checkpoint，否则从 Stage 1 初始化 ──
    # lr 变化时自动切换到 best.pt 权重，optimizer/scheduler 用新 lr 重建
    ckpt_dir = Path(trainer_cfg.save_dir)
    existing_ckpts = sorted(
        ckpt_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if existing_ckpts:
        latest = existing_ckpts[-1]
        # 只 peek lr，不做完整 load
        peek = torch.load(latest, map_location="cpu", weights_only=True)
        ckpt_lr = peek.get("lr", None)
        lr_changed = ckpt_lr is not None and abs(ckpt_lr - trainer_cfg.lr) > 1e-12
        if lr_changed:
            # lr 有变化：优先从 best.pt 加载权重（val 最优），没有才用最新 checkpoint
            best_path = ckpt_dir / "best.pt"
            src = best_path if best_path.exists() else latest
            logger.info(f"lr changed ({ckpt_lr:.2e} → {trainer_cfg.lr:.2e}): loading weights from {src}")
        else:
            src = latest
            logger.info(f"Resuming from {src}")
        del peek
        trainer.load(src)
    elif stage2_ckpt and Path(stage2_ckpt).exists():
        logger.info(f"Loading Stage 2 weights from {stage2_ckpt}")
        s2_ckpt = torch.load(stage2_ckpt, map_location=device, weights_only=True)
        trainer.model.load_state_dict(s2_ckpt["model"], strict=False)
        trainer.target_model.load_state_dict(s2_ckpt["model"], strict=False)
        logger.info("Target network initialized with Stage 2 weights (strict=False)")
    elif stage1_ckpt and Path(stage1_ckpt).exists():
        logger.info(f"Loading Stage 1 weights from {stage1_ckpt}")
        s1_ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=True)
        trainer.model.load_state_dict(s1_ckpt["model"], strict=False)
        # 目标网络也初始化为相同权重
        trainer.target_model.load_state_dict(s1_ckpt["model"], strict=False)
        logger.info("Target network initialized with Stage 1 weights (strict=False)")
    else:
        logger.warning("No checkpoint or stage1_ckpt/stage2_ckpt found, training from scratch")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(cfg.get("total_steps", 100_000))
    val_every   = int(cfg.get("val_every", 2000))
    best_val_loss = float("inf")
    best_gate_delta = float("inf")

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
                        "GRP 2.0 requires grp_reward deltas from fill_grp_rewards.py."
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
                trainer.save(ckpt_dir / "best_val.pt")

            if cfg.get("arena_gate_games", 0):
                gate_ckpt = ckpt_dir / f"gate_eval_step{step}.pt"
                trainer.save(gate_ckpt)
                passed, gate_metrics = _arena_gate(cfg, gate_ckpt, step, ckpt_dir)
                if passed and gate_metrics.get("delta_rank", float("inf")) < best_gate_delta:
                    best_gate_delta = gate_metrics["delta_rank"]
                    trainer.save(ckpt_dir / "best.pt")
                try:
                    gate_ckpt.unlink()
                except FileNotFoundError:
                    pass
            elif val_loss < best_val_loss:
                trainer.save(ckpt_dir / "best.pt")

    logger.info("Stage 3 complete")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
