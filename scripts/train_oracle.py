"""
train_oracle.py — 训练全信息 Oracle 模型

【设计目标】
  Oracle 是一个"上帝视角"的 value predictor：
    - 输入：含三家对手手牌的完整序列（oracle_tokens）
    - 任务1 (BC)      : 预测人类动作（快速收敛，确保 Q 分布有意义）
    - 任务2 (value)   : 预测该局面执行人类动作后的终局价值
                         target = hand_reward（局内得分/1000） + grp_reward（终局排名价值）
    - 任务3 (belief)  : 不需要，Oracle 已看全牌

【与 Stage2 的关系】
  Stage2 用 Oracle Q 分布作为软标签蒸馏 Student。
  Oracle 必须先学会"看到对手手牌后做出更好的决策"，
  否则蒸馏传递的是噪声而非知识。

【训练流程】
  1. 从 Stage1 best.pt 热启动（共享 Transformer encoder 权重）
  2. 纯全信息输入，loss = BC + value regression（不跑 Student，不跑 Belief）
  3. 收敛标准：
       - oracle_bc  < 0.15 (Stage1 是 0.40，Oracle 利用全信息能压到更低)
       - val 连续 patience 次不改善则停止

【可复现性保证】
  - Oracle 独立训练，不依赖任何 Stage2 中间产物
  - train_stage2.py 只需指定 oracle_ckpt 路径，无需任何自动 fine-tune 逻辑
  - 所有超参在 config yaml 中显式指定

Usage:
    python scripts/train_oracle.py configs/oracle_base.yaml
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.utils.config       import load_config
from rinshan.model              import RinshanModel
from rinshan.model.transformer  import TransformerConfig
from rinshan.data               import MjaiDataset, collate_fn
from rinshan.constants          import MAX_ORACLE_SEQ_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_oracle")


def _strip_prefix(sd: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def oracle_loss(
    q:          torch.Tensor,   # (B, N_cand) Oracle Q logits
    action_idx: torch.Tensor,   # (B,) 人类动作 index
    reward:     torch.Tensor,   # (B,) 终局价值 target（hand+grp 混合）
    bc_weight:    float = 1.0,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Oracle 双任务损失：
      BC loss    : cross_entropy(Q, human_action)
                   → 确保 Q 分布有区分度，快速收敛
      Value loss : MSE(Q[human_action], reward)
                   → 让 Q 值的绝对量级逼近真实价值
                   → 这是 Oracle 与 Student 的本质区别
    """
    losses = {}

    # ── 任务1: BC ──────────────────────────────────────────
    bc = F.cross_entropy(q, action_idx)
    losses["bc"] = bc.item()

    # ── 任务2: Value regression ────────────────────────────
    # 取人类执行的那个动作对应的 Q 值
    B = q.size(0)
    q_chosen = q[torch.arange(B, device=q.device), action_idx]   # (B,)
    # reward 已经是 float，scale 到合理范围（hand_reward ~[-5,5], grp ~[-1,1]）
    value = F.mse_loss(q_chosen, reward)
    losses["value"] = value.item()

    total = bc_weight * bc + value_weight * value
    losses["total"] = total.item()
    return total, losses


@torch.no_grad()
def validate(model, loader, device, n_batches=50):
    model.eval()
    bc_sum = val_sum = n = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        oracle_tokens  = batch.get("oracle_tokens")
        if oracle_tokens is None:
            continue
        oracle_tokens  = oracle_tokens.to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        oracle_pad     = batch.get("oracle_pad_mask")
        if oracle_pad is not None:
            oracle_pad = oracle_pad.to(device)
        action_idx  = batch["action_idx"].to(device)
        reward_hand = batch["reward_hand"].to(device).float()
        reward_game = batch["reward"].to(device).float()
        reward      = reward_hand + reward_game

        out = model(
            tokens=oracle_tokens,
            candidate_mask=candidate_mask,
            pad_mask=oracle_pad,
        )
        _, d = oracle_loss(out.q.float(), action_idx, reward)
        bc_sum  += d["bc"]
        val_sum += d["value"]
        n += 1

    model.train()
    if n == 0:
        return {"bc": float("inf"), "value": float("inf"), "score": float("inf")}
    return {
        "bc":    bc_sum / n,
        "value": val_sum / n,
        "score": bc_sum / n + 0.1 * val_sum / n,   # 主指标：BC 为主，value 为辅
    }


def main():
    cfg = load_config(sys.argv[1], sys.argv[2:])
    logger.info(f"Config: {cfg}")

    device = torch.device(cfg.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 数据 ──────────────────────────────────────────────
    import random
    all_files = sorted(Path(cfg["data_dir"]).rglob("*.jsonl"))
    random.seed(cfg.get("seed", 42))
    random.shuffle(all_files)
    val_n     = max(1, int(len(all_files) * cfg.get("val_ratio", 0.02)))
    val_files = all_files[:val_n]
    trn_files = all_files[val_n:]
    logger.info(f"Train files: {len(trn_files)}  Val files: {val_n}")

    train_ds = MjaiDataset(trn_files, shuffle_files=True,
                           shuffle_buffer=cfg.get("shuffle_buffer", 15000), stage=1)
    val_ds   = MjaiDataset(val_files, shuffle_files=False,
                           shuffle_buffer=500, stage=1)

    batch_size   = cfg.get("batch_size", 512)
    num_workers  = cfg.get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2,
                              collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=True)

    # ── 模型（Oracle：全信息序列，无 Belief Net）────────────
    preset = cfg.get("model_preset", "base")
    oracle_cfg = TransformerConfig.from_preset(preset)
    oracle_cfg.max_seq_len = MAX_ORACLE_SEQ_LEN   # 扩展序列长度以容纳对手手牌
    model = RinshanModel(
        transformer_cfg=oracle_cfg,
        use_belief=False,    # Oracle 已看全牌，不需要 Belief Net
        use_aux=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Oracle model parameters: {total_params:,}")

    # ── 从 Stage1 权重热启动 ────────────────────────────────
    stage1_ckpt = cfg.get("stage1_ckpt", "")
    if stage1_ckpt and Path(stage1_ckpt).exists():
        raw = torch.load(stage1_ckpt, map_location=device, weights_only=True)
        sd  = _strip_prefix(raw.get("model", raw.get("model_state_dict", raw)))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded Stage1 weights — missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            logger.info(f"  missing (will be random-init): {missing[:6]}")
    else:
        logger.info("No stage1_ckpt provided — training from scratch")

    # ── 优化器 & 调度器 ─────────────────────────────────────
    total_steps  = cfg.get("total_steps",  150_000)
    warmup_steps = cfg.get("warmup_steps", 1000)
    lr           = float(cfg.get("lr", 1e-4))
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
    )
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                            total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                     eta_min=lr * 0.05)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    # ── AMP ──────────────────────────────────────────────────
    use_amp  = cfg.get("amp", True)
    dtype    = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler   = torch.cuda.GradScaler(enabled=use_amp and dtype == torch.float16)

    # ── Resume ───────────────────────────────────────────────
    save_dir = Path(cfg.get("save_dir", "checkpoints/oracle"))
    save_dir.mkdir(parents=True, exist_ok=True)
    step           = 0
    best_score     = float("inf")
    patience_count = 0
    patience       = cfg.get("patience", 10)

    best_pt = save_dir / "best.pt"
    if best_pt.exists():
        logger.info(f"Resuming from {best_pt}")
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(_strip_prefix(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step       = ckpt["step"]
        best_score = ckpt.get("best_score", float("inf"))
        logger.info(f"Resumed at step {step}, best_score={best_score:.4f}")

    # ── Loss 权重 ─────────────────────────────────────────────
    bc_weight    = float(cfg.get("bc_weight",    1.0))
    value_weight = float(cfg.get("value_weight", 1.0))

    # ── 训练循环 ──────────────────────────────────────────────
    model.train()
    grad_accum   = cfg.get("grad_accum_steps", 1)
    max_grad_norm= float(cfg.get("max_grad_norm", 1.0))
    log_every    = cfg.get("log_every",   100)
    save_every   = cfg.get("save_every",  5000)
    val_every    = cfg.get("val_every",   5000)

    log_bc = log_val = log_total = 0.0
    optimizer.zero_grad(set_to_none=True)

    logger.info(f"Starting Oracle training for {total_steps} steps")
    logger.info(f"  bc_weight={bc_weight}  value_weight={value_weight}")
    logger.info("  Convergence target: oracle_bc < 0.15")

    for batch in train_loader:
        if step >= total_steps:
            break

        oracle_tokens = batch.get("oracle_tokens")
        if oracle_tokens is None:
            continue   # 没有对手手牌的样本跳过

        oracle_tokens  = oracle_tokens.to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        oracle_pad     = batch.get("oracle_pad_mask")
        if oracle_pad is not None:
            oracle_pad = oracle_pad.to(device)
        action_idx  = batch["action_idx"].to(device)
        reward_hand = batch["reward_hand"].to(device).float()
        reward_game = batch["reward"].to(device).float()
        reward      = reward_hand + reward_game   # 局内 + 终局价值

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(
                tokens=oracle_tokens,
                candidate_mask=candidate_mask,
                pad_mask=oracle_pad,
            )
            loss, d = oracle_loss(
                q=out.q.float(),
                action_idx=action_idx,
                reward=reward,
                bc_weight=bc_weight,
                value_weight=value_weight,
            )
            loss = loss / grad_accum

        if use_amp and dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        log_bc    += d["bc"]
        log_val   += d["value"]
        log_total += d["total"]

        if (step + 1) % grad_accum == 0:
            if use_amp and dtype == torch.float16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        step += 1

        # ── 日志 ─────────────────────────────────────────────
        if step % log_every == 0:
            n = log_every
            logger.info(
                f"[step {step}] "
                f"bc={log_bc/n:.4f}  "
                f"value={log_val/n:.4f}  "
                f"total={log_total/n:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            log_bc = log_val = log_total = 0.0

        # ── 定期保存 ──────────────────────────────────────────
        if step % save_every == 0:
            ckpt_path = save_dir / f"checkpoint_{step}.pt"
            torch.save({
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "step":       step,
                "best_score": best_score,
            }, ckpt_path)
            logger.info(f"Saved checkpoint → {ckpt_path}")

        # ── 验证 + early stop ─────────────────────────────────
        if step % val_every == 0:
            metrics = validate(model, val_loader, device, n_batches=100)
            score   = metrics["score"]
            logger.info(
                f"[val step={step}] "
                f"bc={metrics['bc']:.4f}  "
                f"value={metrics['value']:.4f}  "
                f"score={score:.4f}  "
                f"best={best_score:.4f}  "
                f"patience={patience_count}/{patience}"
            )
            if score < best_score:
                best_score = score
                patience_count = 0
                torch.save({
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "step":       step,
                    "best_score": best_score,
                }, best_pt)
                logger.info(f"  ✓ New best saved (score={best_score:.4f})")
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f"  Early stopping at step {step}")
                    break

    logger.info("=" * 60)
    logger.info(f"Oracle training complete. Best score={best_score:.4f}")
    logger.info(f"Checkpoint: {best_pt}")
    logger.info("Next step: use this Oracle in Stage2:")
    logger.info(f"  oracle_ckpt: {best_pt}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
