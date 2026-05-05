"""
diagnose_oracle.py — 一键诊断 Oracle 是否真正利用了对手手牌

测量三个指标：
  1. oracle_bc       : Oracle 在全信息输入下的 BC loss
  2. student_bc      : Student 在公开信息输入下的 BC loss（对比基准）
  3. oracle_kl       : Oracle vs Student 的 Q 分布 KL 散度
                       → 这是 Stage2 蒸馏信号的直接来源
  4. q_gap           : Oracle 在"正确动作"上比 Student 高出多少 Q 值

判断标准：
  oracle_kl > 0.25  → Oracle 信号充足，可以直接跑 Stage2
  oracle_kl 0.1-0.25 → 信号偏弱但可用，Stage2 蒸馏效果打折
  oracle_kl < 0.1   → Oracle 基本没有利用对手手牌，蒸馏无意义

Usage:
    python scripts/diagnose_oracle.py \\
        --oracle_ckpt  checkpoints/oracle_base/best.pt \\
        --student_ckpt checkpoints/stage2_base/best.pt \\
        --data_dir     /root/autodl-tmp/rinshan/data/annotated_v4 \\
        --n_batches    200
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.model             import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.data              import MjaiDataset, collate_fn
from rinshan.constants         import MAX_ORACLE_SEQ_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose_oracle")


def strip(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def load_oracle(ckpt_path: str, preset: str, device):
    cfg = TransformerConfig.from_preset(preset)
    cfg.max_seq_len = MAX_ORACLE_SEQ_LEN
    model = RinshanModel(transformer_cfg=cfg, use_belief=False, use_aux=False).to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(strip(raw.get("model", raw)), strict=False)
    model.eval()
    return model


def load_student(ckpt_path: str, preset: str, device):
    cfg = TransformerConfig.from_preset(preset)
    model = RinshanModel(transformer_cfg=cfg, use_belief=True, use_aux=False).to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(strip(raw.get("model", raw)), strict=False)
    model.eval()
    return model


@torch.no_grad()
def diagnose(oracle, student, loader, device, n_batches):
    oracle_bc_sum = student_bc_sum = kl_sum = q_gap_sum = n = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        oracle_tokens  = batch.get("oracle_tokens")
        if oracle_tokens is None:
            continue

        tokens         = batch["tokens"].to(device)
        oracle_tokens  = oracle_tokens.to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        pad_mask       = batch.get("pad_mask")
        oracle_pad     = batch.get("oracle_pad_mask")
        if pad_mask    is not None: pad_mask    = pad_mask.to(device)
        if oracle_pad  is not None: oracle_pad  = oracle_pad.to(device)
        action_idx     = batch["action_idx"].to(device)

        # ── Oracle forward ──────────────────────────────────────
        o_out = oracle(tokens=oracle_tokens, candidate_mask=candidate_mask,
                       pad_mask=oracle_pad)
        o_q = o_out.q.float()   # (B, N_cand)

        # ── Student forward ─────────────────────────────────────
        s_out = student(tokens=tokens, candidate_mask=candidate_mask,
                        pad_mask=pad_mask)
        s_q = s_out.q.float()   # (B, N_cand)

        # 只看有效候选位
        mask = candidate_mask.bool()   # (B, N_cand)

        # BC loss
        oracle_bc_sum  += F.cross_entropy(o_q, action_idx).item()
        student_bc_sum += F.cross_entropy(s_q, action_idx).item()

        # KL(Oracle || Student)  —  softmax 后在有效候选上算
        o_log_p = F.log_softmax(o_q.masked_fill(~mask, -1e9), dim=-1)
        s_log_p = F.log_softmax(s_q.masked_fill(~mask, -1e9), dim=-1)
        o_p     = o_log_p.exp()
        # KL = sum(p_o * (log_p_o - log_p_s))
        kl = (o_p * (o_log_p - s_log_p)).sum(dim=-1).mean()
        kl_sum += kl.item()

        # Q gap：Oracle 在人类动作上比 Student 高多少
        B = action_idx.size(0)
        idx = action_idx.clamp(0, o_q.size(1) - 1)
        o_chosen = o_q[torch.arange(B, device=device), idx]
        s_chosen = s_q[torch.arange(B, device=device), idx]
        q_gap_sum += (o_chosen - s_chosen).mean().item()

        n += 1

    if n == 0:
        logger.error("No valid batches! Check oracle_tokens in dataset.")
        return

    logger.info("=" * 60)
    logger.info("Oracle Diagnosis Results")
    logger.info("=" * 60)
    logger.info(f"  oracle_bc    = {oracle_bc_sum/n:.4f}   (全信息下 BC loss)")
    logger.info(f"  student_bc   = {student_bc_sum/n:.4f}   (公开信息下 BC loss，对比基准)")
    logger.info(f"  bc_delta     = {(student_bc_sum - oracle_bc_sum)/n:.4f}   "
                f"(Oracle 比 Student 准多少，>0 说明对手手牌有帮助)")
    logger.info(f"  oracle_kl    = {kl_sum/n:.4f}   (KL 散度，Stage2 蒸馏信号强度)")
    logger.info(f"  q_gap        = {q_gap_sum/n:.4f}   (Oracle 在人类动作上的 Q 值优势)")
    logger.info("-" * 60)

    kl = kl_sum / n
    if kl > 0.25:
        logger.info("  ✓ 信号充足 — 直接跑 Stage2")
    elif kl > 0.10:
        logger.info("  ⚠ 信号偏弱 — Stage2 可以跑但蒸馏效果打折")
        logger.info("    建议：续跑 Oracle 或调低 value_weight")
    else:
        logger.info("  ✗ 信号不足 — Oracle 没有利用对手手牌，蒸馏无意义")
        logger.info("    建议：检查 encode_oracle() 是否正确填入 opponent_hands")
    logger.info("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_ckpt",  required=True)
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--data_dir",     required=True)
    ap.add_argument("--preset",       default="base")
    ap.add_argument("--n_batches",    type=int, default=200)
    ap.add_argument("--batch_size",   type=int, default=512)
    ap.add_argument("--num_workers",  type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    import random
    all_files = sorted(Path(args.data_dir).rglob("*.jsonl"))
    random.seed(42)
    random.shuffle(all_files)
    val_n     = max(1, int(len(all_files) * 0.02))
    val_files = all_files[:val_n]
    logger.info(f"Val files: {val_n}")

    val_ds = MjaiDataset(val_files, shuffle_files=False, shuffle_buffer=500, stage=2)
    loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Loading Oracle  from {args.oracle_ckpt}")
    oracle = load_oracle(args.oracle_ckpt, args.preset, device)

    logger.info(f"Loading Student from {args.student_ckpt}")
    student = load_student(args.student_ckpt, args.preset, device)

    diagnose(oracle, student, loader, device, args.n_batches)


if __name__ == "__main__":
    main()
