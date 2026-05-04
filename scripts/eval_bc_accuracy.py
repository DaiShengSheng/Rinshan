"""
eval_bc_accuracy.py — 对比 Stage1 / Stage2 checkpoint 的 BC Accuracy

BC Accuracy = argmax(Q) 与人类动作匹配的比例
用于验证 Stage 2 蒸馏是否真正提升了 policy 质量，
还是只是把 Oracle 的随机扰动学进来了。

Usage:
    python scripts/eval_bc_accuracy.py \
        --stage1 /path/to/stage1/best_v3.pt \
        --stage2 /path/to/stage2/best.pt \
        --data_dir /path/to/annotated_v4 \
        --n_batches 500 \
        --batch_size 256
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.model          import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.data           import MjaiDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_bc_accuracy")


def _strip_prefix(sd: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def load_model(ckpt_path: str, device: torch.device, preset: str = "base") -> RinshanModel:
    cfg = TransformerConfig.from_preset(preset)
    model = RinshanModel(transformer_cfg=cfg, use_belief=True, use_aux=False)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = raw.get("model", raw.get("model_state_dict", raw))
    sd = _strip_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded {ckpt_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(
    model: RinshanModel,
    loader: DataLoader,
    device: torch.device,
    n_batches: int,
    label: str,
) -> dict:
    """
    计算以下指标：
    - top1_acc:    argmax(Q) == human_action
    - top3_acc:    human_action 是否在 top-3 Q 值中
    - mean_rank:   human_action 在 Q 值降序排名的均值（越小越好，最小为 1）
    - mean_q_margin: 最优 Q 与 human Q 的差（越小说明 human 动作越接近最优）
    """
    top1_correct = top3_correct = total = 0
    rank_sum = q_margin_sum = 0.0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        tokens         = batch["tokens"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        pad_mask       = batch.get("pad_mask")
        belief_tokens  = batch.get("belief_tokens")
        belief_pad     = batch.get("belief_pad_mask")
        action_idx     = batch["action_idx"].to(device)          # (B,)

        if pad_mask     is not None: pad_mask     = pad_mask.to(device)
        if belief_tokens is not None: belief_tokens = belief_tokens.to(device)
        if belief_pad   is not None: belief_pad   = belief_pad.to(device)

        out = model(
            tokens=tokens,
            candidate_mask=candidate_mask,
            pad_mask=pad_mask,
            belief_tokens=belief_tokens,
            belief_pad_mask=belief_pad,
        )

        q = out.q.float()                              # (B, N)
        # 非法动作已置 -inf，取合法动作中的 argmax
        pred = q.argmax(dim=-1)                        # (B,)
        B = action_idx.size(0)

        # top-1
        top1_correct += (pred == action_idx).sum().item()

        # top-3（只在 ≥3 个合法动作时有意义）
        _, topk_idx = q.topk(min(3, q.size(-1)), dim=-1)
        for b in range(B):
            if action_idx[b] in topk_idx[b]:
                top3_correct += 1

        # mean rank：把 -inf 替换成极小值后 argsort
        q_safe = q.masked_fill(q == float('-inf'), -1e9)
        sorted_desc = q_safe.argsort(dim=-1, descending=True)  # (B, N)
        for b in range(B):
            rank = (sorted_desc[b] == action_idx[b]).nonzero(as_tuple=True)[0]
            rank_sum += (rank.item() + 1)   # 1-indexed

        # Q margin：best_Q - human_Q
        best_q  = q_safe.max(dim=-1).values                     # (B,)
        human_q = q_safe.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (B,)
        q_margin_sum += (best_q - human_q).sum().item()

        total += B

        if (i + 1) % 100 == 0:
            logger.info(
                f"[{label}] {i+1}/{n_batches} batches  "
                f"top1={top1_correct/total:.4f}  "
                f"top3={top3_correct/total:.4f}  "
                f"mean_rank={rank_sum/total:.2f}  "
                f"q_margin={q_margin_sum/total:.4f}"
            )

    n = max(total, 1)
    return {
        "top1_acc":    top1_correct  / n,
        "top3_acc":    top3_correct  / n,
        "mean_rank":   rank_sum      / n,
        "q_margin":    q_margin_sum  / n,
        "n_samples":   total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1",     required=True,  help="Stage 1 checkpoint 路径")
    parser.add_argument("--stage2",     required=True,  help="Stage 2 checkpoint 路径")
    parser.add_argument("--data_dir",   required=True,  help="annotated jsonl 数据目录")
    parser.add_argument("--n_batches",  type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_ratio",  type=float, default=0.02)
    parser.add_argument("--preset",     default="base")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 数据（只用 val split，保证没见过）──────────────────
    all_files = sorted(Path(args.data_dir).rglob("*.jsonl"))
    random.seed(args.seed)
    random.shuffle(all_files)
    val_n     = max(1, int(len(all_files) * args.val_ratio))
    val_files = all_files[:val_n]
    logger.info(f"Val files: {val_n}  (seed={args.seed})")

    val_ds = MjaiDataset(val_files, shuffle_files=False, shuffle_buffer=500, stage=1)
    loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # ── 加载两个模型 ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading Stage 1 model...")
    s1_model = load_model(args.stage1, device, args.preset)

    logger.info("Loading Stage 2 model...")
    s2_model = load_model(args.stage2, device, args.preset)

    # ── 评估 ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Evaluating Stage 1  ({args.n_batches} batches × {args.batch_size})")
    s1_res = evaluate(s1_model, loader, device, args.n_batches, "S1")

    # 重置 loader（IterableDataset 需要重新创建）
    val_ds2 = MjaiDataset(val_files, shuffle_files=False, shuffle_buffer=500, stage=1)
    loader2 = DataLoader(
        val_ds2, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    logger.info(f"Evaluating Stage 2  ({args.n_batches} batches × {args.batch_size})")
    s2_res = evaluate(s2_model, loader2, device, args.n_batches, "S2")

    # ── 结果汇总 ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info(f"{'Metric':<20} {'Stage1':>10} {'Stage2':>10} {'Delta':>10}")
    logger.info("-" * 54)
    for key in ("top1_acc", "top3_acc", "mean_rank", "q_margin"):
        v1, v2 = s1_res[key], s2_res[key]
        # rank 和 q_margin 越小越好，用负号表示改善方向
        if key in ("mean_rank", "q_margin"):
            delta_str = f"{v2 - v1:+.4f}  {'↓ better' if v2 < v1 else '↑ worse'}"
        else:
            delta_str = f"{v2 - v1:+.4f}  {'↑ better' if v2 > v1 else '↓ worse'}"
        logger.info(f"{key:<20} {v1:>10.4f} {v2:>10.4f} {delta_str}")
    logger.info(f"{'n_samples':<20} {s1_res['n_samples']:>10} {s2_res['n_samples']:>10}")
    logger.info("=" * 60)

    # 给一个综合判断
    top1_delta = s2_res["top1_acc"] - s1_res["top1_acc"]
    rank_delta = s2_res["mean_rank"] - s1_res["mean_rank"]
    if top1_delta > 0.005 and rank_delta < -0.05:
        logger.info("✓ Stage 2 明显提升了 BC accuracy，蒸馏有效，建议继续 Stage 3。")
    elif top1_delta > 0.001 or rank_delta < -0.01:
        logger.info("→ Stage 2 有轻微提升，蒸馏部分有效。可进 Stage 3，但效果增量有限。")
    else:
        logger.info("⚠ Stage 2 对 BC accuracy 无明显提升，蒸馏可能只学了 Oracle 的噪声。")
        logger.info("  建议：先对 Oracle 做有监督 fine-tune 后重跑 Stage 2，再进 Stage 3。")


if __name__ == "__main__":
    main()
