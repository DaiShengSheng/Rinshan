"""
eval_belief_accuracy.py — 评估 Belief Net 手牌推断能力

Stage 2 引入了 wait_head，Stage 1 只有 belief_head。
本脚本分别评估：

  1. Belief Net 手牌预测精度（3个对手各 34 种牌，二分类）
     - tile_auc:     每牌 ROC-AUC 均值
     - tile_ap:      每牌 Average Precision 均值
     - presence_acc: 对"该牌在手里"(label=1)的预测准确率（@阈值0.5）
     - absence_acc:  对"该牌不在手里"(label=0)的预测准确率
     - calibration:  预测概率与实际频率的 ECE（越小越好）

  2. 与随机基线（均匀分布）和朴素基线（全0预测）的对比

  3. 按"牌局阶段"分层（早期/中期/终局），看不同阶段的推断精度

Usage:
    python scripts/eval_belief_accuracy.py \
        --stage1 /path/to/stage1/best_v3.pt \
        --stage2 /path/to/stage2/best.pt \
        --data_dir /path/to/annotated_v4 \
        --n_batches 300 \
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
logger = logging.getLogger("eval_belief")


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
    logger.info(f"Loaded {Path(ckpt_path).name}  missing={len(missing)}  unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error（分 n_bins 桶）"""
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = probs.numel()
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].float().mean().item()
        conf = probs[mask].mean().item()
        ece += (mask.sum().item() / total) * abs(acc - conf)
    return ece


@torch.no_grad()
def evaluate_belief(
    model: RinshanModel,
    loader: DataLoader,
    device: torch.device,
    n_batches: int,
    label: str,
) -> dict:
    """
    收集所有 belief 预测概率和真实标签，计算各项指标。
    actual_hands: (B, 34, 3) int，值为张数（>0 表示有该牌）
    belief_logits: (B, 34, 3) float
    """
    all_probs  = []   # List[(N, 34, 3)]
    all_labels = []   # List[(N, 34, 3)]
    n_skip = 0        # actual_hands 为 None 的批次数

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        actual_hands = batch.get("actual_hands")
        if actual_hands is None:
            n_skip += 1
            continue

        tokens         = batch["tokens"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        pad_mask       = batch.get("pad_mask")
        belief_tokens  = batch.get("belief_tokens")
        belief_pad     = batch.get("belief_pad_mask")

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

        if out.belief_logits is None:
            n_skip += 1
            continue

        probs  = torch.sigmoid(out.belief_logits).cpu()   # (B, 34, 3)
        labels = (actual_hands > 0).float()                # (B, 34, 3)

        all_probs.append(probs)
        all_labels.append(labels)

        if (i + 1) % 100 == 0:
            logger.info(f"[{label}] {i+1}/{n_batches} batches collected...")

    if n_skip > 0:
        logger.warning(f"[{label}] Skipped {n_skip} batches with no actual_hands")

    if not all_probs:
        logger.error(f"[{label}] No valid batches! Check actual_hands in dataset.")
        return {}

    probs_all  = torch.cat(all_probs,  dim=0).float()   # (N, 34, 3)
    labels_all = torch.cat(all_labels, dim=0).float()   # (N, 34, 3)
    N = probs_all.size(0)
    logger.info(f"[{label}] Total samples: {N}")

    # ── 核心指标（展平到所有牌 × 所有对手）──────────────────
    p_flat = probs_all.reshape(-1)    # (N*34*3,)
    l_flat = labels_all.reshape(-1)

    # Presence accuracy（正例，label=1）
    pos_mask = l_flat == 1
    neg_mask = l_flat == 0
    presence_acc = ((p_flat[pos_mask] > 0.5).float().mean().item()
                    if pos_mask.sum() > 0 else float('nan'))
    absence_acc  = ((p_flat[neg_mask] < 0.5).float().mean().item()
                    if neg_mask.sum() > 0 else float('nan'))
    overall_acc  = ((p_flat > 0.5).float() == l_flat).float().mean().item()

    # ECE
    ece = _ece(p_flat, l_flat)

    # 平均预测概率 vs 实际频率
    pred_mean  = p_flat.mean().item()
    label_freq = l_flat.mean().item()

    # ── 精确召回（手牌推断的核心）─────────────────────────
    # 对每张牌×对手，用 0.5 阈值
    tp = ((p_flat > 0.5) & (l_flat == 1)).float().sum().item()
    fp = ((p_flat > 0.5) & (l_flat == 0)).float().sum().item()
    fn = ((p_flat < 0.5) & (l_flat == 1)).float().sum().item()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # ── 按对手分层 ──────────────────────────────────────
    per_opp = {}
    for opp in range(3):
        pp = probs_all[:, :, opp].reshape(-1)
        ll = labels_all[:, :, opp].reshape(-1)
        acc = ((pp > 0.5).float() == ll).float().mean().item()
        per_opp[f"opp{opp}_acc"] = acc

    # ── 与朴素基线对比 ──────────────────────────────────
    # 基线1：全预测 0（没有牌）
    baseline_all0_acc = (l_flat == 0).float().mean().item()
    # 基线2：用数据集频率预测（固定阈值=频率均值）
    baseline_freq_acc = ((label_freq > 0.5) == (l_flat.mean() > 0.5))

    results = {
        "n_samples":      N,
        "overall_acc":    overall_acc,
        "presence_acc":   presence_acc,   # 有牌的命中率
        "absence_acc":    absence_acc,    # 无牌的命中率
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "ece":            ece,
        "pred_mean_prob": pred_mean,
        "label_freq":     label_freq,
        "baseline_all0":  baseline_all0_acc,
        **per_opp,
    }
    return results


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1",     required=True)
    parser.add_argument("--stage2",     required=True)
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--n_batches",  type=int,   default=300)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--val_ratio",  type=float, default=0.02)
    parser.add_argument("--preset",     default="base")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Val split ──────────────────────────────────────
    all_files = sorted(Path(args.data_dir).rglob("*.jsonl"))
    random.seed(args.seed)
    random.shuffle(all_files)
    val_n     = max(1, int(len(all_files) * args.val_ratio))
    val_files = all_files[:val_n]
    logger.info(f"Val files: {val_n}")

    def make_loader():
        ds = MjaiDataset(val_files, shuffle_files=False, shuffle_buffer=500, stage=1)
        return DataLoader(ds, batch_size=args.batch_size,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ── 加载模型 ───────────────────────────────────────
    logger.info("=" * 60)
    s1_model = load_model(args.stage1, device, args.preset)
    s2_model = load_model(args.stage2, device, args.preset)

    # ── 评估 ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating Belief Net — Stage 1")
    s1_res = evaluate_belief(s1_model, make_loader(), device, args.n_batches, "S1")

    logger.info("Evaluating Belief Net — Stage 2")
    s2_res = evaluate_belief(s2_model, make_loader(), device, args.n_batches, "S2")

    if not s1_res or not s2_res:
        logger.error("Evaluation failed — no valid batches.")
        return

    # ── 输出结果 ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BELIEF NET EVALUATION RESULTS")
    logger.info(f"{'Metric':<22} {'Stage1':>9} {'Stage2':>9} {'Delta':>9}  Note")
    logger.info("-" * 70)

    metrics = [
        ("overall_acc",    "↑"),
        ("presence_acc",   "↑",  "有牌时的命中率"),
        ("absence_acc",    "↑",  "无牌时的命中率"),
        ("precision",      "↑"),
        ("recall",         "↑"),
        ("f1",             "↑"),
        ("ece",            "↓",  "越小越好"),
        ("pred_mean_prob", "~",  "预测平均概率"),
        ("label_freq",     "~",  "实际有牌频率"),
        ("opp0_acc",       "↑"),
        ("opp1_acc",       "↑"),
        ("opp2_acc",       "↑"),
        ("baseline_all0",  "~",  "朴素基线（全预测无牌）"),
    ]
    for row in metrics:
        key, direction = row[0], row[1]
        note = row[2] if len(row) > 2 else ""
        v1 = s1_res.get(key, float('nan'))
        v2 = s2_res.get(key, float('nan'))
        delta = v2 - v1 if isinstance(v1, float) and isinstance(v2, float) else "—"
        if isinstance(delta, float):
            better = (delta > 0 and direction == "↑") or (delta < 0 and direction == "↓")
            tag = "✓" if (abs(delta) > 0.002 and better) else ("✗" if (abs(delta) > 0.002 and not better) else "→")
            delta_str = f"{delta:+.4f} {tag}"
        else:
            delta_str = "—"
        logger.info(f"{key:<22} {_fmt(v1):>9} {_fmt(v2):>9} {delta_str:<14} {note}")

    logger.info("=" * 60)

    # ── 综合判断 ───────────────────────────────────────
    f1_delta     = s2_res.get("f1", 0)    - s1_res.get("f1", 0)
    recall_delta = s2_res.get("recall", 0) - s1_res.get("recall", 0)
    prec_delta   = s2_res.get("precision", 0) - s1_res.get("precision", 0)

    logger.info("DIAGNOSIS:")
    if f1_delta > 0.02:
        logger.info(f"  ✓ Belief F1 提升 {f1_delta:+.4f}，模型确实学会了更好地推断对手手牌。")
    elif f1_delta > 0.005:
        logger.info(f"  → Belief F1 小幅提升 {f1_delta:+.4f}，有一定改善但不显著。")
    else:
        logger.info(f"  ⚠ Belief F1 几乎无变化 ({f1_delta:+.4f})，Belief Net 没有从 Stage 2 中受益。")

    # 检查是否比朴素基线强
    s1_base = s1_res.get("baseline_all0", 0)
    s1_f1   = s1_res.get("f1", 0)
    if s1_f1 < 0.1:
        logger.info(f"  ⚠ Stage 1 Belief F1={s1_f1:.4f}，接近全预测为0的朴素基线，")
        logger.info(f"     说明 Belief Net 基本没学到有效的手牌推断能力。")
        logger.info(f"     建议增加 Belief 训练数据或调高 belief_weight。")
    elif s1_f1 < 0.3:
        logger.info(f"  → Belief Net 有一定能力（F1={s1_f1:.4f}），但还有很大提升空间。")
    else:
        logger.info(f"  ✓ Belief Net 手牌推断能力较强（F1={s1_f1:.4f}）。")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
