"""
fill_grp_rewards.py — 用训练好的 GRP 模型为 .jsonl 标注文件填充 grp_reward 字段

必须在 train_grp.py 训练完成后运行。

Usage:
  python scripts/fill_grp_rewards.py configs/grp.yaml
  python scripts/fill_grp_rewards.py --data data/annotated/ --grp checkpoints/grp/grp_best.pt --output data/annotated_grp/

关键设计：
  - 使用 spawn 上下文的 ProcessPoolExecutor，规避 Linux fork+CUDA 崩溃
  - 每个 worker 独立加载 GRP 到 CUDA，并行消化 CPU 瓶颈（JSON解析/tensor构建）
  - 每个文件内做 mega-batch forward，一次 kernel call 覆盖整文件所有游戏
  - 断点续跑：跳过已存在的输出文件
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from rinshan.model.grp import GRP, RewardCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 顶层函数（spawn worker 可以 pickle）
# ─────────────────────────────────────────────────────────────

def _process_file(args_tuple):
    """
    spawn worker 入口：每个 worker 进程独立加载 GRP 到 CUDA。
    用 args_tuple 传参以兼容 ProcessPoolExecutor.map。
    """
    in_path, out_path, grp_path, platform = args_tuple

    import torch
    from rinshan.model.grp import GRP, RewardCalculator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grp = GRP()
    ckpt = torch.load(grp_path, map_location=device, weights_only=True)
    grp.load_state_dict(ckpt["model"])
    grp = grp.to(device).eval()
    rc = RewardCalculator(grp, platform=platform)
    rc.pts = rc.pts.to(device)

    return _fill_file_impl(Path(in_path), Path(out_path), rc)


def _fill_file_impl(in_path: Path, out_path: Path, reward_calc) -> int:
    """
    为一个 .jsonl 文件填写 grp_reward。
    整个文件所有游戏所有前缀序列打成一个 mega-batch，只做一次 GRP forward。
    """
    lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    if not lines:
        return 0

    device = next(reward_calc.grp.parameters()).device
    pts    = reward_calc.pts.to(device)

    # ── 分组 ──────────────────────────────────────────────────
    groups: dict = defaultdict(list)
    for i, d in enumerate(lines):
        groups[(d.get("game_id", ""), d.get("player_id", 0))].append(i)

    # ── 构建 mega-batch ────────────────────────────────────────
    game_meta   = []
    all_seqs    = []
    game_slices = []

    for (_, _player_id), idxs in groups.items():
        idxs.sort(key=lambda i: (
            lines[i].get("round_wind", 0),
            lines[i].get("round_num",  1),
        ))

        seen_rounds: set = set()
        frames = []
        for i in idxs:
            d  = lines[i]
            rw = d.get("round_wind", 0)
            rn = d.get("round_num",  1)
            if (rw, rn) not in seen_rounds:
                seen_rounds.add((rw, rn))
                pid    = d.get("player_id", 0)
                scores = d.get("scores", [250] * 4)
                abs_s  = scores[4 - pid:] + scores[:4 - pid]
                frames.append([
                    float(rw * 4 + rn - 1),
                    float(d.get("honba",   0)),
                    float(d.get("kyotaku", 0)),
                    abs_s[0] / 1e4, abs_s[1] / 1e4,
                    abs_s[2] / 1e4, abs_s[3] / 1e4,
                ])

        if len(frames) < 2:
            continue

        last = lines[idxs[-1]]
        pid  = last.get("player_id", 0)
        if "final_rank" not in last:
            continue

        frames_t = torch.tensor(frames, dtype=torch.float64, device=device)
        T        = frames_t.shape[0]

        start = len(all_seqs)
        for t in range(1, T + 1):
            all_seqs.append(frames_t[:t])
        game_slices.append((start, len(all_seqs)))
        game_meta.append((pid, sorted(seen_rounds), idxs, last["final_rank"]))

    if not all_seqs:
        return 0

    # ── 一次 mega-batch forward ────────────────────────────────
    with torch.no_grad():
        all_logits = reward_calc.grp(all_seqs)
        all_matrix = reward_calc.grp.calc_matrix(all_logits)  # (N_prefixes, 4, 4)

    # ── 写回 rewards ───────────────────────────────────────────
    n_filled = 0
    for (pid, seen_rounds_sorted, idxs, final_rank_val), (start, end) in zip(
        game_meta, game_slices
    ):
        rank_probs = all_matrix[start:end, pid, :]
        exp_pts    = (rank_probs.double() * pts).sum(dim=-1)
        all_pts    = torch.cat([exp_pts, pts[final_rank_val].unsqueeze(0)])
        rewards    = (all_pts[1:] - all_pts[:-1]).cpu().tolist()

        round_to_reward = {
            (rw, rn): rewards[j]
            for j, (rw, rn) in enumerate(seen_rounds_sorted)
            if j < len(rewards)
        }
        for i in idxs:
            d  = lines[i]
            lines[i]["grp_reward"] = round_to_reward.get(
                (d.get("round_wind", 0), d.get("round_num", 1)), 0.0
            )
            n_filled += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return n_filled


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",      nargs="?", help="grp.yaml 配置文件")
    parser.add_argument("--data",   "-d", default=None)
    parser.add_argument("--grp",    "-g", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--platform",     default=None, choices=["tenhou", "jantama"])
    parser.add_argument("--workers", "-w", type=int, default=None, help="并行 worker 数（默认 8）")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if args.data     is None: args.data     = cfg.get("data_dir", None)
        if args.grp      is None:
            args.grp = str(Path(cfg.get("save_dir", "checkpoints/grp")) / "grp_best.pt")
        if args.platform is None: args.platform = cfg.get("platform", "tenhou")
        if args.workers  is None: args.workers  = int(cfg.get("fill_workers", 8))
    if args.platform is None: args.platform = "tenhou"
    if args.workers  is None: args.workers  = 8

    for name, val in [("--data", args.data), ("--grp", args.grp), ("--output", args.output)]:
        if not val:
            parser.error(f"{name} 是必填项")

    in_dir  = Path(args.data)
    out_dir = Path(args.output)
    files   = sorted(in_dir.rglob("*.jsonl"))
    todo    = [p for p in files if not (out_dir / p.relative_to(in_dir)).exists()]
    logger.info(f"Total: {len(files)} | Done: {len(files)-len(todo)} | Todo: {len(todo)}")
    logger.info(f"Workers: {args.workers}  (spawn, each loads GRP onto CUDA independently)")

    if not todo:
        logger.info("Nothing to do.")
        return

    task_args = [
        (str(p), str(out_dir / p.relative_to(in_dir)), args.grp, args.platform)
        for p in todo
    ]

    total     = 0
    completed = 0

    # spawn 上下文：每个 worker 是全新 Python 进程，不继承 CUDA 状态，不会崩溃
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=spawn_ctx) as pool:
        futures = {pool.submit(_process_file, a): a[0] for a in task_args}
        for fut in as_completed(futures):
            try:
                total += fut.result()
            except Exception as e:
                logger.warning(f"Failed {futures[fut]}: {e}")
            completed += 1
            if completed % 50 == 0 or completed == len(todo):
                logger.info(f"  {completed}/{len(todo)} files — {total:,} rewards filled")

    logger.info(f"Done. Total {total:,} grp_reward fields filled.")


if __name__ == "__main__":
    main()
