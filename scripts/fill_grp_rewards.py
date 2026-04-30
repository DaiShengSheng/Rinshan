"""
fill_grp_rewards.py — 用训练好的 GRP 模型为 .jsonl 标注文件填充 grp_reward 字段

必须在 train_grp.py 训练完成后运行。

Usage:
  python scripts/fill_grp_rewards.py configs/grp.yaml
  python scripts/fill_grp_rewards.py --data data/annotated/ --grp checkpoints/grp/grp_best.pt --output data/annotated_grp/

关键设计：
  - 输出格式为 GRP 2.0：每个 round（以 round_wind/round_num/honba 为唯一键）内
    只有最后一个 action 保留 grp_reward 和 hand_reward，其余 action 均置 0.0。
    这与 dataset.py 的 Stage3 逻辑一致（跨局/终局时才结算 game reward）。
  - 使用 spawn 上下文的 ProcessPoolExecutor，规避 Linux fork+CUDA 崩溃
  - 每个 worker 独立加载 GRP 到 CUDA，并行消化 CPU 瓶颈（JSON解析/tensor构建）
  - 每个文件内做 mega-batch forward，一次 kernel call 覆盖整文件所有游戏
  - 断点续跑：跳过已存在的输出文件

注意：
  convert_grp_rewards_to_v2.py 是历史迁移工具（用于将旧 1.0 广播格式数据转换为
  2.0 格式），新填充的数据直接是 2.0 格式，无需再运行该迁移脚本。
  analyze_grp_rewards.py 可用于验证填充结果：broadcast_round_ratio 应接近 0。
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson

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
    with open(in_path, "rb") as f:          # rb + orjson: ~3x faster than json
        for raw in f:
            raw = raw.strip()
            if raw:
                try:
                    lines.append(orjson.loads(raw))
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
            lines[i].get("honba", 0),
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

    # ── 写回 rewards（GRP 2.0：每 round 只有最后一个 action 保留非零值）──
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

        # 按 round key（含 honba）将 idxs 分组，每组只有最后一个 action 写非零值
        # round key = (round_wind, round_num, honba)，与 dataset.py grp_state_key 一致
        from collections import defaultdict as _dd
        round_groups: dict = _dd(list)
        for i in idxs:
            d = lines[i]
            rkey = (d.get("round_wind", 0), d.get("round_num", 1), d.get("honba", 0))
            round_groups[rkey].append(i)

        for rkey, ridxs in round_groups.items():
            rw, rn, _ = rkey
            grp_r = round_to_reward.get((rw, rn), 0.0)

            # 局内中间 action：grp_reward=0，hand_reward=0
            for i in ridxs[:-1]:
                lines[i]["grp_reward"]  = 0.0
                lines[i]["hand_reward"] = 0.0

            # 该 round 最后一个 action：填真实奖励
            last_i = ridxs[-1]
            lines[last_i]["grp_reward"]  = grp_r
            lines[last_i]["hand_reward"] = float(
                lines[last_i].get("round_delta_score", 0.0)
            ) / 1000.0

        n_filled += len(idxs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for d in lines:
            f.write(orjson.dumps(d))
            f.write(b"\n")

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
            # 优先读 grp_ckpt，没有再回退到 save_dir/grp_best.pt
            if "grp_ckpt" in cfg:
                args.grp = cfg["grp_ckpt"]
            else:
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
