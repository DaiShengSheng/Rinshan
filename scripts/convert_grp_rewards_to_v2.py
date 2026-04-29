#!/usr/bin/env python3
"""
convert_grp_rewards_to_v2.py — broadcast grp_reward to GRP 2.0 v2 format.

旧版问题:
  同一 round 内所有 action 共享同一个 grp_reward (broadcast)。
转换目标:
  同一 round 内仅最后一个 action 保留 grp_reward，其余置 0.0。

Usage:
  python scripts/convert_grp_rewards_to_v2.py \
      --input data/annotated_grp \
      --output data/annotated_grp_v2 \
      --workers 16
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# IMPORTANT: Do NOT import torch or any rinshan model at module level here.
# top-level torch + fork() corrupts PyTorch thread pool in child processes,
# silently killing all workers and making the script run single-threaded.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_grp_rewards_to_v2")

RoundKey = tuple[str, int, int, int, int]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def _round_key(d: dict) -> RoundKey:
    return (
        str(d.get("game_id", "")),
        int(d.get("player_id", 0)),
        int(d.get("round_wind", 0)),
        int(d.get("round_num", 0)),
        int(d.get("honba", 0)),
    )


def _convert_lines(lines: list[dict]) -> tuple[list[dict], dict]:
    """仅 round 最后一步保留 grp_reward，其余置 0.0。"""
    grouped: dict[RoundKey, list[int]] = defaultdict(list)
    for i, d in enumerate(lines):
        grouped[_round_key(d)].append(i)

    stats = dict(rounds=0, actions=len(lines),
                 nonzero_before=0, nonzero_after=0, zeroed_actions=0)

    for d in lines:
        if abs(_safe_float(d.get("grp_reward", 0.0))) > 1e-12:
            stats["nonzero_before"] += 1

    for idxs in grouped.values():
        stats["rounds"] += 1
        # 取 round 内最后一个非零 reward（兼容脏数据）
        reward = None
        for i in reversed(idxs):
            r = _safe_float(lines[i].get("grp_reward", 0.0))
            if abs(r) > 1e-12:
                reward = r
                break
        if reward is None:
            reward = _safe_float(lines[idxs[-1]].get("grp_reward", 0.0))

        for i in idxs[:-1]:
            if abs(_safe_float(lines[i].get("grp_reward", 0.0))) > 1e-12:
                stats["zeroed_actions"] += 1
            lines[i]["grp_reward"] = 0.0
            lines[i]["hand_reward"] = 0.0

        lines[idxs[-1]]["grp_reward"] = reward
        lines[idxs[-1]]["hand_reward"] = _safe_float(
            lines[idxs[-1]].get("round_delta_score", 0.0)) / 1000.0
        if abs(reward) > 1e-12:
            stats["nonzero_after"] += 1

    return lines, stats


def _process_file(args: tuple[str, str, bool]) -> dict:
    """Worker 函数（顶层定义，确保 pickle 可序列化）。"""
    in_s, out_s, keep_invalid = args
    try:
        in_path, out_path = Path(in_s), Path(out_s)
        raw_lines, parsed, positions, n_err = [], [], [], 0

        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    if keep_invalid:
                        raw_lines.append(raw)
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    n_err += 1
                    if keep_invalid:
                        raw_lines.append(raw)
                    continue
                positions.append(len(raw_lines))
                parsed.append(obj)
                raw_lines.append(raw)

        if not parsed and n_err == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("", encoding="utf-8")
            return dict(file=in_s, written=True, actions=0, rounds=0,
                        nonzero_before=0, nonzero_after=0,
                        zeroed_actions=0, invalid_json_lines=0)

        converted, stats = _convert_lines(parsed)

        if keep_invalid:
            for pos, d in zip(positions, converted):
                raw_lines[pos] = json.dumps(d, ensure_ascii=False)
            text = "\n".join(raw_lines) + ("\n" if raw_lines else "")
        else:
            text = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in converted)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return dict(file=in_s, written=True, **stats, invalid_json_lines=n_err)

    except Exception as e:
        return dict(file=in_s, written=False, actions=0, rounds=0,
                    nonzero_before=0, nonzero_after=0,
                    zeroed_actions=0, invalid_json_lines=0,
                    error=f"{type(e).__name__}: {e}", pid=os.getpid())


def _build_file_list(d: Path, sample: int | None, seed: int, max_f: int | None) -> list[Path]:
    files = sorted(d.rglob("*.jsonl"))
    if sample and sample < len(files):
        files = sorted(random.Random(seed).sample(files, sample))
    if max_f:
        files = files[:max_f]
    return files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--workers", "-w", type=int, default=16,
                   help="并行 worker 数，建议 = CPU 核心数（默认 16）")
    p.add_argument("--sample-files", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-invalid-lines", action="store_true")
    p.add_argument("--chunksize", type=int, default=8,
                   help="imap_unordered chunksize（默认 8）")
    args = p.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.exists():
        p.error(f"输入目录不存在: {in_dir}")

    files = _build_file_list(in_dir, args.sample_files, args.seed, args.max_files)
    if not files:
        logger.info("No .jsonl files found.")
        return

    tasks, skipped = [], 0
    for f in files:
        out = out_dir / f.relative_to(in_dir)
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        tasks.append((str(f), str(out), bool(args.keep_invalid_lines)))

    logger.info("files=%d  todo=%d  skipped=%d  workers=%d  chunksize=%d",
                len(files), len(tasks), skipped, args.workers, args.chunksize)
    if not tasks:
        logger.info("Nothing to do.")
        return

    tot = dict(files=0, actions=0, rounds=0, nonzero_before=0,
               nonzero_after=0, zeroed_actions=0, invalid_json_lines=0)
    failed = 0

    # mp.Pool — works on Python 3.8+, no ProcessPoolExecutor version issues
    with mp.Pool(processes=args.workers) as pool:
        for idx, r in enumerate(
            pool.imap_unordered(_process_file, tasks, chunksize=args.chunksize),
            start=1
        ):
            if not r.get("written"):
                failed += 1
                logger.warning("FAIL pid=%s  %s  %s",
                               r.get("pid","?"), r["file"], r.get("error",""))
            else:
                tot["files"]           += 1
                tot["actions"]         += r["actions"]
                tot["rounds"]          += r["rounds"]
                tot["nonzero_before"]  += r["nonzero_before"]
                tot["nonzero_after"]   += r["nonzero_after"]
                tot["zeroed_actions"]  += r["zeroed_actions"]
                tot["invalid_json_lines"] += r["invalid_json_lines"]

            if idx % 100 == 0 or idx == len(tasks):
                logger.info("  %d/%d  actions=%s  rounds=%s  zeroed=%s  failed=%d",
                            idx, len(tasks),
                            f"{tot['actions']:,}", f"{tot['rounds']:,}",
                            f"{tot['zeroed_actions']:,}", failed)

    logger.info("Done.")
    for k, v in tot.items():
        logger.info("  %-26s %s", k, f"{v:,}")
    logger.info("  %-26s %d", "failed_files", failed)


if __name__ == "__main__":
    main()
