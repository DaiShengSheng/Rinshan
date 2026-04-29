#!/usr/bin/env python3
"""
convert_grp_rewards_to_v2.py — 将旧版 annotated_grp 广播式 grp_reward 转为 GRP 2.0 第一阶段格式。

旧版问题：
  - 同一 (game_id, player_id, round_wind, round_num, honba) 内的所有 action
    都带着同一个 grp_reward（broadcast）

转换目标：
  - 同一局内仅保留“最后一个 action”的 grp_reward
  - 该局内其余 action 的 grp_reward 统一置 0.0
  - 目录结构保持不变，输出到新的 data_dir

特点：
  - 支持多 worker 并行
  - 使用 spawn 上下文，和现有 fill_grp_rewards.py 风格一致
  - 支持断点续跑（默认跳过已存在输出文件）
  - 支持 sample/max-files，便于小规模验证

Usage:
  python scripts/convert_grp_rewards_to_v2.py \
      --input data/annotated_grp \
      --output data/annotated_grp_v2 \
      --workers 8

  python scripts/convert_grp_rewards_to_v2.py \
      --input data/annotated_grp \
      --output data/annotated_grp_v2 \
      --sample-files 200 --seed 42 --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# torch / GRP are NOT imported at module level — top-level PyTorch import
# combined with fork(2) corrupts the thread pool in child processes and
# causes every worker to silently crash or run single-threaded.
# Import them lazily inside functions that actually need them.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_grp_rewards_to_v2")


RoundKey = tuple[str, int, int, int, int]


def _build_grp_frames(rows: list[dict[str, Any]], player_id: int) -> torch.Tensor:
    seen_rounds: set[tuple[int, int, int]] = set()
    frames: list[list[float]] = []
    for d in rows:
        rw = int(d.get("round_wind", 0))
        rn = int(d.get("round_num", 0))
        hb = int(d.get("honba", 0))
        if (rw, rn, hb) in seen_rounds:
            continue
        seen_rounds.add((rw, rn, hb))
        scores = list(d.get("scores", [25000] * 4))
        abs_scores = scores[4 - player_id:] + scores[:4 - player_id]
        frames.append([
            float(rw * 4 + rn - 1),
            float(hb),
            float(d.get("kyotaku", 0)),
            abs_scores[0] / 1e4,
            abs_scores[1] / 1e4,
            abs_scores[2] / 1e4,
            abs_scores[3] / 1e4,
        ])
    if not frames:
        return torch.zeros((0, 7), dtype=torch.float64)
    return torch.tensor(frames, dtype=torch.float64)


def _compute_transition_hand_rewards(lines: list[dict[str, Any]], device: str, grp_ckpt: str | None, platform: str) -> list[float]:
    # NOTE: torch / GRP are imported lazily (not at module level) to avoid the
    # fork+PyTorch thread-pool corruption that would cause every worker to crash.
    # Current implementation uses only round_delta_score; grp_ckpt is reserved.
    hand_rewards = [0.0 for _ in lines]
    grouped_indices: dict[RoundKey, list[int]] = defaultdict(list)
    for idx, d in enumerate(lines):
        grouped_indices[_round_key(d)].append(idx)
    for _, idxs in grouped_indices.items():
        if not idxs:
            continue
        last_i = idxs[-1]
        hand_rewards[last_i] = _safe_float(lines[last_i].get("round_delta_score", 0.0)) / 1000.0
    return hand_rewards


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_key(d: dict[str, Any]) -> RoundKey:
    return (
        str(d.get("game_id", "")),
        int(d.get("player_id", 0)),
        int(d.get("round_wind", 0)),
        int(d.get("round_num", 0)),
        int(d.get("honba", 0)),
    )


def _convert_lines(lines: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """把单文件的广播式 reward 改成“仅 round 最后一步保留 reward”格式。"""
    grouped_indices: dict[RoundKey, list[int]] = defaultdict(list)
    for idx, d in enumerate(lines):
        grouped_indices[_round_key(d)].append(idx)

    stats = {
        "rounds": 0,
        "actions": len(lines),
        "nonzero_before": 0,
        "nonzero_after": 0,
        "zeroed_actions": 0,
    }

    for d in lines:
        if abs(_safe_float(d.get("grp_reward", 0.0))) > 1e-12:
            stats["nonzero_before"] += 1

    for _, idxs in grouped_indices.items():
        stats["rounds"] += 1
        if not idxs:
            continue

        # 广播式 reward 理论上 round 内一致；为兼容脏数据，取最后一个非零值，
        # 若全为 0，则退化为最后一条的原值。
        reward_value = None
        for i in reversed(idxs):
            r = _safe_float(lines[i].get("grp_reward", 0.0))
            if abs(r) > 1e-12:
                reward_value = r
                break
        if reward_value is None:
            reward_value = _safe_float(lines[idxs[-1]].get("grp_reward", 0.0))

        for i in idxs[:-1]:
            old_r = _safe_float(lines[i].get("grp_reward", 0.0))
            if abs(old_r) > 1e-12:
                stats["zeroed_actions"] += 1
            lines[i]["grp_reward"] = 0.0
            lines[i]["hand_reward"] = _safe_float(lines[i].get("round_delta_score", 0.0)) / 1000.0

        lines[idxs[-1]]["grp_reward"] = reward_value
        lines[idxs[-1]]["hand_reward"] = _safe_float(lines[idxs[-1]].get("round_delta_score", 0.0)) / 1000.0
        if abs(reward_value) > 1e-12:
            stats["nonzero_after"] += 1

    return lines, stats


def _process_file(args_tuple: tuple[str, str, bool, str | None, str]) -> dict[str, Any]:
    in_path_s, out_path_s, keep_invalid_lines, grp_ckpt, platform = args_tuple
    try:
        return _process_file_impl(in_path_s, out_path_s, keep_invalid_lines, grp_ckpt, platform)
    except BaseException as e:
        # 尽量把 worker 内异常转成普通失败结果，而不是直接打爆整个进程池。
        return {
            "file": str(in_path_s),
            "written": False,
            "actions": 0,
            "rounds": 0,
            "nonzero_before": 0,
            "nonzero_after": 0,
            "zeroed_actions": 0,
            "invalid_json_lines": 0,
            "error": f"{type(e).__name__}: {e}",
            "pid": os.getpid(),
        }


def _process_file_impl(in_path_s: str, out_path_s: str, keep_invalid_lines: bool, grp_ckpt: str | None, platform: str) -> dict[str, Any]:
    in_path = Path(in_path_s)
    out_path = Path(out_path_s)

    raw_lines: list[str] = []
    parsed_lines: list[dict[str, Any]] = []
    parsed_positions: list[int] = []
    invalid_count = 0

    with in_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                if keep_invalid_lines:
                    raw_lines.append(raw)
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                invalid_count += 1
                if keep_invalid_lines:
                    raw_lines.append(raw)
                continue
            parsed_positions.append(len(raw_lines))
            parsed_lines.append(parsed)
            raw_lines.append(raw)

    if not parsed_lines and invalid_count == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return {
            "file": str(in_path),
            "written": True,
            "actions": 0,
            "rounds": 0,
            "nonzero_before": 0,
            "nonzero_after": 0,
            "zeroed_actions": 0,
            "invalid_json_lines": 0,
        }

    converted, stats = _convert_lines(parsed_lines)
    hand_rewards = _compute_transition_hand_rewards(converted, device="cpu", grp_ckpt=grp_ckpt, platform=platform)
    for d, hand_r in zip(converted, hand_rewards):
        d["hand_reward"] = float(hand_r)

    if keep_invalid_lines:
        for pos, d in zip(parsed_positions, converted):
            raw_lines[pos] = json.dumps(d, ensure_ascii=False)
        output_text = "\n".join(raw_lines)
        if raw_lines:
            output_text += "\n"
    else:
        output_text = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in converted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_text, encoding="utf-8")

    return {
        "file": str(in_path),
        "written": True,
        **stats,
        "invalid_json_lines": invalid_count,
    }


def _build_file_list(input_dir: Path, sample_files: int | None, seed: int, max_files: int | None) -> list[Path]:
    files = sorted(input_dir.rglob("*.jsonl"))
    if sample_files is not None and sample_files > 0:
        rng = random.Random(seed)
        if sample_files < len(files):
            files = rng.sample(files, sample_files)
            files.sort()
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    return files


def _iter_completed_futures(pool: ProcessPoolExecutor, tasks: list[tuple[str, str, bool, str | None, str]], submit_batch: int):
    """限制 in-flight future 数量，避免一次性把大量大文件任务全压进进程池。"""
    pending = {}
    task_iter = iter(tasks)

    def _submit_up_to(limit: int) -> None:
        while len(pending) < limit:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            fut = pool.submit(_process_file, task)
            pending[fut] = task[0]

    _submit_up_to(max(1, submit_batch))
    while pending:
        for fut in as_completed(list(pending.keys()), timeout=None):
            src = pending.pop(fut)
            yield src, fut
            _submit_up_to(max(1, submit_batch))
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert broadcast grp_reward files to GRP 2.0 v2 format")
    parser.add_argument("--input", "-i", required=True, help="旧版 annotated_grp 输入目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录，例如 data/annotated_grp_v2")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并行 worker 数")
    parser.add_argument("--sample-files", type=int, default=None, help="随机抽样文件数")
    parser.add_argument("--seed", type=int, default=42, help="sample-files 使用的随机种子")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理前 N 个文件")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在输出文件")
    parser.add_argument(
        "--keep-invalid-lines",
        action="store_true",
        help="保留无法解析的原始行（原样抄到输出），默认跳过非法 JSON 行",
    )
    parser.add_argument("--grp-ckpt", type=str, default=None, help="可选：GRP checkpoint 路径（为未来更细粒度 relabel 预留）")
    parser.add_argument("--platform", choices=["tenhou", "jantama"], default="tenhou", help="GRP reward 平台类型")
    parser.add_argument(
        "--start-method",
        choices=["auto", "fork", "spawn", "forkserver"],
        default="auto",
        help="多进程启动方式；Linux 默认建议 fork，Windows 自动用 spawn",
    )
    parser.add_argument(
        "--submit-batch",
        type=int,
        default=None,
        help="限制同时 in-flight 的任务数；默认 = workers * 2，降低大文件并发导致的 OOM 风险",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=20,
        help="每个 worker 最多处理多少个文件后重启，降低长时间运行的内存膨胀",
    )
    parser.add_argument(
        "--fallback-sequential",
        action="store_true",
        help="当进程池崩溃或 worker 被杀时，对失败文件自动单进程重试",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        parser.error(f"输入目录不存在: {input_dir}")

    files = _build_file_list(input_dir, args.sample_files, args.seed, args.max_files)
    if not files:
        logger.info("No .jsonl files found. Nothing to do.")
        return

    tasks: list[tuple[str, str, bool, str | None, str]] = []
    skipped = 0
    for p in files:
        out_path = output_dir / p.relative_to(input_dir)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        tasks.append((str(p), str(out_path), bool(args.keep_invalid_lines), args.grp_ckpt, args.platform))

    submit_batch = args.submit_batch if args.submit_batch is not None else max(1, args.workers * 2)
    if args.start_method == "auto":
        if sys.platform.startswith("win"):
            start_method = "spawn"
        else:
            start_method = "fork"
    else:
        start_method = args.start_method

    logger.info(
        "Total files: %s | Todo: %s | Skipped(existing): %s | Workers: %s | start_method=%s | submit_batch=%s | max_tasks_per_child=%s",
        len(files), len(tasks), skipped, args.workers, start_method, submit_batch, args.max_tasks_per_child,
    )

    if not tasks:
        logger.info("Nothing to do.")
        return

    totals = {
        "files": 0,
        "actions": 0,
        "rounds": 0,
        "nonzero_before": 0,
        "nonzero_after": 0,
        "zeroed_actions": 0,
        "invalid_json_lines": 0,
    }

    failed_tasks: list[tuple[str, str, bool]] = []
    ctx = mp.get_context(start_method)
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            max_tasks_per_child=max(1, args.max_tasks_per_child),
        ) as pool:
            for idx, (src, fut) in enumerate(_iter_completed_futures(pool, tasks, submit_batch), start=1):
                try:
                    result = fut.result()
                except Exception as e:
                    logger.warning("Failed %s: %s", src, e)
                    failed_tasks.append(next(task for task in tasks if task[0] == src))
                    continue

                if not result.get("written", False):
                    logger.warning("Failed %s in worker(pid=%s): %s", src, result.get("pid", "?"), result.get("error", "unknown error"))
                    failed_tasks.append(next(task for task in tasks if task[0] == src))
                    continue

                totals["files"] += 1
                totals["actions"] += int(result["actions"])
                totals["rounds"] += int(result["rounds"])
                totals["nonzero_before"] += int(result["nonzero_before"])
                totals["nonzero_after"] += int(result["nonzero_after"])
                totals["zeroed_actions"] += int(result["zeroed_actions"])
                totals["invalid_json_lines"] += int(result["invalid_json_lines"])

                if idx % 50 == 0 or idx == len(tasks):
                    logger.info(
                        "  %s/%s files done | actions=%s rounds=%s zeroed=%s invalid=%s failed=%s",
                        idx,
                        len(tasks),
                        f"{totals['actions']:,}",
                        f"{totals['rounds']:,}",
                        f"{totals['zeroed_actions']:,}",
                        f"{totals['invalid_json_lines']:,}",
                        len(failed_tasks),
                    )
    except Exception as e:
        logger.warning("Process pool aborted: %s", e)
        remaining_sources = {task[0]: task for task in tasks}
        for task in failed_tasks:
            remaining_sources.pop(task[0], None)
        failed_tasks.extend(remaining_sources.values())

    if failed_tasks and args.fallback_sequential:
        logger.info("Retrying %s failed files sequentially...", len(failed_tasks))
        dedup_failed = []
        seen = set()
        for task in failed_tasks:
            if task[0] in seen:
                continue
            seen.add(task[0])
            dedup_failed.append(task)
        for idx, task in enumerate(dedup_failed, start=1):
            src = task[0]
            try:
                result = _process_file_impl(*task)
            except Exception as e:
                logger.warning("Sequential retry failed %s: %s", src, e)
                continue
            totals["files"] += int(bool(result.get("written", False)))
            totals["actions"] += int(result["actions"])
            totals["rounds"] += int(result["rounds"])
            totals["nonzero_before"] += int(result["nonzero_before"])
            totals["nonzero_after"] += int(result["nonzero_after"])
            totals["zeroed_actions"] += int(result["zeroed_actions"])
            totals["invalid_json_lines"] += int(result["invalid_json_lines"])
            if idx % 20 == 0 or idx == len(dedup_failed):
                logger.info("  sequential retry %s/%s done", idx, len(dedup_failed))

    logger.info("Conversion finished.")
    logger.info("  files converted        : %s", f"{totals['files']:,}")
    logger.info("  actions seen           : %s", f"{totals['actions']:,}")
    logger.info("  rounds seen            : %s", f"{totals['rounds']:,}")
    logger.info("  nonzero rewards before : %s", f"{totals['nonzero_before']:,}")
    logger.info("  nonzero rewards after  : %s", f"{totals['nonzero_after']:,}")
    logger.info("  zeroed actions         : %s", f"{totals['zeroed_actions']:,}")
    logger.info("  invalid json lines     : %s", f"{totals['invalid_json_lines']:,}")


if __name__ == "__main__":
    main()
