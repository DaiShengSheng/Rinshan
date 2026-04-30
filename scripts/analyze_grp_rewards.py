#!/usr/bin/env python3
"""Analyze grp_reward distribution in annotated jsonl files.

Goals:
1. Describe the global grp_reward distribution.
2. Verify whether reward is broadcast at round level
   (same reward repeated for all actions in one round).
3. Quantify potential noise / extreme outliers.
4. Measure whether long rounds get overweighted because the same reward is
   repeated many times.

Linux usage:
    python3 scripts/analyze_grp_rewards.py --data-dir data/annotated_grp
    python3 scripts/analyze_grp_rewards.py --data-dir /path/to/annotated_grp --top-k 20 --json-out out.json
    python3 scripts/analyze_grp_rewards.py --data-dir data/annotated_grp --sample-files 200 --seed 42 --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import random
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from heapq import heappush, heappushpop
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_grp_rewards")


@dataclass
class RoundAccumulator:
    """Per-(game_id, player_id, round) aggregate within one file."""

    count: int = 0
    reward_sum: float = 0.0
    reward_min: float = float("inf")
    reward_max: float = float("-inf")
    unique_rewards: set[float] | None = None

    def __post_init__(self) -> None:
        if self.unique_rewards is None:
            self.unique_rewards = set()

    def add(self, reward: float, rounding: int) -> None:
        self.count += 1
        self.reward_sum += reward
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)
        self.unique_rewards.add(round(reward, rounding))

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.count if self.count else 0.0

    @property
    def unique_count(self) -> int:
        return len(self.unique_rewards) if self.unique_rewards is not None else 0


@dataclass
class CorrAccumulator:
    """Streaming Pearson correlation accumulator."""

    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y

    def merge(self, other: "CorrAccumulator") -> None:
        self.n += other.n
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_x2 += other.sum_x2
        self.sum_y2 += other.sum_y2
        self.sum_xy += other.sum_xy

    def corr(self) -> float | None:
        if self.n < 2:
            return None
        num = self.n * self.sum_xy - self.sum_x * self.sum_y
        den_x = self.n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = self.n * self.sum_y2 - self.sum_y * self.sum_y
        if den_x <= 0 or den_y <= 0:
            return None
        return num / math.sqrt(den_x * den_y)


@dataclass
class Summary:
    files_scanned: int
    lines_scanned: int
    valid_rewards: int
    missing_rewards: int
    invalid_json_lines: int
    total_rounds: int
    single_reward_rounds: int
    multi_reward_rounds: int
    total_actions_in_single_reward_rounds: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_abs_reward: float
    p01: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    pos_ratio: float
    neg_ratio: float
    zero_ratio: float
    near_zero_ratio: float
    broadcast_round_ratio: float
    broadcast_action_ratio: float
    corr_round_len_vs_abs_reward: float | None
    corr_round_len_vs_abs_total_weight: float | None
    mean_round_len: float
    mean_round_len_single: float
    mean_round_len_multi: float
    longest_round_len: int
    longest_round_unique_rewards: int


class RewardAnalyzer:
    def __init__(self, top_k: int, zero_eps: float, reward_rounding: int) -> None:
        self.top_k = top_k
        self.zero_eps = zero_eps
        self.reward_rounding = reward_rounding

        self.files_scanned = 0
        self.lines_scanned = 0
        self.valid_rewards = 0
        self.missing_rewards = 0
        self.invalid_json_lines = 0

        self.rewards = array("d")
        self.sum_reward = 0.0
        self.sum_reward_sq = 0.0
        self.sum_abs_reward = 0.0
        self.pos_count = 0
        self.neg_count = 0
        self.zero_count = 0
        self.near_zero_count = 0

        self.total_rounds = 0
        self.single_reward_rounds = 0
        self.multi_reward_rounds = 0
        self.total_actions_in_single_reward_rounds = 0
        self.total_round_len = 0
        self.total_round_len_single = 0
        self.total_round_len_multi = 0
        self.longest_round_len = 0
        self.longest_round_unique_rewards = 0

        self.round_len_vs_abs_reward = CorrAccumulator()
        self.round_len_vs_abs_total_weight = CorrAccumulator()

        self.top_abs_actions: list[tuple[float, int, dict[str, Any]]] = []
        self.top_abs_rounds: list[tuple[float, int, dict[str, Any]]] = []
        self.top_abs_round_total_weight: list[tuple[float, int, dict[str, Any]]] = []
        self._top_seq = 0

    def _push_top(self, heap: list[tuple[float, int, dict[str, Any]]], score: float, payload: dict[str, Any]) -> None:
        self._top_seq += 1
        item = (score, self._top_seq, payload)
        if len(heap) < self.top_k:
            heappush(heap, item)
        else:
            if score > heap[0][0]:
                heappushpop(heap, item)

    def process_file(self, path: Path, max_lines_per_file: int | None = None) -> None:
        logger.info("Scanning %s", path)
        self.files_scanned += 1
        round_map: dict[tuple[Any, ...], RoundAccumulator] = {}

        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                if max_lines_per_file is not None and line_no > max_lines_per_file:
                    break
                self.lines_scanned += 1
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    self.invalid_json_lines += 1
                    continue

                reward_raw = d.get("grp_reward", None)
                if reward_raw is None:
                    self.missing_rewards += 1
                    continue
                try:
                    reward = float(reward_raw)
                except (TypeError, ValueError):
                    self.missing_rewards += 1
                    continue

                self.valid_rewards += 1
                self.rewards.append(reward)
                self.sum_reward += reward
                self.sum_reward_sq += reward * reward
                self.sum_abs_reward += abs(reward)

                if reward > 0:
                    self.pos_count += 1
                elif reward < 0:
                    self.neg_count += 1
                else:
                    self.zero_count += 1
                if abs(reward) <= self.zero_eps:
                    self.near_zero_count += 1

                action_payload = {
                    "file": str(path),
                    "line_no": line_no,
                    "game_id": d.get("game_id"),
                    "player_id": d.get("player_id"),
                    "round_wind": d.get("round_wind"),
                    "round_num": d.get("round_num"),
                    "reward": reward,
                    "action_chosen": d.get("action_chosen"),
                    "tiles_left": d.get("tiles_left"),
                }
                self._push_top(self.top_abs_actions, abs(reward), action_payload)

                round_key = (
                    d.get("game_id"),
                    d.get("player_id"),
                    d.get("round_wind"),
                    d.get("round_num"),
                )
                acc = round_map.get(round_key)
                if acc is None:
                    acc = RoundAccumulator()
                    round_map[round_key] = acc
                acc.add(reward, self.reward_rounding)

        for (game_id, player_id, round_wind, round_num), acc in round_map.items():
            mean_reward = acc.mean_reward
            abs_mean = abs(mean_reward)
            abs_total_weight = abs_mean * acc.count
            unique_count = acc.unique_count

            self.total_rounds += 1
            self.total_round_len += acc.count
            self.longest_round_len = max(self.longest_round_len, acc.count)
            if acc.count == self.longest_round_len:
                self.longest_round_unique_rewards = unique_count

            if unique_count == 1:
                self.single_reward_rounds += 1
                self.total_actions_in_single_reward_rounds += acc.count
                self.total_round_len_single += acc.count
            else:
                self.multi_reward_rounds += 1
                self.total_round_len_multi += acc.count

            self.round_len_vs_abs_reward.add(acc.count, abs_mean)
            self.round_len_vs_abs_total_weight.add(acc.count, abs_total_weight)

            round_payload = {
                "file": str(path),
                "game_id": game_id,
                "player_id": player_id,
                "round_wind": round_wind,
                "round_num": round_num,
                "n_actions": acc.count,
                "mean_reward": mean_reward,
                "reward_min": acc.reward_min,
                "reward_max": acc.reward_max,
                "unique_rewards": unique_count,
            }
            self._push_top(self.top_abs_rounds, abs_mean, round_payload)
            self._push_top(self.top_abs_round_total_weight, abs_total_weight, round_payload)

    def export_state(self) -> dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "lines_scanned": self.lines_scanned,
            "valid_rewards": self.valid_rewards,
            "missing_rewards": self.missing_rewards,
            "invalid_json_lines": self.invalid_json_lines,
            "rewards": list(self.rewards),
            "sum_reward": self.sum_reward,
            "sum_reward_sq": self.sum_reward_sq,
            "sum_abs_reward": self.sum_abs_reward,
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
            "zero_count": self.zero_count,
            "near_zero_count": self.near_zero_count,
            "total_rounds": self.total_rounds,
            "single_reward_rounds": self.single_reward_rounds,
            "multi_reward_rounds": self.multi_reward_rounds,
            "total_actions_in_single_reward_rounds": self.total_actions_in_single_reward_rounds,
            "total_round_len": self.total_round_len,
            "total_round_len_single": self.total_round_len_single,
            "total_round_len_multi": self.total_round_len_multi,
            "longest_round_len": self.longest_round_len,
            "longest_round_unique_rewards": self.longest_round_unique_rewards,
            "round_len_vs_abs_reward": asdict(self.round_len_vs_abs_reward),
            "round_len_vs_abs_total_weight": asdict(self.round_len_vs_abs_total_weight),
            "top_abs_actions": list(self.top_abs_actions),
            "top_abs_rounds": list(self.top_abs_rounds),
            "top_abs_round_total_weight": list(self.top_abs_round_total_weight),
        }

    def merge_exported_state(self, state: dict[str, Any]) -> None:
        self.files_scanned += int(state.get("files_scanned", 0))
        self.lines_scanned += int(state.get("lines_scanned", 0))
        self.valid_rewards += int(state.get("valid_rewards", 0))
        self.missing_rewards += int(state.get("missing_rewards", 0))
        self.invalid_json_lines += int(state.get("invalid_json_lines", 0))

        self.rewards.extend(state.get("rewards", []))
        self.sum_reward += float(state.get("sum_reward", 0.0))
        self.sum_reward_sq += float(state.get("sum_reward_sq", 0.0))
        self.sum_abs_reward += float(state.get("sum_abs_reward", 0.0))
        self.pos_count += int(state.get("pos_count", 0))
        self.neg_count += int(state.get("neg_count", 0))
        self.zero_count += int(state.get("zero_count", 0))
        self.near_zero_count += int(state.get("near_zero_count", 0))

        self.total_rounds += int(state.get("total_rounds", 0))
        self.single_reward_rounds += int(state.get("single_reward_rounds", 0))
        self.multi_reward_rounds += int(state.get("multi_reward_rounds", 0))
        self.total_actions_in_single_reward_rounds += int(state.get("total_actions_in_single_reward_rounds", 0))
        self.total_round_len += int(state.get("total_round_len", 0))
        self.total_round_len_single += int(state.get("total_round_len_single", 0))
        self.total_round_len_multi += int(state.get("total_round_len_multi", 0))

        other_longest = int(state.get("longest_round_len", 0))
        other_longest_unique = int(state.get("longest_round_unique_rewards", 0))
        if other_longest > self.longest_round_len:
            self.longest_round_len = other_longest
            self.longest_round_unique_rewards = other_longest_unique
        elif other_longest == self.longest_round_len and other_longest > 0:
            self.longest_round_unique_rewards = max(self.longest_round_unique_rewards, other_longest_unique)

        self.round_len_vs_abs_reward.merge(CorrAccumulator(**state.get("round_len_vs_abs_reward", {})))
        self.round_len_vs_abs_total_weight.merge(CorrAccumulator(**state.get("round_len_vs_abs_total_weight", {})))

        for score, _seq, payload in state.get("top_abs_actions", []):
            self._push_top(self.top_abs_actions, float(score), payload)
        for score, _seq, payload in state.get("top_abs_rounds", []):
            self._push_top(self.top_abs_rounds, float(score), payload)
        for score, _seq, payload in state.get("top_abs_round_total_weight", []):
            self._push_top(self.top_abs_round_total_weight, float(score), payload)

    @staticmethod
    def _percentile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        pos = (len(sorted_vals) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_vals[lo]
        frac = pos - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    def build_summary(self) -> Summary:
        if self.valid_rewards == 0:
            raise RuntimeError("No grp_reward values found. Please check --data-dir or whether rewards were filled.")

        rewards_sorted = sorted(self.rewards)
        mean_reward = self.sum_reward / self.valid_rewards
        var = max(self.sum_reward_sq / self.valid_rewards - mean_reward * mean_reward, 0.0)
        std_reward = math.sqrt(var)

        return Summary(
            files_scanned=self.files_scanned,
            lines_scanned=self.lines_scanned,
            valid_rewards=self.valid_rewards,
            missing_rewards=self.missing_rewards,
            invalid_json_lines=self.invalid_json_lines,
            total_rounds=self.total_rounds,
            single_reward_rounds=self.single_reward_rounds,
            multi_reward_rounds=self.multi_reward_rounds,
            total_actions_in_single_reward_rounds=self.total_actions_in_single_reward_rounds,
            mean_reward=mean_reward,
            std_reward=std_reward,
            min_reward=rewards_sorted[0],
            max_reward=rewards_sorted[-1],
            mean_abs_reward=self.sum_abs_reward / self.valid_rewards,
            p01=self._percentile(rewards_sorted, 0.01),
            p05=self._percentile(rewards_sorted, 0.05),
            p25=self._percentile(rewards_sorted, 0.25),
            p50=self._percentile(rewards_sorted, 0.50),
            p75=self._percentile(rewards_sorted, 0.75),
            p95=self._percentile(rewards_sorted, 0.95),
            p99=self._percentile(rewards_sorted, 0.99),
            pos_ratio=self.pos_count / self.valid_rewards,
            neg_ratio=self.neg_count / self.valid_rewards,
            zero_ratio=self.zero_count / self.valid_rewards,
            near_zero_ratio=self.near_zero_count / self.valid_rewards,
            broadcast_round_ratio=self.single_reward_rounds / self.total_rounds if self.total_rounds else 0.0,
            broadcast_action_ratio=(
                self.total_actions_in_single_reward_rounds / self.valid_rewards if self.valid_rewards else 0.0
            ),
            corr_round_len_vs_abs_reward=self.round_len_vs_abs_reward.corr(),
            corr_round_len_vs_abs_total_weight=self.round_len_vs_abs_total_weight.corr(),
            mean_round_len=self.total_round_len / self.total_rounds if self.total_rounds else 0.0,
            mean_round_len_single=(
                self.total_round_len_single / self.single_reward_rounds if self.single_reward_rounds else 0.0
            ),
            mean_round_len_multi=(
                self.total_round_len_multi / self.multi_reward_rounds if self.multi_reward_rounds else 0.0
            ),
            longest_round_len=self.longest_round_len,
            longest_round_unique_rewards=self.longest_round_unique_rewards,
        )

    def report(self) -> dict[str, Any]:
        summary = self.build_summary()
        result = {
            "summary": asdict(summary),
            "top_abs_actions": [payload for _, _, payload in sorted(self.top_abs_actions, key=lambda x: x[0], reverse=True)],
            "top_abs_rounds": [payload for _, _, payload in sorted(self.top_abs_rounds, key=lambda x: x[0], reverse=True)],
            "top_abs_round_total_weight": [payload for _, _, payload in sorted(self.top_abs_round_total_weight, key=lambda x: x[0], reverse=True)],
            "diagnosis": {
                "round_level_broadcast_suspected": summary.broadcast_round_ratio > 0.95,
                "extreme_noise_suspected": abs(summary.p99) > 5 * max(abs(summary.p50), self.zero_eps),
                "long_round_overweight_suspected": (
                    summary.corr_round_len_vs_abs_total_weight is not None
                    and summary.corr_round_len_vs_abs_total_weight > 0.5
                ),
            },
        }
        return result


def _analyze_file_worker(args_tuple: tuple[str, int, float, int, int | None]) -> dict[str, Any]:
    path_str, top_k, zero_eps, reward_rounding, max_lines_per_file = args_tuple
    analyzer = RewardAnalyzer(top_k=top_k, zero_eps=zero_eps, reward_rounding=reward_rounding)
    analyzer.process_file(Path(path_str), max_lines_per_file=max_lines_per_file)
    return analyzer.export_state()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze grp_reward distribution and round-level broadcast artifacts.")
    parser.add_argument("--data-dir", type=str, default="data/annotated_grp", help="Directory containing annotated .jsonl files with grp_reward")
    parser.add_argument("--glob", type=str, default="*.jsonl", help="Glob pattern under data-dir (default: *.jsonl)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of extreme samples/rounds to print")
    parser.add_argument("--zero-eps", type=float, default=1e-8, help="Threshold for near-zero reward")
    parser.add_argument("--reward-rounding", type=int, default=12, help="Rounding digits when counting unique rewards inside a round")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker processes. 0=auto, 1=disable parallelism")
    parser.add_argument("--max-files", type=int, default=0, help="Take only the first N matched files after optional sampling/shuffle. 0=all")
    parser.add_argument("--sample-files", type=int, default=0, help="Randomly sample N files from all matched files before analysis. 0=all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by --sample-files")
    parser.add_argument("--max-lines-per-file", type=int, default=0, help="Read at most N lines from each file. 0=all")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save full JSON report")
    return parser.parse_args()


def print_human_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    d = report["diagnosis"]

    print("=" * 80)
    print("GRP Reward Distribution Report")
    print("=" * 80)
    print(f"Files scanned                 : {s['files_scanned']}")
    print(f"Lines scanned                 : {s['lines_scanned']}")
    print(f"Valid grp_reward rows         : {s['valid_rewards']}")
    print(f"Missing grp_reward rows       : {s['missing_rewards']}")
    print(f"Invalid JSON lines            : {s['invalid_json_lines']}")
    print()

    print("[Global reward distribution]")
    print(f"mean / std                    : {s['mean_reward']:.6f} / {s['std_reward']:.6f}")
    print(f"min / p01 / p05               : {s['min_reward']:.6f} / {s['p01']:.6f} / {s['p05']:.6f}")
    print(f"p25 / p50 / p75              : {s['p25']:.6f} / {s['p50']:.6f} / {s['p75']:.6f}")
    print(f"p95 / p99 / max              : {s['p95']:.6f} / {s['p99']:.6f} / {s['max_reward']:.6f}")
    print(f"mean(|reward|)                : {s['mean_abs_reward']:.6f}")
    print(f"positive / negative / zero    : {s['pos_ratio']:.2%} / {s['neg_ratio']:.2%} / {s['zero_ratio']:.2%}")
    print(f"near-zero (|r|<=eps)          : {s['near_zero_ratio']:.2%}")
    print()

    print("[Round-level broadcast check]")
    print(f"total rounds                  : {s['total_rounds']}")
    print(f"single-reward rounds          : {s['single_reward_rounds']} ({s['broadcast_round_ratio']:.2%})")
    print(f"multi-reward rounds           : {s['multi_reward_rounds']}")
    print(f"actions in single-reward rounds: {s['total_actions_in_single_reward_rounds']} ({s['broadcast_action_ratio']:.2%})")
    print(f"mean round length             : {s['mean_round_len']:.2f}")
    print(f"mean len (single-reward)      : {s['mean_round_len_single']:.2f}")
    print(f"mean len (multi-reward)       : {s['mean_round_len_multi']:.2f}")
    print(f"longest round length          : {s['longest_round_len']} (unique rewards={s['longest_round_unique_rewards']})")
    print()

    print("[Correlation diagnostics]")
    corr1 = s['corr_round_len_vs_abs_reward']
    corr2 = s['corr_round_len_vs_abs_total_weight']
    print(f"corr(round_len, |mean_reward|)      : {corr1:.4f}" if corr1 is not None else "corr(round_len, |mean_reward|)      : N/A")
    print(f"corr(round_len, |mean_reward|*len)  : {corr2:.4f}" if corr2 is not None else "corr(round_len, |mean_reward|*len)  : N/A")
    print()

    print("[Diagnosis]")
    print(f"round-level broadcast suspected : {d['round_level_broadcast_suspected']}")
    print(f"extreme noise suspected         : {d['extreme_noise_suspected']}")
    print(f"long-round overweight suspected : {d['long_round_overweight_suspected']}")
    print()

    print("[Top abs(action reward)]")
    for row in report["top_abs_actions"]:
        print(json.dumps(row, ensure_ascii=False))
    print()

    print("[Top abs(round mean reward)]")
    for row in report["top_abs_rounds"]:
        print(json.dumps(row, ensure_ascii=False))
    print()

    print("[Top abs(round total training weight = |mean_reward| * n_actions)]")
    for row in report["top_abs_round_total_weight"]:
        print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.rglob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob!r} under {data_dir}")

    total_matched = len(files)
    if args.sample_files and args.sample_files > 0 and args.sample_files < len(files):
        rng = random.Random(args.seed)
        files = sorted(rng.sample(files, args.sample_files))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError("No files left after applying sampling/limit arguments")

    analyzer = RewardAnalyzer(top_k=args.top_k, zero_eps=args.zero_eps, reward_rounding=args.reward_rounding)

    max_lines_per_file = args.max_lines_per_file if args.max_lines_per_file > 0 else None
    cpu_count = mp.cpu_count() or 1
    workers = args.workers if args.workers > 0 else min(cpu_count, len(files))
    workers = max(1, min(workers, len(files)))

    logger.info("Matched %d files under %s", total_matched, data_dir)
    logger.info("Selected %d files after sampling/limits", len(files))
    if args.sample_files and args.sample_files > 0:
        logger.info("Sampling enabled: sample_files=%d seed=%d", args.sample_files, args.seed)
    if args.max_files and args.max_files > 0:
        logger.info("Max files enabled: max_files=%d", args.max_files)
    if max_lines_per_file is not None:
        logger.info("Per-file line cap enabled: max_lines_per_file=%d", max_lines_per_file)
    logger.info("Using %d worker process(es)", workers)

    if workers == 1:
        for idx, path in enumerate(files, start=1):
            logger.info("[%d/%d] Scanning %s", idx, len(files), path)
            analyzer.process_file(path, max_lines_per_file=max_lines_per_file)
    else:
        ctx = mp.get_context("spawn")
        job_args = [(str(path), args.top_k, args.zero_eps, args.reward_rounding, max_lines_per_file) for path in files]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_analyze_file_worker, job): job[0] for job in job_args}
            done = 0
            for fut in as_completed(futures):
                path_str = futures[fut]
                done += 1
                state = fut.result()
                analyzer.merge_exported_state(state)
                logger.info("[%d/%d] Done %s", done, len(files), path_str)

    report = analyzer.report()
    print_human_report(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"[+] JSON report written to: {out_path}")


if __name__ == "__main__":
    main()
