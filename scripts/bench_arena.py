"""
bench_arena.py — Rinshan 自对弈引擎吞吐量基准测试

测量指标：
  - 纯引擎速度：RandomAgent × 4，排除 AI 推理开销
  - 批量并发效率：不同 n_games 并发数下的 games/s
  - 每局平均决策次数（用于估算 AI 接入后的推理开销）

Usage:
    python scripts/bench_arena.py [--games 64] [--length hanchan|tonpuu]

说明：
    RandomAgent 代表引擎本身的上限速度。
    接入 RinshanAgent 后速度由 GPU 推理批量化效率决定，
    可通过 --games 调大并发数来均摊 GPU 推理开销。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.self_play.agent import RandomAgent
from rinshan.self_play.arena import Arena


def run_bench(n_games: int, game_length: str, n_rounds: int = 3) -> None:
    agents = [RandomAgent(name=f"rand_{i}", seed=i) for i in range(4)]

    print(f"\n{'='*55}")
    print(f"  Rinshan Arena Benchmark")
    print(f"  game_length={game_length}  n_games={n_games}  rounds={n_rounds}")
    print(f"{'='*55}")

    total_decisions = 0
    total_games_done = 0
    best_gps = 0.0

    for r in range(n_rounds):
        arena = Arena(
            agents=agents,
            n_games=n_games,
            game_length=game_length,
            base_seed=r * 10000,
            agent_rotation="round_robin",
            show_progress=False,
        )
        t0 = time.perf_counter()
        records = arena.run()
        elapsed = time.perf_counter() - t0

        gps = len(records) / elapsed

        # 统计总决策次数（每局的 kyoku_log 事件数近似决策密度）
        decisions = sum(
            sum(len(ev_list) for ev_list in rec.kyoku_logs)
            for rec in records
        )
        dps = decisions / elapsed  # decisions per second

        total_decisions += decisions
        total_games_done += len(records)
        best_gps = max(best_gps, gps)

        avg_decisions_per_game = decisions / max(len(records), 1)

        print(
            f"  Round {r+1}: {len(records)} games in {elapsed:.2f}s  "
            f"| {gps:.1f} games/s  "
            f"| {dps:.0f} events/s  "
            f"| avg {avg_decisions_per_game:.0f} events/game"
        )

    print(f"\n  Peak: {best_gps:.1f} games/s")
    print(f"  Avg decisions/game: {total_decisions / max(total_games_done, 1):.0f}")

    # 估算接入 AI 后的瓶颈
    avg_decs = total_decisions / max(total_games_done, 1)
    print(f"\n  ── AI 接入估算（仅供参考）──")
    for gpu_fps in [500, 1000, 2000, 5000]:
        # AI 需要处理的 token forward 次数（每个决策一次）
        # 每局约 avg_decs/4 次轮到自己决策
        ai_decisions_per_game = avg_decs / 4
        # games/s 受限于 GPU 吞吐
        bottleneck_gps = gpu_fps / ai_decisions_per_game
        print(
            f"    GPU @ {gpu_fps:5d} decisions/s → "
            f"估算 {bottleneck_gps:.1f} games/s "
            f"(需并发 {max(1, int(bottleneck_gps / best_gps * n_games))} 局均摊推理)"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rinshan Arena 引擎吞吐量 benchmark")
    parser.add_argument("--games",  type=int, default=64,
                        help="并发局数（默认 64）")
    parser.add_argument("--length", choices=["hanchan", "tonpuu"],
                        default="hanchan", help="局长（默认 hanchan）")
    parser.add_argument("--rounds", type=int, default=3,
                        help="重复测量轮数（默认 3，取最高值）")
    args = parser.parse_args()

    run_bench(args.games, args.length, args.rounds)

    # 额外跑不同并发数的扩展性曲线
    print("  ── 并发数扩展性（fixed rounds=1）──")
    print(f"  {'n_games':>8}  {'games/s':>10}  {'speedup':>8}")
    base_gps = None
    for n in [1, 4, 16, 32, 64, 128]:
        arena = Arena(
            agents=[RandomAgent(name=f"r{i}", seed=i) for i in range(4)],
            n_games=n,
            game_length=args.length,
            base_seed=99999,
            agent_rotation="round_robin",
            show_progress=False,
        )
        t0 = time.perf_counter()
        records = arena.run()
        elapsed = time.perf_counter() - t0
        gps = len(records) / elapsed
        if base_gps is None:
            base_gps = gps
        speedup = gps / base_gps
        print(f"  {n:>8}  {gps:>10.1f}  {speedup:>7.2f}x")
    print()


if __name__ == "__main__":
    main()
