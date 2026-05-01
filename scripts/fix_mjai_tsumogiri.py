"""
fix_mjai_tsumogiri.py — 原地修正 mjai_json 里所有 dahai.tsumogiri 字段

旧 parse_tenhou.py 使用 riichi_who 代理推断 tsumogiri，导致
只有立直后的打牌才被标为 tsumogiri，其余一律 False。

正确逻辑：dahai.pai == 该玩家上一次 tsumo.pai → tsumogiri=True，否则 False。
mjai_json 里 tsumo 事件含完整 pai 字段，可以直接重算。

用法：
    python scripts/fix_mjai_tsumogiri.py --data /path/to/mjai_json
    python scripts/fix_mjai_tsumogiri.py --data /path/to/mjai_json --dry_run
    python scripts/fix_mjai_tsumogiri.py --data /path/to/mjai_json --workers 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def fix_tsumogiri_in_events(events: list) -> tuple[int, int]:
    """
    原地修正 events 里所有 dahai.tsumogiri。
    返回 (fixed_count, total_dahai)。
    """
    last_draw: dict = {}
    fixed = 0
    total_dahai = 0

    for e in events:
        if not isinstance(e, dict):
            continue
        t = e.get("type")

        if t == "start_kyoku":
            last_draw = {}

        elif t == "tsumo":
            last_draw[e.get("actor", 0)] = e.get("pai")

        elif t in ("chi", "pon", "daiminkan", "ankan", "kakan"):
            last_draw[e.get("actor", 0)] = None

        elif t == "dahai":
            total_dahai += 1
            seat = e.get("actor", 0)
            pai  = e.get("pai")
            correct = (last_draw.get(seat) == pai)
            if e.get("tsumogiri") != correct:
                e["tsumogiri"] = correct
                fixed += 1
            last_draw[seat] = None

    return fixed, total_dahai


def fix_file(args: tuple) -> tuple[str, int, int]:
    """修正单个文件，返回 (filename, fixed_dahai, total_dahai)。供多进程调用。"""
    path, dry_run = args
    path = Path(path)

    games = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))

    total_fixed = total_dahai = 0
    for g in games:
        fx, td = fix_tsumogiri_in_events(g.get("events", []))
        total_fixed += fx
        total_dahai += td

    if not dry_run and total_fixed > 0:
        with open(path, "w", encoding="utf-8") as f:
            for g in games:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

    return path.name, total_fixed, total_dahai


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/raw/mjai_json",
                   help="mjai_json 目录路径")
    p.add_argument("--dry_run", action="store_true",
                   help="只统计，不写文件")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="并行进程数，默认=CPU核数")
    args = p.parse_args()

    data_dir = Path(args.data)
    files = sorted(data_dir.rglob("*.jsonl"))
    log.info(f"Found {len(files)} files in {data_dir}  "
             f"(dry_run={args.dry_run}, workers={args.workers})")

    tasks = [(str(fp), args.dry_run) for fp in files]
    grand_fixed = grand_total = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fix_file, t): t for t in tasks}
        for fut in as_completed(futs):
            name, fx, td = fut.result()
            grand_fixed += fx
            grand_total += td
            if fx:
                log.info(f"  {name}: fixed {fx}/{td} ({100*fx/max(td,1):.1f}%)")

    log.info(f"Done. fixed {grand_fixed}/{grand_total} dahai "
             f"({100*grand_fixed/max(grand_total,1):.1f}%) across {len(files)} files")
    if args.dry_run:
        log.info("dry_run=True, no files written.")


if __name__ == "__main__":
    main()
