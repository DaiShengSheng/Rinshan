"""
fix_mjai_tsumogiri.py — 原地修正 data/raw/mjai_json 里所有 dahai.tsumogiri 字段

旧 parse_tenhou.py 使用 riichi_who 代理推断 tsumogiri，导致
只有立直后的打牌才被标为 tsumogiri，其余一律 False。

正确逻辑：dahai.pai == 该玩家上一次 tsumo.pai → tsumogiri=True，否则 False。
mjai_json 里 tsumo 事件含完整 pai 字段，可以直接重算。

用法：
    python scripts/fix_mjai_tsumogiri.py
    python scripts/fix_mjai_tsumogiri.py --data data/raw/mjai_json --dry_run
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def fix_tsumogiri_in_events(events: list) -> tuple[int, int]:
    """
    原地修正 events 里所有 dahai.tsumogiri。
    返回 (fixed_count, total_dahai)。
    """
    last_draw: dict[int, str | None] = {}
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
            # 鸣牌后出牌不算摸切
            last_draw[e.get("actor", 0)] = None

        elif t == "dahai":
            total_dahai += 1
            seat = e.get("actor", 0)
            pai  = e.get("pai")
            correct = (last_draw.get(seat) == pai)
            if e.get("tsumogiri") != correct:
                e["tsumogiri"] = correct
                fixed += 1
            last_draw[seat] = None  # 打牌后清空

    return fixed, total_dahai


def fix_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """修正单个文件，返回 (fixed_dahai, total_dahai)。"""
    games = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))

    total_fixed = total_dahai = 0
    for g in games:
        events = g.get("events", [])
        fx, td = fix_tsumogiri_in_events(events)
        total_fixed += fx
        total_dahai += td

    if not dry_run and total_fixed > 0:
        with open(path, "w", encoding="utf-8") as f:
            for g in games:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

    return total_fixed, total_dahai


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/raw/mjai_json")
    p.add_argument("--dry_run", action="store_true",
                   help="只统计，不写文件")
    args = p.parse_args()

    data_dir = Path(args.data)
    files = sorted(data_dir.rglob("*.jsonl"))
    log.info(f"Found {len(files)} files in {data_dir}  (dry_run={args.dry_run})")

    grand_fixed = grand_total = 0
    for fp in files:
        fx, td = fix_file(fp, args.dry_run)
        grand_fixed += fx
        grand_total += td
        if fx:
            log.info(f"  {fp.name}: fixed {fx}/{td} dahai ({100*fx/max(td,1):.1f}%)")

    log.info(f"Done. fixed {grand_fixed}/{grand_total} dahai "
             f"({100*grand_fixed/max(grand_total,1):.1f}%) across {len(files)} files")
    if args.dry_run:
        log.info("dry_run=True, no files written.")


if __name__ == "__main__":
    main()
