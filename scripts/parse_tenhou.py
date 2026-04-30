"""
parse_tenhou.py — 天凤牌谱 → .jsonl 标注文件（Annotation 格式）

支持两种输入格式：
  A) fetch_tenhou.py 输出的 mjai_json/*.jsonl
     每行: {"log_id": ..., "events": [...], "players": [...]}
  B) 传统 mjai JSON 文件（每个文件一局，JSON array of events）

使用方法：
  # 处理 fetch_tenhou.py 的输出（推荐）
  python scripts/parse_tenhou.py \\
      --input  data/raw/mjai_json/  \\
      --output data/annotated/      \\
      --workers 4

  # 处理传统 mjai JSON 文件
  python scripts/parse_tenhou.py \\
      --input  data/raw/old_json/   \\
      --output data/annotated/

输出格式：每行一个 Annotation JSON，文件名与输入 .jsonl 对应。
GRP 奖励由 fill_grp_rewards.py 在 GRP 训练后回填。
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Optional

import orjson

from rinshan.tile import Tile
from rinshan.engine.simulator import MjaiSimulator
from rinshan.data.annotation  import Annotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Tile lookup table（比逐个调用 to_mjai() 快 ~4x）
# ─────────────────────────────────────────────

_TILE_STR: dict[int, str] = {Tile(i).tile_id: Tile(i).to_mjai() for i in range(34)}
_TILE_STR[Tile(4).akaize().tile_id]  = "0m"
_TILE_STR[Tile(13).akaize().tile_id] = "0p"
_TILE_STR[Tile(22).akaize().tile_id] = "0s"


def _tile_to_str(t: Tile) -> str:
    return _TILE_STR[t.tile_id]

def _tiles_to_strs(ts: list[Tile]) -> list[str]:
    return [_TILE_STR[t.tile_id] for t in ts]

def annotation_to_dict(ann: Annotation) -> dict:
    """将 Annotation 转为可 JSON 序列化的 dict"""
    return {
        "game_id":         ann.game_id,
        "player_id":       ann.player_id,
        "round_wind":      ann.round_wind,
        "round_num":       ann.round_num,
        "honba":           ann.honba,
        "kyotaku":         ann.kyotaku,
        "scores":          ann.scores,
        "tiles_left":      ann.tiles_left,
        "hand":            _tiles_to_strs(ann.hand),
        "dora_indicators": _tiles_to_strs(ann.dora_indicators),
        "discards":        [_tiles_to_strs(d) for d in ann.discards],
        "melds": [
            [{"type": m[0], "tiles": _tiles_to_strs(m[1])} for m in seat_melds]
            for seat_melds in ann.melds
        ],
        "riichi_declared":  ann.riichi_declared,
        "progression":      ann.progression,
        "action_candidates":ann.action_candidates,
        "action_chosen":    ann.action_chosen,
        "round_delta_score":ann.round_delta_score,
        "final_delta_score":ann.final_delta_score,
        "final_rank":       ann.final_rank,
        "grp_reward":       ann.grp_reward,
        "hand_reward":      ann.hand_reward,
        "is_done":          ann.is_done,
        "aux": {
            "shanten":      ann.aux.shanten,
            "tenpai_prob":  ann.aux.tenpai_prob,
            "deal_in_risk": ann.aux.deal_in_risk,
            "opp_tenpai":   ann.aux.opp_tenpai,
        } if ann.aux else None,
        "opponent_hands": [
            [_tile_to_str(t) for t in hand]
            for hand in ann.opponent_hands
        ] if ann.opponent_hands is not None else None,
    }


# ─────────────────────────────────────────────
# 单文件处理
# ─────────────────────────────────────────────

def _process_events(events: list, game_id: str, sim: MjaiSimulator) -> list:
    """共用的事件流 → Annotation 列表逻辑"""
    annotations = sim.parse_game(events, game_id=game_id)
    if annotations:
        annotations[-1].is_done = True
    return annotations


def process_file(args: tuple[Path, Path]) -> tuple[int, int]:
    """
    处理单个牌谱文件（支持两种输入格式）
    Returns: (n_annotations, n_errors)
    """
    in_path, out_path = args
    sim = MjaiSimulator()
    n_ann = 0
    n_err = 0

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ── 格式 A：fetch_tenhou 输出的 .jsonl（每行一局，含 events 字段）──
        if in_path.suffix == ".jsonl":
            with open(in_path, "r", encoding="utf-8") as fin, \
                 open(out_path, "wb") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        game_id = record.get("log_id", in_path.stem)
                        events  = record.get("events", [])
                        annotations = _process_events(events, game_id, sim)
                        for ann in annotations:
                            fout.write(orjson.dumps(annotation_to_dict(ann)))
                            fout.write(b"\n")
                            n_ann += 1
                    except Exception as e:
                        logger.warning(f"Skip bad line in {in_path}: {e}")
                        n_err += 1
            return n_ann, n_err

        # ── 格式 B：传统 mjai JSON（.json / .json.gz，单局）──────────────────
        if in_path.suffix == ".gz":
            with gzip.open(in_path, "rt", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            with open(in_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

        game_id = in_path.stem.replace(".json", "")
        if isinstance(raw, dict):
            events = raw.get("log", raw.get("events", []))
        elif isinstance(raw, list):
            events = raw
        else:
            logger.warning(f"Unknown format in {in_path}")
            return 0, 1

        annotations = _process_events(events, game_id, sim)
        with open(out_path, "wb") as f:
            for ann in annotations:
                f.write(orjson.dumps(annotation_to_dict(ann)))
                f.write(b"\n")
                n_ann += 1

    except Exception as e:
        logger.error(f"Error processing {in_path}: {e}")
        n_err += 1

    return n_ann, n_err


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse Tenhou mjai logs to annotations")
    parser.add_argument("--input",   "-i", required=True,  help="Input directory with .json.gz files")
    parser.add_argument("--output",  "-o", required=True,  help="Output directory for .jsonl files")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit",   "-n", type=int, default=None, help="Max files to process")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有输入文件（支持 .jsonl / .json.gz / .json）
    in_files = sorted(
        list(in_dir.rglob("*.jsonl")) +
        list(in_dir.rglob("*.json.gz")) +
        list(in_dir.rglob("*.json"))
    )
    if args.limit:
        in_files = in_files[:args.limit]

    logger.info(f"Found {len(in_files)} files in {in_dir}")

    # 构造输出路径列表
    task_args = []
    for in_path in in_files:
        rel = in_path.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix("").with_suffix(".jsonl")
        if out_path.exists():
            continue  # 跳过已处理的
        task_args.append((in_path, out_path))

    logger.info(f"Processing {len(task_args)} new files...")

    if not task_args:
        logger.info("Nothing to do.")
        return

    # 多进程处理
    total_ann = 0
    total_err = 0
    with mp.Pool(args.workers) as pool:
        for i, (n_ann, n_err) in enumerate(
            pool.imap_unordered(process_file, task_args, chunksize=10)
        ):
            total_ann += n_ann
            total_err += n_err
            if (i + 1) % 100 == 0:
                logger.info(
                    f"  {i+1}/{len(task_args)} files — "
                    f"{total_ann:,} annotations, {total_err} errors"
                )

    logger.info(
        f"Done. Total: {total_ann:,} annotations, {total_err} errors"
    )
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
