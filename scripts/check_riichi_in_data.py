#!/usr/bin/env python3
"""
check_riichi_in_data.py — 验证标注数据中立直 token 的完整性

检查项目：
  1. action_candidates 里是否出现 RIICHI_TOKEN (497)
  2. 立直被选择的频率（action_chosen 指向 RIICHI_TOKEN 的比例）
  3. tenpai 样本（shanten==0）里立直候选的出现率
  4. PROG_RIICHI token (665-668) 在 progression 里的出现率（交叉验证）
  5. 立直样本的打印示例，供人工抽查

依赖：纯标准库，无需安装任何第三方包，可直接在服务器上运行。

用法（在项目根目录下执行）：
  # 检查原始标注数据（Stage 1/2 用）
  python scripts/check_riichi_in_data.py data/annotated

  # 检查 GRP 标注数据（Stage 3 用，最关键）
  python scripts/check_riichi_in_data.py data/annotated_grp

  # 随机采样 50 个文件
  python scripts/check_riichi_in_data.py data/annotated --files 50 --sample

  # 扫描全部文件（较慢）
  python scripts/check_riichi_in_data.py data/annotated --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# ── 常量（与 rinshan/constants.py 保持一致，避免 import 依赖）──────────────
RIICHI_TOKEN      = 497
TSUMO_AGARI_TOKEN = 498
RON_AGARI_TOKEN   = 499
RYUKYOKU_TOKEN    = 500
PASS_TOKEN        = 501

DISCARD_OFFSET    = 37
DISCARD_END       = 37 + 37   # 37 种打牌 token (34普通+3赤)

PROG_RIICHI_BASE  = 665
PROG_RIICHI_END   = 665 + 4   # 4 seats

# tenpai 时立直候选"正常出现率"下限（振听/副露/已立直 等情况会排除，故不是100%）
TENPAI_RIICHI_CAND_WARN_THRESHOLD = 60.0


# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(path: Path, stats: dict, riichi_examples: list) -> None:
    """逐行扫描一个 .jsonl 文件，累积统计信息。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                continue

            cands      = d.get("action_candidates", [])
            chosen_idx = d.get("action_chosen", 0)
            progression = d.get("progression", [])
            aux        = d.get("aux") or {}
            shanten    = aux.get("shanten", 99)

            stats["total"] += 1

            # ── 候选集合分析 ──────────────────────────────────────
            cand_set = set(cands)

            has_riichi_cand  = RIICHI_TOKEN in cand_set
            has_discard_cand = any(DISCARD_OFFSET <= c < DISCARD_END for c in cands)
            has_tsumo_cand   = TSUMO_AGARI_TOKEN in cand_set
            has_ron_cand     = RON_AGARI_TOKEN in cand_set

            if has_riichi_cand:
                stats["has_riichi_cand"] += 1

            if not cands:
                stats["empty_cands"] += 1

            stats["cand_len_dist"][len(cands)] += 1

            # ── 实际选择的动作 ────────────────────────────────────
            chosen_token = cands[chosen_idx] if 0 <= chosen_idx < len(cands) else None

            if chosen_token == RIICHI_TOKEN:
                stats["riichi_chosen"] += 1
                # 收集示例（最多 5 条）
                if len(riichi_examples) < 5:
                    riichi_examples.append({
                        "game_id":    d.get("game_id", "?"),
                        "player_id":  d.get("player_id", "?"),
                        "round":      f"{d.get('round_wind','?')}-{d.get('round_num','?')}",
                        "tiles_left": d.get("tiles_left", "?"),
                        "shanten":    shanten,
                        "cands":      cands,
                        "chosen_idx": chosen_idx,
                        "hand":       d.get("hand", []),
                        "grp_reward": d.get("grp_reward", "N/A"),
                    })
            elif chosen_token is not None and DISCARD_OFFSET <= chosen_token < DISCARD_END:
                stats["discard_chosen"] += 1
            elif chosen_token == TSUMO_AGARI_TOKEN:
                stats["tsumo_chosen"] += 1
            elif chosen_token == RON_AGARI_TOKEN:
                stats["ron_chosen"] += 1
            elif chosen_token == PASS_TOKEN:
                stats["pass_chosen"] += 1
            else:
                stats["other_chosen"] += 1

            # ── 越界检测 ──────────────────────────────────────────
            if chosen_idx < 0 or chosen_idx >= len(cands):
                stats["oob_chosen"] += 1

            # ── tenpai 样本里的立直候选 ───────────────────────────
            if shanten == 0:
                stats["tenpai_samples"] += 1
                if has_riichi_cand:
                    stats["tenpai_has_riichi_cand"] += 1

            # ── progression 里的 PROG_RIICHI ─────────────────────
            prog_has_riichi = any(PROG_RIICHI_BASE <= t < PROG_RIICHI_END for t in progression)
            if prog_has_riichi:
                stats["has_prog_riichi_in_prog"] += 1

            # ── grp_reward 分布（Stage 3 专项）────────────────────
            grp = d.get("grp_reward")
            if grp is not None:
                stats["has_grp_reward"] += 1
                if abs(float(grp)) < 1e-9:
                    stats["grp_reward_zero"] += 1


def print_report(stats: dict, n_files: int, riichi_examples: list) -> None:
    total = stats["total"]
    SEP  = "─" * 64
    SEP2 = "═" * 64

    print(f"\n{SEP2}")
    print(f"  立直数据完整性诊断报告")
    print(SEP2)
    print(f"  扫描文件数 : {n_files}")
    print(f"  总样本数   : {total:,}")
    if stats["parse_errors"]:
        print(f"  解析错误   : {stats['parse_errors']}  ⚠️")

    if total == 0:
        print("\n  ❌  未找到任何样本！请检查数据目录路径。")
        return

    def pct(n: int) -> str:
        return f"{n / total * 100:.2f}%"

    # ── 核心检查 1：RIICHI_TOKEN 是否出现在候选中 ──────────────────
    print(f"\n{SEP}")
    print(f"  [核心] RIICHI_TOKEN (497) 在 action_candidates 中")
    print(SEP)
    riichi_cand_pct  = stats["has_riichi_cand"] / total * 100
    riichi_chosen_pct = stats["riichi_chosen"] / total * 100
    print(f"  含立直候选的样本 : {stats['has_riichi_cand']:>9,}  ({riichi_cand_pct:.2f}%)")
    print(f"  实际选择了立直   : {stats['riichi_chosen']:>9,}  ({riichi_chosen_pct:.2f}%)")

    if stats["has_riichi_cand"] == 0:
        print()
        print("  ❌  RIICHI_TOKEN 从未出现在候选动作中！")
        print("  →  数据生成管道存在严重问题，Stage 3 无法学到立直。")
        print("  →  检查 MjaiSimulator._build_candidate_tokens / _compute_discard_candidates")
    elif stats["riichi_chosen"] == 0:
        print()
        print("  ⚠️   立直候选存在，但 action_chosen 从未指向 RIICHI_TOKEN。")
        print("  →  可能原因：action_chosen 编码存在 off-by-one，")
        print("     或人类选择立直时 token 写的是打牌而非 RIICHI。")
        print("  →  检查 MjaiSimulator._handle_dahai 中 RIICHI 判断逻辑。")
    else:
        print()
        print("  ✅  立直候选和立直选择均正常存在。")

    # ── 核心检查 2：tenpai 时的立直候选出现率 ────────────────────────
    print(f"\n{SEP}")
    print(f"  [听牌分析] shanten == 0 时立直候选出现率")
    print(SEP)
    tenpai = stats["tenpai_samples"]
    if tenpai == 0:
        print("  ⚠️   未找到 shanten==0 样本（aux 字段可能缺失，或 shanten 字段未写入）。")
    else:
        t_riichi_pct = stats["tenpai_has_riichi_cand"] / tenpai * 100
        print(f"  听牌样本数           : {tenpai:>9,}  ({tenpai/total*100:.1f}% of total)")
        print(f"  听牌时含立直候选     : {stats['tenpai_has_riichi_cand']:>9,}  ({t_riichi_pct:.1f}%)")
        print()
        if t_riichi_pct < TENPAI_RIICHI_CAND_WARN_THRESHOLD:
            print(f"  ⚠️   听牌时立直候选出现率偏低（{t_riichi_pct:.1f}% < {TENPAI_RIICHI_CAND_WARN_THRESHOLD}%）。")
            print(f"  →  正常偏低原因：振听 / 副露副露 / 已立直 / 副露后无法立直等。")
            print(f"  →  若 < 30% 则需排查 simulator.can_riichi 判断逻辑。")
        else:
            print(f"  ✅  听牌时立直候选出现率正常（{t_riichi_pct:.1f}%）。")

    # ── 交叉验证：progression 里的立直事件 ──────────────────────────
    print(f"\n{SEP}")
    print(f"  [交叉验证] PROG_RIICHI (665-668) 在 progression 中")
    print(SEP)
    prog_riichi_pct = stats["has_prog_riichi_in_prog"] / total * 100
    print(f"  progression 含立直事件 : {stats['has_prog_riichi_in_prog']:>9,}  ({prog_riichi_pct:.2f}%)")
    if prog_riichi_pct < 0.5:
        print("  ⚠️   progression 里几乎没有立直事件 token。数据异常或解析管道存在问题。")
    else:
        print("  ✅  立直事件在进行序列中正常出现。")

    # ── 动作选择分布 ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  [动作分布] action_chosen 各类型占比")
    print(SEP)
    print(f"  打牌 (DISCARD)   : {stats['discard_chosen']:>9,}  ({pct(stats['discard_chosen'])})")
    print(f"  立直 (RIICHI)    : {stats['riichi_chosen']:>9,}  ({pct(stats['riichi_chosen'])})")
    print(f"  自摸 (TSUMO)     : {stats['tsumo_chosen']:>9,}  ({pct(stats['tsumo_chosen'])})")
    print(f"  荣和 (RON)       : {stats['ron_chosen']:>9,}  ({pct(stats['ron_chosen'])})")
    print(f"  PASS             : {stats['pass_chosen']:>9,}  ({pct(stats['pass_chosen'])})")
    if stats["other_chosen"]:
        print(f"  其他/未知        : {stats['other_chosen']:>9,}  ({pct(stats['other_chosen'])})")
    if stats["oob_chosen"]:
        print(f"  ⚠️  越界 chosen_idx : {stats['oob_chosen']:>9,}")

    # ── GRP reward 分布（Stage 3 专项）─────────────────────────────
    if stats["has_grp_reward"] > 0:
        print(f"\n{SEP}")
        print(f"  [Stage3 专项] grp_reward 字段分布")
        print(SEP)
        grp_zero_pct = stats["grp_reward_zero"] / stats["has_grp_reward"] * 100
        grp_nonzero  = stats["has_grp_reward"] - stats["grp_reward_zero"]
        print(f"  含 grp_reward 的样本 : {stats['has_grp_reward']:>9,}")
        print(f"  grp_reward == 0      : {stats['grp_reward_zero']:>9,}  ({grp_zero_pct:.1f}%)")
        print(f"  grp_reward != 0      : {grp_nonzero:>9,}  ({100-grp_zero_pct:.1f}%)")
        if grp_zero_pct > 99:
            print("  ⚠️   几乎所有 grp_reward 都是 0！")
            print("  →  可能未运行 fill_grp_rewards.py，或 GRP 模型尚未训练。")
        elif grp_zero_pct > 90:
            print("  ✅  grp_reward 分布符合 GRP 2.0 设计（局内大量为 0，仅跨局结算）。")
        else:
            print("  ✅  grp_reward 已正常填充。")
    else:
        print(f"\n{SEP}")
        print(f"  [Stage3 专项] grp_reward 字段")
        print(SEP)
        print("  ⚠️  数据中无 grp_reward 字段（此目录可能是原始 annotated 而非 annotated_grp）。")

    # ── 候选集大小分布 ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  [候选集大小分布]")
    print(SEP)
    for size, count in sorted(stats["cand_len_dist"].items()):
        bar = "█" * min(40, max(1, int(count / total * 400)))
        print(f"  cands={size:2d} : {count:>8,}  {bar}")

    # ── 立直样本示例 ────────────────────────────────────────────────
    if riichi_examples:
        print(f"\n{SEP}")
        print(f"  [立直样本抽查] 前 {len(riichi_examples)} 条 action_chosen==RIICHI 的样本")
        print(SEP)
        for i, ex in enumerate(riichi_examples, 1):
            riichi_pos = ex["cands"].index(RIICHI_TOKEN) if RIICHI_TOKEN in ex["cands"] else "N/A"
            print(f"\n  样本 #{i}")
            print(f"    game_id    : {ex['game_id']}")
            print(f"    player_id  : {ex['player_id']}  round: {ex['round']}  tiles_left: {ex['tiles_left']}")
            print(f"    shanten    : {ex['shanten']}  grp_reward: {ex['grp_reward']}")
            print(f"    hand       : {ex['hand']}")
            print(f"    candidates : {ex['cands']}")
            print(f"    chosen_idx : {ex['chosen_idx']}  → token={ex['cands'][ex['chosen_idx']] if 0 <= ex['chosen_idx'] < len(ex['cands']) else 'OOB'}")
            print(f"    RIICHI pos : {riichi_pos} in candidates list")
            ok = (ex["chosen_idx"] == riichi_pos)
            print(f"    chosen==RIICHI pos? : {'✅ YES' if ok else '❌ NO (mismatch!)'}")
    else:
        print(f"\n  ⚠️  未找到任何 action_chosen==RIICHI 的样本，无法展示示例。")

    print(f"\n{SEP2}\n")


def main():
    parser = argparse.ArgumentParser(
        description="验证标注数据中立直 token 完整性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 检查 Stage 3 训练数据（最关键）
  python scripts/check_riichi_in_data.py data/annotated_grp

  # 检查原始标注（对比用）
  python scripts/check_riichi_in_data.py data/annotated --files 30

  # 全量扫描
  python scripts/check_riichi_in_data.py data/annotated_grp --all
        """
    )
    parser.add_argument(
        "data_dir", nargs="?",
        default="data/annotated",
        help="标注数据目录（默认: data/annotated）",
    )
    parser.add_argument(
        "--files", "-n", type=int, default=20,
        help="扫描文件数（默认: 20，与 --all 互斥）",
    )
    parser.add_argument(
        "--sample", "-s", action="store_true",
        help="随机采样文件（默认按文件名排序取前 N 个）",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="扫描全部文件（忽略 --files 限制，较慢）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌  目录不存在: {data_dir}")
        print(f"    请传入正确的数据目录路径，例如：")
        print(f"    python scripts/check_riichi_in_data.py /root/autodl-tmp/rinshan/data/annotated_grp")
        sys.exit(1)

    all_files = sorted(data_dir.rglob("*.jsonl"))
    if not all_files:
        print(f"❌  未找到 .jsonl 文件: {data_dir}")
        sys.exit(1)

    print(f"数据目录    : {data_dir.resolve()}")
    print(f"找到文件数  : {len(all_files)}")

    if args.all:
        files = all_files
    elif args.sample:
        n = min(args.files, len(all_files))
        files = random.sample(all_files, n)
        print(f"随机采样    : {n} 个文件")
    else:
        files = all_files[:args.files]
        print(f"扫描前 {len(files)} 个文件（按文件名排序）")

    stats: dict = {
        "total": 0,
        "parse_errors": 0,
        # 候选
        "has_riichi_cand": 0,
        "empty_cands": 0,
        "cand_len_dist": Counter(),
        # 选择
        "riichi_chosen": 0,
        "discard_chosen": 0,
        "tsumo_chosen": 0,
        "ron_chosen": 0,
        "pass_chosen": 0,
        "other_chosen": 0,
        "oob_chosen": 0,
        # tenpai
        "tenpai_samples": 0,
        "tenpai_has_riichi_cand": 0,
        # progression 交叉验证
        "has_prog_riichi_in_prog": 0,
        # grp reward
        "has_grp_reward": 0,
        "grp_reward_zero": 0,
    }
    riichi_examples: list = []

    for i, f in enumerate(files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(files)}] {f.name}  (累计 {stats['total']:,} 样本)")
        analyze_file(f, stats, riichi_examples)

    print_report(stats, len(files), riichi_examples)


if __name__ == "__main__":
    main()
