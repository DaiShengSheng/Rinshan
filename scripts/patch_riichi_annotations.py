#!/usr/bin/env python3
"""
patch_riichi_annotations.py  —  纯 JSON 修复，不重新 parse，不重新 GRP

Bug 背景：
  simulator._handle_dahai 在 reach→dahai 时序中，若 shanten==-1（摸到和了牌才立直），
  _compute_discard_candidates 会设 can_riichi=False（因为 can_tsumo=True），
  导致 RIICHI_TOKEN(497) 不进入 candidates，_find_action_idx fallback 到 cands[0]（随机 DISCARD）。
  结果：所有立直宣言样本的 action_chosen 指向错误 DISCARD，RIICHI 学习信号为 0。

修复1（立直宣言点）：
  识别：riichi_declared[0]==True AND discard in cands AND melds[0] 为空（门清）
  处理：保留原打牌候选 + 追加 RIICHI_TOKEN，action_chosen 指向 RIICHI
  效果：IQL 能学到「同一局面人类选立直而非 damaten（打牌）」的 Q 对比信号
  注：副露时不处理（melds[0] 非空，不能立直，修复2负责清残留 RIICHI）

修复2（副露无法立直）：
  识别：riichi_declared[0]==False AND 497 in candidates AND melds[0] 非空
  处理：从 candidates 中删除 497，同步修正 action_chosen 的 index

使用方法：
  python scripts/patch_riichi_annotations.py \\
      --input  /root/autodl-tmp/rinshan/data/annotated_grp \\
      --output /root/autodl-tmp/rinshan/data/annotated_grp_patched \\
      --workers 16

  如果想原地修改（先备份！）：--output 同 --input
"""
from __future__ import annotations
import json, gzip, argparse, multiprocessing as mp
from pathlib import Path

RIICHI_TOKEN   = 497
DISCARD_OFFSET = 37
DISCARD_END    = 74   # 37..73 inclusive (34 regular + 3 aka)


def _has_discard(cands: list[int]) -> bool:
    return any(DISCARD_OFFSET <= c < DISCARD_END for c in cands)


def patch_sample(d: dict) -> tuple[dict, bool]:
    """修复单条 annotation，返回 (修复后的 dict, 是否发生了变化)"""
    cands           = d.get("action_candidates", [])
    riichi_declared = d.get("riichi_declared", [False] * 4)
    melds           = d.get("melds", [[], [], [], []])
    player_melds    = melds[0] if melds else []
    action_chosen   = d.get("action_chosen", 0)

    # ── 修复1：立直宣言点 ──────────────────────────────────────────────────
    # riichi_declared[0]=True（reach 事件已触发）+ candidates 含 DISCARD + melds[0] 为空（门清）
    # → 这是被 bug 丢掉了 RIICHI 候选的立直宣言决策点
    # 正确做法：保留原打牌候选（作为对比），追加 RIICHI_TOKEN，action_chosen 指向 RIICHI
    # 这样 IQL 才能学到「在这个局面人类选了立直而非打牌（damaten）」
    # 副露情况（melds[0] 非空）：不能立直，不做处理（修复2会清掉残留的 RIICHI）
    if riichi_declared[0] and _has_discard(cands) and not player_melds:
        new_cands = [c for c in cands if c != RIICHI_TOKEN]  # 防止重复追加
        new_cands.append(RIICHI_TOKEN)
        d["action_candidates"] = new_cands
        d["action_chosen"]     = len(new_cands) - 1          # 末尾的 RIICHI_TOKEN
        return d, True

    # ── 修复2：副露状态不能立直 ────────────────────────────────────────────
    # 有吃/碰副露时 can_riichi 不应为 True，但旧 simulator 没检查
    if (not riichi_declared[0]
            and RIICHI_TOKEN in cands
            and player_melds):
        # 记住当前被选中的 token，然后从候选里删 RIICHI，再重新找 index
        chosen_token  = cands[action_chosen] if action_chosen < len(cands) else None
        new_cands     = [c for c in cands if c != RIICHI_TOKEN]
        if not new_cands:
            # 极端情况：只剩 RIICHI，不应发生，保守跳过
            return d, False
        d["action_candidates"] = new_cands
        if chosen_token is not None and chosen_token in new_cands:
            d["action_chosen"] = new_cands.index(chosen_token)
        elif chosen_token == RIICHI_TOKEN:
            # 原来选的就是 RIICHI（理论上不会出现在旧数据里），给第一个候选
            d["action_chosen"] = 0
        # chosen_token 不在 new_cands 也不是 RIICHI → 保持原 index（可能越界但不常见）
        return d, True

    return d, False


def patch_file(args: tuple[Path, Path]) -> tuple[str, int, int, int]:
    """处理单个文件，返回 (文件名, total, riichi_fixed, meld_fixed)"""
    in_path, out_path = args
    total = riichi_fixed = meld_fixed = 0

    is_gz = in_path.suffix == ".gz"
    opener = gzip.open if is_gz else open
    mode_r = "rt"

    lines_out: list[str] = []
    with opener(in_path, mode_r, encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError:
                lines_out.append(raw_line)
                continue

            cands_before         = list(d.get("action_candidates", []))
            riichi_declared_self = d.get("riichi_declared", [False])[0]
            d, changed           = patch_sample(d)

            if changed:
                if riichi_declared_self and _has_discard(cands_before):
                    riichi_fixed += 1
                else:
                    meld_fixed += 1

            lines_out.append(json.dumps(d, ensure_ascii=False))
            total += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return in_path.name, total, riichi_fixed, meld_fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="快速修复 annotated 数据中立直相关 bug，无需重新 parse/GRP")
    parser.add_argument("--input",   required=True, help="输入目录（annotated_grp/）")
    parser.add_argument("--output",  required=True, help="输出目录（与 input 相同则原地修改，建议先备份）")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数（默认 8）")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)

    files = sorted(in_dir.glob("*.jsonl")) + sorted(in_dir.glob("*.jsonl.gz"))
    if not files:
        print(f"[ERROR] 在 {in_dir} 下找不到 .jsonl / .jsonl.gz 文件")
        return

    print(f"找到 {len(files)} 个文件，使用 {args.workers} 个进程...")

    tasks = [(fp, out_dir / fp.name) for fp in files]
    with mp.Pool(args.workers) as pool:
        results = pool.map(patch_file, tasks)

    grand_total = grand_riichi = grand_meld = 0
    for name, total, rf, mf in results:
        pct = (rf + mf) / max(total, 1) * 100
        print(f"  {name:40s}  {total:7d} samples  riichi_fixed={rf:5d}  meld_fixed={mf:4d}  ({pct:.1f}%)")
        grand_total += total
        grand_riichi += rf
        grand_meld   += mf

    print()
    print(f"汇总: {grand_total} 个样本")
    print(f"  立直宣言修复: {grand_riichi} ({grand_riichi/max(grand_total,1):.2%})")
    print(f"  副露去 RIICHI: {grand_meld} ({grand_meld/max(grand_total,1):.2%})")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
