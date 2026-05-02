"""
export_mjson.py — 把 run_self_play.py (Python Arena) 的 jsonl 输出
转换为 tenhou.net/6 可以直接上传的 .mjson 格式。

用法：
    # 转换 jsonl 里的第 0 局（默认）
    python scripts/export_mjson.py data/self_play/games_XXXXXX.jsonl

    # 转换第 3 局
    python scripts/export_mjson.py data/self_play/games_XXXXXX.jsonl --game_idx 3

    # 指定输出路径
    python scripts/export_mjson.py data/self_play/games_XXXXXX.jsonl -o replay.mjson

输出：
    .mjson 文件，每行一个 JSON 事件，可直接拖入 https://tenhou.net/6/
"""
import argparse
import json
from pathlib import Path


def load_game(jsonl_path: str, game_idx: int) -> dict:
    with open(jsonl_path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i == game_idx:
                return json.loads(line.strip())
    raise IndexError(f"jsonl 里只有 {i+1} 局，找不到 game_idx={game_idx}")


def to_mjson(game: dict) -> list[dict]:
    """把一条 GameRecord 展开成 MJAI 事件列表（tenhou.net/6 格式）"""
    names = game.get("agent_names", ["p0", "p1", "p2", "p3"])
    final_scores = game.get("final_scores", [0, 0, 0, 0])

    events = []

    # ── start_game ──────────────────────────────────────────────
    events.append({
        "type":  "start_game",
        "id":    0,          # 观战视角（0 = 东家全知）
        "names": names,
    })

    # ── 逐局 kyoku 事件 ─────────────────────────────────────────
    for kyoku_events in game.get("kyoku_logs", []):
        for evt in kyoku_events:
            events.append(evt)

    # ── end_game ────────────────────────────────────────────────
    events.append({
        "type":   "end_game",
        "scores": final_scores,
    })

    return events


def main():
    p = argparse.ArgumentParser(description="jsonl → tenhou.net/6 mjson")
    p.add_argument("jsonl", help="run_self_play.py 输出的 .jsonl 文件路径")
    p.add_argument("--game_idx", type=int, default=0,
                   help="要导出的对局序号（0-based，默认 0）")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="输出文件路径（默认与 jsonl 同目录，改后缀为 _g{idx}.mjson）")
    args = p.parse_args()

    game = load_game(args.jsonl, args.game_idx)
    events = to_mjson(game)

    # 输出路径
    if args.output:
        out_path = Path(args.output)
    else:
        stem = Path(args.jsonl).stem
        out_path = Path(args.jsonl).parent / f"{stem}_g{args.game_idx}.mjson"

    with open(out_path, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

    names   = game.get("agent_names", [])
    scores  = game.get("final_scores", [])
    ranks   = game.get("ranks", [])
    n_kyoku = len(game.get("kyoku_logs", []))
    n_events = len(events)

    print(f"[+] 已导出 → {out_path}")
    print(f"    对局：{n_kyoku} 局  事件数：{n_events}")
    for seat, (name, score, rank) in enumerate(zip(names, scores, ranks)):
        print(f"    座位{seat} {name:<16} 得分 {score:>7}  顺位 {rank+1}")
    print(f"\n    上传地址：https://tenhou.net/6/")
    print(f"    （打开页面后将 .mjson 文件拖入即可回放）")


if __name__ == "__main__":
    main()
