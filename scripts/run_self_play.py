"""
run_self_play.py — 自对弈数据生成脚本

用法：
    # 随机 agent 冒烟测试
    python scripts/run_self_play.py --mode random --n_games 4 --seed 42

    # 用已训练的模型生成自对弈数据
    python scripts/run_self_play.py \
        --mode ai \
        --ckpt checkpoints/stage2_base/best.pt \
        --model_preset base \
        --n_games 256 \
        --output data/self_play \
        --device cuda

输出：
    data/self_play/games_YYYYMMDD_HHMMSS.jsonl
    每行一局游戏的完整 mjai 事件流（JSON list）
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Rinshan 自对弈数据生成")
    p.add_argument("--mode",          choices=["random", "ai"], default="random",
                   help="运行模式：random=随机agent测试，ai=AI自对弈")
    p.add_argument("--n_games",       type=int, default=4,
                   help="要生成的对局数量")
    p.add_argument("--seed",          type=int, default=0,
                   help="随机种子（用于局面生成）")
    p.add_argument("--game_length",   choices=["hanchan", "tonpuu"],
                   default="hanchan", help="hanchan=半庄 tonpuu=东风")
    p.add_argument("--output",        type=str, default="data/self_play",
                   help="输出目录")
    p.add_argument("--compress",      action="store_true",
                   help="输出 .jsonl.gz 压缩文件")
    # AI 模式参数
    p.add_argument("--ckpt",          type=str, default=None,
                   help="模型 checkpoint 路径（ai 模式必填）")
    p.add_argument("--model_preset",  choices=["nano", "base", "large"],
                   default="nano", help="模型规模预设")
    p.add_argument("--device",        type=str, default="cpu",
                   help="推理设备 (cpu / cuda / cuda:0 等)")
    p.add_argument("--temperature",   type=float, default=0.8,
                   help="采样温度")
    p.add_argument("--top_p",         type=float, default=0.9,
                   help="Top-p nucleus sampling")
    p.add_argument("--greedy",        action="store_true",
                   help="贪心解码（argmax，不随机）")
    p.add_argument("--n_agents",      type=int, default=1,
                   help="参与自对弈的 AI agent 数量（1=自我对局，4=四家各自独立）")
    # 显示控制
    p.add_argument("--quiet",         action="store_true")
    return p.parse_args()


def load_model(ckpt_path: str, preset: str, device: str):
    """加载 RinshanModel checkpoint"""
    import torch
    from rinshan.model.full_model import RinshanModel
    from rinshan.model.transformer import TransformerConfig
    from rinshan.constants import MODEL_CONFIGS

    cfg_dict = MODEL_CONFIGS[preset]
    cfg = TransformerConfig(**cfg_dict)
    model = RinshanModel(transformer_cfg=cfg, use_belief=True, use_aux=False)

    state = torch.load(ckpt_path, map_location=device)
    # 兼容不同的 checkpoint 格式
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"[+] 模型已加载：{ckpt_path} (preset={preset})")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    参数量：{param_count:.1f}M")
    return model


def build_agents(args):
    """根据参数构建 agent 列表"""
    from rinshan.self_play.agent import RandomAgent, RinshanAgent

    if args.mode == "random":
        agents = [RandomAgent(name=f"random_{i}", seed=args.seed+i)
                  for i in range(4)]
        return agents

    # AI 模式
    if not args.ckpt:
        raise ValueError("ai 模式必须指定 --ckpt 参数")

    model = load_model(args.ckpt, args.model_preset, args.device)
    n = max(1, min(args.n_agents, 4))
    agents = []
    for i in range(n):
        agents.append(RinshanAgent(
            model       = model,
            name        = f"rinshan_{i}",
            device      = args.device,
            temperature = args.temperature,
            top_p       = args.top_p,
            greedy      = args.greedy,
        ))
    return agents


def save_records(records, output_dir: str, compress: bool) -> Path:
    """把 GameRecord 序列化为 jsonl（或 jsonl.gz）"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ".jsonl.gz" if compress else ".jsonl"
    out_path = out_dir / f"games_{timestamp}{suffix}"

    opener = gzip.open if compress else open
    mode   = "wt"

    with opener(out_path, mode, encoding="utf-8") as f:
        for rec in records:
            # 每行写入一局的元信息 + 事件流
            entry = {
                "game_id":      rec.game_id,
                "seed":         list(rec.seed),
                "agent_names":  rec.agent_names,
                "final_scores": rec.final_scores,
                "ranks":        rec.ranks,
                "kyoku_logs":   rec.kyoku_logs,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return out_path


def print_summary(records, elapsed: float) -> None:
    """打印统计摘要"""
    if not records:
        print("[!] 无有效对局记录")
        return

    n = len(records)
    avg_kyoku = sum(len(r.kyoku_logs) for r in records) / n

    # 统计各 agent 平均顺位
    agent_stats: dict[str, list[int]] = {}
    for rec in records:
        for seat, (name, rank) in enumerate(zip(rec.agent_names, rec.ranks)):
            agent_stats.setdefault(name, []).append(rank)

    print(f"\n{'='*50}")
    print(f"自对弈完成  {n} 局 | 用时 {elapsed:.1f}s | "
          f"速度 {n/elapsed:.2f} 局/s")
    print(f"平均 {avg_kyoku:.1f} 局/游戏")
    print("\nAgent 平均顺位：")
    for name, ranks in sorted(agent_stats.items()):
        avg_rank = sum(ranks) / len(ranks) + 1  # 1-indexed
        print(f"  {name}: {avg_rank:.3f} 位（{len(ranks)} 次出场）")
    print("="*50)


def main():
    args = parse_args()
    import time

    print(f"[Rinshan 自对弈] mode={args.mode}  n_games={args.n_games}  "
          f"seed={args.seed}  length={args.game_length}")

    # 构建 agents
    agents = build_agents(args)
    print(f"[+] Agent：{[a.name for a in agents]}")

    # 构建 Arena
    from rinshan.self_play.arena import Arena

    arena = Arena(
        agents         = agents,
        n_games        = args.n_games,
        game_length    = args.game_length,
        base_seed      = args.seed,
        agent_rotation = "round_robin",
        show_progress  = not args.quiet,
    )

    t0 = time.time()
    records = arena.run()
    elapsed = time.time() - t0

    if not args.quiet:
        print_summary(records, elapsed)

    # 保存
    out_path = save_records(records, args.output, args.compress)
    print(f"[+] 已保存 {len(records)} 局对局 → {out_path}")


if __name__ == "__main__":
    main()
