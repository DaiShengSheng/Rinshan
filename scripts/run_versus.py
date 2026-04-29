"""
run_versus.py — 基于 libriichi Rust Arena 的双模型对战评估

用法：
    python scripts/run_versus.py \
        --ckpt  checkpoints/stage3_best.pt \
        --ckpt2 checkpoints/stage2_best.pt \
        --model_preset base \
        --n_games 200 \
        --device cuda \
        --greedy

说明：
    - 使用 libriichi.arena.TwoVsTwo 的 Rust 游戏引擎，比纯 Python Arena 快 10x+
    - Rust 侧通过 RinshanBatchAgent 将 mjai event JSON 传给 Python RinshanAgent
    - 游戏逻辑（牌局推进、向听计算等）全在 Rust 完成，Python 只负责 GPU 推理
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Rinshan libriichi Rust Arena 对战评估")
    p.add_argument("--ckpt",         type=str, required=True,  help="Challenger 模型路径")
    p.add_argument("--ckpt2",        type=str, required=True,  help="Baseline 模型路径")
    p.add_argument("--model_preset", choices=["nano", "base", "large"], default="base")
    p.add_argument("--ckpt2_preset", choices=["nano", "base", "large"], default=None)
    p.add_argument("--n_games",      type=int, default=200,    help="对局数（必须是偶数）")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top_p",        type=float, default=0.9)
    p.add_argument("--greedy",       action="store_true")
    p.add_argument("--log_dir",      type=str, default=None,   help="保存 mjai 日志的目录（可选）")
    p.add_argument("--quiet",        action="store_true")
    return p.parse_args()


def load_model(ckpt_path: str, preset: str, device: str):
    import torch
    from rinshan.model.full_model import RinshanModel
    from rinshan.model.transformer import TransformerConfig
    from rinshan.constants import MODEL_CONFIGS

    cfg = TransformerConfig(**MODEL_CONFIGS[preset])
    model = RinshanModel(transformer_cfg=cfg, use_belief=True, use_aux=False)

    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[+] 模型已加载：{ckpt_path} (preset={preset}, params={n_params:.1f}M)")
    return model


def main():
    args = parse_args()

    if args.n_games % 2 != 0:
        print("[!] n_games 必须是偶数（TwoVsTwo 每轮 2 局），自动+1")
        args.n_games += 1

    preset2 = args.ckpt2_preset or args.model_preset

    print(f"[Rinshan] Rust Arena  n_games={args.n_games}  seed={args.seed}")
    print(f"[+] Challenger : {args.ckpt}")
    print(f"[+] Baseline   : {args.ckpt2}")

    from rinshan.self_play.agent import RinshanAgent

    model_ch = load_model(args.ckpt,  args.model_preset, args.device)
    model_bl = load_model(args.ckpt2, preset2,            args.device)

    agent_ch = RinshanAgent(
        model_ch, name="challenger",
        device=args.device,
        temperature=args.temperature, top_p=args.top_p, greedy=args.greedy,
    )
    agent_bl = RinshanAgent(
        model_bl, name="baseline",
        device=args.device,
        temperature=args.temperature, top_p=args.top_p, greedy=args.greedy,
    )

    from libriichi.arena import TwoVsTwo

    arena = TwoVsTwo(
        disable_progress_bar=args.quiet,
        log_dir=args.log_dir,
    )

    t0 = time.time()
    # seed_start 是 (nonce, key) tuple，seed_count 是每边对局数
    # TwoVsTwo 每个 seed 跑 2 局（互换座位），所以 seed_count = n_games // 2
    result = arena.py_vs_py(agent_ch, agent_bl, (args.seed, 0), args.n_games // 2)
    elapsed = time.time() - t0

    # result 是 libriichi.stat 里的统计对象
    print(f"\n{'='*58}")
    print(f"对战完成  {args.n_games} 局 | 用时 {elapsed:.1f}s | 速度 {args.n_games/elapsed:.2f} 局/s")
    print(result)
    print("="*58)


if __name__ == "__main__":
    main()
