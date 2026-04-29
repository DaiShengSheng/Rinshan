from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from libriichi.arena import SelfPlay
from rinshan.model.full_model import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.constants import MODEL_CONFIGS
from rinshan.self_play.agent import RinshanAgent


def load_model(ckpt_path: str, preset: str, device: str):
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
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_preset", choices=["nano", "base", "large"], default="base")
    p.add_argument("--n_games", type=int, default=32, help="总局数")
    p.add_argument("--parallel_games", type=int, default=None, help="单个 wave 并发局数；默认等于 n_games")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()

    model = load_model(args.ckpt, args.model_preset, args.device)
    agent = RinshanAgent(model, name="selfplay", device=args.device, greedy=args.greedy)
    arena = SelfPlay(disable_progress_bar=True)

    parallel_games = args.parallel_games or args.n_games
    parallel_games = max(1, min(parallel_games, args.n_games))

    all_results = []
    generated = 0
    t0 = time.time()
    while generated < args.n_games:
        wave_games = min(parallel_games, args.n_games - generated)
        results = arena.py_self_play(agent, (args.seed + generated, 0), wave_games)
        all_results.extend(results)
        generated += wave_games
    elapsed = time.time() - t0
    print({
        "games": len(all_results),
        "parallel_games": parallel_games,
        "elapsed_s": round(elapsed, 3),
        "games_per_s": round(len(all_results) / elapsed, 3),
        "s_per_game": round(elapsed / len(all_results), 3),
    })


if __name__ == "__main__":
    main()
