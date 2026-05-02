"""
run_self_play.py — 自对弈 / 对战评估脚本

后端（默认 Rust libriichi，比 Python Arena 快 10x+）：
    不加参数          → 自动使用 Rust；libriichi 不可用时自动降级
    --no_rust         → 强制 Python Arena（调试 / libriichi 不可用时使用）
    --parallel_games  → 每个 wave 并发局数（默认等于 n_games，即一次跑完）
                        调小可降显存压力，调大可提高 GPU 利用率

用法：
    # 随机 agent 冒烟测试（自动 Python Arena）
    python scripts/run_self_play.py --mode random --n_games 4 --seed 42

    # 单模型自对弈（默认 Rust）
    python scripts/run_self_play.py \\
        --mode ai \\
        --ckpt checkpoints/stage3_best.pt \\
        --model_preset base \\
        --n_games 256 \\
        --device cuda

    # 对战评估：stage3 vs stage2（默认 Rust TwoVsTwo）
    python scripts/run_self_play.py \\
        --mode versus \\
        --ckpt  checkpoints/stage3_best.pt \\
        --ckpt2 checkpoints/stage2_best.pt \\
        --model_preset base \\
        --n_games 200 \\
        --device cuda \\
        --greedy

    # 强制 Python Arena（调试用）
    python scripts/run_self_play.py \\
        --mode versus \\
        --ckpt  checkpoints/stage3_best.pt \\
        --ckpt2 checkpoints/stage2_best.pt \\
        --n_games 100 --no_rust --device cuda

输出：
    data/self_play/games_YYYYMMDD_HHMMSS.jsonl
    每行一局游戏记录（Rust 路径含 agent_names/ranks/scores；
    Python 路径额外含 kyoku_logs）
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _libriichi_available() -> bool:
    """检测 libriichi Rust 扩展是否可用"""
    try:
        from libriichi.arena import TwoVsTwo, SelfPlay  # noqa: F401
        return True
    except ImportError:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Rinshan 自对弈 / 对战评估")
    p.add_argument("--mode",          choices=["random", "ai", "versus"], default="random",
                   help="运行模式：random=随机agent测试，ai=AI自对弈，versus=双模型对战")
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
    # 后端控制
    p.add_argument("--no_rust",       action="store_true",
                   help="强制使用 Python Arena（默认自动选 Rust；random 模式始终用 Python）")
    p.add_argument("--parallel_games", type=int, default=None,
                   help="Rust 后端每个 wave 并发局数（默认等于 n_games）；\n"
                        "调小降显存压力，调大提高 GPU 利用率")
    # AI 模式参数
    p.add_argument("--ckpt",          type=str, default=None,
                   help="主模型 checkpoint 路径（ai/versus 模式必填）")
    p.add_argument("--ckpt2",         type=str, default=None,
                   help="对手模型 checkpoint 路径（versus 模式必填）")
    p.add_argument("--ckpt2_preset",  choices=["nano", "base", "large"], default=None,
                   help="对手模型规模预设（不填则与 --model_preset 相同）")
    p.add_argument("--model_preset",  choices=["nano", "base", "large"],
                   default="nano", help="主模型规模预设")
    p.add_argument("--device",        type=str, default="cpu",
                   help="推理设备 (cpu / cuda / cuda:0 等)")
    p.add_argument("--temperature",   type=float, default=0.8,
                   help="采样温度")
    p.add_argument("--top_p",         type=float, default=0.9,
                   help="Top-p nucleus sampling")
    p.add_argument("--greedy",        action="store_true",
                   help="贪心解码（argmax，不随机）")
    p.add_argument("--n_agents",      type=int, default=1,
                   help="ai 模式：参与自对弈的 AI agent 数量（1=自我对局，4=四家各自独立）")
    # 日志 / 显示控制
    p.add_argument("--log_dir",       type=str, default=None,
                   help="保存每局 mjai 原始日志的目录（仅 Rust 后端 versus 模式支持，用于复盘调试）")
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
    # torch.compile 保存的 state_dict key 带 "_orig_mod." 前缀，strip 掉
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
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

    if args.mode == "versus":
        return build_versus_agents(args)

    # AI 自对弈模式
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


def build_versus_agents(args):
    """
    versus 模式：主模型 2 席 vs 对手模型 2 席。
    seats [ch_0, bl_0, ch_1, bl_1]，Arena round_robin 轮换后
    每局始终保持 2 ch + 2 bl，且座位均匀分布。
    """
    from rinshan.self_play.agent import RinshanAgent

    if not args.ckpt:
        raise ValueError("versus 模式必须指定 --ckpt")
    if not args.ckpt2:
        raise ValueError("versus 模式必须指定 --ckpt2")

    preset2 = args.ckpt2_preset or args.model_preset

    def _label(path: str) -> str:
        p = Path(path)
        return f"{p.parent.name}/{p.name}"

    ch_label = _label(args.ckpt)
    bl_label = _label(args.ckpt2)
    print(f"[+] Challenger : {ch_label}")
    print(f"[+] Baseline   : {bl_label}")

    # 两个模型各自加载（共享同一 device，显存允许时没问题）
    model_ch = load_model(args.ckpt,  args.model_preset, args.device)
    model_bl = load_model(args.ckpt2, preset2,            args.device)

    # 构建 2 个 agent 实例（每个模型只实例化一次）：
    #   agent_ch: 承载 model_ch，逻辑上对应 ch_0 / ch_1 两席
    #   agent_bl: 承载 model_bl，逻辑上对应 bl_0 / bl_1 两席
    # Arena 按 id(agent.model) 分组，ch_0/ch_1 同 model 自动合并到同一 batch，
    # 避免原来 4 实例时 batch size 被无意义切半。
    # round_robin 下 game_idx=0 → seats=(ch, bl, ch, bl)
    #              game_idx=1 → seats=(bl, ch, bl, ch)  依次轮换
    # 每局恰好 2 ch + 2 bl，座次均匀覆盖。
    agent_ch = RinshanAgent(model_ch, name="ch", device=args.device,
                            temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)
    agent_bl = RinshanAgent(model_bl, name="bl", device=args.device,
                            temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)
    agents = [agent_ch, agent_bl, agent_ch, agent_bl]
    return agents


# ─────────────────────────────────────────────────────────────────────────────
# Rust 后端运行函数
# ─────────────────────────────────────────────────────────────────────────────

def _save_rust_results(results, output_dir: str, compress: bool) -> None:
    """把 Rust arena 结果序列化为 jsonl（简化格式，无 kyoku_logs）"""
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix    = ".jsonl.gz" if compress else ".jsonl"
    out_path  = out_dir / f"games_{timestamp}{suffix}"
    opener    = gzip.open if compress else open
    with opener(out_path, "wt", encoding="utf-8") as f:
        for i, r in enumerate(results):
            entry = {
                "game_id":      f"rust_{i:06d}",
                "agent_names":  list(r.names),
                "final_scores": list(r.scores),
                "ranks":        list(r.rankings()),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[+] 已保存 {len(results)} 局对局 → {out_path}")


def evaluate_versus_strength(args) -> dict:
    """返回对战汇总指标，供 Stage3 arena gate 使用。"""
    import time
    from rinshan.self_play.agent import RinshanAgent
    from libriichi.arena import TwoVsTwo

    n_games = int(args.n_games)
    if n_games % 2 != 0:
        n_games += 1
    wave = int(args.parallel_games)
    if wave % 2 != 0:
        wave += 1

    preset2 = args.ckpt2_preset or args.model_preset
    model_ch = load_model(args.ckpt, args.model_preset, args.device)
    model_bl = load_model(args.ckpt2, preset2, args.device)
    agent_ch = RinshanAgent(model_ch, name="ch", device=args.device,
                            temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)
    agent_bl = RinshanAgent(model_bl, name="bl", device=args.device,
                            temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)

    arena = TwoVsTwo(disable_progress_bar=args.quiet, log_dir=args.log_dir)
    all_results = []
    generated = 0
    skipped = 0
    t0 = time.time()
    _LIBRIICHI_HAND_ERRS = ("is not in hand", "cannot tsumogiri")
    while generated < n_games:
        this_wave = min(wave, n_games - generated)
        try:
            results = arena.py_vs_py(agent_ch, agent_bl, (args.seed + generated, 0), this_wave // 2)
            all_results.extend(results)
            generated += this_wave
        except RuntimeError as e:
            if any(tag in str(e) for tag in _LIBRIICHI_HAND_ERRS):
                skipped += 1
                generated += 1
            else:
                raise

    elapsed = time.time() - t0
    ch_ranks, bl_ranks = [], []
    ch_scores, bl_scores = [], []
    for r in all_results:
        rr = list(r.rankings())
        sc = list(r.scores)
        names = list(r.names)
        for seat in range(4):
            if names[seat] == "ch":
                ch_ranks.append(rr[seat])
                ch_scores.append(sc[seat])
            elif names[seat] == "bl":
                bl_ranks.append(rr[seat])
                bl_scores.append(sc[seat])

    ch_avg = sum(x + 1 for x in ch_ranks) / max(len(ch_ranks), 1)
    bl_avg = sum(x + 1 for x in bl_ranks) / max(len(bl_ranks), 1)
    n_ch = max(len(ch_ranks), 1)
    n_bl = max(len(bl_ranks), 1)
    ch_first  = sum(1 for x in ch_ranks if x == 0) / n_ch
    ch_second = sum(1 for x in ch_ranks if x == 1) / n_ch
    ch_third  = sum(1 for x in ch_ranks if x == 2) / n_ch
    ch_fourth = sum(1 for x in ch_ranks if x == 3) / n_ch
    bl_first  = sum(1 for x in bl_ranks if x == 0) / n_bl
    bl_second = sum(1 for x in bl_ranks if x == 1) / n_bl
    bl_third  = sum(1 for x in bl_ranks if x == 2) / n_bl
    bl_fourth = sum(1 for x in bl_ranks if x == 3) / n_bl
    ch_score = sum(ch_scores) / max(len(ch_scores), 1)
    bl_score = sum(bl_scores) / max(len(bl_scores), 1)
    delta = ch_avg - bl_avg
    se = (torch.tensor([(x + 1) for x in ch_ranks], dtype=torch.float32).std(unbiased=False).item() / math.sqrt(max(len(ch_ranks), 1))) if ch_ranks else 0.0
    return {
        "games": len(all_results),
        "elapsed": elapsed,
        "challenger_avg_rank": ch_avg,
        "baseline_avg_rank": bl_avg,
        "challenger_first_rate":  ch_first,
        "challenger_second_rate": ch_second,
        "challenger_third_rate":  ch_third,
        "challenger_fourth_rate": ch_fourth,
        "baseline_first_rate":    bl_first,
        "baseline_second_rate":   bl_second,
        "baseline_third_rate":    bl_third,
        "baseline_fourth_rate":   bl_fourth,
        "challenger_avg_score": ch_score,
        "baseline_avg_score": bl_score,
        "delta_rank": delta,
        "challenger_rank_se": se,
    }, all_results


def run_rust_versus(args) -> None:
    """Rust TwoVsTwo 双模型对战（wave 循环，支持 --parallel_games）"""
    summary, all_results = evaluate_versus_strength(args)
    delta = summary["delta_rank"]
    verdict = ("↑ Challenger 胜" if delta < -0.05
                else "↓ Baseline 胜" if delta > 0.05
                else "→ 持平")
    print(f"\n{'='*58}")
    print(f"对战完成  {summary['games']} 局 | 用时 {summary['elapsed']:.1f}s | 速度 {summary['games']/summary['elapsed']:.2f} 局/s")
    print(f"{'─'*58}")
    def _rank_str(s, prefix):
        return (f"一位 {s[prefix+'first_rate']*100:5.1f}%  "
                f"二位 {s[prefix+'second_rate']*100:5.1f}%  "
                f"三位 {s[prefix+'third_rate']*100:5.1f}%  "
                f"四位 {s[prefix+'fourth_rate']*100:5.1f}%")
    print(f"Challenger  平均顺位 {summary['challenger_avg_rank']:.3f}  {_rank_str(summary, 'challenger_')}  平均得分 {summary['challenger_avg_score']:.0f}")
    print(f"Baseline    平均顺位 {summary['baseline_avg_rank']:.3f}  {_rank_str(summary, 'baseline_')}  平均得分 {summary['baseline_avg_score']:.0f}")
    print(f"顺位差 Δ={delta:+.3f}  {verdict}")
    print("="*58)
    _save_rust_results(all_results, args.output, args.compress)


def run_rust_selfplay(args) -> None:
    """Rust SelfPlay 单模型自对弈（wave 循环，支持 --parallel_games）"""
    import time
    from rinshan.self_play.agent import RinshanAgent
    from libriichi.arena import SelfPlay

    model = load_model(args.ckpt, args.model_preset, args.device)
    agent = RinshanAgent(model, name="selfplay", device=args.device,
                         temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)

    arena       = SelfPlay(disable_progress_bar=args.quiet)
    all_results = []
    generated   = 0
    skipped     = 0
    t0          = time.time()

    # libriichi 已知 bug：极少数 seed 下 py_self_play 会抛出
    #   "X is not in hand" 或 "cannot tsumogiri" RuntimeError，
    # 原因是 libriichi 下发给 Python 的 start_kyoku.tehais 与
    # Rust 内部实际手牌存在间歇性不一致。
    # workaround：跳过触发该 bug 的 seed（通常 <1%），继续生成后续局。
    _LIBRIICHI_HAND_ERRS = ("is not in hand", "cannot tsumogiri")

    while generated < args.n_games:
        this_wave = min(args.parallel_games, args.n_games - generated)
        try:
            results = arena.py_self_play(agent,
                                         (args.seed + generated, 0),
                                         this_wave)
            all_results.extend(results)
            generated += this_wave
        except RuntimeError as e:
            emsg = str(e)
            if any(tag in emsg for tag in _LIBRIICHI_HAND_ERRS):
                # 跳过这一 wave，seed 前进 1，让后续 seed 继续生成
                skipped += 1
                generated += 1
                if not args.quiet:
                    print(f"\n[warn] libriichi hand-state bug @ seed={args.seed+generated-1}"
                          f"，已跳过（共跳过 {skipped} 局）", flush=True)
            else:
                raise
        if not args.quiet:
            elapsed_so_far = time.time() - t0
            speed = generated / max(elapsed_so_far, 1e-6)
            print(f"\r[Arena] {generated}/{args.n_games} games  "
                  f"({speed:.2f} 局/s)", end="", flush=True)

    if not args.quiet:
        print()

    elapsed = time.time() - t0

    # ── 顺位统计 ──────────────────────────────────────────────
    from collections import defaultdict
    agent_ranks: dict[str, list[int]] = defaultdict(list)
    agent_scores_map: dict[str, list[int]] = defaultdict(list)
    for r in all_results:
        names = list(r.names)
        rnks  = list(r.rankings())
        scs   = list(r.scores)
        for seat in range(4):
            agent_ranks[names[seat]].append(rnks[seat])
            agent_scores_map[names[seat]].append(scs[seat])

    print(f"\n{'='*68}")
    print(f"自对弈完成  {len(all_results)} 局 | 用时 {elapsed:.1f}s | "
          f"速度 {len(all_results)/elapsed:.2f} 局/s")
    print(f"{'─'*68}")
    print(f"{'名称':<16} {'平均顺位':>8} {'一位':>7} {'二位':>7} {'三位':>7} {'四位':>7} {'平均得分':>10}")
    print(f"{'─'*16} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")
    for name in sorted(agent_ranks.keys()):
        rks = agent_ranks[name]
        scs = agent_scores_map[name]
        n   = max(len(rks), 1)
        avg = sum(x + 1 for x in rks) / n
        r1  = sum(1 for x in rks if x == 0) / n * 100
        r2  = sum(1 for x in rks if x == 1) / n * 100
        r3  = sum(1 for x in rks if x == 2) / n * 100
        r4  = sum(1 for x in rks if x == 3) / n * 100
        sc  = sum(scs) / max(len(scs), 1)
        print(f"{name:<16} {avg:>8.3f} {r1:>6.1f}% {r2:>6.1f}% {r3:>6.1f}% {r4:>6.1f}% {sc:>10.0f}")
    print("="*68)

    _save_rust_results(all_results, args.output, args.compress)


# ─────────────────────────────────────────────────────────────────────────────
# Python Arena 路径（原有逻辑，保持不变）
# ─────────────────────────────────────────────────────────────────────────────

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


def print_summary(records, elapsed: float, mode: str = "ai") -> None:
    """打印统计摘要"""
    if not records:
        print("[!] 无有效对局记录")
        return

    n = len(records)
    avg_kyoku = sum(len(r.kyoku_logs) for r in records) / n

    # 统计各 agent 平均顺位、一位率、平均得分
    agent_stats:  dict[str, list[int]] = {}
    agent_scores: dict[str, list[int]] = {}
    for rec in records:
        for name, rank, score in zip(rec.agent_names, rec.ranks, rec.final_scores):
            agent_stats.setdefault(name, []).append(rank)
            agent_scores.setdefault(name, []).append(score)

    print(f"\n{'='*58}")
    print(f"对局完成  {n} 局 | 用时 {elapsed:.1f}s | 速度 {n/elapsed:.2f} 局/s")
    print(f"平均 {avg_kyoku:.1f} 局/游戏")
    print(f"\n{'名称':<16} {'平均顺位':>8} {'一位':>7} {'二位':>7} {'三位':>7} {'四位':>7} {'平均得分':>10} {'出场数':>6}")
    print(f"{'─'*16} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*10} {'─'*6}")
    for name in sorted(agent_stats.keys()):
        ranks  = agent_stats[name]
        scores = agent_scores[name]
        n = max(len(ranks), 1)
        avg_rank = sum(ranks) / n + 1
        r1 = ranks.count(0) / n * 100
        r2 = ranks.count(1) / n * 100
        r3 = ranks.count(2) / n * 100
        r4 = ranks.count(3) / n * 100
        avg_score = sum(scores) / len(scores)
        print(f"{name:<16} {avg_rank:>8.3f} {r1:>6.1f}% {r2:>6.1f}% {r3:>6.1f}% {r4:>6.1f}% {avg_score:>10.0f} {len(ranks):>6}")

    # versus 模式：额外打印汇总对比
    if mode == "versus":
        ch_names = [nm for nm in agent_stats if nm == "ch" or nm.startswith("ch_")]
        bl_names = [nm for nm in agent_stats if nm == "bl" or nm.startswith("bl_")]
        if ch_names and bl_names:
            ch_ranks  = [r for nm in ch_names for r in agent_stats[nm]]
            bl_ranks  = [r for nm in bl_names for r in agent_stats[nm]]
            ch_scores = [s for nm in ch_names for s in agent_scores[nm]]
            bl_scores = [s for nm in bl_names for s in agent_scores[nm]]
            ch_avg = sum(ch_ranks) / len(ch_ranks) + 1
            bl_avg = sum(bl_ranks) / len(bl_ranks) + 1
            delta  = ch_avg - bl_avg   # 正 = challenger 顺位更差（数字更大）
            verdict = ("↑ Challenger 胜" if delta < -0.05
                       else "↓ Baseline 胜" if delta > 0.05
                       else "→ 持平")
            print(f"\n{'─'*58}")
            print(f"对战汇总（Challenger ch vs Baseline bl）：")
            def _rs(rks):
                n = max(len(rks), 1)
                return (f"一位 {rks.count(0)/n*100:5.1f}%  "
                        f"二位 {rks.count(1)/n*100:5.1f}%  "
                        f"三位 {rks.count(2)/n*100:5.1f}%  "
                        f"四位 {rks.count(3)/n*100:5.1f}%")
            print(f"  Challenger  平均顺位 {ch_avg:.3f}  {_rs(ch_ranks)}  "
                  f"平均得分 {sum(ch_scores)/len(ch_scores):.0f}")
            print(f"  Baseline    平均顺位 {bl_avg:.3f}  {_rs(bl_ranks)}  "
                  f"平均得分 {sum(bl_scores)/len(bl_scores):.0f}")
            print(f"  顺位差 Δ={delta:+.3f}  {verdict}")

    print("="*58)


def main():
    args = parse_args()
    import time

    # parallel_games 默认等于 n_games（单 wave 跑完）
    if args.parallel_games is None:
        args.parallel_games = args.n_games

    # random 模式没有 Rust 支持；其余默认走 Rust
    use_rust = (
        args.mode != "random"
        and not args.no_rust
        and _libriichi_available()
    )

    if args.mode != "random" and not args.no_rust and not _libriichi_available():
        print("[!] libriichi 不可用，自动降级到 Python Arena")

    backend = "Rust libriichi" if use_rust else "Python Arena"
    print(f"[Rinshan] mode={args.mode}  n_games={args.n_games}  "
          f"seed={args.seed}  length={args.game_length}  "
          f"backend={backend}  parallel_games={args.parallel_games}")

    # ── Rust 路径 ────────────────────────────────────────────────
    if use_rust:
        if args.mode == "versus":
            if not args.ckpt:
                raise ValueError("versus 模式必须指定 --ckpt")
            if not args.ckpt2:
                raise ValueError("versus 模式必须指定 --ckpt2")
            run_rust_versus(args)
        elif args.mode == "ai":
            if not args.ckpt:
                raise ValueError("ai 模式必须指定 --ckpt")
            run_rust_selfplay(args)
        return

    # ── Python Arena 路径 ────────────────────────────────────────
    agents = build_agents(args)
    print(f"[+] Agents: {[a.name for a in agents]}")

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
        print_summary(records, elapsed, mode=args.mode)

    out_path = save_records(records, args.output, args.compress)
    print(f"[+] 已保存 {len(records)} 局对局 → {out_path}")


if __name__ == "__main__":
    main()
