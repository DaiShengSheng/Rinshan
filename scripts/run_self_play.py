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
from typing import Any

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
    p.add_argument("--seed",          type=int, default=None,
                   help="随机种子（用于局面生成）；不填则自动用当前时间戳，每次产生不同牌局")
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
    p.add_argument("--mjson",         action="store_true",
                   help="将对局保存为 tenhou.net/6 可直接上传的 .mjson 格式；"
                        "单局输出到 --output 目录，多局自动建子文件夹")
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


# ─────────────────────────────────────────────────────────────────────────────
# MJAI → tenhou.net/6 格式转换器
# 参考：Equim-chan/mjai-reviewer convlog/src/tenhou/tile.rs + json_scheme.rs
# ─────────────────────────────────────────────────────────────────────────────

# MJAI tile string → 天凤数字编码
_MJAI_TO_TENHOU: dict[str, int] = {
    # 万子 11-19
    **{f"{n}m": 10 + n for n in range(1, 10)},
    # 筒子 21-29
    **{f"{n}p": 20 + n for n in range(1, 10)},
    # 索子 31-39
    **{f"{n}s": 30 + n for n in range(1, 10)},
    # 字牌 41-47（MJAI 标准名）
    "E": 41, "S": 42, "W": 43, "N": 44,
    "P": 45, "F": 46, "C": 47,
    # 字牌 41-47（1z-7z 写法，我们引擎用）
    "1z": 41, "2z": 42, "3z": 43, "4z": 44,
    "5z": 45, "6z": 46, "7z": 47,
    # 赤五 51-53（标准名）
    "5mr": 51, "5pr": 52, "5sr": 53,
    # 赤五 51-53（0x 写法，我们引擎用）
    "0m": 51, "0p": 52, "0s": 53,
    "?": 0,
}

def _t(pai: str) -> int:
    """MJAI tile string → 天凤数字"""
    return _MJAI_TO_TENHOU.get(pai, 0)


def _encode_naki_chi(actor: int, target: int, pai: str, consumed: list[str]) -> str:
    """
    吃牌 → 天凤 takes 字符串，格式：c{pai}{c0}{c1}
    target 必须是 kamicha（(actor+3)%4）
    c0/c1 是手牌中用的两张
    """
    return f"c{_t(pai):02d}{_t(consumed[0]):02d}{_t(consumed[1]):02d}"


def _encode_naki_pon(actor: int, target: int, pai: str, consumed: list[str]) -> str:
    """
    碰牌 → 天凤 takes 字符串
    格式根据 target 方向：
      kamicha  (actor+3)%4 → p{pai}{c0}{c1}
      toimen   (actor+2)%4 → {c0}p{pai}{c1}  (p在idx=2)
      shimocha (actor+1)%4 → {c0}{c1}p{pai}  (p在idx=4)
    """
    p = _t(pai)
    c0, c1 = _t(consumed[0]), _t(consumed[1])
    rel = (target - actor) % 4
    if rel == 3:   # kamicha
        return f"p{p:02d}{c0:02d}{c1:02d}"
    elif rel == 2: # toimen
        return f"{c0:02d}p{p:02d}{c1:02d}"
    else:          # shimocha rel==1
        return f"{c0:02d}{c1:02d}p{p:02d}"


def _encode_naki_daiminkan(actor: int, target: int, pai: str, consumed: list[str]) -> str:
    """
    大明杠 → 天凤 takes 字符串
    格式根据 target 方向（与碰类似但用 m，4张）：
      kamicha  → m{pai}{c0}{c1}{c2}
      toimen   → {c0}m{pai}{c1}{c2}
      shimocha → {c0}{c1}{c2}m{pai}
    """
    p = _t(pai)
    cs = [_t(c) for c in consumed]
    rel = (target - actor) % 4
    if rel == 3:   # kamicha
        return f"m{p:02d}{cs[0]:02d}{cs[1]:02d}{cs[2]:02d}"
    elif rel == 2: # toimen
        return f"{cs[0]:02d}m{p:02d}{cs[1]:02d}{cs[2]:02d}"
    else:          # shimocha
        return f"{cs[0]:02d}{cs[1]:02d}{cs[2]:02d}m{p:02d}"


def _encode_naki_kakan(actor: int, pai: str, consumed: list[str]) -> str:
    """
    加杠 → 天凤 discards 字符串（格式同碰，但用 k）
    consumed[0..2] 是已碰的三张，consumed 里的 target 信息需从原碰牌方向推断。
    我们这里用最简单的：从 mjai 的 consumed 判断哪张是「来自对家」。
    天凤格式：
      previously pon from kamicha  → k{pai}{c0}{c1}{c2}
      previously pon from toimen   → {c0}k{pai}{c1}{c2}
      previously pon from shimocha → {c0}{c1}k{c2}{pai} (k在idx=4)
    实际上加杠时 mjai 没有给出原碰方向，用 idx=0 (kamicha) 作为默认即可，
    viewer 只需要知道是杠，不会影响回放正确性。
    """
    p = _t(pai)
    cs = [_t(c) for c in consumed]
    # 天凤格式 k 在 idx=0 位置（previously pon from kamicha）
    return f"k{p:02d}{cs[0]:02d}{cs[1]:02d}{cs[2]:02d}"


def _encode_naki_ankan(actor: int, consumed: list[str]) -> str:
    """
    暗杠 → 天凤 discards 字符串
    格式：{c0}{c1}{c2}a{c3}  （a 固定在 idx=6）
    """
    cs = [_t(c) for c in consumed]
    return f"{cs[0]:02d}{cs[1]:02d}{cs[2]:02d}a{cs[3]:02d}"


def mjai_events_to_tenhou(events: list[dict]) -> dict[str, Any]:
    """
    把一局完整的 MJAI 事件列表（含 start_game … end_game）
    转换为 tenhou.net/6 JSON 格式的 dict。

    tenhou.net/6 格式（参考 mjai-reviewer/convlog/src/tenhou/json_scheme.rs）：
    {
      "title": ["", ""],
      "name": [p0, p1, p2, p3],
      "rule": {"disp": "般南喰赤", "aka": 1},
      "log": [
        [                             ← 一局
          [kyoku_num, honba, kyotaku],
          [score0, score1, score2, score3],
          [dora_num, ...],            ← dora indicators
          [ura_num, ...],             ← ura indicators（和了后填）
          [haipai_0...],  [takes_0...],  [discards_0...],
          [haipai_1...],  [takes_1...],  [discards_1...],
          [haipai_2...],  [takes_2...],  [discards_2...],
          [haipai_3...],  [takes_3...],  [discards_3...],
          results         ← ["和了", deltas, [who,target,...]] or ["流局", deltas]
        ],
        ...
      ]
    }
    """
    names = ["Player0", "Player1", "Player2", "Player3"]
    kyoku_logs: list[list[Any]] = []

    # ── 解析 start_game ──────────────────────────────────────────
    for ev in events:
        if ev.get("type") == "start_game":
            names = list(ev.get("names", names))
            break

    # ── 逐局解析 ─────────────────────────────────────────────────
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.get("type") != "start_kyoku":
            i += 1
            continue

        # --- start_kyoku 字段 ---
        bakaze_map = {"E": 0, "S": 4, "W": 8, "N": 12}
        bakaze_off = bakaze_map.get(ev.get("bakaze", "E"), 0)
        kyoku_in_wind = ev.get("kyoku", 1) - 1   # 0-based
        kyoku_num = bakaze_off + kyoku_in_wind     # 天凤 kyoku_num
        honba    = ev.get("honba", 0)
        kyotaku  = ev.get("kyotaku", 0)
        scores   = list(ev.get("scores", [25000]*4))
        haipai   = [list(h) for h in ev.get("tehais", [[]]*4)]  # list[list[str]]
        dora_ind = [_t(ev.get("dora_marker", "?"))]
        ura_ind:  list[int] = []

        # --- 每家的 takes / discards ---
        takes:    list[list[Any]] = [[], [], [], []]
        discards: list[list[Any]] = [[], [], [], []]

        # --- 结算 ---
        results: list[Any] = []
        pending_reach: list[bool] = [False, False, False, False]

        i += 1
        while i < len(events):
            ev = events[i]
            t_type = ev.get("type", "")
            i += 1

            if t_type == "end_kyoku":
                break

            actor = ev.get("actor", 0)

            if t_type == "tsumo":
                takes[actor].append(_t(ev["pai"]))

            elif t_type == "dahai":
                if pending_reach[actor]:
                    # reach 后的第一张打牌 → 合并为 "r{pai}" 写入 discards
                    pai_num = 60 if ev.get("tsumogiri", False) else _t(ev["pai"])
                    discards[actor].append(f"r{pai_num:02d}")
                    pending_reach[actor] = False
                elif ev.get("tsumogiri", False):
                    discards[actor].append(60)
                else:
                    discards[actor].append(_t(ev["pai"]))

            elif t_type == "reach":
                # reach 本身不写入 discards，仅标记等待下一张打牌
                pending_reach[actor] = True

            elif t_type == "reach_accepted":
                pass  # 天凤格式不单独记录

            elif t_type == "chi":
                takes[actor].append(
                    _encode_naki_chi(actor, ev["target"], ev["pai"], ev["consumed"])
                )

            elif t_type == "pon":
                takes[actor].append(
                    _encode_naki_pon(actor, ev["target"], ev["pai"], ev["consumed"])
                )

            elif t_type == "daiminkan":
                takes[actor].append(
                    _encode_naki_daiminkan(actor, ev["target"], ev["pai"], ev["consumed"])
                )

            elif t_type == "kakan":
                discards[actor].append(
                    _encode_naki_kakan(actor, ev["pai"], ev["consumed"])
                )

            elif t_type == "ankan":
                discards[actor].append(
                    _encode_naki_ankan(actor, ev["consumed"])
                )

            elif t_type == "dora":
                dora_ind.append(_t(ev["dora_marker"]))

            elif t_type == "hora":
                deltas = list(ev.get("deltas", [0, 0, 0, 0]))
                who    = ev["actor"]
                target = ev["target"]
                ura_markers = ev.get("ura_markers", [])
                ura_ind.extend(_t(u) for u in ura_markers)
                if not results:
                    results.append("和了")
                results.append(deltas)
                results.append([who, target])

            elif t_type == "ryukyoku":
                deltas = list(ev.get("deltas", [0, 0, 0, 0]))
                results = ["流局", deltas]

        # haipai 数字化
        haipai_num = [[_t(p) for p in h] for h in haipai]

        kyoku_entry: list[Any] = [
            [kyoku_num, honba, kyotaku],
            scores,
            dora_ind,
            ura_ind,
            haipai_num[0], takes[0], discards[0],
            haipai_num[1], takes[1], discards[1],
            haipai_num[2], takes[2], discards[2],
            haipai_num[3], takes[3], discards[3],
            results,
        ]
        kyoku_logs.append(kyoku_entry)

    return {
        "title": ["", ""],
        "name": names,
        "rule": {"disp": "般南喰赤", "aka": 1},
        "log": kyoku_logs,
    }


def _collect_mjson(log_dir: str, output_dir: str, n_games: int, label: str = "game") -> None:
    """
    把 Rust log_dir 里的 .json.gz 转换为 tenhou.net/6 JSON 格式并保存。
    同时也保留一份 MJAI 格式（.mjai 后缀）。
    多局时建子文件夹；单局时直接放在 output_dir 下。
    """
    src_dir = Path(log_dir)
    out_dir = Path(output_dir)

    gz_files = sorted(src_dir.glob("*.json.gz"))
    if not gz_files:
        print("[!] log_dir 中没有找到 .json.gz 文件，跳过 mjson 转存")
        return

    # 每局 Rust 可能产生 _a / _b 两个文件（TwoVsTwo），取 _a 即可（全知视角一致）
    seen_stems: set[str] = set()
    mjson_files: list[Path] = []
    for gz in gz_files:
        stem = gz.stem.replace(".json", "")
        base = stem.rstrip("_ab").rstrip("_")
        if base not in seen_stems:
            seen_stems.add(base)
            mjson_files.append(gz)

    multi = len(mjson_files) > 1
    if multi:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir  = out_dir / f"tenhou_{label}_{timestamp}"
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = out_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

    for idx, gz_path in enumerate(mjson_files):
        base_name = f"{label}_{idx:04d}" if multi else label
        tenhou_path = dest_dir / f"{base_name}.json"
        mjai_path   = dest_dir / f"{base_name}.mjai"

        with gzip.open(gz_path, "rt", encoding="utf-8") as src:
            events = [json.loads(ln) for ln in src if ln.strip()]

        # 保存 MJAI 格式（每行一个 JSON，原始格式）
        with open(mjai_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

        # 转换并保存 tenhou.net/6 格式
        tenhou_data = mjai_events_to_tenhou(events)
        with open(tenhou_path, "w", encoding="utf-8") as f:
            json.dump(tenhou_data, f, ensure_ascii=False)

    if multi:
        print(f"[+] 已导出 {len(mjson_files)} 局 → {dest_dir}/")
        print(f"    tenhou.net/6 格式: {label}_XXXX.json")
        print(f"    MJAI 格式:         {label}_XXXX.mjai")
    else:
        print(f"[+] 已导出 tenhou格式 → {dest_dir / (label + '.json')}")
        print(f"    MJAI格式         → {dest_dir / (label + '.mjai')}")
    print(f"    上传地址：https://tenhou.net/6/  （将 .json 拖入页面即可回放）")


def evaluate_versus_strength(args, log_dir_override=None) -> dict:
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

    effective_log_dir = log_dir_override if log_dir_override is not None else args.log_dir
    arena = TwoVsTwo(disable_progress_bar=args.quiet, log_dir=effective_log_dir)
    all_results = []
    generated = 0
    skipped = 0
    t0 = time.time()
    _LIBRIICHI_HAND_ERRS = ("is not in hand", "cannot tsumogiri", "not a hora hand")
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
                if not args.quiet:
                    print(f"\n[warn] libriichi hand-state bug @ seed={args.seed + generated - 1}"
                          f"，已跳过（共跳过 {skipped} 局）", flush=True)
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
    import tempfile, shutil

    # mjson 模式：用临时目录接收 log_dir 输出，结束后转存
    _tmp_log_dir = None
    effective_log_dir = args.log_dir
    if args.mjson:
        _tmp_log_dir = tempfile.mkdtemp(prefix="rinshan_logdir_")
        effective_log_dir = _tmp_log_dir

    summary, all_results = evaluate_versus_strength(args, log_dir_override=effective_log_dir)
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

    if args.mjson and _tmp_log_dir:
        _collect_mjson(_tmp_log_dir, args.output, summary["games"], label="versus")
        shutil.rmtree(_tmp_log_dir, ignore_errors=True)


def run_rust_selfplay(args) -> None:
    """Rust SelfPlay 单模型自对弈（wave 循环，支持 --parallel_games）"""
    import time, tempfile, shutil
    from rinshan.self_play.agent import RinshanAgent
    from libriichi.arena import SelfPlay

    model = load_model(args.ckpt, args.model_preset, args.device)
    agent = RinshanAgent(model, name="selfplay", device=args.device,
                         temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)

    # mjson 模式：用临时目录接收 log_dir 输出
    _tmp_log_dir = None
    effective_log_dir = None
    if args.mjson:
        _tmp_log_dir = tempfile.mkdtemp(prefix="rinshan_logdir_")
        effective_log_dir = _tmp_log_dir

    arena       = SelfPlay(disable_progress_bar=args.quiet, log_dir=effective_log_dir)
    all_results = []
    generated   = 0
    skipped     = 0
    t0          = time.time()

    # libriichi 已知 bug：极少数 seed 下 py_self_play 会抛出
    #   "X is not in hand" 或 "cannot tsumogiri" RuntimeError，
    # 原因是 libriichi 下发给 Python 的 start_kyoku.tehais 与
    # Rust 内部实际手牌存在间歇性不一致。
    # workaround：跳过触发该 bug 的 seed（通常 <1%），继续生成后续局。
    _LIBRIICHI_HAND_ERRS = ("is not in hand", "cannot tsumogiri", "not a hora hand")

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

    if args.mjson and _tmp_log_dir:
        _collect_mjson(_tmp_log_dir, args.output, len(all_results), label="selfplay")
        shutil.rmtree(_tmp_log_dir, ignore_errors=True)


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

    # seed 未指定时用时间戳，保证每次配牌不同
    if args.seed is None:
        import time
        args.seed = int(time.time()) & 0x7FFFFFFF
        print(f"[+] 自动 seed={args.seed}（时间戳）")

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

    # Python Arena mjson 导出（kyoku_logs 直接拼接）
    if args.mjson:
        _export_python_mjson(records, args.output)


def _export_python_mjson(records, output_dir: str) -> None:
    """
    Python Arena 路径的导出。
    同时输出两种格式：
      .json  → tenhou.net/6 天凤格式（可直接拖入页面回放）
      .mjai  → MJAI 原始格式（每行一个 JSON）
    """
    out_dir = Path(output_dir)
    multi = len(records) > 1
    if multi:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir  = out_dir / f"tenhou_selfplay_{timestamp}"
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = out_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

    for idx, rec in enumerate(records):
        base_name = f"game_{idx:04d}" if multi else "game"
        tenhou_path = dest_dir / f"{base_name}.json"
        mjai_path   = dest_dir / f"{base_name}.mjai"

        # 展开成 MJAI 事件列表
        events: list[dict] = [{"type": "start_game", "id": 0, "names": rec.agent_names}]
        for kyoku_events in rec.kyoku_logs:
            events.extend(kyoku_events)
        events.append({"type": "end_game", "scores": rec.final_scores})

        # 保存 MJAI 格式（每行一个 JSON）
        with open(mjai_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

        # 转换并保存 tenhou.net/6 格式
        tenhou_data = mjai_events_to_tenhou(events)
        with open(tenhou_path, "w", encoding="utf-8") as f:
            json.dump(tenhou_data, f, ensure_ascii=False)

    if multi:
        print(f"[+] 已导出 {len(records)} 局 → {dest_dir}/")
        print(f"    tenhou.net/6 格式: game_XXXX.json")
        print(f"    MJAI 格式:         game_XXXX.mjai")
    else:
        print(f"[+] 已导出 tenhou格式 → {dest_dir / 'game.json'}")
        print(f"    MJAI格式         → {dest_dir / 'game.mjai'}")
    print(f"    上传地址：https://tenhou.net/6/  （将 .json 拖入页面即可回放）")


if __name__ == "__main__":
    main()
