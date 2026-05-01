#!/usr/bin/env python3
"""
calibrate_aux_weights.py — 从天凤复盘牌谱统计 aux 任务 loss 量级，推算 deal_in_risk 的合理 AUX_WEIGHT

原理
----
对每个 aux 任务计算「最优常数预测器」下的期望 loss（= 边际分布熵 H(μ)）。
以 shanten / tenpai_prob / opp_tenpai（均已设 weight=0.10）的 baseline loss 为参照，
推导让 deal_in_risk 的梯度贡献量级相近的权重：

    w_deal = mean( w_ref × E[loss_ref] ) / E[loss_deal]
    其中 ref ∈ {shanten, tenpai_prob, opp_tenpai}，w_ref = 0.10

还额外输出「随机初始化时」各任务的 loss（logit=0 → p=0.5），
方便对比训练开始 vs 收敛到常数预测时的量级差异。

deal_in_risk Label B 定义（与 simulator._calc_deal_in_risk 一致）：
    risk[tile_id] = 在当前决策点，听牌且未振听的对手中，
                   该牌是其待张的对手数 / 3
    → 复盘时用全知手牌信息计算；模型推理时只看公开信息，学习预测此值。

Usage:
    python scripts/calibrate_aux_weights.py \\
        --input  data/raw/mjai_json/ \\
        --workers 22 \\
        --limit  0          # 0 = 处理所有文件

Output:
    各任务 label 分布统计、baseline loss、随机初始化 loss、推算权重建议值
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.tile import Tile, hand_to_counts
from rinshan.algo.shanten import calc_shanten


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _remove_tile(hand: list, tile: Tile) -> None:
    """从手牌列表中移除一张牌（优先匹配赤牌标记）"""
    for i, t in enumerate(hand):
        if t.tile_id == tile.tile_id and t.is_aka == tile.is_aka:
            hand.pop(i)
            return
    for i, t in enumerate(hand):
        if t.tile_id == tile.tile_id:
            hand.pop(i)
            return


def _calc_dir(
    seat: int,
    hands: list[list],
    melds: list[list],
    in_riichi: list[bool],
    furiten: list[bool],
) -> list[float]:
    """
    Label B：在当前决策点，对手听牌时各牌的放铳危险度。

    返回 list[float] 长度 34：
        risk[tile_id] = 危险对手数 / 3   ∈ {0.0, 0.33, 0.67, 1.0}

    只有「已确定处于 13 张听牌等待状态」或「立直中」的对手才计入，
    14 张（刚摸牌）的对手做保守枚举（所有合法打法的待张并集）。
    振听对手不计入。
    """
    N_OPP = 3
    danger = [0] * 34

    for k in range(N_OPP):
        opp = (seat + k + 1) % 4
        opp_hand = hands[opp]
        if not opp_hand or furiten[opp]:
            continue

        counts = hand_to_counts(opp_hand)
        mc     = min(len(melds[opp]), 4)   # Rust shanten 上限 4 副露
        n_tiles = sum(counts)

        if in_riichi[opp]:
            # 立直：手牌固定，直接枚举待张
            for tid in range(34):
                counts[tid] += 1
                if calc_shanten(counts, mc) == -1:
                    danger[tid] += 1
                counts[tid] -= 1
            continue

        if n_tiles == 13:
            # 已经处于听牌等待（shanten==0 时 tile+1 → -1）
            if calc_shanten(counts, mc) != 0:
                continue
            for tid in range(34):
                counts[tid] += 1
                if calc_shanten(counts, mc) == -1:
                    danger[tid] += 1
                counts[tid] -= 1

        elif n_tiles == 14:
            # 刚摸牌，还没决定打哪张 → 保守：枚举所有打法的待张并集
            waits: set[int] = set()
            has_tenpai = False
            for tid in range(34):
                if counts[tid] == 0:
                    continue
                counts[tid] -= 1
                if calc_shanten(counts, mc) == 0:
                    has_tenpai = True
                    for w in range(34):
                        counts[w] += 1
                        if calc_shanten(counts, mc) == -1:
                            waits.add(w)
                        counts[w] -= 1
                counts[tid] += 1
            if has_tenpai:
                for w in waits:
                    danger[w] += 1

        else:
            # 副露后手牌张数更少，同 13 张逻辑处理
            if calc_shanten(counts, mc) != 0:
                continue
            for tid in range(34):
                counts[tid] += 1
                if calc_shanten(counts, mc) == -1:
                    danger[tid] += 1
                counts[tid] -= 1

    return [danger[t] / N_OPP for t in range(34)]


# ─────────────────────────────────────────────────────────────────────────────
# Worker：处理单个 .jsonl 文件
# ─────────────────────────────────────────────────────────────────────────────

def _process_file(path: Path) -> dict | None:
    """
    解析单个 .jsonl 文件，在每个打牌决策点收集 aux label 统计。

    返回 dict：
        n_decisions    : int           总打牌决策点数
        shanten_hist   : list[int]×10  各 shanten label (0~9) 的计数
        tenpai_count   : int           shanten==0 的决策点数
        dir_sum        : list[float]×34 deal_in_risk 各 tile 的 label 累计（Label B）
        opp_riichi_pos : list[int]×3   三个对手位置处于立直的决策点数
        n_games        : int           处理的游戏局数
        n_errors       : int           跳过的错误行数
    """
    stats: dict = {
        'n_decisions': 0,
        'shanten_hist': [0] * 10,
        'tenpai_count': 0,
        'dir_sum': [0.0] * 34,
        'opp_riichi_pos': [0, 0, 0],
        'n_games': 0,
        'n_errors': 0,
    }

    try:
        with open(path, encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    game = json.loads(line)
                    _process_game(game.get('events', []), stats)
                    stats['n_games'] += 1
                except Exception:
                    stats['n_errors'] += 1
    except Exception:
        return None

    return stats


def _process_game(events: list[dict], stats: dict) -> None:
    """重放一局完整事件，在每个打牌决策点采集 aux label"""
    hands: list[list] = [[] for _ in range(4)]
    melds: list[list] = [[] for _ in range(4)]
    in_riichi = [False] * 4
    furiten   = [False] * 4  # 仅追踪「打出自己待张」的永久振听

    for ev in events:
        etype = ev.get('type', '')

        # ── 新局开始，重置状态 ────────────────────────────────
        if etype == 'start_kyoku':
            hands     = [[] for _ in range(4)]
            melds     = [[] for _ in range(4)]
            in_riichi = [False] * 4
            furiten   = [False] * 4
            for seat, h in enumerate(ev.get('tehais', [[], [], [], []])):
                # 复盘牌谱四家 tehais 均为明文，无 '?'
                hands[seat] = [Tile.from_mjai(t) for t in h if t != '?']

        # ── 摸牌：加入手牌 ────────────────────────────────────
        elif etype == 'tsumo':
            pai_str = ev.get('pai', '?')
            if pai_str != '?':
                hands[ev['actor']].append(Tile.from_mjai(pai_str))

        # ── 打牌：采集 label，然后更新状态 ───────────────────
        elif etype == 'dahai':
            seat = ev['actor']
            tile = Tile.from_mjai(ev['pai'])

            # shanten label
            counts = hand_to_counts(hands[seat])
            mc     = min(len(melds[seat]), 4)   # Rust shanten 上限 4 副露
            sht    = calc_shanten(counts, mc)
            label_sht = max(0, min(9, sht + 1))
            stats['shanten_hist'][label_sht] += 1
            if sht <= 0:
                stats['tenpai_count'] += 1

            # deal_in_risk（Label B）
            dir_vec = _calc_dir(seat, hands, melds, in_riichi, furiten)
            for ti in range(34):
                stats['dir_sum'][ti] += dir_vec[ti]

            # opp_tenpai（立直 proxy）
            for k in range(3):
                if in_riichi[(seat + k + 1) % 4]:
                    stats['opp_riichi_pos'][k] += 1

            stats['n_decisions'] += 1

            # 移除打出的牌
            _remove_tile(hands[seat], tile)

            # 振听检测：打出了自己的待张
            if not furiten[seat] and not in_riichi[seat]:
                test = hand_to_counts(hands[seat])
                test[tile.tile_id] += 1
                if calc_shanten(test, min(len(melds[seat]), 4)) == -1:
                    furiten[seat] = True

        # ── 鸣牌：更新手牌和副露 ──────────────────────────────
        elif etype in ('chi', 'pon', 'daiminkan'):
            actor    = ev['actor']
            pai      = Tile.from_mjai(ev['pai'])
            consumed = [Tile.from_mjai(t) for t in ev.get('consumed', [])]
            for t in consumed:
                _remove_tile(hands[actor], t)
            melds[actor].append((etype, [pai] + consumed))

        elif etype == 'ankan':
            # 暗杠：新增一个 meld，手牌移除 consumed（4张）
            actor    = ev['actor']
            consumed = [Tile.from_mjai(t) for t in ev.get('consumed', [])]
            for t in consumed:
                _remove_tile(hands[actor], t)
            melds[actor].append(('ankan', consumed))

        elif etype == 'kakan':
            # 加杠：升级已有 pon → kakan，meld 数不变，手牌移除那1张
            actor = ev['actor']
            pai   = Tile.from_mjai(ev['pai'])
            _remove_tile(hands[actor], pai)
            # 找到对应的 pon 升级，不 append 新 meld
            for i, (mtype, tiles) in enumerate(melds[actor]):
                if mtype == 'pon' and tiles[0].tile_id == pai.tile_id:
                    melds[actor][i] = ('kakan', tiles + [pai])
                    break

        # ── 立直宣言 ──────────────────────────────────────────
        elif etype == 'reach':
            in_riichi[ev['actor']] = True


# ─────────────────────────────────────────────────────────────────────────────
# 统计聚合 & 权重推算
# ─────────────────────────────────────────────────────────────────────────────

def _H(p: float) -> float:
    """Binary entropy H(p) in nats，数值安全"""
    if p <= 1e-15 or p >= 1 - 1e-15:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def _CE(counts: list[int]) -> float:
    """多分类交叉熵（nats）—— 最优常数预测器的期望 loss"""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def compute_and_report(all_stats: list[dict]) -> None:
    """聚合所有 worker 结果，计算 baseline loss，输出权重建议"""

    # ── 合并 ──────────────────────────────────────────────────────────────
    agg: dict = {
        'n_decisions': 0,
        'shanten_hist': [0] * 10,
        'tenpai_count': 0,
        'dir_sum': [0.0] * 34,
        'opp_riichi_pos': [0, 0, 0],
        'n_games': 0,
        'n_errors': 0,
    }
    for s in all_stats:
        if s is None:
            continue
        agg['n_decisions']  += s['n_decisions']
        agg['tenpai_count'] += s['tenpai_count']
        agg['n_games']      += s['n_games']
        agg['n_errors']     += s['n_errors']
        for i in range(10):
            agg['shanten_hist'][i] += s['shanten_hist'][i]
        for i in range(34):
            agg['dir_sum'][i] += s['dir_sum'][i]
        for i in range(3):
            agg['opp_riichi_pos'][i] += s['opp_riichi_pos'][i]

    N = agg['n_decisions']
    SEP = '=' * 68

    print(f'\n{SEP}')
    print(f'  处理游戏局数 : {agg["n_games"]:,}')
    print(f'  总打牌决策点 : {N:,}')
    print(f'  跳过错误行   : {agg["n_errors"]:,}')
    print(SEP)

    # ──────────────────────────────────────────────────────────────────────
    # 1. shanten（10 分类 CE）
    # ──────────────────────────────────────────────────────────────────────
    h_shanten    = _CE(agg['shanten_hist'])
    loss_init_sh = math.log(10)           # 随机 logit（均匀分布）→ -log(1/10)

    print('\n[1] shanten  （10 分类 CrossEntropy）')
    total_sht = sum(agg['shanten_hist'])
    dist_str = '  '.join(
        f'sht{i-1}={agg["shanten_hist"][i]/total_sht:.3f}'
        for i in range(10) if agg['shanten_hist'][i] > 0
    )
    print(f'    label 分布 : {dist_str}')
    print(f'    baseline loss  (H of marginal) = {h_shanten:.4f} nats')
    print(f'    random-init loss (log 10)       = {loss_init_sh:.4f} nats')
    print(f'    当前 weight=0.10  → contribution (baseline) = {0.10*h_shanten:.4f}')

    # ──────────────────────────────────────────────────────────────────────
    # 2. tenpai_prob（scalar binary BCE）
    # ──────────────────────────────────────────────────────────────────────
    p_tenpai     = agg['tenpai_count'] / max(N, 1)
    h_tenpai     = _H(p_tenpai)
    loss_init_tp = math.log(2)            # BCE at p=0.5

    print('\n[2] tenpai_prob  （scalar binary BCE）')
    print(f'    正例率（tenpai）  = {p_tenpai:.4f}  ({p_tenpai*100:.2f}%)')
    print(f'    baseline loss  H(p)    = {h_tenpai:.4f} nats')
    print(f'    random-init loss log2  = {loss_init_tp:.4f} nats')
    print(f'    当前 weight=0.10  → contribution (baseline) = {0.10*h_tenpai:.4f}')

    # ──────────────────────────────────────────────────────────────────────
    # 3. deal_in_risk（34-dim binary BCE，Label B）
    # ──────────────────────────────────────────────────────────────────────
    tile_means     = [agg['dir_sum'][i] / max(N, 1) for i in range(34)]
    tile_entropies = [_H(m) for m in tile_means]
    h_dir          = sum(tile_entropies) / 34   # mean over 34 tiles
    loss_init_dir  = math.log(2)                # BCE at logit=0

    # 正例率：某决策点至少有一张危险牌
    decisions_with_risk = sum(1 for v in agg['dir_sum'] if v > 0)
    # 更准确：决策点中至少一个 dir_vec[i]>0 的比例
    # 直接统计 mean(dir_sum_i) / N 的最大值作为代理
    max_tile_mean = max(tile_means)
    mean_nonzero_tiles = sum(1 for m in tile_means if m > 1e-6)

    SUIT_NAMES = ['万1', '万2', '万3', '万4', '万5', '万6', '万7', '万8', '万9',
                  '筒1', '筒2', '筒3', '筒4', '筒5', '筒6', '筒7', '筒8', '筒9',
                  '索1', '索2', '索3', '索4', '索5', '索6', '索7', '索8', '索9',
                  '東', '南', '西', '北', '白', '發', '中']

    print('\n[3] deal_in_risk  （34-dim binary BCE，Label B）')
    print(f'    label 均值（各牌危险度 μ）:')
    print('      牌    |  万子                              |  筒子                              |  索子                              |  字牌      ')
    print('      μ     | ' + ' '.join(f'{tile_means[i]:.4f}' for i in range(9)) +
          ' | ' + ' '.join(f'{tile_means[i]:.4f}' for i in range(9, 18)) +
          ' | ' + ' '.join(f'{tile_means[i]:.4f}' for i in range(18, 27)) +
          ' | ' + ' '.join(f'{tile_means[i]:.4f}' for i in range(27, 34)))
    print(f'    全局均值 mean(μ) over 34 tiles = {sum(tile_means)/34:.6f}')
    print(f'    非零 tile 数                   = {mean_nonzero_tiles}/34')
    print(f'    最大单牌危险度 max(μ)           = {max_tile_mean:.4f}  ({SUIT_NAMES[tile_means.index(max_tile_mean)]})')
    print(f'    baseline loss  mean(H(μ_tile)) = {h_dir:.4f} nats')
    print(f'    random-init loss log2          = {loss_init_dir:.4f} nats')

    # ──────────────────────────────────────────────────────────────────────
    # 4. opp_tenpai（3-dim binary BCE，立直 proxy）
    # ──────────────────────────────────────────────────────────────────────
    opp_rates     = [agg['opp_riichi_pos'][i] / max(N, 1) for i in range(3)]
    opp_entropies = [_H(r) for r in opp_rates]
    h_opp         = sum(opp_entropies) / 3
    loss_init_opp = math.log(2)

    print('\n[4] opp_tenpai  （3-dim binary BCE，立直 proxy）')
    print(f'    各座位立直率 = {[f"{r:.4f}" for r in opp_rates]}')
    print(f'    mean H(p_seat) = {h_opp:.4f} nats')
    print(f'    random-init loss log2  = {loss_init_opp:.4f} nats')
    print(f'    当前 weight=0.10  → contribution (baseline) = {0.10*h_opp:.4f}')

    # ──────────────────────────────────────────────────────────────────────
    # 权重推算：让 deal_in_risk 的 baseline contribution 与其他三项均值对齐
    # ──────────────────────────────────────────────────────────────────────
    ref_contributions = [
        0.10 * h_shanten,
        0.10 * h_tenpai,
        0.10 * h_opp,
    ]
    target = sum(ref_contributions) / len(ref_contributions)

    w_raw   = target / max(h_dir, 1e-12)
    # 四舍五入到 2 位小数
    w_final = round(w_raw, 2)

    # 同样计算「随机初始化」基准下的推算值（仅供参考）
    ref_init = [0.10 * loss_init_sh, 0.10 * loss_init_tp, 0.10 * loss_init_opp]
    target_init = sum(ref_init) / len(ref_init)
    w_init = round(target_init / loss_init_dir, 2)

    print(f'\n{SEP}')
    print('  AUX_WEIGHTS["deal_in_risk"] 推算结果')
    print(SEP)
    print(f'  参照任务贡献（baseline）:')
    print(f'    shanten    0.10 × {h_shanten:.4f} = {ref_contributions[0]:.4f}')
    print(f'    tenpai     0.10 × {h_tenpai:.4f} = {ref_contributions[1]:.4f}')
    print(f'    opp_tenpai 0.10 × {h_opp:.4f} = {ref_contributions[2]:.4f}')
    print(f'  目标贡献（三项均值）     = {target:.4f}')
    print(f'  deal_in_risk baseline loss = {h_dir:.4f}')
    print(f'  → 推算 weight = {target:.4f} / {h_dir:.4f} = {w_raw:.4f}')
    print(f'')
    print(f'  ╔══════════════════════════════════════════════════════════╗')
    print(f'  ║  建议值（baseline 对齐）:  AUX_WEIGHTS["deal_in_risk"] = {w_final:<6} ║')
    print(f'  ║  参考值（random-init 对齐）:                          = {w_init:<6} ║')
    print(f'  ╚══════════════════════════════════════════════════════════╝')
    print(f'\n  注：baseline 对齐确保训练稳定后各任务梯度量级均衡；')
    print(f'      random-init 对齐确保训练初期不被 deal_in_risk 主导。')
    print(f'      若两值相差较大，建议取两者均值或偏小值作为保守起点。')
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='统计 aux 任务 label 分布，推算 deal_in_risk 的合理 AUX_WEIGHT'
    )
    parser.add_argument('--input',   '-i', required=True,
                        help='mjai_json 目录（含 .jsonl 文件）')
    parser.add_argument('--workers', '-w', type=int, default=22,
                        help='并行 worker 数（默认 22）')
    parser.add_argument('--limit',   '-n', type=int, default=0,
                        help='最多处理文件数，0=全部')
    args = parser.parse_args()

    in_dir    = Path(args.input)
    all_files = sorted(in_dir.rglob('*.jsonl'))
    if args.limit > 0:
        all_files = all_files[:args.limit]

    print(f'输入目录 : {in_dir}')
    print(f'文件总数 : {len(all_files):,}')
    print(f'Worker 数: {args.workers}')
    print(f'开始处理...\n')

    t0 = time.time()
    results: list[dict | None] = []

    with mp.get_context('fork').Pool(args.workers) as pool:
        done = 0
        for r in pool.imap_unordered(_process_file, all_files, chunksize=8):
            results.append(r)
            done += 1
            if done % 200 == 0 or done == len(all_files):
                elapsed = time.time() - t0
                speed   = done / elapsed
                eta     = (len(all_files) - done) / max(speed, 1e-6)
                print(f'  {done:>6}/{len(all_files)}  '
                      f'elapsed={elapsed:.0f}s  '
                      f'speed={speed:.1f} files/s  '
                      f'ETA={eta:.0f}s',
                      flush=True)

    elapsed = time.time() - t0
    print(f'\n全部处理完毕，耗时 {elapsed:.1f}s')

    valid = [r for r in results if r is not None]
    print(f'有效文件 : {len(valid)}/{len(results)}')

    compute_and_report(valid)


if __name__ == '__main__':
    mp.freeze_support()   # Windows 兼容
    main()
