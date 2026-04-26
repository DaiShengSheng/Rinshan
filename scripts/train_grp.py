"""
train_grp.py — 训练 GRP（局结果预测器）

GRP 是整个奖励体系的基础，必须最先训练。
训练数据：从 .jsonl 标注文件中提取游戏进程特征序列。

Usage:
  python scripts/train_grp.py \\
      --data    data/annotated/   \\
      --save    checkpoints/grp/  \\
      --epochs  30                \\
      --batch   256               \\
      --lr      1e-3
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from rinshan.model.grp import GRP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 从 .jsonl 提取 GRP 训练数据
# ─────────────────────────────────────────────

def _parse_one_file(path: Path) -> dict:
    """
    子进程任务：解析单个 .jsonl 文件，返回该文件内所有游戏的局部数据。
    返回值: {game_id: {"rounds": {...}, "final_scores": [...], "final_player_id": int}}
    """
    local_games: dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue

                gid = d.get("game_id", "?")
                if gid not in local_games:
                    local_games[gid] = {"rounds": {}}

                rw = d.get("round_wind", 0)
                rn = d.get("round_num",  1)
                key = (rw, rn)

                if key not in local_games[gid]["rounds"]:
                    honba   = d.get("honba",   0)
                    kyotaku = d.get("kyotaku", 0)
                    scores  = d.get("scores",  [250]*4)
                    grand_kyoku = rw * 4 + rn - 1
                    local_games[gid]["rounds"][key] = {
                        "grand_kyoku": grand_kyoku,
                        "honba":       honba,
                        "kyotaku":     kyotaku,
                        "scores":      scores,
                        "player_id":   d.get("player_id", 0),
                    }

                if d.get("is_done", False):
                    local_games[gid]["final_scores"]    = d.get("scores", [250]*4)
                    local_games[gid]["final_player_id"] = d.get("player_id", 0)
    except Exception:
        pass
    return local_games


def build_grp_dataset(jsonl_dir: Path, limit: int = None, n_workers: int = 8):
    """
    从标注文件提取每局游戏的进程特征序列 + 最终排名

    GRP 输入（每局一帧，7维）：
      [grand_kyoku, honba, kyotaku, score_p0/1e4, s_p1/1e4, s_p2/1e4, s_p3/1e4]

    返回: list of (frames_tensor, rank_by_player_tensor)
    """
    # ── cache：避免重复 parse 几千个 jsonl ──────────────────────────
    cache_path = jsonl_dir / "_grp_cache.pt"
    files = sorted(jsonl_dir.rglob("*.jsonl"))
    if limit:
        files = files[:limit]

    if not limit and cache_path.exists():
        cache_mtime  = cache_path.stat().st_mtime
        newest_jsonl = max(f.stat().st_mtime for f in files) if files else 0
        if cache_mtime >= newest_jsonl:
            logger.info(f"Loading GRP dataset from cache: {cache_path}")
            data = torch.load(cache_path, weights_only=True)
            logger.info(f"  {len(data['seqs']):,} games loaded from cache")
            return data["seqs"], data["ranks"]
        else:
            logger.info("Cache is stale (new jsonl files detected), rebuilding...")

    games: dict[str, dict] = {}   # game_id → {rounds: {...}, final_scores: [...], ...}

    logger.info(f"Reading {len(files)} .jsonl files with {n_workers} workers...")

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_parse_one_file, p): p for p in files}
        for fut in as_completed(futures):
            try:
                local_games = fut.result()
            except Exception as e:
                logger.warning(f"Failed {futures[fut]}: {e}")
                local_games = {}

            # 合并到全局 games dict
            for gid, gdata in local_games.items():
                if gid not in games:
                    games[gid] = {"rounds": {}}
                # 合并 rounds（保留每局第一条，先到先得）
                for key, rinfo in gdata["rounds"].items():
                    if key not in games[gid]["rounds"]:
                        games[gid]["rounds"][key] = rinfo
                # 合并 final_scores（有就写，后写覆盖前写，同一游戏多玩家各写一次无所谓）
                if "final_scores" in gdata:
                    games[gid]["final_scores"]    = gdata["final_scores"]
                    games[gid]["final_player_id"] = gdata["final_player_id"]

            completed += 1
            if completed % 100 == 0 or completed == len(files):
                logger.info(f"  [{completed}/{len(files)}] {len(games):,} games merged so far")

    # 转换为张量列表
    all_seqs    = []   # list of (T, 7) Tensor
    all_ranks   = []   # list of (4,) Tensor

    for gid, gdata in games.items():
        rounds = sorted(gdata["rounds"].items(), key=lambda x: x[1]["grand_kyoku"])
        if not rounds:
            continue

        frames = []
        for (rw, rn), info in rounds:
            player_id = info["player_id"]
            scores = info["scores"]

            # 反旋转分数到绝对顺序
            abs_scores = scores[4-player_id:] + scores[:4-player_id]

            frame = [
                float(info["grand_kyoku"]),
                float(info["honba"]),
                float(info["kyotaku"]),
                abs_scores[0] / 1e4,
                abs_scores[1] / 1e4,
                abs_scores[2] / 1e4,
                abs_scores[3] / 1e4,
            ]
            frames.append(frame)

        if len(frames) < 2:
            continue

        # 从最终分数计算最终排名
        # 没有 is_done 记录的游戏直接跳过，避免用错误数据污染标签
        if "final_scores" not in gdata:
            continue
        final_s   = gdata["final_scores"]
        final_pid = gdata["final_player_id"]
        # 反旋转：abs[k] = rotated[(k - pid) % 4] = rotated[(4-pid+k) % 4]
        abs_final = final_s[4-final_pid:] + final_s[:4-final_pid]
        # 平局时按起家顺序（seat 0 优先），与天凤/雀魂规则一致
        order = sorted(range(4), key=lambda i: (-abs_final[i], i))
        rank_by_player = [0]*4
        for rank, seat in enumerate(order):
            rank_by_player[seat] = rank

        all_seqs.append(torch.tensor(frames, dtype=torch.float64))
        all_ranks.append(torch.tensor(rank_by_player, dtype=torch.long))

    logger.info(f"Extracted {len(all_seqs)} games for GRP training")

    # 写 cache（--limit 模式下不写，避免缓存不完整数据）
    if not limit and all_seqs:
        logger.info(f"Saving cache to {cache_path} ...")
        torch.save({"seqs": all_seqs, "ranks": all_ranks}, cache_path)
        logger.info("Cache saved.")

    return all_seqs, all_ranks


class GRPDataset(Dataset):
    def __init__(self, seqs, ranks):
        self.seqs  = seqs
        self.ranks = ranks

    def __len__(self):  return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.ranks[i]


def grp_collate(batch):
    seqs, ranks = zip(*batch)
    return list(seqs), torch.stack(ranks)


# ─────────────────────────────────────────────
# 训练循环
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 加载数据
    seqs, ranks = build_grp_dataset(Path(args.data), limit=args.limit, n_workers=args.fill_workers)
    if not seqs:
        logger.error("No training data found. Run parse_tenhou.py first.")
        return

    # 分割训练/验证
    n = len(seqs)
    val_n = max(1, int(n * 0.05))
    idx   = list(range(n))
    random.shuffle(idx)
    train_idx = idx[val_n:]
    val_idx   = idx[:val_n]

    train_ds = GRPDataset([seqs[i] for i in train_idx], [ranks[i] for i in train_idx])
    val_ds   = GRPDataset([seqs[i] for i in val_idx],   [ranks[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=grp_collate, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=grp_collate, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers > 0))

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # 模型
    grp = GRP().to(device)
    grp_double = grp  # GRP 内部使用 float64

    optimizer = AdamW(grp.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        grp.train()
        train_loss = 0.0
        for seqs_b, ranks_b in train_loader:
            seqs_b  = [s.to(device) for s in seqs_b]
            ranks_b = ranks_b.to(device)

            optimizer.zero_grad()
            logits = grp(seqs_b)
            loss   = grp.compute_loss(logits, ranks_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(grp.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        grp.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for seqs_b, ranks_b in val_loader:
                seqs_b  = [s.to(device) for s in seqs_b]
                ranks_b = ranks_b.to(device)
                logits  = grp(seqs_b)
                loss    = grp.compute_loss(logits, ranks_b)
                val_loss += loss.item()
                pred = logits.argmax(-1)
                labels = grp.get_label(ranks_b)
                correct += (pred == labels).sum().item()
                total   += ranks_b.shape[0]

        val_loss /= len(val_loader)
        acc = correct / total if total > 0 else 0.0

        scheduler.step()
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | acc={acc:.3f}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": grp.state_dict(), "epoch": epoch}, save_dir / "grp_best.pt")
            logger.info(f"  ↑ Best model saved (val_loss={best_val_loss:.4f})")

    # 保存最终
    torch.save({"model": grp.state_dict(), "epoch": args.epochs}, save_dir / "grp_final.pt")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="可选 yaml 配置文件")
    parser.add_argument("--data",   "-d", default=None, help=".jsonl 标注目录")
    parser.add_argument("--save",   "-s", default=None)
    parser.add_argument("--epochs", "-e", type=int, default=None)
    parser.add_argument("--batch",  "-b", type=int, default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--limit",  "-n", type=int, default=None)
    parser.add_argument("--workers",   "-j", type=int, default=None, help="DataLoader num_workers")
    parser.add_argument("--fill-workers", type=int, default=None, help="并行读取 jsonl 的进程数")
    args = parser.parse_args()

    # 读 yaml（命令行参数优先级更高）
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if args.data   is None: args.data   = cfg.get("data_dir",   "data/annotated")
        if args.save   is None: args.save   = cfg.get("save_dir",   "checkpoints/grp")
        if args.epochs is None: args.epochs = int(cfg.get("epochs",     30))
        if args.batch  is None: args.batch  = int(cfg.get("batch_size", 256))
        if args.lr     is None: args.lr     = float(cfg.get("lr",       1e-3))
        if args.workers      is None: args.workers      = int(cfg.get("dataloader_workers", 0))
        if args.fill_workers is None: args.fill_workers = int(cfg.get("fill_workers", 8))
    else:
        if args.data         is None: parser.error("--data/-d is required (or pass a yaml config)")
        if args.save         is None: args.save         = "checkpoints/grp"
        if args.epochs       is None: args.epochs       = 30
        if args.batch        is None: args.batch        = 256
        if args.lr           is None: args.lr           = 1e-3
        if args.workers      is None: args.workers      = 0
        if args.fill_workers is None: args.fill_workers = 8

    train(args)


if __name__ == "__main__":
    main()
