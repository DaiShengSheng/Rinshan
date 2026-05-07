"""
migrate_vocab_score.py — 将旧 checkpoint（VOCAB_SIZE=1548）的 token embedding
扩容到新词表大小（VOCAB_SIZE=1612），新增的 64 个分数 token 用小随机值初始化。

Usage:
    python scripts/migrate_vocab_score.py \
        --src  /root/autodl-tmp/rinshan/checkpoints/stage2_base/best.pt \
        --dst  /root/autodl-tmp/rinshan/checkpoints/stage2_base/best_v4.pt

注意：只修改 token_embed.weight，其余参数原样保留。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


OLD_VOCAB = 1548
NEW_VOCAB = 1612  # +64 分数 token（4 seats × 16 bins），SCORE_OFFSET=1548


def migrate(src: Path, dst: Path) -> None:
    print(f"Loading  {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=True)

    model_sd = ckpt.get("model") or ckpt  # 兼容裸权重格式
    key = "transformer.token_embed.weight"

    if key not in model_sd:
        # 可能带 _orig_mod. 前缀
        key_orig = f"_orig_mod.{key}"
        if key_orig in model_sd:
            key = key_orig
        else:
            raise KeyError(f"找不到 token embedding key，已有 keys 前缀：{list(model_sd.keys())[:10]}")

    old_w = model_sd[key]          # (OLD_VOCAB, dim)
    old_v, dim = old_w.shape
    print(f"Old embedding shape: {old_w.shape}")

    if old_v == NEW_VOCAB:
        print("词表已是新版本，无需迁移。")
        torch.save(ckpt, dst)
        return

    if old_v != OLD_VOCAB:
        raise ValueError(f"期望旧词表大小 {OLD_VOCAB}，实际 {old_v}")

    # 新增的 64 个 embedding，用 N(0, 0.02) 初始化（与 nn.Embedding 默认一致）
    new_rows = torch.randn(NEW_VOCAB - OLD_VOCAB, dim, dtype=old_w.dtype) * 0.02
    new_w = torch.cat([old_w, new_rows], dim=0)   # (NEW_VOCAB, dim)
    print(f"New embedding shape: {new_w.shape}")

    model_sd[key] = new_w

    # 如果 target_model 也有同名 key，一并迁移
    target_sd = ckpt.get("target_model")
    if target_sd is not None and key in target_sd:
        old_t = target_sd[key]
        new_rows_t = torch.randn(NEW_VOCAB - OLD_VOCAB, dim, dtype=old_t.dtype) * 0.02
        target_sd[key] = torch.cat([old_t, new_rows_t], dim=0)
        print("target_model embedding 同步扩容。")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"Saved → {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    migrate(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
