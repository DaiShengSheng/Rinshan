"""
migrate_vocab_score.py — 将旧 checkpoint（VOCAB_SIZE=1548）的 token embedding
扩容到新词表大小（VOCAB_SIZE=1612），新增的 64 个分数 token 用小随机值初始化。

同时处理两侧 token_embed：
  - transformer.token_embed.weight : (1548, 768) → (1612, 768)
  - belief_net.token_embed.weight  : (1548, 256) → (1612, 256)

Usage:
    python scripts/migrate_vocab_score.py \
        --src  checkpoints/stage2_best_50000_v3.pt \
        --dst  checkpoints/stage2_best_50000_v4.pt

注意：只修改 token_embed.weight，其余参数原样保留。
新的 Stage2 训练（起点已是 VOCAB_SIZE=1612 的 Stage1 权重）不需要运行此脚本。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


OLD_VOCAB = 1548
NEW_VOCAB = 1612  # +64 分数 token（4 seats × 16 bins），SCORE_OFFSET=1548

# 需要扩容的所有 token_embed key（带/不带 _orig_mod. 前缀均兼容）
_EMBED_KEYS = [
    "transformer.token_embed.weight",
    "belief_net.token_embed.weight",
]


def _expand_embed(sd: dict, key: str, label: str) -> bool:
    """在 state_dict sd 中扩容指定 key。返回是否实际做了修改。"""
    # 兼容 _orig_mod. 前缀（torch.compile 保存格式）
    real_key = key
    if real_key not in sd:
        real_key = f"_orig_mod.{key}"
        if real_key not in sd:
            print(f"  [{label}] key 不存在，跳过：{key}")
            return False

    old_w = sd[real_key]
    old_v, dim = old_w.shape

    if old_v == NEW_VOCAB:
        print(f"  [{label}] {real_key}: 已是新词表 {NEW_VOCAB}，无需迁移。")
        return False

    if old_v != OLD_VOCAB:
        raise ValueError(
            f"[{label}] {real_key}: 期望旧词表大小 {OLD_VOCAB}，实际 {old_v}"
        )

    new_rows = torch.randn(NEW_VOCAB - OLD_VOCAB, dim, dtype=old_w.dtype) * 0.02
    sd[real_key] = torch.cat([old_w, new_rows], dim=0)
    print(f"  [{label}] {real_key}: {tuple(old_w.shape)} → {tuple(sd[real_key].shape)}")
    return True


def migrate(src: Path, dst: Path) -> None:
    print(f"Loading  {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=True)

    model_sd   = ckpt.get("model") or ckpt
    target_sd  = ckpt.get("target_model")

    any_changed = False
    for key in _EMBED_KEYS:
        any_changed |= _expand_embed(model_sd, key, "model")
        if target_sd is not None:
            _expand_embed(target_sd, key, "target")

    if not any_changed:
        print("所有 embedding 已是新版本，无需迁移，原样保存。")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"Saved → {dst}")

    # 最终验证
    print("\n── 验证 ──")
    ckpt2 = torch.load(dst, map_location="cpu", weights_only=True)
    sd2   = ckpt2.get("model") or ckpt2
    for k in sorted(k for k in sd2.keys() if "embed" in k):
        print(f"  {k}: {tuple(sd2[k].shape)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    migrate(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
