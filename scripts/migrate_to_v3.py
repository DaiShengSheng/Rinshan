"""
migrate_to_v3.py
把 stage1_best_30000_v2.pt（旧架构）迁移到 v3 架构：
  1. belief_net.token_embed: (1508,256) → (1548,256)  零初始化新40行
  2. belief_net.pos_embed:   (134, 256) → (146, 256)  零初始化新12行
  3. belief_net.wait_head:   不存在 → 不注入（由 strict=False 随机初始化）
  4. 其余所有权重原样保留

用法：
    python scripts/migrate_to_v3.py \\
        --src checkpoints/stage1_best_30000_v2.pt \\
        --dst checkpoints/stage1_best_30000_v3.pt
"""
import argparse, torch

OLD_VOCAB = 1508
NEW_VOCAB = 1548   # +40: RIICHI_JUNME(36) + RIICHI_FURITEN(4)

OLD_POS = 134      # max_seq_len=133 +1
NEW_POS = 146      # max_seq_len=145 +1


def migrate(src: str, dst: str):
    print(f"loading {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]

    # ── 1. transformer.token_embed (1508,768) → (1548,768) ─────────────
    t_te_key = "_orig_mod.transformer.token_embed.weight"
    old_t_te = sd[t_te_key]
    new_t_te = torch.zeros(NEW_VOCAB, old_t_te.shape[1], dtype=old_t_te.dtype)
    new_t_te[:OLD_VOCAB] = old_t_te
    sd[t_te_key] = new_t_te
    print(f"  transformer.token_embed: {tuple(old_t_te.shape)} → {tuple(new_t_te.shape)}")

    # ── 2. belief_net.token_embed (1508,256) → (1548,256) ───────────────
    b_te_key = "_orig_mod.belief_net.token_embed.weight"
    old_b_te = sd[b_te_key]
    new_b_te = torch.zeros(NEW_VOCAB, old_b_te.shape[1], dtype=old_b_te.dtype)
    new_b_te[:OLD_VOCAB] = old_b_te
    sd[b_te_key] = new_b_te
    print(f"  belief_net.token_embed:  {tuple(old_b_te.shape)} → {tuple(new_b_te.shape)}")

    # ── 3. belief_net.pos_embed (134,256) → (146,256) ───────────────────
    pe_key = "_orig_mod.belief_net.pos_embed.weight"
    old_pe = sd[pe_key]
    new_pe = torch.zeros(NEW_POS, old_pe.shape[1], dtype=old_pe.dtype)
    new_pe[:OLD_POS] = old_pe
    # 新12行用最后一行外推，比零初始化更稳定
    new_pe[OLD_POS:] = old_pe[-1:].expand(NEW_POS - OLD_POS, -1)
    sd[pe_key] = new_pe
    print(f"  belief_net.pos_embed:    {tuple(old_pe.shape)} → {tuple(new_pe.shape)}")

    # ── 4. wait_head 不注入，交给 strict=False 随机初始化 ────────────────
    print("  wait_head:               not injected → randomly initialized on load")

    ckpt["model"] = sd
    torch.save(ckpt, dst)
    print(f"saved → {dst}")

    # ── 验证 ─────────────────────────────────────────────────────────────
    from rinshan.model.full_model import RinshanModel
    from rinshan.model.transformer import TransformerConfig
    from rinshan.constants import MODEL_CONFIGS
    cfg   = TransformerConfig(**MODEL_CONFIGS["base"])
    model = RinshanModel(transformer_cfg=cfg, use_belief=True, use_aux=True)
    # 剥 _orig_mod. 前缀（与 train_stage2.py 保持一致）
    clean_sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    wait_missing  = [k for k in missing   if "wait_head" in k]
    other_missing = [k for k in missing   if "wait_head" not in k]
    print(f"\n  wait_head missing (expected 4): {len(wait_missing)}")
    print(f"  other missing    (should be 0): {len(other_missing)}")
    if other_missing:
        for k in other_missing[:5]: print(f"    {k}")
    print(f"  unexpected       (should be 0): {len(unexpected)}")

    be = dict(model.named_parameters())
    print(f"\n  transformer.token_embed: {tuple(be['transformer.token_embed.weight'].shape)}")
    print(f"  belief_net.token_embed:  {tuple(be['belief_net.token_embed.weight'].shape)}")
    print(f"  belief_net.pos_embed:    {tuple(be['belief_net.pos_embed.weight'].shape)}")
    wait_std   = round(be["belief_net.wait_head.0.weight"].std().item(), 4)
    belief_std = round(be["belief_net.belief_head.0.weight"].std().item(), 4)
    print(f"\n  wait_head std   (random init): {wait_std}")
    print(f"  belief_head std (pretrained):  {belief_std}")
    assert len(other_missing) == 0, "unexpected missing keys!"
    print("\nMigration OK ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="checkpoints/stage1_best_30000_v2.pt")
    p.add_argument("--dst", default="checkpoints/stage1_best_30000_v3.pt")
    args = p.parse_args()
    migrate(args.src, args.dst)
