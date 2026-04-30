"""
migrate_vocab_tsumogiri.py
将旧词表（VOCAB_SIZE=1360）的 checkpoint 扩展为新词表（VOCAB_SIZE=1508）。

新增的 148 个 token [1360-1507] 是摸切版 PROG_DISCARD，
初始 embedding weight = 对应手切 token 的 weight + 微小随机噪声。

用法:
    python scripts/migrate_vocab_tsumogiri.py \
        --src checkpoints/stage1_best.pt \
        --dst checkpoints/stage1_migrated_tsumogiri.pt
"""
import argparse
import torch

OLD_VOCAB = 1360
NEW_VOCAB = 1508
HAND_CUT_BASE  = 513   # PROG_DISCARD_BASE（手切）
TSUMO_CUT_BASE = 1360  # PROG_DISCARD_TSUMOGIRI_BASE（摸切）
DISCARD_TOKENS = 148   # 4 seats × 37 tiles

NOISE_STD = 1e-4       # 摸切 embedding 初始噪声量级


def migrate(src_path: str, dst_path: str) -> None:
    print(f"[load] {src_path}")
    ckpt = torch.load(src_path, map_location="cpu")

    # checkpoint 可能直接是 state_dict，也可能是包含 model_state_dict 的 dict
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    patched = 0
    new_state = {}
    for key, tensor in state.items():
        if "token_embed.weight" in key:
            old_w = tensor  # (OLD_VOCAB, dim)
            assert old_w.shape[0] == OLD_VOCAB, \
                f"{key}: 期望 {OLD_VOCAB} 行，实际 {old_w.shape[0]} 行，可能已经 migrate 过了"

            dim = old_w.shape[1]
            new_w = torch.zeros(NEW_VOCAB, dim, dtype=old_w.dtype)

            # 旧词表全部原样复制
            new_w[:OLD_VOCAB] = old_w

            # 摸切 token [1360-1507]：用手切 weight 做热启动 + 微小噪声
            hand_cut_weights = old_w[HAND_CUT_BASE: HAND_CUT_BASE + DISCARD_TOKENS]
            noise = torch.randn_like(hand_cut_weights) * NOISE_STD
            new_w[TSUMO_CUT_BASE: TSUMO_CUT_BASE + DISCARD_TOKENS] = hand_cut_weights + noise

            new_state[key] = new_w
            patched += 1
            print(f"  [patch] {key}: {old_w.shape} -> {new_w.shape}")
        else:
            new_state[key] = tensor

    assert patched > 0, "没有找到任何 token_embed.weight，请检查 checkpoint 格式"

    # 写回 checkpoint（保留 optimizer / step 等其他字段）
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "model" in ckpt):
        out_ckpt = dict(ckpt)
        if "model_state_dict" in ckpt:
            out_ckpt["model_state_dict"] = new_state
        else:
            out_ckpt["model"] = new_state
        # optimizer state 里的 param 数量不变（token_embed 只改了 size），
        # 但 optimizer 里存的是 param 引用，load 新模型后 optimizer 会自动重建，
        # 这里把旧 optimizer state 丢掉，从 stage 续训时会重新初始化。
        out_ckpt.pop("optimizer_state_dict", None)
        out_ckpt.pop("optimizer", None)
        out_ckpt.pop("scaler_state_dict", None)
        out_ckpt["step"] = out_ckpt.get("step", 0)  # 保留训练步数（供参考）
    else:
        out_ckpt = new_state

    torch.save(out_ckpt, dst_path)
    print(f"[save] {dst_path}  (vocab {OLD_VOCAB} -> {NEW_VOCAB})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="旧 checkpoint 路径")
    parser.add_argument("--dst", required=True, help="输出新 checkpoint 路径")
    args = parser.parse_args()
    migrate(args.src, args.dst)
