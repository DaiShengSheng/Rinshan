"""
migrate_stage1_weights.py
把 stage1_best.pt（旧 QV Head 结构）迁移到新结构，并修复 V 值量级问题。

处理策略：
  1. 键名重映射
       qv_head.v_net.*  →  qv_head.v_game_net.*
       qv_head.a_net.*  →  qv_head.a_game_net.*

  2. v_game_net 最后一层归零（stage1 的 V 值被训到 -7 万量级，无法用于 RL）
       qv_head.v_game_net.3.weight  →  zeros
       qv_head.v_game_net.3.bias    →  zeros
     特征提取层 (.0.*) 保留，让 Stage2 在正确 basis 上快速重学 V。

  3. a_game_net 完整保留（76% top-1 acc，是 Stage1 真正学到的弃牌策略）

  4. v_hand_net / a_hand_net 按正确形状填入标准初始化
     （第一层 kaiming_uniform，最后一层 zeros），消除 missing keys。

用法：
  python scripts/migrate_stage1_weights.py \
      --src checkpoints/stage1_best.pt \
      --dst checkpoints/stage1_migrated.pt \
      --preset base
"""
import argparse
import math
import torch
import torch.nn as nn


# QVHead 各 head 的形状：Sequential(Linear(dim, hidden), SiLU, Dropout, Linear(hidden, 1))
# base: dim=768, hidden=384
PRESET_DIMS = {
    "base": (768, 384),
    "nano": (256, 128),
    "small": (512, 256),
}


def _make_linear_weights(in_f: int, out_f: int, zero_last: bool = False):
    """生成一对 (weight, bias)，zero_last=True 时全零，否则 kaiming_uniform"""
    w = torch.zeros(out_f, in_f)
    b = torch.zeros(out_f)
    if not zero_last:
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        fan_in = in_f
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(b, -bound, bound)
    return w, b


def migrate(src_path: str, dst_path: str, preset: str = "base") -> None:
    dim, hidden = PRESET_DIMS.get(preset, PRESET_DIMS["base"])

    ckpt  = torch.load(src_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    new_state = {}
    log_remap  = []
    log_reset  = []
    log_add    = []

    # ── 1. 键名重映射 ──────────────────────────────────────────────
    for k, v in state.items():
        clean_k = k.replace("_orig_mod.", "")   # 去掉 torch.compile 前缀

        if "qv_head.v_net." in clean_k:
            new_k = clean_k.replace("qv_head.v_net.", "qv_head.v_game_net.")
            new_state[new_k] = v.clone()
            log_remap.append(f"  {clean_k}  →  {new_k}")
        elif "qv_head.a_net." in clean_k:
            new_k = clean_k.replace("qv_head.a_net.", "qv_head.a_game_net.")
            new_state[new_k] = v.clone()
            log_remap.append(f"  {clean_k}  →  {new_k}")
        else:
            new_state[clean_k] = v.clone()

    # ── 2. v_game_net 最后一层归零（消除 -7 万 V 值量级污染）──────
    for suffix in ("qv_head.v_game_net.3.weight", "qv_head.v_game_net.3.bias"):
        if suffix in new_state:
            old_max = new_state[suffix].abs().max().item()
            new_state[suffix] = torch.zeros_like(new_state[suffix])
            log_reset.append(f"  {suffix}  (旧 max_abs={old_max:.4f}  →  0)")

    # ── 3. 补全缺失的 v_hand_net / a_hand_net ─────────────────────
    missing_heads = {
        "qv_head.v_hand_net": (dim, hidden),
        "qv_head.a_hand_net": (dim, hidden),
    }
    for head_prefix, (d, h) in missing_heads.items():
        # 第一层：kaiming init
        w0, b0 = _make_linear_weights(d, h, zero_last=False)
        # 最后一层：零初始化（与 QVHead._init_weights 一致）
        w3, b3 = _make_linear_weights(h, 1, zero_last=True)

        new_state[f"{head_prefix}.0.weight"] = w0
        new_state[f"{head_prefix}.0.bias"]   = b0
        new_state[f"{head_prefix}.3.weight"] = w3
        new_state[f"{head_prefix}.3.bias"]   = b3
        log_add.append(f"  {head_prefix}.{{0,3}}.{{weight,bias}}  (新增零初始化)")

    # ── 打印摘要 ──────────────────────────────────────────────────
    print(f"[migrate] 键名重映射（{len(log_remap)} 个）：")
    for m in log_remap: print(m)

    print(f"\n[migrate] V head 最后一层归零（{len(log_reset)} 个）：")
    for m in log_reset: print(m)

    print(f"\n[migrate] 新增缺失 head（{len(log_add)} 个）：")
    for m in log_add: print(m)

    # ── 保存 ──────────────────────────────────────────────────────
    # 统一使用 "model" 键名，与 train_stage2.py / train_stage3.py 期望一致
    new_ckpt = {k: v for k, v in ckpt.items()
                if k not in ("model_state_dict", "model")}
    new_ckpt["model"] = new_state
    torch.save(new_ckpt, dst_path)
    print(f"\n[migrate] 已保存 → {dst_path}")

    # ── 快速验证 ──────────────────────────────────────────────────
    print("\n[validate]")
    # 确认键名是 "model"
    assert "model" in new_ckpt, "BUG: checkpoint key should be 'model'"
    checks = {
        "a_game_net.3.weight  (应非零，保留弃牌策略)": "qv_head.a_game_net.3.weight",
        "v_game_net.3.weight  (应为零，V 值重学)":     "qv_head.v_game_net.3.weight",
        "v_hand_net.3.weight  (应为零，新增)":          "qv_head.v_hand_net.3.weight",
        "a_hand_net.3.weight  (应为零，新增)":          "qv_head.a_hand_net.3.weight",
    }
    for desc, key in checks.items():
        val = new_state.get(key)
        if val is not None:
            print(f"  {desc}  →  max_abs={val.abs().max().item():.6f}")
        else:
            print(f"  {desc}  →  !! 找不到键 !!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",    default="checkpoints/stage1_best.pt")
    parser.add_argument("--dst",    default="checkpoints/stage1_migrated.pt")
    parser.add_argument("--preset", default="base",
                        choices=list(PRESET_DIMS.keys()),
                        help="模型规模，用于确定 v_hand/a_hand 的形状")
    args = parser.parse_args()
    migrate(args.src, args.dst, args.preset)
