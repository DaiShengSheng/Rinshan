"""
用法: python scripts/inspect_checkpoint.py <checkpoint路径>
例如: python scripts/inspect_checkpoint.py checkpoints/stage1_base/best.pt
"""
import sys
import torch
from pathlib import Path


def inspect(path: str):
    path = Path(path)
    print(f"\n{'='*50}")
    print(f"文件: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    print('='*50)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    print(f"\n包含的 key: {list(ckpt.keys())}")

    # 训练状态
    if "step" in ckpt:
        print(f"\n训练进度: step = {ckpt['step']:,}")
    if "val_loss" in ckpt:
        print(f"最佳 val_loss: {ckpt['val_loss']:.4f}")

    # 模型参数量
    if "model" in ckpt:
        sd = ckpt["model"]
        total_params = sum(v.numel() for v in sd.values())
        print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"模型层数:   {len(sd)} tensors")
        print("\n各层 shape（前10层）:")
        for i, (k, v) in enumerate(sd.items()):
            print(f"  {k:50s} {str(list(v.shape)):20s} {v.dtype}")
            if i >= 9:
                print("  ...")
                break

    # LR
    if "scheduler" in ckpt:
        sc = ckpt["scheduler"]
        if "_last_lr" in sc:
            print(f"\n最后 lr: {sc['_last_lr']}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/inspect_checkpoint.py <checkpoint路径>")
        sys.exit(1)
    inspect(sys.argv[1])
