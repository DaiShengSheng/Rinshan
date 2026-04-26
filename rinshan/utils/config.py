"""
简单 YAML 配置加载，不依赖 Hydra/OmegaConf
支持从 CLI 覆盖: --lr 1e-4 --batch_size 32
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any


def load_config(config_path: str, extra_args: list[str] = None) -> dict:
    """
    加载 YAML 配置文件，并可通过 CLI 参数覆盖

    Usage:
        cfg = load_config("configs/stage1_base.yaml", sys.argv[1:])
        cfg = load_config("configs/stage1_base.yaml", ["--lr", "1e-4"])
    """
    try:
        import yaml
    except ImportError:
        # 简单的 YAML 解析（仅支持 key: value 格式）
        return _simple_yaml_load(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        # 跳过注释行后 load
        cfg = yaml.safe_load(f) or {}

    # CLI 参数覆盖
    if extra_args:
        for kv in _parse_cli_overrides(extra_args):
            key, val = kv
            cfg[key] = _coerce(val, cfg.get(key))

    return cfg


def _parse_cli_overrides(args: list[str]) -> list[tuple[str, str]]:
    """解析 --key value 格式的 CLI 参数"""
    result = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            key = a[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                result.append((k, v))
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                result.append((key, args[i + 1]))
                i += 1
        i += 1
    return result


def _coerce(val: str, ref: Any) -> Any:
    """把字符串值强制转换为与参考值相同的类型"""
    if ref is None:
        # 尝试推断类型
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                if val.lower() in ("true", "yes"):
                    return True
                if val.lower() in ("false", "no"):
                    return False
                return val
    if isinstance(ref, bool):
        return val.lower() in ("true", "yes", "1")
    if isinstance(ref, int):
        return int(float(val))
    if isinstance(ref, float):
        return float(val)
    return val


def _simple_yaml_load(path: str) -> dict:
    """极简 YAML 解析（仅 key: value，不支持嵌套）"""
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip()
                # 处理内联注释
                if "#" in v:
                    v = v[:v.index("#")].strip()
                # 处理数字中的下划线（如 200_000）
                v_clean = v.replace("_", "")
                cfg[k] = _coerce(v_clean, None)
    return cfg
