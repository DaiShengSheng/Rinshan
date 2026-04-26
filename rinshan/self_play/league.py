"""
League — 历史模型池

维护一个过去 checkpoint 的滑动窗口，每隔固定迭代数加入一个快照。
自对弈时按概率从池中抽取对手（最新模型权重最高），防止策略退化和循环。

用法：
    league = League(max_size=8, latest_weight=0.5)
    league.add(model, step=1000)
    opponent_model = league.sample()
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Optional

import torch


class League:
    """
    历史模型池

    Args:
        max_size:       保留的历史 checkpoint 数（不含当前模型本身）
        latest_weight:  当前最新模型被抽中的概率；其余历史模型均分剩余概率
        save_dir:       可选：把历史 checkpoint 序列化到磁盘，节省 GPU 显存
        device:         加载 checkpoint 时放到哪个设备
    """

    def __init__(
        self,
        max_size: int = 8,
        latest_weight: float = 0.5,
        save_dir: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.max_size       = max_size
        self.latest_weight  = latest_weight
        self.save_dir       = Path(save_dir) if save_dir else None
        self.device         = device

        # 历史模型列表（state_dict 或文件路径），从旧到新
        self._pool: list[dict | Path] = []
        self._steps: list[int]        = []
        self._current: Optional[dict] = None   # 当前最新模型的 state_dict

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def update_current(self, model: torch.nn.Module, step: int) -> None:
        """更新当前最新模型（每次训练后调用）"""
        self._current = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def snapshot(self, model: torch.nn.Module, step: int) -> None:
        """
        把当前模型状态加入历史池。

        若 save_dir 已设置，则序列化到磁盘；否则在内存中保存 state_dict 副本。
        超过 max_size 时淘汰最老的一个。
        """
        if self.save_dir:
            path = self.save_dir / f"league_{step:08d}.pt"
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
            entry: dict | Path = path
        else:
            entry = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        self._pool.append(entry)
        self._steps.append(step)

        # 超出容量时淘汰最老的（同时删除磁盘文件）
        while len(self._pool) > self.max_size:
            old = self._pool.pop(0)
            self._steps.pop(0)
            if isinstance(old, Path) and old.exists():
                try:
                    old.unlink()
                except OSError:
                    pass

    def sample_state_dict(self) -> Optional[dict]:
        """
        按加权概率抽取一个 state_dict：
          - latest_weight 概率返回当前最新模型
          - (1 - latest_weight) 平均分配给历史池中的模型
        若历史池为空且当前模型也为 None，返回 None。
        """
        if not self._pool and self._current is None:
            return None

        # 构造候选列表
        candidates: list[dict | Path] = []
        weights:    list[float]       = []

        # 当前模型
        if self._current is not None:
            candidates.append(self._current)
            weights.append(self.latest_weight if self._pool else 1.0)

        # 历史模型
        if self._pool:
            hist_weight = (1.0 - self.latest_weight) / len(self._pool) \
                          if self._current is not None else 1.0 / len(self._pool)
            for entry in self._pool:
                candidates.append(entry)
                weights.append(hist_weight)

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        return self._load(chosen)

    def _load(self, entry: dict | Path) -> dict:
        if isinstance(entry, Path):
            return torch.load(entry, map_location=self.device, weights_only=True)
        return {k: v.to(self.device) for k, v in entry.items()}

    @property
    def size(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        steps_str = ", ".join(str(s) for s in self._steps)
        return f"League(size={self.size}/{self.max_size}, steps=[{steps_str}])"
