# 服务器环境搭建指南

本文档说明如何在 Linux 训练服务器上配置完整的 Rinshan + libriichi 运行环境。

---

## 前置条件

- Ubuntu 20.04 / 22.04（或同等 Linux 发行版）
- Python 3.10 或 3.11
- CUDA 11.8+（训练用，自对弈引擎不需要 GPU）
- 有 sudo 权限

---

## 第一步：安装系统依赖

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    python3-dev \
    python3-pip \
    python3-venv
```

---

## 第二步：安装 Rust 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source $HOME/.cargo/env

# 验证
rustc --version   # 应显示 1.75.0+
cargo --version
```

---

## 第三步：创建 Python 虚拟环境并安装依赖

```bash
cd /path/to/Rinshan
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install maturin numpy
pip install -e .   # 安装 rinshan 包本身
```

---

## 第四步：编译 libriichi

Rust 源码已随项目一起存放在 `libriichi/` 目录，无需另外克隆 mortal。

**注意**：`libriichi/libriichi/Cargo.toml` 中 lib name 已改为 `libriichi`（与 `lib.rs` 中
`fn libriichi(...)` 一致），无需额外修改。

```bash
cd /path/to/Rinshan
source .venv/bin/activate

# --release 约需 5~10 分钟（16 核可用 CARGO_BUILD_JOBS=16 加速）
maturin build --release -m libriichi/Cargo.toml --out dist/
pip install dist/libriichi-*.whl --force-reinstall
```

**注意**：编译前需要确认 `libriichi/Cargo.toml` 中 lib name 与 `lib.rs` 中 `#[pymodule]` 函数名一致，
否则 pyo3 找不到入口符号。已确认需要将 `name = "riichi"` 改为 `name = "libriichi"`：

```toml
# libriichi/Cargo.toml
[lib]
name = "libriichi"      # ← 必须与 lib.rs 中 fn libriichi(...) 一致
crate-type = ["cdylib", "rlib"]
```

```bash
cd /path/to/mortal
source /path/to/Rinshan/.venv/bin/activate   # 必须激活虚拟环境

# 编译成 wheel 再安装（--release 约需 5~10 分钟）
maturin build --release -m libriichi/Cargo.toml --out dist/
pip install dist/libriichi-*.whl --force-reinstall
```

**pyo3 子模块 import patch**：pyo3 编译的子模块需要手动注册到 `sys.modules`，
否则 `from libriichi.arena import OneVsThree` 会失败。
找到安装后的 `__init__.py`（`python -c "import libriichi; print(libriichi.__file__)"`），
确保内容如下：

```python
import sys
from .libriichi import *
from .libriichi import arena, consts, dataset, mjai, stat, state
from . import libriichi as _inner

_pkg = __name__
for _submod_name in ("arena", "consts", "dataset", "mjai", "stat", "state"):
    _key = f"{_pkg}.{_submod_name}"
    if _key not in sys.modules:
        sys.modules[_key] = getattr(_inner, _submod_name)

__doc__ = _inner.__doc__
if hasattr(_inner, "__all__"):
    __all__ = _inner.__all__
```

```bash
# 验证
python -c "
from libriichi.arena import OneVsThree
from libriichi.state import PlayerState
from libriichi.consts import ACTION_SPACE
import libriichi
print('version:', libriichi.__version__)
print('ACTION_SPACE:', ACTION_SPACE)
print('PlayerState:', PlayerState(0))
"
```

> **编译时间参考**：
> - 4 核 CPU：约 8~12 分钟
> - 16 核 CPU：约 2~4 分钟（Cargo 自动并行）
>
> 可用 `export CARGO_BUILD_JOBS=16` 指定并行数。

---

## 第五步：验证完整环境

```bash
source /path/to/Rinshan/.venv/bin/activate
cd /path/to/Rinshan

# 1. 引擎基础验证
python -c "
from rinshan.self_play.agent import RandomAgent
from rinshan.self_play.arena import Arena
arena = Arena([RandomAgent()], n_games=2, game_length='tonpuu', show_progress=False)
records = arena.run()
print(f'engine OK: {len(records)} games, scores={records[0].final_scores}')
"

# 2. libriichi 验证
python -c "
import libriichi
from libriichi.arena import OneVsThree
print(f'libriichi OK: version={libriichi.__version__}')
"

# 3. 模型前向验证
python -c "
import torch
from rinshan.model.full_model import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.constants import MODEL_CONFIGS
cfg = TransformerConfig(**MODEL_CONFIGS['nano'])
model = RinshanModel(cfg)
print(f'model OK: {model.count_parameters()[\"total\"]/1e6:.1f}M params')
"
```

---

## 第六步：跑训练

```bash
source /path/to/Rinshan/.venv/bin/activate
cd /path/to/Rinshan

# Stage 1（行为克隆）
python scripts/train_stage1.py configs/stage1_base.yaml

# Stage 3（离线 IQL）
python scripts/train_stage3.py configs/stage3_base.yaml \
    --stage1_ckpt checkpoints/stage1_base/best.pt

# Stage 4（在线自对弈，长期运行）
nohup python scripts/train_stage4.py configs/stage4_self_play.yaml \
    --stage3_ckpt checkpoints/stage3_base/best.pt \
    > logs/stage4.log 2>&1 &
echo "Stage 4 PID: $!"
```

---

## 引擎速度 benchmark

```bash
# 测量 Python 引擎吞吐量上限
python scripts/bench_arena.py --games 64 --length hanchan

# 典型结果参考（16 核 CPU）：
#   RandomAgent × 4：约 30~80 games/s（半庄）
#   RinshanAgent（GPU 批量推理）：约 5~20 games/s（取决于 GPU 和并发数）
```

---

## 常见问题

### libriichi 编译报 `pyo3` 版本不匹配

```bash
# 检查 pyo3 要求的 Python 版本
grep pyo3 /path/to/mortal/libriichi/Cargo.toml
# 确保 maturin 使用的 Python 和虚拟环境一致
which python && python --version
```

### CUDA out of memory

调小 `configs/stage1_base.yaml` 中的 `batch_size`，或改用 `model_preset: nano` 先跑通。

### Stage 4 自对弈很慢

增大 `self_play.n_games_per_iter`（更多并发局数均摊 GPU 推理），参考 `scripts/bench_arena.py` 的输出估算最优值。
