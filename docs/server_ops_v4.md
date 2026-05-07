# Rinshan 服务器端操作手册（v4 vocab，分数特征版）

> 适用版本：commit `45c7fb3` 之后（VOCAB_SIZE=1612，分数 token 已加入 META 段）
> 服务器环境：AutoDL，单卡 A100/A800，路径约定见下文

---

## 目录

1. [路径约定](#路径约定)
2. [v4 词表说明（必读）](#v4-词表说明)
3. [情况 A：全新复现（从零开始）](#情况-a全新复现)
4. [情况 B：已有旧 checkpoint，接续到 Stage 3](#情况-b已有旧-checkpoint)
5. [通用工具命令](#通用工具命令)
6. [收敛指标速查](#收敛指标速查)
7. [常见问题](#常见问题)

---

## 路径约定

```
/root/autodl-tmp/rinshan/
├── data/
│   ├── annotated_v4/        # parse 后的 .jsonl 标注数据（Stage 1/2 用）
│   └── annotated_grp/       # fill_grp_rewards.py 处理后的数据（Stage 3 用）
└── checkpoints/
    ├── stage1_base/
    │   └── best.pt          # Stage 1 最优（v4 词表从零训练时直接产出）
    ├── oracle_base/
    │   └── best.pt
    ├── stage2_base/
    │   ├── best.pt          # 旧词表（1548），仅已有历史产物
    │   └── best_v4.pt       # 新词表（1612），迁移或重训后的权重
    └── stage3_base/
        └── best.pt
```

---

## v4 词表说明

| 项目 | 旧值 | 新值 | 说明 |
|---|---|---|---|
| `VOCAB_SIZE` | 1548 | 1612 | 新增 64 个分数 token |
| `SCORE_OFFSET` | — | 1548 | 4 seats × 16 bins，覆盖 -60k~+90k |
| `MAX_GAME_META_LEN` | 16 | 20 | META 段末尾追加 4 个分数 token |
| 序列实际长度 | 196 | 200 | +4 个 META token |

**新 checkpoint 与旧 checkpoint 不兼容**。旧 best.pt（vocab=1548）必须先用迁移脚本扩容，否则 embedding 维度不匹配。

---

## 情况 A：全新复现

> 完全从零开始，没有任何旧 checkpoint。全流程约需 4~6 天（单 A100）。

### 步骤 0：环境准备

```bash
cd /root/autodl-fs/Rinshan0503
pip install -r requirements.txt   # 含 mahjong, torch, pyyaml 等

# 验证词表和 encoder
python scripts/validate_score_feature.py
# 预期输出末行: ALL OK
```

### 步骤 1：解析牌谱数据

```bash
# 把天凤/雀魂 .xml / .mjson 解析为 .jsonl 标注
python scripts/parse_tenhou.py \
    --input_dir /root/autodl-tmp/rinshan/data/raw \
    --output_dir /root/autodl-tmp/rinshan/data/annotated_v4 \
    --num_workers 16

# 验证数据量（建议 >=500 万条）
find /root/autodl-tmp/rinshan/data/annotated_v4 -name "*.jsonl" | xargs wc -l | tail -1
```

### 步骤 2：训练 GRP 模型

```bash
python scripts/train_grp.py configs/grp.yaml
# 产出：checkpoints/grp/best.pt
```

### 步骤 3：填入 GRP Reward

```bash
python scripts/fill_grp_rewards.py \
    --data_dir /root/autodl-tmp/rinshan/data/annotated_v4 \
    --output_dir /root/autodl-tmp/rinshan/data/annotated_grp \
    --grp_ckpt /root/autodl-tmp/rinshan/checkpoints/grp/best.pt \
    --num_workers 16
# 产出：annotated_grp/  每条样本含 grp_reward 字段
```

### 步骤 4：Stage 1 行为克隆

```bash
# 编辑 configs/stage1_base.yaml
# 确认 data_dir 指向 annotated_v4
# 确认 init_ckpt 行注释掉（全新训练）

python scripts/train_stage1.py configs/stage1_base.yaml
# 产出：checkpoints/stage1_base/best.pt（v4 词表，直接可用）

# 收敛标准：val_acc > 0.76，val_loss < 0.30
```

### 步骤 5：训练 Oracle

```bash
python scripts/train_oracle.py configs/oracle_base.yaml
# 产出：checkpoints/oracle_base/best.pt
```

### 步骤 6：Stage 2 Oracle 蒸馏

```bash
# 编辑 configs/stage2_base.yaml
#   stage1_ckpt: .../checkpoints/stage1_base/best.pt   ← 步骤 4 产出（v4 词表，无需迁移）
#   oracle_ckpt: .../checkpoints/oracle_base/best.pt
#   data_dir:    .../data/annotated_v4

python scripts/train_stage2.py configs/stage2_base.yaml
# 产出：checkpoints/stage2_base/best.pt（= best_v4.pt，直接可用）

# 收敛标准（step ~100k）：
#   bel_recall  > 0.80
#   val KL      < 0.009（temperature=2.0）
#   val BC      < 0.185
```

### 步骤 7：Stage 3 离线 IQL

```bash
# configs/stage3_base.yaml 已配置：
#   stage2_ckpt: .../checkpoints/stage2_base/best.pt
#   data_dir:    .../data/annotated_grp

python scripts/train_stage3.py configs/stage3_base.yaml
# 产出：checkpoints/stage3_base/best.pt
```

---

## 情况 B：已有旧 checkpoint，接续到 Stage 3

> 已完成 Stage 1 + Stage 2（vocab=1548），现在要迁移到 v4 词表并开始 Stage 3。

### 步骤 B-0：拉取最新代码

```bash
cd /root/autodl-fs/Rinshan0503
git pull origin main

# 验证词表
python scripts/validate_score_feature.py
# 预期末行: ALL OK
```

### 步骤 B-1：修复 Stage 2 checkpoint 的 scheduler（T_max 问题）

> **仅当** Stage 2 是在旧 total_steps=100000 的 config 下训练，后来改成 120000 才需要此步骤。
> 如果 Stage 2 已跑完或用的是 best.pt，可跳过。

```bash
python -c "
import torch, math
from pathlib import Path

ckpt_dir = Path('/root/autodl-tmp/rinshan/checkpoints/stage2_base')
p = sorted(ckpt_dir.glob('checkpoint_*.pt'), key=lambda x: int(x.stem.split('_')[-1]))[-1]
print(f'Patching {p}')
ckpt = torch.load(p, map_location='cpu', weights_only=True)
cosine = ckpt['scheduler']['_schedulers'][1]
t = cosine['last_epoch']
T_new, eta_min, base_lr = 119500, 1e-5, 1e-4
new_lr = eta_min + 0.5*(base_lr-eta_min)*(1+math.cos(math.pi*t/T_new))
cosine['T_max'] = T_new
cosine['_last_lr'] = [new_lr, new_lr]
ckpt['scheduler']['_last_lr'] = [new_lr, new_lr]
torch.save(ckpt, p)
print(f'Done. t={t}, new_lr={new_lr:.4e}')
"
```

### 步骤 B-2：继续跑完 Stage 2（如未跑完）

```bash
python scripts/train_stage2.py configs/stage2_base.yaml
# 直到产出 checkpoints/stage2_base/best.pt（旧词表 1548）
```

### 步骤 B-3：迁移 Stage 2 权重到 v4 词表

```bash
python scripts/migrate_vocab_score.py \
    --src /root/autodl-tmp/rinshan/checkpoints/stage2_base/best.pt \
    --dst /root/autodl-tmp/rinshan/checkpoints/stage2_base/best_v4.pt

# 验证
python -c "
import torch
ckpt = torch.load('/root/autodl-tmp/rinshan/checkpoints/stage2_base/best_v4.pt',
                  map_location='cpu', weights_only=True)
w = ckpt['model']['transformer.token_embed.weight']
print(f'embedding shape: {w.shape}')   # 期望: torch.Size([1612, 768])
assert w.shape[0] == 1612, f'Wrong vocab size: {w.shape[0]}'
print('OK')
"
```

### 步骤 B-4：填入 GRP Reward（如果 annotated_grp 尚未生成）

```bash
python scripts/fill_grp_rewards.py \
    --data_dir /root/autodl-tmp/rinshan/data/annotated_v4 \
    --output_dir /root/autodl-tmp/rinshan/data/annotated_grp \
    --grp_ckpt /root/autodl-tmp/rinshan/checkpoints/grp/best.pt \
    --num_workers 16
```

### 步骤 B-5：Stage 3 离线 IQL

```bash
# configs/stage3_base.yaml 已更新为：
#   stage2_ckpt: .../checkpoints/stage2_base/best_v4.pt
#   arena_gate_baseline_ckpt: .../checkpoints/stage2_base/best_v4.pt
#   data_dir: .../data/annotated_grp

python scripts/train_stage3.py configs/stage3_base.yaml
```

---

## 通用工具命令

```bash
# 查看 checkpoint 的词表大小和 step
python scripts/inspect_checkpoint.py \
    --ckpt /root/autodl-tmp/rinshan/checkpoints/stage2_base/best_v4.pt

# 诊断 Oracle-Student KL（Stage 2 用）
python scripts/diagnose_oracle.py configs/stage2_base.yaml

# 评估 Belief 准确率
python scripts/eval_belief_accuracy.py \
    --ckpt /root/autodl-tmp/rinshan/checkpoints/stage2_base/best_v4.pt \
    --data_dir /root/autodl-tmp/rinshan/data/annotated_v4

# 评估 BC 准确率（行为克隆 top-1 acc）
python scripts/eval_bc_accuracy.py \
    --ckpt /root/autodl-tmp/rinshan/checkpoints/stage3_base/best.pt \
    --data_dir /root/autodl-tmp/rinshan/data/annotated_v4
```

---

## 收敛指标速查

### Stage 1（行为克隆）

| 指标 | 及格 | 目标 | 失败警告 |
|---|---|---|---|
| `val_acc` | > 0.74 | > 0.76 | < 0.70 停滞 |
| `val_loss` | < 0.32 | < 0.28 | > 0.40 不降 |

### Stage 2（Oracle 蒸馏）

| 指标 | 及格 | 目标 | 失败警告 |
|---|---|---|---|
| `bel_recall` | > 0.78 | > 0.82 | < 0.75 |
| `val KL`（temp=2.0）| < 0.009 | < 0.006 | > 0.012 不降 |
| `val BC` | < 0.190 | < 0.185 | > 0.250 反弹 |
| `score(kl+0.3bc)` | < 0.065 | < 0.060 | — |

### Stage 3（IQL）

| 指标 | 及格 | 目标 | 失败警告 |
|---|---|---|---|
| `val q_loss` | 持续下降 | < 5.0 | 反弹 > 10.0 |
| `val v_loss` | 持续下降 | < 5.0 | 反弹 > 10.0 |
| `val bc` | < 0.25 | < 0.22 | > 0.30 说明策略漂移 |
| `val bel_recall` | > 0.82 | > 0.85 | < 0.78 退化 |
| `arena delta_rank` | ≤ 0.05 | ≤ 0.0 | > 0.10 说明 RL 在伤害策略 |

---

## 常见问题

**Q: validate_score_feature.py 报 `AssertionError: gap=...`**
A: 说明代码没有完整拉取。执行 `git pull` 后重试。

**Q: Stage 2 resume 后日志出现 `[scheduler] T_max corrected`**
A: 正常，这是自动修复 scheduler T_max 的提示，说明旧 checkpoint 的 cosine 周期被校正了。

**Q: Stage 3 load checkpoint 报 `size mismatch for transformer.token_embed.weight`**
A: stage2_ckpt 指向的是旧词表（1548）的权重，需要先执行步骤 B-3 迁移到 best_v4.pt。

**Q: Stage 3 arena gate 持续 not pass**
A: 检查 `arena_gate_baseline_ckpt` 是否也指向了 best_v4.pt。
新旧词表模型在 arena 对打时行为对比无意义，两边必须用同词表权重。

**Q: 从头复现但跳过了 Stage 1，直接从旧 Stage 1 权重开始 Stage 2**
A: 旧 Stage 1 权重（vocab=1548）同样需要用 `migrate_vocab_score.py` 迁移后再传给 stage2.yaml 的 `stage1_ckpt`。
