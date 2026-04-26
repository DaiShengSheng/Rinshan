"""
train_stage4.py — Stage 4: 在线强化学习 + League 自对弈闭环

流程（每轮 iteration）：
  1. [Generate]  当前模型 + League 历史对手 → Arena 自对弈 → GameRecord
  2. [Parse]     GameRecord → OnlineBuffer（Transition 队列）
  3. [Train]     从 OnlineBuffer 采样 mini-batch → 在线 IQL 梯度更新
  4. [Snapshot]  每隔 snapshot_every 轮把当前模型存入 League

Usage:
    python scripts/train_stage4.py configs/stage4_self_play.yaml \\
        --stage3_ckpt checkpoints/stage3_base/best.pt

Config 关键字段（见 configs/stage4_self_play.yaml）：
    n_iters              总迭代轮数
    train_steps_per_iter 每轮训练步数
    self_play.n_games_per_iter  每轮自对弈局数
    self_play.game_length       hanchan / tonpuu
    league.max_size             League 历史池大小（默认 8）
    league.latest_weight        当前模型被采样概率（默认 0.5）
    league.snapshot_every       每多少轮存一次 snapshot（默认 10）
    buffer.capacity             Replay buffer 容量（默认 50000）
"""
from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.utils.config    import load_config
from rinshan.training        import Trainer, TrainerConfig
from rinshan.model.full_model import RinshanModel
from rinshan.model.transformer import TransformerConfig
from rinshan.constants       import MODEL_CONFIGS
from rinshan.self_play.agent          import RinshanAgent, RandomAgent
from rinshan.self_play.arena          import Arena
from rinshan.self_play.league         import League
from rinshan.self_play.online_buffer  import OnlineBuffer
from rinshan.self_play.libriichi_agent import LibriichiBoostedAgent, libriichi_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_stage4")

# 启动时报告 libriichi 状态
if libriichi_available():
    logger.info("libriichi available: candidate generation will use Rust engine")
else:
    logger.warning("libriichi NOT available: falling back to pure Python candidate generation")


# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

def _load_model(preset: str, ckpt_path: str, device: torch.device) -> RinshanModel:
    cfg_dict = MODEL_CONFIGS[preset]
    tcfg = TransformerConfig(**cfg_dict)
    model = RinshanModel(transformer_cfg=tcfg, use_belief=True, use_aux=False)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights from {ckpt_path}")
    return model.to(device)


def _build_agents(
    current_model: RinshanModel,
    league: League,
    device: torch.device,
    sp_cfg: dict,
    n_seats: int = 4,
) -> list:
    """
    构建 4 席 agent：
      - 至少 1 席用当前最新模型
      - 其余席位从 League 中采样历史模型（若 League 为空则也用当前模型）

    n_seats 个 agent 对象（RinshanAgent），每个持有独立的模型引用。
    League 的对手模型 CPU 推理，当前模型 GPU 推理（如可用）。
    """
    temp = float(sp_cfg.get("temperature", 0.8))
    top_p = float(sp_cfg.get("top_p", 0.9))
    greedy = bool(sp_cfg.get("greedy", False))

    agents = []
    for i in range(n_seats):
        # 优先使用 LibriichiBoostedAgent（候选生成 Rust 化），不可用时自动降级
        AgentCls = LibriichiBoostedAgent if libriichi_available() else RinshanAgent

        if i == 0:
            # 席位 0 始终用当前最新模型（GPU）
            agent = AgentCls(
                model=current_model,
                name="current",
                device=str(device),
                temperature=temp, top_p=top_p, greedy=greedy,
            )
        else:
            # 其余席位从 League 采样
            sd = league.sample_state_dict()
            if sd is not None:
                from rinshan.constants import MODEL_CONFIGS
                preset = sp_cfg.get("model_preset", "base")
                cfg_dict = MODEL_CONFIGS.get(preset, MODEL_CONFIGS["base"])
                tcfg = TransformerConfig(**cfg_dict)
                opp_model = RinshanModel(tcfg, use_belief=True, use_aux=False)
                opp_model.load_state_dict(sd, strict=False)
                opp_model.eval()
                agent = AgentCls(
                    model=opp_model,
                    name=f"league_{i}",
                    device="cpu",
                    temperature=temp, top_p=top_p, greedy=greedy,
                )
            else:
                # League 为空：用 RandomAgent
                agent = RandomAgent(name=f"random_{i}", seed=i)
        agents.append(agent)
    return agents


# ─────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_stage4.py <config.yaml> [--stage3_ckpt <path>]")
        sys.exit(1)

    cfg = load_config(sys.argv[1], sys.argv[2:])
    logger.info(f"Stage 4 config:\n{cfg}")

    # ── 基础参数 ───────────────────────────────
    device_str    = cfg.get("device", "cpu")
    device        = torch.device(device_str if torch.cuda.is_available() else "cpu")
    preset        = cfg.get("model_preset", "base")
    stage3_ckpt   = cfg.get("stage3_ckpt", "")
    n_iters       = int(cfg.get("n_iters", 100))
    train_steps   = int(cfg.get("train_steps_per_iter", 500))
    batch_size    = int(cfg.get("batch_size", 32))
    save_dir      = Path(cfg.get("save_dir", "checkpoints/stage4"))
    save_every    = int(cfg.get("save_every", 10))      # 单位：iter
    log_every     = int(cfg.get("log_every", 1))        # 单位：iter

    # 自对弈参数
    sp_cfg = cfg.get("self_play", {})
    n_games_per_iter = int(sp_cfg.get("n_games_per_iter", 32))
    game_length      = sp_cfg.get("game_length", "hanchan")

    # League 参数
    lc_cfg           = cfg.get("league", {})
    league_max_size  = int(lc_cfg.get("max_size", 8))
    league_lw        = float(lc_cfg.get("latest_weight", 0.5))
    snapshot_every   = int(lc_cfg.get("snapshot_every", 10))

    # Buffer 参数
    buf_cfg          = cfg.get("buffer", {})
    buf_capacity     = int(buf_cfg.get("capacity", 50_000))
    warmup_samples   = int(buf_cfg.get("warmup_samples", batch_size * 4))

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 初始化组件 ──────────────────────────────
    current_model = _load_model(preset, stage3_ckpt, device)

    league = League(
        max_size       = league_max_size,
        latest_weight  = league_lw,
        save_dir       = save_dir / "league",
        device         = "cpu",
    )
    league.update_current(current_model, step=0)
    league.snapshot(current_model, step=0)   # 把初始模型放入池

    buffer = OnlineBuffer(
        capacity     = buf_capacity,
        rank_scale   = float(cfg.get("rank_scale", 1.0)),
        score_scale  = float(cfg.get("score_scale", 1000.0)),
    )

    # Trainer（Stage 3 模式 = IQL，CQL 权重降低用于在线场景）
    trainer_cfg = TrainerConfig(
        stage        = 3,   # 复用 IQL 损失
        device       = device_str,
        dtype        = cfg.get("dtype", "float32"),
        amp          = bool(cfg.get("amp", False)),
        compile      = bool(cfg.get("compile", False)),
        model_preset = preset,
        lr           = float(cfg.get("lr", 1e-5)),
        weight_decay = float(cfg.get("weight_decay", 0.01)),
        warmup_steps = int(cfg.get("warmup_steps", 100)),
        total_steps  = n_iters * train_steps + 200,
        save_dir     = str(save_dir),
        save_every   = 999_999,   # 由外层控制
        log_every    = int(cfg.get("log_every_steps", 50)),
        target_update_every = int(cfg.get("target_update_every", 200)),
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0)),
    )
    trainer = Trainer(trainer_cfg)

    # 把已加载的权重复制进 Trainer
    trainer.model.load_state_dict(current_model.state_dict(), strict=False)
    if trainer.target_model:
        trainer.target_model.load_state_dict(current_model.state_dict(), strict=False)
    current_model = trainer.model   # 让两者指向同一对象

    # CQL 调低（在线场景无需强保守约束）
    from rinshan.constants import CQL_WEIGHT
    online_cql_weight = float(cfg.get("online_cql_weight", 0.1))

    # ── 主循环 ────────────────────────────────
    logger.info(
        f"Starting Stage 4: {n_iters} iters × "
        f"{n_games_per_iter} games/iter + {train_steps} train_steps/iter"
    )

    total_games    = 0
    total_samples  = 0
    total_train_steps = 0
    t_start        = time.time()

    for iteration in range(1, n_iters + 1):
        t_iter = time.time()

        # ── 1. 自对弈 ────────────────────────────
        agents = _build_agents(
            current_model, league, device, sp_cfg
        )
        arena = Arena(
            agents         = agents,
            n_games        = n_games_per_iter,
            game_length    = game_length,
            base_seed      = iteration * 1000,
            agent_rotation = "round_robin",
            show_progress  = False,
        )
        records = arena.run()
        total_games += len(records)

        n_new = buffer.ingest_records(records)
        total_samples += n_new

        t_gen = time.time() - t_iter

        # ── 2. 训练（等 buffer 预热后才开始）───
        t_train_start = time.time()
        iter_losses: list[float] = []

        if buffer.size >= warmup_samples:
            for batch in buffer.iter_batches(batch_size, train_steps):
                # 在线时关闭 CQL（offline=False）
                from rinshan.training.losses import iql_loss
                # 通过 monkey-patch 调整 cql_weight
                batch["_online_cql_weight"] = online_cql_weight

                loss_dict = trainer.train_step(batch)
                total_train_steps += 1
                iter_losses.append(loss_dict["total"])
        else:
            logger.info(
                f"[iter {iteration}] Buffer warming up "
                f"({buffer.size}/{warmup_samples}), skipping training"
            )

        t_train = time.time() - t_train_start

        # ── 3. 更新 League 的 current 引用 ──────
        league.update_current(current_model, step=total_train_steps)

        # 定期 snapshot
        if iteration % snapshot_every == 0:
            league.snapshot(current_model, step=total_train_steps)
            logger.info(f"  League snapshot added: {league}")

        # ── 4. 日志 ────────────────────────────
        if iteration % log_every == 0:
            avg_loss = sum(iter_losses) / max(len(iter_losses), 1)
            elapsed  = time.time() - t_start
            games_ps = total_games / elapsed
            # 当前局平均分布
            all_scores = [s for r in records for s in r.final_scores]
            avg_score  = sum(all_scores) / max(len(all_scores), 1)
            logger.info(
                f"[iter {iteration:4d}/{n_iters}] "
                f"games={total_games} ({games_ps:.1f}/s)  "
                f"buf={buffer.size}  "
                f"train_steps={total_train_steps}  "
                f"avg_loss={avg_loss:.4f}  "
                f"avg_score={avg_score:.0f}  "
                f"t_gen={t_gen:.1f}s  t_train={t_train:.1f}s"
            )

        # ── 5. 保存 ────────────────────────────
        if iteration % save_every == 0:
            ckpt_path = save_dir / f"iter_{iteration:06d}.pt"
            trainer.save(ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # 训练结束
    trainer.save(save_dir / "final.pt")
    logger.info(
        f"Stage 4 complete. "
        f"Total: {total_games} games, {total_train_steps} train steps, "
        f"{time.time()-t_start:.0f}s"
    )


if __name__ == "__main__":
    main()
