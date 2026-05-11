"""
train_stage3.py — Stage 3: 离线 IQL 强化学习（GRP 2.0）

在 Stage 1/2 的基础上，用 GRP 价值差分信号做离线 RL 精调。
GRP 2.0 的关键变化：
  1. 保留 GRP 作为 learned game-value estimator
  2. 不再把同一局的 grp_reward 广播到局内所有 action
  3. 只在 GRP game-state 真正变化（进入下一局 / 终局）时，
     将该局 delta-value 记到最后一个 action 上；其余 action reward=0
  4. 训练时加入 AWR 风格 BC anchor，防止策略快速偏离 Stage2 基线

需要先跑完：
  1. train_grp.py         → 得到 GRP 模型
  2. fill_grp_rewards.py  → 为标注数据填入 grp_reward 字段（按局 delta）
  3. train_stage1.py / train_stage2.py → 得到 Stage 初始化权重

Usage:
    python scripts/train_stage3.py configs/stage3_base.yaml \
        --stage1_ckpt checkpoints/stage1_base/best.pt
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rinshan.utils.config import load_config
from rinshan.data         import MjaiDataset, collate_fn
from rinshan.training     import Trainer, TrainerConfig
from rinshan.model.full_model import RinshanModel
from rinshan.model.transformer import TransformerConfig


def _arena_gate(cfg: dict, ckpt_path: Path, step: int, save_dir: Path) -> tuple[bool, dict]:
    gate_games = int(cfg.get("arena_gate_games", 0))
    baseline_ckpt = cfg.get("arena_gate_baseline_ckpt", cfg.get("stage2_ckpt", ""))
    if gate_games <= 0 or not baseline_ckpt:
        return False, {}

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_self_play.py")),
        "--mode", "versus",
        "--ckpt", str(ckpt_path),
        "--ckpt2", str(baseline_ckpt),
        "--model_preset", str(cfg.get("model_preset", "base")),
        "--n_games", str(gate_games),
        "--parallel_games", str(cfg.get("arena_parallel_games", gate_games)),
        "--parallel_groups", str(cfg.get("arena_parallel_groups", 1)),
        "--device", str(cfg.get("arena_device", cfg.get("device", "cuda"))),
        "--seed", str(int(cfg.get("arena_seed", 1234)) + int(step)),
        "--greedy",
        "--quiet",
    ]
    ckpt2_preset = cfg.get("arena_gate_baseline_preset")
    if ckpt2_preset:
        cmd.extend(["--ckpt2_preset", str(ckpt2_preset)])

    logger.info("Running arena gate: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(Path(__file__).parents[1]), capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("Arena gate failed (returncode=%s): %s", proc.returncode, proc.stderr.strip())
        return False, {"error": proc.stderr.strip(), "returncode": proc.returncode}

    # 直接在 stdout 中解析最后打印的 delta 行
    metrics = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "顺位差 Δ=" in line:
            try:
                delta_str = line.split("Δ=")[-1].split()[0]
                metrics["delta_rank"] = float(delta_str)
            except Exception:
                pass
        elif "Challenger  平均顺位" in line:
            parts = line.replace("Challenger  平均顺位", "").split()
            if parts:
                try:
                    metrics["challenger_avg_rank"] = float(parts[0])
                except Exception:
                    pass
    if "delta_rank" not in metrics:
        logger.warning("Arena gate parse failed; stdout tail:\n%s", "\n".join(proc.stdout.splitlines()[-20:]))
        return False, {"error": "parse_failed"}

    threshold = float(cfg.get("arena_gate_rank_delta_threshold", 0.0))
    passed = metrics["delta_rank"] <= threshold
    metrics["passed"] = passed
    metrics["threshold"] = threshold
    gate_log = save_dir / f"arena_gate_step{step}.log"
    gate_log.write_text(proc.stdout + "\n\nSTDERR:\n" + proc.stderr, encoding="utf-8")
    logger.info("Arena gate step=%s delta_rank=%.4f threshold=%.4f passed=%s", step, metrics["delta_rank"], threshold, passed)
    return passed, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stage3")


def _init_riichi_embed(model, device: str = "cpu") -> None:
    """
    RIICHI token (497) embedding 冷启动初始化。

    Stage 1/2 的训练数据里，RIICHI_TOKEN 从未作为 *候选动作* 出现在
    candidate 区域中（它只出现在 progression 序列里），导致 token_embed[497]
    的 embedding 向量虽然存在，但自始至终没有受到来自 Q-value head 方向的
    有意义的梯度，等价于随机初始化状态。

    修复方法：用语义最相近的 special-action token 的 embedding 均值来初始化
    RIICHI_TOKEN 的 embedding，给 Stage 3 的 Bellman 回传一个合理的起点。

    邻居选择依据：
      TSUMO_AGARI(498), RON_AGARI(499), RYUKYOKU(500), PASS(501)
      均为 special action token，与 RIICHI 同属"特殊操作"类别，且都经过完整
      的 Stage 1/2 训练，其 embedding 已编码了"特殊决策点"的语义。

    注意：PROG_RIICHI_BASE (665-668) 是进行序列里的"他家立直事件"token，
    语义为「观察到某人立直」，而非「我宣言立直」，两者不可混用。
    """
    from rinshan.constants import RIICHI_TOKEN, TSUMO_AGARI_TOKEN, RON_AGARI_TOKEN
    from rinshan.constants import RYUKYOKU_TOKEN, PASS_TOKEN

    # 找到 token embedding 层（兼容 torch.compile 包装后的 OptimizedModule）
    try:
        embed = model.transformer.token_embed
    except AttributeError:
        # torch.compile 后模型可能被包成 _orig_mod
        embed = model._orig_mod.transformer.token_embed

    neighbor_ids = [TSUMO_AGARI_TOKEN, RON_AGARI_TOKEN, RYUKYOKU_TOKEN, PASS_TOKEN]
    with torch.no_grad():
        neighbors = torch.stack([embed.weight[i] for i in neighbor_ids])  # (4, dim)
        mean_vec  = neighbors.mean(0)                                      # (dim,)
        old_norm  = embed.weight[RIICHI_TOKEN].norm().item()
        embed.weight[RIICHI_TOKEN].copy_(mean_vec)
        new_norm  = embed.weight[RIICHI_TOKEN].norm().item()
    logger.info(
        "RIICHI embed init: neighbors=%s  old_norm=%.4f → new_norm=%.4f",
        neighbor_ids, old_norm, new_norm,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_stage3.py <config.yaml> --stage1_ckpt <path>")
        sys.exit(1)

    cfg = load_config(sys.argv[1], sys.argv[2:])
    logger.info(f"Config: {cfg}")

    stage1_ckpt = cfg.get("stage1_ckpt", "")
    stage2_ckpt = cfg.get("stage2_ckpt", "")

    # ── 数据（Stage 3 需要 s,a,r,s' 对）────────
    data_dir = Path(cfg["data_dir"])
    all_files = sorted(data_dir.rglob("*.jsonl"))

    import random
    random.seed(42)
    random.shuffle(all_files)
    val_n = max(1, int(len(all_files) * cfg.get("val_ratio", 0.02)))
    train_files = all_files[val_n:]
    val_files   = all_files[:val_n]

    batch_size = cfg.get("batch_size", 64)
    train_ds = MjaiDataset(train_files, shuffle_files=True,
                           shuffle_buffer=cfg.get("shuffle_buffer", 2000),
                           stage=3)  # stage=3 会自动配对 (s, s')
    val_ds   = MjaiDataset(val_files, shuffle_files=False,
                           shuffle_buffer=500, stage=3)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,   # 不扩大，Stage3 val 同样有3次forward
                              collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)

    # ── Trainer（Stage 3）─────────────────────
    trainer_cfg = TrainerConfig(
        stage            = 3,
        device           = cfg.get("device", "cuda"),
        dtype            = cfg.get("dtype", "float32"),
        amp              = cfg.get("amp", False),
        compile          = cfg.get("compile", False),
        model_preset     = cfg.get("model_preset", "base"),
        lr               = float(cfg.get("lr", 1e-4)),
        weight_decay     = float(cfg.get("weight_decay", 0.01)),
        max_grad_norm    = float(cfg.get("max_grad_norm", 1.0)),
        grad_accum_steps = int(cfg.get("grad_accum_steps", 1)),   # ← 修复：之前漏传，永远是1
        warmup_steps     = int(cfg.get("warmup_steps", 500)),
        total_steps      = int(cfg.get("total_steps", 100_000)),
        cosine_t_max     = int(cfg.get("cosine_t_max", 0)),
        save_dir         = cfg.get("save_dir", "checkpoints/stage3"),
        save_every       = int(cfg.get("save_every", 5000)),
        log_every        = int(cfg.get("log_every", 100)),
        target_update_every = int(cfg.get("target_update_every", 100)),
        cql_weight          = float(cfg.get("cql_weight", -1.0)),
        weights_only_save   = bool(cfg.get("weights_only_save", False)),
        bc_weight           = float(cfg.get("bc_weight", 0.2)),
        reward_clip         = float(cfg.get("reward_clip", 20.0)),
        value_clip          = float(cfg.get("value_clip", 50.0)),
        adv_clip            = float(cfg.get("adv_clip", 20.0)),
        awr_temperature     = float(cfg.get("awr_temperature", 3.0)),
        awr_max_weight      = float(cfg.get("awr_max_weight", 20.0)),
        game_expectile      = float(cfg.get("game_expectile", 0.95)),
        hand_expectile      = float(cfg.get("hand_expectile", 0.70)),
        game_reward_weight  = float(cfg.get("game_reward_weight", 1.0)),
        hand_reward_weight  = float(cfg.get("hand_reward_weight", 1.0)),
        arena_gate_every  = float(cfg.get("arena_gate_every", 0.0)),
        riichi_legal_sample_weight = float(cfg.get("riichi_legal_sample_weight", 1.0)),
        riichi_bc_scale     = float(cfg.get("riichi_bc_scale", 1.0)),
        stage3_anchor_weight = float(cfg.get("stage3_anchor_weight", 0.0)),
        stage3_anchor_temperature = float(cfg.get("stage3_anchor_temperature", 1.0)),
        riichi_rank_weight  = float(cfg.get("riichi_rank_weight", 0.0)),
        riichi_margin       = float(cfg.get("riichi_margin", 0.2)),
    )
    trainer = Trainer(trainer_cfg)
    device  = trainer.device

    if trainer_cfg.stage3_anchor_weight > 0 and stage2_ckpt and Path(stage2_ckpt).exists():
        logger.info("Building Stage2 anchor model from %s", stage2_ckpt)
        anchor_cfg = TransformerConfig.from_preset(trainer_cfg.model_preset)
        anchor_model = RinshanModel(
            transformer_cfg=anchor_cfg,
            use_belief=True,
            use_aux=False,
            gradient_checkpointing=False,
        )
        s2_anchor_ckpt = torch.load(stage2_ckpt, map_location=device, weights_only=True)
        anchor_state = s2_anchor_ckpt["model"] if "model" in s2_anchor_ckpt else s2_anchor_ckpt
        anchor_model.load_state_dict(anchor_state, strict=False)
        trainer.set_oracle_model(anchor_model)
        logger.info("Stage2 anchor model attached for Stage3 non-riichi KL anchor")

    # ── 断点续训：优先恢复已有 checkpoint，否则从 Stage 1 初始化 ──
    # lr 变化时自动切换到 best.pt 权重，optimizer/scheduler 用新 lr 重建
    ckpt_dir = Path(trainer_cfg.save_dir)
    existing_ckpts = sorted(
        ckpt_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    best_val_path  = ckpt_dir / "best_val.pt"
    best_gate_path = ckpt_dir / "best.pt"
    if existing_ckpts:
        latest = existing_ckpts[-1]
        # 只 peek lr，不做完整 load
        peek = torch.load(latest, map_location="cpu", weights_only=True)
        ckpt_lr = peek.get("lr", None)
        lr_changed = ckpt_lr is not None and abs(ckpt_lr - trainer_cfg.lr) > 1e-12
        if lr_changed:
            # lr 有变化：优先从 best.pt / best_val.pt 加载权重，没有才用最新 checkpoint
            src = best_gate_path if best_gate_path.exists() else (best_val_path if best_val_path.exists() else latest)
            logger.info(f"lr changed ({ckpt_lr:.2e} → {trainer_cfg.lr:.2e}): loading weights from {src}")
        else:
            src = latest
            logger.info(f"Resuming from {src}")
        del peek
        trainer.load(src)
    elif stage2_ckpt and Path(stage2_ckpt).exists():
        logger.info(f"Loading Stage 2 weights from {stage2_ckpt}")
        s2_ckpt = torch.load(stage2_ckpt, map_location=device, weights_only=True)
        trainer.model.load_state_dict(s2_ckpt["model"], strict=False)
        trainer.target_model.load_state_dict(s2_ckpt["model"], strict=False)
        logger.info("Target network initialized with Stage 2 weights (strict=False)")
    elif stage1_ckpt and Path(stage1_ckpt).exists():
        logger.info(f"Loading Stage 1 weights from {stage1_ckpt}")
        s1_ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=True)
        trainer.model.load_state_dict(s1_ckpt["model"], strict=False)
        trainer.target_model.load_state_dict(s1_ckpt["model"], strict=False)
        logger.info("Target network initialized with Stage 1 weights (strict=False)")
    else:
        logger.warning("No checkpoint or stage1_ckpt/stage2_ckpt found, training from scratch")

    # RIICHI embedding 初始化：由 yaml 的 reinit_riichi_embed 独立控制
    # true  → 无论是 resume 还是从 S1/S2 加载，都执行一次 reinit
    #         适用场景：S1/S2 没有训练过 RIICHI 候选（老版本），embedding 是噪声
    # false → 不执行，保留当前 checkpoint 里的 embedding
    #         适用场景：新版本 S1/S2 数据里已有 RIICHI 信号，embedding 已有意义
    if cfg.get("reinit_riichi_embed", False):
        _init_riichi_embed(trainer.model, device=str(device))
        _init_riichi_embed(trainer.target_model, device=str(device))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(cfg.get("total_steps", 100_000))
    val_every   = int(cfg.get("val_every", 2000))
    best_val_loss = float("inf")
    best_gate_delta = float("inf")
    # 恢复历史 best 指标，避免重启后“第一次 val 必定 best”
    if best_val_path.exists():
        try:
            best_val_ckpt = torch.load(best_val_path, map_location="cpu", weights_only=True)
            best_val_loss = float(best_val_ckpt.get("best_val_loss", best_val_loss))
            logger.info(f"Restored best_val_loss={best_val_loss:.4f} from {best_val_path}")
        except Exception as e:
            logger.warning(f"Failed to restore best_val_loss from {best_val_path}: {e}")
    if best_gate_path.exists():
        try:
            best_gate_ckpt = torch.load(best_gate_path, map_location="cpu", weights_only=True)
            best_gate_delta = float(best_gate_ckpt.get("best_gate_delta", best_gate_delta))
            logger.info(f"Restored best_gate_delta={best_gate_delta:.4f} from {best_gate_path}")
        except Exception as e:
            logger.warning(f"Failed to restore best_gate_delta from {best_gate_path}: {e}")

    # 检查 grp_reward 是否存在
    # 取第一个样本验证
    _checked = False

    logger.info(f"Starting Stage 3 (IQL) for {total_steps} steps")

    step = 0
    _ema: dict[str, float] = {}
    _ema_prev: dict[str, float] = {}
    _ema_alpha = 0.05

    def _update_ema(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, float):
                _ema[k] = v if k not in _ema else (1 - _ema_alpha) * _ema[k] + _ema_alpha * v

    def _trend(key: str) -> str:
        if key not in _ema or key not in _ema_prev:
            return ""
        delta = _ema[key] - _ema_prev[key]
        if delta < -0.001: return "↓"
        if delta >  0.001: return "↑"
        return "→"
    for batch in train_loader:
        if step >= total_steps:
            break

        # 首次检查 reward 是否已填入
        if not _checked:
            rewards = batch.get("reward")
            if rewards is not None:
                avg_r = rewards.float().mean().item()
                if abs(avg_r) < 1e-6:
                    logger.warning(
                        "All grp_reward values are ~0. "
                        "GRP 2.0 requires grp_reward deltas from fill_grp_rewards.py."
                    )
            _checked = True

        loss_dict = trainer.train_step(batch)
        step = trainer.step
        did_update = bool(loss_dict.get("did_update", False))
        if not did_update:
            continue
        _update_ema(loss_dict)

        # EMA 摘要行（按 optimizer update step）
        log_every = int(cfg.get("log_every", 100))
        if step % log_every == 0:
            ema_keys = [("q_loss","q"),("v_loss","v"),("bc_loss","bc"),("cql_loss","cql"),
                        ("belief","bel"),("wait","wait"),("total","total")]
            ema_parts = "  ".join(
                f"{s}={_ema[k]:.4f}{_trend(k)}"
                for k, s in ema_keys if k in _ema
            )
            logger.info(f"[ema  {step}] {ema_parts}" + ("" if _ema_prev else "  (warming up)"))
            _ema_prev.update(_ema)

        if step % val_every == 0:
            # 验证集 IQL 损失 + belief 召回率 + Stage3 立直/稳定性看板
            trainer.model.eval()
            val_keys = ["q_loss", "v_loss", "bc_loss", "cql_loss", "anchor_kl", "riichi_rank_loss", "belief", "wait", "total"]
            val_sums: dict[str, float] = {k: 0.0 for k in val_keys}
            from rinshan.constants import MAX_CANDIDATES_LEN, RIICHI_TOKEN
            bel_tp = bel_total = 0
            riichi_legal_count = 0
            riichi_human_count = 0
            riichi_pred_count = 0
            riichi_match_count = 0
            non_riichi_state_count = 0
            non_riichi_anchor_agree = 0
            n_val = 0
            with torch.no_grad():
                for vb in val_loader:
                    _, ld = trainer._forward_and_loss(vb)
                    for k in val_keys:
                        val_sums[k] += ld.get(k, 0.0)
                    # belief 召回率 + 立直看板 + 普通打牌稳定性看板
                    tokens = trainer._to_device(vb["tokens"])
                    candidate_mask = trainer._to_device(vb["candidate_mask"])
                    pad_mask = trainer._to_device(vb.get("pad_mask"))
                    belief_tokens = trainer._to_device(vb.get("belief_tokens"))
                    belief_pad_mask = trainer._to_device(vb.get("belief_pad_mask"))
                    s_out = trainer.model(
                        tokens=tokens,
                        candidate_mask=candidate_mask,
                        pad_mask=pad_mask,
                        belief_tokens=belief_tokens,
                        belief_pad_mask=belief_pad_mask,
                    )
                    if s_out.belief_logits is not None and vb.get("actual_hands") is not None:
                        ah   = trainer._to_device(vb["actual_hands"]).float()
                        pred = (s_out.belief_probs > 0.5).float()
                        tgt  = (ah > 0).float()
                        bel_tp    += (pred * tgt).sum().item()
                        bel_total += tgt.sum().item()

                    cand_region = tokens[:, -MAX_CANDIDATES_LEN:]
                    riichi_candidate_mask = (cand_region == RIICHI_TOKEN)
                    riichi_legal_mask = riichi_candidate_mask.any(dim=-1)
                    pred_action_idx = s_out.q.argmax(dim=-1)
                    pred_token = cand_region[torch.arange(pred_action_idx.shape[0], device=pred_action_idx.device), pred_action_idx]
                    human_action_idx = vb["action_idx"].to(trainer.device)
                    human_token = cand_region[torch.arange(human_action_idx.shape[0], device=human_action_idx.device), human_action_idx]
                    pred_riichi = (pred_token == RIICHI_TOKEN)
                    human_riichi = (human_token == RIICHI_TOKEN)
                    riichi_legal_count += int(riichi_legal_mask.sum().item())
                    riichi_human_count += int((riichi_legal_mask & human_riichi).sum().item())
                    riichi_pred_count += int((riichi_legal_mask & pred_riichi).sum().item())
                    riichi_match_count += int((riichi_legal_mask & human_riichi & pred_riichi).sum().item())

                    if trainer.oracle_model is not None:
                        anchor_out = trainer.oracle_model(
                            tokens=tokens,
                            candidate_mask=candidate_mask,
                            pad_mask=pad_mask,
                            belief_tokens=belief_tokens,
                            belief_pad_mask=belief_pad_mask,
                        )
                        non_riichi_state_mask = ~riichi_legal_mask
                        if non_riichi_state_mask.any():
                            anchor_pred_idx = anchor_out.q.argmax(dim=-1)
                            non_riichi_state_count += int(non_riichi_state_mask.sum().item())
                            non_riichi_anchor_agree += int(((pred_action_idx == anchor_pred_idx) & non_riichi_state_mask).sum().item())
                    n_val += 1
                    if n_val >= 50:
                        break
            trainer.model.train()
            n = max(n_val, 1)
            bel_recall = bel_tp / max(bel_total, 1)
            avgs = {k: val_sums[k] / n for k in val_keys}
            val_loss = avgs["total"]
            riichi_human_rate = riichi_human_count / max(riichi_legal_count, 1)
            riichi_pred_rate = riichi_pred_count / max(riichi_legal_count, 1)
            riichi_recall = riichi_match_count / max(riichi_human_count, 1)
            riichi_precision = riichi_match_count / max(riichi_pred_count, 1)
            non_riichi_agree = non_riichi_anchor_agree / max(non_riichi_state_count, 1)
            logger.info(
                f"[val step={step}] "
                + "  ".join(f"{k}={avgs[k]:.4f}" for k in val_keys if avgs[k] != 0.0)
                + f"  bel_recall={bel_recall:.3f}"
                + f"  riichi_legal={riichi_legal_count}"
                + f"  riichi_human_rate={riichi_human_rate:.3f}"
                + f"  riichi_pred_rate={riichi_pred_rate:.3f}"
                + f"  riichi_recall={riichi_recall:.3f}"
                + f"  riichi_precision={riichi_precision:.3f}"
                + (f"  nonriichi_anchor_agree={non_riichi_agree:.3f}" if non_riichi_state_count > 0 else "")
            )
            is_best_val = val_loss < best_val_loss
            if is_best_val:
                best_val_loss = val_loss
                trainer.save(ckpt_dir / "best_val.pt")
                # 把历史 best 指标写进 best_val.pt，供重启恢复
                best_val_ckpt = torch.load(ckpt_dir / "best_val.pt", map_location="cpu", weights_only=True)
                best_val_ckpt["best_val_loss"] = best_val_loss
                best_val_ckpt["best_gate_delta"] = best_gate_delta
                torch.save(best_val_ckpt, ckpt_dir / "best_val.pt")

            # best.pt 以 arena 为主，但 arena 开销大，按 arena_gate_every 节流
            # arena_gate_every=N 表示每 N 次 val 触发一次 gate（0=每次都跑）
            _arena_gate_every = int(trainer_cfg.arena_gate_every) if trainer_cfg.arena_gate_every > 0 else 1
            _val_count = step // val_every   # 已经到第几次 val
            _should_gate = cfg.get("arena_gate_games", 0) and (_val_count % _arena_gate_every == 0)
            if _should_gate:
                gate_ckpt = ckpt_dir / f"gate_eval_step{step}.pt"
                trainer.save(gate_ckpt)
                passed, gate_metrics = _arena_gate(cfg, gate_ckpt, step, ckpt_dir)
                gate_delta = gate_metrics.get("delta_rank", float("inf"))
                if passed and gate_delta < best_gate_delta:
                    best_gate_delta = gate_delta
                    trainer.save(ckpt_dir / "best.pt")
                    best_gate_ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
                    best_gate_ckpt["best_val_loss"] = best_val_loss
                    best_gate_ckpt["best_gate_delta"] = best_gate_delta
                    torch.save(best_gate_ckpt, ckpt_dir / "best.pt")
                    logger.info("[best step=%s] arena improved: delta_rank=%.4f", step, best_gate_delta)
                try:
                    gate_ckpt.unlink()
                except FileNotFoundError:
                    pass
            elif is_best_val and not cfg.get("arena_gate_games", 0):
                # 没有 arena gate 时，由 best_val 居个指导 best.pt
                trainer.save(ckpt_dir / "best.pt")
                best_ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
                best_ckpt["best_val_loss"] = best_val_loss
                best_ckpt["best_gate_delta"] = best_gate_delta
                torch.save(best_ckpt, ckpt_dir / "best.pt")

    logger.info("Stage 3 complete")
    trainer.save(ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
