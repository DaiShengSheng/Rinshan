# Rinshan

*"Rinshan Kaihou. The winning tile from where you least expect it."*

Rinshan is a training framework for 4-player Japanese Riichi Mahjong AI, built around explicit incomplete-information modeling and a four-stage curriculum. It is a personal research project, not affiliated with any game platform or corporation.

This repository provides the model architecture, training pipeline, self-play engine, and data processing tools. It does **not** include training data or pretrained weights.

---

## Design

The central research question Rinshan addresses is: **how should a Mahjong AI reason under partial observability without oracle access?**

Most existing approaches either ignore the incomplete-information structure entirely (treating Mahjong as if all hands were visible) or handle it implicitly through large-scale behavioral cloning. Rinshan instead models uncertainty explicitly at inference time via a dedicated Belief Network, and uses a learned dense reward signal to make offline RL tractable.

### Belief Network

At every decision point, only a fraction of the game state is observable — the current player cannot see opponents' hand tiles. Rinshan addresses this with a Belief Network that runs alongside the policy: it takes all public information (discards, melds, riichi declarations, tile counts) and produces a probability distribution over each opponent's concealed hand tiles. This belief vector is injected into the policy encoder at every step, giving the policy a structured summary of what is unknown.

This is architecturally distinct from oracle distillation approaches, which train a full-information teacher and distill into a partial-information student. Rinshan's belief module operates entirely from observable information and is jointly trainable with the policy.

### Policy Transformer

The main policy network is a decoder-style Transformer following the LLaMA design (RoPE positional encoding, RMSNorm, SwiGLU activations, Grouped Query Attention). The input sequence encodes the full observable game history as discrete tokens: meta information (seat wind, round, scores), dora indicators, hand tiles, melds, and the progression of all players' discards and declarations. The belief vector from the Belief Network is concatenated as an additional context embedding.

The architecture scales across three presets:

| Preset | Hidden dim | Layers | Heads | Parameters |
|--------|-----------|--------|-------|------------|
| nano   | 256        | 4      | 4     | ~6.7M      |
| base   | 768        | 12     | 12    | ~81M       |
| large  | 1536       | 24     | 16    | ~500M      |

### Action Space

The action space contains 241 discrete tokens, covering all strategically distinct decisions:

- **Discard**: 37 tokens (34 tile types + 3 aka-dora variants)
- **Chi**: 63 tokens (3 suits × 7 low positions × 3 forms)
- **Pon / Ankan / Kakan / Daiminkan**: 34 tokens each
- **Special**: Riichi, Tsumo agari, Ron agari, Ryuukyoku, Pass

Candidate tokens are masked per decision point so the model only picks from the legal set.

### QV Head and Mixed Strategy

The output head uses a Dueling Q/V architecture. Rather than outputting a deterministic argmax policy, it samples from a Top-p distribution over legal actions weighted by Q-values. This mixed strategy is more appropriate for the adversarial, partially observable setting of Mahjong — a deterministic policy is exploitable once opponents model it.

Auxiliary heads are trained jointly to predict: shanten number, tenpai probability, deal-in risk per tile, and opponent tenpai state. These serve as auxiliary supervision and as interpretable diagnostic signals.

### GRP: Game Result Predictor

Offline RL in Mahjong faces a sparse reward problem — terminal outcomes (rank, score) only arrive at game end, hundreds of steps after most decisions. Rinshan trains a separate LSTM-based Game Result Predictor (GRP) on annotated game records to predict the expected terminal rank distribution from any mid-game state. The GRP output is used as a dense per-step reward signal during Stage 3 training.

The GRP is a learned module, not a hand-crafted reward shaping function. Its predictions are bounded to the human-data distribution, which is a known limitation for Stage 3 but a deliberate trade-off against the complexity of reward engineering.

### Mahjong Engine

Rinshan includes a pure-Python 4-player Mahjong game engine implementing full Tenhou Phoenix table rules:

- Red dora (one each for 5m / 5p / 5s)
- Riichi, ippatsu, chankan, rinshan kaihou
- All abortive draws: four-wind discard, four-kan, four-player riichi, kyushu kyuhai
- Nagashi mangan
- Correct furiten tracking (permanent, riichi, and same-junme)
- Deterministic seeded RNG for reproducibility

The engine drives the self-play scheduler (`Arena`) which runs multiple games concurrently and feeds transitions into the online training buffer.

---

## Training Pipeline

Rinshan uses a four-stage curriculum. Records are first converted to mjai-compatible JSON, annotated into decision-point samples, and back-filled with GRP reward signals before training begins.

**Stage 1 — Behavioral Cloning**
Supervised learning on high-level human game records. Trains a base policy that imitates expert play. Convergence is fast and stable; this stage serves as initialization for all subsequent stages.

**Stage 2 — Oracle Distillation**
A full-information teacher observes the complete game state; a partial-information student learns to match the teacher's decisions given only observable information. When full-information data is unavailable, the stage falls back to self-distillation using the Stage 1 model as a soft teacher.

**Stage 3 — Offline IQL**
Implicit Q-Learning on the annotated dataset with GRP-derived dense rewards. The online model is trained against an EMA target network. Conservative Q-Learning (CQL) regularization prevents value overestimation on out-of-distribution actions.

**Stage 4 — Online Self-Play**
Agents play against themselves and past checkpoints, updating from real game outcomes rather than GRP estimates. A `League` pool of historical checkpoints is maintained to prevent strategy collapse.

---

## Setup

```bash
git clone https://github.com/DaiShengSheng/Rinshan.git
cd Rinshan
pip install -e .
```

---

## Roadmap

- [ ] Stage 1: Behavioral Cloning (nano + base configs)
- [ ] Stage 2: Oracle Distillation — framework complete; full-information data dependent
- [ ] Stage 3: Offline IQL — framework complete; base-scale convergence in progress
- [ ] Pure Python Mahjong Engine & Rust Mahjong Engine(full Tenhou rules, seeded RNG)
- [ ] Stage 4: Online Self-Play + League training — framework complete; base-scale convergence pending
- [ ] Inference server / live game interface
- [ ] Evaluation tooling and benchmark comparisons

---

## Acknowledgements

[Mortal](https://github.com/Equim-chan/Mortal) and [Kanachan](https://github.com/Cryolite/kanachan) are the primary prior works this project builds on and differs from. The mjai event format is used as the internal game representation standard.

---

## License

MIT. See [LICENSE](LICENSE).
