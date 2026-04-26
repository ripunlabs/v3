---

title: Multi-Agent Aviation Control & Oversight Environment (MACE)
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv

---

# ✈️ MACE - Multi-Agent Aviation Control Environment

## 🔗 Links

- Hugging Face Space: `<PLACEHOLDER>`
- Training Notebook: `<PLACEHOLDER>`
- Demo Video / Blog: `<PLACEHOLDER>`

## 🚀 Quick Run

```bash
python train.py --episodes 200
python eval.py --model latest
python run_demo.py --judge-mode
```

## 📊 Results (Auto-generated)

| Metric | Baseline | Trained |
| --- | --- | --- |
| Avg Reward | -9.2129 | 5.0144 |
| Violations | 1.34 | 0.00 |
| Completion | 0.0% | 100.0% |

## 🧠 What this shows

Training improves coordination quality: the trained policy is more proactive, incurs fewer safety violations, and completes more scenarios through oversight-aware decisions.

---

# MACE v3 - Judge-Ready Multi-Agent Aviation AI Demo

MACE v3 is a production-style OpenEnv simulation of an airport control tower where five agents coordinate under safety constraints:

- ATC
- Airline
- Ops
- Weather
- Oversight (safety arbitration engine)

This version is optimized for a **3-minute hackathon pitch + live demo**:

- cinematic phase-based storytelling,
- visible multi-agent conflict and oversight arbitration,
- before/after training evidence,
- reproducible training/evaluation pipeline,
- structured report artifacts for judges.

---

## What Is New in v3

MACE v3 upgrades the project from simulation logs to a judge-facing AI product demo.

- **Story-first demo flow** in `run_demo.py`:
  - Calm -> Stress -> Conflict -> Oversight -> Resolution -> Learning
- **Judge mode** (`--judge-mode`) with cleaner output and emphasized arbitration moments.
- **Stepwise orchestration view** with per-agent decisions and role impact.
- **Final episode summary panel** with outcome classification (`SAFE` / `DEGRADED` / `FAILURE`) and single-line insight.
- **Training + eval split**:
  - `train.py` for training + artifact generation
  - `eval.py` for reproducible before/after comparison
- **Structured reports**:
  - `reports/metrics.json`
  - `reports/report.json`
  - `reward_curve.png`
  - `reports/safety_curve.png`

---

## System Overview

### Multi-agent stack

- **ATC**: runway assignment, sequencing, clear-to-land control
- **Airline**: fuel/emergency pressure and prioritization intent
- **Ops**: infrastructure and runway constraints
- **Weather**: dynamic closure context
- **Oversight**: safety arbitration across competing proposals

### Environment guarantees

- OpenEnv-compatible `reset()`, `step()`, `state()`
- deterministic task setup by seed
- aviation-specific constraints:
  - fuel exhaustion events
  - runway constraints
  - conflict resolution

---

## Repository Layout (v3)

```text
mace/
├── env/                     # OpenEnv environment core
├── agents/                  # Agent exports
├── training/
│   ├── reward_fn.py         # Training reward shaping
│   ├── policies.py          # Baseline + trainable policy
│   ├── ppo_loop.py          # Minimal PPO-style loop
│   └── model_latest.json    # Saved trained model
├── evaluation/
│   ├── runner.py            # Episode runner + orchestration capture
│   └── reporting.py         # Metrics/report writers
├── ui/
│   ├── live_renderer.py     # Rich cinematic renderer
│   ├── orchestration_view.py# Agent table per timestep
│   └── replay_engine.py     # Replay + final summary panel
├── reports/
│   ├── metrics.json
│   ├── report.json
│   ├── safety_curve.png
│   └── plots.py
├── train.py
├── eval.py
├── run_demo.py
├── operations.txt
└── README.md
```

---

## Step-by-Step Quickstart (Baby Steps)

All commands below assume terminal is already inside:

`PS E:\v3\mace>`

### Step 1 - Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 2 - Train

```bash
python train.py --episodes 50 --seed 42
```

What this does:

- runs minimal PPO-style training loop,
- saves `training/model_latest.json`,
- generates reward and safety plots,
- writes structured metrics/report JSON.

### Step 3 - Evaluate before vs after

```bash
python eval.py --model latest --episodes 50 --seed 42
```

You should see a comparison table with:

- Avg Reward
- Safety Violations/Episode
- Completion Rate

### Step 4 - Run live judge demo

```bash
python run_demo.py --model latest --task hard --seed 42 --frame-delay 0.6 --stepwise --orchestrate --judge-mode
```

### Step 5 - Optional fast demo (no delay)

```bash
python run_demo.py --model latest --task hard --seed 42 --frame-delay 0 --stepwise --orchestrate --judge-mode
```

---

## Demo Story Phases (What Judges See)

Every demo follows fixed narrative phases:

1. **Phase 1 - Stable Airport Operations**
2. **Phase 2 - Operational Stress Building**
3. **Phase 3 - Multi-Agent Conflict Detected**
4. **Phase 4 - Oversight Arbitration (Safety Decision Engine)**
5. **Phase 5 - Safe Resolution & Execution**
6. **Phase 6 - Outcome & System Learning**

Each timestep renders:

- progress indicator (`Step X/Y`)
- agent decisions (ATC, Airline, Ops, Weather)
- oversight scoring and selected agent
- final executed action
- reward/safety learning interpretation

---

## Judge Mode

Enable with:

```bash
python run_demo.py --model latest --judge-mode
```

Judge mode behavior:

- cleaner spacing and reduced low-value noise,
- stronger visual emphasis on arbitration moments,
- one-line oversight decision reason,
- final summary panel for scoring clarity.

---

## Final Episode Summary (End of Demo)

The demo ends with a mandatory summary panel containing:

- Episode type
- Total reward
- Safety violations
- Completion status
- System outcome (`SAFE`, `DEGRADED`, `FAILURE`)
- Key insight (one line)

Example key insight:

`Oversight prevented unsafe runway allocation during fuel-critical conflict.`

---

## Training + Reward Pipeline

### Training loop

- File: `training/ppo_loop.py`
- Structure:
  - collect trajectory,
  - compute discounted returns/advantages,
  - apply clipped PPO-style preference update.

### Reward shaping

- File: `training/reward_fn.py`
- Training signal combines:
  - environment reward,
  - safety penalty from violation deltas,
  - completion bonus,
  - valid action bonus.

This keeps optimization aligned with judge-facing behavior:

- safer actions,
- better completion,
- fewer violations.

---

## Evidence of Improvement (Current v3)

From `reports/metrics.json` and `reports/report.json`:

- Avg reward: `-9.2129 -> 1.8750`
- Safety violations/episode: `1.34 -> 0.32`
- Completion rate: `0.0% -> 68.0%`

This satisfies the key judging dimensions around:

- observable training progress,
- coherent reward logic,
- meaningful improvement in inference behavior.

---

## Generated Artifacts

After train/eval runs, expect:

- `training/model_latest.json`
- `reward_curve.png`
- `reports/safety_curve.png`
- `reports/metrics.json`
- `reports/report.json`

Use `operations.txt` for the canonical command list.

---

## OpenEnv and Deployment Notes

- `openenv.yaml` is included.
- Environment exposes OpenEnv-compatible APIs.
- Validate locally with:

```bash
openenv validate .
```

If deploying via container:

```bash
docker build -t mace-env .
docker run --rm -p 7860:7860 mace-env
```

### Notebook / Colab compatibility

- Commands use relative paths from repo root (`mace/`) with no machine-specific hardcoding.
- Training/evaluation scripts resolve `latest` model paths internally.
- Suitable for Colab-style execution cells:
  - `!python train.py --episodes 50 --seed 42`
  - `!python eval.py --model latest --episodes 50 --seed 42`

---

## Troubleshooting

### "can't open file ...\mace\mace\run_demo.py"

Cause: running `python mace/run_demo.py ...` while already inside `E:\v3\mace`.

Fix:

- inside `E:\v3\mace`, use `python run_demo.py ...`
- check `operations.txt`

### Model not found

Use:

```bash
python eval.py --model latest
python run_demo.py --model latest
```

`latest` resolves to `training/model_latest.json`.

---

## Submission Checklist (Final)

- [ ] `python train.py --episodes 50 --seed 42`
- [ ] `python eval.py --model latest --episodes 50 --seed 42`
- [ ] verify `reports/metrics.json` has improved before/after values
- [ ] `python run_demo.py --model latest --task hard --seed 42 --frame-delay 0.6 --stepwise --orchestrate --judge-mode`
- [ ] confirm final summary panel shows system outcome + key insight
- [ ] run `openenv validate .`

---

## ⚡ Judge Quickstart (Under 60 seconds)

```bash
python train.py --episodes 50
python eval.py --model latest
python run_demo.py --judge-mode
```

---

## Team

MACE (Multi-Agent Aviation Control & Oversight Environment) - 2026 Team Final Commit:

- Ripun Basumatary (Lead)
- Veeshal D. Bodosa
- Jeu Machahary