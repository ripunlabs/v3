# Building MACE v3: A Cinematic Multi-Agent AI Control Tower

When we started MACE, we were not trying to build another toy simulator.

We wanted to build something judges could *feel*.

Not just "the code runs."
Not just "we have a reward function."
But a system where you can watch multiple AI agents disagree during an airport crisis, see a safety engine arbitrate in real time, and clearly observe behavior improving after training.

That is what MACE v3 became.

---

## The problem we chose

Aviation control is one of those real-world domains where "good enough" is not enough.

At every moment, operations teams balance:

- runway capacity,
- arrival sequencing,
- fuel pressure,
- weather closures,
- emergency handling,
- and strict safety requirements.

This is exactly the kind of environment where multi-agent AI can be valuable and dangerous at the same time.

So MACE is intentionally designed as a **safety-critical, coordination-heavy benchmark**:

- ATC proposes operational actions,
- Airline pushes urgency,
- Ops enforces infrastructure limits,
- Weather injects external constraints,
- Oversight arbitrates safely when proposals conflict.

In short: this is not a scheduling game. It is a structured simulation of decision quality under pressure.

---

## What MACE v3 does in one line

**MACE v3 is a multi-agent OpenEnv airport control environment where a trained policy and oversight arbitration jointly turn conflict into safe execution under constraints.**

---

## Why we moved to "story-first"

Early versions had the classic issue many technical projects have:

the model might be working, but the output looked like logs.

For hackathon judging, that is risky.
Judges have limited time.
If they need 2 minutes just to decode what is happening, you lose points even if the internals are good.

So we redesigned the demo as a narrative with fixed phases:

1. Calm operations
2. Stress buildup
3. Multi-agent conflict
4. Oversight arbitration (climax)
5. Resolution
6. Learning outcome

Now, the first 10 seconds are enough for a judge to understand:

- where the conflict is,
- who proposed what,
- why oversight selected one action,
- and whether safety improved.

---

## System architecture (practical view)

### Core simulation layer

Located under `env/`.

This remains the deterministic aviation simulation engine:

- tasks (`easy`, `medium`, `hard`),
- runway/weather/fuel dynamics,
- action validation,
- shaped environment reward,
- OpenEnv `reset/step/state` contract.

### Training layer

Located under `training/`.

- `reward_fn.py`: training-side reward shaping pressure
- `policies.py`: baseline and trainable preference policy
- `ppo_loop.py`: minimal PPO-style update loop with exploration

### Evaluation layer

Located under `evaluation/`.

- `runner.py`: before/after rollouts and orchestration capture
- `reporting.py`: JSON metrics and judge-friendly summaries

### Demo/UX layer

Located under `ui/` and `run_demo.py`.

- phase-rendered live output,
- arbitration tables,
- oversight highlight,
- final episode summary panel,
- optional baseline-vs-trained comparison mode.

---

## Training: what changed and why it finally learned

The biggest turning point was fixing incentive design.

We diagnosed a common failure mode:

- weak penalties on inactivity,
- insufficient positive signal for meaningful action,
- poor learning pressure under safety constraints.

### The reward learning signal now strongly encodes:

**Positive pressure**
- successful landings,
- conflict reduction,
- meaningful control actions (assignment/sequencing),
- emergency handling.

**Negative pressure**
- safety violations (strong penalty),
- fuel-critical risk,
- invalid actions,
- no-op under active traffic,
- delay worsening.

We also added:

- epsilon-greedy exploration in training,
- online per-step preference updates,
- plus PPO-style trajectory updates.

This created behavior pressure toward proactive, safer decisions.

---

## Observable improvement (the most important proof)

From `reports/metrics.json` and `reports/report.json` (latest run):

| Metric | Baseline | Trained |
| --- | --- | --- |
| Avg Reward | -9.2129 | 5.0144 |
| Safety Violations / Episode | 1.34 | 0.00 |
| Completion Rate | 0.0% | 100.0% |

Improvement summary:

- Reward improved by **+14.23**
- Violations reduced by **-1.34**
- Completion improved by **+100%**

This is exactly what we wanted for judging:

clear before-vs-after, not ambiguous progress.

---

## Demo experience: what judges see

Run:

```bash
python run_demo.py --model latest --task hard --seed 42 --stepwise --orchestrate --judge-mode
```

What appears:

- colored phase headers,
- step progress indicator,
- per-agent decisions each timestep,
- oversight scoring + selected proposal,
- one-line arbitration reason,
- final action execution and reward update,
- final summary panel with outcome and key insight.

If you want quick visible contrast:

```bash
python run_demo.py --model latest --judge-mode --compare-baseline
```

This first prints a baseline snapshot, then runs the full trained cinematic demo.

---

## Why oversight is the hero in MACE

In many multi-agent systems, arbitration is hidden.
In MACE v3, arbitration is explicit and central.

We made this a deliberate design decision because safety is not just a metric, it is the product.

So the demo always makes it visible:

- competing proposals,
- per-agent scores,
- selected agent highlight,
- readable reason text ("Selected because ...").

This turns oversight from backend logic into a judge-visible safety decision engine.

---

## Reproducibility and judge confidence

All core runs are deterministic by seed and generate structured artifacts.

### Standard run flow

```bash
python train.py --episodes 200 --seed 42
python eval.py --model latest --episodes 50 --seed 42
python run_demo.py --model latest --judge-mode
```

### Generated artifacts

- `training/model_latest.json`
- `reward_curve.png`
- `reports/safety_curve.png`
- `reports/metrics.json`
- `reports/report.json`

This gives both:

- visual proof (plots),
- machine-readable proof (JSON),
- live behavioral proof (demo).

---

## Submission packaging mindset

For hackathon finals, we treated packaging as part of engineering quality.

That means:

- README starts with links and quick run,
- metrics are top-level and readable,
- demo explains itself without narration,
- commands are context-safe (`operations.txt`),
- no dependence on hidden local paths.

The target was simple:

> A judge should understand value in one shot.

---

## Lessons learned

1. **Reward design decides behavior quality**
   - If no-op is cheap, models stall.

2. **Visibility is not optional**
   - If improvement is not obvious, it will be treated as absent.

3. **Narrative beats raw logs**
   - Story-driven phase output dramatically improves comprehension.

4. **Safety must be explicit**
   - Arbitration has to be seen, not inferred.

5. **Before/after evidence closes trust gap**
   - Tables + plots + live behavior change are the strongest combo.

---

## What we are proud of in v3

- We did not hide behind "model complexity."
- We made behavior legible.
- We made safety visible.
- We made improvement measurable.

MACE v3 now behaves like a real control-tower AI demo:

conflict appears, oversight resolves it, and trained behavior is observably better.

That was the goal from day one.

---

## Quick links reminder (fill before final submission)

- Hugging Face Space: `<PLACEHOLDER>`
- Training Notebook: `<PLACEHOLDER>`
- Demo Video / Blog: `<PLACEHOLDER>`

