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

# Multi-Agent Aviation Control & Oversight Environment (MACE)

MACE is a benchmark-grade **OpenEnv** environment for aviation operations control. It evaluates AI agents on a real-world, safety-critical task through the standard `step()`, `reset()`, and `state()` API.

The simulator covers runway allocation, arrival sequencing, delay management, weather-aware runway usage, fuel-critical prioritization, and emergency coordination. It is **deterministic and reproducible** - suitable for agent evaluation, reward-shaping research, and OpenEnv-compliant deployment.

Unlike toy scheduling games, MACE is grounded in operational pressure: agents must respect safety and capacity constraints; poor decisions can create conflicts, amplify delays, misuse closed infrastructure, or cause fuel-critical failures. That aligns with the requirements for environments that mirror work humans actually do.

---

## Table of contents

1. Motivation
2. Why this problem matters
3. OpenEnv hackathon fit
4. Environment overview
5. Action space
6. Observation space
7. Reward function
8. Tasks and difficulty progression
9. Graders and evaluation logic
10. Episode boundaries and termination
11. Repository structure
12. Setup
13. Running the environment
14. Running inference
15. Baseline results
16. Docker
17. OpenEnv compliance
18. Hugging Face Spaces
19. Design decisions
20. Limitations
21. Why MACE is novel
22. Copyright

---

## Motivation

Modern aviation operations require continuous, high-stakes decisions under limited time, incomplete flexibility, and strict safety rules. Human controllers and operations teams constantly balance:

- which runway a flight should use,
- which arrival should be sequenced first,
- whether a delay or hold is safer than immediate clearance,
- whether deteriorating weather makes a runway unusable,
- whether a low-fuel or emergency flight must be prioritized,
- and how to maintain throughput without violating operational constraints.

Despite the importance of these decisions, there are very few benchmarks that let AI agents be evaluated on this kind of work in a standardized, reproducible, programmatically graded environment.

MACE was built to fill that gap.

---

## Why this problem matters

Aviation decision-making is one of the clearest examples of a real-world control problem where performance is not measured by a single objective. Good decisions must balance:

- Safety
- Efficiency
- Throughput
- Responsiveness to disruptions
- Robustness under constrained resources

This makes it an excellent benchmark domain for evaluating agent quality.

MACE is relevant to workflows such as:

- Air Traffic Control (ATC)
- Airport runway and terminal-area operations
- Airline Operations Control (AOC)
- Arrival management and disruption handling systems
- Decision-support tooling for safety-critical logistics

Potential use cases include:

- evaluating AI copilots for traffic prioritization,
- benchmarking conflict-aware sequencing policies,
- stress-testing runway allocation strategies,
- simulating weather and congestion recovery logic,
- and studying safety-first agent behavior in constrained operational systems.

---

## OpenEnv hackathon fit

MACE is explicitly designed to satisfy the Round 1 OpenEnv Hackathon requirements.

### Real-world task simulation

The environment simulates an operational task humans genuinely perform in aviation systems. It is not a game, toy, or abstract synthetic puzzle.

### Full OpenEnv spec compliance

MACE implements:

- typed Observation, Action, and Reward models,
- `step(action)` returning an `AACEObservation` that carries the next state, scalar `reward`, boolean `done`, `last_action_valid` / `last_action_message`, and `metadata` (not a separate Gym-style `info` dict),
- `reset()` returning a clean initial observation,
- `state()` returning the current environment state,
- an `openenv.yaml` manifest,
- and validation via `openenv validate`.

### Minimum 3 tasks with graders

MACE includes three deterministic tasks. In code they are selected with `reset(..., task_id=...)` (or `AACE_TASK` for the server default). `inference.py` labels them **EASY / MEDIUM / HARD** with the titles below; graders live in `env/graders/` and are wired in `env/graders/__init__.py` as `GRADERS[task_id]`.

#### `easy` - Efficient Landing Scheduling

- **Grader:** `grade_easy` (`env/graders/grader_easy.py`).
- **Level:** Introductory. Teaches the control loop without weather stress.
- **Scenario (`env/tasks/task_easy.py`):** three inbound flights (`AC001`–`AC003`), two parallel runways **R1** and **R2**, good visibility, no weather closures. Horizon `**max_steps` = 36**.
- **What you must handle:** runway assignment, approach sequencing, `clear_to_land`, and keeping delays/violations low. Baseline policy should complete in a small number of steps.

#### `medium` - Weather-Constrained Runway Operations

- **Grader:** `grade_medium` (`env/graders/grader_medium.py`).
- **Level:** Intermediate. Forces reading `weather.closed_runways` every step and replanning when capacity returns.
- **Scenario (`env/tasks/task_medium.py`):** four arrivals (`AC010`–`AC013`), same two runways. Runway **R2** is on a **deterministic closure schedule** (closed for early steps, then reopens). Horizon `**max_steps` = 44**.
- **What you must handle:** avoid assigning or clearing to closed runways, re-sequence when R2 is unavailable, and balance safety vs delay when both runways are usable.

#### `hard` - Emergency Conflict Resolution

- **Grader:** `grade_hard` (`env/graders/grader_hard.py`).
- **Level:** Advanced. Congestion plus **pre-declared emergency** and **multiple fuel-critical** aircraft under intermittent runway loss.
- **Scenario (`env/tasks/task_hard.py`):** five flights (`AC020`–`AC024`). **AC020** starts in emergency with very low fuel; **AC023** and **AC024** are fuel-critical. Runway **R1** is closed during a fixed mid-episode window (steps **4–9**). Task config sets `emergency_flight_id`, `legitimate_emergency_ids`, and `priority_flight_ids` for grading. Horizon `**max_steps` = 48**.
- **What you must handle:** prioritize emergency and fuel-critical traffic, survive the R1 outage without `fuel_exhaustion`, and still land or divert everyone safely.

Each task has a concrete `objective_summary` on `TaskConfig` (surfaced in observations) and is scored in **[0.0, 1.0]** by its grader on the terminal `EnvironmentState`.

### Meaningful reward function

Rewards are shaped across the full episode, not only at the terminal state. Partial progress is rewarded, harmful behavior is penalized, and safety dominates optimization.

### Baseline inference script

The project includes a root-level `inference.py` that supports reproducible evaluation and reads required inference configuration from environment variables.

### Deployment readiness

The repository includes:

- a working Dockerfile,
- Hugging Face Space compatibility,
- and OpenEnv-compliant serving.

---

## Environment overview

MACE models terminal-area aviation control as a sequential decision-making environment.

At each step, the agent receives an observation describing the current state of traffic, runways, weather conditions, operational alerts, and performance metrics. It must issue an action that changes or maintains the operational plan. The simulator then updates the world state deterministically and returns:

- the next observation,
- a shaped reward,
- whether the episode is done,
- and structured metadata about the transition.

The environment abstracts aviation control at the level of decision logic, not full aerodynamics. This is intentional. The goal is to benchmark agent reasoning about operational constraints, prioritization, and control sequencing rather than low-level flight physics.

### Core system components


| Component          | Description                                                                                                             |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| **Flight Engine**  | Simulates aircraft states, ETA progression, delay accumulation, fuel burn, approach progression, and landing completion |
| **Runway Manager** | Tracks runway availability, closures, occupancy windows, and assignment conflicts                                       |
| **Weather Engine** | Applies visibility and weather-driven runway restrictions through deterministic schedules                               |
| **Risk Engine**    | Measures conflict pressure, unsafe sequencing, and fuel-critical risk                                                   |
| **Reward Engine**  | Computes trajectory-level reward with safety, efficiency, and progress components                                       |
| **Task Engine**    | Loads deterministic scenarios with fixed initial conditions and difficulty progression                                  |
| **Grader Engine**  | Scores completed episodes using deterministic aviation-relevant criteria                                                |


*These names describe responsibilities implemented across `env/` modules (`transitions.py`, `metrics.py`, `reward.py`, `tasks/`, `graders/`, etc.), not separate runtime services or class names.*

---

## Action space

The environment uses a typed action model, typically represented as `AACEAction`.

Each step consists of one discrete operational decision. Actions are validated before state transitions are applied, and invalid actions are penalized.

### Supported actions


| Action Kind         | Purpose                                                 | Required Fields               |
| ------------------- | ------------------------------------------------------- | ----------------------------- |
| `noop`              | Make no operational change for the current step         | none                          |
| `assign_runway`     | Assign a runway to a specific flight                    | `flight_id`, `runway_id`      |
| `sequence_landing`  | Reorder the approach queue for incoming flights         | `ordered_flight_ids`          |
| `hold_pattern`      | Keep a flight in holding for a limited number of rounds | `flight_id`, `hold_rounds`    |
| `delay_flight`      | Apply tactical delay before approach progression        | `flight_id`, `delay_steps`    |
| `reroute_flight`    | Divert or reroute a flight to an alternate plan         | `flight_id`, `alternate_code` |
| `declare_emergency` | Mark a flight as emergency traffic when appropriate     | `flight_id`                   |
| `clear_to_land`     | Issue landing clearance to a selected flight            | `flight_id`                   |


### Action design rationale

The action space was designed to reflect decisions that are:

- operationally meaningful,
- understandable to both humans and agents,
- constrained enough to grade deterministically,
- and expressive enough to create genuine strategy differences.

This supports the hackathon's environment-design criteria by keeping the control surface realistic while avoiding unnecessary ambiguity.

---

## Observation space

The observation model, typically represented as `AACEObservation`, exposes a structured snapshot of the current terminal-control state.

### Observation contents


| Field Group             | Description                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Identity & Progress** | `task_id`, `step_index`, `max_steps`, `simulation_minutes`, plus `done` and scalar `reward` from the OpenEnv observation base |
| **Flights**             | Per-flight state including phase, ETA, fuel, delay, emergency flags, runway assignment, and status                            |
| **Runways**             | Open/closed state, current occupancy, capacity, and runway metadata                                                           |
| **Weather**             | Visibility and weather-driven runway restrictions, including closed runway signals                                            |
| **Approach Queue**      | Current landing order for active arrivals                                                                                     |
| **Alerts**              | Structured operational warnings and event flags                                                                               |
| **Metrics**             | Flattened KPIs such as safety, conflict risk, delay, utilization, and fuel risk                                               |
| **Transition Feedback** | `last_action_valid`, `last_action_message`, and other execution feedback                                                      |


### Why the observation space is useful

The observation model is designed so that:

- an agent has enough information to act sensibly,
- grader behavior is explainable relative to observable state,
- reward shaping remains aligned with what the agent can see,
- and actions can be interpreted in a human-auditable way.

This matters for both evaluation quality and hackathon judging around clean environment design.

---

## Reward function

MACE uses a dense, meaningful reward function rather than sparse terminal-only scoring.

The reward is modeled as a typed structure with a total score and component signals such as:

- safety
- efficiency
- progress

### Reward philosophy

The reward function is intended to guide agents toward behavior that is operationally sensible, not just grader exploitation.

It rewards:

- safe runway usage,
- orderly sequencing,
- successful landing progression,
- prioritization of emergency or fuel-critical traffic,
- productive advancement toward task completion.

It penalizes:

- invalid actions,
- runway misuse,
- unsafe sequencing,
- unnecessary delay,
- repeated passivity,
- and failure modes such as fuel exhaustion.

### Important shaping properties

- Rewards are provided throughout the episode, not only at the end.
- Partial progress is explicitly recognized.
- Repeated no-op behavior is penalized when traffic remains active.
- Safety-oriented shaping dominates opportunistic throughput gains.
- Emergency and fuel-critical resolution provide additional positive signal.

This directly addresses the hackathon requirement for a meaningful reward function with partial progress signals.

---

## Tasks and difficulty progression

MACE includes three benchmark tasks with clear progression from easier coordination to harder safety-critical prioritization.

### Easy task

A small arrival set with stable conditions and multiple usable runways.

**Objective:** Learn the core loop of assigning runways, sequencing arrivals, and issuing landing clearances without creating violations.

**Why it matters:** This task verifies that an agent can understand the basic mechanics of the environment.

**As implemented:** three arrivals on two runways (R1/R2), good weather, `max_steps` 36 (`env/tasks/task_easy.py`).

### Medium task

A more complex arrival set with deterministic weather disruption and runway closure windows.

**Objective:** Maintain safe and efficient traffic flow while adapting to temporary infrastructure loss.

**Why it matters:** This task tests whether the agent actually reads and reacts to changing constraints instead of following a static policy.

**As implemented:** four arrivals; runway R2 is closed by weather for early steps via a deterministic schedule, `max_steps` 44 (`env/tasks/task_medium.py`).

### Hard task

A high-pressure scenario with emergency traffic, multiple fuel-critical flights, and intermittent runway availability changes.

**Objective:** Preserve safety while prioritizing urgent traffic and preventing catastrophic outcomes.

**Why it matters:** This task is meant to challenge stronger models and satisfy the hackathon requirement that harder tasks meaningfully stress capable agents.

**As implemented:** five arrivals including one pre-declared emergency and two fuel-critical aircraft; runway R1 is intermittently closed by weather, `max_steps` 48 (`env/tasks/task_hard.py`).

### Difficulty progression summary


| Task   | Focus                                  | Main difficulty driver                          |
| ------ | -------------------------------------- | ----------------------------------------------- |
| Easy   | Basic sequencing and runway assignment | Learning valid control flow                     |
| Medium | Weather-aware control                  | Adapting to changing runway constraints         |
| Hard   | Safety-critical prioritization         | Emergencies, low fuel, and constrained capacity |


All tasks are deterministic and reproducible under a fixed seed.

---

## Graders and evaluation logic

Each task is paired with a deterministic programmatic grader that returns a score from 0.0 to 1.0.

The graders are designed to satisfy the hackathon's requirements for:

- clear success and failure criteria,
- reproducibility,
- meaningful differentiation across easy, medium, and hard,
- and resistance to trivial exploitation.

### Grading principles

- **Safety dominates** - Unsafe policies should never receive strong scores purely from throughput.
- **Mission completion matters** - Successfully resolving all traffic contributes positively, but unsafe completion is not rewarded equally.
- **Efficiency matters secondarily** - Delay and runway usage quality affect the score, but they do not override safety.
- **Task-specific criteria are explicit** - Weather discipline, emergency management, and survival of priority flights are graded where relevant.

### Example graded dimensions

- safety compliance,
- runway misuse,
- conflict pressure,
- delay performance,
- successful resolution of emergency traffic,
- handling of fuel-critical flights,
- mission completion quality.

### Determinism

The graders depend on deterministic environment state and task configuration. They do not rely on hidden stochastic behavior. This is important for both local testing and automated validation.

---

## Episode boundaries and termination

Episodes terminate when one of the defined conditions is met.

### Terminal conditions


| Terminal reason    | Meaning                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| `mission_complete` | All active arrivals have been resolved through landing or controlled rerouting |
| `max_steps`        | The task horizon has been reached                                              |
| `fuel_exhaustion`  | An active aircraft exceeds the fuel failure threshold                          |


### Why this boundary design is sensible

The episode logic is designed to reflect real operational outcomes:

- successful completion,
- failure due to exhaustion of time budget,
- or catastrophic operational breakdown.

This aligns with the hackathon evaluation criterion around proper episode boundaries.

---

## Repository structure

Source layout (omits generated dirs such as `.pytest_cache/`, `aace.egg-info/`, and local `.env`):

```text
aace/
├── env/
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # AACEConfig, reward/simulation defaults
│   ├── environment.py              # AACEEnvironment: reset(), step(), state()
│   ├── inference_policy.py         # Scripted baseline: deterministic action choice
│   ├── metrics.py                  # KPI dict for observations (safety, delay, fuel risk, etc.)
│   ├── models.py                   # Pydantic models: actions, observations, state, tasks
│   ├── reward.py                   # Per-step shaped reward (safety / efficiency / progress)
│   ├── transitions.py              # apply_control_action, advance_physics, weather, validation
│   ├── utils.py                    # Shared helpers
│   ├── tasks/
│   │   ├── __init__.py             # TASK_BUILDERS: task_id -> build_*_state(seed)
│   │   ├── task_easy.py            # Easy scenario initial state
│   │   ├── task_medium.py          # Medium scenario (weather closure schedule)
│   │   └── task_hard.py            # Hard scenario (emergency, fuel-critical, R1 closures)
│   └── graders/
│       ├── __init__.py             # GRADERS map: task_id → grade_* function
│       ├── common.py               # Shared deterministic grading helpers
│       ├── grader_easy.py          # Easy task score in [0, 1]
│       ├── grader_medium.py        # Medium task score in [0, 1]
│       └── grader_hard.py          # Hard task score in [0, 1]
├── server/
│   ├── __init__.py
│   └── app.py                      # FastAPI app via openenv create_app; uvicorn entry (server)
├── tests/
│   ├── test_contract.py            # OpenEnv-style: reset, step, state, task selection
│   ├── test_env.py                 # Dynamics, safety rules, rewards, terminal semantics
│   ├── test_graders.py             # Grader ranges; good vs bad outcomes; task stress
│   └── test_tasks.py               # Builder determinism; weather schedules; hard-task shape
├── inference.py                    # CLI: scripted or LLM rollouts across all tasks
├── app.py                          # HF ``app_file``, ``openenv.yaml`` ``app:app``, Docker CMD (wraps ``server.app``)
├── openenv.yaml                    # OpenEnv space manifest (name, runtime, app, port)
├── Dockerfile                      # CPU image: pip install -e ., uvicorn on 7860
├── .dockerignore                   # Excludes .env, venv, tests, caches from image build
├── pyproject.toml                  # Package metadata, deps, [project.scripts] server
├── requirements.txt                # Runtime deps only (matches pyproject); dev: pip install -e ".[dev]"
├── uv.lock                         # uv lockfile (optional; present if using uv)
├── .env.example                    # Template for LLM inference env vars
├── .gitignore
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+
- Docker
- OpenEnv tooling
- Optional Hugging Face CLI for deployment

### Local installation

```bash
cd aace
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -e ".[dev]"
pytest -q
```

### Validate the environment

```bash
openenv validate .
```

This should confirm that the project structure, typed models, and OpenEnv manifest are valid.

---

## Running the environment

### Start the server locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Equivalent: `uvicorn server.app:app` (same `FastAPI` instance). After `pip install -e .`, you can also run the console script:

```bash
server
```

If you use [uv](https://github.com/astral-sh/uv), `uv run server` is equivalent once the project environment is synced.

The server listens on **port 7860** (see `openenv.yaml`). To hit a running instance with the CLI validator, use e.g. `openenv validate --url http://127.0.0.1:7860`.

Once running, the environment should respond through the OpenEnv-compatible API surface and support the expected reset/step/state flow.

---

## Running inference

The hackathon expects a root-level `inference.py` that can call an OpenAI-compatible chat API and read configuration from the environment. This project uses `python-dotenv`: if a `.env` file exists in the `aace/` directory, it is loaded first, then normal process environment variables override.

**Never commit `.env`** (it usually contains `HF_TOKEN`). Start from `.env.example` and copy to `.env` locally.

### Example `.env` layout

Match the template in `.env.example` (values below are illustrative; use your own model id and token for LLM runs):

```dotenv
# --- Mode (pick one) ---
AACE_INFERENCE_MODE=scripted
# AACE_INFERENCE_MODE=llm
# --- Terminal output ---
AACE_OUTPUT_MODE=compact
# AACE_OUTPUT_MODE=verbose
# --- LLM endpoint (required when AACE_INFERENCE_MODE=llm) ---
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=
HF_TOKEN=
# Optional: larger model - change MODEL_NAME only, no code edits
# MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
# Smaller / often faster for smoke tests on hosted routers:
# MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
# --- Reproducibility (both modes) ---
AACE_INFERENCE_SEED=42
# --- LLM reliability (llm mode only) ---
AACE_LLM_MAX_RETRIES=3
AACE_LLM_RETRY_BACKOFF_MS=750
AACE_LLM_TIMEOUT_S=120
```

### Configuration reference


| Variable                    | Used in | Default / empty behavior                                           |
| --------------------------- | ------- | ------------------------------------------------------------------ |
| `AACE_INFERENCE_MODE`       | Both    | `scripted` (no API calls). Set to `llm` to use the chat model.     |
| `AACE_OUTPUT_MODE`          | Both    | `compact` table summary. `verbose` prints per-task detail blocks.  |
| `AACE_INFERENCE_SEED`       | Both    | Passed to `reset()`; default **42** if unset (see `inference.py`). |
| `API_BASE_URL`              | `llm`   | Must be non-empty for LLM mode (OpenAI-compatible base URL).       |
| `MODEL_NAME`                | `llm`   | Must be non-empty for LLM mode.                                    |
| `HF_TOKEN`                  | `llm`   | API key for the OpenAI client (e.g. Hugging Face router token).    |
| `AACE_LLM_MAX_RETRIES`      | `llm`   | Max completion attempts per step (default **3**).                  |
| `AACE_LLM_RETRY_BACKOFF_MS` | `llm`   | Initial backoff in ms, then doubles (default **750**).             |
| `AACE_LLM_TIMEOUT_S`        | `llm`   | HTTP timeout seconds (default **120**).                            |


In `**scripted`** mode, `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN` are not required; the policy in `env/inference_policy.py` drives actions.

### Variants

**Inference mode**

- `**scripted`** - Deterministic baseline, no credentials. Use for official reproducible scores and CI.
- `**llm`** - Each step asks the model for a JSON action; parse/validate against `AACEAction`. Retries and timeouts use the `AACE_LLM_`* variables.

**Output mode**

- `**compact`** - Single summary table after all three tasks.
- `**verbose`** - Per-task sections with more step-level detail.

**Shell equivalents (no `.env`)**

Scripted:

```powershell
$env:AACE_INFERENCE_MODE = "scripted"
$env:AACE_OUTPUT_MODE = "compact"
$env:AACE_INFERENCE_SEED = "42"
python inference.py
```

LLM (PowerShell):

```powershell
$env:AACE_INFERENCE_MODE = "llm"
$env:AACE_OUTPUT_MODE = "compact"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "your-org/your-model"
$env:HF_TOKEN = "your-secret-token"
python inference.py
```

Bash (LLM):

```bash
export AACE_INFERENCE_MODE=llm
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=your-org/your-model
export HF_TOKEN=your-secret-token
python inference.py
```

### Notes

- **Hackathon checklist:** for LLM runs, `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are the credentials the instructions expect.
- Scripted runs are repeatable; LLM runs can differ across models, temperature, and API behavior.
- Full behavior and defaults are documented in the module docstring at the top of `inference.py`.

---

## Training, evaluation, and demo

MACE now includes a clean RL-ready pipeline split into training, evaluation, and judge-facing demo modes:

```bash
python train.py --episodes 50
python eval.py --model latest
python run_demo.py --frame-delay 0.6 --stepwise --orchestrate
```

### Outputs generated automatically

- `training/model_latest.json` (PPO-style trained policy preferences)
- `reports/metrics.json` (structured metrics including before-vs-after table)
- `reports/report.json` (pitch-ready summary payload)
- `reward_curve.png` and `reports/safety_curve.png` (reward/safety visualization)

### Live demo view

`run_demo.py` provides timestep orchestration replay with:

- sync/loading animation between steps,
- ATC / Airline / Ops / Weather decisions,
- oversight arbitration highlight,
- final selected environment action,
- per-step reward updates with safety signals.

---

## Baseline results

The current deterministic scripted baseline produces the following scores (reproduced with `AACE_INFERENCE_MODE=scripted` and default `AACE_INFERENCE_SEED=42`):


| Task   | Score  |
| ------ | ------ |
| Easy   | 0.9452 |
| Medium | 0.9649 |
| Hard   | 0.9327 |
| Mean   | 0.9476 |


Additional run diagnostics from the current benchmark execution show:

- Total steps: 38
- Invalid actions: 0
- Fallbacks: 0

These results are based on the current benchmark outputs in your project materials.

### Why deterministic baseline scores matter

The hackathon explicitly checks that the baseline script reproduces. Deterministic scripted runs provide:

- stable verification across environments,
- clear regression detection,
- and reproducible benchmark numbers for local and hosted evaluation.

---

## Docker

The environment includes a working Dockerfile for containerized execution.

### Build

```bash
docker build -t aace-env .
```

### Run

```bash
docker run --rm -p 7860:7860 aace-env
```

This is necessary for both local validation and Hugging Face Spaces deployment.

---

## OpenEnv compliance

MACE is designed to satisfy the OpenEnv interface and packaging requirements.

### Compliance checklist

- typed action model,
- typed observation model,
- typed reward model,
- `step(action)` interface,
- `reset()` interface,
- `state()` interface,
- `openenv.yaml` manifest,
- deterministic task definitions,
- grader outputs in the range 0.0–1.0,
- baseline inference script,
- container support.

### Validation target

```bash
openenv validate .
```

This should pass before submission.

---

## Hugging Face Spaces

MACE is structured for deployment as a containerized Hugging Face Space tagged with openenv. The API listens on **port 7860** (`openenv.yaml` and `Dockerfile` must agree). `**GET /`** returns `200` JSON for platform readiness probes; OpenEnv also exposes `**GET /health**`, `**POST /reset**`, `**POST /step**`, `**GET /docs**`, and `**GET /openapi.json**`.

### Deployment goals

- the Space starts successfully,
- the API responds with HTTP 200,
- `reset()` works,
- inference and validation work inside the expected runtime envelope.

### Suggested deployment flow

```bash
openenv push --repo-id your-username/aace-env
```

After deployment, verify:

- reset endpoint responsiveness,
- environment startup,
- and consistency between local and hosted behavior.

---

## Design decisions

Several design decisions were made specifically to improve benchmark quality.

### Determinism over stochastic realism

MACE prioritizes reproducibility so that task grading, baseline scores, and validation remain stable.

### Operational abstraction over full physics

The environment models the decision problem, not high-fidelity flight dynamics. This keeps the benchmark focused, inspectable, and computationally practical.

### Safety-first scoring

Unsafe success should not be treated as true success. This is encoded in both reward shaping and grader design.

### Structured observations

Observations expose enough information to support intelligent policies while preserving operational realism.

### Progressive task design

The difficulty curve is intended to verify baseline competence first and frontier reasoning later.

---

## Limitations

MACE is a benchmark abstraction, not a certified aviation simulator.

It does not aim to model:

- full aircraft physics,
- complete ATC communications,
- all regulatory edge cases,
- or stochastic real-world weather distributions.

Instead, it focuses on a narrower but highly useful target: benchmarking agent decision quality in terminal-area aviation control.

That tradeoff is deliberate. It keeps the environment computationally lightweight, deterministic, and suitable for OpenEnv evaluation infrastructure.

---

## Why MACE is novel

MACE explores a domain that is still underrepresented in public agent benchmarks.

Its novelty comes from combining:

- a real-world aviation operations setting,
- explicit safety-critical priorities,
- deterministic benchmark structure,
- meaningful shaped rewards,
- and task grading that reflects operational quality rather than simple completion.

Most benchmark environments either focus on abstract planning or lightweight interaction tasks. MACE instead presents a real operational control problem where the agent must reason about:

- constrained shared infrastructure,
- urgency under fuel limits,
- dynamic restrictions,
- and asymmetric cost of mistakes.

That makes it a strong candidate for the hackathon's creativity and novelty criterion while staying grounded in real utility.

### Final summary

MACE is a complete OpenEnv environment for evaluating AI agents on a genuine, safety-critical, human-relevant control task. It includes:

- a real-world aviation operations domain,
- typed OpenEnv models and API methods,
- three deterministic tasks with increasing difficulty,
- programmatic graders scoring from 0.0 to 1.0,
- shaped rewards with partial progress signals,
- reproducible baseline inference,
- Docker and Hugging Face Spaces readiness,
- and documentation aligned to the OpenEnv Hackathon submission requirements.

---

## © Copyright

**MACE** (Multi-Agent Aviation Control & Oversight Environment) - ©2026 Team Final Commit, **Ripun Basumatary (LEAD)**, **Veeshal D. Bodosa**, and **Jeu Machahary**.

This repository is **original work** submitted for **OpenEnv Hackathon Round 1** evaluation. The team retains intellectual property; redistribution or reuse outside the hackathon context requires permission. Third-party libraries are used under their respective licenses.

*Final submission reference: 29 March 2026 · version 1.0.*