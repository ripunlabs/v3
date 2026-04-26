"""
MACE baseline inference - OpenAI-compatible client against the local environment.

Environment variables:
  AACE_INFERENCE_MODE   If set: ``scripted`` or ``llm``. If unset: ``llm`` when both
                        ``API_BASE_URL`` and ``API_KEY`` are set (validator / proxy), else ``scripted``.
  API_BASE_URL          Required non-empty for ``llm`` (OpenAI-compatible chat endpoint)
  API_KEY               Preferred API key (hackathon validator / LLM proxy injects this)
  MODEL_NAME            Optional; defaults to ``DEFAULT_MODEL_NAME`` when unset
  HF_TOKEN              Local fallback key if ``API_KEY`` is unset
  OPENAI_API_KEY        Local fallback key if ``API_KEY`` and ``HF_TOKEN`` are unset
  AACE_INFERENCE_SEED   Optional int for ``reset()`` (default: 42)
  AACE_OUTPUT_MODE      ``verbose`` (default) or ``compact`` - terminal layout
  AACE_LLM_MAX_RETRIES  Max API attempts per step in ``llm`` mode (default: 3)
  AACE_LLM_RETRY_BACKOFF_MS  First backoff delay in ms, then doubles (default: 750)
  AACE_LLM_TIMEOUT_S    HTTP timeout seconds for the OpenAI client (default: 120)

Load order: optional ``.env`` (python-dotenv), then process environment.

**Official reproducible baseline:** ``AACE_INFERENCE_MODE=scripted`` (no credentials).
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Literal, NamedTuple

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from env.environment import AACEEnvironment
from env.graders import GRADERS
from env.inference_policy import scripted_action
from env.models import AACEAction, ActionKind, AACEObservation

DEFAULT_API_BASE = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_STEPS_PER_TASK = 64
TEMPERATURE = 0.0
DEFAULT_SEED = 42
MAX_LLM_TOKENS = 200

# Exceptions after which we retry the chat completion (not parse/validation).
_LLM_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    TimeoutError,
    ConnectionError,
)
if httpx is not None:
    _extra: tuple[type[BaseException], ...] = (
        httpx.TimeoutException,
        httpx.ConnectError,
    )
    _read_err = getattr(httpx, "ReadError", None)
    if _read_err is not None:
        _extra = _extra + (_read_err,)
    _LLM_RETRY_EXCEPTIONS = _LLM_RETRY_EXCEPTIONS + _extra

# Terminal layout (box + separators). Total width includes side borders.
_TERM_W = 70
# Text between "║ " and " ║" on content rows (70 = 1+1 + _BOX_TEXT_W + 1+1).
_BOX_TEXT_W = _TERM_W - 4

TASK_HEADING = {
    "easy": ("EASY", "Efficient Landing Scheduling"),
    "medium": ("MEDIUM", "Weather-Constrained Runway Operations"),
    "hard": ("HARD", "Emergency Conflict Resolution"),
}


def _inference_mode() -> str:
    raw = (os.getenv("AACE_INFERENCE_MODE") or "").strip().lower()

    if raw:
        if raw not in ("scripted", "llm"):
            print(
                f"Error: AACE_INFERENCE_MODE must be 'scripted' or 'llm', got {raw!r}.",
                file=sys.stderr,
            )
            sys.exit(1)
        return raw

    # Auto-detect validator / proxy environment.
    if (os.getenv("API_BASE_URL") or "").strip() and (os.getenv("API_KEY") or "").strip():
        return "llm"

    return "scripted"


def _output_mode() -> str:
    raw = (os.getenv("AACE_OUTPUT_MODE") or "verbose").strip().lower()
    if raw not in ("compact", "verbose"):
        return "verbose"
    return raw


def _inference_seed() -> int:
    raw = (os.getenv("AACE_INFERENCE_SEED") or str(DEFAULT_SEED)).strip()
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_SEED


def _llm_max_attempts() -> int:
    raw = (os.getenv("AACE_LLM_MAX_RETRIES") or "3").strip()
    try:
        n = int(raw)
        return max(1, min(n, 8))
    except ValueError:
        return 3


def _llm_retry_backoff_sec() -> float:
    raw = (os.getenv("AACE_LLM_RETRY_BACKOFF_MS") or "750").strip()
    try:
        ms = float(raw)
        return max(0.05, ms / 1000.0)
    except ValueError:
        return 0.75


def _llm_timeout_s() -> float:
    raw = (os.getenv("AACE_LLM_TIMEOUT_S") or "120").strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 120.0


def _llm_credentials() -> tuple[OpenAI, str, str]:
    """
    Build client and return (client, api_base, model).

    Hackathon validator injects:
      - API_BASE_URL
      - API_KEY

    MODEL_NAME is optional there; fall back to a safe default.

    Local convenience fallbacks:
      - HF_TOKEN
      - OPENAI_API_KEY
    """

    base = (os.getenv("API_BASE_URL") or "").strip()
    key = (
        (os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "").strip()
    )
    model = (os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME).strip()

    missing: list[str] = []
    if not base:
        missing.append("API_BASE_URL")
    if not key:
        missing.append("API_KEY (or HF_TOKEN / OPENAI_API_KEY for local use)")

    if missing:
        print(
            "Error: llm mode requires non-empty "
            + ", ".join(missing)
            + ". Set them in the environment or `.env`.",
            file=sys.stderr,
        )
        sys.exit(1)

    timeout = _llm_timeout_s()
    return OpenAI(base_url=base, api_key=key, timeout=timeout), base, model


# --- Terminal presentation (box-drawing, no color) ---------------------------------

def _sep() -> str:
    return "─" * _TERM_W


def _box_top() -> str:
    return "╔" + "═" * (_TERM_W - 2) + "╗"


def _box_bottom() -> str:
    return "╚" + "═" * (_TERM_W - 2) + "╝"


def _box_line(text: str) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) > _BOX_TEXT_W:
        t = t[: _BOX_TEXT_W - 1] + "…"
    return "║ " + t.ljust(_BOX_TEXT_W) + " ║"


def _box_title_bar(title: str) -> str:
    inner = _TERM_W - 2  # ═ count between corners
    t = f" {title.strip()} "
    if len(t) > inner:
        t = t[:inner]
    pad = inner - len(t)
    left = pad // 2
    right = pad - left
    return "╔" + "═" * left + t + "═" * right + "╗"


def _kv(label: str, value: str, label_w: int = 12) -> str:
    return f"{label:<{label_w}}: {value}"


def _obs_metadata(obs: AACEObservation) -> dict[str, Any]:
    d = obs.model_dump(mode="python")
    meta = d.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _visibility_label(obs: AACEObservation) -> str:
    v = obs.weather.visibility
    name = getattr(v, "name", None) or str(v).split(".")[-1]
    return str(name).upper()


def _weather_line(obs: AACEObservation) -> str:
    vis = _visibility_label(obs)
    closed = obs.weather.closed_runways
    if closed:
        return f"{vis}  |  closed now: {', '.join(closed)}"
    return vis


def _result_label(reason: str | None) -> str:
    if reason == "mission_complete":
        return "COMPLETE"
    if reason == "max_steps":
        return "MAX_STEPS"
    if reason == "fuel_exhaustion":
        return "FUEL_EXHAUSTION"
    return "COMPLETE" if reason is None else reason.upper()


def _format_reward(r: float) -> str:
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.2f}"


def _format_action(action: AACEAction) -> str:
    k = action.kind.value
    if action.kind == ActionKind.NOOP:
        return "noop"
    if action.kind == ActionKind.ASSIGN_RUNWAY:
        f, r = action.flight_id or "?", action.runway_id or "?"
        return f"assign_runway {f} @ {r}"
    if action.kind == ActionKind.CLEAR_TO_LAND:
        return f"clear_to_land {action.flight_id or '?'}"
    if action.kind == ActionKind.SEQUENCE_LANDING:
        ids = action.ordered_flight_ids or []
        if not ids:
            return "sequence_landing"
        s = ",".join(ids[:5])
        if len(ids) > 5:
            s += ",…"
        return f"sequence_landing {s}"
    if action.kind == ActionKind.HOLD_PATTERN:
        return f"hold_pattern {action.flight_id or '?'} x{action.hold_rounds}"
    if action.kind == ActionKind.DELAY_FLIGHT:
        return f"delay_flight {action.flight_id or '?'} +{action.delay_steps}"
    if action.kind == ActionKind.REROUTE_FLIGHT:
        return f"reroute_flight {action.flight_id or '?'} →{action.alternate_code or '?'}"
    if action.kind == ActionKind.DECLARE_EMERGENCY:
        return f"declare_emergency {action.flight_id or '?'}"
    return k


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a terminal-area air traffic flow manager. Reply with exactly one JSON
    object (no fences, no extra text). Invalid actions are rejected and lower graded
    performance.

    Keys: kind (noop|assign_runway|sequence_landing|hold_pattern|delay_flight|
    reroute_flight|declare_emergency|clear_to_land); flight_id; runway_id;
    ordered_flight_ids; hold_rounds; delay_steps; alternate_code - only fields
    needed for the chosen kind.

    Rules: never assign_runway or clear_to_land to runways in
    observation.weather.closed_runways. For clear_to_land, non-emergency flights must
    use observation.approach_queue[0]; emergencies may preempt. Assign runway before
    clear; respect occupancy; check closed_runways each step. Prefer clearing
    fuel-critical flights at the fix after assignment.
    """
).strip()

ACTION_CHECKLIST = [
    "assign_runway before clear_to_land for that flight/runway",
    "do not clear_to_land if the runway is occupied when disallowed",
    "check observation.weather.closed_runways every step; never assign or clear closed runways",
    "for clear_to_land, non-emergency: flight_id must match approach_queue[0]; emergencies may preempt",
]


def _first_balanced_brace_object(raw: str) -> str | None:
    """Return substring of the first top-level ``{...}`` using brace depth (strings aware)."""

    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    i = start
    in_string = False
    escape = False
    while i < len(raw):
        ch = raw[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
        i += 1
    return None


def _parse_json_object_from_slice(raw: str) -> dict[str, Any] | None:
    """Parse the first balanced JSON object in *raw* (trimmed whole string or substring)."""

    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{") and raw.endswith("}"):
        try:
            out = json.loads(raw)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
    candidate = _first_balanced_brace_object(raw)
    if not candidate:
        return None
    try:
        out = json.loads(candidate)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract a JSON object from model text: fenced ``` / ```json blocks first,
    then a ``{...}`` scan on the full string (tolerates prose before/after).
    """

    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        got = _parse_json_object_from_slice(inner)
        if got is not None:
            return got
    return _parse_json_object_from_slice(text)


def parse_action_payload(payload: dict[str, Any]) -> AACEAction:
    """Map a parsed JSON object into a validated ``AACEAction``."""

    kind_raw = str(payload.get("kind", "noop")).lower()
    try:
        kind = ActionKind(kind_raw)
    except ValueError:
        kind = ActionKind.NOOP

    try:
        return AACEAction(
            kind=kind,
            flight_id=payload.get("flight_id"),
            runway_id=payload.get("runway_id"),
            ordered_flight_ids=payload.get("ordered_flight_ids"),
            hold_rounds=int(payload.get("hold_rounds", 1)),
            delay_steps=int(payload.get("delay_steps", 1)),
            alternate_code=payload.get("alternate_code"),
        )
    except Exception:
        return AACEAction(kind=ActionKind.NOOP)


LLMIssue = Literal["ok", "parse_failure", "validation_failure"]


def _chat_completion_with_retries(
    client: OpenAI,
    model: str,
    *,
    messages: list[dict[str, str]],
) -> tuple[str | None, int, int]:
    """
    Returns ``(assistant_text, api_attempts, api_retries)``.
    ``api_retries`` is how many backoff waits preceded a retry (0 if the first
    attempt succeeded). ``assistant_text`` is ``None`` only if every attempt failed
    with a retryable API error.
    """

    max_attempts = _llm_max_attempts()
    delay = _llm_retry_backoff_sec()
    attempts = 0
    retries = 0

    for i in range(max_attempts):
        if i > 0:
            time.sleep(delay)
            retries += 1
            delay = min(delay * 2.0, 30.0)
        attempts += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                max_tokens=MAX_LLM_TOKENS,
                messages=messages,
            )
            choice0 = completion.choices[0]
            msg = choice0.message
            text = (msg.content if msg else None) or ""
            return text, attempts, retries
        except _LLM_RETRY_EXCEPTIONS:
            continue
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            raise

    return None, attempts, retries


def llm_action(
    client: OpenAI, model: str, user_text: str
) -> tuple[AACEAction, LLMIssue, int, int, bool]:
    """
    Returns ``(action, issue, api_attempts, api_retries, api_failed)``.
    If ``api_failed``, the caller should use ``scripted_action`` instead (all API
    attempts exhausted).
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text, att, ret = _chat_completion_with_retries(client, model, messages=messages)
    if text is None:
        return AACEAction(kind=ActionKind.NOOP), "ok", att, ret, True

    data = _extract_json_object(text)
    if data is None:
        return AACEAction(kind=ActionKind.NOOP), "parse_failure", att, ret, False
    raw_kind = str(data.get("kind", "noop") or "noop").lower()
    action = parse_action_payload(data)
    if raw_kind != "noop" and action.kind == ActionKind.NOOP:
        return action, "validation_failure", att, ret, False
    return action, "ok", att, ret, False


class TaskRunResult(NamedTuple):
    score: float
    total_steps: int
    fallback_steps: int
    invalid_actions: int
    parse_failures: int
    validation_failures: int
    terminal_reason: str | None
    step_traces: list[tuple[int, str, float, bool]]
    initial_observation: AACEObservation
    llm_api_attempts: int
    llm_api_retries: int


def _llm_diag_display(res: TaskRunResult) -> str:
    return (
        f"att={res.llm_api_attempts}  ret={res.llm_api_retries}  "
        f"api_fb={res.fallback_steps}  parse_fail={res.parse_failures}  "
        f"validation_fail={res.validation_failures}"
    )


def _emit_structured_start(task_id: str) -> None:
    print(f"[START] task={task_id}", flush=True)


def _emit_structured_step(
    task_id: str,
    step: int,
    reward: float,
    valid: bool,
    action: str,
) -> None:
    print(
        f"[STEP] task={task_id} step={step} reward={reward:.4f} "
        f"valid={str(valid).lower()} action={json.dumps(action)}",
        flush=True,
    )


def _emit_structured_end(
    task_id: str,
    score: float,
    steps: int,
    invalid: int,
    fallbacks: int,
    reason: str | None,
) -> None:
    reason_val = reason or "-"
    print(
        f"[END] task={task_id} score={score:.4f} steps={steps} "
        f"invalid={invalid} fallbacks={fallbacks} reason={reason_val}",
        flush=True,
    )


def run_task(
    env: AACEEnvironment,
    task_id: str,
    use_llm: bool,
    client: OpenAI | None,
    model: str,
    seed: int,
    *,
    verbose: bool,
) -> TaskRunResult:
    """
    Run one episode. Counting semantics unchanged from prior versions; ``step_traces``
    is filled only when ``verbose`` is True (for display only).

    Emits ``[START]``, ``[STEP]``, ``[END]`` lines to stdout for machine-readable validation.
    """

    obs = env.reset(seed=seed, episode_id=f"infer-{task_id}", task_id=task_id)
    initial_obs = obs

    _emit_structured_start(task_id)

    total_steps = 0
    fallback_steps = 0
    invalid_actions = 0
    parse_failures = 0
    validation_failures = 0
    llm_api_attempts = 0
    llm_api_retries = 0
    step_traces: list[tuple[int, str, float, bool]] = []

    for _ in range(MAX_STEPS_PER_TASK):
        if obs.done:
            break

        if use_llm:
            assert client is not None
            payload = {
                "task": task_id,
                "action_checklist": ACTION_CHECKLIST,
                "observation": obs.model_dump(mode="json"),
            }
            user_text = json.dumps(payload, separators=(",", ":"), default=str)
            action, issue, att, ret, api_failed = llm_action(client, model, user_text)
            llm_api_attempts += att
            llm_api_retries += ret
            if api_failed:
                fallback_steps += 1
                action = scripted_action(obs)
            elif issue == "parse_failure":
                parse_failures += 1
            elif issue == "validation_failure":
                validation_failures += 1
        else:
            action = scripted_action(obs)

        action_str = _format_action(action)
        obs = env.step(action)
        total_steps += 1

        if not obs.last_action_valid:
            invalid_actions += 1

        _emit_structured_step(
            task_id=task_id,
            step=total_steps,
            reward=float(obs.reward),
            valid=bool(obs.last_action_valid),
            action=action_str,
        )

        if verbose:
            step_traces.append(
                (total_steps, action_str, float(obs.reward), obs.last_action_valid)
            )

    meta = _obs_metadata(obs)
    terminal_reason = meta.get("terminal_reason")
    tr = terminal_reason if isinstance(terminal_reason, str) else None

    snap = env.snapshot_for_grader()
    result = GRADERS[task_id](snap)
    score = float(result.score)

    _emit_structured_end(
        task_id=task_id,
        score=score,
        steps=total_steps,
        invalid=invalid_actions,
        fallbacks=fallback_steps,
        reason=tr,
    )

    return TaskRunResult(
        score,
        total_steps,
        fallback_steps,
        invalid_actions,
        parse_failures,
        validation_failures,
        tr,
        step_traces,
        initial_obs,
        llm_api_attempts if use_llm else 0,
        llm_api_retries if use_llm else 0,
    )


def _print_header(*, mode: str, seed: int, api_base: str, model_line: str, use_llm: bool) -> None:
    print(_box_top())
    print(_box_line("MACE - Multi-Agent Aviation Control & Oversight Environment"))
    print(_box_line("OpenEnv Benchmark Inference Runner"))
    print(_box_bottom())
    print()
    print(_kv("Mode", mode))
    print(_kv("Seed", str(seed)))
    print(_kv("API base", api_base))
    print(_kv("Model", model_line))
    print(_kv("Output", _output_mode()))
    print(_kv("Task suite", "easy • medium • hard"))
    if use_llm:
        print(_kv("Note", "Hosted LLM runs may vary slightly."))
        print(f"{'':14}Scripted mode is the official reproducible baseline.")
    print()


def _print_verbose_task_block(
    idx: int,
    task_id: str,
    res: TaskRunResult,
    *,
    use_llm: bool,
) -> None:
    start_obs = res.initial_observation
    tag, subtitle = TASK_HEADING[task_id]
    print(_sep())
    print(f"[{idx}/3] TASK: {tag}  - {subtitle}")
    print(_sep())
    print(_kv("Status", "RUNNING"))
    print(_kv("Flights", str(len(start_obs.flights))))
    print(_kv("Runways", str(len(start_obs.runways))))
    print(_kv("Weather", _weather_line(start_obs)))
    obj = _obs_metadata(start_obs).get("objective", "")
    if isinstance(obj, str) and obj:
        for i, line in enumerate(textwrap.wrap(obj, width=52)):
            lab = "Objective" if i == 0 else ""
            print(_kv(lab, line, label_w=12))
    print()
    for step_i, act_s, rew, valid in res.step_traces:
        v = "yes" if valid else "no"
        step_lbl = f"Step {step_i:02d}"
        act_col = act_s[:42].ljust(42)
        print(f"{step_lbl}  Action: {act_col}  Reward: {_format_reward(rew):>7}   Valid: {v}")
    print()
    print(_kv("Result", _result_label(res.terminal_reason)))
    print(_kv("Score", f"{res.score:.4f}"))
    print(_kv("Steps", str(res.total_steps)))
    print(_kv("Invalid", str(res.invalid_actions)))
    print(_kv("Fallbacks", str(res.fallback_steps)))
    if use_llm:
        print(_kv("Diagnostics", _llm_diag_display(res)))
    print(_kv("Reason", res.terminal_reason or "-"))


def _print_compact_task_row(
    idx: int,
    task_id: str,
    res: TaskRunResult,
    use_llm: bool,
) -> None:
    name = task_id.capitalize()
    line = (
        f"[{idx}/3] {name:<6}  Score: {res.score:.4f}   "
        f"Steps: {res.total_steps:>3}   Invalid: {res.invalid_actions:<2}   "
        f"Fallbacks: {res.fallback_steps:<2}"
    )
    print(line)
    if use_llm:
        print(f"         Diagnostics:  {_llm_diag_display(res)}")


def _print_final_scoreboard(
    *,
    results: list[tuple[str, TaskRunResult]],
    mean: float,
    total_steps: int,
    total_invalid: int,
    total_fallback: int,
    total_llm_attempts: int,
    total_llm_retries: int,
    scripted: bool,
    use_llm: bool,
) -> None:
    print()
    print(_box_title_bar("FINAL SCOREBOARD"))
    for task_id, res in results:
        name = task_id.capitalize()
        body = (
            f"{name:<7} {res.score:>7.4f}   │ Steps:{res.total_steps:>3}  "
            f"│ Invalid:{res.invalid_actions:>2}  │ FB:{res.fallback_steps:>3}"
        )
        print(_box_line(body))
    print(_box_line("─" * _BOX_TEXT_W))
    print(_box_line(f"Mean score : {mean:.4f}"))
    print(_box_line(f"Total steps: {total_steps}"))
    if use_llm:
        print(
            _box_line(
                f"LLM: att={total_llm_attempts}  ret={total_llm_retries}  "
                f"api_fb={total_fallback}"
            )
        )
    print(_box_line(f"Total invalid actions: {total_invalid}"))
    baseline = (
        "Deterministic scripted policy (official reproducible baseline)"
        if scripted
        else "Hosted LLM evaluation (optional; may vary or fall back)"
    )
    print(_box_line(baseline))
    print(_box_bottom())


def _configure_stdio_utf8() -> None:
    """Best-effort UTF-8 for box-drawing on Windows consoles (cp1252 default)."""

    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    load_dotenv()
    _configure_stdio_utf8()
    mode = _inference_mode()
    out_mode = _output_mode()
    verbose = out_mode == "verbose"
    seed = _inference_seed()
    use_llm = mode == "llm"
    client: OpenAI | None = None
    api_base = (os.getenv("API_BASE_URL") or DEFAULT_API_BASE).strip()
    model_display = (os.getenv("MODEL_NAME") or "").strip()

    if use_llm:
        client, api_base, model_used = _llm_credentials()
        model_display = model_used
    else:
        if not (os.getenv("API_BASE_URL") or "").strip():
            api_base = DEFAULT_API_BASE
        model_display = ""

    model_line = model_display if use_llm else "<unset>"
    _print_header(
        mode=mode,
        seed=seed,
        api_base=api_base,
        model_line=model_line,
        use_llm=use_llm,
    )

    env = AACEEnvironment()
    scores: dict[str, float] = {}
    total_steps_all = 0
    total_fallback_all = 0
    total_invalid_all = 0
    total_llm_attempts_all = 0
    total_llm_retries_all = 0
    ordered_results: list[tuple[str, TaskRunResult]] = []

    if not verbose:
        print(_sep())

    for idx, task_id in enumerate(("easy", "medium", "hard"), start=1):
        res = run_task(
            env,
            task_id,
            use_llm,
            client,
            model_display if use_llm else "",
            seed,
            verbose=verbose,
        )
        scores[task_id] = res.score
        total_steps_all += res.total_steps
        total_fallback_all += res.fallback_steps
        total_invalid_all += res.invalid_actions
        total_llm_attempts_all += res.llm_api_attempts
        total_llm_retries_all += res.llm_api_retries
        ordered_results.append((task_id, res))

        if verbose:
            _print_verbose_task_block(idx, task_id, res, use_llm=use_llm)
        else:
            _print_compact_task_row(idx, task_id, res, use_llm)

    mean = sum(scores.values()) / max(1, len(scores))

    if not verbose:
        print(_sep())
        print(_kv("Mean score", f"{mean:.4f}"))
        print(_kv("Total steps", str(total_steps_all)))
        print(_kv("Total invalid actions", str(total_invalid_all)))
        if use_llm:
            print(
                _kv(
                    "LLM API (totals)",
                    f"attempts={total_llm_attempts_all}  retries={total_llm_retries_all}",
                )
            )
        print(_sep())

    _print_final_scoreboard(
        results=ordered_results,
        mean=mean,
        total_steps=total_steps_all,
        total_invalid=total_invalid_all,
        total_fallback=total_fallback_all,
        total_llm_attempts=total_llm_attempts_all,
        total_llm_retries=total_llm_retries_all,
        scripted=not use_llm,
        use_llm=use_llm,
    )


if __name__ == "__main__":
    main()
