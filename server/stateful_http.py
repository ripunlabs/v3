"""
Stateful HTTP /reset, /step, and /state for MACE.

OpenEnv's default HTTP handlers create a fresh Environment per request and call
``close()`` afterward, so ``POST /step`` cannot follow ``POST /reset``. This module
replaces those routes with session-scoped instances keyed by optional header
``X-OpenEnv-Session-Id`` (default session id: ``default``).

WebSocket ``/ws`` behavior is unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections import defaultdict
from typing import Any, Callable, Type

from fastapi import Body, FastAPI, Header, HTTPException, status
from fastapi.routing import APIRoute
from pydantic import ValidationError

try:
    from openenv.core.env_server.serialization import deserialize_action, serialize_observation
    from openenv.core.env_server.types import ResetRequest, ResetResponse, StepRequest, StepResponse
except ImportError:
    from core.env_server.serialization import deserialize_action, serialize_observation  # type: ignore[no-redef]
    from core.env_server.types import (  # type: ignore[no-redef]
        ResetRequest,
        ResetResponse,
        StepRequest,
        StepResponse,
    )

from env.environment import AACEEnvironment
from env.models import AACEAction

SESSION_HEADER = "X-OpenEnv-Session-Id"


def _filter_sig_kwargs(
    sig: inspect.Signature,
    kwargs: dict[str, Any],
    *,
    skip: set[str] | None = None,
) -> dict[str, Any]:
    skip = skip or set()
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in skip:
            continue
        if k in sig.parameters or has_var_kw:
            out[k] = v
    return out


def _strip_openenv_control_routes(app: FastAPI) -> None:
    kept: list[Any] = []
    for route in app.router.routes:
        if isinstance(route, APIRoute) and route.methods:
            m = route.methods
            if route.path == "/reset" and "POST" in m:
                continue
            if route.path == "/step" and "POST" in m:
                continue
            if route.path == "/state" and "GET" in m:
                continue
        kept.append(route)
    app.router.routes = kept


def install_stateful_http_sessions(
    app: FastAPI,
    env_factory: Callable[[], AACEEnvironment] | Type[AACEEnvironment],
    action_cls: type[AACEAction],
) -> None:
    if getattr(app.state, "aace_stateful_http", False):
        return
    app.state.aace_stateful_http = True

    factory: Callable[[], AACEEnvironment]
    if isinstance(env_factory, type):
        factory = env_factory
    else:
        factory = env_factory

    _locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
    _envs: dict[str, AACEEnvironment | None] = {}

    _strip_openenv_control_routes(app)

    def _sid(x_openenv_session_id: str | None) -> str:
        s = (x_openenv_session_id or "").strip()
        return s if s else "default"

    @app.post(
        "/reset",
        response_model=ResetResponse,
        tags=["Environment Control"],
        summary="Reset the environment (session-scoped HTTP)",
    )
    async def http_reset(
        request_body: ResetRequest = Body(default_factory=ResetRequest),
        x_openenv_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
    ) -> ResetResponse:
        sid = _sid(x_openenv_session_id)
        with _locks[sid]:
            env = _envs.get(sid)
            if env is None:
                env = factory()
                _envs[sid] = env

            kwargs = request_body.model_dump(exclude_unset=True)
            sig = inspect.signature(env.reset)
            valid = _filter_sig_kwargs(sig, kwargs)

            obs = await asyncio.to_thread(env.reset, **valid)
        return ResetResponse(**serialize_observation(obs))

    @app.post(
        "/step",
        response_model=StepResponse,
        tags=["Environment Control"],
        summary="Execute an action (session-scoped HTTP)",
    )
    async def http_step(
        request_body: StepRequest,
        x_openenv_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
    ) -> StepResponse:
        sid = _sid(x_openenv_session_id)
        try:
            action = deserialize_action(request_body.action, action_cls)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=e.errors(),
            ) from e

        with _locks[sid]:
            env = _envs.get(sid)
            if env is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No environment for this session; call POST /reset first (or use the same "
                    f"{SESSION_HEADER} header).",
                )

            kwargs = request_body.model_dump(exclude_unset=True, exclude={"action"})
            sig = inspect.signature(env.step)
            valid = _filter_sig_kwargs(sig, kwargs, skip={"action"})

            obs = await asyncio.to_thread(env.step, action, **valid)
        return StepResponse(**serialize_observation(obs))

    @app.get(
        "/state",
        tags=["State Management"],
        summary="Current environment state (session-scoped HTTP)",
    )
    async def http_state(
        x_openenv_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
    ) -> dict[str, Any]:
        sid = _sid(x_openenv_session_id)
        with _locks[sid]:
            env = _envs.get(sid)
            if env is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No environment for this session; call POST /reset first.",
                )
            st = env.state
            if hasattr(st, "model_dump"):
                return st.model_dump()
            return dict(st) if st else {}
