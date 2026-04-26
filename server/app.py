"""
FastAPI application exposing MACE over the OpenEnv HTTP runtime.
"""

from __future__ import annotations

try:
    from openenv.core.env_server import create_app
except ImportError:
    from core.env_server import create_app  # type: ignore[no-redef]

from starlette.responses import Response

from env.environment import AACEEnvironment
from env.models import AACEAction, AACEObservation

from server.stateful_http import install_stateful_http_sessions

app = create_app(
    AACEEnvironment,
    AACEAction,
    AACEObservation,
    env_name="mace",
)

install_stateful_http_sessions(app, AACEEnvironment, AACEAction)


@app.get(
    "/",
    tags=["Health"],
    summary="Root readiness (e.g. Hugging Face Spaces probe)",
    response_model=None,
)
async def root() -> dict[str, str | bool]:
    """
    Lightweight 200 JSON for platforms that probe ``GET /``.
    OpenEnv control API remains on ``/reset``, ``/step``, ``/health``, etc.
    """

    return {
        "status": "ok",
        "service": "mace",
        "openenv": True,
    }


@app.head(
    "/",
    tags=["Health"],
    summary="Root readiness HEAD (some platforms probe HEAD /)",
    include_in_schema=False,
    response_model=None,
)
async def root_head() -> Response:
    """Empty 200 for load balancers that use HEAD instead of GET."""
    return Response(status_code=200)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
