"""Stateful HTTP /reset + /step share one environment per session (default)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_http_reset_then_step_noop_advances() -> None:
    c = TestClient(app)
    r0 = c.post("/reset", json={"task_id": "easy", "seed": 0})
    assert r0.status_code == 200
    assert r0.json()["observation"]["step_index"] == 0

    r1 = c.post("/step", json={"action": {"kind": "noop", "metadata": {}}})
    assert r1.status_code == 200
    body = r1.json()
    assert body["observation"]["step_index"] == 1


def test_http_step_without_reset_400() -> None:
    c = TestClient(app)
    # Unique session with no reset
    r = c.post(
        "/step",
        json={"action": {"kind": "noop", "metadata": {}}},
        headers={"X-OpenEnv-Session-Id": "never-reset-this-id"},
    )
    assert r.status_code == 400
