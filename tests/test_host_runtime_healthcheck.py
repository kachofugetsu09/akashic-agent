from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_healthcheck():
    path = Path(__file__).parents[1] / "docker/host-runtime/healthcheck.py"
    spec = importlib.util.spec_from_file_location("host_runtime_healthcheck", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _HealthResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"status":"ready"}'


class _NotReadyHealthResponse(_HealthResponse):
    def read(self) -> bytes:
        return b'{"status":"ok"}'


def test_healthcheck_requires_identity_and_web_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_healthcheck()
    readiness = {
        "bootId": "boot-1",
        "pid": 123,
        "state": "ready",
        "sourceCommit": "a" * 40,
        "hostCheckout": "/srv/runtime",
    }
    (tmp_path / ".runtime-ready.json").write_text(json.dumps(readiness))
    monkeypatch.setenv("AKASHIC_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AKASHIC_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("AKASHIC_RUNTIME_CHECKOUT", "/srv/runtime")
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(module.os, "kill", lambda pid, signal: killed.append((pid, signal)))
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: _HealthResponse())

    module.main()

    assert killed == [(123, 0)]


def test_healthcheck_rejects_stale_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_healthcheck()
    readiness = {
        "bootId": "boot-1",
        "pid": 123,
        "state": "ready",
        "sourceCommit": "b" * 40,
        "hostCheckout": "/srv/runtime",
    }
    (tmp_path / ".runtime-ready.json").write_text(json.dumps(readiness))
    monkeypatch.setenv("AKASHIC_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AKASHIC_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("AKASHIC_RUNTIME_CHECKOUT", "/srv/runtime")

    with pytest.raises(RuntimeError, match="identity"):
        module.main()


def test_healthcheck_rejects_non_ready_web_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_healthcheck()
    readiness = {
        "bootId": "boot-1",
        "pid": 123,
        "state": "ready",
        "sourceCommit": "a" * 40,
        "hostCheckout": "/srv/runtime",
    }
    (tmp_path / ".runtime-ready.json").write_text(json.dumps(readiness))
    monkeypatch.setenv("AKASHIC_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AKASHIC_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("AKASHIC_RUNTIME_CHECKOUT", "/srv/runtime")
    monkeypatch.setattr(module.os, "kill", lambda *_args: None)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _NotReadyHealthResponse(),
    )

    with pytest.raises(RuntimeError, match="payload"):
        module.main()
