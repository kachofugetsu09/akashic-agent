from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from docker.debug.model_gate import build_server


@contextmanager
def _server_url() -> Iterator[str]:
    server = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _json(
    method: str, url: str, payload: object | None = None
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, method=method, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=2) as response:
        decoded = json.loads(response.read())
    if not isinstance(decoded, dict):
        raise ValueError("model gate response 必须是 JSON object")
    return cast(dict[str, Any], decoded)


def test_complete_and_error_scripts_are_consumed_in_order() -> None:
    with _server_url() as url:
        assert _json(
            "PUT",
            f"{url}/control/script",
            [
                {"mode": "complete", "content": "first"},
                {"mode": "error", "status": 429},
            ],
        ) == {"loaded": 2}

        complete = _json(
            "POST",
            f"{url}/v1/chat/completions",
            {"model": "gate", "messages": [], "stream": False},
        )
        assert complete["choices"][0]["message"]["content"] == "first"
        with pytest.raises(HTTPError) as raised:
            _json(
                "POST",
                f"{url}/v1/chat/completions",
                {"model": "gate", "messages": [], "stream": False},
            )
        assert raised.value.code == 429
        raised.value.close()

        requests = _json("GET", f"{url}/control/requests")
        assert [item["state"] for item in requests["requests"]] == [
            "completed",
            "error_sent",
        ]


def test_stream_emits_content_tool_usage_and_done() -> None:
    with _server_url() as url:
        _json(
            "PUT",
            f"{url}/control/script",
            {
                "mode": "stream",
                "deltas": ["hello", {"reasoning_content": "think"}],
                "tool_calls": [
                    {"id": "call-1", "name": "lookup", "arguments": {"q": "x"}}
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        )
        request = Request(
            f"{url}/v1/chat/completions",
            method="POST",
            data=json.dumps({"model": "gate", "messages": [], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=2) as response:
            body = response.read().decode()

        assert '"content":"hello"' in body
        assert '"reasoning_content":"think"' in body
        assert '"name":"lookup"' in body
        assert '"total_tokens":6' in body
        assert body.endswith("data: [DONE]\n\n")


def test_stream_can_pace_deltas_for_visual_debugging() -> None:
    with _server_url() as url:
        _json(
            "PUT",
            f"{url}/control/script",
            {
                "mode": "stream",
                "deltas": ["one", "two", "three"],
                "delay_ms": 20,
            },
        )
        request = Request(
            f"{url}/v1/chat/completions",
            method="POST",
            data=json.dumps({"model": "gate", "messages": [], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )

        started = time.monotonic()
        with urlopen(request, timeout=2) as response:
            body = response.read().decode()

        assert time.monotonic() - started >= 0.05
        assert body.index('"content":"one"') < body.index('"content":"three"')


def test_complete_script_honors_stream_request() -> None:
    with _server_url() as url:
        _json(
            "PUT",
            f"{url}/control/script",
            {"mode": "complete", "content": "one chunk"},
        )
        request = Request(
            f"{url}/v1/chat/completions",
            method="POST",
            data=json.dumps({"model": "gate", "messages": [], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=2) as response:
            body = response.read().decode()

        assert '"content":"one chunk"' in body
        assert body.endswith("data: [DONE]\n\n")


def test_barrier_waits_for_exact_request_until_release() -> None:
    with _server_url() as url:
        _json("PUT", f"{url}/control/barriers/held")
        _json(
            "PUT",
            f"{url}/control/script",
            {"mode": "complete", "content": "released", "barrier": "held"},
        )
        response: dict[str, object] = {}

        def send_request() -> None:
            response["payload"] = _json(
                "POST",
                f"{url}/v1/chat/completions",
                {"model": "gate", "messages": [], "stream": False},
            )

        worker = threading.Thread(target=send_request)
        worker.start()
        wait = _json("GET", f"{url}/control/barriers/held/wait?timeout=2")
        assert wait == {"name": "held", "reached": True, "released": False}
        assert worker.is_alive()

        release = _json("POST", f"{url}/control/barriers/held/release")
        assert release == {"name": "held", "reached": True, "released": True}
        worker.join(timeout=2)
        assert not worker.is_alive()
        payload = response["payload"]
        assert isinstance(payload, dict)
        assert payload["choices"][0]["message"]["content"] == "released"


def test_truncated_stream_closes_without_done_marker() -> None:
    with _server_url() as url:
        _json(
            "PUT",
            f"{url}/control/script",
            {"mode": "truncate", "deltas": ["partial"]},
        )
        parsed = urlsplit(url)
        assert parsed.hostname is not None
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({"model": "gate", "messages": [], "stream": True}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()

        assert response.status == 200
        assert '"content":"partial"' in body
        assert "[DONE]" not in body


def test_timeout_script_holds_until_client_cancels() -> None:
    with _server_url() as url:
        _json("PUT", f"{url}/control/script", {"mode": "timeout"})
        parsed = urlsplit(url)
        assert parsed.hostname is not None
        connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port, timeout=0.1
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({"model": "gate", "messages": [], "stream": False}),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(TimeoutError):
            connection.getresponse()
        connection.close()

        deadline = time.monotonic() + 2
        state = ""
        while time.monotonic() < deadline:
            requests = _json("GET", f"{url}/control/requests")
            state = requests["requests"][0]["state"]
            if state == "client_disconnected":
                break
            threading.Event().wait(0.01)
        assert state == "client_disconnected"


def test_invalid_script_is_rejected_without_partial_load() -> None:
    with _server_url() as url:
        with pytest.raises(HTTPError) as raised:
            _json("PUT", f"{url}/control/script", {"mode": "error", "status": 200})
        assert raised.value.code == 400
        body = json.loads(raised.value.read())
        raised.value.close()
        assert body["error"] == "invalid_request"


@pytest.mark.parametrize("delay_ms", [-1, 5001, 1.5, True])
def test_invalid_stream_delay_is_rejected(delay_ms: object) -> None:
    with _server_url() as url:
        with pytest.raises(HTTPError) as raised:
            _json(
                "PUT",
                f"{url}/control/script",
                {"mode": "stream", "deltas": ["x"], "delay_ms": delay_ms},
            )
        assert raised.value.code == 400
        body = json.loads(raised.value.read())
        raised.value.close()
        assert body["error"] == "invalid_request"
