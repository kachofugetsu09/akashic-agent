#!/usr/bin/env python3
"""Exercise Akasha learning through the public programmatic Message API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from docker.debug.model_plugin_fixture import add_openai_models
from docker.debug.programmatic_control_probe import (
    CheckResult,
    GateFailure,
    JsonRpcSocketClient,
    _connect_client,
    _configure_model_gate,
    _http_json,
    _model_requests,
    _prepare_host_sandbox,
    _repository_digest,
    _runtime_identity,
    _wait_barrier,
    _wait_http_ready,
    _wait_programmatic_result,
    _wait_socket,
)
from plugins.akasha.infrastructure.consumption import Consumption
from plugins.akasha.infrastructure.persistence import (
    load_consumption,
    logical_state_sha256,
)

READINESS_DEADLINE_S = 30.0
TURN_DEADLINE_S = 60.0
SESSION_ID = "programmatic:akasha-v2"
FIRST_INPUT_ID = "akv2-first-input"
SECOND_INPUT_ID = "akv2-second-input"


def _load_scripts(model_url: str, scripts: object) -> None:
    """Load deterministic model responses through the fixture control API."""

    _http_json("PUT", f"{model_url}/control/script", scripts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
    }


def _formal_identity(workspace: Path) -> dict[str, dict[str, object]]:
    return {
        relative: _file_identity(workspace / relative)
        for relative in ("sessions.db", "memory/akasha.db")
    }


def _source_identity(repo: Path) -> dict[str, object]:
    """Capture the source revision, dirty state, and content digest."""

    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {"head": head, "dirty": dirty, "digest": _repository_digest(repo)}


def _tree_digest(root: Path) -> str:
    """Hash every regular file in a copied sandbox tree in stable order."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8")
            digest.update(b"symlink")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
            continue
        if not path.is_file():
            continue
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _write_runtime_config(sandbox: Path) -> None:
    """Write a private config whose runtime state stays in the sandbox."""

    config = """\
[agent.plugins]
disabled_builtin = ["subagent"]

[app_server]
enabled = true
listen = "/sandbox/akashic.sock"
max_connections = 8
ingress_queue_size = 32
outbound_queue_size = 64

[channels.chat]
enabled = true

[channels.telegram]
enabled = false
token = ""

[channels.qq]
enabled = false
bot_uin = ""

"""
    path = sandbox / "config.toml"
    path.write_text(config, encoding="utf-8")
    path.chmod(0o600)
    reply_config = sandbox / "workspace/plugin-data/reply-builtin/config.local.toml"
    reply_config.parent.mkdir(parents=True, exist_ok=True)
    reply_config.write_text("max_steps = 4\n", encoding="utf-8")
    compaction_config = sandbox / "workspace/plugin-data/compaction-builtin/config.local.toml"
    compaction_config.parent.mkdir(parents=True, exist_ok=True)
    compaction_config.write_text("keep_recent_tokens = 20000\n", encoding="utf-8")


def _embedding_environment() -> tuple[str, str, str]:
    """Require the real embedding provider before a container is started."""

    values = tuple(
        os.environ.get(name, "").strip()
        for name in (
            "AKASHIC_E2E_EMBEDDING_API_KEY",
            "AKASHIC_E2E_EMBEDDING_BASE_URL",
            "AKASHIC_E2E_EMBEDDING_MODEL",
        )
    )
    if not all(values):
        raise GateFailure(
            "Akasha E2E embedding environment is incomplete; "
            "set AKASHIC_E2E_EMBEDDING_API_KEY, "
            "AKASHIC_E2E_EMBEDDING_BASE_URL, and "
            "AKASHIC_E2E_EMBEDDING_MODEL before starting Docker"
        )
    return cast(tuple[str, str, str], values)


def _configure_embedding(settings_url: str) -> None:
    """Register the external embedding binding without exposing its credential."""

    key, url, model = _embedding_environment()
    add_openai_models(
        settings_url,
        connection_id="akasha-embedding",
        endpoint=url,
        api_key=key,
        embedding_model=model,
        embedding_dimensions=1024,
    )


class _EmbeddingFixtureState:
    """Count requests served by the explicit local embedding fixture."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._lock = threading.Lock()
        self._requests: list[dict[str, object]] = []

    def record(self, payload: dict[str, object]) -> None:
        with self._lock:
            self._requests.append(payload)

    def count(self) -> int:
        with self._lock:
            return len(self._requests)


class _EmbeddingFixtureHandler(BaseHTTPRequestHandler):
    """Serve the minimum OpenAI-compatible model and embedding contract."""

    server: "_EmbeddingFixtureServer"

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [{"id": self.server.state.model, "object": "model"}],
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/embeddings":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("embedding fixture request must be an object")
            inputs = payload.get("input")
            model = payload.get("model")
            if (
                not isinstance(inputs, list)
                or not inputs
                or not all(isinstance(item, str) for item in inputs)
                or model != self.server.state.model
            ):
                raise ValueError("embedding fixture request is invalid")
        except (ValueError, TypeError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self.server.state.record(cast(dict[str, object], payload))
        vector = [0.0] * 1023 + [1.0]
        self._json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector}
                    for index, _text in enumerate(inputs)
                ],
                "model": self.server.state.model,
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            },
        )

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _EmbeddingFixtureServer(ThreadingHTTPServer):
    """Threaded local server used only by --local-fixture."""

    daemon_threads = True

    def __init__(self, state: _EmbeddingFixtureState) -> None:
        super().__init__(("0.0.0.0", 0), _EmbeddingFixtureHandler)
        self.state = state


def _start_embedding_fixture() -> tuple[_EmbeddingFixtureServer, threading.Thread, str]:
    """Start a host listener reachable from the Docker bridge."""

    state = _EmbeddingFixtureState("akasha-local-embedding")
    server = _EmbeddingFixtureServer(state)
    thread = threading.Thread(target=server.serve_forever, name="akasha-embedding-fixture")
    thread.start()
    return server, thread, f"http://host.docker.internal:{server.server_port}/v1"


def _stop_embedding_fixture(server: _EmbeddingFixtureServer, thread: threading.Thread) -> None:
    """Stop the local fixture and fail if its serving thread remains alive."""

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    if thread.is_alive():
        raise GateFailure("local embedding fixture thread did not stop")


def _wait_learning(
    memory_path: Path,
    *,
    minimum_applied: int,
    timeout: float = TURN_DEADLINE_S,
) -> tuple[str, Consumption]:
    """Wait until the learning owner publishes the requested graph nodes."""

    deadline = time.monotonic() + timeout
    last_state: Consumption | None = None
    while time.monotonic() < deadline:
        if memory_path.exists():
            state = load_consumption(memory_path)
            last_state = state
            if state is not None and len(state.applied) >= minimum_applied:
                return logical_state_sha256(memory_path), state
        threading.Event().wait(0.05)
    raise GateFailure(
        "Akasha learning did not settle: "
        f"expected applied>={minimum_applied}, state={last_state!r}"
    )


def _message_text(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    body = row.get("body")
    if not isinstance(body, dict):
        return ""
    return "".join(
        str(part.get("value", ""))
        for part in body.get("parts", [])
        if isinstance(part, dict) and part.get("kind") == "text"
    )


def _page_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    items = page.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise GateFailure(f"message/read items 非法：{page!r}")
    return cast(list[dict[str, Any]], items)


def _message_rows(database: Path) -> list[dict[str, object]]:
    """Read canonical Message rows, including the encoded body."""

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, session_key, seq, ts, author, source, body "
            "FROM messages ORDER BY session_key, seq"
        ).fetchall()
    return [dict(row) for row in rows]


def _embedding_rows(database: Path) -> list[dict[str, object]]:
    """Read all durable embedding bytes without decoding provider data."""

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT message_id, content_hash, model, hex(embedding) AS embedding_hex, "
            "dim, created_at, updated_at FROM message_embeddings "
            "ORDER BY message_id, model"
        ).fetchall()
    return [dict(row) for row in rows]


def _akasha_recall_records(database: Path) -> dict[str, dict[str, object]]:
    """Read Akasha's durable recall owner records without re-running a query."""

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT key, value FROM owner_records "
            "WHERE owner = 'plugin:akasha' AND key LIKE 'recall:%' ORDER BY key"
        ).fetchall()
    records: dict[str, dict[str, object]] = {}
    for row in rows:
        value = json.loads(str(row["value"]))
        if not isinstance(value, dict):
            raise GateFailure(f"Akasha recall owner record 非 object: {row['key']!r}")
        records[str(row["key"])] = cast(dict[str, object], value)
    return records


def _context_rows(requests: list[object]) -> list[dict[str, object]]:
    """Extract JSON recall rows from real provider request context messages."""

    rows: list[dict[str, object]] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        payload = request.get("payload")
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                encoded = json.loads(content)
            except json.JSONDecodeError:
                continue
            if not isinstance(encoded, dict) or not isinstance(encoded.get("context"), list):
                continue
            for part in encoded["context"]:
                if not isinstance(part, dict) or part.get("kind") != "text":
                    continue
                value = part.get("value")
                if not isinstance(value, str):
                    continue
                for line in value.splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and {"message_id", "text"} <= set(row):
                        rows.append(cast(dict[str, object], row))
    return rows


def _database_snapshot(sessions_db: Path, memory_db: Path) -> dict[str, object]:
    """Capture the complete published graph and its source-side durable facts."""

    state = load_consumption(memory_db)
    if state is None:
        raise GateFailure("Akasha memory snapshot lacks consumer_state_json")
    return {
        "logicalState": logical_state_sha256(memory_db),
        "consumerProgress": state.model_dump(mode="json"),
        "messages": _message_rows(sessions_db),
        "embeddings": _embedding_rows(sessions_db),
    }



def _inside_scenario(report_dir: Path) -> int:
    """Run first learning, read-only recall, and second learning online."""

    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "akasha-v2-events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    settings_url = "http://akashic-control-gate:2236/api/settings/model"
    endpoint = Path("/sandbox/akashic.sock")
    memory_path = Path("/sandbox/workspace/memory/akasha.db")
    checks: list[CheckResult] = []
    client: JsonRpcSocketClient | None = None
    scenario_state: dict[str, object] = {}
    try:
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _configure_model_gate()
        _configure_embedding(settings_url)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, events_path)
        status_before = client.request("server/status", {}).get("result")
        if not isinstance(status_before, dict) or not isinstance(
            status_before.get("bootId"), str
        ):
            raise GateFailure(f"server/status 缺少 bootId：{status_before!r}")

        admission = client.admit_programmatic(SESSION_ID, persist_memory=True)
        if admission.get("learning") != "eligible":
            raise GateFailure(f"programmatic Session 未取得 eligible 准入：{admission!r}")
        initial_hash, initial_progress = _wait_learning(memory_path, minimum_applied=0)

        _load_scripts(model_url, {"mode": "complete", "content": "first remembered answer"})
        first_ack = client.send_programmatic(
            SESSION_ID, FIRST_INPUT_ID, "alpha medical story"
        )
        first_result = _wait_programmatic_result(
            client, SESSION_ID, FIRST_INPUT_ID, timeout=TURN_DEADLINE_S
        )
        first_page = client.read_messages(SESSION_ID)
        first_items = _page_items(first_page)
        first_hash, first_progress = _wait_learning(memory_path, minimum_applied=1)

        barrier = "akasha-v2-after-recall"
        _http_json("PUT", f"{model_url}/control/barriers/{barrier}")
        _load_scripts(
            model_url,
            [
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_akasha_tool_search",
                            "name": "tool_search",
                            "arguments": {"query": "select:recall_memory"},
                        }
                    ],
                },
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_akasha_recall",
                            "name": "recall_memory",
                            "arguments": {
                                "query": "alpha medical story",
                                "limit": 5,
                            },
                        }
                    ],
                },
                {
                    "mode": "complete",
                    "content": "second answer after recall",
                    "barrier": barrier,
                },
            ],
        )
        second_ack = client.send_programmatic(
            SESSION_ID, SECOND_INPUT_ID, "what happened next"
        )
        _wait_barrier(model_url, barrier)
        during_recall_hash = logical_state_sha256(memory_path)
        _http_json("POST", f"{model_url}/control/barriers/{barrier}/release")
        second_result = _wait_programmatic_result(
            client, SESSION_ID, SECOND_INPUT_ID, timeout=TURN_DEADLINE_S
        )
        second_page = client.read_messages(SESSION_ID)
        second_items = _page_items(second_page)
        second_hash, second_progress = _wait_learning(memory_path, minimum_applied=2)

        requests = _model_requests(_http_json("GET", f"{model_url}/control/requests"))
        recall_records = _akasha_recall_records(Path("/sandbox/workspace/sessions.db"))
        context_rows = _context_rows(requests)
        context_by_id = {
            str(row["message_id"]): row
            for row in context_rows
            if isinstance(row.get("message_id"), str)
        }
        first_message_texts = {
            str(item["id"]): _message_text(item)
            for item in first_items
            if isinstance(item.get("id"), str)
        }
        context_records: list[dict[str, object]] = []
        for key, record in recall_records.items():
            source = record.get("source")
            ids = record.get("presented_message_ids")
            if (
                not isinstance(source, dict)
                or source.get("kind") != "context"
                or source.get("session_id") != SESSION_ID
                or not isinstance(ids, list)
                or not ids
                or not all(isinstance(item, str) for item in ids)
            ):
                continue
            expected_ids = [str(item) for item in ids]
            exact_content = all(
                message_id in context_by_id
                and message_id in first_message_texts
                and context_by_id[message_id].get("text") == first_message_texts[message_id]
                for message_id in expected_ids
            )
            context_records.append(
                {
                    "key": key,
                    "sourceMessageIds": expected_ids,
                    "exactContent": exact_content,
                    "record": record,
                }
            )
        automatic_context_seen = bool(context_records) and any(
            bool(record["exactContent"]) for record in context_records
        )

        wire_tool_calls: list[dict[str, object]] = []
        wire_tool_results: list[dict[str, object]] = []
        for item in second_items:
            body = item.get("body")
            if not isinstance(body, dict):
                continue
            if body.get("kind") == "output":
                parts = body.get("parts")
                if not isinstance(parts, list):
                    continue
                for part_index, part in enumerate(parts):
                    if isinstance(part, dict) and part.get("kind") == "tool_call":
                        wire_tool_calls.append(
                            {
                                "messageId": item.get("id"),
                                "partIndex": part_index,
                                "bindingId": part.get("binding_id"),
                                "name": part.get("name"),
                                "arguments": part.get("arguments"),
                            }
                        )
            elif body.get("kind") == "tool_result":
                call_ref = body.get("call_ref")
                if isinstance(call_ref, dict):
                    wire_tool_results.append(
                        {
                            "messageId": item.get("id"),
                            "callRef": call_ref,
                            "outcome": body.get("outcome"),
                        }
                    )
        matched_tool_results = [
            {
                **call,
                "result": next(
                    (
                        result
                        for result in wire_tool_results
                        if result.get("callRef")
                        == {
                            "message_id": call.get("messageId"),
                            "part_index": call.get("partIndex"),
                        }
                    ),
                    None,
                ),
            }
            for call in wire_tool_calls
        ]
        tool_wire_ok = (
            [call.get("name") for call in wire_tool_calls]
            == ["tool_search", "recall_memory"]
            and len(matched_tool_results) == 2
            and all(
                isinstance(item.get("result"), dict)
                and item["result"].get("outcome") == "success"
                for item in matched_tool_results
            )
        )
        output_texts = [
            _message_text(item)
            for item in second_items
            if item.get("body", {}).get("kind") == "output"
        ]

        checks.extend(
            [
                CheckResult(
                    "AKV2-01",
                    first_ack.get("message_id") == FIRST_INPUT_ID
                    and first_result.get("status") == "complete"
                    and any(
                        text == "first remembered answer"
                        for text in (_message_text(item) for item in first_items)
                    ),
                    {
                        "admission": admission,
                        "ack": first_ack,
                        "result": first_result,
                        "messagePage": first_page,
                        "learning": first_progress.model_dump(mode="json"),
                    },
                ),
                CheckResult(
                    "AKV2-02",
                    automatic_context_seen,
                    {
                        "modelRequestCount": len(requests),
                        "contextRows": context_rows,
                        "contextRecallRecords": context_records,
                    },
                ),
                CheckResult(
                    "AKV2-03",
                    first_hash == during_recall_hash
                    and tool_wire_ok,
                    {
                        "beforeRecall": first_hash,
                        "duringRecall": during_recall_hash,
                        "toolCalls": wire_tool_calls,
                        "toolResults": wire_tool_results,
                        "matchedToolResults": matched_tool_results,
                    },
                ),
                CheckResult(
                    "AKV2-04",
                    second_ack.get("message_id") == SECOND_INPUT_ID
                    and second_result.get("status") == "complete"
                    and "second answer after recall" in output_texts
                    and second_hash != first_hash,
                    {
                        "ack": second_ack,
                        "result": second_result,
                        "messagePage": second_page,
                        "beforeCommit": first_hash,
                        "afterCommit": second_hash,
                        "learning": second_progress.model_dump(mode="json"),
                    },
                ),
            ]
        )
        scenario_state = {
            "sessionId": SESSION_ID,
            "inputIds": [FIRST_INPUT_ID, SECOND_INPUT_ID],
            "messagePage": second_page,
            "learningHash": second_hash,
            "consumerProgress": second_progress.model_dump(mode="json"),
            "providerRequestCount": len(requests),
            "initialLearningHash": initial_hash,
            "initialConsumerProgress": initial_progress.model_dump(mode="json"),
            "bootIdBefore": status_before["bootId"],
        }
    except Exception as error:
        checks.append(
            CheckResult(
                "AKV2-controller",
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
        )
    finally:
        if client is not None:
            client.close()

    report = {
        "checks": [asdict(check) for check in checks],
        "passed": bool(checks) and all(check.passed for check in checks),
        "scenario": scenario_state,
    }
    (report_dir / "akasha-v2-inside.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


def _inside_restart_check(report_dir: Path) -> int:
    """Reload the published graph and read it without sending or embedding."""

    state = json.loads((report_dir / "akasha-v2-state.json").read_text(encoding="utf-8"))
    endpoint = Path("/sandbox/akashic.sock")
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    client: JsonRpcSocketClient | None = None
    try:
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        ready_deadline = time.monotonic() + READINESS_DEADLINE_S
        while True:
            try:
                client = _connect_client(endpoint, report_dir / "restart-events.jsonl")
                break
            except GateFailure:
                if time.monotonic() >= ready_deadline:
                    raise
                threading.Event().wait(0.05)
        status = client.request("server/status", {}).get("result")
        session_id = state.get("sessionId")
        input_ids = state.get("inputIds")
        expected_page = state.get("messagePage")
        if (
            not isinstance(session_id, str)
            or not isinstance(input_ids, list)
            or not isinstance(expected_page, dict)
        ):
            raise GateFailure(f"restart state fields invalid: {state!r}")
        page = client.read_messages(session_id)
        results = [
            client.programmatic_result(session_id, input_id)
            for input_id in input_ids
            if isinstance(input_id, str)
        ]
        request_count = len(_model_requests(_http_json("GET", f"{model_url}/control/requests")))
        passed = (
            isinstance(status, dict)
            and status.get("ready") is True
            and status.get("protocolVersion") == "2.0"
            and isinstance(state.get("bootIdBefore"), str)
            and isinstance(status.get("bootId"), str)
            and state.get("bootIdBefore") != status.get("bootId")
            and page == expected_page
            and all(result.get("status") == "complete" for result in results)
            and request_count == state.get("providerRequestCount")
        )
        evidence = {
            "status": status,
            "bootIdBefore": state.get("bootIdBefore"),
            "bootIdAfter": status.get("bootId") if isinstance(status, dict) else None,
            "messagePageBefore": expected_page,
            "messagePageAfter": page,
            "results": results,
            "providerRequestCountBefore": state.get("providerRequestCount"),
            "providerRequestCountAfter": request_count,
        }
    except Exception as error:
        passed = False
        evidence = {"type": type(error).__name__, "message": str(error)}
    finally:
        if client is not None:
            client.close()
    result = CheckResult("AKV2-05-read", passed, evidence)
    (report_dir / "akasha-v2-restart.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if passed else 1


def _run_controller(
    repo: Path,
    formal_workspace: Path | None,
    *,
    local_fixture: bool = False,
) -> int:
    """Run the isolated online scenario, restart it, and compare durable state."""

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    report_dir = repo / "docker/debug/reports/akasha-v2-runtime" / run_id
    report_dir.mkdir(parents=True)
    sandbox: Path | None = None
    compose_override: Path | None = None
    embedding_server: _EmbeddingFixtureServer | None = None
    embedding_thread: threading.Thread | None = None
    embedding_observation: dict[str, object]
    formal_before = _formal_identity(formal_workspace) if formal_workspace else None
    source_before = _source_identity(repo)
    if local_fixture:
        embedding_server, embedding_thread, embedding_url = _start_embedding_fixture()
        embedding_observation = {
            "scope": "local-fixture",
            "provider": "local",
            "externalProvider": "unverified",
            "url": embedding_url,
        }
    else:
        _embedding_environment()
        embedding_observation = {
            "scope": "external-provider",
            "provider": "external",
            "externalProvider": "unverified",
            "calls": "unavailable",
        }
    checks: list[CheckResult] = []
    controller_error = ""
    cleanup_returncode = -1
    residual: list[str] = []
    sandbox_app_before: str | None = None
    sandbox_app_after: str | None = None
    sandbox_cleanup_error: str | None = None
    compose_cleanup_error: str | None = None
    compose_override_cleanup_error: str | None = None
    fixture_cleanup_error: str | None = None
    sandbox_preserved = False
    sandbox_path = ""
    env: dict[str, str] = {}
    compose: list[str] = []
    try:
        sandbox = Path(tempfile.mkdtemp(prefix="akashic-akasha-v2-gate-", dir="/tmp"))
        sandbox_path = str(sandbox)
        _prepare_host_sandbox(sandbox, repo)
        _write_runtime_config(sandbox)
        sandbox_app_before = _tree_digest(sandbox / "app")
        env = {
            **os.environ,
            "AKASHIC_CONTROL_SANDBOX": str(sandbox),
            "UID": str(os.getuid()),
            "GID": str(os.getgid()),
        }
        if local_fixture:
            assert embedding_server is not None
            env.update(
                {
                    "AKASHIC_E2E_EMBEDDING_API_KEY": "akasha-local-fixture-key",
                    "AKASHIC_E2E_EMBEDDING_BASE_URL": embedding_observation["url"],
                    "AKASHIC_E2E_EMBEDDING_MODEL": embedding_server.state.model,
                }
            )
        project = f"akashic-akasha-v2-{run_id.lower()}"
        compose = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(repo / "docker/debug/docker-compose.control-gate.yml"),
        ]
        if local_fixture:
            override_fd, override_name = tempfile.mkstemp(
                prefix="akashic-akasha-v2-compose-", suffix=".yml", dir="/tmp"
            )
            os.close(override_fd)
            compose_override = Path(override_name)
            compose_override.write_text(
                "services:\n"
                "  akashic-control-gate:\n"
                "    extra_hosts:\n"
                "      - host.docker.internal:host-gateway\n"
                "  control-probe:\n"
                "    extra_hosts:\n"
                "      - host.docker.internal:host-gateway\n",
                encoding="utf-8",
            )
            compose.extend(["-f", str(compose_override)])

        build = subprocess.run([*compose, "build", "model-gate"], cwd=repo, env=env, check=False)
        if build.returncode != 0:
            raise GateFailure(f"control-gate image build failed: {build.returncode}")
        up = subprocess.run(
            [*compose, "up", "-d", "model-gate", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if up.returncode != 0:
            raise GateFailure(f"compose up failed: {up.returncode}")
        inside = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "control-probe",
                "python",
                "docker/debug/akasha_v2_runtime_probe.py",
                "--inside-container",
                "--report-dir",
                "/sandbox/reports",
            ],
            cwd=repo,
            env=env,
            check=False,
        )
        inside_report = sandbox / "reports/akasha-v2-inside.json"
        if not inside_report.exists():
            raise GateFailure("inside Akasha V2 report missing")
        inside_payload = json.loads(inside_report.read_text(encoding="utf-8"))
        checks.extend(CheckResult(**item) for item in inside_payload.get("checks", []))
        scenario = inside_payload.get("scenario")
        if not isinstance(scenario, dict):
            raise GateFailure("inside Akasha V2 scenario state missing")
        (report_dir / "akasha-v2-inside.json").write_text(
            json.dumps(inside_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if inside.returncode != 0:
            raise GateFailure(f"inside Akasha V2 probe failed: {inside.returncode}")

        sessions_db = sandbox / "workspace/sessions.db"
        memory_db = sandbox / "workspace/memory/akasha.db"
        before_state = _database_snapshot(sessions_db, memory_db)
        scenario_state = dict(scenario)
        if (
            before_state.get("logicalState") != scenario_state.get("learningHash")
            or before_state.get("consumerProgress")
            != scenario_state.get("consumerProgress")
        ):
            raise GateFailure("inside learning evidence differs from host snapshot")
        scenario_state["beforeRestart"] = before_state
        (sandbox / "reports/akasha-v2-state.json").write_text(
            json.dumps(scenario_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if embedding_server is not None:
            embedding_observation["callsBeforeRestart"] = embedding_server.state.count()
        runtime_before = _runtime_identity(compose, repo, env)
        stop = subprocess.run(
            [*compose, "stop", "-t", "15", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if stop.returncode != 0:
            raise GateFailure(f"gateway stop failed: {stop.returncode}")
        start = subprocess.run(
            [*compose, "start", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if start.returncode != 0:
            raise GateFailure(f"gateway restart failed: {start.returncode}")
        runtime_after = _runtime_identity(compose, repo, env)
        restart = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "control-probe",
                "python",
                "docker/debug/akasha_v2_runtime_probe.py",
                "--inside-container",
                "--phase",
                "restart-check",
                "--report-dir",
                "/sandbox/reports",
            ],
            cwd=repo,
            env=env,
            check=False,
        )
        restart_path = sandbox / "reports/akasha-v2-restart.json"
        if not restart_path.exists():
            raise GateFailure("restart Akasha V2 report missing")
        restart_result = CheckResult(**json.loads(restart_path.read_text(encoding="utf-8")))
        checks.append(restart_result)
        (report_dir / "akasha-v2-restart.json").write_text(
            restart_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        after_state = _database_snapshot(sessions_db, memory_db)
        state_equal = before_state == after_state
        runtime_changed = (
            runtime_before.get("pid") != runtime_after.get("pid")
            or runtime_before.get("starttime") != runtime_after.get("starttime")
        )
        logical_equal = before_state.get("logicalState") == after_state.get("logicalState")
        progress_equal = before_state.get("consumerProgress") == after_state.get("consumerProgress")
        messages_equal = before_state.get("messages") == after_state.get("messages")
        embeddings_equal = before_state.get("embeddings") == after_state.get("embeddings")
        if embedding_server is not None:
            embedding_observation["callsAfterRestart"] = embedding_server.state.count()
            embedding_observation["delta"] = (
                embedding_observation["callsAfterRestart"]
                - embedding_observation["callsBeforeRestart"]
            )
            embedding_unchanged = (
                embedding_observation["callsBeforeRestart"] > 0
                and embedding_observation["delta"] == 0
            )
        else:
            embedding_unchanged = True
        restart_evidence = restart_result.evidence
        boot_changed = (
            isinstance(restart_evidence, dict)
            and isinstance(restart_evidence.get("bootIdBefore"), str)
            and isinstance(restart_evidence.get("bootIdAfter"), str)
            and restart_evidence.get("bootIdBefore")
            != restart_evidence.get("bootIdAfter")
        )
        checks.append(
            CheckResult(
                "AKV2-05",
                restart.returncode == 0
                and restart_result.passed
                and runtime_changed
                and boot_changed
                and logical_equal
                and progress_equal
                and messages_equal
                and embeddings_equal
                and state_equal
                and embedding_unchanged,
                {
                    "runtimeBefore": runtime_before,
                    "runtimeAfter": runtime_after,
                    "runtimeChanged": runtime_changed,
                    "bootIdBefore": (
                        restart_evidence.get("bootIdBefore")
                        if isinstance(restart_evidence, dict)
                        else None
                    ),
                    "bootIdAfter": (
                        restart_evidence.get("bootIdAfter")
                        if isinstance(restart_evidence, dict)
                        else None
                    ),
                    "bootChanged": boot_changed,
                    "logicalStateBefore": before_state.get("logicalState"),
                    "logicalStateAfter": after_state.get("logicalState"),
                    "logicalStateUnchanged": logical_equal,
                    "consumerProgressUnchanged": progress_equal,
                    "rawMessagesUnchanged": messages_equal,
                    "embeddingsUnchanged": embeddings_equal,
                    "embeddingCallObservation": embedding_observation,
                    "restartProbeReturncode": restart.returncode,
                },
            )
        )
        (report_dir / "akasha-v2-state-before.json").write_text(
            json.dumps(before_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (report_dir / "akasha-v2-state-after.json").write_text(
            json.dumps(after_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        if compose:
            try:
                logs = subprocess.run(
                    [*compose, "logs", "--no-color"],
                    cwd=repo,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                (report_dir / "compose.log").write_text(logs.stdout, encoding="utf-8")
            except Exception as error:
                controller_error = controller_error or f"compose logs failed: {error}"
            try:
                cleanup = subprocess.run(
                    [*compose, "down", "--remove-orphans", "--volumes"],
                    cwd=repo,
                    env=env,
                    check=False,
                )
                cleanup_returncode = cleanup.returncode
            except Exception as error:
                compose_cleanup_error = f"{type(error).__name__}: {error}"
                controller_error = controller_error or compose_cleanup_error
            try:
                residual = subprocess.run(
                    [*compose, "ps", "-aq"],
                    cwd=repo,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                ).stdout.split()
            except Exception as error:
                compose_cleanup_error = compose_cleanup_error or f"{type(error).__name__}: {error}"
                controller_error = controller_error or compose_cleanup_error
        if sandbox is not None and (sandbox / "app").exists():
            try:
                sandbox_app_after = _tree_digest(sandbox / "app")
            except Exception as error:
                sandbox_cleanup_error = f"sandbox app digest failed: {error}"
                controller_error = controller_error or sandbox_cleanup_error
        try:
            formal_after = _formal_identity(formal_workspace) if formal_workspace else None
        except Exception as error:
            formal_after = None
            controller_error = controller_error or f"formal workspace snapshot failed: {error}"
        try:
            source_after = _source_identity(repo)
        except Exception as error:
            source_after = {}
            controller_error = controller_error or f"source snapshot failed: {error}"
        if sandbox is not None:
            try:
                shutil.rmtree(sandbox)
            except OSError as error:
                sandbox_cleanup_error = f"{type(error).__name__}: {error}"
                sandbox_preserved = True
                controller_error = controller_error or sandbox_cleanup_error
        if compose_override is not None:
            try:
                compose_override.unlink()
            except OSError as error:
                compose_override_cleanup_error = f"{type(error).__name__}: {error}"
                controller_error = controller_error or compose_override_cleanup_error
        if embedding_server is not None and embedding_thread is not None:
            try:
                _stop_embedding_fixture(embedding_server, embedding_thread)
            except Exception as error:
                fixture_cleanup_error = f"{type(error).__name__}: {error}"
                controller_error = controller_error or fixture_cleanup_error
        checks.append(
            CheckResult(
                "AKV2-06",
                cleanup_returncode == 0
                and not residual
                and formal_before == formal_after
                and source_before == source_after
                and sandbox_app_before == sandbox_app_after
                and sandbox_cleanup_error is None
                and compose_cleanup_error is None
                and compose_override_cleanup_error is None
                and fixture_cleanup_error is None,
                {
                    "cleanupReturncode": cleanup_returncode,
                    "residualContainers": residual,
                    "formalWorkspaceUnchanged": formal_before == formal_after,
                    "sourceBefore": source_before,
                    "sourceAfter": source_after,
                    "sourceUnchanged": source_before == source_after,
                    "sandboxPath": sandbox_path,
                    "sandboxAppDigestBefore": sandbox_app_before,
                    "sandboxAppDigestAfter": sandbox_app_after,
                    "sandboxAppUnchanged": sandbox_app_before == sandbox_app_after,
                    "sandboxCleanupError": sandbox_cleanup_error,
                    "sandboxPreserved": sandbox_preserved,
                    "composeCleanupError": compose_cleanup_error,
                    "composeOverrideCleanupError": compose_override_cleanup_error,
                    "fixtureCleanupError": fixture_cleanup_error,
                },
            )
        )

    passed = not controller_error and bool(checks) and all(check.passed for check in checks)
    report = {
        "runId": run_id,
        "status": "passed" if passed else "failed",
        "checks": [asdict(check) for check in checks],
        "controllerError": controller_error,
        "reportDir": str(report_dir),
        "embeddingCallObservation": embedding_observation,
    }
    (report_dir / "gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--phase", choices=("scenario", "restart-check"), default="scenario")
    parser.add_argument("--report-dir", type=Path, default=Path("/sandbox/reports"))
    parser.add_argument("--formal-workspace", type=Path, default=None)
    parser.add_argument(
        "--local-fixture",
        action="store_true",
        help="use an observable local OpenAI-compatible embedding fixture",
    )
    arguments = parser.parse_args()
    if arguments.inside_container:
        if arguments.phase == "restart-check":
            return _inside_restart_check(arguments.report_dir)
        return _inside_scenario(arguments.report_dir)
    formal_workspace = (
        arguments.formal_workspace.resolve(strict=True)
        if arguments.formal_workspace
        else None
    )
    return _run_controller(
        Path(__file__).resolve().parents[2],
        formal_workspace,
        local_fixture=arguments.local_fixture,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error
