#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import venv
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugins.reload_journal import ReloadJournal
from docker.debug.programmatic_control_probe import (
    CheckResult,
    GateFailure,
    JsonRpcSocketClient,
    _configure_model_gate,
    _connect_client,
    _http_json,
    _initialize_current_workspace,
    _model_requests,
    _prepare_host_sandbox,
    _repository_digest,
    _wait_barrier,
    _wait_http_ready,
    _wait_socket,
    _write_json,
)
from infra.persistence.json_store import atomic_write_text
from plugins.turn_projection.plugin import TurnProjection
from session.message import Message
from session.message_codec import decode_body

READINESS_DEADLINE_S = 30.0
SCENARIO_DEADLINE_S = 15.0
MODEL_URL = "http://model-gate:8090"
ENDPOINT = Path("/sandbox/akashic.sock")
WORKSPACE = Path("/sandbox/workspace")

MCP_SERVER_SOURCE = r"""from __future__ import annotations
import json
import os
from pathlib import Path
import sys

log = Path(os.environ["LIFECYCLE_LOG"])
version = os.environ["VERSION"]
pid = os.getpid()

def record(event: str) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "pid": pid, "version": version}) + "\n")

record("started")
try:
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25"}
        elif method == "tools/list":
            result = {"tools": [{"name": "version", "description": "Return version", "inputSchema": {"type": "object", "properties": {}}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": version}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
finally:
    record("stopped")
"""


def _load_scripts(scripts: list[dict[str, object]]) -> int:
    payload = _http_json("PUT", f"{MODEL_URL}/control/script", scripts)
    loaded = cast(dict[str, object], payload)["loaded"]
    if not isinstance(loaded, int):
        raise GateFailure(f"model gate loaded 响应非法: {payload!r}")
    return loaded


def _requests() -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        _model_requests(_http_json("GET", f"{MODEL_URL}/control/requests")),
    )


def _page_items(page: object) -> list[dict[str, Any]]:
    if not isinstance(page, dict) or not isinstance(page.get("items"), list):
        raise GateFailure(f"message/read 页面非法：{page!r}")
    return [item for item in page["items"] if isinstance(item, dict)]


def _body_kind(item: dict[str, Any], kind: str) -> bool:
    body = item.get("body")
    return isinstance(body, dict) and body.get("kind") == kind


def _tool_calls(page: object) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _page_items(page):
        body = item.get("body")
        if not isinstance(body, dict) or body.get("kind") != "output":
            continue
        parts = body.get("parts")
        if not isinstance(parts, list):
            continue
        for index, part in enumerate(parts):
            if isinstance(part, dict) and part.get("kind") == "tool_call":
                calls.append({
                    "messageId": item.get("id"),
                    "partIndex": index,
                    "bindingId": part.get("binding_id"),
                    "name": part.get("name"),
                    "arguments": part.get("arguments"),
                })
    return calls


def _tool_results(page: object) -> list[dict[str, Any]]:
    return [item for item in _page_items(page) if _body_kind(item, "tool_result")]


def _tool_result_for_call(
    page: object, call: dict[str, Any]
) -> dict[str, Any] | None:
    call_ref = (call.get("messageId"), call.get("partIndex"))
    for item in _tool_results(page):
        body = item.get("body")
        ref = body.get("call_ref") if isinstance(body, dict) else None
        if isinstance(ref, dict) and (ref.get("message_id"), ref.get("part_index")) == call_ref:
            return item
    return None


def _tool_result_json(item: dict[str, Any] | None) -> dict[str, Any] | None:
    """Decode the JSON text returned by a discovery ToolResult."""

    try:
        value = json.loads(_body_text(item))
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _output_text(item: dict[str, Any] | None) -> str:
    if item is None or not isinstance(item.get("body"), dict):
        return ""
    parts = item["body"].get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part.get("value", ""))
        for part in parts
        if isinstance(part, dict) and part.get("kind") == "text"
    )


def _body_text(item: dict[str, Any] | None) -> str:
    """Extract text ContentParts from an Output or ToolResult row."""

    if item is None or not isinstance(item.get("body"), dict):
        return ""
    parts = item["body"].get("parts")
    if not isinstance(parts, list):
        return ""
    values = [
        part.get("value")
        for part in parts
        if isinstance(part, dict) and part.get("kind") == "text"
    ]
    return "".join(value if isinstance(value, str) else str(value) for value in values)


def _final_output(page: object, result: dict[str, Any]) -> dict[str, Any] | None:
    ending_id = result.get("ending_message_id")
    if not isinstance(ending_id, str):
        return None
    for item in _page_items(page):
        if item.get("id") == ending_id and _body_kind(item, "output"):
            body = item.get("body")
            if isinstance(body, dict) and body.get("finish") == "complete":
                return item
    return None


def _raw_messages(session_id: str) -> list[Message]:
    """Read the append-only Message log for projection evidence."""

    try:
        with sqlite3.connect(WORKSPACE / "sessions.db") as connection:
            rows = connection.execute(
                "SELECT id, session_key, seq, ts, author, source, body "
                "FROM messages WHERE session_key = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [
            Message(
                message_id=str(row[0]),
                session_id=str(row[1]),
                seq=int(row[2]),
                recorded_at=datetime.fromisoformat(str(row[3])),
                author=str(row[4]),
                source=str(row[5]),
                body=decode_body(str(row[6])),
            )
            for row in rows
        ]
    except (sqlite3.Error, KeyError, TypeError, ValueError) as error:
        raise GateFailure(f"读取 raw Message 失败：{session_id}") from error


def _raw_message_rows(session_id: str) -> list[dict[str, str | int]]:
    """Capture exact append-only SQLite rows for a restart continuity check."""

    try:
        with sqlite3.connect(WORKSPACE / "sessions.db") as connection:
            rows = connection.execute(
                "SELECT id, session_key, seq, ts, author, source, body "
                "FROM messages WHERE session_key = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise GateFailure(f"读取 raw Message 行失败：{session_id}") from error
    return [
        {
            "id": str(row[0]),
            "sessionKey": str(row[1]),
            "seq": int(row[2]),
            "ts": str(row[3]),
            "author": str(row[4]),
            "source": str(row[5]),
            "body": str(row[6]),
        }
        for row in rows
    ]


def _projection_evidence(session_id: str, page: object) -> dict[str, Any]:
    raw = _raw_messages(session_id)
    projection = TurnProjection().project(raw, "programmatic") if raw else ()
    wire = _page_items(page)
    raw_ids = [item.message_id for item in raw]
    wire_ids = [str(item.get("id")) for item in wire]
    raw_seqs = [item.seq for item in raw]
    wire_seqs = [item.get("seq") for item in wire]
    return {
        "rawMessageCount": len(raw),
        "rawMessageIds": raw_ids,
        "rawMessageSeqs": raw_seqs,
        "wireMessageIds": wire_ids,
        "wireMessageSeqs": wire_seqs,
        "wireMatchesRaw": wire_ids == raw_ids and wire_seqs == raw_seqs,
        "turnProjection": [
            {
                "afterSeq": turn.after_seq,
                "throughSeq": turn.through_seq,
                "endingMessageId": turn.ending_message_id,
                "status": turn.status,
                "messageIds": list(turn.message_ids),
                "toolResultRefs": [
                    {"messageId": ref.message_id, "partIndex": ref.part_index}
                    for ref, _ in turn.observations
                ],
            }
            for turn in projection
        ],
    }


def _wait_page_event(
    client: JsonRpcSocketClient,
    session_id: str,
    subscription_id: str,
    predicate: object,
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drain follow events until the durable page satisfies a predicate."""

    deadline = time.monotonic() + timeout
    events: list[dict[str, Any]] = []
    page: dict[str, Any] = {
        "version": 2,
        "session_id": session_id,
        "items": [],
        "after_seq": -1,
        "through_seq": -1,
        "next_after_seq": -1,
        "has_more": False,
    }
    check = cast(Any, predicate)
    while time.monotonic() < deadline:
        event = client.wait_session_event(
            "messages.appended",
            subscription_id=subscription_id,
            timeout=max(0.01, deadline - time.monotonic()),
        )
        events.append(event)
        params = event.get("params")
        payload = params.get("event") if isinstance(params, dict) else None
        items = payload.get("items") if isinstance(payload, dict) else None
        if isinstance(items, list):
            by_id = {
                str(item.get("id")): item
                for item in page["items"]
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    by_id[str(item["id"])] = item
            page["items"] = sorted(
                by_id.values(), key=lambda item: int(item.get("seq", -1))
            )
        if isinstance(payload, dict):
            for key in ("after_seq", "through_seq", "next_after_seq", "has_more"):
                if key in payload:
                    page[key] = payload[key]
        if check(page):
            return page, events
    raise GateFailure(f"{session_id} Message drain 超时：{page!r}")


def _wait_programmatic_result(
    client: JsonRpcSocketClient,
    session_id: str,
    input_id: str,
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    result: dict[str, Any] = {}
    while time.monotonic() < deadline:
        result = client.programmatic_result(session_id, input_id)
        if result.get("status") != "open":
            return result
        time.sleep(0.05)
    raise GateFailure(f"programmatic result 超时：{session_id}/{input_id} {result!r}")


def _admit_follow(
    client: JsonRpcSocketClient,
    session_id: str,
    subscription_id: str,
    *,
    after_seq: int = -1,
) -> dict[str, Any]:
    admission = client.admit_programmatic(session_id)
    follow = client.follow_session(session_id, subscription_id, after_seq=after_seq)
    if admission.get("session_id") != session_id or follow.get("subscription_id") != subscription_id:
        raise GateFailure(f"programmatic Session 准入/follow 异常：{admission!r} {follow!r}")
    return follow


def _send_input(
    client: JsonRpcSocketClient,
    session_id: str,
    message_id: str,
    text: str,
    subscription_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Follow before sending one Input, then return durable ACK and result."""

    _admit_follow(client, session_id, subscription_id)
    ack = client.send_programmatic(session_id, message_id, text)
    if ack.get("message_id") != message_id or not isinstance(ack.get("seq"), int):
        raise GateFailure(f"programmatic Input ACK 异常：{ack!r}")
    result = _wait_programmatic_result(client, session_id, message_id)
    return ack, result


def _tool_names(request: dict[str, Any]) -> set[str]:
    payload = request.get("payload")
    if not isinstance(payload, dict):
        raise GateFailure(f"model request 缺少 payload: {request!r}")
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return set()
    return {
        str(item.get("function", {}).get("name"))
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }


def _read_ready() -> dict[str, Any]:
    path = WORKSPACE / ".runtime-ready.json"
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        if payload.get("state") == "ready":
            return cast(dict[str, Any], payload)
        time.sleep(0.02)
    raise GateFailure("runtime readiness 超时")


def _connect_new_boot(
    old_boot: str, events_path: Path
) -> tuple[JsonRpcSocketClient, dict[str, Any]]:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        ready = _read_ready()
        if ready.get("bootId") != old_boot:
            try:
                return _connect_client(ENDPOINT, events_path), ready
            except (ConnectionError, OSError, GateFailure):
                pass
        time.sleep(0.02)
    raise GateFailure(f"等待新 boot 超时: old={old_boot}")


def _restart_scripts(index: int, barrier: str) -> list[dict[str, object]]:
    return [
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [
                {
                    "id": f"call_search_{index}",
                    "name": "tool_search",
                    "arguments": {"query": "select:agent_restart"},
                }
            ],
        },
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [
                {
                    "id": f"call_restart_{index}",
                    "name": "agent_restart",
                    "arguments": {"reason": f"restart gate iteration {index}"},
                }
            ],
        },
        {
            "mode": "complete",
            "content": f"restart-complete-{index}",
            "barrier": barrier,
        },
    ]


def _mcp_scripts(version: str) -> list[dict[str, object]]:
    return [
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [
                {
                    "id": f"call_mcp_search_{version}",
                    "name": "tool_search",
                    "arguments": {"query": "select:mcp_restart_probe__version"},
                }
            ],
        },
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [
                {
                    "id": f"call_mcp_version_{version}",
                    "name": "mcp_restart_probe__version",
                    "arguments": {},
                }
            ],
        },
        {"mode": "complete", "content": f"mcp-{version}"},
    ]


def _process_identity(pid: int) -> dict[str, int]:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = stat[stat.rfind(")") + 2 :].split()
    return {"pid": pid, "starttime": int(fields[19])}


def _identity_alive(identity: dict[str, int]) -> bool:
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = stat[stat.rfind(")") + 2 :].split()
    return (
        fields[0] != "Z"
        and {
            "pid": identity["pid"],
            "starttime": int(fields[19]),
        }
        == identity
    )


def _running_mcp_identity(
    version: str,
    *,
    workspace: Path = WORKSPACE,
    previous: dict[str, int] | None = None,
) -> dict[str, int]:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    lifecycle = workspace / "mcp/restart-probe-lifecycle.jsonl"
    while time.monotonic() < deadline:
        records = (
            [json.loads(line) for line in lifecycle.read_text().splitlines() if line]
            if lifecycle.exists()
            else []
        )
        started = [
            int(item["pid"])
            for item in records
            if item["event"] == "started" and item["version"] == version
        ]
        for pid in reversed(started):
            try:
                identity = _process_identity(pid)
            except FileNotFoundError:
                continue
            if identity != previous:
                return identity
        time.sleep(0.02)
    raise GateFailure(f"MCP {version} 子进程未启动")


def _wait_identity_exit(identity: dict[str, int]) -> None:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        if not _identity_alive(identity):
            return
        time.sleep(0.02)
    raise GateFailure(f"旧进程 identity 未退出: {identity}")


def _wait_reload_complete(
    journal: ReloadJournal,
    *,
    previous_tx_id: str | None,
) -> None:
    """等待 watcher 发布新的 restart-probe generation。"""

    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        current = journal.latest(plugin_id="restart_probe")
        if current is None or current.tx_id == previous_tx_id:
            time.sleep(0.02)
            continue
        if current.phase == "complete":
            return
        if current.phase in {"aborted", "recovered", "cleanup_failed", "degraded"}:
            raise GateFailure(
                "restart_probe 热重载失败: "
                f"phase={current.phase}, error={current.error}"
            )
        time.sleep(0.02)
    raise GateFailure("restart_probe 热重载未在限时内完成")


def _write_mcp_plugin(
    version: str,
    *,
    plugin_root: Path,
    workspace: Path = WORKSPACE,
    runtime_workspace: Path | None = None,
    stage_runtime: bool = True,
) -> None:
    """Write one disposable pure-v3 MCP plugin generation."""

    # 1. 写入与真实 Root 声明一致的 v3 module。
    lifecycle_file = Path("mcp/restart-probe-lifecycle.jsonl")
    (workspace / lifecycle_file).parent.mkdir(parents=True, exist_ok=True)
    lifecycle = (runtime_workspace or workspace) / lifecycle_file
    plugin_root.mkdir(parents=True, exist_ok=True)
    module = plugin_root / "plugin.py"
    if not module.exists():
        module.write_text(
            "from collections.abc import AsyncIterator, Mapping\n"
            "from contextlib import asynccontextmanager\n"
            "from pathlib import Path\n"
            "import tomllib\n"
            "from agent.plugin_composition import MCP_SERVERS, McpServerDefinition\n"
            "from plugins.tools.api import BoundTool, CallSource, ContentPart, Result\n"
            "from plugins.tools.plugin import TOOLS\n"
            "_manifest = tomllib.loads(\n"
            "    Path(__file__).with_name('akashic.plugin.toml').read_text(encoding='utf-8')\n"
            ")\n"
            "api_version = 3\n"
            "name = 'restart_probe'\n"
            "version = str(_manifest['version'])\n"
            "inject = (MCP_SERVERS, TOOLS)\n"
            "\n"
            "class VersionTool:\n"
            "    def __init__(self, ctx):\n"
            "        self._ctx = ctx\n"
            "\n"
            "    idempotent = True\n"
            "\n"
            "    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:\n"
            "        if arguments:\n"
            "            raise ValueError('version 工具不接受参数')\n"
            "        return {}\n"
            "\n"
            "    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:\n"
            "        if arguments:\n"
            "            raise ValueError('version 工具不接受参数')\n"
            "        async with self._ctx.require(MCP_SERVERS).open(self._ctx, 'restart_probe') as server:\n"
            "            async with server.route() as route:\n"
            "                call = await route.call('version', {})\n"
            "        if call.status != 'success':\n"
            "            return Result('error', (ContentPart('text', call.output),))\n"
            "        return Result('success', (ContentPart('text', call.output),))\n"
            "\n"
            "    async def query(self, key: str) -> Result | None:\n"
            "        return None\n"
            "\n"
            "async def apply(ctx, config):\n"
            "    @asynccontextmanager\n"
            "    async def open_version_for_context(_state: Mapping[str, object]) -> AsyncIterator[BoundTool]:\n"
            "        yield VersionTool(ctx)\n"
            "    await ctx.require(TOOLS).register(\n"
            "        ctx, name='mcp_restart_probe__version',\n"
            "        description='Read the live restart probe MCP server version.',\n"
            "        parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},\n"
            "        open=open_version_for_context, idempotent=True, risk='read-only',\n"
            "        preloadable=False, requires_search=True,\n"
            "        search_hint='MCP restart probe version',\n"
            "    )\n"
            "    await ctx.require(MCP_SERVERS).register(\n"
            "        ctx, McpServerDefinition(\n"
            "            name='restart_probe',\n"
            "            command=('python', 'restart_probe_server.py'),\n"
            f"            env={{'VERSION': version, "
            f"'LIFECYCLE_LOG': {str(lifecycle)!r}}},\n"
            "            required_tools=('version',),\n"
            "            candidate_read_only_tools=('version',),\n"
            "        ),\n"
            "    )\n",
            encoding="utf-8",
        )

    # 2. 静态 manifest 冻结同一 MCP 合同，server 只写 disposable lifecycle。
    server_source = plugin_root / "restart_probe_server.py"
    if (
        not server_source.exists()
        or server_source.read_text(encoding="utf-8") != MCP_SERVER_SOURCE
    ):
        server_source.write_text(MCP_SERVER_SOURCE, encoding="utf-8")
    requirements = plugin_root / "requirements.txt"
    if not requirements.exists():
        requirements.write_text("", encoding="utf-8")
    runtime = plugin_root / ".venv"
    if stage_runtime and not runtime.exists():
        venv.EnvBuilder(with_pip=False).create(runtime)
    atomic_write_text(
        plugin_root / "akashic.plugin.toml",
        "schema_version = 1\n"
        "name = 'restart_probe'\n"
        f"version = {version!r}\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[[python]]\n"
        "requirements = 'requirements.txt'\n\n"
        "[[mcp]]\n"
        "name = 'restart_probe'\n"
        "command = ['python', 'restart_probe_server.py']\n"
        f"env = {{VERSION = {version!r}, "
        f"LIFECYCLE_LOG = {str(lifecycle)!r}}}\n"
        "required_tools = ['version']\n"
        "candidate_read_only_tools = ['version']\n",
        domain="restart_gate_fixture",
    )


def _run_mcp_call(
    client: JsonRpcSocketClient,
    version: str,
    label: str,
) -> CheckResult:
    """Run one MCP call through a real programmatic Message Session."""

    before = len(_requests())
    _load_scripts(_mcp_scripts(version))
    session_id = f"programmatic:mcp-{label}-{uuid.uuid4().hex[:8]}"
    input_id = f"mcp-input-{label}"
    subscription_id = f"mcp-follow-{label}"
    follow = _admit_follow(client, session_id, subscription_id)
    ack = client.send_programmatic(session_id, input_id, f"call MCP {version}")
    result = _wait_programmatic_result(client, session_id, input_id)
    page = client.read_messages(session_id)
    if _final_output(page, result) is None:
        page, events = _wait_page_event(
            client,
            session_id,
            subscription_id,
            lambda value: _final_output(value, result) is not None,
        )
    else:
        events = []
    requests = _requests()[before:]
    calls = _tool_calls(page)
    results = _tool_results(page)
    version_call = next((item for item in calls if item.get("name") == "mcp_restart_probe__version"), None)
    version_result = _tool_result_for_call(page, version_call) if version_call is not None else None
    tool_result_names = [
        item.get("body", {}).get("outcome")
        for item in results
        if isinstance(item.get("body"), dict)
    ]
    projection = _projection_evidence(session_id, page)
    projected = projection["turnProjection"]
    final_output = _final_output(page, result)
    passed = (
        len(requests) == 3
        and "mcp_restart_probe__version" not in _tool_names(requests[0])
        and "mcp_restart_probe__version" in _tool_names(requests[1])
        and version_call is not None
        and version_result is not None
        and version_result.get("body", {}).get("outcome") == "success"
        and _body_text(version_result) == version
        and len(results) >= 2
        and all(outcome == "success" for outcome in tool_result_names)
        and result.get("status") == "complete"
        and final_output is not None
        and isinstance(projected, list)
        and projected[-1].get("status") == "complete"
        and projection["wireMatchesRaw"]
    )
    return CheckResult(
        f"MCP-{label}",
        passed,
        {
            "sessionId": session_id,
            "inputId": input_id,
            "version": version,
            "follow": follow,
            "ack": ack,
            "result": result,
            "initialTools": sorted(_tool_names(requests[0])) if requests else [],
            "postSearchTools": (
                sorted(_tool_names(requests[1])) if len(requests) > 1 else []
            ),
            "toolCall": version_call,
            "versionToolResult": version_result,
            "toolResults": results,
            "messageEvents": events,
            "messagePage": page,
            "projection": projection,
        },
    )


def _sample_supervisor_children(
    supervisor_pid: int,
    stop: threading.Event,
    samples: list[int],
) -> None:
    path = Path(f"/proc/{supervisor_pid}/task/{supervisor_pid}/children")
    while not stop.is_set():
        try:
            pids = path.read_text(encoding="utf-8").split()
        except FileNotFoundError:
            pids = []
        samples.append(sum(Path(f"/proc/{pid}").exists() for pid in pids))
        stop.wait(0.002)


def _run_restart_iteration(
    index: int,
    client: JsonRpcSocketClient,
    report_dir: Path,
) -> tuple[JsonRpcSocketClient, CheckResult]:
    """Run one restart Input and recover its Message result after the new boot."""

    before = _requests()
    ready_before = _read_ready()
    old_identity = _process_identity(int(ready_before["pid"]))
    supervisor_pid = int((WORKSPACE / ".supervisor.pid").read_text())
    child_samples: list[int] = []
    stop_sampling = threading.Event()
    sampler = threading.Thread(
        target=_sample_supervisor_children,
        args=(supervisor_pid, stop_sampling, child_samples),
        daemon=True,
    )
    sampler.start()
    barrier = f"restart-final-{index}-{uuid.uuid4().hex[:8]}"
    session_id = f"programmatic:restart-{index}"
    input_id = f"restart-input-{index}"
    subscription_id = f"restart-follow-{index}"
    _http_json("PUT", f"{MODEL_URL}/control/barriers/{barrier}")
    _load_scripts(_restart_scripts(index, barrier))

    # 1. Follow the same Message Session before sending the restart Input.
    follow = _admit_follow(client, session_id, subscription_id)
    ack = client.send_programmatic(session_id, input_id, f"restart iteration {index}")
    if ack.get("message_id") != input_id or not isinstance(ack.get("seq"), int):
        raise GateFailure(f"restart Input ACK 异常：{ack!r}")

    def has_restart_result(page: object) -> bool:
        call = next((item for item in _tool_calls(page) if item.get("name") == "agent_restart"), None)
        result = _tool_result_for_call(page, call) if call is not None else None
        return result is not None and result.get("body", {}).get("outcome") == "success"

    tool_page, tool_events = _wait_page_event(
        client, session_id, subscription_id, has_restart_result
    )
    restart_call = next(
        (item for item in _tool_calls(tool_page) if item.get("name") == "agent_restart"),
        None,
    )
    if restart_call is None:
        raise GateFailure(f"restart ToolCall 缺失：{tool_page!r}")
    restart_result = _tool_result_for_call(tool_page, restart_call)
    if restart_result is None:
        raise GateFailure(f"restart ToolResult 缺失：{tool_page!r}")
    _wait_barrier(MODEL_URL, barrier)

    # 2. The watcher has prepared the gate, while the final Output is held.
    concurrent = _connect_client(
        ENDPOINT, report_dir / f"events-{index}-concurrent.jsonl"
    )
    concurrent_session = f"programmatic:restart-concurrent-{index}"
    concurrent_subscription = f"restart-concurrent-follow-{index}"
    try:
        _admit_follow(concurrent, concurrent_session, concurrent_subscription)
        rejected = concurrent.request_raw(
            "programmatic/message/send",
            {
                "session_id": concurrent_session,
                "message_id": f"concurrent-input-{index}",
                "text": "must retry",
            },
        )
    finally:
        concurrent.close()

    # 3. Release the model Output barrier and drain the complete Output on the
    # same follow connection before allowing the old runtime to disappear.
    _http_json("POST", f"{MODEL_URL}/control/barriers/{barrier}/release")
    terminal_page, terminal_events = _wait_page_event(
        client,
        session_id,
        subscription_id,
        lambda value: any(
            _body_kind(item, "output")
            and isinstance(item.get("body"), dict)
            and item["body"].get("finish") == "complete"
            and _output_text(item) == f"restart-complete-{index}"
            for item in _page_items(value)
        ),
    )
    raw_before_restart = _raw_message_rows(session_id)
    old_child_alive_at_terminal = _identity_alive(old_identity)
    client.close()
    new_client, ready_after = _connect_new_boot(
        str(ready_before["bootId"]),
            report_dir / f"events-{index}-after.jsonl",
    )
    stop_sampling.set()
    sampler.join(timeout=2)
    max_concurrent_child = max(child_samples, default=0)
    new_identity = _process_identity(int(ready_after["pid"]))

    # 4. Re-follow the same Session and recover ACK/result/raw Message page.
    recovery_subscription = f"restart-recovery-follow-{index}"
    recovery_follow = _admit_follow(new_client, session_id, recovery_subscription)
    result = _wait_programmatic_result(new_client, session_id, input_id)
    page = new_client.read_messages(session_id)
    if _final_output(page, result) is None:
        page, recovery_events = _wait_page_event(
            new_client,
            session_id,
            recovery_subscription,
            lambda value: _final_output(value, result) is not None,
        )
    else:
        recovery_events = []
    stable_ready = _read_ready()
    stable_identity = _process_identity(int(stable_ready["pid"]))
    after = _requests()
    iteration_requests = after[len(before) :]
    calls = _tool_calls(page)
    restart_result = _tool_result_for_call(page, restart_call)
    projection = _projection_evidence(session_id, page)
    final_output = _final_output(page, result)
    raw_after_restart = _raw_message_rows(session_id)
    raw_rows_match = raw_after_restart == raw_before_restart
    replay_request_count_before = len(after)
    replay_ack = new_client.send_programmatic(
        session_id, input_id, f"restart iteration {index}"
    )
    replay_result = _wait_programmatic_result(new_client, session_id, input_id)
    replay_request_count_after = len(_requests())
    raw_after_replay = _raw_message_rows(session_id)
    replay_stable = (
        replay_ack == ack
        and replay_result.get("status") == result.get("status") == "complete"
        and replay_result.get("ending_message_id") == result.get("ending_message_id")
        and replay_request_count_after == replay_request_count_before
        and raw_after_replay == raw_after_restart
    )
    error = rejected.get("error")
    passed = (
        len(iteration_requests) == 3
        and "agent_restart" not in _tool_names(iteration_requests[0])
        and "agent_restart" in _tool_names(iteration_requests[1])
        and {"tool_search", "agent_restart"} <= {str(item.get("name")) for item in calls}
        and restart_result is not None
        and restart_result.get("body", {}).get("outcome") == "success"
        and result.get("status") == "complete"
        and final_output is not None
        and _output_text(final_output) == f"restart-complete-{index}"
        and projection["wireMatchesRaw"]
        and raw_rows_match
        and replay_stable
        and projection["turnProjection"]
        and projection["turnProjection"][-1].get("status") == "complete"
        and old_child_alive_at_terminal
        and ready_after["bootId"] != ready_before["bootId"]
        and new_identity != old_identity
        and int((WORKSPACE / ".supervisor.pid").read_text()) == supervisor_pid
        and max_concurrent_child == 1
        and stable_ready["bootId"] == ready_after["bootId"]
        and stable_identity == new_identity
        and isinstance(error, dict)
        and error.get("data", {}).get("retryable") is True
    )
    return new_client, CheckResult(
        f"RESTART-{index}",
        passed,
        {
            "sessionId": session_id,
            "inputId": input_id,
            "before": ready_before,
            "beforeIdentity": old_identity,
            "after": ready_after,
            "afterIdentity": new_identity,
            "supervisorPid": supervisor_pid,
            "maxConcurrentChild": max_concurrent_child,
            "restartCount": 1,
            "stableAfterRestart": stable_ready,
            "oldChildAliveAtTerminal": old_child_alive_at_terminal,
            "terminalPage": terminal_page,
            "terminalEvents": terminal_events,
            "follow": follow,
            "ack": ack,
            "toolEvents": tool_events,
            "restartToolCall": restart_call,
            "restartToolResult": restart_result,
            "recoveryFollow": recovery_follow,
            "recoveryEvents": recovery_events,
            "result": result,
            "messagePage": page,
            "projection": projection,
            "rawRowsBeforeRestart": raw_before_restart,
            "rawRowsAfterRestart": raw_after_restart,
            "rawRowsMatch": raw_rows_match,
            "replayAck": replay_ack,
            "replayResult": replay_result,
            "replayRequestCountBefore": replay_request_count_before,
            "replayRequestCountAfter": replay_request_count_after,
            "rawRowsAfterReplay": raw_after_replay,
            "replayStable": replay_stable,
            "initialTools": sorted(_tool_names(iteration_requests[0])),
            "postSearchTools": sorted(_tool_names(iteration_requests[1])),
            "calledTools": sorted({str(item.get("name")) for item in calls}),
            "concurrentResponse": rejected,
        },
    )

def _disconnect_before_terminal_check(report_dir: Path) -> CheckResult:
    """Abort the client connection, settle the old Input, then reuse its Session."""

    client = _connect_client(ENDPOINT, report_dir / "events-disconnect.jsonl")
    ready_before = _read_ready()
    old_identity = _process_identity(int(ready_before["pid"]))
    session_id = "programmatic:restart-disconnect"
    input_id = "disconnect-restart-input"
    subscription_id = "disconnect-follow"
    barrier = f"restart-disconnect-{uuid.uuid4().hex[:8]}"
    _http_json("PUT", f"{MODEL_URL}/control/barriers/{barrier}")
    _load_scripts(_restart_scripts(999, barrier))
    _admit_follow(client, session_id, subscription_id)
    ack = client.send_programmatic(session_id, input_id, "disconnect before terminal")

    def has_restart_result(page: object) -> bool:
        call = next((item for item in _tool_calls(page) if item.get("name") == "agent_restart"), None)
        result = _tool_result_for_call(page, call) if call is not None else None
        return result is not None and result.get("body", {}).get("outcome") == "success"

    _wait_page_event(client, session_id, subscription_id, has_restart_result)
    _wait_barrier(MODEL_URL, barrier)
    client.close()
    disconnected_at = time.monotonic()
    _http_json("POST", f"{MODEL_URL}/control/barriers/{barrier}/release")

    # The old request must be settled before loading the recovery model script.
    recovery = _connect_client(ENDPOINT, report_dir / "events-recovery.jsonl")
    recovery_subscription = "disconnect-recovery-follow"
    recovery_follow = _admit_follow(recovery, session_id, recovery_subscription)
    old_result = _wait_programmatic_result(recovery, session_id, input_id)
    old_page = recovery.read_messages(session_id)
    if _final_output(old_page, old_result) is None:
        old_page, old_events = _wait_page_event(
            recovery,
            session_id,
            recovery_subscription,
            lambda value: _final_output(value, old_result) is not None,
        )
    else:
        old_events = []
    if old_result.get("status") == "open":
        raise GateFailure(f"断线旧 Input 未结算：{old_result!r}")

    # Only now configure the next provider response; this cannot race the old one.
    _load_scripts([{"mode": "complete", "content": "admission-restored"}])
    recovery_input = "disconnect-recovery-input"
    recovery_follow_2 = _admit_follow(recovery, session_id, "disconnect-recovery-input-follow")
    recovery_ack: dict[str, Any] | None = None
    rejected: list[dict[str, Any]] = []
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = recovery.request_raw(
            "programmatic/message/send",
            {"session_id": session_id, "message_id": recovery_input, "text": "recovery after disconnect"},
        )
        if isinstance(response.get("result"), dict):
            recovery_ack = cast(dict[str, Any], response["result"])
            break
        rejected.append(response)
        time.sleep(0.02)
    if recovery_ack is None:
        raise GateFailure("disconnect 后 admission 未在 3 秒内恢复")
    recovery_result = _wait_programmatic_result(recovery, session_id, recovery_input)
    recovery_page = recovery.read_messages(session_id)
    if _final_output(recovery_page, recovery_result) is None:
        recovery_page, recovery_events = _wait_page_event(
            recovery,
            session_id,
            "disconnect-recovery-input-follow",
            lambda value: _final_output(value, recovery_result) is not None,
        )
    else:
        recovery_events = []
    elapsed = time.monotonic() - disconnected_at
    ready_after = _read_ready()
    new_identity = _process_identity(int(ready_after["pid"]))
    projection = _projection_evidence(session_id, recovery_page)
    recovery_output = _final_output(recovery_page, recovery_result)
    recovery_calls = _tool_calls(recovery_page)
    recovery.close()
    return CheckResult(
        "RESTART-DISCONNECT",
        recovery_result.get("status") == "complete"
        and _output_text(recovery_output) == "admission-restored"
        and ready_after["bootId"] == ready_before["bootId"]
        and new_identity == old_identity
        and elapsed < 3
        and recovery_ack.get("message_id") == recovery_input
        and projection["wireMatchesRaw"]
        and projection["turnProjection"]
        and projection["turnProjection"][-1].get("status") == "complete",
        {
            "sessionId": session_id,
            "restartInputId": input_id,
            "restartAck": ack,
            "oldResult": old_result,
            "oldPage": old_page,
            "oldEvents": old_events,
            "recoveryInputId": recovery_input,
            "recoveryAck": recovery_ack,
            "recoveryResult": recovery_result,
            "recoveryPage": recovery_page,
            "recoveryEvents": recovery_events,
            "recoveryFollow": recovery_follow,
            "recoveryFollowAfterSettle": recovery_follow_2,
            "before": ready_before,
            "after": ready_after,
            "beforeIdentity": old_identity,
            "afterIdentity": new_identity,
            "recoverySeconds": elapsed,
            "rejectedBeforeRecovery": rejected,
            "recoveryToolCalls": recovery_calls,
            "projection": projection,
        },
    )

def _process_metrics(pid: int) -> dict[str, int]:
    status = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            status[key] = int(value.split()[0])
    return {
        "pid": pid,
        "fds": len(list(Path(f"/proc/{pid}/fd").iterdir())),
        "threads": len(list(Path(f"/proc/{pid}/task").iterdir())),
        "vmRssKiB": status["VmRSS"],
        "vmHwmKiB": status["VmHWM"],
    }


def _descendant_pids(pid: int) -> set[int]:
    descendants: set[int] = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        path = Path(f"/proc/{parent}/task/{parent}/children")
        if not path.exists():
            continue
        children = {int(item) for item in path.read_text().split()}
        new = children - descendants
        descendants.update(new)
        pending.extend(new)
    return descendants


def _zombie_descendants(supervisor_pid: int) -> list[int]:
    zombies: list[int] = []
    for pid in _descendant_pids(supervisor_pid):
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            zombies.append(pid)
    return sorted(zombies)


def _peak_memory_deltas(
    supervisor_before: dict[str, int],
    child_before: dict[str, int],
    resource_samples: list[dict[str, dict[str, int]]],
) -> dict[str, int]:
    """计算各进程跨轮次 RSS 与 HWM 最大增量。"""

    return {
        "supervisorRssKiB": max(
            sample["supervisor"]["vmRssKiB"] - supervisor_before["vmRssKiB"]
            for sample in resource_samples
        ),
        "supervisorHwmKiB": max(
            sample["supervisor"]["vmHwmKiB"] - supervisor_before["vmHwmKiB"]
            for sample in resource_samples
        ),
        "childRssKiB": max(
            sample["child"]["vmRssKiB"] - child_before["vmRssKiB"]
            for sample in resource_samples
        ),
        "childHwmKiB": max(
            sample["child"]["vmHwmKiB"] - child_before["vmHwmKiB"]
            for sample in resource_samples
        ),
    }


def _memory_within_limit(deltas: dict[str, int], limit_kib: int = 64 * 1024) -> bool:
    return all(delta <= limit_kib for delta in deltas.values())


def _message_projection_health(session_ids: list[str]) -> dict[str, Any]:
    """Use raw Message plus pure TurnProjection to prove no open work remains."""

    sessions: list[dict[str, Any]] = []
    open_turns: list[dict[str, Any]] = []
    for session_id in sorted(set(session_ids)):
        raw = _raw_messages(session_id)
        projected = TurnProjection().project(raw, "programmatic") if raw else ()
        session = {
            "sessionId": session_id,
            "messageCount": len(raw),
            "throughSeq": raw[-1].seq if raw else -1,
            "turns": [
                {
                    "afterSeq": turn.after_seq,
                    "throughSeq": turn.through_seq,
                    "status": turn.status,
                    "endingMessageId": turn.ending_message_id,
                }
                for turn in projected
            ],
        }
        sessions.append(session)
        open_turns.extend(
            {"sessionId": session_id, **turn}
            for turn in session["turns"]
            if turn["status"] == "open"
        )
    return {"sessions": sessions, "openTurns": open_turns}


def _resource_check(
    supervisor_before: dict[str, int],
    child_before: dict[str, int],
    zombie_samples: list[list[int]],
    resource_samples: list[dict[str, dict[str, int]]],
    session_ids: list[str],
) -> CheckResult:
    time.sleep(0.2)
    supervisor_after = _process_metrics(supervisor_before["pid"])
    child_after = _process_metrics(int(_read_ready()["pid"]))
    supervisor_delta = {
        "fds": supervisor_after["fds"] - supervisor_before["fds"],
        "threads": supervisor_after["threads"] - supervisor_before["threads"],
    }
    child_delta = {
        "fds": child_after["fds"] - child_before["fds"],
        "threads": child_after["threads"] - child_before["threads"],
    }
    memory_deltas = _peak_memory_deltas(
        supervisor_before, child_before, resource_samples
    )
    zombies = _zombie_descendants(supervisor_before["pid"])
    message_health = _message_projection_health(session_ids)
    return CheckResult(
        "RESTART-SOAK-RESOURCES",
        supervisor_delta["fds"] <= 2
        and supervisor_delta["threads"] <= 0
        and child_delta["fds"] <= 4
        and child_delta["threads"] <= 2
        and _memory_within_limit(memory_deltas)
        and not zombies
        and not any(zombie_samples)
        and not message_health["openTurns"],
        {
            "supervisorBefore": supervisor_before,
            "supervisorAfter": supervisor_after,
            "supervisorDelta": supervisor_delta,
            "childBefore": child_before,
            "childAfter": child_after,
            "childDelta": child_delta,
            "resourceSamples": resource_samples,
            "peakMemoryDeltasKiB": memory_deltas,
            "zombies": zombies,
            "zombieSamples": zombie_samples,
            "messageHealth": message_health,
            "thresholds": {
                "supervisorFds": 2,
                "supervisorThreads": 0,
                "childFds": 4,
                "childThreads": 2,
                "supervisorRssKiB": 64 * 1024,
                "supervisorHwmKiB": 64 * 1024,
                "childRssKiB": 64 * 1024,
                "childHwmKiB": 64 * 1024,
            },
        },
    )


def _unsupervised_tool_absence_check(report_dir: Path) -> CheckResult:
    config = Path("/sandbox/restart-config-template.toml").read_text(encoding="utf-8")
    config = config.replace(
        'listen = "/sandbox/akashic.sock"',
        'listen = "/sandbox/unsupervised.sock"',
    ).replace(
        "[channels.chat]\nenabled = true",
        "[channels.chat]\nenabled = false",
    )
    config_path = Path("/sandbox/unsupervised.toml")
    workspace = Path("/sandbox/unsupervised-workspace")
    endpoint = Path("/sandbox/unsupervised.sock")
    config_path.write_text(config, encoding="utf-8")
    _initialize_current_workspace(workspace, Path("/app"))
    source_registry = WORKSPACE / "model-registry.sqlite3"
    target_registry = workspace / "model-registry.sqlite3"
    with (
        sqlite3.connect(f"file:{source_registry}?mode=ro", uri=True) as source,
        sqlite3.connect(target_registry) as target,
    ):
        source.backup(target)
    process = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "gateway",
            "--config",
            str(config_path),
            "--workspace",
            str(workspace),
        ]
    )
    try:
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, report_dir / "events-unsupervised.jsonl")
        before = len(_requests())
        _load_scripts(
            [
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": [
                        {
                            "id": "call_unsupervised_search",
                            "name": "tool_search",
                            "arguments": {"query": "select:agent_restart"},
                        }
                    ],
                },
                {"mode": "complete", "content": "unsupervised-complete"},
            ]
        )
        session_id = "programmatic:restart-unsupervised"
        input_id = "unsupervised-input"
        subscription_id = "unsupervised-follow"
        follow = _admit_follow(client, session_id, subscription_id)
        ack = client.send_programmatic(session_id, input_id, "find restart")
        result = _wait_programmatic_result(client, session_id, input_id)
        page = client.read_messages(session_id)
        if _final_output(page, result) is None:
            page, events = _wait_page_event(
                client,
                session_id,
                subscription_id,
                lambda value: _final_output(value, result) is not None,
            )
        else:
            events = []
        client.close()
        requests = _requests()[before:]
        calls = _tool_calls(page)
        search_call = next((item for item in calls if item.get("name") == "tool_search"), None)
        search_result = _tool_result_for_call(page, search_call) if search_call is not None else None
        search_payload = _tool_result_json(search_result)
        projection = _projection_evidence(session_id, page)
        final_output = _final_output(page, result)
        passed = (
            result.get("status") == "complete"
            and _output_text(final_output) == "unsupervised-complete"
            and all("agent_restart" not in _tool_names(request) for request in requests)
            and all(item.get("name") != "agent_restart" for item in calls)
            and search_result is not None
            and search_result.get("body", {}).get("outcome") == "success"
            and search_payload is not None
            and search_payload.get("selected") == []
            and projection["wireMatchesRaw"]
        )
        return CheckResult(
            "RESTART-UNSUPERVISED",
            passed,
            {
                "sessionId": session_id,
                "inputId": input_id,
                "follow": follow,
                "ack": ack,
                "result": result,
                "messagePage": page,
                "messageEvents": events,
                "requestTools": [sorted(_tool_names(item)) for item in requests],
                "toolCalls": calls,
                "toolSearchResult": search_result,
                "toolSearchPayload": search_payload,
                "projection": projection,
            },
        )
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=15)
        endpoint.unlink(missing_ok=True)


def _inside(iterations: int, report_dir: Path, *, resource_gate: bool) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    _wait_http_ready(f"{MODEL_URL}/readyz", READINESS_DEADLINE_S)
    _configure_model_gate()
    _wait_socket(ENDPOINT, READINESS_DEADLINE_S)
    events_path = report_dir / "events.jsonl"
    client = _connect_client(ENDPOINT, events_path)
    reload_journal = ReloadJournal(WORKSPACE)
    checks: list[CheckResult] = []
    try:
        isolation = {
            "extraPluginDirs": os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS"),
            "pluginCacheExists": Path("/sandbox/home/.akashic-plugin/cache").exists(),
        }
        checks.append(
            CheckResult(
                "RESTART-ISOLATION",
                isolation["extraPluginDirs"] == "/sandbox/restart-plugins"
                and isolation["pluginCacheExists"] is False,
                isolation,
            )
        )
        previous_mcp_identity: dict[str, int] | None = None
        supervisor_baseline: dict[str, int] | None = None
        child_baseline: dict[str, int] | None = None
        zombie_samples: list[list[int]] = []
        resource_samples: list[dict[str, dict[str, int]]] = []
        session_ids: list[str] = []
        for index in range(iterations):
            version = f"v{index + 1}"
            previous_reload = reload_journal.latest(plugin_id="restart_probe")
            _write_mcp_plugin(
                version,
                plugin_root=Path("/sandbox/restart-plugins/restart_probe"),
            )
            mcp_identity = _running_mcp_identity(
                version, previous=previous_mcp_identity
            )
            _wait_reload_complete(
                reload_journal,
                previous_tx_id=(
                    None if previous_reload is None else previous_reload.tx_id
                ),
            )
            if previous_mcp_identity is not None:
                _wait_identity_exit(previous_mcp_identity)
            hot_mcp = _run_mcp_call(client, version, f"{index}-HOT")
            checks.append(hot_mcp)
            session_ids.append(str(cast(dict[str, Any], hot_mcp.evidence)["sessionId"]))

            if supervisor_baseline is None:
                supervisor_pid = int((WORKSPACE / ".supervisor.pid").read_text())
                supervisor_baseline = _process_metrics(supervisor_pid)
                child_baseline = _process_metrics(int(_read_ready()["pid"]))

            client, result = _run_restart_iteration(
                index,
                client,
                report_dir,
            )
            checks.append(result)
            session_ids.append(str(cast(dict[str, Any], result.evidence)["sessionId"]))
            recovered_mcp_identity = _running_mcp_identity(
                version, previous=mcp_identity
            )
            _wait_identity_exit(mcp_identity)
            checks.append(
                CheckResult(
                    f"MCP-{index}-RECOVERED",
                    recovered_mcp_identity != mcp_identity,
                    {
                        "version": version,
                        "oldIdentity": mcp_identity,
                        "oldIdentityAlive": _identity_alive(mcp_identity),
                        "newIdentity": recovered_mcp_identity,
                    },
                )
            )
            recovered_mcp = _run_mcp_call(client, version, f"{index}-AFTER-RESTART")
            checks.append(recovered_mcp)
            session_ids.append(str(cast(dict[str, Any], recovered_mcp.evidence)["sessionId"]))
            previous_mcp_identity = recovered_mcp_identity
            zombie_samples.append(_zombie_descendants(supervisor_baseline["pid"]))

            # 同一 Message Session 追加新 Input，同时证明工具快照不再含 restart。
            before = len(_requests())
            _load_scripts([{"mode": "complete", "content": f"resume-{index}"}])
            restart_session = str(cast(dict[str, Any], result.evidence)["sessionId"])
            resume_id = f"resume-input-{index}"
            resume_subscription = f"resume-follow-{index}"
            resume_follow = _admit_follow(client, restart_session, resume_subscription)
            resume_ack = client.send_programmatic(
                restart_session, resume_id, f"resume {index}"
            )
            resume_result = _wait_programmatic_result(client, restart_session, resume_id)
            resume_page = client.read_messages(restart_session)
            if _final_output(resume_page, resume_result) is None:
                resume_page, resume_events = _wait_page_event(
                    client,
                    restart_session,
                    resume_subscription,
                    lambda value: _final_output(value, resume_result) is not None,
                )
            else:
                resume_events = []
            request = _requests()[before]
            resume_projection = _projection_evidence(restart_session, resume_page)
            checks.append(
                CheckResult(
                    f"RESTART-{index}-RESUME",
                    resume_result.get("status") == "complete"
                    and _output_text(_final_output(resume_page, resume_result))
                    == f"resume-{index}"
                    and not any(item.get("name") == "agent_restart" for item in _tool_calls(resume_page))
                    and resume_projection["wireMatchesRaw"],
                    {
                        "sessionId": restart_session,
                        "inputId": resume_id,
                        "follow": resume_follow,
                        "ack": resume_ack,
                        "result": resume_result,
                        "messagePage": resume_page,
                        "messageEvents": resume_events,
                        "initialTools": sorted(_tool_names(request)),
                        "projection": resume_projection,
                    },
                )
            )
            resource_samples.append(
                {
                    "supervisor": _process_metrics(supervisor_baseline["pid"]),
                    "child": _process_metrics(int(_read_ready()["pid"])),
                }
            )
        disconnect = _disconnect_before_terminal_check(report_dir)
        checks.append(disconnect)
        session_ids.append(str(cast(dict[str, Any], disconnect.evidence)["sessionId"]))
        if resource_gate:
            if supervisor_baseline is None or child_baseline is None:
                raise GateFailure("soak resource baseline 缺失")
            checks.append(
                _resource_check(
                    supervisor_baseline,
                    child_baseline,
                    zombie_samples,
                    resource_samples,
                    session_ids,
                )
            )
    finally:
        client.close()

    report = {
        "gate": "restart",
        "iterations": iterations,
        "status": "passed" if all(check.passed for check in checks) else "failed",
        "checks": [asdict(check) for check in checks],
    }
    _write_json(report_dir / "restart-gate.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


def _inside_unsupervised(report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    check = _unsupervised_tool_absence_check(report_dir)
    _write_json(report_dir / "unsupervised.json", asdict(check))
    print(json.dumps(asdict(check), ensure_ascii=False))
    return 0 if check.passed else 1


def _isolated_config(name: str) -> tuple[Path, Path, Path]:
    source = Path("/sandbox/restart-config-template.toml").read_text(encoding="utf-8")
    endpoint = Path(f"/sandbox/{name}.sock")
    source = source.replace(
        'listen = "/sandbox/akashic.sock"',
        f'listen = "{endpoint}"',
    ).replace(
        "[channels.chat]\nenabled = true",
        "[channels.chat]\nenabled = false",
    )
    config = Path(f"/sandbox/{name}.toml")
    workspace = Path(f"/sandbox/{name}-workspace")
    config.write_text(source, encoding="utf-8")
    _initialize_current_workspace(workspace, Path("/app"))
    return config, workspace, endpoint


def _install_startup_plugin(home: Path, name: str, source: str) -> Path:
    """Install a disposable v3 source plugin for one startup failure scenario."""

    root = home / "plugins"
    plugin = root / name
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.py").write_text(
        "api_version = 3\n" f"name = {name!r}\n" "version = '1.0.0'\n" f"{source}",
        encoding="utf-8",
    )
    (plugin / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {name!r}\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    return root


def _wait_scenario_ready(path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        if payload.get("state") == "ready":
            return cast(dict[str, Any], payload)
        time.sleep(0.02)
    raise GateFailure(f"场景 readiness 超时: {path}")


def _failure_mode_checks(report_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. child 裸 75 没有私有 commit，supervisor 必须以 70 失败且只启动一次。
    naked_config, naked_workspace, _ = _isolated_config("naked75")
    naked_home = Path("/sandbox/naked75-home")
    count_path = Path("/sandbox/naked75-count.txt")
    naked_plugins = _install_startup_plugin(
        naked_home,
        "naked75",
        "import os, pathlib\n"
        f"p=pathlib.Path({str(count_path)!r})\n"
        "p.write_text((p.read_text() if p.exists() else '') + '1\\n')\n"
        "os._exit(75)\n",
    )
    naked = subprocess.run(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(naked_config),
            "--workspace",
            str(naked_workspace),
        ],
        env={
            **os.environ,
            "HOME": str(naked_home),
            "AKASHIC_EXTRA_PLUGIN_DIRS": str(naked_plugins),
        },
        timeout=20,
    )
    naked_starts = count_path.read_text().splitlines() if count_path.exists() else []
    checks.append(
        CheckResult(
            "RESTART-NAKED-75",
            naked.returncode == 70 and len(naked_starts) == 1,
            {"returncode": naked.returncode, "childStarts": len(naked_starts)},
        )
    )

    # 2. stale readiness 不能满足新 boot；故障场景单独使用 15 秒失败门。
    stale_config, stale_workspace, _ = _isolated_config("stale-ready")
    stale_home = Path("/sandbox/stale-ready-home")
    stale_payload = {"bootId": "stale", "pid": 1, "state": "ready"}
    (stale_workspace / ".runtime-ready.json").write_text(json.dumps(stale_payload))
    stale_plugins = _install_startup_plugin(
        stale_home,
        "stale_ready",
        "import time\ntime.sleep(30)\n",
    )
    stale_started = time.monotonic()
    stale = subprocess.run(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(stale_config),
            "--workspace",
            str(stale_workspace),
        ],
        env={
            **os.environ,
            "HOME": str(stale_home),
            "AKASHIC_EXTRA_PLUGIN_DIRS": str(stale_plugins),
            "AKASHIC_READINESS_TIMEOUT_S": "15",
        },
        timeout=25,
    )
    stale_duration = time.monotonic() - stale_started
    stale_after = json.loads(
        (stale_workspace / ".runtime-ready.json").read_text(encoding="utf-8")
    )
    checks.append(
        CheckResult(
            "RESTART-STALE-READY",
            stale.returncode == 70
            and stale_duration >= 14
            and stale_after == stale_payload,
            {
                "returncode": stale.returncode,
                "durationSeconds": stale_duration,
                "readiness": stale_after,
            },
        )
    )

    # 3. supervisor SIGTERM 必须清理 gateway、MCP 及全部既有后代。
    stop_config, stop_workspace, _ = _isolated_config("supervisor-stop")
    stop_home = Path("/sandbox/supervisor-stop-home")
    stop_home.mkdir(exist_ok=True)
    stop_plugins = stop_home / "plugins"
    _write_mcp_plugin(
        "sigterm",
        plugin_root=stop_plugins / "restart_probe",
        workspace=stop_workspace,
    )
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(stop_config),
            "--workspace",
            str(stop_workspace),
        ],
        env={
            **os.environ,
            "HOME": str(stop_home),
            "AKASHIC_EXTRA_PLUGIN_DIRS": str(stop_plugins),
        },
    )
    ready_path = stop_workspace / ".runtime-ready.json"
    ready = _wait_scenario_ready(ready_path)
    child_identity = _process_identity(int(ready["pid"]))
    mcp_identity = _running_mcp_identity("sigterm", workspace=stop_workspace)
    descendant_identities = [
        _process_identity(pid) for pid in _descendant_pids(supervisor.pid)
    ]
    supervisor.send_signal(signal.SIGTERM)
    stop_exit = supervisor.wait(timeout=15)
    _wait_identity_exit(child_identity)
    _wait_identity_exit(mcp_identity)
    live_descendants = [
        identity for identity in descendant_identities if _identity_alive(identity)
    ]
    checks.append(
        CheckResult(
            "RESTART-SUPERVISOR-SIGTERM",
            stop_exit == 0
            and not _identity_alive(child_identity)
            and not _identity_alive(mcp_identity)
            and not live_descendants
            and not ready_path.exists(),
            {
                "returncode": stop_exit,
                "childIdentity": child_identity,
                "childAlive": _identity_alive(child_identity),
                "mcpIdentity": mcp_identity,
                "mcpAlive": _identity_alive(mcp_identity),
                "descendantIdentities": descendant_identities,
                "liveDescendants": live_descendants,
                "readinessExists": ready_path.exists(),
            },
        )
    )

    # 4. Supervisor SIGKILL 后，Guardian 必须通过 lease EOF 清空完整 boot。
    kill_config, kill_workspace, _ = _isolated_config("supervisor-kill")
    kill_home = Path("/sandbox/supervisor-kill-home")
    kill_home.mkdir(exist_ok=True)
    kill_plugins = kill_home / "plugins"
    _write_mcp_plugin(
        "supervisor-kill",
        plugin_root=kill_plugins / "restart_probe",
        workspace=kill_workspace,
    )
    killed_supervisor = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(kill_config),
            "--workspace",
            str(kill_workspace),
        ],
        env={
            **os.environ,
            "HOME": str(kill_home),
            "AKASHIC_EXTRA_PLUGIN_DIRS": str(kill_plugins),
        },
    )
    kill_ready_path = kill_workspace / ".runtime-ready.json"
    _ = _wait_scenario_ready(kill_ready_path)
    kill_descendants = [
        _process_identity(pid) for pid in _descendant_pids(killed_supervisor.pid)
    ]
    os.kill(killed_supervisor.pid, signal.SIGKILL)
    kill_exit = killed_supervisor.wait(timeout=5)
    for identity in kill_descendants:
        _wait_identity_exit(identity)
    kill_live = [identity for identity in kill_descendants if _identity_alive(identity)]
    checks.append(
        CheckResult(
            "RESTART-SUPERVISOR-SIGKILL",
            kill_exit == -signal.SIGKILL
            and not kill_live
            and not kill_ready_path.exists(),
            {
                "returncode": kill_exit,
                "descendantIdentities": kill_descendants,
                "liveDescendants": kill_live,
                "readinessExists": kill_ready_path.exists(),
            },
        )
    )

    # 5. Guardian SIGKILL 后，Supervisor 必须兜底清空 boot 并非零退出。
    guardian_config, guardian_workspace, _ = _isolated_config("guardian-kill")
    guardian_home = Path("/sandbox/guardian-kill-home")
    guardian_home.mkdir(exist_ok=True)
    guardian_plugins = guardian_home / "plugins"
    _write_mcp_plugin(
        "guardian-kill",
        plugin_root=guardian_plugins / "restart_probe",
        workspace=guardian_workspace,
    )
    guardian_supervisor = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(guardian_config),
            "--workspace",
            str(guardian_workspace),
        ],
        env={
            **os.environ,
            "HOME": str(guardian_home),
            "AKASHIC_EXTRA_PLUGIN_DIRS": str(guardian_plugins),
        },
    )
    guardian_ready_path = guardian_workspace / ".runtime-ready.json"
    _ = _wait_scenario_ready(guardian_ready_path)
    guardian_descendants = _descendant_pids(guardian_supervisor.pid)
    guardian_children_path = Path(
        f"/proc/{guardian_supervisor.pid}/task/" f"{guardian_supervisor.pid}/children"
    )
    guardian_children = [int(pid) for pid in guardian_children_path.read_text().split()]
    if len(guardian_children) != 1:
        raise GateFailure(f"Guardian 数量异常: {guardian_children}")
    guardian_identities = [_process_identity(pid) for pid in guardian_descendants]
    os.kill(guardian_children[0], signal.SIGKILL)
    guardian_supervisor_exit = guardian_supervisor.wait(timeout=15)
    for identity in guardian_identities:
        _wait_identity_exit(identity)
    guardian_live = [
        identity for identity in guardian_identities if _identity_alive(identity)
    ]
    checks.append(
        CheckResult(
            "RESTART-GUARDIAN-SIGKILL",
            guardian_supervisor_exit != 0
            and not guardian_live
            and not guardian_ready_path.exists(),
            {
                "returncode": guardian_supervisor_exit,
                "guardianPid": guardian_children[0],
                "descendantIdentities": guardian_identities,
                "liveDescendants": guardian_live,
                "readinessExists": guardian_ready_path.exists(),
            },
        )
    )
    _write_json(
        report_dir / "failure-modes.json",
        {"checks": [asdict(check) for check in checks]},
    )
    return checks


def _inside_failures(report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    checks = _failure_mode_checks(report_dir)
    passed = all(check.passed for check in checks)
    print(
        json.dumps(
            {
                "status": "passed" if passed else "failed",
                "checks": [asdict(check) for check in checks],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


def _configure_restart_gate(sandbox: Path) -> None:
    config = sandbox / "config.toml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("max_iterations = 2", "max_iterations = 5")
    config.write_text(text, encoding="utf-8")
    _write_mcp_plugin(
        "bootstrap",
        plugin_root=sandbox / "restart-plugins/restart_probe",
        workspace=sandbox / "workspace",
        runtime_workspace=WORKSPACE,
        stage_runtime=False,
    )
    # 启动迁移会改写活动配置；隔离场景必须从不可变模板各自迁移。
    (sandbox / "restart-config-template.toml").write_text(text, encoding="utf-8")


def _digest_summary(files: dict[str, str]) -> dict[str, object]:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "fileCount": len(files)}


def _copied_source_digests(
    source: dict[str, str],
    app: Path,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    """按同一 source manifest 比较宿主源码与 sandbox app。"""

    manifest = {
        path: digest
        for path, digest in source.items()
        if not path.startswith("static/")
    }
    app_files = {
        path: hashlib.sha256((app / path).read_bytes()).hexdigest()
        for path in manifest
        if (app / path).is_file() and not (app / path).is_symlink()
    }
    missing = sorted(set(manifest) - set(app_files))
    return _digest_summary(manifest), _digest_summary(app_files), missing


def _host(iterations: int, *, soak: bool) -> int:
    repo = Path(__file__).resolve().parents[2]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    report_dir = repo / "docker/debug/reports/restart" / run_id
    report_dir.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-restart-gate-", dir="/tmp"))
    _prepare_host_sandbox(sandbox, repo)
    _configure_restart_gate(sandbox)
    before = _repository_digest(repo)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    source_digest, app_digest, app_missing = _copied_source_digests(
        before, sandbox / "app"
    )
    env = {
        **os.environ,
        "AKASHIC_CONTROL_SANDBOX": str(sandbox),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    env["AKASHIC_EXTRA_PLUGIN_DIRS"] = "/sandbox/restart-plugins"
    project = f"akashic-restart-{run_id.lower()}"
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(repo / "docker/debug/docker-compose.control-gate.yml"),
    ]
    error = ""
    inside_returncode = -1
    unsupervised_returncode = -1
    failures_returncode = -1
    cleanup_returncode = -1
    residual: dict[str, list[str]] = {
        "containers": [],
        "networks": [],
        "volumes": [],
    }
    image: dict[str, object] = {}
    try:
        build = subprocess.run([*compose, "build", "model-gate"], cwd=repo, env=env)
        if build.returncode != 0:
            raise GateFailure(f"image build failed: {build.returncode}")
        stage_runtime = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "--no-deps",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--entrypoint",
                "python",
                "control-probe",
                "-m",
                "venv",
                "/sandbox/restart-plugins/restart_probe/.venv",
            ],
            cwd=repo,
            env=env,
        )
        if stage_runtime.returncode != 0:
            raise GateFailure(
                f"restart fixture runtime staging failed: {stage_runtime.returncode}"
            )
        image_inspect = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "akashic-agent-control-gate:latest",
                "--format",
                "{{json .}}",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        inspected = json.loads(image_inspect.stdout)
        image = {
            "name": "akashic-agent-control-gate:latest",
            "id": inspected["Id"],
            "repoDigests": inspected.get("RepoDigests", []),
        }
        up = subprocess.run(
            [*compose, "up", "-d", "model-gate", "akashic-control-gate"],
            cwd=repo,
            env=env,
        )
        if up.returncode != 0:
            raise GateFailure(f"compose up failed: {up.returncode}")
        inside_command = [
            *compose,
            "exec",
            "-T",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "akashic-control-gate",
            "python",
            "docker/debug/restart_probe.py",
            "--inside",
            "--iterations",
            str(iterations),
            "--report-dir",
            "/sandbox/reports/restart",
        ]
        if soak:
            inside_command.append("--resource-gate")
        inside = subprocess.run(
            inside_command,
            cwd=repo,
            env=env,
        )
        inside_returncode = inside.returncode
        if inside.returncode != 0:
            raise GateFailure(f"inside gate failed: {inside.returncode}")
        unsupervised = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "--no-deps",
                "--env",
                "AKASHIC_PLUGIN_HOME=/sandbox/unsupervised-plugin-home",
                "control-probe",
                "python",
                "docker/debug/restart_probe.py",
                "--inside-unsupervised",
                "--report-dir",
                "/sandbox/reports/restart",
            ],
            cwd=repo,
            env=env,
        )
        unsupervised_returncode = unsupervised.returncode
        if unsupervised.returncode != 0:
            raise GateFailure(f"unsupervised gate failed: {unsupervised.returncode}")
        failures = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "--no-deps",
                "--env",
                "AKASHIC_PLUGIN_HOME=/sandbox/failure-plugin-home",
                "control-probe",
                "python",
                "docker/debug/restart_probe.py",
                "--inside-failures",
                "--report-dir",
                "/sandbox/reports/restart",
            ],
            cwd=repo,
            env=env,
        )
        failures_returncode = failures.returncode
        if failures.returncode != 0:
            raise GateFailure(f"failure modes failed: {failures.returncode}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        logs = subprocess.run(
            [*compose, "logs", "--no-color"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (report_dir / "compose.log").write_text(logs.stdout, encoding="utf-8")
        if (sandbox / "reports/restart").exists():
            shutil.copytree(
                sandbox / "reports/restart",
                report_dir,
                dirs_exist_ok=True,
            )
        cleanup = subprocess.run(
            [*compose, "down", "--remove-orphans", "--volumes"],
            cwd=repo,
            env=env,
        )
        cleanup_returncode = cleanup.returncode
        for kind, command in {
            "containers": [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
            "networks": [
                "docker",
                "network",
                "ls",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
            "volumes": [
                "docker",
                "volume",
                "ls",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
        }.items():
            result = subprocess.run(
                command, check=True, text=True, stdout=subprocess.PIPE
            )
            residual[kind] = result.stdout.split()

    after = _repository_digest(repo)
    passed = (
        not error
        and inside_returncode == 0
        and unsupervised_returncode == 0
        and failures_returncode == 0
        and cleanup_returncode == 0
        and not any(residual.values())
        and before == after
        and source_digest == app_digest
        and not app_missing
    )
    report = {
        "runId": run_id,
        "gate": "restart",
        "head": head,
        "dirtyStatus": dirty_status,
        "sourceDigest": source_digest,
        "sandboxAppDigest": app_digest,
        "sandboxMissingSourceFiles": app_missing,
        "composeProject": project,
        "image": image,
        "iterations": iterations,
        "status": "passed" if passed else "failed",
        "insideReturncode": inside_returncode,
        "unsupervisedReturncode": unsupervised_returncode,
        "failuresReturncode": failures_returncode,
        "cleanupReturncode": cleanup_returncode,
        "residualResources": residual,
        "repositoriesUnchanged": before == after,
        "error": error,
        "reportDir": str(report_dir),
    }
    _write_json(report_dir / "gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    shutil.rmtree(sandbox)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="restart Docker 真实验收 Gate")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--inside-unsupervised", action="store_true")
    parser.add_argument("--inside-failures", action="store_true")
    parser.add_argument("--resource-gate", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--soak", action="store_true")
    parser.add_argument(
        "--report-dir", type=Path, default=Path("/sandbox/reports/restart")
    )
    args = parser.parse_args()
    iterations = 20 if args.soak else args.iterations
    if iterations < 1:
        raise SystemExit("--iterations 必须大于 0")
    if args.inside:
        return _inside(iterations, args.report_dir, resource_gate=args.resource_gate)
    if args.inside_unsupervised:
        return _inside_unsupervised(args.report_dir)
    if args.inside_failures:
        return _inside_failures(args.report_dir)
    return _host(iterations, soak=args.soak)


if __name__ == "__main__":
    raise SystemExit(main())
