#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import struct
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import httpx
from control import endpoint_name

BASE_URL = os.environ.get("COMPUTER_URL", "").rstrip("/")
PROTOCOL = "2025-11-25"
_SCREENSHOT_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_MAX_SCREENSHOT_FILES = 32

TOOLS = (
    {
        "name": "browser_observe",
        "description": (
            "Inspect the persistent Chromium browser without changing page state. "
            "Use snapshot before actions so later calls can target stable element refs. "
            "Screenshots are saved as local files; use read_file with the returned path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "observe": {
                    "type": "string",
                    "enum": [
                        "snapshot",
                        "get_content",
                        "get_url",
                        "get_title",
                        "screenshot",
                        "tab_list",
                    ],
                },
                "target_id": {"type": "string", "maxLength": 128},
            },
            "required": ["observe"],
            "additionalProperties": False,
        },
    },
    {
        "name": "browser_action",
        "description": (
            "Operate the persistent Chromium browser through CDP. Prefer a ref from "
            "browser_observe snapshot over coordinates or guessed selectors."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "navigate",
                        "click",
                        "fill",
                        "type",
                        "press",
                        "scroll",
                        "wait",
                        "go_back",
                        "go_forward",
                        "reload",
                        "tab_new",
                        "tab_select",
                        "tab_close",
                    ],
                },
                "url": {"type": "string", "maxLength": 8192},
                "ref": {"type": "string", "pattern": "^e[1-9][0-9]{0,3}$"},
                "snapshot_id": {"type": "string", "maxLength": 64},
                "text": {"type": "string", "maxLength": 16384},
                "key": {"type": "string", "maxLength": 80},
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                },
                "amount": {"type": "integer", "minimum": 1, "maximum": 5000},
                "timeout": {"type": "integer", "minimum": 0, "maximum": 30000},
                "target_id": {"type": "string", "maxLength": 128},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "computer_observe",
        "description": (
            "Read the current desktop screenshot or Computer activity state. Screenshots "
            "are saved as local files; use read_file with the returned path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "observe": {"type": "string", "enum": ["screenshot", "activity"]}
            },
            "required": ["observe"],
            "additionalProperties": False,
        },
    },
    {
        "name": "computer_action",
        "description": (
            "Control the persistent desktop. Use browser first for web pages; use this "
            "for login screens, native dialogs, or visual-coordinate fallback."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "click",
                        "double_click",
                        "move",
                        "drag",
                        "type",
                        "key",
                        "scroll",
                        "wait",
                    ],
                },
                "mouse_button": {"type": "string", "enum": ["left", "middle", "right"]},
                "x": {"type": "integer", "minimum": 0, "maximum": 1279},
                "y": {"type": "integer", "minimum": 0, "maximum": 799},
                "to_x": {"type": "integer", "minimum": 0, "maximum": 1279},
                "to_y": {"type": "integer", "minimum": 0, "maximum": 799},
                "text": {"type": "string", "maxLength": 16384},
                "key": {"type": "string", "maxLength": 80},
                "amount": {"type": "integer", "minimum": -100, "maximum": 100},
                "ms": {"type": "integer", "minimum": 0, "maximum": 30000},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
)


def request(path: str, payload: dict[str, Any] | None = None) -> tuple[bytes, str]:
    """Call the Workload gateway and return its bounded response."""

    if not BASE_URL:
        raise RuntimeError("COMPUTER_URL is missing")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        BASE_URL + path,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(req, timeout=125) as response:
            body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise RuntimeError("Computer response is too large")
            return body, response.headers.get_content_type()
    except HTTPError as error:
        detail = error.read(16 * 1024).decode("utf-8", "replace")
        raise RuntimeError(f"Computer returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Computer is unavailable: {error.reason}") from error


def save_screenshot(raw: bytes, media_type: str) -> str:
    """Save one screenshot inside this plugin's data root and return its path."""

    extension = _SCREENSHOT_EXTENSIONS.get(media_type)
    if extension is None:
        raise RuntimeError(
            f"Computer returned an unsupported screenshot type: {media_type}"
        )
    if not raw:
        raise RuntimeError("Computer returned an empty screenshot")
    data_dir = os.environ.get("AKA_PLUGIN_DATA_DIR") or os.environ.get(
        "AKASHIC_PLUGIN_DATA_DIR"
    )
    if not data_dir:
        raise RuntimeError("Computer plugin data directory is missing")

    root = Path(data_dir).resolve()
    screenshot_dir = root / "screenshots"
    screenshot_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not screenshot_dir.resolve().is_relative_to(root):
        raise RuntimeError("Computer screenshot directory escaped plugin data")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = screenshot_dir / f"computer-{stamp}-{uuid.uuid4().hex}{extension}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    prune_screenshots(screenshot_dir, keep=path)
    return str(path)


def prune_screenshots(screenshot_dir: Path, *, keep: Path) -> None:
    """Remove abandoned temporary files and keep a bounded screenshot history."""

    for temporary in screenshot_dir.glob("computer-*.tmp"):
        if not temporary.is_symlink() and temporary.is_file():
            temporary.unlink()

    files: list[tuple[int, str, Path]] = []
    extensions = frozenset(_SCREENSHOT_EXTENSIONS.values())
    for path in screenshot_dir.iterdir():
        if (
            path.is_symlink()
            or not path.name.startswith("computer-")
            or path.suffix not in extensions
            or not path.is_file()
        ):
            continue
        files.append((path.stat().st_mtime_ns, path.name, path))
    files.sort(reverse=True)
    older = [entry for entry in files if entry[2] != keep]
    for _mtime, _name, path in older[_MAX_SCREENSHOT_FILES - 1 :]:
        path.unlink()


def screenshot_result(raw: bytes, media_type: str) -> dict[str, object]:
    """Return a file reference that both text and multimodal agents can consume."""

    path = save_screenshot(raw, media_type)
    value = {
        "kind": "screenshot_file",
        "path": path,
        "mime_type": media_type,
        "next": "Call read_file with this path to inspect the screenshot.",
    }
    return {"content": [{"type": "text", "text": json.dumps(value)}]}


def call_tool(name: str, arguments: object) -> dict[str, object]:
    """Execute one known tool and build MCP content blocks."""

    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    if name == "browser_observe":
        raw, _ = request("/browser/observe", arguments)
        value = json.loads(raw)
        if arguments.get("observe") == "screenshot":
            media_type = value.get("mimeType")
            data = value.get("data")
            if not isinstance(media_type, str) or not isinstance(data, str):
                raise RuntimeError("Computer returned an invalid browser screenshot")
            try:
                raw = base64.b64decode(data, validate=True)
            except ValueError as error:
                raise RuntimeError(
                    "Computer returned invalid screenshot data"
                ) from error
            return screenshot_result(raw, media_type)
        return {
            "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]
        }
    if name == "browser_action":
        raw, _ = request("/browser/action", arguments)
        return {"content": [{"type": "text", "text": raw.decode("utf-8")}]}
    if name == "computer_observe":
        observe = arguments.get("observe")
        if observe == "activity":
            raw, _ = request("/activity")
            return {"content": [{"type": "text", "text": raw.decode("utf-8")}]}
        if observe == "screenshot":
            raw, media_type = request("/screenshot")
            return screenshot_result(raw, media_type)
        raise ValueError("observe must be screenshot or activity")
    if name == "computer_action":
        raw, _ = request("/input", arguments)
        return {"content": [{"type": "text", "text": raw.decode("utf-8")}]}
    raise ValueError(f"unknown tool: {name}")


def reply(message: dict[str, Any]) -> dict[str, object] | None:
    """Handle one MCP JSON-RPC message."""

    request_id = message.get("id")
    method = message.get("method")
    if method == "notifications/initialized":
        return None
    if "id" not in message:
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "akashic-computer", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": list(TOOLS)}}
    if method == "tools/call":
        params = message.get("params")
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            raise ValueError("tools/call params are invalid")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": call_tool(params["name"], params.get("arguments", {})),
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def driver_content(value: dict[str, Any]) -> dict[str, object]:
    """驱动图片继续使用原截图 owner 保存，避免绕开保留与路径合同。"""
    content = []
    for item in value["content"]:
        if item.get("type") == "image":
            content.extend(
                screenshot_result(
                    base64.b64decode(item["data"], validate=True), item["mimeType"]
                )["content"]
            )
        else:
            content.append(item)
    return {"content": content, "call_id": value["call_id"]}


async def control_connection(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """控制连接断开或取消时，等待容器 drain 后才给调用者回执。"""
    task = None
    cancellation = None
    context = None
    try:
        peer = writer.get_extra_info("socket")
        _, uid, _ = struct.unpack(
            "3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if uid != os.getuid():
            raise PermissionError("Computer control peer has a different uid")
        raw = await reader.readline()
        payload = json.loads(raw)
        context = payload["context"]
        async with httpx.AsyncClient(
            base_url=BASE_URL, timeout=330 if payload["op"] == "end_turn" else 170
        ) as client:

            async def send(path, body):
                response = await client.post(path, json=body)
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Computer returned {response.status_code}: {response.text[:16000]}"
                    )
                if len(response.content) > 8 * 1024 * 1024:
                    raise RuntimeError("Computer response is too large")
                return response.json()

            if payload["op"] not in {"run", "end_turn"}:
                raise ValueError("Unknown Computer control operation")
            if payload.get("reset") is True:
                await send("/driver/reset", {"session_id": context["session_id"]})
            task = asyncio.create_task(
                send(
                    "/driver/run",
                    {
                        "context": context,
                        "code": payload.get("code", ""),
                        "endTurn": payload["op"] == "end_turn",
                        "timeoutMs": payload.get("timeoutMs", 60000),
                    },
                )
            )
            cancellation = asyncio.create_task(reader.readline())
            try:
                done, _ = await asyncio.wait(
                    {task, cancellation}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                await send("/driver/cancel", {"call_id": context["call_id"]})
                await asyncio.gather(task, return_exceptions=True)
                raise
            if cancellation in done:
                await send("/driver/cancel", {"call_id": context["call_id"]})
                # 原请求仍须读完，确认它不再执行；副作用不当作已经回滚。
                await asyncio.gather(task, return_exceptions=True)
                result = {"cancelled": True, "released": True, "effects": "may_remain"}
            else:
                result = driver_content(await task)
        writer.write(json.dumps(result, ensure_ascii=False).encode() + b"\n")
        await writer.drain()
    except (
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        OSError,
        httpx.HTTPError,
    ) as error:
        if not writer.is_closing():
            writer.write(json.dumps({"error": str(error)}).encode() + b"\n")
            try:
                await writer.drain()
            except (ConnectionError, BrokenPipeError):
                pass
    finally:
        if cancellation is not None:
            cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        writer.close()
        await writer.wait_closed()


async def serve() -> None:
    """MCP 和本代控制连接共用一个事件循环，旧工具保持原协议。"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=15) as client:
        status = await client.get("/driver/status")
        status.raise_for_status()
        info = status.json()
        if (
            info.get("version") != 2
            or info.get("source") is not True
            or info.get("ready") is not True
        ):
            raise RuntimeError(
                "Computer requires the v2 source driver image; publish it and update the plugin image digest"
            )
    data_root = os.environ.get("AKA_PLUGIN_DATA_DIR") or os.environ.get(
        "AKASHIC_PLUGIN_DATA_DIR"
    )
    control_name = endpoint_name(Path(data_root)) if data_root else None
    active_connections = set()

    async def connection(reader, writer):
        task = asyncio.current_task()
        active_connections.add(task)
        try:
            await control_connection(reader, writer)
        finally:
            active_connections.remove(task)

    control = None
    if control_name:
        control = await asyncio.start_unix_server(
            connection, path="\0" + control_name, limit=256 * 1024
        )
    reader = asyncio.StreamReader(limit=256 * 1024)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    try:
        while line := await reader.readline():
            request_id: object = None
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise TypeError("message must be an object")
                request_id = message.get("id")
                response = await asyncio.to_thread(reply, message)
            except Exception as error:  # noqa: BLE001 - JSON-RPC 边界将工具失败显式返回。
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(error)},
                }
            if response is not None:
                print(
                    json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                    flush=True,
                )
    finally:
        transport.close()
        if control is not None:
            control.close()
            await control.wait_closed()
        # 每条连接有自己的 HTTP deadline；停止接收后先收完回执再退出进程。
        if active_connections:
            pending = list(active_connections)
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*pending, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    raise result


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
