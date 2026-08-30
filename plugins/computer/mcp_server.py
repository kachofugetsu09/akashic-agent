#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = os.environ.get("COMPUTER_URL", "").rstrip("/")
PROTOCOL = "2025-11-25"

TOOLS = (
    {
        "name": "browser_observe",
        "description": (
            "Inspect the persistent Chromium browser without changing page state. "
            "Use snapshot before actions so later calls can target stable element refs."
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
        "description": "Read the current desktop screenshot or Computer activity state.",
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
                    "enum": ["click", "double_click", "move", "drag", "type", "key", "scroll", "wait"],
                },
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


def call_tool(name: str, arguments: object) -> dict[str, object]:
    """Execute one known tool and build MCP content blocks."""

    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    if name == "browser_observe":
        raw, _ = request("/browser/observe", arguments)
        value = json.loads(raw)
        if arguments.get("observe") == "screenshot":
            media_type = value.get("mimeType")
            data = value.get("data")
            if not isinstance(media_type, str) or not isinstance(data, str):
                raise RuntimeError("Computer returned an invalid browser screenshot")
            return {"content": [{"type": "image", "mimeType": media_type, "data": data}]}
        return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
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
            return {
                "content": [
                    {"type": "image", "mimeType": media_type, "data": base64.b64encode(raw).decode("ascii")}
                ]
            }
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


def main() -> None:
    """Serve newline-delimited MCP messages over stdio."""

    for line in sys.stdin:
        request_id: object = None
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            request_id = message.get("id")
            response = reply(message)
        except Exception as error:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(error)},
            }
        if response is not None:
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
