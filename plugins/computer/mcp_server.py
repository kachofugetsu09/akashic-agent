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
        "name": "browser",
        "description": (
            "Run an OpenCLI browser or site command in the persistent Chromium profile. "
            "Pass arguments exactly as they follow the opencli executable."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 64,
                }
            },
            "required": ["args"],
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
                    "enum": ["click", "double_click", "move", "type", "key", "scroll", "wait"],
                },
                "x": {"type": "integer", "minimum": 0, "maximum": 1279},
                "y": {"type": "integer", "minimum": 0, "maximum": 799},
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
    if name == "browser":
        raw, _ = request("/opencli", {"args": arguments.get("args")})
        value = json.loads(raw)
        return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
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
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            response = reply(message)
        except Exception as error:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(error)},
            }
        if response is not None:
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
