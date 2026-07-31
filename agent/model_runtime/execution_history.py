from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

_TOOL_EXECUTION_PREFIX = "<tool_execution "
_SUCCESS_TOOL_EXECUTION_MARKER = '<tool_execution transport_status="success" />'
_TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "timed_out", "stopped", "unknown"}
)


def active_shell_execution_origins(
    messages: Sequence[dict[str, Any]],
) -> dict[int, str]:
    """返回仍在运行的 execution 及其原始 shell tool_call_id。"""

    # 1. 按 tool_call_id 解码内部工具结果 envelope。
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        call_id = message.get("tool_call_id")
        if message.get("role") != "tool" or not isinstance(call_id, str):
            continue
        result = _tool_result_object(message.get("content"))
        if result is not None:
            results[call_id] = result

    # 2. 按调用顺序建立 execution，并在 terminal 调用时释放。
    active: dict[int, str] = {}
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            continue
        for raw_call in calls:
            if not isinstance(raw_call, dict):
                continue
            call = cast(dict[str, Any], raw_call)
            call_id = str(call.get("id") or "")
            function = call.get("function")
            if not call_id or not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = _json_object(function.get("arguments"))
            result = results.get(call_id)
            _update_active(
                active,
                call_id=call_id,
                name=name,
                arguments=arguments,
                result=result,
            )
    return active


def _update_active(
    active: dict[int, str],
    *,
    call_id: str,
    name: object,
    arguments: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> None:
    if result is None:
        return
    status = result.get("process_status")
    if name == "shell" and status == "running":
        execution_id = _execution_id(result.get("execution_id"))
        if execution_id is not None:
            active[execution_id] = call_id
        return
    if name not in {"write_stdin", "task_stop"} or arguments is None:
        return
    execution_id = _execution_id(arguments.get("execution_id"))
    if execution_id is None:
        return
    if status == "running":
        active.setdefault(execution_id, call_id)
    elif status in _TERMINAL_STATUSES:
        active.pop(execution_id, None)


def _tool_result_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.startswith(_TOOL_EXECUTION_PREFIX):
        return None
    marker, separator, value = value.partition("\n")
    if not separator or marker != _SUCCESS_TOOL_EXECUTION_MARKER:
        return None
    return _json_object(value)


def _json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, Any], decoded) if isinstance(decoded, dict) else None


def _execution_id(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value
