from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore
from proactive_v2.tools import TOOL_SCHEMAS, _get_recent_chat, _recall_memory, _web_fetch

logger = logging.getLogger(__name__)

FORCED_WRITE_PROMPT = (
    "步数即将用尽。你必须现在调用 writefile，\n"
    "将当前 working files 更新到最新状态。\n"
    "下一步将强制结束，请确保进度已完整写入。"
)

FORCED_FINISH_PROMPT = (
    "你正在执行 Drift 任务，步数已用尽。\n"
    "根据上方对话历史，立即调用 finish_drift，填写：\n"
    "- skill_used：你执行的技能名\n"
    "- one_line：一句话总结本次做了什么\n"
    "- next：下次从哪里继续（一句话）\n"
    "- note：全局备注（可选）"
)

DRIFT_SEND_MESSAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "发送一条消息给用户。单次 Drift run 最多只能调用一次。",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要发送的消息内容"},
            },
            "required": ["content"],
        },
    },
}

DRIFT_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "readfile",
            "description": "读取 drift 工作区内的文件。path 相对于 drift/ 目录，只要目标在 drift_dir 内即可。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "writefile",
            "description": "写入 drift 工作区内的文件。path 相对于 drift/ 目录，只要目标在 drift_dir 内即可。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    next(
        schema for schema in TOOL_SCHEMAS
        if schema["function"]["name"] == "recall_memory"
    ),
    next(
        schema for schema in TOOL_SCHEMAS
        if schema["function"]["name"] == "web_fetch"
    ),
    next(
        schema for schema in TOOL_SCHEMAS
        if schema["function"]["name"] == "get_recent_chat"
    ),
    DRIFT_SEND_MESSAGE_SCHEMA,
    {
        "type": "function",
        "function": {
            "name": "finish_drift",
            "description": "【终止工具】结束本次 Drift，保存进度状态。调用后 loop 立即结束。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_used": {"type": "string"},
                    "one_line": {"type": "string"},
                    "next": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["skill_used", "one_line", "next"],
            },
        },
    },
]


@dataclass
class DriftToolDeps:
    drift_dir: Path
    store: DriftStateStore
    memory: Any = None
    recent_chat_fn: Any = None
    web_fetch_tool: Any = None
    max_chars: int = 8_000
    send_message_fn: Any = None


def _json_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _resolve_path(drift_dir: Path, raw_path: str) -> Path | None:
    try:
        raw = str(raw_path or "").strip()
        if not raw:
            return None
        raw_obj = Path(raw)
        if raw_obj.is_absolute():
            target = raw_obj.resolve()
        else:
            parts = raw_obj.parts
            if any(part == ".." for part in parts):
                return None
            if parts and parts[0] != "skills":
                candidate = drift_dir / "skills" / raw_obj
                if candidate.exists() or (drift_dir / "skills" / parts[0]).is_dir():
                    target = candidate.resolve()
                else:
                    target = (drift_dir / raw_obj).resolve()
            else:
                target = (drift_dir / raw_obj).resolve()
        root = drift_dir.resolve()
        if not target.is_relative_to(root):
            return None
        return target
    except Exception:
        return None


def _readfile(args: dict, *, drift_dir: Path) -> str:
    target = _resolve_path(drift_dir, args.get("path", ""))
    if target is None:
        logger.info("[drift_tools] readfile rejected: path outside drift dir raw=%r", args.get("path"))
        return _json_error("path outside drift directory")
    try:
        if not target.exists():
            logger.info("[drift_tools] readfile missing: %s", target)
            return _json_error("file not found")
        if not target.is_file():
            logger.info("[drift_tools] readfile rejected non-file: %s", target)
            return _json_error("path is not a file")
        logger.info("[drift_tools] readfile ok: %s", target)
        return json.dumps({"content": target.read_text(encoding="utf-8")}, ensure_ascii=False)
    except Exception as e:
        logger.warning("[drift_tools] readfile failed: path=%s err=%s", target, e)
        return _json_error(f"readfile failed: {e}")


def _writefile(args: dict, *, drift_dir: Path) -> str:
    target = _resolve_path(drift_dir, args.get("path", ""))
    if target is None:
        logger.info("[drift_tools] writefile rejected: path outside drift dir raw=%r", args.get("path"))
        return _json_error("path outside drift directory")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(args.get("content", "")), encoding="utf-8")
        logger.info("[drift_tools] writefile ok: %s", target)
        return json.dumps({"ok": True}, ensure_ascii=False)
    except Exception as e:
        logger.warning("[drift_tools] writefile failed: path=%s err=%s", target, e)
        return _json_error(f"writefile failed: {e}")


async def _send_message(ctx: AgentTickContext, args: dict, *, send_message_fn) -> str:
    if send_message_fn is None:
        logger.info("[drift_tools] send_message unavailable")
        return _json_error("send_message not configured")
    if getattr(ctx, "drift_message_sent", False):
        logger.info("[drift_tools] send_message rejected: already used")
        return _json_error("send_message already used in this drift run")
    content = str(args.get("content", "") or "")
    if not content.strip():
        logger.info("[drift_tools] send_message rejected: empty content")
        return _json_error("content is required")
    ok = await send_message_fn(content)
    if not ok:
        logger.warning("[drift_tools] send_message failed")
        return _json_error("send_message failed")
    ctx.drift_message_sent = True
    logger.info("[drift_tools] send_message ok")
    return json.dumps({"ok": True}, ensure_ascii=False)


def _finish_drift(ctx: AgentTickContext, args: dict, *, store: DriftStateStore) -> str:
    skill_used = str(args.get("skill_used", "") or "").strip()
    if skill_used not in store.valid_skill_names():
        logger.info("[drift_tools] finish_drift rejected unknown skill=%s", skill_used)
        return _json_error(f"unknown skill: {skill_used}")
    one_line = str(args.get("one_line", "") or "").strip()
    next_action = str(args.get("next", "") or "").strip()
    if not one_line:
        return _json_error("one_line is required")
    if not next_action:
        return _json_error("next is required")
    note_raw = args.get("note")
    note = str(note_raw).strip() if note_raw is not None else None
    store.save_finish(
        skill_used=skill_used,
        one_line=one_line,
        next_action=next_action,
        note=note,
        now_utc=ctx.now_utc,
    )
    ctx.drift_finished = True
    logger.info(
        "[drift_tools] finish_drift ok: skill=%s one_line=%s next=%s",
        skill_used,
        one_line[:120],
        next_action[:100],
    )
    return json.dumps({"ok": True}, ensure_ascii=False)


async def dispatch(tool_name: str, args: dict, ctx: AgentTickContext, deps: DriftToolDeps) -> str:
    if tool_name == "readfile":
        return _readfile(args, drift_dir=deps.drift_dir)
    if tool_name == "writefile":
        return _writefile(args, drift_dir=deps.drift_dir)
    if tool_name == "send_message":
        return await _send_message(ctx, args, send_message_fn=deps.send_message_fn)
    if tool_name == "finish_drift":
        return _finish_drift(ctx, args, store=deps.store)
    if tool_name == "recall_memory":
        return await _recall_memory(ctx, args, memory=deps.memory)
    if tool_name == "web_fetch":
        return await _web_fetch(ctx, args, web_fetch_tool=deps.web_fetch_tool, max_chars=deps.max_chars)
    if tool_name == "get_recent_chat":
        return await _get_recent_chat(ctx, args, recent_chat_fn=deps.recent_chat_fn)
    raise ValueError(f"unknown drift tool: {tool_name!r}")
