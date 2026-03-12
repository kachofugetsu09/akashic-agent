from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from core.memory.port import MemoryPort


def extract_action_tokens(tool_name: str, tool_arguments: dict) -> list[str]:
    """Extract tokens from a tool call for procedure keyword matching."""
    tokens = [tool_name]
    if tool_name == "shell":
        cmd = str(tool_arguments.get("command", "") or "").strip()
        if cmd:
            cleaned = re.sub(r"^(sudo|env)\s+", "", cmd)
            parts = cleaned.split()
            first = parts[0] if parts else ""
            if first in ("bash", "sh"):
                rest = [p.strip("\"'") for p in parts[1:] if not p.startswith("-")]
                first = rest[0] if rest else ""
            if first and first not in ("sudo", "env", "bash", "sh"):
                tokens.append(first)
    elif tool_name == "web_fetch":
        url = str(tool_arguments.get("url", "") or "").strip()
        if url:
            parsed = urlparse(url)
            host = (parsed.hostname or "").strip().lower()
            if host:
                tokens.append(host)
                if host.startswith("www.") and len(host) > 4:
                    tokens.append(host[4:])
            path_parts = [
                part.strip().lower() for part in parsed.path.split("/") if part.strip()
            ]
            tokens.extend(path_parts[:3])
    elif tool_name in ("read_file", "write_file", "edit_file", "list_dir"):
        path = str(next(iter(tool_arguments.values()), "") or "")
        match = re.search(r"/skills/([^/]+)", path)
        if match:
            tokens.append(match.group(1))
    return tokens


def build_procedure_hint(
    items: list[dict],
    injected_ids: set[str],
) -> tuple[str, list[str]]:
    """Format newly matched procedure items into a hint block."""
    new_items = [i for i in items if str(i.get("id", "")) not in injected_ids]
    if not new_items:
        return "", []

    lines: list[str] = []
    new_ids: list[str] = []
    for item in new_items:
        summary = (item.get("summary") or "").strip()
        if summary:
            lines.append(f"- {summary}")
            new_ids.append(str(item.get("id", "")))

    if not lines:
        return "", []

    hint = "⚠️ 【操作规范提醒】以下规范适用于当前操作，必须遵守：\n" + "\n".join(lines)
    return hint, new_ids


def prepend_procedure_hint(
    *,
    memory: "MemoryPort | None",
    tool_name: str,
    tool_arguments: dict,
    result: str,
    injected_ids: set[str],
    logger: logging.Logger | None = None,
) -> tuple[str, list[str]]:
    """Prepend a procedure hint to a tool result when keyword matching hits."""
    if memory is None:
        return result, []

    try:
        action_tokens = extract_action_tokens(tool_name, tool_arguments)
        proc_items = memory.keyword_match_procedures(action_tokens)
        if not proc_items:
            return result, []
        hint, new_ids = build_procedure_hint(proc_items, injected_ids)
        if not hint:
            return result, []
        if logger is not None:
            logger.info("  [proc_hint] 注入 %d 条规范: %s", len(new_ids), new_ids)
        return hint + "\n\n---\n\n" + result, new_ids
    except Exception as exc:
        if logger is not None:
            logger.debug("  [proc_hint] keyword match 失败: %s", exc)
        return result, []
