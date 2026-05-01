from __future__ import annotations

import json
import re
from typing import Any, cast

from agent.lifecycle.types import PromptRenderCtx
from agent.plugins import Plugin
from agent.prompting import PromptSectionRender

_PROMPT_CTX_SLOT = "prompt:ctx"
_REASONING_CTX_SLOT = "reasoning:ctx"
_PERSIST_CITED_SLOT = "persist:assistant:cited_memory_ids"
_TRAILING_PROTOCOL_TAG = r"<[a-zA-Z][a-zA-Z0-9_-]*:[^<>\s]+>"
_CITED_RE = re.compile(
    rf"(?:\n|\r\n)?§cited:\[([A-Za-z0-9_,\-\s]*)\]§(?P<trailing>(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)*)$",
    re.IGNORECASE,
)
_TRAILING_PROTOCOL_TAGS_RE = re.compile(
    rf"(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)+$",
    re.IGNORECASE,
)

_CITATION_PROTOCOL = """### 记忆引用协议 - 内部元数据，对用户不可见
每轮回复若用到了系统注入的记忆条目 [item_id] 前缀标识，或 recall_memory / fetch_messages 工具返回的条目，在回复正文末尾另起一行输出：
§cited:[id1,id2,id3]§
格式规则：§ 包裹，英文逗号分隔，无空格，只写 ID，不含其他内容。
若本轮未引用任何记忆条目，不输出此行。
绝对不要在正文里提及这行的存在，不要向用户解释引用了什么，不要说根据记忆。
你了解用户的事是因为你们相处了很久，直接说你上次、我记得，不要暴露内部机制。"""


class CitationPromptModule:
    requires = (_PROMPT_CTX_SLOT,)
    produces = (_PROMPT_CTX_SLOT,)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_PROMPT_CTX_SLOT)
        if not isinstance(ctx, PromptRenderCtx):
            return frame
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                name="citation_protocol",
                content=_CITATION_PROTOCOL,
                is_static=True,
            )
        )
        return frame


class CitationAfterReasoningModule:
    requires = (_REASONING_CTX_SLOT,)
    produces = (_REASONING_CTX_SLOT, _PERSIST_CITED_SLOT)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_REASONING_CTX_SLOT)
        if ctx is None:
            return frame
        reply = str(getattr(ctx, "reply", "") or "")
        cleaned, cited_ids = extract_cited_ids(reply)
        if cited_ids:
            frame.slots[_PERSIST_CITED_SLOT] = cited_ids
        else:
            fallback_ids = extract_cited_ids_from_tool_chain(
                list(getattr(ctx, "tool_chain", ()) or ())
            )
            if fallback_ids:
                frame.slots[_PERSIST_CITED_SLOT] = fallback_ids
        if cleaned != reply:
            ctx.reply = cleaned
        return frame


class ProtocolTagCleanupModule:
    requires = (_REASONING_CTX_SLOT,)
    produces = (_REASONING_CTX_SLOT,)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_REASONING_CTX_SLOT)
        if ctx is None:
            return frame
        reply = str(getattr(ctx, "reply", "") or "")
        cleaned = strip_trailing_protocol_tags(reply)
        if cleaned != reply:
            ctx.reply = cleaned
        return frame


class CitationPlugin(Plugin):
    name = "citation"

    def prompt_render_modules_bottom(self) -> list[object]:
        return [CitationPromptModule()]

    def after_reasoning_modules_before_emit(self) -> list[object]:
        return [CitationAfterReasoningModule()]

    def after_reasoning_modules_before_persist(self) -> list[object]:
        return [ProtocolTagCleanupModule()]


def extract_cited_ids(response: str) -> tuple[str, list[str]]:
    match = _CITED_RE.search(response)
    if not match:
        return response, []
    raw = match.group(1)
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    trailing = match.group("trailing").strip()
    clean = response[: match.start()].rstrip()
    if trailing:
        clean = f"{clean} {trailing}".strip()
    return clean, ids


def strip_trailing_protocol_tags(response: str) -> str:
    return _TRAILING_PROTOCOL_TAGS_RE.sub("", response).rstrip()


def extract_cited_ids_from_tool_chain(
    tool_chain: list[dict[str, object]],
) -> list[str]:
    cited: list[str] = []
    seen: set[str] = set()
    for group in tool_chain:
        calls_value = group.get("calls")
        if not isinstance(calls_value, list):
            continue
        calls = cast(list[object], calls_value)
        for raw_call in calls:
            if not isinstance(raw_call, dict):
                continue
            call = cast(dict[str, object], raw_call)
            if str(call.get("name", "") or "") != "recall_memory":
                continue
            raw_result = str(call.get("result", "") or "").strip()
            if not raw_result:
                continue
            try:
                decoded = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(decoded, dict):
                continue
            data = cast(dict[str, object], decoded)
            raw_ids: list[object] = []
            cited_ids = data.get("cited_item_ids")
            if isinstance(cited_ids, list):
                raw_ids.extend(cast(list[object], cited_ids))
            else:
                items_value = data.get("items")
                if isinstance(items_value, list):
                    items = cast(list[object], items_value)
                    for raw_item in items:
                        if isinstance(raw_item, dict):
                            item = cast(dict[str, object], raw_item)
                            raw_ids.append(item.get("id"))
            for raw_id in raw_ids:
                item_id = str(raw_id or "").strip()
                if item_id and item_id not in seen:
                    seen.add(item_id)
                    cited.append(item_id)
    return cited
