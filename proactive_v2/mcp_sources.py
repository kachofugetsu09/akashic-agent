from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from agent.plugins.specs import RegisteredProactiveSource, proactive_source_key
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
MAX_QUARANTINE_ITEMS_PER_SOURCE = 256


@dataclass(frozen=True, slots=True)
class QuarantinedItem:
    """保留单条 MCP 输入的可查询诊断，而不阻断同批合法记录。"""

    source_id: str
    item_id: str
    reason: str
    payload: object


class SourceChannels(dict[str, list[dict[str, Any]]]):
    """三类 source 结果及同批被隔离的单条输入。"""

    def __init__(self) -> None:
        super().__init__({"alert": [], "content": [], "context": []})
        self.quarantined: list[QuarantinedItem] = []
        self.quarantine_overflow: dict[str, int] = {}
        self.quarantine_overflow_count = 0


class McpGateway(Protocol):
    async def call(
        self,
        server: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any: ...


class SharedMcpGateway:
    def __init__(self, workspace: Path, tools: ToolRegistry | None) -> None:
        self._workspace = workspace
        self._tools = tools

    async def call(
        self,
        server: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._tools is None:
            raise RuntimeError("共享 ToolRegistry 不可用")
        names = self._tools.get_tool_names_by_source("mcp", server)
        registered_name = tool_name if tool_name in names else f"mcp_{server}__{tool_name}"
        if registered_name not in names:
            raise RuntimeError(f"MCP tool 不可用: {server}.{tool_name}")
        result = await self._tools.execute(
            registered_name,
            args,
            raise_errors=True,
            execution_timeout=timeout,
        )
        text = result.text if isinstance(result, ToolResult) else str(result)
        if text.strip().startswith(("[", "{")):
            return json.loads(text)
        return text


def source_key(source: RegisteredProactiveSource) -> str:
    return proactive_source_key(source)


async def fetch_sources_async(
    pool: McpGateway,
    sources: list[RegisteredProactiveSource],
) -> dict[str, list[dict[str, Any]]]:
    results = await asyncio.gather(
        *(
            fetch_source_strict_async(
                pool, source, quarantine_invalid=True
            )
            for source in sources
        ),
        return_exceptions=True,
    )
    channels = SourceChannels()
    succeeded = 0
    failures: list[str] = []
    for source, result in zip(sources, results):
        key = source_key(source)
        if isinstance(result, BaseException):
            failures.append(key)
            logger.warning("[proactive.source] fetch 失败 %s: %s", key, result)
            continue
        succeeded += 1
        for channel, items in result.items():
            channels[channel].extend(items)
        channels.quarantined.extend(
            getattr(result, "quarantined", [])
        )
        overflow_count = int(getattr(result, "quarantine_overflow_count", 0))
        if overflow_count:
            channels.quarantine_overflow[key] = (
                channels.quarantine_overflow.get(key, 0) + overflow_count
            )
    if failures and succeeded == 0:
        raise RuntimeError(f"所有 proactive sources 拉取失败: {failures}")
    return channels


async def fetch_source_strict_async(
    pool: McpGateway,
    source: RegisteredProactiveSource,
    *,
    quarantine_invalid: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """拉取并严格校验单个 source，保留原始失败。"""

    spec = source.spec
    key = source_key(source)
    result = SourceChannels()
    if spec.fetch_page_size > 0:
        data = await _fetch_pages(pool, source)
    else:
        data = await pool.call(spec.server, spec.fetch_tool, {})
    if "context" in spec.channels and isinstance(data, dict):
        item = dict(data)
        item.setdefault("_source", key)
        result["context"].append(item)
        return result
    if not isinstance(data, list):
        raise RuntimeError(f"source 返回值必须是 list 或 context dict: {key}")
    for index, raw in enumerate(data):
        item_id = _item_identity(raw, index)
        try:
            item = _validate_item(raw, spec.channels, key)
        except ValueError as exc:
            if not quarantine_invalid:
                raise RuntimeError(str(exc)) from exc
            if len(result.quarantined) < MAX_QUARANTINE_ITEMS_PER_SOURCE:
                result.quarantined.append(
                    QuarantinedItem(key, item_id, str(exc), raw)
                )
            else:
                result.quarantine_overflow_count += 1
            logger.warning(
                "[proactive.source] item quarantined source=%s item=%s reason=%s",
                key,
                item_id,
                exc,
            )
            continue
        kind = str(item.get("kind") or "")
        if kind not in spec.channels:
            continue
        if kind == "context":
            item.setdefault("_source", key)
        else:
            item.setdefault("ack_server", key)
        result[kind].append(item)
    return result


def _item_identity(raw: object, index: int) -> str:
    if isinstance(raw, dict):
        value = raw.get("event_id") or raw.get("id")
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"index:{index}"


def _validate_item(
    raw: object,
    declared_channels: tuple[str, ...],
    source_id: str,
) -> dict[str, Any]:
    """在 MCP 信任边界验证一条记录，失败只返回该条 quarantine。"""

    if not isinstance(raw, dict):
        raise ValueError(f"source item 必须是 object ({type(raw).__name__})")
    item = dict(raw)
    kind = str(item.get("kind") or "").strip()
    if not kind and len(declared_channels) == 1:
        kind = declared_channels[0]
    if kind not in declared_channels:
        raise ValueError(f"kind 未声明或为空: {source_id}")
    item["kind"] = kind
    if kind in {"alert", "content"}:
        if not str(item.get("event_id") or item.get("id") or "").strip():
            raise ValueError(f"source item 缺少 event_id/id: {source_id}")
        score_value = item.get("preprocess_score")
        if score_value is None:
            score_value = item.get("rank_score", 0.0)
        try:
            score = float(score_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score 非数字: {source_id}") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"score 超出 [0,1] 或非 finite: {source_id}")
        item["preprocess_score"] = score
    for field in ("published_at", "triggered_at", "first_seen_at"):
        if field not in item or item[field] in (None, ""):
            continue
        try:
            parsed = item[field]
            if isinstance(parsed, datetime):
                value = parsed
            else:
                value = datetime.fromisoformat(str(parsed))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} 不是 ISO timestamp: {source_id}") from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field} 必须带 timezone: {source_id}")
    return item


async def _fetch_pages(
    pool: McpGateway,
    source: RegisteredProactiveSource,
) -> list[Any]:
    page_size = source.spec.fetch_page_size
    result: list[Any] = []
    offset = 0
    for _ in range(256):
        page = await pool.call(
            source.spec.server,
            source.spec.fetch_tool,
            {"offset": offset, "limit": page_size},
        )
        if not isinstance(page, list):
            raise RuntimeError(
                f"分页 source 返回值必须是 list: {source_key(source)}"
            )
        result.extend(page)
        if len(page) < page_size:
            return result
        offset += len(page)
    raise RuntimeError(f"分页 source 超过 256 页: {source_key(source)}")


async def acknowledge_async(
    pool: McpGateway,
    sources: list[RegisteredProactiveSource],
    source_id: str,
    event_ids: list[str],
    *,
    feedback: str | None = None,
) -> None:
    source = next((item for item in sources if source_key(item) == source_id), None)
    if not event_ids:
        return
    if source is None:
        raise RuntimeError(f"MCP ack source 不存在: {source_id}")
    if not source.spec.ack_tool:
        raise RuntimeError(f"MCP source 未声明 ack tool: {source_id}")
    args: dict[str, Any] = {"event_ids": event_ids}
    if feedback is not None:
        args["feedback"] = feedback
    await pool.call(source.spec.server, source.spec.ack_tool, args)
