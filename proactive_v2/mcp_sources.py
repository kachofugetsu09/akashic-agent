from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from agent.plugins.specs import RegisteredProactiveSource, proactive_source_key
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

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
        *(fetch_source_strict_async(pool, source) for source in sources),
        return_exceptions=True,
    )
    channels: dict[str, list[dict[str, Any]]] = {
        "alert": [],
        "content": [],
        "context": [],
    }
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
    if failures and succeeded == 0:
        raise RuntimeError(f"所有 proactive sources 拉取失败: {failures}")
    return channels


async def fetch_source_strict_async(
    pool: McpGateway,
    source: RegisteredProactiveSource,
) -> dict[str, list[dict[str, Any]]]:
    """拉取并严格校验单个 source，保留原始失败。"""

    spec = source.spec
    key = source_key(source)
    result: dict[str, list[dict[str, Any]]] = {
        "alert": [],
        "content": [],
        "context": [],
    }
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
    for raw in data:
        if not isinstance(raw, dict):
            raise RuntimeError(
                f"source item 必须是 object: {key} ({type(raw).__name__})"
            )
        kind = str(raw.get("kind") or "").strip()
        if not kind and len(spec.channels) == 1:
            kind = spec.channels[0]
        if kind not in spec.channels:
            continue
        if kind in {"alert", "content"} and not str(
            raw.get("event_id") or raw.get("id") or ""
        ).strip():
            raise RuntimeError(f"source item 缺少 event_id/id: {key}")
        item = dict(raw)
        if kind == "context":
            item.setdefault("_source", key)
        else:
            item.setdefault("ack_server", key)
        result[kind].append(item)
    return result


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
    if source is None or not source.spec.ack_tool or not event_ids:
        return
    args: dict[str, Any] = {"event_ids": event_ids}
    if feedback is not None:
        args["feedback"] = feedback
    await pool.call(source.spec.server, source.spec.ack_tool, args)
