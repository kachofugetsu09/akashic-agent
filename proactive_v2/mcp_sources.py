"""
proactive/mcp_sources.py — 从 MCP server 拉取主动链路数据的通用客户端。

读取 ~/.akashic/workspace/proactive_sources.json 中的配置，
动态调用各 MCP server 的 get_tool / ack_tool。

通过共享 ToolRegistry 调用已连接的 MCP 工具。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = Path.home() / ".akashic" / "workspace"
_POLL_TOOL_TIMEOUT = 180.0


class McpGateway(Protocol):
    _workspace: Path

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
        registered_name = (
            tool_name
            if tool_name in names
            else f"mcp_{server}__{tool_name}"
        )
        if registered_name not in names:
            raise RuntimeError(f"MCP tool 不可用: {server}.{tool_name}")
        execution = self._tools.execute(registered_name, args, raise_errors=True)
        result = (
            await asyncio.wait_for(execution, timeout=timeout)
            if timeout is not None
            else await execution
        )
        text = result.text if isinstance(result, ToolResult) else str(result)
        if text.strip().startswith(("[", "{")):
            return json.loads(text)
        return text


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_sources(workspace: Path) -> list[dict]:
    path = workspace / "proactive_sources.json"
    try:
        data = json.loads(path.read_text())
        return [s for s in data.get("sources", []) if s.get("enabled", True)]
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("[mcp_sources] proactive_sources.json 读取失败: %s", e)
        return []


async def fetch_alert_events_async(pool: McpGateway) -> list[dict]:
    return await _fetch_by_channel_async(pool, channel="alert")


async def fetch_content_events_async(pool: McpGateway) -> list[dict]:
    return await _fetch_by_channel_async(pool, channel="content")


async def fetch_context_data_async(pool: McpGateway) -> list[dict]:
    return await _fetch_by_channel_async(pool, channel="context")


def _extract_proactive_events(data: Any, *, server: str, kind: str) -> list[dict]:
    # 1. proactive 事件源约定返回 list[dict]。
    # 2. 这里只保留 kind 匹配当前 channel 的事件，并补上 ack_server。
    if not isinstance(data, list):
        return []
    result: list[dict] = []
    for event in data:
        if not isinstance(event, dict) or event.get("kind") != kind:
            continue
        enriched = dict(event)
        enriched.setdefault("ack_server", server)
        result.append(enriched)
    return result


def _extract_context_items(data: Any, *, server: str) -> list[dict]:
    if not isinstance(data, list):
        return []
    result: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        enriched.setdefault("_source", server)
        result.append(enriched)
    return result


async def _fetch_by_channel_async(pool: McpGateway, *, channel: str) -> list[dict]:
    result: list[dict] = []
    failed_servers: list[str] = []
    succeeded_count = 0
    # 1. 先按 channel 从 proactive_sources.json 中挑出本轮该访问的源。
    for src in _iter_sources_by_channel(channel, pool._workspace):
        server = src.get("server", "")
        # 2. 每个源默认调用：
        #    - context 走 get_context
        #    - alert/content 走 get_proactive_events
        #    也允许在配置里用 get_tool 覆盖。
        get_tool = src.get(
            "get_tool",
            "get_context" if channel == "context" else "get_proactive_events",
        )
        try:
            # 3. 通过共享 MCP Gateway 调远端工具。
            data = await pool.call(server, get_tool, {})
            succeeded_count += 1
            if channel == "context":
                # 4a. context 通道不看 kind，直接把返回值规范成 list[dict]。
                items = _extract_context_items(data, server=server)
                result.extend(items)
                logger.debug("[mcp_sources] context 源 %s 返回 %d 条", server, len(items))
            else:
                # 4b. alert/content 通道要求远端返回 proactive event 列表，
                #     再按 kind 过滤出当前通道的事件。
                events = _extract_proactive_events(data, server=server, kind=channel)
                result.extend(events)
                logger.debug("[mcp_sources] %s 返回 %d 条 %s 事件", server, len(events), channel)
        except Exception as e:
            logger.warning(
                "[mcp_sources] fetch_%s %s.%s failed: %s",
                channel,
                server,
                get_tool,
                e,
            )
            failed_servers.append(server)
    if failed_servers and succeeded_count == 0:
        raise RuntimeError(f"fetch_{channel} 以下源失败: {failed_servers}")
    if failed_servers:
        logger.warning(
            "[mcp_sources] fetch_%s 部分源失败，保留其他源结果: %s",
            channel,
            failed_servers,
        )
    return result


def _iter_sources_by_channel(channel: str, workspace: Path = _DEFAULT_WORKSPACE) -> list[dict]:
    sources = _load_sources(workspace)
    result: list[dict] = []
    # 根据 channel 做一层静态路由：
    # - context 只取 channel=context 的源
    # - alert 排除纯 content 源
    # - content 排除纯 alert 源
    for src in sources:
        src_channel = str(src.get("channel", "")).strip().lower()
        if channel == "context":
            if src_channel == "context":
                result.append(src)
            continue
        if src_channel in ("context",):
            continue
        if channel == "alert" and src_channel in ("content",):
            continue
        if channel == "content" and src_channel in ("alert",):
            continue
        result.append(src)
    return result


def _build_ack_map(sources: list[dict]) -> dict[str, tuple[str, list[str]]]:
    ack_map: dict[str, tuple[str, list[str]]] = {}
    for src in sources:
        ack_tool = src.get("ack_tool")
        if ack_tool:
            ack_map[src["server"]] = (ack_tool, [])
    return ack_map


async def poll_content_feeds_async(pool: McpGateway) -> None:
    failed_servers: list[str] = []
    for src in _iter_sources_by_channel("content", pool._workspace):
        poll_tool = src.get("poll_tool")
        if not poll_tool:
            continue
        server = src.get("server", "")
        try:
            result = await pool.call(server, poll_tool, {}, timeout=_POLL_TOOL_TIMEOUT)
            if isinstance(result, str) and result.startswith("error:"):
                raise RuntimeError(f"poll_feeds 系统级失败: {result}")
            logger.info("[mcp_sources] poll_content_feeds: %s.%s 完成", server, poll_tool)
        except Exception as e:
            logger.warning(
                "[mcp_sources] poll_content_feeds: %s.%s 失败: %s",
                server, poll_tool, e, exc_info=True,
            )
            failed_servers.append(server)
    if failed_servers:
        raise RuntimeError(f"poll_content_feeds 以下源失败: {failed_servers}")


async def acknowledge_events_async(
    pool: McpGateway,
    events: list[tuple[str, str]],
) -> None:
    ack_map = _build_ack_map(_load_sources(pool._workspace))
    failed_servers: list[str] = []
    for ack_server, ack_id in events:
        if ack_server in ack_map and ack_id:
            ack_map[ack_server][1].append(ack_id)
    for server, (ack_tool, ids) in ack_map.items():
        if not ids:
            continue
        try:
            await pool.call(server, ack_tool, {"event_ids": ids})
            logger.info("[mcp_sources] acked %d 事件 via %s.%s ids=%s", len(ids), server, ack_tool, ids)
        except Exception as e:
            logger.warning("[mcp_sources] ack failed %s.%s: %s", server, ack_tool, e)
            failed_servers.append(server)
    if failed_servers:
        raise RuntimeError(f"ack 以下源失败: {failed_servers}")


async def acknowledge_content_entries_async(
    pool: McpGateway,
    entries: list[tuple[str, str]],
    *,
    feedback: str,
) -> None:
    if not entries:
        return
    if feedback not in {"interesting", "not_interesting"}:
        raise ValueError(f"invalid feedback: {feedback}")
    ack_map = _build_ack_map(_load_sources(pool._workspace))
    failed_servers: list[str] = []
    for source_key, item_id in entries:
        if not source_key.startswith("mcp:"):
            continue
        parts = source_key.split(":", 2)
        server = parts[1] if len(parts) >= 2 else ""
        ack_id = parts[2] if len(parts) >= 3 else item_id
        if server in ack_map and ack_id:
            ack_map[server][1].append(ack_id)
    for server, (ack_tool, ids) in ack_map.items():
        if not ids:
            continue
        args: dict = {"event_ids": ids}
        args["feedback"] = feedback
        try:
            await pool.call(server, ack_tool, args)
        except Exception as e:
            logger.warning("[mcp_sources] content ack failed %s.%s: %s", server, ack_tool, e)
            failed_servers.append(server)
    if failed_servers:
        raise RuntimeError(f"content ack 以下源失败: {failed_servers}")
