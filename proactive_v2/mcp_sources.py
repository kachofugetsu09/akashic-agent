"""
proactive/mcp_sources.py — 从 MCP server 拉取 ProactiveEvent 的通用客户端。

读取 ~/.akashic/workspace/proactive_sources.json 中的配置，
动态调用各 MCP server 的 get_tool / ack_tool。

使用项目自带的 agent.mcp.client.McpClient，无需额外依赖。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from proactive_v2.event import AlertEvent

logger = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = Path.home() / ".akashic" / "workspace"


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


def _get_server_cfg(server_name: str, workspace: Path) -> dict | None:
    path = workspace / "mcp_servers.json"
    try:
        data = json.loads(path.read_text())
        return data.get("servers", {}).get(server_name)
    except Exception as e:
        logger.warning("[mcp_sources] mcp_servers.json 读取失败: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def poll_content_feeds() -> None:
    raise RuntimeError("mcp_sources.sync API 已移除，请使用 poll_content_feeds_async + McpClientPool")


def fetch_alert_events() -> list[dict]:
    raise RuntimeError("mcp_sources.sync API 已移除，请使用 fetch_alert_events_async + McpClientPool")


def fetch_content_events() -> list[dict]:
    raise RuntimeError("mcp_sources.sync API 已移除，请使用 fetch_content_events_async + McpClientPool")


def fetch_context_data() -> list[dict]:
    raise RuntimeError("mcp_sources.sync API 已移除，请使用 fetch_context_data_async + McpClientPool")


def acknowledge_events(events: list[AlertEvent]) -> None:
    _ = events
    raise RuntimeError("mcp_sources.sync API 已移除，请使用 acknowledge_events_async + McpClientPool")


def acknowledge_content_entries(entries: list[tuple[str, str]], ttl_hours: int | None = None) -> None:
    _ = (entries, ttl_hours)
    raise RuntimeError(
        "mcp_sources.sync API 已移除，请使用 acknowledge_content_entries_async + McpClientPool"
    )


# ── Persistent connection pool ────────────────────────────────────────────────


class McpClientPool:
    """每个 MCP server 保持一个常驻连接，避免每次调用重启子进程。

    用法:
        pool = McpClientPool()
        await pool.connect_all()      # agent 启动时
        await pool.call(server, tool, args)
        await pool.disconnect_all()   # agent 关闭时（finally 块）
    """

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = workspace or _DEFAULT_WORKSPACE
        self._clients: dict[str, Any] = {}               # server -> McpClient
        self._configs: dict[str, tuple[list, dict]] = {}  # server -> (command, env)
        self._locks: dict[str, asyncio.Lock] = {}         # server -> per-server lock（MCP stdio 不支持并发调用）

    async def connect_all(self) -> None:
        """按当前配置连接所有 server，连接失败的 server 跳过。"""
        seen: set[str] = set()
        for src in _load_sources(self._workspace):
            server = src.get("server", "")
            if not server or server in seen:
                continue
            seen.add(server)
            cfg = _get_server_cfg(server, self._workspace)
            if not cfg:
                continue
            command = cfg.get("command", [])
            env = cfg.get("env") or {}
            if not command:
                continue
            self._configs[server] = (command, env)
            await self._connect(server)

    async def _connect(self, server: str) -> bool:
        from agent.mcp.client import McpClient

        command, env = self._configs.get(server, ([], {}))
        if not command:
            return False
        try:
            client = McpClient(name=server, command=command, env=env)
            await client.connect()
            self._clients[server] = client
            logger.info("[mcp_pool] connected: %s", server)
            return True
        except Exception as e:
            logger.warning("[mcp_pool] connect failed %s: %s", server, e, exc_info=True)
            return False

    async def call(self, server: str, tool_name: str, args: dict) -> Any:
        """调用 tool，连接断开时自动重连一次。

        MCP stdio 传输不支持并发调用，per-server lock 保证串行。
        """
        if server not in self._locks:
            self._locks[server] = asyncio.Lock()
        async with self._locks[server]:
            if server not in self._clients:
                if server not in self._configs:
                    raise RuntimeError(f"[mcp_pool] unknown server: {server}")
                if not await self._connect(server):
                    raise RuntimeError(f"[mcp_pool] could not connect: {server}")
            client = self._clients[server]
            try:
                raw = await client.call(tool_name, args)
                return json.loads(raw) if raw and raw.strip().startswith(("[", "{")) else raw
            except Exception as e:
                logger.warning(
                    "[mcp_pool] call failed %s.%s, reconnecting: %s", server, tool_name, e
                )
                self._clients.pop(server, None)
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if await self._connect(server):
                    raw = await self._clients[server].call(tool_name, args)
                    return json.loads(raw) if raw and raw.strip().startswith(("[", "{")) else raw
                raise

    async def disconnect_all(self) -> None:
        """断开所有连接。agent 关闭时在 finally 块调用。"""
        for server, client in list(self._clients.items()):
            try:
                await client.disconnect()
                logger.info("[mcp_pool] disconnected: %s", server)
            except Exception as e:
                logger.warning("[mcp_pool] disconnect error %s: %s", server, e)
        self._clients.clear()


# ── Async pool-based variants ─────────────────────────────────────────────────


async def fetch_alert_events_async(pool: McpClientPool) -> list[dict]:
    return await _fetch_by_channel_async(pool, channel="alert")


async def fetch_content_events_async(pool: McpClientPool) -> list[dict]:
    return await _fetch_by_channel_async(pool, channel="content")


async def fetch_context_data_async(pool: McpClientPool) -> list[dict]:
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
    # context 源兼容两种返回形态：
    # 1. 单个 dict：包装成长度为 1 的列表
    # 2. list[dict]：逐条补 _source 后原样返回
    if isinstance(data, dict):
        item = dict(data)
        item.setdefault("_source", server)
        return [item]
    if isinstance(data, list):
        result: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            enriched = dict(item)
            enriched.setdefault("_source", server)
            result.append(enriched)
        return result
    return []


async def _fetch_by_channel_async(pool: McpClientPool, *, channel: str) -> list[dict]:
    result: list[dict] = []
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
            # 3. 通过常驻 McpClientPool 调远端 MCP 工具。
            #    pool.call() 内部会负责串行、断线重连、JSON 反序列化。
            data = await pool.call(server, get_tool, {})
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
            # 5. 单个源失败只记日志，不阻断其他源。
            logger.warning(
                "[mcp_sources] fetch_%s %s.%s failed: %s",
                channel,
                server,
                get_tool,
                e,
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


async def poll_content_feeds_async(pool: McpClientPool) -> None:
    failed_servers: list[str] = []
    for src in _iter_sources_by_channel("content", pool._workspace):
        poll_tool = src.get("poll_tool")
        if not poll_tool:
            continue
        server = src.get("server", "")
        try:
            result = await pool.call(server, poll_tool, {})
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


async def acknowledge_events_async(pool: McpClientPool, events: list) -> None:
    ack_map = _build_ack_map(_load_sources(pool._workspace))
    for e in events:
        ack_server: str = getattr(e, "_ack_server", None) or ""
        if not ack_server:
            ack_server = getattr(e, "source_name", "") or ""
        ack_id: str | None = getattr(e, "ack_id", None)
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


async def acknowledge_content_entries_async(
    pool: McpClientPool,
    entries: list[tuple[str, str]],
    ttl_hours: int | None = None,
) -> None:
    if not entries:
        return
    ack_map = _build_ack_map(_load_sources(pool._workspace))
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
        if ttl_hours is not None and ttl_hours > 0:
            args["ttl_hours"] = ttl_hours
        try:
            await pool.call(server, ack_tool, args)
        except Exception as e:
            logger.warning("[mcp_sources] content ack failed %s.%s: %s", server, ack_tool, e)
