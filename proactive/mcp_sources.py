"""
proactive/mcp_sources.py — 从 MCP server 拉取 ProactiveEvent 的通用客户端。

读取 ~/.akasic/workspace/proactive_sources.json 中的配置，
动态调用各 MCP server 的 get_tool / ack_tool。

使用项目自带的 agent.mcp.client.McpClient，无需额外依赖。
同步入口通过 ThreadPoolExecutor 在子线程里跑 asyncio，
避免与外层 event loop 冲突。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from proactive.event import AlertEvent

logger = logging.getLogger(__name__)

_SOURCES_CONFIG = Path.home() / ".akasic/workspace/proactive_sources.json"
_MCP_SERVERS_CONFIG = Path.home() / ".akasic/workspace/mcp_servers.json"


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_sources() -> list[dict]:
    try:
        data = json.loads(_SOURCES_CONFIG.read_text())
        return [s for s in data.get("sources", []) if s.get("enabled", True)]
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("[mcp_sources] proactive_sources.json 读取失败: %s", e)
        return []


def _get_server_cfg(server_name: str) -> dict | None:
    try:
        data = json.loads(_MCP_SERVERS_CONFIG.read_text())
        return data.get("servers", {}).get(server_name)
    except Exception as e:
        logger.warning("[mcp_sources] mcp_servers.json 读取失败: %s", e)
        return None


# ---------------------------------------------------------------------------
# Async caller (uses project McpClient)
# ---------------------------------------------------------------------------

async def _call_tool_async(
    server_name: str,
    command: list[str],
    env: dict,
    tool_name: str,
    args: dict,
) -> Any:
    from agent.mcp.client import McpClient

    client = McpClient(name=server_name, command=command, env=env)
    try:
        await client.connect()
        raw = await client.call(tool_name, args)
        return json.loads(raw) if raw and raw.strip().startswith(("[", "{")) else raw
    finally:
        await client.disconnect()


def _run_in_thread(coro, *, timeout: float = 30.0) -> Any:
    """在子线程中运行 coroutine，避免与外层 event loop 冲突。"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def poll_content_feeds() -> None:
    """调用所有配置了 poll_tool 的 content 源，触发 feed 侧轮询更新 DB。
    单源失败不影响其他源继续 poll；所有源尝试完毕后，若有失败则统一抛聚合异常。
    由 proactive loop 按固定 1800s 周期调用。
    """
    sources = _load_sources()
    failed_servers: list[str] = []
    for src in sources:
        if str(src.get("channel", "")).strip().lower() != "content":
            continue
        poll_tool = src.get("poll_tool")
        if not poll_tool:
            continue
        server = src.get("server", "")
        cfg = _get_server_cfg(server)
        if not cfg:
            logger.warning("[mcp_sources] poll_content_feeds: 找不到 server 配置: %s", server)
            continue
        command = cfg.get("command", [])
        env = cfg.get("env") or {}
        if not command:
            continue
        try:
            result = _run_in_thread(
                _call_tool_async(server, command, env, poll_tool, {})
            )
            if isinstance(result, str) and result.startswith("error:"):
                raise RuntimeError(f"poll_feeds 系统级失败: {result}")
            logger.info("[mcp_sources] poll_content_feeds: %s.%s 完成", server, poll_tool)
        except Exception as e:
            logger.warning(
                "[mcp_sources] poll_content_feeds: %s.%s 失败: %s",
                server, poll_tool, e,
                exc_info=True,
            )
            failed_servers.append(server)
    if failed_servers:
        raise RuntimeError(f"poll_content_feeds 以下源失败: {failed_servers}")


def fetch_alert_events() -> list[dict]:
    """从所有已配置 MCP 源拉取 alert 类 proactive events。

    返回标准 ProactiveEvent schema 的 dict 列表（kind=="alert"）。
    调用失败的源跳过，不影响其他源。
    """
    sources = _load_sources()
    result: list[dict] = []

    for src in sources:
        if str(src.get("channel", "")).strip().lower() in ("content", "context"):
            continue
        server = src.get("server", "")
        get_tool = src.get("get_tool", "get_proactive_events")
        cfg = _get_server_cfg(server)
        if not cfg:
            logger.warning("[mcp_sources] 找不到 server 配置: %s", server)
            continue
        command = cfg.get("command", [])
        env = cfg.get("env") or {}
        if not command:
            continue
        try:
            events = _run_in_thread(
                _call_tool_async(server, command, env, get_tool, {})
            )
            if isinstance(events, list):
                alerts = []
                for event in events:
                    if not isinstance(event, dict) or event.get("kind") != "alert":
                        continue
                    enriched = dict(event)
                    enriched.setdefault("ack_server", server)
                    alerts.append(enriched)
                result.extend(alerts)
                logger.debug("[mcp_sources] %s 返回 %d 条 alert 事件", server, len(alerts))
        except Exception as e:
            logger.warning(
                "[mcp_sources] 调用 %s.%s 失败: %s (%r)",
                server,
                get_tool,
                e,
                e,
            )

    return result


def fetch_content_events() -> list[dict]:
    """从所有已配置 MCP 源拉取 content 类 proactive events。"""
    sources = _load_sources()
    result: list[dict] = []

    for src in sources:
        if str(src.get("channel", "")).strip().lower() in ("alert", "context"):
            continue
        server = src.get("server", "")
        get_tool = src.get("get_tool", "get_proactive_events")
        cfg = _get_server_cfg(server)
        if not cfg:
            continue
        command = cfg.get("command", [])
        env = cfg.get("env") or {}
        if not command:
            continue
        try:
            events = _run_in_thread(
                _call_tool_async(server, command, env, get_tool, {})
            )
            if isinstance(events, list):
                contents = []
                for event in events:
                    if not isinstance(event, dict) or event.get("kind") != "content":
                        continue
                    enriched = dict(event)
                    enriched.setdefault("ack_server", server)
                    contents.append(enriched)
                result.extend(contents)
        except Exception as e:
            logger.warning(
                "[mcp_sources] 调用 %s.%s 失败: %s (%r)",
                server,
                get_tool,
                e,
                e,
            )
    return result


def fetch_context_data() -> list[dict]:
    """从 channel=context 的源拉取持久背景上下文。不涉及 ack，每次返回最新状态。"""
    sources = _load_sources()
    result: list[dict] = []

    for src in sources:
        if str(src.get("channel", "")).strip().lower() != "context":
            continue
        server = src.get("server", "")
        get_tool = src.get("get_tool", "get_context")
        cfg = _get_server_cfg(server)
        if not cfg:
            logger.warning("[mcp_sources] 找不到 context server 配置: %s", server)
            continue
        command = cfg.get("command", [])
        env = cfg.get("env") or {}
        if not command:
            continue
        try:
            data = _run_in_thread(
                _call_tool_async(server, command, env, get_tool, {})
            )
            if isinstance(data, dict):
                data = dict(data)
                data.setdefault("_source", server)
                result.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item = dict(item)
                        item.setdefault("_source", server)
                        result.append(item)
            logger.debug("[mcp_sources] context 源 %s 返回数据 available=%s", server, (data or {}).get("available"))
        except Exception as e:
            logger.warning("[mcp_sources] context 源 %s 拉取失败: %s", server, e)

    return result


def acknowledge_events(events: list[AlertEvent]) -> None:
    """按显式 ack_server 分组，调用对应 MCP server 的 ack_tool。"""
    sources = _load_sources()
    ack_map: dict[str, tuple[str, list[str]]] = {}

    for src in sources:
        ack_tool = src.get("ack_tool")
        if ack_tool:
            ack_map[src["server"]] = (ack_tool, [])

    for e in events:
        # 1. 优先走事件携带的 ack_server，避免把 source_name 和 server 名硬绑定。
        ack_server: str = getattr(e, "_ack_server", None) or ""
        # 2. 兼容旧事件：若没有显式 ack_server，再退回 source_name。
        if not ack_server:
            ack_server = getattr(e, "source_name", "") or ""
        ack_id: str | None = getattr(e, "ack_id", None)
        if ack_server in ack_map and ack_id:
            ack_map[ack_server][1].append(ack_id)

    for server, (ack_tool, ids) in ack_map.items():
        if not ids:
            continue
        cfg = _get_server_cfg(server)
        if not cfg:
            continue
        command = cfg.get("command", [])
        env = cfg.get("env") or {}
        try:
            _run_in_thread(
                _call_tool_async(server, command, env, ack_tool, {"event_ids": ids})
            )
            logger.info("[mcp_sources] acked %d 事件 via %s.%s ids=%s", len(ids), server, ack_tool, ids)
        except Exception as e:
            logger.warning("[mcp_sources] ack 失败 %s.%s: %s", server, ack_tool, e)


def acknowledge_content_entries(entries: list[tuple[str, str]], ttl_hours: int | None = None) -> None:
    """按 mcp:<server> source_key 回调内容事件 ack。"""
    if not entries:
        return
    sources = _load_sources()
    ack_map: dict[str, tuple[str, list[str]]] = {}
    for src in sources:
        ack_tool = src.get("ack_tool")
        if ack_tool:
            ack_map[src["server"]] = (ack_tool, [])
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
        cfg = _get_server_cfg(server)
        if not cfg:
            continue
        command = cfg.get("command", [])
        env = cfg.get("env") or {}
        try:
            args: dict = {"event_ids": ids}
            if ttl_hours is not None and ttl_hours > 0:
                args["ttl_hours"] = ttl_hours
            _run_in_thread(
                _call_tool_async(server, command, env, ack_tool, args)
            )
        except Exception as e:
            logger.warning("[mcp_sources] content ack 失败 %s.%s: %s", server, ack_tool, e)


# ── Persistent connection pool ────────────────────────────────────────────────


class McpClientPool:
    """每个 MCP server 保持一个常驻连接，避免每次调用重启子进程。

    用法:
        pool = McpClientPool()
        await pool.connect_all()      # agent 启动时
        await pool.call(server, tool, args)
        await pool.disconnect_all()   # agent 关闭时（finally 块）
    """

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}               # server -> McpClient
        self._configs: dict[str, tuple[list, dict]] = {}  # server -> (command, env)

    async def connect_all(self) -> None:
        """按当前配置连接所有 server，连接失败的 server 跳过。"""
        seen: set[str] = set()
        for src in _load_sources():
            server = src.get("server", "")
            if not server or server in seen:
                continue
            seen.add(server)
            cfg = _get_server_cfg(server)
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
        """调用 tool，连接断开时自动重连一次。"""
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
    for src in _iter_sources_by_channel(channel):
        server = src.get("server", "")
        get_tool = src.get(
            "get_tool",
            "get_context" if channel == "context" else "get_proactive_events",
        )
        try:
            data = await pool.call(server, get_tool, {})
            if channel == "context":
                items = _extract_context_items(data, server=server)
                result.extend(items)
                logger.debug("[mcp_sources] context 源 %s 返回 %d 条", server, len(items))
            else:
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
    return result


def _iter_sources_by_channel(channel: str) -> list[dict]:
    sources = _load_sources()
    result: list[dict] = []
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
    for src in _iter_sources_by_channel("content"):
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
    ack_map = _build_ack_map(_load_sources())
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
    ack_map = _build_ack_map(_load_sources())
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
