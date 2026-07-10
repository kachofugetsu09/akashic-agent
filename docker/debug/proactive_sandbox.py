#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.mcp.registry import McpServerRegistry
from agent.config_models import Config
from agent.plugins.manager import PluginManager
from agent.provider import LLMResponse, ToolCall
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bootstrap.proactive import _build_proactive_provider
from bootstrap.providers import build_providers
from proactive_v2.config import ProactiveConfig
from proactive_v2.loop import ProactiveLoop
from proactive_v2.state import ProactiveStateStore
from session.manager import SessionManager

EVENT_ID = "sandbox_content_001"
SKILL_NAME = "sandbox_probe"


class SandboxProvider:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[str] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        names = {
            str(schema.get("function", {}).get("name") or "")
            for schema in kwargs.get("tools", [])
        }
        name, arguments = self._next_call(names)
        self.calls.append(name)
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"sandbox_call_{len(self.calls)}",
                    name=name,
                    arguments=arguments,
                )
            ],
        )

    def _next_call(self, names: set[str]) -> tuple[str, dict[str, Any]]:
        if self.mode == "content":
            item_id = f"feed:{EVENT_ID}"
            if "mark_interesting" in names and "mark_interesting" not in self.calls:
                return "mark_interesting", {"item_ids": [item_id], "reason": "沙盒验证"}
            if "message_push" in names and "message_push" not in self.calls:
                return "message_push", {
                    "message": "沙盒主动推送：已读取 Feed MCP content。",
                    "evidence": [item_id],
                }
            return "finish_turn", {"decision": "reply"}

        if "select_skill" in names and "select_skill" not in self.calls:
            return "select_skill", {"skill_name": SKILL_NAME}
        if "message_push" in names and "message_push" not in self.calls:
            return "message_push", {"message": "沙盒 Drift：空闲链路已执行。"}
        return "finish_drift", {
            "skill_used": SKILL_NAME,
            "status": "completed",
            "briefing": "完成沙盒 Drift 验证",
        }


def _workspace_from_env() -> Path:
    root = Path(os.environ.get("AKASHIC_DEBUG_WORKSPACE", "/sandbox/workspace"))
    return root / "proactive-sandbox"


def _installed_plugin_root(
    cache_root: Path,
    marketplace: str,
    plugin_name: str,
) -> Path:
    plugin_root = cache_root / marketplace / plugin_name
    versions = sorted(path for path in plugin_root.iterdir() if path.is_dir())
    if not versions:
        raise RuntimeError(f"外置插件未安装: {plugin_name}@{marketplace}")
    return versions[-1]


def _assert_sandbox_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if Path("/sandbox") not in resolved.parents:
        raise RuntimeError(f"沙盒路径必须位于 /sandbox: {resolved}")


def reset(workspace: Path) -> None:
    _assert_sandbox_path(workspace)
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    _write_sources(workspace)
    skill_dir = workspace / "drift" / "skills" / SKILL_NAME
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {SKILL_NAME}\n"
        "description: 验证主动链路能够进入并完成 Drift\n"
        "---\n\n"
        "选择本 skill 后发送一条沙盒消息并正常收尾。\n",
        encoding="utf-8",
    )
    (workspace / "PROACTIVE_CONTEXT.md").write_text(
        "# Proactive Context\n\n"
        "- 当前是主动链路沙盒验证。遇到 Sandbox Feed content 时标记为 interesting，发送简短测试消息并正常结束。\n"
        "- 没有 content 时选择 sandbox_probe 完成一次 Drift，并发送简短测试消息。\n",
        encoding="utf-8",
    )


def inject_content(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    _write_sources(workspace)
    db_path = workspace / "feed-data" / "feed_mcp.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                event_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                source_name TEXT NOT NULL, source_type TEXT NOT NULL,
                title TEXT, content TEXT NOT NULL, url TEXT, author TEXT,
                published_at TEXT, first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL, emitted_at TEXT,
                content_hash TEXT NOT NULL, interest_ok INTEGER,
                interest_scored_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS acked_items (
                event_id TEXT PRIMARY KEY, acked_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM acked_items WHERE event_id = ?", (EVENT_ID,))
        conn.execute("DELETE FROM items WHERE event_id = ?", (EVENT_ID,))
        conn.execute(
            """
            INSERT INTO items (
                event_id, source_id, source_name, source_type, title, content,
                url, author, published_at, first_seen_at, last_seen_at,
                emitted_at, content_hash, interest_ok, interest_scored_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
            """,
            (
                EVENT_ID,
                "sandbox-source",
                "Sandbox Feed",
                "manual",
                "主动链路沙盒内容",
                "这条 content 由操作者手动注入，用于验证 MCP 拉取和主动发送。",
                "https://example.com/akashic-proactive-sandbox",
                "sandbox",
                now,
                now,
                now,
                "sandbox-content-hash",
            ),
        )


def clear_content(workspace: Path) -> None:
    db_path = workspace / "feed-data" / "feed_mcp.sqlite3"
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM acked_items")
        conn.execute("DELETE FROM items")


def _write_sources(workspace: Path) -> None:
    (workspace / "proactive_sources.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "server": "feed",
                        "channel": "content",
                        "get_tool": "get_proactive_events",
                        "ack_tool": "acknowledge_events",
                        "enabled": True,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _config(mode: str) -> ProactiveConfig:
    return ProactiveConfig(
        enabled=True,
        lifecycle="default",
        default_channel="sandbox",
        default_chat_id="operator",
        model="sandbox-model",
        feed_poller_interval_seconds=3600,
        anyaction_daily_max_actions=999,
        anyaction_min_interval_seconds=0,
        anyaction_probability_min=1.0,
        anyaction_probability_max=1.0,
        agent_tick_max_steps=8,
        agent_tick_content_limit=5,
        agent_tick_context_prob=0.0,
        agent_tick_delivery_cooldown_hours=0,
        message_dedupe_enabled=False,
        delivery_dedupe_hours=0,
        drift_enabled=mode == "drift",
        drift_max_steps=5,
        drift_min_interval_hours=0,
    )


async def tick(
    workspace: Path,
    mode: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    event_bus = EventBus()
    tools = ToolRegistry()
    installed_cache = Path.home() / ".akashic-plugin" / "cache"
    feed_root = _installed_plugin_root(installed_cache, "lab", "feed")
    sessions = SessionManager(workspace)
    plugins = PluginManager(
        plugin_dirs=[Path("/app/plugins")],
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
        session_manager=sessions,
        plugin_configs={"daynight_gate": {"enabled": False}},
        installed_cache_root=installed_cache,
    )
    mcp = McpServerRegistry(workspace / "mcp_servers.json", tools)
    state = ProactiveStateStore(workspace / "proactive.db")
    provider: Any = SandboxProvider(mode)
    proactive_config = _config(mode)
    model = "sandbox-model"
    max_tokens = 1024
    if config_path is not None:
        app_config = Config.load(config_path)
        main_provider, _, _ = build_providers(app_config)
        provider = _build_proactive_provider(app_config, main_provider)
        proactive_config = app_config.proactive
        proactive_config.lifecycle = "default"
        proactive_config.default_channel = "sandbox"
        proactive_config.default_chat_id = "operator"
        proactive_config.anyaction_daily_max_actions = 999
        proactive_config.anyaction_min_interval_seconds = 0
        proactive_config.anyaction_probability_min = 1.0
        proactive_config.anyaction_probability_max = 1.0
        proactive_config.agent_tick_max_steps = 12
        proactive_config.agent_tick_content_limit = 5
        proactive_config.agent_tick_context_prob = 0.0
        proactive_config.agent_tick_delivery_cooldown_hours = 0
        proactive_config.message_dedupe_enabled = False
        proactive_config.delivery_dedupe_hours = 0
        proactive_config.drift_enabled = mode == "drift"
        proactive_config.drift_max_steps = 8
        proactive_config.drift_min_interval_hours = 0
        model = app_config.model
        max_tokens = app_config.max_tokens
    sent: list[dict[str, str]] = []
    push = MessagePushTool()

    async def send_text(chat_id: str, message: str) -> None:
        sent.append({"chat_id": chat_id, "message": message})

    push.register_channel("sandbox", text=send_text)
    await plugins.load_all()
    connect_result = await mcp.add(
        "feed",
        [sys.executable, str(feed_root / "mcp" / "run_mcp.py")],
        env={"AKA_PLUGIN_DATA_DIR": str(workspace / "feed-data")},
        cwd=str(feed_root),
    )
    if not any(
        name.endswith("__get_proactive_events")
        for name in tools.get_registered_names()
    ):
        raise RuntimeError(f"Feed MCP 未连接: {connect_result}")
    if len(plugins.proactive_runtime_factories) != 1:
        raise RuntimeError("主动 Runtime factory 数量异常")
    if len(plugins.proactive_module_factories) != 3:
        raise RuntimeError("主动 Module factory 未完整加载")

    loop = ProactiveLoop(
        session_manager=sessions,
        provider=provider,
        push_tool=push,
        config=proactive_config,
        model=model,
        max_tokens=max_tokens,
        state_store=state,
        rng=random.Random(0),
        shared_tools=tools,
        event_bus=event_bus,
        proactive_modules=plugins.proactive_modules,
        proactive_lifecycles=plugins.proactive_lifecycles,
        proactive_module_factories=plugins.proactive_module_factories,
        proactive_runtime_factories=plugins.proactive_runtime_factories,
    )
    try:
        await loop._proactive_kernel.start()
        await loop._tick()
        await event_bus.drain()
        latest_tick = status(workspace).get("tick") or {}
        return {
            "mode": mode,
            "decision": latest_tick.get("terminal_action"),
            "sent": sent,
            "model_tools": list(getattr(provider, "calls", [])),
            "mcp_tools": sorted(tools.get_tool_names_by_source("mcp", "feed")),
            "lifecycle": loop._proactive_kernel.inspect(),
        }
    finally:
        await loop._proactive_kernel.stop()
        await plugins.terminate_all()
        await mcp.shutdown()
        await event_bus.aclose()
        state.close()
        sessions._store.close()


def status(workspace: Path) -> dict[str, Any]:
    return {
        "feed": _query_one(
            workspace / "feed-data" / "feed_mcp.sqlite3",
            "SELECT event_id, interest_ok, interest_scored_at FROM items WHERE event_id = ?",
            (EVENT_ID,),
        ),
        "tick": _query_one(
            workspace / "proactive.db",
            """
            SELECT terminal_action, content_count, drift_entered, final_message
            FROM tick_log ORDER BY rowid DESC LIMIT 1
            """,
        ),
        "steps": _query_all(
            workspace / "proactive.db",
            """
            SELECT step_index, phase, tool_name, terminal_action_after
            FROM tick_step_log
            WHERE tick_id = (SELECT tick_id FROM tick_log ORDER BY rowid DESC LIMIT 1)
            ORDER BY step_index, id
            """,
        ),
        "drift": _query_one(
            workspace / "drift" / "drift.db",
            """
            SELECT skill_name, status, briefing, message_result
            FROM runs ORDER BY id DESC LIMIT 1
            """,
        ),
        "session": _query_one(
            workspace / "sessions.db",
            """
            SELECT role, content, extra FROM messages
            ORDER BY seq DESC LIMIT 1
            """,
        ),
    }


def _query_one(
    path: Path,
    sql: str,
    params: tuple[object, ...] = (),
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(sql, params).fetchone()
        except sqlite3.OperationalError:
            return None
    return dict(row) if row is not None else None


def _query_all(path: Path, sql: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


async def run_all(
    workspace: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    reset(workspace)
    inject_content(workspace)
    content_result = await tick(workspace, "content", config_path)
    content_status = status(workspace)
    clear_content(workspace)
    drift_result = await tick(workspace, "drift", config_path)
    drift_status = status(workspace)
    if content_result["decision"] != "reply" or not content_result["sent"]:
        raise AssertionError("content 主动发送未完成")
    feed = content_status["feed"] or {}
    if feed.get("interest_ok") != 1:
        raise AssertionError("content ACK 未写回 Feed MCP")
    if drift_result["decision"] != "reply" or not drift_result["sent"]:
        raise AssertionError("Drift 主动发送未完成")
    drift = drift_status["drift"] or {}
    if drift.get("status") != "completed" or drift.get("message_result") != "sent":
        raise AssertionError("Drift 状态未完整提交")
    return {
        "content": content_result,
        "content_status": content_status,
        "drift": drift_result,
        "drift_status": drift_status,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="主动链路 Docker 操作沙盒")
    parser.add_argument(
        "command",
        choices=["reset", "inject-content", "clear-content", "tick-content", "tick-drift", "status", "run-all"],
    )
    parser.add_argument("--workspace", type=Path, default=_workspace_from_env())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    workspace = args.workspace
    _assert_sandbox_path(workspace)
    if args.command == "reset":
        reset(workspace)
        result: object = {"ok": True, "workspace": str(workspace)}
    elif args.command == "inject-content":
        inject_content(workspace)
        result = status(workspace)
    elif args.command == "clear-content":
        clear_content(workspace)
        result = status(workspace)
    elif args.command == "tick-content":
        result = asyncio.run(tick(workspace, "content", args.config))
    elif args.command == "tick-drift":
        result = asyncio.run(tick(workspace, "drift", args.config))
    elif args.command == "status":
        result = status(workspace)
    else:
        result = asyncio.run(run_all(workspace, args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
