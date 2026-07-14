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
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.config_models import Config
from agent.plugins.manager import PluginManager
from agent.provider import LLMResponse, ToolCall
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.web_fetch import WebFetchTool
from bus.event_bus import EventBus
from bootstrap.proactive import _build_proactive_provider
from bootstrap.providers import build_providers
from proactive_v2.config import ProactiveConfig
from proactive_v2.loop import ProactiveLoop
from proactive_v2.state import ProactiveStateStore
from plugins.drift_flow.state import DriftStateStore
from plugins.wake_proactive.state import WakeStateStore
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from session.manager import SessionManager

EVENT_ID = "sandbox_content_001"
SKILL_NAME = "sandbox_probe"
PAUSED_SKILL_NAME = "paused-plan-probe"


class SandboxProvider:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[str] = []
        self.content_item_id = ""

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
            item_id = self.content_item_id
            if "scratchpad" in names:
                return "scratchpad", {
                    "items": [
                        {
                            "item_id": item_id,
                            "initial_interest": "likely_interesting",
                            "question": "确认沙盒正文能否被读取并分享",
                        }
                    ]
                }
            if "share_content" in names:
                return "share_content", {
                    "message": "沙盒主动推送：已读取 Feed MCP content。",
                    "items": [
                        {
                            "item_id": item_id,
                            "summary": "主动链路沙盒内容",
                            "why_it_matters": "验证 Wake 与 Feed MCP 的完整协作链",
                        }
                    ],
                }
            if "mark_interesting" in names and "mark_interesting" not in self.calls:
                return "mark_interesting", {"item_ids": [item_id], "reason": "沙盒验证"}
            if "message_push" in names and "message_push" not in self.calls:
                return "message_push", {
                    "message": "沙盒主动推送：已读取 Feed MCP content。",
                    "evidence": [item_id],
                }
            return "finish_turn", {"decision": "reply"}

        if "select_skill" in names and "select_skill" not in self.calls:
            return "select_skill", {
                "skill_name": SKILL_NAME,
                "decision": "explore",
                "intention": "验证主动 Drift 完整协作链",
                "reason": "沙盒当前没有 content，执行固定验证 skill",
            }
        if "message_push" in names and "message_push" not in self.calls:
            return "message_push", {"message": "沙盒 Drift：空闲链路已执行。"}
        return "finish_drift", {
            "skill_used": SKILL_NAME,
            "status": "completed",
            "briefing": "完成沙盒 Drift 验证",
            "self_update": {
                "pattern": "ordinary",
                "reflection": "本轮只执行固定的沙盒协作链验证，没有形成新的行为模式。",
                "next_tendency": "下次按当时来源和上下文决定是否继续验证。",
            },
        }


def _workspace_from_env() -> Path:
    root = Path(os.environ.get("AKASHIC_DEBUG_WORKSPACE", "/sandbox/workspace"))
    return root / "proactive-sandbox"


def _installed_plugin_root(
    cache_root: Path,
    plugin_name: str,
) -> Path:
    plugin_roots = sorted(cache_root.glob(f"*/{plugin_name}"))
    if len(plugin_roots) != 1:
        raise RuntimeError(
            f"外置插件来源不唯一: {plugin_name}={plugin_roots}"
        )
    plugin_root = plugin_roots[0]
    versions = sorted(path for path in plugin_root.iterdir() if path.is_dir())
    if not versions:
        raise RuntimeError(f"外置插件没有可运行版本: {plugin_root}")
    return versions[-1]


def _assert_sandbox_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if Path("/sandbox") not in resolved.parents:
        raise RuntimeError(f"沙盒路径必须位于 /sandbox: {resolved}")


def reset(workspace: Path) -> None:
    _assert_sandbox_path(workspace)
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
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
                "https://example.com/",
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


def prime_wake_hazard(workspace: Path) -> None:
    """把 Wake 水位置于确定可触发状态，只验证阈值后的协作链。"""

    # 1. 固定阈值，避免随机采样掩盖后续链路问题
    now = datetime.now(UTC)
    store = WakeStateStore(workspace / "wake_proactive.db")
    try:
        store.save_hazard(
            session_key="sandbox:operator",
            hazard=0.0,
            threshold=0.0,
            updated_at=now,
            last_wake_at=None,
        )
    finally:
        store.close()


def prepare_paused_resume(workspace: Path) -> None:
    _assert_sandbox_path(workspace)
    shutil.rmtree(workspace, ignore_errors=True)
    skill_dir = workspace / "drift" / "skills" / PAUSED_SKILL_NAME
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {PAUSED_SKILL_NAME}\n"
        "description: 根据需求制定计划并产出结果，用于验证 paused 续接认知。\n"
        "---\n\n"
        "# Paused Plan Probe\n\n"
        "这是一份能力说明书。完整工作路径如下：\n\n"
        "1. 读取 `skills/paused-plan-probe/requirements.md`。\n"
        "2. 根据需求创建 `skills/paused-plan-probe/plan.json`。\n"
        "3. 读取计划，将其中的 `output` 写入 `skills/paused-plan-probe/result.txt`。\n"
        "4. 调用 `finish_drift(status=\"completed\", message_result=\"silent\")`。\n\n"
        "结合 local_context 判断当前位于哪一步，只执行本轮需要的部分。\n",
        encoding="utf-8",
    )
    (skill_dir / "requirements.md").write_text(
        "产出内容必须是：continued-from-existing-plan\n",
        encoding="utf-8",
    )
    (skill_dir / "plan.json").write_text(
        json.dumps(
            {
                "output": "continued-from-existing-plan",
                "next_action": "write_result",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "PROACTIVE_CONTEXT.md").write_text(
        "# Proactive Context\n\n当前是 paused 断点续接沙盒。\n",
        encoding="utf-8",
    )
    store = DriftStateStore(workspace / "drift")
    store.save_finish(
        skill_used=PAUSED_SKILL_NAME,
        status="paused",
        briefing="计划已生成，但执行阶段遇到临时 502，结果文件尚未写入。",
        message_result="silent",
        scratchpad_update=(
            "requirements.md 已读取，plan.json 已创建。上次在执行计划时遇到临时 502；"
            "result.txt 尚未生成，下次仍从 execute_plan 继续。也允许暂时延后。"
        ),
        global_note_update=None,
        now_utc=datetime.now(UTC),
        cursor_update={
            "phase": "execute_plan",
            "plan_path": f"skills/{PAUSED_SKILL_NAME}/plan.json",
            "next_action": "write_result",
        },
    )


async def verify_paused_resume(
    workspace: Path,
    config_path: Path,
) -> dict[str, Any]:
    prepare_paused_resume(workspace)
    result = await tick(workspace, "drift", "default", config_path)
    drift_steps = _query_all(
        workspace / "drift" / "drift.db",
        """
        SELECT step_index, tool_name, input_preview, output_preview
        FROM run_steps
        WHERE run_id = (SELECT id FROM runs ORDER BY id DESC LIMIT 1)
        ORDER BY step_index, id
        """,
    )
    skill_dir = workspace / "drift" / "skills" / PAUSED_SKILL_NAME
    result_path = skill_dir / "result.txt"
    plan_path = f"skills/{PAUSED_SKILL_NAME}/plan.json"
    requirements_path = f"skills/{PAUSED_SKILL_NAME}/requirements.md"
    restarted = False
    for step in drift_steps:
        raw = str(step.get("input_preview") or "")
        if step.get("tool_name") == "read_file" and requirements_path in raw:
            restarted = True
        if step.get("tool_name") in {"write_file", "edit_file"} and plan_path in raw:
            restarted = True
    if not result_path.exists():
        raise AssertionError(f"模型没有从 paused 计划产出 result.txt: {drift_steps}")
    if restarted:
        raise AssertionError(f"模型重新执行了已完成的前置步骤: {drift_steps}")
    result_text = result_path.read_text(encoding="utf-8").strip()
    if "continued-from-existing-plan" not in result_text:
        raise AssertionError(f"result.txt 未使用已有计划: {result_text!r}")
    return {
        "ok": True,
        "runtime": result,
        "drift": _query_one(
            workspace / "drift" / "drift.db",
            "SELECT id, skill_name, status, briefing FROM runs ORDER BY id DESC LIMIT 1",
        ),
        "drift_steps": drift_steps,
        "result": result_text,
        "restarted_completed_steps": restarted,
    }


def _config(mode: str, lifecycle: str) -> ProactiveConfig:
    return ProactiveConfig(
        enabled=True,
        lifecycle=lifecycle,
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
    lifecycle: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    event_bus = EventBus()
    tools = ToolRegistry()
    http_resources = SharedHttpResources()
    configure_default_shared_http_resources(http_resources)
    tools.register(WebFetchTool(http_resources.external_default))
    installed_cache = Path.home() / ".akashic-plugin" / "cache"
    feed_root = _installed_plugin_root(installed_cache, "feed")
    sessions = SessionManager(workspace)
    plugins = PluginManager(
        plugin_dirs=[Path("/app/plugins")],
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=installed_cache,
    )
    state = ProactiveStateStore(workspace / "proactive.db")
    provider: Any = SandboxProvider(mode)
    proactive_config = _config(mode, lifecycle)
    model = "sandbox-model"
    max_tokens = 1024
    if config_path is not None:
        app_config = Config.load(config_path)
        main_provider, _, _ = build_providers(app_config)
        provider = _build_proactive_provider(app_config, main_provider)
        proactive_config = app_config.proactive
        proactive_config.lifecycle = lifecycle
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
        proactive_config.drift_min_interval_hours = 0
        model = app_config.model
        max_tokens = app_config.max_tokens
    sent: list[dict[str, str]] = []
    push = MessagePushTool()

    async def send_text(chat_id: str, message: str) -> None:
        sent.append({"chat_id": chat_id, "message": message})

    push.register_channel("sandbox", text=send_text)
    await plugins.load_all()
    if isinstance(provider, SandboxProvider):
        feed_sources = [
            source
            for source in plugins.proactive_sources
            if source.spec.server == "feed"
        ]
        if len(feed_sources) != 1:
            raise RuntimeError(
                f"Feed 主动来源数量异常: {feed_sources}; "
                f"gate={plugins.latest_gate('feed@github')}"
            )
        source = feed_sources[0]
        provider.content_item_id = (
            f"{source.plugin_id}:{source.spec.id}:{EVENT_ID}"
        )
    snapshot = plugins.current_snapshot
    snapshot_tools = snapshot.tool_registry if snapshot is not None else None
    registered_names = (
        snapshot_tools.get_registered_names() if snapshot_tools is not None else set()
    )
    if not any(
        name.endswith("__get_proactive_events")
        for name in registered_names
    ):
        raise RuntimeError("Feed MCP 未连接")
    if len(plugins.proactive_runtime_factories) != 1:
        raise RuntimeError("主动 Runtime factory 数量异常")
    if len(plugins.proactive_module_factories) != 3:
        raise RuntimeError("主动 Module factory 未完整加载")

    loop = ProactiveLoop(
        session_manager=sessions,
        provider=cast(Any, provider),
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
        proactive_sources=plugins.proactive_sources,
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
            "mcp_tools": sorted(
                snapshot_tools.get_tool_names_by_source("mcp", "feed")
                if snapshot_tools is not None
                else set()
            ),
            "lifecycle": loop._proactive_kernel.inspect(),
        }
    finally:
        await loop._proactive_kernel.stop()
        await plugins.terminate_all()
        await event_bus.aclose()
        clear_default_shared_http_resources(http_resources)
        await http_resources.aclose()
        state.close()
        sessions._store.close()


def status(workspace: Path) -> dict[str, Any]:
    return {
        "feed": _query_one(
            workspace / "feed-data" / "feed_mcp.sqlite3",
            "SELECT event_id, interest_ok, interest_scored_at FROM items WHERE event_id = ?",
            (EVENT_ID,),
        ),
        "feed_ack": _query_one(
            workspace / "feed-data" / "feed_mcp.sqlite3",
            "SELECT event_id, acked_at FROM acked_items WHERE event_id = ?",
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
    lifecycle: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    reset(workspace)
    inject_content(workspace)
    if lifecycle == "wake":
        prime_wake_hazard(workspace)
    content_result = await tick(workspace, "content", lifecycle, config_path)
    content_status = status(workspace)
    if lifecycle == "wake":
        drift_result: dict[str, Any] = {"skipped": "content tick 已覆盖 wake.drift.decide"}
        drift_status = content_status
    else:
        clear_content(workspace)
        drift_result = await tick(workspace, "drift", lifecycle, config_path)
        drift_status = status(workspace)
    if not content_result["sent"]:
        raise AssertionError("content 主动发送未完成")
    if lifecycle == "wake":
        if not content_status["feed_ack"]:
            raise AssertionError("Wake content ACK 未写回 Feed MCP")
    else:
        if content_result["decision"] != "reply":
            raise AssertionError("default content 未以 reply 收尾")
        feed = content_status["feed"] or {}
        if feed.get("interest_ok") != 1:
            raise AssertionError("content 兴趣反馈未写回 Feed MCP")
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
        choices=[
            "reset",
            "inject-content",
            "clear-content",
            "tick-content",
            "tick-drift",
            "verify-paused-resume",
            "status",
            "run-all",
        ],
    )
    parser.add_argument("--workspace", type=Path, default=_workspace_from_env())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lifecycle", choices=("default", "wake"), default="default")
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
        result = asyncio.run(tick(workspace, "content", args.lifecycle, args.config))
    elif args.command == "tick-drift":
        result = asyncio.run(tick(workspace, "drift", args.lifecycle, args.config))
    elif args.command == "verify-paused-resume":
        if args.config is None:
            raise SystemExit("verify-paused-resume 必须通过 --config 使用真实模型")
        result = asyncio.run(verify_paused_resume(workspace, args.config))
    elif args.command == "status":
        result = status(workspace)
    else:
        result = asyncio.run(run_all(workspace, args.lifecycle, args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
