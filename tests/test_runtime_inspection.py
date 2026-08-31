from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from infra.mobile_realtime.runtime_inspection import (
    RuntimeInspectionError,
    RuntimeInspectionService,
    _mcp_items,
)
from agent.plugins.snapshot import RuntimeSnapshot
from agent.plugins.manager import PluginManager
from agent.scheduler import JobStore, ScheduledJob
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus


def _service(tmp_path: Path) -> tuple[RuntimeInspectionService, JobStore]:
    store = JobStore(tmp_path / "schedules.json")
    service = RuntimeInspectionService(
        workspace=tmp_path,
        snapshot_store=None,
    )
    return service, store


def _write_plugin(root: Path, name: str, source: str) -> None:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")


class _InspectionMcpTool(Tool):
    @property
    def name(self) -> str:
        return "mcp_calendar__list_events"

    @property
    def description(self) -> str:
        return "[MCP:calendar] List calendar events"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"date": {"type": "string"}},
        }

    async def execute(self, **kwargs: object) -> str:
        return "unused"


def test_documents_use_fixed_allowlist_and_return_markdown(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    path = tmp_path / "memory/MEMORY.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Memory\n\n真实内容", encoding="utf-8")

    listed = service.list_documents()
    document = service.get_document("memory")

    assert len(cast(list[object], listed["items"])) == 3
    assert document["relative_path"] == "memory/MEMORY.md"
    assert document["markdown"] == "# Memory\n\n真实内容"
    with pytest.raises(RuntimeInspectionError, match="未知运行时文档"):
        service.get_document("../../config.toml")
    with pytest.raises(RuntimeInspectionError, match="未知运行时文档"):
        service.get_document("proactive-context")


def test_scheduler_projection_reads_live_service_state(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    job = ScheduledJob(
        id="morning",
        name="晨间提醒",
        trigger="every",
        tier="instant",
        fire_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        channel="mobile",
        chat_id="mobile:test",
        interval_seconds=3600,
        message="起来走一走",
    )
    store.save({job.id: job})

    listed = service.list_jobs()
    detail = service.get_job("morning")

    assert cast(list[dict[str, object]], listed["items"])[0]["id"] == "morning"
    assert "起来走一走" in cast(str, detail["markdown"])
    with pytest.raises(RuntimeInspectionError, match="定时任务不存在"):
        service.get_job("missing")


@pytest.mark.asyncio
async def test_capabilities_fail_loud_without_runtime_snapshot(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    with pytest.raises(RuntimeInspectionError, match="快照尚未就绪"):
        await service.list_capabilities()


def test_mcp_projection_uses_exact_v3_registry_and_live_tool_view() -> None:
    registry = ToolRegistry()
    registry.register(
        _InspectionMcpTool(),
        source_type="mcp",
        source_name="calendar",
    )
    snapshot = cast(
        RuntimeSnapshot,
        SimpleNamespace(
            mcp_server_registry=SimpleNamespace(
                descriptors=(
                    SimpleNamespace(owner="calendar-plugin", name="calendar"),
                )
            ),
            tool_registry=registry,
        ),
    )

    assert _mcp_items(snapshot) == [
        {
            "owner_id": "calendar-plugin",
            "name": "calendar",
            "tool_count": 1,
            "tools": [
                {
                    "name": "list_events",
                    "description": "List calendar events",
                    "input_schema": {
                        "type": "object",
                        "properties": {"date": {"type": "string"}},
                    },
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_capabilities_project_bounded_v3_composition_facts(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "inspected_v3",
        "api_version = 3\n"
        "name = 'inspected_v3'\n"
        "version = '1.0.0'\n"
        "async def worker(ctx):\n"
        "    health = await ctx.health('poller', required=False)\n"
        "    health.degrade('paused')\n"
        "    for index in range(140):\n"
        "        ctx.report_incident('poll', f'failure {index}')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.mount(worker, name='worker', required_for_readiness=False)\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "inactive_v3",
        "api_version = 3\n"
        "name = 'inactive_v3'\n"
        "version = '1.0.0'\n"
        "def is_active(services): return False\n"
        "async def worker(ctx): pass\n"
        "async def apply(ctx, config):\n"
        "    await ctx.mount(worker, name='inactive_worker')\n",
    )
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    service = RuntimeInspectionService(
        workspace=workspace,
        snapshot_store=manager.snapshot_store,
    )
    await manager.load_all()

    payload = await service.list_capabilities()

    plugins = {
        cast(str, item["id"]): item
        for item in cast(list[dict[str, object]], payload["plugins"])
    }
    assert "inactive_v3" not in plugins
    v3 = plugins["inspected_v3"]
    assert v3["api_version"] == 3
    composition = cast(dict[str, object], v3["composition"])
    assert composition["ready"] is True
    fibers = cast(list[dict[str, object]], composition["fibers"])
    assert [(item["name"], item["parent"]) for item in fibers] == [
        ("inspected_v3", None),
        ("worker", "inspected_v3"),
    ]
    health = cast(list[dict[str, object]], composition["health"])
    assert health == [
        {
            "owner": "worker",
            "name": "poller",
            "required": False,
            "healthy": False,
            "reason": "paused",
        }
    ]
    assert composition["incident_count"] == 140
    incidents = cast(
        list[dict[str, object]],
        composition["recent_incidents"],
    )
    assert len(incidents) == 20
    assert incidents[0]["message"] == "failure 120"
    assert incidents[-1]["message"] == "failure 139"

    await manager.terminate_all()

    with pytest.raises(RuntimeInspectionError, match="快照尚未就绪"):
        await service.list_capabilities()
