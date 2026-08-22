"""Project core-owned runtime facts into the read-only mobile protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugin_composition import TopologyFiberView
from agent.plugins.composable import ComposablePlugin
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotLease,
    RuntimeSnapshotStore,
)
from agent.scheduler import JobStore, ScheduledJob
from agent.skills import SkillRecord, SkillsLoader

_MAX_DOCUMENT_BYTES = 192 * 1024
_MAX_RECENT_PLUGIN_INCIDENTS = 20


@dataclass(frozen=True, slots=True)
class RuntimeDocument:
    id: str
    title: str
    relative_path: str
    group: str
    description: str


_DOCUMENTS = (
    RuntimeDocument(
        "memory",
        "长期记忆",
        "memory/MEMORY.md",
        "memory",
        "沉淀后的长期事实、偏好与经验。",
    ),
    RuntimeDocument(
        "self",
        "自我认知",
        "memory/SELF.md",
        "identity",
        "Agent 对自身状态与能力边界的认识。",
    ),
    RuntimeDocument(
        "veda",
        "VEDA 人格",
        "memory/VEDA.md",
        "identity",
        "Main、Proactive 与 Drift 共用的人格真源。",
    ),
    RuntimeDocument(
        "pending",
        "待处理线索",
        "memory/PENDING.md",
        "memory",
        "尚未沉淀或仍需处理的记忆线索。",
    ),
    RuntimeDocument(
        "proactive-context",
        "主动上下文",
        "PROACTIVE_CONTEXT.md",
        "context",
        "主动任务使用的长期上下文。",
    ),
)
_DOCUMENT_BY_ID = {document.id: document for document in _DOCUMENTS}


class RuntimeInspectionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RuntimeInspectionService:
    """从运行时 owner 投影移动端可读的文档、任务与能力。"""

    def __init__(
        self,
        *,
        workspace: Path,
        snapshot_store: RuntimeSnapshotStore | None,
    ) -> None:
        self._workspace = workspace.expanduser().resolve()
        self._job_store = JobStore(self._workspace / "schedules.json")
        self._snapshot_store = snapshot_store

    def list_documents(self) -> dict[str, object]:
        return {"items": [self._document_summary(item) for item in _DOCUMENTS]}

    def get_document(self, document_id: str) -> dict[str, object]:
        document = _DOCUMENT_BY_ID.get(document_id)
        if document is None:
            raise RuntimeInspectionError(
                "document_not_found",
                f"未知运行时文档: {document_id}",
            )
        path = self._workspace / document.relative_path
        try:
            size = path.stat().st_size
        except FileNotFoundError as exc:
            raise RuntimeInspectionError(
                "document_unavailable",
                f"运行时文档不存在: {document.relative_path}",
            ) from exc
        if size > _MAX_DOCUMENT_BYTES:
            raise RuntimeInspectionError(
                "document_too_large",
                f"运行时文档超过 192 KiB: {document.relative_path}",
            )
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeInspectionError(
                "document_invalid_utf8",
                f"运行时文档不是合法 UTF-8: {document.relative_path}",
            ) from exc
        return {**self._document_summary(document), "markdown": content}

    def list_jobs(self) -> dict[str, object]:
        jobs = sorted(
            self._active_jobs(),
            key=lambda job: (job.fire_at, job.id),
        )
        return {"items": [self._job_summary(job) for job in jobs]}

    def get_job(self, job_id: str) -> dict[str, object]:
        job = next(
            (candidate for candidate in self._active_jobs() if candidate.id == job_id),
            None,
        )
        if job is None:
            raise RuntimeInspectionError("job_not_found", f"定时任务不存在: {job_id}")
        return {**self._job_summary(job), "markdown": _job_markdown(job)}

    def _active_jobs(self) -> list[ScheduledJob]:
        return [job for job in self._job_store.load() if job.enabled]

    async def list_capabilities(self) -> dict[str, object]:
        async with await self._acquire_snapshot() as snapshot:
            return {
                "snapshot_id": snapshot.snapshot_id,
                "plugins": _plugin_items(snapshot),
                "skills": _skill_items(self._workspace, snapshot),
                "mcp_servers": _mcp_items(snapshot),
            }

    async def get_mcp(self, owner_id: str, server_name: str) -> dict[str, object]:
        async with await self._acquire_snapshot() as snapshot:
            server = _find_mcp_item(snapshot, owner_id, server_name)
            if server is None:
                raise RuntimeInspectionError(
                    "mcp_not_found",
                    f"MCP server 不存在: {owner_id}/{server_name}",
                )
            tools = cast(list[dict[str, object]], server["tools"])
            return {
                "owner_id": owner_id,
                "name": server_name,
                "tool_count": len(tools),
                "tools": tools,
                "markdown": _mcp_markdown(owner_id, server_name, tools),
            }

    async def _acquire_snapshot(self) -> RuntimeSnapshotLease:
        if self._snapshot_store is None or self._snapshot_store.current is None:
            raise RuntimeInspectionError(
                "runtime_snapshot_unavailable",
                "运行时能力快照尚未就绪",
            )
        return await self._snapshot_store.acquire()

    def _document_summary(self, document: RuntimeDocument) -> dict[str, object]:
        path = self._workspace / document.relative_path
        return {
            "id": document.id,
            "title": document.title,
            "relative_path": document.relative_path,
            "group": document.group,
            "description": document.description,
            "available": path.is_file(),
        }

    @staticmethod
    def _job_summary(job: ScheduledJob) -> dict[str, object]:
        return {
            "id": job.id,
            "name": job.name,
            "trigger": job.trigger,
            "tier": job.tier,
            "fire_at": job.fire_at.isoformat(),
            "timezone": job.timezone,
            "enabled": job.enabled,
            "run_count": job.run_count,
        }


def _plugin_items(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    """Project generation and composition facts from one leased snapshot."""

    composition = _plugin_composition_items(snapshot)
    items: list[dict[str, object]] = []
    for generation in sorted(
        snapshot.active_generations(),
        key=lambda item: item.plugin_id,
    ):
        api_version = cast(ComposablePlugin, generation.instance).api_version
        current = composition.get(generation.plugin_id)
        if current is None:
            raise RuntimeError(
                f"stable v3 插件缺少 composition inspection: {generation.plugin_id}"
            )
        items.append(
            {
                "id": generation.plugin_id,
                "revision": generation.source_revision,
                "generation_id": generation.generation_id,
                "api_version": api_version,
                "composition": current,
            }
        )
    return items


def _plugin_composition_items(
    snapshot: RuntimeSnapshot,
) -> dict[str, dict[str, object]]:
    """Group current Root facts by the top-level plugin Fiber owner."""

    root = snapshot.composition_root
    topology = snapshot.composition_topology
    if root is None and topology is None:
        return {}
    if root is None or topology is None:
        raise RuntimeError(
            "stable snapshot 的 composition Root 与 Topology 必须成对存在"
        )

    # 1. Frozen parent edges assign every nested Fiber to one top-level plugin.
    all_v3_plugin_ids = set(snapshot.generations)
    active_v3_plugin_ids = {
        generation.plugin_id for generation in snapshot.active_generations()
    }
    all_owners = _top_level_plugin_owners(topology.fibers, all_v3_plugin_ids)
    receipt = root.receipt()
    current_fibers = {fiber.name: fiber for fiber in receipt.fibers}
    if current_fibers.keys() != all_owners.keys():
        raise RuntimeError("stable snapshot 的 current Fiber 与冻结 Topology 不一致")
    owner_by_fiber = {
        name: owner
        for name, owner in all_owners.items()
        if owner in active_v3_plugin_ids
    }
    incident_counts = dict(receipt.incident_counts)

    # 2. Current Health/Incident state is bounded; cumulative counts remain exact.
    result: dict[str, dict[str, object]] = {}
    for plugin_id in sorted(active_v3_plugin_ids):
        owned_names = {
            name for name, owner in owner_by_fiber.items() if owner == plugin_id
        }
        topology_fibers = tuple(
            fiber for fiber in topology.fibers if fiber.name in owned_names
        )
        health = tuple(item for item in receipt.health if item.owner in owned_names)
        recent_incidents = tuple(
            item for item in receipt.incidents if item.owner in owned_names
        )[-_MAX_RECENT_PLUGIN_INCIDENTS:]
        fibers = tuple(
            current_fibers[fiber.name]
            for fiber in topology_fibers
            if fiber.name in current_fibers
        )
        ready = all(
            not fiber.required_for_readiness or fiber.state.value == "active"
            for fiber in fibers
        ) and all(not item.required or item.healthy for item in health)
        result[plugin_id] = {
            "ready": ready,
            "topology_identity": topology.identity,
            "composition_revision": topology.composition_revision,
            "fibers": [
                {
                    "name": topology_fiber.name,
                    "parent": topology_fiber.parent,
                    "state": current_fibers[topology_fiber.name].state.value,
                    "required": topology_fiber.required_for_readiness,
                    "static_active": topology_fiber.static_active,
                    "dependencies": list(topology_fiber.dependencies),
                    "missing_services": list(
                        current_fibers[topology_fiber.name].missing_services
                    ),
                    "error": current_fibers[topology_fiber.name].error,
                }
                for topology_fiber in topology_fibers
            ],
            "health": [
                {
                    "owner": item.owner,
                    "name": item.name,
                    "required": item.required,
                    "healthy": item.healthy,
                    "reason": item.reason,
                }
                for item in health
            ],
            "incident_count": sum(incident_counts.get(name, 0) for name in owned_names),
            "recent_incidents": [
                {
                    "sequence": item.sequence,
                    "owner": item.owner,
                    "kind": item.kind,
                    "message": item.message,
                    "error_type": item.error_type,
                }
                for item in recent_incidents
            ],
            "incident_overflowed": receipt.incident_overflowed,
        }
    return result


def _top_level_plugin_owners(
    fibers: tuple[TopologyFiberView, ...],
    plugin_ids: set[str],
) -> dict[str, str]:
    """Resolve each frozen Fiber name to its top-level plugin Fiber."""

    parent_by_name = {fiber.name: fiber.parent for fiber in fibers}
    owners: dict[str, str] = {}
    for name in parent_by_name:
        current = name
        seen: set[str] = set()
        while parent_by_name[current] is not None:
            if current in seen:
                raise RuntimeError(f"composition parent edge 构成循环: {name}")
            seen.add(current)
            parent = parent_by_name[current]
            if not isinstance(parent, str) or parent not in parent_by_name:
                raise RuntimeError(f"composition parent edge 缺失: {name} -> {parent}")
            current = parent
        if current not in plugin_ids:
            raise RuntimeError(
                f"composition 顶层 Fiber 不属于 active v3 插件: {current}"
            )
        owners[name] = current
    return owners


def _skill_items(workspace: Path, snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    records: dict[str, SkillRecord] = {
        record.name: record
        for record in SkillsLoader(workspace).list_skill_records(
            filter_unavailable=False
        )
    }
    if snapshot.plugin_skill_index is not None:
        for name, record in snapshot.plugin_skill_index.records.items():
            _ = records.setdefault(name, record)
    return [
        {
            "name": record.name,
            "display_name": record.display_name,
            "description": record.description,
            "source": record.source,
            "source_id": record.source_id,
            "available": record.available,
            "missing": record.missing,
        }
        for record in sorted(records.values(), key=lambda item: item.name)
    ]


def _mcp_items(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    """Project exact v3 MCP servers from the stable Root registry."""

    # 1. v3 declarations provide owner identity; ToolRegistry provides live schemas.
    items: list[dict[str, object]] = []
    registry = snapshot.mcp_server_registry
    if registry is not None:
        if snapshot.tool_registry is None:
            raise RuntimeError("stable v3 MCP registry 缺少 exact ToolRegistry")
        for descriptor in registry.descriptors:
            tools = _mcp_tools_from_registry(snapshot, descriptor.name)
            items.append(
                {
                    "owner_id": descriptor.owner,
                    "name": descriptor.name,
                    "tool_count": len(tools),
                    "tools": tools,
                }
            )

    return sorted(items, key=lambda item: (str(item["owner_id"]), str(item["name"])))


def _mcp_tools_from_registry(
    snapshot: RuntimeSnapshot,
    server_name: str,
) -> list[dict[str, object]]:
    """Read one exact live MCP server projection from the frozen ToolRegistry."""

    registry = snapshot.tool_registry
    if registry is None:
        raise RuntimeError("stable v3 MCP registry 缺少 exact ToolRegistry")
    prefix = f"mcp_{server_name}__"
    tools: list[dict[str, object]] = []
    for name in registry.get_registered_order(
        registry.get_source_tool_names("mcp", server_name)
    ):
        tool = registry.get_tool(name)
        if tool is None:
            raise RuntimeError(f"stable MCP ToolRegistry 缺少已登记工具: {name}")
        remote_name = name.removeprefix(prefix)
        description = tool.description.removeprefix(f"[MCP:{server_name}] ")
        tools.append(
            {
                "name": remote_name,
                "description": description,
                "input_schema": tool.parameters or {},
            }
        )
    return tools


def _find_mcp_item(
    snapshot: RuntimeSnapshot,
    owner_id: str,
    server_name: str,
) -> dict[str, object] | None:
    return next(
        (
            item
            for item in _mcp_items(snapshot)
            if item["owner_id"] == owner_id and item["name"] == server_name
        ),
        None,
    )


def _job_markdown(job: ScheduledJob) -> str:
    content = job.message if job.tier == "instant" else job.prompt
    schedule = job.cron_expr or (
        f"每 {job.interval_seconds} 秒"
        if job.interval_seconds is not None
        else job.fire_at.isoformat()
    )
    return "\n".join(
        (
            f"# {job.name or '未命名定时任务'}",
            "",
            f"- **状态：** {'启用' if job.enabled else '停用'}",
            f"- **触发：** `{job.trigger}` / `{job.tier}`",
            f"- **计划：** {schedule}",
            f"- **时区：** `{job.timezone}`",
            f"- **运行次数：** {job.run_count}",
            "",
            "## 内容",
            "",
            content or "",
        )
    )


def _mcp_markdown(
    owner_id: str,
    server_name: str,
    tools: list[dict[str, object]],
) -> str:
    lines = [
        f"# {server_name}",
        "",
        f"归属：`{owner_id}`",
        "",
        "## 工具",
        "",
    ]
    for tool in tools:
        lines.extend(
            (
                f"### `{tool['name']}`",
                "",
                str(tool["description"]),
                "",
                "```json",
                json.dumps(
                    tool["input_schema"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            )
        )
    return "\n".join(lines)
