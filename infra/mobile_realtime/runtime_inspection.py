"""Project core-owned runtime facts into the read-only mobile protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent.mcp.host import PreparedMcpServer
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotLease,
    RuntimeSnapshotStore,
)
from agent.scheduler import ScheduledJob, SchedulerService
from agent.skills import SkillRecord, SkillsLoader

_MAX_DOCUMENT_BYTES = 192 * 1024


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
        scheduler: SchedulerService,
        snapshot_store: RuntimeSnapshotStore | None,
    ) -> None:
        self._workspace = workspace.expanduser().resolve()
        self._scheduler = scheduler
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
            self._scheduler.list_jobs(),
            key=lambda job: (job.fire_at, job.id),
        )
        return {"items": [self._job_summary(job) for job in jobs]}

    def get_job(self, job_id: str) -> dict[str, object]:
        job = next(
            (
                candidate
                for candidate in self._scheduler.list_jobs()
                if candidate.id == job_id
            ),
            None,
        )
        if job is None:
            raise RuntimeInspectionError("job_not_found", f"定时任务不存在: {job_id}")
        return {**self._job_summary(job), "markdown": _job_markdown(job)}

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
            server = _find_mcp_server(snapshot, owner_id, server_name)
            if server is None:
                raise RuntimeInspectionError(
                    "mcp_not_found",
                    f"MCP server 不存在: {owner_id}/{server_name}",
                )
            tools: list[dict[str, object]] = [
                {
                    "name": info.name,
                    "description": info.description,
                    "input_schema": info.input_schema,
                }
                for info in server.client.tool_infos
            ]
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
    return [
        {
            "id": generation.plugin_id,
            "revision": generation.source_revision,
            "generation_id": generation.generation_id,
        }
        for generation in sorted(
            snapshot.active_generations(),
            key=lambda item: item.plugin_id,
        )
    ]


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


def _mcp_catalogs(snapshot: RuntimeSnapshot):
    for generation in sorted(
        snapshot.active_generations(),
        key=lambda item: item.plugin_id,
    ):
        if generation.mcp_catalog is not None:
            yield generation.plugin_id, generation.mcp_catalog
    if snapshot.workspace_mcp_generation is not None:
        yield "workspace", snapshot.workspace_mcp_generation.catalog


def _mcp_items(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    return [
        {
            "owner_id": owner_id,
            "name": server.name,
            "tool_count": len(server.tools),
            "tools": [
                {
                    "name": info.name,
                    "description": info.description,
                }
                for info in server.client.tool_infos
            ],
        }
        for owner_id, catalog in _mcp_catalogs(snapshot)
        for server in sorted(catalog.servers.values(), key=lambda item: item.name)
    ]


def _find_mcp_server(
    snapshot: RuntimeSnapshot,
    owner_id: str,
    server_name: str,
) -> PreparedMcpServer | None:
    return next(
        (
            catalog.servers.get(server_name)
            for candidate_owner, catalog in _mcp_catalogs(snapshot)
            if candidate_owner == owner_id
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
