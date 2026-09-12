"""普通 runtime_inspection 插件拥有的中性检查 provider。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent.plugin_composition import ServiceKey


_MAX_DOCUMENT_BYTES = 192 * 1024


class SchedulerReader(Protocol):
    """scheduler 只读投影的窄输入。"""

    def list_jobs(self) -> tuple[Mapping[str, object], ...]: ...

    def get_job(self, job_id: str) -> Mapping[str, object] | None: ...


class SkillReader(Protocol):
    """技能目录只读投影的窄输入。"""

    def list_skills(self) -> tuple[Mapping[str, object], ...]: ...


SCHEDULER_INSPECTION = ServiceKey[SchedulerReader]("scheduler.inspection.v1")
SKILL_INSPECTION = ServiceKey[SkillReader]("standard_tools.skill_inspection.v1")


@dataclass(frozen=True, slots=True)
class _Document:
    id: str
    title: str
    relative_path: str
    group: str
    description: str


_DOCUMENTS = (
    _Document(
        "memory",
        "长期记忆",
        "memory/MEMORY.md",
        "memory",
        "沉淀后的长期事实、偏好与经验。",
    ),
    _Document(
        "self",
        "自我认知",
        "memory/SELF.md",
        "identity",
        "Agent 对自身状态与能力边界的认识。",
    ),
    _Document(
        "veda",
        "VEDA 人格",
        "memory/VEDA.md",
        "identity",
        "Agent 的人格真源。",
    ),
)


class RuntimeInspectionProvider:
    """拥有文档 allowlist，并组合可选的 scheduler/skill 只读输入。"""

    def __init__(self, document_paths: Mapping[str, Path]) -> None:
        self._documents = tuple(
            (document, document_paths[document.id]) for document in _DOCUMENTS
        )
        self._document_by_id = {
            document.id: (document, path) for document, path in self._documents
        }
        self._scheduler: SchedulerReader | None = None
        self._skills: SkillReader | None = None

    def bind_scheduler(self, service: SchedulerReader) -> None:
        self._scheduler = service

    def unbind_scheduler(self, service: SchedulerReader) -> None:
        if self._scheduler is service:
            self._scheduler = None

    def bind_skills(self, service: SkillReader) -> None:
        self._skills = service

    def unbind_skills(self, service: SkillReader) -> None:
        if self._skills is service:
            self._skills = None

    def list_documents(self) -> tuple[Mapping[str, object], ...]:
        """返回固定 allowlist 的元数据，不暴露任意文件路径入口。"""

        return tuple(
            self._summary(document, path) for document, path in self._documents
        )

    def get_document(self, document_id: str) -> Mapping[str, object] | None:
        """读取一个固定文档；损坏或缺失以明确 provider 状态返回。"""

        item = self._document_by_id.get(document_id)
        if item is None:
            return None
        document, path = item
        summary = self._summary(document, path)
        if not bool(summary["available"]):
            return {
                **summary,
                "unavailable": {
                    "code": "document_unavailable",
                    "message": f"运行时文档不存在: {document.relative_path}",
                },
            }
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return {
                **summary,
                "unavailable": {
                    "code": "document_unavailable",
                    "message": f"运行时文档不存在: {document.relative_path}",
                },
            }
        if size > _MAX_DOCUMENT_BYTES:
            return {
                **summary,
                "unavailable": {
                    "code": "document_too_large",
                    "message": f"运行时文档超过 192 KiB: {document.relative_path}",
                },
            }
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {
                **summary,
                "unavailable": {
                    "code": "document_invalid_utf8",
                    "message": f"运行时文档不是合法 UTF-8: {document.relative_path}",
                },
            }
        return {**summary, "markdown": content}

    def list_skills(self) -> tuple[Mapping[str, object], ...] | None:
        """读取当前 generation 的技能 provider；缺失时保持局部 unavailable。"""

        service = self._skills
        return None if service is None else service.list_skills()

    def list_jobs(self) -> tuple[Mapping[str, object], ...] | None:
        """读取当前 generation 的 scheduler provider。"""

        service = self._scheduler
        return None if service is None else service.list_jobs()

    def get_job(self, job_id: str) -> Mapping[str, object] | None:
        """读取一个任务；缺 scheduler 用状态对象区别于未知任务。"""

        service = self._scheduler
        if service is None:
            return {
                "unavailable": {
                    "code": "scheduler_unavailable",
                    "message": "调度检查服务尚未绑定",
                }
            }
        return service.get_job(job_id)

    @staticmethod
    def _summary(document: _Document, path: Path) -> dict[str, object]:
        return {
            "id": document.id,
            "title": document.title,
            "relative_path": document.relative_path,
            "group": document.group,
            "description": document.description,
            "available": path.is_file(),
        }


__all__ = [
    "RuntimeInspectionProvider",
    "SCHEDULER_INSPECTION",
    "SKILL_INSPECTION",
]
