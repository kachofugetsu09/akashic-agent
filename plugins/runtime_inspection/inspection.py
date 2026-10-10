"""普通 runtime_inspection 插件拥有的中性检查 provider。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from agent.plugin_composition import Context, Effect

from plugins.runtime_inspection.contract import Document
from plugins.scheduler.contract import SCHEDULER_INSPECTION
from plugins.standard_tools.contract import SkillReader

_MAX_DOCUMENT_BYTES = 192 * 1024


class RuntimeInspectionProvider:
    """汇总 owner 发布的文档，并组合可选 scheduler/skill 只读输入。"""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._documents: dict[str, Document] = {}
        self._skills: SkillReader | None = None

    async def register(self, ctx: Context, document: Document) -> Effect:
        """注册随贡献者释放；读取入口保护真实文档 owner 的寿命。"""
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("文档注册不能跨 Root")
        if not document.id or not callable(document.read):
            raise ValueError("文档必须有明确 ID 和读取口")
        owned = replace(document, read=ctx.entrypoint(document.read))

        def setup():
            if document.id in self._documents:
                raise ValueError(f"文档 ID 已有 owner: {document.id}")
            self._documents[document.id] = owned
            return lambda: self._documents.pop(document.id)

        return await ctx.effect(setup, label=f"document:{document.id}")

    def bind_skills(self, service: SkillReader) -> None:
        self._skills = service

    def unbind_skills(self, service: SkillReader) -> None:
        if self._skills is service:
            self._skills = None

    def list_documents(self) -> tuple[Mapping[str, object], ...]:
        """只列出实际 owner 的贡献；零字节读取只检查当前可用性。"""
        rows: list[Mapping[str, object]] = []
        for document in sorted(self._documents.values(), key=lambda item: (item.order, item.id)):
            try:
                document.read(0)
            except (FileNotFoundError, IsADirectoryError):
                available = False
            else:
                available = True
            rows.append(self._summary(document, available))
        return tuple(rows)

    def get_document(self, document_id: str) -> Mapping[str, object] | None:
        """只请求有界字节；文件缺失、过大和解码失败保留原外部状态。"""
        document = self._documents.get(document_id)
        if document is None:
            return None
        try:
            payload = document.read(_MAX_DOCUMENT_BYTES + 1)
        except (FileNotFoundError, IsADirectoryError):
            return {**self._summary(document, False), "unavailable": {
                "code": "document_unavailable",
                "message": f"运行时文档不存在: {document.relative_path}",
            }}
        summary = self._summary(document, True)
        if len(payload) > _MAX_DOCUMENT_BYTES:
            return {**summary, "unavailable": {
                "code": "document_too_large",
                "message": f"运行时文档超过 192 KiB: {document.relative_path}",
            }}
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            return {**summary, "unavailable": {
                "code": "document_invalid_utf8",
                "message": f"运行时文档不是合法 UTF-8: {document.relative_path}",
            }}
        return {**summary, "markdown": content}

    async def list_skills(self) -> tuple[Mapping[str, object], ...] | None:
        """读取当前 generation 的技能 provider；缺失时保持局部 unavailable。"""

        service = self._skills
        return None if service is None else await service.list_skills()

    async def list_skill_sources(self) -> tuple[Mapping[str, object], ...] | None:
        """读取本地来源探测结果；缺失与不可读也原样透出。"""

        service = self._skills
        return None if service is None else await service.list_sources()

    async def list_jobs(self) -> tuple[Mapping[str, object], ...] | None:
        """借用本次实际 provider，读取排空后才允许其卸载。"""
        with self._ctx.borrow(SCHEDULER_INSPECTION) as service:
            return None if service is None else await service.list_jobs()

    async def get_job(self, job_id: str) -> Mapping[str, object] | None:
        """读取一个任务；缺 scheduler 用状态对象区别于未知任务。"""
        with self._ctx.borrow(SCHEDULER_INSPECTION) as service:
            if service is None:
                return {
                    "unavailable": {
                        "code": "scheduler_unavailable",
                        "message": "调度检查服务尚未绑定",
                    }
                }
            return await service.get_job(job_id)

    @staticmethod
    def _summary(document: Document, available: bool) -> dict[str, object]:
        return {
            "id": document.id,
            "title": document.title,
            "relative_path": document.relative_path,
            "group": document.group,
            "description": document.description,
            "available": available,
        }
