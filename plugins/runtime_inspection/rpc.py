"""普通检查插件的值级 RPC；外部调用与 Web/Mobile 使用同一合同。"""
from pydantic import BaseModel, ConfigDict

from agent.plugin_composition.rpc import RpcMethod

from .inspection import RuntimeInspectionProvider


class EmptyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentParams(EmptyParams):
    document_id: str


class JobParams(EmptyParams):
    job_id: str


def _unavailable(code: str, message: str) -> dict[str, object]:
    return {"unavailable": {"code": code, "message": message}}


def rpc_methods(provider: RuntimeInspectionProvider) -> dict[str, RpcMethod]:
    """发布独立请求入口，具体模型和缺失语义由本插件拥有。"""
    async def documents(params: BaseModel) -> object:
        return {"items": [dict(row) for row in provider.list_documents()]}

    async def document(params: BaseModel) -> object:
        assert isinstance(params, DocumentParams)
        item = provider.get_document(params.document_id)
        return (dict(item) if item is not None else
                _unavailable("document_not_found", f"未知运行时文档: {params.document_id}"))

    async def jobs(params: BaseModel) -> object:
        rows = provider.list_jobs()
        return ({"items": [dict(row) for row in rows]} if rows is not None else
                _unavailable("scheduler_unavailable", "调度检查服务尚未绑定"))

    async def job(params: BaseModel) -> object:
        assert isinstance(params, JobParams)
        item = provider.get_job(params.job_id)
        return (dict(item) if item is not None else
                _unavailable("job_not_found", f"定时任务不存在: {params.job_id}"))

    async def skills(params: BaseModel) -> object:
        rows = provider.list_skills()
        return ({"items": [dict(row) for row in rows]} if rows is not None else
                _unavailable("skills_unavailable", "技能检查服务尚未绑定"))

    return {
        "inspection/documents.list": RpcMethod(EmptyParams, documents),
        "inspection/documents.get": RpcMethod(DocumentParams, document),
        "inspection/jobs.list": RpcMethod(EmptyParams, jobs),
        "inspection/jobs.get": RpcMethod(JobParams, job),
        "inspection/skills.list": RpcMethod(EmptyParams, skills),
    }
