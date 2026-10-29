"""Authenticated dashboard routes; paths refer to the execution host, not Core."""
from fastapi import FastAPI, Query
from agent.plugin_composition import DashboardContext
from plugins.host_execution.contract import FILES
from plugins.working_directory.contract import WORKING_DIRECTORY

inject = (FILES, WORKING_DIRECTORY)


def register(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/markdown_reader/document")
    async def document(
        session: str = Query(min_length=1, max_length=512),
        path: str = Query(min_length=1, max_length=4096),
    ) -> dict[str, object]:
        target, _ = await context.require(WORKING_DIRECTORY).resolve_target(session, path)
        async with context.require(FILES).paths() as reader:
            result = await reader.read("markdown_read", target)
        return result.model_dump(exclude_none=True)

    @app.get("/api/dashboard/markdown_reader/files")
    async def files(
        session: str = Query(min_length=1, max_length=512),
        path: str = Query(default="", max_length=4096),
        after: str | None = Query(default=None, max_length=512),
    ) -> dict[str, object]:
        directories = context.require(WORKING_DIRECTORY)
        root = directories.snapshot(session).path
        if not root and not path:
            return {"status": "not_directory", "message": "尚未设置工作目录，请输入绝对路径打开文档。"}
        target, _ = await directories.resolve_target(session, path or root or ".")
        async with context.require(FILES).paths() as reader:
            result = await reader.read("markdown_list", target, after=after, limit=200)
        return {**result.model_dump(exclude_none=True), "root": root}
