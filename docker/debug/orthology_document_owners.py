"""真实文档 owner 与诊断目录的安装、卸载和有界读取验证。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.channel_io import unavailable
from agent.plugin_composition.messages import MESSAGE_CATALOG
from session.log import MessageCatalog, MessageLog
from agent.plugin_composition.model import FiberState, PluginRuntime
from agent.plugin_contracts.context import CONTEXT, MATERIALS_V4 as MATERIALS
from agent.plugin_contracts.inspection import DOCUMENTS, Document
from plugins.context import plugin as context
from plugins.markdown_memory import plugin as memory
from plugins.prompt import plugin as prompt
from plugins.runtime_inspection import plugin as inspection


async def check(workspace: Path) -> None:
    root = CompositionRoot("document-owner-scenario")
    log = MessageLog(workspace / "sessions.db")
    def runtime(name, files=(), config=None):
        return PluginRuntime(name, name, workspace, workspace / name, workspace, config or {}, workspace_files=files)
    try:
        await root.mount(context.apply, name="context", runtime=runtime("context", config={
            "prompt_sources": {"default_prompt": "prompt", "markdown_memory": "markdown_memory"},
        }))
        for key in memory.inject:
            if key not in (CONTEXT, MATERIALS, MESSAGE_CATALOG):
                await root.context.provide(key, unavailable)
        await root.context.provide(MESSAGE_CATALOG, MessageCatalog(log))
        persona = await root.mount(prompt.apply, name="prompt", inject=prompt.inject,
                                   runtime=runtime("prompt", prompt.workspace_files))
        profiles = await root.mount(memory.apply, name="memory", inject=memory.inject,
                                    runtime=runtime("markdown_memory", memory.workspace_files))
        assert persona.state is FiberState.ACTIVE and profiles.state is FiberState.ACTIVE
        observer = await root.mount(inspection.apply, name="inspection", runtime=runtime("inspection"))
        documents = root.context.require(DOCUMENTS)
        assert [row["id"] for row in documents.list_documents()] == ["memory", "self", "veda"]
        path = workspace / "memory/MEMORY.md"
        path.rename(workspace / "original-memory.md")
        assert documents.get_document("memory")["unavailable"]["code"] == "document_unavailable"
        assert documents.get_document("veda")["markdown"]
        path = workspace / "memory/MEMORY.md"
        path.write_text("owner text\n")
        assert documents.get_document("memory")["markdown"] == "owner text\n"
        path.write_bytes(b"\xff")
        assert documents.get_document("memory")["unavailable"]["code"] == "document_invalid_utf8"
        path.write_bytes(b"x" * (192 * 1024 + 1))
        assert documents.get_document("memory")["unavailable"]["code"] == "document_too_large"
        before = path.read_bytes()
        await profiles.dispose()
        assert documents.get_document("memory") is None and path.read_bytes() == before
        assert [row["id"] for row in documents.list_documents()] == ["veda"]

        # 新增异名 owner 不修改诊断目录源码；每次读取都经过其入口。
        sizes = []
        async def custom(ctx):
            def read(limit):
                sizes.append(limit)
                return b"other"[:limit]
            await ctx.require(DOCUMENTS).register(ctx, Document("other", "Other", "owner/file", "other", "custom", read))
        other = await root.mount(custom, name="other", inject=(DOCUMENTS,))
        assert documents.get_document("other")["markdown"] == "other"
        assert sizes == [192 * 1024 + 1]
        await other.dispose()
        assert documents.get_document("other") is None
        await observer.dispose()
        assert persona.state is FiberState.ACTIVE
        assert (workspace / "memory/VEDA.md").is_file()
        observer = await root.mount(inspection.apply, name="inspection-again", runtime=runtime("inspection"))
        assert [row["id"] for row in root.context.require(DOCUMENTS).list_documents()] == ["veda"]
    finally:
        await root.dispose()
        log.close()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-documents-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: real owners, optional inspection, missing/UTF-8/size, alternate owner, reload; no data deletion")
