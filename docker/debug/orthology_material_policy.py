"""材料用途独立于名称；验证真实目录、授权、旧恢复入口和贡献者排空。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.plugin_composition import CompositionRoot, ServiceKey
from plugins.ledger.contract import unavailable
from agent.plugin_composition.model import PluginRuntime
from plugins.context.contract import MATERIALS_V4
from plugins.context import plugin as context
from plugins.context.materials import ContextMaterials
from plugins.reply_program import plugin as reply
from plugins.reply_program.contract import (
    REPLY_EXECUTE_V4,
)

REPLY_EXECUTE = ServiceKey[object]("reply.execute.v1")
REPLY_EXECUTE_V2 = ServiceKey[object]("reply.execute.v2")
REPLY_EXECUTE_V3 = ServiceKey[object]("reply.execute.v3")


async def check(workspace: Path) -> None:
    root = CompositionRoot("material-policy-scenario")
    def runtime(name, config=None):
        return PluginRuntime(name, name, workspace, workspace / name, workspace, config or {})
    def provider(name, kind, *, prompt=False):
        async def apply(ctx):
            async def prepare(_snapshot, _source):
                return {"system_prompt": name} if prompt else {"reminders": ({"name": name, "text": name, "priority": 0},)}
            await ctx.require(MATERIALS_V4).register(ctx, name=name, kind=kind, prepare=prepare, prompt=prompt)
        return apply
    try:
        await root.mount(context.apply, name="context", runtime=runtime("context", {
            "prompt_sources": {"biography": "archive-owner", "instructions": "rules-owner"},
        }))
        await root.mount(provider("instructions", "context", prompt=True), name="rules", inject=(MATERIALS_V4,), runtime=runtime("rules-owner"))
        profile = await root.mount(provider("biography", "profile", prompt=True), name="profile", inject=(MATERIALS_V4,), runtime=runtime("archive-owner"))
        await root.mount(provider("retrieved-events", "recall"), name="recall", inject=(MATERIALS_V4,), runtime=runtime("different-memory-engine"))
        materials = root.context.require(MATERIALS_V4)
        async with materials.bind(exclude_kinds=frozenset({"recall", "profile"})) as view:
            value = await view.prepare((), "scheduler")
            assert value["system_prompt"] == "instructions" and not value["reminders"]
        async with materials.bind(exclude_kinds=frozenset({"recall"})) as view:
            value = await view.prepare((), "subagent")
            assert "biography" in value["system_prompt"] and not value["reminders"]
        async with materials.bind(exclude_kinds=frozenset()) as view:
            value = await view.prepare((), "conversation")
            assert value["reminders"][0]["name"] == "retrieved-events"

        # 当前 view 持有实际贡献者；热卸载不能关闭正在使用的 provider。
        async with materials.bind(exclude_kinds=frozenset()) as view:
            disposing = asyncio.create_task(profile.dispose())
            barrier = asyncio.Event()
            asyncio.get_running_loop().call_soon(barrier.set)
            await barrier.wait()
            assert not disposing.done()
        await disposing
        async with materials.bind(exclude_kinds=frozenset({"profile", "recall"})) as view:
            assert (await view.prepare((), "scheduler"))["system_prompt"] == "instructions"
        assert root.context.get(ServiceKey[object]("context.materials.v3")) is None
        invalid = await root.mount(provider("old-provider", None), name="unclassified",
                                   inject=(MATERIALS_V4,), runtime=runtime("old-owner"))
        assert isinstance(invalid.error, ValueError) and "kind" in str(invalid.error)
        assert "old-provider" not in materials._sources
        await invalid.dispose()
        assert "akasha" not in materials._sources and "markdown_memory" not in materials._sources
        # 只有新材料接口时，真实 Reply 的新入口已就绪，旧入口局部缺席。
        isolated = CompositionRoot("new-material-api-only")
        try:
            new_materials = ContextMaterials(isolated.context, prompt_sources={})
            for key in reply.inject:
                await isolated.context.provide(key, new_materials if key == MATERIALS_V4 else unavailable)
            mounted = await isolated.mount(reply.apply, name="reply", inject=reply.inject, runtime=runtime("reply"))
            assert mounted.error is None
            assert isolated.context.get(REPLY_EXECUTE_V3) is None
            assert isolated.context.get(REPLY_EXECUTE_V4) is not None
            assert isolated.context.get(REPLY_EXECUTE_V2) is None
            assert isolated.context.get(REPLY_EXECUTE) is None
        finally:
            await isolated.dispose()
    finally:
        await root.dispose()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-materials-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: renamed providers, kind selection, grant independence, drain, retired API absent, unknown kind rejected")
