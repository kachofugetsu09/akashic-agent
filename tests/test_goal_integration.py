"""目标插件与默认回复链路的端到端组合测试。"""

from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from datetime import datetime, timezone

from agent.plugin_composition.config_input import save_config
from agent.plugins.manager import PluginManager
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.goal.store import decode_goal, goal_key
from plugins.ledger.contract import MESSAGE_CATALOG, ContentPart, Input, Output
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


_PROVIDER = r'''
from contextlib import asynccontextmanager
from types import SimpleNamespace

from agent.plugin_composition import ServiceKey
from plugins.models.contract import CHAT_MODELS, MODEL_CALL_STATS, MODEL_CATALOG
from plugins.models.contract import LLMResponse, ToolCall
from plugins.models.contract import (
    BoundModelDescriptor,
    CapabilitySources,
    ModelCapabilities,
)
from plugins.models.projection import (
    MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner,
)
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat, ModelsState
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.store import ModelsStore
from plugins.tools.api import Result
from agent.plugin_composition.tasks import TASKS
from plugins.ledger.contract import BINDINGS
from plugins.tools.plugin import TOOLS
from plugins.ledger.contract import ContentPart

api_version = 3
name = "test_provider"
version = "1.0.0"
inject = (TOOLS, TASKS, BINDINGS)

async def apply(ctx):
    calls = []
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    await ctx.effect(lambda: store.close, label="model-store")
    settings = ModelsState(store, context=ctx)
    await ctx.provide(MODEL_SETTINGS, settings.settings)
    await ctx.provide(MODEL_CATALOG, settings.catalog)

    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            calls.append(request)
            if len(calls) == 1:
                return LLMResponse(None, [ToolCall("call-1", "update_goal", {"status": "complete"})])
            return LLMResponse("finished")

    descriptor = BoundModelDescriptor(
        binding_id="fixture-model", plugin_snapshot_id="fixture", model_revision=0,
        model_id="fixture", connection_id="fixture", driver_id="fixture",
        driver_contract_version="1", auth_identity="fixture", model="fixture", role="agent",
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="fixture",
    )
    model = _BoundChat(descriptor, Driver(), store)

    async def use(use, choice):
        return model

    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            async with ctx.runtime_scope():
                yield SimpleNamespace(chat=lambda: model, use=use)

    class Target:
        idempotent = False
        async def prepare(self, args, source=None):
            return args
        async def invoke(self, key, args):
            return Result("success", (ContentPart("text", "ok"),))
        async def query(self, key):
            return None

    @asynccontextmanager
    async def open_tool(state):
        yield Target()

    await ctx.require(TOOLS).declare_group(ctx, always_on=True)
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(MODEL_CALL_STATS, store.read_call_stats)
    await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    await ctx.provide(MODEL_CONTENT, ContentOwner())
    await ctx.provide(MODEL_SELECTION, SelectionOwner())
    await ctx.provide(ServiceKey("fixture.calls"), calls)
'''


@asynccontextmanager
async def application(tmp_path):
    sources = tmp_path / "plugins"
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    save_config(workspace / "plugin-data/reply-builtin", {"max_steps": 8})
    for name in (
        "ledger", "commands", "sources", "content", "context", "tools",
        "conversation", "react", "turn_projection", "reply_program", "reply", "goal",
    ):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    provider = sources / "test_provider"
    provider.mkdir()
    (provider / "plugin.py").write_text(_PROVIDER)
    host = PluginManager(
        [sources], workspace=workspace,
        installed_cache_root=tmp_path / "home/cache",
    )
    try:
        await host.load_all()
        yield host
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_goal_command_runs_until_model_completes(tmp_path):
    async with application(tmp_path) as host:
        root = host.live_root
        assert root is not None
        message = ChannelInboundMessage(
            "test", "user", "room", "/goal ship the feature",
            datetime.now(timezone.utc), {},
        )
        accept = root.context.require(CHANNEL_INPUT)
        accepted = await accept("test:room", "u1", message)
        assert isinstance(accepted.body, Input)

        log = root.context.require(MESSAGE_CATALOG)._log
        deadline = asyncio.get_running_loop().time() + 10
        while asyncio.get_running_loop().time() < deadline:
            rows = log.reader("test:room").snapshot()
            if any(
                isinstance(row.body, Output) and row.body.finish == "complete"
                and any(
                    isinstance(part, ContentPart) and part.kind == "text"
                    and part.value == "finished"
                    for part in row.body.parts
                )
                for row in rows
            ):
                break
            await asyncio.sleep(0.05)
        rows = log.reader("test:room").snapshot()
        goal_inputs = [row for row in rows if row.author == "goal" and isinstance(row.body, Input)]
        assert goal_inputs, [type(row.body).__name__ for row in rows]
        record = log.owner("plugin:goal").read(goal_key("test:room"))
        assert record is not None
        goal = decode_goal("test:room", record.value)
        assert goal.status.value == "complete"
        assert goal.continuations == 1
