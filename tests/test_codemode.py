from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
import shutil
import pytest

from agent.plugin_composition.config_input import save_config
from agent.plugins.bundles import set_plugin_choice
from agent.plugins.manager import PluginManager
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    CallRef,
    ContentPart,
    Input,
    Output,
    ToolCall,
    ToolResult,
)
from plugins.models.contract import ToolCall as ModelToolCall
from plugins.tools.contract import TOOL_PROGRAM_V2, TOOLS, Result
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def _write_mock_phone_plugin(sources: Path) -> None:
    plugin = sources / "mock_phone"
    plugin.mkdir()
    (plugin / "plugin.py").write_text(
        '''
from contextlib import asynccontextmanager
from plugins.ledger.contract import ContentPart, json_value
from plugins.tools.contract import TOOLS, Result

api_version = 3
name = "mock_phone"
version = "1.0.0"
desc = "模拟后台手机控制"
inject = (TOOLS,)

async def apply(ctx):
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=False, description="模拟手机控制")
    state = {"step": 0, "ref": "snap-1", "text": ""}

    class ObserveTool:
        idempotent = True
        async def prepare(self, arguments, source=None):
            if source is None:
                return "缺少 CallSource"
            return dict(json_value(arguments))
        async def invoke(self, key, arguments):
            return Result("success", (ContentPart(
                "text",
                f"Snapshot: snapshot_ref={state['ref']}\\n[1] EditText: '输入框' [click,edit] center=(200,300)",
            ),))
        async def query(self, key):
            return None

    class ActionTool:
        idempotent = False
        async def prepare(self, arguments, source=None):
            if source is None:
                return "缺少 CallSource"
            return dict(json_value(arguments))
        async def invoke(self, key, arguments):
            act = arguments.get("action")
            ref = arguments.get("snapshot_ref")
            if ref != state["ref"]:
                return Result("error", (ContentPart("text", f"PHONE_SNAPSHOT_STALE: {ref} != {state['ref']}"),))
            state["step"] += 1
            state["ref"] = f"snap-{state['step'] + 1}"
            if act == "type_text":
                state["text"] = str(arguments.get("text") or "")
            return Result("success", (ContentPart(
                "text",
                f"done={act} text={state['text']} next_ref={state['ref']}\\nSnapshot: snapshot_ref={state['ref']}",
            ),))
        async def query(self, key):
            return None

    @asynccontextmanager
    async def open_obs(_state):
        yield ObserveTool()

    @asynccontextmanager
    async def open_act(_state):
        yield ActionTool()

    await catalog.register(
        ctx,
        name="phone_observe",
        description="观察模拟手机屏幕",
        parameters={"type": "object", "properties": {}},
        open=open_obs,
        idempotent=True,
        parallel=False,
    )
    await catalog.register(
        ctx,
        name="phone_action",
        description="执行模拟手机动作",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "node_id": {"type": "integer"},
                "snapshot_ref": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["action"],
        },
        open=open_act,
        idempotent=False,
        parallel=False,
    )
''',
        encoding="utf-8",
    )


def _copy_core_plugins(sources: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "plugins"
    for name in (
        "ledger",
        "commands",
        "sources",
        "content",
        "context",
        "tools",
        "conversation",
        "react",
        "turn_projection",
        "reply_program",
        "models",
        "reply",
        "working_directory",
        "tool_search",
        "codemode",
    ):
        shutil.copytree(
            root / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )


def _write_e2e_provider(sources: Path, script_code: str) -> None:
    provider = sources / "test_provider"
    provider.mkdir()
    (provider / "plugin.py").write_text(
        f'''
from contextlib import asynccontextmanager
from types import SimpleNamespace
from agent.plugin_composition import ServiceKey
from plugins.models.contract import (
    CHAT_MODELS, MODEL_CATALOG, LLMResponse, ToolCall,
    BoundModelDescriptor, CapabilitySources, ModelCapabilities,
)
from plugins.models.projection import MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat, ModelsState
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.store import ModelsStore

api_version = 3
name = "test_provider"
version = "1.0.0"
inject = ()

async def apply(ctx):
    calls = []
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    await ctx.effect(lambda: store.close, label="model-store")
    settings = ModelsState(store, context=ctx)
    await ctx.provide(MODEL_SETTINGS, settings.settings)
    await ctx.provide(MODEL_CATALOG, settings.catalog)
    code_arg = {script_code!r}
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            calls.append(request)
            if len(calls) == 1:
                return LLMResponse(None, [ToolCall("cm-1", "codemode", {{"code": code_arg}})])
            return LLMResponse("all done")
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
        independent_execution = execution
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    await ctx.provide(MODEL_CONTENT, ContentOwner())
    await ctx.provide(MODEL_SELECTION, SelectionOwner())
    await ctx.provide(ServiceKey("fixture.calls"), calls)
''',
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_codemode_full_reply_e2e_multistep_phone_and_sandbox_isolation(tmp_path):
    sources = tmp_path / "plugins"
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    save_config(workspace / "plugin-data/reply-builtin", {"max_output_tokens": 2048})
    _copy_core_plugins(sources)
    _write_mock_phone_plugin(sources)

    script_code = """
    const list = tools.list();
    if (!list.some(t => t.name === 'phone_observe')) throw new Error('missing phone_observe');
    if (typeof process !== 'undefined' || typeof require !== 'undefined') {
        throw new Error('sandbox leaked node globals');
    }
    let evalBlocked = false;
    try { new Function('return 1')(); } catch (e) { evalBlocked = true; }
    if (!evalBlocked) throw new Error('Function codegen was not blocked');

    const obs = await tools.phone_observe({});
    let ref = obs.text.match(/snapshot_ref=(\\S+)/)[1];
    const tapRes = await tools.phone_action({ action: 'tap', node_id: 1, snapshot_ref: ref });
    ref = tapRes.text.match(/snapshot_ref=(\\S+)/)[1];
    const typeRes = await tools.phone_action({ action: 'type_text', node_id: 1, snapshot_ref: ref, text: '10:36' });
    const denied = await tools.nonexistent_tool({});
    const rec = await tools.codemode({ code: 'return 1' });
    return {
        typedOk: typeRes.ok && typeRes.text.includes('text=10:36'),
        deniedOk: denied.ok,
        recOk: rec.ok,
    };
    """
    _write_e2e_provider(sources, script_code)
    for disabled in ("models", "working_directory"):
        set_plugin_choice(workspace, disabled, enabled=False)

    host = PluginManager(
        [sources],
        workspace=workspace,
        installed_cache_root=tmp_path / "home/cache",
    )
    await host.load_all()
    try:
        root = host.live_root
        assert root is not None
        log = root.context.require(MESSAGE_CATALOG)._log
        accept = root.context.require(CHANNEL_INPUT)
        accepted = await accept(
            "test:room",
            "u1",
            ChannelInboundMessage("test", "user", "room", "run codemode", datetime(2026, 11, 16, tzinfo=UTC), {}),
        )
        assert isinstance(accepted.body, Input)

        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader("test:room").snapshot()
                if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                    return rows

        rows = await asyncio.wait_for(completed(), 10)
        assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
        tool_result = rows[2].body
        assert isinstance(tool_result, ToolResult)
        assert tool_result.outcome == "success", tool_result.parts[0].value
        text = str(tool_result.parts[0].value)
        assert "calls=5" in text
        assert '"typedOk": true' in text
        assert '"deniedOk": false' in text
        assert '"recOk": false' in text
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_codemode_respects_max_calls_and_timeout(tmp_path):
    async def _run_case(sub: Path, code: str, *, max_calls: int, timeout_ms: int) -> ToolResult:
        sources = sub / "plugins"
        workspace = sub / "workspace"
        initialize_plugin_workspace(workspace)
        save_config(workspace / "plugin-data/reply-builtin", {"max_output_tokens": 2048})
        _copy_core_plugins(sources)
        _write_mock_phone_plugin(sources)
        provider = sources / "test_provider"
        provider.mkdir()
        (provider / "plugin.py").write_text(
            f'''
from contextlib import asynccontextmanager
from types import SimpleNamespace
from plugins.models.contract import (
    CHAT_MODELS, MODEL_CATALOG, LLMResponse, ToolCall,
    BoundModelDescriptor, CapabilitySources, ModelCapabilities,
)
from plugins.models.projection import MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat, ModelsState
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.store import ModelsStore

api_version = 3
name = "test_provider"
version = "1.0.0"
inject = ()

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
                return LLMResponse(None, [ToolCall("cm-1", "codemode", {{"code": {code!r}, "max_calls": {max_calls}, "timeout_ms": {timeout_ms}}})])
            return LLMResponse("done")
    descriptor = BoundModelDescriptor(
        binding_id="fixture-model", plugin_snapshot_id="fixture", model_revision=0,
        model_id="fixture", connection_id="fixture", driver_id="fixture",
        driver_contract_version="1", auth_identity="fixture", model="fixture", role="agent",
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="fixture",
    )
    model = _BoundChat(descriptor, Driver(), store)
    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            async with ctx.runtime_scope():
                yield SimpleNamespace(chat=lambda: model, use=lambda *_a: None)
        independent_execution = execution
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    await ctx.provide(MODEL_CONTENT, ContentOwner())
    await ctx.provide(MODEL_SELECTION, SelectionOwner())
''',
            encoding="utf-8",
        )
        for disabled in ("models", "working_directory"):
            set_plugin_choice(workspace, disabled, enabled=False)
        host = PluginManager([sources], workspace=workspace, installed_cache_root=sub / "home/cache")
        await host.load_all()
        try:
            root = host.live_root
            assert root is not None
            log = root.context.require(MESSAGE_CATALOG)._log
            await root.context.require(CHANNEL_INPUT)(
                "test:room", "u1",
                ChannelInboundMessage("test", "user", "room", "go", datetime(2026, 11, 16, tzinfo=UTC), {}),
            )

            async def completed():
                async for _ in log.catalog().follow():
                    rows = log.reader("test:room").snapshot()
                    if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                        return rows

            rows = await asyncio.wait_for(completed(), 10)
            tool_res = rows[2].body
            assert isinstance(tool_res, ToolResult)
            return tool_res
        finally:
            await host.terminate_all()

    res_cap = await _run_case(
        tmp_path / "cap",
        "await tools.phone_observe({}); await tools.phone_observe({}); await tools.phone_observe({});",
        max_calls=2,
        timeout_ms=10_000,
    )
    assert res_cap.outcome == "error"
    assert "Exceeded max_calls limit (2)" in str(res_cap.parts[0].value)

    res_timeout = await _run_case(
        tmp_path / "timeout",
        "await sleep(5000); return 1;",
        max_calls=10,
        timeout_ms=1000,
    )
    assert res_timeout.outcome == "error"
    assert "超时" in str(res_timeout.parts[0].value)
