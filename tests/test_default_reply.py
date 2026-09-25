import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
import shutil
import pytest
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from agent.plugin_composition.config_input import save_config
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.log import MessageLog
from session.message import Input, Output, ToolResult

@asynccontextmanager
async def live_root(host: PluginManager):
    """Read the one formal Root used by the installed reply consumer."""
    root = host.live_root
    assert root is not None
    yield root

@asynccontextmanager
async def application(tmp_path, *, replying, start=True, missing_tool=False, discovery=False, compaction=False,
                      output_tokens=4096, keep_recent_tokens=128, summary_padding=0, provider_effect_data=False,
                      updates=False, validation_passed=True, extra_sources=None):
    sources = tmp_path / "plugins"
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    for name in (
        "commands",
        "sources",
        "content",
        "context",
        "tools",
        "conversation",
        "react",
        "turn_projection",
        "reply_program",
        *(("reply", "tool_search") if replying else ()),
    ):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    if updates:
        from tests.support.delivery_sources import sources as delivery_sources
        delivery_sources(sources)
        shutil.copytree(
            Path(__file__).parents[1] / "plugins/delivery_policy",
            sources / "delivery_policy",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        for name in ("assets", "plugin_update"):
            shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
    if compaction:
        shutil.copytree(Path(__file__).parents[1] / "plugins/compaction", sources / "compaction",
                        ignore=shutil.ignore_patterns("__pycache__"))
        settings = tmp_path / "workspace/plugin-data/context-builtin"
        settings.parent.mkdir(parents=True, exist_ok=True)
        save_config(settings, {"summary_source": ["compaction", "compaction"]})
        module = sources / "compaction/plugin.py"
        module.write_text(module.read_text().replace('Field(default=20_000,', f'Field(default={keep_recent_tokens},'))
        reply = sources / 'reply/plugin.py'
        reply.write_text(reply.read_text().replace('Field(default=4096,', f'Field(default={output_tokens},'))
    if missing_tool:
        settings = tmp_path / "workspace/plugin-data/reply-builtin"
        settings.parent.mkdir(parents=True, exist_ok=True)
        save_config(settings, {"tools": ["gone"]})
    provider = sources / "test_provider"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from pathlib import Path
from agent.plugin_composition import CHAT_MODELS, ServiceKey
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ToolCall
from plugins.models.projection import MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat, ModelsState
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.store import ModelsStore
from plugins.tools.api import Result
from plugins.standard_tools.shell import ShellOwners, shell_cleanup
from agent.plugin_composition.tasks import TASKS
from agent.plugin_composition.bindings import BINDINGS
from plugins.tools.plugin import TOOLS
from session.message import ContentPart
api_version = 3
name = "test_provider"
version = "1.0.0"
inject = (TOOLS, TASKS, BINDINGS)
async def apply(ctx):
    calls = []
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    settings = ModelsState(store, context=ctx)
    await ctx.provide(MODEL_SETTINGS, settings.settings)
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            calls.append(request)
            if len(calls) == 1:
                return LLMResponse(None, [ToolCall("provider-call", "write_evidence", {})])
            return LLMResponse("finished")
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
            yield SimpleNamespace(chat=lambda role: model)
    class Target:
        idempotent = False
        async def prepare(self, args, source=None):
            return args
        async def invoke(self, key, args):
            with Path(EFFECT_PATH).open("a") as handle:
                handle.write("once\\n")
            return Result("success", (ContentPart("text", "written"),))
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open(state):
        yield Target()
    await ctx.require(TOOLS).declare_group(ctx, always_on=True)
    await ctx.require(TOOLS).register(ctx, name="write_evidence", description="record local test evidence",
        parameters={"type":"object"}, open=open)
    await ctx.provide(ServiceKey("tools.cleanup.v1"), partial(
        shell_cleanup, ctx, ShellOwners(ctx), ctx.require(TASKS).open(ctx),
    ))
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    await ctx.provide(MODEL_CONTENT, ContentOwner())
    await ctx.provide(MODEL_SELECTION, SelectionOwner())
    await ctx.provide(ServiceKey("fixture.calls"), calls)
'''.replace("EFFECT_PATH", repr(str(tmp_path / "effect.txt"))))
    if provider_effect_data:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace(
            repr(str(tmp_path / "effect.txt")), 'ctx.runtime.data_dir / "effect.txt"'))
    if updates and not validation_passed:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace(
            'return LLMResponse("finished")', 'raise RuntimeError("candidate execution failed")'))
    if discovery:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace(
            'if len(calls) == 1:',
            'if len(calls) == 1:\n                return LLMResponse(None, [ToolCall("search-call", "tool_search", {"query": "write_evidence"})])\n            if len(calls) == 2:').replace('declare_group(ctx, always_on=True)', 'declare_group(ctx, description="Write local evidence")').replace(
            'ToolCall("provider-call", "write_evidence", {})',
            'ToolCall("provider-call", "tool_call", {"name": "write_evidence", "arguments": {}})'))
    if compaction:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace('calls = []', 'calls = []\n    business = []').replace(
            'return 10', 'return len(str(messages)) // 4').replace('if len(calls) == 1:', '''if "[Source messages]" in str(request.messages):
                from plugins.compaction.message_summary import HEADINGS
                return LLMResponse("\\n".join(heading + "\\nPreserved facts." for heading in HEADINGS))
            business.append(request)
            if len(business) == 1:''').replace("Preserved facts.", "Preserved facts." + "z" * summary_padding))
    if extra_sources is not None:
        extra_sources(sources)
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore

    log = MessageLog(tmp_path / "sessions.db")
    artifact_store = ArtifactStore(tmp_path / "sessions.db")
    artifacts = ChannelAttachmentArtifactStore(
        workspace=workspace, metadata_store=artifact_store
    )
    event_bus = EventBus()
    host = PluginManager(
        [sources],
        event_bus=event_bus,
        workspace=workspace,
        installed_cache_root=tmp_path / "home/cache",
        message_log=log,
        channel_attachment_store=artifacts,
    )
    try:
        await host.load_all()
        if start:
            await host.start_runtime()
        yield log, host
    finally:
        termination_error = None
        try:
            await host.terminate_all()
        except BaseException as error:
            termination_error = error
        cleanup_errors = []
        for cleanup in (log.close, artifact_store.close):
            try:
                cleanup()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            await event_bus.aclose()
        except BaseException as error:
            cleanup_errors.append(error)
        if termination_error is not None:
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Manager termination and fixture cleanup failed",
                    [termination_error, *cleanup_errors],
                ) from termination_error
            raise termination_error
        if cleanup_errors:
            raise BaseExceptionGroup("fixture cleanup failed", cleanup_errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("replying", [False, True])
async def test_installed_default_reply_is_an_independent_log_consumer(tmp_path, replying):
    from agent.plugin_composition import ServiceKey
    async with application(tmp_path, replying=replying) as (log, host):
        message = ChannelInboundMessage("test", "user", "room", "do the work",
                                        datetime(2026, 9, 5, tzinfo=UTC), {})
        async with live_root(host) as root:
            accept = root.context.require(CHANNEL_INPUT)
            accepted = await accept("test:room", "u1", message)
        assert isinstance(accepted.body, Input)
        if replying:
            async def completed():
                async for _ in log.catalog().follow():
                    rows = log.reader("test:room").snapshot()
                    if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                        return rows
            rows = await asyncio.wait_for(completed(), 5)
            assert rows is not None
            assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
            assert (tmp_path / "effect.txt").read_text() == "once\n"
        else:
            async with live_root(host) as root:
                calls = root.context.require(ServiceKey("fixture.calls"))
                assert calls == []
                assert all("[Source messages]" in str(call.messages) for call in calls)
            assert log.reader("test:room").snapshot() == (accepted,)
            assert not (tmp_path / "effect.txt").exists()
