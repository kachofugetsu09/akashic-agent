from plugins.context.api import check_summary as _model_summary_check
import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, nullcontext
from typing import Any
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, ModelCapabilities, ToolCall as ModelToolCall
from agent.plugin_composition.tasks import Task, Tasks
from plugins.content.plugin import _decode_text, check_text
from plugins.context.api import Materials, check_summary, material_data
from plugins.context.plugin import ContextBuilder
from plugins.sources.session import SourceSession as Conversation
from plugins.models.content import render_content
from plugins.models.projection import MessageProjection, check_facts, check_tool_rejection
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.react.plugin import react
from plugins.tools.execution import ToolExecution, MessageReply, Result
from plugins.tools.abandon import follow_abandon, reject_start
from plugins.tools.menu import NativePresentation, ToolMenu, ToolCallDecode
from session.log import MessageLog
from session.message import CallRef, Control, Input, Output, ToolCall, ToolResult

@asynccontextmanager
async def runtime(tmp_path, complete, invoke, *, max_steps=4, authorize_hook=None,
                  reducer=None, material_source=None, estimate=None, preview_state=None, terminal_tools=frozenset(),
                  state_owner=None, model_max_attempts=1):
    log = MessageLog(tmp_path / "sessions.db")
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    tasks = Tasks()
    descriptor = BoundModelDescriptor(
        binding_id="model", plugin_snapshot_id="snapshot", model_revision=0,
        model_id="model", connection_id="connection", driver_id="driver",
        driver_contract_version="1", auth_identity="test", model="test", role="agent",
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="test",
    )
    class Driver:
        async def complete(self, request):
            return await complete(request)
        def estimate_context_tokens(self, messages, tools=()):
            return 100 if estimate is None else estimate(messages, tools)
        def estimate_appended_message_tokens(self, messages):
            return 0
        max_tool_schemas = None
    model = _BoundChat(descriptor, Driver(), store, max_attempts=model_max_attempts)
    log.save_binding("tool", {"target": "test-file-effect"})
    def writer(body, call_ref=None):
        return log.writer(
            "s", author="test", source="conversation", body_types=(body,),
            content={"text": check_text, "model.facts": check_facts, "model.tool_rejection": check_tool_rejection, "context.summary": check_summary} if body is Output else {"text": check_text},
            call_ref=call_ref, check_call=lambda call: None,
        )
    class Target:
        idempotent = False
        async def prepare(self, arguments, source=None):
            return arguments
        async def invoke(self, key, arguments):
            return await invoke(key, arguments)
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open_tool(binding):
        assert binding == "tool"
        yield Target()
    async def authorize(binding, arguments):
        if authorize_hook is not None:
            await authorize_hook()
        return {"decision": "allowed"}
    execution = ToolExecution(
        log.owner("tools"), tasks, open_tool, authorize, task_key="tools",
    )
    class Menu(ToolMenu):
        def __init__(self, task: Task) -> None:
            self.task = task

        @property
        def schemas(self) -> tuple[Mapping[str, Any], ...]:
            return ({"type": "function", "function": {
                "name": "example", "parameters": {"type": "object"},
            }},)

        def decode(self, call: ModelToolCall):
            if call.name == "tool_call":
                assert call.arguments["name"] == "example"
                return ToolCallDecode("tool", call.arguments["arguments"])
            decoded = NativePresentation({"example": {}}).decode(call)
            if isinstance(decoded, str):
                return ToolCallDecode(None, {}, {"name": call.name, "arguments": call.arguments, "error": decoded})
            return ToolCallDecode("tool", decoded[1])

        def name(self, binding_id: str) -> str:
            assert binding_id == "tool"
            return "example"

        async def execute(self, call: CallRef) -> Result:
            return await execution.execute_call(MessageReply(
                "result:" + call.message_id + ":" + str(call.part_index), call,
                log.reader("s"), writer(ToolResult, call), self.check_start,
            ))

        async def settle_abandoned(self, call: CallRef) -> Result:
            reply = MessageReply(
                "result:" + call.message_id + ":" + str(call.part_index), call,
                log.reader("s"), writer(ToolResult, call), self.check_start,
            )
            try:
                return await execution.settle_abandoned(reply)
            finally:
                reply.writer.expire()

        def check_start(self) -> None:
            if not self.task.active:
                raise asyncio.CancelledError

        def check_call(self, call: ToolCall) -> None:
            raise AssertionError(f"controlled menu unexpectedly checked {call}")

    class Content:
        def check_metadata(self, metadata):
            assert not metadata
        prompts = ()
        checks = {}
        async def decode(self, text, references=()):
            return await _decode_text(text, (), references)
    projection = MessageProjection(model, check_summary=_model_summary_check, source="conversation",
                                   render_content=lambda p: render_content(p, artifacts={}),
                                   tool_name=lambda binding: "example", read_call=store.read_call)
    async def materials(snapshot):
        return material_data(Materials("system")) if material_source is None else await material_source(snapshot)
    async def run(task, reader, source):
        output = writer(Output)
        assert output.source == source
        task.on_close(output.expire)
        with preview_state.open(task, reader.session_id, source) if preview_state is not None else nullcontext(None) as preview:
            return await react(reader, output, model=model, context=ContextBuilder(),
                               projection=projection, materials=materials, content=Content(), tools=Menu(task),
                               max_output_tokens=100, max_steps=max_steps, reduce=reducer, preview=preview, terminal_tools=terminal_tools,
                               state=None if state_owner is None else log.owner(state_owner))
    conversation = Conversation(reader=log.reader("s"), inputs=writer(Input), controls=writer(Control),
                                tasks=tasks)
    @asynccontextmanager
    async def interrupted_reply(reader, source, ref):
        reply = MessageReply("result:" + ref.message_id + ":" + str(ref.part_index), ref,
                             reader, writer(ToolResult, ref), reject_start)
        try:
            yield reply
        finally:
            reply.writer.expire()
    watcher = asyncio.create_task(follow_abandon(
        log.catalog(), log.owner("tools"), tasks, interrupted_reply, task_key="tools",
        report_incident=lambda kind, message: None,
    ))
    try:
        yield conversation, log, store, run
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await tasks.close()
        store.close()
        log.close()
