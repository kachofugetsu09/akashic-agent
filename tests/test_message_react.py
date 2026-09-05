import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from agent.plugin_composition.models import (
    BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities,
    ModelRole, ToolCall as ModelToolCall,
)
from agent.plugin_composition.tasks import Tasks
from plugins.content.plugin import _decode_text, check_text
from plugins.context.api import Materials, check_summary
from plugins.context.plugin import ContextBuilder
from plugins.conversation.source import Conversation, needs_reply
from plugins.models.content import render_content
from plugins.models.projection import MessageProjection, check_facts
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.react.plugin import react, UnknownToolEffect, StepLimit
from plugins.tools.execution import ToolExecution, MessageReply, Result
from session.log import MessageConflict, MessageLog
from session.message import Input, Output, ContentPart, ToolResult, CallRef, Control


@asynccontextmanager
async def runtime(tmp_path, complete, invoke, *, max_steps=4, authorize_hook=None,
                  reducer=None, material_source=None, estimate=None):
    log = MessageLog(tmp_path / "sessions.db")
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    tasks = Tasks()
    descriptor = BoundModelDescriptor(
        binding_id="model", plugin_snapshot_id="snapshot", model_revision=0,
        model_id="model", connection_id="connection", driver_id="driver",
        driver_contract_version="1", auth_identity="test", model="test", role=ModelRole.AGENT,
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="test",
    )
    class Driver:
        async def complete(self, request):
            return await complete(request)
        def estimate_context_tokens(self, messages, tools):
            return 100 if estimate is None else estimate(messages, tools)
        max_tool_schemas = None
    model = _BoundChat(descriptor, Driver(), store)
    log.save_binding("tool", {"target": "test-file-effect"})
    def writer(body, call_ref=None):
        return log.writer(
            "s", author="test", source="conversation", body_types=(body,),
            content={"text": check_text, "model.facts": check_facts, "context.summary": check_summary} if body is Output else {"text": check_text},
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
    execution = ToolExecution(log.owner("tools"), tasks, open_tool, authorize, task_key="tools")
    class Menu:
        def __init__(self, task):
            self.task = task
        schemas = ({"type": "function", "function": {"name": "example", "parameters": {"type": "object"}}},)
        def bind(self, name):
            assert name == "example"
            return "tool"
        async def execute(self, ref):
            return await execution.execute_call(MessageReply(
                "result:" + ref.message_id + ":" + str(ref.part_index), ref,
                log.reader("s"), writer(ToolResult, ref), self.check_start,
            ))
        def check_start(self):
            if not self.task.active:
                raise asyncio.CancelledError
    class Content:
        async def decode(self, text, references=()):
            return await _decode_text(text, (), references)
    projection = MessageProjection(model, source="conversation",
                                   render_content=lambda p: render_content(p, artifacts={}),
                                   tool_name=lambda binding: "example", read_call=store.read_call)
    async def materials(snapshot):
        return Materials("system") if material_source is None else await material_source(snapshot)
    async def run(task, reader, source):
        output = writer(Output)
        assert output.source == source
        task.on_close(output.expire)
        return await react(reader, output, model=model, context=ContextBuilder(),
                           projection=projection, materials=materials, content=Content(), tools=Menu(task),
                           max_output_tokens=100, max_steps=max_steps, reduce=reducer)
    conversation = Conversation(reader=log.reader("s"), inputs=writer(Input), controls=writer(Control),
                                tasks=tasks)
    try:
        yield conversation, log, store, run
    finally:
        await tasks.close()
        log.close()


@pytest.mark.asyncio
async def test_react_commits_each_real_model_call_tool_request_and_result(tmp_path):
    requests = []
    async def complete(request):
        requests.append(request)
        if len(requests) == 1:
            return LLMResponse("working", [ModelToolCall("provider-call", "example", {"value": "written"})])
        assert [row["role"] for row in request.messages] == ["system", "user", "assistant", "tool"]
        assert request.messages[-1]["tool_call_id"] == "provider-call"
        return LLMResponse("finished")
    async def invoke(key, arguments):
        (tmp_path / "effect.txt").write_text(arguments["value"])
        return Result("success", (ContentPart("text", "real result"),))
    async with runtime(tmp_path, complete, invoke) as (conversation, log, store, run):
        await conversation.accept("u1", Input((ContentPart("text", "input"),)))
        result = await (await conversation.start(run)).join()
        messages = log.reader("s").snapshot()
        assert [type(m.body) for m in messages] == [Input, Output, ToolResult, Output]
        assert result == messages[-1]
        assert (tmp_path / "effect.txt").read_text() == "written"
        for message in (messages[1], messages[-1]):
            facts = message.body.parts[-1].value
            assert store.read_call(facts["call_record_id"])["state"] == "success"
        assert await conversation.start(run) is None


@pytest.mark.asyncio
async def test_new_input_during_tool_drains_real_effect_before_next_model(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    requests = []
    effects = []
    async def complete(request):
        requests.append(request)
        if len(requests) == 1:
            return LLMResponse(None, [ModelToolCall("call", "example", {})])
        assert release.is_set()
        assert "u1" in str(request.messages) and "u2" in str(request.messages)
        return LLMResponse("all inputs answered")
    async def invoke(key, arguments):
        effects.append(key)
        entered.set()
        await release.wait()
        return Result("success", (ContentPart("text", "settled"),))
    async with runtime(tmp_path, complete, invoke) as (conversation, log, store, run):
        await conversation.accept("u1", Input((ContentPart("text", "u1"),)))
        old = await conversation.start(run)
        await entered.wait()
        await conversation.accept("u2", Input((ContentPart("text", "u2"),)))
        replacement = asyncio.create_task(conversation.start(run))
        assert len(requests) == 1
        release.set()
        latest = await replacement
        await latest.join()
        assert len(effects) == 1
        assert [type(m.body) for m in log.reader("s").snapshot()] == [Input, Output, Input, ToolResult, Output]
        with pytest.raises(asyncio.CancelledError):
            await old.join()


@pytest.mark.asyncio
async def test_stale_model_output_has_no_tool_effect_and_inputs_stay_open(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    async def complete(request):
        nonlocal calls
        calls += 1
        if calls > 1:
            return LLMResponse("new reply")
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # 模拟不能及时取消的 provider；提交 grant 仍是最终边界。
            await release.wait()
        return LLMResponse(None, [ModelToolCall("call", "example", {})])
    async def invoke(key, arguments):
        pytest.fail("stale draft must never execute")
    async with runtime(tmp_path, complete, invoke) as (conversation, log, store, run):
        await conversation.accept("u1", Input(()))
        old = await conversation.start(run)
        await entered.wait()
        await conversation.accept("u2", Input(()))
        replacement = asyncio.create_task(conversation.start(run))
        release.set()
        with pytest.raises(RuntimeError, match="writer 已失效"):
            await old.join()
        await (await replacement).join()
        messages = log.reader("s").snapshot()
        assert [m.message_id for m in messages[:2]] == ["u1", "u2"]
        assert len(messages) == 3 and messages[-1].body.finish == "complete"
        assert not needs_reply(messages, "conversation")


@pytest.mark.asyncio
async def test_unknown_tool_result_pauses_and_never_reexecutes_after_new_input(tmp_path):
    effects = []
    requests = []
    async def complete(request):
        requests.append(request)
        return LLMResponse(None, [ModelToolCall("call", "example", {})])
    async def invoke(key, arguments):
        effects.append(key)
        return Result("unknown", ())
    async with runtime(tmp_path, complete, invoke) as (conversation, log, store, run):
        await conversation.accept("u1", Input(()))
        with pytest.raises(UnknownToolEffect):
            await (await conversation.start(run)).join()
        assert await conversation.start(run) is None
        await conversation.accept("u2", Input(()))
        with pytest.raises(UnknownToolEffect):
            await (await conversation.start(run)).join()
        assert len(effects) == len(requests) == 1
        assert isinstance(log.reader("s").snapshot()[-1].body, Control)


@pytest.mark.asyncio
async def test_request_limit_settles_last_committed_tool_then_pauses(tmp_path):
    effects = []
    async def complete(request):
        return LLMResponse(None, [ModelToolCall("call", "example", {})])
    async def invoke(key, arguments):
        effects.append(key)
        return Result("success", ())
    async with runtime(tmp_path, complete, invoke, max_steps=1) as (conversation, log, store, run):
        await conversation.accept("u1", Input(()))
        with pytest.raises(StepLimit):
            await (await conversation.start(run)).join()
        assert len(effects) == 1
        assert [type(m.body) for m in log.reader("s").snapshot()] == [Input, Output, ToolResult, Control]
        assert await conversation.start(run) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wake", ["resume", "input"])
async def test_pause_during_authorization_does_not_start_effect_resume_uses_original_call(tmp_path, wake):
    authorizing = asyncio.Event()
    released = asyncio.Event()
    requests = []
    effects = []
    async def authorize():
        authorizing.set()
        await released.wait()
    async def complete(request):
        requests.append(request)
        if len(requests) == 1:
            return LLMResponse(None, [ModelToolCall("call", "example", {})])
        return LLMResponse("resumed")
    async def invoke(key, arguments):
        effects.append(key)
        return Result("success", ())
    async with runtime(tmp_path, complete, invoke, authorize_hook=authorize) as (conversation, log, store, run):
        await conversation.accept("u1", Input(()))
        old = await conversation.start(run)
        await authorizing.wait()
        call = log.reader("s").snapshot()[-1]
        stopping = asyncio.create_task(conversation.control(
            "pause", Control("pause", call.seq), expected_head=call.seq, handle=old.handle))
        async def pause_committed():
            async for _ in log.catalog().follow():
                if log.reader("s").get("pause") is not None:
                    return
        await pause_committed()
        assert not stopping.done()
        released.set()
        await stopping
        with pytest.raises(asyncio.CancelledError):
            await old.join()
        assert effects == []
        assert await conversation.start(run) is None
        assert not any(isinstance(m.body, ToolResult) for m in log.reader("s").snapshot())
        head = log.reader("s").head()
        if wake == "resume":
            await conversation.control("resume", Control("resume", call.seq), expected_head=head, handle=None)
        else:
            await conversation.accept("u2", Input(()))
        await (await conversation.start(run)).join()
        assert len(effects) == 1
        result = next(m for m in log.reader("s").snapshot() if isinstance(m.body, ToolResult))
        assert result.body.call_ref == CallRef(call.message_id, 0)
        assert len(requests) == 2


@pytest.mark.asyncio
async def test_new_input_does_not_reset_step_budget_and_abandon_starts_new_work(tmp_path):
    requests = []
    effects = []
    async def complete(request):
        requests.append(request)
        if len(requests) == 1:
            return LLMResponse(None, [ModelToolCall("call", "example", {})])
        return LLMResponse("new work")
    async def invoke(key, arguments):
        effects.append(key)
        return Result("success", ())
    async with runtime(tmp_path, complete, invoke, max_steps=1) as (conversation, log, store, run):
        await conversation.accept("u1", Input(()))
        with pytest.raises(StepLimit):
            await (await conversation.start(run)).join()
        await conversation.accept("u2", Input(()))
        with pytest.raises(StepLimit):
            await (await conversation.start(run)).join()
        assert len(requests) == len(effects) == 1
        head = log.reader("s").head()
        await conversation.control("abandon", Control("abandon", head), expected_head=head, handle=None)
        await conversation.accept("u3", Input(()))
        await (await conversation.start(run)).join()
        assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["soft", "local", "provider", "no_progress", "second_overflow"])
async def test_react_reduces_one_prepared_request_and_bounds_provider_retry(tmp_path, case):
    from agent.plugin_composition.models import ContextLengthError
    from plugins.context.api import Summary

    prepared_count = 0
    requests, reductions = [], []
    state = None
    original = None

    async def materials(snapshot):
        nonlocal prepared_count
        prepared_count += 1
        return Materials("fixed prompt", (ContentPart("retrieval", "actual query result"),))

    async def reduce(snapshot, materials, request, model, projection, *, source, force):
        assert source == "conversation"
        assert model.descriptor.binding_id == "model"
        assert request.tools and request.max_output_tokens == 100
        assert materials.system_prompt == "fixed prompt"
        reductions.append(force)
        if case == "no_progress" or (case in {"provider", "second_overflow"} and not force):
            return materials.summary
        summary = Summary("published", ("old-user", "old-reply"), "durable old history")
        state.transact(lambda tx: tx.save("published", {"summary": summary.content}, expected_version=None))
        return summary

    def estimate(messages, tools):
        return 9901 if case == "local" and '"summary":' not in str(messages) else 100

    async def complete(request):
        requests.append(request)
        if case == "no_progress" or case == "second_overflow" or (case == "provider" and len(requests) == 1):
            raise ContextLengthError("provider rejected actual payload")
        assert state.read("published").value["summary"] == "durable old history"
        assert "old verbatim" not in str(request.messages)
        assert "current input" in str(request.messages)
        assert "actual query result" in str(request.messages)
        assert request.messages[0] == {"role": "system", "content": "fixed prompt"}
        return LLMResponse("finished")

    async def invoke(key, arguments):
        pytest.fail("this request has no tool effects")

    async with runtime(tmp_path, complete, invoke, reducer=reduce,
                       material_source=materials, estimate=estimate) as (conversation, log, store, run):
        state = log.owner("summary-test")
        log.save_binding("published", {"target": "summary-test"})
        old = log.writer("s", author="test", source="conversation", body_types=(Input, Output),
                         content={"text": check_text})
        old.append("old-user", Input((ContentPart("text", "old verbatim input"),)))
        old.append("old-reply", Output((ContentPart("text", "old verbatim reply"),), "complete"))
        await conversation.accept("current", Input((ContentPart("text", "current input"),)))
        original = log.reader("s").snapshot()
        task = await conversation.start(run)
        if case in {"no_progress", "second_overflow"}:
            with pytest.raises(ContextLengthError, match="provider rejected"):
                await task.join()
            failed = log.reader("s").snapshot()
            assert failed[:len(original)] == original
            assert len(failed) == len(original) + 1
            assert isinstance(failed[-1].body, Control) and failed[-1].body.action == "failure"
        else:
            await task.join()
            assert log.reader("s").snapshot()[:len(original)] == original
            used = log.reader("s").snapshot()[-1].body.parts[-1]
            assert used == ContentPart("context.summary", {"reference": "published"})
        assert prepared_count == 1
        assert len(requests) == (2 if case in {"provider", "second_overflow"} else 1)
        assert reductions == ([True] if case == "local" else [False] if case == "soft" else [False, True])
