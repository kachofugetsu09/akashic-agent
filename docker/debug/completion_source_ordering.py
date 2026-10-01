"""A background report keeps the controlling Source through first effect and Output.

Uses real Subagents._announce, SourceSession.complete, Task, run_reply, ReAct,
ModelsStore and MessageLog. Only the model driver and destination are local
fixtures. Notifications are deliberately delayed after real SQL commit; no
production files, credentials, external model request or delivery are used.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from docker.debug import source_reply_boundaries as fixture
from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
    MessageWriters, OwnerState, SessionAdmission,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks
from agent.plugin_contracts import ContentPart, ContentReferences, Control, Input, Output, ToolResult
from agent.plugin_composition.models import ToolCall as ModelToolCall
from plugins.tools.execution import ToolExecution
from plugins.tools.api import MessageReply, Result, result_message_id
from plugins.tools.menu import ToolCallDecode
from agent.plugin_contracts.content import CONTENT
from agent.plugin_contracts.delivery import DELIVERY
from agent.plugin_contracts.reply import REPLY_PROGRAM_V3
from agent.plugin_contracts.sources import CHECK_ORIGIN, CONVERSATION_COMPLETE_V2
from plugins.sources.session import SourceSession, check_source
from plugins.subagent.request import PROFILE_TOOLS, Request
from plugins.subagent.runtime import Subagents
from session.log import MessageLog, OwnerStore


class LocalDelivery(fixture.LocalDelivery):
    def prepare(self, reader, message, sinks):
        if message.message_id not in self.selections:
            self.selections[message.message_id] = SimpleNamespace(sinks=sinks)
        return super().prepare(reader, message, sinks)


async def check(directory: Path, stage: str, control: bool):
    log = MessageLog(directory / "sessions.db")
    models = fixture.ModelsStore(directory / "models.db", directory / "backups")
    models.initialize()
    root, tasks = CompositionRoot("completion-source-ordering"), PluginTasks()
    writers, state = MessageWriters(log), OwnerState(log)
    contexts, calls, report_tasks, effects = {}, [], [], []
    tool_state = log.owner("tools")
    reached, proceed = asyncio.Event(), asyncio.Event()
    committed, notify = asyncio.Event(), asyncio.Event()
    rejected = asyncio.Event()
    source = None

    async def hold():
        reached.set()
        await proceed.wait()

    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, *_args):
            return 10
        async def complete(self, request):
            calls.append(request)
            if stage in {"tool", "tool-success", "tool-finished"} and len(calls) == 1:
                return fixture.LLMResponse(None, [ModelToolCall("local-call", "local_effect", {})])
            return fixture.LLMResponse("background report")

    descriptor = fixture.BoundModelDescriptor(
        binding_id="local-model", plugin_snapshot_id="scenario", model_revision=0,
        model_id="local-model", connection_id="local", driver_id="local",
        driver_contract_version="1", auth_identity="none", model="local", role="agent",
        reasoning_effort=None, capabilities=fixture.ModelCapabilities(context_window=10000),
        capability_sources=fixture.CapabilitySources(), capability_digest="scenario",
    )
    bound = fixture._BoundChat(descriptor, Driver(), models)

    class Models:
        @asynccontextmanager
        async def execution(self, **_kwargs):
            yield SimpleNamespace(chat=lambda _role: bound)

    class Content(fixture.TextContent):
        async def decode(self, text, references=()):
            if stage == "output":
                await hold()
            return await super().decode(text, references)

    class Menu(fixture.EmptyTools):
        names = frozenset({"local_effect"})
        schemas = ({"type": "function", "function": {"name": "local_effect", "parameters": {"type": "object"}}},)

        async def create_menu(self, reader, output_source, *, check_start, **_kwargs):
            self.reader, self.source, self.check_start = reader, output_source, check_start
            self.tasks = tasks.open(contexts["reply"])
            return self

        def decode(self, call):
            assert call.name == "local_effect"
            return ToolCallDecode("local-tool", call.arguments)

        def name(self, binding):
            assert binding == "local-tool"
            return "local_effect"

        def check_call(self, call):
            assert call.binding_id == "local-tool"

        def parallel(self, _binding):
            return False

        async def execute(self, call, *, commit_after=None):
            async def authorize(_binding, _arguments):
                if stage == "tool":
                    await hold()
                return {"scenario": "allowed"}

            class Target:
                idempotent = False
                async def prepare(self, arguments, _source=None):
                    return arguments
                async def query(self, _key):
                    return None
                async def invoke(self, key, _arguments):
                    with (directory / "effect.txt").open("a") as stream:
                        stream.write(key + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    effects.append(key)
                    if stage == "tool-finished":
                        await hold()
                    return Result("success", (ContentPart("text", "effect"),))

            @asynccontextmanager
            async def open_tool(binding):
                assert binding == "local-tool"
                yield Target()

            execution = ToolExecution(tool_state, self.tasks, open_tool,
                                      authorize, task_key="tool-effect")
            writer = log.writer(self.reader.session_id, author="tool", source=self.source,
                body_types=(ToolResult,), content={"text": fixture.check_text}, call_ref=call)
            try:
                return await execution.execute_call(MessageReply(result_message_id(call), call,
                    self.reader, writer, self.check_start), commit_after=commit_after)
            finally:
                writer.expire()

    async def complete(session_id, program):
        assert source is not None and session_id == "parent"
        async with contexts["conversation"].runtime_scope():
            return await source.complete(program)

    async def report(task, reader, output_source, _reminders, *, check_admission):
        report_tasks.append(task)
        try:
            async with contexts["reply"].runtime_scope():
                if stage == "entry":
                    await hold()
                return await fixture.run_reply(
                    contexts["reply"], task, reader, output_source,
                    models=Models(), content=Content(), context=fixture.ContextBuilder(),
                    tools=fixture.EmptyTools(), cleanup=fixture.cleanup,
                    check_admission=check_admission, selection=fixture.SelectionOwner(),
                    tool_program=Menu() if stage in {"tool", "tool-success", "tool-finished"} else fixture.EmptyTools(), model_checks=fixture.MessageChecksOwner(),
                    model_content=fixture.ContentOwner(), model_projection=fixture.ProjectionOwner(),
                    writers=writers, owner_state=state, artifact_reader=None, read_call=models.read_call,
                    react=fixture.react, materials=fixture.material_scope(),
                    turn_projection=fixture.TurnProjection(), authorize=None,
                    max_output_tokens=100, max_steps=4, fixed_bindings={},
                )
        except asyncio.CancelledError:
            rejected.set()
            raise

    async def storage(ctx):
        for key, value in (
            (MESSAGE_CATALOG, log.catalog()), (MESSAGE_WRITERS, writers), (OWNER_STATE, state),
            (SESSION_ADMISSION, SessionAdmission(log)), (TASKS, tasks), (CONTENT, Content()),
            (CHECK_ORIGIN, lambda _part: ContentReferences()), (DELIVERY, LocalDelivery()),
            (CONVERSATION_COMPLETE_V2, complete), (REPLY_PROGRAM_V3, report),
        ):
            await ctx.provide(key, value)

    dependencies = (MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION, TASKS,
                    CONTENT, CHECK_ORIGIN, DELIVERY, CONVERSATION_COMPLETE_V2, REPLY_PROGRAM_V3)
    await root.mount(storage, name="storage")
    for name in ("conversation", "reply", "subagent"):
        async def owner(ctx, name=name):
            contexts[name] = ctx
        await root.mount(owner, name=name, inject=dependencies,
            runtime=PluginRuntime(name, name, directory, directory, directory, {},
                                 workspace_roots=("subagent-runs",), workspace_files=("memory/spawn_trace.jsonl",)))
    def writer(kind, source_name="conversation", session="parent"):
        return log.writer(session, author=source_name, source=source_name,
                          body_types=(kind,), content={"text": fixture.check_text})
    inputs, controls = writer(Input), writer(Control)
    async with contexts["conversation"].runtime_scope():
        source = SourceSession(reader=log.reader("parent"), inputs=inputs, controls=controls,
                               tasks=tasks.open(contexts["conversation"]))
        await source.accept("original", Input(()))
        writer(Output).append("original-answer", Output((), "complete"))
    request = Request(job_id="c" * 32, label="finished child", profile="research", background=True,
        retry_count=0, parent_session_id="parent", parent_message_id="original", parent_part_index=0,
        origin={"channel": "local", "chat_id": "one", "sender": "user"},
        sink={"name": "local", "binding_id": "sink", "address": "one"},
        program_binding="program", tools={name: name for name in PROFILE_TOOLS["research"]})
    for identity in ("program", "sink", "local-tool", *PROFILE_TOOLS["research"]):
        log.save_binding(identity, {"scenario": identity})
    async with contexts["subagent"].runtime_scope():
        jobs = Subagents(contexts["subagent"])
        jobs.accept("job", request, "child task")
    writer(Output, "subagent", request.session_id).append("child-result", Output((ContentPart("text", "child done"),), "complete"))
    originals = tuple(message for session in ("parent", request.session_id) for message in log.reader(session).snapshot())
    original_transact = OwnerStore.transact_async

    async def transact(store, callback, **kwargs):
        # Pause before the real generation-claim transaction acquires SQL ownership.
        if stage == "claim" and callback.__name__ == "advance":
            await hold()
        return await original_transact(store, callback, **kwargs)

    async def announce():
        async with contexts["subagent"].runtime_scope():
            return await jobs._announce("job", request, log.reader(request.session_id), ("completed", "child done"))

    announce_job = accepting = None
    try:
        with patch.object(OwnerStore, "transact_async", transact):
            announce_job = asyncio.create_task(announce())
            if stage in {"success", "tool-success"}:
                assert await announce_job is True
                assert len(calls) == (2 if stage == "tool-success" else 1)
                assert len(effects) == int(stage == "tool-success")
                assert any(m.source == request.session_id for m in log.reader("parent").snapshot())
            else:
                await asyncio.wait_for(reached.wait(), 3)
                selected = controls if control else inputs
                original_append = selected.append_async

                async def delayed(identity, body, **kwargs):
                    callback = kwargs.pop("on_commit")
                    receipts = []
                    message = await original_append(identity, body,
                        on_commit=lambda saved, created: receipts.append((saved, created)), **kwargs)
                    committed.set()
                    await notify.wait()
                    for receipt in receipts:
                        callback(*receipt)
                    return message

                selected.append_async = delayed
                accepting = asyncio.create_task(source.pause("new-control") if control else source.accept("new-input", Input(())))
                await asyncio.wait_for(committed.wait(), 3)
                assert report_tasks and report_tasks[0].active
                assert log.reader("parent").get("new-control" if control else "new-input") is not None
                proceed.set()
                await asyncio.wait_for(rejected.wait(), 3)
                assert len(calls) == int(stage in {"output", "tool", "tool-finished"})
                assert len(effects) == int(stage == "tool-finished")
                if stage == "tool-finished":
                    assert [row.value["phase"] for _, row in tool_state.list()] == ["done"]
                    assert any(isinstance(m.body, ToolResult) and m.body.outcome == "success" for m in log.reader("parent").snapshot())
                if stage == "tool":
                    assert [row.value["phase"] for _, row in tool_state.list()] == ["prepared"]
                assert not any(m.source == request.session_id and isinstance(m.body, Output)
                               and m.body.finish != "continue" for m in log.reader("parent").snapshot())
                announce_job.cancel()
                await asyncio.gather(announce_job, return_exceptions=True)
                notify.set()
                await accepting
            effect_file = directory / "effect.txt"
            assert (effect_file.read_text().splitlines() if effect_file.exists() else []) == effects
            assert all(log.reader(m.session_id).get(m.message_id) == m for m in originals)
            with sqlite3.connect(directory / "sessions.db") as db:
                assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert db.execute("PRAGMA foreign_key_check").fetchall() == []
            with sqlite3.connect(directory / "models.db") as db:
                statuses = [row[0] for row in db.execute("SELECT state FROM model_calls")]
                assert statuses == ["success"] * len(calls), statuses
            return {"stage": stage, "control": control, "driver_calls": len(calls), "models_receipts": statuses, "tool_effects": len(effects)}
    finally:
        proceed.set()
        notify.set()
        for job in (announce_job, accepting):
            if job is not None and not job.done():
                job.cancel()
        await asyncio.gather(*(job for job in (announce_job, accepting) if job is not None), return_exceptions=True)
        await tasks.close()
        await root.dispose()
        models.close()
        log.close()


async def main(directory):
    results = []
    for stage, control in [("success", False), ("tool-success", False), *((stage, control) for stage in ("entry", "claim", "tool", "tool-finished", "output") for control in (False, True))]:
        path = directory / f"{stage}-{control}"
        path.mkdir()
        results.append(await check(path, stage, control))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-completion-order-") as directory:
        results = asyncio.run(asyncio.wait_for(main(Path(directory)), 30))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))
