"""Real source owners keep their admitted Input through shared reply preparation.

Temporary SQLite/files only. The model driver and delivery endpoint are local
fixtures; Task, source admission, reply program, ReAct and Models receipts are
production code. No credentials, network requests or production workspace.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--slow-writes", action="store_true")
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--owner", choices=("scheduler", "subagent", "wake"))
parser.add_argument("--case", choices=("fresh", "recovered", "newer-input", "newer-control", "two-phases"))
args = parser.parse_args()
if args.slow_writes and args.case not in {None, "fresh", "two-phases"}:
    parser.error("--slow-writes 只用于 fresh 或 two-phases；其他模式会同步写入恢复/竞争 fixture")
sys.path.insert(0, str(args.source))

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
    MessageWriters, OwnerState, SessionAdmission,
)
from agent.plugin_composition.models import (
    BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks
from agent.plugin_contracts import ContentPart, ContentReferences, Control, Input, Output
from plugins.content.contract import CONTENT
from plugins.delivery.contract import (
    DELIVERY_GUARDED_START as DELIVERY,
    DELIVERY_SENDERS,
)
from plugins.models.contract import MODEL_SELECTION
from plugins.conversation.contract import (
    CHECK_ORIGIN,
)
from plugins.content.plugin import _decode_text, check_text
from plugins.content.api import check_artifact
from plugins.context.api import Materials, material_data
from plugins.context.plugin import ContextBuilder
from plugins.models.content import ContentOwner
from plugins.models.projection import MessageChecksOwner, ProjectionOwner
from plugins.models.selection import SelectionOwner
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.react.plugin import react
from plugins.reply_program.program import run_reply
from plugins.scheduler.runtime import SchedulerRuntime
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler.store import JobStore
from plugins.sources.session import check_source
from plugins.subagent.request import PROFILE_TOOLS, Request as SubagentRequest
from plugins.subagent.runtime import SUBAGENT_PROGRAM, Subagents
from plugins.turn_projection.plugin import TurnProjection
from plugins.wake.api import DeliveryTarget
from plugins.wake.request import Request as WakeRequest, WAKE_PROGRAM
from plugins.wake.source import Source as WakeSource
from plugins.wake.state import WakeState
from session.log import MessageLog, SessionAttributes


class TextContent:
    check_text = staticmethod(check_text)
    check_artifact = staticmethod(check_artifact)
    prompts = ()
    checks = {"text": check_text}

    @asynccontextmanager
    async def bind(self):
        yield self

    def check_metadata(self, value):
        assert not value

    async def decode(self, text, references=()):
        return await _decode_text(text, (), references)


class LocalMaterials:
    async def prepare(self, *_args, **_kwargs):
        return material_data(Materials("Local source boundary scenario."))

    async def reduce(self, *_args, **_kwargs):
        return None


class EmptyTools:
    names = frozenset()
    schemas = ()
    system_prompt = ""

    async def create_menu(self, *_args, **_kwargs):
        return self

    async def drain_calls(self, *_args, **_kwargs):
        return None

    def name(self, _binding):
        raise AssertionError("scenario has no tools")

    def check_call(self, _call):
        raise AssertionError("scenario has no tools")


@asynccontextmanager
async def cleanup(*_args, **_kwargs):
    yield


@asynccontextmanager
async def material_scope():
    yield LocalMaterials()


class LocalDelivery:
    """A local successful sink; production source still writes real Message rows."""
    def __init__(self):
        self.selections = {}

    def open(self, _ctx):
        return self

    async def wait_idle(self, *_args):
        return None

    def selection(self, identity):
        return self.selections.get(identity)

    def publish(self, writer, identity, body, sinks):
        message = writer.append(identity, body)
        selected = SimpleNamespace(sinks=sinks)
        self.selections[identity] = selected
        return message, selected

    def prepare(self, _reader, message, _sinks):
        return self.selections[message.message_id]

    async def publish_async(self, writer, identity, body, sinks):
        message = await writer.append_async(identity, body)
        selected = SimpleNamespace(sinks=sinks)
        self.selections[identity] = selected
        return message, selected

    async def prepare_async(self, reader, message, sinks):
        return self.prepare(reader, message, sinks)

    async def send(self, _identity, _sink):
        return SimpleNamespace(status="delivered")

    def bind(self, *_args):
        return "sink"


async def check(directory: Path, kind: str, mode: str):
    log = MessageLog(directory / "sessions.db")
    models = ModelsStore(directory / "models.db", directory / "backups")
    models.initialize()
    root = CompositionRoot("source-boundary-scenario")
    tasks = PluginTasks()
    writers, state = MessageWriters(log), OwnerState(log)
    bindings = Bindings(log, root)
    calls = []
    receipts = []
    inputs = []
    contexts = {}
    stale = mode in {"newer-input", "newer-control"}
    originals = []

    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, *_args):
            return 10
        async def complete(self, request):
            calls.append(request)
            return LLMResponse("local result")

    descriptor = BoundModelDescriptor(
        binding_id="model", plugin_snapshot_id="scenario", model_revision=0,
        model_id="model", connection_id="local", driver_id="local",
        driver_contract_version="1", auth_identity="no-credentials", model="local", role="agent",
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="scenario",
    )
    bound = _BoundChat(descriptor, Driver(), models)

    class Models:
        @asynccontextmanager
        async def execution(self, **_options):
            yield SimpleNamespace(chat=lambda _role: bound)

    async def reply(task, reader, source):
        # This runs after the owner's admission and before the actual reply guard.
        # A new Input must not be adopted by sampling a later head.
        originals.extend(reader.snapshot())
        admitted = reader.head(source=source)
        inputs.append((source, task.boundary_hint, admitted))
        if stale:
            log.writer(reader.session_id, author=source, source=source,
                       body_types=(Input, Control), content={}).append(
                           "newer:" + source, Control("pause", admitted) if mode == "newer-control" else Input(()))
        result = await run_reply(
            contexts[kind], task, reader, source,
            models=Models(), content=TextContent(), context=ContextBuilder(),
            tools=EmptyTools(), cleanup=cleanup,
            check_admission=partial(check_source, task, reader, source, task.boundary_hint),
            selection=SelectionOwner(), tool_program=EmptyTools(),
            model_checks=MessageChecksOwner(), model_content=ContentOwner(),
            model_projection=ProjectionOwner(), writers=writers, owner_state=state,
            artifact_reader=None, read_call=models.read_call, react=react,
            materials=material_scope(), turn_projection=TurnProjection(),
            authorize=None, max_output_tokens=100, max_steps=4, fixed_bindings={},
        )
        receipts.append(result)
        return result

    async def subagent_program(task, reader, _request):
        return await reply(task, reader, "subagent")

    async def wake_program(task, reader, _request):
        return await reply(task, reader, "wake")

    async def storage(ctx):
        for key, value in (
            (MESSAGE_CATALOG, log.catalog()), (MESSAGE_WRITERS, writers),
            (OWNER_STATE, state), (SESSION_ADMISSION, SessionAdmission(log)),
            (TASKS, tasks), (BINDINGS, bindings), (CONTENT, TextContent()),
            (MODEL_SELECTION, SelectionOwner()), (CHECK_ORIGIN, lambda _part: ContentReferences()),
            (DELIVERY, LocalDelivery()), (DELIVERY_SENDERS, LocalDelivery()),
            (SUBAGENT_PROGRAM, subagent_program), (WAKE_PROGRAM, wake_program),
        ):
            await ctx.provide(key, value)

    async def owner(ctx):
        contexts[kind] = ctx

    dependencies = (MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
                    TASKS, BINDINGS, CONTENT, MODEL_SELECTION, CHECK_ORIGIN, DELIVERY, DELIVERY_SENDERS)
    await root.mount(storage, name="storage")
    await root.mount(owner, name=kind, inject=dependencies,
                     runtime=PluginRuntime(kind, kind, directory, directory, directory, {},
                         workspace_roots=("subagent-runs",), workspace_files=("memory/spawn_trace.jsonl",)))
    ctx = contexts[kind]
    log.save_binding("program", {"version": 1, "service": (
        SUBAGENT_PROGRAM.name if kind == "subagent" else WAKE_PROGRAM.name), "metadata": {}})
    for identity in ("sink", *PROFILE_TOOLS["research"], "screen_content", "share_content", "skip_content"):
        log.save_binding(identity, {"scenario": identity})
    delays = []
    original_write = log._write
    loop = asyncio.get_running_loop()

    def slow_write(callback):
        def commit():
            result = callback()
            release, started = threading.Event(), time.perf_counter()
            def heartbeat():
                delays.append(time.perf_counter() - started)
                release.set()
            loop.call_soon_threadsafe(heartbeat)
            release.wait(1)
            return result
        return original_write(commit)

    if args.slow_writes:
        log._write = slow_write
    try:
        async with ctx.runtime_scope():
            if kind == "scheduler":
                store = JobStore(directory / "schedules.json")
                job = ScheduledJob(id="schedule", trigger="at", tier="soft", channel="local", chat_id="one",
                                   prompt="scheduled input", fire_at=datetime.now(UTC))
                await store.add("add", job, "scheduled")
                fire = await store.start_fire(job)
                assert fire is not None
                if mode == "recovered":
                    log.ensure_session(fire.session_id, SessionAttributes(visibility="internal", learning="excluded"))
                    log.writer(fire.session_id, author="scheduler", source="scheduler", body_types=(Input,),
                               content={"text": check_text}).append("scheduler-input:" + fire.key,
                                                                      Input((ContentPart("text", job.prompt),)))
                runtime = SchedulerRuntime(ctx, store, lambda task, reader: reply(task, reader, "scheduler"))
                task = await runtime.start(fire)
                try:
                    await task.join()
                except asyncio.CancelledError:
                    assert stale, "valid scheduler input was rejected before the model"
                assert store.read().fires[fire.key].status == ("pending" if stale else "delivered")
            elif kind == "subagent":
                request = SubagentRequest(job_id="a" * 32, label="scenario", profile="research", background=False,
                    retry_count=0, parent_session_id="parent", parent_message_id="parent-input", parent_part_index=0,
                    origin=None, sink=None, program_binding="program", tools={key: key for key in PROFILE_TOOLS["research"]})
                jobs = Subagents(ctx)
                await jobs.accept("job", request, "child input")
                if mode == "recovered":
                    jobs = Subagents(ctx)
                task = await jobs.start("job")
                assert task is not None
                await task.join()
                outcome = await jobs.outcome(log.reader(request.session_id))
                expected_outcome = "cancelled" if mode == "newer-control" else "failed" if stale else "completed"
                assert outcome is not None and outcome[0] == expected_outcome, outcome
            else:
                request = WakeRequest(flow_id="b" * 32, owner="content", now=datetime.now(UTC), timezone="UTC",
                    target=DeliveryTarget(channel="local", recipient="one", session_id="parent"),
                    sink={"name": "local", "binding_id": "sink", "address": "one"},
                    program_binding="program", tools={name: name for name in ("screen_content", "share_content", "skip_content")},
                    snapshot_seq=0, rules="", history="")
                source = WakeSource(ctx, WakeState(directory / "wake.db"))
                await source.accept(request)
                reader = log.reader(request.session_id)
                stages = ("screen", "investigate") if mode == "two-phases" else ("screen",)
                if mode == "recovered":
                    from plugins.wake.request import Phase, check_phase
                    log.writer(request.session_id, author="wake", source="wake", body_types=(Input,),
                        content={"wake.phase": check_phase, "text": check_text, "model.selection": SelectionOwner.check}).append(
                            request.phase_id("screen"), Input((ContentPart("wake.phase", Phase(input_id=request.input_id, stage="screen").model_dump()),
                                ContentPart("text", "original phase"), ContentPart("model.selection", {"model_id": None, "reasoning_effort": None}))))
                    source = WakeSource(ctx, source.state)
                async def phases(task):
                    for stage in stages:
                        await source._phase(task, request, reader, stage, {})
                task = await ctx.require(TASKS).open(ctx).admit("flow", lambda slot: slot.start(phases))
                try:
                    await task.join()
                except asyncio.CancelledError:
                    assert stale, "valid Wake phase was rejected before the model"
            if args.slow_writes:
                assert delays and max(delays) < 0.2, delays
            expected = 0 if stale else (2 if mode == "two-phases" else 1)
            assert len(calls) == len(receipts) == expected, (kind, mode, len(calls), len(receipts))
            assert all(boundary >= 0 and boundary <= head for _, boundary, head in inputs), inputs
            if mode == "two-phases":
                assert inputs[0][1] < inputs[1][1], inputs
            assert all(log.reader(message.session_id).get(message.message_id) == message for message in originals)
            with sqlite3.connect(directory / "sessions.db") as raw:
                assert raw.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert raw.execute("PRAGMA foreign_key_check").fetchall() == []
            return {"slow_write_delays": delays, "source": kind, "case": mode, "driver_calls": len(calls), "boundaries": inputs}
    finally:
        await tasks.close()
        await root.dispose()
        models.close()
        log.close()


async def main(directory):
    results = []
    for source in ((args.owner,) if args.owner else ("scheduler", "subagent", "wake")):
        for mode in ((args.case,) if args.case else ("fresh",) if args.slow_writes else ("fresh", "recovered", "newer-input", "newer-control", *(("two-phases",) if source == "wake" else ()))):
            path = directory / (source + "-" + mode)
            path.mkdir()
            results.append(await check(path, source, mode))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-source-boundaries-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"source": str(args.source.resolve()), "cases": results, "cleanup": "passed"}, indent=2))
