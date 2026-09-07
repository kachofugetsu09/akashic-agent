import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.delivery.api import Sink
from plugins.delivery.plugin import DELIVERY
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.drift.plugin import DRIFT_PROPOSALS
from plugins.tools.plugin import TOOLS
from plugins.wake.api import DeliveryTarget, DRIFT_WAKE, DRIFT_DELIVERY, EVENTMAIL_WAKE
from plugins.wake.request import Request, TOOLS as WAKE_TOOLS, WAKE_PROGRAM
from plugins.wake.source import Source
from plugins.wake.state import WakeState
from session.message import Input, Output, ToolResult
from tests.test_standard_tools import environment


CONTROLS = {}


@asynccontextmanager
async def application(tmp_path, *, wake_delivery=False):
    host, store, log, artifacts, sources = environment(tmp_path, reply=True)
    for name in ("sources", "conversation", "react", "wake", "delivery", "eventmail", "drift"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (sources / "wake/akashic.plugin.toml").write_text(
        'schema_version = 1\nname = "wake"\nversion = "4.0.0"\napi_version = 3\nentrypoint = "message_plugin.py"\n')
    module = sources / "wake/message_plugin.py"
    module.write_text(module.read_text() + '''
from agent.plugin_composition import ServiceKey
_original_apply = apply
async def apply(ctx, config):
    await _original_apply(ctx, config)
    await ctx.provide(ServiceKey("fixture.wake"), ctx)
''')
    text = module.read_text()
    if wake_delivery:
        text = text.replace("await _original_apply(ctx, config)",
            'await _original_apply(ctx, Config.model_validate({"delivery": {"channel": "test", "recipient": "room", "session_id": "test:room"}}))')
    text += "\nfrom tests.test_wake_messages import CONTROLS\n_original_runtime = Runtime\ndef Runtime(ctx, config):\n    runtime = _original_runtime(ctx, config)\n    control = CONTROLS[" + repr(str(tmp_path)) + "]\n    control['runtime'] = runtime\n    deadline = runtime.duties.deadline\n    def observe(now):\n        value = deadline(now)\n        control.setdefault('deadlines', []).append(value)\n        control['due_read'].set()\n        return value\n    runtime.duties.deadline = observe\n    return runtime\n"
    module.write_text(text)
    provider = sources / "models_fixture"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from types import SimpleNamespace
from agent.plugin_composition import CHAT_MODELS
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ModelRole, ToolCall
from plugins.models.projection import MODEL_CALLS
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.delivery.api import Receipt
from plugins.tools.plugin import TOOLS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_EMBEDDINGS
from plugins.akasha.interest import SEMANTIC_INTEREST, SemanticInterest
from plugins.akasha.learning import Learning, LearningConfig
from plugins.turn_projection.plugin import TURN_PROJECTION
from tests.test_wake_messages import CONTROLS
api_version = 3
name = "models_fixture"
version = "1.0.0"
inject = (DELIVERY_SENDERS, MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, TURN_PROJECTION, TOOLS)
async def apply(ctx, config):
    control = CONTROLS[CONTROL_PATH]
    async def embed(texts):
        return [[1.0, 0.0] for _ in texts]
    await ctx.provide(SEMANTIC_INTEREST, SemanticInterest(Learning(ctx.require(TURN_PROJECTION), owner="akasha"),
        ctx.require(MESSAGE_CATALOG), ctx.require(MESSAGE_EMBEDDINGS),
        lambda: (LearningConfig(embedding_model="fixture", dimension=2, sources=("conversation",)), embed)))
    @asynccontextmanager
    async def unused_recall(state):
        raise AssertionError("this fixture never invokes memory recall")
        yield
    await ctx.require(TOOLS).register(ctx, name="recall_memory", description="fixture unused recall boundary",
        parameters={"type": "object", "properties": {}}, open=unused_recall, idempotent=True)
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            control["calls"].append(request)
            control["entered"].put_nowait(request)
            await control["release"].wait()
            if control["failure"] is not None:
                raise control["failure"]
            if control.get("content") and len(control["calls"]) == 1:
                return LLMResponse(None, [ToolCall("screen", "screen_content", {"items": [{
                    "candidate_id": control["candidate"], "initial_interest": "relevant", "question": "verify this"}]})])
            name = control["tool"]
            args = {"reason": "nothing useful"} if name == "skip_content" else {"message": "useful notification"}
            if name == "share_content":
                args["items"] = [control["candidate"]] if control.get("content") else []
            return LLMResponse(None, [ToolCall("decision", name, args)])
    descriptor = BoundModelDescriptor(binding_id="fixture-model", plugin_snapshot_id="fixture", model_revision=0,
        model_id="fixture", connection_id="fixture", driver_id="fixture", driver_contract_version="1",
        auth_identity="fixture", model="fixture", role=ModelRole.AGENT, reasoning_effort=None,
        capabilities=ModelCapabilities(context_window=10000), capability_sources=CapabilitySources(), capability_digest="fixture")
    model = _BoundChat(descriptor, Driver(), store)
    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            control.setdefault("models", []).append((model_id, reasoning_effort))
            yield SimpleNamespace(chat=lambda role: model)
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    class Sender:
        idempotent = True
        async def send(self, key, address, message):
            control["sent"].append((key, address, message))
            return Receipt(status="delivered", provider_ids=(key,))
        async def query(self, key, address):
            return Receipt(status="delivered", provider_ids=(key,)) if any(row[0] == key for row in control["sent"]) else None
    @asynccontextmanager
    async def sender():
        yield Sender()
    await ctx.require(DELIVERY_SENDERS).register(ctx, name="test", idempotent=True, open=sender)
'''.replace("CONTROL_PATH", repr(str(tmp_path))))
    control = {"calls": [], "sent": [], "entered": asyncio.Queue(), "release": asyncio.Event(),
               "tool": "share_content", "failure": None, "due_read": asyncio.Event()}
    control["release"].set()
    CONTROLS[str(tmp_path)] = control
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context.require(ServiceKey("fixture.wake"))
            async with ctx.runtime_scope():
                state = WakeState(ctx.data_root / "wake.sqlite3")
                state.initialize()
                source = Source(ctx, state)
                yield host, log, ctx, source, control
    finally:
        await host.terminate_all()
        log.close()
        store.close()


def request(ctx, owner, now, *, proposals=(), alert_ref=None):
    bindings = ctx.require(BINDINGS)
    return Request(flow_id="a" * 32, owner=owner, now=now, timezone="UTC",
        target=DeliveryTarget(channel="test", recipient="room", session_id="test:room"),
        sink=Sink(name="test", binding_id=ctx.require(DELIVERY_SENDERS).bind("test", bindings), address="room"),
        program_binding=bindings.bind(WAKE_PROGRAM, {}),
        tools={name: ctx.require(TOOLS).bind(name, bindings) for name in WAKE_TOOLS[owner]},
        snapshot_seq=0, proposals=tuple(dict(item) for item in proposals), alert_ref=alert_ref,
        rules="", history="")


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["share_content", "skip_content"])
async def test_drift_runs_actual_private_tool_and_settles_once(tmp_path, action):
    async with application(tmp_path) as (host, log, ctx, source, control):
        control["tool"] = action
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "check something"}, now)
        original = request(ctx, "drift", now, proposals=ctx.require(DRIFT_WAKE).snapshot(now)["proposals"])
        source.accept(original)
        task = await source.start(original.flow_id)
        result = await asyncio.wait_for(task.join(), 10)
        assert result == ("shared" if action == "share_content" else "model_skip")
        rows = log.reader(original.session_id).snapshot()
        assert [type(row.body) for row in rows] == [Input, Input, Output, ToolResult, Output]
        assert log.reader(original.session_id).attributes.visibility == "internal"
        assert log.reader(original.session_id).attributes.learning == "excluded"
        assert len(control["calls"]) == 1
        assert set(tool["name"] for tool in ctx.require(TOOLS).descriptions()).isdisjoint(WAKE_TOOLS["drift"])
        assert len(control["sent"]) == (1 if action == "share_content" else 0)
        if action == "share_content":
            assert ctx.require(DRIFT_DELIVERY).lookup(original.accepted)["status"] == "settled"
        assert source.pending() == ()
        assert await source.start(original.flow_id) is None
        assert log.reader(original.session_id).snapshot() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["decision", "ready", "delivered", "skipped"])
async def test_drift_recovers_original_terminal_across_domain_and_delivery_commits(tmp_path, monkeypatch, fault):
    async with application(tmp_path) as (host, log, ctx, source, control):
        control["tool"] = "skip_content" if fault == "skipped" else "share_content"
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "check something"}, now)
        original = request(ctx, "drift", now, proposals=ctx.require(DRIFT_WAKE).snapshot(now)["proposals"])
        source.accept(original)
        target, name = ((ctx.require(DRIFT_WAKE), "transition") if fault == "decision" else
                        (source, "_notify") if fault == "ready" else
                        (ctx.require(DRIFT_DELIVERY), "settle") if fault == "delivered" else
                        (source, "_settled"))
        previous = getattr(target, name)
        def fail(*args, **kwargs):
            raise OSError("controlled commit interruption")
        async def fail_async(*args, **kwargs):
            fail()
        monkeypatch.setattr(target, name, fail_async if fault == "ready" else fail)
        task = await source.start(original.flow_id)
        with pytest.raises(OSError, match="controlled commit"):
            await asyncio.wait_for(task.join(), 10)
        assert len(control["calls"]) == 1
        before = log.reader(original.session_id).snapshot()
        monkeypatch.setattr(target, name, previous)
        recovered = Source(ctx, source.state)
        task = await recovered.start(original.flow_id)
        await asyncio.wait_for(task.join(), 10)
        assert len(control["calls"]) == 1
        assert log.reader(original.session_id).snapshot() == before
        assert len(control["sent"]) == (0 if fault == "skipped" else 1)
        assert recovered.pending() == ()
        if fault != "skipped":
            assert ctx.require(DRIFT_DELIVERY).lookup(original.accepted)["status"] == "settled"
        else:
            assert ctx.require(DRIFT_WAKE).snapshot(now)["proposals"] == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("superseded,interrupt_expire", [(False, False), (True, False), (False, True)])
async def test_alert_queue_rechecks_original_expiry_and_never_closes_new_envelope(
    tmp_path, monkeypatch, superseded, interrupt_expire,
):
    from plugins.eventmail.plugin import EVENTMAIL_ALERT_SOURCE
    async with application(tmp_path) as (host, log, ctx, source, control):
        control["tool"] = "share_alert"
        control["release"].clear()
        now = datetime.now(timezone.utc)
        current_time = now
        source.now = lambda: current_time
        producer = ctx.require(EVENTMAIL_ALERT_SOURCE).bind("source")
        producer.report(event_id="event", payload={"body": "original"},
                              observed_at=now, expires_at=now + timedelta(seconds=1))
        domain = ctx.require(EVENTMAIL_WAKE)
        original = request(ctx, "alert", now, alert_ref=dict(domain.peek_alert(now)))
        source.accept(original)
        task = await source.start(original.flow_id)
        await asyncio.wait_for(control["entered"].get(), 10)
        delivery = ctx.require(DELIVERY).open(ctx)
        queued = asyncio.Event()
        exclusive = delivery._tasks.exclusive
        @asynccontextmanager
        async def observe(*args, **kwargs):
            queued.set()
            async with exclusive(*args, **kwargs):
                yield
        monkeypatch.setattr(type(delivery._tasks), "exclusive", lambda self, *args, **kwargs: observe(*args, **kwargs))
        change = domain.change_alert
        def fail_expire(ref, accepted, action, when, **kwargs):
            if action == "expire":
                raise OSError("interrupt before domain expiry")
            return change(ref, accepted, action, when, **kwargs)
        with delivery.activity("test", "room"):
            control["release"].set()
            await asyncio.wait_for(queued.wait(), 10)
            current_time = now + timedelta(seconds=2)
            if superseded:
                producer.report(event_id="event", payload={"body": "new"}, observed_at=current_time)
            if interrupt_expire:
                monkeypatch.setattr(domain, "change_alert", fail_expire)
        if interrupt_expire:
            with pytest.raises(OSError, match="domain expiry"):
                await asyncio.wait_for(task.join(), 10)
            assert source.pending() == (original.flow_id,)
            monkeypatch.setattr(domain, "change_alert", change)
            task = await source.start(original.flow_id)
        assert await asyncio.wait_for(task.join(), 10) == "model_skip"
        assert not control["sent"]
        assert len(control["calls"]) == 1
        assert delivery.receipt(original.notification_id, "test").status == "rejected"
        assert domain.alert_status("source", "event") == ("pending" if superseded else "expired")
        assert source.pending() == ()


@pytest.mark.asyncio
async def test_content_screen_and_investigation_keep_original_refs_until_provider_ack(tmp_path):
    from plugins.eventmail.plugin import EVENTMAIL_CONTENT_SOURCE
    from plugins.wake.api import EVENTMAIL_DELIVERY
    from plugins.wake.content import _candidate_id
    async with application(tmp_path) as (host, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        producer = ctx.require(EVENTMAIL_CONTENT_SOURCE).bind("feed")
        producer.submit("first", [{"item_id": "one", "revision": "1", "not_before": now,
            "requires_ack": True, "payload": {"title": "useful", "url": "https://example.com/original"}}])
        snapshot = ctx.require(EVENTMAIL_WAKE).snapshot(now)
        control["content"] = True
        control["candidate"] = _candidate_id(snapshot["items"][0]["ref"])
        original = request(ctx, "content", now).model_copy(update={
            "snapshot_seq": snapshot["snapshot_seq"], "items": tuple(dict(item) for item in snapshot["items"])})
        source.accept(original)
        task = await source.start(original.flow_id)
        assert await asyncio.wait_for(task.join(), 10) == "shared"
        assert len(control["calls"]) == 2 and len(control["sent"]) == 1
        rows = log.reader(original.session_id).snapshot()
        assert [type(row.body) for row in rows] == [Input, Input, Output, ToolResult, Output, Input, Output, ToolResult, Output]
        delivered = ctx.require(EVENTMAIL_DELIVERY).lookup(original.accepted)
        assert delivered["status"] == "delivered"
        assert len(producer.unsettled()) == 1  # 上游 ACK 仍由来源自己提交。
        assert "https://example.com/original" in control["sent"][0][2].body.parts[0].value
        producer.ack(original.notification_id)
        assert ctx.require(EVENTMAIL_DELIVERY).lookup(original.accepted)["status"] == "settled"
        assert await source.start(original.flow_id) is None
        assert len(control["calls"]) == 2


@pytest.mark.asyncio
async def test_runtime_timer_captures_original_drift_and_records_real_completion(tmp_path, monkeypatch):
    from plugins.wake.api import Config
    from plugins.wake.runtime import Runtime
    async with application(tmp_path) as (host, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "runtime flow"}, now)
        runtime = Runtime(ctx, Config(delivery=DeliveryTarget(channel="test", recipient="room", session_id="test:room")))
        completed = asyncio.Event()
        finish = runtime.state.finish_attempt
        def observe(**kwargs):
            finish(**kwargs)
            if kwargs["outcome"] == "shared":
                completed.set()
        monkeypatch.setattr(runtime.state, "finish_attempt", observe)
        running = asyncio.create_task(runtime.follow())
        try:
            await asyncio.wait_for(completed.wait(), 10)
            attempts = runtime.state.list_attempts()
            assert len(attempts) == 1 and attempts[0]["outcome"] == "shared"
            assert attempts[0]["mail_watermark"] == 0
            assert len(control["calls"]) == 1 and len(control["sent"]) == 1
            assert runtime.source.pending() == ()
        finally:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running


@pytest.mark.asyncio
@pytest.mark.parametrize("can_retry", [False, True])
async def test_model_failure_keeps_real_control_and_original_retry_classification(tmp_path, can_retry):
    from agent.plugin_composition.models import AuthenticationError, RateLimitError
    from plugins.wake.request import retryable
    from session.message import Control
    async with application(tmp_path) as (host, log, ctx, source, control):
        control["failure"] = (RateLimitError if can_retry else AuthenticationError)("provider refused")
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "try once"}, now, next_due=now + timedelta(minutes=5))
        original = request(ctx, "drift", now, proposals=ctx.require(DRIFT_WAKE).snapshot(now)["proposals"])
        source.accept(original)
        task = await source.start(original.flow_id)
        assert await asyncio.wait_for(task.join(), 10) == ("deferred" if can_retry else "model_skip")
        rows = log.reader(original.session_id).snapshot()
        assert isinstance(rows[-1].body, Control) and retryable(rows[-1]) is can_retry
        assert bool(ctx.require(DRIFT_WAKE).snapshot(now)["proposals"]) is can_retry
        assert not control["sent"] and len(control["calls"]) == 1
        assert await source.start(original.flow_id) is None


@pytest.mark.asyncio
async def test_capture_freezes_target_model_and_phase_text_remains_a_real_memory_cue(tmp_path):
    from plugins.content.plugin import check_text
    from plugins.conversation.source import update_selection
    from plugins.models.selection import check_selection
    from plugins.wake.api import Config
    from plugins.wake.runtime import Runtime
    from plugins.akasha.learning import Learning
    from plugins.turn_projection.plugin import TURN_PROJECTION
    from session.message import ContentPart
    async with application(tmp_path) as (host, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        writer = log.writer("test:room", author="user", source="fixture", body_types=(Input,),
            content={"text": check_text, "model.selection": check_selection},
            metadata_keys=frozenset({"model_selection", "model_runtime_override"}), update_metadata=update_selection)
        def select(identity, model):
            writer.append(identity, Input((ContentPart("model.selection", {"model_id": model, "reasoning_effort": "high"}),)))
        select("old", "chosen-original")
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "my interests"}, now)
        runtime = Runtime(ctx, Config(delivery=DeliveryTarget(channel="test", recipient="room", session_id="test:room")))
        original = runtime.capture("b" * 32, await runtime.duties.check(now), now)
        assert original.model_id == "chosen-original"
        source.accept(original)
        select("new", "changed-later")
        task = await source.start(original.flow_id)
        assert await asyncio.wait_for(task.join(), 10) == "shared"
        assert control["models"] == [("chosen-original", "high")]
        phase = log.reader(original.session_id).get(original.phase_id("drift"))
        cue = Learning(ctx.require(TURN_PROJECTION), owner="akasha").text(phase)
        assert "my interests" in cue
        assert str(control["calls"][0].messages).count("my interests") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["input", "ready", "delivered"])
async def test_reopen_uses_original_program_sender_and_input_after_source_changes(tmp_path, monkeypatch, fault):
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore
    from session.log import MessageLog
    async with application(tmp_path) as (host, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "original duty"}, now)
        original = request(ctx, "drift", now, proposals=ctx.require(DRIFT_WAKE).snapshot(now)["proposals"])
        source.accept(original)
        if fault != "input":
            async def fail_notify(*args, **kwargs):
                raise OSError("interrupt before notification")
            def fail_settle(*args, **kwargs):
                raise OSError("interrupt before domain confirmation")
            if fault == "ready":
                monkeypatch.setattr(source, "_notify", fail_notify)
            else:
                monkeypatch.setattr(ctx.require(DRIFT_DELIVERY), "settle", fail_settle)
            task = await source.start(original.flow_id)
            with pytest.raises(OSError, match="interrupt before"):
                await asyncio.wait_for(task.join(), 10)
        saved = log.reader(original.session_id).snapshot()
    module = tmp_path / "plugins/models_fixture/plugin.py"
    module.write_text(module.read_text().replace("async def complete(self, request):",
        'async def complete(self, request):\n            raise RuntimeError("changed model must not run")').replace(
        "async def send(self, key, address, message):",
        'async def send(self, key, address, message):\n            raise RuntimeError("changed sender must not run")'))
    workspace = tmp_path / "workspace"
    log = MessageLog(workspace / "sessions.db")
    metadata = ArtifactStore(workspace / "sessions.db")
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
    host = PluginManager([tmp_path / "plugins"], event_bus=EventBus(), workspace=workspace,
        installed_cache_root=tmp_path / "cache", message_log=log, channel_attachment_store=artifacts)
    try:
        await host.load_all()
        await host.start_runtime()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context.require(ServiceKey("fixture.wake"))
            async with ctx.runtime_scope():
                source = Source(ctx, WakeState(ctx.data_root / "wake.sqlite3"))
                task = await source.start(original.flow_id)
                if task is not None:
                    await asyncio.wait_for(task.join(), 10)
                assert source.pending() == ()
                assert await source.start(original.flow_id) is None
                assert ctx.require(DRIFT_DELIVERY).lookup(original.accepted)["status"] == "settled"
        assert len(control["calls"]) == 1 and len(control["sent"]) == 1
        assert log.reader(original.session_id).snapshot()[:len(saved)] == saved
        assert len(log.reader(original.session_id).snapshot()) == 5
    finally:
        await host.terminate_all()
        log.close()
        metadata.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["admission", "source", "maintenance"])
async def test_runtime_failure_closes_timer_audit_before_stopping_both_loops(tmp_path, monkeypatch, where):
    from plugins.wake.api import Config
    from plugins.wake.runtime import Runtime
    async with application(tmp_path) as (host, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        runtime = Runtime(ctx, Config(delivery=DeliveryTarget(channel="test", recipient="room", session_id="test:room")))
        async def fail(*args, **kwargs):
            raise OSError("controlled runtime failure")
        if where == "maintenance":
            monkeypatch.setattr(runtime.state, "next_maintenance_deadline", lambda now, **kwargs: now)
            monkeypatch.setattr(runtime.duties, "maintain", fail)
        else:
            ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "failure"}, now)
            monkeypatch.setattr(runtime.duties if where == "admission" else runtime.source,
                                "check" if where == "admission" else "_notify", fail)
        with pytest.raises(ExceptionGroup) as failure:
            await asyncio.wait_for(runtime.follow(), 10)
        assert "controlled runtime failure" in str(failure.value.exceptions[0])
        attempts = runtime.state.list_attempts()
        assert len(attempts) == 1
        assert attempts[0]["outcome"] == ("delivery_unknown" if where == "source" else "failed")
        assert bool(runtime.source.pending()) is (where == "source")
        assert not control["sent"]


@pytest.mark.asyncio
async def test_runtime_stop_drains_its_active_source_before_returning(tmp_path, monkeypatch):
    from plugins.wake.api import Config
    from plugins.wake.runtime import Runtime
    async with application(tmp_path) as (host, log, ctx, source, control):
        control["release"].clear()
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "blocked model"}, now)
        runtime = Runtime(ctx, Config(delivery=DeliveryTarget(channel="test", recipient="room", session_id="test:room")))
        started = []
        start = runtime.source.start
        async def observe(flow_id):
            task = await start(flow_id)
            started.append(task)
            return task
        monkeypatch.setattr(runtime.source, "start", observe)
        running = asyncio.create_task(runtime.follow())
        await asyncio.wait_for(control["entered"].get(), 10)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert len(started) == 1 and started[0].done and not started[0].active
        assert len(runtime.source.pending()) == 1 and not control["sent"]
        assert runtime.state.list_attempts()[0]["outcome"] == "delivery_unknown"
        control["release"].set()
        flow_id = runtime.source.pending()[0]
        assert await runtime._run(flow_id) == "shared"
        assert len(control["calls"]) == 2 and len(control["sent"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("future_mail", [False, True])
async def test_new_drift_wakes_idle_runtime_and_replaces_later_deadline(tmp_path, monkeypatch, future_mail):
    from plugins.eventmail.plugin import EVENTMAIL_CONTENT_SOURCE
    async with application(tmp_path, wake_delivery=True) as (host, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        if future_mail:
            producer = ctx.require(EVENTMAIL_CONTENT_SOURCE).bind("feed")
            producer.submit("future", [{"item_id": "later", "revision": "1", "not_before": now + timedelta(days=1),
                "payload": {"title": "future content"}}])
        await host.start_runtime()
        runtime = control["runtime"]
        await asyncio.wait_for(control["due_read"].wait(), 10)
        assert (control["deadlines"][0] is not None) is future_mail
        completed = asyncio.Event()
        finish = runtime.state.finish_attempt
        def observe(**kwargs):
            finish(**kwargs)
            if kwargs["outcome"] == "shared":
                completed.set()
        monkeypatch.setattr(runtime.state, "finish_attempt", observe)
        ctx.require(DRIFT_PROPOSALS).propose("new-duty", "1", {"summary": "newly due"}, now)
        await asyncio.wait_for(completed.wait(), 10)
        assert len(control["calls"]) == 1 and len(control["sent"]) == 1
        assert runtime.source.pending() == ()
        if future_mail:
            assert ctx.require(EVENTMAIL_WAKE).snapshot(now)["items"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_missing_target_only_maintains_pool_then_reload_can_admit_original_duty(tmp_path, monkeypatch):
    from plugins.wake.api import Config
    from plugins.wake.runtime import Runtime
    async with application(tmp_path) as (host, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "wait for target"}, now)
        runtime = Runtime(ctx, Config())
        completed = asyncio.Event()
        finish = runtime.state.finish_attempt
        first = True
        def deadline(now, **kwargs):
            nonlocal first
            if first:
                first = False
                return now
            return now + timedelta(days=1)
        def observe(**kwargs):
            finish(**kwargs)
            completed.set()
        monkeypatch.setattr(runtime.state, "next_maintenance_deadline", deadline)
        monkeypatch.setattr(runtime.state, "finish_attempt", observe)
        running = asyncio.create_task(runtime.follow())
        try:
            await asyncio.wait_for(completed.wait(), 10)
            assert not control["calls"] and not control["sent"]
            assert runtime.source.pending() == () and runtime.state.count_runs() == 0
            assert ctx.require(DRIFT_WAKE).snapshot(now)["proposals"][0]["ref"]["state_version"] == 1
            assert runtime.capture("c" * 32, await runtime.duties.check(now), now) is None
        finally:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        enabled = Runtime(ctx, Config(delivery=DeliveryTarget(channel="test", recipient="room", session_id="test:room")))
        original = enabled.capture("c" * 32, await enabled.duties.check(now), now)
        enabled.source.accept(original)
        assert await enabled._run(original.flow_id) == "shared"
        assert len(control["calls"]) == 1


@pytest.mark.asyncio
async def test_cancel_during_timer_cleanup_closes_fired_audit_and_drains_handle(tmp_path, monkeypatch):
    from agent.control.timer import TimerReceipt, TimerStatus
    from agent.plugin_composition.timers import TIMERS
    from plugins.wake.api import Config
    from plugins.wake.runtime import Runtime
    async with application(tmp_path) as (host, log, ctx, source, control):
        runtime = Runtime(ctx, Config())
        entered, release = asyncio.Event(), asyncio.Event()
        now = datetime.now(timezone.utc)
        class Handle:
            async def result(self):
                return TimerReceipt(timer_id="controlled", deadline=now, settled_at=now, status=TimerStatus.FIRED)
            async def cancel(self):
                return await self.result()
            async def cleanup(self):
                entered.set()
                await release.wait()
        monkeypatch.setattr(ctx.require(TIMERS), "schedule", lambda deadline: Handle())
        waiting = asyncio.create_task(runtime._wait(now, changed=False))
        await asyncio.wait_for(entered.wait(), 10)
        assert runtime.state.list_attempts()[0]["outcome"] == "checking"
        waiting.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert runtime.state.list_attempts()[0]["outcome"] == "cancelled_after_fire"
        assert runtime.source.pending() == ()
