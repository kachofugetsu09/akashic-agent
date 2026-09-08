from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import agent.plugins.manager as plugin_manager_module
import plugins.wake.message_plugin as wake_plugin_module
from agent.control.timer import TimerReceipt, TimerStatus
from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CHAT_MODELS,
    CapabilitySources,
    LLMResponse,
    ModelCapabilities,
    ModelKind,
    ModelRole,
    SetDefaultModel,
    ToolCall,
)
from agent.plugins.manager import PluginManager
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from session.log import MessageLog
from plugins.wake.request import Request, read_request
from plugins.wake.source import Pointer
from tests.fixtures.content_clock_source.plugin import FixtureSourceStore
from tests.model_plugin_fakes import (
    register_test_model_provider,
    unregister_test_model_provider,
)

MODEL = os.environ.get("PR_G_DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
_SELECTED_CONTEXT_WINDOW = 1_000_000
_SELECTED_REASONING_EFFORT = "max"
_OLD_ISLAND_NAMES = frozenset(
    {
        "proactive.db",
        "wake_proactive.db",
        "PROACTIVE_CONTEXT.md",
        "proactive_pending.md",
        "proactive_quota.json",
    }
)
_PROTECTED_RELATIVE_TARGETS = (
    Path("sessions.db"),
    Path("proactive.db"),
    Path("wake_proactive.db"),
    Path("drift/drift.db"),
    Path("PROACTIVE_CONTEXT.md"),
    Path("proactive_pending.md"),
    Path("proactive_quota.json"),
)
_RUNTIME_FAILURE_CODES = {
    "formal_before": "FORMAL_BASELINE_RUNTIME_ERROR",
    "deterministic_recovery": "DETERMINISTIC_RECOVERY_RUNTIME_ERROR",
    "deterministic_quiet": "DETERMINISTIC_QUIET_RUNTIME_ERROR",
    "credential": "CREDENTIAL_RUNTIME_ERROR",
    "selected_chain": "SELECTED_PROVIDER_OR_CHAIN_RUNTIME_ERROR",
    "formal_after": "FORMAL_AFTER_RUNTIME_ERROR",
    "selected_oracles": "SELECTED_ORACLE_RUNTIME_ERROR",
}


class GateFailure(RuntimeError):
    """Represent one report-safe E2E contract failure."""

    def __init__(self, code: str, *, stage: str = "unassigned") -> None:
        self.code = code
        self.stage = stage
        super().__init__(code)


class SafeRuntimeFailure(RuntimeError):
    """Carry only a fixed failure code and stage across the report boundary."""

    def __init__(self, code: str, stage: str) -> None:
        self.code = code
        self.stage = stage
        super().__init__(code)


class _FixtureSettlementInterruption(RuntimeError):
    """Mark only the deliberate recovery interruption in the isolated fixture."""


class ControlledTimerHandle:
    def __init__(self, timer_id: str, deadline: datetime) -> None:
        self._id = timer_id
        self.deadline = deadline
        self._future: asyncio.Future[TimerReceipt] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def pending(self) -> bool:
        return not self._future.done()

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self._future)

    async def cancel(self) -> TimerReceipt:
        if not self._future.done():
            self._future.set_result(self._receipt(TimerStatus.CANCELLED))
        return await self._future

    async def cleanup(self) -> None:
        _ = await self.cancel()

    def fire(self) -> None:
        if self._future.done():
            raise RuntimeError(f"timer 已终结: {self.id}")
        self._future.set_result(self._receipt(TimerStatus.FIRED))

    def _receipt(self, status: TimerStatus) -> TimerReceipt:
        return TimerReceipt(
            self.id,
            self.deadline,
            datetime.now(UTC),
            status,
        )


class ControlledTimer:
    """Expose deterministic firing while preserving the ordinary Timer protocol."""

    def __init__(self) -> None:
        self.handles: list[ControlledTimerHandle] = []

    def schedule(self, deadline: datetime) -> ControlledTimerHandle:
        handle = ControlledTimerHandle(f"timer:e2e:{len(self.handles) + 1}", deadline)
        self.handles.append(handle)
        return handle

    def fire_earliest(self) -> None:
        pending = [handle for handle in self.handles if handle.pending]
        if not pending:
            raise RuntimeError("没有可触发的 E2E Timer")
        min(pending, key=lambda handle: (handle.deadline, handle.id)).fire()

    def pending_count(self) -> int:
        return sum(handle.pending for handle in self.handles)


class CountingProvider:
    """Count logical model calls while leaving HTTP retry ownership to the provider."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.logical_requests = 0

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.logical_requests += 1
        chat = getattr(self._delegate, "chat")
        return cast(LLMResponse, await chat(**kwargs))

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class ScriptedProvider:
    """Return one typed result for each Wake Content phase."""

    context_window = 64_000

    def __init__(self, response: str = "E2E wake response") -> None:
        self.response = response
        self.tool_batches: list[tuple[str, ...]] = []

    async def chat(self, **kwargs: object) -> LLMResponse:
        tools = kwargs.get("tools")
        if isinstance(tools, (list, tuple)) and tools:
            prompt = str(kwargs.get("messages"))
            candidate = re.search(r"candidate_[0-9a-f]{16}", prompt)
            if candidate is None:
                raise RuntimeError("Wake E2E prompt 缺少 candidate_id")
            names = {
                str(item.get("function", {}).get("name"))
                for item in tools
                if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
            }
            self.tool_batches.append(tuple(sorted(names)))
            if "screen_content" in names:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call:wake-screen",
                            name="screen_content",
                            arguments={
                                "items": [
                                    {
                                        "candidate_id": candidate.group(0),
                                        "initial_interest": "likely_interesting",
                                        "question": "这是否有用户真正关心的新能力？",
                                    }
                                ]
                            },
                        )
                    ],
                )
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call:wake-share",
                        name="share_content",
                        arguments={
                            "message": self.response,
                            "items": [candidate.group(0)],
                        },
                    )
                ],
            )
        return LLMResponse(content="Wake decision recorded.", tool_calls=[])

    def estimate_context_tokens(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> int:
        return max(1, len(json.dumps([messages, tools], ensure_ascii=False)) // 4)


class ProviderMilestones(logging.Handler):
    """Collect provider attempt identities without retaining prompts or bodies."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[dict[str, object]] = []
        self.nonstream_retries = 0

    def emit(self, record: logging.LogRecord) -> None:
        fields = getattr(record, "akashic_fields", None)
        if not isinstance(fields, dict):
            return
        fields = cast(dict[str, object], fields)
        event = fields.get("event")
        if isinstance(event, str) and event.startswith("tl:provider."):
            self.events.append(
                {
                    "event": event,
                    "turn_id": fields.get("turn_id"),
                    "counts": fields.get("counts"),
                }
            )

    def http_attempts(self) -> int:
        attempts: set[tuple[str, str]] = set()
        for event in self.events:
            if event["event"] != "tl:provider.http.start":
                continue
            counts = str(event.get("counts") or "")
            attempts.add((_field(counts, "span_id"), _field(counts, "http_attempt")))
        if attempts:
            return len(attempts)
        nonstream_starts = sum(
            event["event"] == "tl:provider.nonstream.start" for event in self.events
        )
        return nonstream_starts + self.nonstream_retries

    def safe_evidence(self) -> dict[str, object]:
        """Summarize provider identities as counts and optional single digests."""

        starts = [
            event
            for event in self.events
            if event["event"]
            in {
                "tl:provider.call.start",
                "tl:provider.transport.start",
                "tl:provider.http.start",
            }
        ]
        call_ids = {
            _field(str(event.get("counts") or ""), "provider_call_id")
            for event in starts
        }
        call_ids.discard("")
        turn_ids = {str(event.get("turn_id") or "") for event in starts}
        turn_ids.discard("")
        return {
            "http_attempts": self.http_attempts(),
            "provider_terminal_counts": self.terminal_counts(),
            "provider_call_identity_count": len(call_ids),
            "provider_call_id_digest": (
                _digest_text(next(iter(call_ids))) if len(call_ids) == 1 else None
            ),
            "provider_control_identity_count": len(turn_ids),
            "provider_control_id_digest": (
                _digest_text(next(iter(turn_ids))) if len(turn_ids) == 1 else None
            ),
        }

    def terminal_counts(self) -> dict[str, int]:
        """Count only fixed provider terminal event classes."""

        allowed = (
            "tl:provider.call.done",
            "tl:provider.call.error",
            "tl:provider.call.cancelled",
            "tl:provider.nonstream.done",
            "tl:provider.nonstream.error",
            "tl:provider.nonstream.cancelled",
        )
        return {
            event.removeprefix("tl:provider.").replace(".", "_"): sum(
                item["event"] == event for item in self.events
            )
            for event in allowed
        }


@dataclass
class RuntimeStack:
    workspace: Path
    timer: ControlledTimer
    provider: CountingProvider
    event_bus: EventBus
    message_log: MessageLog
    manager: PluginManager
    after_load: Callable[[], Awaitable[None]] | None = None
    uses_test_model: bool = True

    async def start(self) -> None:
        await self.manager.load_all()
        if self.after_load is not None:
            await self.after_load()
        await self.manager.start_runtime()

    async def close(self) -> None:
        """Close every isolated runtime owner while preserving its durable workspace."""

        try:
            await self.manager.terminate_all()
        finally:
            try:
                await self.event_bus.aclose()
            finally:
                self.message_log.close()
                if self.uses_test_model:
                    unregister_test_model_provider(self.workspace)


async def run_suite(
    root: Path,
    *,
    provider: object,
    request_counter: CountingProvider | None = None,
    model_plugin_dirs: tuple[Path, ...] = (),
    inject_settlement_failure: bool = False,
    ack_failures: int = 0,
) -> dict[str, object]:
    """Run one formal Wake chain and return only identity/state evidence."""

    # 1. Seed only the fixture-owned external source and plugin configuration.
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    receipt_db = workspace / "recording-receipts.sqlite3"
    _write_plugin_configs(workspace, receipt_db)
    source_store = FixtureSourceStore(
        workspace / "plugin-data" / "content_clock_source-builtin" / "source.sqlite3"
    )
    seeded_at = datetime.now(UTC)
    source_store.seed(
        (
            {
                "kind": "fixture",
                "wake_action": "select",
                "preprocess_score": 0.9,
                "published_at": seeded_at.isoformat(),
            },
        ),
        seeded_at,
    )
    source_store.fail_next_acks(ack_failures)
    counted = request_counter or CountingProvider(provider)
    timer = ControlledTimer()
    original_timer = plugin_manager_module.AsyncioOneShotTimer
    plugin_manager_module.AsyncioOneShotTimer = lambda: timer
    settlement_failures = 0

    first: RuntimeStack | None = None
    restarted: RuntimeStack | None = None
    try:
        # 2. Install through the formal manager and run the ordinary source Timer.
        if model_plugin_dirs:
            # Model settings are durable, while a running Root keeps the exact
            # plugin generation that was loaded before the settings write.  Seed
            # the registry in a short bootstrap Root, then run the chain against
            # a fresh Root that loads the committed binding and its driver
            # together.
            bootstrap = _build_stack(
                workspace,
                root,
                timer,
                counted,
                model_plugin_dirs=model_plugin_dirs,
            )
            try:
                await bootstrap.manager.load_all()
                if bootstrap.after_load is None:
                    raise GateFailure("MODEL_BOOTSTRAP_CONFIG_MISSING")
                await bootstrap.after_load()
            finally:
                await bootstrap.close()
            first = _build_stack(
                workspace,
                root,
                timer,
                counted,
                model_plugin_dirs=model_plugin_dirs,
                configure_selected_model=False,
            )
        else:
            first = _build_stack(
                workspace,
                root,
                timer,
                counted,
                model_plugin_dirs=model_plugin_dirs,
            )
        await first.start()
        if inject_settlement_failure:
            snapshot = first.manager.current_snapshot
            if snapshot is None or snapshot.composition_root is None:
                raise GateFailure("EVENTMAIL_SETTLEMENT_SERVICE_MISSING")
            delivery_service = cast(
                Any,
                snapshot.composition_root.context.require(
                    wake_plugin_module.EVENTMAIL_DELIVERY
                ),
            )

            def fail_before_restart(
                selection_token: str,
                settlement_ref: str,
            ) -> dict[str, object]:
                nonlocal settlement_failures
                del selection_token, settlement_ref
                settlement_failures += 1
                raise _FixtureSettlementInterruption()

            delivery_service.settle = fail_before_restart
        await _eventually(lambda: timer.pending_count() >= 1, "SOURCE_TIMER_NOT_ARMED")
        timer.fire_earliest()
        await _eventually(
            lambda: source_store.state(datetime.now(UTC))["cursor"] == 1,
            "SOURCE_CURSOR_NOT_COMMITTED",
        )
        await _eventually(lambda: timer.pending_count() >= 1, "WAKE_TIMER_NOT_ARMED")
        timer.fire_earliest()
        terminal = "delivered"
        await _eventually(
            lambda: _delivery_state(workspace) == terminal,
            (
                "DELIVERY_PRE_RESTART_NOT_TERMINAL"
                if inject_settlement_failure
                else "SELECTED_DELIVERY_NOT_TERMINAL"
            ),
        )

        # 3. A projected interruption restarts the formal stack and only moves forward.
        if inject_settlement_failure:
            try:
                await first.close()
            except BaseException as error:
                # The injected domain failure is surfaced by the Wake watcher during stop;
                # its durable owner records remain the recovery evidence.
                if not _is_fixture_settlement_failure(error):
                    raise
            first = None
            restarted = _build_stack(
                workspace,
                root,
                timer,
                counted,
                model_plugin_dirs=model_plugin_dirs,
            )
            await restarted.start()
            await _eventually(
                lambda: _delivery_state(workspace) == "delivered",
                "DELIVERY_RESTART_NOT_SETTLED",
            )

        # 4. Drive competing source and Wake heartbeats until source ACK settles.
        max_fires = 2 * (ack_failures + 1) + 2
        for _ in range(max_fires):
            if source_store.acknowledgements():
                break
            await _eventually(lambda: timer.pending_count() >= 1, "ACK_TIMER_NOT_ARMED")
            timer.fire_earliest()
            await asyncio.sleep(0)
        await _eventually(
            lambda: len(source_store.acknowledgements()) == 1,
            "SOURCE_ACK_NOT_COMMITTED",
        )
        await _eventually(
            lambda: _content_state_counts(workspace) == {"settled": 1},
            "CONTENT_NOT_SETTLED",
        )

        # 5. Read every oracle from its durable owner, never from callback counters alone.
        active = first if first is not None else restarted
        if active is None:
            raise GateFailure("RUNTIME_STACK_MISSING")
        request, pointer = _wake_request(active.message_log)
        channel_rows = _rows(receipt_db, "deliveries")
        target_messages = active.message_log.reader(request.target.session_id).snapshot()
        delivery = _single_delivery(workspace)
        if len(channel_rows) != 1 or len(target_messages) != 1:
            raise GateFailure("DURABLE_ORACLE_MULTIPLICITY_MISMATCH")
        if target_messages[0].message_id != request.notification_id:
            raise GateFailure("DELIVERY_NOTIFICATION_MISMATCH")
        if pointer.settled is not True:
            raise GateFailure("WAKE_POINTER_NOT_SETTLED")
        identities = {str(delivery["message_id"]), str(channel_rows[0]["delivery_id"]), request.notification_id}
        if len(identities) != 1:
            raise GateFailure("DELIVERY_IDENTITY_MISMATCH")
        if channel_rows[0]["recipient"] != request.target.recipient:
            raise GateFailure("DELIVERY_RECIPIENT_MISMATCH")
        model_evidence: dict[str, object] = {}
        if model_plugin_dirs:
            catalog = await RuntimeModelControl(active.manager.snapshot_store).catalog()
            async with lease_runtime_snapshot(active.manager.snapshot_store) as snapshot:
                composition_root = snapshot.composition_root
                if composition_root is None:
                    raise GateFailure("MODEL_SNAPSHOT_ROOT_MISSING")
                chat_models = composition_root.context.require(CHAT_MODELS)
                async with chat_models.execution() as execution:
                    selected_model = execution.chat(ModelRole.DEFAULT)
                    model_evidence = {
                        "revision": catalog.revision,
                        "model_id": selected_model.descriptor.model_id,
                        "driver_id": selected_model.descriptor.driver_id,
                        "snapshot_id": selected_model.descriptor.plugin_snapshot_id,
                    }
        return {
            "model": MODEL,
            "logical_provider_requests": counted.logical_requests,
            "delivery_count": len(channel_rows),
            "session_projection_count": len(target_messages),
            "content_counts": {"settled": 1},
            "source_ack_count": len(source_store.acknowledgements()),
            "source_ack_attempts": _source_count(source_store, "ack_attempts"),
            "content_submission_count": len(
                _rows(
                    workspace
                    / "plugin-data"
                    / "eventmail-builtin"
                    / "eventmail.sqlite3",
                    "submissions",
                )
            ),
            "delivery_id_digest": _digest_text(next(iter(identities))),
            "wake_session_id_digest": _digest_text(request.session_id),
            "notification_id_digest": _digest_text(request.notification_id),
            "settlement_failure_count": settlement_failures,
            "final_state": str(delivery["state"]),
            "restart_count": int(inject_settlement_failure),
            "model_binding": model_evidence,
        }
    finally:
        plugin_manager_module.AsyncioOneShotTimer = original_timer
        if first is not None:
            await first.close()
        if restarted is not None:
            await restarted.close()


def _build_stack(
    workspace: Path,
    root: Path,
    timer: ControlledTimer,
    provider: CountingProvider,
    *,
    model_plugin_dirs: tuple[Path, ...] = (),
    configure_selected_model: bool = True,
) -> RuntimeStack:
    """Assemble the formal MessageLog, Wake, Models, and Delivery chain."""

    event_bus = EventBus()
    message_log = MessageLog(workspace / "sessions.db")

    plugin_dirs = [
        Path(__file__).resolve().parents[2] / "plugins" / name
        for name in (
            "content",
            "context",
            "delivery",
            "drift",
            "eventmail",
            "react",
            "tools",
            "turn_projection",
            "wake",
        )
    ] + [
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / name
        for name in (
            "content_clock_source",
            "memory_recall",
            "semantic_interest",
        )
    ]
    fixture_plugin = _write_e2e_fixture_plugin(root, include_models=not model_plugin_dirs)
    plugin_dirs.append(fixture_plugin)
    plugin_dirs.extend(model_plugin_dirs)
    manager = PluginManager(
        plugin_dirs=plugin_dirs,
        event_bus=event_bus,
        workspace=workspace,
        message_log=message_log,
        installed_cache_root=root / "plugin-home" / "cache",
    )
    if not model_plugin_dirs:
        provider.model = MODEL
        register_test_model_provider(workspace, provider)
    return RuntimeStack(
        workspace,
        timer,
        provider,
        event_bus,
        message_log,
        manager,
        after_load=(
            (
                (lambda: _configure_selected_model(manager))
                if configure_selected_model
                else None
            )
            if model_plugin_dirs
            else None
        ),
        uses_test_model=not model_plugin_dirs,
    )


def _write_e2e_fixture_plugin(root: Path, *, include_models: bool) -> Path:
    """Create the isolated provider/tool boundary required by the formal chain."""

    directory = root / ("wake_e2e_models" if include_models else "wake_e2e_support")
    directory.mkdir(parents=True, exist_ok=True)
    models = """
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from agent.plugin_composition import CHAT_MODELS
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, ModelCapabilities, ModelRole
from plugins.models.projection import MODEL_CALLS
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from tests.model_plugin_fakes import _MODEL_PROVIDERS
""" if include_models else ""
    model_apply = """
    provider = _MODEL_PROVIDERS.get(Path(ctx.runtime.workspace).resolve())
    if provider is None:
        raise RuntimeError(f"E2E model provider 未注册: {ctx.runtime.workspace}")
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return max(1, len(str((messages, tools))) // 4)
        def estimate_appended_message_tokens(self, messages):
            return max(1, len(str(messages)) // 4)
        async def complete(self, request):
            return await provider.chat(
                messages=request.messages,
                tools=request.tools,
                model=descriptor.model,
                max_tokens=request.max_output_tokens,
                tool_choice=request.tool_choice,
                disable_thinking=request.disable_reasoning,
                on_content_delta=request.on_delta,
                cache_namespace=request.prompt_cache_key,
                **({"model_state": request.continuation.payload}
                   if request.continuation is not None else {}),
            )
    descriptor = BoundModelDescriptor(
        binding_id="wake-e2e-fixture-model", plugin_snapshot_id="wake-e2e-fixture",
        model_revision=1, model_id="wake-e2e-fixture", connection_id="fixture",
        driver_id="fixture", driver_contract_version="1", auth_identity="fixture",
        model=getattr(provider, "model", "wake-e2e-fixture"), role=ModelRole.AGENT,
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=64_000),
        capability_sources=CapabilitySources(), capability_digest="wake-e2e-fixture",
    )
    model = _BoundChat(descriptor, Driver(), store)
    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            del model_id, reasoning_effort
            yield SimpleNamespace(chat=lambda role: model)
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
""" if include_models else ""
    text = f'''from contextlib import asynccontextmanager, closing
from agent.plugin_composition import Context
from plugins.akasha.interest import SEMANTIC_INTEREST
from plugins.akasha.message_plugin import AKASHA_TOOLS
from plugins.delivery.api import Receipt
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.standard_web.plugin import STANDARD_WEB_TOOLS
from plugins.tools.api import Result
from plugins.tools.plugin import TOOLS
from session.message import ContentPart
from session.message_codec import encode_body
{models}
api_version = 3
name = "wake_e2e_{"models" if include_models else "support"}"
version = "1.0.0"
inject = (TOOLS, DELIVERY_SENDERS)

class ZeroSemanticInterest:
    async def score(self, texts, *, cutoff):
        del cutoff
        return tuple(0.0 for _ in texts)

class NoopTool:
    idempotent = True
    async def prepare(self, arguments, source=None):
        del source
        return dict(arguments)
    async def invoke(self, key, arguments):
        del key, arguments
        return Result("success", (ContentPart("text", "fixture tool result"),))
    async def query(self, key):
        del key
        return None

async def apply(ctx: Context, config: object):
    del config
    await ctx.provide(SEMANTIC_INTEREST, ZeroSemanticInterest())
    @asynccontextmanager
    async def open_tool(state):
        del state
        yield NoopTool()
    refs = {{}}
    for name in ("recall_memory", "web_fetch"):
        refs[name] = await ctx.require(TOOLS).register(
            ctx, name=name, description="isolated Wake E2E fixture tool",
            parameters={{"type": "object", "additionalProperties": True}},
            open=open_tool, idempotent=True, public=False,
        )
    await ctx.provide(AKASHA_TOOLS, ctx.require(TOOLS).view(refs["recall_memory"]))
    await ctx.provide(STANDARD_WEB_TOOLS, ctx.require(TOOLS).view(refs["web_fetch"]))
    class RecordingSender:
        idempotent = True
        def __init__(self):
            self.path = ctx.runtime.workspace / "recording-receipts.sqlite3"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            import sqlite3
            with closing(sqlite3.connect(self.path)) as connection:
                with connection:
                    connection.execute("CREATE TABLE IF NOT EXISTS deliveries("
                        "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "delivery_id TEXT NOT NULL UNIQUE, recipient TEXT NOT NULL, "
                        "message_json TEXT NOT NULL, receipt_json TEXT NOT NULL)")
        async def send(self, key, address, message):
            import sqlite3
            import json
            receipt = Receipt(status="delivered", provider_ids=(key,))
            with closing(sqlite3.connect(self.path)) as connection:
                with connection:
                    connection.execute(
                        "INSERT OR IGNORE INTO deliveries(delivery_id, recipient, message_json, receipt_json) VALUES (?, ?, ?, ?)",
                        (message.message_id, address, encode_body(message.body), receipt.model_dump_json()),
                    )
            return receipt
        async def query(self, key, address):
            del key, address
            return None
    @asynccontextmanager
    async def open_sender():
        yield RecordingSender()
    await ctx.require(DELIVERY_SENDERS).register(
        ctx, name="recording", idempotent=True, open=open_sender,
    )
{model_apply}'''
    path = directory / "plugin.py"
    path.write_text(text, encoding="utf-8")
    return directory


def _copy_selected_model_plugins(root: Path) -> tuple[Path, Path]:
    """Copy the real model store and HTTP driver with an archive-visible edge."""

    external = root / "external-model-plugins"
    models = external / "models"
    provider = external / "openai_compatible"
    shutil.copytree(_SOURCE_ROOT / "plugins" / "models", models)
    shutil.copytree(_SOURCE_ROOT / "plugins" / "openai_compatible", provider)
    marker = 'ServiceKey("wake-e2e.openai-provider.v1")'
    plugin = models / "plugin.py"
    text = plugin.read_text(encoding="utf-8")
    text = text.replace(
        "from agent.plugin_composition import (\n",
        "from agent.plugin_composition import (\n    ServiceKey,\n",
    )
    text = text.replace("inject = ()", f"inject = ({marker},)")
    plugin.write_text(text, encoding="utf-8")
    plugin = provider / "plugin.py"
    text = plugin.read_text(encoding="utf-8")
    text = text.replace(
        "from agent.plugin_composition import MODEL_DRIVERS, Context\n",
        "from agent.plugin_composition import MODEL_DRIVERS, SNAPSHOT_SEALING, Context, ServiceKey\n",
    )
    text = text.replace("inject = (MODEL_DRIVERS,)", "inject = ()")
    text = text.replace(
        "    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())\n",
        "    await ctx.provide(ServiceKey(\"wake-e2e.openai-provider.v1\"), object())\n"
        "    async def register(_event: object) -> None:\n"
        "        _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())\n"
        "    _ = await ctx.on(SNAPSHOT_SEALING, register)\n",
    )
    plugin.write_text(text, encoding="utf-8")
    return models, provider


async def _configure_selected_model(manager: PluginManager) -> None:
    """Configure the real endpoint through the ordinary models service."""

    control = RuntimeModelControl(manager.snapshot_store)
    receipt = await control.apply(
        AddConnection(
            expected_revision=0,
            connection_id="wake-e2e",
            name="Wake E2E",
            driver_id="openai-compatible",
            endpoint=os.environ["PR_G_DEEPSEEK_BASE_URL"].strip(),
            auth_identity="wake-e2e",
            credential={
                "driver": "api_key",
                "access_token": os.environ["PR_G_DEEPSEEK_API_KEY"],
            },
            driver_config={"format_version": 1, "max_retries": 3},
        )
    )
    receipt = await control.apply(
        AddModel(
            expected_revision=receipt.revision,
            model_id="wake-e2e-model",
            connection_id="wake-e2e",
            kind=ModelKind.CHAT,
            model=MODEL,
            default_reasoning_effort=_SELECTED_REASONING_EFFORT,
            capabilities=ModelCapabilities(
                context_window=_SELECTED_CONTEXT_WINDOW,
                input_modalities=("text",),
                supports_tool_calls=True,
                supported_reasoning_efforts=(_SELECTED_REASONING_EFFORT,),
            ),
            capability_sources=CapabilitySources(context_window="e2e-profile"),
        )
    )
    for role in (ModelRole.DEFAULT, ModelRole.FAST, ModelRole.AGENT):
        receipt = await control.apply(
            SetDefaultModel(receipt.revision, role, "wake-e2e-model")
        )


def _write_plugin_configs(workspace: Path, receipt_db: Path) -> None:
    """Write only isolated plugin-local configuration needed by the fixture chain."""

    wake = workspace / "plugin-data" / "wake-builtin"
    recording = workspace / "plugin-data" / "recording_channel-builtin"
    wake.mkdir(parents=True)
    recording.mkdir(parents=True)
    _ = (wake / "config.local.toml").write_text(
        '[delivery]\nchannel = "recording"\n'
        'recipient = "fixture-recipient"\n'
        'session_id = "wake-provider-e2e"\n',
        encoding="utf-8",
    )
    escaped = str(receipt_db).replace("\\", "\\\\").replace('"', '\\"')
    _ = (recording / "config.local.toml").write_text(
        f'receipt_db = "{escaped}"\ntoken = "isolated-fixture-token"\n',
        encoding="utf-8",
    )


async def _eventually(
    predicate: Callable[[], bool], code: str, *, timeout: float = 10.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise GateFailure(code)


async def run_quiet_suite(root: Path) -> dict[str, object]:
    """Prove a declined Content duty stays inside the diagnostic control Turn."""

    # 1. Install the same formal chain with a declined source fact.
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    _write_plugin_configs(workspace, workspace / "recording-receipts.sqlite3")
    source_store = FixtureSourceStore(
        workspace / "plugin-data" / "content_clock_source-builtin" / "source.sqlite3"
    )
    seeded_at = datetime.now(UTC)
    source_store.seed(
        (
            {
                "kind": "fixture",
                "wake_action": "decline",
                "preprocess_score": 0.9,
                "published_at": seeded_at.isoformat(),
            },
        ),
        seeded_at,
    )
    counted = CountingProvider(ScriptedProvider("unexpected"))
    timer = ControlledTimer()
    original_timer = plugin_manager_module.AsyncioOneShotTimer
    plugin_manager_module.AsyncioOneShotTimer = lambda: timer
    stack: RuntimeStack | None = None
    try:
        stack = _build_stack(workspace, root, timer, counted)
        await stack.start()
        await _eventually(
            lambda: timer.pending_count() >= 1, "QUIET_SOURCE_TIMER_NOT_ARMED"
        )
        timer.fire_earliest()
        await _eventually(
            lambda: source_store.state(datetime.now(UTC))["cursor"] == 1,
            "QUIET_SOURCE_CURSOR_NOT_COMMITTED",
        )
        await _eventually(
            lambda: timer.pending_count() >= 1, "QUIET_WAKE_TIMER_NOT_ARMED"
        )
        timer.fire_earliest()
        await _eventually(
            lambda: any(
                session_id.startswith("wake:")
                for session_id in stack.message_log.catalog().snapshot_heads()
            ),
            "QUIET_CONTROL_TURN_MISSING",
        )

        # 2. One later empty poll creates no duplicate Content history.
        await _eventually(
            lambda: timer.pending_count() >= 1, "QUIET_EMPTY_POLL_TIMER_NOT_ARMED"
        )
        poll_count = _source_count(source_store, "poll_count")
        for _ in range(4):
            if _source_count(source_store, "poll_count") > poll_count:
                break
            await _eventually(
                lambda: timer.pending_count() >= 1,
                "QUIET_EMPTY_POLL_TIMER_NOT_ARMED",
            )
            timer.fire_earliest()
            await asyncio.sleep(0)
        await _eventually(
            lambda: _source_count(source_store, "poll_count") >= 2,
            "QUIET_EMPTY_POLL_NOT_COMMITTED",
        )
        content_db = (
            workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3"
        )
        wake_sessions = [
            session_id
            for session_id in stack.message_log.catalog().snapshot_heads()
            if session_id.startswith("wake:")
        ]
        wake_messages = stack.message_log.reader(wake_sessions[0]).snapshot()
        messages = stack.message_log.reader("wake-provider-e2e").snapshot()
        if (
            len(wake_sessions) != 1
            or len(wake_messages) != 2
            or getattr(wake_messages[-1].body, "finish", None) != "quiet"
            or counted.logical_requests != 0
            or messages
            or _delivery_rows(workspace)
        ):
            raise GateFailure("QUIET_CONTRACT_MISMATCH")
        return {
            "logical_provider_requests": 0,
            "control_turn_count": 1,
            "session_projection_count": 0,
            "delivery_count": 0,
            "content_submission_count": len(_rows(content_db, "submissions")),
            "source_poll_count": _source_count(source_store, "poll_count"),
        }
    finally:
        plugin_manager_module.AsyncioOneShotTimer = original_timer
        if stack is not None:
            await stack.close()


def _delivery_rows(workspace: Path) -> list[dict[str, object]]:
    """Read the Delivery plugin's owner records without recreating old ledger state."""

    rows = _read_failure_rows(
        workspace / "sessions.db",
        "owner_records",
        "SELECT key, value FROM owner_records WHERE owner = ? AND key LIKE 'delivery:%'",
        ("plugin:delivery",),
    )
    result: list[dict[str, object]] = []
    for key, raw in rows:
        if not isinstance(key, str) or not isinstance(raw, str):
            raise GateFailure("DELIVERY_OWNER_RECORD_INVALID")
        try:
            identity = json.loads(key.removeprefix("delivery:"))
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise GateFailure("DELIVERY_OWNER_RECORD_INVALID") from error
        if (
            not isinstance(identity, list)
            or len(identity) != 2
            or not all(isinstance(item, str) and item for item in identity)
            or not isinstance(value, dict)
        ):
            raise GateFailure("DELIVERY_OWNER_RECORD_INVALID")
        sink = value.get("sink")
        if not isinstance(sink, dict) or not isinstance(sink.get("name"), str):
            raise GateFailure("DELIVERY_OWNER_RECORD_INVALID")
        result.append(
            {
                "message_id": identity[0],
                "sink": sink,
                "state": value.get("phase"),
                "receipt": value.get("receipt"),
            }
        )
    return result


def _delivery_state(workspace: Path) -> str:
    rows = _delivery_rows(workspace)
    return "" if not rows else str(rows[0]["state"])


def _source_count(store: FixtureSourceStore, name: str) -> int:
    value = store.state(datetime.now(UTC))[name]
    if not isinstance(value, int):
        raise GateFailure("FIXTURE_SOURCE_COUNT_INVALID")
    return value


def _content_state_counts(workspace: Path) -> dict[str, int]:
    """Read Content item states through a short-lived read-only SQLite handle."""

    rows = _read_failure_rows(
        workspace / "plugin-data/eventmail-builtin/eventmail.sqlite3",
        "items",
        "SELECT status, COUNT(*) FROM items GROUP BY status ORDER BY status",
    )
    return {str(row[0]): _evidence_int(row[1]) for row in rows}


def _is_fixture_settlement_failure(error: BaseException) -> bool:
    """Accept only the exact injected interruption during the recovery exercise."""

    if isinstance(error, BaseExceptionGroup):
        return bool(error.exceptions) and all(
            _is_fixture_settlement_failure(item) for item in error.exceptions
        )
    return type(error) is _FixtureSettlementInterruption


def _wake_request(log: MessageLog) -> tuple[Request, Pointer]:
    rows = log.owner("plugin:wake").list()
    flows = [(key, row) for key, row in rows if key.startswith("flow:")]
    if len(flows) != 1:
        raise GateFailure("WAKE_POINTER_MULTIPLICITY_MISMATCH")
    _, row = flows[0]
    pointer = Pointer.model_validate(dict(row.value))
    request = read_request(log.reader(pointer.session_id).snapshot())
    if request.input_id != pointer.input_id:
        raise GateFailure("WAKE_POINTER_REQUEST_MISMATCH")
    return request, pointer


def _single_delivery(workspace: Path) -> dict[str, object]:
    rows = _delivery_rows(workspace)
    if len(rows) != 1:
        raise GateFailure("DURABLE_DELIVERY_MULTIPLICITY_MISMATCH")
    return rows[0]


def _rows(path: Path, table: str) -> list[dict[str, object]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
    finally:
        connection.close()


def _read_failure_rows(
    path: Path, table: str, query: str, parameters: tuple[object, ...] = ()
) -> list[tuple[object, ...]]:
    """Read an optional isolated oracle without creating a missing database."""

    if not path.is_file():
        return []
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if exists is None:
            return []
        return [tuple(row) for row in connection.execute(query, parameters)]
    finally:
        connection.close()


def _evidence_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GateFailure("ISOLATED_EVIDENCE_COUNT_INVALID")
    return value


def _selected_failure_evidence(
    root: Path,
    milestones: ProviderMilestones,
) -> dict[str, object]:
    """Read safe selected-chain identities and counts after any terminal outcome."""

    # 1. Read only identity/state columns from each isolated durable owner.
    workspace = root / "workspace"
    delivery_rows = _delivery_rows(workspace) if workspace.is_dir() else []
    channel_rows = _read_failure_rows(
        workspace / "recording-receipts.sqlite3",
        "deliveries",
        "SELECT delivery_id, recipient FROM deliveries",
    )
    session_rows = _read_failure_rows(
        workspace / "sessions.db",
        "messages",
        "SELECT id, session_key, body FROM messages",
    )
    model_call_rows = _read_failure_rows(
        workspace / "model-registry.sqlite3",
        "model_calls",
        "SELECT id, state, failure FROM model_calls",
    )
    content_rows = _read_failure_rows(
        workspace / "plugin-data/eventmail-builtin/eventmail.sqlite3",
        "items",
        "SELECT status, COUNT(*) FROM items GROUP BY status ORDER BY status",
    )
    ack_rows = _read_failure_rows(
        workspace / "plugin-data/content_clock_source-builtin/source.sqlite3",
        "acknowledgements",
        "SELECT settlement_ref FROM acknowledgements",
    )
    source_rows = _read_failure_rows(
        workspace / "plugin-data/content_clock_source-builtin/source.sqlite3",
        "source_state",
        "SELECT ack_attempts FROM source_state WHERE singleton = 1",
    )

    # 2. Decode only the fixed Wake failure envelope; never return provider text.
    wake_failure_retryable: list[bool] = []
    wake_message_count = 0
    target_message_count = 0
    for message_id, session_key, raw_body in session_rows:
        if not isinstance(session_key, str) or not isinstance(raw_body, str):
            raise GateFailure("SESSION_MESSAGE_ROW_INVALID")
        if session_key.startswith("wake:"):
            wake_message_count += 1
        if session_key == "wake-provider-e2e":
            target_message_count += 1
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise GateFailure("SESSION_MESSAGE_BODY_INVALID") from error
        if not isinstance(body, dict) or body.get("kind") != "control":
            continue
        if body.get("action") != "failure":
            continue
        reason = body.get("reason")
        if not isinstance(reason, str):
            raise GateFailure("WAKE_FAILURE_REASON_INVALID")
        try:
            envelope = json.loads(reason)
        except json.JSONDecodeError as error:
            raise GateFailure("WAKE_FAILURE_REASON_INVALID") from error
        if not isinstance(envelope, dict) or envelope.get("kind") != "wake.failure.v1":
            raise GateFailure("WAKE_FAILURE_REASON_INVALID")
        retryable = envelope.get("retryable")
        if not isinstance(retryable, bool):
            raise GateFailure("WAKE_FAILURE_REASON_INVALID")
        wake_failure_retryable.append(retryable)

    # 3. Cross-owner delivery identities leave the process only as cardinality and digest.
    delivery_ids = {
        str(value) for value in (
            *(row["message_id"] for row in delivery_rows),
            *(row[0] for row in channel_rows),
        )
        if value is not None
    }
    model_call_ids = {str(row[0]) for row in model_call_rows}
    model_call_states = {str(row[1]) for row in model_call_rows}
    model_call_failures = {str(row[2]) for row in model_call_rows if row[2] is not None}
    provider_evidence = milestones.safe_evidence()
    # The selected external driver records calls in the models owner but does
    # not emit the legacy passive-turn provider log events.  Use that durable
    # call ledger as the provider identity source for this path.
    if not provider_evidence["provider_call_identity_count"] and model_call_ids:
        provider_evidence["provider_call_identity_count"] = len(model_call_ids)
        provider_evidence["provider_call_id_digest"] = _digest_text(
            "\x00".join(sorted(model_call_ids))
        )
        terminal_counts = provider_evidence["provider_terminal_counts"]
        if not isinstance(terminal_counts, dict):
            raise GateFailure("PROVIDER_TERMINAL_EVIDENCE_INVALID")
        provider_evidence["provider_terminal_counts"] = {
            **terminal_counts,
            "call_done": sum(state == "success" for state in model_call_states),
            "call_error": sum(state == "unknown" for state in model_call_states),
        }
    return {
        **provider_evidence,
        "logical_provider_requests": sum(
            item.get("event") == "tl:provider.call.start" for item in milestones.events
        ),
        "delivery_count": len(delivery_rows),
        "delivery_state_counts": {
            state: sum(str(row["state"]) == state for row in delivery_rows)
            for state in sorted({str(row["state"]) for row in delivery_rows})
        },
        "channel_receipt_count": len(channel_rows),
        "message_count": len(session_rows),
        "wake_message_count": wake_message_count,
        "session_projection_count": target_message_count,
        "wake_failure_count": len(wake_failure_retryable),
        "wake_failure_retryable_counts": {
            label: sum(value is expected for value in wake_failure_retryable)
            for expected, label in ((False, "false"), (True, "true"))
        },
        "model_call_count": len(model_call_rows),
        "model_call_identity_count": len(model_call_ids),
        "model_call_id_digest": (
            _digest_text("\x00".join(sorted(model_call_ids)))
            if model_call_ids
            else None
        ),
        "model_call_state_counts": {
            state: sum(str(row[1]) == state for row in model_call_rows)
            for state in sorted(model_call_states)
        },
        "model_call_failure_type_count": len(model_call_failures),
        "model_call_failure_type_digest": (
            _digest_text(next(iter(model_call_failures)))
            if len(model_call_failures) == 1
            else None
        ),
        "content_counts": {str(row[0]): _evidence_int(row[1]) for row in content_rows},
        "source_ack_count": len(ack_rows),
        "source_ack_attempts": _evidence_int(source_rows[0][0]) if source_rows else 0,
        "delivery_identity_count": len(delivery_ids),
        "delivery_id_digest": (
            _digest_text(next(iter(delivery_ids))) if len(delivery_ids) == 1 else None
        ),
    }


def _empty_selected_evidence() -> dict[str, object]:
    return _selected_failure_evidence(
        Path("/nonexistent"),
        ProviderMilestones(),
    )


def snapshot_protected_workspace(path: Path) -> dict[str, object]:
    """Read the protected Session and old-island targets without workspace writes."""

    if not path.is_dir():
        raise GateFailure("PROTECTED_WORKSPACE_MISSING")
    targets = tuple(
        path / relative
        for relative in _PROTECTED_RELATIVE_TARGETS
        if (path / relative).is_file()
    )
    sqlite_state: dict[str, object] = {}
    for candidate in targets:
        if candidate.suffix in {".db", ".sqlite", ".sqlite3"}:
            sqlite_state[str(candidate.relative_to(path))] = _sqlite_state(candidate)
    files: dict[str, object] = {}
    old_island: dict[str, object] = {}
    for candidate in targets:
        relative = str(candidate.relative_to(path))
        item: dict[str, object] = {
            "inode": candidate.stat().st_ino,
            "size": candidate.stat().st_size,
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
        files[relative] = item
        if candidate.name in _OLD_ISLAND_NAMES:
            old_island[relative] = item
    return {"files": files, "sqlite": sqlite_state, "old_island": old_island}


def _sqlite_state(path: Path) -> dict[str, object]:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise GateFailure("PROTECTED_SQLITE_INTEGRITY_FAILED")
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise GateFailure("PROTECTED_SQLITE_QUICK_CHECK_FAILED")
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            "integrity": "ok",
            "quick_check": "ok",
            "rows": {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in tables
            },
        }
    finally:
        connection.close()


def _field(counts: str, name: str) -> str:
    prefix = name + "="
    return next(
        (part[len(prefix) :] for part in counts.split() if part.startswith(prefix)),
        "",
    )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot_changes(
    before: dict[str, object], after: dict[str, object]
) -> tuple[dict[str, object], ...]:
    """Describe changed protected paths without exposing their contents."""

    changes: dict[str, set[str]] = {}
    for section, kind in (("files", "digest_or_size"), ("sqlite", "sqlite_state")):
        left = cast(dict[str, object], before[section])
        right = cast(dict[str, object], after[section])
        for path in set(left) | set(right):
            if left.get(path) != right.get(path):
                changes.setdefault(path, set()).add(kind)
    return tuple(
        {"path": path, "types": sorted(kinds)}
        for path, kinds in sorted(changes.items())
    )


def _formal_evidence(
    before_a: dict[str, object],
    before_b: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    """Separate live formal concurrency from isolated product-chain evidence."""

    baseline_changes = _snapshot_changes(before_a, before_b)
    after_changes = _snapshot_changes(before_b, after)
    verified = not baseline_changes and not after_changes
    return {
        "status": "unchanged" if verified else "formal_concurrent_change",
        "deployment_gate_verified": verified,
        "baseline_stable": not baseline_changes,
        "baseline_change_count": len(baseline_changes),
        "baseline_changes": baseline_changes,
        "after_change_count": len(after_changes),
        "after_changes": after_changes,
        "digest": _digest_text(json.dumps(before_b, sort_keys=True)),
        "sqlite_count": len(cast(dict[str, object], before_b["sqlite"])),
        "old_island_archive_count": len(
            cast(dict[str, object], before_b["old_island"])
        ),
    }


def _process_isolation_evidence(protected: Path, isolated: Path) -> dict[str, object]:
    """Prove no open descriptor or environment value points at the formal workspace."""

    protected_text = str(protected)
    formal_fds = 0
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target == protected_text or target.startswith(protected_text + os.sep):
            formal_fds += 1
    formal_env = sum(protected_text in value for value in os.environ.values() if value)
    if formal_fds or formal_env:
        raise GateFailure("PROCESS_FORMAL_REFERENCE_PRESENT")
    return {
        "formal_fd_reference_count": 0,
        "formal_env_reference_count": 0,
        "isolated_data_root_count": 3,
        "isolated_root_digest": _digest_text(str(isolated)),
    }


async def _run(args: argparse.Namespace) -> dict[str, object]:
    """Run deterministic recovery first, then exactly one real selected provider call."""

    protected = Path(args.protected_workspace).resolve()
    milestones = ProviderMilestones()
    provider_loggers = (logging.getLogger("agent.core.passive_turn"),)
    prior_levels = tuple(logger.level for logger in provider_loggers)
    attached = False
    stage = "formal_before"
    before_a: dict[str, object] | None = None
    before_b: dict[str, object] | None = None
    failure: GateFailure | SafeRuntimeFailure | None = None
    report: dict[str, object] = {
        "status": "failed",
        "model": MODEL,
        "selected_evidence": _empty_selected_evidence(),
    }
    try:
        before_a = snapshot_protected_workspace(protected)
        before_b = snapshot_protected_workspace(protected)
        with TemporaryDirectory(prefix="akashic-wake-provider-e2e-") as temporary:
            root = Path(temporary)
            selected_root = root / "selected"
            try:
                isolation = _process_isolation_evidence(protected, root)
                stage = "deterministic_recovery"
                deterministic = await run_suite(
                    root / "deterministic",
                    provider=ScriptedProvider(),
                    inject_settlement_failure=True,
                    ack_failures=1,
                )
                stage = "deterministic_quiet"
                quiet = await run_quiet_suite(root / "quiet")
                stage = "credential"
                api_key = os.environ.get("PR_G_DEEPSEEK_API_KEY", "")
                if not api_key:
                    raise GateFailure("MISSING_DEEPSEEK_CREDENTIAL")
                if not os.environ.get("PR_G_DEEPSEEK_BASE_URL", "").strip():
                    raise GateFailure("MISSING_DEEPSEEK_ENDPOINT")
                model_plugin_dirs = _copy_selected_model_plugins(selected_root)
                # 2. Provider evidence starts after every deterministic gate is green.
                for logger in provider_loggers:
                    logger.addHandler(milestones)
                    logger.setLevel(logging.INFO)
                attached = True
                stage = "selected_chain"
                selected = await run_suite(
                    selected_root,
                    provider=ScriptedProvider(),
                    model_plugin_dirs=tuple(model_plugin_dirs),
                )
                stage = "selected_oracles"
                selected_evidence = _selected_failure_evidence(
                    selected_root, milestones
                )
                model_call_states = selected_evidence["model_call_state_counts"]
                if selected_evidence["model_call_count"] != 2 or model_call_states != {
                    "success": 2
                }:
                    raise GateFailure("SELECTED_MODEL_CALL_LEDGER_MISMATCH")
                selected["logical_provider_requests"] = selected_evidence[
                    "model_call_count"
                ]
                binding = selected.get("model_binding")
                if not isinstance(binding, dict) or (
                    binding.get("model_id") != "wake-e2e-model"
                    or binding.get("driver_id") != "openai-compatible"
                    or int(binding.get("revision", 0)) < 5
                    or not str(binding.get("snapshot_id", ""))
                ):
                    raise GateFailure("SELECTED_PLUGIN_BINDING_MISMATCH")
                selected["provider_control_id_digest"] = None
                report.update(
                    {
                        "selected": selected,
                        "deterministic_recovery": deterministic,
                        "deterministic_quiet": quiet,
                        "http_attempts": milestones.http_attempts(),
                        "provider_call_id_digest": selected_evidence[
                            "model_call_id_digest"
                        ],
                        "process_isolation": isolation,
                    }
                )
            finally:
                report["selected_evidence"] = _selected_failure_evidence(
                    selected_root, milestones
                )
    except GateFailure as error:
        if error.stage == "unassigned":
            error.stage = stage
        failure = error
    except BaseException as error:
        failure = SafeRuntimeFailure(
            _RUNTIME_FAILURE_CODES.get(stage, "E2E_RUNTIME_ERROR"), stage
        )
    finally:
        if attached:
            for logger, level in zip(provider_loggers, prior_levels, strict=True):
                logger.removeHandler(milestones)
                logger.setLevel(level)
        if before_a is not None and before_b is not None:
            try:
                stage = "formal_after"
                after = snapshot_protected_workspace(protected)
                report["protected_workspace"] = _formal_evidence(
                    before_a, before_b, after
                )
            except BaseException:
                report["protected_workspace"] = {
                    "status": "after_unavailable",
                    "deployment_gate_verified": False,
                }
                if failure is None:
                    failure = SafeRuntimeFailure(
                        _RUNTIME_FAILURE_CODES["formal_after"], "formal_after"
                    )
    if failure is None:
        report["status"] = "passed"
        return report
    report.update(
        {
            "status": "failed",
            "error_type": type(failure).__name__,
            "error_category": (
                "contract" if isinstance(failure, GateFailure) else "runtime"
            ),
            "failure_code": failure.code,
            "failure_stage": failure.stage,
        }
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated Wake v3 real-provider E2E"
    )
    _ = parser.add_argument("--protected-workspace", required=True)
    _ = parser.add_argument("--report", required=True)
    return parser


def _main_fallback_report() -> dict[str, object]:
    """Build a fixed-code fallback without retaining the triggering exception."""

    return {
        "status": "failed",
        "model": MODEL,
        "error_type": "SafeRuntimeFailure",
        "error_category": "runtime",
        "failure_code": "UNHANDLED_MAIN_ERROR",
        "failure_stage": "main",
        "selected_evidence": _empty_selected_evidence(),
        "protected_workspace": {
            "status": "after_unavailable",
            "deployment_gate_verified": False,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    report_path = Path(args.report)
    try:
        report = asyncio.run(_run(args))
        exit_code = 0 if report.get("status") == "passed" else 1
    except BaseException:
        report = _main_fallback_report()
        exit_code = 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
