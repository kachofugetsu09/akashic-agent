from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import agent.plugins.manager as plugin_manager_module
import plugins.wake.plugin as wake_plugin_module
from agent.config import load_config
from agent.config_models import Config
from agent.control.models import TurnRequest, TurnStatus
from agent.control.runtime import ConversationRuntime
from agent.control.timer import TimerReceipt, TimerStatus
from agent.model_runtime.types import ToolCall
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from agent.plugins.manager import PluginManager
from agent.provider import LLMProvider, LLMResponse
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bootstrap.control_execution import execute_control_turn
from bootstrap.providers import build_providers
from bootstrap.tools import _dispatch_v3_durable_delivery
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.memory.markdown import build_markdown_memory_runtime
from core.memory.runtime import MemoryRuntime
from plugins.eventmail.store import EventMailStore
from session.manager import SessionManager
from tests.fixtures.content_clock_source.plugin import FixtureSourceStore

MODEL = os.environ.get("PR_G_DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
_SELECTED_CONTEXT_WINDOW = 1_000_000
_SELECTED_REASONING_EFFORT = "max"
_BUILDER_SYSTEM_MARKER = "Wake provider E2E control turn."
_CALLER_SYSTEM_MARKER = "wake-v3-e2e-caller-system"
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

    async def chat(self, **kwargs: object) -> LLMResponse:
        tools = kwargs.get("tools")
        if isinstance(tools, list) and tools:
            prompt = json.dumps(kwargs.get("messages"), ensure_ascii=False)
            candidate = re.search(r"candidate_[0-9a-f]{16}", prompt)
            if candidate is None:
                raise RuntimeError("Wake E2E prompt 缺少 candidate_id")
            names = {
                str(item.get("function", {}).get("name"))
                for item in tools
                if isinstance(item, dict)
                and isinstance(item.get("function"), dict)
            }
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


class FixtureWebFetch(Tool):
    """Expose the production Tool shape with deterministic isolated evidence."""

    name = "web_fetch"
    description = "Fetch a candidate URL for evidence."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: object) -> str:
        _ = kwargs
        return "The update reports benchmark gains but no new model capability."


class ProviderMilestones(logging.Handler):
    """Collect provider attempt identities without retaining prompts or bodies."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[dict[str, object]] = []
        self.nonstream_retries = 0

    def emit(self, record: logging.LogRecord) -> None:
        if (
            record.name == "agent.provider"
            and isinstance(record.msg, str)
            and record.msg.startswith("[llm] 请求失败，将重试")
        ):
            self.nonstream_retries += 1
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

    def logical_identity(
        self,
        expected_calls: int = 1,
        expected_turns: int = 1,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return exact provider call identities bound to one control Turn."""

        # 1. Every logical/transport/HTTP start must retain one provider call id.
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
        turn_ids = tuple(
            dict.fromkeys(str(event.get("turn_id") or "") for event in starts)
        )
        if len(call_ids) != expected_calls or "" in call_ids:
            raise GateFailure("PROVIDER_CALL_IDENTITY_MISMATCH")
        if len(turn_ids) != expected_turns or "" in turn_ids:
            raise GateFailure("PROVIDER_CONTROL_IDENTITY_MISMATCH")
        return tuple(sorted(call_ids)), turn_ids

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
    bus: MessageBus
    event_bus: EventBus
    sessions: SessionManager
    loop: AgentLoop
    conversation: ConversationRuntime
    manager: PluginManager
    dispatch_task: asyncio.Task[None]
    lifecycle_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.manager.load_all()
        self.lifecycle_task = asyncio.create_task(self.manager.run_runtime_services())

    async def close(self) -> None:
        """Close every isolated runtime owner while preserving its durable workspace."""

        if self.lifecycle_task is not None:
            _ = self.lifecycle_task.cancel()
            _ = await asyncio.gather(self.lifecycle_task, return_exceptions=True)
        await self.manager.terminate_all()
        await self.conversation.shutdown()
        self.bus.stop()
        _ = self.dispatch_task.cancel()
        _ = await asyncio.gather(self.dispatch_task, return_exceptions=True)
        await self.event_bus.aclose()
        self.sessions.close()


async def run_suite(
    root: Path,
    *,
    provider: object,
    request_counter: CountingProvider | None = None,
    llm_config: LLMConfig | None = None,
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
        first = _build_stack(workspace, root, timer, counted, llm_config=llm_config)
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
                raise RuntimeError("fixture settlement interruption")

            delivery_service.settle = fail_before_restart
        await _eventually(lambda: timer.pending_count() >= 1, "SOURCE_TIMER_NOT_ARMED")
        timer.fire_earliest()
        await _eventually(
            lambda: source_store.state(datetime.now(UTC))["cursor"] == 1,
            "SOURCE_CURSOR_NOT_COMMITTED",
        )
        await _eventually(lambda: timer.pending_count() >= 1, "WAKE_TIMER_NOT_ARMED")
        timer.fire_earliest()
        ledger = DurableDeliveryStore(
            workspace / "runtime" / "deliveries" / "settlements.sqlite"
        )
        terminal = "projected" if inject_settlement_failure else "settled"
        await _eventually(
            lambda: _delivery_state(ledger) == terminal,
            (
                "DELIVERY_PRE_RESTART_NOT_TERMINAL"
                if inject_settlement_failure
                else "SELECTED_DELIVERY_NOT_TERMINAL"
            ),
        )

        # 3. A projected interruption restarts the formal stack and only moves forward.
        if inject_settlement_failure:
            await first.close()
            first = None
            restarted = _build_stack(
                workspace,
                root,
                timer,
                counted,
                llm_config=llm_config,
            )
            await restarted.start()
            await _eventually(
                lambda: _delivery_state(ledger) == "settled",
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
            lambda: EventMailStore(
                workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3"
            ).state_counts()
            == {"settled": 1},
            "CONTENT_NOT_SETTLED",
        )

        # 5. Read every oracle from its durable owner, never from callback counters alone.
        delivery = _single_delivery(ledger)
        channel_rows = _rows(receipt_db, "deliveries")
        active = first if first is not None else restarted
        if active is None:
            raise GateFailure("RUNTIME_STACK_MISSING")
        session_rows = active.sessions.control_store.fetch_session_messages(
            "wake-provider-e2e"
        )
        turns = active.sessions.control_store.list_turns("wake-provider-e2e")
        if len(channel_rows) != 1 or len(session_rows) != 1 or len(turns) != 2:
            raise GateFailure("DURABLE_ORACLE_MULTIPLICITY_MISMATCH")
        if any(turn.status is not TurnStatus.COMPLETED for turn in turns):
            raise GateFailure("CONTROL_TURN_NOT_COMPLETED")
        accepted_turn_id = str(delivery["accepted_turn_id"])
        turn = next((item for item in turns if item.id == accepted_turn_id), None)
        if turn is None:
            raise GateFailure("DELIVERY_CONTROL_TURN_MISSING")
        identities = {
            str(delivery["logical_delivery_id"]),
            str(channel_rows[0]["delivery_id"]),
            str(session_rows[0]["delivery_id"]),
        }
        if len(identities) != 1:
            raise GateFailure("DELIVERY_IDENTITY_MISMATCH")
        control_ids = {
            str(delivery["accepted_turn_id"]),
            str(channel_rows[0]["control_turn_id"]),
            str(session_rows[0]["control_turn_id"]),
            str(turn.id),
        }
        if len(control_ids) != 1:
            raise GateFailure("CONTROL_IDENTITY_MISMATCH")
        return {
            "model": MODEL,
            "logical_provider_requests": counted.logical_requests,
            "delivery_count": len(channel_rows),
            "session_projection_count": len(session_rows),
            "content_counts": {"settled": 1},
            "source_ack_count": len(source_store.acknowledgements()),
            "source_ack_attempts": _source_count(source_store, "ack_attempts"),
            "content_submission_count": len(
                _rows(
                    workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3",
                    "submissions",
                )
            ),
            "delivery_id_digest": _digest_text(next(iter(identities))),
            "control_id_digest": _digest_text(next(iter(control_ids))),
            "settlement_failure_count": settlement_failures,
            "final_state": str(delivery["state"]),
            "restart_count": int(inject_settlement_failure),
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
    llm_config: LLMConfig | None = None,
) -> RuntimeStack:
    """Assemble the formal plugin, control, react, and Channel runtime chain."""

    bus = MessageBus()
    event_bus = EventBus()
    sessions = SessionManager(workspace)
    tools = ToolRegistry()
    tools.register(FixtureWebFetch(), always_on=True, risk="read-only")
    markdown = build_markdown_memory_runtime(
        workspace=workspace,
        provider=cast(Any, provider),
        model=MODEL,
        event_bus=event_bus,
    )
    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            provider=cast(Any, provider),
            light_provider=cast(Any, provider),
            tools=tools,
            session_manager=sessions,
            workspace=workspace,
            event_bus=event_bus,
            memory_runtime=MemoryRuntime(markdown=markdown),
        ),
        AgentLoopConfig(
            llm=llm_config
            or LLMConfig(
                model=MODEL,
                max_iterations=1,
                tool_search_enabled=False,
                multimodal=False,
            )
        ),
    )
    plugin_dirs = [
        Path(__file__).resolve().parents[2] / "plugins" / name
        for name in ("eventmail", "drift", "wake")
    ] + [
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / name
        for name in (
            "content_clock_source",
            "memory_recall",
            "recording_channel",
            "semantic_interest",
        )
    ]
    manager = PluginManager(
        plugin_dirs=plugin_dirs,
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=root / "plugin-home" / "cache",
    )
    loop.bind_runtime_snapshot_store(manager.snapshot_store)

    async def execute(request: TurnRequest):
        return await execute_control_turn(loop, event_bus, request)

    conversation = ConversationRuntime(sessions.control_store, execute)
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=sessions.control_store.create_session,
        programmatic_session_reader=sessions.control_store.get_session_meta,
    )
    manager.bind_durable_delivery_sender(
        lambda request, started: _dispatch_v3_durable_delivery(
            manager, bus, request, started
        )
    )
    bus.bind_channel_outbound_dispatcher(
        manager.channel_generation_host.dispatch_outbound
    )
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    return RuntimeStack(
        workspace,
        timer,
        provider,
        bus,
        event_bus,
        sessions,
        loop,
        conversation,
        manager,
        dispatch_task,
    )


def _write_selected_runtime_config(root: Path) -> Path:
    """Write a secret-free runtime profile for the formal config loader."""

    # 1. Keep credential values in environment interpolation, never on disk.
    path = root / "selected-runtime.toml"
    _ = path.write_text(
        f"""
[llm]
main = "main"

[llm.runtimes.main]
provider = "deepseek"
model = "{MODEL}"
api_key = "${{PR_G_DEEPSEEK_API_KEY}}"
base_url = "https://runtime-endpoint-injected.invalid/v1"
context_window = 1000000
max_output_tokens = 0
reasoning_effort = "max"
enable_thinking = true

[agent]
system_prompt = "Wake provider E2E control turn."
max_iterations = 1

[agent.tools]
search_enabled = false
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _build_selected_provider(
    root: Path,
    workspace: Path,
) -> tuple[LLMProvider, LLMConfig, Config]:
    """Load and build the selected provider through the production config path."""

    # 1. Parse the isolated profile with the same boundary used by runtime startup.
    config = load_config(_write_selected_runtime_config(root), workspace=workspace)
    endpoint = os.environ["PR_G_DEEPSEEK_BASE_URL"].strip()
    runtime = config.model_runtimes[config.runtime_id]
    runtime = replace(runtime, base_url=endpoint)
    config = replace(
        config,
        base_url=endpoint,
        model_runtimes={**config.model_runtimes, config.runtime_id: runtime},
    )
    if (
        runtime.provider != "deepseek"
        or runtime.model != MODEL
        or runtime.context_window != _SELECTED_CONTEXT_WINDOW
        or runtime.max_output_tokens != 0
        or runtime.reasoning_effort != _SELECTED_REASONING_EFFORT
        or config.extra_body.get("enable_thinking") is not True
        or config.extra_body.get("reasoning_effort") != _SELECTED_REASONING_EFFORT
    ):
        raise GateFailure("SELECTED_RUNTIME_PROFILE_MISMATCH")

    # 2. Reuse production provider construction and project only harness loop limits.
    provider, _, _ = build_providers(config)
    loop_config = LLMConfig(
        model=config.model,
        max_iterations=1,
        max_tokens=config.max_tokens,
        tool_search_enabled=False,
        multimodal=False,
    )
    return provider, loop_config, config


def _write_plugin_configs(workspace: Path, receipt_db: Path) -> None:
    """Write only isolated plugin-local configuration needed by the fixture chain."""

    wake = workspace / "plugin-data" / "wake-builtin"
    recording = workspace / "plugin-data" / "recording_channel-builtin"
    memory = workspace / "memory"
    wake.mkdir(parents=True)
    recording.mkdir(parents=True)
    memory.mkdir(parents=True)
    template = Path(__file__).resolve().parents[2] / "prompts" / "VEDA.md"
    caller_veda = (
        template.read_text(encoding="utf-8").rstrip()
        + "\n\n"
        + _CALLER_SYSTEM_MARKER
        + "\n"
    )
    _ = (memory / "VEDA.md").write_text(caller_veda, encoding="utf-8")
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
            lambda: bool(stack.sessions.control_store.list_turns("wake-provider-e2e")),
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
        turns = stack.sessions.control_store.list_turns("wake-provider-e2e")
        content_db = workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3"
        messages = stack.sessions.control_store.fetch_session_messages(
            "wake-provider-e2e"
        )
        ledger = DurableDeliveryStore(
            workspace / "runtime" / "deliveries" / "settlements.sqlite"
        )
        ledger.initialize()
        if (
            len(turns) != 1
            or turns[0].status is not TurnStatus.COMPLETED
            or turns[0].final_response != ""
            or counted.logical_requests != 0
            or messages
            or ledger.recoverable()
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


def _delivery_state(store: DurableDeliveryStore) -> str:
    store.initialize()
    connection = sqlite3.connect(store.path)
    try:
        rows = connection.execute("SELECT state FROM deliveries").fetchall()
        return "" if not rows else str(rows[0][0])
    finally:
        connection.close()


def _source_count(store: FixtureSourceStore, name: str) -> int:
    value = store.state(datetime.now(UTC))[name]
    if not isinstance(value, int):
        raise GateFailure("FIXTURE_SOURCE_COUNT_INVALID")
    return value


def _single_delivery(store: DurableDeliveryStore) -> dict[str, object]:
    connection = sqlite3.connect(store.path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM deliveries").fetchall()
        if len(rows) != 1:
            raise GateFailure("DURABLE_DELIVERY_MULTIPLICITY_MISMATCH")
        return dict(rows[0])
    finally:
        connection.close()


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
    counter: CountingProvider,
    milestones: ProviderMilestones,
) -> dict[str, object]:
    """Read safe selected-chain identities and counts after any terminal outcome."""

    # 1. Read only identity/state columns from each isolated durable owner.
    workspace = root / "workspace"
    ledger_rows = _read_failure_rows(
        workspace / "runtime/deliveries/settlements.sqlite",
        "deliveries",
        "SELECT logical_delivery_id, accepted_turn_id, state FROM deliveries",
    )
    channel_rows = _read_failure_rows(
        workspace / "recording-receipts.sqlite3",
        "deliveries",
        "SELECT delivery_id, control_turn_id FROM deliveries",
    )
    session_rows = _read_failure_rows(
        workspace / "sessions.db",
        "messages",
        "SELECT json_extract(extra, '$.delivery_id'), "
        "json_extract(extra, '$.control_turn_id') "
        "FROM messages WHERE session_key = ?",
        ("wake-provider-e2e",),
    )
    turn_rows = _read_failure_rows(
        workspace / "sessions.db",
        "turns",
        "SELECT id, status, json_extract(error_json, '$.type'), "
        "json_extract(error_json, '$.retryable'), final_response IS NOT NULL "
        "FROM turns WHERE session_key = ?",
        ("wake-provider-e2e",),
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

    # 2. Cross-owner identities leave the process only as cardinality and digest.
    delivery_ids = {
        str(value)
        for value in (
            *(row[0] for row in ledger_rows),
            *(row[0] for row in channel_rows),
            *(row[0] for row in session_rows),
        )
        if value is not None
    }
    control_ids = {
        str(value)
        for value in (
            *(row[1] for row in ledger_rows),
            *(row[1] for row in channel_rows),
            *(row[1] for row in session_rows),
        )
        if value is not None
    }
    turn_ids = {str(row[0]) for row in turn_rows}
    error_types = {str(row[2]) for row in turn_rows if row[2] is not None}
    retryable_labels: dict[int | None, str] = {
        None: "none",
        0: "false",
        1: "true",
    }
    return {
        **milestones.safe_evidence(),
        "logical_provider_requests": counter.logical_requests,
        "delivery_count": len(ledger_rows),
        "delivery_state_counts": {
            state: sum(str(row[2]) == state for row in ledger_rows)
            for state in sorted({str(row[2]) for row in ledger_rows})
        },
        "channel_receipt_count": len(channel_rows),
        "session_projection_count": len(session_rows),
        "turn_count": len(turn_rows),
        "turn_status_counts": {
            status: sum(str(row[1]) == status for row in turn_rows)
            for status in sorted({str(row[1]) for row in turn_rows})
        },
        "turn_error_type_count": len(error_types),
        "turn_error_type_digest": (
            _digest_text(next(iter(error_types))) if len(error_types) == 1 else None
        ),
        "turn_retryable_counts": {
            label: sum(row[3] == value for row in turn_rows)
            for value, label in retryable_labels.items()
        },
        "turn_final_response_present_count": sum(bool(row[4]) for row in turn_rows),
        "turn_identity_count": len(turn_ids),
        "turn_id_digest": (
            _digest_text(next(iter(turn_ids))) if len(turn_ids) == 1 else None
        ),
        "content_counts": {str(row[0]): _evidence_int(row[1]) for row in content_rows},
        "source_ack_count": len(ack_rows),
        "source_ack_attempts": _evidence_int(source_rows[0][0]) if source_rows else 0,
        "delivery_identity_count": len(delivery_ids),
        "delivery_id_digest": (
            _digest_text(next(iter(delivery_ids))) if len(delivery_ids) == 1 else None
        ),
        "control_identity_count": len(control_ids),
        "control_id_digest": (
            _digest_text(next(iter(control_ids))) if len(control_ids) == 1 else None
        ),
    }


def _empty_selected_evidence() -> dict[str, object]:
    return _selected_failure_evidence(
        Path("/nonexistent"),
        CountingProvider(ScriptedProvider()),
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
    provider_loggers = (
        logging.getLogger("agent.provider"),
        logging.getLogger("agent.core.passive_turn"),
    )
    prior_levels = tuple(logger.level for logger in provider_loggers)
    attached = False
    stage = "formal_before"
    before_a: dict[str, object] | None = None
    before_b: dict[str, object] | None = None
    failure: GateFailure | SafeRuntimeFailure | None = None
    selected_counter = CountingProvider(ScriptedProvider())
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
                selected_workspace = selected_root / "workspace"
                selected_workspace.mkdir(parents=True)
                real_provider, selected_llm, _ = _build_selected_provider(
                    selected_root,
                    selected_workspace,
                )
                # 2. Provider evidence starts after every deterministic gate is green.
                for logger in provider_loggers:
                    logger.addHandler(milestones)
                    logger.setLevel(logging.INFO)
                attached = True
                selected_counter = CountingProvider(real_provider)
                stage = "selected_chain"
                selected = await run_suite(
                    selected_root,
                    provider=real_provider,
                    request_counter=selected_counter,
                    llm_config=selected_llm,
                )
                stage = "selected_oracles"
                if selected["logical_provider_requests"] != 2:
                    raise GateFailure("SELECTED_LOGICAL_REQUEST_COUNT_MISMATCH")
                if milestones.http_attempts() < 1:
                    raise GateFailure("SELECTED_HTTP_ATTEMPT_MISSING")
                provider_call_ids, provider_turn_ids = milestones.logical_identity(2, 2)
                if (
                    _digest_text(provider_turn_ids[-1])
                    != selected["control_id_digest"]
                ):
                    raise GateFailure("SELECTED_PROVIDER_CONTROL_IDENTITY_MISMATCH")
                report.update(
                    {
                        "selected": selected,
                        "deterministic_recovery": deterministic,
                        "deterministic_quiet": quiet,
                        "http_attempts": milestones.http_attempts(),
                        "provider_call_id_digest": _digest_text(
                            "\x00".join(provider_call_ids)
                        ),
                        "process_isolation": isolation,
                    }
                )
            finally:
                report["selected_evidence"] = _selected_failure_evidence(
                    selected_root, selected_counter, milestones
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
