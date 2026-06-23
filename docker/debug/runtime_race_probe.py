#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.config_models import Config
from agent.core.passive_turn import Reasoner
from agent.core.runtime_support import TurnRunResult
from agent.core.types import ReasonerResult
from agent.looping.core import AgentLoop
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    MemoryConfig,
    MemoryServices,
)
from agent.provider import LLMResponse
from agent.retrieval.protocol import RetrievalRequest, RetrievalResult
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import BusOutboundPort, OutboundDispatch, PushToolOutboundPort
from bootstrap.tools import build_core_runtime
from bus.events import InboundMessage, OutboundMessage
from bus.event_bus import EventBus
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.engine import (
    EngineProfile,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryToolProfile,
)
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from session.manager import SessionManager


CHANNEL = "race"
CHAT = "same-chat"
OTHER_CHAT = "other-chat"


@dataclass
class SendRecord:
    seq: int
    event: str
    source: str
    channel: str
    chat_id: str
    message: str
    ts: float


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    records: list[dict[str, object]]


class _ProbeMemoryEngine:
    def describe(self) -> MemoryEngineDescriptor:
        return MemoryEngineDescriptor(
            name="race_probe",
            profile=EngineProfile.CLASSIC_MEMORY_SERVICE,
            capabilities=frozenset({MemoryCapability.RETRIEVE_CONTEXT_BLOCK}),
        )

    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile()

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        return MemoryQueryResult(text_block="")

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        return MemoryMutationResult(accepted=True, item_id="race-memory")

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        return MemoryIngestResult(accepted=True)

    def read_long_term(self) -> str:
        return ""

    def read_self(self) -> str:
        return ""

    def read_recent_context(self) -> str:
        return ""

    def get_memory_context(self) -> str:
        return ""

    def read_history(self, max_chars: int = 0) -> str:
        return ""

    def read_recent_history(self, *, max_chars: int = 0) -> str:
        return ""

    def has_long_term_memory(self) -> bool:
        return False


class _NoopProvider:
    async def chat(self, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="noop", tool_calls=[])


class _NoopRetrieval:
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult(block="")


class _BlockingReasoner(Reasoner):
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.started: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}
        self.events: list[str] = []
        self.active = 0
        self.max_active = 0

    def block(self, content: str) -> asyncio.Event:
        release = asyncio.Event()
        self.release[content] = release
        _ = self.started.setdefault(content, asyncio.Event())
        return release

    async def wait_started(self, content: str) -> None:
        event = self.started.setdefault(content, asyncio.Event())
        _ = await asyncio.wait_for(event.wait(), timeout=self.timeout)

    async def run(
        self,
        initial_messages: list[dict[str, Any]],
        *,
        request_time: Any = None,
        preloaded_tools: set[str] | None = None,
        preloaded_tool_order: list[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Any = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
        disabled_tools: set[str] | None = None,
    ) -> ReasonerResult:
        return ReasonerResult(reply="agent-loop")

    async def run_turn(
        self,
        *,
        msg: Any,
        session: Any,
        skill_names: list[str] | None = None,
        base_history: list[dict[str, Any]] | None = None,
        retrieved_memory_block: str = "",
        extra_hints: list[str] | None = None,
    ) -> TurnRunResult:
        content = str(getattr(msg, "content", ""))
        key = str(getattr(session, "key", ""))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.events.append(f"start:{key}:{content}")
        _ = self.started.setdefault(content, asyncio.Event()).set()
        try:
            release = self.release.get(content)
            if release is not None:
                _ = await asyncio.wait_for(release.wait(), timeout=self.timeout)
            return TurnRunResult(reply=f"passive:{content}")
        finally:
            self.events.append(f"end:{key}:{content}")
            self.active -= 1

    async def render_prompt(self, input: Any) -> Any:
        raise NotImplementedError


class RaceHarness:
    def __init__(
        self,
        timeout: float,
        *,
        config_path: Path | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.timeout = timeout
        self._tmpdir = TemporaryDirectory(prefix="akashic-race-")
        self._config_explicit = config_path is not None
        self.workspace = workspace or Path(self._tmpdir.name) / "workspace"
        self.config_path = config_path or Path(self._tmpdir.name) / "config.toml"
        self.bus = MessageBus()
        self.push_tool = MessagePushTool(chat_lane=self.bus.chat_lane)
        self.push_port = PushToolOutboundPort(self.push_tool)
        self.bus_port = BusOutboundPort(self.bus)
        self.records: list[SendRecord] = []
        self._seq = 0
        self._blocked: dict[str, asyncio.Event] = {}
        self._started: dict[str, asyncio.Event] = {}
        self._ended: dict[str, asyncio.Event] = {}
        self._dispatch_task: asyncio.Task[None] | None = None
        self.register_runtime_channel(CHANNEL, self.bus, self.push_tool)

    def load_config(self) -> Config:
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            _ = self.config_path.write_text(
                "\n".join(
                    [
                        'provider = "openai"',
                        'model = "race-model"',
                        'api_key = ""',
                        'system_prompt = "race probe"',
                        "max_iterations = 3",
                        "max_tokens = 128",
                        "memory_window = 8",
                        "",
                        "[channels]",
                        'socket = "/tmp/akashic-race.sock"',
                        "",
                        "[channels.telegram]",
                        "enabled = false",
                        'token = ""',
                        "",
                        "[channels.qq]",
                        "enabled = false",
                        'bot_uin = ""',
                        "",
                        "[proactive]",
                        'profile = "quiet"',
                        "enabled = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return Config.load(self.config_path)

    def load_repo_config(self) -> Config:
        path = self.config_path if self._config_explicit else Path.cwd() / "config.toml"
        if not path.exists():
            raise AssertionError(f"config.toml not found: {path}")
        return Config.load(path)

    def register_runtime_channel(
        self,
        channel: str,
        bus: MessageBus,
        push_tool: MessagePushTool,
    ) -> None:
        async def _text(chat_id: str, message: str) -> None:
            await self._send_text_for(channel, chat_id, message)

        push_tool.register_channel(channel, text=_text)
        bus.subscribe_outbound(channel, self._send_outbound)

    async def start(self) -> None:
        self._dispatch_task = asyncio.create_task(self.bus.dispatch_outbound())

    async def close(self) -> None:
        self.bus.stop()
        if self._dispatch_task is None:
            self._tmpdir.cleanup()
            return
        try:
            _ = self._dispatch_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dispatch_task
        finally:
            self._tmpdir.cleanup()

    def make_agent_loop(self, reasoner: Reasoner) -> AgentLoop:
        config = self.load_config()
        self.workspace.mkdir(parents=True, exist_ok=True)
        session_manager = SessionManager(self.workspace)
        return AgentLoop(
            AgentLoopDeps(
                bus=self.bus,
                provider=cast(Any, _NoopProvider()),
                tools=ToolRegistry(),
                session_manager=session_manager,
                workspace=self.workspace,
                event_bus=EventBus(),
                processing_state=ProcessingState(),
                memory_services=MemoryServices(
                    engine=cast(Any, _ProbeMemoryEngine()),
                ),
                retrieval_pipeline=_NoopRetrieval(),
                reasoner=reasoner,
            ),
            AgentLoopConfig(
                llm=LLMConfig(
                    model=config.agent_model or config.model,
                    light_model=config.light_model,
                    max_iterations=config.max_iterations,
                    max_tokens=config.max_tokens,
                    tool_search_enabled=config.tool_search_enabled,
                    multimodal=config.multimodal,
                    vl_available=bool(config.vl_model),
                ),
                memory=MemoryConfig(window=config.memory_window),
            ),
        )

    def block_message(self, message: str) -> asyncio.Event:
        release = asyncio.Event()
        self._blocked[message] = release
        _ = self._started.setdefault(message, asyncio.Event())
        _ = self._ended.setdefault(message, asyncio.Event())
        return release

    async def wait_started(self, message: str) -> None:
        event = self._started.setdefault(message, asyncio.Event())
        _ = await asyncio.wait_for(event.wait(), timeout=self.timeout)

    async def wait_ended(self, message: str) -> None:
        if any(
            record.event == "end" and record.message == message
            for record in self.records
        ):
            return
        event = self._ended.setdefault(message, asyncio.Event())
        _ = await asyncio.wait_for(event.wait(), timeout=self.timeout)

    async def publish_user(self, chat_id: str = CHAT) -> InboundMessage:
        item = InboundMessage(
            channel=CHANNEL,
            sender="user",
            chat_id=chat_id,
            content=f"user:{chat_id}",
        )
        await self.bus.publish_inbound(item)
        return item

    async def passive_once(self, reply: str) -> None:
        item = await asyncio.wait_for(self.bus.consume_inbound(), timeout=self.timeout)
        _ = await self.bus_port.dispatch(
            OutboundDispatch(
                channel=item.channel,
                chat_id=item.chat_id,
                content=reply,
            )
        )
        await self.bus.complete_inbound(item)

    async def non_passive(self, message: str, chat_id: str = CHAT) -> bool:
        return await self.push_port.dispatch(
            OutboundDispatch(
                channel=CHANNEL,
                chat_id=chat_id,
                content=message,
            )
        )

    async def _send_text_for(self, channel: str, chat_id: str, message: str) -> None:
        await self._record("start", channel, chat_id, message)
        _ = self._started.setdefault(message, asyncio.Event()).set()
        release = self._blocked.get(message)
        if release is not None:
            _ = await asyncio.wait_for(release.wait(), timeout=self.timeout)
        await self._record("end", channel, chat_id, message)
        _ = self._ended.setdefault(message, asyncio.Event()).set()

    async def _send_outbound(self, msg: OutboundMessage) -> None:
        await self._record("start", msg.channel, msg.chat_id, msg.content)
        _ = self._started.setdefault(msg.content, asyncio.Event()).set()
        release = self._blocked.get(msg.content)
        if release is not None:
            _ = await asyncio.wait_for(release.wait(), timeout=self.timeout)
        await self._record("end", msg.channel, msg.chat_id, msg.content)
        _ = self._ended.setdefault(msg.content, asyncio.Event()).set()

    async def _record(
        self,
        event: str,
        channel: str,
        chat_id: str,
        message: str,
    ) -> None:
        self._seq += 1
        source = message.split(":", 1)[0]
        self.records.append(
            SendRecord(
                seq=self._seq,
                event=event,
                source=source,
                channel=channel,
                chat_id=chat_id,
                message=message,
                ts=time.perf_counter(),
            )
        )

    def assert_end_order(self, expected: list[str]) -> None:
        actual = [
            record.message
            for record in self.records
            if record.event == "end" and record.message in expected
        ]
        if actual != expected:
            raise AssertionError(f"发送顺序异常: expected={expected!r}, actual={actual!r}")

    def dump_records(self) -> list[dict[str, object]]:
        return [
            {
                "seq": record.seq,
                "event": record.event,
                "source": record.source,
                "channel": record.channel,
                "chat_id": record.chat_id,
                "message": record.message,
                "ts": record.ts,
            }
            for record in self.records
        ]


ScenarioFn = Callable[[RaceHarness], Awaitable[None]]


async def _run_harness(
    name: str,
    timeout: float,
    scenario: ScenarioFn,
    *,
    config_path: Path | None = None,
    workspace: Path | None = None,
) -> ScenarioResult:
    harness = RaceHarness(
        timeout=timeout,
        config_path=config_path,
        workspace=workspace,
    )
    try:
        await scenario(harness)
        return ScenarioResult(name=name, ok=True, records=harness.dump_records())
    finally:
        await harness.close()


async def scenario_drift_before_push(harness: RaceHarness) -> None:
    await harness.start()
    passive = asyncio.create_task(harness.passive_once("passive:A1"))
    drift_ready = asyncio.Event()
    release_drift = asyncio.Event()

    async def drift() -> None:
        _ = drift_ready.set()
        _ = await asyncio.wait_for(release_drift.wait(), timeout=harness.timeout)
        ok = await harness.non_passive("drift:A1")
        if not ok:
            raise AssertionError("drift message_push failed")

    drift_task = asyncio.create_task(drift())
    _ = await asyncio.wait_for(drift_ready.wait(), timeout=harness.timeout)
    _ = await harness.publish_user()
    _ = release_drift.set()
    _ = await asyncio.wait_for(
        asyncio.gather(passive, drift_task),
        timeout=harness.timeout,
    )
    harness.assert_end_order(["passive:A1", "drift:A1"])


async def scenario_drift_sending_then_user(harness: RaceHarness) -> None:
    await harness.start()
    release_drift = harness.block_message("drift:A3")
    drift = asyncio.create_task(harness.non_passive("drift:A3"))
    await harness.wait_started("drift:A3")
    passive = asyncio.create_task(harness.passive_once("passive:A3"))
    _ = await harness.publish_user()
    _ = release_drift.set()
    _ = await asyncio.wait_for(asyncio.gather(passive, drift), timeout=harness.timeout)
    await harness.wait_ended("passive:A3")
    harness.assert_end_order(["drift:A3", "passive:A3"])


async def scenario_scheduler_after_user(harness: RaceHarness) -> None:
    await harness.start()
    passive = asyncio.create_task(harness.passive_once("passive:B1"))
    _ = await harness.publish_user()
    scheduler = asyncio.create_task(harness.non_passive("scheduler:B1"))
    _ = await asyncio.wait_for(
        asyncio.gather(passive, scheduler),
        timeout=harness.timeout,
    )
    harness.assert_end_order(["passive:B1", "scheduler:B1"])


async def scenario_fifo_with_passive_insert(harness: RaceHarness) -> None:
    await harness.start()
    release_first = harness.block_message("proactive:D1")
    first = asyncio.create_task(harness.non_passive("proactive:D1"))
    await harness.wait_started("proactive:D1")
    second = asyncio.create_task(harness.non_passive("scheduler:D1"))
    await asyncio.sleep(0)
    third = asyncio.create_task(harness.non_passive("drift:D1"))
    await asyncio.sleep(0)
    passive = asyncio.create_task(harness.passive_once("passive:D1"))
    _ = await harness.publish_user()
    _ = release_first.set()
    _ = await asyncio.wait_for(
        asyncio.gather(first, second, third, passive),
        timeout=harness.timeout,
    )
    harness.assert_end_order(
        ["proactive:D1", "passive:D1", "scheduler:D1", "drift:D1"]
    )


async def scenario_cross_chat_isolated(harness: RaceHarness) -> None:
    item = await harness.publish_user()
    same = asyncio.create_task(harness.non_passive("drift:C2"))
    await asyncio.sleep(0.02)
    if same.done():
        raise AssertionError("same chat non_passive should wait for passive turn")
    other = asyncio.create_task(harness.non_passive("proactive:C2", chat_id=OTHER_CHAT))
    _ = await asyncio.wait_for(other, timeout=harness.timeout)
    await harness.bus.complete_inbound(item)
    _ = await asyncio.wait_for(same, timeout=harness.timeout)
    harness.assert_end_order(["proactive:C2", "drift:C2"])


async def scenario_silent_passive_releases_lane(harness: RaceHarness) -> None:
    item = await harness.publish_user()
    drift = asyncio.create_task(harness.non_passive("drift:E1"))
    await asyncio.sleep(0.02)
    if drift.done():
        raise AssertionError("non_passive should wait while passive turn is pending")
    await harness.bus.complete_inbound(item)
    _ = await asyncio.wait_for(drift, timeout=harness.timeout)
    harness.assert_end_order(["drift:E1"])


async def scenario_cancelled_non_passive_ticket(harness: RaceHarness) -> None:
    await harness.bus.chat_lane.mark_passive_pending(CHANNEL, CHAT)
    try:
        _ = await asyncio.wait_for(
            harness.non_passive("drift:E6"),
            timeout=0.05,
        )
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("first non_passive should be cancelled while waiting")
    await harness.bus.chat_lane.mark_passive_done(CHANNEL, CHAT)
    ok = await asyncio.wait_for(
        harness.non_passive("scheduler:E6"),
        timeout=harness.timeout,
    )
    if not ok:
        raise AssertionError("second non_passive failed after cancelled ticket")
    harness.assert_end_order(["scheduler:E6"])


async def scenario_agent_loop_runtime(harness: RaceHarness) -> None:
    config = harness.load_config()
    if config.channels.telegram is not None or config.channels.qq is not None:
        raise AssertionError("agent-loop runtime probe config must not enable telegram/qq")

    await harness.start()
    reasoner = _BlockingReasoner(timeout=harness.timeout)
    loop = harness.make_agent_loop(reasoner)
    loop_task = asyncio.create_task(loop.run())
    release_passive = reasoner.block("user:same-chat")

    async def scheduler_soft() -> None:
        content = await loop.process_direct(
            "scheduler-soft",
            session_key="scheduler:job",
            busy_session_key=f"{CHANNEL}:{CHAT}",
            channel=CHANNEL,
            chat_id=CHAT,
            skip_post_memory=True,
            skip_memory_retrieval=True,
        )
        if content != "passive:scheduler-soft":
            raise AssertionError(f"scheduler soft content mismatch: {content!r}")
        ok = await harness.non_passive("scheduler:agent-loop")
        if not ok:
            raise AssertionError("scheduler soft message_push failed")

    try:
        _ = await harness.publish_user()
        await reasoner.wait_started("user:same-chat")

        drift = asyncio.create_task(harness.non_passive("drift:agent-loop"))
        scheduler = asyncio.create_task(scheduler_soft())
        await asyncio.sleep(0.02)
        if drift.done():
            raise AssertionError("drift should wait for real AgentLoop passive turn")
        if any(event.startswith("start:scheduler:job:") for event in reasoner.events):
            raise AssertionError("scheduler entered reasoner while passive held RTL")

        _ = release_passive.set()
        _ = await asyncio.wait_for(
            asyncio.gather(drift, scheduler),
            timeout=harness.timeout,
        )
        harness.assert_end_order(
            ["passive:user:same-chat", "drift:agent-loop", "scheduler:agent-loop"]
        )
        if reasoner.max_active != 1:
            raise AssertionError(f"reasoner concurrent execution: {reasoner.max_active}")
        if reasoner.events != [
            "start:race:same-chat:user:same-chat",
            "end:race:same-chat:user:same-chat",
            "start:scheduler:job:scheduler-soft",
            "end:scheduler:job:scheduler-soft",
        ]:
            raise AssertionError(f"reasoner event order mismatch: {reasoner.events!r}")
    finally:
        loop.stop()
        _ = loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await loop_task


async def scenario_config_runtime_llm(harness: RaceHarness) -> None:
    config = harness.load_repo_config()
    channel = config.proactive.default_channel or CHANNEL
    chat_id = config.proactive.default_chat_id or CHAT
    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    core = None
    loop_task: asyncio.Task[None] | None = None
    dispatch_task: asyncio.Task[None] | None = None
    try:
        harness.workspace.mkdir(parents=True, exist_ok=True)
        core = build_core_runtime(config, harness.workspace, resources)
        harness.register_runtime_channel(channel, core.bus, core.push_tool)
        await core.start()
        loop_task = asyncio.create_task(core.loop.run())
        dispatch_task = asyncio.create_task(core.bus.dispatch_outbound())

        user = InboundMessage(
            channel=channel,
            sender="race-user",
            chat_id=chat_id,
            content="竞态验证：请只用一句中文回复，内容包含“收到竞态验证”。",
        )
        await core.bus.publish_inbound(user)

        drift = asyncio.create_task(
            core.push_tool.execute(
                channel=channel,
                chat_id=chat_id,
                message="drift:config-runtime",
            )
        )
        proactive = asyncio.create_task(
            core.push_tool.execute(
                channel=channel,
                chat_id=chat_id,
                message="proactive:config-runtime",
            )
        )

        async def scheduler_soft() -> str:
            _ = await core.loop.process_direct(
                "竞态验证 scheduler soft：请只用一句中文回复，内容包含“scheduler done”。",
                session_key="scheduler:config-runtime",
                busy_session_key=f"{channel}:{chat_id}",
                channel=channel,
                chat_id=chat_id,
                skip_post_memory=True,
                skip_memory_retrieval=True,
            )
            result = await core.push_tool.execute(
                channel=channel,
                chat_id=chat_id,
                message="scheduler:config-runtime",
            )
            return str(result)

        scheduler = asyncio.create_task(scheduler_soft())
        _ = await asyncio.wait_for(
            asyncio.gather(drift, proactive, scheduler),
            timeout=max(harness.timeout, 30.0),
        )
        await harness.wait_ended("scheduler:config-runtime")

        ended = [record for record in harness.records if record.event == "end"]
        non_passive_messages = {
            "drift:config-runtime",
            "proactive:config-runtime",
            "scheduler:config-runtime",
        }
        passive_indexes = [
            index
            for index, record in enumerate(ended)
            if record.channel == channel
            and record.chat_id == chat_id
            and record.message not in non_passive_messages
        ]
        if not passive_indexes:
            raise AssertionError("real passive AgentLoop reply was not sent")
        first_passive = passive_indexes[0]
        for message in non_passive_messages:
            indexes = [
                index for index, record in enumerate(ended) if record.message == message
            ]
            if not indexes:
                raise AssertionError(f"non-passive message missing: {message}")
            if indexes[0] < first_passive:
                raise AssertionError(
                    f"non-passive sent before passive reply: {message}"
                )
    finally:
        if core is not None:
            core.loop.stop()
            core.bus.stop()
        for task in (loop_task, dispatch_task):
            if task is not None:
                _ = task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if core is not None:
            with suppress(Exception):
                await core.stop()
            with suppress(Exception):
                await core.memory_runtime.aclose()
        clear_default_shared_http_resources(resources)
        await resources.aclose()


DEFAULT_SCENARIOS = [
    "agent-loop-runtime",
    "a1-drift-before-push",
    "a3-drift-sending-then-user",
    "b1-scheduler-after-user",
    "d1-fifo-passive-insert",
    "c2-cross-chat-isolated",
    "e1-silent-passive",
    "e6-cancelled-nonpassive-ticket",
]


SCENARIOS: dict[str, ScenarioFn] = {
    "agent-loop-runtime": scenario_agent_loop_runtime,
    "config-runtime-llm": scenario_config_runtime_llm,
    "a1-drift-before-push": scenario_drift_before_push,
    "a3-drift-sending-then-user": scenario_drift_sending_then_user,
    "b1-scheduler-after-user": scenario_scheduler_after_user,
    "d1-fifo-passive-insert": scenario_fifo_with_passive_insert,
    "c2-cross-chat-isolated": scenario_cross_chat_isolated,
    "e1-silent-passive": scenario_silent_passive_releases_lane,
    "e6-cancelled-nonpassive-ticket": scenario_cancelled_non_passive_ticket,
}


async def _run(args: argparse.Namespace) -> int:
    names = list(DEFAULT_SCENARIOS) if args.scenario == "all" else [args.scenario]
    results: list[ScenarioResult] = []
    for name in names:
        scenario = SCENARIOS.get(name)
        if scenario is None:
            raise SystemExit(f"未知场景: {name}")
        try:
            result = await _run_harness(
                name,
                args.timeout,
                scenario,
                config_path=args.config,
                workspace=args.workspace,
            )
        except Exception as exc:
            result = ScenarioResult(name=name, ok=False, records=[])
            results.append(result)
            print(
                json.dumps(
                    {
                        "ok": False,
                        "failed": name,
                        "error": repr(exc),
                        "results": [
                            {
                                "name": item.name,
                                "ok": item.ok,
                                "records": item.records,
                            }
                            for item in results
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        results.append(result)

    payload: dict[str, object] = {
        "ok": True,
        "scenario": args.scenario,
        "results": [
            {
                "name": result.name,
                "ok": result.ok,
                "records": result.records,
            }
            for result in results
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(text + "\n", encoding="utf-8")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docker runtime 竞态探针")
    _ = parser.add_argument(
        "--scenario",
        default=os.environ.get("AKASHIC_RACE_SCENARIO", "all"),
        choices=["all", *SCENARIOS.keys()],
    )
    _ = parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("AKASHIC_RACE_TIMEOUT", "2")),
    )
    _ = parser.add_argument(
        "--trace",
        type=Path,
        default=(
            Path(os.environ["AKASHIC_RACE_TRACE"])
            if os.environ.get("AKASHIC_RACE_TRACE")
            else None
        ),
    )
    _ = parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(os.environ["AKASHIC_RACE_CONFIG"])
            if os.environ.get("AKASHIC_RACE_CONFIG")
            else None
        ),
    )
    _ = parser.add_argument(
        "--workspace",
        type=Path,
        default=(
            Path(os.environ["AKASHIC_RACE_WORKSPACE"])
            if os.environ.get("AKASHIC_RACE_WORKSPACE")
            else None
        ),
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parse_args())))


if __name__ == "__main__":
    main()
