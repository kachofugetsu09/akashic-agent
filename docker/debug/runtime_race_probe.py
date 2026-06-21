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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tools.message_push import MessagePushTool
from agent.turns.outbound import BusOutboundPort, OutboundDispatch, PushToolOutboundPort
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus


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


class RaceHarness:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
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
        self.push_tool.register_channel(CHANNEL, text=self._send_text)
        self.bus.subscribe_outbound(CHANNEL, self._send_outbound)

    async def start(self) -> None:
        self._dispatch_task = asyncio.create_task(self.bus.dispatch_outbound())

    async def close(self) -> None:
        self.bus.stop()
        if self._dispatch_task is None:
            return
        _ = self._dispatch_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._dispatch_task

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

    async def _send_text(self, chat_id: str, message: str) -> None:
        await self._record("start", CHANNEL, chat_id, message)
        _ = self._started.setdefault(message, asyncio.Event()).set()
        release = self._blocked.get(message)
        if release is not None:
            _ = await asyncio.wait_for(release.wait(), timeout=self.timeout)
        await self._record("end", CHANNEL, chat_id, message)
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
) -> ScenarioResult:
    harness = RaceHarness(timeout=timeout)
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


SCENARIOS: dict[str, ScenarioFn] = {
    "a1-drift-before-push": scenario_drift_before_push,
    "a3-drift-sending-then-user": scenario_drift_sending_then_user,
    "b1-scheduler-after-user": scenario_scheduler_after_user,
    "d1-fifo-passive-insert": scenario_fifo_with_passive_insert,
    "c2-cross-chat-isolated": scenario_cross_chat_isolated,
    "e1-silent-passive": scenario_silent_passive_releases_lane,
    "e6-cancelled-nonpassive-ticket": scenario_cancelled_non_passive_ticket,
}


async def _run(args: argparse.Namespace) -> int:
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results: list[ScenarioResult] = []
    for name in names:
        scenario = SCENARIOS.get(name)
        if scenario is None:
            raise SystemExit(f"未知场景: {name}")
        try:
            result = await _run_harness(name, args.timeout, scenario)
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
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parse_args())))


if __name__ == "__main__":
    main()
