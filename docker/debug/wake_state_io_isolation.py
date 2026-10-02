"""临时真实 Root、Timer 与 SQLite；屏障不替代原 SQL、提交或关闭。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import threading
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

from agent.control.timer import AsyncioOneShotTimer, TimerHandle
from agent.plugin_composition import CompositionRoot, Context
from agent.plugin_composition.model import FiberState, PluginRuntime
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, MessageWriters, OwnerState
from agent.plugin_composition.tasks import TASKS, PluginTasks
from agent.plugin_composition.timers import TIMERS, PluginTimers
from plugins.drift.plugin import _AsyncWakeServices as DriftServices
from plugins.drift.store import DriftStore
from plugins.eventmail.plugin import _WakeServices as MailServices
from plugins.eventmail.plugin import _DeliveryServices as MailDelivery
from plugins.eventmail.store import EventMailStore
from plugins.wake._boundary import SEMANTIC_INTEREST, SemanticInterest
from plugins.wake.api import DRIFT_WAKE, EVENTMAIL_WAKE, EVENTMAIL_DELIVERY, Config, DeliveryTarget
from plugins.wake.request import Request, TOOLS
from plugins.wake.source import Pointer
from plugins.wake.runtime import Runtime
from plugins.wake.state import ContentScore, WakeState
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input


def leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


async def scenario(directory: Path, kind: str, expect_blocking: bool) -> dict[str, Any]:
    """保持真实 SQL 未结束，观察 loop、取消排空与独立连接重开的终态。"""
    root = CompositionRoot("wake-state-io-" + kind)
    main_thread = threading.get_ident()
    now = datetime.now(UTC)
    clock = [now]
    first_fire = True

    async def maintenance_sleep(delay: float) -> None:
        """只推进第一次真实 Timer 的时钟；后续 deadline 留给原 cancel/cleanup。"""
        nonlocal first_fire
        if first_fire:
            first_fire = False
            clock[0] += timedelta(seconds=delay)
            await asyncio.sleep(0)
        else:
            await asyncio.Event().wait()
    path = directory / "wake.sqlite3"
    state = WakeState(path)
    state.initialize()
    messages = MessageLog(directory / "sessions.db")
    mail = EventMailStore(directory / "mail.db")
    mail.initialize()
    drift = DriftStore(directory / "drift.db")
    drift.initialize()
    mail.submit("feed", "batch", [{"item_id": "one", "revision": "r1", "payload": {
        "title": "news", **({"wake_action": "decline"} if kind in {"source_decline", "source_cancel"} else {})}}])
    envelope_before = mail.snapshot(now)
    receipt: dict[str, Any] = {"kind": kind, "connections": [], "score_calls": 0}
    runtimes: list[Runtime] = []
    contexts: list[Context] = []

    class Interest:
        async def score(self, texts, *, cutoff):
            assert threading.get_ident() == main_thread
            receipt["score_calls"] += 1
            return tuple(0.5 for _ in texts)

        def status(self):
            assert threading.get_ident() == main_thread
            return None

    class GuardedMail(MailServices):
        def snapshot(self, now):
            assert threading.get_ident() == main_thread
            return super().snapshot(now)

        def expire(self, refs, now):
            assert threading.get_ident() == main_thread
            return super().expire(refs, now)

    actual_timer = PluginTimers(AsyncioOneShotTimer(
        clock=lambda: clock[0], sleeper=maintenance_sleep)
        if kind in {"finish_cancel", "blocked_stop"} else AsyncioOneShotTimer())

    class CleanupFailure(PluginTimers):
        def schedule(self, deadline):
            handle = actual_timer.schedule(deadline)

            class Handle:
                id = handle.id

                async def result(self):
                    return await handle.result()

                async def cancel(self):
                    return await handle.cancel()

                async def cleanup(self):
                    await handle.cleanup()
                    raise OSError("controlled failure after actual Timer cleanup")

            return cast(TimerHandle, Handle())

    async def providers(ctx):
        await ctx.provide(EVENTMAIL_WAKE, GuardedMail(mail))
        await ctx.provide(DRIFT_WAKE, DriftServices(drift))
        await ctx.provide(SEMANTIC_INTEREST, cast(SemanticInterest, Interest()))
        await ctx.provide(TIMERS, CleanupFailure(None) if kind == "cleanup_cancel" else actual_timer)
        await ctx.provide(MESSAGE_CATALOG, messages.catalog())
        await ctx.provide(MESSAGE_WRITERS, MessageWriters(messages))
        await ctx.provide(OWNER_STATE, OwnerState(messages))
        await ctx.provide(TASKS, PluginTasks())
        await ctx.provide(EVENTMAIL_DELIVERY, MailDelivery(mail))

    async def consumer(ctx):
        contexts.append(ctx)
        runtimes.append(Runtime(ctx, Config(), now=lambda: clock[0]))

    await root.mount(providers, name="providers")
    fiber = await root.mount(consumer, name="wake",
                            inject=(EVENTMAIL_WAKE, DRIFT_WAKE, SEMANTIC_INTEREST, TIMERS,
                                    MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, TASKS, EVENTMAIL_DELIVERY),
                            runtime=PluginRuntime("wake", "io-scenario", directory, directory, directory, {}))
    assert fiber.state is FiberState.ACTIVE, fiber.error
    runtime, ctx = runtimes[0], contexts[0]
    request: Request | None = None
    original_rows = []
    if kind in {"source_decline", "source_cancel"}:
        # 原 Input 与 Pointer 是明确 fixture；未使用的 binding 不作为正式准入/工具验收。
        state.record_content_scores((ContentScore("feed", "one", "r1", 3.0, 0.5, now),))
        request = Request(flow_id="a" * 32, owner="content", now=now, timezone="UTC",
                          target=DeliveryTarget(channel="unused", recipient="unused", session_id="target"),
                          sink={"name": "unused", "address": "unused", "binding_id": "unused-sink"},
                          program_binding="unused-program", tools={name: "unused-" + name for name in TOOLS["content"]},
                          snapshot_seq=cast(int, envelope_before["snapshot_seq"]),
                          items=tuple(dict(item) for item in state.scored_items(envelope_before["items"])),
                          rules="", history="")
        writer = messages.writer(request.session_id, author="wake", source="wake", body_types=(Input,),
                                 content={"wake.request": lambda _: ContentReferences()})
        writer.append(request.input_id, Input((ContentPart("wake.request", request.model_dump(mode="json")),)))
        writer.expire()
        async with ctx.runtime_scope():
            pointer = Pointer(session_id=request.session_id, input_id=request.input_id)
            ctx.require(OWNER_STATE).open(ctx).transact(lambda tx: tx.save(
                "flow:" + request.flow_id, pointer.model_dump(), expected_version=None))
        with closing(sqlite3.connect(directory / "sessions.db")) as connection:
            original_rows = connection.execute("SELECT * FROM messages ORDER BY rowid").fetchall()
    if kind in {"finish_cancel", "blocked_stop"}:
        # 本场景只检查维护 SQL/诊断边界，不用真实模型伪造 availability 验收。
        runtime._blocked = lambda **_: "controlled service unavailable" if kind == "blocked_stop" else None
    connect = sqlite3.connect
    entered, release, checkpoint, ready = (threading.Event() for _ in range(4))
    recovery_entered, recovery_release, recovery_checkpoint = (threading.Event() for _ in range(3))
    loop = asyncio.get_running_loop()
    caller: asyncio.Task[Any] | None = None
    disposal: asyncio.Task[None] | None = None
    observer: asyncio.Task[None] | None = None
    thread: threading.Thread | None = None
    paused = False
    recovery_observer: asyncio.Task[None] | None = None

    class TrackedConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.record = {"opened_thread": threading.get_ident(), "closed": False}
            receipt["connections"].append(self.record)
            self.inserted = False
            self.finished: str | None = None
            self.screened = False

        def execute(self, sql, parameters: Any = (), /):
            nonlocal paused
            if kind == "write_error_cancel" and "INSERT OR IGNORE INTO wake_attempts" in sql:
                if not paused:
                    paused = True
                    receipt["entered_at"] = time.monotonic()
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError("actual readonly INSERT barrier was not released")
                super().execute("PRAGMA query_only=ON")
            target = ((kind == "score_lock" and "INSERT OR IGNORE INTO content_scores" in sql)
                      or (kind == "fire_lock" and "INSERT OR IGNORE INTO wake_attempts" in sql))
            if target and not paused:
                paused = True
                receipt["entered_at"] = time.monotonic()
                entered.set()
            result = super().execute(sql, parameters)
            self.inserted |= "INSERT OR IGNORE INTO wake_attempts" in sql
            self.screened |= "INSERT OR IGNORE INTO wake_runs" in sql
            if "UPDATE wake_attempts SET outcome = ?" in sql:
                self.finished = parameters[0]
            return result

        def executemany(self, sql, parameters, /):
            nonlocal paused
            if kind == "score_lock" and "INSERT OR IGNORE INTO content_scores" in sql and not paused:
                paused = True
                receipt["entered_at"] = time.monotonic()
                entered.set()
            return super().executemany(sql, parameters)

        def __exit__(self, exc_type, exc_value, traceback):
            nonlocal paused
            result = super().__exit__(exc_type, exc_value, traceback)
            pause_commit = (self.finished in {"content_insufficient", "admission_rejected"}
                            if kind in {"finish_cancel", "blocked_stop"}
                            else self.screened if kind in {"source_decline", "source_cancel"} else self.inserted)
            if pause_commit and exc_type is None and kind not in {"score_lock", "fire_lock"} and not paused:
                paused = True
                receipt["entered_at"] = time.monotonic()
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("native committed attempt barrier was not released")
            if kind == "recovery_cancel" and self.finished == "cancelled_after_fire" and exc_type is None:
                receipt["recovery_entered_at"] = time.monotonic()
                recovery_entered.set()
                if not recovery_release.wait(10):
                    raise TimeoutError("native recovery commit barrier was not released")
            return result

        def close(self):
            super().close()
            self.record.update(closed=True, closed_thread=threading.get_ident(), closed_at=time.monotonic())

    def tracked_connect(database, *args, **kwargs):
        if str(database) == str(path) or str(database).startswith(path.resolve().as_uri()):
            kwargs["factory"] = TrackedConnection
        return connect(database, *args, **kwargs)

    async def observe() -> None:
        nonlocal disposal
        try:
            assert caller is not None
            receipt["checkpoint_at"] = time.monotonic()
            receipt["caller_pending"] = not caller.done()
            if kind not in {"score_lock", "fire_lock", "source_decline"}:
                if kind in {"root_stop", "blocked_stop"}:
                    disposal = asyncio.create_task(root.dispose())
                    async with asyncio.timeout(3):
                        while not caller.cancelling():
                            await asyncio.sleep(0)
                    assert not disposal.done()
                    receipt["root_draining"] = True
                else:
                    caller.cancel()
                await asyncio.sleep(0)
                caller.cancel()
                await asyncio.sleep(0)
                assert not caller.done(), "cancel returned before real close"
                receipt["cancel_draining"] = True
            checkpoint.set()
        except BaseException as error:
            receipt["observer_failure"] = repr(error)
            checkpoint.set()
            release.set()
            raise

    def controller() -> None:
        nonlocal recovery_observer
        connection = None
        try:
            if kind in {"score_lock", "fire_lock"}:
                connection = connect(path)
                connection.execute("BEGIN IMMEDIATE")
            ready.set()
            assert entered.wait(10), "original SQL/commit was not reached"
            loop.call_soon_threadsafe(lambda: start_observer())
            reached = checkpoint.wait(1)
            receipt["checkpoint_before_release"] = reached
            if kind == "recovery_cancel":
                release.set()
                assert recovery_entered.wait(10), "cancel diagnostic commit was not reached"

                def start_recovery_observer() -> None:
                    nonlocal recovery_observer
                    recovery_observer = asyncio.create_task(observe_recovery())

                loop.call_soon_threadsafe(start_recovery_observer)
                assert recovery_checkpoint.wait(3), "loop stopped during recovery commit"
        except BaseException as error:
            receipt["controller_failure"] = repr(error)
        finally:
            receipt["released_at"] = time.monotonic()
            if connection is not None:
                connection.rollback()
                connection.close()
            release.set()
            recovery_release.set()

    async def observe_recovery() -> None:
        try:
            assert caller is not None
            caller.cancel()
            await asyncio.sleep(0)
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done()
            receipt["recovery_cancel_draining"] = True
            recovery_checkpoint.set()
        except BaseException as error:
            receipt["observer_failure"] = repr(error)
            recovery_checkpoint.set()
            recovery_release.set()
            raise

    def start_observer() -> None:
        nonlocal observer
        observer = asyncio.create_task(observe())

    async def operation():
        async with ctx.runtime_scope():
            if kind == "score_lock":
                return await runtime.duties.maintain(now)
            if kind in {"source_decline", "source_cancel"}:
                assert request is not None
                return await runtime._run(request.flow_id)
            if kind in {"finish_cancel", "blocked_stop"}:
                return await runtime._maintenance()
            return await runtime._wait(now - timedelta(seconds=1), changed=False)

    try:
        # 1. 临时真实 Root 拥有调用方，foreign SQLite connection 或原生提交屏障持有实际工作。
        with patch.object(sqlite3, "connect", tracked_connect):
            thread = threading.Thread(target=controller, daemon=True)
            thread.start()
            assert await asyncio.to_thread(ready.wait, 5)
            caller = await ctx.spawn(operation(), name="wake-io-" + kind)
            try:
                result = await asyncio.wait_for(asyncio.shield(caller), 15)
                if kind == "score_lock":
                    assert result.scored_count == 1
                elif kind == "source_decline":
                    assert result == "model_skip"
            except BaseException as error:
                errors = leaves(error)
                receipt["errors"] = [type(x).__name__ for x in errors]
                receipt["caller_classified_cancelled"] = caller.cancelled()
                assert kind not in {"score_lock", "fire_lock"}
                assert any(isinstance(x, asyncio.CancelledError) for x in errors)
                if kind == "cleanup_cancel":
                    assert any(isinstance(x, OSError) for x in errors)
                elif kind == "write_error_cancel":
                    assert any(isinstance(x, sqlite3.OperationalError) and "readonly" in str(x) for x in errors)
                    assert not caller.cancelled(), "SQLite failure was classified as plain cancel"
                else:
                    assert len(errors) == 1
                    assert caller.cancelled(), "pure repeated cancel became a program failure"
            assert observer is not None
            await observer
            if recovery_observer is not None:
                await recovery_observer
            if disposal is not None:
                await disposal
                receipt["root_disposed_at"] = time.monotonic()
                assert all(x["closed_at"] <= receipt["root_disposed_at"] for x in receipt["connections"])
            thread.join(2)
            assert not thread.is_alive()
            assert "controller_failure" not in receipt and "observer_failure" not in receipt
            assert receipt["checkpoint_before_release"] is not expect_blocking
            assert all(x["closed"] and x["closed_thread"] == x["opened_thread"] for x in receipt["connections"])
        # 2. 用新的原始连接核对实际 rows/schema；不把 Task 的 prose 当提交证据。
        with closing(connect(path)) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert connection.execute("PRAGMA user_version").fetchone() == (9,)
            attempts = connection.execute("SELECT attempt_id, timer_id, outcome, mail_watermark FROM wake_attempts").fetchall()
            scores = connection.execute("SELECT * FROM content_scores").fetchall()
        receipt.update(attempts=attempts, scores=scores,
                       checkpoint_lag_seconds=receipt["checkpoint_at"] - receipt["entered_at"])
        if kind == "score_lock":
            assert len(scores) == 1 and not attempts and receipt["score_calls"] == 1
            async with ctx.runtime_scope():
                await runtime.duties.maintain(now)
            with closing(connect(path)) as connection:
                assert connection.execute("SELECT * FROM content_scores").fetchall() == scores
            assert receipt["score_calls"] == 1, "same revision was rescored after reopening"
        elif kind in {"source_decline", "source_cancel"}:
            assert request is not None and not attempts and len(scores) == 1 and receipt["score_calls"] == 0
            if kind == "source_cancel":
                async with ctx.runtime_scope():
                    found = runtime.source.read(request.flow_id)
                    assert found is not None and not Pointer.model_validate(dict(found[0].value)).settled
                    assert mail.snapshot(now) == envelope_before
                    assert await runtime._run(request.flow_id) == "model_skip"
                receipt["source_resumed_after_cancel"] = True
            row = state.get_run(request.flow_id)
            assert row is not None and row["decision"] == "skip"
            async with ctx.runtime_scope():
                found = runtime.source.read(request.flow_id)
                assert found is not None and Pointer.model_validate(dict(found[0].value)).settled
                assert await runtime.source.start(request.flow_id) is None
            with closing(connect(directory / "sessions.db")) as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                actual_rows = connection.execute("SELECT * FROM messages ORDER BY rowid").fetchall()
                assert actual_rows[:len(original_rows)] == original_rows and len(actual_rows) == len(original_rows) + 1
            receipt["source_pointer_settled"] = True
            receipt["source_input_preserved"] = True
            receipt["source_not_replayed"] = True
        elif kind == "write_error_cancel":
            assert not attempts and not scores
            receipt["fire_not_falsely_persisted"] = True
        else:
            assert len(attempts) == 1
            expected = ("checking" if kind == "fire_lock" else "failed" if kind == "cleanup_cancel"
                        else "content_insufficient" if kind == "finish_cancel"
                        else "admission_rejected" if kind == "blocked_stop" else "cancelled_after_fire")
            assert attempts[0][2] == expected
            if kind == "finish_cancel":
                assert len(scores) == 1 and receipt["score_calls"] == 1 and attempts[0][3] is not None
            else:
                assert not scores and attempts[0][3] is None
        if kind not in {"source_decline", "source_cancel"}:
            assert mail.snapshot(now) == envelope_before, "Wake I/O changed authoritative EventMail"
        return receipt
    finally:
        release.set()
        recovery_release.set()
        if thread is not None:
            thread.join(2)
        if caller is not None and not caller.done():
            caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
        await root.dispose()
        messages.close()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="旧源码锁等待负对照，只运行两份 lock 场景")
    args = parser.parse_args()
    results = []
    with TemporaryDirectory(prefix="wake-state-io-") as directory:
        kinds = ("score_lock", "fire_lock") if args.baseline else (
            "score_lock", "fire_lock", "fire_cancel", "root_stop", "cleanup_cancel",
            "finish_cancel", "blocked_stop", "recovery_cancel", "source_decline", "source_cancel",
            "write_error_cancel")
        for kind in kinds:
            path = Path(directory) / kind
            path.mkdir()
            results.append(await scenario(path, kind, args.baseline))
    print(json.dumps({"scenarios": results, "scope": "temporary real Root/Timer/SQLite; deterministic interest fixture. Source decline uses real Task/Message/Owner/domain with manually seeded Input/Pointer and unused binding fixture; not Source admission/tool validation. No models, sends or formal install"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
