"""Source abandon/resume read isolation on real SQLite and Tasks admission.

Run with PYTHONPATH=. .venv/bin/python docker/debug/source_control_read_isolation.py.
--source can select an unmodified checkout for the decode-barrier negative control.
Only temporary messages and local Tasks are used; no provider or delivery runs.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch
import weakref

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--action", choices=("abandon", "resume"))
OPTIONS = parser.parse_args()
SOURCE = OPTIONS.source.resolve()
sys.path.insert(0, str(SOURCE))

from agent.plugin_composition.tasks import RestartGate, Task, TaskSlot, Tasks
from agent.restart import RestartPendingError
from plugins.sources.session import SourceSession
import plugins.ledger.log as storage
from plugins.ledger.contract import ContentPart, ContentReferences, Control, Input, Output


LOOP_THREAD = threading.get_ident()


def on_loop():
    assert threading.get_ident() == LOOP_THREAD, "Task/handle/restart/callback left loop"


class LoopGate(RestartGate):
    def check_open(self):
        on_loop()
        return super().check_open()


class Fixture:
    def __init__(self, directory):
        self.log = storage.MessageLog(directory / "sessions.db")
        self.tasks = Tasks()
        self.reader = self.log.reader("session")
        self.gate = LoopGate(boot_id="scenario", supervised=True, commit=lambda _: None)
        self.notified = []
        self.inputs, self.outputs, self.controls = (
            self.writer(kind) for kind in (Input, Output, Control)
        )
        self.source = SourceSession(reader=self.reader, inputs=self.inputs, controls=self.controls,
            tasks=self.tasks, changed=self.changed, restart_gate=self.gate)
        self.peer = SourceSession(reader=self.reader, inputs=self.writer(Input, "peer"),
            controls=self.writer(Control, "peer"), tasks=self.tasks, changed=self.changed)
        self.target = self.inputs.append("input", Input(()))
        text = "x" * 65536
        for index in range(192):
            self.outputs.append(f"tail-{index}", Output((ContentPart("text", text),), "continue"))
        self.failure = self.controls.append("failure", Control("failure", self.reader.head(), "fixture"))
        self.original = self.digest()
        self.writer(Output, "peer").append("peer-completed", Output((), "complete"))

    def writer(self, kind, source="conversation"):
        def content(_):
            on_loop()
            return ContentReferences()
        return self.log.writer("session", author=source, source=source,
            body_types=(kind,), content={"text": content})

    def changed(self, _reader, source, _pending):
        on_loop()
        self.notified.append(source)

    def operation(self, action, identity="operation"):
        if action == "resume":
            return self.source.resume(identity, "input")
        return self.source.control(identity, Control("abandon", self.failure.seq),
            expected_head=self.failure.seq, handle=None)

    def digest(self):
        digest = hashlib.sha256()
        for row in self.log._connection.execute(
            "SELECT * FROM messages WHERE session_key=? AND seq<=? ORDER BY seq",
            ("session", self.failure.seq),
        ):
            digest.update(repr(tuple(row)).encode())
        return digest.hexdigest()

    async def verify(self):
        assert self.digest() == self.original, "original messages changed"
        assert self.log._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        await self.tasks.close()
        self.log.close()


@asynccontextmanager
async def fixture(directory):
    directory.mkdir()
    value = Fixture(directory)
    try:
        yield value
    finally:
        await value.verify()


class DecodeBarrier:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.references = []
        self.decoded = 0
        self.peak_live = 0
        self.original = storage._message

    def decode(self, row):
        message = self.original(row)
        if row["id"].startswith("tail-"):
            assert threading.get_ident() != LOOP_THREAD, "long Source history decoded on event loop"
            self.references.append(weakref.ref(message))
            self.decoded += 1
            self.peak_live = max(self.peak_live, sum(ref() is not None for ref in self.references))
            assert self.peak_live <= 128, "predicate retained more than adjacent scan pages"
            if self.decoded == 1:
                self.loop.call_soon_threadsafe(self.entered.set)
                assert self.release.wait(10), "decode barrier was not released"
                self.exited.set()
        return message

    async def wait(self, operation):
        marker = asyncio.create_task(self.entered.wait())
        try:
            done, _ = await asyncio.wait((marker, operation), timeout=10,
                                         return_when=asyncio.FIRST_COMPLETED)
            if operation in done:
                await operation
                raise AssertionError("Source operation did not enter the decode barrier")
            assert marker in done, "decode barrier was not reached"
        finally:
            marker.cancel()
            await asyncio.gather(marker, return_exceptions=True)

    def assert_released(self):
        gc.collect()
        assert self.exited.is_set()
        assert not any(reference() is not None for reference in self.references), "history escaped predicate"


async def conflict(awaitable, error=storage.MessageConflict):
    try:
        await awaitable
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


async def isolated_read(directory, action, case):
    async with fixture(directory) as f:
        barrier = DecodeBarrier()
        queued = closing = None
        with patch.object(storage, "_message", barrier.decode):
            operation = asyncio.create_task(f.operation(action))
            try:
                await barrier.wait(operation)
                # This actual Source admission must commit while the first reader is pinned.
                peer = await f.peer.accept("peer-input", Input(()))
                assert peer.source == "peer" and not operation.done()
                if case == "finish":
                    terminal = f.outputs.append("finished", Output((), "complete"))
                    assert terminal.seq > f.failure.seq
                else:
                    attempted = asyncio.Event()
                    async def next_input():
                        attempted.set()
                        return await f.source.accept("next-input", Input(()))
                    queued = asyncio.create_task(next_input())
                    await attempted.wait()
                    assert not queued.done() and f.reader.get("next-input") is None
                if case in {"cancel", "close"}:
                    operation.cancel()
                    await asyncio.sleep(0)
                    operation.cancel()
                    await asyncio.sleep(0)
                    assert not operation.done() and not barrier.exited.is_set()
                    assert queued is not None and not queued.done()
                    if case == "close":
                        closing = asyncio.create_task(f.tasks.close())
                        await asyncio.sleep(0)
                        assert not closing.done(), "Tasks.close escaped the physical read"
                barrier.release.set()
                if case == "finish":
                    await conflict(operation, storage.SourceHeadConflict)
                    assert f.reader.get("operation") is None
                    await conflict(f.operation(action, "after-finish"))
                elif case in {"cancel", "close"}:
                    await conflict(operation, asyncio.CancelledError)
                    assert f.reader.get("operation") is None
                    assert queued is not None
                    if closing is None:
                        await queued
                    else:
                        from agent.plugin_composition.tasks import TaskServiceClosed
                        await conflict(queued, TaskServiceClosed)
                        await closing
                else:
                    committed = await operation
                    assert queued is not None
                    later = await queued
                    assert committed.seq < later.seq
                    assert isinstance(committed.body, Control)
                    assert committed.body.through_seq == f.failure.seq
                    assert f.notified.count("conversation") == 2
                barrier.assert_released()
                return {"action": action, "case": case, "tail_decodes_off_loop": barrier.decoded,
                        "peak_live_tail_messages": barrier.peak_live}
            finally:
                barrier.release.set()
                await asyncio.gather(operation, *([queued] if queued else []),
                    *([closing] if closing else []), return_exceptions=True)


async def replay_and_validation(directory, action):
    async with fixture(directory) as f:
        committed = await f.operation(action)
        notifications = len(f.notified)
        newer = f.inputs.append("newer", Input(()))
        f.outputs.append("complete", Output((), "complete"))
        release = asyncio.Event()
        async def work(_):
            await release.wait()
        running = await f.tasks.admit(("session", "conversation"), lambda slot: slot.start(work))
        f.gate.prepare("restart")
        try:
            # Replay precedes current head/handle/gate/closed-tail validation.
            if action == "resume":
                replay = await f.operation(action)
                await conflict(f.source.resume("operation", "newer"))
            else:
                replay = await f.source.control("operation", committed.body,
                    expected_head=-1, handle="obsolete-handle")
            assert replay == committed and running.active
            assert len(f.notified) == notifications
            peer = f.writer(Input, "peer").append("foreign", Input(()))
            await conflict(f.source.resume("invalid-resume", "foreign"))
            await conflict(f.source.control("invalid-abandon", Control("abandon", peer.seq),
                expected_head=peer.seq, handle=None))
            await conflict(f.source.resume("newer", "newer"))
            assert f.reader.get("invalid-resume") is None
            assert f.reader.get("invalid-abandon") is None
            assert f.reader.get("newer") == newer
            assert f.log._connection.execute("SELECT COUNT(*) FROM messages WHERE id='operation'").fetchone()[0] == 1
        finally:
            release.set()
            await running.join()
        return {"action": action, "case": "replay_source_identity_validation"}


async def loop_checks(directory):
    async with fixture(directory) as f:
        release = asyncio.Event()
        async def work(_):
            await release.wait()
        running = await f.tasks.admit(("session", "conversation"), lambda slot: slot.start(work))
        await conflict(f.operation("resume"))
        await conflict(f.operation("abandon"))
        result = await f.source.control("with-handle", Control("abandon", f.failure.seq),
            expected_head=f.failure.seq, handle=running.handle)
        assert result.body.action == "abandon" and running.superseded
        release.set()
        await conflict(running.join(), asyncio.CancelledError)
        fresh = f.inputs.append("fresh", Input(()))
        f.controls.append("fresh-pause", Control("pause", fresh.seq))
        f.gate.prepare("restart")
        await conflict(f.source.resume("gate-rejected", "fresh"), RestartPendingError)
        assert f.reader.get("gate-rejected") is None
        return {"case": "loop_task_handle_restart_checks"}


async def run(directory):
    # Instrument owner operations, not the predicates being tested.
    current = TaskSlot.current.fget
    require, cancel, supersede = TaskSlot.require, Task.cancel, Task.supersede
    def loop_current(slot):
        on_loop()
        return current(slot)
    def loop_require(slot, handle):
        on_loop()
        return require(slot, handle)
    def loop_cancel(task):
        on_loop()
        return cancel(task)
    def loop_supersede(task):
        on_loop()
        return supersede(task)
    results = []
    with patch.object(TaskSlot, "current", property(loop_current)), \
         patch.object(TaskSlot, "require", loop_require), \
         patch.object(Task, "cancel", loop_cancel), \
         patch.object(Task, "supersede", loop_supersede):
        for action in ((OPTIONS.action,) if OPTIONS.action else ("abandon", "resume")):
            for case in ("serialize", "finish", "cancel", "close"):
                results.append(await isolated_read(directory / f"{action}-{case}", action, case))
            results.append(await replay_and_validation(directory / f"{action}-replay", action))
        results.append(await loop_checks(directory / "loop-checks"))
    return {"source": str(SOURCE), "tail_rows_per_case": 192,
        "tail_text_bytes_per_case": 192 * 65536, "cases": results,
        "production_provider_delivery": "not_run"}


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-source-control-read-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary))), indent=2))
