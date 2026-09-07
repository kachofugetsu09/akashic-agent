from __future__ import annotations

import asyncio
from typing import cast

import pytest

from agent.control.protocol.method import OutputReservation
from agent.plugin_composition.tasks import Task, TaskSlot, Tasks
from agent.restart import RestartGate
from infra.control.connection import _FrameReservation
from plugins.programmatic.control import Programmatic


def _output(session_id: str, message_id: str) -> dict[str, object]:
    return {
        "id": message_id,
        "session_id": session_id,
        "body": {"kind": "output", "finish": "complete", "parts": []},
    }


@pytest.mark.asyncio
async def test_frame_seen_before_expect_waits_for_drain() -> None:
    reservation = _FrameReservation("session:a", "input:a")
    written = asyncio.get_running_loop().create_future()
    reservation.observe({"result": {"items": [_output("session:a", "output:a")]}}, written)

    waiter = asyncio.create_task(reservation.wait_output("output:a"))
    await asyncio.sleep(0)
    assert not waiter.done()

    written.set_result(None)
    await waiter


@pytest.mark.asyncio
async def test_frame_drain_failure_rejects_delivery() -> None:
    reservation = _FrameReservation("session:a", "input:a")
    written = asyncio.get_running_loop().create_future()
    reservation.observe({"result": {"items": [_output("session:a", "output:a")]}}, written)
    error = ConnectionError("writer failed")
    written.set_exception(error)

    with pytest.raises(ConnectionError, match="writer failed"):
        await reservation.wait_output("output:a")


@pytest.mark.asyncio
async def test_same_message_id_from_another_session_is_ignored() -> None:
    reservation = _FrameReservation("session:a", "input:a")
    other = asyncio.get_running_loop().create_future()
    reservation.observe({"result": {"items": [_output("session:b", "same")]}}, other)
    other.set_result(None)

    waiter = asyncio.create_task(reservation.wait_output("same"))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), 0.01)
    assert not waiter.done()

    current = asyncio.get_running_loop().create_future()
    reservation.observe({"result": {"items": [_output("session:a", "same")]}}, current)
    current.set_result(None)
    await waiter


class _Transport:
    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id
        self.calls = 0
        self.reservation = cast(OutputReservation, object())

    def reserve_input(self, session_id: str, input_id: str) -> OutputReservation:
        self.calls += 1
        return self.reservation


@pytest.mark.asyncio
async def test_duplicate_input_on_second_connection_keeps_first_owner() -> None:
    programmatic = object.__new__(Programmatic)
    programmatic._reservations = {}
    first = _Transport("first")
    second = _Transport("second")

    owned = programmatic.reserve_input("session:a", "input:a", first)
    duplicate = programmatic.reserve_input("session:a", "input:a", second)

    assert duplicate is owned
    assert first.calls == 1
    assert second.calls == 0


@pytest.mark.asyncio
async def test_root_permit_releases_when_task_is_cancelled_before_run() -> None:
    gate = RestartGate(boot_id="boot", supervised=True, commit=lambda _request_id: None)
    tasks = Tasks()

    async def never_runs(_task: Task) -> None:
        raise AssertionError("早取消的 Task 不应进入 operation")

    def admit(slot: TaskSlot) -> Task:
        permit = gate.acquire()
        task = slot.start(never_runs, child_permit=permit.child)
        task.on_done(permit.release)
        return task

    try:
        task = await tasks.admit("root", admit)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task.join()
        await asyncio.sleep(0)
        assert gate.permit_count == 0
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_root_permit_releases_after_running_task_cleanup() -> None:
    gate = RestartGate(boot_id="boot", supervised=True, commit=lambda _request_id: None)
    tasks = Tasks()
    started = asyncio.Event()

    async def runs(_task: Task) -> None:
        started.set()
        await asyncio.Event().wait()

    def admit(slot: TaskSlot) -> Task:
        permit = gate.acquire()
        task = slot.start(runs, child_permit=permit.child)
        task.on_done(permit.release)
        return task

    try:
        task = await tasks.admit("root", admit)
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task.join()
        assert gate.permit_count == 0
    finally:
        await tasks.close()
