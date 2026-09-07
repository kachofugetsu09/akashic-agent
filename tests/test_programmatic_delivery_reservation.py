from __future__ import annotations

import asyncio

import pytest

from agent.control.frame_book import FrameBook
from agent.plugin_composition.tasks import Task, TaskSlot, Tasks
from agent.restart import RestartGate
from session.message import CallRef


def _output(session_id: str, message_id: str) -> dict[str, object]:
    return {
        "id": message_id,
        "session_id": session_id,
        "seq": 0,
        "timestamp": "2026-09-07T00:00:00+00:00",
        "author": "assistant",
        "source": "conversation",
        "attachments": [],
        "body": {"kind": "output", "finish": "complete", "parts": []},
    }


def _page(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": 2,
        "session_id": rows[0]["session_id"] if rows else "session:a",
        "items": rows,
        "after_seq": -1,
        "through_seq": 0,
        "next_after_seq": 0,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_frame_seen_before_expect_waits_for_drain() -> None:
    book = FrameBook()
    reservation = book.route_input("session:a", "input:a", "connection:a", lambda: "output:a")
    written = asyncio.get_running_loop().create_future()
    tracked = book.resolve_page("connection:a", _page([_output("session:a", "output:a")]))
    book.attach_page(tracked, written)

    waiter = asyncio.create_task(reservation.wait_output("output:a"))
    await asyncio.sleep(0)
    assert not waiter.done()

    written.set_result(None)
    await waiter


@pytest.mark.asyncio
async def test_frame_drain_failure_rejects_delivery() -> None:
    book = FrameBook()
    reservation = book.route_input("session:a", "input:a", "connection:a", lambda: "output:a")
    written = asyncio.get_running_loop().create_future()
    tracked = book.resolve_page("connection:a", _page([_output("session:a", "output:a")]))
    book.attach_page(tracked, written)
    error = ConnectionError("writer failed")
    written.set_exception(error)

    with pytest.raises(ConnectionError, match="writer failed"):
        await reservation.wait_output("output:a")


@pytest.mark.asyncio
async def test_frame_book_releases_normal_route_after_exact_frame_drain() -> None:
    book = FrameBook()
    ending = "output:a"
    reservation = book.route_input(
        "session:a", "input:a", "connection:a", lambda: ending,
    )
    page = _page([_output("session:a", ending)])
    tracked = book.resolve_page("connection:a", page)
    assert tracked
    written = asyncio.get_running_loop().create_future()
    book.attach_page(tracked, written)
    waiter = asyncio.create_task(reservation.wait_output(ending))
    await asyncio.sleep(0)
    assert book._routes  # type: ignore[attr-defined]
    written.set_result(None)
    await waiter
    assert not book._routes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_frame_book_ignores_wrong_connection_and_releases_result_only_route() -> None:
    book = FrameBook()
    book.route_input("session:a", "input:a", "connection:a", lambda: "output:a")
    page = _page([_output("session:a", "output:a")])
    assert not book.resolve_page("connection:b", page)
    assert book._routes  # type: ignore[attr-defined]
    book.release_input("session:a", "input:a")
    assert not book._routes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_frame_book_disconnect_fails_and_releases_route() -> None:
    book = FrameBook()
    reservation = book.route_input("session:a", "input:a", "connection:a", lambda: "output:a")
    waiter = asyncio.create_task(reservation.wait_output("output:a"))
    await asyncio.sleep(0)
    error = ConnectionError("connection closed")
    book.fail_connection("connection:a", error)
    with pytest.raises(ConnectionError, match="connection closed"):
        await waiter
    assert not book._routes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_frame_book_settle_releases_completed_route_without_a_read() -> None:
    book = FrameBook()
    reservation = book.route_input("session:a", "input:a", "connection:a", lambda: "output:a")
    book.settle_input("session:a", "input:a")
    assert not book._routes  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="没有最终 Output"):
        await reservation.wait_output("output:a")


def test_frame_book_resume_stage_preserves_old_owner_until_commit() -> None:
    book = FrameBook()
    old = book.route_input("session:a", "input:a", "connection:a", lambda: "old")
    stage = book.stage_input("session:a", "input:a", "connection:b", lambda: "new")
    stage.abort()
    assert book.route_input("session:a", "input:a", "connection:c", lambda: "old") is old
    committed = book.stage_input("session:a", "input:a", "connection:b", lambda: "new")
    new = committed.commit()
    assert new is not old
    assert not book.resolve_page("connection:a", _page([_output("session:a", "old")]))
    assert book.resolve_page("connection:b", _page([_output("session:a", "new")]))


@pytest.mark.asyncio
async def test_frame_book_claim_binds_ending_on_page_and_survives_until_consume() -> None:
    book = FrameBook()
    book.route_input("session:a", "input:a", "connection:a", lambda: "output:a")
    claim = book.arm_claim("session:a", "input:a", CallRef("call", 0))
    assert claim.ending_message_id is None
    page = _page([_output("session:a", "output:a")])
    tracked = book.resolve_page("connection:a", page)
    written = asyncio.get_running_loop().create_future()
    book.attach_page(tracked, written)
    waiter = asyncio.create_task(claim.wait_output())
    await asyncio.sleep(0)
    assert not waiter.done()
    written.set_result(None)
    await waiter
    assert claim.ending_message_id == "output:a"
    assert book._routes  # type: ignore[attr-defined]
    claim.consume()
    assert not book._routes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_frame_book_claim_keeps_disconnect_error_after_route_is_removed() -> None:
    book = FrameBook()
    book.route_input("session:a", "input:a", "connection:a", lambda: "output:a")
    claim = book.arm_claim("session:a", "input:a", CallRef("call", 0))
    error = ConnectionError("connection closed")
    book.fail_connection("connection:a", error)
    assert not book._routes  # type: ignore[attr-defined]
    assert book.claim_for("session:a", CallRef("call", 0)) is claim
    with pytest.raises(ConnectionError, match="connection closed"):
        await claim.wait_output()
    claim.abort()
    assert book.claim_for("session:a", CallRef("call", 0)) is None


@pytest.mark.asyncio
async def test_frame_book_stage_old_drain_cannot_complete_new_route() -> None:
    book = FrameBook()
    old = book.route_input("session:a", "input:a", "connection:a", lambda: "old")
    old_tracked = book.resolve_page("connection:a", _page([_output("session:a", "old")]))
    old_written = asyncio.get_running_loop().create_future()
    book.attach_page(old_tracked, old_written)
    stage = book.stage_input("session:a", "input:a", "connection:b", lambda: "new")
    new = stage.commit()
    new_tracked = book.resolve_page("connection:b", _page([_output("session:a", "new")]))
    new_written = asyncio.get_running_loop().create_future()
    book.attach_page(new_tracked, new_written)
    waiter = asyncio.create_task(new.wait_output("new"))
    await asyncio.sleep(0)
    old_written.set_result(None)
    await asyncio.sleep(0)
    assert not waiter.done()
    new_written.set_result(None)
    await waiter
    assert old is not new


@pytest.mark.asyncio
async def test_frame_book_old_failed_drain_cannot_fail_new_route() -> None:
    book = FrameBook()
    book.route_input("session:a", "input:a", "connection:a", lambda: "old")
    old_tracked = book.resolve_page("connection:a", _page([_output("session:a", "old")]))
    old_written = asyncio.get_running_loop().create_future()
    book.attach_page(old_tracked, old_written)
    stage = book.stage_input("session:a", "input:a", "connection:b", lambda: "new")
    new = stage.commit()
    new_tracked = book.resolve_page("connection:b", _page([_output("session:a", "new")]))
    new_written = asyncio.get_running_loop().create_future()
    book.attach_page(new_tracked, new_written)
    waiter = asyncio.create_task(new.wait_output("new"))
    await asyncio.sleep(0)
    old_written.set_exception(ConnectionError("old writer failed"))
    await asyncio.sleep(0)
    assert not waiter.done()
    new_written.set_result(None)
    await waiter


@pytest.mark.asyncio
async def test_frame_book_stage_drains_before_resume_commit_without_losing_new_owner() -> None:
    book = FrameBook()
    book.route_input("session:a", "input:a", "connection:a", lambda: "old")
    entered = asyncio.Event()
    release = asyncio.Event()
    stage = book.stage_input("session:a", "input:a", "connection:b", lambda: "new")

    async def resume() -> None:
        entered.set()
        await release.wait()
        stage.commit()

    task = asyncio.create_task(resume())
    await entered.wait()
    tracked = book.resolve_page("connection:b", _page([_output("session:a", "new")]))
    written = asyncio.get_running_loop().create_future()
    book.attach_page(tracked, written)
    written.set_result(None)
    await asyncio.sleep(0)
    release.set()
    await task
    await stage.reservation.wait_output("new")


@pytest.mark.asyncio
async def test_frame_book_aborted_stage_drain_cannot_release_old_owner() -> None:
    book = FrameBook()
    old = book.route_input("session:a", "input:a", "connection:a", lambda: "old")
    stage = book.stage_input("session:a", "input:a", "connection:b", lambda: "new")
    tracked = book.resolve_page("connection:b", _page([_output("session:a", "new")]))
    staged_written = asyncio.get_running_loop().create_future()
    book.attach_page(tracked, staged_written)
    stage.abort()
    old_tracked = book.resolve_page("connection:a", _page([_output("session:a", "old")]))
    old_written = asyncio.get_running_loop().create_future()
    book.attach_page(old_tracked, old_written)
    staged_written.set_result(None)
    waiter = asyncio.create_task(old.wait_output("old"))
    await asyncio.sleep(0)
    assert not waiter.done()
    old_written.set_result(None)
    await waiter


@pytest.mark.asyncio
async def test_same_message_id_from_another_session_is_ignored() -> None:
    book = FrameBook()
    reservation = book.route_input("session:a", "input:a", "connection:a", lambda: "same")
    other = asyncio.get_running_loop().create_future()
    assert not book.resolve_page("connection:a", _page([_output("session:b", "same")]))
    other.set_result(None)

    waiter = asyncio.create_task(reservation.wait_output("same"))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), 0.01)
    assert not waiter.done()

    current = asyncio.get_running_loop().create_future()
    tracked = book.resolve_page("connection:a", _page([_output("session:a", "same")]))
    book.attach_page(tracked, current)
    current.set_result(None)
    await waiter


@pytest.mark.asyncio
async def test_duplicate_input_on_second_connection_keeps_first_owner() -> None:
    book = FrameBook()
    owned = book.route_input("session:a", "input:a", "first", lambda: "output:a")
    duplicate = book.route_input("session:a", "input:a", "second", lambda: "output:a")

    assert duplicate is owned


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
