"""真实 Message/Turn/FrameBook/RestartGate 验证重启按回执而非来源名称。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.plugin_composition import CompositionRoot
from plugins.gateway.contract import CONTROL_FRAMES
from plugins.gateway.frames import FrameBook
from agent.restart import RestartGate, RestartRejectedError
from plugins.delivery.api import FinalOutputDelivery
from plugins.message_push.restart import RestartRequest, RestartWatcher
from plugins.programmatic.control import Programmatic
from plugins.turn_projection.plugin import TurnProjection
from session.log import MessageLog
from session.message import CallRef, ContentPart, ContentReferences, Input, Output, ToolCall


async def check(workspace: Path) -> None:
    root = CompositionRoot("restart-receipt-scenario")
    log = MessageLog(workspace / "sessions.db")
    frames = FrameBook()
    await root.context.provide(CONTROL_FRAMES, frames)
    log.save_binding("tool-binding", {"scenario": "reference-only"})
    writers = []
    async def barrier():
        event = asyncio.Event()
        asyncio.get_running_loop().call_soon(event.set)
        await event.wait()
    try:
        for mode in ("frame", "provider", "missing", "disconnect", "wrong-ending", "cancel"):
            source = "programmatic" if mode == "missing" else "another-source"
            session = "scenario:" + mode
            writer = log.writer(session, source=source, author="scenario", body_types=(Input, Output),
                                content={"text": lambda _: ContentReferences()}, check_call=lambda _: None)
            writers.append(writer)
            input_id, call_id, ending_id = (mode + ":" + suffix for suffix in ("input", "call", "ending"))
            writer.append(input_id, Input(()))
            writer.append(call_id, Output((ToolCall("tool-binding", {}),), "continue"))
            ending = writer.append(ending_id, Output((ContentPart("text", "done"),), "complete"))
            reader = log.reader(session)
            commits = []
            gate = RestartGate(boot_id=mode, supervised=True, commit=commits.append)
            watcher = RestartWatcher(root.context)
            watcher._gate = gate
            delivery = FinalOutputDelivery()
            request = RestartRequest(mode, session, source, CallRef(call_id, 0))
            entered, release = asyncio.Event(), asyncio.Event()
            if mode == "provider":
                class Receipt:
                    async def wait(self, reader, turn):
                        assert reader.session_id == session and turn.ending_message_id == ending.message_id
                        entered.set()
                        await release.wait()
                delivery.register(source, Receipt())
            elif mode == "missing":
                delivery.register(source, Programmatic(root.context))
            else:
                expected = "different-output" if mode == "wrong-ending" else ending.message_id
                frames.route_input_with_owner(session, input_id, mode, lambda expected=expected: expected)
                frames.arm_claim(session, input_id, request.call_ref)
                written = asyncio.get_running_loop().create_future()
                assert frames.track_page(mode, {"items": [{"id": expected, "session_id": session,
                    "body": {"kind": "output", "finish": "complete", "parts": []}}]}, written)
            pending = asyncio.create_task(watcher._wait_for_request(request, reader, TurnProjection(), delivery))
            await barrier()
            if mode == "missing":
                try:
                    await pending
                    raise AssertionError("无 route/claim 仍允许重启")
                except (RestartRejectedError, ConnectionError):
                    pass
            else:
                assert not pending.done() and not commits and not gate.accepting
                if mode == "provider":
                    await entered.wait()
                    release.set()
                elif mode == "disconnect":
                    frames.fail_connection(mode, ConnectionError("writer disconnected"))
                    written.set_result(None)
                elif mode == "cancel":
                    pending.cancel()
                    written.set_result(None)
                else:
                    written.set_result(None)
                result = (await asyncio.gather(pending, return_exceptions=True))[0]
                if mode in {"frame", "provider"}:
                    assert result is None and commits == [mode]
                else:
                    assert isinstance(result, (ConnectionError, RestartRejectedError, asyncio.CancelledError)), result
            if mode not in {"frame", "provider"}:
                assert not commits and gate.accepting
            assert frames.claim_for(session, request.call_ref) is None
            assert [row.message_id for row in reader.snapshot()] == [input_id, call_id, ending_id]
    finally:
        frames.close()
        for writer in writers:
            writer.expire()
        await root.dispose()
        log.close()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-restart-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: renamed source frame/provider, missing receipt, disconnect, wrong ending, cancel; no process restart")
