"""用真实未闭合 Input/ToolResult 量测 Source 判定、恢复与材料准入。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

from agent.plugin_composition.tasks import Tasks
from plugins.sources.session import SourceSession
import session.log as storage
from session.log import SourceHeadConflict
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult


async def run(directory: Path) -> dict:
    """使用实际 Message owner 创建事实，测量阶段不读取或改写原正文。"""
    log = storage.MessageLog(directory / "sessions.db")
    tasks = Tasks()
    writers = {}
    loop_thread = threading.get_ident()
    original_decode = storage._message
    decoded = []
    measurements = []
    release = asyncio.Event()
    log.save_binding("tool", {"target": "native-read-fixture"})

    def writer(session, body, *, call_ref=None):
        return log.writer(session, source="conversation", author="fixture", body_types=(body,),
                          content={"text": lambda _: ContentReferences()}, call_ref=call_ref,
                          check_call=lambda _: None)

    async def measure(case, operation):
        decoded.clear()
        tick = asyncio.Event()
        delay = []
        began = time.perf_counter()
        last_tick = began
        gaps = []
        heartbeat = None
        loop = asyncio.get_running_loop()

        def sample_gap():
            nonlocal last_tick, heartbeat
            now = time.perf_counter()
            gaps.append(now - last_tick)
            last_tick = now
            heartbeat = loop.call_later(0.001, sample_gap)

        def peer():
            delay.append(time.perf_counter() - began)
            tick.set()
        loop.call_soon(peer)
        heartbeat = loop.call_later(0.001, sample_gap)
        try:
            value = await operation()
            elapsed = time.perf_counter() - began
            await tick.wait()
        finally:
            heartbeat.cancel()
            gaps.append(time.perf_counter() - last_tick)
        on_loop = sum(thread == loop_thread for _, thread in decoded)
        old_on_loop = sum(thread == loop_thread and identity in before for identity, thread in decoded)
        measurements.append({"case": case, "seconds": elapsed,
                             "peer_callback_delay_seconds": delay[0],
                             "max_loop_gap_seconds": max(gaps),
                             "decoded_rows": len(decoded), "loop_decoded_rows": on_loop,
                             "loop_decoded_original_rows": old_on_loop})
        if not args.baseline:
            assert old_on_loop == 0, (case, decoded)
        return value

    try:
        # 1. 原 writer 保存巨大输入、真实 call ref 与完整工具结果；不直接灌 SQL。
        for session, closed in (("open", False), ("closed", True)):
            inputs, outputs, controls = (writer(session, kind) for kind in (Input, Output, Control))
            target = inputs.append(session + "-input", Input((ContentPart("text", "i" * 1_048_576),)))
            call = outputs.append(session + "-call", Output((ToolCall("tool", {}),), "continue"))
            if closed:
                outputs.append(session + "-complete", Output((), "complete"))
            result = writer(session, ToolResult, call_ref=CallRef(call.message_id, 0))
            result.append(session + "-tool", ToolResult(CallRef(call.message_id, 0), "success",
                                                      (ContentPart("text", "t" * 12_582_912),)))
            if not closed:
                controls.append("failure", Control("failure", log.reader(session).head(), "native fixture"))
            writers[session] = (outputs, SourceSession(reader=log.reader(session), inputs=inputs,
                                                      controls=controls, tasks=tasks))
        before = {row["id"]: tuple(row) for row in log._connection.execute("SELECT * FROM messages")}
        body_bytes = log._connection.execute("SELECT SUM(length(CAST(body AS BLOB))) FROM messages").fetchone()[0]

        def decode(row):
            decoded.append((row["id"], threading.get_ident()))
            return original_decode(row)
        storage._message = decode
        # 2. 同一实际消费者在旧基线和新源码执行，分别报告解码和 peer 响应。
        async def pending(session):
            if args.baseline:
                return SourceSession.needs_reply(log.reader(session), "conversation")
            return await SourceSession.needs_reply(log.reader(session), "conversation")

        assert not await measure("pending_paused_open_tail", lambda: pending("open"))
        source = writers["open"][1]
        retry = await measure("resume_open_tail", lambda: source.resume("retry", "open-input"))
        assert isinstance(retry.body, Control) and retry.body.action == "resume"
        assert await measure("pending_resumed_open_tail", lambda: pending("open"))
        assert await measure("resume_same_id_open_tail", lambda: source.resume("retry", "open-input")) == retry

        async def wait_program(_task, _reader, _source):
            await release.wait()
        task = await measure("start_open_tail", lambda: source.start(wait_program))
        assert task is not None and task.active
        await measure("pause_open_tail", lambda: source.pause("stop"))
        assert not task.active
        # 3. 已关闭 Turn 的晚到工具结果仍完整保存；材料准入不重新执行原工具。
        output, source = writers["closed"]
        async def complete_program(_task, reader, check_control=None):
            if check_control is not None:
                check_control()
            return output.append("material-complete", Output((), "complete"),
                                 expected_source_head=reader.head(source="conversation"))
        completion = await measure("complete_with_late_large_tool_result", lambda: source.complete(complete_program))
        assert completion.message_id == "material-complete"
        storage._message = original_decode
        after = {row["id"]: tuple(row) for row in log._connection.execute("SELECT * FROM messages")}
        assert all(after[identity] == row for identity, row in before.items())
        assert log._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not log._connection.execute("PRAGMA foreign_key_check").fetchall()
        if args.baseline:
            assert any(item["loop_decoded_rows"] for item in measurements)
        return {"source": str(args.source.resolve()), "baseline": args.baseline,
                "body_bytes": body_bytes, "original_rows_equal": True,
                "integrity": "ok", "foreign_keys": [], "measurements": measurements,
                "original_rows_sha256": hashlib.sha256(repr(before).encode()).hexdigest(),
                "paid_provider": "unrun", "production_p99": "unrun"}
    finally:
        storage._message = original_decode
        release.set()
        await tasks.close()
        log.close()


async def run_race(directory: Path, *, cancel: bool) -> dict:
    """屏障固定真实解码，核对其他来源、旧快照 CAS 与关闭排空。"""
    log = storage.MessageLog(directory / "sessions.db")
    tasks = Tasks()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered, release = asyncio.Event(), threading.Event()
    original_decode = storage._message
    blocked = False
    notifications = []
    operation = closing = None

    def writer(session, body, *, call_ref=None):
        return log.writer(session, source="conversation", author="fixture", body_types=(body,),
                          content={"text": lambda _: ContentReferences()}, call_ref=call_ref,
                          check_call=lambda _: None)

    def changed(reader, source, pending):
        assert threading.get_ident() == loop_thread
        notifications.append((reader.session_id, source, pending))

    async def next_turn():
        ready = loop.create_future()
        loop.call_soon(ready.set_result, None)
        await ready

    try:
        # 1. 用原 writer 建立开放前缀；屏障只拦一次真实大工具结果解码。
        inputs, controls, outputs = (writer("blocked", body) for body in (Input, Control, Output))
        target = inputs.append("input", Input((ContentPart("text", "original"),)))
        log.save_binding("tool", {"target": "native-race-fixture"})
        call = outputs.append("call", Output((ToolCall("tool", {}),), "continue"))
        result = writer("blocked", ToolResult, call_ref=CallRef(call.message_id, 0))
        result.append("blocked-tool", ToolResult(CallRef(call.message_id, 0), "success",
                      (ContentPart("text", "t" * 12_582_912),)))
        controls.append("failure", Control("failure", log.reader("blocked").head(), "fixture"))
        before = {row["id"]: tuple(row) for row in log._connection.execute("SELECT * FROM messages")}
        source = SourceSession(reader=log.reader("blocked"), inputs=inputs, controls=controls,
                               tasks=tasks, changed=changed)
        peer = SourceSession(reader=log.reader("peer"), inputs=writer("peer", Input),
                             controls=writer("peer", Control), tasks=tasks, changed=changed)

        def decode(row):
            nonlocal blocked
            if row["id"] == "blocked-tool" and not blocked:
                assert threading.get_ident() != loop_thread
                blocked = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(10), "worker 屏障未释放"
            return original_decode(row)

        storage._message = decode
        operation = asyncio.create_task(source.resume("resume", target.message_id))
        await asyncio.wait_for(entered.wait(), 5)
        # 2. 被固定的只读连接不占其他来源准入；通知仍同步在原 loop 发生。
        accepted = await peer.accept("peer-input", Input((ContentPart("text", "work"),)))
        assert notifications == [("peer", "conversation", True)]
        await peer.pause("peer-pause")
        assert notifications[-1] == ("peer", "conversation", False)
        assert accepted.message_id == "peer-input" and not operation.done()

        if cancel:
            operation.cancel()
            closing = asyncio.create_task(tasks.close())
            await next_turn()
            await next_turn()
            assert not operation.done() and not closing.done(), "关闭早于实际只读连接排空"
            release.set()
            result = await asyncio.gather(operation, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError), result
            await closing
            conclusion = "caller cancelled; Tasks.close waited for physical reader"
        else:
            await inputs.append_async("new-input", Input((ContentPart("text", "new"),)),
                                      on_commit=lambda _message, _created: None)
            release.set()
            result = await asyncio.gather(operation, return_exceptions=True)
            assert isinstance(result[0], SourceHeadConflict), result
            conclusion = "fixed-prefix resume rejected later same-source Input by CAS"
        # 3. 冲突/取消都不追加 resume，不改正文；peer 的 Input/Control 正常保留。
        assert log.reader("blocked").get("resume") is None
        after = {row["id"]: tuple(row) for row in log._connection.execute("SELECT * FROM messages")}
        assert all(after[identity] == row for identity, row in before.items())
        assert log._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not log._connection.execute("PRAGMA foreign_key_check").fetchall()
        return {"case": "cancel_and_close" if cancel else "pinned_source_head_conflict",
                "conclusion": conclusion, "peer_append_and_pause_before_release": True,
                "notification_thread": "original loop", "resume_rows": 0,
                "original_rows_equal": True, "integrity": "ok", "foreign_keys": []}
    finally:
        release.set()
        for pending in (operation, closing):
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
        storage._message = original_decode
        await tasks.close()
        log.close()


async def main(directory: Path) -> dict:
    """每个交错使用独立数据库，全部物理 owner 退出后才删除临时目录。"""
    performance = directory / "performance"
    performance.mkdir()
    report = await run(performance)
    if not args.baseline:
        report["races"] = []
        for cancel in (False, True):
            path = directory / str(cancel)
            path.mkdir()
            report["races"].append(await run_race(path, cancel=cancel))
    return report


with tempfile.TemporaryDirectory(prefix="source-open-tail-") as temporary:
    print(json.dumps(asyncio.run(main(Path(temporary))), ensure_ascii=False, indent=2))
