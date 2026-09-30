"""用真实重试准入量测已关闭历史是否被重复解码。"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
SOURCE = parser.parse_args().source
sys.path.insert(0, str(SOURCE))
from agent.plugin_composition.tasks import Tasks
from plugins.sources.session import SourceSession
import session.log as storage
from session.message import ContentPart, ContentReferences, Control, Input, Output
from plugins.subagent.runtime import Subagents


async def check_semantics(directory):
    """用真实准入核对关闭、来源、活动任务及条件追加。"""
    checked = []
    for case in ("completed", "abandoned", "missing_control", "old_control",
                 "active_task", "new_input_before_commit"):
        path = directory / case
        path.mkdir()
        log, tasks = storage.MessageLog(path / "sessions.db"), Tasks()
        inputs = log.writer("s", author="user", source="conversation", body_types=(Input,), content={})
        outputs = log.writer("s", author="agent", source="conversation", body_types=(Output,), content={})
        controls = log.writer("s", author="app", source="conversation", body_types=(Control,), content={})
        source = SourceSession(reader=log.reader("s"), inputs=inputs, controls=controls, tasks=tasks)
        release = asyncio.Event()
        try:
            # 1. 保留原规则：当前 Input 前的最后 Control 也参与选择。
            if case == "old_control":
                previous = inputs.append("previous", Input(()))
                controls.append("old-failure", Control("failure", previous.seq, "old"))
            target = inputs.append("current", Input(()))
            if case == "completed":
                outputs.append("closed", Output((), "complete"))
            elif case == "abandoned":
                controls.append("closed", Control("abandon", target.seq))
            if case not in {"missing_control", "old_control"}:
                controls.append("failure", Control("failure", log.reader("s").head(), "fixture"))
            if case == "active_task":
                entered = asyncio.Event()
                async def work(_):
                    entered.set()
                    await release.wait()
                await tasks.admit(("s", "conversation"), lambda slot: slot.start(work))
                await entered.wait()
            if case == "new_input_before_commit":
                append = controls.append
                def interleaved(identity, body, **kwargs):
                    inputs.append("newer", Input(()))
                    return append(identity, body, **kwargs)
                controls.append = interleaved
            # 2. 拒绝不得落 resume；允许只落同一事实并可重放。
            try:
                result = await source.resume("retry", "current")
            except storage.MessageConflict:
                assert case != "old_control"
                assert log.reader("s").get("retry") is None
            else:
                assert case == "old_control"
                assert await source.resume("retry", "current") == result
            checked.append(case)
        finally:
            release.set()
            await tasks.close()
            log.close()
    return checked


async def run(directory):
    """分开量测同步准入和已合入的异步子任务读取。"""
    log = storage.MessageLog(directory / "sessions.db")
    tasks = Tasks()
    inputs = log.writer("s", author="user", source="conversation", body_types=(Input,),
                        content={"text": lambda _: ContentReferences()})
    outputs = log.writer("s", author="agent", source="conversation", body_types=(Output,),
                         content={"text": lambda _: ContentReferences()})
    controls = log.writer("s", author="app", source="conversation", body_types=(Control,), content={})
    source = SourceSession(reader=log.reader("s"), inputs=inputs, controls=controls, tasks=tasks)
    results = []
    try:
        # 1. 通过实际 writer 创建大闭合历史，不直接注入 SQL 或调用正文改写。
        text = "x" * 8192
        for index in range(1300):
            inputs.append(f"old-input-{index}", Input(()))
            outputs.append(f"old-output-{index}", Output((ContentPart("text", text),), "complete"))
        target = inputs.append("current-input", Input(()))
        controls.append("current-failure", Control("failure", target.seq, "fixture failure"))
        body_bytes = log._connection.execute("SELECT SUM(length(CAST(body AS BLOB))) FROM messages").fetchone()[0]
        original = storage._message
        loop_thread = threading.get_ident()
        decoded = []
        def decode(row):
            decoded.append((row["id"], threading.get_ident() == loop_thread))
            return original(row)
        storage._message = decode
        try:
            # 2. 准入自身没有等待；已排队的无关回调给出实际事件循环延迟。
            for label in ("first_resume", "same_identity_replay", "subagent_outcome"):
                decoded.clear()
                event = asyncio.Event()
                marker = []
                start = time.perf_counter()
                def tick():
                    marker.append(time.perf_counter() - start)
                    event.set()
                asyncio.get_running_loop().call_soon(tick)
                if label == "subagent_outcome":
                    assert await Subagents.outcome(log.reader("s")) is None
                else:
                    message = await source.resume("resume", "current-input")
                    assert isinstance(message.body, Control) and message.body.action == "resume"
                elapsed = time.perf_counter() - start
                await event.wait()
                results.append({"case": label, "seconds": elapsed,
                    "peer_callback_delay_seconds": marker[0], "decoded_rows": len(decoded),
                    "decoded_old_rows": sum(identity.startswith("old-") for identity, _ in decoded),
                    "loop_decoded_rows": sum(on_loop for _, on_loop in decoded)})
        finally:
            storage._message = original
        assert log._connection.execute("SELECT COUNT(*) FROM messages WHERE id='resume'").fetchone()[0] == 1
        return {"source": str(SOURCE), "history_body_bytes": body_bytes, "closed_turns": 1300,
                "measurements": results, "semantic_cases": await check_semantics(directory),
                "production_p99": "unrun"}
    finally:
        await tasks.close()
        log.close()


with tempfile.TemporaryDirectory(prefix="akashic-resume-read-") as temporary:
    print(json.dumps(asyncio.run(run(Path(temporary))), ensure_ascii=False, indent=2))
