"""开放来源不阻挡其他已结算批次；跨代、重启与迟到结束均保全真实消息（#1705）。"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition import ServiceKey
from agent.plugins.manager import PluginManager
from plugins.compaction.contract import COMPACTION_READER, COMPACTION_SUMMARIES, StoredSummary
from plugins.content.contract import CONTENT
from plugins.content.plugin import check_text
from plugins.context.contract import CONTEXT
from plugins.ledger.contract import BINDINGS, MESSAGE_CATALOG, CallRef, ContentPart, Control, Input, Message, Output, ToolCall, ToolResult
from plugins.markdown_memory.plugin import _unapplied_groups
from plugins.markdown_memory.store import MarkdownProfileStore
from plugins.tools.contract import ALL_TOOLS, TOOLS
from plugins.turn_projection.contract import TURN_PROJECTION
from tests.test_default_reply import application

QUESTION = "Please open the pull request."
KEEP_INPUT_ERROR = "保留输入必须是当前来源的真实 Input，且不能重复"


# 只替换外部模型：10K 窗口按文本长度估算；回答里报告最终请求是否含当前输入、摘要、开放来源正文。
def configure(sources: Path) -> None:
    provider = sources / "test_provider/plugin.py"
    text = provider.read_text()
    start, end = text.index("    class Driver:"), text.index("    descriptor =")
    _ = provider.write_text(text[:start] + '''    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return len(str(messages)) // 4
        async def complete(self, request):
            from plugins.compaction.message_summary import HEADINGS
            calls.append(request)
            text = str(request.messages)
            if "[Source messages]" in text:
                leak = " LEAKED-OPEN-SOURCE" if "still working" in text else ""
                return LLMResponse("\\n".join(heading + "\\nRecorded progress." + leak for heading in HEADINGS))
            return LLMResponse(f"input={__QUESTION__ in text} summarized={'Recorded progress.' in text} "
                               f"leaked={'LEAKED-OPEN-SOURCE' in text}")
'''.replace("__QUESTION__", repr(QUESTION)) + text[end:])


def outcome(terminal: Message) -> dict[str, object]:
    body = terminal.body
    if isinstance(body, Control):
        return {"outcome": "failure", "reason": body.reason}
    assert isinstance(body, Output)
    texts = [part.value for part in body.parts if isinstance(part, ContentPart) and part.kind == "text"]
    return {"outcome": "complete", "answer": str(texts[0])}


async def run(directory: Path, *, other_open: bool, long_history: bool, second_turn: bool = False) -> dict[str, object]:
    """真实插件、Channel 输入、Reply follower、Compaction 与 SQLite；只有模型是确定性对端。"""
    directory.mkdir()
    async with application(directory, replying=True, compaction=True, output_tokens=512,
                           extra_sources=configure) as (log, host):
        root = host.live_root
        assert root is not None
        accept = root.context.require(CHANNEL_INPUT)

        def write(source: str, identity: str, body: Input | Output) -> None:
            writer = log.writer("test:room", author="fixture", source=source,
                                body_types=(type(body),), content={"text": check_text})
            _ = writer.append(identity, body)

        async def ask(identity: str) -> Message:
            accepted = await accept("test:room", identity, ChannelInboundMessage(
                "test", "user", "room", QUESTION, datetime.now(UTC), {}))
            async for _ in log.catalog().follow():
                for row in log.reader("test:room").snapshot(after_seq=accepted.seq):
                    if row.source == "conversation" and (
                        isinstance(row.body, Control) and row.body.action == "failure"
                        or isinstance(row.body, Output) and row.body.finish == "complete"
                    ):
                        return row
            raise AssertionError("message subscription closed")

        # 1. 已结束的长历史、另一个来源进行中，再提交普通用户输入。
        write("conversation", "old-input", Input((ContentPart("text", "earlier task"),)))
        write("conversation", "old-final", Output((ContentPart("text", "x" * (33000 if long_history else 100)),), "complete"))
        write("job:background", "background-progress", Output((ContentPart("text", "still working"),), "continue"))
        if not other_open:
            write("job:background", "background-final", Output((ContentPart("text", "done"),), "complete"))
        before = tuple(log.reader("test:room").snapshot())
        result = outcome(await asyncio.wait_for(ask("current-input"), 20))
        assert tuple(log.reader("test:room").snapshot())[:len(before)] == before, "existing messages changed"
        if second_turn:
            # 2. 父摘要已存在、开放来源仍在：后续主对话也能摘要。
            write("conversation", "mid-input", Input((ContentPart("text", "another task"),)))
            write("conversation", "mid-final", Output((ContentPart("text", "y" * 33000),), "complete"))
            result = outcome(await asyncio.wait_for(ask("second-input"), 20))
    assert result.get("reason") != KEEP_INPUT_ERROR, result
    return {"other_open": other_open, "long_history": long_history, "second_turn": second_turn, **result}


async def recovery(directory: Path, *, pending_call: bool = False) -> dict[str, object]:
    """从真实频道触发硬容量恢复，再重启同一 SQLite，补学后来结束的后台来源。"""
    directory.mkdir()
    database = directory / "workspace/sessions.db"
    async with application(directory, replying=True, compaction=True, output_tokens=512,
                           extra_sources=configure) as (log, host):
        root = host.live_root
        assert root is not None
        tool = root.context.require(ALL_TOOLS)().refs[0]
        binding = await root.context.require(TOOLS).bind(tool, root.context.require(BINDINGS))

        def write(source, identity, body):
            writer = log.writer("test:room", author="user" if isinstance(body, Input) else "assistant",
                                source=source, body_types=(type(body),),
                                call_ref=body.call_ref if isinstance(body, ToolResult) else None,
                                content={"text": check_text}, check_call=lambda call: None)
            return writer.append(identity, body)

        def rows():
            with sqlite3.connect(database) as connection:
                assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
                return connection.execute("SELECT * FROM messages ORDER BY seq").fetchall()

        async def ask(identity: str) -> StoredSummary:
            original = rows()
            accepted = await root.context.require(CHANNEL_INPUT)("test:room", identity, ChannelInboundMessage(
                "test", "user", "room", QUESTION, datetime.now(UTC), {}))
            async def finish() -> Message:
                async for _ in log.catalog().follow():
                    for message in log.reader("test:room").snapshot(after_seq=accepted.seq):
                        if message.source == "conversation" and (
                            isinstance(message.body, Control) and message.body.action == "failure"
                            or isinstance(message.body, Output) and message.body.finish == "complete"
                        ):
                            return message
                raise AssertionError("message subscription closed")
            terminal = await asyncio.wait_for(finish(), 20)
            assert terminal is not None
            result = outcome(terminal)
            assert result["outcome"] == "complete" and "input=True" in str(result["answer"]), result
            assert rows()[:len(original)] == original
            summary = root.context.require(COMPACTION_SUMMARIES).head("test:room")
            assert summary is not None
            return summary

        def long_batch(identity):
            write("conversation", identity + "-input", Input((ContentPart("text", "next task"),)))
            write("conversation", identity + "-call", Output((ToolCall(binding, {}),), "continue"))
            write("conversation", identity + "-result", ToolResult(
                call_ref=CallRef(identity + "-call", 0), parts=(ContentPart("text", "x" * 46_000),), outcome="success"))
            write("conversation", identity + "-final", Output((ContentPart("text", "done"),), "complete"))

        # 1. 与事故相同：后台调用已有回执但没有来源终态，主对话仍不断追加长工具结果。
        long_batch("old")
        write("job:background", "background-input", Input((ContentPart("text", "background task"),)))
        write("job:background", "background-call", Output((ToolCall(binding, {}),), "continue"))
        if not pending_call:
            write("job:background", "background-result", ToolResult(
                call_ref=CallRef("background-call", 0), parts=(ContentPart("text", "still working"),), outcome="success"))
        first = await ask("first-input")
        assert first is not None
        for index in range(2):
            long_batch(f"later-{index}")
            head = await ask(f"later-input-{index}")
            assert head is not None and head.generation > first.generation
            assert f"later-{index}-result" in head.source_message_ids
            assert not any(identity.startswith("background-") for identity in head.source_message_ids)
        requests = root.context.require(ServiceKey("fixture.calls"))
        assert all("still working" not in str(request.messages) for request in requests)
        assert sum("[Source messages]" in str(request.messages) for request in requests) == 3

        store = MarkdownProfileStore(directory / "learning", directory / "learning/receipts.db")
        async def learning(record: StoredSummary) -> tuple[tuple[Message, ...], ...]:
            groups = await _unapplied_groups(
                record, root.context.require(COMPACTION_SUMMARIES), log.reader("test:room"), store,
                ("conversation", "job:background"), root.context.require(TURN_PROJECTION),
                compaction=root.context.require(COMPACTION_READER),
                post_commit_effect=root.context.require(CONTENT).legacy_post_commit_effect,
                summary_indexes=root.context.require(CONTEXT).summary_indexes,
            )
            assert groups is not None
            return groups
        learned = {message.message_id for group in await learning(head) for message in group}
        assert learned and not any(identity.startswith("background-") for identity in learned)
        store.write_draft(head.reference, {"version": 3, "scope": None, "ops": []},
                          session_key="test:room", generation=head.generation)
        store.apply(head.reference)

        # 2. 全部插件冷启动，稀疏摘要不依赖旧内存；后台仍未结束时继续隐藏。
        original = rows()
        await host.terminate_all()
        restarted = PluginManager([directory / "plugins"], workspace=directory / "workspace",
                                  installed_cache_root=directory / "home/cache")
        try:
            await restarted.load_all()
            root = restarted.live_root
            assert root is not None
            log = root.context.require(MESSAGE_CATALOG)._log
            reloaded = await ask("reloaded-input")
            assert reloaded is not None and reloaded.reference == head.reference
            requests = root.context.require(ServiceKey("fixture.calls"))
            assert "still working" not in str(requests[-1].messages)
            assert rows()[:len(original)] == original

            # 3. 后台结束后旧空洞恢复原文；下一代可补入旧身份，学习不漏掉后台也不重复旧主对话。
            if pending_call:
                write("job:background", "background-result", ToolResult(
                    call_ref=CallRef("background-call", 0), parts=(ContentPart("text", "still working"),), outcome="success"))
            write("job:background", "background-final", Output((ContentPart("text", "background done"),), "complete"))
            await ask("closed-input")
            assert "still working" in str(requests[-1].messages)
            long_batch("after-close")
            closed = await ask("compact-closed-input")
            assert closed is not None and "background-result" in closed.source_message_ids
            assert set(head.source_message_ids) <= set(closed.source_message_ids)
            new_learning = {message.message_id for group in await learning(closed) for message in group}
            assert "background-result" in new_learning and not learned.intersection(new_learning)
        finally:
            await restarted.terminate_all()
    return {"case": "sparse-recovery", "pending_background_call": pending_call, "hard_overflow_recovered": True, "restarted": True,
            "late_source_visible_and_learned": True, "original_rows_unchanged": True}


async def main(directory: Path) -> list[dict[str, object]]:
    results = [
        await run(directory / "open-long", other_open=True, long_history=True),
        await run(directory / "closed-long", other_open=False, long_history=True),
        await run(directory / "open-short", other_open=True, long_history=False),
        await run(directory / "open-second", other_open=True, long_history=True, second_turn=True),
    ]
    for result in results[:3]:
        answer = str(result.get("answer"))
        assert result["outcome"] == "complete" and "input=True" in answer, result
        assert ("summarized=True" in answer) == result["long_history"], result
        if result["other_open"]:
            assert "leaked=False" in answer, result
    assert results[3]["outcome"] == "complete" and "leaked=False" in str(results[3].get("answer")), results[3]
    results.append(await recovery(directory / "recovery"))
    results.append(await recovery(directory / "pending-recovery", pending_call=True))
    return results


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-compaction-open-") as directory:
        print(json.dumps(asyncio.run(main(Path(directory))), ensure_ascii=False))
