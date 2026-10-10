"""隔离安装真实回复插件，经本地 HTTP 模型检查长 Turn 压缩和消息保全。"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import shutil
import sys
import tempfile
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.plugin_composition.channels import CHANNEL_INPUT_V2, ChannelInboundMessage
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_contracts.ui import MESSAGE_DISPLAY
from agent.plugin_contracts import CallRef, ContentPart, Control, Input, Output, ToolCall, ToolResult
from plugins.compaction.message_summary import HEADINGS
from plugins.content.plugin import check_text
from plugins.context.api import check_notice
from plugins.compaction.contract import COMPACTION_READER, COMPACTION_SUMMARIES
from agent.plugin_contracts.content import CONTENT
from agent.plugin_contracts.context import CONTEXT
from agent.plugin_contracts.turns import TURN_PROJECTION
from plugins.markdown_memory.plugin import _unapplied_groups
from plugins.markdown_memory.store import MarkdownProfileStore
from agent.plugin_contracts.tools import ALL_TOOLS, TOOLS
from tests.test_default_reply import application


async def run(folder: Path, case: str) -> dict[str, object]:
    """只替换外部模型对端；插件安装、回复、投影和持久提交均走真实实现。"""
    # 1. 本地 HTTP 对端区分摘要与业务调用，不读取凭据或正式 workspace。
    requests: list[dict] = []
    handlers: set[asyncio.Task] = set()

    async def serve(reader, writer):
        task = asyncio.current_task()
        assert task is not None
        handlers.add(task)
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = int(next(row.split(":", 1)[1] for row in header.split("\r\n")
                              if row.lower().startswith("content-length:")))
            request = json.loads(await reader.readexactly(length))
            requests.append(request)
            summary = "[Source messages]" in str(request["messages"])
            text = ("invalid summary" if case.endswith("failure") else
                    "\n".join(heading + "\nRecorded progress." for heading in HEADINGS)) if summary else "work completed"
            body = json.dumps({"text": text}).encode()
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/complete"

    def configure(sources: Path) -> None:
        shutil.copytree(Path(__file__).resolve().parents[1] / "plugins/ui", sources / "ui",
                        ignore=shutil.ignore_patterns("__pycache__"))
        entry = sources / "test_provider/plugin.py"
        source = entry.read_text()
        start, end = source.index("    class Driver:"), source.index("    descriptor =")
        source = source[:start] + f'''    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return len(str(messages)) // 4
        async def complete(self, request):
            import httpx
            from agent.plugin_contracts import json_value
            async with httpx.AsyncClient(trust_env=False) as client:
                result = await client.post({url!r}, json={{"messages": json_value(request.messages)}})
                result.raise_for_status()
                return LLMResponse(result.json()["text"])
''' + source[end:]
        entry.write_text(source)

    try:
        async with application(folder, replying=True, start=False, compaction=True,
                               output_tokens=512, keep_recent_tokens=20_000 if case == "short_tail" else 2000,
                               extra_sources=configure) as (log, host):
            root = host.live_root
            assert root is not None
            # 2. 建立含工具配对的一个长 Turn，再从真实频道入口提交当前输入。
            tool = root.context.require(ALL_TOOLS)().refs[0]
            binding = await root.context.require(TOOLS).bind(tool, root.context.require(BINDINGS))
            def writer(body, ref=None):
                return log.writer("test:room", author="user" if body is Input else "assistant",
                                  source="conversation", body_types=(body,), call_ref=ref,
                                  content={"text": check_text, "context.notice": check_notice}, check_call=lambda call: None)
            if case == "split_turn":
                writer(Input).append("earlier-input", Input((ContentPart("text", "earlier task"),)))
                writer(Output).append("earlier-final", Output((ContentPart("text", "earlier finished"),), "complete"))
            writer(Input).append("old-input", Input((ContentPart("text", "original task"),)))
            padding = 6500 if case == "hard_failure" else 5500
            for index in range(6):
                identity = f"old-call-{index}"
                writer(Output).append(identity, Output((ToolCall(binding, {}),), "continue"))
                ref = CallRef(identity, 0)
                writer(ToolResult, ref).append(f"old-result-{index}", ToolResult(
                    call_ref=ref, parts=(ContentPart("text", f"evidence {index}:" + "x" * padding),), outcome="success"))
            if case != "open_turn":
                writer(Output).append("old-final", Output((ContentPart("text", "old task finished"),), "complete"))
            before = tuple(log.reader("test:room").snapshot())
            with sqlite3.connect(folder / "sessions.db") as connection:
                original_rows = connection.execute("SELECT * FROM messages ORDER BY seq").fetchall()
            await host.start_runtime()
            accept = root.context.require(CHANNEL_INPUT_V2)
            accepted = await accept("test:room", "current-input", ChannelInboundMessage(
                "test", "user", "room", "implement now", datetime.now(UTC), {}))
            async def finish():
                async for _ in log.catalog().follow():
                    rows = log.reader("test:room").snapshot()
                    if any(row.seq > accepted.seq and (
                        isinstance(row.body, Output) and row.body.finish == "complete"
                        or isinstance(row.body, Control) and row.body.action == "failure") for row in rows):
                        return rows
            rows = await asyncio.wait_for(finish(), 15)
            assert rows is not None
            with sqlite3.connect(folder / "sessions.db") as connection:
                after_rows = connection.execute("SELECT * FROM messages ORDER BY seq").fetchall()
                summaries = [json.loads(row[0]) for row in connection.execute(
                    "SELECT value FROM owner_records WHERE owner='plugin:compaction' AND key LIKE 'summary:%'")]
                assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            assert after_rows[:len(original_rows)] == original_rows
            assert rows[:len(before)] == before
            business = [request for request in requests if "[Source messages]" not in str(request["messages"])]
            if case == "hard_failure":
                assert not business and not summaries
                reason = next(row.body.reason for row in rows if isinstance(row.body, Control) and row.body.action == "failure")
                assert reason is not None
                assert "本次请求被阻断" in reason and "窗口 10,000" in reason
            else:
                assert len(business) == 1 and "implement now" in str(business[0])
                assert "context.notice" not in str(business[0]) and "上下文压缩失败" not in str(business[0])
                assert isinstance(rows[-1].body, Output) and rows[-1].body.finish == "complete"
                notices = [part.value for part in rows[-1].body.parts if isinstance(part, ContentPart) and part.kind == "context.notice"]
                assert all(isinstance(notice, str) for notice in notices)
                notices = cast(list[str], notices)
                if case == "soft_failure":
                    assert not summaries and any("仍满足硬容量" in notice for notice in notices)
                else:
                    assert len(summaries) == 1
                    record = summaries[0]
                    assert "current-input" not in record["source_message_ids"]
                    if case != "short_tail":
                        assert "old-final" not in record["source_message_ids"]
                    if case == "open_turn":
                        assert "original task" in str(business[0])
                    if case == "short_tail":
                        assert any("原文目标 20,000" in notice for notice in notices)
                    pending: set[str] = set()
                    for message in business[0]["messages"]:
                        if message["role"] == "assistant":
                            for call in message.get("tool_calls", ()):
                                assert call["id"] not in pending
                                pending.add(call["id"])
                        elif message["role"] == "tool":
                            pending.remove(message["tool_call_id"])
                    assert not pending
                page = log.reader("test:room").read_page(limit=200)
                display = cast(list[dict[str, Any]], await root.context.require(MESSAGE_DISPLAY)(page, display_only=True))
                shown = [part["value"]["text"] for row in display if row["body"]["kind"] == "output"
                         for part in row["body"]["parts"] if part["kind"] == "context.notice"]
                assert shown == notices
                if case == "split_turn":
                    # 3. 真正应用前一代完整 Turn，再由第二次压缩补学上一代延后的尾部。
                    store = MarkdownProfileStore(folder / "learning/memory.md", folder / "learning/self.md",
                                                 folder / "learning/receipts.db")
                    lookup = root.context.require(COMPACTION_SUMMARIES)
                    async def learning():
                        record = lookup.head("test:room")
                        assert record is not None
                        groups = await _unapplied_groups(
                            record, lookup, log.reader("test:room"), store, ("conversation",),
                            root.context.require(TURN_PROJECTION), compaction=root.context.require(COMPACTION_READER),
                            post_commit_effect=root.context.require(CONTENT).legacy_post_commit_effect,
                            summary_range=root.context.require(CONTEXT).summary_range,
                        )
                        return record, {message.message_id for group in groups or () for message in group}
                    record, learned = await learning()
                    assert learned == {"earlier-input", "earlier-final"}
                    draft = {"version": 1, "memory_before": store.read_memory(), "memory": store.read_memory(),
                             "self_before": store.read_self(), "self": store.read_self()}
                    store.write_draft(record.reference, draft, session_key="test:room", generation=record.generation)
                    store.apply_draft(record.reference, draft)
                    writer(Input).append("next-task", Input((ContentPart("text", "next task"),)))
                    for index in range(6):
                        identity = f"next-call-{index}"
                        writer(Output).append(identity, Output((ToolCall(binding, {}),), "continue"))
                        ref = CallRef(identity, 0)
                        writer(ToolResult, ref).append(f"next-result-{index}", ToolResult(
                            call_ref=ref, parts=(ContentPart("text", "y" * padding),), outcome="success"))
                    accepted = await accept("test:room", "next-input", ChannelInboundMessage(
                        "test", "user", "room", "continue now", datetime.now(UTC), {}))
                    await asyncio.wait_for(finish(), 15)
                    next_record, learned = await learning()
                    assert next_record.generation == record.generation + 1
                    assert "old-input" in learned and "old-final" in learned
                    assert not {"earlier-input", "earlier-final"}.intersection(learned)
            with sqlite3.connect(folder / "sessions.db") as connection:
                assert connection.execute("SELECT * FROM messages ORDER BY seq").fetchall()[:len(original_rows)] == original_rows
                generations = connection.execute("SELECT count(*) FROM owner_records WHERE owner='plugin:compaction' AND key LIKE 'summary:%'").fetchone()[0]
            return {"case": case, "http_requests": len(requests), "summaries": generations,
                    "original_rows_preserved": len(original_rows), "result": "passed"}
    finally:
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="compaction-workflow-") as path:
        for case in ("split_turn", "open_turn", "short_tail", "soft_failure", "hard_failure"):
            folder = Path(path) / case
            folder.mkdir()
            print(json.dumps(await run(folder, case), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
