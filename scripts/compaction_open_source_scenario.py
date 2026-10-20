"""其他来源仍有未完成轮次时压缩长历史：当前输入保留，摘要不越过开放消息（#1544）。"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.content.plugin import check_text
from plugins.ledger.contract import ContentPart, Control, Input, Message, Output
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
            # 2. 父摘要已存在、开放来源仍在：其后的新长历史不能进摘要，只能如实报告容量或保留原文完成。
            write("conversation", "mid-input", Input((ContentPart("text", "another task"),)))
            write("conversation", "mid-final", Output((ContentPart("text", "y" * 33000),), "complete"))
            result = outcome(await asyncio.wait_for(ask("second-input"), 20))
    assert result.get("reason") != KEEP_INPUT_ERROR, result
    return {"other_open": other_open, "long_history": long_history, "second_turn": second_turn, **result}


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
    assert "leaked=True" not in str(results[3].get("answer")), results[3]
    return results


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-compaction-open-") as directory:
        print(json.dumps(asyncio.run(main(Path(directory))), ensure_ascii=False))
