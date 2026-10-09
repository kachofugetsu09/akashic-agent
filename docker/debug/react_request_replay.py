"""真实 ReAct 执行工具后重开数据库，核对冻结请求的准确重放。"""

from __future__ import annotations
import asyncio, json, sqlite3, tempfile
from pathlib import Path
from agent.plugin_composition.models import LLMResponse, ToolCall
from plugins.react.plugin import _encode_request, _load_entry
from plugins.tools.execution import Result
from session.message import Input, ContentPart
from session.log import MessageLog
from tests.support.message_react import runtime


def encode(value):
    return json.dumps(
        value,
        default=dict,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


async def scenario(target, label):
    """执行跨越完整检查点的工具链，再核对保存、重放与正文保留。"""
    calls = {}
    effects = []
    count = 40

    async def complete(request):
        calls[request.request_key] = encode(_encode_request(request))
        if len(calls) == count:
            return LLMResponse("done")
        return LLMResponse(
            None,
            tool_calls=[
                ToolCall(
                    f"call-{len(calls)}",
                    "example",
                    {"step": len(calls), "value": [1, 1.0, True, -0.0][len(calls) % 4]},
                )
            ],
        )

    async def invoke(key, arguments):
        path = target / f"effect-{arguments['step']}.json"
        with path.open("x") as output:
            output.write(encode(arguments))
        effects.append(key)
        return Result("success", (ContentPart("text", path.read_text()),))

    def schemas():
        # 工具 schema 经过真实请求边界，区分 Python 相等的 JSON 类型和正负零。
        value = [1, 1.0, True, -0.0, 0.0][len(calls) % 5]
        return (
            {
                "type": "function",
                "function": {
                    "name": "example",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"default": value}},
                    },
                },
            },
        )

    async def materials(snapshot):
        # Prompt 中途改变；工具结果保留实际数值的 JSON 类型。
        return {
            "system_prompt": "\n "
            + label
            + " rules "
            + ("A" if len(calls) < 20 else "B")
            + " \n"
        }

    # 1. 本地 driver 返回受控响应，工具真实写入文件。
    target.mkdir()
    async with runtime(
        target,
        complete,
        invoke,
        max_steps=50,
        state_owner="react",
        material_source=materials,
        tool_schemas=schemas,
    ) as (source, log, models, run):
        await source.accept(
            "original",
            Input((ContentPart("text", label + " request " + "long context " * 800),)),
        )
        original = log.reader("s").snapshot()
        task = await source.start(run)
        await task.join()
        messages = log.reader("s").snapshot()
        assert messages[: len(original)] == original
        assert len(calls) == count and len(effects) == count - 1
    # 2. 重开独立存储，每条准备记录必须还原真实 driver 收到的请求。
    log = MessageLog(target / "sessions.db")
    try:
        assert log.reader("s").snapshot() == messages
        state = log.owner("react")
        restored = 0
        deltas = 0
        for key, record in state.list():
            if "attempts" not in record.value:
                continue
            for index, entry in enumerate(record.value["attempts"]):
                if entry is None:
                    continue
                request, _ = _load_entry(entry, state)
                assert (
                    encode(_encode_request(request))
                    == calls[record.value["request_keys"][index]]
                ), key
                deltas += entry.get("encoding") == "request-delta-v1"
                restored += 1
        assert restored == count and deltas == count - 3, (restored, deltas)
        with sqlite3.connect(target / "sessions.db") as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        return {
            "case": label,
            "calls": restored,
            "deltas": deltas,
            "effects": len(effects),
            "reopened_messages_equal": True,
        }
    finally:
        log.close()


async def main():
    with tempfile.TemporaryDirectory(prefix="react-request-replay-") as temporary:
        root = Path(temporary)
        # 同名 Session、Input 和 owner 的独立数据库不能共享请求基线。
        result = await asyncio.gather(
            scenario(root / "first", "first"), scenario(root / "second", "second")
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
