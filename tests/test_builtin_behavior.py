"""强场景：交错的两个会话执行真实工具，再跨进程核对正文、重放和后续工作。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import socket

from aiohttp import web
from akashic_sdk import AsyncAkashic
import pytest

from tests.fixtures.builtin_runtime import runtime



class WireModel:
    """固定外部模型响应；每一步按实际工具结果推进，不共用调用次数。"""

    def __init__(self, root: Path):
        self.root = root
        self.requests: list[dict] = []
        self.entered = {label: asyncio.Event() for label in "ABCD"}
        self.responded = {label: asyncio.Event() for label in "ABCD"}
        self.release = {label: asyncio.Event() for label in "ABCD"}
        for event in self.release.values():
            event.set()

    @staticmethod
    def contents(label: str) -> str:
        return f"ONLY_{label}\n" + (f"{label}：汉字、🧪、引号\"与换行\n" * 80) + f"END_{label}"

    async def chat(self, request: web.Request) -> web.Response:
        """发送真实 SSE 工具调用，并保存收到的协议请求供测试独立核对。"""
        body = await request.json()
        self.requests.append(body)
        with (self.root / "model-requests.jsonl").open("a", encoding="utf-8") as evidence:
            evidence.write(json.dumps(body, ensure_ascii=False) + "\n")
        users = [item for item in body["messages"] if item["role"] == "user"]
        candidates = [re.findall(r"CASE_([ABCD])", json.dumps(item, ensure_ascii=False)) for item in users]
        matches = next((items for items in reversed(candidates) if items), [])
        if len(matches) != 1:
            return web.json_response({"error": {"message": "fixture requires one case"}}, status=400)
        label = matches[0]
        self.entered[label].set()
        await self.release[label].wait()
        results = [item for item in body["messages"] if item["role"] == "tool"]
        path = str(self.root / f"effect-{label}.txt")
        steps = (
            ("write_file", {"path": path, "content": self.contents(label)}),
            ("read_file", {"path": path}),
            ("message_push", {"target_channel": "akashic", "target_chat_id": f"sink-{label}",
                              "message": f"DELIVERY_{label}"}),
        )
        if len(results) < len(steps):
            name, arguments = steps[len(results)]
            encoded = json.dumps(arguments, ensure_ascii=False)
            middle = len(encoded) // 2
            deltas = [
                {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{label}_{len(results)}",
                    "type": "function", "function": {"name": name, "arguments": encoded[:middle]}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": encoded[middle:]}}]},
            ]
            finish = "tool_calls"
        else:
            deltas = [{"role": "assistant", "content": f"FINAL_{label}"}]
            finish = "stop"
        chunks: list[dict[str, object]] = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})
        payload = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
        self.responded[label].set()
        return web.Response(text=payload + "data: [DONE]\n\n", content_type="text/event-stream")

    @asynccontextmanager
    async def serve(self) -> AsyncIterator[str]:
        """模型服务只监听一次性 loopback 端口。"""
        app = web.Application()

        async def models(_request: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": "fixture"}]})

        app.router.add_get("/v1/models", models)
        app.router.add_post("/v1/chat/completions", self.chat)
        runner = web.AppRunner(app)
        await runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        try:
            await web.SockSite(runner, sock).start()
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            for event in self.release.values():
                event.set()
            await runner.cleanup()




async def finish(client: AsyncAkashic, session: str, label: str) -> list[dict]:
    """通过真实订阅观察终稿，再从公开读取接口核对持久消息。"""
    async with await client.session_follow(session) as subscription:
        await client.message_send(session, f"CASE_{label}", message_id=f"input-{label}")
        async with asyncio.timeout(25):
            async for event in subscription.events():
                if event["type"] == "messages.appended" and any(
                    item["body"].get("finish") == "complete" for item in event["items"]
                ):
                    break
    return (await client.message_read(session))["items"]


def check_turn(rows: list[dict], label: str) -> None:
    """每个结果必须引用本会话的实际调用，终稿不能代替缺失的工具轨迹。"""
    bodies = [row["body"] for row in rows]
    assert [body["kind"] for body in bodies] == [
        "input", "output", "tool_result", "output", "tool_result", "output", "tool_result", "output",
    ]
    assert [row["seq"] for row in rows] == list(range(8))
    assert len({row["id"] for row in rows}) == 8
    assert rows[0]["id"] == f"input-{label}"
    for call, result in zip(rows[1::2][:-1], rows[2::2], strict=True):
        part_index = next(i for i, part in enumerate(call["body"]["parts"]) if part["kind"] == "tool_call")
        assert result["body"]["call_ref"] == {"message_id": call["id"], "part_index": part_index}
        assert result["body"]["outcome"] == "success"
    assert any(part.get("value") == f"FINAL_{label}" for part in bodies[-1]["parts"])


@pytest.mark.asyncio
async def test_two_sessions_keep_tools_and_delivery_after_process_crash(tmp_path: Path) -> None:
    """A 阻塞时 B 完成；强杀后重放不重复执行，C 仍能完成同一条真实链路。"""
    model = WireModel(tmp_path)
    model.release["A"].clear()
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint) as (first, address):
            async with await AsyncAkashic.connect(address) as client:
                a = (await client.session_create())["session_id"]
                b = (await client.session_create())["session_id"]
                pending = asyncio.create_task(finish(client, a, "A"))
                try:
                    await asyncio.wait_for(model.entered["A"].wait(), 10)
                    rows_b = await finish(client, b, "B")
                    assert not pending.done(), "B 应独立完成，A 仍在外部模型边界等待"
                    assert (tmp_path / "effect-B.txt").read_text() == model.contents("B")
                    assert not (tmp_path / "effect-A.txt").exists()
                    model.release["A"].set()
                    rows_a = await pending
                finally:
                    model.release["A"].set()
                    if not pending.done():
                        pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                check_turn(rows_a, "A")
                check_turn(rows_b, "B")
                assert (tmp_path / "effect-A.txt").read_text() == model.contents("A")
                for label in "AB":
                    delivered = (await client.message_read(f"akashic:sink-{label}"))["items"]
                    assert len(delivered) == 1
                    assert delivered[0]["body"]["parts"][0]["value"] == f"DELIVERY_{label}"
                saved = {a: rows_a, b: rows_b}
            first.kill()
            await first.wait()

        # 新 OS 进程不能借用旧 Task、Model、连接或缓存证明恢复。
        async with runtime(tmp_path, endpoint) as (second, address):
            assert second.pid != first.pid
            async with await AsyncAkashic.connect(address) as client:
                for session, rows in saved.items():
                    assert (await client.message_read(session))["items"] == rows
                await client.message_send(a, "CASE_A", message_id="input-A")
                c = (await client.session_create())["session_id"]
                check_turn(await finish(client, c, "C"), "C")
                assert (await client.message_read(a))["items"] == rows_a
                assert len((await client.message_read("akashic:sink-A"))["items"]) == 1
        assert not (tmp_path / "workspace/.runtime-ready.json").exists()
        for label in "ABC":
            requests = [body for body in model.requests if f"CASE_{label}" in json.dumps(body)]
            assert len(requests) == 4, "一次输入应只执行写入、读取、推送和终稿四次模型请求"
            read_result = [row for row in requests[-1]["messages"] if row["role"] == "tool"][1]
            assert f"ONLY_{label}" in str(read_result) and f"END_{label}" in str(read_result)
            for other in set("ABC") - {label}:
                assert all(f"CASE_{other}" not in json.dumps(body) for body in requests)


@pytest.mark.asyncio
async def test_stop_rejects_late_model_tools_and_next_input_recovers(tmp_path: Path) -> None:
    """停止 A 后才返回它的工具调用；B 正常执行，A 的下一条输入也能完成。"""
    model = WireModel(tmp_path)
    model.release["A"].clear()
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint) as (_process, address):
            async with await AsyncAkashic.connect(address) as client:
                a = (await client.session_create())["session_id"]
                b = (await client.session_create())["session_id"]
                await client.message_send(a, "CASE_A", message_id="input-A")
                await asyncio.wait_for(model.entered["A"].wait(), 10)
                await asyncio.wait_for(client.message_send(a, "/stop", message_id="stop-A"), 10)
                stopped = (await client.message_read(a))["items"]
                assert [item["body"]["kind"] for item in stopped] == ["input", "control"]
                assert stopped[-1]["body"]["action"] == "pause"
                model.release["A"].set()
                await asyncio.wait_for(model.responded["A"].wait(), 10)
                check_turn(await finish(client, b, "B"), "B")
                assert (await client.message_read(a))["items"] == stopped
                assert not (tmp_path / "effect-A.txt").exists(), "被停止请求的迟到工具调用产生了效果"
                continued = await finish(client, a, "C")
                assert continued[:len(stopped)] == stopped
                assert continued[-1]["body"]["finish"] == "complete"
                assert any(part.get("value") == "FINAL_C" for part in continued[-1]["body"]["parts"])
                assert (tmp_path / "effect-C.txt").read_text() == model.contents("C")
                assert (await client.message_read("akashic:sink-C"))["items"][0]["body"]["parts"][0]["value"] == "DELIVERY_C"
