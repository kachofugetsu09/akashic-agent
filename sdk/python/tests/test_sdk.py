from __future__ import annotations

import asyncio
import json

import pytest

from akashic_sdk import AsyncAkashic, SlowConsumerError


@pytest.mark.asyncio
async def test_slow_subscription_does_not_close_other_sessions_or_requests():
    """可控服务端批次先灌满慢队列，再证明另一订阅和 request 仍工作。"""
    ready = asyncio.Event()
    sent = asyncio.Event()
    finished = asyncio.Event()
    subscriptions = {}
    writers = []

    async def accept(reader, writer):
        writers.append(writer)
        async def send(frame):
            writer.write((json.dumps({"jsonrpc": "2.0", **frame}) + "\n").encode())
            await writer.drain()
        try:
            while line := await reader.readline():
                frame = json.loads(line)
                if "id" not in frame:
                    continue
                params = frame["params"]
                if frame["method"] == "session/follow":
                    subscriptions[params["session_id"]] = params["subscription_id"]
                await send({"id": frame["id"], "result": {"ok": True}})
                if frame["method"] == "test/flood":
                    for index in range(3):
                        await send({"method": "session/event", "params": {
                            "subscription_id": subscriptions["slow"],
                            "event": {"session_id": "slow", "index": index},
                        }})
                    for index in range(600):
                        await send({"method": "session/event", "params": {
                            "subscription_id": subscriptions["fast"],
                            "event": {"session_id": "fast", "index": index},
                        }})
                    sent.set()
                if frame["method"] == "session/unfollow":
                    ready.set()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    address = f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        async with await AsyncAkashic.connect(address) as client:
            slow = await client.session_follow("slow", queue_size=1)
            fast = await client.session_follow("fast")
            async def consume():
                events = fast.events()
                try:
                    return [(await anext(events))["index"] for _ in range(600)]
                finally:
                    await events.aclose()
            consumer = asyncio.create_task(consume())
            await client.request("test/flood", {})
            async with asyncio.timeout(5):
                await sent.wait()
                await ready.wait()
                with pytest.raises(SlowConsumerError):
                    await anext(slow.events())
                assert await consumer == list(range(600))
                assert await client.request("server/status", {}) == {"ok": True}
                await fast.close()
        await asyncio.wait_for(finished.wait(), 5)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_malformed_response_fails_pending_request_instead_of_losing_it():
    from akashic_sdk import ProtocolError

    finished = asyncio.Event()
    async def accept(reader, writer):
        try:
            frame = json.loads(await reader.readline())
            # 初始化响应的 error 缺 code；解析失败仍须唤醒已登记的等待者。
            writer.write((json.dumps({"jsonrpc": "2.0", "id": frame["id"],
                                      "error": {"message": "invalid"}}) + "\n").encode())
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    endpoint = f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        async with asyncio.timeout(2):
            with pytest.raises(ProtocolError):
                await AsyncAkashic.connect(endpoint)
            await finished.wait()
    finally:
        server.close()
        await server.wait_closed()
