import asyncio
from contextlib import asynccontextmanager

import pytest

from akashic_sdk import Akashic, AsyncAkashic, RemoteError
from infra.control.socket import SocketAppServer
from tests.test_message_control import runtime


@asynccontextmanager
async def endpoint(tmp_path, monkeypatch):
    async with runtime(tmp_path, monkeypatch) as (core, service):
        server = SocketAppServer(tmp_path / "control.sock", service)
        await server.start()
        try:
            yield str(server.endpoint), core
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_async_sdk_real_socket_replay_and_subscription_replacement(tmp_path, monkeypatch):
    async with endpoint(tmp_path, monkeypatch) as (address, core):
        async with await AsyncAkashic.connect(address) as client:
            session = (await client.session_create())["session_id"]
            original = "original:" + "x" * (128 * 1024)
            old = await client.session_follow(session)
            new = await client.session_follow(session)
            await old.close()
            async with new:
                first = await client.message_send(session, original, message_id="input-one")
                assert first == await client.message_send(session, original, message_id="input-one")
                async with asyncio.timeout(5):
                    async for event in new.events():
                        if event["type"] == "messages.appended":
                            assert event["items"][0]["id"] == "input-one"
                            assert event["items"][0]["body"]["parts"][1]["value"] == original
                            break
            with pytest.raises(RemoteError):
                await client.message_send(session, "changed", message_id="input-one")
        async with await AsyncAkashic.connect(address) as resumed:
            assert [row["id"] for row in (await resumed.message_read(session))["items"]] == ["input-one"]
        assert len(core.message_log.reader(session).snapshot()) == 1


@pytest.mark.asyncio
async def test_sync_sdk_uses_same_message_contract(tmp_path, monkeypatch):
    async with endpoint(tmp_path, monkeypatch) as (address, _):
        def use_client():
            with Akashic.connect(address) as client:
                session = client.session_create()["session_id"]
                with client.session_follow(session) as subscription:
                    ack = client.message_send(session, "sync", message_id="sync-input")
                    for event in subscription.events():
                        if event["type"] == "messages.appended":
                            return ack, event
        ack, event = await asyncio.wait_for(asyncio.to_thread(use_client), 5)
        assert event["items"][0]["id"] == ack["message_id"]




@pytest.mark.asyncio
async def test_subscription_claims_one_consumer_before_first_iteration(tmp_path, monkeypatch):
    async with endpoint(tmp_path, monkeypatch) as (address, _):
        async with await AsyncAkashic.connect(address) as client:
            session = (await client.session_create())["session_id"]
            async with await client.session_follow(session) as subscription:
                first = subscription.events()
                with pytest.raises(RuntimeError, match="一个消费者"):
                    subscription.events()
                await client.message_send(session, "retained", message_id="retained")
                try:
                    async with asyncio.timeout(3):
                        async for event in first:
                            if event["type"] == "messages.appended":
                                assert event["items"][0]["id"] == "retained"
                                break
                finally:
                    await first.aclose()
                with pytest.raises(RuntimeError, match="一个消费者"):
                    subscription.events()


@pytest.mark.asyncio
async def test_sync_subscription_concurrent_consumers_cannot_split_one_queue(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    async with endpoint(tmp_path, monkeypatch) as (address, _):
        def use_client():
            with Akashic.connect(address) as client:
                session = client.session_create()["session_id"]
                with client.session_follow(session) as subscription:
                    barrier = Barrier(2)
                    def claim():
                        barrier.wait()
                        try:
                            return subscription.events()
                        except RuntimeError as error:
                            return error
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(claim) for _ in range(2)]
                        claimed = [future.result() for future in futures]
                    errors = [item for item in claimed if isinstance(item, RuntimeError)]
                    assert len(errors) == 1
                    iterator = next(item for item in claimed if not isinstance(item, RuntimeError))
                    client.message_send(session, "one queue", message_id="only")
                    try:
                        for event in iterator:
                            if event["type"] == "messages.appended":
                                assert event["items"][0]["id"] == "only"
                                break
                    finally:
                        iterator.close()
        await asyncio.wait_for(asyncio.to_thread(use_client), 5)
