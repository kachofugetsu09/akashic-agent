"""实际插件绑定、回执恢复与 loopback provider wire，不连接正式渠道。"""
import asyncio
import base64
from contextlib import asynccontextmanager
from functools import partial
import json
import logging
from pathlib import Path
import shutil

from aiohttp import web
import pytest
from websockets.asyncio.server import serve

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.tasks import Tasks
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.delivery.api import Sink
from plugins.delivery.execution import Deliveries
from plugins.delivery.records import DeliveryRecords
from plugins.delivery.senders import DELIVERY_SENDERS, open_sender
from session.artifact_store import ArtifactStore
from session.artifacts import AttachmentKind
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Output


@asynccontextmanager
async def telegram_server(respond):
    calls = []

    async def handle(request):
        if request.content_type == "application/json":
            payload = await request.json()
        else:
            fields = await request.post()
            payload = {key: value.file.read() if isinstance(value, web.FileField) else value
                       for key, value in fields.items()}
        calls.append((request.path, payload))
        return await respond(request, len(calls), payload)

    app = web.Application()
    app.router.add_post('/{tail:.*}', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", calls
    finally:
        await runner.cleanup()


@asynccontextmanager
async def qq_server(respond):
    calls = []
    headers = []

    async def handle(connection):
        headers.append(connection.request.headers.get("Authorization"))
        async for wire in connection:
            request = json.loads(wire)
            calls.append(request)
            response = await respond(connection, len(calls), request)
            if response is not None:
                await connection.send(json.dumps(response))

    async with serve(handle, "127.0.0.1", 0) as server:
        yield f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/api", calls, headers


@asynccontextmanager
async def application(tmp_path, channel, endpoint):
    source = tmp_path / "plugins"
    for name in ("delivery", channel + "_sender"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    workspace = tmp_path / "workspace"
    config = workspace / f"plugin-data/{channel}_sender-builtin/config.local.toml"
    config.parent.mkdir(parents=True)
    field = "api_base" if channel == "telegram" else "endpoint"
    config.write_text(f'enabled=true\ntoken="wire-fixture-secret"\n{field}="{endpoint}"\ntimeout_seconds=2\n')
    log = MessageLog(workspace / "sessions.db")
    artifacts = ArtifactStore(workspace / "sessions.db")
    physical = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=artifacts)
    host = PluginManager([source], event_bus=EventBus(), workspace=workspace, message_log=log,
                         installed_cache_root=tmp_path / "cache", channel_attachment_store=physical)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            bindings = Bindings(log, host._archive, host.open_binding)
            binding = snapshot.composition_root.context.require(DELIVERY_SENDERS).bind(channel, bindings)
        yield log, host, bindings, binding, physical, config
    finally:
        await host.terminate_all()
        artifacts.close()
        log.close()


def message(log, parts, identity="answer"):
    return log.writer("chat", author="reply", source="conversation", body_types=(Output,), content={
        "text": lambda part: ContentReferences(),
        "artifact_ref": lambda part: ContentReferences(artifact_ids=(part.value,)),
        "model.facts": lambda part: ContentReferences(),
    }).append(identity, Output(tuple(parts), "complete"))


async def file_part(physical, kind=AttachmentKind.FILE):
    payload = b"file payload" if kind == AttachmentKind.FILE else base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=")
    ref = await physical.import_bytes(payload, kind=kind, filename="sample.txt" if kind == AttachmentKind.FILE else "sample.png",
                                      media_type="text/plain" if kind == AttachmentKind.FILE else "image/png")
    return ContentPart("artifact_ref", ref.artifact_id), payload, ref


async def tg_success(request, index, payload):
    return web.json_response({"ok": True, "result": {"message_id": index}})


@pytest.mark.asyncio
async def test_telegram_archived_sender_preserves_rich_text_and_artifact_order(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    async with telegram_server(tg_success) as (endpoint, calls):
        async with application(tmp_path, "telegram", endpoint) as (log, host, bindings, binding, physical, _):
            attachment, payload, _ = await file_part(physical)
            photo, photo_bytes, _ = await file_part(physical, AttachmentKind.IMAGE)
            msg = message(log, [ContentPart("text", "**Hello 🌙**\n" + "长" * 4300), attachment, photo,
                                ContentPart("model.facts", {"private": "never render"})])
            before = tuple(log._connection.iterdump())
            await host.terminate_all()
            shutil.rmtree(tmp_path / "plugins")
            # 绑定只用归档代码和原凭据，不需要当前插件源码或收件实例。
            async with open_sender(bindings, binding) as sender:
                assert not sender.idempotent
                assert calls == []
                receipt = await sender.send("once", "-100123", msg)
                assert await sender.query("once", "-100123") is None
            assert receipt.status == "delivered" and len(receipt.provider_ids) == 5
            assert [path.rsplit("/", 1)[-1] for path, _ in calls] == ["sendMessage", "sendMessage", "sendMessage", "sendDocument", "sendPhoto"]
            assert calls[0][1]["entities"], [(path, str(body)[:240]) for path, body in calls]
            assert calls[0][1]["entities"][0]["type"] == "bold"
            assert all(len(body["text"].encode("utf-16-le")) // 2 <= 4090 for _, body in calls[:3])
            assert "never render" not in json.dumps(calls[:3])
            assert calls[3][1]["document"] == payload and calls[4][1]["photo"] == photo_bytes
            assert tuple(log._connection.iterdump()) == before
            assert all("wire-fixture-secret" not in item.getMessage() for item in caplog.records
                       if item.name != "aiohttp.access")
            for path in host._archive.path.rglob("*.json"):
                assert b"wire-fixture-secret" not in path.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rejected", "wrong_body", "partial", "disconnect"])
async def test_telegram_unknown_never_replays_a_successful_prefix(tmp_path, failure):
    async def respond(request, index, payload):
        if failure == "disconnect":
            request.transport.close()
            return web.Response()
        if failure == "partial" and index == 1:
            return await tg_success(request, index, payload)
        if failure == "wrong_body":
            return web.json_response({"ok": True, "result": {}})
        return web.json_response({"ok": False, "error_code": 429}, status=429)

    async with telegram_server(respond) as (endpoint, calls):
        async with application(tmp_path, "telegram", endpoint) as (log, host, bindings, binding, _, _):
            msg = message(log, [ContentPart("text", "first"), ContentPart("text", "second")])
            tasks = Tasks()
            records = DeliveryRecords(log.owner("plugin:delivery"), "test")
            execution = Deliveries(records, log.catalog(), tasks, partial(open_sender, bindings), task_key="delivery")
            execution.prepare(log.reader("chat"), msg, (Sink(name="telegram", binding_id=binding, address="123"),))
            try:
                receipt = await execution.send(msg.message_id, "telegram")
                assert receipt.status == ("rejected" if failure == "rejected" else "unknown")
                assert len(calls) == (2 if failure == "partial" else 1)
                count = len(calls)
                await host.terminate_all()
                shutil.rmtree(tmp_path / "plugins")
                recovered = await execution.send(msg.message_id, "telegram")
                assert recovered == receipt
                assert recovered.provider_ids == (("1",) if failure == "partial" else ())
                assert len(calls) == count
            finally:
                await tasks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["gqq:42", "17"])
async def test_qq_uses_echo_and_sends_files_without_inventing_message_ids(tmp_path, address, caplog):
    caplog.set_level(logging.DEBUG)
    async def respond(connection, index, request):
        await connection.send(json.dumps({"post_type": "meta_event", "meta_event_type": "heartbeat"}))
        data = None if request["action"].startswith("upload_") else {"message_id": index}
        return {"status": "ok", "retcode": 0, "data": data, "echo": request["echo"]}

    async with qq_server(respond) as (endpoint, calls, headers):
        async with application(tmp_path, "qq", endpoint) as (log, host, bindings, binding, physical, _):
            file, payload, _ = await file_part(physical)
            photo, photo_payload, _ = await file_part(physical, AttachmentKind.IMAGE)
            msg = message(log, [ContentPart("text", "literal [CQ:at,qq=1]"), file, photo])
            await host.terminate_all()
            shutil.rmtree(tmp_path / "plugins")
            async with open_sender(bindings, binding) as sender:
                assert calls == []
                receipt = await sender.send("once", address, msg)
            assert receipt.status == "delivered" and receipt.provider_ids == ("1", "3")
            assert headers == ["Bearer wire-fixture-secret"]
            assert all("wire-fixture-secret" not in item.getMessage() for item in caplog.records
                       if item.name != "websockets.server")
            kind = "group" if address.startswith("gqq:") else "private"
            assert [item["action"] for item in calls] == [f"send_{kind}_msg", f"upload_{kind}_file", f"send_{kind}_msg"]
            assert calls[0]["params"]["message"][0]["data"]["text"] == "literal [CQ:at,qq=1]"
            assert base64.b64decode(calls[1]["params"]["file"][9:]) == payload
            assert base64.b64decode(calls[2]["params"]["message"][0]["data"]["file"][9:]) == photo_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rejected", "wrong_echo", "partial", "disconnect"])
async def test_qq_uncertain_receipt_does_not_resend(tmp_path, failure):
    async def respond(connection, index, request):
        if failure == "disconnect":
            await connection.close()
            return None
        if failure == "partial" and index == 1:
            return {"status": "ok", "retcode": 0, "data": None, "echo": request["echo"]}
        return {"status": "failed", "retcode": 100, "echo": "wrong" if failure == "wrong_echo" else request["echo"]}

    async with qq_server(respond) as (endpoint, calls, _):
        async with application(tmp_path, "qq", endpoint) as (log, _, bindings, binding, physical, _):
            file, _, _ = await file_part(physical)
            msg = message(log, [file, ContentPart("text", "after upload")])
            tasks = Tasks()
            records = DeliveryRecords(log.owner("plugin:delivery"), "test")
            execution = Deliveries(records, log.catalog(), tasks, partial(open_sender, bindings), task_key="delivery")
            execution.prepare(log.reader("chat"), msg, (Sink(name="qq", binding_id=binding, address="gqq:42"),))
            try:
                receipt = await execution.send(msg.message_id, "qq")
                assert receipt.status == ("rejected" if failure == "rejected" else "unknown")
                assert receipt.provider_ids == ()
                assert len(calls) == (2 if failure == "partial" else 1)
                count = len(calls)
                assert (await execution.send(msg.message_id, "qq")).status == receipt.status
                assert len(calls) == count
            finally:
                await tasks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["telegram", "qq"])
async def test_native_sender_reads_all_artifacts_before_any_provider_effect(tmp_path, channel):
    async def respond(connection, index, request):
        return {"status": "ok", "retcode": 0, "data": {"message_id": index}, "echo": request["echo"]}

    server = telegram_server(tg_success) if channel == "telegram" else qq_server(respond)
    async with server as wire:
        endpoint, calls = wire[:2]
        async with application(tmp_path, channel, endpoint) as (log, _, bindings, binding, physical, _):
            file, _, ref = await file_part(physical)
            msg = message(log, [ContentPart("text", "must not send"), file])
            path = tmp_path / "workspace/uploads/artifacts" / (ref.artifact_id + ".bin")
            path.write_bytes(b"corrupt")
            tasks = Tasks()
            records = DeliveryRecords(log.owner("plugin:delivery"), "test")
            execution = Deliveries(records, log.catalog(), tasks, partial(open_sender, bindings), task_key="delivery")
            execution.prepare(log.reader("chat"), msg, (Sink(name=channel, binding_id=binding, address="123"),))
            try:
                receipt = await execution.send(msg.message_id, channel)
                assert receipt.status == "rejected" and "本地材料" in receipt.error
                assert calls == []
                path.write_bytes(b"file payload")
                assert (await execution.retry(msg.message_id, channel)).status == "delivered"
                assert len(calls) == 2
            finally:
                await tasks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["telegram", "qq"])
async def test_native_address_rejection_and_credential_revocation_have_no_effect(tmp_path, channel):
    async def respond(connection, index, request):
        return {"status": "ok", "retcode": 0, "data": {"message_id": index}, "echo": request["echo"]}

    server = telegram_server(tg_success) if channel == "telegram" else qq_server(respond)
    async with server as wire:
        endpoint, calls = wire[:2]
        async with application(tmp_path, channel, endpoint) as (log, _, bindings, binding, _, config):
            msg = message(log, [ContentPart("text", "must not send")])
            tasks = Tasks()
            records = DeliveryRecords(log.owner("plugin:delivery"), "test")
            execution = Deliveries(records, log.catalog(), tasks, partial(open_sender, bindings), task_key="delivery")
            execution.prepare(log.reader("chat"), msg, (Sink(name=channel, binding_id=binding, address="not-a-chat"),))
            try:
                receipt = await execution.send(msg.message_id, channel)
                assert receipt.status == "rejected" and "地址" in receipt.error
                assert records.read(msg.message_id, channel)[1].phase == "rejected"
                config.write_text(config.read_text().replace("wire-fixture-secret", "revoked-replacement"))
                # 显式 retry 仍使用原 binding；不能悄悄切到新 token。
                with pytest.raises(RuntimeError, match="revision 已漂移"):
                    await execution.retry(msg.message_id, channel)
                assert records.read(msg.message_id, channel)[1].phase == "prepared"
                assert calls == []
            finally:
                await tasks.close()
