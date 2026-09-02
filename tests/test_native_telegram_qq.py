from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import AsyncMock

import pytest

from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentReadLease,
    AttachmentRef,
    ChannelAttachmentImportPort,
    ChannelAttachmentReadPort,
    ChannelFactoryContext,
    ChannelIngressPort,
    ChannelRuntimePorts,
    ChannelReady,
    CredentialRef,
    DeliveryStatus,
    RawInbound,
    ProviderDeliveryRequest,
    ProviderClient,
)

from tests.test_channel_clients import (
    _Bus,
    _SessionManager,
    _import_qq_channel,
    _import_telegram_channel,
)


class _ProviderFactory:
    async def create(
        self,
        credentials: Mapping[str, CredentialRef],
    ) -> ProviderClient:
        raise AssertionError("native channel must reuse the existing provider owner")

    async def aclose(self) -> None:
        raise AssertionError(
            "native channel must not close the existing provider owner"
        )


class _Lease:
    def __init__(self, ref: AttachmentRef, data: bytes, events: list[str]) -> None:
        self.ref = ref
        self._data = data
        self._events = events

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        assert max_bytes >= len(self._data)
        self._events.append(f"read:{self.ref.artifact_id}")
        return self._data

    async def aclose(self) -> None:
        self._events.append(f"close:{self.ref.artifact_id}")


class _ReadPort:
    def __init__(self, blobs: dict[str, bytes], events: list[str]) -> None:
        self._blobs = blobs
        self._events = events

    async def acquire(self, ref: AttachmentRef) -> _Lease:
        self._events.append(f"acquire:{ref.artifact_id}")
        return _Lease(ref, self._blobs[ref.artifact_id], self._events)


def _context(
    binding_token: str,
    read_port: ChannelAttachmentReadPort,
    *,
    ingress: ChannelIngressPort | None = None,
    attachment_import: ChannelAttachmentImportPort | None = None,
) -> ChannelFactoryContext:
    return ChannelFactoryContext(
        snapshot_id="snapshot-1",
        generation_id="generation-1",
        binding_token=binding_token,
        config={},
        credentials={},
        provider_client_factory=_ProviderFactory(),
        ingress=ingress,
        identity=None,
        attachment_import=attachment_import,
        attachment_read=read_port,
    )


class _Ingress:
    def __init__(self) -> None:
        self.messages: list[RawInbound] = []

    async def admit(self, raw: RawInbound) -> bool:
        self.messages.append(raw)
        return True


class _ImportPort:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, AttachmentKind, str | None, str | None]] = []
        self._counter = 0

    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        self.calls.append((data, kind, filename, media_type))
        self._counter += 1
        return _ref(
            f"inbound-{self._counter}",
            kind,
            filename or f"inbound-{self._counter}",
            media_type or "application/octet-stream",
            data,
        )


class _FailingReadPort:
    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease:
        raise RuntimeError("read rejected")


def _runtime_context(
    binding_token: str,
    ingress: _Ingress,
    attachment_import: _ImportPort,
) -> ChannelFactoryContext:
    return _context(
        binding_token,
        _ReadPort({}, []),
        ingress=ingress,
        attachment_import=attachment_import,
    )


def _ref(
    artifact_id: str,
    kind: AttachmentKind,
    filename: str,
    media_type: str,
    data: bytes,
) -> AttachmentRef:
    return AttachmentRef(
        artifact_id=artifact_id,
        kind=kind,
        filename=filename,
        media_type=media_type,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


@pytest.mark.asyncio
async def test_telegram_v3_adapter_delivers_in_order_from_exact_leases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mod = _import_telegram_channel(monkeypatch)
    channel = mod.TelegramChannel("token", _Bus(), _SessionManager(tmp_path))
    channel._telegram_outbound_limiter = mod.TelegramOutboundLimiter(
        send_interval_s=0.0,
        edit_interval_s=0.0,
        typing_interval_s=0.0,
        global_interval_s=0.0,
        retry_padding_s=0.0,
    )
    channel._app.initialize = AsyncMock()
    channel._app.start = AsyncMock()
    events: list[str] = []
    photo = b"photo"
    document = b"document"
    image_ref = _ref("image-1", AttachmentKind.IMAGE, "image.png", "image/png", photo)
    file_ref = _ref(
        "file-1", AttachmentKind.FILE, "report.pdf", "application/pdf", document
    )
    read_port = _ReadPort(
        {image_ref.artifact_id: photo, file_ref.artifact_id: document},
        events,
    )
    context = _context("telegram-binding", read_port)

    async def send_message(**kwargs: Any) -> None:
        events.append(f"text:{kwargs['text']}")

    async def send_photo(**kwargs: Any) -> None:
        events.append(f"photo:{kwargs['photo'].getvalue().decode()}")

    async def send_document(**kwargs: Any) -> None:
        events.append(
            f"file:{kwargs['filename']}:{kwargs['document'].getvalue().decode()}"
        )

    channel._app.bot.send_message = send_message
    channel._app.bot.send_photo = send_photo
    channel._app.bot.send_document = send_document
    adapter = channel.build_v3_adapter(context)

    assert await adapter.start() == ChannelReady("telegram-binding")
    channel._app.initialize.assert_not_awaited()
    channel._app.start.assert_not_awaited()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="telegram-binding",
            delivery_id="delivery-1",
            recipient="123",
            body="hello",
            attachments=(image_ref, file_ref),
        )
    )

    assert receipt.status is DeliveryStatus.DELIVERED
    assert events == [
        "text:hello",
        "acquire:image-1",
        "read:image-1",
        "close:image-1",
        "photo:photo",
        "acquire:file-1",
        "read:file-1",
        "close:file-1",
        "file:report.pdf:document",
    ]
    assert (await adapter.stop()).resources_closed is True

    with pytest.raises(RuntimeError, match="binding token"):
        await adapter.deliver(
            ProviderDeliveryRequest(
                binding_token="wrong-binding",
                delivery_id="wrong-binding",
                recipient="123",
                body="must not send",
            )
        )


@pytest.mark.asyncio
async def test_telegram_v3_adapter_maps_pre_and_post_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mod = _import_telegram_channel(monkeypatch)
    channel = mod.TelegramChannel("token", _Bus(), _SessionManager(tmp_path))
    events: list[str] = []
    data = b"payload"
    ref = _ref("file-1", AttachmentKind.FILE, "a.txt", "text/plain", data)
    adapter = channel.build_v3_adapter(
        _context("telegram-binding", _ReadPort({ref.artifact_id: data}, events))
    )

    rejected = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="telegram-binding",
            delivery_id="invalid-recipient",
            recipient="not-a-chat",
            body="hello",
        )
    )
    assert rejected.status is DeliveryStatus.REJECTED

    async def fail_send_message(**_kwargs: Any) -> None:
        raise RuntimeError("provider failed")

    channel._app.bot.send_message = fail_send_message
    unknown = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="telegram-binding",
            delivery_id="provider-failure",
            recipient="123",
            body="hello",
        )
    )
    assert unknown.status is DeliveryStatus.UNKNOWN

    no_provider = channel.build_v3_adapter(
        _context("telegram-binding-2", _FailingReadPort())
    )
    rejected_attachment = await no_provider.deliver(
        ProviderDeliveryRequest(
            binding_token="telegram-binding-2",
            delivery_id="read-failure",
            recipient="123",
            body="",
            attachments=(ref,),
        )
    )
    assert rejected_attachment.status is DeliveryStatus.REJECTED


@pytest.mark.asyncio
async def test_qq_v3_adapter_delivers_group_text_and_binary_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mod = _import_qq_channel(monkeypatch)
    channel = mod.QQChannel(
        "42",
        _Bus(),
        _SessionManager(tmp_path),
        http_requester=SimpleNamespace(),
    )
    events: list[tuple[object, ...]] = []

    class _Api:
        async def send_group_text(self, group_id: int, body: str) -> None:
            events.append(("text", group_id, body))

        async def send_group_image(self, group_id: int, uri: str) -> None:
            events.append(
                ("image", group_id, base64.b64decode(uri.removeprefix("base64://")))
            )

        async def send_group_file(self, group_id: int, uri: str, filename: str) -> None:
            events.append(
                (
                    "file",
                    group_id,
                    filename,
                    base64.b64decode(uri.removeprefix("base64://")),
                )
            )

    channel._api = _Api()

    async def run(coro: Any) -> object:
        return await coro

    channel._run_on_bot_loop = AsyncMock(side_effect=run)
    image = b"qq-image"
    document = b"qq-document"
    image_ref = _ref("image-1", AttachmentKind.IMAGE, "image.jpg", "image/jpeg", image)
    file_ref = _ref("file-1", AttachmentKind.FILE, "report.txt", "text/plain", document)
    events_lease: list[str] = []
    adapter = channel.build_v3_adapter(
        _context(
            "qq-binding",
            _ReadPort(
                {image_ref.artifact_id: image, file_ref.artifact_id: document},
                events_lease,
            ),
        )
    )

    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="qq-binding",
            delivery_id="delivery-1",
            recipient="gqq:100",
            body="hello",
            attachments=(image_ref, file_ref),
        )
    )

    assert receipt.status is DeliveryStatus.DELIVERED
    assert events == [
        ("text", 100, "hello"),
        ("image", 100, image),
        ("file", 100, "report.txt", document),
    ]
    assert events_lease == [
        "acquire:image-1",
        "read:image-1",
        "close:image-1",
        "acquire:file-1",
        "read:file-1",
        "close:file-1",
    ]


@pytest.mark.asyncio
async def test_qq_v3_adapter_rejects_invalid_recipient_before_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mod = _import_qq_channel(monkeypatch)
    channel = mod.QQChannel(
        "42",
        _Bus(),
        _SessionManager(tmp_path),
        http_requester=SimpleNamespace(),
    )
    channel._api = SimpleNamespace()
    adapter = channel.build_v3_adapter(_context("qq-binding", _ReadPort({}, [])))

    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="qq-binding",
            delivery_id="invalid-recipient",
            recipient="gqq:not-a-number",
            body="hello",
        )
    )
    assert receipt.status is DeliveryStatus.REJECTED


@pytest.mark.asyncio
async def test_telegram_v3_inbound_waits_for_open_and_imports_reply_media(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mod = _import_telegram_channel(monkeypatch)
    channel = mod.TelegramChannel("token", _Bus(), _SessionManager(tmp_path))
    ingress = _Ingress()
    attachment_import = _ImportPort()
    context = _runtime_context("telegram-inbound", ingress, attachment_import)
    adapter = channel.build_v3_adapter(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )

    reply = SimpleNamespace(
        text="原消息",
        caption="",
        photo=[SimpleNamespace(file_id="reply-photo")],
        document=SimpleNamespace(
            file_id="reply-file", file_name="note.txt", mime_type="text/plain"
        ),
        from_user=SimpleNamespace(id=8, username="bob"),
        message_id=8,
    )
    message = SimpleNamespace(
        message_id=9,
        text="你好",
        caption="",
        photo=None,
        document=None,
        reply_to_message=reply,
        date=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    update = SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=7, username="alice"),
    )

    class _File:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        async def download_as_bytearray(self) -> bytearray:
            return bytearray(self.payload)

    channel._app.bot.get_file = AsyncMock(
        side_effect=[_File(b"reply-image"), _File(b"reply-file")]
    )
    handler = asyncio.create_task(
        channel._on_message(update, SimpleNamespace(bot=channel.bot))
    )
    await asyncio.sleep(0)
    assert ingress.messages == []

    adapter.open_admission()
    await handler

    assert len(ingress.messages) == 1
    raw = ingress.messages[0]
    assert raw.message_id == "9"
    assert raw.provider_identity == "7"
    assert raw.recipient == "123"
    assert "原消息" in raw.message.content
    assert "你好" in raw.message.content
    assert [ref.kind for ref in raw.message.attachments] == [
        AttachmentKind.IMAGE,
        AttachmentKind.FILE,
    ]
    assert [call[0] for call in attachment_import.calls] == [
        b"reply-image",
        b"reply-file",
    ]
    adapter.close_admission()
    assert (await adapter.stop()).resources_closed is True


@pytest.mark.asyncio
async def test_qq_v3_inbound_preserves_identity_and_imports_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mod = _import_qq_channel(monkeypatch)

    class _Response:
        status_code = 200
        headers = {"content-type": "image/png"}

        async def aiter_bytes(self, *, chunk_size: int):
            _ = chunk_size
            yield b"qq-image"

    from contextlib import asynccontextmanager

    class _Requester:
        @asynccontextmanager
        async def stream(self, *_args: Any, **_kwargs: Any):
            yield _Response()

    channel = mod.QQChannel(
        "42",
        _Bus(),
        _SessionManager(tmp_path),
        http_requester=_Requester(),
    )
    ingress = _Ingress()
    attachment_import = _ImportPort()
    context = _runtime_context("qq-inbound", ingress, attachment_import)
    adapter = channel.build_v3_adapter(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    event = SimpleNamespace(message_id="qq-77", time=1724140800)
    pending = asyncio.create_task(
        channel._handle_private(
            "10001",
            "hello",
            ["http://qq.invalid/a.png"],
            message_id="qq-77",
            event=event,
        )
    )
    await asyncio.sleep(0)
    assert ingress.messages == []
    adapter.open_admission()
    await pending

    assert len(ingress.messages) == 1
    raw = ingress.messages[0]
    assert raw.message_id == "qq-77"
    assert raw.provider_identity == "10001"
    assert raw.recipient == "10001"
    assert raw.message.content == "hello"
    assert raw.message.attachments[0].kind is AttachmentKind.IMAGE
    assert attachment_import.calls[0][0] == b"qq-image"
    adapter.close_admission()
    assert (await adapter.stop()).resources_closed is True


@pytest.mark.asyncio
async def test_telegram_v3_inflight_callback_cannot_cross_binding_after_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mod = _import_telegram_channel(monkeypatch)
    channel = mod.TelegramChannel("token", _Bus(), _SessionManager(tmp_path))
    ingress_formal = _Ingress()
    import_formal = _ImportPort()
    formal = _runtime_context("telegram-formal", ingress_formal, import_formal)
    adapter = channel.build_v3_adapter(formal)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=formal.snapshot_id,
            generation_id=formal.generation_id,
            binding_token=formal.binding_token,
            ingress=formal.ingress,
            identity=formal.identity,
            attachment_import=formal.attachment_import,
        )
    )
    adapter.open_admission()

    release_download = asyncio.Event()

    class _File:
        async def download_as_bytearray(self) -> bytearray:
            await release_download.wait()
            return bytearray(b"old-binding")

    channel._app.bot.get_file = AsyncMock(return_value=_File())
    message = SimpleNamespace(
        message_id=21,
        text="",
        caption="photo",
        photo=[SimpleNamespace(file_id="old-photo")],
        document=None,
        reply_to_message=None,
        date=datetime.now(timezone.utc),
    )
    update = SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=321),
        effective_user=SimpleNamespace(id=7, username="alice"),
    )
    old_callback = asyncio.create_task(
        channel._on_photo(update, SimpleNamespace(bot=channel.bot))
    )
    await asyncio.sleep(0)

    adapter.close_admission()
    ingress_candidate = _Ingress()
    import_candidate = _ImportPort()
    candidate = _runtime_context(
        "telegram-candidate", ingress_candidate, import_candidate
    )
    candidate_adapter = channel.build_v3_adapter(candidate)
    candidate_adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=candidate.snapshot_id,
            generation_id=candidate.generation_id,
            binding_token=candidate.binding_token,
            ingress=candidate.ingress,
            identity=candidate.identity,
            attachment_import=candidate.attachment_import,
        )
    )
    candidate_adapter.open_admission()
    release_download.set()
    with pytest.raises(RuntimeError, match="admission 已关闭"):
        await old_callback
    assert ingress_formal.messages == []
    assert ingress_candidate.messages == []
    assert import_formal.calls == []
    assert import_candidate.calls == []
    candidate_adapter.close_admission()
    assert (await candidate_adapter.stop()).resources_closed is True
