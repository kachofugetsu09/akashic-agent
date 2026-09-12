"""每个 Telegram 请求只发送一次；不确定效果交还 Delivery。"""
import asyncio
import json
from dataclasses import dataclass
from typing import Literal, cast

import aiohttp
from telegramify_markdown.converter import convert_with_segments
from telegramify_markdown.entity import split_entities

from infra.channels.telegram_utils import strip_chunk
from agent.plugin_composition.artifacts import ArtifactRead
from session.artifacts import AttachmentKind, AttachmentRef
from session.log import MessageCatalog
from session.message import ContentPart, Control, Message


Status = Literal["delivered", "rejected", "failed"]


@dataclass(frozen=True, slots=True)
class SendResult:
    """Telegram provider 的本地结果；Delivery 在注册边界重新校验它。"""

    status: Status
    provider_ids: tuple[str, ...] = ()
    error: str | None = None


class AttachmentReadError(ValueError):
    """已验证引用的字节当前不可读取；修复后可重新准备发送。"""


@dataclass(frozen=True, slots=True)
class File:
    ref: AttachmentRef
    data: bytes


async def read_content(message: Message, catalog: MessageCatalog, artifacts: ArtifactRead) -> tuple[str | File, ...]:
    """先读完全部附件，避免发送正文后才发现文件损坏。"""
    if isinstance(message.body, Control):
        return ()
    refs = {ref.artifact_id: ref for ref in catalog.reader(message.session_id).attachments(message.message_id)}
    parts: list[str | File] = []
    for part in message.body.parts:
        if not isinstance(part, ContentPart):
            continue
        if part.kind == "text":
            text = cast(str, part.value)
            if text.strip():
                parts.append(text)
        elif part.kind == "artifact_ref":
            ref = refs[cast(str, part.value)]
            try:
                lease = await artifacts.acquire(ref)
                try:
                    data = await lease.read_bytes(max_bytes=ref.size_bytes)
                finally:
                    await lease.aclose()
            except (ValueError, OSError) as error:
                raise AttachmentReadError(f"{type(error).__name__}: {error}") from error
            parts.append(File(ref, data))
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    fields: dict[str, object]
    file: File | None = None


class TelegramSender:
    idempotent = False

    def __init__(self, client: aiohttp.ClientSession, api_base: str, token: str,
                 catalog: MessageCatalog, artifacts: ArtifactRead):
        self._client = client
        self._url = api_base.rstrip("/") + "/bot" + token
        self._catalog = catalog
        self._artifacts = artifacts

    async def query(self, key: str, address: str) -> SendResult | None:
        return None

    async def send(self, key: str, address: str, message: Message) -> SendResult:
        """先准备所有正文与附件，再逐项记录明确的 provider 回执。"""
        # 1. 地址和本地材料错误发生在第一个发送请求之前。
        try:
            chat_id = int(address)
        except ValueError:
            return SendResult(status="rejected", error="Telegram 地址必须是整数 chat ID")
        try:
            parts = await read_content(message, self._catalog, self._artifacts)
        except AttachmentReadError as error:
            return SendResult(status="rejected", error=f"Telegram 本地材料读取失败：{error}")
        requests: list[Request] = []
        for part in parts:
            if isinstance(part, str):
                try:
                    text, entities, _ = convert_with_segments(part)
                    chunks = split_entities(text, entities, 4090)
                except ValueError:
                    return SendResult(status="rejected", error="Telegram 正文格式无法转换")
                for text, entities in chunks:
                    text, entities = strip_chunk(text, entities)
                    if text:
                        requests.append(Request("sendMessage", {"chat_id": chat_id, "text": text,
                                                               "entities": [item.to_dict() for item in entities]}))
            else:
                method = "sendPhoto" if part.ref.kind == AttachmentKind.IMAGE else "sendDocument"
                requests.append(Request(method, {"chat_id": chat_id}, part))
        if not requests:
            return SendResult(status="rejected", error="消息没有可发送正文或附件")

        # 2. 不重试超时、断连或已成功前缀；错误文本不包含凭据 URL。
        provider_ids: list[str] = []
        for index, request in enumerate(requests):
            if index:
                # Telegram 建议同一 chat 每秒最多一条；这里只平滑本条消息的分片。
                await asyncio.sleep(1.05)
            try:
                status, body = await self._post(request)
            except (aiohttp.ClientError, TimeoutError):
                return SendResult(status="failed", provider_ids=tuple(provider_ids), error="Telegram 连接或回执未确认")
            try:
                result = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return SendResult(status="failed", provider_ids=tuple(provider_ids), error="Telegram 回执不是 JSON")
            if not isinstance(result, dict):
                return SendResult(status="failed", provider_ids=tuple(provider_ids), error="Telegram 回执结构无效")
            result = cast(dict[str, object], result)
            if result.get("ok") is False and 400 <= status < 500:
                return SendResult(status="failed" if provider_ids else "rejected", provider_ids=tuple(provider_ids),
                                  error=f"Telegram 拒绝请求（HTTP {status}）")
            raw_data = result.get("result")
            data = cast(dict[str, object], raw_data) if isinstance(raw_data, dict) else None
            if (status != 200 or result.get("ok") is not True or not isinstance(data, dict)
                    or type(data.get("message_id")) is not int):
                return SendResult(status="failed", provider_ids=tuple(provider_ids), error="Telegram 回执缺少已确认消息")
            provider_ids.append(str(data["message_id"]))
        return SendResult(status="delivered", provider_ids=tuple(provider_ids))


    async def _post(self, request: Request) -> tuple[int, bytes]:
        """只提交一次 HTTP 请求；不把含 token 的 URL 写入日志或错误回执。"""
        url = self._url + "/" + request.method
        if request.file is None:
            async with self._client.post(url, json=request.fields, allow_redirects=False) as response:
                return response.status, await response.read()
        file = request.file
        field = "photo" if request.method == "sendPhoto" else "document"
        form = aiohttp.FormData()
        for name, value in request.fields.items():
            form.add_field(name, str(value))
        form.add_field(field, file.data, filename=file.ref.filename or "attachment",
                       content_type=file.ref.media_type or "application/octet-stream")
        async with self._client.post(url, data=form, allow_redirects=False) as response:
            return response.status, await response.read()
