"""每个 Telegram 请求只发送一次；不确定效果交还 Delivery。"""
import asyncio
import json
from dataclasses import dataclass
from typing import cast

import aiohttp
from telegramify_markdown.converter import convert_with_segments
from telegramify_markdown.entity import split_entities

from agent.plugin_composition.artifacts import ArtifactRead
from infra.channels.telegram_utils import strip_chunk
from plugins.delivery.api import Receipt
from plugins.delivery.content import AttachmentReadError, File, read_content
from session.artifacts import AttachmentKind
from session.log import MessageCatalog
from session.message import Message


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

    async def query(self, key: str, address: str) -> Receipt | None:
        return None

    async def send(self, key: str, address: str, message: Message) -> Receipt:
        """先准备所有正文与附件，再逐项记录明确的 provider 回执。"""
        # 1. 地址和本地材料错误发生在第一个发送请求之前。
        try:
            chat_id = int(address)
        except ValueError:
            return Receipt(status="rejected", error="Telegram 地址必须是整数 chat ID")
        try:
            parts = await read_content(message, self._catalog, self._artifacts)
        except AttachmentReadError as error:
            return Receipt(status="rejected", error=f"Telegram 本地材料读取失败：{error}")
        requests: list[Request] = []
        for part in parts:
            if isinstance(part, str):
                try:
                    text, entities, _ = convert_with_segments(part)
                    chunks = split_entities(text, entities, 4090)
                except ValueError:
                    return Receipt(status="rejected", error="Telegram 正文格式无法转换")
                for text, entities in chunks:
                    text, entities = strip_chunk(text, entities)
                    if text:
                        requests.append(Request("sendMessage", {"chat_id": chat_id, "text": text,
                                                               "entities": [item.to_dict() for item in entities]}))
            else:
                method = "sendPhoto" if part.ref.kind == AttachmentKind.IMAGE else "sendDocument"
                requests.append(Request(method, {"chat_id": chat_id}, part))
        if not requests:
            return Receipt(status="rejected", error="消息没有可发送正文或附件")

        # 2. 不重试超时、断连或已成功前缀；错误文本不包含凭据 URL。
        provider_ids: list[str] = []
        for index, request in enumerate(requests):
            if index:
                # Telegram 建议同一 chat 每秒最多一条；这里只平滑本条消息的分片。
                await asyncio.sleep(1.05)
            try:
                status, body = await self._post(request)
            except (aiohttp.ClientError, TimeoutError):
                return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="Telegram 连接或回执未确认")
            try:
                result = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="Telegram 回执不是 JSON")
            if not isinstance(result, dict):
                return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="Telegram 回执结构无效")
            result = cast(dict[str, object], result)
            if result.get("ok") is False and 400 <= status < 500:
                return Receipt(status="unknown" if provider_ids else "rejected", provider_ids=tuple(provider_ids),
                               error=f"Telegram 拒绝请求（HTTP {status}）")
            raw_data = result.get("result")
            data = cast(dict[str, object], raw_data) if isinstance(raw_data, dict) else None
            if (status != 200 or result.get("ok") is not True or not isinstance(data, dict)
                    or type(data.get("message_id")) is not int):
                return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="Telegram 回执缺少已确认消息")
            provider_ids.append(str(data["message_id"]))
        return Receipt(status="delivered", provider_ids=tuple(provider_ids))


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
