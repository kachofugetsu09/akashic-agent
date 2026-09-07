"""OneBot 请求和 echo 回执一一对应，不重试外部效果。"""
import asyncio
import base64
import json
import secrets
from dataclasses import dataclass
from typing import cast

from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from agent.plugin_composition.artifacts import ArtifactRead
from plugins.delivery.api import Receipt
from plugins.delivery.content import AttachmentReadError, read_content
from session.artifacts import AttachmentKind
from session.log import MessageCatalog
from session.message import Message


@dataclass(frozen=True, slots=True)
class Request:
    action: str
    params: dict[str, object]
    upload: bool = False


class QQSender:
    idempotent = False

    def __init__(self, connection: ClientConnection, timeout: float, catalog: MessageCatalog, artifacts: ArtifactRead):
        self._connection = connection
        self._timeout = timeout
        self._catalog = catalog
        self._artifacts = artifacts

    async def query(self, key: str, address: str) -> Receipt | None:
        return None

    async def send(self, key: str, address: str, message: Message) -> Receipt:
        """按正文顺序提交 OneBot 消息和文件，保留部分成功证据。"""
        # 1. 准备全部字节，不把群号、私聊号或文件路径留给 provider 猜测。
        group = address.startswith("gqq:")
        number = address[4:] if group else address
        if not number.isascii() or not number.isdecimal() or int(number) <= 0:
            return Receipt(status="rejected", error="QQ 地址必须是正整数私聊号或 gqq:群号")
        target = {"group_id" if group else "user_id": int(number)}
        kind = "group" if group else "private"
        try:
            parts = await read_content(message, self._catalog, self._artifacts)
        except AttachmentReadError as error:
            return Receipt(status="rejected", error=f"QQ 本地材料读取失败：{error}")
        requests: list[Request] = []
        for part in parts:
            if isinstance(part, str):
                requests.append(Request(f"send_{kind}_msg", {**target, "message": [{"type": "text", "data": {"text": part}}]}))
            else:
                encoded = "base64://" + base64.b64encode(part.data).decode("ascii")
                if part.ref.kind == AttachmentKind.IMAGE:
                    requests.append(Request(f"send_{kind}_msg", {**target, "message": [{"type": "image", "data": {"file": encoded}}]}))
                else:
                    requests.append(Request(f"upload_{kind}_file", {**target, "file": encoded,
                                                                  "name": part.ref.filename or part.ref.artifact_id}, True))
        if not requests:
            return Receipt(status="rejected", error="消息没有可发送正文或附件")

        # 2. 文件成功可没有消息 ID；独立计数防止把已成功前缀误报为拒绝。
        provider_ids: list[str] = []
        completed = 0
        for request in requests:
            echo = secrets.token_hex(16)
            try:
                async with asyncio.timeout(self._timeout):
                    await self._connection.send(json.dumps({"action": request.action, "params": request.params, "echo": echo}))
                    result = await self._read_response(echo)
            except (ConnectionClosed, TimeoutError, OSError):
                return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="QQ 连接或回执未确认")
            except (json.JSONDecodeError, ValueError):
                return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="QQ 回执结构或 echo 无效")
            if result.get("status") == "failed" and type(result.get("retcode")) is int and result["retcode"] != 0:
                return Receipt(status="unknown" if completed else "rejected", provider_ids=tuple(provider_ids), error="QQ 拒绝请求")
            if result.get("status") != "ok" or type(result.get("retcode")) is not int or result["retcode"] != 0:
                return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="QQ 回执未确认成功")
            raw_data = result.get("data")
            data = cast(dict[str, object], raw_data) if isinstance(raw_data, dict) else raw_data
            if request.upload:
                if data is not None and not isinstance(data, dict):
                    return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="QQ 文件回执无效")
                if isinstance(data, dict) and cast(dict[str, object], data).get("file_id") is not None:
                    file_id = cast(dict[str, object], data)["file_id"]
                    if type(file_id) not in (str, int) or str(file_id) == "":
                        return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="QQ 文件 ID 无效")
                    provider_ids.append("file:" + str(file_id))
            else:
                message_id = cast(dict[str, object], data).get("message_id") if isinstance(data, dict) else None
                if type(message_id) not in (str, int) or str(message_id) == "":
                    return Receipt(status="unknown", provider_ids=tuple(provider_ids), error="QQ 回执缺少消息 ID")
                provider_ids.append(str(message_id))
            completed += 1
        return Receipt(status="delivered", provider_ids=tuple(provider_ids))

    async def _read_response(self, echo: str) -> dict[str, object]:
        """兼容同连接的事件帧；只有 exact echo 的响应能确认当前请求。"""
        while True:
            result = json.loads(await self._connection.recv())
            if not isinstance(result, dict):
                raise ValueError("OneBot 回执必须是对象")
            result = cast(dict[str, object], result)
            if "post_type" in result and "echo" not in result and "retcode" not in result:
                continue
            if result.get("echo") != echo:
                raise ValueError("OneBot echo 不匹配")
            return result
