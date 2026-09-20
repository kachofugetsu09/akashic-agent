from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_composition.artifacts import AttachmentKind
from agent.plugin_contracts import ContentPart, Output
from agent.plugin_contracts import decode_body, encode_body
from agent.plugin_contracts import json_value

from .boundary import CONTENT, DELIVERY, CallSource, ReceiptView, ToolResult


class PushInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target_channel: str = Field(min_length=1)
    target_chat_id: str = Field(min_length=1)
    message: str | None = None
    file: str | None = None
    image: str | None = None

    @model_validator(mode="after")
    def check_content(self) -> Self:
        if not self.message and not self.file and not self.image:
            raise ValueError("message、file、image 至少提供一个")
        return self


class PushSink(BaseModel):
    """message_push 固定到工具参数中的目的地结构。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: str = Field(min_length=1)
    binding_id: str = Field(min_length=1)
    address: str = Field(min_length=1)


class PreparedPush(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sink: PushSink
    body: str


def message_id(key: str) -> str:
    return "message-push:" + hashlib.sha256(key.encode()).hexdigest()


def result(message_id: str, receipt: ReceiptView) -> ToolResult:
    payload: dict[str, object] = {
        "message_id": message_id,
        "status": receipt.status,
        "provider_ids": receipt.provider_ids,
        "error": receipt.error,
    }
    outcome = "success" if receipt.status == "delivered" else "error"
    return ToolResult(outcome, (ContentPart("text", json.dumps(payload, ensure_ascii=False)),))


class MessagePush:
    """工具只拥有参数准备和目标消息；发送事实与恢复由 Delivery 拥有。"""

    idempotent = True

    def __init__(self, ctx: Context, senders: Mapping[str, str]):
        self._ctx = ctx
        self._senders = senders

    async def prepare(
        self,
        arguments: Mapping[str, object],
        source: CallSource | None = None,
    ) -> Mapping[str, object] | str:
        """先固定出站绑定，再把原文件与 URL 保存为不可变内容，恢复不重读来源。"""
        try:
            request = PushInput.model_validate(dict(arguments))
        except ValidationError as error:
            return str(error)
        ctx = self._ctx
        try:
            binding = self._senders[request.target_channel]
        except KeyError:
            return f"未知目标渠道：{request.target_channel}"
        parts = [ContentPart("text", request.message)] if request.message else []
        for path, kind in ((request.file, AttachmentKind.FILE), (request.image, AttachmentKind.IMAGE)):
            if path:
                ref = await ctx.require(ARTIFACT_IMPORT).import_source(path, kind)
                parts.append(ContentPart("artifact_ref", ref.artifact_id))
        prepared = PreparedPush(sink=PushSink(name=request.target_channel, binding_id=binding,
                                              address=request.target_chat_id),
                                body=encode_body(Output(tuple(parts), "complete")))
        return prepared.model_dump(mode="json")

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResult:
        """目标正文与首次选路同事务提交，重复调用只恢复原发送。"""
        prepared = PreparedPush.model_validate(json_value(arguments))
        body = decode_body(prepared.body)
        if not isinstance(body, Output) or body.finish != "complete":
            raise ValueError("原推送参数必须保存完整 Output")
        ctx = self._ctx
        delivery = ctx.require(DELIVERY).open(ctx)
        identity = message_id(key)
        async with ctx.require(CONTENT).bind() as content:
            checks = content.checks
            try:
                text_check = checks["text"]
                artifact_check = checks["artifact_ref"]
            except KeyError as error:
                raise RuntimeError("Content owner 缺少基础内容检查器") from error
            writer = ctx.require(MESSAGE_WRITERS).bind(
                ctx, author="message_push", source="message_push", body_types=(Output,),
                content={"text": text_check, "artifact_ref": artifact_check},
            )(f"{prepared.sink.name}:{prepared.sink.address}")
            try:
                _, selected = delivery.publish(
                    writer,
                    identity,
                    body,
                    (prepared.sink.model_dump(mode="json"),),
                    passive=True,
                )
            finally:
                writer.expire()
        if selected.sinks != (prepared.sink.name,):
            raise ValueError("原推送的发送选择不一致")
        receipt = await delivery.send(identity, prepared.sink.name)
        return result(identity, receipt)

    async def query(self, key: str) -> ToolResult | None:
        """沿原选择恢复发送；未发布时允许同 key 的 invoke 完成原授权操作。"""
        delivery = self._ctx.require(DELIVERY).open(self._ctx)
        identity = message_id(key)
        selected = delivery.selection(identity)
        if selected is None:
            return None
        if len(selected.sinks) != 1:
            raise ValueError("原推送必须只有一个发送目的地")
        receipt = await delivery.send(identity, selected.sinks[0])
        return result(identity, receipt)
