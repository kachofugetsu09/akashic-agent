from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES, UpdateStatus
from plugins.delivery.api import Sink
from plugins.delivery_policy.plugin import input_origin
from plugins.tools.api import CallSource, InvalidArguments, Result
from session.message import ContentPart
from session.message_codec import json_value


class InstallInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str = Field(min_length=1)
    marketplace: str = Field(min_length=1)
    ref: str = ""
    sparse: list[str] = Field(default_factory=list)
    validation_prompt: str = Field(min_length=1)
    validation_tools: list[str] | None = None
    excluded_materials: list[str] = Field(default_factory=list)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    install: InstallInput
    session_id: str = Field(min_length=1)
    sink: Sink | None


def update_id(key: str) -> str:
    return "plugin-update:" + hashlib.sha256(key.encode()).hexdigest()


def receipt(status: UpdateStatus) -> Result:
    return Result("error" if status.phase == "rolled_back" else "success",
                  (ContentPart("text", json.dumps(asdict(status), ensure_ascii=False)),))


class InstallPlugin:
    """保存原请求与通知地址；更新事实和重复查询归 Core journal。"""

    idempotent = False

    def __init__(self, ctx: Context, senders: Mapping[str, str]):
        self._ctx = ctx
        self._senders = senders

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        """固定实际发起消息的 Session 与发送者，不读取后续输入改变通知地址。"""
        try:
            install = InstallInput.model_validate(json_value(arguments))
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        if source is None or not source.messages:
            raise InvalidArguments("插件更新需要实际发起消息")
        message = source.messages[-1]
        reader = self._ctx.require(MESSAGE_CATALOG).reader(message.session_id)
        route = input_origin(reader, message.source, through_seq=message.seq)
        sink = None
        if route is not None:
            name, address = route
            try:
                binding = self._senders[name]
            except KeyError as error:
                raise InvalidArguments(f"发起渠道没有可恢复发送者：{name}") from error
            sink = Sink(name=name, binding_id=binding, address=address)
        return Request(install=install, session_id=message.session_id, sink=sink).model_dump(mode="json")

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """一次新调用写入通知意图，再准备候选；不等待验证或父 Turn 结束。"""
        request = Request.model_validate(json_value(arguments))
        identity = update_id(key)
        ctx = self._ctx
        store = ctx.require(OWNER_STATE).open(ctx)
        if store.read(identity) is not None:
            raise RuntimeError("已有插件更新请求只能查询")
        _ = store.transact(lambda tx: tx.save(identity, request.model_dump(mode="json"), expected_version=None))
        install = request.install
        status = await ctx.require(PLUGIN_UPDATES).install(ctx, identity, source=install.source,
            marketplace=install.marketplace, ref=install.ref, sparse=tuple(install.sparse))
        return receipt(status)

    async def query(self, key: str) -> Result | None:
        """只读取原更新；进程死亡后的回退是失败，不自动重跑安装或验证。"""
        ctx = self._ctx
        identity = update_id(key)
        status = ctx.require(PLUGIN_UPDATES).read(ctx, identity)
        if status is not None:
            return receipt(status)
        if ctx.require(OWNER_STATE).open(ctx).read(identity) is None:
            return None
        return Result("error", (ContentPart("text", "原更新未完成候选准备；请发起新的更新请求。"),))
