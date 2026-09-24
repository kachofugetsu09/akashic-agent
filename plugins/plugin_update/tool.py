from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict

from typing_extensions import TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES, UpdateStatus
from .inputs import INPUT_ORIGIN
from .inputs import CallSource, Result
from agent.plugin_contracts import ContentPart
from agent.plugin_contracts import json_value


class SinkInput(TypedDict):
    name: str
    binding_id: str
    address: str


class InstallInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str = Field(min_length=1)
    marketplace: str = Field(min_length=1)
    ref: str = ""
    sparse: list[str] = Field(default_factory=list)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    install: InstallInput
    session_id: str = Field(min_length=1)
    sink: SinkInput | None


_RETIRED_INSTALL_FIELDS = frozenset({
    "validation_prompt",
    "validation_tools",
    "excluded_materials",
})


def update_id(key: str) -> str:
    return "plugin-update:" + hashlib.sha256(key.encode()).hexdigest()


def receipt(status: UpdateStatus) -> Result:
    return Result("success" if status.state in {"accepted", "active"} else "error",
                  (ContentPart("text", json.dumps(asdict(status), ensure_ascii=False)),))


def decode_request(value: object) -> Request:
    """Decode history while dropping only the three retired validation fields."""
    raw = json_value(value)
    if not isinstance(raw, Mapping):
        raise ValueError("插件更新请求必须是对象")
    install = raw.get("install")
    if not isinstance(install, Mapping):
        raise ValueError("插件更新 install 必须是对象")
    current = {
        key: item for key, item in install.items()
        if key not in _RETIRED_INSTALL_FIELDS
    }
    payload = dict(raw)
    payload["install"] = current
    return Request.model_validate(payload)


class InstallPlugin:
    """保存原请求与通知地址；更新事实和重复查询归 Core journal。"""

    idempotent = False

    def __init__(self, ctx: Context, senders: Mapping[str, str]):
        self._ctx = ctx
        self._senders = senders

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        """固定实际发起消息的 Session 与发送者，不读取后续输入改变通知地址。"""
        try:
            install = InstallInput.model_validate(json_value(arguments))
        except ValidationError as error:
            return str(error)
        if source is None or not source.messages:
            return '插件更新需要实际发起消息'
        message = source.messages[-1]
        reader = self._ctx.require(MESSAGE_CATALOG).reader(message.session_id)
        route = self._ctx.require(INPUT_ORIGIN)(reader, message.source, through_seq=message.seq)
        sink = None
        if route is not None:
            name, address = route
            try:
                binding = self._senders[name]
            except KeyError as error:
                return f'发起渠道没有可恢复发送者：{name}'
            sink = SinkInput(name=name, binding_id=binding, address=address)
        return Request(install=install, session_id=message.session_id, sink=sink).model_dump(mode="json")

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """一次新调用写入通知意图，再提交公开安装；不等待父 Turn 结束。"""
        request = Request.model_validate(json_value(arguments))
        identity = update_id(key)
        ctx = self._ctx
        store = ctx.require(OWNER_STATE).open(ctx)
        updates = ctx.require(PLUGIN_UPDATES)
        if store.read(identity) is not None or updates.read(ctx, identity) is not None:
            raise RuntimeError("已有插件更新请求只能查询")
        _ = store.transact(lambda tx: tx.save(identity, request.model_dump(mode="json"), expected_version=None))
        install = request.install
        status = await updates.install(ctx, identity, source=install.source,
            marketplace=install.marketplace, ref=install.ref, sparse=tuple(install.sparse))
        return receipt(status)

    async def query(self, key: str) -> Result | None:
        """只读取原安装请求；未知结果不自动重跑或猜测当前制品。"""
        ctx = self._ctx
        identity = update_id(key)
        status = ctx.require(PLUGIN_UPDATES).read(ctx, identity)
        if status is not None:
            return receipt(status)
        if ctx.require(OWNER_STATE).open(ctx).read(identity) is None:
            return None
        return Result("error", (ContentPart("text", "原安装请求没有可读取的当前状态；请发起新的更新请求。"),))
