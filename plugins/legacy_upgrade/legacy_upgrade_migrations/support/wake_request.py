"""Frozen Wake request reader used by the historical empty-response migration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Literal, TypedDict

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from agent.plugin_contracts import ContentPart, ContentReferences, Input, Message, json_value


Owner = Literal["content", "drift", "alert"]
Stage = Literal["screen", "investigate", "drift", "alert"]


class SinkValue(TypedDict):
    name: str
    binding_id: str
    address: str


class DeliveryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    channel: str
    recipient: str
    session_id: str


TOOLS: dict[Owner, tuple[str, ...]] = {
    "content": ("screen_content", "recall_memory", "web_fetch", "share_content", "skip_content"),
    "drift": ("share_content", "skip_content"),
    "alert": ("share_alert",),
}


class Request(BaseModel):
    """One immutable historical Wake request and its binding identities."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    flow_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    owner: Owner
    now: AwareDatetime
    timezone: str
    target: DeliveryTarget
    sink: SinkValue
    program_binding: str = Field(min_length=1)
    tools: dict[str, str]
    snapshot_seq: int = Field(ge=0)
    items: tuple[dict[str, object], ...] = ()
    proposals: tuple[dict[str, object], ...] = ()
    alert_ref: dict[str, str] | None = None
    model_id: str | None = Field(default=None, min_length=1)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    rules: str
    history: str
    events: tuple[dict[str, object], ...] = ()

    @model_validator(mode="after")
    def check_choices(self) -> "Request":
        if set(self.tools) != set(TOOLS[self.owner]) or any(not value for value in self.tools.values()):
            raise ValueError("Wake 原工具集合与职责不一致")
        if self.target.channel != self.sink["name"] or self.target.recipient != self.sink["address"]:
            raise ValueError("Wake 原目标与 Sink 不一致")
        if self.owner == "alert":
            if self.alert_ref is None or set(self.alert_ref) != {"source_id", "event_id", "mail_id"} or any(
                not value or value.strip() != value for value in self.alert_ref.values()
            ):
                raise ValueError("Wake Alert 缺少准确原 envelope 身份")
        elif self.alert_ref is not None:
            raise ValueError("非 Alert 职责不能声明告警身份")
        return self

    @property
    def session_id(self) -> str:
        return "wake:" + self.flow_id

    @property
    def input_id(self) -> str:
        return "wake-input:" + self.flow_id

    def phase_id(self, stage: Stage) -> str:
        return self.input_id + ":" + stage


class Phase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    input_id: str = Field(min_length=1)
    stage: Stage


class WakeFailure(BaseModel):
    """Wake failure details are stored as an ordinary Control reason."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["wake.failure.v1"] = "wake.failure.v1"
    message: str
    retryable: bool


def read_request(messages: Sequence[Message]) -> Request:
    """Read and verify the one original Wake request from the actual session."""
    found = [
        (message, part)
        for message in messages
        if isinstance(message.body, Input)
        for part in message.body.parts
        if part.kind == "wake.request"
    ]
    if len(found) != 1:
        raise ValueError("Wake 内部会话缺少唯一原请求")
    message, part = found[0]
    request = Request.model_validate_json(json.dumps(json_value(part.value)))
    if (message.author, message.source, message.session_id, message.message_id) != (
        "wake", "wake", request.session_id, request.input_id
    ):
        raise ValueError("Wake 原请求身份不一致")
    return request


def read_phase(messages: Sequence[Message], request: Request) -> tuple[Message, Phase]:
    """Read and verify the latest phase input for the frozen request."""
    for message in reversed(messages):
        if not isinstance(message.body, Input):
            continue
        parts = [part for part in message.body.parts if part.kind == "wake.phase"]
        if not parts:
            continue
        if len(parts) != 1:
            raise ValueError("Wake 阶段 Input 缺少唯一材料")
        phase = Phase.model_validate(json_value(parts[0].value))
        if (phase.input_id, message.source, message.author, message.session_id, message.message_id) != (
            request.input_id, "wake", "wake", request.session_id, request.phase_id(phase.stage)
        ):
            raise ValueError("Wake 阶段身份不一致")
        if phase.stage not in ({"screen", "investigate"} if request.owner == "content" else {request.owner}):
            raise ValueError("Wake 阶段不属于原职责")
        return message, phase
    raise ValueError("Wake 程序缺少已提交阶段输入")
