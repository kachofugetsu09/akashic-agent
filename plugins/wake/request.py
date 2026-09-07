from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
import json
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.tasks import Task
from session.log import MessageReader

from plugins.delivery.api import Sink
from session.message import ContentPart, ContentReferences, Control, Input, Message
from session.message_codec import json_value

from .api import DeliveryTarget

Owner = Literal["content", "drift", "alert"]
Stage = Literal["screen", "investigate", "drift", "alert"]
TOOLS: dict[Owner, tuple[str, ...]] = {
    "content": ("screen_content", "recall_memory", "web_fetch", "share_content", "skip_content"),
    "drift": ("share_content", "skip_content"),
    "alert": ("share_alert",),
}
STAGE_TOOLS: dict[Stage, tuple[str, ...]] = {
    "screen": ("screen_content",),
    "investigate": ("recall_memory", "web_fetch", "share_content", "skip_content"),
    "drift": ("share_content", "skip_content"),
    "alert": ("share_alert",),
}


class Request(BaseModel):
    """一次 Wake 选择的固定输入；阶段进度只从后续 Message 和领域回执读取。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    flow_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    owner: Owner
    now: AwareDatetime
    timezone: str
    target: DeliveryTarget
    sink: Sink
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
    def check_choices(self) -> Self:
        if set(self.tools) != set(TOOLS[self.owner]) or any(not value for value in self.tools.values()):
            raise ValueError("Wake 原工具集合与职责不一致")
        if (
            self.target.channel != self.sink.name or self.target.recipient != self.sink.address
        ):
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

    @property
    def notification_id(self) -> str:
        return "wake-notification:" + self.flow_id

    @property
    def accepted(self) -> dict[str, str]:
        # EventMail/Drift 既有边界字段叫 turn_id；新来源明确引用自己的原 Input。
        return {"session_id": self.session_id, "turn_id": self.input_id}

    def phase_id(self, stage: Stage) -> str:
        return self.input_id + ":" + stage


class Phase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_id: str = Field(min_length=1)
    stage: Stage


class WakeFailure(BaseModel):
    """Wake 自己保存模型失败分类；Core 仍只保存普通 Control.reason。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["wake.failure.v1"] = "wake.failure.v1"
    message: str
    retryable: bool


def retryable(message: Message | None) -> bool | None:
    if message is None or not isinstance(message.body, Control) or message.body.action != "failure" or message.body.reason is None:
        return None
    try:
        value = json.loads(message.body.reason)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("kind") != "wake.failure.v1":
        return None
    return WakeFailure.model_validate(value).retryable


def check_request(part: ContentPart) -> ContentReferences:
    request = Request.model_validate_json(json.dumps(json_value(part.value)))
    return ContentReferences(binding_ids=(request.program_binding, *request.tools.values(),
        request.sink.binding_id))


def check_phase(part: ContentPart) -> ContentReferences:
    _ = Phase.model_validate(json_value(part.value))
    return ContentReferences()


def read_request(messages: Sequence[Message]) -> Request:
    """从实际内部 Input 读取唯一原选择，不接受来自另一来源或会话的伪造材料。"""
    found = [(message, part) for message in messages if isinstance(message.body, Input)
             for part in message.body.parts if part.kind == "wake.request"]
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


WAKE_PROGRAM = ServiceKey[Callable[[Task, MessageReader, Request], Awaitable[Message]]]("wake.program.v1")
