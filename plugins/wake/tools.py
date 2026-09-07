from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from plugins.tools.api import CallSource, Denied, InvalidArguments, Result
from session.message import ContentPart, Output, ToolCall
from session.message_codec import json_value

from .request import STAGE_TOOLS, read_phase, read_request


class ScreenedItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    candidate_id: str = Field(min_length=1)
    initial_interest: str = Field(min_length=1)
    question: str = Field(min_length=1)


class Screen(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    items: list[ScreenedItem] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len({item.candidate_id for item in self.items}) != len(self.items):
            raise ValueError("初筛不能重复同一个候选")
        return self


class Share(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str = Field(min_length=1)
    items: list[str] = Field(max_length=5)

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len(set(self.items)) != len(self.items):
            raise ValueError("分享不能重复同一个候选")
        return self


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str = Field(min_length=1)


class Skip(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(min_length=1)


SCHEMAS: dict[str, type[BaseModel]] = {
    "screen_content": Screen, "share_content": Share, "share_alert": Alert, "skip_content": Skip,
}


class DecisionTool:
    """只在真实 Wake 调用里记录结构化决定；发送和领域确认仍归来源处理。"""

    idempotent = True

    def __init__(self, name: str):
        self.name = name

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        if source is None:
            raise Denied("Wake 决策只能由内部消息程序调用")
        try:
            request = read_request(source.messages)
            _, phase = read_phase(source.messages, request)
        except ValueError as error:
            raise Denied("Wake 决策缺少真实内部请求") from error
        message = next(message for message in source.messages if message.message_id == source.call_ref.message_id)
        if message.source != "wake" or not isinstance(message.body, Output):
            raise Denied("当前来源无权提交 Wake 决策")
        call = message.body.parts[source.call_ref.part_index]
        if not isinstance(call, ToolCall) or self.name not in STAGE_TOOLS[phase.stage] or call.binding_id != request.tools.get(self.name):
            raise Denied("当前阶段未授予此 Wake 工具")
        try:
            result = SCHEMAS[self.name].model_validate(json_value(arguments))
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        return result.model_dump(mode="json")

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        return Result("success", (ContentPart("text", "Wake 决定已记录。"),))

    async def query(self, key: str) -> Result | None:
        return None
