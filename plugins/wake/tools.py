from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from plugins.ledger.contract import ContentPart, Message, Output, ToolCall, ToolResult
from plugins.ledger.contract import json_value

from ._boundary import CallSource, ToolResultValue
from .request import TOOLS, Request, read_phase, read_request
from .content import _candidate_payloads
from .selection import propose_content


class ScreenedItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    candidate_id: str = Field(min_length=1, description="逐字复制本轮候选的完整 candidate_id，不得截短、改写或自行生成。")
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


# 读取本会话某些 Wake 工具的成功决定；Wake 只认原归档 binding 的调用。
def successful_decisions(messages: Sequence[Message], request: Request, names: Collection[str], *,
                         after_seq: int = -1) -> list[tuple[str, BaseModel]]:
    by_id = {message.message_id: message for message in messages}
    found: list[tuple[str, BaseModel]] = []
    for message in messages:
        result = message.body
        if message.seq <= after_seq or message.source != "wake" or not isinstance(result, ToolResult) or result.outcome != "success":
            continue
        call_message = by_id[result.call_ref.message_id]
        assert isinstance(call_message.body, Output)
        call = call_message.body.parts[result.call_ref.part_index]
        assert isinstance(call, ToolCall)
        name = next((name for name, binding in request.tools.items() if binding == call.binding_id), None)
        if name in SCHEMAS and name in names:
            found.append((name, SCHEMAS[name].model_validate(json_value(call.arguments))))
    return found


class DecisionTool:
    """只在 Wake 会话的模型调用里记录结构化决定；发送和领域确认仍归来源处理。"""

    idempotent = True

    def __init__(self, name: str):
        self.name = name

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        # 1. 决定只属于持有原请求的 Wake 会话，且必须使用请求归档的工具 binding。
        if source is None:
            raise PermissionError('Wake 决定必须来自 Wake 会话中的模型调用')
        try:
            request = read_request(source.messages)
            start, _ = read_phase(source.messages, request)
        except ValueError as error:
            raise PermissionError('Wake 决定缺少真实内部请求') from error
        message = next(message for message in source.messages if message.message_id == source.call_ref.message_id)
        if message.source != "wake" or not isinstance(message.body, Output):
            raise PermissionError('当前来源无权提交 Wake 决定')
        call = message.body.parts[source.call_ref.part_index]
        if not isinstance(call, ToolCall) or self.name not in TOOLS[request.owner] or call.binding_id != request.tools.get(self.name):
            raise PermissionError('原 Wake 职责未授予此工具')
        try:
            result = SCHEMAS[self.name].model_validate(json_value(arguments))
        except ValidationError as error:
            return str(error)
        # 2. 顺序由技能说明，工具只核对前置事实；错误反馈给本轮模型纠正。
        if request.owner == "content":
            feedback = self._check_content(result, request, successful_decisions(
                source.messages, request, ("screen_content",), after_seq=start.seq))
            if feedback is not None:
                return feedback
        return result.model_dump(mode="json")

    # Content 先初筛一次，再只分享初筛选中的候选。
    def _check_content(self, result: BaseModel, request: Request, screens: list[tuple[str, BaseModel]]) -> str | None:
        if isinstance(result, Screen):
            if screens:
                return '本轮已经提交过初筛，不能重复初筛；继续调查后调用 share_content 或 skip_content。本次决定未记录。'
            proposal = propose_content(request.items, now=request.now)
            if proposal is None:
                raise RuntimeError("Wake 初筛缺少原候选集合")
            allowed = {item["candidate_id"] for item in _candidate_payloads(proposal)}
            unknown = [item.candidate_id for item in result.items if item.candidate_id not in allowed]
            if unknown:
                return f'candidate_id 不属于本轮候选: {unknown}。请逐字复制候选中的完整 ID，不得截短；本次决定未记录。'
            return None
        if not screens:
            return '还没有提交初筛：先按 proactive-wake 技能调用 screen_content，再分享或跳过。本次决定未记录。'
        if isinstance(result, Share):
            screened = {item.candidate_id for item in cast(Screen, screens[0][1]).items}
            outside = [item for item in result.items if item not in screened]
            if not result.items or outside:
                return f'share_content.items 必须是初筛选中的 1..5 个 candidate_id，越界: {outside}。本次决定未记录。'
        return None

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        if self.name == "screen_content":
            return ToolResultValue("success", (ContentPart("text", "初筛已记录。继续按技能调查这些候选，最后调用 share_content 或 skip_content。"),))
        return ToolResultValue("success", (ContentPart("text", "Wake 决定已记录。"),))

    async def query(self, key: str) -> ToolResultValue | None:
        return None
