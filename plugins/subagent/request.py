from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from plugins.delivery.api import Sink
from session.message import ContentPart, ContentReferences
from session.message_codec import json_value


PROFILE_TOOLS: dict[str, tuple[str, ...]] = {
    "research": ("read_file", "list_dir", "web_fetch", "web_search"),
    "scripting": ("read_file", "list_dir", "write_file", "edit_file", "shell", "write_stdin", "task_stop"),
    "general": ("read_file", "list_dir", "web_fetch", "web_search", "write_file", "edit_file", "shell", "write_stdin", "task_stop"),
}


class SpawnInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: str = Field(min_length=1, description="独立任务的目标、约束、上下文和预期结果")
    label: str | None = None
    profile: Literal["research", "scripting", "general"] = "research"
    run_in_background: bool = False
    retry_count: int = Field(default=0, ge=0)


class Request(BaseModel):
    """输入消息保存原任务选择；owner 回执只保存指针与结算状态。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    label: str
    profile: Literal["research", "scripting", "general"]
    background: bool
    retry_count: int = Field(ge=0)
    parent_session_id: str = Field(min_length=1)
    parent_message_id: str = Field(min_length=1)
    parent_part_index: int = Field(ge=0)
    origin: dict[str, str] | None
    sink: Sink | None
    program_binding: str = Field(min_length=1)
    tools: dict[str, str]

    @property
    def session_id(self) -> str:
        return "subagent:" + self.job_id

    @property
    def input_id(self) -> str:
        return "subagent-input:" + self.job_id


def check_request(part: ContentPart) -> ContentReferences:
    request = Request.model_validate(json_value(part.value))
    if set(request.tools) != set(PROFILE_TOOLS[request.profile]):
        raise ValueError("子任务工具选择与 profile 不一致")
    if request.origin is not None:
        from plugins.conversation.plugin import check_origin
        _ = check_origin(ContentPart("channel.origin", request.origin))
    if request.background and (request.origin is None or request.sink is None):
        raise ValueError("后台子任务缺少原发送目标")
    return ContentReferences(binding_ids=(request.program_binding, *request.tools.values(),
                                         *((request.sink.binding_id,) if request.sink is not None else ())))
