from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from plugins.ledger.contract import ContentPart, ContentReferences, json_value

# 兼容旧测试与历史消息夹具；新派发不再受这三套固定白名单限制，改为继承父会话工具池 + read_only 开关。
PROFILE_TOOLS: dict[str, tuple[str, ...]] = {
    "research": ("read_file", "list_dir", "web_fetch", "web_search"),
    "scripting": ("read_file", "list_dir", "write_file", "edit_file", "shell", "write_stdin", "task_stop"),
    "general": ("read_file", "list_dir", "web_fetch", "web_search", "write_file", "edit_file", "shell", "write_stdin", "task_stop"),
}

EXCLUDED_ALWAYS_TOOLS: frozenset[str] = frozenset({
    "message_push",
    "plugin_install",
    "plugin_update",
    "plugin_uninstall",
    "deploy_akashic",
})

DELEGATION_TOOLS: frozenset[str] = frozenset({"spawn", "spawn_manage"})

READ_ONLY_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "shell",
    "write_stdin",
    "task_stop",
    "set_working_directory",
    "spawn",
    "spawn_manage",
})


def is_delegation_tool(name: str) -> bool:
    return name in DELEGATION_TOOLS or name.startswith("delegate_")


def is_read_only_excluded(name: str) -> bool:
    return name in READ_ONLY_EXCLUDED_TOOLS or name.startswith("delegate_")


class SinkInput(TypedDict):
    name: str
    binding_id: str
    address: str


class SpawnInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: str = Field(min_length=1, description="独立任务的目标、约束、上下文和预期结果")
    label: str | None = None
    read_only: bool = Field(
        default=False,
        description="可选：只读模式，仅保留读取与检索类工具，禁止修改文件、执行命令或继续派发子任务",
    )
    profile: Literal["research", "scripting", "general"] | None = Field(
        default=None,
        description="可选兼容字段；传入 research 等价于 read_only=true",
    )
    cwd: str | None = Field(default=None, description="可选：子任务的初始工作目录（相对路径以父会话当前目录解析，省略时继承父会话工作目录）")
    run_in_background: bool = False
    retry_count: int = Field(default=0, ge=0)
    model: str | None = Field(default=None, description="可选：指定子任务使用的模型（型号名或 model_id）；留空按子任务设置解析")
    connection: str | None = Field(default=None, description="可选：指定模型所属的连接渠道（连接名称或 connection_id）")
    reasoning_effort: str | None = Field(default=None, description="可选：配合指定模型的思考强度（如 low、medium、high）")


class Request(BaseModel):
    """输入消息保存原任务选择；owner 回执只保存指针与结算状态。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    label: str
    profile: Literal["research", "scripting", "general"] = "general"
    read_only: bool = False
    depth: int = Field(default=1, ge=1)
    root_session_id: str | None = Field(default=None, min_length=1)
    background: bool
    retry_count: int = Field(ge=0)
    parent_session_id: str = Field(min_length=1)
    parent_message_id: str = Field(min_length=1)
    parent_part_index: int = Field(ge=0)
    origin: dict[str, str] | None
    sink: SinkInput | None
    program_binding: str = Field(min_length=1)
    tools: dict[str, str]
    # 为空时子任务用默认模型；旧请求没有这两个字段，恢复时同样回到默认。
    model_id: str | None = Field(default=None, min_length=1)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    # 委派时冻结的子任务初始工作目录；旧请求没有该字段时回退到子任务目录。
    cwd: str | None = Field(default=None, min_length=1)

    @property
    def session_id(self) -> str:
        return "subagent:" + self.job_id

    @property
    def input_id(self) -> str:
        return "subagent-input:" + self.job_id


def check_request(part: ContentPart, *, check_origin: Callable[[ContentPart], ContentReferences]) -> ContentReferences:
    request = Request.model_validate(json_value(part.value))
    if not request.tools or any(not name or not binding for name, binding in request.tools.items()):
        raise ValueError("子任务工具选择不能为空")
    if request.read_only and any(is_read_only_excluded(name) for name in request.tools):
        raise ValueError("只读子任务不能包含写入或执行类工具")
    if request.origin is not None:
        _ = check_origin(ContentPart("channel.origin", request.origin))
    if request.background and (request.origin is None or request.sink is None):
        raise ValueError("后台子任务缺少原发送目标")
    return ContentReferences(binding_ids=(request.program_binding, *request.tools.values(),
                                         *((request.sink["binding_id"],) if request.sink is not None else ())))
