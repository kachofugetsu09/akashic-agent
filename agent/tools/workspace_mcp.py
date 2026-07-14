from __future__ import annotations

import json
from typing import Any

from agent.mcp.admin import WorkspaceMcpAdmin
from agent.tools.base import Tool


class WorkspaceMcpApplyTool(Tool):
    name = "workspace_mcp_apply"
    description = (
        "创建或更新一个非插件 workspace MCP server 声明，立即校验并热发布。"
        "不修改 config.toml，也不需要重启 Agent；新工具从下一轮开始可用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 64},
            "command": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "cwd": {"type": "string", "minLength": 1},
            "env": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "watch_paths": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["name", "command"],
        "additionalProperties": False,
    }

    def __init__(self, admin: WorkspaceMcpAdmin) -> None:
        self._admin = admin

    async def execute(self, **kwargs: Any) -> str:
        result = await self._admin.apply(
            name=str(kwargs["name"]),
            command=list(kwargs["command"]),
            cwd=kwargs.get("cwd"),
            env=dict(kwargs.get("env", {})),
            watch_paths=list(kwargs.get("watch_paths", [])),
        )
        return json.dumps(result, ensure_ascii=False)


class WorkspaceMcpRemoveTool(Tool):
    name = "workspace_mcp_remove"
    description = (
        "删除一个非插件 workspace MCP server 声明并热发布；旧代际会在已有 turn 释放后排空。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 64},
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def __init__(self, admin: WorkspaceMcpAdmin) -> None:
        self._admin = admin

    async def execute(self, **kwargs: Any) -> str:
        result = await self._admin.remove(str(kwargs["name"]))
        return json.dumps(result, ensure_ascii=False)


class WorkspaceMcpStatusTool(Tool):
    name = "workspace_mcp_status"
    description = (
        "查看非插件 workspace MCP 声明、已发布 generation、工具列表和最近热加载错误；"
        "环境变量只显示键名。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 64},
        },
        "additionalProperties": False,
    }

    def __init__(self, admin: WorkspaceMcpAdmin) -> None:
        self._admin = admin

    async def execute(self, **kwargs: Any) -> str:
        result = self._admin.status(kwargs.get("name"))
        return json.dumps(result, ensure_ascii=False)
