from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.tools.base import Tool
from agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from agent.tools.registry import ToolRegistry
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore

logger = logging.getLogger(__name__)

FORCED_WRITE_PROMPT = (
    "步数即将用尽。你必须现在调用 write_file 或 edit_file，\n"
    "将当前 working files 更新到最新状态。\n"
    "下一步将强制结束，请确保进度已完整写入。"
)

FORCED_FINISH_PROMPT = (
    "你正在执行 Drift 任务，步数已用尽。\n"
    "根据上方对话历史，立即调用 finish_drift，填写：\n"
    "- skill_used：你执行的技能名\n"
    "- one_line：一句话总结本次做了什么\n"
    "- next：下次从哪里继续（一句话）\n"
    "- note：全局备注（可选）"
)


@dataclass
class DriftToolDeps:
    drift_dir: Path
    store: DriftStateStore
    memory: Any = None
    shared_tools: ToolRegistry | None = None
    send_message_fn: Any = None
    max_web_fetch_chars: int = 8_000


class SendMessageTool(Tool):
    def __init__(self, ctx: AgentTickContext, send_message_fn: Any) -> None:
        self._ctx = ctx
        self._send_message_fn = send_message_fn

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        return "发送一条消息给用户。单次 Drift run 最多只能调用一次。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要发送的消息内容"},
            },
            "required": ["content"],
        }

    async def execute(self, content: str, **_: Any) -> str:
        if self._send_message_fn is None:
            logger.info("[drift_tools] send_message unavailable")
            return json.dumps({"error": "send_message not configured"}, ensure_ascii=False)
        if self._ctx.drift_message_sent:
            logger.info("[drift_tools] send_message rejected: already used")
            return json.dumps(
                {"error": "send_message already used in this drift run"},
                ensure_ascii=False,
            )
        if not str(content or "").strip():
            logger.info("[drift_tools] send_message rejected: empty content")
            return json.dumps({"error": "content is required"}, ensure_ascii=False)
        ok = await self._send_message_fn(content)
        if not ok:
            logger.warning("[drift_tools] send_message failed")
            return json.dumps({"error": "send_message failed"}, ensure_ascii=False)
        self._ctx.drift_message_sent = True
        logger.info("[drift_tools] send_message ok")
        return json.dumps({"ok": True}, ensure_ascii=False)


class FinishDriftTool(Tool):
    def __init__(
        self,
        ctx: AgentTickContext,
        store: DriftStateStore,
    ) -> None:
        self._ctx = ctx
        self._store = store

    @property
    def name(self) -> str:
        return "finish_drift"

    @property
    def description(self) -> str:
        return "【终止工具】结束本次 Drift，保存进度状态。调用后 loop 立即结束。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_used": {"type": "string"},
                "one_line": {"type": "string"},
                "next": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["skill_used", "one_line", "next"],
        }

    async def execute(
        self,
        skill_used: str,
        one_line: str,
        next: str,
        note: str | None = None,
        **_: Any,
    ) -> str:
        skill_name = str(skill_used or "").strip()
        if skill_name not in self._store.valid_skill_names():
            logger.info("[drift_tools] finish_drift rejected unknown skill=%s", skill_name)
            return json.dumps(
                {"error": f"unknown skill: {skill_name}"},
                ensure_ascii=False,
            )
        summary = str(one_line or "").strip()
        next_action = str(next or "").strip()
        if not summary:
            return json.dumps({"error": "one_line is required"}, ensure_ascii=False)
        if not next_action:
            return json.dumps({"error": "next is required"}, ensure_ascii=False)
        note_text = str(note).strip() if note is not None else None
        self._store.save_finish(
            skill_used=skill_name,
            one_line=summary,
            next_action=next_action,
            note=note_text,
            now_utc=self._ctx.now_utc,
        )
        self._ctx.drift_finished = True
        logger.info(
            "[drift_tools] finish_drift ok: skill=%s one_line=%s next=%s",
            skill_name,
            summary[:120],
            next_action[:100],
        )
        return json.dumps({"ok": True}, ensure_ascii=False)


class MountServerTool(Tool):
    """挂载一个已连接的 MCP server，使其工具在本次 drift 中可用。"""

    def __init__(self, shared_tools: ToolRegistry, mounted: set[str]) -> None:
        self._shared = shared_tools
        self._mounted = mounted

    @property
    def name(self) -> str:
        return "mount_server"

    @property
    def description(self) -> str:
        return (
            "挂载一个已连接的 MCP server，使其工具在本次 drift 中可用。"
            "挂载后即可直接调用该 server 的工具。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "要挂载的 MCP server 名称",
                },
            },
            "required": ["server"],
        }

    async def execute(self, server: str, **_: Any) -> str:
        server = str(server or "").strip()
        if not server:
            return json.dumps({"error": "server is required"}, ensure_ascii=False)
        names = self._shared.get_tool_names_by_source("mcp", server)
        if not names:
            return json.dumps(
                {"error": f"MCP server '{server}' 不存在或未连接"},
                ensure_ascii=False,
            )
        new = names - self._mounted
        if not new:
            return json.dumps(
                {"ok": True, "message": f"'{server}' 已挂载，无新增工具", "tools": sorted(names)},
                ensure_ascii=False,
            )
        self._mounted |= new
        logger.info("[drift_tools] mount_server ok: server=%s new=%s", server, sorted(new))
        return json.dumps(
            {"ok": True, "tools": sorted(names), "new": sorted(new)},
            ensure_ascii=False,
        )


class DriftWebFetchTool(Tool):
    def __init__(self, wrapped: Tool, max_chars: int) -> None:
        self._wrapped = wrapped
        self._max_chars = max(1, int(max_chars))

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def description(self) -> str:
        return self._wrapped.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._wrapped.parameters

    async def execute(self, **kwargs: Any) -> str:
        result = await self._wrapped.execute(**kwargs)
        if not isinstance(result, str):
            return result
        try:
            payload = json.loads(result)
        except Exception:
            return result
        text = payload.get("text")
        if not isinstance(text, str) or len(text) <= self._max_chars:
            return result
        payload["text"] = text[: self._max_chars]
        payload["length"] = len(payload["text"])
        payload["truncated"] = True
        payload["note"] = (
            f"内容已截断至 {self._max_chars} 字符，"
            "如需更多内容请缩小范围或改用更精确的读取方式"
        )
        return json.dumps(payload, ensure_ascii=False)


class DriftReadFileTool(Tool):
    def __init__(self, drift_dir: Path) -> None:
        self._drift_dir = drift_dir
        self._relative_reader = ReadFileTool(allowed_dir=drift_dir)
        self._absolute_reader = ReadFileTool()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return self._absolute_reader.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._absolute_reader.parameters

    async def execute(self, path: str, **kwargs: Any) -> Any:
        raw = str(path or "").strip()
        if not raw:
            return await self._absolute_reader.execute(path=path, **kwargs)
        reader = self._absolute_reader if Path(raw).expanduser().is_absolute() else self._relative_reader
        return await reader.execute(path=path, **kwargs)


def build_drift_tool_registry(
    *,
    ctx: AgentTickContext,
    deps: DriftToolDeps,
    mounted_tool_names: set[str] | None = None,
) -> ToolRegistry:
    tools = ToolRegistry()
    drift_dir = deps.drift_dir
    tools.register(DriftReadFileTool(drift_dir), risk="read-only")
    tools.register(WriteFileTool(allowed_dir=drift_dir), risk="write")
    tools.register(EditFileTool(allowed_dir=drift_dir), risk="write")

    shared = deps.shared_tools
    for name in (
        "recall_memory",
        "web_fetch",
        "web_search",
        "fetch_messages",
        "search_messages",
        "shell",
    ):
        if shared is None:
            continue
        tool = shared.get_tool(name)
        if tool is not None:
            if name == "web_fetch":
                tool = DriftWebFetchTool(tool, deps.max_web_fetch_chars)
            risk = "external-side-effect" if name == "shell" else "read-only"
            tools.register(tool, risk=risk)

    # mount_server: 只有 shared registry 里有 MCP 工具时才注册
    if shared is not None and shared.get_mcp_server_names():
        mounted = mounted_tool_names if mounted_tool_names is not None else set()
        tools.register(MountServerTool(shared, mounted), risk="read-only")

    tools.register(
        SendMessageTool(ctx, deps.send_message_fn),
        risk="external-side-effect",
    )
    tools.register(FinishDriftTool(ctx, deps.store), risk="write")
    return tools
