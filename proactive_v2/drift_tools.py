from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from agent.tools.base import Tool, ToolResult
from agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from agent.tools.registry import ToolRegistry
from bus.events_lifecycle import DriftFinished
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore
from proactive_v2.outbound_text import normalize_outbound_text

logger = logging.getLogger(__name__)


def _clip_text(text: object, limit: int) -> str:
    value = str(text or "").strip()
    return value[:limit]


@dataclass
class DriftToolDeps:
    drift_dir: Path
    store: DriftStateStore
    workspace_dir: Path | None = None
    builtin_skills_dir: Path | None = None
    memory: Any = None
    recent_chat_fn: Any = None
    shared_tools: ToolRegistry | None = None
    send_message_fn: Any = None
    event_bus: Any = None


class SendMessageTool(Tool):
    def __init__(self, ctx: AgentTickContext, send_message_fn: Any) -> None:
        self._ctx = ctx
        self._send_message_fn = send_message_fn

    @property
    def name(self) -> str:
        return "message_push"

    @property
    def description(self) -> str:
        return (
            "向用户发送一条消息，可附带图片。单次 Drift run 最多只能调用一次。\n"
            "channel 和 chat_id 在 Drift 上下文中已由配置预设，可省略不填。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "要发送的消息内容"},
                "image": {"type": "string", "description": "要发送的一张图片本地路径或 URL"},
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要随消息发送的图片路径或 URL 列表",
                },
                "channel": {
                    "type": "string",
                    "description": "目标渠道（Drift 上下文可省略，已由配置预设）",
                },
                "chat_id": {
                    "type": "string",
                    "description": "目标会话 ID（Drift 上下文可省略，已由配置预设）",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        message: str = "",
        image: str = "",
        media: list[str] | str | None = None,
        channel: str = "",
        chat_id: str = "",
    ) -> str:
        _ = (channel, chat_id)
        text = normalize_outbound_text(message or "").strip()
        media_paths = self._normalize_media(image=image, media=media)
        if self._send_message_fn is None:
            logger.info("[drift_tools] message_push unavailable")
            return json.dumps({"error": "message_push not configured"}, ensure_ascii=False)
        if self._ctx.drift_message_sent:
            logger.info("[drift_tools] message_push rejected: already used")
            return json.dumps(
                {"error": "message_push already used in this drift run"},
                ensure_ascii=False,
            )
        if not text and not media_paths:
            logger.info("[drift_tools] message_push rejected: empty message and media")
            return json.dumps({"error": "message or media is required"}, ensure_ascii=False)
        ok = await self._send_message_fn(text, media_paths)
        if not ok:
            logger.warning("[drift_tools] message_push failed")
            return json.dumps({"error": "message_push failed"}, ensure_ascii=False)
        self._ctx.drift_message_sent = True
        logger.info("[drift_tools] message_push ok")
        return json.dumps({"ok": True}, ensure_ascii=False)

    @staticmethod
    def _normalize_media(*, image: str = "", media: list[str] | str | None = None) -> list[str]:
        paths: list[str] = []
        if image:
            paths.append(str(image).strip())
        if isinstance(media, str):
            paths.append(media.strip())
        elif media:
            paths.extend(str(item).strip() for item in media)
        return [path for path in paths if path]


class FinishDriftTool(Tool):
    def __init__(
        self,
        ctx: AgentTickContext,
        store: DriftStateStore,
        event_bus: Any = None,
    ) -> None:
        self._ctx = ctx
        self._store = store
        self._event_bus = event_bus

    @property
    def name(self) -> str:
        return "finish_drift"

    @property
    def description(self) -> str:
        return "【终止工具】结束本次 Drift，保存本轮摘要和连续性前情。调用后 loop 立即结束。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_used": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["completed", "paused", "waiting"],
                    "description": (
                        "completed 表示本轮小闭环完成；"
                        "paused 表示本轮未完成但可下次继续；"
                        "waiting 表示正在等待用户回复或外部条件。"
                    ),
                },
                "briefing": {"type": "string", "description": "本轮做了什么的一句话摘要"},
                "message_result": {
                    "type": "string",
                    "enum": ["sent", "silent"],
                    "description": (
                        "sent 表示本轮已经成功调用 message_push；"
                        "silent 表示本轮确认不该打扰用户，静默结束。"
                    ),
                },
                "scratchpad_update": {
                    "type": "string",
                    "description": "下次进入本 skill 时需要注入的自然语言前情",
                },
                "cursor_update": {
                    "type": "object",
                    "description": "结构化游标，供下轮脚本或流程直接决定下一步",
                },
                "journal_append": {
                    "type": "array",
                    "description": "追加本轮已完成事实，例如已问过、已生成、已审计",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entry_type": {"type": "string"},
                            "key": {"type": "string"},
                            "payload": {"type": "object"},
                        },
                        "required": ["entry_type"],
                    },
                },
                "global_note_update": {"type": "string"},
            },
            "required": ["skill_used", "status", "briefing", "message_result"],
        }

    async def execute(
        self,
        skill_used: str,
        status: str = "",
        briefing: str = "",
        message_result: str = "",
        scratchpad_update: str | None = None,
        cursor_update: dict[str, Any] | None = None,
        journal_append: list[dict[str, Any]] | dict[str, Any] | None = None,
        global_note_update: str | None = None,
    ) -> str:
        skill_name = str(skill_used or "").strip()
        if skill_name not in self._store.valid_skill_names():
            logger.info("[drift_tools] finish_drift rejected unknown skill=%s", skill_name)
            return json.dumps(
                {"error": f"unknown skill: {skill_name}"},
                ensure_ascii=False,
            )
        selected = str(self._ctx.drift_selected_skill or "").strip()
        if selected and skill_name != selected:
            return json.dumps(
                {"error": f"skill_used must match selected skill: {selected}"},
                ensure_ascii=False,
            )
        status_value = str(status or "").strip()
        if status_value not in {"completed", "paused", "waiting"}:
            return json.dumps(
                {"error": "status must be one of: completed, paused, waiting"},
                ensure_ascii=False,
            )
        summary = str(briefing or "").strip()
        if not summary:
            return json.dumps({"error": "briefing is required"}, ensure_ascii=False)
        scratchpad_text = str(scratchpad_update or "").strip()
        if status_value in {"paused", "waiting"} and not scratchpad_text:
            return json.dumps(
                {
                    "error": (
                        "scratchpad_update is required when "
                        "status is paused or waiting"
                    )
                },
                ensure_ascii=False,
            )
        message_result_value = str(message_result or "").strip()
        if message_result_value not in {"sent", "silent"}:
            return json.dumps(
                {"error": "message_result must be one of: sent, silent"},
                ensure_ascii=False,
            )
        if message_result_value == "sent" and not self._ctx.drift_message_sent:
            return json.dumps(
                {"error": "message_result=sent requires successful message_push first"},
                ensure_ascii=False,
            )
        if message_result_value == "silent" and self._ctx.drift_message_sent:
            return json.dumps(
                {"error": "message_result=silent conflicts with successful message_push"},
                ensure_ascii=False,
            )
        if cursor_update is not None and not isinstance(cursor_update, dict):
            return json.dumps(
                {"error": "cursor_update must be an object"},
                ensure_ascii=False,
            )
        journal_entries, journal_error = self._normalize_journal_append(journal_append)
        if journal_error:
            return json.dumps({"error": journal_error}, ensure_ascii=False)
        note_text = (
            str(global_note_update).strip()
            if global_note_update is not None
            else None
        )
        if not selected:
            self._ctx.drift_selected_skill = skill_name
        self._store.save_finish(
            skill_used=skill_name,
            status=status_value,
            briefing=summary,
            message_result=message_result_value,
            scratchpad_update=scratchpad_text or None,
            global_note_update=note_text,
            now_utc=self._ctx.now_utc,
            cursor_update=cursor_update,
            journal_append=journal_entries,
        )
        self._ctx.drift_finished = True
        if self._event_bus is not None:
            self._event_bus.enqueue(
                DriftFinished(
                    session_key=self._ctx.session_key,
                    skill_name=skill_name,
                    status=status_value,
                    briefing=summary,
                    message_result=message_result_value,
                    timestamp=self._ctx.now_utc,
                )
            )
        logger.info(
            "[drift_tools] finish_drift ok: skill=%s status=%s briefing=%s",
            skill_name,
            status_value,
            summary[:120],
        )
        return json.dumps({"ok": True}, ensure_ascii=False)

    @staticmethod
    def _normalize_journal_append(
        raw: list[dict[str, Any]] | dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str]:
        if raw is None:
            return [], ""
        items: list[Any] = raw if isinstance(raw, list) else [raw]
        result: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                return [], "journal_append items must be objects"
            data = cast(dict[str, Any], item)
            entry_type = str(data.get("entry_type") or "").strip()
            if not entry_type:
                return [], "journal_append.entry_type is required"
            payload = data.get("payload")
            if payload is not None and not isinstance(payload, dict):
                return [], "journal_append.payload must be an object"
            result.append(
                {
                    "entry_type": entry_type,
                    "key": str(data.get("key") or "").strip(),
                    "payload": payload if isinstance(payload, dict) else {},
                }
            )
        return result, ""


class SelectSkillTool(Tool):
    def __init__(self, ctx: AgentTickContext, store: DriftStateStore) -> None:
        self._ctx = ctx
        self._store = store

    @property
    def name(self) -> str:
        return "select_skill"

    @property
    def description(self) -> str:
        return "声明本轮 Drift 选中的 skill，并返回该 skill 的 SKILL.md 内容和 local_context。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "本轮要执行的 drift skill 名称",
                },
            },
            "required": ["skill_name"],
        }

    async def execute(self, skill_name: str) -> str:
        name = str(skill_name or "").strip()
        if name not in self._store.valid_skill_names():
            return json.dumps({"error": f"unknown skill: {name}"}, ensure_ascii=False)
        selected = str(self._ctx.drift_selected_skill or "").strip()
        if selected and selected != name:
            return json.dumps(
                {"error": f"selected skill already fixed: {selected}"},
                ensure_ascii=False,
            )
        skill_dir = self._store.skill_dir_for(name)
        if skill_dir is None:
            return json.dumps({"error": f"skill not mounted: {name}"}, ensure_ascii=False)
        skill_file = skill_dir / "SKILL.md"
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        self._ctx.drift_selected_skill = name
        continuum = self._store.load_skill_continuum(name)
        journal_recent = self._store.load_skill_journal(name, limit=8)
        return json.dumps(
            {
                "ok": True,
                "skill": name,
                "content": content,
                "local_context": {
                    "run_count": int(continuum.get("run_count") or 0),
                    "last_status": _clip_text(continuum.get("last_status"), 40),
                    "last_run_at": _clip_text(continuum.get("last_run_at"), 80),
                    "updated_at": _clip_text(continuum.get("updated_at"), 80),
                    "last_briefing": _clip_text(continuum.get("last_briefing"), 500),
                    "scratchpad": _clip_text(continuum.get("scratchpad"), 2000),
                    "cursor": continuum.get("cursor") or {},
                    "journal_recent": journal_recent,
                },
            },
            ensure_ascii=False,
        )


class IdleDriftTool(Tool):
    def __init__(self, ctx: AgentTickContext, store: DriftStateStore) -> None:
        self._ctx = ctx
        self._store = store

    @property
    def name(self) -> str:
        return "idle_drift"

    @property
    def description(self) -> str:
        return "【终止工具】不选择 skill，静默结束本次 Drift；reason 必填。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "为什么本轮什么都不做，例如气氛不合适、没有足够价值或刚刚发过消息。",
                },
            },
            "required": ["reason"],
        }

    async def execute(self, reason: str = "") -> str:
        reason_text = str(reason or "").strip()
        if not reason_text:
            return json.dumps({"error": "reason is required"}, ensure_ascii=False)
        selected = str(self._ctx.drift_selected_skill or "").strip()
        if selected:
            return json.dumps(
                {"error": "idle_drift must be called before select_skill"},
                ensure_ascii=False,
            )

        self._ctx.drift_selected_skill = "idle"
        self._store.save_finish(
            skill_used="idle",
            status="completed",
            briefing=_clip_text(f"空闲不行动：{reason_text}", 500),
            message_result="silent",
            scratchpad_update=None,
            global_note_update=None,
            now_utc=self._ctx.now_utc,
        )
        self._ctx.drift_finished = True
        logger.info("[drift_tools] idle_drift ok reason=%s", reason_text[:120])
        return json.dumps({"ok": True}, ensure_ascii=False)


class MountServerTool(Tool):
    """挂载一个已连接的 MCP server，使其工具在本次 drift 中可用。"""

    def __init__(self, shared_tools: ToolRegistry, target_tools: ToolRegistry) -> None:
        self._shared = shared_tools
        self._target = target_tools

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

    async def execute(self, server: str) -> str:
        server = str(server or "").strip()
        if not server:
            return json.dumps({"error": "server is required"}, ensure_ascii=False)
        names = self._shared.get_tool_names_by_source("mcp", server)
        if not names:
            return json.dumps(
                {"error": f"MCP server '{server}' 不存在或未连接"},
                ensure_ascii=False,
            )
        new = names - self._target.get_registered_names()
        if not new:
            return json.dumps(
                {"ok": True, "message": f"'{server}' 已挂载，无新增工具", "tools": sorted(names)},
                ensure_ascii=False,
            )
        for name in sorted(new):
            tool = self._shared.get_tool(name)
            if tool is not None:
                self._target.register(
                    tool,
                    risk="external-side-effect",
                    source_type="mcp",
                    source_name=server,
                )
        logger.info("[drift_tools] mount_server ok: server=%s new=%s", server, sorted(new))
        return json.dumps(
            {"ok": True, "tools": sorted(names), "new": sorted(new)},
            ensure_ascii=False,
        )


class DriftRecallMemoryTool(Tool):
    def __init__(self, wrapped: Tool, ctx: AgentTickContext) -> None:
        self._wrapped = wrapped
        self._ctx = ctx

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def description(self) -> str:
        return self._wrapped.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._wrapped.parameters

    async def execute(self, **kwargs: Any) -> str | ToolResult:
        args = dict(kwargs)
        args.setdefault("current_timestamp", self._ctx.now_utc.isoformat())
        if ":" in self._ctx.session_key:
            channel, chat_id = self._ctx.session_key.split(":", 1)
            args.setdefault("channel", channel)
            args.setdefault("chat_id", chat_id)
        return await self._wrapped.execute(**args)


class DriftShellTool(Tool):
    def __init__(self, wrapped: Tool, drift_dir: Path) -> None:
        self._wrapped = wrapped
        self._drift_dir = drift_dir

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def description(self) -> str:
        return self._wrapped.description + "\nDrift 中默认工作目录是 drift 工作区。"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._wrapped.parameters

    async def execute(self, **kwargs: Any) -> str | ToolResult:
        args = dict(kwargs)
        raw_cwd = str(args.get("cwd") or "").strip()
        if raw_cwd:
            cwd = Path(raw_cwd).expanduser()
            if not cwd.is_absolute():
                cwd = self._drift_dir / cwd
        else:
            cwd = self._drift_dir
        args["cwd"] = str(cwd)
        return await self._wrapped.execute(**args)


class DriftPathResolver:
    def __init__(self, drift_dir: Path, store: DriftStateStore) -> None:
        self._drift_dir = drift_dir
        self._store = store

    def resolve(self, path: str) -> Path | None:
        raw = str(path or "").strip()
        if not raw:
            return None
        raw_path = Path(raw).expanduser()
        if raw_path.is_absolute():
            return raw_path
        parts = PurePosixPath(raw).parts
        if len(parts) >= 2 and parts[0] == "skills":
            skill_dir = self._store.skill_dir_for(parts[1])
            if skill_dir is not None:
                return skill_dir.joinpath(*parts[2:])
        return self._drift_dir / raw


class DriftReadFileTool(Tool):
    def __init__(self, resolver: DriftPathResolver) -> None:
        self._resolver = resolver
        self._reader = ReadFileTool()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return self._reader.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._reader.parameters

    async def execute(self, path: str, **kwargs: Any) -> Any:
        resolved = self._resolver.resolve(path)
        if resolved is None:
            return await self._reader.execute(path=path, **kwargs)
        return await self._reader.execute(path=str(resolved), **kwargs)


class DriftListDirTool(Tool):
    def __init__(self, resolver: DriftPathResolver) -> None:
        self._resolver = resolver
        self._lister = ListDirTool()

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return self._lister.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._lister.parameters

    async def execute(self, path: str, **kwargs: Any) -> Any:
        resolved = self._resolver.resolve(path)
        if resolved is None:
            return await self._lister.execute(path=path, **kwargs)
        return await self._lister.execute(path=str(resolved), **kwargs)


class DriftWriteFileTool(Tool):
    def __init__(self, resolver: DriftPathResolver, allowed_dir: Path) -> None:
        self._resolver = resolver
        self._writer = WriteFileTool(allowed_dir=allowed_dir)

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return self._writer.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._writer.parameters

    async def execute(self, path: str, content: str, **kwargs: Any) -> Any:
        resolved = self._resolver.resolve(path)
        if resolved is None:
            return await self._writer.execute(path=path, content=content, **kwargs)
        return await self._writer.execute(
            path=str(resolved),
            content=content,
            **kwargs,
        )


class DriftEditFileTool(Tool):
    def __init__(self, resolver: DriftPathResolver, allowed_dir: Path) -> None:
        self._resolver = resolver
        self._editor = EditFileTool(allowed_dir=allowed_dir)

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return self._editor.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._editor.parameters

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        **kwargs: Any,
    ) -> Any:
        resolved = self._resolver.resolve(path)
        if resolved is None:
            return await self._editor.execute(
                path=path,
                old_text=old_text,
                new_text=new_text,
                **kwargs,
            )
        return await self._editor.execute(
            path=str(resolved),
            old_text=old_text,
            new_text=new_text,
            **kwargs,
        )


def build_drift_tool_registry(
    *,
    ctx: AgentTickContext,
    deps: DriftToolDeps,
) -> ToolRegistry:
    tools = ToolRegistry()
    drift_dir = deps.drift_dir
    resolver = DriftPathResolver(drift_dir, deps.store)
    tools.register(SelectSkillTool(ctx, deps.store), risk="read-only")
    tools.register(IdleDriftTool(ctx, deps.store), risk="write")
    tools.register(
        DriftReadFileTool(resolver),
        risk="read-only",
    )
    tools.register(
        DriftListDirTool(resolver),
        risk="read-only",
    )
    write_allowed_dir = deps.workspace_dir or drift_dir
    tools.register(DriftWriteFileTool(resolver, write_allowed_dir), risk="write")
    tools.register(DriftEditFileTool(resolver, write_allowed_dir), risk="write")

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
            if name == "recall_memory":
                tool = DriftRecallMemoryTool(tool, ctx)
            elif name == "shell":
                tool = DriftShellTool(tool, drift_dir)
            risk = "external-side-effect" if name == "shell" else "read-only"
            tools.register(tool, risk=risk)

    # mount_server: 只有 shared registry 里有 MCP 工具时才注册
    if shared is not None and shared.get_mcp_server_names():
        tools.register(MountServerTool(shared, tools), risk="read-only")

    tools.register(
        SendMessageTool(ctx, deps.send_message_fn),
        risk="external-side-effect",
    )
    tools.register(FinishDriftTool(ctx, deps.store, deps.event_bus), risk="write")
    return tools
