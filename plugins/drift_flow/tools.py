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
from plugins.default_proactive.context import AgentTickContext
from plugins.drift_flow.state import DriftStateStore
from plugins.default_proactive.outbound_text import normalize_outbound_text

logger = logging.getLogger(__name__)

_DRIFT_DECISIONS = {"continue", "defer", "switch", "explore"}


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
    event_bus: Any = None


class SendMessageTool(Tool):
    def __init__(self, ctx: AgentTickContext) -> None:
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "message_push"

    @property
    def description(self) -> str:
        return (
            "向用户发送一条消息，可附带图片。单次 Drift run 最多只能调用一次。\n"
            "channel 和 chat_id 在 Drift 上下文中已由配置预设，可省略不填。\n"
            "这是 fire-and-forget：发送成功即完成本轮动作，不创建等待回复的状态。"
            "未来若出现用户回答，它会作为新的会话上下文和记忆自然进入；"
            "不得记录‘等用户回复’，也不得把‘没有回复’当成可观测事实。"
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
        if self._ctx.drift_message_staged:
            logger.info("[drift_tools] message_push rejected: already used")
            return json.dumps(
                {"error": "message_push already used in this drift run"},
                ensure_ascii=False,
            )
        if not text and not media_paths:
            logger.info("[drift_tools] message_push rejected: empty message and media")
            return json.dumps({"error": "message or media is required"}, ensure_ascii=False)
        self._ctx.draft_message = text
        self._ctx.draft_media = media_paths
        self._ctx.drift_message_staged = True
        logger.info("[drift_tools] message_push staged")
        return json.dumps(
            {
                "ok": True,
                "delivery_semantics": "completed_fire_and_forget",
                "reply_state": "not_tracked",
                "next": "finish_drift_without_waiting_for_user",
            },
            ensure_ascii=False,
        )

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
                    "enum": ["completed", "paused"],
                    "description": (
                        "completed 表示本轮主动行为已闭环，包含已行动、检查后无事可做、"
                        "或判断当前不合时宜后静默结束；"
                        "paused 表示本轮因工具、外部服务、步数上限或中间处理未完成而中断，"
                        "scratchpad_update 必须写清已经做到哪里、下次从哪里继续。"
                    ),
                },
                "briefing": {"type": "string", "description": "本轮做了什么的一句话摘要"},
                "scratchpad_update": {
                    "type": "string",
                    "description": (
                        "下次进入本 skill 时需要注入的自然语言前情，只记录系统自己的进度。"
                        "<example>已生成 plan.json，下次从执行计划继续。</example>"
                        "<example>不要：已向用户提问，等待用户回复。</example>"
                    ),
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
                "self_update": {
                    "type": "object",
                    "description": (
                        "收尾后的自我连续性更新，不保存执行断点或长期人格事实。"
                        "先对照本轮与 recent_drift_runs：有真实重复、反例或变化时写 observation，普通一轮则省略。"
                        "<example name=\"发现重复模式\">{\"pattern\":\"repeat\","
                        "\"reflection\":\"最近几轮反复选择了提问\","
                        "\"next_tendency\":\"下次看当时心情，不急着继续提问\","
                        "\"observation\":{\"statement\":\"我不确定时似乎容易选择向用户提问\","
                        "\"basis\":\"最近三轮有两轮都选择了提问\",\"effect\":\"question\"}}</example>"
                        "<example name=\"普通一轮无新发现\">{\"pattern\":\"ordinary\","
                        "\"reflection\":\"本轮只是一次普通闭环，没有形成新的重复或变化\",\"next_tendency\":"
                        "\"下次可能想安静整理点东西，也可以临时换别的活动\"}</example>"
                    ),
                    "properties": {
                        "current_intention": {
                            "type": "string",
                            "description": "如果本轮改变了原意图，写更新后的意图",
                        },
                        "reflection": {
                            "type": "string",
                            "description": (
                                "用一句话对照本轮与 recent_drift_runs：是在自由延续、主动变化，"
                                "还是可能只走了最省事的路径；只负责看见，不强迫换活动。"
                                "<example>最近三轮都在审计，因为它低打扰且工具可用；"
                                "这可能是当下自然节奏，也可能是我在选择最省事的现成活动。</example>"
                            ),
                        },
                        "pattern": {
                            "type": "string",
                            "enum": ["ordinary", "repeat", "change", "contradiction"],
                            "description": (
                                "本轮与近期行为的关系。ordinary=没有形成模式；repeat=重复出现；"
                                "change=主动换了做法；contradiction=出现反例。"
                                "选择 repeat/change/contradiction 时必须同时写 observation，"
                                "让看见的变化能够跨轮保留。"
                                "<example>连续三轮选择同类活动用 repeat；首次普通执行用 ordinary。</example>"
                            ),
                        },
                        "next_tendency": {
                            "type": "string",
                            "description": (
                                "下次空闲时可能想继续、搁置或探索的宽松倾向，不是下一轮必须执行的题目或步骤。"
                                "不能写等待用户回答、如果用户没回答就怎样；未来回答出现时会自然成为新上下文。"
                                "<example>可以：下次可能想做点安静、不打扰用户的整理，也可以看当时心情换别的。</example>"
                                "<example>不要：下次问用户最近单曲循环哪首歌。</example>"
                            ),
                        },
                        "observation": {
                            "type": "object",
                            "description": (
                                "仅当本轮与近期多轮的实际选择形成了重复、反例或变化时，"
                                "留下可被后续 Drift 质疑或修正的暂定观察；没有则省略。"
                                "<example>{\"statement\":\"我不确定时似乎容易连续选择向用户提问\","
                                "\"basis\":\"最近三轮都选择了提问\",\"effect\":\"question\"}</example>"
                            ),
                            "properties": {
                                "statement": {
                                    "type": "string",
                                    "description": (
                                        "关于自己在 Drift 中如何选择或行动的暂定观察，避免写成稳定人格结论。"
                                        "<example>我似乎会在没有明确念头时选择最容易执行的活动。</example>"
                                    ),
                                },
                                "basis": {
                                    "type": "string",
                                    "description": (
                                        "本轮以及可见近期 runs 支持这条观察的具体行为证据。"
                                        "<example>四轮里三次选择同一 skill，理由都直接沿用了上轮 next_tendency。</example>"
                                    ),
                                },
                                "effect": {
                                    "type": "string",
                                    "enum": ["question", "reinforce", "revise"],
                                    "description": (
                                        "question=首次提出暂定观察；reinforce=后来再次出现同类证据；"
                                        "revise=出现反例或情境变化。"
                                        "<example>先写 question；后续确实再次发生才写 reinforce；主动换了做法可写 revise。</example>"
                                    ),
                                },
                            },
                            "required": ["statement", "basis", "effect"],
                        },
                    },
                    "required": ["next_tendency", "reflection", "pattern"],
                },
            },
            "required": ["skill_used", "status", "briefing", "self_update"],
        }

    async def execute(
        self,
        skill_used: str,
        status: str = "",
        briefing: str = "",
        scratchpad_update: str | None = None,
        cursor_update: dict[str, Any] | None = None,
        journal_append: list[dict[str, Any]] | dict[str, Any] | None = None,
        global_note_update: str | None = None,
        self_update: dict[str, Any] | None = None,
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
        if status_value not in {"completed", "paused"}:
            return json.dumps(
                {"error": "status must be one of: completed, paused"},
                ensure_ascii=False,
            )
        summary = str(briefing or "").strip()
        if not summary:
            return json.dumps({"error": "briefing is required"}, ensure_ascii=False)
        scratchpad_text = str(scratchpad_update or "").strip()
        if status_value == "paused" and not scratchpad_text:
            return json.dumps(
                {
                    "error": (
                        "scratchpad_update is required when "
                        "status is paused"
                    )
                },
                ensure_ascii=False,
            )
        message_result_value = "staged" if self._ctx.drift_message_staged else "silent"
        if cursor_update is not None and not isinstance(cursor_update, dict):
            return json.dumps(
                {"error": "cursor_update must be an object"},
                ensure_ascii=False,
            )
        journal_entries, journal_error = self._normalize_journal_append(journal_append)
        if journal_error:
            return json.dumps({"error": journal_error}, ensure_ascii=False)
        if not isinstance(self_update, dict):
            return json.dumps({"error": "self_update must be an object"}, ensure_ascii=False)
        next_tendency = str(self_update.get("next_tendency") or "").strip()
        if not next_tendency:
            return json.dumps(
                {"error": "self_update.next_tendency is required"},
                ensure_ascii=False,
            )
        reflection = str(self_update.get("reflection") or "").strip()
        if not reflection:
            return json.dumps(
                {"error": "self_update.reflection is required"},
                ensure_ascii=False,
            )
        pattern = str(self_update.get("pattern") or "").strip()
        if pattern not in {"ordinary", "repeat", "change", "contradiction"}:
            return json.dumps(
                {"error": "self_update.pattern must be one of: ordinary, repeat, change, contradiction"},
                ensure_ascii=False,
            )
        normalized_self_update = {
            "current_intention": str(self_update.get("current_intention") or "").strip(),
            "next_tendency": next_tendency,
        }
        observation, observation_error = self._normalize_self_observation(
            self_update.get("observation")
        )
        if observation_error:
            return json.dumps({"error": observation_error}, ensure_ascii=False)
        if pattern != "ordinary" and observation is None:
            return json.dumps(
                {"error": f"self_update.observation is required when pattern is {pattern}"},
                ensure_ascii=False,
            )
        if observation is not None:
            journal_entries.append(
                {
                    "entry_type": "self_observation",
                    "key": observation["effect"],
                    "payload": observation,
                }
            )
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
            self_update=normalized_self_update,
        )
        self._ctx.drift_finished = True
        self._ctx.drift_finish_status = status_value
        self._ctx.drift_finish_briefing = summary
        if self._event_bus is not None and not self._ctx.drift_message_staged:
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
    def _normalize_self_observation(raw: Any) -> tuple[dict[str, str] | None, str]:
        if raw is None:
            return None, ""
        if not isinstance(raw, dict):
            return None, "self_update.observation must be an object"
        effect = str(raw.get("effect") or "").strip()
        if effect not in {"question", "reinforce", "revise"}:
            return None, "self_update.observation.effect must be one of: question, reinforce, revise"
        statement = str(raw.get("statement") or "").strip()
        basis = str(raw.get("basis") or "").strip()
        if not statement:
            return None, "self_update.observation.statement is required"
        if not basis:
            return None, "self_update.observation.basis is required"
        return {
            "statement": _clip_text(statement, 500),
            "basis": _clip_text(basis, 500),
            "effect": effect,
        }, ""

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
        return "声明本轮 Drift 的意图与选择，并返回所选 skill 的说明和 local_context。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "本轮要执行的 drift skill 名称",
                },
                "decision": {
                    "type": "string",
                    "enum": sorted(_DRIFT_DECISIONS),
                    "description": "本轮与既有意图的关系：继续、延后、切换或自由探索",
                },
                "intention": {
                    "type": "string",
                    "description": (
                        "这轮此刻真正想做的一件小事，不照抄上轮 next_tendency。"
                        "<example>翻一小段旧记录，看看有没有值得继续发展的兴趣。</example>"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "结合当前状态、近期 runs 和已有 skill 覆盖范围，说明为什么此刻这样选择；"
                        "可以延续上轮倾向，但必须是现在仍然想做，而不是因为它写在那里。"
                        "<example>最近三轮都在主动提问，这轮不再复制上轮建议，"
                        "改选一个安静的活动。</example>"
                        "<example>现有 skill 都偏向提问或维护，但我此刻想反复做一种尚未被覆盖的小活动，"
                        "因此选择候选中的创建元能力，先把它设计成以后可自由选择的 skill。</example>"
                        "<example>不要只写：上轮说下次问音乐，所以这轮问音乐。</example>"
                    ),
                },
            },
            "required": ["skill_name", "decision", "intention", "reason"],
        }

    async def execute(
        self,
        skill_name: str,
        decision: str = "",
        intention: str = "",
        reason: str = "",
    ) -> str:
        name = str(skill_name or "").strip()
        decision_value = str(decision or "").strip()
        intention_text = str(intention or "").strip()
        reason_text = str(reason or "").strip()
        if name not in self._store.valid_skill_names():
            return json.dumps({"error": f"unknown skill: {name}"}, ensure_ascii=False)
        if decision_value not in _DRIFT_DECISIONS:
            return json.dumps(
                {"error": "decision must be one of: continue, defer, switch, explore"},
                ensure_ascii=False,
            )
        if not intention_text:
            return json.dumps({"error": "intention is required"}, ensure_ascii=False)
        if not reason_text:
            return json.dumps({"error": "reason is required"}, ensure_ascii=False)
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
        self._store.save_self_choice(
            skill_name=name,
            intention=intention_text,
            decision=decision_value,
            reason=reason_text,
            now_utc=self._ctx.now_utc,
        )
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
                "runtime_guidance": (
                    "这是 paused skill 的可续接停点。SKILL.md 是完整能力说明书，不是本轮从头执行清单。"
                    "先用 local_context 区分已完成与未完成步骤；如果继续，只执行停点后的最小下一步。"
                    "不要仅为遵循完整流程而重复读取、查重、规划或重建已有产物。"
                    if str(continuum.get("last_status") or "") == "paused"
                    else "本 skill 上轮已闭环；根据当前目标选择本轮实际需要的说明书部分。"
                ),
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
        return "【例外终止工具】仅在近期气氛、频率或风险明确不合适时，不选择 skill 并静默结束；reason 必填。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "具体时机或风险原因，例如刚主动发过消息、丧亲/疾病/强压力语境、当前行动会明显低价值重复。",
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
        self._store.save_self_choice(
            skill_name="idle",
            intention="本轮暂时不行动",
            decision="rest",
            reason=reason_text,
            now_utc=self._ctx.now_utc,
        )
        self._store.save_finish(
            skill_used="idle",
            status="completed",
            briefing=_clip_text(f"空闲不行动：{reason_text}", 500),
            message_result="silent",
            scratchpad_update=None,
            global_note_update=None,
            now_utc=self._ctx.now_utc,
            self_update={"next_tendency": "等待更合适的时机再自由选择"},
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
    tools = ToolRegistry(follow_runtime_snapshot=False)
    drift_dir = deps.drift_dir
    resolver = DriftPathResolver(drift_dir, deps.store)
    tools.register(SelectSkillTool(ctx, deps.store), risk="write")
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
        SendMessageTool(ctx),
        risk="external-side-effect",
    )
    tools.register(FinishDriftTool(ctx, deps.store, deps.event_bus), risk="write")
    return tools
