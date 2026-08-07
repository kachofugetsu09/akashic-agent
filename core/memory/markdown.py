from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from agent.llm_json import load_json_object_loose
from agent.memory import MemoryStore
from agent.prompting import is_context_frame
from agent.provider import LLMProvider
from bus.events_lifecycle import TurnCommitted
from core.memory.events import ConsolidationCommitted
from infra.persistence.json_store import atomic_write_text
from session.manager import (
    logical_history_tail_start,
    logical_history_unit_count,
    logical_history_unit_ranges,
)
from session.memory_policy import excludes_memory

if TYPE_CHECKING:
    from bus.event_bus import EventBus

logger = logging.getLogger("memory.markdown")

_EVENT_EXTRACTION_TIMEOUT_S = 300.0
_RECENT_CONTEXT_TIMEOUT_S = 180.0


@dataclass(frozen=True)
class ConsolidateRequest:
    session: object
    archive_all: bool = False
    force: bool = False
    drain_backlog: bool = True
    scope_channel: str = ""
    scope_chat_id: str = ""


@dataclass
class ConsolidateResult:
    consolidated_count: int = 0
    trace: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RefreshRecentTurnsRequest:
    session: object


@dataclass(frozen=True)
class MemoryLifecycleBindRequest:
    get_session: Callable[[str], object]
    save_session: Callable[[object], Awaitable[None]]


@runtime_checkable
class MemoryProfileApi(Protocol):
    def read_long_term(self) -> str: ...

    def write_long_term(self, content: str) -> None: ...

    def read_self(self) -> str: ...

    def write_self(self, content: str) -> None: ...

    def read_recent_context(self) -> str: ...

    def write_recent_context(self, content: str) -> None: ...

    def backup_long_term(self, backup_name: str = "MEMORY.bak.md") -> None: ...

    def backup_self(self, backup_name: str = "SELF.bak.md") -> None: ...

    def get_memory_context(self) -> str: ...

    def has_long_term_memory(self) -> bool: ...


_ALLOWED_PENDING_TAGS = frozenset(
    {
        "identity",
        "preference",
        "key_info",
        "health_long_term",
        "requested_memory",
        "correction",
        "agent_context",
    }
)


class _ConsolidationPayloadError(ValueError):
    """模型返回结构不符合 consolidation 合约。"""


def _format_pending_items(raw_items: object) -> str:
    """校验并整理模型输出为 PENDING.md 接受的 Markdown 列表。"""
    if not isinstance(raw_items, list):
        raise _ConsolidationPayloadError("pending_items must be an array")

    lines: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            raise _ConsolidationPayloadError("pending_items entries must be objects")
        raw_tag = item.get("tag", "")
        raw_content = item.get("content", "")
        if not isinstance(raw_tag, str) or not isinstance(raw_content, str):
            raise _ConsolidationPayloadError(
                "pending_items tag and content must be strings"
            )
        tag = raw_tag.strip().lower()
        content = raw_content.strip()
        if tag not in _ALLOWED_PENDING_TAGS or not content:
            continue
        line = f"- [{tag}] {content}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def _parse_consolidation_payload(text: str) -> dict[str, object] | None:
    result = load_json_object_loose(text)
    return cast(dict[str, object] | None, result)


def _format_consolidation_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


@dataclass(frozen=True)
class _ConsolidationWindow:
    old_messages: list[dict]
    keep_count: int
    consolidate_up_to: int


@dataclass(frozen=True)
class _ConsolidationDraft:
    window: _ConsolidationWindow
    source_ref: str
    history_entry_payloads: list[tuple[str, int]]
    pending_items: str
    conversation: str
    recent_context_text: str
    scope_channel: str
    scope_chat_id: str
    archive_all: bool = False


@dataclass(frozen=True)
class _ConsolidationFailure:
    step: str
    error: str
    elapsed_ms: int = 0


def _resolve_consolidation_cursor(
    messages: list[dict],
    last_consolidated: int,
    *,
    force: bool,
) -> tuple[list[tuple[int, int]], int]:
    """校验 consolidation 游标，并在显式 force 时返回最近合法边界。"""
    unit_ranges = logical_history_unit_ranges(messages)
    boundaries = {0, *(end for _, end in unit_ranges)}
    if last_consolidated in boundaries:
        return unit_ranges, last_consolidated
    if not force:
        raise ValueError(
            "last_consolidated 落在逻辑历史单元内部: " f"{last_consolidated}"
        )
    return unit_ranges, max(
        boundary for boundary in boundaries if boundary <= last_consolidated
    )


def _select_consolidation_window(
    session,
    *,
    keep_count: int,
    consolidation_min_new_messages: int,
    archive_all: bool,
    force: bool = False,
    continuing: bool = False,
) -> _ConsolidationWindow | None:
    total_messages = len(session.messages)
    if archive_all:
        return _ConsolidationWindow(
            old_messages=list(session.messages),
            keep_count=0,
            consolidate_up_to=total_messages,
        )

    # 1. 游标只能停在完整逻辑单元之后；新写入不再制造半 turn 游标。
    unit_ranges, consolidation_start = _resolve_consolidation_cursor(
        session.messages,
        session.last_consolidated,
        force=force,
    )

    if total_messages - consolidation_start <= 0:
        return None

    if force:
        consolidate_up_to = total_messages
    else:
        if len(unit_ranges) <= keep_count:
            return None
        consolidate_up_to = unit_ranges[-keep_count][0]
    old_messages = session.messages[consolidation_start:consolidate_up_to]
    if not old_messages:
        return None
    new_unit_count = logical_history_unit_count(
        session.messages,
        start_index=consolidation_start,
    ) - (0 if force else keep_count)
    if (
        not force
        and not continuing
        and new_unit_count < max(1, int(consolidation_min_new_messages))
    ):
        return None
    return _ConsolidationWindow(
        old_messages=old_messages,
        keep_count=0 if force else keep_count,
        consolidate_up_to=consolidate_up_to,
    )


def _build_consolidation_source_ref(window: _ConsolidationWindow) -> str:
    """返回本次 consolidation 窗口内所有消息 ID 的 JSON 列表。
    缺失 id 的消息（迁移前的历史脏数据）直接跳过。
    """
    ids = [
        str(msg["id"])
        for msg in window.old_messages
        if msg.get("id") and not _is_context_frame_message(msg)
    ]
    return json.dumps(ids, ensure_ascii=False)


def _build_entry_source_ref(base_source_ref: str, entry: str) -> str:
    """为单条 history_entry 生成稳定子键，避免同窗口多条写入互相覆盖。"""
    text = (entry or "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else "empty"
    return f"{base_source_ref}#h:{digest}"


def _format_conversation_for_consolidation(old_messages: list[dict]) -> str:
    lines = []
    for message in old_messages:
        if _is_context_frame_message(message):
            continue
        if not message.get("content") or message.get("role") == "tool":
            continue
        if message.get("role") == "assistant" and message.get("proactive"):
            continue
        role = str(message.get("role", "")).upper()
        ts = str(message.get("timestamp", "?"))[:16]
        lines.append(f"[{ts}] {role}: {message['content']}")
    return "\n".join(lines)


def _limit_consolidation_window(
    window: _ConsolidationWindow,
    *,
    max_conversation_chars: int,
) -> _ConsolidationWindow:
    """按完整语义回合限制单页正文，并返回稳定的绝对游标。"""

    # 1. 与 prompt history 共用同一分组，显式多 U turn 与 proactive 都保持原子。
    groups = [
        window.old_messages[start:end]
        for start, end in logical_history_unit_ranges(window.old_messages)
    ]

    # 2. 再按实际送给模型的文本长度装页；单个超长回合保持完整并 fail-loud。
    selected: list[dict] = []
    used_chars = 0
    for group in groups:
        rendered = _format_conversation_for_consolidation(group)
        added_chars = len(rendered) + (1 if rendered and selected else 0)
        if selected and used_chars + added_chars > max_conversation_chars:
            break
        selected.extend(group)
        used_chars += added_chars
    if len(selected) == len(window.old_messages):
        return window
    return _ConsolidationWindow(
        old_messages=selected,
        keep_count=window.keep_count,
        consolidate_up_to=(
            window.consolidate_up_to - len(window.old_messages) + len(selected)
        ),
    )


def _replace_consolidation_prompt_conversation(
    prompt: str,
    *,
    old_conversation: str,
    new_conversation: str,
) -> str:
    """只替换事件提取 prompt 尾部的待处理对话。"""

    marker = "\n## 待处理对话\n"
    suffix = "\n\n只返回合法 JSON，不要 markdown 代码块。"
    prefix, separator, tail = prompt.rpartition(marker)
    if not separator or tail != old_conversation + suffix:
        raise RuntimeError("consolidation prompt 尾部结构已改变")
    return prefix + marker + new_conversation + suffix


def _clip_context_text(text: str, max_chars: int = 16000) -> str:
    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


def _coerce_emotional_weight(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
        raise _ConsolidationPayloadError(
            "history entry emotional_weight must be an integer from 0 to 10"
        )
    return value


def _normalize_history_entries(
    raw_entries: object,
    fallback_entry: object = None,
) -> list[tuple[str, int]]:
    """校验并整理模型输出的 history 条目，保留合法条目的原有规则。"""
    # 1. 收集当前数组格式和旧版单条格式的候选条目。
    entries: list[tuple[str, int]] = []
    seen: set[str] = set()
    candidates: list[object] = []
    if isinstance(raw_entries, list):
        candidates.extend(raw_entries)
    elif raw_entries is not None:
        if not isinstance(raw_entries, str | dict):
            raise _ConsolidationPayloadError(
                "history_entries must be an array or object"
            )
        candidates.append(raw_entries)
    if fallback_entry is not None and not isinstance(raw_entries, list):
        candidates.append(fallback_entry)
    # 2. 校验字段类型，保留合法条目并按摘要去重。
    for item in candidates:
        if isinstance(item, str):
            summary = item.strip()
            emotional_weight = 0
        elif isinstance(item, dict):
            raw_summary = item.get("summary", "")
            if not isinstance(raw_summary, str):
                raise _ConsolidationPayloadError(
                    "history entry summary must be a string"
                )
            summary = raw_summary.strip()
            emotional_weight = _coerce_emotional_weight(item.get("emotional_weight"))
        else:
            raise _ConsolidationPayloadError(
                "history_entries entries must be strings or objects"
            )
        if not summary or summary in seen:
            continue
        seen.add(summary)
        entries.append((summary, emotional_weight))
    return entries


def _recent_turn_count(keep_count: int) -> int:
    """把旧的消息预览预算换算为不切分的逻辑历史单元数。"""
    return max(1, keep_count // 4)


def _message_time(message: dict) -> str:
    return str(message.get("timestamp") or "").strip()


def _is_context_frame_message(message: dict) -> bool:
    content = str(message.get("content") or "")
    return is_context_frame(content)


def _format_recent_context_messages(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        if _is_context_frame_message(message):
            continue
        content = str(message.get("content") or "").strip()
        role = str(message.get("role") or "").lower()
        if not content or role not in {"user", "assistant"}:
            continue
        if role == "assistant" and message.get("proactive"):
            continue
        if role == "assistant":
            preview = content[:60]
            if preview:
                lines.append(f"[a-preview] {preview}")
            continue
        lines.append(f"[user] {content}")
    return "\n".join(lines).strip()


def _replace_recent_turns_block(existing_text: str, recent_turns: str) -> str:
    block_lines = [
        "## Recent Turns",
        "<!-- a-preview = assistant reply preview only -->",
        recent_turns.strip() or "- none",
    ]
    block = "\n".join(block_lines).rstrip() + "\n"
    marker = "\n## Recent Turns\n"
    text = (existing_text or "").strip()
    if marker in text:
        prefix, _ = text.split(marker, 1)
        return prefix.rstrip() + "\n\n" + block
    if text:
        return text + "\n\n" + block
    return _render_recent_context(
        compression=None,
        compression_until="none",
        recent_turns=recent_turns,
    )


def _format_conversation_for_recent_context(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        if _is_context_frame_message(message):
            continue
        content = str(message.get("content") or "").strip()
        role = str(message.get("role") or "").upper()
        if not content or role not in {"USER", "ASSISTANT"}:
            continue
        if role == "ASSISTANT" and message.get("proactive"):
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _render_recent_context(
    *,
    compression: dict[str, list[str]] | None,
    compression_until: str,
    recent_turns: str,
) -> str:
    compression = {} if compression is None else compression
    ongoing_threads = [
        item.strip() for item in compression.get("ongoing_threads", []) if item.strip()
    ]
    sections = [
        ("最近持续关注", compression.get("active_topics", [])),
        ("最近明确偏好", compression.get("user_preferences", [])),
        ("最近待延续话题", compression.get("follow_ups", [])),
        ("最近避免事项", compression.get("avoidances", [])),
    ]
    lines = [
        "# Recent Context",
        "",
        "## Compression",
        f"until: {compression_until or 'none'}",
    ]
    rendered_any = False
    for title, items in sections:
        cleaned = [item.strip() for item in items if item.strip()]
        if not cleaned:
            continue
        rendered_any = True
        lines.append(f"- {title}：{'；'.join(cleaned[:3])}")
    if not rendered_any:
        lines.append("- none")
    lines.extend(["", "## Ongoing Threads"])
    if ongoing_threads:
        for item in ongoing_threads[:3]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(
        ["", "## Recent Turns", "<!-- a-preview = assistant reply preview only -->"]
    )
    if recent_turns.strip():
        lines.append(recent_turns.strip())
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


class _MarkdownConsolidationWorker:
    def __init__(
        self,
        *,
        profile_maint: "MarkdownMemoryStore",
        provider: "LLMProvider",
        model: str,
        keep_count: int,
        consolidation_input_budget: int | None = None,
        provider_system_prompt: str = "",
        recent_context_provider: "LLMProvider | None" = None,
        recent_context_model: str | None = None,
    ) -> None:
        self._profile_maint = profile_maint
        self._provider = provider
        self._model = model
        self._recent_context_provider = recent_context_provider or provider
        self._recent_context_model = str(recent_context_model or "").strip() or model
        self._keep_count = keep_count
        self._consolidation_input_budget = consolidation_input_budget
        self._provider_system_prompt = provider_system_prompt
        self._consolidation_min_new_messages = max(5, keep_count // 2)

    @staticmethod
    def _build_recent_context_prompt(
        *,
        old_recent_context: str,
        conversation: str,
        recent_turns: str,
    ) -> str:
        return f"""你是近期语境压缩代理。你的任务不是自由总结，而是为后续 proactive 和 drift 保守地抽取近期语境。

目标：
1. 提取用户最近持续关注的话题
2. 提取最近新暴露、但尚未沉淀为长期记忆的显式偏好
3. 提取最近适合自然续接的话题
4. 提取最近应避免打扰、应避免推荐、或明显不想聊的方向
5. 提取跨窗口持续存在的重要现实线索（ongoing_threads）

规则：
- 只允许依据 USER 明确表达过的内容输出；ASSISTANT 的建议、解释、命名、延伸，一律不得当作证据
- recent_topics 可以总结“用户最近在讨论什么”，但必须贴近 USER 原话，不得升级成长期偏好
- active_topics 和 follow_ups 要优先写“话题层级”的概括，不要写 JSON Schema、函数名、字段名、具体术语翻译这类实现细节，除非用户明确把该细节当作核心关注点反复强调
- user_preferences 只允许在 USER 出现明确偏好/要求/禁忌表达时输出，例如：喜欢、偏好、希望、别、不要、避免、不想
- 不要把技术方案讨论、架构设想、问题求证、头脑风暴自动写成“用户偏好”
- 对技术讨论场景，只有当 USER 明确表达“以后都这样做 / 我就是偏好这种方式 / 我不要另一种方式 / 以后统一按这个来”时，才允许写 user_preferences；否则一律视为 active_topics 或 follow_ups
- 用户用“为什么不……”“能不能……”“是不是可以……”“只要不是最后一轮就……”这类方式提出方案设想或追问时，默认视为设计提议，不视为稳定偏好
- avoidances 只允许在 USER 明确表达“不要/别/避免/不想”时输出；没有明确否定表达就留空
- 如果最新 recent turns 显示话题已经明显切换，不要把较早窗口的技术讨论升级成当前偏好或避免事项
- 只保留未来几轮仍会影响主动行为的信息
- 不要记录工具细节、推理过程、普通寒暄
- 每个字段最多 3 条，每条尽量 1 句
- 没有把握就留空；宁可漏掉，也不要脑补

ongoing_threads 严格限制：
- 只记录用户正在经历、推进或承受的重要事情
- 必须是对用户当前生活、情绪、工作、学习、关系或健康有持续影响的线索
- 普通提问、技术讨论、方案脑暴、一次性 ask、知识求证，一律不得写入 ongoing_threads
- 若旧的 ongoing_threads 中已有某条重要线索，而当前窗口没有明确终结它，默认保留
- 只有当用户明确表示这件事已解决、结束、过去了、不再关心，才允许删除
- ongoing_threads 的写入门槛高于 active_topics；宁可少写，也不要把普通话题升级进去

专项禁令：
- 用户讨论“某个设计有没有依据/有没有实践/是否可行/为什么不这样做”，这是方案讨论，不是偏好；默认只能进入 active_topics 或 follow_ups，不能进入 user_preferences
- 用户说“为什么不让前台……只要不是最后一轮就……”是在提出一种实现设想，不等于“用户偏好以后统一这样做”
- 用户说“这样也不会引入额外延迟”“有没有这样的设计”，这是在分析方案目标，不等于稳定偏好
- 用户讨论“零延迟”“预加载”“流式预取”“前瞻性检索”这类设计目标时，默认视为当前方案讨论，不得直接提炼成 user_preferences
- 对方案讨论里的具体实现细节，优先上收一层概括，例如写“下一轮检索规划”“流式预取方案”，不要写“JSON Schema”“结构化预取指令”这类细碎实现点
- 用户说“睡觉了”“头有点疼”“身体不适”，这只是当前状态；除非用户明确说“别再聊这个”“不要继续”“我不想讨论”，否则不得生成 avoidances
- assistant 说“今晚先别想架构和代码了”“先休息”，这是 assistant 建议，不是用户 avoidances
- 如果较早窗口是技术方案讨论，而最新 recent turns 已切到睡眠/头痛/身体状态，则 user_preferences 和 avoidances 默认应为空；技术方案最多保留在 active_topics / follow_ups
- “最近在讨论前瞻性检索/流式预取方案”只能进入 active_topics / follow_ups，不能进入 ongoing_threads
- “用户最近几天反复因面试失败而情绪低落”“用户近期持续受睡眠紊乱影响”这类重要现实线索，才允许进入 ongoing_threads

反例：
- 错误：把“在 React 过程中同时输出下一轮检索内容”写成“用户偏好在对话中实时生成下一轮检索指令”
- 错误：把“这样也不会引入额外延迟”写成“用户偏好零延迟预加载”
- 错误：把“为什么不让前台在进行时同时输出自己想要什么”写成“用户偏好实时生成下一轮检索指令”
- 错误：把“睡觉了，吃了褪黑素头有点疼”写成“避免在身体不适时继续讨论技术架构”
- 错误：把“最近在讨论 React / 流式预取方案”写成 ongoing_threads
- 正确：active_topics 可写“用户最近在讨论前瞻性检索/流式预取方案”
- 正确：ongoing_threads 可写“用户最近几天反复提到面试受挫，持续影响情绪”
- 正确：如果用户没有明确说“希望/不要/避免/不想”，user_preferences 和 avoidances 可以为空

输出前自检：
1. 检查 user_preferences 中每一条，是否都能在 USER 原话里找到明确偏好/要求词（如“希望/不要/避免/不想/偏好/喜欢”）
2. 若找不到明确偏好/要求词，删除该条
3. 检查 avoidances 中每一条，是否都能在 USER 原话里找到明确否定/回避表达
4. 若找不到明确否定/回避表达，删除该条
5. 如果删除后为空，返回空数组，不要为了“信息完整”硬填

【上一版 recent context（仅供延续，不要机械复述）】
{old_recent_context or "（空）"}

【较早窗口（本次待压缩）】
{conversation or "（空）"}

【最新 recent turns（只用于判断是否已切话题，不可把 assistant 内容当证据）】
{recent_turns or "（空）"}

返回 JSON：
{{
  "active_topics": [],
  "user_preferences": [],
  "follow_ups": [],
  "avoidances": [],
  "ongoing_threads": []
}}
"""

    @staticmethod
    def _extract_recent_context_compression(text: str) -> dict[str, list[str]] | None:
        if not text.strip():
            return None
        section_match = re.search(
            r"## Compression\n(?P<body>.*?)(?:\n## Ongoing Threads\n|\Z)",
            text,
            flags=re.S,
        )
        if not section_match:
            return None
        body = section_match.group("body")
        parsed: dict[str, list[str]] = {
            "active_topics": [],
            "user_preferences": [],
            "follow_ups": [],
            "avoidances": [],
            "ongoing_threads": [],
        }
        title_map = {
            "最近持续关注": "active_topics",
            "最近明确偏好": "user_preferences",
            "最近待延续话题": "follow_ups",
            "最近避免事项": "avoidances",
        }
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("until:") or line == "- none":
                continue
            if not line.startswith("- "):
                continue
            payload = line[2:]
            if "：" not in payload:
                continue
            title, value = payload.split("：", 1)
            key = title_map.get(title.strip())
            if key is None:
                continue
            items = [part.strip() for part in value.split("；") if part.strip()]
            parsed[key] = items[:3]
        ongoing_match = re.search(
            r"## Ongoing Threads\n(?P<body>.*?)(?:\n## Recent Turns\n|\Z)",
            text,
            flags=re.S,
        )
        if ongoing_match:
            ongoing_items = []
            for raw_line in ongoing_match.group("body").splitlines():
                line = raw_line.strip()
                if line.startswith("- "):
                    item = line[2:].strip()
                    if item and item != "none":
                        ongoing_items.append(item)
            parsed["ongoing_threads"] = ongoing_items[:3]
        return parsed

    @staticmethod
    def _normalize_recent_context_compression(
        payload: dict[str, object],
    ) -> dict[str, list[str]]:
        """校验近期语境字段，避免把错误 JSON 当作字符序列写入记忆。"""
        # 1. 逐字段确认模型返回数组或明确缺省。
        fields = (
            "active_topics",
            "user_preferences",
            "follow_ups",
            "avoidances",
            "ongoing_threads",
        )
        compression: dict[str, list[str]] = {}
        # 2. 校验数组元素并保留原有最多三条限制。
        for key in fields:
            if key not in payload:
                compression[key] = []
                continue
            raw_items = payload[key]
            if not isinstance(raw_items, list):
                raise _ConsolidationPayloadError(
                    f"recent context {key} must be an array"
                )
            items: list[str] = []
            for item in raw_items:
                if not isinstance(item, str):
                    raise _ConsolidationPayloadError(
                        f"recent context {key} entries must be strings"
                    )
                value = item.strip()
                if value:
                    items.append(value)
            compression[key] = items[:3]
        return compression

    async def _call_llm_step(
        self,
        *,
        step: str,
        provider: "LLMProvider",
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_s: float,
    ) -> tuple[str, int] | _ConsolidationFailure:
        started_at = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                provider.chat(
                    messages=messages,
                    tools=[],
                    model=model,
                    max_tokens=max_tokens,
                    disable_thinking=True,
                ),
                timeout=timeout_s,
            )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            error = _format_consolidation_error(e)
            logger.error(
                "Memory consolidation llm step failed: step=%s elapsed_ms=%d error=%s",
                step,
                elapsed_ms,
                error,
            )
            return _ConsolidationFailure(step=step, error=error, elapsed_ms=elapsed_ms)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        content = response.content
        return (content.strip() if content is not None else ""), elapsed_ms

    async def _build_recent_context_snapshot(
        self,
        *,
        session,
        window: _ConsolidationWindow | None,
        archive_all: bool,
        old_recent_context: str,
    ) -> str | _ConsolidationFailure:
        """读取会话窗口并生成近期语境快照。"""
        # 1. 复用调用方已读取的 recent context，保证两个模型步骤看到同一版本。
        session_messages = list(session.messages)
        recent_count = _recent_turn_count(self._keep_count)
        recent_start = logical_history_tail_start(session_messages, recent_count)
        recent_turns = session_messages[recent_start:]
        if archive_all:
            compact_source = session_messages[:recent_start]
        else:
            compact_source = list(window.old_messages) if window is not None else []
        compression_until = _message_time(compact_source[-1]) if compact_source else ""
        rendered_recent_turns = _format_recent_context_messages(recent_turns)
        recent_turns_for_prompt = _format_conversation_for_recent_context(recent_turns)
        conversation = _format_conversation_for_recent_context(compact_source)
        # 2. 只有存在待压缩对话时才调用近期语境模型。
        compression: dict[str, list[str]] | None = None
        if conversation:
            prompt = self._build_recent_context_prompt(
                old_recent_context=old_recent_context,
                conversation=conversation,
                recent_turns=recent_turns_for_prompt,
            )
            call_result = await self._call_llm_step(
                step="recent_context",
                provider=self._recent_context_provider,
                model=self._recent_context_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是近期语境压缩代理，只返回合法 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                timeout_s=_RECENT_CONTEXT_TIMEOUT_S,
            )
            if isinstance(call_result, _ConsolidationFailure):
                return call_result
            text, elapsed_ms = call_result
            logger.info(
                "Memory consolidation recent_context raw: elapsed_ms=%d chars=%d preview=%r",
                elapsed_ms,
                len(text),
                text[:300],
            )
            if not text:
                return _ConsolidationFailure(
                    step="recent_context",
                    error="empty_response",
                    elapsed_ms=elapsed_ms,
                )
            parsed = _parse_consolidation_payload(text)
            if isinstance(parsed, dict):
                try:
                    compression = self._normalize_recent_context_compression(parsed)
                except _ConsolidationPayloadError as exc:
                    return _ConsolidationFailure(
                        step="recent_context",
                        error=f"invalid_schema: {exc}",
                        elapsed_ms=elapsed_ms,
                    )
            else:
                return _ConsolidationFailure(
                    step="recent_context",
                    error="invalid_json",
                    elapsed_ms=elapsed_ms,
                )
        elif old_recent_context.strip():
            compression = self._extract_recent_context_compression(old_recent_context)
        return _render_recent_context(
            compression=compression,
            compression_until=(
                compression_until
                or (
                    match.group(1).strip()
                    if old_recent_context.strip()
                    and (
                        match := re.search(
                            r"^until:\s*(.+)$", old_recent_context, flags=re.M
                        )
                    )
                    else ""
                )
            ),
            recent_turns=rendered_recent_turns,
        )

    async def refresh_recent_turns(self, *, session, profile_maint=None) -> None:
        """刷新 RECENT_CONTEXT.md 的 recent turns 区块。"""
        # 1. 读取当前会话尾部并渲染稳定的 recent turns 格式。
        profile = self._profile_maint if profile_maint is None else profile_maint
        recent_count = _recent_turn_count(self._keep_count)
        start = logical_history_tail_start(session.messages, recent_count)
        recent_turns = list(session.messages[start:])
        rendered_recent_turns = _format_recent_context_messages(recent_turns)
        # 2. 保留压缩内容，只替换 recent turns。
        existing_text = await asyncio.to_thread(profile.read_recent_context)
        updated = _replace_recent_turns_block(existing_text, rendered_recent_turns)
        await asyncio.to_thread(profile.write_recent_context, updated)

    # 只做窗口选择和 LLM 提取，写入由 MemoryEngine 统一提交。
    async def prepare_consolidation(
        self,
        session,
        archive_all: bool = False,
        force: bool = False,
        continuing: bool = False,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> _ConsolidationDraft | _ConsolidationFailure | None:
        profile_maint = self._profile_maint
        # 1. 先决定这次要归档哪一段消息窗口；没有新窗口就直接返回。
        window = _select_consolidation_window(
            session,
            keep_count=self._keep_count,
            consolidation_min_new_messages=self._consolidation_min_new_messages,
            archive_all=archive_all,
            force=force,
            continuing=continuing,
        )
        if archive_all:
            logger.info(
                "Memory consolidation (archive_all): %d total messages archived",
                len(session.messages),
            )
        else:
            if window is None:
                pending_units = logical_history_unit_count(
                    session.messages,
                    start_index=session.last_consolidated,
                )
                ready_count = pending_units - self._keep_count
                if pending_units <= self._keep_count:
                    logger.debug(
                        "Session %s: No consolidation needed (units=%d, keep=%d)",
                        session.key,
                        pending_units,
                        self._keep_count,
                    )
                else:
                    logger.debug(
                        "Session %s: Not enough messages to consolidate yet (ready=%d, min=%d, last_consolidated=%d, total=%d)",
                        session.key,
                        ready_count,
                        self._consolidation_min_new_messages,
                        session.last_consolidated,
                        len(session.messages),
                    )
                return
            logger.info(
                "Memory consolidation started: %d total, %d new to consolidate, %d keep, force=%s",
                len(session.messages),
                len(window.old_messages),
                window.keep_count,
                force,
            )

        if window is None:
            return

        # 2. 把窗口消息格式化成一段对话文本，并准备好 source_ref / 现有长期记忆 / 近期语境。
        source_ref = _build_consolidation_source_ref(window)
        conversation = _format_conversation_for_consolidation(window.old_messages)
        current_memory = await asyncio.to_thread(profile_maint.read_long_term)
        old_recent_context = await asyncio.to_thread(profile_maint.read_recent_context)
        recent_context_block = _clip_context_text(old_recent_context)

        prompt = f"""你是记忆提取代理（Memory Extraction Agent）。从对话中精确提取结构化信息，返回 JSON。

## 字段说明

### 1. "history_entries" → 记忆事件条目（数组，每条对应一个独立主题）
按主题拆分，每个独立话题写一条对象，格式为 {{"summary":"...", "emotional_weight":0}}。
summary 仍然要求 1-2 句，以 [YYYY-MM-DD HH:MM] 开头，保留足够细节便于后续向量写入和回源判断。
不同主题必须拆成独立条目，不得合并。若整段对话只有一个主题，返回只含一条的数组。

history_entries.emotional_weight 规则：
- 范围 0-10
- 普通技术讨论、普通事务记录、无明显情绪色彩 → 0
- 用户明确表达强烈喜欢/厌恶、明显受挫、关系冲突、情绪波动时按强度给 3-9
- 不确定时保守输出 0

**history_entries 提取规则（严格遵守）**：
1. 只提取 USER 明确表达的行动、经历、计划和状态；ASSISTANT 的建议、推荐、解释一律不写入，即使其中提到了地名、店名或活动。
2. 每条必须是简洁的第三人称摘要句，绝对不能包含 "USER:" 或 "ASSISTANT:" 等原始对话标记，不得复制粘贴原始对话文本。
3. 商家名称、地点、人名、数量、价格、型号等具体细节必须保留，不得用"某商店""某地方"概括。
4. 先判断当前 USER 内容的材料类型：是“用户此刻直接自述”，还是“用户正在展示一段外部聊天记录、截图 OCR、转贴 transcript 给助手看”。
5. 若 USER 内容属于外部聊天记录 / transcript，必须先做层级理解：
   - 外层：当前 USER 正在把一段材料发给助手看。
   - 内层：材料中可能有多个 speaker；这些 speaker 不自动等于当前 USER。
   - 只有当材料中某个 speaker 与当前 USER 的映射在当前会话里被明确确认时，才允许把该 speaker 的事实写入摘要。
6. 对 transcript 场景，默认认为 speaker 映射不明确；除非当前会话中有非常明确的显式说明，否则不要尝试判断材料里的某个昵称/说话人就是用户或对方。
7. 若 speaker 映射不明确，history_entries 只允许写 1 条高层 event，例如“用户向助手展示了一段与某人的聊天记录，内容涉及求职、学校、兴趣等话题”。
8. 对 transcript 场景，禁止输出任何未确认关系的句子，例如：
   - “用户向对方透露……”
   - “对方是……”
   - “双方确认……”
   - 把聊天记录里的具体事实直接写成用户个人经历
9. transcript 场景下，默认最多输出 1 条高层 history_entry；不要下钻成人物小传，不要替材料里的 speaker 自动补全身份关系，不要写任何昵称归属、学校归属、出生年份归属、爱好归属。

**transcript 场景示例（严格遵守）**：
- 错误：用户贴出一段聊天记录，speaker 归属未确认，却写成“用户向对方透露自己正在找暑期实习”。
- 错误：用户贴出一段聊天记录，直接写成“对方位于北京大兴区，就读于二外 MPAcc 专业”。
- 错误：用户贴出一段聊天记录，直接写成“对方昵称为‘一只快乐的小奶龙’”。
- 错误：用户贴出一段聊天记录，直接写成“用户曾为打 FGO 日服选修日语”。
- 正确：用户向助手展示了一段与匹配对象的聊天记录，聊天内容涉及学校背景、兴趣爱好和求职话题。

### 2. "pending_items" → PENDING.md 候选缓冲
只写用户的长期记忆候选，返回对象数组。每个对象格式：
{{"tag": "<tag>", "content": "<string>"}}

允许的 tag 只有 7 个：
- "identity"：稳定背景事实，如身份、学校/专业、长期技术方向、实习/工作经历、长期设备、长期维护项目
- "preference"：稳定偏好、禁忌、审美、游戏口味、价值取向
- "key_info"：用户明确允许保存的 key / token / id / 账号信息
- "health_long_term"：长期健康状态的一阶事实，只写长期状态，不写动态指标、基线、最近波动
- "requested_memory"：用户明确要求"长期记住"的关键内容，可比普通事实更连贯
- "correction"：对当前 MEMORY.md 现有事实的明确纠正
- "agent_context"：助手操作用户环境所需的工具性配置，如已部署服务的端口、环境变量名、工具分工约定、常用登录站点列表；不是用户画像，但对助手执行操作有长期价值；具体参数（端口号、变量名）必须完整保留。**硬规则：只有当对话明确表明该配置当前有效且助手已被授权使用时才提取；方案讨论、架构设计、网络诊断中出现的端口和地址一律不提取**

必须遵守：
- 只写跨对话仍有长期价值的内容
- 不写 agent 执行规则、SOP、工具调用顺序、流程规范
- 不写短期状态、近期计划、日程、课表、一次性操作
- 不写动态健康数据、实时指标、最近状态
- 不写对话过程总结
- 不写 self_insights、行为规律总结、关系演进感悟
- "requested_memory" 只能在用户明确表达"记住这个 / 写进长期记忆 / 以后要能聊到 / 希望你记住"时使用

进阶过滤（四条硬规则，任一触发即不提取）：

1. **网络运维细节不提取**
内网 IP、路由模式（如"CGNAT""桥接模式""NAT"）、运营商名称、MAC 地址等网络层配置属于瞬时运维信息，不提取。项目路径、配置文件名、环境变量名等与用户开发环境直接相关的信息可以提取。
✗ "家庭网络是联通宽带，光猫路由模式，内网 IP 192.168.1.x" → 不提取（网络层瞬时配置）
✓ "项目位于 /home/user/project，配置文件 config.toml" → 可提取（开发环境画像）

2. **临时状态不提取，规律习惯可提取**
带"最近""这周""目前""正在"等时间限定词的瞬时状态不提取。每周/每天持续的规律性行为模式可以提取为偏好或习惯标识。
✗ "用户最近加班频繁，靠咖啡撑着" → 不提取（瞬时状态，随时会变）
✓ "用户每周去健身房，主要做力量训练" → 可提取（规律性习惯，是长期生活方式）

3. **时效性数字和瞬时情绪不提取**
带有具体数值的动态指标（如 Star 数、增长率、评分）、瞬时情绪描述（如"失落""焦虑"）、正在进行中的短期状态。保留背后的价值判断，不提取数字和情绪本身。
✗ "项目刚突破 500 Star，但增速降到每天 2 个，用户为此很焦虑" → 不提取（数字过期、情绪瞬时）
✓ "用户长期维护某开源项目并重视社区增长" → 可提取（稳定身份信息）

4. **Agent 执行规则不放入 pending_items**
以"偏好"开头但语义上描述 agent 应如何执行的内容（如检索策略、元数据标注规范、输出格式要求等），属于 procedure，应由隐式提取路径写入向量库。
✗ "偏好搜索结果按来源可信度分层展示" → 不提取为 pending_item（agent 输出规范）
✗ "希望以后推荐前先查最新评测和社区反馈" → 不提取为 pending_item（agent 执行规则）

5. **agent_context 只提取已部署的配置，不提取方案讨论**
判断标准：对话中是否明确表明该服务/工具**当前已在运行**，且助手**已被告知可以使用**。
对话中提出的架构方案、网络诊断信息、假设性配置，即使出现了具体端口、地址或变量名，也不提取。

<example id="agent_context_proposal_vs_deployed">
反例（方案讨论 → 不提取）：
- 用户在讨论"可以搭一个 X 服务监听某端口"或"我们可以用 Y 工具穿透"——这是在设计方案，不是在告知助手已有的可用工具
- 用户问助手"这个配置怎么搭"——这是提问，不是已部署事实
- 对话中出现了 IP 地址或端口是为了排查问题、讲解原理——这是诊断/教学内容，不是可调用的配置

正例（已部署、已授权 → 提取）：
- 用户明确告知助手"X 服务现在跑着，你可以直接用"或"以后遇到 Y 场景就调这个接口"
- 用户描述了某个长期运行的工具，并期望助手在后续任务中利用它
</example>

若没有合格条目，返回空数组 []。

---

## 当前用户档案（用于查重）
{current_memory or "（空）"}

## 当前 RECENT_CONTEXT.md（仅用于主题延续参考）
使用原则（严格遵守）：
- 这份近期语境只能帮助你理解“当前窗口大概在延续什么话题”，不能作为人物身份、说话人归属、关系判断或具体事实归属的直接证据。
- 若近期语境与当前窗口原文在昵称、身份、关系、事实归属上存在冲突或不一致，必须以当前窗口原文为准。
- 不要因为近期语境里出现了某个昵称、人设或关系描述，就在新的 history_entries 中继续沿用这些判断。
- 对 transcript / 聊天截图 / 转贴聊天场景，近期语境绝不能用于推断“谁是当前用户、谁是对方、哪句话归谁”。
{recent_context_block or "（空）"}

## 待处理对话
{conversation}

只返回合法 JSON，不要 markdown 代码块。"""

        # 3. 普通 consolidation 按实际模型预算切成完整语义页。
        if self._consolidation_input_budget is not None and not archive_all:
            fixed_chars = (
                len(self._provider_system_prompt) + len(prompt) - len(conversation)
            )
            available_tokens = self._consolidation_input_budget - max(
                1, fixed_chars // 3
            )
            page_char_budget = max(1, available_tokens * 3 * 95 // 100)
            paged_window = _limit_consolidation_window(
                window,
                max_conversation_chars=page_char_budget,
            )
            if paged_window is not window:
                old_conversation = conversation
                window = paged_window
                conversation = _format_conversation_for_consolidation(
                    window.old_messages
                )
                source_ref = _build_consolidation_source_ref(window)
                prompt = _replace_consolidation_prompt_conversation(
                    prompt,
                    old_conversation=old_conversation,
                    new_conversation=conversation,
                )
                logger.info(
                    "Memory consolidation page selected: messages=%d up_to=%d "
                    "conversation_chars=%d budget_chars=%d",
                    len(window.old_messages),
                    window.consolidate_up_to,
                    len(conversation),
                    page_char_budget,
                )

        # 4. 调主模型把这页旧对话提炼成结构化结果。
        call_result = await self._call_llm_step(
            step="event_extract",
            provider=self._provider,
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            timeout_s=_EVENT_EXTRACTION_TIMEOUT_S,
        )
        if isinstance(call_result, _ConsolidationFailure):
            return call_result
        text, event_elapsed_ms = call_result
        logger.info(
            "Memory consolidation event llm raw: elapsed_ms=%d chars=%d preview=%r",
            event_elapsed_ms,
            len(text),
            text[:300],
        )

        if not text:
            logger.warning("Memory consolidation: LLM returned empty response")
            return _ConsolidationFailure(
                step="event_extract",
                error="empty_response",
                elapsed_ms=event_elapsed_ms,
            )
        result = _parse_consolidation_payload(text)
        if result is None:
            logger.warning(
                "Memory consolidation: unexpected response type. Response: %r",
                text[:200],
            )
            return _ConsolidationFailure(
                step="event_extract",
                error="invalid_json",
                elapsed_ms=event_elapsed_ms,
            )

        # 5. 归一化文本产物，并把后续写入所需信息交给 engine。
        try:
            history_entry_payloads = _normalize_history_entries(
                result.get("history_entries"),
                result.get("history_entry"),
            )
            pending_items = _format_pending_items(result.get("pending_items", []))
        except _ConsolidationPayloadError as exc:
            logger.warning("Memory consolidation: invalid event payload: %s", exc)
            return _ConsolidationFailure(
                step="event_extract",
                error=f"invalid_schema: {exc}",
                elapsed_ms=event_elapsed_ms,
            )
        # 6. 生成 markdown 产物，向量写入由 engine 订阅提交事件完成。
        recent_context_text = await self._build_recent_context_snapshot(
            session=session,
            window=window,
            archive_all=archive_all,
            old_recent_context=old_recent_context,
        )
        if isinstance(recent_context_text, _ConsolidationFailure):
            return recent_context_text
        return _ConsolidationDraft(
            window=window,
            source_ref=source_ref,
            history_entry_payloads=history_entry_payloads,
            pending_items=pending_items,
            conversation=conversation,
            recent_context_text=recent_context_text,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            archive_all=archive_all,
        )


class MarkdownMemoryStore(MemoryStore):
    def backup_long_term(self, backup_name: str = "MEMORY.bak.md") -> None:
        self._backup_profile(self.memory_file, backup_name)

    def backup_self(self, backup_name: str = "SELF.bak.md") -> None:
        self._backup_profile(self.self_file, backup_name)

    def _backup_profile(self, source: Path, latest_name: str) -> None:
        """原子保存最新备份和不可覆盖的历史版本。"""
        if not source.exists():
            return

        # 1. 先保存唯一历史版本，保证后续优化无法覆盖恢复点
        content = source.read_text(encoding="utf-8")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        nanoseconds = time.time_ns() % 1_000_000_000
        history_path = (
            self.memory_dir
            / "backups"
            / f"{source.stem}.{timestamp}-{nanoseconds:09d}.bak{source.suffix}"
        )
        atomic_write_text(history_path, content, domain="memory_backup")

        # 2. 刷新固定名称，保留现有手工恢复入口
        atomic_write_text(
            source.with_name(latest_name),
            content,
            domain="memory_backup",
        )

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())


@dataclass
class MarkdownMemoryRuntime:
    store: MarkdownMemoryStore
    maintenance: "MarkdownMemoryMaintenance"


class MarkdownMemoryMaintenance:
    def __init__(
        self,
        *,
        store: MarkdownMemoryStore,
        provider: "LLMProvider",
        model: str,
        keep_count: int,
        consolidation_input_budget: int | None = None,
        provider_system_prompt: str = "",
        event_bus: "EventBus | None" = None,
        recent_context_provider: "LLMProvider | None" = None,
        recent_context_model: str | None = None,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._worker = _MarkdownConsolidationWorker(
            profile_maint=store,
            provider=provider,
            model=model,
            keep_count=keep_count,
            consolidation_input_budget=consolidation_input_budget,
            provider_system_prompt=provider_system_prompt,
            recent_context_provider=recent_context_provider,
            recent_context_model=recent_context_model,
        )
        self._keep_count = keep_count
        self._consolidation_min_new_messages = max(5, keep_count // 2)
        self._get_session: Callable[[str], object] | None = None
        self._save_session: Callable[[object], Awaitable[None]] | None = None
        self._maintenance_queues: dict[str, deque[TurnCommitted]] = {}
        self._maintenance_tasks: dict[str, asyncio.Task[None]] = {}
        self._maintenance_locks: dict[str, asyncio.Lock] = {}
        if event_bus is not None:
            event_bus.on(TurnCommitted, self.on_turn_committed)

    def bind_lifecycle(self, request: MemoryLifecycleBindRequest) -> None:
        self._get_session = request.get_session
        self._save_session = request.save_session

    def on_turn_committed(self, event: TurnCommitted) -> None:
        if bool((event.extra or {}).get("skip_post_memory")):
            return
        self._enqueue_maintenance(event)

    def _enqueue_maintenance(self, event: TurnCommitted) -> None:
        if self._get_session is None or self._save_session is None:
            return
        session_key = event.session_key
        queue = self._maintenance_queues.setdefault(session_key, deque())
        queue.append(event)
        if session_key in self._maintenance_tasks:
            return
        task = asyncio.create_task(
            self._run_maintenance_queue(session_key),
            name=f"markdown-memory-maintenance:{session_key}",
        )
        self._maintenance_tasks[session_key] = task
        task.add_done_callback(lambda t: self._on_maintenance_done(t, session_key))

    async def _run_maintenance_queue(self, session_key: str) -> None:
        """按 session 顺序执行维护任务，并保留每个提交事件的刷新语义。"""
        # 1. 固定本次任务的生命周期 owner，后续不再重复判空。
        get_session = self._get_session
        save_session = self._save_session
        if get_session is None or save_session is None:
            raise RuntimeError("markdown memory lifecycle is not bound")
        lock = self._maintenance_locks.setdefault(session_key, asyncio.Lock())
        # 2. 在同一 session 锁内逐个消费提交事件。
        async with lock:
            while True:
                queue = self._maintenance_queues.get(session_key)
                if not queue:
                    return
                event = queue.popleft()
                session = get_session(session_key)
                if self._should_consolidate_session(session):
                    result = await self._consolidate_unlocked(
                        ConsolidateRequest(
                            session=session,
                            scope_channel=event.channel,
                            scope_chat_id=event.chat_id,
                        )
                    )
                    if result.trace.get("mode") == "markdown":
                        await save_session(session)
                    elif result.trace.get("mode") == "failed":
                        logger.warning(
                            "markdown memory maintenance consolidation failed: "
                            "session=%s step=%s error=%s elapsed_ms=%s",
                            session_key,
                            result.trace.get("step"),
                            result.trace.get("error"),
                            result.trace.get("elapsed_ms"),
                        )
                else:
                    await self.refresh_recent_turns(
                        RefreshRecentTurnsRequest(session=session)
                    )

    def _on_maintenance_done(
        self,
        task: asyncio.Task[None],
        session_key: str,
    ) -> None:
        if self._maintenance_tasks.get(session_key) is task:
            _ = self._maintenance_tasks.pop(session_key, None)
        queue = self._maintenance_queues.get(session_key)
        if queue:
            next_task = asyncio.create_task(
                self._run_maintenance_queue(session_key),
                name=f"markdown-memory-maintenance:{session_key}",
            )
            self._maintenance_tasks[session_key] = next_task
            next_task.add_done_callback(
                lambda t: self._on_maintenance_done(t, session_key)
            )
        else:
            _ = self._maintenance_queues.pop(session_key, None)
        if task.cancelled():
            logger.info("markdown memory maintenance cancelled: %s", session_key)
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "markdown memory maintenance failed: session=%s err=%s",
                session_key,
                exc,
            )

    def _should_consolidate_session(self, session: object) -> bool:
        return (
            _select_consolidation_window(
                session,
                keep_count=self._keep_count,
                consolidation_min_new_messages=self._consolidation_min_new_messages,
                archive_all=False,
                force=False,
            )
            is not None
        )

    async def consolidate(self, request: ConsolidateRequest) -> ConsolidateResult:
        # 1. session 级记忆排除：命中统一谓词的 session 不进入 markdown 沉淀。
        session_key = request.session.key
        if not session_key:
            return await self._consolidate_unlocked(request)
        session_metadata = getattr(request.session, "metadata", None)
        if isinstance(session_metadata, dict) and excludes_memory(
            session_key, session_metadata
        ):
            return ConsolidateResult(
                trace={
                    "mode": "skipped",
                    "reason": "session_memory_excluded",
                }
            )
        lock = self._maintenance_locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            return await self._consolidate_unlocked(request)

    async def _consolidate_unlocked(
        self, request: ConsolidateRequest
    ) -> ConsolidateResult:
        """逐页压缩 backlog，并在每页提交后保存绝对游标。"""

        # 1. 显式 force 先把失配游标持久回退到最近的完整逻辑单元边界。
        cursor_repair: dict[str, int] = {}
        if request.force and not request.archive_all:
            previous_cursor = request.session.last_consolidated
            _, recovered_cursor = _resolve_consolidation_cursor(
                request.session.messages,
                previous_cursor,
                force=True,
            )
            if recovered_cursor != previous_cursor:
                request.session.last_consolidated = recovered_cursor
                cursor_repair = {
                    "cursor_rewound_from": previous_cursor,
                    "cursor_rewound_to": recovered_cursor,
                }
                logger.warning(
                    "Force consolidation rewound invalid cursor: session=%s from=%d to=%d",
                    request.session.key,
                    previous_cursor,
                    recovered_cursor,
                )
                if self._save_session is not None:
                    await self._save_session(request.session)

        # 2. 每页重新读取 RECENT_CONTEXT，让下一页基于上一页的滚动摘要继续压缩。
        consolidated_count = 0
        source_refs: list[str] = []
        while True:
            draft = await self._worker.prepare_consolidation(
                request.session,
                archive_all=request.archive_all,
                force=request.force,
                continuing=bool(source_refs),
                scope_channel=request.scope_channel,
                scope_chat_id=request.scope_chat_id,
            )
            if draft is None:
                if consolidated_count == 0:
                    return ConsolidateResult(
                        trace={"mode": "skipped", **cursor_repair}
                    )
                break
            if isinstance(draft, _ConsolidationFailure):
                failure_trace: dict[str, object] = {
                    "mode": "failed",
                    "step": draft.step,
                    "error": draft.error,
                    "elapsed_ms": draft.elapsed_ms,
                }
                if source_refs:
                    failure_trace["completed_pages"] = len(source_refs)
                failure_trace.update(cursor_repair)
                return ConsolidateResult(
                    consolidated_count=consolidated_count,
                    trace=failure_trace,
                )

            # 3. 提交一页后立即保存游标；失败重启只会重做未提交页。
            await self._commit_markdown_draft(request.session, draft)
            consolidated_count += len(draft.window.old_messages)
            source_refs.append(draft.source_ref)
            if self._save_session is not None:
                await self._save_session(request.session)
            if (
                not request.drain_backlog
                or request.archive_all
                or draft.window.consolidate_up_to >= len(request.session.messages)
            ):
                break

        return ConsolidateResult(
            consolidated_count=consolidated_count,
            trace={
                "mode": "markdown",
                "source_ref": source_refs[-1],
                "source_refs": source_refs,
                "pages": len(source_refs),
                **cursor_repair,
            },
        )

    async def _commit_markdown_draft(
        self,
        session: object,
        draft: "_ConsolidationDraft",
    ) -> None:
        if draft.pending_items:
            appended = await asyncio.to_thread(
                self._store.append_pending_once,
                draft.pending_items,
                source_ref=draft.source_ref,
                kind="pending_items",
            )
            if appended:
                logger.info(
                    "Markdown memory: appended %d pending_items",
                    len(draft.pending_items.splitlines()),
                )
        self._store.write_recent_context(draft.recent_context_text)
        if draft.archive_all:
            session.last_consolidated = 0
        else:
            session.last_consolidated = draft.window.consolidate_up_to
        if self._event_bus is not None:
            await self._event_bus.emit(
                ConsolidationCommitted(
                    history_entry_payloads=list(draft.history_entry_payloads),
                    source_ref=draft.source_ref,
                    scope_channel=draft.scope_channel,
                    scope_chat_id=draft.scope_chat_id,
                    conversation=draft.conversation,
                )
            )

    async def refresh_recent_turns(
        self,
        request: RefreshRecentTurnsRequest,
    ) -> None:
        await self._worker.refresh_recent_turns(session=request.session)


def build_markdown_memory_runtime(
    *,
    workspace: Path,
    provider: "LLMProvider",
    model: str,
    keep_count: int,
    consolidation_input_budget: int | None = None,
    provider_system_prompt: str = "",
    event_bus: "EventBus | None" = None,
    recent_context_provider: "LLMProvider | None" = None,
    recent_context_model: str | None = None,
) -> MarkdownMemoryRuntime:
    store = MarkdownMemoryStore(workspace)
    maintenance = MarkdownMemoryMaintenance(
        store=store,
        provider=provider,
        model=model,
        keep_count=keep_count,
        consolidation_input_budget=consolidation_input_budget,
        provider_system_prompt=provider_system_prompt,
        event_bus=event_bus,
        recent_context_provider=recent_context_provider,
        recent_context_model=recent_context_model,
    )
    return MarkdownMemoryRuntime(store=store, maintenance=maintenance)
