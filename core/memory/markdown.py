from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from agent.llm_json import load_json_object_loose
from agent.memory import MemoryStore
from agent.plugin_composition import (
    CHAT_MODELS,
    BoundChatModel,
    ModelError,
    ModelRequest,
    ModelRole,
)
from agent.plugins.snapshot import RuntimeSnapshotStore, lease_runtime_snapshot
from agent.prompting import is_context_frame
from core.memory.events import ConsolidationCommitted
from infra.persistence.json_store import atomic_write_text

if TYPE_CHECKING:
    from bus.event_bus import EventBus

logger = logging.getLogger("memory.markdown")

_EVENT_EXTRACTION_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class CompactionMarkdownDraft:
    """Validated Markdown side effects for one session compaction source_ref."""

    source_ref: str
    history_entry_payloads: tuple[tuple[str, int], ...] = ()
    pending_items: str = ""
    conversation: str = ""
    scope_channel: str = ""
    scope_chat_id: str = ""


@runtime_checkable
class MemoryProfileApi(Protocol):
    def read_long_term(self) -> str: ...

    def write_long_term(self, content: str) -> None: ...

    def read_self(self) -> str: ...

    def write_self(self, content: str) -> None: ...

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
class _ConsolidationDraft:
    source_ref: str
    history_entry_payloads: list[tuple[str, int]]
    pending_items: str
    conversation: str
    scope_channel: str
    scope_chat_id: str


@dataclass(frozen=True)
class _ConsolidationFailure:
    step: str
    error: str
    elapsed_ms: int = 0


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


def _is_context_frame_message(message: dict) -> bool:
    content = str(message.get("content") or "")
    return is_context_frame(content)


def _group_compaction_source_plan(
    selected_source_messages: tuple[dict[str, object], ...],
) -> list[list[dict[str, object]]]:
    """Validate the exact source plan and group only consecutive logical units."""

    groups: list[list[dict[str, object]]] = []
    seen_unit_refs: set[str] = set()
    current_ref = ""
    for item in selected_source_messages:
        message = item.get("message")
        message_id = item.get("id")
        raw_seq = item.get("seq")
        unit_ref = item.get("unit_ref")
        if (
            not isinstance(message, dict)
            or not isinstance(message_id, str)
            or not message_id
            or not isinstance(raw_seq, int)
            or isinstance(raw_seq, bool)
            or raw_seq < 0
            or not isinstance(unit_ref, str)
            or not unit_ref.strip()
        ):
            raise ValueError("compaction Markdown source plan 无效")
        normalized = dict(item)
        normalized["message"] = dict(message)
        if not groups or unit_ref != current_ref:
            if unit_ref in seen_unit_refs:
                raise ValueError("compaction Markdown source plan 的 unit_ref 非连续")
            groups.append([])
            current_ref = unit_ref
            seen_unit_refs.add(unit_ref)
        groups[-1].append(normalized)
    return groups


def _merge_pending_pages(pending_pages: list[str]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for page in pending_pages:
        for line in page.splitlines():
            value = line.strip()
            if value and value not in seen:
                seen.add(value)
                lines.append(value)
    return "\n".join(lines)


class _MarkdownConsolidationWorker:
    def __init__(
        self,
        *,
        profile_maint: "MarkdownMemoryStore",
        provider_input_budget: int | None,
    ) -> None:
        self._profile_maint = profile_maint
        self._configured_provider_input_budget = provider_input_budget

    @staticmethod
    def _summary_output_tokens(provider: BoundChatModel) -> int:
        """Resolve the current provider's bounded event-extraction output budget."""

        raw_cap = provider.descriptor.capabilities.max_output_tokens or 0
        if raw_cap is None:
            raw_cap = 0
        if not isinstance(raw_cap, int) or isinstance(raw_cap, bool) or raw_cap < 0:
            raise ValueError("Markdown provider max_output_tokens 必须是非负整数")
        return min(1024, raw_cap) if raw_cap > 0 else 1024

    def _input_budget(
        self,
        provider: BoundChatModel,
        summary_output_tokens: int,
    ) -> int | None:
        """Resolve input budget from explicit policy or the current provider window."""

        if self._configured_provider_input_budget is not None:
            return self._configured_provider_input_budget
        context_window = provider.descriptor.capabilities.context_window or 0
        if context_window <= summary_output_tokens:
            return None
        return context_window - summary_output_tokens

    async def _call_llm_step(
        self,
        *,
        step: str,
        provider: BoundChatModel,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_s: float,
    ) -> tuple[str, int] | _ConsolidationFailure:
        started_at = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                provider.complete(
                    ModelRequest(
                        messages=messages,
                        max_output_tokens=max_tokens,
                        disable_reasoning=True,
                    )
                ),
                timeout=timeout_s,
            )
        except (ModelError, TimeoutError) as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            error = _format_consolidation_error(exc)
            logger.error(
                "Memory consolidation llm step failed: step=%s elapsed_ms=%d error=%s",
                step,
                elapsed_ms,
                error,
            )
            return _ConsolidationFailure(
                step=step,
                error=error,
                elapsed_ms=elapsed_ms,
            )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        content = response.content
        return (content.strip() if content is not None else ""), elapsed_ms

    async def prepare_page(
        self,
        provider: BoundChatModel,
        messages: list[dict],
        *,
        source_ref: str,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> _ConsolidationDraft | _ConsolidationFailure:
        profile_maint = self._profile_maint
        # 1. 使用 ContextCompactor 已提交的精确 source plan，不自行读取 session/cursor。
        conversation = _format_conversation_for_consolidation(messages)
        current_memory = await asyncio.to_thread(profile_maint.read_long_term)

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

## 待处理对话
{conversation}

只返回合法 JSON，不要 markdown 代码块。"""

        # 3. 按当前 frozen provider 解析输入和输出边界，禁止超预算请求。
        summary_output_tokens = self._summary_output_tokens(provider)
        provider_input_budget = self._input_budget(provider, summary_output_tokens)
        if provider_input_budget is None:
            return _ConsolidationFailure(
                step="input_budget",
                error=(
                    "markdown provider input budget unavailable: "
                    "context_window="
                    f"{provider.descriptor.capabilities.context_window or 0} "
                    f"summary_output={summary_output_tokens}"
                ),
            )
        estimated_tokens = provider.estimate_context_tokens(
            [{"role": "user", "content": prompt}],
            [],
        )
        if estimated_tokens >= provider_input_budget:
            return _ConsolidationFailure(
                step="input_budget",
                error=(
                    "markdown provider input exceeds hard budget: "
                    f"estimated={estimated_tokens} "
                    f"budget={provider_input_budget}"
                ),
            )

        # 4. 调主模型把这页精确历史提炼成结构化结果。
        call_result = await self._call_llm_step(
            step="event_extract",
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=summary_output_tokens,
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
        return _ConsolidationDraft(
            source_ref=source_ref,
            history_entry_payloads=history_entry_payloads,
            pending_items=pending_items,
            conversation=conversation,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
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
        runtime_snapshot_store: RuntimeSnapshotStore,
        provider_input_budget: int | None = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._runtime_snapshot_store = runtime_snapshot_store
        if provider_input_budget is not None and provider_input_budget <= 0:
            raise ValueError("provider_input_budget 必须大于 0")
        self._worker = _MarkdownConsolidationWorker(
            profile_maint=store,
            provider_input_budget=provider_input_budget,
        )
        self._provider_input_budget = provider_input_budget

    def read_compaction_receipt(
        self,
        source_ref: str,
    ) -> dict[str, object] | None:
        """Read the immutable prepared compaction receipt."""

        return self._store.read_consolidation_receipt(
            source_ref,
            kind="session_compaction_receipt",
        )

    def write_compaction_receipt(
        self,
        source_ref: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Persist the immutable prepared compaction receipt."""

        return self._store.write_consolidation_receipt(
            source_ref,
            payload,
            kind="session_compaction_receipt",
        )

    async def prepare_compaction_markdown(
        self,
        selected_source_messages: tuple[dict[str, object], ...],
        *,
        source_ref: str,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> CompactionMarkdownDraft:
        """Extract Markdown/PENDING effects from an exact committed source plan."""

        if not selected_source_messages:
            raise ValueError("compaction Markdown source plan 不能为空")
        source_ref = source_ref.strip()
        if not source_ref:
            raise ValueError("compaction Markdown source_ref 不能为空")

        # 1. 按 exact plan 的连续 unit_ref 装页；任何失败都中止整个 checkpoint。
        groups = _group_compaction_source_plan(selected_source_messages)
        page_index = 0
        history_entries: list[tuple[str, int]] = []
        pending_pages: list[str] = []
        conversations: list[str] = []
        remaining = list(groups)
        async with lease_runtime_snapshot(self._runtime_snapshot_store) as snapshot:
            root = snapshot.composition_root
            if root is None:
                raise RuntimeError("RuntimeSnapshot 缺少 composition Root")
            chat_models = root.context.require(CHAT_MODELS)
            async with chat_models.execution() as execution:
                provider = execution.chat(ModelRole.DEFAULT)
                while remaining:
                    page_groups = list(remaining)
                    while page_groups:
                        rows: list[dict[str, object]] = []
                        for group in page_groups:
                            for item in group:
                                raw_message = item["message"]
                                if not isinstance(raw_message, dict):
                                    raise RuntimeError(
                                        "compaction Markdown source plan message 无效"
                                    )
                                rows.append(dict(raw_message))
                        draft = await self._worker.prepare_page(
                            provider,
                            rows,
                            source_ref=source_ref,
                            scope_channel=scope_channel,
                            scope_chat_id=scope_chat_id,
                        )
                        if isinstance(draft, _ConsolidationFailure):
                            if draft.step == "input_budget" and len(page_groups) > 1:
                                page_groups.pop()
                                continue
                            raise RuntimeError(
                                "compaction Markdown prepare failed: "
                                f"page={page_index} {draft.step}: {draft.error}"
                            )
                        break
                    else:
                        raise RuntimeError("compaction Markdown page selection failed")

                    history_entries.extend(draft.history_entry_payloads)
                    pending_pages.append(draft.pending_items)
                    if draft.conversation:
                        conversations.append(draft.conversation)
                    remaining = remaining[len(page_groups) :]
                    page_index += 1

        return CompactionMarkdownDraft(
            source_ref=source_ref,
            history_entry_payloads=tuple(history_entries),
            pending_items=_merge_pending_pages(pending_pages),
            conversation="\n".join(conversations),
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )

    async def commit_compaction_markdown(
        self,
        draft: CompactionMarkdownDraft,
    ) -> None:
        """Commit Markdown/PENDING effects for a ledger source without moving its cursor."""

        source_ref = draft.source_ref.strip()
        if not source_ref:
            raise ValueError("compaction markdown source_ref 不能为空")
        if draft.pending_items.strip():
            self._store.append_pending_once(
                draft.pending_items,
                source_ref=source_ref,
                kind="pending_items",
            )
        if self._event_bus is not None:
            await self._event_bus.emit(
                ConsolidationCommitted(
                    history_entry_payloads=list(draft.history_entry_payloads),
                    source_ref=source_ref,
                    scope_channel=draft.scope_channel,
                    scope_chat_id=draft.scope_chat_id,
                    conversation=draft.conversation,
                )
            )


def build_markdown_memory_runtime(
    *,
    workspace: Path,
    runtime_snapshot_store: RuntimeSnapshotStore,
    provider_input_budget: int | None = None,
    event_bus: "EventBus | None" = None,
) -> MarkdownMemoryRuntime:
    store = MarkdownMemoryStore(workspace)
    maintenance = MarkdownMemoryMaintenance(
        store=store,
        runtime_snapshot_store=runtime_snapshot_store,
        provider_input_budget=provider_input_budget,
        event_bus=event_bus,
    )
    return MarkdownMemoryRuntime(store=store, maintenance=maintenance)
