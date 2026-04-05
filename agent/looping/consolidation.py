import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

import json_repair

from agent.llm_json import load_json_object_loose

logger = logging.getLogger("agent.loop")

if TYPE_CHECKING:
    from agent.looping.ports import TurnScheduler
    from agent.provider import LLMProvider
    from core.memory.port import MemoryPort
    from core.memory.profile import ProfileMaintenanceStore
    from memory2.profile_extractor import ProfileFactExtractor
    from session.manager import SessionManager

_ALLOWED_PENDING_TAGS = frozenset(
    {
        "identity",
        "preference",
        "key_info",
        "health_long_term",
        "requested_memory",
        "correction",
    }
)


def _format_pending_items(raw_items) -> str:
    """Normalize LLM pending_items into markdown bullets accepted by PENDING.md."""
    if not isinstance(raw_items, list):
        return ""

    lines = []
    seen = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if tag not in _ALLOWED_PENDING_TAGS or not content:
            continue
        line = f"- [{tag}] {content}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def _parse_consolidation_payload(text: str) -> dict | None:
    return load_json_object_loose(text)


@dataclass(frozen=True)
class ConsolidationWindow:
    old_messages: list[dict]
    keep_count: int
    consolidate_up_to: int


def _select_consolidation_window(
    session,
    *,
    memory_window: int,
    consolidation_min_new_messages: int,
    archive_all: bool,
) -> ConsolidationWindow | None:
    total_messages = len(session.messages)
    if archive_all:
        return ConsolidationWindow(
            old_messages=list(session.messages),
            keep_count=0,
            consolidate_up_to=total_messages,
        )

    keep_count = memory_window // 2
    if total_messages <= keep_count:
        return None
    if total_messages - session.last_consolidated <= 0:
        return None

    consolidate_up_to = total_messages - keep_count
    old_messages = session.messages[session.last_consolidated : consolidate_up_to]
    if len(old_messages) < max(1, int(consolidation_min_new_messages)):
        return None
    return ConsolidationWindow(
        old_messages=old_messages,
        keep_count=keep_count,
        consolidate_up_to=consolidate_up_to,
    )


def _build_consolidation_source_ref(window: ConsolidationWindow) -> str:
    """返回本次 consolidation 窗口内所有消息 ID 的 JSON 列表。
    缺失 id 的消息（迁移前的历史脏数据）直接跳过。
    """
    ids = [str(msg["id"]) for msg in window.old_messages if msg.get("id")]
    return json.dumps(ids, ensure_ascii=False)


def _build_entry_source_ref(base_source_ref: str, entry: str) -> str:
    """为单条 history_entry 生成稳定子键，避免同窗口多条写入互相覆盖。"""
    text = (entry or "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else "empty"
    return f"{base_source_ref}#h:{digest}"


def _format_conversation_for_consolidation(old_messages: list[dict]) -> str:
    lines = []
    for message in old_messages:
        if not message.get("content") or message.get("role") == "tool":
            continue
        if message.get("role") == "assistant" and message.get("proactive"):
            continue
        role = str(message.get("role", "")).upper()
        ts = str(message.get("timestamp", "?"))[:16]
        lines.append(f"[{ts}] {role}: {message['content']}")
    return "\n".join(lines)


def _select_recent_history_entries(history_text: str, *, limit: int = 3) -> list[str]:
    if not history_text.strip() or limit <= 0:
        return []
    chunks = re.split(r"\n\s*\n+", history_text.strip())
    entries = [chunk.strip() for chunk in chunks if chunk.strip()]
    return entries[-limit:]


def _coerce_history_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


class ConsolidationService:
    def __init__(
        self,
        *,
        memory_port: "MemoryPort",
        profile_maint: "ProfileMaintenanceStore | None" = None,
        provider: "LLMProvider",
        model: str,
        memory_window: int,
        consolidation_min_new_messages: int = 10,
        profile_extractor: "ProfileFactExtractor | None" = None,
    ) -> None:
        self._memory_port = memory_port
        self._profile_maint = profile_maint or memory_port
        self._provider = provider
        self._model = model
        self._memory_window = memory_window
        self._consolidation_min_new_messages = max(
            1, int(consolidation_min_new_messages)
        )
        self._profile_extractor = profile_extractor

    async def _extract_and_save_profile_facts(
        self,
        *,
        extractor,
        conversation: str,
        existing_profile: str,
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
    ) -> None:
        try:
            facts = await extractor.extract(
                conversation,
                existing_profile=existing_profile,
            )
            if not facts:
                return

            for fact in facts:
                await self._memory_port.save_item(
                    summary=fact.summary,
                    memory_type="profile",
                    extra={
                        "category": fact.category,
                        "scope_channel": scope_channel,
                        "scope_chat_id": scope_chat_id,
                    },
                    source_ref=f"{source_ref}#profile",
                    happened_at=fact.happened_at,
                )
                logger.info(
                    "memory2 profile fact saved: category=%s %r",
                    fact.category,
                    fact.summary[:60],
                )
        except Exception as e:
            logger.warning("profile fact extraction failed: %s", e)

    @staticmethod
    def _build_long_term_prompt(*, conversation: str, existing_profile: str) -> str:
        return f"""你是长期记忆提取专家。从对话窗口中一次性提取三类长期记忆，返回 JSON。

默认答案是所有数组为空。提取门槛要高，宁可不提取，也不要把临时信息写进长期记忆。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【核心判断标准】
把这条信息放进 6 个月后的一次全新对话，它还有用吗？
→ 是 → 可能是长期记忆，继续检查
→ 否 → 不是长期记忆，留空

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【三类记忆的语义】

profile — 关于用户本人或其客观处境的事实
  语义：身份背景、持有物、爱好、健康事实、长期状态、重要决定
  允许 category：personal_fact / purchase / decision / status
  要求：只有 USER 在对话中直接陈述自身的事实，才允许提取
  禁止：用户提问、追问、反问、记忆测试句一律不算事实披露，绝对禁止反推
    · "你还记得我什么时候开始戴 fitbit 手环的吗" → 返回空
    · "你记得我住哪里吗" → 返回空
    · "我之前是不是买过这个" → 返回空

preference — 用户希望怎样被服务、怎样被讲解、怎样被推荐
  语义：跨 session 稳定成立的偏好/厌恶/倾向，而非硬约束
  来自 USER 明确表达

procedure — agent 在未来类似场景下应遵守的长期执行规则
  语义：面向 agent 的行为规则，跨任务可复用
  来自 USER 的长期要求，或被 USER 明确确认过的非显然做法

绝对不输出：event（有时间性的具体事件）

区分三类：
- "用户是什么/拥有什么/处在什么客观背景里" → profile
- "用户希望 agent 怎么服务他、怎么讲解、怎么推荐" → preference
- "agent 在某类请求下必须怎么做/用什么工具" → procedure（有明确执行步骤/工具要求）
- 只是方向性偏好 → preference（优先选 preference）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【preference / procedure 提取前四项检查，顺序执行，任一不通过即不提取】

▸ 检查 0 — 元讨论/举例说明
先判断 USER 是在提供长期规则，还是在讨论"什么该记、怎么记、你是否理解、请举例说明"。
  - 元讨论场景：只允许提取 USER 自己明确说出的长期规则/筛选标准
  - ASSISTANT 为说明概念而举出的任何例子、类比、假设场景一律不得提取
  - 即使 ASSISTANT 的示例内容本身合理、未来有用，也不能因"看起来像长期规则"就入库

▸ 检查 A — USER 原话锚点
在 USER 消息里找到支撑这条记忆的直接原句（逐字存在，不是推断）。
  - 找不到 USER 的直接原句 → 不提取
  - ASSISTANT 的解释、建议、工具返回的数据，不算 USER 原句
  - USER 没有反驳 ASSISTANT ≠ USER 认同且希望长期记忆
  - USER 消息是纯状态汇报（"复习中"/"在看书"/"工作中"等）→ 不提取

▸ 检查 B — 时效性
  - 涉及当前任务、当前时间段、当前情境（本次/今天/这个项目） → 不提取
  - 只有明确跨 session 稳定成立，才继续

▸ 检查 C — 来源方向
  - 核心内容来自 ASSISTANT（解释/建议/工具结果） → 不提取
  - ASSISTANT 主动给出建议，USER 没有明确说"以后都这样"/"记住这个" → 不提取
  - "USER 没有反驳"不等于"USER 授权 AGENT 长期执行这条规则"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【profile 专用规则】

仅允许以下 4 类 category：
- purchase：用户购买 / 下单了什么
- decision：用户明确拍板了什么方案 / 计划
- status：用户某件事的状态变化（等待/完成/放弃/里程碑达成）
- personal_fact：用户关于自身的事实性披露（身份/背景/持有物/爱好/习惯/经验背景）

必须遵守：
- 纯技术讨论、闲聊、打招呼不输出
- 若 existing_profile 已有相同事实，不重复输出
- summary 简洁、可独立检索；personal_fact 默认不填 happened_at
- 每一件具体的事单独一条，绝对不合并
  ✗ 错误："用户购买了多件商品"
  ✓ 正确：每件商品单独一条，写出具体名称/型号
- ASSISTANT 的回复只作背景参考，不作提取证据
  即使 ASSISTANT 说"你之前买了 X""你是 XX 方向的学生"，也不得作为事实来源

额外禁止：
- 工程操作（安装/更新/配置工具/依赖）→ 这些是工程 event，不是 profile
- 项目内讨论（架构决策/重构方案/代码评审）
- 用户表达的观点/意见 → 必须是客观事实
- 纯 event：例如"这周日去徒步""昨晚去了超市""明天要开会"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【示例】

<example id="keep_profile_personal_fact">
USER: 我在互联网公司做产品经理，今年30岁，住在上海，有一块 Fitbit 手表，爱好是弹钢琴。
→ profile: [
  {{"summary": "用户在互联网公司做产品经理", "category": "personal_fact"}},
  {{"summary": "用户今年30岁", "category": "personal_fact"}},
  {{"summary": "用户住在上海", "category": "personal_fact"}},
  {{"summary": "用户有一块 Fitbit 手表", "category": "personal_fact"}},
  {{"summary": "用户的爱好是弹钢琴", "category": "personal_fact"}}
]
</example>

<example id="drop_profile_memory_test">
USER: 你还记得我什么时候开始戴 fitbit 手环的吗
→ profile: []（提问不是事实披露，绝对不反推）
</example>

<example id="profile_event_split">
USER: 这周日朋友约我去徒步，我其实不常徒步，不知道该买什么装备。
→ profile: [
  {{"summary": "用户不常徒步", "category": "personal_fact"}},
  {{"summary": "用户目前缺少徒步相关装备准备", "category": "personal_fact"}}
]
不提取："这周日去徒步"（是 event）
</example>

<example id="profile_not_preference">
USER: 我家有 10 套房，我平时爱弹钢琴，而且我有一块 Fitbit 手表
→ profile: [以上三条 personal_fact]
→ preference/procedure: []
（这些是用户身份事实，不是"用户希望被怎样服务"）
</example>

<example id="keep_explicit_rule">
USER: 以后帮我查菜谱只给 20 分钟以内能做完的，我没时间搞复杂的
检查A: "以后帮我查菜谱只给20分钟以内能做完的" ✓
检查B: "以后"明确跨 session ✓
检查C: 来自 USER 主动要求 ✓
→ procedure: [{{"summary": "查询菜谱时只推荐 20 分钟内可完成的菜式"}}]
</example>

<example id="keep_multi_source_research">
USER: 以后帮我查耳机先看 B 站评测和 Reddit 讨论，别只看官网参数
→ procedure: [{{"summary": "查询耳机时先看 B 站评测和 Reddit 讨论，不只依赖官网参数"}}]
</example>

<example id="keep_preference_trimmed">
USER: 我不喜欢这种悬疑风格的游戏，太压抑了
ASSISTANT: 明白！你是偏好轻松明快风格的玩家，喜欢治愈系或休闲类游戏……
→ preference: [{{"summary": "不喜欢悬疑压抑风格的游戏"}}]
✗ 不能写："偏好治愈系或休闲类游戏"（USER 没说过，来自 ASSISTANT 延伸）
</example>

<example id="keep_preference_service_style">
USER: 你给我讲内容的时候最好附带一个很棒的例子，并且最好贯穿始终
→ preference: [{{"summary": "讲解内容时最好附带贯穿始终的例子"}}]
（这是"希望被怎样讲解"，是 preference 不是 profile）
</example>

<example id="drop_situational">
USER: 今晚几个同学来，想找个气氛好的日料店
→ 全部为空（"今晚"是当前情境，不跨 session）
✗ 不能提取："用户喜欢日料"（推断）
</example>

<example id="drop_knowledge">
USER: TCP 和 UDP 的区别是什么
ASSISTANT: TCP 是可靠传输协议，有拥塞控制和重传机制……
→ 全部为空（USER 在提问，知识内容来自 ASSISTANT）
✗ 不能提取："TCP 是可靠传输协议"
</example>

<example id="drop_assistant_proactive_advice">
USER: 在赶代码
ASSISTANT: 别忘了每隔一段时间起来活动下，喝点水，久坐对颈椎不好……
→ 全部为空
✗ 不能提取："每隔45分钟应起身活动并补水"（来自 ASSISTANT，USER 没有授权）
关键判断：ASSISTANT 建议得再具体再合理，只要 USER 没有明确授权，就不是长期记忆
</example>

<example id="drop_meta_discussion_example">
USER: 我希望只有每轮对话里真正重要的参考信息才值得存入 memory.md，你举个例子我看看你理解没有
ASSISTANT: 明白。比如智能家居架构应坚持纯本地化部署，拒绝云端依赖……
检查0: USER 在讨论记忆标准并要求举例，是元讨论
可提取：USER 自己说出的筛选标准
ASSISTANT 的智能家居举例只是教学示范，不是 USER 新提供的规则
→ procedure: [{{"summary": "每轮对话中真正重要的参考信息才值得存入 memory.md"}}]
✗ 不能提取："智能家居架构坚持纯本地化部署"
</example>

<example id="drop_workaround">
USER: 那就直接写个脚本绕过去吧
→ 全部为空（当前任务临时策略，不跨 session）
✗ 不能提取："遇到此类问题应优先用 Python 脚本绕过"
</example>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【summary 写法约束】
- 只包含 USER 原话中直接出现的内容，不能加推断或延伸
- summary 语气不得强于 USER 原话（"不太喜欢" ≠ "强烈反感且要求永久避免"）
- summary 脱离对话也能独立成立，不含"这次""今天""当前"等时间锚
- 不能只是原话碎片，必须是完整句
- profile：每条 summary 只表达一条完整事实，绝对不合并

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前已有 profile（用于 profile 查重）】
{existing_profile or "（空）"}

【待处理对话】
{conversation}

只返回合法 JSON，不要 markdown 代码块：
{{
  "profile": [
    {{"summary": "...", "category": "personal_fact|purchase|decision|status", "happened_at": null}}
  ],
  "preference": [
    {{"summary": "..."}}
  ],
  "procedure": [
    {{"summary": "...", "tool_requirement": null, "steps": [], "rule_schema": {{"required_tools": [], "forbidden_tools": [], "mentioned_tools": []}}}}
  ]
}}"""

    async def _extract_implicit_long_term(
        self,
        *,
        conversation: str,
        existing_profile: str = "",
    ) -> dict | None:
        """窗口级隐式长期记忆 LLM 提取（只提取，不写库），与 event 并行运行。
        返回原始 dict（含 profile/preference/procedure），失败返回 None。
        写库由调用方在 event 路径确认成功后统一执行，确保幂等。
        """
        try:
            started_at = time.perf_counter()
            prompt = self._build_long_term_prompt(
                conversation=conversation,
                existing_profile=existing_profile,
            )
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=600,
            )
            text = (resp.content or "").strip()
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Memory consolidation implicit llm raw: elapsed_ms=%d chars=%d preview=%r",
                elapsed_ms,
                len(text),
                text[:300],
            )
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                return None
            return result
        except Exception as e:
            logger.warning("consolidation long_term extraction failed: %s", e)
            return None

    async def _save_implicit_long_term(
        self,
        result: dict,
        *,
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
    ) -> dict[str, int]:
        """将已提取的隐式长期记忆写入向量库。仅在 event 路径确认成功后调用。"""
        saved_counts = {
            "profile": 0,
            "preference": 0,
            "procedure": 0,
        }
        # profile
        for item in (result.get("profile") or []):
            if not isinstance(item, dict):
                continue
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue
            category = (item.get("category") or "personal_fact").strip()
            happened_at = item.get("happened_at") or None
            await self._memory_port.save_item_with_supersede(
                summary=summary,
                memory_type="profile",
                extra={
                    "category": category,
                    "scope_channel": scope_channel,
                    "scope_chat_id": scope_chat_id,
                },
                source_ref=f"{source_ref}#profile",
                happened_at=happened_at,
            )
            saved_counts["profile"] += 1
            logger.info("consolidation long_term saved: type=profile %r", summary[:60])

        # preference + procedure
        for mtype in ("preference", "procedure"):
            for item in (result.get(mtype) or []):
                if not isinstance(item, dict):
                    continue
                summary = (item.get("summary") or "").strip()
                if not summary:
                    continue
                extra: dict = {
                    "tool_requirement": item.get("tool_requirement"),
                    "steps": item.get("steps") or [],
                    "scope_channel": scope_channel,
                    "scope_chat_id": scope_chat_id,
                }
                if mtype == "procedure":
                    rule_schema = item.get("rule_schema")
                    if rule_schema and isinstance(rule_schema, dict):
                        extra["rule_schema"] = rule_schema
                await self._memory_port.save_item_with_supersede(
                    summary=summary,
                    memory_type=mtype,
                    extra=extra,
                    source_ref=f"{source_ref}#implicit",
                )
                saved_counts[mtype] += 1
                logger.info(
                    "consolidation long_term saved: type=%s %r", mtype, summary[:60]
                )
        return saved_counts

    async def consolidate(
        self,
        session,
        archive_all: bool = False,
        await_vector_store: bool = False,
    ) -> None:
        memory = self._memory_port
        profile_maint = self._profile_maint
        # 1. 先决定这次要归档哪一段消息窗口；没有新窗口就直接返回。
        window = _select_consolidation_window(
            session,
            memory_window=self._memory_window,
            consolidation_min_new_messages=self._consolidation_min_new_messages,
            archive_all=archive_all,
        )
        if archive_all:
            logger.info(
                "Memory consolidation (archive_all): %d total messages archived",
                len(session.messages),
            )
        else:
            if window is None:
                keep_count = self._memory_window // 2
                ready_count = (
                    len(session.messages) - keep_count - session.last_consolidated
                )
                if len(session.messages) <= keep_count:
                    logger.debug(
                        "Session %s: No consolidation needed (messages=%d, keep=%d)",
                        session.key,
                        len(session.messages),
                        keep_count,
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
                "Memory consolidation started: %d total, %d new to consolidate, %d keep",
                len(session.messages),
                len(window.old_messages),
                window.keep_count,
            )

        if window is None:
            return

        # 2. 把窗口消息格式化成一段对话文本，并准备好 source_ref / 现有长期记忆 / 最近 history。
        source_ref = _build_consolidation_source_ref(window)
        conversation = _format_conversation_for_consolidation(window.old_messages)
        current_memory = await asyncio.to_thread(profile_maint.read_long_term)
        history_text = ""
        if hasattr(profile_maint, "read_history"):
            history_text = _coerce_history_text(
                await asyncio.to_thread(profile_maint.read_history, 16000)
            )
        recent_history_entries = _select_recent_history_entries(
            history_text,
            limit=3,
        )
        recent_history_block = "\n".join(
            f"- {entry}" for entry in recent_history_entries
        )

        # scope 信息提前取出，隐式提取任务需要用
        scope_channel = getattr(session, "_channel", "")
        scope_chat_id = getattr(session, "_chat_id", "")

        # 隐式长期记忆提取（procedure/preference/profile）与 event 提取并行启动。
        # 只并行提取（不写库），等 event 路径确认 JSON 合法后再统一写库，确保幂等。
        implicit_task: asyncio.Task[dict | None] = asyncio.create_task(
            self._extract_implicit_long_term(
                conversation=conversation,
                existing_profile=current_memory or "",
            )
        )

        prompt = f"""你是记忆提取代理（Memory Extraction Agent）。从对话中精确提取结构化信息，返回 JSON。

## 字段说明

### 1. "history_entries" → HISTORY.md（数组，每条对应一个独立主题）
按主题拆分，每个独立话题写一条，1-2 句，以 [YYYY-MM-DD HH:MM] 开头，保留足够细节便于未来 grep 检索。
不同主题必须拆成独立条目，不得合并。若整段对话只有一个主题，返回只含一条的数组。

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

允许的 tag 只有 6 个：
- "identity"：稳定背景事实，如身份、学校/专业、长期技术方向、实习/工作经历、长期设备、长期维护项目
- "preference"：稳定偏好、禁忌、审美、游戏口味、价值取向
- "key_info"：用户明确允许保存的 key / token / id / 账号信息
- "health_long_term"：长期健康状态的一阶事实，只写长期状态，不写动态指标、基线、最近波动
- "requested_memory"：用户明确要求“长期记住”的关键内容，可比普通事实更连贯
- "correction"：对当前 MEMORY.md 现有事实的明确纠正

必须遵守：
- 只写跨对话仍有长期价值的内容
- 不写 agent 执行规则、SOP、工具调用顺序、流程规范
- 不写短期状态、近期计划、日程、课表、一次性操作
- 不写动态健康数据、实时指标、最近状态
- 不写对话过程总结
- 不写 self_insights、行为规律总结、关系演进感悟
- "requested_memory" 只能在用户明确表达“记住这个 / 写进长期记忆 / 以后要能聊到 / 希望你记住”时使用

若没有合格条目，返回空数组 []。

---

## 当前用户档案（用于查重）
{current_memory or "（空）"}

## 最近三次 consolidation event（仅用于主题延续参考）
使用原则（严格遵守）：
- 这些旧 event 只能帮助你理解“当前窗口大概在延续什么话题”，不能作为人物身份、说话人归属、关系判断或具体事实归属的直接证据。
- 若旧 event 与当前窗口原文在昵称、身份、关系、事实归属上存在冲突或不一致，必须以当前窗口原文为准。
- 不要因为旧 event 里出现了某个昵称、人设或关系描述，就在新的 history_entries 中继续沿用这些判断。
- 对 transcript / 聊天截图 / 转贴聊天场景，旧 event 绝不能用于推断“谁是当前用户、谁是对方、哪句话归谁”。
{recent_history_block or "（空）"},

## 待处理对话
{conversation}

只返回合法 JSON，不要 markdown 代码块。"""

        try:
            # 3. 调主模型把这段旧对话提炼成结构化结果。
            event_started_at = time.perf_counter()
            response = await self._provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是记忆提取代理，只返回合法 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=1024,
            )
            text = (response.content or "").strip()
            event_elapsed_ms = int((time.perf_counter() - event_started_at) * 1000)
            logger.info(
                "Memory consolidation event llm raw: elapsed_ms=%d chars=%d preview=%r",
                event_elapsed_ms,
                len(text),
                text[:300],
            )

            if not text:
                logger.warning(
                    "Memory consolidation: LLM returned empty response, skipping"
                )
                implicit_task.cancel()
                await asyncio.gather(implicit_task, return_exceptions=True)
                return
            result = _parse_consolidation_payload(text)
            if result is None:
                logger.warning(
                    "Memory consolidation: unexpected response type, skipping. Response: %r",
                    text[:200],
                )
                implicit_task.cancel()
                await asyncio.gather(implicit_task, return_exceptions=True)
                return

            # 4. 先处理 history_entries / pending_items 这两类文本产物。
            raw_entries = result.get("history_entries")
            if isinstance(raw_entries, list):
                history_entries = [
                    e for e in raw_entries if isinstance(e, str) and e.strip()
                ]
            elif result.get("history_entry"):
                history_entries = [result["history_entry"]]
            else:
                history_entries = []

            if history_entries:
                combined = "\n".join(history_entries)
                await asyncio.to_thread(
                    profile_maint.append_history_once,
                    combined,
                    source_ref=source_ref,
                    kind="history_entry",
                )

            pending_items = _format_pending_items(result.get("pending_items", []))
            if pending_items:
                appended = await asyncio.to_thread(
                    profile_maint.append_pending_once,
                    pending_items,
                    source_ref=source_ref,
                    kind="pending_items",
                )
                if appended:
                    logger.info(
                        "Memory consolidation: appended %d pending_items to PENDING",
                        len(pending_items.splitlines()),
                    )

            # 5. 再把 history_entries 写入向量记忆，供后续 retrieval 使用。
            save_tasks: list[asyncio.Task] = []
            for entry in history_entries:
                entry_source_ref = _build_entry_source_ref(source_ref, entry)
                task = asyncio.create_task(
                    self._memory_port.save_from_consolidation(
                        history_entry=entry,
                        behavior_updates=[],
                        source_ref=entry_source_ref,
                        scope_channel=scope_channel,
                        scope_chat_id=scope_chat_id,
                    )
                )
                save_tasks.append(task)
            if await_vector_store and save_tasks:
                await asyncio.gather(*save_tasks)
            if history_entries:
                logger.info(
                    "Memory consolidation: saved %d history entries to vector store",
                    len(history_entries),
                )

            # 6. 等待隐式提取完成（仅 LLM 调用，无写库），然后统一写库。
            # event 路径走到这里说明 JSON 合法，两侧提取均已成功后才提交，保证幂等。
            # 进程在此之前崩溃，last_consolidated 不更新，下次重跑同窗口。
            implicit_result = await implicit_task
            if implicit_result:
                extracted_profile = [
                    (item.get("summary") or "").strip()
                    for item in (implicit_result.get("profile") or [])
                    if isinstance(item, dict) and (item.get("summary") or "").strip()
                ]
                extracted_preference = [
                    (item.get("summary") or "").strip()
                    for item in (implicit_result.get("preference") or [])
                    if isinstance(item, dict) and (item.get("summary") or "").strip()
                ]
                extracted_procedure = [
                    (item.get("summary") or "").strip()
                    for item in (implicit_result.get("procedure") or [])
                    if isinstance(item, dict) and (item.get("summary") or "").strip()
                ]
                logger.info(
                    "Memory consolidation implicit extracted: profile=%d preference=%d procedure=%d profile_items=%s preference_items=%s procedure_items=%s",
                    len(extracted_profile),
                    len(extracted_preference),
                    len(extracted_procedure),
                    [s[:60] for s in extracted_profile],
                    [s[:60] for s in extracted_preference],
                    [s[:60] for s in extracted_procedure],
                )
                saved_counts = await self._save_implicit_long_term(
                    implicit_result,
                    source_ref=source_ref,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                )
                logger.info(
                    "Memory consolidation implicit saved: profile=%d preference=%d procedure=%d",
                    saved_counts["profile"],
                    saved_counts["preference"],
                    saved_counts["procedure"],
                )
            else:
                logger.info(
                    "Memory consolidation implicit extracted: no result"
                )

            # 7. 更新 session.last_consolidated，表示这批旧消息已经被归档过。
            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = window.consolidate_up_to
            logger.info(
                "Memory consolidation done: %d messages, last_consolidated=%d",
                len(session.messages),
                session.last_consolidated,
            )
        except Exception as e:
            logger.error("Memory consolidation failed: %s", e)


class ConsolidationRuntime:
    """
    ┌──────────────────────────────────────┐
    │ ConsolidationRuntime                 │
    ├──────────────────────────────────────┤
    │ 1. 暴露手动 consolidation 入口       │
    │ 2. 等待后台 consolidation 空闲       │
    │ 3. 复用 service 做 consolidate/save  │
    └──────────────────────────────────────┘
    """

    def __init__(
        self,
        *,
        session_manager: "SessionManager",
        scheduler: "TurnScheduler",
        consolidation: ConsolidationService,
        memory_window: int,
        consolidation_min_new_messages: int = 10,
        wait_timeout_s: float,
    ) -> None:
        self._session_manager = session_manager
        self._scheduler = scheduler
        self._consolidation = consolidation
        self._memory_window = memory_window
        self._consolidation_min_new_messages = max(
            1, int(consolidation_min_new_messages)
        )
        self._wait_timeout_s = wait_timeout_s

    async def consolidate_memory(
        self,
        session,
        *,
        archive_all: bool = False,
        await_vector_store: bool = False,
    ) -> None:
        await self._consolidation.consolidate(
            session,
            archive_all=archive_all,
            await_vector_store=await_vector_store,
        )

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        consolidate_fn: Callable[..., Awaitable[None]] | None = None,
    ) -> bool:
        # 1. 先读取真实 session，并判断当前是否真的需要 consolidation。
        session = self._session_manager.get_or_create(session_key)
        window = _select_consolidation_window(
            session,
            memory_window=self._memory_window,
            consolidation_min_new_messages=self._consolidation_min_new_messages,
            archive_all=archive_all,
        )
        if window is None:
            return False

        # 2. 若后台已在跑，同步等待那次 consolidation 完成，避免返回语义含糊的 False。
        if self._scheduler.is_consolidating(session_key):
            await self.wait_for_consolidation_idle(session_key)
            session = self._session_manager.get_or_create(session_key)
            window = _select_consolidation_window(
                session,
                memory_window=self._memory_window,
                consolidation_min_new_messages=self._consolidation_min_new_messages,
                archive_all=archive_all,
            )
            if window is None:
                return True

        # 3. 再复用现有真实 consolidation 逻辑执行一次，避免测试绕过主实现。
        if not self._scheduler.mark_manual_start(session_key):
            return False
        try:
            runner = consolidate_fn or self.consolidate_memory
            await runner(
                session,
                archive_all=archive_all,
                await_vector_store=True,
            )
            await self._session_manager.save_async(session)
            return True
        finally:
            self._scheduler.mark_manual_end(session_key)

    async def wait_for_consolidation_idle(self, session_key: str) -> None:
        deadline = time.perf_counter() + self._wait_timeout_s
        while self._scheduler.is_consolidating(session_key):
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"等待 consolidation 完成超时: session_key={session_key}"
                )
            await asyncio.sleep(0.05)

    async def consolidate_and_save(self, session: object) -> None:
        await self.consolidate_memory(session)  # type: ignore[arg-type]
        await self._session_manager.save_async(session)  # type: ignore[arg-type]
