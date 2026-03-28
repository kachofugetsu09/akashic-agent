import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.llm_json import load_json_object_loose

logger = logging.getLogger("agent.loop")

if TYPE_CHECKING:
    from agent.provider import LLMProvider
    from core.memory.port import MemoryPort
    from memory2.profile_extractor import ProfileFactExtractor

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
    if not old_messages:
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


class ConsolidationService:
    def __init__(
        self,
        *,
        memory_port: "MemoryPort",
        provider: "LLMProvider",
        model: str,
        memory_window: int,
        profile_extractor: "ProfileFactExtractor | None" = None,
    ) -> None:
        self._memory_port = memory_port
        self._provider = provider
        self._model = model
        self._memory_window = memory_window
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

    async def consolidate(
        self,
        session,
        archive_all: bool = False,
        await_vector_store: bool = False,
    ) -> None:
        memory = self._memory_port
        # 1. 先决定这次要归档哪一段消息窗口；没有新窗口就直接返回。
        window = _select_consolidation_window(
            session,
            memory_window=self._memory_window,
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
                if len(session.messages) <= keep_count:
                    logger.debug(
                        "Session %s: No consolidation needed (messages=%d, keep=%d)",
                        session.key,
                        len(session.messages),
                        keep_count,
                    )
                else:
                    logger.debug(
                        "Session %s: No new messages to consolidate (last_consolidated=%d, total=%d)",
                        session.key,
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
        current_memory = await asyncio.to_thread(memory.read_long_term)
        recent_history_entries = _select_recent_history_entries(
            await asyncio.to_thread(memory.read_history, 16000),
            limit=3,
        )
        recent_history_block = "\n".join(
            f"- {entry}" for entry in recent_history_entries
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

            if not text:
                logger.warning(
                    "Memory consolidation: LLM returned empty response, skipping"
                )
                return
            result = _parse_consolidation_payload(text)
            if result is None:
                logger.warning(
                    "Memory consolidation: unexpected response type, skipping. Response: %r",
                    text[:200],
                )
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
                    memory.append_history_once,
                    combined,
                    source_ref=source_ref,
                    kind="history_entry",
                )

            pending_items = _format_pending_items(result.get("pending_items", []))
            if pending_items:
                appended = await asyncio.to_thread(
                    memory.append_pending_once,
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
            scope_channel = getattr(session, "_channel", "")
            scope_chat_id = getattr(session, "_chat_id", "")
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

            # 6. 最后按需补做 profile facts 提取，这条链和 history_entries 分开。
            profile_task = None
            if self._profile_extractor and conversation.strip():
                profile_task = asyncio.create_task(
                    self._extract_and_save_profile_facts(
                        extractor=self._profile_extractor,
                        conversation=conversation,
                        existing_profile=current_memory or "",
                        source_ref=source_ref,
                        scope_channel=scope_channel,
                        scope_chat_id=scope_chat_id,
                    )
                )
            if await_vector_store and profile_task is not None:
                await profile_task

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
