import asyncio
import json
import json_repair
import logging
import re
from datetime import datetime
from pathlib import Path

from agent.context import ContextBuilder
from agent.memory import MemoryStore
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from agent.provider import ContentSafetyError, LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager
from proactive.presence import PresenceStore

# 安全拦截时递减历史窗口的倍率序列：全量 → 减半 → 清空
_SAFETY_RETRY_RATIOS = (1.0, 0.5, 0.0)

logger = logging.getLogger(__name__)

# 内部注入的反思提示，不应持久化到 session
_REFLECT_PROMPT = "根据上述工具执行结果，决定下一步操作。"

_RISKY_ASSERTION_TOKENS = (
    "正在进行",
    "已开始",
    "已经开始",
    "已结束",
    "已经结束",
    "一定会",
    "必然",
    "肯定会",
)

_INFERENCE_MARKERS = ("我猜", "我推断", "我觉得", "我不确定", "可能", "也许", "似乎", "大概")

_PROFILE_CLAIM_TOKENS = (
    "你之前",
    "你一直",
    "你是",
    "你有",
    "你喜欢",
    "你偏好",
    "你会对",
    "你更想",
    "你不喜欢",
    "你的库里",
    "你玩过",
)

_NATURAL_INFERENCE_PREFIXES = (
    "我觉得，",
    "我感觉，",
    "我猜，",
    "我推测，",
    "可能，",
)


def _needs_inference_tone_pass(text: str) -> bool:
    return any(token in text for token in _RISKY_ASSERTION_TOKENS)


def _truncate_for_prompt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep_head = int(limit * 0.7)
    keep_tail = max(0, limit - keep_head)
    return text[:keep_head] + "\n...（中间截断）...\n" + text[-keep_tail:]


def _build_inference_tone_prompt(numbered_lines: str, evidence_block: str) -> str:
    return f"""你是语气校准助手。你的任务是：避免把“推断”写成“事实”。

规则（严格执行）：
1. 只做“行级判定”：指出哪些行需要加“我推测/我觉得/我猜”语气前缀。
2. 若该行已有推断语气（如“我不确定/可能/我觉得/我猜”），不要选它。
3. 若【工具结果】有直接证据支持该行断言，不要选它。
4. 你不能改写原文内容，只能返回行号数组。
5. 输出严格 JSON：{{"prefix_line_numbers":[整数行号,...]}}；若无需修改，返回空数组。

{evidence_block}

【待校准回复（带行号）】：
{numbered_lines}

只输出 JSON："""


def _has_inference_marker(text: str) -> bool:
    return any(marker in text for marker in _INFERENCE_MARKERS)


def _has_profile_claim_candidate(text: str) -> bool:
    if "你" not in text:
        return False
    return any(token in text for token in _PROFILE_CLAIM_TOKENS)


def _split_line_prefix(raw: str) -> tuple[str, str]:
    leading = len(raw) - len(raw.lstrip(" "))
    head = raw[:leading]
    rest = raw[leading:]

    for bullet in ("- ", "* ", "• "):
        if rest.startswith(bullet):
            return head + bullet, rest[len(bullet) :]

    idx = 0
    while idx < len(rest) and rest[idx].isdigit():
        idx += 1
    if idx > 0 and rest[idx : idx + 2] == ". ":
        return head + rest[: idx + 2], rest[idx + 2 :]

    return head, rest


def _apply_inference_prefix_by_line_numbers(
    response: str, prefix_line_numbers: list[int]
) -> str:
    if not response.strip() or not prefix_line_numbers:
        return response

    line_no_set = {int(n) for n in prefix_line_numbers if isinstance(n, int) and n > 0}
    if not line_no_set:
        return response

    lines = response.splitlines()
    out: list[str] = []
    for i, raw in enumerate(lines, start=1):
        if i not in line_no_set:
            out.append(raw)
            continue

        if not raw.strip():
            out.append(raw)
            continue

        prefix, body = _split_line_prefix(raw)
        body_stripped = body.strip()
        if _has_inference_marker(body_stripped):
            out.append(raw)
            continue

        out.append(f"{prefix}我推测，{body}")
    return "\n".join(out)


def _apply_line_replacement_by_numbers(
    response: str,
    replace_line_numbers: list[int],
) -> str:
    """对指定行做“语气降级”，而不是整行替换。"""
    if not response.strip() or not replace_line_numbers:
        return response
    line_no_set = {
        int(n) for n in replace_line_numbers if isinstance(n, int) and n > 0
    }
    if not line_no_set:
        return response

    lines = response.splitlines()
    out: list[str] = []
    for i, raw in enumerate(lines, start=1):
        if i not in line_no_set:
            out.append(raw)
            continue
        prefix, body = _split_line_prefix(raw)
        body_stripped = body.strip()
        if _has_inference_marker(body_stripped):
            out.append(raw)
            continue
        marker = _NATURAL_INFERENCE_PREFIXES[(i - 1) % len(_NATURAL_INFERENCE_PREFIXES)]
        out.append(f"{prefix}{marker}{body}")
    return "\n".join(out)


def _build_session_window_block(session_window: list[dict], max_lines: int = 12) -> str:
    if not session_window:
        return "【会话窗口】（无）"
    lines: list[str] = []
    for m in session_window[-max_lines:]:
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content.strip()[:220]}")
    return "【会话窗口】\n" + ("\n".join(lines) if lines else "（无）")


def _build_memory_block(memory_snapshot: str) -> str:
    if not memory_snapshot.strip():
        return "【记忆】（无）"
    return "【记忆】\n" + _truncate_for_prompt(memory_snapshot, 2200)


def _build_skills_block(skills_snapshot: str) -> str:
    if not skills_snapshot.strip():
        return "【可用技能】（无）"
    return "【可用技能】\n" + _truncate_for_prompt(skills_snapshot, 3000)


def _format_tool_result_for_self_check(name: str, result: str) -> str:
    raw = str(result or "")
    if name == "shell":
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                exit_code = parsed.get("exit_code")
                output = str(parsed.get("output", "")).strip()
                if output:
                    return f"exit_code={exit_code}\n{output}"
                return f"exit_code={exit_code}"
        except Exception:
            pass
    return raw


def _build_verification_feedback_prompt(
    response: str,
    tools_used: list[str],
    tool_evidence: str,
    session_window: list[dict],
    memory_snapshot: str,
    skills_snapshot: str,
) -> str:
    tool_block = (
        f"【工具证据】\n{tool_evidence}" if tool_evidence else "【工具证据】（无）"
    )
    used = "、".join(tools_used) if tools_used else "（无）"
    session_block = _build_session_window_block(session_window)
    memory_block = _build_memory_block(memory_snapshot)
    skills_block = _build_skills_block(skills_snapshot)
    skill_catalog = _extract_skill_catalog(skills_snapshot)
    if skill_catalog:
        catalog_lines = [
            f"- {it['name']} | {it['description']} | {it['location']}"
            for it in skill_catalog[:20]
        ]
        skill_catalog_block = "【技能目录】\n" + "\n".join(catalog_lines)
    else:
        skill_catalog_block = "【技能目录】（无）"
    return f"""你是“回复风险评估器”。

目标：判断当前回复是否需要：
1) 追加一次工具验证步骤；
2) 将部分无证据断言降级成“推测语气”。

判定规则：
1. 若存在“可验证但未验证”的断言（尤其用户画像/兴趣/拥有关系），且有可行技能/工具路径，则 needs_verification=true。
2. 若存在“证据不足但不必强行验证”的断言，则 needs_tone_downgrade=true，并建议改成“我觉得/我猜/可能/感觉”等自然推测语气。
3. 若两者都不需要，两个字段都为 false。

若需要验证，feedback 要明确：
- 先尝试读取匹配技能说明（read_file）
- 再尝试调用对应工具验证（例如 shell），并尽量遵循技能中的步骤
- 验证失败时必须明确不确定，不得强断言
- 从【技能目录】中选最相关技能名放入 suggested_skill_names（可空，可多个）
- 若断言是“用户兴趣/拥有关系/画像推断”，优先选“用户状态验证类技能”（inventory/profile/history），不要选仅做资讯拉取的技能

若需要降级语气，tone_reasons 中要给出“为什么应降级”的简短理由（1-3条）。

输出严格 JSON：
{{"needs_verification":true|false,"needs_tone_downgrade":true|false,"feedback":"...","suggested_skill_names":["skill-a","skill-b"],"tone_reasons":["理由1","理由2"]}}

【已使用工具】{used}
{tool_block}
{session_block}
{memory_block}
{skills_block}
{skill_catalog_block}

【当前回复】
{response}

只输出 JSON："""


def _extract_skill_catalog(skills_snapshot: str) -> list[dict[str, str]]:
    if not skills_snapshot.strip():
        return []
    out: list[dict[str, str]] = []
    for block in re.findall(r"<skill\b.*?>.*?</skill>", skills_snapshot, flags=re.S):
        name_m = re.search(r"<name>(.*?)</name>", block, flags=re.S)
        desc_m = re.search(r"<description>(.*?)</description>", block, flags=re.S)
        loc_m = re.search(r"<location>(.*?)</location>", block, flags=re.S)
        if not name_m:
            continue
        out.append(
            {
                "name": name_m.group(1).strip(),
                "description": (desc_m.group(1).strip() if desc_m else ""),
                "location": (loc_m.group(1).strip() if loc_m else ""),
            }
        )
    return out


def _looks_like_feed_source_query(session_window: list[dict]) -> bool:
    if not session_window:
        return False
    user_texts = [
        str(m.get("content", ""))
        for m in session_window[-4:]
        if str(m.get("role", "")) == "user"
    ]
    text = "\n".join(user_texts).lower()
    if not text.strip():
        return False
    keywords = (
        "订阅",
        "信息源",
        "来源",
        "feed",
        "rss",
        "都有哪些",
        "清单",
    )
    return any(k in text for k in keywords)


class AgentLoop:
    """
    主循环：从 MessageBus 消费 InboundMessage，
    驱动 LLM + 工具调用，将结果发回 MessageBus。
    对话历史按 session_key 独立维护，格式为 OpenAI messages。
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        tools: ToolRegistry,
        session_manager: SessionManager,
        workspace: Path,
        model: str = "deepseek-chat",
        max_iterations: int = 10,
        max_tokens: int = 8192,
        memory_window: int = 40,
        presence: PresenceStore | None = None,
        light_model: str = "",
        light_provider: LLMProvider | None = None,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.session_manager = session_manager
        self.workspace = workspace
        self.context = ContextBuilder(workspace)
        self.model = model
        # light_model / light_provider 用于 self-check 等辅助推理
        # 留空则退化到主模型/主 provider
        self.light_model = light_model or model
        self.light_provider = light_provider or provider
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self._presence = presence
        self._running = False
        self._consolidating: set[str] = set()  # 正在后台压缩的 session key

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"AgentLoop 启动  model={self.model}  max_iter={self.max_iterations}"
        )
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                try:
                    response = await self._process(msg)
                    await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"处理消息出错: {e}", exc_info=True)
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"出错：{e}",
                        )
                    )
            except asyncio.TimeoutError:
                continue

    async def _run_with_safety_retry(
        self, msg: InboundMessage, session
    ) -> tuple[str, list[str], list[dict]]:
        """递减历史窗口重试，处理 LLM 安全拦截错误。

        重试顺序：全量历史 → 减半 → 无历史。
        降级成功后同步修剪 session，防止下次继续触发。
        所有窗口均失败时说明当前消息本身违规，返回友好提示。
        """
        for attempt, ratio in enumerate(_SAFETY_RETRY_RATIOS):
            window = int(self.memory_window * ratio)
            initial_messages = self.context.build_messages(
                history=session.get_history(max_messages=window),
                current_message=msg.content,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
            )
            try:
                result = await self._run_agent_loop(initial_messages)
                if attempt > 0:
                    # 降级后成功：修剪 session，避免违规内容继续存在于历史
                    logger.warning(
                        f"安全拦截后以 window={window} 成功，修剪 session 历史"
                    )
                    if window == 0:
                        session.messages.clear()
                    else:
                        session.messages = session.messages[-window:]
                    session.last_consolidated = 0
                    self.session_manager.save(session)
                return result
            except ContentSafetyError:
                if attempt < len(_SAFETY_RETRY_RATIOS) - 1:
                    next_window = int(
                        self.memory_window * _SAFETY_RETRY_RATIOS[attempt + 1]
                    )
                    logger.warning(
                        f"安全拦截 (attempt={attempt + 1})，"
                        f"缩小历史窗口重试 {window} → {next_window}"
                    )
                else:
                    logger.warning("安全拦截：所有窗口均失败，当前消息本身可能违规")
                    return "你的消息触发了安全审查，无法处理。", [], []

        return "（安全重试异常）", [], []

    def stop(self) -> None:
        self._running = False
        logger.info("AgentLoop 停止")

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """将当前会话的 channel/chat_id 注入工具，供主动推送时使用。"""
        self.tools.set_context(channel=channel, chat_id=chat_id)

    # ── 私有方法 ──────────────────────────────────────────────────

    async def _process(
        self, msg: InboundMessage, session_key: str | None = None
    ) -> OutboundMessage:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender}: {preview}")

        key = session_key or msg.session_key
        session = self.session_manager.get_or_create(key)

        # 超过记忆窗口时后台压缩（不阻塞当前消息处理）
        if (
            len(session.messages) > self.memory_window
            and key not in self._consolidating
        ):
            self._consolidating.add(key)
            asyncio.create_task(self._consolidate_memory_bg(session, key))

        self._set_tool_context(msg.channel, msg.chat_id)
        final_content, tools_used, tool_chain = await self._run_with_safety_retry(
            msg, session
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Self-Check Pass：始终验证回复中的事实声明
        session_window = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in session.messages[-self.memory_window :]
        ]
        memory_snapshot = self.context.memory.read_long_term()
        skills_snapshot = self.context.skills.build_skills_summary()
        final_content, verification_directive = await self._self_check(
            final_content,
            tool_chain,
            tools_used=tools_used,
            session_window=session_window,
            memory_snapshot=memory_snapshot,
            skills_snapshot=skills_snapshot,
        )

        if verification_directive:
            verification_feedback = str(verification_directive.get("feedback", "")).strip()
            need_verification = bool(verification_directive.get("needs_verification", False))
            need_tone_downgrade = bool(verification_directive.get("needs_tone_downgrade", False))
            suggested_skills = verification_directive.get("suggested_skill_names", [])
            if not isinstance(suggested_skills, list):
                suggested_skills = []
            suggested_skills = [str(s).strip() for s in suggested_skills if str(s).strip()]
            tone_reasons = verification_directive.get("tone_reasons", [])
            if not isinstance(tone_reasons, list):
                tone_reasons = []
            tone_reasons = [str(r).strip() for r in tone_reasons if str(r).strip()]

            logger.info(
                "Self-Check 请求追加验证回合：%s",
                verification_feedback[:200],
            )
            skill_catalog = _extract_skill_catalog(skills_snapshot)
            hint_lines: list[str] = []
            if suggested_skills:
                for name in suggested_skills[:3]:
                    hit = next((s for s in skill_catalog if s.get("name") == name), None)
                    if hit:
                        hint_lines.append(f"- {name}: {hit.get('location', '')}")
                    else:
                        hint_lines.append(f"- {name}")
            skills_hint = (
                "优先尝试以下技能：\n" + "\n".join(hint_lines)
                if hint_lines
                else "若技能目录中有匹配技能，请优先 read_file 读取其 SKILL.md。"
            )
            tone_hint = ""
            if need_tone_downgrade:
                reason_block = "\n".join(f"- {r}" for r in tone_reasons[:3]) or "- 存在证据不足的强断言"
                tone_hint = (
                    "【语气降级建议】\n"
                    "以下内容建议改为推测语气（我觉得/我猜/可能/感觉），不要当成确定事实：\n"
                    f"{reason_block}\n"
                )

            action_hint = ""
            if need_verification:
                action_hint += (
                    "请先尝试调用工具完成验证，再给最终答复。"
                    "尽量按技能说明中的步骤执行，不要自创无关验证路径。"
                    "只有拿到有效验证结果时，才能写“已验证/验证完成”；否则必须明确说明不确定。"
                )
            if need_tone_downgrade:
                action_hint += (
                    "若当前证据不足，不要硬断言；请自然改成推测语气，避免新增事实。"
                )
            if not action_hint:
                action_hint = "请根据 self-check 建议优化最终答复，不要新增事实。"
            retry_messages = self.context.build_messages(
                history=session.get_history(max_messages=self.memory_window),
                current_message=msg.content,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
            )
            retry_messages.append({"role": "assistant", "content": final_content})
            retry_messages.append(
                {
                    "role": "user",
                    "content": (
                        "【内部质量反馈（不要原样对用户复述）】\n"
                        f"{verification_feedback}\n"
                        f"{skills_hint}\n"
                        f"{tone_hint}"
                        f"{action_hint}\n\n"
                        "【最终输出要求（面向用户）】\n"
                        "1) 直接回答用户原问题，不要复盘内部流程。\n"
                        "2) 禁止出现“验证完成/最终修正版/作废/第三步/去伪存真”等内部措辞。\n"
                        "3) 不要长篇自我纠错叙事；只保留结论、依据与必要的不确定性。\n"
                        "4) 语气自然口语，简洁，优先给用户可用结论。"
                    ),
                }
            )

            retry_content, retry_tools_used, retry_tool_chain = await self._run_agent_loop(
                retry_messages
            )
            final_content = retry_content
            if retry_tools_used:
                tools_used.extend(retry_tools_used)
            if retry_tool_chain:
                tool_chain.extend(retry_tool_chain)

            # 验证回合后再做一次收尾核查，但不再触发额外验证回合，避免循环。
            final_content, _ = await self._self_check(
                final_content,
                tool_chain,
                tools_used=tools_used,
                session_window=session_window,
                memory_snapshot=memory_snapshot,
                skills_snapshot=skills_snapshot,
                allow_verification_feedback=False,
            )

        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info(f"Response to {msg.channel}:{msg.sender}: {preview}")

        if self._presence:
            self._presence.record_user_message(key)
        session.add_message("user", msg.content)
        session.add_message(
            "assistant",
            final_content,
            tools_used=tools_used if tools_used else None,
            tool_chain=tool_chain if tool_chain else None,
        )
        self.session_manager.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata={
                **(
                    msg.metadata or {}
                ),  # Pass through for channel-specific needs (e.g. Slack thread_ts)
                "tools_used": tools_used,
                "tool_chain": tool_chain,
            },
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
    ) -> tuple[str, list[str], list[dict]]:
        """迭代调用 LLM，直到无工具调用或达到上限。返回 (final_content, tools_used, tool_chain)

        tool_chain 是按迭代分组的工具调用记录，每个元素：
          {"text": str|None, "calls": [{"call_id", "name", "arguments", "result"}]}
        """
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []

        for iteration in range(self.max_iterations):
            logger.debug(f"LLM 调用  iteration={iteration + 1}")
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_schemas(),
                model=self.model,
                max_tokens=self.max_tokens,
            )

            if response.tool_calls:
                logger.info(
                    f"LLM 请求调用 {len(response.tool_calls)} 个工具: "
                    f"{[tc.name for tc in response.tool_calls]}"
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(
                                        tc.arguments, ensure_ascii=False
                                    ),
                                },
                            }
                            for tc in response.tool_calls
                        ],
                    }
                )
                iter_calls: list[dict] = []
                for tc in response.tool_calls:
                    tools_used.append(tc.name)
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info(f"  → 工具 {tc.name}  参数: {args_str[:120]}")
                    result = await self.tools.execute(tc.name, tc.arguments)
                    result_preview = result[:80] + "..." if len(result) > 80 else result
                    logger.info(f"  ← 工具 {tc.name}  结果: {result_preview!r}")
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
                    iter_calls.append(
                        {
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": result,
                        }
                    )
                tool_chain.append({"text": response.content, "calls": iter_calls})

                # 工具结果注入后，提示 LLM 反思并决定下一步
                messages.append({"role": "user", "content": _REFLECT_PROMPT})
            else:
                logger.info(f"LLM 返回最终回复  iteration={iteration + 1}")
                messages.append({"role": "assistant", "content": response.content})
                return response.content or "（无响应）", tools_used, tool_chain

        logger.warning(f"已达到最大迭代次数 {self.max_iterations}")
        return "（已达到最大迭代次数）", tools_used, tool_chain

    async def _self_check(
        self,
        response: str,
        tool_chain: list[dict],
        tools_used: list[str],
        session_window: list[dict],
        memory_snapshot: str,
        skills_snapshot: str,
        allow_verification_feedback: bool = True,
    ) -> tuple[str, dict | None]:
        """Self-Check Pass：只做风险评估与建议，不直接改写回复正文。"""
        _MAX_RESULT = 900
        lines: list[str] = []
        for iter_group in tool_chain:
            for call in iter_group.get("calls", []):
                name = call.get("name", "tool")
                result = _format_tool_result_for_self_check(
                    name, str(call.get("result", ""))
                )
                if len(result) > _MAX_RESULT:
                    result = result[:_MAX_RESULT] + "…（已截断）"
                lines.append(f"[{name}]\n{result}")
        tool_evidence = "\n\n".join(lines)

        if tool_evidence:
            evidence_block = f"【工具结果】（可信事实来源）：\n{tool_evidence}"
            no_tool_rule = ""
        else:
            evidence_block = "【工具结果】：（本次无工具调用，无任何外部事实来源）"
            no_tool_rule = (
                "\n**特别规则（无工具调用时）**：回复中出现的具体数字、"
                "产品名称/型号、价格、版本号、发布状态、事件结论、人物动态等，"
                '凡是需要查证才能确认的，一律改为"我不确定"或删除。'
                "无需查证的常识不改。"
            )

        session_block = _build_session_window_block(session_window)
        memory_block = _build_memory_block(memory_snapshot)
        skills_block = _build_skills_block(skills_snapshot)

        _ = no_tool_rule, evidence_block, session_block, memory_block, skills_block
        candidate = response
        verification_directive = None
        if allow_verification_feedback:
            verification_directive = await self._verification_feedback_check(
                candidate,
                tools_used=tools_used,
                tool_evidence=tool_evidence,
                session_window=session_window,
                memory_snapshot=memory_snapshot,
                skills_snapshot=skills_snapshot,
            )
        return candidate, verification_directive

    async def _verification_feedback_check(
        self,
        response: str,
        tools_used: list[str],
        tool_evidence: str,
        session_window: list[dict],
        memory_snapshot: str,
        skills_snapshot: str,
    ) -> dict | None:
        """让 self-check 告诉主循环：是否应追加一次“先验证再回答”的回合。"""
        if not response.strip():
            return None
        prompt = _build_verification_feedback_prompt(
            response=response,
            tools_used=tools_used,
            tool_evidence=tool_evidence,
            session_window=session_window,
            memory_snapshot=memory_snapshot,
            skills_snapshot=skills_snapshot,
        )
        try:
            judged = await self.light_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self.light_model,
                max_tokens=512,
            )
            text = (judged.content or "").strip()
            if not text:
                return None
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json_repair.loads(text)
            if not isinstance(parsed, dict):
                return None
            need = bool(parsed.get("needs_verification"))
            need_tone_downgrade = bool(parsed.get("needs_tone_downgrade"))
            feedback = str(parsed.get("feedback", "")).strip()
            suggested = parsed.get("suggested_skill_names", [])
            if not isinstance(suggested, list):
                suggested = []
            suggested_names = [
                str(it).strip() for it in suggested if isinstance(it, (str, int, float))
            ]
            tone_reasons = parsed.get("tone_reasons", [])
            if not isinstance(tone_reasons, list):
                tone_reasons = []
            tone_reasons = [
                str(it).strip() for it in tone_reasons if isinstance(it, (str, int, float))
            ]
            if not need and not need_tone_downgrade:
                return None

            if _looks_like_feed_source_query(session_window):
                # 订阅来源查询优先走 feed 工具，避免被 RSSHub 路由类技能带偏。
                suggested_names = [
                    s for s in suggested_names if "rsshub-route-finder" not in s.lower()
                ]
                feed_guidance = (
                    "这是订阅来源/清单问题：优先调用 "
                    "feed_manage(action=list) 获取完整订阅列表，必要时再用 "
                    "feed_query(action=summary|catalog) 交叉核对；"
                    "不要先读取 RSSHub 路由技能。"
                )
                feedback = f"{feed_guidance} {feedback}".strip()
                need = True

            if not feedback:
                if need:
                    feedback = (
                        "检测到证据不足但可验证的断言。"
                        "请先 read_file 读取相关 SKILL.md，再调用对应工具验证后回答；"
                        "若验证失败，明确说不确定。"
                    )
                else:
                    feedback = "检测到证据不足的强断言，建议降级为自然推测语气。"
            return {
                "needs_verification": need,
                "needs_tone_downgrade": need_tone_downgrade,
                "feedback": feedback[:400],
                "suggested_skill_names": suggested_names[:4],
                "tone_reasons": tone_reasons[:4],
            }
        except Exception as e:
            logger.warning(f"Verification feedback check 失败，忽略追加验证: {e}")
            return None

    async def _inference_tone_check(self, response: str, tool_evidence: str) -> str:
        """LLM 语气校准：将无证据断言降级为显式推断语气。"""
        if not response.strip():
            return response
        if not _needs_inference_tone_pass(response):
            return response

        if tool_evidence:
            evidence_block = f"【工具结果】（可信事实来源）：\n{tool_evidence}"
        else:
            evidence_block = "【工具结果】：（本次无工具调用，无任何外部事实来源）"

        lines = response.splitlines()
        numbered_lines = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))
        prompt = _build_inference_tone_prompt(numbered_lines, evidence_block)
        try:
            tuned = await self.light_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self.light_model,
                max_tokens=self.max_tokens,
            )
            content = (tuned.content or "").strip()
            if not content:
                return response
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json_repair.loads(content)
            nums = parsed.get("prefix_line_numbers", []) if isinstance(parsed, dict) else []
            if isinstance(nums, list):
                return _apply_inference_prefix_by_line_numbers(response, nums)
        except Exception as e:
            logger.warning(f"Inference tone check 失败，返回原始回复: {e}")
        return response

    async def _profile_grounding_check(
        self,
        response: str,
        tool_evidence: str,
        session_window: list[dict],
        memory_snapshot: str,
        skills_snapshot: str,
    ) -> str:
        """LLM 用户画像落地校验：阻止无依据的“你如何/你之前如何”断言。"""
        if not response.strip():
            return response
        if not _has_profile_claim_candidate(response):
            return response

        if tool_evidence:
            tool_block = f"【工具结果】\n{tool_evidence}"
        else:
            tool_block = "【工具结果】（无）"

        history_block = _build_session_window_block(session_window)
        memory_block = _build_memory_block(memory_snapshot)
        skills_block = _build_skills_block(skills_snapshot)

        lines = response.splitlines()
        numbered_lines = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))
        prompt = f"""你是用户画像事实校验器。

任务：识别【待校验回复】中“关于用户本人”的断言行（例如“你之前提过X”“你喜欢Y”“你有Z”）。
若该断言在【会话窗口】【记忆】【工具结果】中没有直接依据，标记该行需要“降级为推测语气”。
如果用户偏好有明确限定词（如“开放世界RPG”），而回复把更宽泛类别（如“RPG”）当成完全匹配，也视为需要替换。
若【可用技能】中存在可用于验证该断言的技能，而回复未验证就下结论，也视为需要替换。

输出要求：
- 只输出 JSON：{{"replace_line_numbers":[整数行号,...]}}
- 若都可被证据支持，则输出空数组。
- 不要改写原文，不要输出解释。
- 你标出的行会被系统自动改成“自然推测语气”，不要把所有行都标上。

{history_block}

{memory_block}

{tool_block}
{skills_block}

【待校验回复（带行号）】
{numbered_lines}
"""
        try:
            checked = await self.light_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self.light_model,
                max_tokens=self.max_tokens,
            )
            text = (checked.content or "").strip()
            if not text:
                return response
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json_repair.loads(text)
            nums = (
                parsed.get("replace_line_numbers", [])
                if isinstance(parsed, dict)
                else []
            )
            if isinstance(nums, list):
                return _apply_line_replacement_by_numbers(response, nums)
        except Exception as e:
            logger.warning(f"Profile grounding check 失败，返回原始回复: {e}")
        return response

    async def _consolidate_memory_bg(self, session, key: str) -> None:
        """后台异步压缩，完成后持久化 last_consolidated 并释放锁。"""
        try:
            await self._consolidate_memory(session)
            self.session_manager.save(session)
        finally:
            self._consolidating.discard(key)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """

        memory = MemoryStore(self.workspace)
        if archive_all:
            old_messages = list(session.messages)
            keep_count = 0
            logger.info(
                f"Memory consolidation (archive_all): {len(session.messages)} total messages archived"
            )
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(
                    f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})"
                )
                return
            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(
                    f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})"
                )
                return
            old_messages = session.messages[session.last_consolidated : -keep_count]
            if not old_messages:
                return
            logger.info(
                f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep"
            )

        # 以下逻辑对 archive_all 和普通压缩均适用
        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}: {m['content']}"
            )
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()
        current_questions = memory.read_questions()

        prompt = f"""你是记忆提取代理（Memory Extraction Agent）。从对话中提取需要长期记住的新事实，返回 JSON。

JSON 包含以下三个键：

1. "history_entry"：2-5 句话的事件摘要，以 [YYYY-MM-DD HH:MM] 开头，保留足够细节便于未来 grep 检索。

2. "new_facts"：本次对话中出现的**新持久化事实**，格式为带分类标注的 bullet 列表。
   规则：
   - 只写现有档案中**没有**的新信息（对照下方档案查重）
   - 只写持久性事实：姓名/设备/账号/偏好/技能/项目经历/游戏数据等
   - 不写一次性操作记录（"帮用户执行了X"、"已完成Y"）
   - 不写对话本身的过程描述
   - 若无新事实，返回空字符串 ""
   - 格式示例：
     - [用户画像] 用户确认正在准备秋招
     - [硬件与环境] 新增显示器：Dell U2723D

3. "answered_question_indices"：从待了解问题列表中，本次对话**已得到解答**的问题序号列表（1-based int）。若无则返回 []。

## 当前用户档案（用于查重，不要重复已有内容）
{current_memory or "（空）"}

## 待了解的问题
{current_questions or "（无）"}

## 待处理对话
{conversation}

只返回合法 JSON，不要 markdown 代码块。"""

        try:
            response = await self.provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是记忆提取代理，只返回合法 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self.model,
                max_tokens=1024,
            )
            text = (response.content or "").strip()

            if not text:
                logger.warning(
                    "Memory consolidation: LLM returned empty response, skipping"
                )
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(
                    f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}"
                )
                return

            if "history_entry" in result:
                memory.append_history(result["history_entry"])
            # 增量事实写入 PENDING.md，不触碰 MEMORY.md
            # MEMORY.md 由夜间 MemoryOptimizer 统一合并维护
            new_facts = result.get("new_facts", "")
            if new_facts and isinstance(new_facts, str) and new_facts.strip():
                memory.append_pending(new_facts)
                logger.info(
                    f"Memory consolidation: appended {len(new_facts.splitlines())} new facts to PENDING"
                )
            answered = result.get("answered_question_indices", [])
            if answered and isinstance(answered, list):
                indices = [
                    int(i) for i in answered if str(i).isdigit() or isinstance(i, int)
                ]
                if indices:
                    memory.remove_questions_by_indices(indices)
                    logger.info(
                        f"Memory consolidation: removed answered questions {indices}"
                    )

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(
                f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}"
            )
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).

        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender="user",
            chat_id=chat_id,
            content=content,
        )

        response = await self._process(msg, session_key=session_key)
        return response.content if response else ""
