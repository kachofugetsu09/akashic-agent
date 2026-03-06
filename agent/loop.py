import asyncio
import json
import json_repair
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.context import ContextBuilder
from bus.events import InboundMessage, OutboundMessage
from bus.processing import ProcessingState
from bus.queue import MessageBus
from agent.provider import ContentSafetyError, ContextLengthError, LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager
from proactive.presence import PresenceStore
from memory2.post_response_worker import PostResponseMemoryWorker

if TYPE_CHECKING:
    from core.memory.port import MemoryPort

# 安全拦截时递减历史窗口的倍率序列：全量 → 减半 → 清空
_SAFETY_RETRY_RATIOS = (1.0, 0.5, 0.0)
# 单条工具结果的字符上限，防止大文件/长网页撑爆当轮上下文
_MAX_TOOL_RESULT_CHARS = 100_000
_TOOL_LOOP_REPEAT_LIMIT = 3  # 连续同签名工具调用达到该次数时判定循环
_SUMMARY_MAX_TOKENS = 512

logger = logging.getLogger(__name__)

# 内部注入的反思提示，不应持久化到 session
_REFLECT_PROMPT = """根据上述工具执行结果，决定下一步操作。

【自检，无需在回复中说明，只用于内部决策】
1. 用户原始消息是否对 **agent（你）** 明确表达行为偏好/操作规范（要求 agent 以后遇到 X 就做 Y）？注意：描述第三方行为规律、用户自述观察/印象，均不属于此类。用户明确说「记住/以后/下次」时，可调用 memorize 做即时确认；其余隐式偏好由后台流程自动提取。
2. 当前任务是否有匹配的技能尚未读取 SKILL.md？若有，必须先 read_file 读取完整指令再继续。
2. 即将输出的结论是否有本轮工具返回的事实支撑？无支撑时允许合理推断，但必须显式标注“我推测/可能/更像是”，并保持可追溯到本轮事实；禁止把推断写成事实。
3. 涉及用户状态/数据/画像的陈述，若未经本轮工具验证，禁止以事实语气输出。
4. 禁止把历史会话中的旧工具结果冒充本轮实测——若用户问的是"现在/当前"的数据，必须本轮重新调用工具。
5. 涉及时间判断（现在/当前/最新/是否已发生）时，统一以本轮 request_time 为时间锚点；若证据只有计划时间而无实际发生证据，不得断言“已经发生”。
6. 若用户问“动机/来源/身世/含义”这类解释问题，可结合事实做联想，但最终要区分“已证据事实”和“待用户确认的推测”。"""

# 每轮对话开始前注入的初始自检提示，不应持久化到 session
_PRE_FLIGHT_PROMPT = """【回复前必须完成以下自检，无需在回复中说明】
0. 【SOP 优先级最高，强制执行】系统 prompt 中已通过向量检索注入了本轮相关 SOP 内容（见”【强制约束】”和”【流程规范】”段），直接参照执行，**无需再 read_file 读取 SOP 文件**。仅当用户明确要求新增/修改 SOP 时，才需要 read_file 读取对应文件。
1. 用户是否在对 **agent（你）** 表达行为偏好或操作规范——即明确要求 agent 以后按某种方式行动？判断标志：**主语必须是"你/agent"**，且含「记住/以后/下次/每次/你要/你最好/帮我…」等显式指令词。以下情况**不触发 memorize**：① 用户在描述第三方（大厂/竞品/他人）的行为规律；② 用户在陈述自己的观察/印象（"我印象里…""我觉得…"）；③ 纯粹的讨论/提问/知识分享。用户明确说「记住/以后/下次」时，可调用 memorize 工具即时确认；隐式偏好由系统后台自动提取，无需手动 memorize。
1a. 用户是否在指出 agent 某个**现有行为/流程有误或需要废弃**（如"你之前X是错的/忘掉之前的X/X不对"）？若是，必须：① 承认该问题；② 主动追问正确的做法是什么（"那正确的流程/方式是？"）。后台会自动清除旧的错误记忆，但需要用户提供新的正确方式才能写入替代规则。
2. 用户是否要求执行某项操作，且该操作与 # Skills 中某个技能的描述明确匹配？若是，禁止在未调用工具的情况下直接回答——必须先 read_file 读取对应 SKILL.md，再按指令执行工具，最后基于工具返回结果作答。（注意：用户只是询问技能列表/能力范围，不触发此规则，直接根据摘要回答即可。）
3. 用户问的内容是否需要实时/当前数据（订阅列表、天气、最新动态、用户状态等）？若需要，同样禁止凭记忆直接回答，必须本轮调用工具获取。
4. 遇到”现在/当前/最新/今天/是否已发生”等时间敏感判断，先以 request_time 锚定时间，再给结论；若缺少可核验事实，明确说不确定。
5. 回答允许做合理联想，但必须显式标注推测语气，不得冒充事实；必要时给出”待确认”。
6. 确认以上规则均满足后，才允许输出最终回复。"""

_INCOMPLETE_SUMMARY_PROMPT = """当前任务未在预算内完成，请直接输出给用户的中文收尾说明（不要提及系统/工具内部细节）。
必须包含三点：
1) 已完成到哪一步（基于当前上下文的事实）；
2) 目前还缺什么信息或步骤；
3) 下一步你会怎么继续。
禁止输出“已达到最大迭代次数”这类模板句；不要输出 JSON。"""


def _tool_call_signature(tool_calls) -> str:
    """生成本轮 tool_calls 的稳定签名，用于检测循环调用。"""
    parts: list[str] = []
    for tc in tool_calls:
        args = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
        parts.append(f"{tc.name}:{args}")
    return "|".join(parts)


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
        processing_state: ProcessingState | None = None,
        memorizer=None,
        retriever=None,
        disable_full_memory: bool = False,
        memory_port: "MemoryPort | None" = None,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.session_manager = session_manager
        self.workspace = workspace
        self.model = model
        # light_model / light_provider 保留接口兼容，不再用于 self-check
        self.light_model = light_model or model
        self.light_provider = light_provider or provider
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self._presence = presence
        self._running = False
        self._consolidating: set[str] = set()  # 正在后台压缩的 session key
        self._processing_state = processing_state
        self._disable_full_memory = disable_full_memory

        if memorizer and retriever:
            self._post_mem_worker = PostResponseMemoryWorker(
                memorizer=memorizer,
                retriever=retriever,
                light_provider=light_provider or provider,
                light_model=light_model or model,
            )
        else:
            self._post_mem_worker = None

        # 1. Build or accept a unified MemoryPort
        if memory_port is not None:
            self._memory_port = memory_port
        else:
            from agent.memory import MemoryStore
            from core.memory.port import DefaultMemoryPort

            self._memory_port: "MemoryPort" = DefaultMemoryPort(
                MemoryStore(workspace),
                memorizer=memorizer,
                retriever=retriever,
            )

        # 2. Wire ContextBuilder with the unified memory port
        self.context = ContextBuilder(workspace, memory=self._memory_port)

        # 3. Keep legacy references for callers that may still use them directly
        self._memorizer = memorizer
        self._retriever = retriever

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
        self,
        msg: InboundMessage,
        session,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
    ) -> tuple[str, list[str], list[dict]]:
        """递减历史窗口重试，处理 LLM 安全拦截错误。

        重试顺序：全量历史 → 减半 → 无历史。
        降级成功后同步修剪 session，防止下次继续触发。
        所有窗口均失败时说明当前消息本身违规，返回友好提示。
        """
        source_history = base_history or session.get_history(
            max_messages=self.memory_window
        )
        total_history = len(source_history)

        for attempt, ratio in enumerate(_SAFETY_RETRY_RATIOS):
            window = int(total_history * ratio)
            if window <= 0:
                history_for_attempt: list[dict] = []
            elif window >= total_history:
                history_for_attempt = source_history
            else:
                history_for_attempt = source_history[-window:]
            initial_messages = self.context.build_messages(
                history=history_for_attempt,
                current_message=msg.content,
                media=msg.media if msg.media else None,
                skill_names=skill_names,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
                retrieved_memory_block=retrieved_memory_block,
                disable_full_memory=self._disable_full_memory,
            )
            try:
                result = await self._run_agent_loop(
                    initial_messages,
                    request_time=msg.timestamp,
                )
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
                    # 持有写锁全量重写，防止与后台 consolidation save_async 竞争
                    await self.session_manager.save_async(session)
                return result
            except ContentSafetyError:
                if attempt < len(_SAFETY_RETRY_RATIOS) - 1:
                    next_window = int(total_history * _SAFETY_RETRY_RATIOS[attempt + 1])
                    logger.warning(
                        f"安全拦截 (attempt={attempt + 1})，"
                        f"缩小历史窗口重试 {window} → {next_window}"
                    )
                else:
                    logger.warning("安全拦截：所有窗口均失败，当前消息本身可能违规")
                    return "你的消息触发了安全审查，无法处理。", [], []
            except ContextLengthError:
                if attempt < len(_SAFETY_RETRY_RATIOS) - 1:
                    next_window = int(total_history * _SAFETY_RETRY_RATIOS[attempt + 1])
                    logger.warning(
                        f"上下文超长 (attempt={attempt + 1})，"
                        f"缩小历史窗口重试 {window} → {next_window}"
                    )
                else:
                    logger.warning("上下文超长：所有窗口均失败，清空历史后仍超长")
                    return "上下文过长无法处理，请尝试新建对话。", [], []

        return "（安全重试异常）", [], []

    @property
    def processing_state(self) -> ProcessingState | None:
        """暴露被动处理信号，供 ProactiveLoop 注入 passive_busy_fn。"""
        return self._processing_state

    def stop(self) -> None:
        self._running = False
        logger.info("AgentLoop 停止")

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """将当前会话的 channel/chat_id 注入工具，供主动推送时使用。"""
        self.tools.set_context(channel=channel, chat_id=chat_id)

    # ── 私有方法 ──────────────────────────────────────────────────

    def _collect_skill_mentions(self, user_message: str) -> list[str]:
        """解析用户消息中 $skill-name 的显式提及，返回命中的技能名列表。"""
        raw_names = re.findall(r"\$([a-zA-Z0-9_-]+)", user_message)
        if not raw_names:
            return []
        available = {
            s["name"] for s in self.context.skills.list_skills(filter_unavailable=False)
        }
        seen: set[str] = set()
        result: list[str] = []
        for name in raw_names:
            if name in available and name not in seen:
                seen.add(name)
                result.append(name)
        return result

    # 单条消息处理的总超时（秒）。覆盖工具挂起、LLM 超时累积等极端场景。
    _MESSAGE_TIMEOUT_S: float = 600.0

    async def _process(
        self, msg: InboundMessage, session_key: str | None = None
    ) -> OutboundMessage:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender}: {preview}")

        key = session_key or msg.session_key
        if self._processing_state:
            self._processing_state.enter(key)
        try:
            return await asyncio.wait_for(
                self._process_inner(msg, key),
                timeout=self._MESSAGE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"消息处理超时 ({self._MESSAGE_TIMEOUT_S}s)  "
                f"channel={msg.channel} chat_id={msg.chat_id}"
            )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="（处理超时，请重试）",
            )
        finally:
            if self._processing_state:
                self._processing_state.exit(key)

    async def _process_inner(self, msg: InboundMessage, key: str) -> OutboundMessage:
        session = self.session_manager.get_or_create(key)

        # 超过记忆窗口时后台压缩（不阻塞当前消息处理）
        if (
            len(session.messages) > self.memory_window
            and key not in self._consolidating
        ):
            self._consolidating.add(key)
            asyncio.create_task(self._consolidate_memory_bg(session, key))

        # 解析 $skill 语法，命中时直接注入完整 SKILL.md（Codex 风格：事前注入，而非事后检测）
        skill_mentions = self._collect_skill_mentions(msg.content)
        if skill_mentions:
            logger.info(f"检测到 $skill 提及，直接注入完整内容: {skill_mentions}")

        main_history = session.get_history(max_messages=self.memory_window)

        # memory v2 检索
        retrieved_block = ""
        try:
            items = await self._memory_port.retrieve_related(msg.content)
            retrieved_block = self._memory_port.format_injection_block(items)
            if retrieved_block:
                logger.info(f"memory2 retrieve: {len(items)} 条命中，注入检索块")
        except Exception as e:
            logger.warning(f"memory2 retrieve 失败，跳过: {e}")

        self._set_tool_context(msg.channel, msg.chat_id)
        final_content, tools_used, tool_chain = await self._run_with_safety_retry(
            msg,
            session,
            skill_names=skill_mentions or None,
            base_history=main_history,
            retrieved_memory_block=retrieved_block,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info(f"Response to {msg.channel}:{msg.sender}: {preview}")

        if self._presence:
            self._presence.record_user_message(key)
        session.add_message("user", msg.content, media=msg.media if msg.media else None)
        session.add_message(
            "assistant",
            final_content,
            tools_used=tools_used if tools_used else None,
            tool_chain=tool_chain if tool_chain else None,
        )
        # 普通对话只追加 2 条消息，避免全量重写阻塞事件循环
        await self.session_manager.append_messages(session, session.messages[-2:])

        if self._post_mem_worker:
            asyncio.create_task(
                self._post_mem_worker.run(
                    user_msg=msg.content,
                    agent_response=final_content,
                    tool_chain=tool_chain,
                    source_ref=f"{key}@post_response",
                )
            )

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
        request_time: datetime | None = None,
    ) -> tuple[str, list[str], list[dict]]:
        """迭代调用 LLM，直到无工具调用或达到上限。返回 (final_content, tools_used, tool_chain)

        tool_chain 是按迭代分组的工具调用记录，每个元素：
          {"text": str|None, "calls": [{"call_id", "name", "arguments", "result"}]}
        """
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []
        last_tool_signature = ""
        repeat_count = 0

        # 第一轮调用前注入预检提示，让 LLM 在未调用任何工具时也做技能匹配自检
        preflight_prompt = (
            f"【本轮时间锚点】{self._format_request_time_anchor(request_time)}\n"
            "所有时间相关判断必须与该锚点一致；无法验证时必须明确不确定。\n\n"
            + _PRE_FLIGHT_PROMPT
        )
        messages = messages + [{"role": "user", "content": preflight_prompt}]

        for iteration in range(self.max_iterations):
            logger.debug(f"LLM 调用  iteration={iteration + 1}")
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_schemas(),
                model=self.model,
                max_tokens=self.max_tokens,
                tool_choice="auto",
            )

            if response.tool_calls:
                signature = _tool_call_signature(response.tool_calls)
                if signature and signature == last_tool_signature:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_tool_signature = signature

                if repeat_count >= _TOOL_LOOP_REPEAT_LIMIT:
                    logger.warning(
                        "检测到工具调用循环 signature=%s repeat=%d，提前收尾",
                        signature[:160],
                        repeat_count,
                    )
                    summary = await self._summarize_incomplete_progress(
                        messages,
                        reason="tool_call_loop",
                        iteration=iteration + 1,
                        tools_used=tools_used,
                    )
                    return summary, tools_used, tool_chain

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
        summary = await self._summarize_incomplete_progress(
            messages,
            reason="max_iterations",
            iteration=self.max_iterations,
            tools_used=tools_used,
        )
        return summary, tools_used, tool_chain

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict],
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
    ) -> str:
        """预算耗尽/循环时额外生成进度总结，避免模板化失败回复。"""
        summary_prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n"
            f"[已调用工具] {', '.join(tools_used[-8:]) if tools_used else '无'}\n\n"
            + _INCOMPLETE_SUMMARY_PROMPT
        )
        try:
            resp = await self.provider.chat(
                messages=messages + [{"role": "user", "content": summary_prompt}],
                tools=[],
                model=self.model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self.max_tokens),
            )
            text = (resp.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("生成预算收尾总结失败: %s", e)

        done = f"已尝试 {iteration} 轮，调用工具 {len(tools_used)} 次。"
        return (
            f"这次任务还没完全收束。{done}"
            "我先停在当前进度，后续会继续补齐缺失信息并给你最终结论。"
        )

    @staticmethod
    def _format_request_time_anchor(ts: datetime | None) -> str:
        if ts is None:
            ts = datetime.now().astimezone()
        elif ts.tzinfo is None:
            ts = ts.astimezone()
        return f"request_time={ts.isoformat()} ({ts.strftime('%Y-%m-%d %H:%M:%S %Z')})"

    async def _consolidate_memory_bg(self, session, key: str) -> None:
        """后台异步压缩，完成后持久化 last_consolidated 并释放锁。"""
        try:
            await self._consolidate_memory(session)
            # consolidation 更新了 last_consolidated，需全量重写 metadata
            await self.session_manager.save_async(session)
        finally:
            self._consolidating.discard(key)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """

        memory = self._memory_port
        consolidate_up_to = len(session.messages)
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
            # 在所有 await 之前捕获边界索引，避免 LLM call 期间新消息追加后
            # 用错误的 len(session.messages) 回写 last_consolidated。
            consolidate_up_to = len(session.messages) - keep_count
            old_messages = session.messages[
                session.last_consolidated : consolidate_up_to
            ]
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
            role = m["role"].upper()
            # 跳过纯工具结果消息（role=tool），它们是内部往返，不是对话内容
            if m["role"] == "tool":
                continue
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {role}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""你是记忆提取代理（Memory Extraction Agent）。从对话中精确提取结构化信息，返回 JSON。

## 字段说明

### 1. "history_entry" → HISTORY.md
2-5 句事件摘要，以 [YYYY-MM-DD HH:MM] 开头，保留足够细节便于未来 grep 检索。

### 2. "user_facts" → MEMORY.md 候选缓冲
只写用户的**稳定持久事实**，格式为带分类标注的 bullet 列表（对照下方用户档案查重，只写新增内容）。

✓ 写：用户身份/设备/账号/凭证/技能/经历/偏好/立场/游戏数据
✓ 写：用户对知识库内容（小说/游戏/作品）的情感倾向——但用指针格式，不复制内容本体
✗ 不写：工具或系统的限制、bug、API 状态（不是用户事实）
✗ 不写：一次性操作记录（"用户执行了X"、"已完成Y"）
✗ 不写：对话过程描述
✗ 不写：短期/临时状态（由 agent 通过 update_now 工具实时维护）
✗ 不写：面向 agent 的执行流程、工具顺序、SOP 条款（此类应放在 SOP 或行为记忆，不放用户画像）
✗ 不写：某次任务的操作性细节（配置参数、具体数值、当次选择的方案）——这些不是用户画像
✗ 不写：知识库内容本体（世界观细节、剧情、角色设定）——内容留在 KB 文件，这里只存用户反应

判断标准："如果同一个用户开始一段新对话，这条信息还有意义吗？"若否，不写。

**知识库内容的指针化格式**（用户对 KB 内容有明确情感/立场时使用）：
`用户 [情感/立场描述] [内容简称] -> [KB文件路径]`
格式示例：
- [2236偏好] 用户最喜爱的设定：正三角形（具体举例非抽象隐喻）-> ~/.akasic/workspace/kb/2236/summaries/chunks/hime_ending-chunk-0003.source.md
- [2236偏好] 用户对伸司的评价从负面反转为认可其悲剧性 -> ~/.akasic/workspace/kb/2236/summaries/chunks/common-chunk-0039.source.md

若无新事实，返回 ""。
普通格式示例：
- [账号与凭证] UCD Moodle Token: xxxxx (User ID: 22578)
- [偏好] 用户偏好按课程单独分配日历颜色

### 3. "corrections" → 覆盖 MEMORY.md 错误记录
仅当对话中有明确信息推翻档案现有记录时才写，必须同时写旧值和新值。若无需纠正，返回 ""。
✗ 不写：工具加载失败/报错（属于运行时故障，不代表用户现实状态发生改变）
✗ 不写：单次工具调用返回异常（需用户本人明确确认才算纠正）
格式示例：
- [纠正] 主力机显示器：档案记录 Dell U2723D → 用户确认实际为 LG 27GP950

### 4. "self_insights" → SELF.md 候选洞察
agent 从本次对话中发现的用户**行为模式新规律**，格式为带 [SELF] 标注的 bullet 列表。
必须是跨对话可泛化的规律，不是描述这次发生了什么。若无新洞察，返回 []。
格式示例：
- [SELF] 用户在涉及日程/课表时要求精确时间，拒绝模糊描述
- [SELF] 用户对工具调用中间步骤不感兴趣，只关心最终结论

---

## 当前用户档案（用于 user_facts 查重）
{current_memory or "（空）"}

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

            # user_facts / corrections → PENDING.md（MemoryOptimizer 稍后合并到 MEMORY.md）
            user_facts = result.get("user_facts", "")
            if user_facts and isinstance(user_facts, str) and user_facts.strip():
                memory.append_pending(user_facts)
                logger.info(
                    f"Memory consolidation: appended {len(user_facts.splitlines())} user_facts to PENDING"
                )
            corrections = result.get("corrections", "")
            if corrections and isinstance(corrections, str) and corrections.strip():
                memory.append_pending(corrections)
                logger.info(
                    f"Memory consolidation: appended {len(corrections.splitlines())} corrections to PENDING"
                )

            # self_insights → PENDING.md 带 [SELF] 标记，供 Optimizer 路由到 SELF.md
            self_insights = result.get("self_insights", [])
            if self_insights and isinstance(self_insights, list):
                insight_lines = [
                    s if s.strip().startswith("[SELF]") else f"[SELF] {s.strip()}"
                    for s in self_insights
                    if isinstance(s, str) and s.strip()
                ]
                if insight_lines:
                    memory.append_pending("\n".join(insight_lines))
                    logger.info(
                        f"Memory consolidation: appended {len(insight_lines)} self_insights to PENDING"
                    )

            # memory v2 写入（非阻塞）：通过 MemoryPort 统一写口
            history_entry = result.get("history_entry", "")
            _source_ref = (
                f"{session.key}@{session.last_consolidated}-{consolidate_up_to}"
                if not archive_all
                else f"{session.key}@archive_all"
            )
            asyncio.create_task(
                self._memory_port.save_from_consolidation(
                    history_entry=history_entry,
                    behavior_updates=[],
                    source_ref=_source_ref,
                    scope_channel=getattr(session, "_channel", ""),
                    scope_chat_id=getattr(session, "_chat_id", ""),
                )
            )

            if archive_all:
                session.last_consolidated = 0
            else:
                # 使用 await 前捕获的边界，而非 await 后可能已增长的长度
                session.last_consolidated = consolidate_up_to
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
