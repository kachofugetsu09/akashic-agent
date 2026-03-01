import asyncio
import json
import json_repair
import logging
import re
from datetime import datetime
from pathlib import Path

from agent.context import ContextBuilder
from agent.memory import MemoryStore
from agent.query_analyzer import QueryAnalysis, QueryAnalyzer
from bus.events import InboundMessage, OutboundMessage
from bus.processing import ProcessingState
from bus.queue import MessageBus
from agent.provider import ContentSafetyError, LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager
from proactive.presence import PresenceStore

# 安全拦截时递减历史窗口的倍率序列：全量 → 减半 → 清空
_SAFETY_RETRY_RATIOS = (1.0, 0.5, 0.0)
_RECENT_CONTEXT_COUNT = 10
_EXTRA_CONTEXT_MAX_BLOCKS = 8

logger = logging.getLogger(__name__)

# 内部注入的反思提示，不应持久化到 session
_REFLECT_PROMPT = """根据上述工具执行结果，决定下一步操作。

【自检，无需在回复中说明，只用于内部决策】
1. 当前任务是否有匹配的技能尚未读取 SKILL.md？若有，必须先 read_file 读取完整指令再继续。
2. 即将输出的结论是否有本轮工具返回的事实支撑？无支撑时允许合理推断，但必须显式标注“我推测/可能/更像是”，并保持可追溯到本轮事实；禁止把推断写成事实。
3. 涉及用户状态/数据/画像的陈述，若未经本轮工具验证，禁止以事实语气输出。
4. 禁止把历史会话中的旧工具结果冒充本轮实测——若用户问的是"现在/当前"的数据，必须本轮重新调用工具。
5. 涉及时间判断（现在/当前/最新/是否已发生）时，统一以本轮 request_time 为时间锚点；若证据只有计划时间而无实际发生证据，不得断言“已经发生”。
6. 若用户问“动机/来源/身世/含义”这类解释问题，可结合事实做联想，但最终要区分“已证据事实”和“待用户确认的推测”。"""

# 每轮对话开始前注入的初始自检提示，不应持久化到 session
_PRE_FLIGHT_PROMPT = """【回复前必须完成以下自检，无需在回复中说明】
0. 【SOP 优先级最高，强制执行】系统 prompt 中已通过向量检索注入了本轮相关 SOP 内容（见"【强制约束】"和"【流程规范】"段），直接参照执行，**无需再 read_file 读取 SOP 文件**。仅当用户明确要求新增/修改 SOP 时，才需要 read_file 读取对应文件。
1. 用户是否要求执行某项操作，且该操作与 # Skills 中某个技能的描述明确匹配？若是，禁止在未调用工具的情况下直接回答——必须先 read_file 读取对应 SKILL.md，再按指令执行工具，最后基于工具返回结果作答。（注意：用户只是询问技能列表/能力范围，不触发此规则，直接根据摘要回答即可。）
2. 用户问的内容是否需要实时/当前数据（订阅列表、天气、最新动态、用户状态等）？若需要，同样禁止凭记忆直接回答，必须本轮调用工具获取。
3. 遇到”现在/当前/最新/今天/是否已发生”等时间敏感判断，先以 request_time 锚定时间，再给结论；若缺少可核验事实，明确说不确定。
4. 回答允许做合理联想，但必须显式标注推测语气，不得冒充事实；必要时给出”待确认”。
5. 确认以上规则均满足后，才允许输出最终回复。"""


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
        query_analyzer_enabled: bool = True,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.session_manager = session_manager
        self.workspace = workspace
        self.context = ContextBuilder(workspace)
        self.model = model
        # light_model / light_provider 保留接口兼容，不再用于 self-check
        self.light_model = light_model or model
        self.light_provider = light_provider or provider
        self.query_analyzer = QueryAnalyzer(
            provider=self.light_provider,
            model=self.light_model,
            workspace=workspace,
            get_tool_schemas=self.tools.get_schemas,
            tool_executor=self.tools.execute,
        )
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self._presence = presence
        self._running = False
        self._consolidating: set[str] = set()  # 正在后台压缩的 session key
        self._processing_state = processing_state
        self._memorizer = memorizer
        self._retriever = retriever
        self._disable_full_memory = disable_full_memory
        self._query_analyzer_enabled = query_analyzer_enabled

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
        analysis: QueryAnalysis | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
    ) -> tuple[str, list[str], list[dict]]:
        """递减历史窗口重试，处理 LLM 安全拦截错误。

        重试顺序：全量历史 → 减半 → 无历史。
        降级成功后同步修剪 session，防止下次继续触发。
        所有窗口均失败时说明当前消息本身违规，返回友好提示。
        """
        source_history = base_history or session.get_history(max_messages=self.memory_window)
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
                relevant_sops=analysis.relevant_sops if analysis else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
                retrieved_memory_block=retrieved_memory_block,
                disable_full_memory=self._disable_full_memory,
            )
            try:
                result = await self._run_agent_loop(
                    initial_messages,
                    analysis=analysis,
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
                    next_window = int(
                        total_history * _SAFETY_RETRY_RATIOS[attempt + 1]
                    )
                    logger.warning(
                        f"安全拦截 (attempt={attempt + 1})，"
                        f"缩小历史窗口重试 {window} → {next_window}"
                    )
                else:
                    logger.warning("安全拦截：所有窗口均失败，当前消息本身可能违规")
                    return "你的消息触发了安全审查，无法处理。", [], []

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

    def _assemble_main_history(
        self,
        history: list[dict],
        analysis: QueryAnalysis,
        max_blocks: int | None = _EXTRA_CONTEXT_MAX_BLOCKS,
    ) -> list[dict]:
        """根据 QueryAnalyzer 指针拼装上下文，并保证 tool 调用链合法。"""
        if not history:
            return []

        n = len(history)
        # 将历史分组成“合法块”：
        # - 普通消息块：单条 user/assistant
        # - 工具块：assistant(tool_calls) + 紧随其后的 tool 结果
        blocks: list[list[int]] = []
        index_to_block: dict[int, int] = {}

        i = 0
        while i < n:
            msg = history[i]
            role = msg.get("role")

            if role == "assistant" and msg.get("tool_calls"):
                block = [i]
                j = i + 1
                while j < n and history[j].get("role") == "tool":
                    block.append(j)
                    j += 1
                # 不完整工具块（assistant 有 tool_calls 但缺 tool 回包）会触发 provider 400，直接丢弃
                if len(block) == 1:
                    i = j
                    continue
                block_id = len(blocks)
                blocks.append(block)
                for idx in block:
                    index_to_block[idx] = block_id
                i = j
                continue

            if role == "tool":
                # 孤立 tool（没有前置 assistant tool_calls）会触发 provider 400，直接丢弃
                i += 1
                continue

            block_id = len(blocks)
            blocks.append([i])
            index_to_block[i] = block_id
            i += 1

        if not blocks:
            return []

        selected_blocks: set[int] = set()

        for idx in analysis.history_pointers:
            if isinstance(idx, int) and 0 <= idx < n and idx in index_to_block:
                selected_blocks.add(index_to_block[idx])

        if not selected_blocks:
            tail_blocks = min(8, len(blocks))
            selected_blocks.update(range(len(blocks) - tail_blocks, len(blocks)))

        # extra context 限流：仅保留靠近当前的若干块，避免再次撑爆主上下文
        selected_sorted = sorted(selected_blocks)
        if max_blocks is not None and max_blocks > 0 and len(selected_sorted) > max_blocks:
            selected_sorted = selected_sorted[-max_blocks:]

        assembled: list[dict] = []
        for bid in selected_sorted:
            for idx in blocks[bid]:
                assembled.append(history[idx])
        return assembled

    @staticmethod
    def _split_history_for_analyzer(
        history: list[dict],
        recent_count: int = _RECENT_CONTEXT_COUNT,
    ) -> tuple[list[dict], list[dict]]:
        """拆分历史：旧上下文给 analyzer 选 extra，最近 N 条始终保留给主循环。"""
        if not history:
            return [], []
        k = max(0, int(recent_count))
        if k <= 0:
            return history, []
        n = len(history)
        if n <= k:
            return [], history
        split_idx = n - k

        # 避免在 tool chain 中间切断：
        # 1) 若切分点落在 tool 消息上，回退到对应 assistant(tool_calls) 起点
        if split_idx < n and history[split_idx].get("role") == "tool":
            j = split_idx - 1
            while j >= 0 and history[j].get("role") == "tool":
                j -= 1
            if j >= 0 and history[j].get("role") == "assistant" and history[j].get("tool_calls"):
                split_idx = j

        # 2) 若切分点刚好落在 assistant(tool_calls) 之后，也回退一位，保持整块在 recent_tail
        if split_idx > 0:
            prev = history[split_idx - 1]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                split_idx -= 1

        split_idx = max(0, min(split_idx, n))
        return history[:split_idx], history[split_idx:]

    async def _process(
        self, msg: InboundMessage, session_key: str | None = None
    ) -> OutboundMessage:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender}: {preview}")

        key = session_key or msg.session_key
        if self._processing_state:
            self._processing_state.enter(key)
        try:
            return await self._process_inner(msg, key)
        finally:
            if self._processing_state:
                self._processing_state.exit(key)

    async def _process_inner(
        self, msg: InboundMessage, key: str
    ) -> OutboundMessage:
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

        # QueryAnalyzer 仅筛选旧上下文；最近 N 条始终作为保底上下文传给主循环
        analysis_history = session.get_history(max_messages=self.memory_window)
        analyzer_scope, recent_tail = self._split_history_for_analyzer(
            analysis_history, recent_count=_RECENT_CONTEXT_COUNT
        )
        if self._query_analyzer_enabled:
            # Analyzer 可读完整历史（含 recent_tail），但只能为旧上下文区间提供指针。
            analysis = await self.query_analyzer.analyze(
                msg.content,
                analysis_history,
                message_timestamp=msg.timestamp,
                selectable_history_len=len(analyzer_scope),
                forced_recent_count=len(recent_tail),
            )
            extra_history = self._assemble_main_history(
                analyzer_scope,
                analysis,
                max_blocks=_EXTRA_CONTEXT_MAX_BLOCKS,
            )
            main_history = extra_history + recent_tail
            logger.info(
                "[query_analyzer] needs_tool=%s required=%s sops=%s targets=%s pointers=%s keep_recent=%s history=%d analyzer_scope=%d extra=%d recent_tail=%d main=%d reason=%s",
                analysis.needs_tool,
                analysis.required_evidence,
                analysis.relevant_sops,
                analysis.target_files,
                analysis.history_pointers,
                analysis.keep_recent,
                len(analysis_history),
                len(analyzer_scope),
                len(extra_history),
                len(recent_tail),
                len(main_history),
                analysis.reasoning,
            )
        else:
            analysis = QueryAnalysis()
            main_history = recent_tail
            logger.info("[query_analyzer] disabled, using recent_tail only (len=%d)", len(recent_tail))

        # memory v2 检索
        retrieved_block = ""
        if self._retriever:
            try:
                items = await self._retriever.retrieve(msg.content)
                retrieved_block = self._retriever.format_injection_block(items)
                if retrieved_block:
                    logger.info(f"memory2 retrieve: {len(items)} 条命中，注入检索块")
            except Exception as e:
                logger.warning(f"memory2 retrieve 失败，跳过: {e}")

        self._set_tool_context(msg.channel, msg.chat_id)
        final_content, tools_used, tool_chain = await self._run_with_safety_retry(
            msg,
            session,
            skill_names=skill_mentions or None,
            analysis=analysis,
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
        analysis: QueryAnalysis | None = None,
        request_time: datetime | None = None,
    ) -> tuple[str, list[str], list[dict]]:
        """迭代调用 LLM，直到无工具调用或达到上限。返回 (final_content, tools_used, tool_chain)

        tool_chain 是按迭代分组的工具调用记录，每个元素：
          {"text": str|None, "calls": [{"call_id", "name", "arguments", "result"}]}
        """
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []

        # 第一轮调用前注入预检提示，让 LLM 在未调用任何工具时也做技能匹配自检
        preflight_prompt = (
            f"【本轮时间锚点】{self._format_request_time_anchor(request_time)}\n"
            "所有时间相关判断必须与该锚点一致；无法验证时必须明确不确定。\n\n"
            + _PRE_FLIGHT_PROMPT
        )
        # 将前置分析器的取证建议注入预检提示，主模型必须按顺序执行后再作答
        if analysis and analysis.required_evidence:
            evidence_lines = "\n".join(
                f"- [{e.get('tool', 'tool')}] {e.get('hint', '')}"
                for e in analysis.required_evidence
            )
            preflight_prompt += (
                "\n\n【前置分析器取证建议（高优先级，请按顺序执行，收集事实依据后再作答）】\n"
                + evidence_lines
            )
        if analysis and analysis.relevant_sops:
            sop_lines = "\n".join(f"- {s}" for s in analysis.relevant_sops)
            preflight_prompt += (
                "\n\n【前置分析器判定本轮相关 SOP，必须优先 read_file 读取并遵循】\n"
                + sop_lines
            )
        # target_files 中的 SOP 文件：注入必读提示，是否修改由 agent 根据用户意图判断
        if analysis and analysis.target_files:
            sop_targets = [
                t for t in analysis.target_files
                if "/sop/" in t and t.endswith(".md") and not t.endswith("README.md")
            ]
            if sop_targets:
                sop_lines = "\n".join(f"- {t}" for t in sop_targets)
                preflight_prompt += (
                    "\n\n【本轮涉及以下 SOP 文件，必须先 read_file 读取】\n"
                    + sop_lines
                    + "\n若用户要求修改/改进/优化该 SOP，读取后按要求改写并 write_file 写回；"
                    "若用户是要执行某项操作，读取后按规范行动。"
                )
        messages = messages + [{"role": "user", "content": preflight_prompt}]
        first_tool_choice = "required" if (analysis and analysis.needs_tool) else "auto"

        for iteration in range(self.max_iterations):
            logger.debug(f"LLM 调用  iteration={iteration + 1}")
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_schemas(),
                model=self.model,
                max_tokens=self.max_tokens,
                tool_choice=first_tool_choice if iteration == 0 else "auto",
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
            # 在所有 await 之前捕获边界索引，避免 LLM call 期间新消息追加后
            # 用错误的 len(session.messages) 回写 last_consolidated。
            consolidate_up_to = len(session.messages) - keep_count
            old_messages = session.messages[session.last_consolidated : consolidate_up_to]
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
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {role}: {m['content']}"
            )
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()
        current_questions = memory.read_questions()
        current_ongoing = memory.read_now_ongoing()

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
✗ 不写：短期/临时状态（改用 now_updates 字段）
✗ 不写：agent 行为规则（改用 behavior_updates 字段）
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

### 4. "now_updates" → 直接更新 NOW.md"近期进行中"
包含两个子字段（均为字符串列表）：
- "add_ongoing"：新增到"近期进行中"的条目（自然语言一句话，不带 bullet 符号）
  - 只写需要跨对话持续追踪的进行中状态，如"正在阅读《西历2236》TE线"
  - 不写技术坐标（chunk 号等）
  - 不写已完成的事项
- "remove_ongoing_keywords"：要从"近期进行中"删除的条目关键词（模糊匹配，命中即删）

均无变化时两者返回 []。

### 5. "self_insights" → SELF.md 候选洞察
agent 从本次对话中发现的用户**行为模式新规律**，格式为带 [SELF] 标注的 bullet 列表。
必须是跨对话可泛化的规律，不是描述这次发生了什么。若无新洞察，返回 []。
格式示例：
- [SELF] 用户在涉及日程/课表时要求精确时间，拒绝模糊描述
- [SELF] 用户对工具调用中间步骤不感兴趣，只关心最终结论

### 6. "answered_question_indices" → 清理 NOW.md 问题列表
本次对话中已得到答复的问题序号（1-based int 列表）。无则返回 []。

### 7. "behavior_updates" → memory2 DB
用户明确要求改变 agent 未来行为的规则（"记住/以后/下次"等触发词才写）。
格式：JSON 数组，每项 {{"summary": "...", "memory_type": "procedure|preference",
"tool_requirement": null或"工具名", "steps": [], "persist_file": null或"文件名"}}
若无则返回 []。

---

## 当前用户档案（用于 user_facts 查重）
{current_memory or "（空）"}

## 待了解的问题（用于 answered_question_indices）
{current_questions or "（无）"}

## 当前进行中事项（用于 now_updates 去重）
{current_ongoing or "（无）"}

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

            # now_updates → 直接写入 NOW.md，不经过 PENDING
            now_updates = result.get("now_updates", {})
            if isinstance(now_updates, dict):
                add_ongoing = now_updates.get("add_ongoing", [])
                remove_kws = now_updates.get("remove_ongoing_keywords", [])
                if (add_ongoing or remove_kws):
                    try:
                        memory.update_now_ongoing(
                            add=[s for s in add_ongoing if isinstance(s, str)],
                            remove_keywords=[s for s in remove_kws if isinstance(s, str)],
                        )
                        logger.info(
                            f"Memory consolidation: now_updates add={add_ongoing} remove={remove_kws}"
                        )
                    except Exception as e:
                        logger.warning(f"Memory consolidation: now_updates 写入失败: {e}")

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

            # memory v2 写入（非阻塞）
            if self._memorizer:
                behavior_updates = result.get("behavior_updates", [])
                history_entry = result.get("history_entry", "")
                _source_ref = (
                    f"{session.key}@{session.last_consolidated}-{consolidate_up_to}"
                    if not archive_all
                    else f"{session.key}@archive_all"
                )
                asyncio.create_task(self._memorizer.save_from_consolidation(
                    history_entry=history_entry,
                    behavior_updates=behavior_updates if isinstance(behavior_updates, list) else [],
                    source_ref=_source_ref,
                    scope_channel=getattr(session, '_channel', ''),
                    scope_chat_id=getattr(session, '_chat_id', ''),
                ))

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
