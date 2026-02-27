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
1. 用户是否要求执行某项操作，且该操作与 # Skills 中某个技能的描述明确匹配？若是，禁止在未调用工具的情况下直接回答——必须先 read_file 读取对应 SKILL.md，再按指令执行工具，最后基于工具返回结果作答。（注意：用户只是询问技能列表/能力范围，不触发此规则，直接根据摘要回答即可。）
2. 用户问的内容是否需要实时/当前数据（订阅列表、天气、最新动态、用户状态等）？若需要，同样禁止凭记忆直接回答，必须本轮调用工具获取。
3. 遇到“现在/当前/最新/今天/是否已发生”等时间敏感判断，先以 request_time 锚定时间，再给结论；若缺少可核验事实，明确说不确定。
4. 若用户在提出“以后请这样做/新增规则/修改 SOP”，先 read_file `sop/README.md` 确认目录索引，再决定读取或编辑具体 SOP 文件。
5. 回答允许做合理联想，但必须显式标注推测语气，不得冒充事实；必要时给出“待确认”。
6. 确认以上规则均满足后，才允许输出最终回复。"""


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
            tool_schemas=self.tools.get_schemas(),
            tool_executor=self.tools.execute,
        )
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self._presence = presence
        self._running = False
        self._consolidating: set[str] = set()  # 正在后台压缩的 session key
        self._processing_state = processing_state

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
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
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

        keep_recent = max(0, min(int(analysis.keep_recent), n))
        selected_blocks: set[int] = set()

        for idx in analysis.history_pointers:
            if isinstance(idx, int) and 0 <= idx < n and idx in index_to_block:
                selected_blocks.add(index_to_block[idx])

        if keep_recent > 0:
            for idx in range(n - keep_recent, n):
                bid = index_to_block.get(idx)
                if bid is not None:
                    selected_blocks.add(bid)

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
        analysis = await self.query_analyzer.analyze(
            msg.content,
            analyzer_scope,
            message_timestamp=msg.timestamp,
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

        self._set_tool_context(msg.channel, msg.chat_id)
        final_content, tools_used, tool_chain = await self._run_with_safety_retry(
            msg,
            session,
            skill_names=skill_mentions or None,
            analysis=analysis,
            base_history=main_history,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

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
