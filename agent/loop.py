import asyncio
import json
import json_repair
import logging
from datetime import datetime
from pathlib import Path

from agent.context import ContextBuilder
from agent.memory import MemoryStore
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager

logger = logging.getLogger(__name__)

# 内部注入的反思提示，不应持久化到 session
_REFLECT_PROMPT = "根据上述工具执行结果，决定下一步操作。"

# 检测到"零工具调用却声称完成操作"时的纠正提示
_NO_TOOL_BUT_CLAIMED_PROMPT = (
    "你在上一条回复中没有调用任何工具，但描述了某些操作的执行结果。"
    "请现在实际调用工具完成任务，不要描述假设的执行过程。"
)

# 响应中暗示"已完成实际操作"的特征词（检测幻觉用）
_ACTION_CLAIM_MARKERS = (
    "successfully", "已下载", "下载完成", "已执行", "执行成功", "执行了",
    "已安装", "安装完成", "i called", "i ran", "i executed", "shell tool",
    "yt-dlp", "命令执行", "文件已保存", "下载到",
)


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
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.session_manager = session_manager
        self.workspace = workspace
        self.context = ContextBuilder(workspace)
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info(f"AgentLoop 启动  model={self.model}  max_iter={self.max_iterations}")
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                try:
                    response = await self._process(msg)
                    await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"处理消息出错: {e}", exc_info=True)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"出错：{e}",
                    ))
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._running = False
        logger.info("AgentLoop 停止")

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """将当前会话的 channel/chat_id 注入工具，供主动推送时使用。"""
        self.tools.set_context(channel=channel, chat_id=chat_id)

    # ── 私有方法 ──────────────────────────────────────────────────

    async def _process(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender}: {preview}")

        key = session_key or msg.session_key
        session = self.session_manager.get_or_create(key)

        # 超过记忆窗口时压缩
        if len(session.messages) > self.memory_window:
            await self._consolidate_memory(session)

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used, tool_chain = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender}: {preview}")

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None,
                            tool_chain=tool_chain if tool_chain else None)
        self.session_manager.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata={
                **(msg.metadata or {}),  # Pass through for channel-specific needs (e.g. Slack thread_ts)
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
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })
                iter_calls: list[dict] = []
                for tc in response.tool_calls:
                    tools_used.append(tc.name)
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info(f"  → 工具 {tc.name}  参数: {args_str[:120]}")
                    result = await self.tools.execute(tc.name, tc.arguments)
                    result_preview = result[:80] + "..." if len(result) > 80 else result
                    logger.info(f"  ← 工具 {tc.name}  结果: {result_preview!r}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    iter_calls.append({
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result": result,
                    })
                tool_chain.append({"text": response.content, "calls": iter_calls})

                # 工具结果注入后，提示 LLM 反思并决定下一步
                messages.append({"role": "user", "content": _REFLECT_PROMPT})
            else:
                # 零工具调用但声称完成了操作 → 纠正一次
                if not tools_used and _claims_action_without_tools(response.content):
                    logger.warning("检测到零工具调用但声称完成操作，注入纠正提示")
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": _NO_TOOL_BUT_CLAIMED_PROMPT})
                    continue
                logger.info(f"LLM 返回最终回复  iteration={iteration + 1}")
                messages.append({"role": "assistant", "content": response.content})
                return response.content or "（无响应）", tools_used, tool_chain

        logger.warning(f"已达到最大迭代次数 {self.max_iterations}")
        return "（已达到最大迭代次数）", tools_used, tool_chain

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
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(
                    f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return
            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(
                    f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(
                f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        # 以下逻辑对 archive_all 和普通压缩均适用
        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system",
                     "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self.model,
                max_tokens=2048,
            )
            text = (response.content or "").strip()

            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if "history_entry" in result:
                memory.append_history(result["history_entry"])
            if "memory_update" in result:
                memory.write_long_term(result["memory_update"])

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(
                f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
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


def _claims_action_without_tools(content: str | None) -> bool:
    """检测响应是否在没有工具调用的情况下声称完成了实际操作。

    仅用于校验模型自身的输出，而非猜测用户意图。
    触发条件：响应里含有"已执行/下载完成/successfully"等暗示操作已完成的词语。
    """
    if not content:
        return False
    lower = content.lower()
    return any(marker in lower for marker in _ACTION_CLAIM_MARKERS)
