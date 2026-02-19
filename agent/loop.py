import asyncio
import json
import logging

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager

logger = logging.getLogger(__name__)


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
        model: str = "deepseek-chat",
        max_iterations: int = 10,
        max_tokens: int = 8192,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.session_manager = session_manager
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
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

    # ── 私有方法 ──────────────────────────────────────────────────

    async def _process(self, msg: InboundMessage) -> OutboundMessage:
        session = self.session_manager.get_or_create(msg.session_key)
        history = session.get_history()
        is_new = len(history) == 0

        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(
            f"[{msg.channel}] 收到消息  session={msg.session_key}"
            f"  {'(新会话)' if is_new else f'(历史{len(history)}条)'}"
            f"  内容: {preview!r}"
        )

        # 将用户消息写入 session，并加入本轮 LLM 上下文
        session.add_message("user", msg.content)
        history.append({"role": "user", "content": msg.content})

        reply = await self._run_agent_loop(history)

        # 将助手回复写入 session 并持久化
        session.add_message("assistant", reply)
        self.session_manager.save(session)

        reply_preview = reply[:60] + "..." if len(reply) > 60 else reply
        logger.info(f"[{msg.channel}] 回复完成  session={msg.session_key}  内容: {reply_preview!r}")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=reply,
            metadata=msg.metadata,
        )

    async def _run_agent_loop(self, history: list[dict]) -> str:
        """迭代调用 LLM，直到无工具调用或达到上限"""
        iteration = 0
        for iteration in range(self.max_iterations):
            logger.debug(f"LLM 调用  iteration={iteration + 1}")
            response = await self.provider.chat(
                messages=history,
                tools=self.tools.get_schemas(),
                model=self.model,
                max_tokens=self.max_tokens,
            )

            if response.tool_calls:
                logger.info(f"LLM 请求调用 {len(response.tool_calls)} 个工具: "
                            f"{[tc.name for tc in response.tool_calls]}")
                history.append({
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
                for tc in response.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info(f"  → 工具 {tc.name}  参数: {args_str[:120]}")
                    result = await self.tools.execute(tc.name, tc.arguments)
                    result_preview = result[:80] + "..." if len(result) > 80 else result
                    logger.info(f"  ← 工具 {tc.name}  结果: {result_preview!r}")
                    history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                logger.info(f"LLM 返回最终回复  iteration={iteration + 1}")
                history.append({"role": "assistant", "content": response.content})
                return response.content or "（无响应）"

        logger.warning(f"已达到最大迭代次数 {self.max_iterations}")
        return "（已达到最大迭代次数）"
