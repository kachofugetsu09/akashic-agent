import asyncio
import json
import logging

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry

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
        model: str = "deepseek-chat",
        max_iterations: int = 10,
        max_tokens: int = 8192,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self._running = False
        self._history: dict[str, list[dict]] = {}  # session_key → OpenAI messages

    async def run(self) -> None:
        self._running = True
        logger.info("AgentLoop 启动")
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
        history = self._history.setdefault(msg.session_key, [])
        history.append({"role": "user", "content": msg.content})
        reply = await self._run_agent_loop(history)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=reply,
            metadata=msg.metadata,
        )

    async def _run_agent_loop(self, history: list[dict]) -> str:
        """迭代调用 LLM，直到无工具调用或达到上限"""
        for _ in range(self.max_iterations):
            response = await self.provider.chat(
                messages=history,
                tools=self.tools.get_schemas(),
                model=self.model,
                max_tokens=self.max_tokens,
            )

            if response.tool_calls:
                # assistant 消息（含 tool_calls 字段）
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
                # 每个工具结果独立一条 role=tool 消息
                for tc in response.tool_calls:
                    logger.info(f"调用工具: {tc.name}  参数: {tc.arguments}")
                    result = await self.tools.execute(tc.name, tc.arguments)
                    history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                history.append({"role": "assistant", "content": response.content})
                return response.content or "（无响应）"

        return "（已达到最大迭代次数）"
