"""
入口：读取 config.json，启动 AgentLoop + CLI channel
"""
import asyncio
import logging
import sys
from pathlib import Path

from bus.queue import MessageBus
from agent.config import Config
from agent.loop import AgentLoop
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from agent.tools.shell import ShellTool
from agent.tools.web_fetch import WebFetchTool
from channels.cli import CLIChannel

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def build_agent(config_path: str = "config.json") -> tuple[AgentLoop, MessageBus]:
    config = Config.load(config_path)

    bus = MessageBus()

    tools = ToolRegistry()
    tools.register(ShellTool())
    tools.register(WebFetchTool())

    provider = LLMProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=config.system_prompt,
    )

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        tools=tools,
        model=config.model,
        max_iterations=config.max_iterations,
        max_tokens=config.max_tokens,
    )
    return loop, bus


async def main(config_path: str = "config.json") -> None:
    agent_loop, bus = build_agent(config_path)
    cli = CLIChannel(bus)

    await asyncio.gather(
        agent_loop.run(),
        cli.run(),
        bus.dispatch_outbound(),
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    if not Path(path).exists():
        print(f"找不到配置文件 {path!r}，请先复制 config.example.json 为 config.json 并填写 API Key。")
        sys.exit(1)
    asyncio.run(main(path))
