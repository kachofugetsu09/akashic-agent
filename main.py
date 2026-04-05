"""
入口

两种模式：
  python main.py          启动 agent 服务（AgentLoop + 所有 channel + IPC server）
  python main.py cli      连接到运行中的 agent（CLI 客户端）
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agent.config import Config
from bootstrap.app import build_app_runtime


def connect_cli(config_path: str = "config.json") -> None:
    socket_path = Config.load(config_path).channels.socket
    try:
        from infra.channels.cli_tui import run_tui
    except RuntimeError as exc:
        print(exc)
        print("回退到纯文本 CLI。")
        from infra.channels.cli import CLIClient

        asyncio.run(CLIClient(socket_path).run())
        return

    run_tui(socket_path)


async def serve(
    config_path: str = "config.json",
    workspace: Path | None = None,
) -> None:
    config = Config.load(config_path)
    runtime = build_app_runtime(
        config,
        workspace=workspace or (Path.home() / ".akasic" / "workspace"),
    )
    await runtime.run()


if __name__ == "__main__":
    args = sys.argv[1:]
    config_path = "config.json"
    workspace: Path | None = None

    if "--config" in args:
        idx = args.index("--config")
        config_path = args[idx + 1]
    if "--workspace" in args:
        idx = args.index("--workspace")
        workspace = Path(args[idx + 1])

    if not Path(config_path).exists():
        print(
            f"找不到配置文件 {config_path!r}，请先复制 config.example.json 为 config.json。"
        )
        sys.exit(1)

    if "cli" in args:
        connect_cli(config_path)
    else:
        asyncio.run(serve(config_path, workspace))
