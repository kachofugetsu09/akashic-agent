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
from bootstrap.dashboard_api import run_dashboard_api
from bootstrap.init_workspace import InitSummary, init_workspace


def _default_workspace() -> Path:
    return Path.home() / ".akashic" / "workspace"


def _get_flag_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        raise ValueError(f"参数 {flag} 缺少值")
    return args[idx + 1]


def _print_init_summary(summary: InitSummary) -> None:
    def _print_group(title: str, paths: list[Path]) -> None:
        if not paths:
            return
        print(title)
        for path in paths:
            print(f"- {path}")

    _print_group("已创建：", summary.created)
    _print_group("已覆盖：", summary.overwritten)
    _print_group("已跳过：", summary.skipped)
    if summary.notes:
        print("说明：")
        for note in summary.notes:
            print(f"- {note}")


def connect_cli(config_path: str = "config.toml") -> None:
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
    config_path: str = "config.toml",
    workspace: Path | None = None,
) -> None:
    config = Config.load(config_path)
    runtime = build_app_runtime(
        config,
        workspace=workspace or _default_workspace(),
    )
    await runtime.run()


if __name__ == "__main__":
    args = sys.argv[1:]
    config_path = "config.toml"
    workspace: Path | None = None
    force = "--force" in args
    with_fitbit = "--with-fitbit" in args
    dashboard_host = "0.0.0.0"
    dashboard_port = 2236

    try:
        config_value = _get_flag_value(args, "--config")
        workspace_value = _get_flag_value(args, "--workspace")
        host_value = _get_flag_value(args, "--host")
        port_value = _get_flag_value(args, "--port")
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if config_value is not None:
        config_path = config_value
    if workspace_value is not None:
        workspace = Path(workspace_value)
    if host_value is not None:
        dashboard_host = host_value
    if port_value is not None:
        dashboard_port = int(port_value)

    if args and args[0] == "init":
        summary = init_workspace(
            config_path=config_path,
            workspace=workspace or _default_workspace(),
            force=force,
            with_fitbit=with_fitbit,
        )
        _print_init_summary(summary)
        sys.exit(0)

    if args and args[0] == "gateway":
        asyncio.run(serve(config_path, workspace))
        sys.exit(0)

    if args and args[0] == "dashboard":
        run_dashboard_api(
            workspace=workspace or _default_workspace(),
            host=dashboard_host,
            port=dashboard_port,
        )
        sys.exit(0)

    if not Path(config_path).exists():
        print(
            f"找不到配置文件 {config_path!r}，请先复制 config.example.toml 为 config.toml。"
        )
        sys.exit(1)

    if "cli" in args:
        connect_cli(config_path)
    else:
        asyncio.run(serve(config_path, workspace))
