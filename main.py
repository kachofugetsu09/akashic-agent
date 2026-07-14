"""
入口

主要模式：
  python main.py gateway            启动完整 agent 与 app-server
  python main.py supervise          以固定 supervisor 托管 gateway
  python main.py app-server --stdio 启动父进程托管控制面
  python main.py exec ...           非交互执行一个 turn
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from contextlib import suppress
from pathlib import Path


def _run_lightweight_setup_command() -> bool:
    """在加载 Agent runtime 依赖前分发纯配置命令。"""
    args = sys.argv[1:]
    if not args or args[0] != "setup-main":
        return False
    config_path = "config.toml"
    if "--config" in args:
        index = args.index("--config")
        if index + 1 >= len(args):
            raise SystemExit("参数 --config 缺少值")
        config_path = args[index + 1]
    import click

    from bootstrap.setup_main import run_main_model_setup

    try:
        run_main_model_setup(Path(config_path))
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        click.echo("已取消。", err=True)
        raise SystemExit(1) from exc
    return True


if __name__ == "__main__" and _run_lightweight_setup_command():
    raise SystemExit(0)


from agent.config import Config, resolve_app_server_endpoint
from agent.control.client import ControlClient, RemoteControlError
from agent.restart import RestartCoordinator, SupervisorCommitChannel
from agent.supervisor import RESTART_EXIT_CODE, run_supervisor
from agent.plugins.doctor import format_plugin_doctor_report, run_plugin_doctor
from agent.plugins.install import (
    install_git_plugin,
    set_installed_plugin_enabled,
    uninstall_plugin,
)
from bootstrap.app import build_app_runtime
from bootstrap.dashboard_api import run_dashboard_api
from bootstrap.init_workspace import InitSummary, init_workspace
from bootstrap.memory import build_memory_admin_runtime
from bootstrap.runtime_readiness import RuntimeReadiness
from bootstrap.workspace_token import read_workspace_token
from bootstrap.providers import build_providers
from core.net.http import SharedHttpResources
from infra.control.socket import is_tcp_endpoint


_HELP = """\
用法: python main.py [命令] [选项]

命令:
  setup                         运行交互式初始化向导
  setup-main                    仅切换主模型并保留其他配置
  init                          非交互初始化配置和工作区
  gateway                       启动 Agent 服务
  supervise                     以固定 supervisor 托管 Agent 服务
  app-server --stdio            在 stdio 上运行程序化控制面
  exec --new|--thread ID PROMPT 执行一个非交互 turn
  dashboard                     单独启动 Dashboard
  plugin-install                安装 Git 插件
  plugin-enable PLUGIN_ID       启用插件
  plugin-disable PLUGIN_ID      禁用插件
  plugin-uninstall PLUGIN_ID    卸载插件
  plugin-doctor [PLUGIN_ID]     检查插件状态

通用选项:
  --config PATH                 配置文件，默认 config.toml
  --workspace PATH              工作区，默认 ~/.akashic/workspace
  -h, --help                    显示帮助

无命令时启动 Agent 服务。
"""


def _default_workspace() -> Path:
    return Path.home() / ".akashic" / "workspace"


def _get_flag_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        raise ValueError(f"参数 {flag} 缺少值")
    return args[idx + 1]


def _validate_supervise_args(args: list[str]) -> None:
    """限制 supervise 只能接收固定 gateway 所需路径参数。"""

    index = 0
    seen: set[str] = set()
    while index < len(args):
        flag = args[index]
        if flag not in {"--config", "--workspace"}:
            raise ValueError(f"supervise 不支持参数: {flag}")
        if flag in seen or index + 1 >= len(args):
            raise ValueError(f"supervise 参数无效: {flag}")
        seen.add(flag)
        index += 2


def _print_init_summary(summary: InitSummary) -> None:
    def _print_group(title: str, paths: list[Path]) -> None:
        if not paths:
            return
        print(title)
        for path in paths:
            print(f"  {path}")

    _print_group("已创建：", summary.created)
    _print_group("已覆盖：", summary.overwritten)
    _print_group("已跳过：", summary.skipped)
    if summary.notes:
        print("说明：")
        for note in summary.notes:
            print(f"  {note}")
    if summary.next_steps:
        print("\n下一步：")
        for step in summary.next_steps:
            print(f"  {step}")


def _parse_csv_flag(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _wait_plugin_disabled(
    config_path: str,
    plugin_id: str,
    workspace: Path | None = None,
) -> None:
    if not Path(config_path).is_file():
        return
    config = Config.load(config_path)
    endpoint = resolve_app_server_endpoint(
        config.app_server.listen,
        workspace or _default_workspace(),
    )
    asyncio.run(
        _request_plugin_drain(endpoint, plugin_id, workspace or _default_workspace())
    )


async def _request_plugin_drain(
    endpoint: str,
    plugin_id: str,
    workspace: Path,
) -> None:
    token = read_workspace_token(workspace) if is_tcp_endpoint(endpoint) else None
    async with await ControlClient.connect(endpoint, workspace_token=token) as client:
        _ = await client.request("plugin/disable-and-drain", {"pluginId": plugin_id})


def _exec_prompt(args: list[str]) -> str:
    values_with_argument = {"--config", "--workspace", "--endpoint", "--thread"}
    positional: list[str] = []
    skip = False
    for index, value in enumerate(args[1:], start=1):
        if skip:
            skip = False
            continue
        if value in values_with_argument:
            skip = True
            continue
        if value.startswith("--"):
            continue
        positional.append(value)
    if not positional:
        raise ValueError("exec 缺少 prompt；使用 - 可从 stdin 读取")
    if len(positional) != 1:
        raise ValueError("exec 只接受一个 prompt 参数")
    return sys.stdin.read() if positional[0] == "-" else positional[0]


async def run_exec(args: list[str], config_path: str, workspace: Path) -> int:
    """连接现有 gateway，执行一轮并输出稳定机器结果。"""

    # 1. 校验 thread 选择和输入来源。
    new_thread = "--new" in args
    thread_id = _get_flag_value(args, "--thread")
    if new_thread == (thread_id is not None):
        raise ValueError("exec 必须且只能指定 --new 或 --thread ID")
    prompt = _exec_prompt(args)
    if "--json" in args and "--final-only" in args:
        raise ValueError("exec 的 --json 与 --final-only 不能同时使用")
    endpoint = _get_flag_value(args, "--endpoint")
    if endpoint is None:
        config = Config.load(config_path)
        endpoint = resolve_app_server_endpoint(config.app_server.listen, workspace)

    # 2. turn events 与最终文本严格按选定 stdout 模式输出。
    workspace_token = (
        read_workspace_token(workspace) if is_tcp_endpoint(endpoint) else None
    )
    async with await ControlClient.connect(
        endpoint,
        workspace_token=workspace_token,
    ) as client:
        if new_thread:
            thread = await client.start_thread()
            thread_id = str(thread["id"])
        assert thread_id is not None
        handle = await client.start_turn(thread_id, prompt)
        interrupt_requested = asyncio.Event()
        loop = asyncio.get_running_loop()
        previous_sigint: object | None = None
        native_handler = False
        try:
            loop.add_signal_handler(signal.SIGINT, interrupt_requested.set)
            native_handler = True
        except NotImplementedError:
            previous_sigint = signal.getsignal(signal.SIGINT)
            _ = signal.signal(
                signal.SIGINT,
                lambda _sig, _frame: loop.call_soon_threadsafe(interrupt_requested.set),
            )

        async def consume_events() -> dict[str, object]:
            async for event in handle.events():
                if "--json" in args:
                    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                if event.get("method") == "turn/completed":
                    params = event["params"]
                    assert isinstance(params, dict)
                    value = params["turn"]
                    assert isinstance(value, dict)
                    return value
            raise ConnectionError("turn event stream closed without terminal event")

        event_task = asyncio.create_task(consume_events(), name=f"exec-events:{handle.id}")
        interrupt_task = asyncio.create_task(interrupt_requested.wait(), name="exec-sigint")
        interrupted_by_user = False
        try:
            done, _ = await asyncio.wait(
                {event_task, interrupt_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if interrupt_task in done and not event_task.done():
                interrupted_by_user = True
                _ = await handle.interrupt()
            terminal = await event_task
        finally:
            interrupt_task.cancel()
            with suppress(asyncio.CancelledError):
                await interrupt_task
            if native_handler:
                _ = loop.remove_signal_handler(signal.SIGINT)
            elif previous_sigint is not None:
                _ = signal.signal(signal.SIGINT, previous_sigint)

        if "--final-only" in args:
            print(str(terminal.get("finalResponse") or ""))
        status = terminal["status"]
        if interrupted_by_user:
            return 130
        if status == "completed":
            return 0
        if status in {"interrupted", "cancelled"}:
            return 130
        if not terminal.get("error"):
            print(json.dumps(terminal, ensure_ascii=False), file=sys.stderr)
        return 1


async def inspect_modules(
    config_path: str = "config.toml",
    workspace: Path | None = None,
) -> None:
    import logging
    from bootstrap.cleanup import run_cleanup_steps
    from bootstrap.tools import build_core_runtime

    logging.getLogger().setLevel(logging.WARNING)
    config = Config.load(config_path)
    http_resources = SharedHttpResources()
    runtime = build_core_runtime(
        config,
        workspace or _default_workspace(),
        http_resources,
    )
    try:
        print(await runtime.inspect_modules())
    finally:
        await run_cleanup_steps(
            ("core.stop", runtime.stop),
            ("memory_runtime.aclose", runtime.memory_runtime.aclose),
            ("http_resources.aclose", http_resources.aclose),
        )


async def serve(
    config_path: str = "config.toml",
    workspace: Path | None = None,
) -> int:
    config = Config.load(config_path)
    workspace = workspace or _default_workspace()
    commit_channel = SupervisorCommitChannel.from_environment()
    restart_coordinator = (
        RestartCoordinator(
            commit_channel.boot_id,
            supervised=True,
            commit=commit_channel.commit,
        )
        if commit_channel is not None
        else None
    )
    readiness = (
        RuntimeReadiness(workspace, commit_channel.boot_id)
        if commit_channel is not None
        else None
    )
    runtime = build_app_runtime(
        config,
        workspace=workspace,
        restart_coordinator=restart_coordinator,
        readiness=readiness,
    )
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    watched_signals = (signal.SIGINT, signal.SIGTERM)
    signal_handlers_registered = False
    for sig in watched_signals:
        try:
            loop.add_signal_handler(sig, stop_event.set)
            signal_handlers_registered = True
        except NotImplementedError:
            # Windows 默认事件循环不支持 add_signal_handler。
            _ = signal.signal(
                sig,
                lambda _sig, _frame: loop.call_soon_threadsafe(stop_event.set),
            )

    runtime_task = asyncio.create_task(runtime.run(), name="app_runtime")
    stop_task = asyncio.create_task(stop_event.wait(), name="shutdown_signal")
    restart_task = (
        asyncio.create_task(
            restart_coordinator.wait_committed(),
            name="restart_committed",
        )
        if restart_coordinator is not None
        else None
    )
    try:
        watched = {runtime_task, stop_task}
        if restart_task is not None:
            watched.add(restart_task)
        done, _ = await asyncio.wait(
            watched,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if runtime_task in done:
            _ = stop_task.cancel()
            await runtime_task
            return 0
        restart_requested = False
        if restart_task is not None and restart_task in done:
            await restart_task
            restart_requested = True
        _ = runtime_task.cancel()
        with suppress(asyncio.CancelledError):
            await runtime_task
        return RESTART_EXIT_CODE if restart_requested else 0
    finally:
        if signal_handlers_registered:
            for sig in watched_signals:
                _ = loop.remove_signal_handler(sig)
        _ = stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        if restart_task is not None:
            _ = restart_task.cancel()
            with suppress(asyncio.CancelledError):
                await restart_task


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(_HELP)
        sys.exit(0)
    config_path = "config.toml"
    workspace: Path | None = None
    force = "--force" in args
    dashboard_host = "0.0.0.0"
    dashboard_port = 2236

    try:
        config_value = _get_flag_value(args, "--config")
        workspace_value = _get_flag_value(args, "--workspace")
        host_value = _get_flag_value(args, "--host")
        port_value = _get_flag_value(args, "--port")
        source_value = _get_flag_value(args, "--source")
        marketplace_value = _get_flag_value(args, "--marketplace")
        ref_value = _get_flag_value(args, "--ref")
        sparse_value = _get_flag_value(args, "--sparse")
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

    if args and args[0] == "setup":
        from bootstrap.setup_wizard import run_setup_wizard
        run_setup_wizard(
            config_path=Path(config_path),
            workspace=workspace or _default_workspace(),
        )
        sys.exit(0)

    if args and args[0] == "setup-main":
        from bootstrap.setup_main import run_main_model_setup

        run_main_model_setup(Path(config_path))
        sys.exit(0)

    if args and args[0] == "init":
        summary = init_workspace(
            config_path=config_path,
            workspace=workspace or _default_workspace(),
            force=force,
        )
        _print_init_summary(summary)
        sys.exit(0)

    if args and args[0] == "plugin-install":
        if not source_value:
            print("plugin-install 缺少 --source")
            sys.exit(1)
        marketplace = marketplace_value or "local"
        result = install_git_plugin(
            source=source_value,
            marketplace=marketplace,
            ref_name=ref_value or "",
            sparse_paths=_parse_csv_flag(sparse_value),
        )
        print(f"已安装插件: {result.plugin_name}@{result.marketplace}")
        print(f"版本: {result.plugin_version}")
        print(f"代码: {result.installed_path}")
        print(f"数据: {result.data_path}")
        sys.exit(0)

    if args and args[0] in {"plugin-enable", "plugin-disable"}:
        if len(args) < 2 or args[1].startswith("--"):
            print(f"{args[0]} 缺少插件 ID")
            sys.exit(1)
        plugin_id = args[1]
        enabled = args[0] == "plugin-enable"
        try:
            manifest = set_installed_plugin_enabled(plugin_id, enabled=enabled)
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)
        print(f"插件已{'启用' if enabled else '禁用'}: {plugin_id}")
        print(f"清单: {manifest}")
        sys.exit(0)

    if args and args[0] == "plugin-uninstall":
        if len(args) < 2 or args[1].startswith("--"):
            print("plugin-uninstall 缺少插件 ID")
            sys.exit(1)
        plugin_id = args[1]
        try:
            cache_path, data_path = uninstall_plugin(
                plugin_id,
                wait_until_disabled=lambda target: _wait_plugin_disabled(
                    config_path,
                    target,
                    workspace,
                ),
            )
        except (ValueError, RuntimeError) as exc:
            print(str(exc))
            sys.exit(1)
        print(f"插件已卸载: {plugin_id}")
        print(f"已删除代码: {cache_path}")
        print(f"已保留数据: {data_path}")
        sys.exit(0)

    if args and args[0] == "plugin-doctor":
        target_plugin_id = ""
        if len(args) >= 2 and not args[1].startswith("--"):
            target_plugin_id = args[1]
        report = run_plugin_doctor(
            plugin_id=target_plugin_id,
            config_path=config_path,
            workspace=workspace or _default_workspace(),
        )
        if "--json" in args:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_plugin_doctor_report(report))
        sys.exit(1 if report.get("status") == "broken" else 0)

    if args and args[0] == "supervise":
        try:
            _validate_supervise_args(args[1:])
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
        sys.exit(
            run_supervisor(
                config_path=Path(config_path),
                workspace=workspace or _default_workspace(),
            )
        )

    if args and args[0] == "gateway":
        sys.exit(asyncio.run(serve(config_path, workspace)))

    if args and args[0] == "app-server":
        if "--stdio" not in args:
            print("app-server 当前必须指定 --stdio", file=sys.stderr)
            sys.exit(2)
        from bootstrap.app_server import run_stdio_app_server

        config = Config.load(config_path)
        asyncio.run(run_stdio_app_server(config, workspace or _default_workspace()))
        sys.exit(0)

    if args and args[0] == "exec":
        try:
            exit_code = asyncio.run(
                run_exec(args, config_path, workspace or _default_workspace())
            )
        except (ValueError, ConnectionError, OSError, RemoteControlError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
        sys.exit(exit_code)

    if args and args[0] == "dashboard":
        config = Config.load(config_path)
        dashboard_workspace = workspace or _default_workspace()
        http_resources = SharedHttpResources()
        provider, light_provider, _ = build_providers(config)
        memory_runtime = build_memory_admin_runtime(
            config=config,
            workspace=dashboard_workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
        )
        try:
            run_dashboard_api(
                workspace=dashboard_workspace,
                host=dashboard_host,
                port=dashboard_port,
                memory_admin=memory_runtime.engine,
            )
        finally:
            asyncio.run(memory_runtime.aclose())
        sys.exit(0)

    if not Path(config_path).exists():
        print(
            f"找不到配置文件 {config_path!r}，请先复制 config.example.toml 为 config.toml。"
        )
        sys.exit(1)

    if "--inspect-modules" in args:
        asyncio.run(inspect_modules(config_path, workspace))
    else:
        asyncio.run(serve(config_path, workspace))
