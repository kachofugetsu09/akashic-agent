"""
入口

主要模式：
  python main.py                    Linux/macOS 由 supervisor 托管，其他平台直接运行 gateway
  python main.py gateway            显式启动未托管 gateway（调试）
  python main.py supervise          显式进入 supervisor（兼容别名）
  python main.py app-server --stdio 启动父进程托管控制面
  python main.py exec ...           非交互执行一个 turn
  python main.py veda-reset         重建 workspace 默认人格
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tomllib
from contextlib import suppress
from pathlib import Path
from typing import cast
from uuid import uuid4

_DEFAULT_WORKSPACE = "~/.akashic/workspace"
_PLUGIN_ROLLOUT_OWNER_TURN_ENV = "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN"
_PLUGIN_ROLLOUT_CAPABILITY_ENV = "AKASHIC_PLUGIN_ROLLOUT_CAPABILITY"
_AGENT_INTERNAL_PLUGIN_COMMANDS = frozenset(
    {
        "plugin-status",
        "plugin-promote",
        "plugin-discard",
        "plugin-enable",
        "plugin-disable",
    }
)


def _reject_agent_internal_plugin_action(command: str) -> None:
    if (
        os.environ.get(_PLUGIN_ROLLOUT_OWNER_TURN_ENV)
        and command in _AGENT_INTERNAL_PLUGIN_COMMANDS
    ):
        raise ValueError(
            f"{command} 是 Core 内部维护动作。当前 turn 只应使用 "
            "plugin-install、plugin-uninstall 或 plugin-revert；"
            "安装验证正确后直接结束本轮，系统会自动切换。"
        )


def _supervisor_readiness_timeout() -> float:
    return float(os.environ.get("AKASHIC_READINESS_TIMEOUT_S", "300"))


def _supervisor_supported(platform: str | None = None) -> bool:
    current = platform or sys.platform
    return current.startswith("linux") or current == "darwin"


def _workspace_from_config(config_path: Path) -> str:
    """从主配置读取 workspace，并拒绝缺失或错误的边界值。"""

    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    runtime: object = data.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"配置文件 {config_path!s} 缺少 [runtime] table")
    workspace: object = cast(dict[str, object], runtime).get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError(f"配置文件 {config_path!s} 缺少 runtime.workspace")
    return workspace


def _workspace_from_args(
    args: list[str],
    config_path: Path,
    *,
    allow_default: bool = False,
) -> Path:
    """按命令行、环境变量、配置文件的顺序解析 workspace。"""

    # 1. 显式启动参数拥有最高优先级
    if "--workspace" in args:
        index = args.index("--workspace")
        if index + 1 >= len(args):
            raise ValueError("参数 --workspace 缺少值")
        value = args[index + 1]
    else:
        value = os.environ.get("AKASHIC_WORKSPACE", "")

    # 2. 环境变量为空时读取 config.toml；首次初始化使用可移植默认值
    if not value.strip():
        if config_path.exists():
            value = _workspace_from_config(config_path)
        elif allow_default:
            value = _DEFAULT_WORKSPACE
        else:
            raise ValueError(
                f"找不到配置文件 {config_path!s}，且未指定 --workspace PATH"
            )
    value = value.strip()
    return Path(value).expanduser().resolve()


def _get_flag_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        raise ValueError(f"参数 {flag} 缺少值")
    return args[idx + 1]


def _run_lightweight_command() -> bool:
    """在加载 Agent runtime 依赖前分发恢复与纯配置命令。"""
    args = sys.argv[1:]
    if not args or args[0] not in {
        "plugin-install-trusted-batch",
        "setup-main",
        "veda-reset",
    }:
        return False
    command = args[0]
    config_path = "config.toml"
    if "--config" in args:
        index = args.index("--config")
        if index + 1 >= len(args):
            raise SystemExit("参数 --config 缺少值")
        config_path = args[index + 1]
    try:
        workspace = _workspace_from_args(
            args,
            Path(config_path),
            allow_default=command == "veda-reset",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if command == "plugin-install-trusted-batch":
        from agent.plugins.trusted_install import install_trusted_plugin_batch
        from agent.plugins.manifest import plugins_root
        from bootstrap.workspace_lock import (
            PluginPublicationLock,
            WorkspaceMaintenanceLock,
        )

        if os.environ.get(_PLUGIN_ROLLOUT_OWNER_TURN_ENV):
            raise SystemExit(
                "plugin-install-trusted-batch 只接受外部 operator，不能由 active turn 调用"
            )
        if "--confirm-trusted" not in args:
            raise SystemExit(
                "plugin-install-trusted-batch 需要 --confirm-trusted 明确信任整个 batch"
            )
        try:
            batch_value = _get_flag_value(args, "--batch")
            plugins_home_value = _get_flag_value(args, "--plugins-home")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if batch_value is None:
            raise SystemExit("plugin-install-trusted-batch 缺少 --batch PATH")
        plugins_home = (
            plugins_root().resolve(strict=False)
            if plugins_home_value is None
            else Path(plugins_home_value).expanduser().resolve()
        )
        workspace_lock = WorkspaceMaintenanceLock(workspace)
        publication_lock = PluginPublicationLock(plugins_home)
        try:
            workspace_lock.acquire()
            publication_lock.acquire()
            receipt = install_trusted_plugin_batch(
                workspace=workspace,
                batch_path=Path(batch_value).expanduser().resolve(),
                plugins_home=plugins_home,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        finally:
            publication_lock.release()
            workspace_lock.release()
        if "--json" in args:
            print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")))
        else:
            print("可信离线批量安装完成；本次未执行 programmatic 验证。")
            for item in cast(list[dict[str, object]], receipt["plugins"]):
                print(f"{item['pluginId']}: {item['sourceRevision']}")
        return True

    if command == "veda-reset":
        from agent.persona import reset_veda

        try:
            result = reset_veda(workspace)
        except (OSError, RuntimeError) as exc:
            raise SystemExit(f"Veda 重建失败: {exc}") from exc
        if not result.changed:
            print(f"Veda 已是默认内容: {result.path}")
            print(f"sha256={result.default_sha256}")
            return True
        print(f"Veda 已重建: {result.path}")
        if result.backup_path is not None:
            print(f"原内容备份: {result.backup_path}")
            print(f"原内容 sha256={result.previous_sha256}")
        print(f"默认内容 sha256={result.default_sha256}")
        print("新人格从下一次提示词组装开始生效。")
        return True

    import click

    from agent.migrations import migrate_installation
    from bootstrap.setup_main import run_main_model_setup

    try:
        _ = migrate_installation(Path(config_path), workspace)
        run_main_model_setup(Path(config_path), workspace)
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        click.echo("已取消。", err=True)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        raise SystemExit(f"启动迁移失败: {exc}") from exc
    return True


if __name__ == "__main__" and _run_lightweight_command():
    raise SystemExit(0)


from agent.config import Config, resolve_app_server_endpoint
from agent.control.client import ControlClient, RemoteControlError
from agent.migrations import (
    MigrationOutcome,
    migrate_installation,
)
from agent.restart import RestartCoordinator, SupervisorCommitChannel
from agent.supervisor import RESTART_EXIT_CODE, run_supervisor
from agent.persona import read_veda
from agent.plugins.doctor import format_plugin_doctor_report, run_plugin_doctor
from agent.plugins.install import (
    set_installed_plugin_enabled,
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
  veda-reset                    备份并重建 workspace 默认人格
  gateway                       启动未托管 Agent 服务（调试）
  supervise                     显式进入 supervisor（兼容别名）
  app-server --stdio            在 stdio 上运行程序化控制面
  exec --new|--thread ID PROMPT 执行一个非交互 turn
  dashboard                     单独启动 Dashboard
  plugin-install                安装 Git 插件
  plugin-install-trusted-batch  离线安装 operator 已信任的 exact v3 插件批次
  plugin-uninstall PLUGIN_ID    卸载插件
  plugin-revert                 撤销本 turn 最近一次插件操作
  plugin-doctor [PLUGIN_ID]     检查插件状态

通用选项:
  --config PATH                 配置文件，默认 config.toml
  --workspace PATH              覆盖 config.toml 中的 runtime.workspace
  -h, --help                    显示帮助

无命令时启动 Agent 服务。
"""


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


def _prepare_startup_migrations(
    args: list[str],
    config_path: Path,
    workspace: Path,
) -> MigrationOutcome | None:
    """只为会加载本地 runtime 的命令执行启动迁移。"""

    command = args[0] if args and not args[0].startswith("--") else ""
    if command not in {
        "",
        "setup",
        "init",
        "supervise",
        "gateway",
        "app-server",
        "dashboard",
    }:
        return None
    if command == "gateway" and os.environ.get("AKASHIC_SUPERVISED") == "1":
        return None
    outcome = migrate_installation(config_path, workspace)
    if outcome.state == "migrated":
        print(f"启动迁移完成: migrations={len(outcome.migrations)}")
    return outcome


def _parse_csv_flag(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


async def _request_runtime_control(
    config_path: str,
    workspace: Path,
    method: str,
    params: dict[str, object],
) -> dict[str, object]:
    """向当前 Gateway 发起一个 runtime-owned control 操作。"""

    config = Config.load(config_path, workspace=workspace)
    endpoint = resolve_app_server_endpoint(config.app_server.listen, workspace)
    token = read_workspace_token(workspace) if is_tcp_endpoint(endpoint) else None
    async with await ControlClient.connect(endpoint, workspace_token=token) as client:
        result = await client.request(method, params)
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} 响应无效")
    return cast(dict[str, object], result)


def _uninstall_via_runtime(
    config_path: str,
    plugin_id: str,
    workspace: Path,
) -> dict[str, object]:
    if not Path(config_path).is_file():
        raise RuntimeError("plugin-uninstall 需要正在运行的 Core 和有效配置")
    config = Config.load(config_path, workspace=workspace)
    endpoint = resolve_app_server_endpoint(
        config.app_server.listen,
        workspace,
    )
    owner_turn_id = os.environ.get(_PLUGIN_ROLLOUT_OWNER_TURN_ENV, "")
    return asyncio.run(
        _request_plugin_uninstall(
            endpoint,
            plugin_id,
            workspace,
            owner_turn_id=owner_turn_id,
        )
    )


async def _request_plugin_uninstall(
    endpoint: str,
    plugin_id: str,
    workspace: Path,
    *,
    owner_turn_id: str,
) -> dict[str, object]:
    """Register a turn-owned uninstall without waiting on its own lease."""

    # 1. Runtime records intent only; terminal resolution owns drain and cleanup.
    token = read_workspace_token(workspace) if is_tcp_endpoint(endpoint) else None
    async with await ControlClient.connect(endpoint, workspace_token=token) as client:
        result = await client.request(
            "plugin/uninstall/start",
            {"pluginId": plugin_id, "ownerTurnId": owner_turn_id},
        )
        if not isinstance(result, dict):
            raise RuntimeError("插件卸载登记响应无效")
        return cast(dict[str, object], result)


def _exec_prompt(args: list[str]) -> str:
    values_with_argument = {
        "--config",
        "--workspace",
        "--endpoint",
        "--thread",
        "--runtime",
    }
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
    runtime_value = _get_flag_value(args, "--runtime")
    if runtime_value is not None and runtime_value not in {"stable", "latest"}:
        raise ValueError("exec --runtime 必须是 stable 或 latest")
    rollout_capability = os.environ.get(_PLUGIN_ROLLOUT_CAPABILITY_ENV, "")
    if rollout_capability and runtime_value is not None:
        raise ValueError("插件自验证由 Core 自动选择候选版本，请移除 --runtime")
    if "--persist-memory" in args and not new_thread:
        raise ValueError("exec --persist-memory 只能与 --new 一起使用")
    if "--detach" in args and "--final-only" in args:
        raise ValueError("exec --detach 不能与 --final-only 一起使用")
    prompt = _exec_prompt(args)
    if "--json" in args and "--final-only" in args:
        raise ValueError("exec 的 --json 与 --final-only 不能同时使用")
    endpoint = _get_flag_value(args, "--endpoint")
    if endpoint is None:
        config = Config.load(config_path, workspace=workspace)
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
            metadata: dict[str, object] = (
                {} if "--persist-memory" in args else {"skip_post_memory": True}
            )
            thread = await client.start_thread(
                metadata,
                runtime=runtime_value or "stable",
                plugin_rollout_capability=rollout_capability,
            )
            thread_id = str(thread["id"])
        assert thread_id is not None
        handle = await client.start_turn(
            thread_id,
            prompt,
            runtime=(runtime_value if not new_thread else None),
            detached="--detach" in args,
        )
        if "--detach" in args:
            detached_result = {"threadId": thread_id, "turn": handle.record}
            if "--json" in args:
                print(
                    json.dumps(
                        detached_result, ensure_ascii=False, separators=(",", ":")
                    )
                )
            else:
                print(f"thread: {thread_id}")
                print(f"turn: {handle.id}")
            return 0
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

        event_task = asyncio.create_task(
            consume_events(), name=f"exec-events:{handle.id}"
        )
        interrupt_task = asyncio.create_task(
            interrupt_requested.wait(), name="exec-sigint"
        )
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


async def inspect_modules(config_path: str, workspace: Path) -> None:
    import logging
    from bootstrap.cleanup import run_cleanup_steps
    from bootstrap.tools import build_core_runtime

    logging.getLogger().setLevel(logging.WARNING)
    config = Config.load(config_path, workspace=workspace)
    http_resources = SharedHttpResources()
    runtime = build_core_runtime(
        config,
        workspace,
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


async def serve(config_path: str, workspace: Path) -> int:
    commit_channel = SupervisorCommitChannel.from_environment()
    if commit_channel is not None:
        commit_channel.stage("gateway.starting")
    _ = read_veda(workspace)
    config = Config.load(config_path, workspace=workspace)
    if commit_channel is not None:
        commit_channel.stage("config.loaded")
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
        RuntimeReadiness(workspace, commit_channel.boot_id, commit_channel)
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
    settings_restart_event = asyncio.Event()
    settings_reload_event = asyncio.Event()
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
    if commit_channel is not None and hasattr(signal, "SIGUSR2"):
        loop.add_signal_handler(signal.SIGUSR2, settings_restart_event.set)
    if commit_channel is not None and hasattr(signal, "SIGUSR1"):
        loop.add_signal_handler(signal.SIGUSR1, settings_reload_event.set)

    async def commit_settings_restart() -> None:
        await settings_restart_event.wait()
        while runtime.conversation_runtime is None:
            await asyncio.sleep(0.05)
        await runtime.conversation_runtime.quiesce_and_drain()
        assert commit_channel is not None
        commit_channel.commit_settings(f"settings_{uuid4().hex}")

    async def apply_settings_reloads() -> None:
        """Apply every Supervisor-requested model reload in the live Gateway."""

        assert commit_channel is not None
        while True:
            await settings_reload_event.wait()
            settings_reload_event.clear()
            try:
                result = await runtime.reload_model_config(config_path)
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "运行时模型配置重载失败",
                    exc_info=exc,
                )
                commit_channel.settings_reloaded(
                    success=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            else:
                commit_channel.settings_reloaded(
                    success=True,
                    detail=str(result["configDigest"]),
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
    settings_restart_task = (
        asyncio.create_task(commit_settings_restart(), name="settings_restart")
        if commit_channel is not None and hasattr(signal, "SIGUSR2")
        else None
    )
    settings_reload_task = (
        asyncio.create_task(apply_settings_reloads(), name="settings_reload")
        if commit_channel is not None and hasattr(signal, "SIGUSR1")
        else None
    )
    try:
        watched = {runtime_task, stop_task}
        if restart_task is not None:
            watched.add(restart_task)
        if settings_restart_task is not None:
            watched.add(settings_restart_task)
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
        if settings_restart_task is not None and settings_restart_task in done:
            await settings_restart_task
            restart_requested = True
        _ = runtime_task.cancel()
        with suppress(asyncio.CancelledError):
            await runtime_task
        return RESTART_EXIT_CODE if restart_requested else 0
    finally:
        if signal_handlers_registered:
            for sig in watched_signals:
                _ = loop.remove_signal_handler(sig)
        if commit_channel is not None and hasattr(signal, "SIGUSR2"):
            _ = loop.remove_signal_handler(signal.SIGUSR2)
        if commit_channel is not None and hasattr(signal, "SIGUSR1"):
            _ = loop.remove_signal_handler(signal.SIGUSR1)
        _ = stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        if restart_task is not None:
            _ = restart_task.cancel()
            with suppress(asyncio.CancelledError):
                await restart_task
        if settings_restart_task is not None:
            _ = settings_restart_task.cancel()
            with suppress(asyncio.CancelledError):
                await settings_restart_task
        if settings_reload_task is not None:
            _ = settings_reload_task.cancel()
            with suppress(asyncio.CancelledError):
                await settings_reload_task


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(_HELP)
        sys.exit(0)
    config_path = "config.toml"
    workspace: Path
    force = "--force" in args
    dashboard_host = "0.0.0.0"
    dashboard_port = 2236

    try:
        config_value = _get_flag_value(args, "--config")
        if config_value is not None:
            config_path = config_value
        bootstrap_command = bool(args and args[0] in {"setup", "init"})
        supervisor_command = not args or args[0] == "supervise"
        workspace = _workspace_from_args(
            args,
            Path(config_path),
            allow_default=bootstrap_command or supervisor_command,
        )
        host_value = _get_flag_value(args, "--host")
        port_value = _get_flag_value(args, "--port")
        source_value = _get_flag_value(args, "--source")
        marketplace_value = _get_flag_value(args, "--marketplace")
        ref_value = _get_flag_value(args, "--ref")
        sparse_value = _get_flag_value(args, "--sparse")
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    os.environ["AKASHIC_WORKSPACE"] = str(workspace)
    try:
        _reject_agent_internal_plugin_action(args[0] if args else "")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    if args and args[0] == "supervise" and not _supervisor_supported():
        print("supervise 仅支持 Linux 和 macOS", file=sys.stderr)
        sys.exit(2)
    if host_value is not None:
        dashboard_host = host_value
    if port_value is not None:
        dashboard_port = int(port_value)

    try:
        migration_outcome = _prepare_startup_migrations(
            args,
            Path(config_path),
            workspace,
        )
    except RuntimeError as exc:
        print(f"启动迁移失败: {exc}", file=sys.stderr)
        sys.exit(1)

    if args and args[0] == "setup":
        from bootstrap.setup_wizard import run_setup_wizard

        run_setup_wizard(
            config_path=Path(config_path),
            workspace=workspace,
        )
        sys.exit(0)

    if args and args[0] == "setup-main":
        from bootstrap.setup_main import run_main_model_setup

        run_main_model_setup(Path(config_path), workspace)
        sys.exit(0)

    if args and args[0] == "init":
        summary = init_workspace(
            config_path=config_path,
            workspace=workspace,
            force=force,
        )
        _print_init_summary(summary)
        sys.exit(0)

    if args and args[0] == "plugin-install":
        if not source_value:
            print("plugin-install 缺少 --source")
            sys.exit(1)
        marketplace = marketplace_value or "local"
        try:
            result = asyncio.run(
                _request_runtime_control(
                    config_path,
                    workspace,
                    "plugin/install",
                    {
                        "source": source_value,
                        "marketplace": marketplace,
                        "ref": ref_value or "",
                        "sparse": _parse_csv_flag(sparse_value),
                        "ownerTurnId": os.environ.get(
                            _PLUGIN_ROLLOUT_OWNER_TURN_ENV, ""
                        ),
                    },
                )
            )
        except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        if "--json" in args:
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        else:
            print(str(result["message"]))
            print(f"版本: {result['version']}")
            print(f"代码: {result['installedPath']}")
            print(f"数据: {result['dataPath']}")
        sys.exit(0)

    if args and args[0] == "plugin-revert":
        owner_turn_id = os.environ.get(_PLUGIN_ROLLOUT_OWNER_TURN_ENV, "")
        try:
            result = asyncio.run(
                _request_runtime_control(
                    config_path,
                    workspace,
                    "plugin/revert",
                    {"ownerTurnId": owner_turn_id},
                )
            )
        except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        if "--json" in args:
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        else:
            print(str(result["message"]))
        sys.exit(0)

    if args and args[0] == "plugin-status":
        try:
            result = asyncio.run(
                _request_runtime_control(
                    config_path,
                    workspace,
                    "plugin/status",
                    {},
                )
            )
        except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        sys.exit(0)

    if args and args[0] in {"plugin-promote", "plugin-discard"}:
        if len(args) < 2 or args[1].startswith("--"):
            print(f"{args[0]} 缺少插件 ID", file=sys.stderr)
            sys.exit(1)
        method = "plugin/promote" if args[0] == "plugin-promote" else "plugin/discard"
        try:
            result = asyncio.run(
                _request_runtime_control(
                    config_path,
                    workspace,
                    method,
                    {"pluginId": args[1]},
                )
            )
        except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
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
            runtime_result = _uninstall_via_runtime(
                config_path,
                plugin_id,
                workspace,
            )
            if "--json" in args:
                print(
                    json.dumps(
                        runtime_result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            else:
                print(str(runtime_result["message"]))
            sys.exit(0)
        except (ValueError, RuntimeError) as exc:
            print(str(exc))
            sys.exit(1)
        raise AssertionError("plugin-uninstall 应在 runtime response 后退出")

    if args and args[0] == "plugin-doctor":
        target_plugin_id = ""
        if len(args) >= 2 and not args[1].startswith("--"):
            target_plugin_id = args[1]
        report = run_plugin_doctor(
            plugin_id=target_plugin_id,
            config_path=config_path,
            workspace=workspace,
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
                workspace=workspace,
                readiness_timeout_s=_supervisor_readiness_timeout(),
            )
        )

    if args and args[0] == "gateway":
        sys.exit(asyncio.run(serve(config_path, workspace)))

    if args and args[0] == "app-server":
        if "--stdio" not in args:
            print("app-server 当前必须指定 --stdio", file=sys.stderr)
            sys.exit(2)
        from bootstrap.app_server import run_stdio_app_server

        _ = read_veda(workspace)
        config = Config.load(config_path, workspace=workspace)
        asyncio.run(run_stdio_app_server(config, workspace))
        sys.exit(0)

    if args and args[0] == "exec":
        try:
            exit_code = asyncio.run(run_exec(args, config_path, workspace))
        except (ValueError, ConnectionError, OSError, RemoteControlError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
        sys.exit(exit_code)

    if args and args[0] == "dashboard":
        config = Config.load(config_path, workspace=workspace)
        dashboard_workspace = workspace
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

    if "--inspect-modules" in args:
        asyncio.run(inspect_modules(config_path, workspace))
    elif not _supervisor_supported():
        print(
            "警告：当前平台不支持 Supervisor；将以 unmanaged gateway 运行，"
            "agent_restart、设置重启和 boot 进程树清理不可用。",
            file=sys.stderr,
        )
        sys.exit(asyncio.run(serve(config_path, workspace)))
    else:
        sys.exit(
            run_supervisor(
                config_path=Path(config_path),
                workspace=workspace,
                readiness_timeout_s=_supervisor_readiness_timeout(),
            )
        )
