"""
入口

主要模式：
  python main.py                    Linux/macOS 由 supervisor 托管，其他平台直接运行 gateway
  python main.py gateway            显式启动未托管 gateway（调试）
  python main.py supervise          显式进入 supervisor（兼容别名）
  python main.py app-server --stdio 启动父进程托管控制面
  python main.py exec ...           非交互执行一个 turn
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tomllib
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

_DEFAULT_WORKSPACE = "~/.akashic/workspace"
_PLUGIN_ROLLOUT_OWNER_TURN_ENV = "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN"
_AGENT_INTERNAL_PLUGIN_COMMANDS = frozenset(
    {
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
            "plugin-install 或 plugin-uninstall；"
            "安装结果需查询 accepted、selected 和 active 状态。"
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
            allow_default=False,
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


    return False


if __name__ == "__main__" and _run_lightweight_command():
    raise SystemExit(0)


from agent.config import Config
from agent.migrations import (
    MigrationOutcome,
    migrate_installation,
)
from agent.restart import RestartGate, SupervisorCommitChannel
from agent.supervisor import RESTART_EXIT_CODE, run_supervisor
from agent.plugins.doctor import format_plugin_doctor_report, run_plugin_doctor
from agent.plugins.manifest import set_plugin_enabled
from bootstrap.app import build_app_runtime
from agent.plugins.entrypoints import invoke_plugin_command
from bootstrap.init_workspace import InitSummary, init_workspace
from bootstrap.runtime_readiness import RuntimeReadiness
from core.net.http import SharedHttpResources

_HELP = """\
用法: python main.py [命令] [选项]

命令:
  setup                         初始化 Core（业务配置在 Web 页面完成）
  init                          非交互初始化配置和工作区
  gateway                       启动未托管 Agent 服务（调试）
  supervise                     显式进入 supervisor（兼容别名）
  app-server --stdio            在 stdio 上运行程序化控制面
  exec --new|--session ID PROMPT 提交程序输入并等待结果
  dashboard                     单独启动 Dashboard
  plugin-install [--update-id ID] 安装 Git 插件
  plugin-install-trusted-batch  离线安装 operator 已信任的 exact v3 插件批次
  plugin-uninstall PLUGIN_ID    卸载插件
  plugin-status [UPDATE_ID]      查询当前插件或指定更新
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
    }:
        return None
    if command in {"", "supervise"} and (not config_path.exists() or not workspace.exists()):
        # 首次启动必须先建立空选择，迁移不能抢先把新目录变成旧 workspace。
        init_workspace(config_path=config_path, workspace=workspace)
    if command in {"init", "setup"} and not workspace.exists():
        # 新建 workspace 由 init_workspace 独占建立基线与空选择；启动迁移
        # 先落 migrations.sqlite3 会把新目录误判成既有 workspace。
        return None
    if command == "gateway" and os.environ.get("AKASHIC_SUPERVISED") == "1":
        return None
    outcome = migrate_installation(config_path, workspace)
    if outcome.state == "migrated":
        print(f"启动迁移完成: migrations={len(outcome.migrations)}")
    return outcome


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
            ("http_resources.aclose", http_resources.aclose),
        )


async def serve(config_path: str, workspace: Path) -> int:
    commit_channel = SupervisorCommitChannel.from_environment()
    if commit_channel is not None:
        commit_channel.stage("gateway.starting")
    config = Config.load(config_path, workspace=workspace)
    if commit_channel is not None:
        commit_channel.stage("config.loaded")
    restart_committed = asyncio.Event()
    restart_commit_error: list[BaseException] = []

    def commit_opaque(request_id: str) -> None:
        if commit_channel is None:
            raise RuntimeError("unmanaged runtime 没有 restart commit channel")
        try:
            commit_channel.commit_opaque(request_id)
        except BaseException as error:
            restart_commit_error.append(error)
            raise
        finally:
            restart_committed.set()

    restart_gate = (
        RestartGate(
            boot_id=commit_channel.boot_id,
            supervised=True,
            commit=commit_opaque,
        )
        if commit_channel is not None
        else None
    )
    readiness = None
    if commit_channel is not None or os.environ.get("AKASHIC_DOCKER_RUNTIME") == "1":
        boot_id = commit_channel.boot_id if commit_channel else uuid4().hex
        os.environ["AKASHIC_BOOT_ID"] = boot_id
        readiness = RuntimeReadiness(workspace, boot_id, commit_channel)
    runtime = build_app_runtime(
        config,
        workspace=workspace,
        restart_gate=restart_gate,
        readiness=readiness,
    )
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    settings_restart_event = asyncio.Event()
    watched_signals = (signal.SIGINT, signal.SIGTERM)
    registered_signal_handlers: set[int] = set()
    fallback_signal_handlers: dict[int, Any] = {}
    for sig in watched_signals:
        try:
            loop.add_signal_handler(sig, stop_event.set)
            registered_signal_handlers.add(sig)
        except NotImplementedError:
            # Windows 默认事件循环不支持 add_signal_handler。
            previous_handler = signal.signal(
                sig,
                lambda _sig, _frame: loop.call_soon_threadsafe(stop_event.set),
            )
            fallback_signal_handlers[sig] = previous_handler
    restart_signal_registered = False
    if commit_channel is not None and hasattr(signal, "SIGUSR2"):
        loop.add_signal_handler(signal.SIGUSR2, settings_restart_event.set)
        restart_signal_registered = True

    async def commit_settings_restart() -> None:
        await settings_restart_event.wait()
        while runtime.core is None:
            await asyncio.sleep(0.05)
        request_id = "settings_" + uuid4().hex
        runtime.core.restart_gate.prepare(request_id)
        await runtime.core.restart_gate.commit(request_id)

    runtime_task: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[bool] | None = None
    restart_task: asyncio.Task[bool] | None = None
    settings_restart_task: asyncio.Task[None] | None = None
    runtime_cancel_requested = False
    result = 0
    primary_error: BaseException | None = None
    deferred_cancellation: asyncio.CancelledError | None = None

    def _cause_chain_contains(
        root: BaseException,
        target: BaseException,
    ) -> bool:
        """Check one standard exception cause chain by identity."""

        current: BaseException | None = root
        seen: set[int] = set()
        while current is not None:
            if current is target:
                return True
            identity = id(current)
            if identity in seen:
                return False
            seen.add(identity)
            current = current.__cause__
        return False

    def _append_visible_error(error: BaseException) -> None:
        """Append one later error to the visible cause chain without cycles."""

        assert primary_error is not None
        existing_ids: set[int] = set()
        tail = primary_error
        while True:
            identity = id(tail)
            if identity in existing_ids:
                return
            existing_ids.add(identity)
            if tail.__cause__ is None:
                break
            tail = tail.__cause__

        candidate: BaseException | None = error
        candidate_ids: set[int] = set()
        while candidate is not None:
            identity = id(candidate)
            if identity in existing_ids or identity in candidate_ids:
                return
            candidate_ids.add(identity)
            candidate = candidate.__cause__
        tail.__cause__ = error

    def _record_error(error: BaseException | None) -> None:
        """Keep a later task error in the existing Python exception chain."""

        nonlocal primary_error
        if error is None:
            return
        if primary_error is None:
            primary_error = error
            return
        if (
            isinstance(primary_error, asyncio.CancelledError)
            and isinstance(error, asyncio.CancelledError)
            and primary_error.__cause__ is None
            and error.__cause__ is None
        ):
            return
        if primary_error is error or _cause_chain_contains(primary_error, error):
            return
        if (
            isinstance(primary_error, asyncio.CancelledError)
            and primary_error.__cause__ is None
            and isinstance(error, asyncio.CancelledError)
            and error.__cause__ is not None
        ):
            primary_error = error
            return
        if _cause_chain_contains(error, primary_error):
            return
        _append_visible_error(error)

    async def _settle_owned_task(
        task: asyncio.Task[object],
        *,
        request_cancel: bool,
    ) -> tuple[BaseException | None, asyncio.CancelledError | None]:
        """Retrieve one CLI-owned task once while deferring caller cancellation."""

        cancel_sent = False
        if request_cancel and not task.done():
            cancel_sent = task.cancel()
        caller_cancel: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.wait((task,))
            except asyncio.CancelledError as error:
                caller_cancel = error
        try:
            task.result()
        except asyncio.CancelledError as task_error:
            if cancel_sent and task_error.__cause__ is None:
                return None, caller_cancel
            return task_error, caller_cancel
        except BaseException as task_error:
            return task_error, caller_cancel
        return None, caller_cancel

    try:
        runtime_task = asyncio.create_task(runtime.run(), name="app_runtime")
        stop_task = asyncio.create_task(stop_event.wait(), name="shutdown_signal")
        restart_task = (
            asyncio.create_task(restart_committed.wait(), name="restart_committed")
            if commit_channel is not None
            else None
        )
        settings_restart_task = (
            asyncio.create_task(commit_settings_restart(), name="settings_restart")
            if commit_channel is not None and hasattr(signal, "SIGUSR2")
            else None
        )
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
            pass
        else:
            restart_requested = False
            if restart_task is not None and restart_task in done:
                restart_requested = True
            if settings_restart_task is not None and settings_restart_task in done:
                restart_requested = True
            runtime_cancel_requested = True
            result = RESTART_EXIT_CODE if restart_requested else 0
    except BaseException as error:
        primary_error = error
        runtime_cancel_requested = (
            runtime_task is not None and not runtime_task.done()
        )
    finally:
        for sig in registered_signal_handlers:
            _ = loop.remove_signal_handler(sig)
        for sig, previous_handler in fallback_signal_handlers.items():
            _ = signal.signal(sig, previous_handler)
        if restart_signal_registered:
            _ = loop.remove_signal_handler(signal.SIGUSR2)
        if stop_task is not None:
            stop_error, caller_cancel = await _settle_owned_task(
                stop_task,
                request_cancel=True,
            )
            deferred_cancellation = caller_cancel or deferred_cancellation
            _record_error(stop_error)
        if restart_task is not None:
            restart_error, caller_cancel = await _settle_owned_task(
                restart_task,
                request_cancel=True,
            )
            deferred_cancellation = caller_cancel or deferred_cancellation
            _record_error(restart_error)
        if settings_restart_task is not None:
            settings_error, caller_cancel = await _settle_owned_task(
                settings_restart_task,
                request_cancel=True,
            )
            deferred_cancellation = caller_cancel or deferred_cancellation
            _record_error(settings_error)
        if restart_commit_error:
            _record_error(restart_commit_error[0])
        if runtime_task is not None:
            runtime_error, caller_cancel = await _settle_owned_task(
                runtime_task,
                request_cancel=runtime_cancel_requested,
            )
            deferred_cancellation = caller_cancel or deferred_cancellation
            _record_error(runtime_error)
        _record_error(deferred_cancellation)
    if primary_error is not None:
        raise primary_error
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(_HELP)
        sys.exit(0)
    config_path = "config.toml"
    workspace: Path
    force = "--force" in args

    try:
        config_value = _get_flag_value(args, "--config")
        if config_value is not None:
            config_path = config_value
        bootstrap_command = bool(args and args[0] in {"setup", "init"})
        supervisor_command = not args or args[0].startswith("--") or args[0] == "supervise"
        workspace = _workspace_from_args(
            args,
            Path(config_path),
            allow_default=bootstrap_command or supervisor_command,
        )
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

    if args and args[0] == "init":
        summary = init_workspace(
            config_path=config_path,
            workspace=workspace,
            force=force,
        )
        _print_init_summary(summary)
        sys.exit(0)

    if args and args[0] in {"plugin-enable", "plugin-disable"}:
        if len(args) < 2 or args[1].startswith("--"):
            print(f"{args[0]} 缺少插件 ID")
            sys.exit(1)
        plugin_id = args[1]
        enabled = args[0] == "plugin-enable"
        try:
            manifest = set_plugin_enabled(plugin_id, enabled=enabled)
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)
        print(f"插件已{'启用' if enabled else '禁用'}: {plugin_id}")
        print(f"清单: {manifest}")
        sys.exit(0)

    if args and args[0] == "plugin-doctor":
        target_plugin_id = ""
        if len(args) >= 2 and not args[1].startswith("--"):
            target_plugin_id = args[1]
        report = run_plugin_doctor(
            plugin_id=target_plugin_id,
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

        config = Config.load(config_path, workspace=workspace)
        asyncio.run(run_stdio_app_server(config, workspace))
        sys.exit(0)

    if args and not args[0].startswith("--") and args[0] != "gateway":
        command_args = list(args[1:])
        for flag in ("--config", "--workspace"):
            while flag in command_args:
                index = command_args.index(flag)
                del command_args[index:index + 2]
        try:
            sys.exit(asyncio.run(invoke_plugin_command(args[0], tuple(command_args),
                workspace=workspace, config_path=Path(config_path))))
        except (LookupError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)

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
