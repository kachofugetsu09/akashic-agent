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

import argparse
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

    return False


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
from agent.plugins.manifest import set_plugin_enabled
from bootstrap.app import build_app_runtime
from bootstrap.dashboard_api import run_dashboard_api
from bootstrap.init_workspace import InitSummary, init_workspace
from bootstrap.runtime_readiness import RuntimeReadiness
from bootstrap.workspace_token import read_workspace_token
from core.net.http import SharedHttpResources
from infra.control.socket import is_tcp_endpoint

_HELP = """\
用法: python main.py [命令] [选项]

命令:
  setup                         运行交互式初始化向导
  init                          非交互初始化配置和工作区
  veda-reset                    备份并重建 workspace 默认人格
  gateway                       启动未托管 Agent 服务（调试）
  supervise                     显式进入 supervisor（兼容别名）
  app-server --stdio            在 stdio 上运行程序化控制面
  exec --new|--session ID PROMPT 提交程序输入并等待结果
  dashboard                     单独启动 Dashboard
  plugin-install --update-id ID 安装 Git 插件候选
  plugin-install-trusted-batch  离线安装 operator 已信任的 exact v3 插件批次
  plugin-uninstall PLUGIN_ID    卸载插件
  plugin-status [UPDATE_ID]      查询当前插件或指定更新
  plugin-promote UPDATE_ID       提交候选发布
  plugin-discard UPDATE_ID       丢弃候选更新
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
    return asyncio.run(_request_plugin_uninstall(endpoint, plugin_id, workspace))


async def _request_plugin_uninstall(
    endpoint: str, plugin_id: str, workspace: Path,
) -> dict[str, object]:
    """由应用 owner 停用、排空并卸载，控制连接只等待结果。"""
    token = read_workspace_token(workspace) if is_tcp_endpoint(endpoint) else None
    async with await ControlClient.connect(endpoint, workspace_token=token) as client:
        result = await client.request("plugin/uninstall", {"plugin_id": plugin_id})
        if not isinstance(result, dict):
            raise RuntimeError("插件卸载响应无效")
        return cast(dict[str, object], result)


async def _wait_exec_result(client: ControlClient, session_id: str, input_id: str,
                            *, json_events: bool) -> dict[str, object]:
    """从当前结果的 seq 继续跟随；订阅建立期间的新消息仍能补读。"""
    query: dict[str, object] = {"session_id": session_id, "input_id": input_id}
    result = cast(dict[str, object], await client.request("programmatic/message/result", query))
    if result["status"] != "open":
        return result
    async with await client.session_follow(session_id, after_seq=cast(int, result["through_seq"])) as feed:
        async for event in feed.events():
            if json_events:
                print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
            if event["type"] == "messages.appended":
                result = cast(dict[str, object], await client.request("programmatic/message/result", query))
                if result["status"] != "open":
                    return result
    raise ConnectionError("消息订阅已关闭；使用原 Session 和 Input 身份恢复查询")


async def _exec_until_stop(client: ControlClient, session_id: str, input_id: str,
                           *, json_events: bool) -> tuple[dict[str, object], bool]:
    """显式 SIGINT 提交 pause；普通连接关闭只停止本地读取。"""
    interrupt = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGINT)
    native_handler = False
    try:
        loop.add_signal_handler(signal.SIGINT, interrupt.set)
        native_handler = True
    except NotImplementedError:
        def on_sigint(_signal: int, _frame: object) -> None:
            _ = loop.call_soon_threadsafe(interrupt.set)
        _ = signal.signal(signal.SIGINT, on_sigint)
    result_task = asyncio.create_task(_wait_exec_result(client, session_id, input_id,
                                                       json_events=json_events), name="exec-result")
    interrupt_task = asyncio.create_task(interrupt.wait(), name="exec-sigint")
    stopped = False
    try:
        done, _ = await asyncio.wait((result_task, interrupt_task), return_when=asyncio.FIRST_COMPLETED)
        if interrupt_task in done and not result_task.done():
            stopped = True
            _ = await client.request("programmatic/message/pause", {
                "session_id": session_id, "message_id": uuid4().hex,
            })
        return await result_task, stopped
    finally:
        _ = result_task.cancel()
        _ = interrupt_task.cancel()
        _ = await asyncio.gather(result_task, interrupt_task, return_exceptions=True)
        if native_handler:
            _ = loop.remove_signal_handler(signal.SIGINT)
        _ = signal.signal(signal.SIGINT, previous)


async def run_exec(args: list[str], config_path: str, workspace: Path) -> int:
    """通过普通程序来源提交 Message，按原 Input 的持久结果退出。"""
    # 1. 每个可重试写入都有调用方身份；CLI 不接受来源或学习属性覆盖。
    parser = argparse.ArgumentParser(prog="exec")
    _ = parser.add_argument("prompt", nargs="?")
    _ = parser.add_argument("--new", action="store_true")
    _ = parser.add_argument("--session")
    _ = parser.add_argument("--message-id")
    _ = parser.add_argument("--resume")
    _ = parser.add_argument("--persist-memory", action="store_true")
    _ = parser.add_argument("--detach", action="store_true")
    output = parser.add_mutually_exclusive_group()
    _ = output.add_argument("--json", action="store_true")
    _ = output.add_argument("--final-only", action="store_true")
    for option in ("--endpoint", "--config", "--workspace"):
        _ = parser.add_argument(option)
    options = parser.parse_args(args[1:])
    if not options.new and options.session is None:
        raise ValueError("exec 需要 --new 或 --session ID")
    if options.persist_memory and not options.new:
        raise ValueError("--persist-memory 只能在 --new 准入时选择")
    if options.detach and options.final_only:
        raise ValueError("--detach 不能与 --final-only 一起使用")
    if options.resume is not None:
        if options.new or options.prompt is not None:
            raise ValueError("--resume 只引用原 Session 的 Input，不接收新 prompt")
    elif options.prompt is None:
        raise ValueError("exec 缺少 prompt；使用 - 从 stdin 读取")
    session_id = options.session or "programmatic:" + uuid4().hex
    message_id = options.message_id or uuid4().hex
    input_id = options.resume or message_id
    endpoint = options.endpoint
    if endpoint is None:
        config = Config.load(config_path, workspace=workspace)
        endpoint = resolve_app_server_endpoint(config.app_server.listen, workspace)
    token = read_workspace_token(workspace) if is_tcp_endpoint(endpoint) else None
    identity: dict[str, object] = {"session_id": session_id, "message_id": message_id, "input_id": input_id}
    print(json.dumps({"type": "message.submitting", **identity}, ensure_ascii=False),
          file=sys.stdout if options.json else sys.stderr, flush=True)

    # 2. 先固定 Session 属性，再提交输入；ACK 不等默认回复。
    async with await ControlClient.connect(endpoint, workspace_token=token) as client:
        if options.new:
            _ = await client.request("programmatic/session/admit", {
                "session_id": session_id, "persist_memory": options.persist_memory,
            })
        if options.resume is not None:
            receipt = await client.request("programmatic/message/resume", identity)
        else:
            prompt = sys.stdin.read() if options.prompt == "-" else options.prompt
            receipt = await client.request("programmatic/message/send", {
                "session_id": session_id, "message_id": message_id, "text": prompt,
            })
        if options.json:
            print(json.dumps({"type": "message.accepted", "receipt": receipt}, ensure_ascii=False), flush=True)
        if options.detach:
            return 0

        # 3. 完成、暂停、失败都来自日志；读取关闭不会伪造成功。
        result, stopped = await _exec_until_stop(client, session_id, input_id, json_events=options.json)
        if options.json:
            print(json.dumps({"type": "message.result", **result}, ensure_ascii=False), flush=True)
        elif result["status"] in {"complete", "quiet"}:
            ending = cast(int, result["ending_seq"])
            page = await client.message_read(session_id, after_seq=ending - 1, through_seq=ending, limit=1)
            row = page["items"][0]
            if row["id"] != result["ending_message_id"]:
                raise RuntimeError("程序结果引用与读取的 Message 不一致")
            print("\n".join(part["value"] for part in row["body"]["parts"] if part["kind"] == "text"))
        else:
            print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        if stopped or result["status"] == "pause":
            return 130
        return 0 if result["status"] in {"complete", "quiet"} else 1


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

    async def commit_settings_restart() -> None:
        await settings_restart_event.wait()
        while runtime.conversation_runtime is None:
            await asyncio.sleep(0.05)
        await runtime.conversation_runtime.quiesce_and_drain()
        assert commit_channel is not None
        commit_channel.commit_settings(f"settings_{uuid4().hex}")

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
                        "update_id": _get_flag_value(args, "--update-id") or "",
                    },
                )
            )
        except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        sys.exit(0)

    if args and args[0] in {"plugin-status", "plugin-promote", "plugin-discard"}:
        update_id = args[1] if len(args) > 1 and not args[1].startswith("--") else None
        if args[0] != "plugin-status" and update_id is None:
            print(f"{args[0]} 缺少更新 ID", file=sys.stderr)
            sys.exit(1)
        method = (
            "plugin/status" if update_id is None else
            "plugin/update" if args[0] == "plugin-status" else
            "plugin/promote" if args[0] == "plugin-promote" else "plugin/discard"
        )
        try:
            result = asyncio.run(_request_runtime_control(
                config_path, workspace, method,
                {} if update_id is None else {"update_id": update_id},
            ))
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
            manifest = set_plugin_enabled(plugin_id, enabled=enabled)
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
                print(json.dumps(runtime_result, ensure_ascii=False))
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
        run_dashboard_api(
            workspace=workspace,
            host=dashboard_host,
            port=dashboard_port,
        )
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
