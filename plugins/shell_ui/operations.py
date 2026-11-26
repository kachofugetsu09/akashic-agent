"""设置·插件页的 Self-CLI operation：与 Dashboard 路由共用同一个 PluginInventory。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from agent.plugin_composition import Context
from plugins.operations.contract import (
    OPERATIONS_V1,
    OperationHandler,
    OperationRequest,
    OperationResult,
    OperationSpec,
    OptionSpec,
)

from .installable import InstallBusy
from .inventory import PluginInventory, Row


# 与路由相同的错误边界：忙、名字不对、子命令失败都是这次请求的拒收，不当程序错误。
def _guarded(summary: str, call: Callable[[OperationRequest], Awaitable[Row]]) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        try:
            data = await call(req)
        except InstallBusy as error:
            return OperationResult(False, str(error), error_code="busy")
        except PermissionError as error:
            return OperationResult(False, str(error), error_code="forbidden")
        except LookupError as error:
            return OperationResult(False, str(error), error_code="not_found")
        except (RuntimeError, ValueError, OSError) as error:
            return OperationResult(False, str(error), error_code="failed")
        return OperationResult(True, summary, data=data)
    return handle


async def register_ops(ctx: Context, service: PluginInventory) -> None:
    installer = service.installer

    async def read(_req: OperationRequest) -> Row:
        return service.read()

    async def set_enabled(req: OperationRequest) -> Row:
        return await service.set_enabled(str(req.args["plugin_id"]), bool(req.args["enabled"]))

    async def install(req: OperationRequest) -> Row:
        names = req.args["names"]
        items = cast(list[object], names) if isinstance(names, list) else []
        if not isinstance(names, list) or not all(isinstance(name, str) for name in items):
            raise ValueError("--names 须为包名的 JSON 数组")
        return await installer.start([str(name) for name in items], bool(req.args.get("first_install", False)))

    async def decline(_req: OperationRequest) -> Row:
        await installer.decline()
        return {"first_install": False}

    specs = (
        OperationSpec("shell_ui.plugins", ("plugin", "list"), "列出插件、启用状态与阶段", _guarded("插件列表已读取", read),
                      mirrors=("http:shell_ui:GET /api/dashboard/plugins",)),
        OperationSpec("shell_ui.set_enabled", ("plugin", "set-enabled"), "启用或停用插件，等待调和后返回", _guarded("插件启停已提交", set_enabled), (
            OptionSpec("plugin_id", "插件 ID", required=True), OptionSpec("enabled", "true 启用，false 停用", "bool", required=True),
        ), mirrors=("http:shell_ui:POST /api/dashboard/plugins/{plugin_id}/enabled",)),
        OperationSpec("shell_ui.installable", ("plugin", "installable"), "列出可安装的扩展包", _guarded("可安装列表已读取", lambda _req: installer.available()),
                      mirrors=("http:shell_ui:GET /api/dashboard/plugins/installable",)),
        OperationSpec("shell_ui.install", ("plugin", "install"), "后台安装扩展包（同一时间只允许一个任务）", _guarded("安装任务已开始", install), (
            OptionSpec("names", "包名 JSON 数组，如 [\"feeds\"]", "json", required=True),
            OptionSpec("first_install", "来自引导「选择功能」的首次挑选或其重试", "bool"),
        ), mirrors=("http:shell_ui:POST /api/dashboard/plugins/install",)),
        OperationSpec("shell_ui.install_status", ("plugin", "install-status"), "读取最近一次安装任务的进度", _guarded("安装进度已读取", lambda _req: installer.status()),
                      mirrors=("http:shell_ui:GET /api/dashboard/plugins/install",)),
        OperationSpec("shell_ui.decline_first_install", ("plugin", "first-install", "decline"), "首装只用核心，不装扩展包", _guarded("已跳过首装挑选", decline),
                      mirrors=("http:shell_ui:DELETE /api/dashboard/plugins/first-install",)),
        OperationSpec("shell_ui.runtime_version", ("runtime", "version"), "读取运行版本、部署记录与宿主桥状态",
                      _guarded("运行版本已读取", lambda _req: service.read_runtime_version()),
                      mirrors=("http:shell_ui:GET /api/dashboard/runtime-version",)),
    )
    operations = ctx.require(OPERATIONS_V1)
    for spec in specs:
        _ = await operations.register(ctx, spec)
