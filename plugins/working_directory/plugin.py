from plugins.host_execution.contract import FILES
from agent.plugin_composition import Context
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest
from plugins.ledger.contract import OWNER_STATE, SESSION_ADMISSION
from plugins.working_directory.contract import WORKING_DIRECTORY

from plugins.context.contract import MATERIALS_V4
from ._tool_boundary import TOOLS
from .working_directory import WorkingDirectories
from .directory_tool import register_directory
from .agents import read_agents, directory_material
from plugins.ledger.contract import Message

api_version = 3
name = "working_directory"
version = "1.0.0"
desc = "管理会话工作目录"
inject = (TOOLS, FILES, MATERIALS_V4,
          OWNER_STATE, SESSION_ADMISSION)



async def apply(ctx: Context) -> None:
    """注册既有工具的普通入口；安装和归档装配不访问文件、进程或网络。"""
    catalog = ctx.require(TOOLS)
    directories = WorkingDirectories(ctx.require(OWNER_STATE).open(ctx), ctx.require(FILES))
    _ = await ctx.provide(WORKING_DIRECTORY, directories)
    _ = await ctx.require(SESSION_ADMISSION).register_initializer(
        ctx, name="working-directory", initialize=directories.initialize,
    )

    async def materials(snapshot: tuple[Message, ...], _source: str) -> dict[str, object]:
        if not snapshot:
            return {}
        current = directories.snapshot(snapshot[-1].session_id)
        if current.path is None:
            return {}
        rules = await read_agents(ctx.require(FILES), current.path)
        return {"reminders": ({"name": "working-directory", "priority": 400, "replay": False,
                               "text": directory_material(current.path, str(rules["directory_status"]), rules)},)}

    _ = await ctx.require(MATERIALS_V4).register(
        ctx, name="working-directory", kind="context", prepare=materials,
    )

    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    await register_directory(ctx, directories)

    # 顶栏工作目录是本插件自己的界面；宿主只提供插槽。
    async def plugin_ui(child: Context) -> None:
        async def query(method: str, payload: dict[str, object], *,
                        session_id: str | None, turn_id: str | None) -> dict[str, object]:
            if method != "directory.current" or payload or not session_id:
                raise PluginUiRpcInvalidRequest("请选择 Session 读取当前目录")
            return await directories.current_info(session_id)

        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="plugin_ui.js", slots=("header.end",)), query=query,
        )

    _ = await ctx.inject((UI_SLOTS,), plugin_ui, name="plugin-ui")
