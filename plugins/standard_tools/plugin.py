from plugins.host_execution.contract import PROCESSES, FILES
from agent.plugin_composition import Context
from importlib import import_module

from plugins.ui.contract import UI, UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest
from plugins.ledger.contract import ARTIFACT_IMPORT
from plugins.assets.contract import INSTALLED_ASSETS
from plugins.ledger.contract import BINDINGS
from agent.plugin_composition.tasks import TASKS
from plugins.ledger.contract import OWNER_STATE, SESSION_ADMISSION
from plugins.standard_tools.contract import WORKING_DIRECTORY

from plugins.context.contract import MATERIALS_V4
from ._tool_boundary import TOOLS
from .files import register_file
from .filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from .shell import register_shell
from .command_log import COMMAND_LOG_VIEW, CommandLog
from .skills import register_skills
from .working_directory import WorkingDirectories
from .directory_tool import register_directory
from .vision import register_vision
from plugins.models.contract import CHAT_MODELS
from .agents import read_agents, directory_material
from plugins.ledger.contract import Message

api_version = 3
name = "standard_tools"
version = "1.0.0"
desc = "提供文件、命令与技能读取工具"
inject = (BINDINGS, TASKS, TOOLS, PROCESSES, FILES, ARTIFACT_IMPORT, MATERIALS_V4, INSTALLED_ASSETS,
          OWNER_STATE, SESSION_ADMISSION, CHAT_MODELS)



async def apply(ctx: Context) -> None:
    """注册既有工具的普通入口；安装和归档装配不访问文件、进程或网络。"""
    _ = await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")
    catalog = ctx.require(TOOLS)
    store = ctx.require(OWNER_STATE).open(ctx)
    directories = WorkingDirectories(store, ctx.require(FILES))
    commands = CommandLog(ctx, store)
    _ = await ctx.provide(COMMAND_LOG_VIEW, commands)
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

    async def directory_ui(child: Context) -> None:
        async def query(method: str, payload: dict[str, object], *,
                        session_id: str | None, turn_id: str | None) -> dict[str, object]:
            if method != "directory.current" or payload or not session_id:
                raise PluginUiRpcInvalidRequest("请选择 Session 读取当前目录")
            return await directories.current_info(session_id)

        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="plugin_ui.js"), query=query,
        )

    _ = await ctx.inject((UI_SLOTS,), directory_ui, name="directory-ui")

    # 终端标签与它读取的命令记录接口；记录不发布给其他插件。没有对话工具区时标签不出现。
    async def terminal_ui(child: Context) -> None:
        _ = await child.require(UI).register(
            child, web="web_module.js",
            dashboard=lambda: import_module(".dashboard", __package__),
            requires=("conversation.tools.v2",),
            contract_digests={
                "conversation.tools.v2": "438131c0fee0caf2ba0c8509d9312ccda6c2c195ad4ce29c26da404e8d15292c",
            },
        )

    _ = await ctx.inject((UI, COMMAND_LOG_VIEW), terminal_ui, name="terminal-ui")
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    await register_directory(ctx, directories)
    await register_vision(ctx, directories)
    for backend in (ReadFileTool, ListDirTool, WriteFileTool, EditFileTool):
        await register_file(
            ctx,
            backend,
            directories=directories,
            allowed_dir=(
                ctx.runtime.workspace
                if backend in (ReadFileTool, ListDirTool)
                else None
            ),
        )
    await register_shell(ctx, directories, commands)
    await register_skills(ctx)
