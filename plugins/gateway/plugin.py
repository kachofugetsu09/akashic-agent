"""Gateway 提供远程 CLI；命令作用域不启动插件 Runtime。"""
from agent.plugin_composition import Context
from .migrations.gateway_migrations.helpers.settings import GatewayConfig

api_version = 3
name = "gateway"
version = "1.0.0"
desc = "JSON-RPC 客户端命令"
Config = GatewayConfig
entrypoints = {"exec": "cli.exec_main", "plugin-install": "cli.install_main",
               "plugin-status": "cli.status_main", "plugin-uninstall": "cli.uninstall_main"}


async def apply(ctx: Context) -> None:
    """命令由独立入口执行，组合激活不创建额外状态。"""
    Config.model_validate(ctx.config)
