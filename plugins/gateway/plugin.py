"""Gateway 拥有 JSON-RPC 监听、连接、控制协议和远程命令。"""
from __future__ import annotations

import asyncio
import os
from agent.plugin_composition import CompositionError, Context
from agent.plugin_composition.channel_io import CHANNEL_ATTACHMENT_READ
from agent.plugin_composition.channels import CHANNEL_INPUT_V2
from agent.plugin_composition.control_frames import CONTROL_FRAMES
from agent.plugin_composition.host import HOST_INFO
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugin_composition.tasks import RESTART_GATE
from .factory import build_control_service
from .migrations.gateway_migrations.helpers.settings import GatewayConfig
from .socket import SocketAppServer, is_tcp_endpoint, resolve_endpoint
from .stdio import StdioAppServer
from .token import ensure_token

api_version = 3
name = "gateway"
version = "1.0.0"
desc = "JSON-RPC 控制面与远程命令"
Config = GatewayConfig
inject = (HOST_INFO,)
workspace_files = (".app-server-token",)
entrypoints = {"exec": "cli.exec_main", "plugin-install": "cli.install_main",
               "plugin-status": "cli.status_main", "plugin-uninstall": "cli.uninstall_main",
               "app-server": "cli.app_server_main"}


async def apply(ctx: Context) -> None:
    """端口缺席只挂起监听子 Fiber，不阻塞 Gateway 命令声明。"""
    config = Config.model_validate(ctx.config)
    output_fd = os.environ.get("AKASHIC_GATEWAY_STDIO")
    if not config.enabled and output_fd is None:
        return
    workspace = ctx.runtime.workspace
    endpoint = resolve_endpoint(config.listen, workspace)
    tcp = output_fd is None and is_tcp_endpoint(endpoint)
    if ctx.require(HOST_INFO).validation:
        return

    async def listen(context: Context) -> None:
        token = ensure_token(context.workspace_file(".app-server-token")) if tcp else None
        service = build_control_service(context, workspace_token=token)
        await context.effect(lambda: service.shutdown, label="control-service")
        if output_fd is not None:
            server = StdioAppServer(service, max_message_bytes=config.max_message_bytes,
                                    output_fd=int(output_fd))
            gate = context.require(RESTART_GATE)

            async def run() -> None:
                try:
                    await server.run()
                except asyncio.CancelledError:
                    raise
                except CompositionError as error:
                    # 换代拒绝旧连接的新请求，只结束旧传输，不结束宿主。
                    if error.code not in {"OWNER_UNAVAILABLE", "STALE_ACTIVATION"}:
                        gate.request_shutdown(error)
                except BaseException as error:
                    gate.request_shutdown(error)
                else:
                    gate.request_shutdown()

            async def start_stdio():
                task = asyncio.create_task(run(), name="gateway-stdio")
                async def close() -> None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                return close

            # 原生任务由 effect 排空；调用回执写完后才取消读取循环。
            await context.effect(start_stdio, label="stdio-listener")
        else:
            server = SocketAppServer(endpoint, service, max_connections=config.max_connections,
                max_pending_requests=config.ingress_queue_size, max_message_bytes=config.max_message_bytes,
                outbound_queue_size=config.outbound_queue_size)
            async def start_socket():
                await server.start()
                return server.stop
            await context.effect(start_socket, label="socket-listener")
            await context.endpoint("gateway", protocol="jsonrpc+tcp" if is_tcp_endpoint(str(server.endpoint)) else "jsonrpc+unix",
                                   address=str(server.endpoint))

    await ctx.inject((HOST_INFO, MESSAGE_CATALOG, CHANNEL_INPUT_V2, CHANNEL_ATTACHMENT_READ,
                      PLUGIN_UPDATES, RESTART_GATE, CONTROL_FRAMES), listen, name="listener")
