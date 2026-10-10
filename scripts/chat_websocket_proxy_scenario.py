"""真实 TCP Web Shell → UDS Chat 链路验证目录通知与查询参数透传。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, closing
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket

from agent.plugin_composition.endpoints import Endpoint, save_endpoint_plan
from bootstrap.web_shell import create_web_shell_app
from docker.debug.session_metadata_scenario import append, fixture
from session.log import MessageLog


@asynccontextmanager
async def serve(app, *, uds=None):
    """用启动完成事件协调真实服务，只绑定一次性 socket。"""
    ready = asyncio.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            ready.set()

    with socket.socket(socket.AF_UNIX if uds else socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(str(uds) if uds else ('127.0.0.1', 0))
        server = Server(uvicorn.Config(app, log_level='error', timeout_graceful_shutdown=1))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            yield sock.getsockname()
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)


async def run(workspace: Path):
    """通过公网入口相同的代理路径观察元数据提交，并保留旧客户端合同。"""
    # 1. Chat 使用真实日志与目录；Dashboard 对端只回显原始传输参数。
    with closing(MessageLog(workspace / 'sessions.db')) as log:
        append(log, 'akashic:proxy', '代理查询参数验收')
        before = log.reader('akashic:proxy').snapshot()
        dashboard = FastAPI()

        @dashboard.websocket('/api/dashboard/echo')
        async def echo(ws: WebSocket):
            await ws.accept()
            await ws.send_text(ws.scope['query_string'].decode('ascii'))
            await ws.close()

        chat_socket = workspace / "chat.sock"
        dashboard_socket = workspace / "dashboard.sock"
        shell = create_web_shell_app(workspace)
        async with serve(fixture(log, workspace), uds=chat_socket), \
                serve(dashboard, uds=dashboard_socket), serve(shell) as address:
            save_endpoint_plan(workspace / "runtime/endpoints.json", "fixture", (
                Endpoint("client", "http+unix", str(chat_socket), ("/api/chat", "/ws"), "client", "fixture"),
                Endpoint("dashboard", "http+unix", str(dashboard_socket), ("/",), "ui", "fixture"),
            ))
            origin = f'127.0.0.1:{address[1]}'
            async with httpx.AsyncClient(base_url=f'http://{origin}') as client:
                async with websockets.connect(f'ws://{origin}/ws?watch_sessions=true', proxy=None) as ws:
                    assert json.loads(await asyncio.wait_for(ws.recv(), 3)) == {'type': 'sessions.changed', 'version': 2}
                    # 2. 真实 HTTP 管理提交必须经同一代理触发 WebSocket 通知。
                    response = await client.post('/api/chat/sessions/akashic:proxy/rename', json={'title': '代理后的标题'})
                    response.raise_for_status()
                    assert json.loads(await asyncio.wait_for(ws.recv(), 3)) == {'type': 'sessions.changed', 'version': 2}
                    response = await client.get('/api/chat/sessions')
                    response.raise_for_status()
                    assert response.json()['items'][0]['title'] == '代理后的标题'
                # 3. 无订阅及显式关闭仍仅收到 pong；不因透传而默认启用新协议。
                for suffix in ('', '?watch_sessions=false'):
                    async with websockets.connect(f'ws://{origin}/ws{suffix}', proxy=None) as ws:
                        await ws.send(json.dumps({'type': 'ping', 'request_id': 'legacy'}))
                        assert json.loads(await asyncio.wait_for(ws.recv(), 3)) == {'type': 'pong', 'request_id': 'legacy'}
                query = 'slot=drawer.panel&value=a%2Fb%20c&value=second&empty='
                async with websockets.connect(f'ws://{origin}/api/dashboard/echo?{query}', proxy=None) as ws:
                    assert await asyncio.wait_for(ws.recv(), 3) == query
        assert log.reader('akashic:proxy').snapshot() == before
        assert not log._listeners
    return {'status': 'passed', 'transport': 'TCP shell -> UDS chat', 'metadata_notifications': 2,
            'legacy_and_false_opt_out': True, 'dashboard_raw_query_preserved': True,
            'messages_preserved': len(before), 'listeners_after_close': 0}


if __name__ == '__main__':
    with TemporaryDirectory(prefix='chat-ws-proxy-') as folder:
        print(json.dumps(asyncio.run(run(Path(folder)))))
