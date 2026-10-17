"""没有 Docker 时让电脑插件在真实应用里加载，用于验收电脑标签的界面。

它扮演宿主 Workload Controller：不起容器，start 时把电脑的全部 endpoint 指向本机一个
只回 /health 的假 gateway。插件因此能正常加载，标签、新标签页与分屏可以照常验收；
标签里没有桌面画面，会显示“请确认电脑容器在运行”。

用法：
    python -I scripts/computer_ui_fixture.py --socket ~/.cache/akashic-try-all/computer-fake/controller.sock
    接真实容器时用 --endpoint 把端口指过去，例如经 ssh -L 转发的远端测试容器：
    --endpoint gateway=http://127.0.0.1:18080 --endpoint stream=http://127.0.0.1:18081
    启用电脑插件：功能设置里打开，或用同一 workspace 执行 main.py plugin-enable computer@release
    AKASHIC_WORKLOAD_SOCKET=<同一 socket> python scripts/start.py --state <state> --port 4405
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

JsonObject = dict[str, object]
LEASE_FIELDS = ("workspace_id", "plugin_id", "workload", "mode", "transaction_id", "generation_id", "spec_digest")


# 假 gateway：健康检查成功，其余请求一律 503，电脑面板据此显示“容器未运行”。
class _Gateway(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        healthy = self.path.startswith("/health")
        body = b'{"ok":true}' if healthy else b""
        self.send_response(200 if healthy else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    def do_POST(self) -> None:
        self.do_GET()

    def log_message(self, format: str, *args: object) -> None:
        return


class FakeController:
    def __init__(self, gateway_port: int, overrides: dict[str, str]) -> None:
        self._gateway = f"http://127.0.0.1:{gateway_port}"
        self._overrides = overrides
        self._leases: dict[str, JsonObject] = {}

    # 按真实 Controller 的四个动作回执；start 记下 lease，stop 视为容器已删、挂载已释放。
    def dispatch(self, action: object, body: JsonObject) -> JsonObject:
        if action == "status":
            return {"controller_id": "computer-ui-fixture", "leases": len(self._leases)}
        if action == "cleanup_candidates":
            return {"receipts": []}
        if action == "start":
            lease: JsonObject = {name: str(body.get(name, "")) for name in LEASE_FIELDS}
            lease["container_id"] = "fixture-" + secrets.token_hex(6)
            self._leases[cast(str, lease["container_id"])] = lease
            return {"lease": lease, "endpoints": self._endpoints(body), "adopted_from_generation": None}
        if action == "stop":
            lease = cast(JsonObject, body["lease"])
            _ = self._leases.pop(str(lease.get("container_id")), None)
            return {"lease": lease, "container_absent": True, "mounts_released": True}
        raise ValueError(f"Controller action 无效: {action}")

    # 声明的端口默认映射到假 gateway，--endpoint 指定的改用真实地址；端口在请求里是 [name, port, ...] 列表。
    def _endpoints(self, body: JsonObject) -> list[JsonObject]:
        ports = cast(list[list[object]], body.get("ports", []))
        return [{"name": str(port[0]), "url": self._overrides.get(str(port[0]), self._gateway)} for port in ports]

    # 一行 JSON 请求、一行 JSON 回执，与 UnixWorkloadController 的协议一致。
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = cast(JsonObject, json.loads(await reader.readline()))
            body = cast(JsonObject, request.get("body", {}))
            response: JsonObject = {"ok": True, "body": self.dispatch(request.get("action"), body)}
        except (ValueError, KeyError, TypeError, IndexError) as error:
            response = {"ok": False, "error": repr(error)}
        writer.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
        await writer.drain()
        writer.close()


async def serve(socket_path: Path, gateway_port: int, overrides: dict[str, str]) -> None:
    # 1. 假 gateway 在后台线程里跑，只给 Workloads 的健康检查用。
    gateway = ThreadingHTTPServer(("127.0.0.1", gateway_port), _Gateway)
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    # 2. 清掉上次留下的 socket 后监听；应用通过 AKASHIC_WORKLOAD_SOCKET 连过来。
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(FakeController(gateway_port, overrides).handle, str(socket_path))
    print(f"Workload Controller 夹具：{socket_path}，gateway http://127.0.0.1:{gateway_port}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="没有 Docker 时让电脑插件在真实应用里加载")
    _ = parser.add_argument("--socket", type=Path, required=True)
    _ = parser.add_argument("--gateway-port", type=int, default=4417)
    _ = parser.add_argument("--endpoint", action="append", default=[], help="name=url，可重复")
    arguments = parser.parse_args()
    overrides = dict(cast(str, item).split("=", 1) for item in cast(list[str], arguments.endpoint))
    asyncio.run(serve(cast(Path, arguments.socket).expanduser(), cast(int, arguments.gateway_port), overrides))


if __name__ == "__main__":
    main()
