#!/usr/bin/env python3
"""真实 Web Shell 进程 + 假守卫与假后端（UDS）：验证入口守卫扩展点的放行、拒绝、失败关闭与换代。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
import httpx
import uvicorn
import websockets

ROOT = Path(__file__).resolve().parents[1]
seen: dict[str, dict[str, str]] = {}


def fake_guard() -> FastAPI:
    app = FastAPI()

    @app.get("/_guard/check")
    async def check(request: Request) -> Response:
        seen["guard"] = dict(request.headers)
        if request.headers.get("x-test-allow") == "1":
            return Response(status_code=204)
        return JSONResponse({"code": "unauthorized"}, 401, headers={"www-authenticate": "Bearer"})
    return app


def fake_backend() -> FastAPI:
    app = FastAPI()

    @app.get("/api/echo")
    async def echo(request: Request) -> dict[str, str]:
        seen["backend"] = dict(request.headers)
        return {"ok": "1"}

    @app.websocket("/api/ws")
    async def socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.close()
    return app


def serve(app: FastAPI, path: Path) -> None:
    server = uvicorn.Server(uvicorn.Config(app, uds=str(path), log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not path.exists():
        time.sleep(0.05)


def publish(workspace: Path, *endpoints: tuple[str, str, str, list[str]], guarded: bool = False) -> None:
    """写一份与 Root 相同形状的 plan；guarded 即选中插件静态声明了 ingress_guard。"""
    rows = [{"name": name, "protocol": protocol, "address": address, "routes": routes,
             "owner": "e2e", "generation_id": "g"} for name, protocol, address, routes in endpoints]
    payload: dict[str, object] = {"generation_id": "g", "endpoints": rows}
    if guarded:
        payload["requires"] = ["guard+unix"]
    (workspace / "runtime/endpoints.json").write_text(json.dumps(payload))


def ready(workspace: Path, boot: str) -> None:
    (workspace / ".runtime-ready.json").write_text(json.dumps({"bootId": boot, "pid": os.getpid(), "state": "ready"}))


async def websocket_status(url: str, allow: bool) -> str:
    try:
        async with websockets.connect(url, additional_headers={"x-test-allow": "1"} if allow else {}):
            return "open"
    except websockets.InvalidStatus as error:
        return str(error.response.status_code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="短路径，避免 AF_UNIX 长度限制")
    parser.add_argument("--port", type=int, default=2298)
    args = parser.parse_args()
    shutil.rmtree(args.output, ignore_errors=True)
    workspace = args.output / "workspace"
    (workspace / "runtime").mkdir(parents=True)
    guard, backend = workspace / "runtime/guard.sock", workspace / "runtime/back.sock"
    serve(fake_guard(), guard)
    serve(fake_backend(), backend)
    shell = subprocess.Popen([sys.executable, "-c", (
        "import sys; from pathlib import Path; from bootstrap.web_shell import create_web_shell_server; "
        "create_web_shell_server(Path(sys.argv[1]), port=int(sys.argv[2]), wait_for_ready=True).run()"),
        str(workspace), str(args.port)], cwd=ROOT)
    base = f"http://127.0.0.1:{args.port}"
    http = httpx.Client(base_url=base, timeout=10)
    route = ("api", "http+unix", str(backend), ["/api"])
    gate = ("gate", "guard+unix", str(guard), [])
    report: dict[str, object] = {}

    def check(name: str, actual: object, expected: object) -> None:
        report[name] = {"actual": actual, "ok": actual == expected}
        print(f"{'PASS' if actual == expected else 'FAIL'} {name}: {actual!r}", flush=True)

    try:
        while True:
            try:
                http.get("/")
                break
            except httpx.TransportError:
                time.sleep(0.1)
        # 1. 外壳等待就绪：Core 就绪前一律拒绝，即使后端端点已发布。
        publish(workspace, route)
        check("before_ready_rejected", http.get("/api/echo").status_code, 503)
        # 2. 就绪且没有插件要求守卫：保持无门直通。
        ready(workspace, "boot-a")
        check("no_guard_passthrough", http.get("/api/echo").status_code, 200)
        # 3. 守卫插件已选中但从未发布守卫（apply 失败或尚未执行）：失败关闭。
        publish(workspace, route, guarded=True)
        check("required_guard_never_published_rejected", http.get("/api/echo").status_code, 503)
        # 4. 守卫发布：拒绝原样返回守卫响应；放行时后端拿到外壳写入的头，客户端伪造的被覆盖。
        publish(workspace, route, gate, guarded=True)
        denied = http.get("/api/echo", headers={"x-akashic-uri": "/_auth/forged"})
        check("guard_denies_with_its_response", (denied.status_code, denied.json(), denied.headers.get("www-authenticate")),
              (401, {"code": "unauthorized"}, "Bearer"))
        check("guard_sees_real_uri", seen["guard"].get("x-akashic-uri"), "/api/echo")
        check("guard_sees_original_host", seen["guard"].get("x-akashic-host"), f"127.0.0.1:{args.port}")
        check("guard_allows", http.get("/api/echo?x=1", headers={"x-test-allow": "1"}).status_code, 200)
        check("backend_gets_peer_and_host", (seen["backend"].get("x-akashic-peer"), seen["backend"].get("x-akashic-host")),
              ("127.0.0.1", f"127.0.0.1:{args.port}"))
        ws = base.replace("http", "ws") + "/api/ws"
        check("websocket_denied", asyncio.run(websocket_status(ws, allow=False)), "403")
        check("websocket_allowed", asyncio.run(websocket_status(ws, allow=True)), "open")
        # 5. 守卫撤下（热重载换代或崩溃）而插件仍被选中：失败关闭，不退回无门。
        publish(workspace, route, guarded=True)
        check("guard_gone_while_required_rejected", http.get("/api/echo").status_code, 503)
        # 6. 两个守卫：拒绝而不是任选其一。
        publish(workspace, route, gate, ("gate2", "guard+unix", str(guard), []), guarded=True)
        check("two_guards_rejected", http.get("/api/echo", headers={"x-test-allow": "1"}).status_code, 503)
        # 7. 插件已停用并重启：plan 不再要求守卫，放开。
        publish(workspace, route)
        ready(workspace, "boot-b")
        check("guard_not_required_passthrough", http.get("/api/echo").status_code, 200)
        # 8. 守卫端点已发布但进程不在：503，不放行。
        publish(workspace, route, ("gate", "guard+unix", str(workspace / "runtime/missing.sock"), []), guarded=True)
        check("guard_unreachable_rejected", http.get("/api/echo").status_code, 503)
    finally:
        shell.terminate()
        shell.wait(timeout=30)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    failed = [name for name, item in report.items() if not item["ok"]]  # type: ignore[index]
    print("全部通过" if not failed else f"失败: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
