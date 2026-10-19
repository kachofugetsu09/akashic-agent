#!/usr/bin/env python3
"""真实 ./start 实例 + Chromium：验证门禁插件的拦截、登录、本机免密、重置、停用与程序调用。"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

import httpx
from playwright.sync_api import sync_playwright
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk/python/src"))
sys.path.insert(0, str(ROOT))
from akashic_sdk import AsyncAkashic  # noqa: E402
from plugins.gateway.cli import find_endpoint  # noqa: E402


class Instance:
    """一个隔离 state 的 ./start；浏览器打开请求被记录到文件而非真的打开。"""

    def __init__(self, state: Path, cache: Path, port: int) -> None:
        self.state, self.cache, self.port = state, cache, port
        self.base = f"http://127.0.0.1:{port}"
        self.opened = state.parent / f"{state.name}-opened.txt"
        self.child: subprocess.Popen[bytes] | None = None

    def start(self, *, browser: bool = True) -> None:
        """browser=False 时不等启动器打开浏览器（门失败关闭时启动器探测不会成功），改等聊天端点进入 plan。"""
        self.opened.unlink(missing_ok=True)
        started = time.time()
        capture = self.state.parent / "capture-browser.py"
        capture.write_text(f"import sys; open({str(self.opened)!r}, 'w').write(sys.argv[1])\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("AKASHIC_")}
        env["BROWSER"] = f"{sys.executable} {capture} %s"
        self.log = (self.state.parent / f"{self.state.name}-start.log").open("ab")
        self.child = subprocess.Popen([sys.executable, str(ROOT / "scripts/start.py"), "--state", str(self.state),
                                       "--cache", str(self.cache), "--port", str(self.port)],
                                      cwd=ROOT, env=env, stdout=self.log, stderr=self.log, start_new_session=True)
        deadline = time.monotonic() + 1800
        plan = self.state / "workspace/runtime/endpoints.json"
        while not (self.opened.exists() if browser else plan.exists() and plan.stat().st_mtime > started
                   and '"/ws"' in plan.read_text()):
            if self.child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"./start 未就绪，见 {self.log.name}")
            time.sleep(1)
        if not browser:
            time.sleep(5)

    def stop(self) -> None:
        assert self.child is not None
        os.killpg(self.child.pid, signal.SIGTERM)
        self.child.wait(timeout=120)

    def choose(self, *, enabled: bool) -> None:
        """内置插件的启停写在 workspace bundle patch；与界面里的开关同一入口。"""
        stage = next(path for path in self.cache.glob("prepare-*") if (path / "complete").exists())
        subprocess.run([str(stage / "env/bin/python"), "-c",
                        "import sys; from pathlib import Path; from agent.plugins.bundles import set_plugin_choice; "
                        "set_plugin_choice(Path(sys.argv[1]), 'web-auth@release', enabled=sys.argv[2] == '1', "
                        "distribution=Path(sys.argv[3]))",
                        str(self.state / "workspace"), "1" if enabled else "0", str(stage / "distribution")],
                       cwd=stage / "core", check=True)

    def command(self, *args: str) -> str:
        stage = next(path for path in self.cache.glob("prepare-*") if (path / "complete").exists())
        core = stage / "core"
        result = subprocess.run([str(stage / "env/bin/python"), str(core / "main.py"), *args, "--config", str(self.state / "config.toml"),
                                 "--workspace", str(self.state / "workspace")], cwd=core, check=True, text=True,
                                capture_output=True, env={**os.environ, "AKASHIC_PLUGIN_HOME": str(self.state / "plugin-home")})
        return result.stdout


def login(http: httpx.Client, password: str, remember: bool = True) -> httpx.Response:
    return http.post("/_auth/login", json={"username": "akashic", "password": password, "remember": remember})


async def websocket_status(url: str, cookie: str, origin: str) -> str:
    headers = {"Cookie": f"akashic_session={cookie}"} if cookie else {}
    try:
        async with websockets.connect(url, additional_headers=headers, origin=origin):  # type: ignore[arg-type]
            return "open"
    except websockets.InvalidStatus as error:
        return str(error.response.status_code)


async def programmatic(workspace: Path) -> int:
    async with await AsyncAkashic.connect(find_endpoint(workspace)) as client:
        return len((await client.session_list(limit=1))["items"]) + 1


def off_loop[T](coroutine: Coroutine[object, object, T]) -> T:
    """Playwright 同步 API 占着当前线程的事件循环；异步检查放到独立线程里跑。"""
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def check(report: dict[str, object], name: str, actual: object, expected: object) -> None:
    report[name] = {"actual": actual, "ok": actual == expected}
    print(f"{'PASS' if actual == expected else 'FAIL'} {name}: {actual!r}", flush=True)


def run(instance: Instance, report: dict[str, object]) -> None:
    """按主人与陌生人的真实路径逐项核对；每项结果写入报告。"""
    secret_dir = instance.state / "workspace/web-auth"
    instance.start()
    anon = httpx.Client(base_url=instance.base, follow_redirects=False)
    ws_url = instance.base.replace("http", "ws") + "/ws"
    # 1. 陌生人：页面跳登录，API 401，WS 拒绝；健康检查公开；本机 Bearer 放行。
    check(report, "page_redirects_to_login", anon.get("/chat", headers={"accept": "text/html"}).headers.get("location"),
          "/_auth/login?next=%2Fchat")
    check(report, "api_unauthorized", anon.get("/api/chat/sessions").status_code, 401)
    check(report, "health_public", [anon.get(p).status_code for p in ("/api/shell/state", "/api/chat/health")], [200, 200])
    check(report, "websocket_anonymous", off_loop(websocket_status(ws_url, "", instance.base)), "403")
    local = {"Authorization": "Bearer " + (secret_dir / "secret").read_text().strip()}
    check(report, "local_bearer", anon.get("/api/chat/web-ui/bootstrap", headers=local).status_code, 200)
    initial = (secret_dir / "initial-password").read_text().strip()
    check(report, "initial_password_0600", stat.S_IMODE((secret_dir / "initial-password").stat().st_mode), 0o600)
    check(report, "initial_password_logged", initial in "".join(
        p.read_text(errors="ignore") for p in instance.state.glob("startup-*.log")), True)
    # 2. 初始密码登录后必须先改密码，此前 API 不放行。
    first = httpx.Client(base_url=instance.base)
    check(report, "initial_login_must_change", login(first, initial).json(), {"mustChange": True})
    check(report, "api_before_change", first.get("/api/chat/sessions").status_code, 403)
    with sync_playwright() as playwright:
        # 用系统 Chromium，避免 Playwright 包与缓存浏览器版本不一致。
        browser = playwright.chromium.launch(executable_path=shutil.which("chromium") or None)
        # 3. ./start 自动打开的一次性链接：免密进入，设置密码后进入产品，链接不能复用。
        owner = browser.new_context()
        page = owner.new_page()
        failures: list[str] = []
        page.on("response", lambda r: failures.append(f"{r.status} {r.url}") if r.status in {401, 403} else None)
        sockets: list[str] = []
        page.on("websocket", lambda ws: sockets.append(ws.url))
        ticket_url = instance.opened.read_text().strip()
        page.goto(ticket_url)
        page.wait_for_selector("#change:not([hidden])")
        page.fill("#new1", "owner-password-1")
        page.fill("#new2", "owner-password-1")
        page.click("#change button")
        page.wait_for_url(instance.base + "/")
        page.goto(instance.base + "/")
        page.wait_for_timeout(4000)
        check(report, "owner_chat_loads_without_auth_errors", failures, [])
        check(report, "owner_chat_websocket_opened", any(url.endswith("/ws") or "/ws?" in url for url in sockets), True)
        cookie = next(c for c in owner.cookies() if c.get("name") == "akashic_session")
        session = cookie.get("value", "")
        check(report, "remembered_cookie_90d", round((cookie.get("expires", 0) - time.time()) / 86400), 90)
        check(report, "cookie_httponly_lax", (cookie.get("httpOnly"), cookie.get("sameSite")), (True, "Lax"))
        check(report, "websocket_cross_site", off_loop(websocket_status(ws_url, session, "https://evil.example")), "403")
        check(report, "websocket_same_origin", off_loop(websocket_status(ws_url, session, instance.base)), "open")
        check(report, "initial_password_retired", ((secret_dir / "initial-password").exists(),
              login(httpx.Client(base_url=instance.base), initial).status_code), (False, 401))
        reused = browser.new_context().new_page()
        reused.goto(ticket_url)
        check(report, "ticket_single_use", reused.wait_for_selector("#login .msg").inner_text().startswith("这条登录链接已失效"), True)
        # 4. 别人的电脑：取消“记住”，得到浏览器会话 cookie。
        borrowed = browser.new_context()
        other = borrowed.new_page()
        other.goto(instance.base + "/")
        other.wait_for_selector("#login:not([hidden])")
        other.fill("#password", "owner-password-1")
        other.uncheck("#remember")
        other.click("#login button")
        other.wait_for_url(instance.base + "/")
        temp = next(c for c in borrowed.cookies() if c.get("name") == "akashic_session")
        check(report, "unremembered_session_cookie", temp.get("expires"), -1)
        browser.close()
    owner_http = httpx.Client(base_url=instance.base, cookies={"akashic_session": session})
    # 5. 程序调用不经过门；本机命令重置后旧设备全部失效。
    check(report, "programmatic_unaffected", off_loop(programmatic(instance.state / "workspace")) >= 1, True)
    output = instance.command("web-passwd")
    temporary = output.split("临时密码 ")[1].split()[0]
    check(report, "reset_revokes_devices", owner_http.get("/api/chat/sessions").status_code, 401)
    check(report, "temporary_login_must_change", login(httpx.Client(base_url=instance.base), temporary).json(), {"mustChange": True})
    # 6. 重启后记住的会话仍有效。
    keeper = httpx.Client(base_url=instance.base, headers={"origin": instance.base})
    login(keeper, temporary)
    keeper.post("/_auth/password", json={"password": "owner-password-2"})
    instance.stop()
    instance.start()
    check(report, "session_survives_restart", keeper.get("/api/chat/sessions").status_code, 200)
    # 6b. 门禁插件 apply 失败（socket 路径被占）：门失败关闭，不退回无门；修复后恢复。
    instance.stop()
    blocker = instance.state / "workspace/runtime/web-auth.sock"
    blocker.unlink(missing_ok=True)
    blocker.write_text("not a socket")
    instance.start(browser=False)
    check(report, "guard_plugin_failed_closed", (anon.get("/api/chat/sessions").status_code,
          keeper.get("/api/chat/sessions").status_code), (503, 503))
    instance.stop()
    blocker.unlink()
    instance.start()
    # 7. 停用插件并重启：门随插件消失；重新启用后同一账号继续生效。
    instance.choose(enabled=False)
    instance.stop()
    instance.start()
    check(report, "disabled_plugin_no_gate", anon.get("/api/chat/sessions").status_code, 200)
    # 运行中在 Dashboard 插件列表里打开开关（唯一的在线启用入口）：门立即生效，plan 在本 boot 内记下守卫要求。
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which("chromium") or None)
        page = browser.new_page()
        page.goto(instance.base + "/dashboard#plugins")
        page.fill("input[aria-label='搜索插件']", "web-auth")
        page.locator(".shell-inventory-head", has_text="web-auth").first.click()
        page.check(".shell-inventory-switch input[role='switch']")
        page.wait_for_timeout(5000)
        browser.close()
    plan = json.loads((instance.state / "workspace/runtime/endpoints.json").read_text())
    check(report, "online_enable_gate_and_requires", (anon.get("/api/chat/sessions").status_code,
          keeper.get("/api/chat/sessions").status_code, plan.get("requires")), (401, 200, ["guard+unix"]))
    instance.stop()
    instance.start()
    check(report, "reenabled_gate_and_account", (anon.get("/api/chat/sessions").status_code,
          keeper.get("/api/chat/sessions").status_code), (401, 200))
    # 同一来源连续输错 5 次后退避；另一个来源不受影响。
    for attempt in range(5):
        login(anon, f"wrong-{attempt}")
    other = httpx.Client(base_url=instance.base, headers={"x-forwarded-for": "198.51.100.7"})
    check(report, "throttled_after_5_per_peer", (login(anon, "wrong-6").status_code, login(other, "wrong-7").status_code),
          (429, 401))
    instance.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="短路径，避免 AF_UNIX 长度限制")
    parser.add_argument("--cache", type=Path, required=True, help="./start 构建缓存，可复用")
    parser.add_argument("--port", type=int, default=2296)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instance = Instance(args.output / "state", args.cache.resolve(), args.port)
    report: dict[str, object] = {}
    try:
        run(instance, report)
    finally:
        if instance.child is not None and instance.child.poll() is None:
            instance.stop()
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    failed = [name for name, item in report.items() if not item["ok"]]  # type: ignore[index]
    print("全部通过" if not failed else f"失败: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
