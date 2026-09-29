#!/usr/bin/env python3
"""Prepare the default product, then hand its Web port to the Supervisor."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


class Preparation:
    """Own one launch attempt and expose only fixed retry and status actions."""

    def __init__(self, log: Path) -> None:
        self.log = log
        self.stage = "准备启动"
        self.error = ""
        self.retry = threading.Event()
        self.token = secrets.token_urlsafe(32)

    def step(self, title: str) -> None:
        self.stage = title
        self.error = ""
        print(title, flush=True)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{title}\n")

    def run(self, command: list[str], *, cwd: Path = ROOT) -> None:
        """Keep command output in the log and stop the entire child on cancellation."""
        with self.log.open("ab") as stream:
            child = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=stream,
                                     start_new_session=True)
            try:
                result = child.wait()
            except BaseException:
                # A cancelled build must not continue writing after its owner exits.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        if result:
            raise RuntimeError(f"{self.stage}失败（退出码 {result}）。查看日志，处理原因后重试。")


def create_server(preparation: Preparation, host: str, port: int) -> ThreadingHTTPServer:
    """Serve an asset-free preparation page until the runtime takes the same port."""
    page = (ROOT / "scripts/start.html").read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed_host(self) -> bool:
            host = urlsplit("//" + self.headers.get("Host", "")).hostname
            if host not in {"localhost", "127.0.0.1"}:
                self.reply(403, b"Use localhost or 127.0.0.1", "text/plain")
                return False
            return True

        def do_GET(self) -> None:
            if not self.allowed_host():
                return
            if self.path == "/api/startup":
                data = {"stage": preparation.stage, "error": preparation.error,
                        "token": preparation.token}
                self.reply(200, json.dumps(data).encode(), "application/json")
            elif self.path == "/startup.log":
                # Download only the current launch log, never arbitrary filesystem paths.
                self.reply(200, preparation.log.read_bytes(), "text/plain; charset=utf-8")
            elif self.path == "/":
                self.reply(200, page.encode(), "text/html; charset=utf-8")
            else:
                self.reply(404, b"Not found", "text/plain")

        def do_POST(self) -> None:
            if not self.allowed_host():
                return
            origin = self.headers.get("Origin", "")
            if (self.path != "/api/startup/retry"
                    or urlsplit(origin).netloc != self.headers.get("Host")
                    or self.headers.get("X-Startup-Token") != preparation.token):
                self.reply(403, b"Forbidden", "text/plain")
                return
            if not preparation.error:
                self.reply(409, b"Already running", "text/plain")
                return
            preparation.retry.set()
            self.reply(202, b"{}", "application/json")

    return ThreadingHTTPServer((host, port), Handler)


def prepare_source(preparation: Preparation, cache: Path) -> tuple[Path, Path, Path]:
    """Build one committed source revision and reuse its completed distribution."""
    # 1. Build exact source, without silently substituting HEAD for local edits.
    if sys.version_info < (3, 12):
        raise RuntimeError("需要 Python 3.12 或更新版本。安装后重新运行 ./start。")
    for command in ("git",):
        if shutil.which(command) is None:
            raise RuntimeError(f"缺少 {command}。请安装 Git 和 Node.js 20+，然后点击重试。")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT).returncode:
        raise RuntimeError("源码有未提交修改。请先提交，再重试；开发调试可直接使用 main.py。")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / revision
    if (target / "complete").is_file():
        preparation.step("复用已准备的界面和默认功能")
        ready = target.resolve()
        return ready / "core", ready / "distribution", ready / "env/bin/python"

    for command in ("node", "npm"):
        if shutil.which(command) is None:
            raise RuntimeError(f"缺少 {command}。请安装 Node.js 20+，然后点击重试。")

    # 2. Stage all outputs together. Failed attempts remain available for diagnosis.
    stage = Path(tempfile.mkdtemp(prefix="prepare-", dir=cache))
    python = stage / "env/bin/python"
    preparation.step("安装运行依赖 · 首次启动可能需要几分钟")
    uv = shutil.which("uv")
    if uv:
        preparation.run([uv, "venv", "--python", sys.executable, str(stage / "env")])
        preparation.run([uv, "pip", "install", "--python", str(python),
                         "-r", str(ROOT / "requirements.txt"), str(ROOT / "sdk/python")])
    else:
        preparation.run([sys.executable, "-m", "venv", str(stage / "env")])
        preparation.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"),
                         str(ROOT / "sdk/python")])
    preparation.step("构建界面和默认功能")
    distribution = stage / "distribution"
    preparation.run([str(python), str(ROOT / "scripts/build_plugin_distribution.py"),
                     "--revision", revision, "--output", str(distribution)])
    preparation.run([str(python), "-c",
                     "from pathlib import Path; from scripts.install_plugin_distribution import extract_core; "
                     "import sys; extract_core(Path(sys.argv[1]), Path(sys.argv[2]))",
                     str(distribution), str(stage / "core")])
    # Virtual environments contain absolute paths; keep the stage directory in place.
    (stage / "complete").touch()
    link = cache / (".ready-" + secrets.token_hex(8))
    link.symlink_to(stage.name, target_is_directory=True)
    os.replace(link, target)
    return stage / "core", distribution, python


def prepare_install(preparation: Preparation, core: Path, distribution: Path,
                    python: Path, state: Path) -> None:
    """Initialize only launcher-owned data and retain the installer's existing choices."""
    workspace = state / "workspace"
    config = state / "config.toml"
    plugin_home = state / "plugin-home"
    receipt = workspace / "runtime/distribution-install.json"
    marker = state / "startup.json"
    revision = json.loads((distribution / "distribution.json").read_text())["source_commit"]
    if marker.exists():
        try:
            previous = json.loads(marker.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"安装标记损坏：{marker}。请从备份恢复该文件；已有运行数据不会被重装。") from error
        if previous["source_commit"] != revision:
            raise RuntimeError("此数据目录属于另一软件版本。请使用原版本启动；升级需走正式发布流程，试用新版本可指定新的 --state 目录。")
    # 1. Do not treat an existing installation without a receipt as a fresh product.
    if not marker.exists():
        if config.exists() or (workspace.exists() and any(workspace.iterdir())) or (plugin_home.exists() and any(plugin_home.iterdir())):
            raise RuntimeError("此目录已有运行数据，启动器不会重装默认组合。请使用原入口，或用 --state 指定新的空目录。")
        temporary = marker.with_name(".startup-" + secrets.token_hex(8) + ".json")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "source_commit": revision}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        directory = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    preparation.step("准备数据目录")
    preparation.run([str(python), str(core / "main.py"), "init", "--config", str(config),
                     "--workspace", str(workspace)], cwd=core)
    # 2. The formal installer is the only owner of profile installation and receipts.
    preparation.step("检查已安装功能" if receipt.exists() else "安装默认功能")
    preparation.run([str(python), str(core / "scripts/install_plugin_distribution.py"),
                     "--distribution", str(distribution), "--profile", str(distribution / "profiles/default.json"),
                     "--workspace", str(workspace), "--plugins-home", str(plugin_home), "--config", str(config),
                     "--ensure-profile", "--receipt", str(receipt)], cwd=core)


def main() -> int:
    """Keep preparation retryable, then replace this process with the Supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path.home() / ".akashic")
    parser.add_argument("--cache", type=Path, default=ROOT / ".akashic-start")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AKASHIC_WEB_PORT", "2236")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--non-interactive", action="store_true", help="Exit on preparation failure instead of waiting for a Web retry")
    parser.add_argument("--distribution", type=Path, help="Use the distribution shipped in the container")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    state = args.state.expanduser().resolve()
    os.umask(0o077)
    # Existing direct-runtime owners also block preparation, even on another port.
    for name in (".supervisor.lock", ".instance.lock"):
        path = state / "workspace" / name
        if path.exists():
            with path.open("rb") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print("此数据目录的服务已在运行，请打开原来的网页。", file=sys.stderr)
                    return 2
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (state / ".startup.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("此数据目录已有启动任务。请回到已打开的页面。", file=sys.stderr)
        return 2
    # Keep this product-instance lock in the Supervisor after exec.
    os.set_inheritable(lock.fileno(), True)
    preparation = Preparation(state / f"startup-{secrets.token_hex(4)}.log")
    preparation.log.touch(mode=0o600)
    host = "0.0.0.0" if args.distribution else "127.0.0.1"
    try:
        server = create_server(preparation, host, args.port)
    except OSError as error:
        print(f"无法打开端口 {args.port}: {error}。请停止占用该端口的服务，或使用 --port。", file=sys.stderr)
        return 2
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"打开 {url} 查看准备进度。日志：{preparation.log}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    try:
        while True:
            try:
                if args.distribution:
                    core, distribution, python = ROOT, args.distribution.resolve(), Path(sys.executable)
                else:
                    core, distribution, python = prepare_source(preparation, args.cache.expanduser().resolve())
                prepare_install(preparation, core, distribution, python, state)
                break
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                preparation.error = str(error)
                print(preparation.error, file=sys.stderr, flush=True)
                if args.non_interactive:
                    return 1
                preparation.retry.wait()
                preparation.retry.clear()
                preparation.error = ""
        preparation.step("启动服务 · 即将进入初始配置")
    except KeyboardInterrupt:
        return 130
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    # 3. One listener at a time: release preparation HTTP before Supervisor binds.
    os.environ.update(AKASHIC_PLUGIN_HOME=str(state / "plugin-home"),
                      AKASHIC_WEB_PORT=str(args.port), AKASHIC_WEB_HOST=host,
                      AKASHIC_WEB_ALLOW_NON_LOOPBACK="1" if args.distribution else "0")
    os.chdir(core)
    os.execv(str(python), [str(python), "main.py", "--config", str(state / "config.toml"),
                          "--workspace", str(state / "workspace")])


if __name__ == "__main__":
    raise SystemExit(main())
