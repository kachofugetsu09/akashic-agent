#!/usr/bin/env python3
"""Prepare the product in the terminal and open its ready Web UI."""
from __future__ import annotations

import argparse
import hashlib
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
import socket
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


class Preparation:
    """Report preparation stages and keep detailed command output in one log."""

    def __init__(self, log: Path) -> None:
        self.log = log
        self.stage = "准备启动"

    def step(self, title: str) -> None:
        self.stage = title
        print(title, flush=True)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{title}\n")

    def run(self, command: list[str], *, cwd: Path = ROOT) -> None:
        """Keep command output in the log and stop the entire child on cancellation."""
        started = time.monotonic()
        result = None
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
            finally:
                stream.write((json.dumps({"event": "release.timing", "stage": "product.command",
                                          "command": command, "status": "complete" if result == 0 else "failed",
                                          "seconds": round(time.monotonic() - started, 6)}) + "\n").encode())
        if result:
            raise RuntimeError(f"{self.stage}失败（退出码 {result}）。查看日志，处理原因后重新运行 ./start。")


def prepare_source(preparation: Preparation, cache: Path) -> tuple[Path, Path, Path]:
    """Build one committed source revision and reuse its completed distribution."""
    # 1. Build exact source, without silently substituting HEAD for local edits.
    if sys.version_info < (3, 12):
        raise RuntimeError("需要 Python 3.12 或更新版本。安装后重新运行 ./start。")
    for command in ("git",):
        if shutil.which(command) is None:
            raise RuntimeError(f"缺少 {command}。请安装 Git 和 Node.js 20+，然后重新运行 ./start。")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT).returncode:
        raise RuntimeError("源码有未提交修改。请先提交，再运行 ./start；开发调试可直接使用 main.py。")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / revision
    if (target / "complete").is_file():
        preparation.step("复用已准备的界面和默认功能")
        ready = target.resolve()
        return ready / "core", ready / "distribution", ready / "env/bin/python"

    for command in ("node", "npm"):
        if shutil.which(command) is None:
            raise RuntimeError(f"缺少 {command}。请安装 Node.js 20+，然后重新运行 ./start。")

    # 2. Stage all outputs together. Failed attempts remain available for diagnosis.
    stage = Path(tempfile.mkdtemp(prefix="prepare-", dir=cache))
    environment = prepare_environment(preparation, cache, revision)
    (stage / "env").symlink_to(environment, target_is_directory=True)
    python = environment / "bin/python"
    preparation.step("构建界面和默认功能")
    distribution = stage / "distribution"
    preparation.run([str(python), str(ROOT / "scripts/build_plugin_distribution.py"),
                     "--revision", revision, "--output", str(distribution)])
    preparation.run([str(python), str(ROOT / "scripts/distribution_runtime.py"),
                     "prepare-wheels", "--distribution", str(distribution)])
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


def prepare_environment(preparation: Preparation, cache: Path, revision: str) -> Path:
    """Reuse runtime dependencies when Python, requirements, and SDK are unchanged."""
    # 1. SDK is installed as a package, so its complete tree belongs in this key.
    sdk_tree = subprocess.check_output(["git", "ls-tree", "-r", revision, "--", "sdk/python"], cwd=ROOT)
    inputs = json.dumps({"python_base": str(Path(sys.base_prefix).resolve()), "python_version": sys.version,
                         "requirements_sha256": hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest(),
                         "sdk_tree_sha256": hashlib.sha256(sdk_tree).hexdigest()}, sort_keys=True).encode()
    target = cache / ("dependencies-" + hashlib.sha256(inputs).hexdigest())
    if (target / "complete").is_file():
        preparation.step("复用已准备的运行依赖")
        return target.resolve() / "env"
    # 2. Keep absolute venv paths fixed and publish only a complete dependency set.
    stage = Path(tempfile.mkdtemp(prefix="dependencies-", dir=cache))
    python = stage / "env/bin/python"
    preparation.step("安装运行依赖 · 首次启动可能需要几分钟")
    uv = shutil.which("uv")
    if uv:
        preparation.run([uv, "venv", "--seed", "--python", sys.executable, str(stage / "env")])
        preparation.run([uv, "pip", "install", "--python", str(python),
                         "-r", str(ROOT / "requirements.txt"), str(ROOT / "sdk/python")])
    else:
        preparation.run([sys.executable, "-m", "venv", str(stage / "env")])
        preparation.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"),
                         str(ROOT / "sdk/python")])
    (stage / "complete").touch()
    link = cache / (".ready-" + secrets.token_hex(8))
    link.symlink_to(stage.name, target_is_directory=True)
    os.replace(link, target)
    return stage / "env"


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
        if previous.get("schema_version") != 1 or not isinstance(previous.get("source_commit"), str):
            raise ValueError("安装标记格式不受支持；请使用原入口核对已有安装。")
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
    if not receipt.exists():
        preparation.step("准备数据目录")
        preparation.run([str(python), str(core / "main.py"), "init", "--config", str(config),
                         "--workspace", str(workspace)], cwd=core)
    # 2. The formal installer is the only owner of profile installation and receipts.
    preparation.step("检查已安装功能" if receipt.exists() else "安装默认功能")
    preparation.run([str(python), str(core / "scripts/install_plugin_distribution.py"),
                     "--distribution", str(distribution), "--bundle", str(distribution / "bundles/base.toml"),
                     "--workspace", str(workspace), "--plugins-home", str(plugin_home), "--config", str(config),
                     "--ensure-bundle", "--receipt", str(receipt)], cwd=core)


def run_service(preparation: Preparation, core: Path, python: Path,
                state: Path, port: int, container: bool, open_browser: bool,
                distribution: Path) -> int:
    """Wait for loaded Web modules, then keep the Supervisor attached to this launch."""
    environment = dict(os.environ, AKASHIC_PLUGIN_HOME=str(state / "plugin-home"),
                       AKASHIC_PLUGIN_DISTRIBUTION=str(distribution),
                       AKASHIC_WEB_PORT=str(port),
                       AKASHIC_WEB_HOST="0.0.0.0" if container else "127.0.0.1",
                       AKASHIC_WEB_ALLOW_NON_LOOPBACK="1" if container else "0")
    url = f"http://127.0.0.1:{port}"
    # 1. The Supervisor alone owns HTTP; the launcher only observes readiness.
    preparation.step("启动服务并加载插件")
    with preparation.log.open("ab") as stream:
        child = subprocess.Popen(
            [str(python), str(core / "main.py"), "--config", str(state / "config.toml"),
             "--workspace", str(state / "workspace")], cwd=core, env=environment,
            stdout=stream, stderr=stream, start_new_session=True)
        try:
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 180
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"服务在就绪前退出（退出码 {child.returncode}）。")
                try:
                    with opener.open(url + "/api/shell/state", timeout=3) as response:
                        ready = json.load(response)["chatReady"]
                    if ready:
                        with opener.open(url + "/api/chat/web-ui/bootstrap", timeout=3) as response:
                            json.load(response)["modules"]
                        break
                except (URLError, TimeoutError, ConnectionError):
                    # The Supervisor may listen before the plugin gateway is ready.
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("等待 WebUI 和插件就绪超时（180 秒）。请查看日志后重新启动。")
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            # 2. Present the product only after its plugin gateway can serve the UI.
            if container:
                preparation.step("WebUI 已就绪。请打开 Compose 映射的本机地址（默认 http://localhost:2236）。")
            else:
                preparation.step(f"WebUI 已就绪：{url}")
                if open_browser:
                    try:
                        webbrowser.open(url)
                    except webbrowser.Error as error:
                        print(f"无法自动打开浏览器：{error}。请手动打开上方地址。", flush=True)
            print(f"按 Ctrl+C 停止服务。运行日志：{preparation.log}", flush=True)
            return child.wait()
        finally:
            # 3. Keep shutdown and the state lock under the same launch owner.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main() -> int:
    """Prepare once, report failures in the terminal, and keep the service attached."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path.home() / ".akashic")
    parser.add_argument("--cache", type=Path, default=ROOT / ".akashic-start")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AKASHIC_WEB_PORT", "2236")))
    parser.add_argument("--no-browser", action="store_true")
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
    with (state / ".startup.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("此数据目录已有启动任务，请查看原来的终端。", file=sys.stderr)
            return 2
        preparation = Preparation(state / f"startup-{secrets.token_hex(4)}.log")
        preparation.log.touch(mode=0o600)
        print(f"正在准备 Akashic，完成后将显示 WebUI 地址。日志：{preparation.log}", flush=True)
        for shutdown_signal in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(shutdown_signal, lambda signum, _frame: sys.exit(128 + signum))
        try:
            # Reserve the port during slow builds, then release it for Supervisor.
            with socket.socket() as reservation:
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                reservation.bind(("0.0.0.0" if args.distribution else "127.0.0.1", args.port))
                if args.distribution:
                    core, distribution, python = ROOT, args.distribution.resolve(), Path(sys.executable)
                else:
                    core, distribution, python = prepare_source(preparation, args.cache.expanduser().resolve())
                prepare_install(preparation, core, distribution, python, state)
            return run_service(preparation, core, python, state, args.port,
                               args.distribution is not None, not args.no_browser, distribution)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"启动失败：{error}\n日志：{preparation.log}\n处理原因后重新运行同一启动命令。", file=sys.stderr, flush=True)
            return 1
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
