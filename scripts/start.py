#!/usr/bin/env python3
"""Prepare the product in the terminal and open its ready Web UI."""
from __future__ import annotations

import argparse
from collections.abc import Callable
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
from urllib.request import ProxyHandler, Request, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.plugins.extra_repo import resolve_repo  # noqa: E402


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

    def run(self, command: list[str], *, cwd: Path = ROOT, progress: Callable[[dict[str, object]], None] | None = None) -> None:
        """Keep command output in the log and stop the entire child on cancellation."""
        started = time.monotonic()
        result = None
        with self.log.open("ab") as stream:
            offset = self.log.stat().st_size
            child = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=stream,
                                     start_new_session=True)
            try:
                while progress is not None and child.poll() is None:
                    offset = _relay_events(self.log, offset, progress)
                    time.sleep(0.3)
                result = child.wait()
                if progress is not None:
                    _ = _relay_events(self.log, offset, progress)
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


# 把子进程日志里新增的计时记录（真实耗时步骤的完成）转给显示函数；返回新的读取位置。
def _relay_events(log: Path, offset: int, progress: Callable[[dict[str, object]], None]) -> int:
    with log.open("rb") as stream:
        _ = stream.seek(offset)
        data = stream.read()
    complete = data.rfind(b"\n") + 1
    from scripts.first_install import relay_line

    for line in data[:complete].decode("utf-8", errors="replace").splitlines():
        _ = relay_line(line, progress)
    return offset + complete


# 汇总：已安装但因外部前置条件未满足而保持关闭的包（来自 runtime/disabled-reasons.json）。
def _print_closed_summary(workspace: Path) -> None:
    from agent.plugins.prerequisites import read_disabled

    for plugin_id, row in sorted(read_disabled(workspace).items()):
        print(f"已安装但未启用：{plugin_id.partition('@')[0]}（{row['reason']}）", flush=True)


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


# 首次安装（或上次没做完的首次安装）：返回 True；已有选择而回执缺失等异常存量直接报错，不提问不写文件。
def _first_install_pending(state: Path, workspace: Path, receipt: Path) -> bool:
    from scripts.first_install import clear_intent, load_intent, selection_committed

    committed = selection_committed(workspace)
    if not receipt.exists():
        if committed:
            raise RuntimeError("既有 selection 缺少首次分发 receipt；不能猜测插件来源或重装默认组合。请使用原入口核对已有安装。")
        return True
    if load_intent(state) is None:
        return False
    if committed:
        clear_intent(state)
        return False
    return True


# 提问（或沿用上次的意图）并把选择写成 patch：首装的内容放在标记块里，每次重跑整块重写，块外原样保留（见 first_install.write_patch）。
# 返回要装的扩展包。
def _choose_first_install(state: Path, workspace: Path, distribution: Path, plugin_home: Path, extra_repo: str,
                          profile: str | None, groups: list[str], assume_yes: bool, reselect: bool) -> tuple[str, ...]:
    from agent.plugins.absorbed import read_absorbed
    from scripts.first_install import choose, preset_extra_rows, write_patch

    rows = preset_extra_rows(distribution, os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"))
    choice = choose(repo=extra_repo, home=plugin_home, preset_rows=rows, profile=profile, groups=groups, assume_yes=assume_yes,
                    state=state, reselect=reselect, absorbed=frozenset(item.partition("@")[0] for item in read_absorbed(distribution)))
    write_patch(choice, rows, workspace / "bundle.patch.toml")
    return choice.install


# 重跑时 init 会读 patch：用户在块外新定义了与块内同名的 row 会让它重复而失败，所以 init 之前先按上次的意图整块重写一遍。
def _refresh_patch(state: Path, workspace: Path, distribution: Path) -> None:
    from scripts.first_install import load_intent, preset_extra_rows, write_patch

    intent = load_intent(state)
    patch = workspace / "bundle.patch.toml"
    if intent is not None and patch.exists():
        write_patch(intent, preset_extra_rows(distribution, os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base")), patch)


def prepare_install(preparation: Preparation, core: Path, distribution: Path,
                    python: Path, state: Path, extra_repo: str, extra_inputs: Path | None = None,
                    allow_retire: bool = False, profile: str | None = None, groups: list[str] | None = None,
                    assume_yes: bool = False, reselect: bool = False) -> None:
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
    # 是否首次安装必须在提问与写任何文件之前定下来，与 ensure 的拒绝条件同一个判定：有选择记录而回执缺失就直接报错退出。
    first = _first_install_pending(state, workspace, receipt)
    if first:
        _refresh_patch(state, workspace, distribution)
    if not receipt.exists():
        preparation.step("准备数据目录")
        preparation.run([str(python), str(core / "main.py"), "init", "--config", str(config),
                         "--workspace", str(workspace)], cwd=core)
    # 2. The formal installer is the only owner of profile installation and receipts.
    # 首次安装：先让用户选装哪些扩展包（有终端才提问；--profile/--groups 可覆盖），选择先落成意图记录，再据此写 bundle.patch.toml。
    install_extra: tuple[str, ...] = ()
    if first and extra_inputs is None:
        install_extra = _choose_first_install(state, workspace, distribution, plugin_home, extra_repo, profile, groups or [],
                                              assume_yes, reselect)
    # 首次安装：联网部分（取包、求依赖 wheel）先单独做完，再用离线输入安装；失败在这一步退出，状态不变。
    prefetched: Path | None = None
    from scripts.first_install import Progress
    shown = Progress()
    if first and extra_inputs is None:
        prefetched = state / ".extra-prefetch"
        shutil.rmtree(prefetched, ignore_errors=True)
        preparation.step("下载扩展包")
        try:
            preparation.run([str(python), str(core / "scripts/install_plugin_distribution.py"),
                             "--distribution", str(distribution), "--bundle", str(distribution / "bundles" / (os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base") + ".toml")),
                             "--workspace", str(workspace), "--plugins-home", str(plugin_home), "--config", str(config),
                             "--extra-repo", extra_repo, "--prefetch-extra", str(prefetched),
                             "--install-extra", ",".join(install_extra)], cwd=core, progress=shown)
        except RuntimeError as error:
            shutil.rmtree(prefetched, ignore_errors=True)
            lines = [line for line in preparation.log.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("RuntimeError")]
            raise RuntimeError(f"{error} {lines[-1]}" if lines else str(error)) from error
        extra_inputs = prefetched
    preparation.step("检查已安装功能" if receipt.exists() else "安装默认功能")
    # 发行版已不再自带而 preset 仍启用的插件从 extra 仓库补装；只有存在缺口时才联网。
    command = [str(python), str(core / "scripts/install_plugin_distribution.py"),
               "--distribution", str(distribution), "--bundle", str(distribution / "bundles" / (os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base") + ".toml")),
               "--workspace", str(workspace), "--plugins-home", str(plugin_home), "--config", str(config),
               "--ensure-bundle", "--receipt", str(receipt), "--extra-repo", extra_repo]
    if extra_inputs is not None:
        command += ["--extra-inputs", str(extra_inputs)]
    if install_extra:
        command += ["--install-extra", ",".join(install_extra)]
    if allow_retire:
        command.append("--allow-retire")
    try:
        preparation.run(command, cwd=core, progress=shown if install_extra else None)
        if prefetched is not None:
            shutil.rmtree(prefetched, ignore_errors=True)
        if first:
            from scripts.first_install import clear_intent
            clear_intent(state)
        _print_closed_summary(workspace)
    except RuntimeError as error:
        # 子进程的原因在日志里；补装失败时把"缺哪些包、怎么办"带到终端。
        lines = [line for line in preparation.log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if "无法取得 preset" in line or "preset 启用的插件没有进入选择" in line or "在用的内置插件" in line or "依赖" in line and "wheel" in line or "依赖冲突" in line]
        raise RuntimeError(f"{error} {lines[-1]}" if lines else str(error)) from error


def _local_request(state: Path, url: str) -> Request:
    """本机进程读得到门禁密钥就是主人；未启用门禁插件时不带凭据。"""
    secret = state / "workspace/web-auth/secret"
    headers = {"Authorization": "Bearer " + secret.read_text(encoding="utf-8").strip()} if secret.exists() else {}
    return Request(url, headers=headers)


def _login_url(core: Path, python: Path, state: Path, url: str) -> str:
    """向门禁插件要一条一次性登录链接；插件缺席时打开普通地址。"""
    result = subprocess.run([str(python), str(core / "main.py"), "web-login", "--url", url,
                             "--config", str(state / "config.toml"), "--workspace", str(state / "workspace")],
                            cwd=core, capture_output=True, text=True, env=dict(
                                os.environ, AKASHIC_PLUGIN_HOME=str(state / "plugin-home")))
    lines = result.stdout.strip().splitlines()
    return lines[-1] if result.returncode == 0 and lines and "/_auth/login#t=" in lines[-1] else url


# 最近一次成功启动用的 Core、解释器和地址；--login 以它为准，换端口或更新源码后未重启也能拿到对的链接。
LAUNCH_RECORD = "launch.json"


def _record_launch(state: Path, core: Path, python: Path, url: str) -> None:
    record = {"core": str(core.resolve()), "python": str(python), "url": url}
    (state / LAUNCH_RECORD).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def _launched(state: Path) -> tuple[Path, Path, str] | None:
    try:
        record = json.loads((state / LAUNCH_RECORD).read_text(encoding="utf-8"))
        core, python, url = Path(record["core"]), Path(record["python"]), str(record["url"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (core, python, url) if (core / "main.py").is_file() and python.exists() else None


# ./start --login：用这个数据目录最近一次启动的环境要一条登录链接；服务在不在运行都可以。
def print_login(cache: Path, state: Path, port: int | None, host: str) -> int:
    # 1. 优先用启动记录；没有记录时退回本源码提交已准备好的环境。
    launched = _launched(state)
    if launched is None:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        ready = (cache / revision).resolve()
        if (ready / "complete").is_file():
            launched = (ready / "core", ready / "env/bin/python", f"http://{host}:{port or 2236}")
    if launched is None or not (state / "config.toml").is_file():
        print("还没有安装完成。先运行 ./start，装好后再用 ./start --login。", file=sys.stderr)
        return 1
    # 2. 显式给了 --port 就按它拼地址，否则用启动时的地址。
    core, python, url = launched
    if port is not None:
        url = f"http://{host}:{port}"
    link = _login_url(core, python, state, url)
    if "/_auth/login#t=" not in link:
        print("没能生成登录链接：网页登录插件可能没有启用。直接打开 " + url, file=sys.stderr)
        return 1
    print(link)
    return 0


def run_service(preparation: Preparation, core: Path, python: Path,
                state: Path, port: int, container: bool, open_browser: bool,
                distribution: Path, host: str = "127.0.0.1") -> int:
    """Wait for loaded Web modules, then keep the Supervisor attached to this launch."""
    environment = dict(os.environ, AKASHIC_PLUGIN_HOME=str(state / "plugin-home"),
                       AKASHIC_PLUGIN_DISTRIBUTION=str(distribution),
                       AKASHIC_WEB_PORT=str(port),
                       AKASHIC_WEB_HOST="0.0.0.0" if container else host,
                       AKASHIC_WEB_ALLOW_NON_LOOPBACK="1" if container or host != "127.0.0.1" else "0")
    url = f"http://{'127.0.0.1' if container or host == '0.0.0.0' else host}:{port}"
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
                        with opener.open(_local_request(state, url + "/api/chat/web-ui/bootstrap"),
                                         timeout=3) as response:
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
                public = os.environ.get("AKASHIC_PUBLIC_PORT", "2236")
                preparation.step(f"WebUI 已就绪：http://localhost:{public}。登录链接用 docker compose exec akashic python main.py web-login 获取。")
            else:
                preparation.step(f"WebUI 已就绪：{url}")
                _record_launch(state, core, python, url)
                if open_browser:
                    try:
                        webbrowser.open(_login_url(core, python, state, url))
                    except webbrowser.Error as error:
                        print(f"无法自动打开浏览器：{error}。请手动打开上方地址。", flush=True)
            if not container:
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
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--distribution", type=Path, help="Use the distribution shipped in the container")
    parser.add_argument("--extra-repo", help="extra 插件仓库地址（可为本地路径）；默认取 AKASHIC_EXTRA_REPO 或官方仓库")
    parser.add_argument("--extra-inputs", type=Path,
                        help="预先下载好的 extra 输入目录，不联网；用于离线环境和 CI")
    parser.add_argument("--profile", choices=("full", "minimal", "custom"), help="首次安装的扩展包选择；不给且有终端就提问，没有终端等于 full")
    parser.add_argument("--groups", default="", help="自选的包组（逗号分隔，隐含 custom）")
    parser.add_argument("--yes", action="store_true", help="首次安装不提问不确认")
    parser.add_argument("--login", action="store_true", help="只打印一条一次性登录链接（2 分钟内有效）后退出，不启动服务")
    parser.add_argument("--reselect", action="store_true", help="首次安装失败后重跑时重新选择（默认沿用上次的选择，不再提问）")
    parser.add_argument("--allow-retire", action="store_true",
                        help="确认让在用而发行版不再提供、extra 上游也没有的内置插件消失；默认启动失败并列出它们")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址；computer 浏览器经宿主出口访问 localhost，开发实例不需要改它")
    args = parser.parse_args()
    if args.extra_inputs and (args.profile or args.groups):
        parser.error("--extra-inputs 是预先下载好的输入目录，已经决定了要装的包，不能同时给 --profile/--groups")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    state = args.state.expanduser().resolve()
    if args.login:
        return print_login(args.cache.expanduser().resolve(), state, args.port, args.host)
    if args.port is None:
        args.port = int(os.environ.get("AKASHIC_WEB_PORT", "2236"))
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
                reservation.bind(("0.0.0.0" if args.distribution else args.host, args.port))
                if args.distribution:
                    core, distribution, python = ROOT, args.distribution.resolve(), Path(sys.executable)
                else:
                    core, distribution, python = prepare_source(preparation, args.cache.expanduser().resolve())
                prepare_install(preparation, core, distribution, python, state, resolve_repo(args.extra_repo),
                                args.extra_inputs.resolve() if args.extra_inputs else None, args.allow_retire,
                                args.profile, [item for item in args.groups.split(",") if item], args.yes, args.reselect)
            return run_service(preparation, core, python, state, args.port,
                               args.distribution is not None, not args.no_browser, distribution, args.host)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"启动失败：{error}\n日志：{preparation.log}\n处理原因后重新运行同一启动命令。", file=sys.stderr, flush=True)
            return 1
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
