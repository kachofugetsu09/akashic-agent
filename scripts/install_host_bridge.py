#!/usr/bin/env python3
"""用户级安装：Core 跑在发行镜像里，Shell 与文件工具经宿主 Host Bridge 以当前用户执行。不需要 root。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

ORIGIN = "https://github.com/kachofugetsu09/akashic-agent.git"
IMAGE_REPO = "ghcr.io/kachofugetsu09/akashic-agent"
UNIT_DIR = Path.home() / ".config/systemd/user"
BRIDGE_UNIT = "akashic-host-bridge.service"
CORE_UNIT = "akashic.service"


def run(*command: str, capture: bool = False, cwd: Path | None = None) -> str:
    result = subprocess.run(command, check=True, text=True, cwd=cwd,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def step(text: str) -> None:
    print(f"==> {text}", flush=True)


@dataclass(frozen=True)
class Layout:
    """一个安装根目录下的全部路径；同一版本的源码与 venv 按提交分目录，升级时并存。"""
    root: Path
    commit: str

    @property
    def source(self) -> Path:
        return self.root / "src" / self.commit

    @property
    def venv(self) -> Path:
        return self.root / "venv" / self.commit

    @property
    def python(self) -> Path:
        return self.venv / "bin/python"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def run_dir(self) -> Path:
        return self.root / "run"

    @property
    def socket(self) -> Path:
        return self.run_dir / "host-bridge.sock"

    @property
    def token_file(self) -> Path:
        return self.root / "secrets/host-bridge.token"

    @property
    def artifacts(self) -> Path:
        return self.root / "runtime/host-executions"

    @property
    def env_file(self) -> Path:
        return self.root / "compose.env"

    @property
    def overlay(self) -> Path:
        return self.root / "compose.host-bridge.yaml"


def compose_command(layout: Layout, project: str) -> list[str]:
    return ["docker", "compose", "-p", project, "--env-file", str(layout.env_file),
            "-f", str(layout.source / "compose.yaml"), "-f", str(layout.overlay)]


# 没给 --image 时取官方仓库最新的 v* 标签，镜像与源码都固定到它。
def resolve_image(image: str | None) -> str:
    if image:
        return image
    tags = run("git", "ls-remote", "--tags", "--refs", ORIGIN, "v*", capture=True)
    versions = [line.rsplit("/", 1)[1] for line in tags.splitlines()]
    numeric = [v for v in versions if re.fullmatch(r"v\d+(\.\d+)*", v)]
    if not numeric:
        raise RuntimeError("读不到官方发行版本；请用 --image 指定镜像")
    return f"{IMAGE_REPO}:{max(numeric, key=lambda v: tuple(int(x) for x in v[1:].split('.')))}"


# 镜像标签上记录的源码提交，以及镜像里的 Python 主次版本；宿主 bridge 必须与之一致。
def image_identity(image: str) -> tuple[str, str]:
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        run("docker", "pull", image)
    labels = json.loads(run("docker", "image", "inspect", image, "--format", "{{json .Config.Labels}}", capture=True))
    commit = (labels or {}).get("org.opencontainers.image.revision", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"镜像缺少源码提交标签：{image}")
    version = run("docker", "run", "--rm", "--entrypoint", "python", image, "-c",
                  "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')", capture=True)
    return commit, version


def prepare_source(layout: Layout, origin: str) -> None:
    if (layout.source / "main.py").is_file():
        return
    # 旧版本的 compose 不支持状态目录外置，安装器只认自带本脚本的版本。
    step(f"取同版本源码 {layout.commit[:12]}")
    layout.source.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.source.with_name(layout.source.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    run("git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", origin, str(temporary))
    run("git", "-C", str(temporary), "checkout", "--quiet", "--detach", layout.commit)
    if not (temporary / "scripts/install_host_bridge.py").is_file():
        shutil.rmtree(temporary)
        raise RuntimeError("这个版本还不支持用户级安装，请换更新的镜像，或改用 Docker 方式")
    temporary.rename(layout.source)


# bridge 用 uv 管理的独立 Python，主次版本与镜像一致；插件环境两侧才能共用。
# 依赖全部装完才写完成标记；中途失败的半成品环境在重跑时整体重建。
def prepare_venv(layout: Layout, python_version: str) -> Path:
    ready = layout.venv / ".akashic-ready"
    if not ready.exists():
        step(f"准备 Host Bridge 的 Python {python_version} 环境")
        shutil.rmtree(layout.venv, ignore_errors=True)
        run("uv", "python", "install", python_version)
        run("uv", "venv", "--quiet", "--python", python_version, str(layout.venv))
        run("uv", "pip", "install", "--quiet", "--python", str(layout.python), "--require-hashes",
            "-r", str(layout.source / "docker/host-runtime/requirements.lock"))
        run("uv", "pip", "install", "--quiet", "--python", str(layout.python), "--no-deps",
            str(layout.source / "sdk/python"))
        ready.write_text(python_version + "\n")
    return Path(run(str(layout.python), "-c", "import sys; print(sys.base_prefix)", capture=True))


def ensure_token(layout: Layout) -> str:
    layout.token_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not layout.token_file.exists():
        fd = os.open(layout.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
    return layout.token_file.read_text().strip()


# 宿主 /etc/localtime 通常链接到 zoneinfo 下的 IANA 名称；读不出就留空，容器按 UTC。
def host_timezone() -> str:
    target = os.path.realpath("/etc/localtime")
    return target.split("/zoneinfo/", 1)[1] if "/zoneinfo/" in target else ""


def toolchain_digest(commit: str, python_version: str) -> str:
    return hashlib.sha256(f"{commit}:python{python_version}".encode()).hexdigest()


# 1. 状态、bridge 目录、Python 前缀同路径挂进容器，宿主与容器看到的路径一致。
# 2. 工作区和 plugin-home 不预建，由 Core 首次启动自己建立。
def write_compose(layout: Layout, image: str, python_prefix: Path, settings: dict[str, str]) -> None:
    for directory in (layout.state / "home", layout.run_dir, layout.artifacts):
        directory.mkdir(parents=True, exist_ok=True)
    layout.run_dir.chmod(0o700)
    q = json.dumps
    overlay = f"""services:
  akashic:
    image: {q(image)}
    pull_policy: missing
    user: "{os.getuid()}:{os.getgid()}"
    command: ["--state", {q(str(layout.state))}]
    environment:
      AKASHIC_EXECUTION_MODE: host-bridge
      AKASHIC_HOST_BRIDGE_SOCKET: {q(str(layout.socket))}
      AKASHIC_HOST_BRIDGE_TOKEN: ${{AKASHIC_HOST_BRIDGE_TOKEN}}
      AKASHIC_RUNTIME_COMMIT: {q(layout.commit)}
      AKASHIC_HOST_TOOLCHAIN_DIGEST: {q(settings["digest"])}
      AKASHIC_WORKSPACE: {q(str(layout.state / "workspace"))}
      AKASHIC_PLUGIN_HOME: {q(str(layout.state / "plugin-home"))}
      HOME: {q(str(layout.state / "home"))}
      NPM_CONFIG_PREFIX: {q(str(layout.state / "tools"))}
    volumes: !override
      - {q(f"{layout.state}:{layout.state}")}
      - {q(f"{layout.run_dir}:{layout.run_dir}")}
      - {q(f"{python_prefix}:{python_prefix}:ro")}
"""
    layout.overlay.write_text(overlay)
    lines = [f"AKASHIC_REVISION={layout.commit}", f"AKASHIC_PORT={settings['port']}",
             f"AKASHIC_PROXY={settings['proxy']}", f"AKASHIC_NO_PROXY={settings['no_proxy']}", f"TZ={settings['tz']}",
             f"AKASHIC_HOST_BRIDGE_TOKEN={settings['token']}"]
    fd = os.open(layout.env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("\n".join(lines) + "\n")


# bridge 发布给命令用的 akashic-runtime 要指向本实例的配置与工作区。
def bridge_environment(layout: Layout) -> dict[str, str]:
    return {"PYTHONPATH": str(layout.source), "AKASHIC_CONFIG": str(layout.state / "config.toml"),
            "AKASHIC_WORKSPACE": str(layout.state / "workspace"),
            "AKASHIC_PLUGIN_HOME": str(layout.state / "plugin-home")}


# systemd 单元里的参数逐项加引号，路径含空格或 % 也不会被拆开。
def unit_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def bridge_command(layout: Layout, digest: str) -> list[str]:
    return [str(layout.python), "-m", "plugins.host_execution.bridge.server",
            "--socket", str(layout.socket), "--token-file", str(layout.token_file), "--lease-timeout", "305",
            "--artifact-root", str(layout.artifacts), "--release-commit", layout.commit,
            "--toolchain-digest", digest, "--runtime-checkout", str(layout.source),
            "--bridge-python", str(layout.python)]


# 两个用户级单元：bridge 常驻；Core 单元依赖它，负责 compose 起停。开机自启需要 linger。
def install_units(layout: Layout, digest: str, project: str) -> None:
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    compose = " ".join(map(unit_quote, compose_command(layout, project)))
    environment = "\n".join(f"Environment={unit_quote(f'{k}={v}')}" for k, v in bridge_environment(layout).items())
    (UNIT_DIR / BRIDGE_UNIT).write_text(f"""[Unit]
Description=Akashic Host Bridge

[Service]
WorkingDirectory={str(layout.artifacts).replace("%", "%%")}
{environment}
ExecStart={" ".join(map(unit_quote, bridge_command(layout, digest)))}
Restart=on-failure
RestartSec=3s

[Install]
WantedBy=default.target
""")
    (UNIT_DIR / CORE_UNIT).write_text(f"""[Unit]
Description=Akashic
Requires={BRIDGE_UNIT}
After={BRIDGE_UNIT}

[Service]
Type=oneshot
RemainAfterExit=yes
TimeoutStartSec=1200
ExecStart={compose} up -d --no-build --wait --wait-timeout 1200
ExecStop={compose} stop

[Install]
WantedBy=default.target
""")
    run("systemctl", "--user", "daemon-reload")
    run("systemctl", "--user", "enable", BRIDGE_UNIT, CORE_UNIT)
    run("systemctl", "--user", "restart", BRIDGE_UNIT)
    run("systemctl", "--user", "restart", CORE_UNIT)
    if subprocess.run(["loginctl", "enable-linger"], capture_output=True).returncode != 0:
        print("提示：没能开启 linger，退出登录后服务会停；可以运行 sudo loginctl enable-linger $USER")


# 不用 systemd 时（容器、CI、没开 systemd 的 WSL）：bridge 放后台进程，Core 直接 compose up。
def start_foreground(layout: Layout, digest: str, project: str) -> None:
    stop_foreground_bridge(layout)
    log = (layout.root / "host-bridge.log").open("a")
    environment = {**os.environ, **bridge_environment(layout)}
    process = subprocess.Popen(bridge_command(layout, digest), cwd=layout.artifacts, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (layout.root / "host-bridge.pid").write_text(f"{process.pid}\n")
    run(*compose_command(layout, project), "up", "-d", "--no-build", "--wait", "--wait-timeout", "1200")


# 重跑时先停掉上一次留下的 bridge，等它退出（它退出时会删 socket），再起新的。
def stop_foreground_bridge(layout: Layout) -> None:
    pid_file = layout.root / "host-bridge.pid"
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text().strip())
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists() and b"plugins.host_execution.bridge.server" in cmdline.read_bytes():
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not cmdline.exists():
                break
            time.sleep(0.1)
    pid_file.unlink()


def preflight(systemd: bool) -> None:
    needed = ["docker", "git", "uv"] + (["systemctl"] if systemd else [])
    missing = [name for name in needed if shutil.which(name) is None]
    if missing:
        hint = "；uv 用 curl -LsSf https://astral.sh/uv/install.sh | sh 安装" if "uv" in missing else ""
        raise RuntimeError(f"缺少命令：{', '.join(missing)}{hint}")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise RuntimeError("当前用户连不上 Docker；把自己加进 docker 组后重新登录")
    if systemd and subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True).returncode != 0:
        raise RuntimeError("systemd 用户服务不可用；WSL2 需在 /etc/wsl.conf 里开启 systemd，或改用 --no-systemd")


def install(args: argparse.Namespace) -> None:
    preflight(not args.no_systemd)
    image = resolve_image(args.image)
    step(f"使用镜像 {image}")
    commit, python_version = image_identity(image)
    layout = Layout(args.root.expanduser().resolve(), commit)
    prepare_source(layout, args.origin)
    prefix = prepare_venv(layout, python_version)
    digest = toolchain_digest(commit, python_version)
    settings = {"digest": digest, "port": str(args.port), "proxy": args.proxy, "no_proxy": args.no_proxy,
                "tz": args.tz or host_timezone(),
                "token": ensure_token(layout)}
    write_compose(layout, image, prefix, settings)
    # 回执先于启动写下：启动失败时 login、排查与清理也能找到这次的 compose 参数。
    (layout.root / "install.json").write_text(json.dumps(
        {"image": image, "commit": commit, "project": args.project, "systemd": not args.no_systemd}, indent=2) + "\n")
    step("启动 Host Bridge 与 Core（首次约 1–2 分钟）")
    if args.no_systemd:
        start_foreground(layout, digest, args.project)
    else:
        install_units(layout, digest, args.project)
    login(args)


def login(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    receipt = json.loads((root / "install.json").read_text())
    layout = Layout(root, receipt["commit"])
    run(*compose_command(layout, receipt["project"]), "exec", "-T", "akashic", "python", "main.py", "web-login")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "login"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("~/.local/share/akashic"))
    install_parser = sub.choices["install"]
    install_parser.add_argument("--image", help="发行镜像；默认官方最新版本")
    install_parser.add_argument("--origin", default=ORIGIN, help="源码仓库（可为本地路径）")
    install_parser.add_argument("--project", default="akashic")
    install_parser.add_argument("--port", type=int, default=2236)
    install_parser.add_argument("--proxy", default=os.environ.get("AKASHIC_PROXY", ""),
                                help="容器能到的出网代理，如 http://host.docker.internal:7897")
    install_parser.add_argument("--no-proxy", default=os.environ.get("AKASHIC_NO_PROXY", ""),
                                help="不走代理的主机，逗号分隔，如局域网模型服务")
    install_parser.add_argument("--tz", default=os.environ.get("TZ", ""), help="IANA 时区；默认读宿主 /etc/localtime")
    install_parser.add_argument("--no-systemd", action="store_true", help="不装用户服务，bridge 以后台进程运行")
    args = parser.parse_args()
    try:
        install(args) if args.command == "install" else login(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"安装失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
