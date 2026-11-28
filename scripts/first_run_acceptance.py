#!/usr/bin/env python3
"""新用户首装端到端验收：空数据卷起镜像，emulate 当模型，容器里的浏览器走完引导、装包、对话与工具，再重建容器核对保留。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
BROWSER_IMAGE = "mcr.microsoft.com/playwright/python:v1.59.0-noble"
EMULATE_IMAGE = "node:22-bookworm-slim"
EMULATE_VERSION = "0.15.0"
SEED = {"ai": {"baseUrl": "http://emulate:4000", "strict": False, "timing": {"ttftMs": 0, "chunkDelayMs": 0}, "models": ["akashic-mock"], "rules": []}}


def run(*command: str, check: bool = True, capture: bool = False) -> str:
    result = subprocess.run(command, check=check, text=True, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    return result.stdout if capture else ""


class Stack:
    """一次性 Compose project 加 emulate 容器；结束后只删本次创建的东西。"""

    def __init__(self, image: str, proxy: str, output: Path) -> None:
        self.project = "akashic-acc-" + uuid.uuid4().hex[:8]
        self.output = output
        self.proxy = proxy
        self.emulate = self.project + "-emulate"
        self.port = 20000 + int(self.project[-4:], 16) % 10000
        self.image = image
        self.override = output / "compose.override.json"
        self.override.write_text(json.dumps({"services": {"akashic": {"image": image, "pull_policy": "never"}}}))
        (output / "emulate.json").write_text(json.dumps(SEED))

    def compose(self, *args: str, capture: bool = False, check: bool = True) -> str:
        files = ("--env-file", str(self._env_file()), "-f", str(ROOT / "compose.yaml"), "-f", str(self.override))
        return run("docker", "compose", "-p", self.project, *files, *args, capture=capture, check=check)

    def _env_file(self) -> Path:
        path = self.output / "compose.env"
        # 容器网络里的 emulate 走直连；其余出网仍走 AKASHIC_PROXY。
        path.write_text(f"AKASHIC_REVISION=acceptance\nAKASHIC_PROXY={self.proxy}\nAKASHIC_NO_PROXY=emulate\n"
                        f"AKASHIC_PORT={self.port}\n")
        return path

    def up(self) -> None:
        self.compose("up", "-d", "--no-build", "--wait", "--wait-timeout", "600")

    def start_emulate(self) -> None:
        proxy = ("-e", f"npm_config_proxy={self.proxy}", "-e", f"npm_config_https_proxy={self.proxy}") if self.proxy else ()
        run("docker", "run", "-d", "--name", self.emulate, "--network", self.project + "_default",
            "--network-alias", "emulate", "--add-host", "host.docker.internal:host-gateway", *proxy, "-v", f"{self.output}:/cfg:ro", EMULATE_IMAGE,
            "npx", "-y", f"emulate@{EMULATE_VERSION}", "start", "--service", "ai", "--port", "4000",
            "--host", "0.0.0.0", "--seed", "/cfg/emulate.json")

    # 登录链接里是宿主映射端口；浏览器与 Core 共用网络命名空间，改回容器内的 2236。
    def login_url(self) -> str:
        text = ""
        for _ in range(5):
            text = self.compose("exec", "-T", "akashic", "python", "main.py", "web-login", capture=True, check=False)
            if re.search(r"https?://\S+", text):
                break
            time.sleep(3)
        match = re.search(r"https?://\S+", text)
        if match is None:
            raise RuntimeError("web-login 没有打印链接：" + text)
        return re.sub(r"^https?://[^/]+", "http://localhost:2236", match.group(0))

    def browser(self, phase: str) -> int:
        container = run("docker", "compose", "-p", self.project, "ps", "-q", "akashic", capture=True).strip().splitlines()[-1]
        args = ["--phase", phase, "--login", self.login_url(), "--emulate", "http://emulate:4000",
                "--model-base", "http://emulate:4000/v1", "--output", f"/out/{phase}", *self.browser_extra()]
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "0:0"
        # 官方镜像只带浏览器与系统库，Python 包按版本现装到临时 HOME。
        proxy = ("-e", f"HTTPS_PROXY={self.proxy}", "-e", "NO_PROXY=localhost,127.0.0.1,emulate") if self.proxy else ()
        script = ("pip install -q --user --break-system-packages playwright==1.59.0"
                  ' && exec python /w/first_run_browser.py "$@"')
        result = subprocess.run(["docker", "run", "--rm", "--network", f"container:{container}", "--user", user,
                                 "-e", "HOME=/tmp", *proxy, "-v", f"{ROOT / 'scripts'}:/w:ro", "-v", f"{self.output}:/out",
                                 BROWSER_IMAGE, "sh", "-c", script, "browser", *args], check=False)
        return result.returncode

    def browser_extra(self) -> list[str]:
        return []

    def logs(self) -> None:
        (self.output / "core.log").write_text(self.compose("logs", "--no-color", capture=True, check=False))

    def down(self) -> None:
        run("docker", "rm", "-f", self.emulate, check=False)
        self.compose("down", "--volumes", "--remove-orphans")


class HostBridgeStack(Stack):
    """默认安装方式：用 install_host_bridge.py（不装 systemd 单元）装到输出目录，Shell 在宿主执行。"""

    def __init__(self, image: str, proxy: str, output: Path) -> None:
        super().__init__(image, proxy, output)
        self.root = output / "install"
        self.shell_expect = socket.gethostname()
        self.files: list[str] = []

    # 宿主主机名与一个不挂进容器的宿主文件：两者都出现在工具结果里，才说明命令和文件读写在宿主执行。
    def browser_extra(self) -> list[str]:
        marker = self.output / "host-only" / "marker.txt"
        marker.parent.mkdir(exist_ok=True)
        marker.write_text("ACC-HOST-FILE\n")
        return ["--shell-expect", self.shell_expect, "--host-file", str(marker)]

    def compose(self, *args: str, capture: bool = False, check: bool = True) -> str:
        return run(*self.files, *args, capture=capture, check=check)

    def up(self) -> None:
        run(sys.executable, str(ROOT / "scripts/install_host_bridge.py"), "install", "--root", str(self.root),
            "--image", self.image, "--origin", str(ROOT), "--project", self.project, "--port", str(self.port),
            "--proxy", self.proxy, "--no-proxy", "emulate", "--no-systemd")
        sys.path.insert(0, str(ROOT / "scripts"))
        from install_host_bridge import Layout, compose_command
        receipt = json.loads((self.root / "install.json").read_text())
        self.files = compose_command(Layout(self.root, receipt["commit"]), self.project)

    def down(self) -> None:
        super().down()
        pid = self.root / "host-bridge.pid"
        if pid.exists():
            run("kill", pid.read_text().strip(), check=False)


# 汇总两轮浏览器结果：任一步失败即整体失败。
def summarize(output: Path) -> bool:
    ok = True
    for phase in ("fresh", "restart"):
        path = output / phase / "steps.json"
        steps = json.loads(path.read_text()) if path.exists() else [{"step": phase, "ok": False, "detail": "没有结果"}]
        for step in steps:
            ok = ok and bool(step["ok"])
            print(("PASS " if step["ok"] else "FAIL ") + f"{phase}/{step['step']}: {str(step['detail'])[:300]}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="本地已构建的 standalone 镜像")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", default=os.environ.get("AKASHIC_PROXY", ""),
                        help="容器能到的出网代理，如 http://host.docker.internal:7897")
    parser.add_argument("--keep", action="store_true", help="结束后保留实例，便于排查")
    parser.add_argument("--host-bridge", action="store_true", help="按默认安装方式验收：宿主 Host Bridge + 容器 Core")
    args = parser.parse_args()
    output = args.output.resolve()
    # 每轮用新目录，旧的 steps.json 不能冒充本轮结果。
    if output.exists() and any(output.iterdir()):
        parser.error(f"--output 必须是新目录或空目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    stack = (HostBridgeStack if args.host_bridge else Stack)(args.image, args.proxy, output)
    try:
        # 1. 空数据卷首装，emulate 先就位再开浏览器。
        stack.up()
        stack.start_emulate()
        codes = [stack.browser("fresh")]
        # 2. 保留数据卷重建容器，核对会话与配置都还在。
        stack.compose("up", "-d", "--no-build", "--force-recreate", "--wait", "--wait-timeout", "600")
        codes.append(stack.browser("restart"))
    finally:
        stack.logs()
        if not args.keep:
            stack.down()
    return 0 if summarize(output) and codes == [0, 0] else 1


if __name__ == "__main__":
    sys.exit(main())
