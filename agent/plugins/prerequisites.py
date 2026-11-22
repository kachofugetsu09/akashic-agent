"""插件的外部前置条件（docker、command:<命令>）：检测，以及"已装但未启用"的原因记录。只依赖标准库。

声明来自 extra 仓库 index 的 packages.<名>.needs。不满足的包照装但保持关闭，原因写进
<workspace>/runtime/disabled-reasons.json，供设置页、pkg list 与 Agent 查询；启用时重新检测。
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

DOCKER_SOCKET = Path("/var/run/docker.sock")
_RECORD = "runtime/disabled-reasons.json"


def _connects(path: Path, timeout: float) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(timeout)
            stream.connect(str(path))
            return True
    except OSError:
        return False


# docker 可用：Core 容器里问 Workload Controller（AKASHIC_WORKLOAD_SOCKET，computer 的工作负载就经它起容器，是容器里
# 唯一能到达宿主 docker 的通道），不看容器里有没有 docker socket；没有配置控制器（本机直接运行）时看宿主 docker socket 能否 ping。
def _docker_available(timeout: float = 3.0) -> bool:
    controller = os.environ.get("AKASHIC_WORKLOAD_SOCKET", "").strip()
    if controller:
        return _connects(Path(controller), timeout)
    if not DOCKER_SOCKET.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(timeout)
            stream.connect(str(DOCKER_SOCKET))
            stream.sendall(b"GET /_ping HTTP/1.0\r\nHost: docker\r\n\r\n")
            return b"200 OK" in stream.recv(256)
    except OSError:
        return False


# 返回没满足的条件（空表示都满足）；不认识的声明按不满足处理，不猜。
def unmet(needs: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for need in needs:
        if need == "docker":
            satisfied = _docker_available()
        elif need.startswith("command:"):
            satisfied = shutil.which(need.removeprefix("command:")) is not None
        else:
            satisfied = False
        if not satisfied:
            missing.append(need)
    return missing


def describe(missing: Iterable[str]) -> str:
    names = [("Docker（docker socket 不可用）" if item == "docker" else f"命令 {item.removeprefix('command:')}") for item in missing]
    return "需要" + "、".join(names)


# 读原因记录：返回 (有效条目, 问题说明)。文件不存在是正常（没有问题）；坏 JSON、顶层不是对象、条目不是
# {needs: [字符串], reason: 字符串} 都算问题，条目按"未知"处理，不当作没有记录。
def load_disabled(workspace: Path) -> tuple[dict[str, dict[str, object]], str | None]:
    path = workspace / _RECORD
    if not path.exists():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {}, f"{path} 无法读取（{error}）"
    if not isinstance(value, dict):
        return {}, f"{path} 格式无效（顶层必须是对象）"
    rows: dict[str, dict[str, object]] = {}
    for plugin_id, row in cast(dict[str, object], value).items():
        if (isinstance(row, dict) and isinstance(cast(dict[str, object], row).get("reason"), str)
                and isinstance(cast(dict[str, object], row).get("needs"), list)):
            rows[plugin_id] = cast(dict[str, object], row)
        else:
            return rows, f"{path} 里 {plugin_id} 的条目无效"
    return rows, None


def read_disabled(workspace: Path) -> dict[str, dict[str, object]]:
    return load_disabled(workspace)[0]


def _write(workspace: Path, rows: dict[str, dict[str, object]]) -> None:
    path = workspace / _RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            _ = stream.write(json.dumps(rows, ensure_ascii=False, indent=1, sort_keys=True))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


# 读-改-写在跨进程文件锁里做（pkg、Gateway 各自进程都会更新）；记录损坏时不覆盖，原样报错让人处理。
def _update(workspace: Path, change: Callable[[dict[str, dict[str, object]]], bool]) -> None:
    path = workspace / _RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, problem = load_disabled(workspace)
        if problem is not None:
            raise ValueError(f"前置条件原因记录损坏，未改动：{problem}；修复或删除该文件后重试")
        if change(rows):
            _write(workspace, rows)


# 记录某插件因前置条件未满足而保持关闭。
def record_disabled(workspace: Path, plugin_id: str, missing: list[str]) -> None:
    def change(rows: dict[str, dict[str, object]]) -> bool:
        rows[plugin_id] = {"needs": missing, "reason": describe(missing)}
        return True

    _update(workspace, change)


def clear_disabled(workspace: Path, plugin_id: str) -> None:
    _update(workspace, lambda rows: rows.pop(plugin_id, None) is not None)


# 启用前的统一检查：有记录且条件仍不满足就拒绝并说明；满足了清掉记录放行。
def check_before_enable(workspace: Path, plugin_id: str) -> None:
    rows, problem = load_disabled(workspace)
    if problem is not None:
        raise ValueError(f"{plugin_id} 不能启用：前置条件原因记录损坏，无法确认条件是否满足（{problem}）；修复或删除该文件后重试")
    record = rows.get(plugin_id)
    if record is None:
        return
    still = unmet(cast(list[str], record["needs"]))
    if still:
        raise ValueError(f"{plugin_id} 不能启用：{describe(still)}；补齐后再启用")
    clear_disabled(workspace, plugin_id)


# 启用声明了镜像的插件前，确认镜像在 docker 里：本机有 docker 命令就 inspect，缺的先 pull（都有进程级超时）；
# 没有 docker 命令（正式 Core 容器形态）就无法确认镜像存在——Workload Controller 目前没有镜像查询/拉取接口，
# 不能放行，明确拒绝并给出要在宿主上执行的确切命令。run 可注入（演练与验证用）。
INSPECT_TIMEOUT_S = 30.0
PULL_TIMEOUT_S = 600.0


def ensure_images(images: Iterable[str], run: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run) -> list[str]:
    pulled: list[str] = []
    images = tuple(images)
    if images and shutil.which("docker") is None:
        raise ValueError("当前环境没有 docker 命令，无法确认镜像已在宿主上；请先在宿主上拉取后再启用：" + "；".join(f"docker pull {image}" for image in images))
    for image in images:
        try:
            if run(["docker", "image", "inspect", image], capture_output=True, text=True, check=False, timeout=INSPECT_TIMEOUT_S).returncode == 0:
                continue
            done = run(["docker", "pull", image], capture_output=True, text=True, check=False, timeout=PULL_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"镜像 {image} 的 docker 操作超时；请在宿主上手动执行 docker pull {image}") from error
        if done.returncode != 0:
            raise ValueError(f"镜像 {image} 拉取失败：{(done.stderr or '').strip().splitlines()[-1:] or done.returncode}")
        pulled.append(image)
    return pulled
