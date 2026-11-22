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
_DECLARED = "runtime/declared-needs.json"
# 前置条件相关的两份状态文件：安装事务的恢复点（snapshot）一并保存与还原。
RECORD_FILES = (_RECORD, _DECLARED)


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


# 前置条件声明只有两种合法形式：docker、command:<命令>。
def _valid_need(value: object) -> bool:
    return isinstance(value, str) and (value == "docker" or (value.startswith("command:") and len(value) > len("command:")))


# 校验一份前置条件声明（来自 index、旁注或记录）：必须是字符串列表，每项是 docker 或 command:<命令>；不合法抛 ValueError。
def parse_needs(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not all(_valid_need(item) for item in cast(list[object], value)):
        raise ValueError(f"{where} 的前置条件声明无效（必须是 docker 或 command:<命令> 组成的列表）: {value!r}")
    return cast(list[str], value)


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
        needs = cast(dict[str, object], row).get("needs") if isinstance(row, dict) else None
        if (isinstance(row, dict) and isinstance(cast(dict[str, object], row).get("reason"), str)
                and isinstance(needs, list) and all(_valid_need(item) for item in cast(list[object], needs))):
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
# 包声明的前置条件（安装时随包记下：index 或首装旁注的 needs），不依赖联网；启用检查以它为准，原因记录只用于展示。
def load_declared(workspace: Path) -> tuple[dict[str, list[str]], str | None]:
    path = workspace / _DECLARED
    if not path.exists():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("顶层必须是对象")
        return {key: parse_needs(item, key) for key, item in cast(dict[str, object], value).items()}, None
    except (OSError, ValueError) as error:
        return {}, f"{path} 无法读取或格式无效（{error}）"


def record_declared(workspace: Path, plugin_id: str, needs: list[str]) -> None:
    path = workspace / _DECLARED
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        declared, problem = load_declared(workspace)
        if problem is not None:
            raise ValueError(f"前置条件声明记录损坏，未改动：{problem}；修复或删除该文件后重试")
        if needs:
            declared[plugin_id] = needs
        elif declared.pop(plugin_id, None) is None:
            return
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                _ = stream.write(json.dumps(declared, ensure_ascii=False, indent=1, sort_keys=True))
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)


# 启用前的统一检查：以包声明的 needs 为准（没有声明时退回原因记录里的 needs，都没有就放行）；
# 仍不满足就拒绝并说明，满足了清掉展示用的原因记录；任一记录文件损坏都拒绝（无法确认条件）。
def check_before_enable(workspace: Path, plugin_id: str) -> None:
    declared, declared_problem = load_declared(workspace)
    rows, problem = load_disabled(workspace)
    if declared_problem is not None or problem is not None:
        raise ValueError(f"{plugin_id} 不能启用：前置条件记录损坏，无法确认条件是否满足（{declared_problem or problem}）；修复或删除该文件后重试")
    needs = declared.get(plugin_id)
    if needs is None and plugin_id in rows:
        needs = cast(list[str], rows[plugin_id]["needs"])
    if not needs:
        return
    still = unmet(needs)
    if still:
        raise ValueError(f"{plugin_id} 不能启用：{describe(still)}；补齐后再启用")
    clear_disabled(workspace, plugin_id)


# 启用声明了镜像的插件前，确认镜像在 docker 里。三种执行端，按可用性选：
#   1. Core 容器里：问 Workload Controller 的 image_ensure（本地有就成功，没有就拉取，控制器侧 600 秒上限）；
#   2. 没有控制器、本机有 docker 命令：docker image inspect / pull，进程级超时；
#   3. 都没有，或控制器是不支持该接口的旧版本：无法确认，明确拒绝并给出要在宿主执行的 docker pull 命令。
# run 可注入（演练与验证用，只影响第 2 种）。
INSPECT_TIMEOUT_S = 30.0
PULL_TIMEOUT_S = 600.0
CONTROLLER_TIMEOUT_S = 660.0


class _ControllerUnsupported(Exception):
    pass


# 与 Workload Controller 的一行 JSON 协议（version=1）；失败回包里 error 是文字，旧版控制器对未知 action 报"action 无效"。
def _controller_ensure(socket_path: Path, image: str, timeout: float = CONTROLLER_TIMEOUT_S) -> None:
    request = json.dumps({"version": 1, "action": "image_ensure", "body": {"image": image}}, separators=(",", ":")).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        stream.connect(str(socket_path))
        stream.sendall(request)
        raw = b""
        while not raw.endswith(b"\n") and len(raw) < 1_048_576:
            chunk = stream.recv(65536)
            if not chunk:
                break
            raw += chunk
    if not raw.strip():
        raise ValueError("Workload Controller 在处理期间退出或断开；稍后再试启用")
    reply = json.loads(raw)
    if isinstance(reply, dict) and reply.get("ok") is True:
        return
    error = str(reply.get("error", "")) if isinstance(reply, dict) else ""
    if "action 无效" in error:
        raise _ControllerUnsupported(error)
    if "正在拉取" in error:
        raise ValueError(f"宿主上的控制器{error}（同一时间只拉一个镜像）；稍后再试启用")
    raise ValueError(f"镜像 {image} 在宿主上准备失败：{error or '控制器没有返回原因'}")


def _host_commands(images: tuple[str, ...]) -> str:
    return "；".join(f"docker pull {image}" for image in images)


def ensure_images(images: Iterable[str], run: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run) -> list[str]:
    pulled: list[str] = []
    images = tuple(images)
    controller = os.environ.get("AKASHIC_WORKLOAD_SOCKET", "").strip()
    if images and controller:
        try:
            for image in images:
                _controller_ensure(Path(controller), image)
            return pulled
        except _ControllerUnsupported:
            if shutil.which("docker") is None:
                raise ValueError("宿主上的 Workload Controller 版本不支持确保镜像；请先在宿主上拉取后再启用：" + _host_commands(images)) from None
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError(f"无法联系 Workload Controller 确认镜像（{error}）；请在宿主上确认：" + _host_commands(images)) from error
    if images and shutil.which("docker") is None:
        raise ValueError("当前环境没有 docker 命令，也没有可用的 Workload Controller，无法确认镜像已在宿主上；请先在宿主上拉取后再启用：" + _host_commands(images))
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
