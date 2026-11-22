"""插件的外部前置条件（docker、command:<命令>）：检测，以及"已装但未启用"的原因记录。只依赖标准库。

声明来自 extra 仓库 index 的 packages.<名>.needs。不满足的包照装但保持关闭，原因写进
<workspace>/runtime/disabled-reasons.json，供设置页、pkg list 与 Agent 查询；启用时重新检测。
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import cast

DOCKER_SOCKET = Path("/var/run/docker.sock")
_RECORD = "runtime/disabled-reasons.json"


# docker 可用 = 宿主 docker socket 存在且 GET /_ping 返回 OK（与 host_execution 控制器同一个 socket 路径）。
def _docker_available(timeout: float = 3.0) -> bool:
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


def read_disabled(workspace: Path) -> dict[str, dict[str, object]]:
    path = workspace / _RECORD
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return cast(dict[str, dict[str, object]], value) if isinstance(value, dict) else {}


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


# 记录某插件因前置条件未满足而保持关闭。
def record_disabled(workspace: Path, plugin_id: str, missing: list[str]) -> None:
    rows = read_disabled(workspace)
    rows[plugin_id] = {"needs": missing, "reason": describe(missing)}
    _write(workspace, rows)


def clear_disabled(workspace: Path, plugin_id: str) -> None:
    rows = read_disabled(workspace)
    if rows.pop(plugin_id, None) is not None:
        _write(workspace, rows)


# 启用前的统一检查：有记录且条件仍不满足就拒绝并说明；满足了清掉记录放行。
def check_before_enable(workspace: Path, plugin_id: str) -> None:
    record = read_disabled(workspace).get(plugin_id)
    if record is None:
        return
    still = unmet(cast(list[str], record.get("needs", [])))
    if still:
        raise ValueError(f"{plugin_id} 不能启用：{describe(still)}；补齐后再启用")
    clear_disabled(workspace, plugin_id)
