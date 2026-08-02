from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, TypedDict

BENCHMARK_PREFIX = "akasic-bench-v4flash-"
BENCHMARK_NETWORK_POOL = ipaddress.IPv4Network("10.240.0.0/16")
BENCHMARK_NETWORK_PREFIX = 28
_IGNORED_TREE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


class IsolationError(RuntimeError):
    pass


class ProcessSnapshot(TypedDict):
    pid: int
    start_ticks: str
    cmdline: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_digests(
    root: Path,
    *,
    exclude: set[Path] | None = None,
) -> dict[str, str]:
    """散列证据目录中的普通文件，并显式排除自引用 manifest。"""

    excluded = {path.resolve() for path in (exclude or set())}
    values: dict[str, str] = {}
    if not root.exists():
        return values
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.resolve() not in excluded:
            values[path.relative_to(root).as_posix()] = sha256_file(path)
    return values


def create_source_bundle(
    source_root: Path,
    bundle_path: Path,
) -> dict[str, str]:
    """导出并校验可恢复当前源码历史的 Git 包。"""

    # 1. bundle 只承载 Git 对象与 refs；当前未提交内容仍由源码目录上传。
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(source_root), "bundle", "create", str(bundle_path), "--all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 2. 用 Git 自身校验 bundle 并记录当前源码身份
    subprocess.run(
        ["git", "-C", str(source_root), "bundle", "verify", str(bundle_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    return {
        "path": str(bundle_path),
        "digest": sha256_file(bundle_path),
        "head": head,
    }


def source_tree_digest(root: Path) -> str:
    """计算上传源码的稳定内容摘要。"""

    # 1. 排除版本控制与本地缓存，只散列会进入 runtime 的普通文件。
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in _IGNORED_TREE_PARTS for part in path.relative_to(root).parts)
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path).removeprefix("sha256:")))
    return f"sha256:{digest.hexdigest()}"


def compose_project_name(session_id: str) -> str:
    value = session_id.lower()
    if not re.match(r"^[a-z0-9]", value):
        value = "0" + value
    return re.sub(r"[^a-z0-9_-]", "-", value)


def reserve_compose_network(
    project_name: str,
    *,
    network_pool: ipaddress.IPv4Network = BENCHMARK_NETWORK_POOL,
    network_prefix: int = BENCHMARK_NETWORK_PREFIX,
) -> dict[str, str]:
    """为 retained benchmark project 原子预留一个小型独立网络。"""

    # 1. 只允许 benchmark owner 在合法 IPv4 池内分配子网。
    if not project_name.startswith(BENCHMARK_PREFIX):
        raise IsolationError(f"compose project 缺少 benchmark 前缀：{project_name}")
    if network_prefix <= network_pool.prefixlen or network_prefix > 30:
        raise IsolationError(
            f"benchmark network prefix 无效：{network_pool} /{network_prefix}"
        )

    # 2. Docker 负责原子判定重叠；hash 只分散起点，冲突时顺序探测。
    subnet_count = 1 << (network_prefix - network_pool.prefixlen)
    subnet_size = 1 << (32 - network_prefix)
    start = int.from_bytes(hashlib.sha256(project_name.encode()).digest()[:4]) % (
        subnet_count
    )
    network_name = f"{project_name}_default"
    for step in range(subnet_count):
        index = (start + step) % subnet_count
        candidate = ipaddress.IPv4Network(
            (int(network_pool.network_address) + index * subnet_size, network_prefix)
        )
        result = subprocess.run(
            [
                "docker",
                "network",
                "create",
                "--driver",
                "bridge",
                "--subnet",
                str(candidate),
                "--label",
                f"com.docker.compose.project={project_name}",
                "--label",
                "com.docker.compose.network=default",
                "--label",
                "akasic.benchmark.managed=true",
                network_name,
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            network_id = result.stdout.strip()
            if not network_id:
                raise IsolationError("docker network create 成功但未返回 network id")
            return {
                "id": network_id,
                "name": network_name,
                "subnet": str(candidate),
                "pool": str(network_pool),
            }
        error = "\n".join((result.stdout, result.stderr)).strip()
        if "overlaps with other one on this address space" in error:
            continue
        raise IsolationError(
            f"无法创建 benchmark network {network_name}：{error}"
        )
    raise IsolationError(f"benchmark network pool 已耗尽：{network_pool}")


def inspect_compose_project(project_name: str) -> list[dict[str, object]]:
    """读取一个 Harbor compose project 的安全容器投影。"""

    result = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    container_ids = [item for item in result.stdout.splitlines() if item]
    if not container_ids:
        raise IsolationError(f"未找到 compose project：{project_name}")
    inspected = subprocess.run(
        ["docker", "inspect", *container_ids],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload: object = json.loads(inspected.stdout)
    if not isinstance(payload, list):
        raise IsolationError("docker inspect 必须返回数组")
    projection: list[dict[str, object]] = []
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            raise IsolationError("docker inspect 数组元素必须是对象")
        item: dict[str, Any] = raw_item
        raw_mounts = item.get("Mounts")
        if not isinstance(raw_mounts, list):
            raise IsolationError("docker inspect Mounts 必须是数组")
        mounts: list[dict[str, object]] = []
        for raw_mount in raw_mounts:
            if not isinstance(raw_mount, dict):
                raise IsolationError("docker inspect Mounts 元素必须是对象")
            mount: dict[str, Any] = raw_mount
            mounts.append(
                {
                    "type": mount.get("Type"),
                    "name": mount.get("Name"),
                    "source": mount.get("Source"),
                    "destination": mount.get("Destination"),
                    "rw": bool(mount.get("RW")),
                }
            )
        projection.append(
            {
                "id": item.get("Id"),
                "name": str(item.get("Name") or "").lstrip("/"),
                "image": item.get("Config", {}).get("Image"),
                "image_id": item.get("Image"),
                "status": item.get("State", {}).get("Status"),
                "running": bool(item.get("State", {}).get("Running")),
                "exit_code": item.get("State", {}).get("ExitCode"),
                "oom_killed": bool(item.get("State", {}).get("OOMKilled")),
                "memory_limit_bytes": item.get("HostConfig", {}).get("Memory"),
                "mounts": mounts,
                "ports": item.get("HostConfig", {}).get("PortBindings") or {},
                "project": item.get("Config", {})
                .get("Labels", {})
                .get("com.docker.compose.project"),
            }
        )
    return projection


def validate_isolation(
    containers: list[dict[str, object]],
    *,
    project_name: str,
    allowed_bind_root: Path,
    forbidden_host_paths: list[Path],
    allowed_volume_mounts: list[tuple[str, str]],
) -> dict[str, object]:
    """拒绝非 trial bind、非 allowlist volume、Docker 控制面和端口。"""

    # 1. 所有容器必须属于唯一 benchmark project。
    if not project_name.startswith(BENCHMARK_PREFIX):
        raise IsolationError(f"compose project 缺少 benchmark 前缀：{project_name}")
    if not containers:
        raise IsolationError("compose project 没有容器")
    allowed_root = allowed_bind_root.resolve()
    forbidden = [path.resolve() for path in forbidden_host_paths]
    allowed_volumes = set(allowed_volume_mounts)
    if len(allowed_volumes) != len(allowed_volume_mounts):
        raise IsolationError("volume allowlist 含重复项")

    # 2. bind 只能来自本 trial；named volume 必须精确匹配且只读。
    checked_mounts = 0
    checked_volume_mounts = 0
    seen_volumes: set[tuple[str, str]] = set()
    for container in containers:
        if container.get("project") != project_name:
            raise IsolationError(f"容器 project 标签不一致：{container!r}")
        ports = container.get("ports")
        if ports:
            raise IsolationError(f"benchmark 容器禁止发布主机端口：{ports!r}")
        raw_mounts = container.get("mounts")
        if not isinstance(raw_mounts, list):
            raise IsolationError("容器投影 mounts 必须是数组")
        for raw_mount in raw_mounts:
            if not isinstance(raw_mount, dict):
                raise IsolationError("容器投影 mount 必须是对象")
            mount_type = raw_mount.get("type")
            if mount_type == "volume":
                checked_mount = (
                    str(raw_mount.get("name") or ""),
                    str(raw_mount.get("destination") or ""),
                )
                if bool(raw_mount.get("rw")):
                    raise IsolationError(
                        f"benchmark runtime volume 必须只读：{checked_mount!r}"
                    )
                if checked_mount not in allowed_volumes:
                    raise IsolationError(
                        f"volume mount 不在 allowlist：{checked_mount!r}"
                    )
                if checked_mount in seen_volumes:
                    raise IsolationError(
                        f"volume mount 重复：{checked_mount!r}"
                    )
                seen_volumes.add(checked_mount)
                checked_volume_mounts += 1
                continue
            if mount_type != "bind":
                raise IsolationError(f"benchmark 禁止 mount 类型：{mount_type!r}")
            checked_mounts += 1
            source = Path(str(raw_mount.get("source") or "")).resolve()
            destination = str(raw_mount.get("destination") or "")
            if destination == "/var/run/docker.sock" or source == Path(
                "/var/run/docker.sock"
            ):
                raise IsolationError("benchmark 容器禁止挂载 Docker socket")
            if source != allowed_root and allowed_root not in source.parents:
                raise IsolationError(f"bind mount 超出 trial 目录：{source}")
            for path in forbidden:
                if source == path or path in source.parents or source in path.parents:
                    raise IsolationError(f"bind mount 触及受保护路径：{source}")
    if seen_volumes != allowed_volumes:
        missing = sorted(allowed_volumes - seen_volumes)
        raise IsolationError(f"缺少 allowlist volume mount：{missing!r}")

    return {
        "status": "passed",
        "project": project_name,
        "container_count": len(containers),
        "checked_bind_mounts": checked_mounts,
        "checked_volume_mounts": checked_volume_mounts,
        "allowed_volume_mounts": [
            {"source": source, "destination": destination, "rw": False}
            for source, destination in sorted(allowed_volumes)
        ],
        "containers": containers,
    }


def online_process_snapshot() -> list[ProcessSnapshot]:
    """记录正式 workspace owner 的只读进程身份。"""

    workspace = "/home/huashen/.akashic/workspace"
    rows: list[ProcessSnapshot] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            cmdline = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
            stat = (entry / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if workspace not in cmdline:
            continue
        rows.append(
            {
                "pid": int(entry.name),
                "start_ticks": stat.rsplit(")", 1)[1].split()[19],
                "cmdline": cmdline,
            }
        )
    return sorted(rows, key=lambda row: row["pid"])


def validate_online_processes_unchanged(
    before: list[ProcessSnapshot],
    after: list[ProcessSnapshot],
) -> dict[str, object]:
    before_by_pid = {row["pid"]: row for row in before}
    after_by_pid = {row["pid"]: row for row in after}
    missing = sorted(set(before_by_pid) - set(after_by_pid))
    changed = sorted(
        pid
        for pid in set(before_by_pid) & set(after_by_pid)
        if before_by_pid[pid] != after_by_pid[pid]
    )
    return {
        "status": "passed" if not missing and not changed else "failed",
        "missing_pids": missing,
        "changed_pids": changed,
        "before": before,
        "after": after,
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
