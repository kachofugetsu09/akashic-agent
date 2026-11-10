"""只改内置插件时的热更新路径（A 路径）。

与当前生效的发行版逐项比较：Core、依赖锁、镜像入口、preset 行和插件集合都不变，变化插件没有
新迁移、依赖文件不变、不是 Host Bridge 时，复用当前镜像，只把新发行版放进制品库，
再让运行中的实例换代这些插件。任何一项不满足都返回原因，由调用方走完整发布。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from scripts.akashic_release.doctor import read_environment
from scripts.akashic_release.manifest import read_json, write_json
from scripts.akashic_release.model import ReleasePaths
from scripts.plugin_store import materialize

# 这些文件决定镜像内容；任何一个变化都必须重建镜像。core.tar 另按成员内容比较。
_IMAGE_INPUTS = ("requirements.lock", "Dockerfile.distribution", "distribution-entrypoint.sh")
# 构建时附加的记录带源提交号，运行时不读取；插件仓库前进时 Core 提交号总会变。
_COMMIT_RECORDS = frozenset({"runtime-dependencies.json"})
# Bridge 在宿主上从 runtime checkout 运行，换它要重启 Bridge，属于完整发布。
_HOST_PLUGINS = frozenset({"host_execution"})
# 宿主外观与 SDK：它们变化时其它插件的样式与导入也要重新核对。
_SHELL_PLUGINS = frozenset({"ui", "akashic_clients"})
_DEPENDENCY_FILES = ("requirements.txt", "pyproject.toml", "migration.catalog.toml")


class GateRejected(RuntimeError):
    """目标发行版自身不兼容；完整发布也会带上同样的问题，所以直接拒绝。"""


@dataclass(frozen=True)
class HotPlan:
    active_distribution: Path
    changed: tuple[str, ...]
    gate_all: bool


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


# Core 身份：tar 成员的路径与内容（不含 mtime 和带提交号的记录），加上依赖文件摘要。
def _core_identity(distribution: Path) -> str:
    digest = hashlib.sha256()
    with tarfile.open(distribution / "core.tar") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if member.name in _COMMIT_RECORDS:
                continue
            digest.update(f"{member.name}\0{member.mode}\0{member.type!r}\0".encode())
            handle = archive.extractfile(member) if member.isfile() else None
            if handle is not None:
                digest.update(hashlib.sha256(handle.read()).digest())
    report = json.loads((distribution / "distribution.json").read_text())
    dependencies = {key: value for key, value in report["runtime_dependencies"].items()
                    if key not in {"source_commit", "source_tree"}}
    digest.update(json.dumps(dependencies, sort_keys=True).encode())
    return digest.hexdigest()


# 1. 镜像输入与 preset 行不变；2. 插件集合不变；3. 变化插件逐个排除需要重启的情形。
def classify(active: Path, target: Path) -> HotPlan | str:
    if _core_identity(active) != _core_identity(target):
        return "Core 变化"
    for name in _IMAGE_INPUTS:
        if _digest(active / name) != _digest(target / name):
            return f"镜像输入变化: {name}"
    if _tree_digest(active / "bundles") != _tree_digest(target / "bundles"):
        return "preset 行变化"
    before = {row["name"]: row for row in json.loads((active / "distribution.json").read_text())["plugins"]}
    after = {row["name"]: row for row in json.loads((target / "distribution.json").read_text())["plugins"]}
    if set(before) != set(after):
        return "插件集合变化"
    changed = sorted(name for name in after if after[name]["content_commit"] != before[name]["content_commit"])
    if not changed:
        return "没有插件变化"
    for name in changed:
        if name in _HOST_PLUGINS:
            return f"Host Bridge 插件变化: {name}"
        old, new = (active / "sources" / name).resolve(), target / "sources" / name
        for file in _DEPENDENCY_FILES:
            olds = sorted(p.relative_to(old) for p in old.rglob(file))
            news = sorted(p.relative_to(new) for p in new.rglob(file))
            if olds != news or any(_digest(old / p) != _digest(new / p) for p in news):
                return f"{name} 的 {file} 变化"
    return HotPlan(active, tuple(changed), gate_all=bool(_SHELL_PLUGINS.intersection(changed)))


# 依赖文件不变，所以 wheels 与摘要直接沿用当前发行版。
def _reuse_wheels(active: Path, target: Path) -> None:
    if (active / "wheels").is_dir():
        _ = shutil.copytree(active / "wheels", target / "wheels", symlinks=True, dirs_exist_ok=True)


def _gate(checkout: Path, image: str, distribution: Path, plan: HotPlan) -> None:
    command = ["docker", "run", "--rm", "--network=none", "--entrypoint", "/opt/venv/bin/python",
               "-v", f"{distribution}:{distribution}:ro",
               "-v", f"{checkout / 'scripts/distribution_compat_gate.py'}:/gate.py:ro",
               image, "/gate.py", "--distribution", str(distribution)]
    if not plan.gate_all:
        command += ["--plugins", ",".join(plan.changed)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise GateRejected("制品兼容门未通过:\n" + (result.stdout + result.stderr)[-4000:])


async def _adopt(environment: dict[str, str], root: Path) -> dict[str, object]:
    # 只走本机 unix 端点：不需要 token，也不必导入另一个版本的网关插件代码。
    from akashic_sdk import AsyncAkashic
    from agent.plugin_composition import load_endpoint_plan

    workspace = Path(environment["AKASHIC_WORKSPACE"])
    endpoints = [endpoint for endpoint in load_endpoint_plan(workspace / "runtime/endpoints.json")
                 if endpoint.protocol == "jsonrpc+unix"]
    if len(endpoints) != 1:
        raise ConnectionError(f"热更新需要唯一的本机控制端点，当前有 {len(endpoints)} 个")
    async with await AsyncAkashic.connect(endpoints[0].address) as client:
        result = await client.request("plugin/adopt-distribution", {"root": str(root)})
    if not isinstance(result, dict):
        raise RuntimeError("实例没有返回热更新结果")
    return {str(key): value for key, value in result.items()}


# 尝试热更新；返回 None 表示需要完整发布，原因写进 reason 列表。
def hot_install(*, paths: ReleasePaths, checkout: Path, commit: str, runtime_env: Path,
                reasons: list[str]) -> dict[str, object] | None:
    environment = read_environment(runtime_env)
    active = Path(environment.get("AKASHIC_PLUGIN_DISTRIBUTION", ""))
    if not active.is_relative_to(paths.plugin_store / "distributions") or not active.is_dir():
        reasons.append("当前实例还没有使用制品库")
        return None
    from scripts.build_plugin_distribution import build

    with tempfile.TemporaryDirectory(prefix="akashic-hot-", dir=paths.plugin_store) as directory:
        # 兼容门在镜像里以运行用户读取；发布器可能以 root 运行。
        Path(directory).chmod(0o755)
        target = Path(directory) / "distribution"
        _ = build(checkout, commit, target)
        plan = classify(active, target)
        if isinstance(plan, str):
            reasons.append(plan)
            return None
        _reuse_wheels(active, target)
        _gate(checkout, environment["AKASHIC_IMAGE"], target, plan)
        stored = materialize(target, paths.plugin_store, commit)
    try:
        result = asyncio.run(_adopt(environment, stored))
    except Exception as error:
        # 旧实例没有这个方法、实例未就绪等：实例要么没收到请求，要么已自行整体退回；完整发布会重新提交全部选择。
        reasons.append(f"实例不能热更新: {type(error).__name__}: {error}")
        return None
    if result.get("state") != "active":
        reasons.append(f"实例拒绝热更新: {json.dumps(result, ensure_ascii=False)[:2000]}")
        return None
    _record(paths, runtime_env, environment, stored, commit, plan)
    return {"status": "hot-activated", "commit": commit, "changed": list(plan.changed), "adopt": result}


# 重启后沿用新发行版；回执记下镜像版本不变、发行版换成目标提交。
def _record(paths: ReleasePaths, runtime_env: Path, environment: dict[str, str], stored: Path,
            commit: str, plan: HotPlan) -> None:
    lines = [line for line in runtime_env.read_text().splitlines()
             if not line.startswith("AKASHIC_PLUGIN_DISTRIBUTION=")]
    lines.append(f"AKASHIC_PLUGIN_DISTRIBUTION={stored}")
    runtime_env.write_text("\n".join(lines) + "\n")
    active_path = paths.activation / "active.json"
    active = read_json(active_path) if active_path.exists() else {}
    active["distributionCommit"] = commit
    active["hotUpdate"] = {"from": str(plan.active_distribution), "to": str(stored),
                           "changed": list(plan.changed), "image": environment["AKASHIC_IMAGE"]}
    write_json(active_path, active)
