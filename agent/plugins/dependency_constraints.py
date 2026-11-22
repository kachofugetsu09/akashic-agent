"""已安装外置插件的依赖约束：主进程里同名包只能有一份，所以环境必须与 Core 和其他插件一致。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from agent.plugins.python_environment import INPROCESS_KEY
from agent.plugins.selection import PluginSelection

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\;]+)", re.MULTILINE)
_REQUESTED = re.compile(r"^\s*The user requested \(constraint\) ([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", re.MULTILINE)
CORE_ORIGIN = "Core 锁定版本"
_CORE_LOCK = Path("/opt/akashic/distribution/requirements.lock")


class EnvironmentOpener(Protocol):
    def open(self, ref: str) -> Path: ...


class DependencyConflict(ValueError):
    """插件依赖与 Core 锁定版本或其他已装插件的版本不兼容；安装在改动任何状态之前失败。"""


class DependencyConstraintUnavailable(ValueError):
    """无法取得做冲突检查所需的事实（缺 Core 锁、读不到其他插件的环境）；安装失败而不是当作没有约束。"""


@dataclass(frozen=True)
class Constraints:
    plugin_name: str
    files: tuple[Path, ...]
    pins: Mapping[str, str]
    origins: Mapping[str, str]

    # 约束内容的摘要：同一组约束下重建出的环境可作为复用候选。
    @property
    def digest(self) -> str:
        text = "".join(f"{name}=={version}\n" for name, version in sorted(self.pins.items()))
        return hashlib.sha256(text.encode()).hexdigest()[:16]


# 规范化包名：大小写与 - _ . 视为相同。
def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# Core 锁文件：显式环境变量只认它指向的文件；否则依次是镜像里的固定位置、开发 checkout。找不到返回 None 并带上期望路径。
def core_lock() -> tuple[Path | None, tuple[Path, ...]]:
    configured = os.environ.get("AKASHIC_CORE_LOCK", "").strip()
    candidates = (Path(configured),) if configured else (
        _CORE_LOCK, Path(__file__).resolve().parents[2] / "docker/host-runtime/requirements.lock")
    return next((path for path in candidates if path.is_file()), None), candidates


def _parse_pins(text: str) -> dict[str, str]:
    return {normalize(name): version for name, version in _PIN.findall(text)}


# 环境里已安装的 name==version，直接读 dist-info，不起 pip。
def installed_pins(site: Path) -> dict[str, str]:
    return {normalize(dist.metadata["Name"]): dist.version
            for dist in importlib.metadata.distributions(path=[str(site)])}


# 其他已安装插件的进程内环境版本，取自选择里随安装固定的环境引用；与 Core 重名的丢掉（Core 优先），
# 彼此重名时按 plugin_id 排序先出现者为准。读不到别人的环境时失败，不当作没有约束。
def _other_pins(workspace: Path, plugin_name: str, core: Mapping[str, str], environments: EnvironmentOpener) -> tuple[dict[str, str], dict[str, str]]:
    from agent.plugins.dependency_paths import site_packages

    selection = PluginSelection(workspace)
    pins: dict[str, str] = {}
    origins: dict[str, str] = {}
    if not selection.path.exists():
        return pins, origins
    try:
        reference = selection.read()
        records = [] if reference is None else [selection.read_input(ref) for ref in selection.components(reference)]
    except (OSError, ValueError, RuntimeError) as error:
        raise DependencyConstraintUnavailable(f"无法读取插件选择，不能确认依赖版本约束：{error}") from error
    for record in sorted(records, key=lambda item: str(item["plugin_id"])):
        plugin_id = str(record["plugin_id"])
        ref = cast(Mapping[str, str], record["python_environments"]).get(INPROCESS_KEY)
        if record["source_type"] != "installed" or plugin_id.partition("@")[0] == plugin_name or ref is None:
            continue
        try:
            site = site_packages(environments.open(ref))
        except (OSError, ValueError, RuntimeError) as error:
            raise DependencyConstraintUnavailable(
                f"无法读取已装插件 {plugin_id} 的依赖环境（{error}），不能确认版本约束；请先重新安装或卸载该插件") from error
        for name, version in installed_pins(site).items():
            if name not in core and name not in pins:
                pins[name], origins[name] = version, plugin_id
    return pins, origins


# 生成本次安装的约束：Core 锁文件去掉哈希后作为 -c（带哈希的文件会让 pip 进入哈希校验模式），
# 其他插件的版本写成另一个临时约束文件。缺 Core 锁时失败：不能把"没做检查"当成"检查通过"。
def build_constraints(workspace: Path, plugin_name: str, environments: EnvironmentOpener, scratch: Path) -> Constraints:
    lock, expected = core_lock()
    if lock is None:
        raise DependencyConstraintUnavailable(
            "找不到 Core 锁文件，无法检查依赖版本冲突；期望路径：" + "、".join(str(path) for path in expected))
    core = _parse_pins(lock.read_text(encoding="utf-8"))
    others, origins = _other_pins(workspace, plugin_name, core, environments)
    files: list[Path] = []
    for label, pins in (("core-lock", core), ("other-plugins", others)):
        if pins:
            target = scratch / f"{label}.txt"
            _ = target.write_text("".join(f"{name}=={version}\n" for name, version in sorted(pins.items())), encoding="utf-8")
            files.append(target)
    return Constraints(plugin_name, tuple(files), {**core, **others}, {**{name: CORE_ORIGIN for name in core}, **origins})


# 已有环境是否仍满足约束：重名的包版本必须相同。
def environment_matches(site: Path, constraints: Constraints) -> bool:
    return all(constraints.pins.get(name, version) == version for name, version in installed_pins(site).items())


# 把 pip 的解析失败翻译成冲突说明：指出哪个包、被谁约束；不是冲突的失败返回 None。
def explain_failure(output: str, constraints: Constraints) -> DependencyConflict | None:
    if "ResolutionImpossible" not in output and "conflicting dependencies" not in output:
        return None
    caused = _REQUESTED.findall(output)
    if not caused:
        return None
    name, version = caused[0]
    origin = constraints.origins.get(normalize(name), CORE_ORIGIN)
    detail = next((line.strip() for line in output.splitlines() if line.startswith("ERROR: Cannot install")), "")
    return DependencyConflict(
        f"依赖冲突：插件 {constraints.plugin_name} 的依赖与 {origin}要求的 {name}=={version} 不兼容"
        f"（{detail}）；请放宽该插件的依赖声明，或先升级 Core 后重试")
