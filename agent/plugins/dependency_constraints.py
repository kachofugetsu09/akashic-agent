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
_NO_DISTRIBUTION = re.compile(r"no matching distributions available for your environment:\n((?:[ \t]+\S+\n)+)")
CORE_ORIGIN = "Core 锁定版本"
_CORE_LOCK = Path("/opt/akashic/distribution/requirements.lock")


@dataclass(frozen=True)
class DependencyBatch:
    """同一次部署里将要生效的包：每个插件（按 name）的进程内解析结果 {包: 版本}。

    约束必须基于同一份批次事实：本批的包以新解析结果替换盘上旧选择里的旧版本，不在本批的已装包照旧。
    """

    pins: Mapping[str, Mapping[str, str]]


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


# 其他已装插件"即将生效"的进程内环境：cache 当前指针指向的 artifact 优先（离线 pkg 只推进指针、下次启动才进选择，
# 在线时两者一致），其次是选择记录里的旧引用；都不可用返回 None。
def _effective_site(
    plugin_id: str, selected_ref: str, environments: EnvironmentOpener, plugins_home: Path | None,
) -> Path | None:
    from agent.plugins.artifacts import read_pointers, resolve_pointer
    from agent.plugins.dependency_paths import site_packages
    from agent.plugins.python_environment import inprocess_ref

    refs: list[str] = []
    if plugins_home is not None:
        name, _, marketplace = plugin_id.partition("@")
        base = plugins_home / "cache" / marketplace / name
        try:
            pointers = read_pointers(base, validate_content=False)
            artifact = None if pointers is None else resolve_pointer(base, pointers.stable, validate_content=False)
            pointed = None if artifact is None else inprocess_ref(artifact)
        except (OSError, ValueError, RuntimeError):
            pointed = None
        if pointed is not None:
            refs.append(pointed)
    refs.append(selected_ref)
    for ref in dict.fromkeys(refs):
        try:
            return site_packages(environments.open(ref))
        except (OSError, ValueError, RuntimeError):
            continue
    return None


# 其他已安装插件的进程内环境版本，取自选择里随安装固定的环境引用；与 Core 重名的丢掉（Core 优先），
# 彼此重名时按 plugin_id 排序先出现者为准。读不到别人的环境时失败，不当作没有约束。
def _other_pins(
    workspace: Path, plugin_name: str, core: Mapping[str, str], environments: EnvironmentOpener,
    batch: DependencyBatch | None = None, plugins_home: Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    selection = PluginSelection(workspace)
    pins: dict[str, str] = {}
    origins: dict[str, str] = {}
    replaced = frozenset(batch.pins) if batch is not None else frozenset()
    if not selection.path.exists():
        return _with_batch(pins, origins, plugin_name, core, batch)
    try:
        reference = selection.read()
        records = [] if reference is None else [selection.read_input(ref) for ref in selection.components(reference)]
    except (OSError, ValueError, RuntimeError) as error:
        raise DependencyConstraintUnavailable(f"无法读取插件选择，不能确认依赖版本约束：{error}") from error
    for record in sorted(records, key=lambda item: str(item["plugin_id"])):
        plugin_id = str(record["plugin_id"])
        ref = cast(Mapping[str, str], record["python_environments"]).get(INPROCESS_KEY)
        owner = plugin_id.partition("@")[0]
        if record["source_type"] != "installed" or owner == plugin_name or owner in replaced or ref is None:
            continue
        site = _effective_site(plugin_id, ref, environments, plugins_home)
        if site is None:
            # 选择里和 cache 指针里的环境都不可用：这个已装插件现在无法加载，没有生效的版本可约束；它由
            # environment_repair_reason 标为需重建，重建时带着本批结果作为约束。进程里若仍挂着它的环境，
            # 由 DependencyPaths 保守要求重启，不靠这里。
            continue
        for name, version in installed_pins(site).items():
            if name not in core and name not in pins:
                pins[name], origins[name] = version, plugin_id
    return _with_batch(pins, origins, plugin_name, core, batch)


# 并入本批其他插件的新解析结果（本插件自己的不算）。
def _with_batch(
    pins: dict[str, str], origins: dict[str, str], plugin_name: str, core: Mapping[str, str], batch: DependencyBatch | None,
) -> tuple[dict[str, str], dict[str, str]]:
    for owner, owned in sorted((batch.pins if batch is not None else {}).items()):
        if owner == plugin_name:
            continue
        for name, version in owned.items():
            if name not in core and name not in pins:
                pins[name], origins[name] = version, owner
    return pins, origins


# 由 Core 锁与给定的"其他插件版本"写出约束文件：Core 锁文件去掉哈希后作为 -c（带哈希的文件会让 pip 进入哈希校验模式）。
# 缺 Core 锁时失败：不能把"没做检查"当成"检查通过"。
def constraints_from_pins(plugin_name: str, scratch: Path, others: Mapping[str, str], origins: Mapping[str, str]) -> Constraints:
    lock, expected = core_lock()
    if lock is None:
        raise DependencyConstraintUnavailable(
            "找不到 Core 锁文件，无法检查依赖版本冲突；期望路径：" + "、".join(str(path) for path in expected))
    core = _parse_pins(lock.read_text(encoding="utf-8"))
    others = {name: version for name, version in others.items() if name not in core}
    files: list[Path] = []
    for label, pins in (("core-lock", core), ("other-plugins", others)):
        if pins:
            target = scratch / f"{label}.txt"
            _ = target.write_text("".join(f"{name}=={version}\n" for name, version in sorted(pins.items())), encoding="utf-8")
            files.append(target)
    return Constraints(plugin_name, tuple(files), {**core, **others}, {**{name: CORE_ORIGIN for name in core}, **origins})


# 生成本次安装的约束：Core 锁 + 其他插件的进程内版本（选择里的已装插件；有 batch 时本批的包取新解析结果）。
def build_constraints(
    workspace: Path, plugin_name: str, environments: EnvironmentOpener, scratch: Path, batch: DependencyBatch | None = None,
    plugins_home: Path | None = None,
) -> Constraints:
    lock, _ = core_lock()
    core = _parse_pins(lock.read_text(encoding="utf-8")) if lock is not None else {}
    others, origins = _other_pins(workspace, plugin_name, core, environments, batch, plugins_home)
    return constraints_from_pins(plugin_name, scratch, others, origins)


# 已有环境是否仍满足约束：重名的包版本必须相同。
def environment_matches(site: Path, constraints: Constraints) -> bool:
    return all(constraints.pins.get(name, version) == version for name, version in installed_pins(site).items())


# 把 pip 的解析失败翻译成冲突说明：指出哪个包、被谁约束；不是冲突的失败返回 None。
def explain_failure(output: str, constraints: Constraints) -> DependencyConflict | None:
    if "ResolutionImpossible" not in output and "conflicting dependencies" not in output:
        return None
    # 有包没有可用 wheel（只有源码包）时 pip 也报成"冲突"，先点破真正原因：不现编，要在插件 wheels/ 里放 wheel。
    missing = _NO_DISTRIBUTION.search(output)
    if missing is not None:
        names = " ".join(missing.group(1).split())
        return DependencyConflict(
            f"依赖 {names} 没有可用的 wheel（只有源码包）：进程内依赖不现编源码包；"
            "请换用有 wheel 的版本，或在插件根的 wheels/ 目录里放一个 py3-none-any wheel")
    caused = _REQUESTED.findall(output)
    if not caused:
        return None
    name, version = caused[0]
    # 新旧 pip 的输出不同：没有"Additionally"段时，用户要求的恰好就是约束钉的同一个版本却仍然无解，只能是该版本没有 wheel。
    if re.search(rf"^\s*The user requested {re.escape(name)}=={re.escape(version)}\s*$", output, re.MULTILINE):
        return DependencyConflict(
            f"依赖 {name}=={version} 没有可用的 wheel（只有源码包或不适配当前 Python）：进程内依赖不现编源码包；"
            "请换用有 wheel 的版本，或在插件根的 wheels/ 目录里放一个 py3-none-any wheel")
    origin = constraints.origins.get(normalize(name), CORE_ORIGIN)
    detail = next((line.strip() for line in output.splitlines() if line.startswith("ERROR: Cannot install")), "")
    return DependencyConflict(
        f"依赖冲突：插件 {constraints.plugin_name} 的依赖与 {origin}要求的 {name}=={version} 不兼容"
        f"（{detail}）；请放宽该插件的依赖声明，或先升级 Core 后重试")


# 提交选择前整体核对一次：新选择里所有插件的进程内环境，同一个包不能有两个版本，也不能与 Core 锁不同。
def verify_selected_environments(
    entries: Mapping[str, Mapping[str, str]], core: Mapping[str, str] | None = None,
) -> None:
    lock, _ = core_lock()
    pins = core if core is not None else (_parse_pins(lock.read_text(encoding="utf-8")) if lock is not None else {})
    seen: dict[str, tuple[str, str]] = {name: (version, CORE_ORIGIN) for name, version in pins.items()}
    for plugin_id, owned in sorted(entries.items()):
        for name, version in owned.items():
            previous = seen.setdefault(name, (version, plugin_id))
            if previous[0] != version:
                raise DependencyConflict(
                    f"依赖冲突：{plugin_id} 的进程内依赖 {name}=={version} 与 {previous[1]} 的 {name}=={previous[0]} 不能同时存在于主进程")
