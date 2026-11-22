"""宿主侧（停机之前、有网络）产出 extra 插件的部署输入：求包、取代码成 bundle、校验、预拉镜像。

只依赖标准库：宿主 bootstrap 不加载 Agent 依赖。权威的静态 manifest 校验在目标镜像的预检里再做一遍。
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, Iterator, cast

from utils.timing import measure
from agent.plugins.absorbed import read_absorbed
from agent.plugins.resources import stage_archives
from agent.plugins.prerequisites import parse_needs, unmet
from agent.plugins.package_index import load_index
from agent.plugins.extra_repo import (
    ExtraAction,
    SelectedPackage,
    classify_gaps,
    git_env,
    list_release_heads,
    resolve_repo,
    upgrade_actions,
)
from agent.plugins.python_environment import PythonEnvironments, environment_repair_reason
from scripts.akashic_release.wheels import (
    Resolver, target_python, docker_resolver, installed_pins, local_resolver, resolve_package_wheels, tree_sha256,
)

Run = Callable[..., subprocess.CompletedProcess[str]]
MARKETPLACE = "release"
_API_VERSION = 3
_SIDECAR = "extra-{name}.json"
_BUNDLE = "extra-{name}.bundle"


@dataclass(frozen=True)
class ExtraOptions:
    enabled: bool = True
    repo: str | None = None
    inputs: Path | None = None
    pull_images: bool = True
    gaps_only: bool = False
    allow_retire: bool = False
    # 带第三方依赖的包在哪里求 wheel：显式的执行点优先；否则有目标镜像就在镜像里（docker），都没有就用本机 Core 解释器。
    image: str | None = None
    wheel_resolver: Resolver | None = None
    # 首次安装：把 index 声明的外部前置条件（needs）记进目标，供离线 ensure 按同一套策略装上但保持关闭；也不为缺前置条件的包预拉镜像。
    record_needs: bool = False
    # 首次安装选定要装的 extra 包（manifest 名）：并入应装集合，patch 里启用的非预设包也按首装处理而不是 optional。
    install_extra: tuple[str, ...] = ()
    # 首次安装的选择文件（bundle.patch.toml 内容）：workspace 还没初始化、patch 还不存在时，用它代替 workspace 里的 patch 计算应装集合。
    patch_file: Path | None = None
    # 首次安装的选择参数（akashic-release install 的 --profile/--groups/--yes）；不在 prepare_extra_inputs 里用，由首次激活分支读取。
    profile: str | None = None
    groups: tuple[str, ...] = ()
    assume_yes: bool = False
    reselect: bool = False


@dataclass(frozen=True)
class ExtraResult:
    targets: tuple[dict[str, object], ...]
    skipped: tuple[tuple[str, str], ...]
    # 在用的内置插件，发行版不再提供、上游也不存在，部署者已确认让它们消失（写进清单的 retire）。
    retiring: tuple[str, ...] = ()


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False, env=git_env())
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {result.stderr.strip()}")
    return result.stdout.strip()


# 旧选择里的包：来源由 artifact 的 origin 与 HEAD 推导，选择文件是普通 JSON。
def read_selected(
    workspace: Path, target_python: Callable[[], tuple[int, int]],
) -> tuple[str | None, dict[str, SelectedPackage]]:
    path = workspace / "runtime/plugin-stable.json"
    if not path.exists():
        return None, {}
    document = json.loads(path.read_text(encoding="utf-8"))
    selected: dict[str, SelectedPackage] = {}
    for record in document["inputs"].values():
        builtin = record["source_type"] == "builtin"
        code = Path(record["code"])
        origin = head = repair = ""
        if not builtin and (code / ".git").exists():
            origin = _git("config", "--get", "remote.origin.url", cwd=code)
            head = _git("rev-parse", "HEAD", cwd=code)
            repair = environment_repair_reason(code, PythonEnvironments(workspace), target_python) or ""
        selected[record["plugin_id"]] = SelectedPackage(record["plugin_id"], builtin, origin, head, repair)
    return document["root_ref"], selected


# 目标发行版自带的身份：发行版目录读 distribution.json；源码检出读 plugins/ 下各 plugin.py 声明的名字。
def provided_ids(source: Path) -> frozenset[str]:
    report = source / "distribution.json"
    if report.exists():
        document = json.loads(report.read_text(encoding="utf-8"))
        return frozenset(f"{row['name']}@{document['marketplace']}" for row in document["plugins"])
    # 身份取 plugin.py 里的 name，不能用目录名：目录 conversation_ui 的插件名是 conversation-ui。
    return frozenset(f"{read_literals(path)[0]}@{MARKETPLACE}" for path in (source / "plugins").glob("*/plugin.py"))


def _read_rows(path: Path) -> dict[str, dict[str, object]]:
    return dict(tomllib.loads(path.read_text(encoding="utf-8"))["rows"])


# preset 行（base 加 mode）叠加用户 patch，语义同 Core 的 load_bundles：patch 行先按 id、再按同插件找到 preset 行；
# 同一插件时按字段继承（只覆盖 disabled），不同插件（同 id 换 provider）则整行替换。
# 返回 (实际生效的启用身份, preset 里出现的身份)，与 pkg sync 口径一致。
def composition_ids(checkout: Path, mode: str, patch: Path) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    rows = _read_rows(checkout / "bundles/base.toml")
    if mode != "base":
        rows.update(_read_rows(checkout / "bundles" / f"{mode}.toml"))
    preset = frozenset(str(row["plugin"]) for row in rows.values())
    for identity, row in (_read_rows(patch) if patch.exists() else {}).items():
        key = identity if identity in rows else next((k for k, v in rows.items() if v["plugin"] == row["plugin"]), None)
        if key is None:
            rows[identity] = row
        elif rows[key]["plugin"] == row["plugin"]:
            rows[key] = {**rows[key], **{k: v for k, v in row.items() if k == "disabled"}}
        else:
            rows[key] = row
    # 被吸收的身份无论 patch 怎么写都不参与：不启用、不算已知，所以不会被安装、升级或补装。
    absorbed = read_absorbed(checkout)
    enabled = frozenset(str(r["plugin"]) for r in rows.values() if not r.get("disabled", False)) - absorbed.keys()
    return enabled, preset - absorbed.keys(), frozenset(str(r["plugin"]) for r in rows.values()) - absorbed.keys()


# 历史归属或首次回执里的旧制品：cache 里虽有目录，但属于被发行版接管的遗留，不算已装。
def _legacy_ids(workspace: Path, home: Path) -> frozenset[str]:
    ids: set[str] = set()
    selection = workspace / "runtime/plugin-stable.json"
    adoption = json.loads(selection.read_text(encoding="utf-8")).get("distribution_adoption") if selection.exists() else None
    ids.update(str(item["plugin_id"]) for item in (adoption or {}).get("entries", ()))
    receipt = workspace / "runtime/distribution-install.json"
    for row in (json.loads(receipt.read_text(encoding="utf-8")).get("installed", ()) if receipt.exists() else ()):
        plugin_id = f"{row['name']}@{row['marketplace']}"
        pointers = home / "cache" / row["marketplace"] / row["name"] / ".pointers.json"
        if pointers.exists():
            stable = json.loads(pointers.read_text(encoding="utf-8")).get("stable")
            if stable and str((pointers.parent / stable).resolve()) == str(Path(row["installed_path"]).resolve()):
                ids.add(plugin_id)
    return frozenset(ids)


# cache 里当前可用的已装制品（含尚未被选择采用、或已移出选择的），与 pkg sync 的"已装"同口径。
def cache_installed_ids(workspace: Path, home: Path) -> frozenset[str]:
    ignored = _legacy_ids(workspace, home)
    ids: set[str] = set()
    for pointers in sorted((home / "cache").glob("*/*/.pointers.json")):
        stable = json.loads(pointers.read_text(encoding="utf-8")).get("stable")
        if stable and (pointers.parent / stable).is_dir():
            ids.add(f"{pointers.parent.name}@{pointers.parent.parent.name}")
    return frozenset(ids - ignored)


# 只读 plugin.py 的顶层字面量：name、api_version、images。
def read_literals(plugin_py: Path) -> tuple[str, int, tuple[str, ...]]:
    name, api_version, images = "", -1, ()
    for statement in ast.parse(plugin_py.read_text(encoding="utf-8")).body:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            continue
        key = statement.targets[0].id
        if key == "name":
            name = str(ast.literal_eval(statement.value))
        elif key == "api_version":
            api_version = int(ast.literal_eval(statement.value))
        elif key == "images":
            images = tuple(str(item) for item in ast.literal_eval(statement.value))
    return name, api_version, images


# 取指定提交成 git bundle；release 分支无父提交，按 SHA 取到的就是全部历史。
# store 给出时从本地仓库（fetch_commits 一次取回的）取，产物与直接从上游取同一个提交逐字节相同。
def fetch_bundle(repo: str, name: str, commit: str, destination: Path, store: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="akashic-extra-") as directory:
        work = Path(directory)
        _git("init", "--quiet", cwd=work)
        _git("fetch", "--quiet", "--", repo if store is None else f"file://{store}", commit, cwd=work)
        ref = f"refs/heads/release/{name}"
        _git("update-ref", ref, commit, cwd=work)
        _git("bundle", "create", str(destination), ref, cwd=work)


# 本次需要的全部提交用一次 git fetch 取进 store（逐包取每次都是一次网络往返，全新安装时占掉大半时间）。
# 不退回逐个取：失败原因是确定的（提交不存在）就逐个取也不会成功，是网络问题就重试一次整批；
# 错误信息点名取不到的提交对应的包。
def fetch_commits(repo: str, wants: Mapping[str, str], store: Path) -> None:
    store.mkdir()
    _git("init", "--quiet", cwd=store)
    shas = list(dict.fromkeys(wants.values()))
    for attempt in (1, 2):
        try:
            _git("fetch", "--quiet", "--no-tags", "--", repo, *shas, cwd=store)
            return
        except RuntimeError as error:
            message = str(error)
            reason = message.split("失败: ", 1)[-1]
            missing = [f"{name}@{commit[:12]}" for name, commit in wants.items() if commit in reason]
            if attempt == 2 or any(hint in reason for hint in ("not our ref", "couldn't find remote ref", "no such remote ref")):
                culprit = "、".join(missing) if missing else "、".join(f"{name}@{commit[:12]}" for name, commit in wants.items())
                raise RuntimeError(f"一次取回 {len(shas)} 个提交失败（涉及: {culprit}）: {reason.strip().splitlines()[0]}") from error


# 把 bundle 检出到临时目录；调用方在目录里校验并求依赖 wheel。
@contextmanager
def _checkout(bundle: Path, commit: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="akashic-extra-check-") as directory:
        code = Path(directory) / "code"
        _git("clone", "--quiet", "--no-local", "--no-checkout", str(bundle), str(code))
        _git("checkout", "--quiet", "--detach", commit, cwd=code)
        yield code


# 校验代码：身份与 api_version；返回声明的镜像。依赖声明由 resolve_package_wheels 处理。
def check_code(code: Path, name: str) -> tuple[str, ...]:
    found, api_version, images = read_literals(code / "plugin.py")
    if found != name or api_version != _API_VERSION:
        raise RuntimeError(f"extra 包 {name} 身份或 api_version 不兼容: name={found} api_version={api_version}")
    return images


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 离线输入目录：每个包一份 bundle 与旁注 json，沿用此前联网部署留下的 .inputs 目录即可。
def _offline_heads(directory: Path) -> dict[str, str]:
    sidecars = (json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("extra-*.json"))
    return {str(item["plugin_id"]).partition("@")[0]: str(item["target_commit"]) for item in sidecars}


# 为一个动作产出 bundle 与清单条目：联网取代码或取离线输入，校验，预拉镜像。
def _produce(
    action: ExtraAction, repo: str, options: ExtraOptions, out: Path, run: Run, *, workspace: Path,
    pins: dict[str, tuple[str, str]], resolver: Resolver, store: Path | None = None, needs: tuple[str, ...] = (),
) -> dict[str, object]:
    name = action.plugin_id.partition("@")[0]
    bundle = out / _BUNDLE.format(name=name)
    sidecar: dict[str, object] = {}
    if options.inputs is not None:
        sidecar_path = options.inputs / _SIDECAR.format(name=name)
        if not sidecar_path.is_file():
            raise RuntimeError(f"离线输入目录里没有 {name} 的输入（extra-{name}.json）；该包需要处理，请补齐输入")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar["target_commit"] != action.commit:
            raise RuntimeError(f"离线输入里 {name} 的提交是 {str(sidecar['target_commit'])[:12]}，需要的是 {action.commit[:12]}")
        repo = str(sidecar["upstream"])
        source = options.inputs / _BUNDLE.format(name=name)
        if source.resolve() != bundle.resolve():
            shutil.copyfile(source, bundle)
    else:
        with measure("extra.fetch", plugin=name):
            fetch_bundle(repo, name, action.commit, bundle, store)
    wheels: dict[str, object] | None = None
    with _checkout(bundle, action.commit) as code:
        images = check_code(code, name)
        if options.inputs is not None and "offline_wheels" in sidecar:
            # 离线输入目录已带 wheel：原样复用，并核对树哈希。
            offline = cast(dict[str, str], sidecar["offline_wheels"])
            shutil.copytree(options.inputs / offline["relative_path"], out / offline["relative_path"], dirs_exist_ok=True)
            if tree_sha256(out / offline["relative_path"]) != offline["tree_sha256"]:
                raise RuntimeError(f"离线输入里 {name} 的 wheel 摘要与旁注不一致")
            carried = {key: sidecar[key] for key in ("inprocess_pins",) if key in sidecar}
            for key, version in cast(dict[str, str], carried.get("inprocess_pins", {})).items():
                pins[key] = (version, action.plugin_id)
            wheels = {"offline_wheels": offline, "core_lock_sha256": sidecar.get("core_lock_sha256"), **carried}
        else:
            with measure("extra.wheels", plugin=name):
                wheels = resolve_package_wheels(code=code, plugin_id=action.plugin_id, workspace=workspace, pins=pins,
                                                resolver=resolver, out=out)
        # 资源归档（resources.json）：与求 wheel 同属停机前宿主侧准备；离线输入目录里已有的原样复用。
        with measure("extra.resources", plugin=name):
            resources = stage_archives(code, out / f"extra-{name}-resources",
                                       reuse=None if options.inputs is None else options.inputs / f"extra-{name}-resources")
    needs = needs or tuple(parse_needs(sidecar.get("needs", []), name))
    # 前置条件在宿主侧（预取运行的地方）判定一次，结果写进目标（unmet：不满足的条件列表，空表示都满足）；
    # 容器里的离线安装只读取它，不再自己探测——容器既没有控制器通路也没有 docker socket。
    unmet_now: list[str] | None = (parse_needs(sidecar["unmet"], name) if "unmet" in sidecar
                                   else unmet(needs) if needs and options.record_needs else None)
    # 前置条件不满足的包照装但保持关闭，也就不为它拉镜像（没有 docker 的主机上首装不能因此失败）。
    for image in images if options.pull_images and not unmet_now else ():
        run(["docker", "pull", image], check=True, capture_output=True, text=True)
    target: dict[str, object] = {
        "plugin_id": action.plugin_id, "kind": action.kind, "bundle_relative_path": bundle.name,
        "bundle_sha256": _sha256(bundle), "target_commit": action.commit, "upstream": repo, "images": list(images),
        **(wheels or {}), **({"offline_resources": resources} if resources is not None else {}),
        **({"needs": list(needs)} if needs else {}), **({"unmet": unmet_now} if unmet_now is not None else {}),
    }
    (out / _SIDECAR.format(name=name)).write_text(json.dumps(target, sort_keys=True), encoding="utf-8")
    return target


# 首次安装时按 index 取各包的外部前置条件；取不到 index（或不需要）就是空，沿用旧行为。
def _needs_by_name(repo: str, workspace: Path, options: ExtraOptions) -> dict[str, tuple[str, ...]]:
    if not options.record_needs or options.inputs is not None:
        return {}
    try:
        index = load_index(repo, workspace.parent / "plugin-home")
    except RuntimeError:
        return {}
    return {name: package.needs for name, package in index.packages.items() if package.needs}


# 宿主侧入口：任何一步失败都抛错，调用方在停机之前整体失败。
def prepare_extra_inputs(
    *, workspace: Path, plugins_home: Path, source: Path, mode: str, root_ref: str | None, options: ExtraOptions,
    out: Path, run: Run,
) -> ExtraResult:
    # 1. 旧选择、目标发行版自带集合、preset 与 patch 的启用集合。
    current_root, selected = read_selected(workspace, target_python(options.image, run, options.wheel_resolver))
    if current_root != root_ref:
        raise RuntimeError("选择在读取期间改变，停止部署")
    patch_path = options.patch_file or workspace / "bundle.patch.toml"
    enabled, preset, known = composition_ids(source, mode, patch_path)
    patched = {str(row["plugin"]) for row in _read_rows(patch_path).values()} if patch_path.exists() else set()
    absorbed = read_absorbed(source)
    absorbed_notes = tuple((identity, f"已并入 Core 的 {provider}：移出选择，忽略用户 patch 里的行")
                           for identity, provider in sorted(absorbed.items()) if identity in selected or identity in patched)
    # 首装选定要装的包并入应装集合；已被 Core 吸收的身份（如 index 里仍列着的 session_reference）不能被装回来。
    wanted = frozenset(f"{name}@{MARKETPLACE}" for name in options.install_extra) - absorbed.keys()
    enabled, preset, known = enabled | wanted, preset | wanted, known | wanted
    repo = resolve_repo(options.repo)
    remote: dict[str, str] = {}

    installed = frozenset(pid for pid, package in selected.items() if not package.builtin) | cache_installed_ids(
        workspace, plugins_home)
    missing = sorted(enabled - provided_ids(source) - installed)
    in_use = frozenset(pid for pid, package in selected.items() if package.builtin)
    repairs = any(package.repair and not package.builtin for package in selected.values())
    # 启动补装只为 preset 要求的缺口、在用的内置插件和需要重建环境的已装包联网；只在 patch 里启用而没在用的行不是启动的前置条件。
    if options.gaps_only and not repairs and not set(missing) & (preset | in_use):
        return ExtraResult((), absorbed_notes)

    def heads() -> dict[str, str]:
        if not remote:
            try:
                with measure("extra.ls_remote"):
                    remote.update(_offline_heads(options.inputs) if options.inputs is not None else list_release_heads(repo))
            except (RuntimeError, OSError) as error:
                raise RuntimeError(f"{str(error).splitlines()[0]}；需要装的插件: {', '.join(missing)}") from error
        return remote

    # 2. 缺口分类与 pkg sync 共用；preset 缺失且上游也没有会在这里失败，换来源与首装由旧选择区分。
    gaps = classify_gaps(enabled=sorted(enabled), present=provided_ids(source) | installed, preset=preset,
                         heads=heads, needs_switch=lambda pid: pid.endswith("@" + MARKETPLACE), in_use=in_use)
    actions: list[ExtraAction] = []
    for gap in gaps:
        if gap.action in ("install", "switch"):
            if gap.upstream != "available":
                raise RuntimeError(f"preset 启用的插件目标发行版不再自带，extra 上游也没有: {gap.plugin_id}")
            kind = "switch" if gap.plugin_id in selected else "install"
            actions.append(ExtraAction(kind, gap.plugin_id, heads()[gap.plugin_id.partition("@")[0]]))
    # 已装包若在目标发行版的预设与用户 patch 里都没有行（例如能力并回了 Core、预设行被删），候选组合会把它移出选择，
    # 这里也不能再给它安排升级，否则会把它装回来、与 Core 里的同名能力重复。
    kept = {plugin_id: package for plugin_id, package in selected.items() if package.builtin or plugin_id in known}
    actions.extend(upgrade_actions(kept, heads, repo, gaps_only=options.gaps_only))
    retiring = tuple(gap.plugin_id for gap in gaps if gap.action == "retire")
    if retiring and not options.allow_retire:
        raise RuntimeError("以下在用的内置插件目标发行版不再提供、extra 上游也没有，部署后会消失: "
                           + ", ".join(retiring) + "；确认要移除请加 --allow-retire")
    notes = {"optional": "上游存在，未自动安装", "retired": "已退役，上游不存在", "retire": "将被移除（在用的内置插件，上游不存在）"}
    skipped = absorbed_notes + tuple((gap.plugin_id, notes[gap.action]) for gap in gaps if gap.action in notes)
    if not actions:
        return ExtraResult((), skipped, retiring)
    # 3. 逐个产出输入。
    out.mkdir(parents=True, exist_ok=True)
    # 取包仓库与 pip 缓存放在输出目录旁的同一个临时目录里：各包共用一次 git fetch 的结果与一份 pip 缓存，结束即删。
    # 不用系统临时目录：启动补装在服务容器里运行，那里的 /tmp 是 256MB 内存盘。
    with tempfile.TemporaryDirectory(prefix="akashic-extra-store-", dir=out.parent) as store_directory:
        store = Path(store_directory)
        cache = store / "pip-cache"
        cache.mkdir()
        resolver = options.wheel_resolver or (docker_resolver(options.image, run, cache) if options.image is not None
                                              else local_resolver(Path(__file__).resolve().parents[2], cache))
        # 本批要生效的包以新解析结果替换旧 pin：旧选择里它们的 pin 不参与约束；不在本批的已装包照旧。
        pins = installed_pins(workspace, frozenset(action.plugin_id.partition("@")[0] for action in actions))
        if options.inputs is None:
            with measure("extra.fetch_all", packages=len(actions)):
                fetch_commits(repo, {action.plugin_id.partition("@")[0]: action.commit for action in actions}, store / "git")
        needs = _needs_by_name(repo, workspace, options)
        targets = tuple(_produce(action, repo, options, out, run, workspace=workspace, pins=pins, resolver=resolver,
                                 store=None if options.inputs is not None else store / "git",
                                 needs=needs.get(action.plugin_id.partition("@")[0], ())) for action in actions)
    return ExtraResult(targets, skipped, retiring)


# 把额外输入目录并入部署者给的 inputs（硬链接优先），返回供容器只读挂载的目录。
def merge_inputs(operator_inputs: Path | None, extra: Path) -> Path:
    if operator_inputs is not None:
        shutil.copytree(operator_inputs, extra, dirs_exist_ok=True, copy_function=_link_or_copy)
    return extra


def _link_or_copy(source: str, destination: str) -> None:
    try:
        Path(destination).hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

