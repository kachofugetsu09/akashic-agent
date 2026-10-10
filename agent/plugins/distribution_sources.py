"""Read image-owned sources without changing installed artifacts or choices."""
from __future__ import annotations

import json
import os
import subprocess
import hashlib
import io
import tarfile
import tempfile
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.manifest import workspace_plugin_data_dir
from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugins.selection import PluginSelection, SelectionConflictError
from agent.plugins.static_manifest import load_static_plugin_manifest
from agent.plugins.files import encode_tree, tree_entries
from agent.plugins.python_environment import ENVIRONMENT_FILE


def _matches_receipt_tree(artifact: Path, revision: str) -> bool:
    """HEAD alone does not prove that the mutable legacy checkout is unchanged."""
    content = subprocess.check_output(["git", "-C", str(artifact), "archive", revision])
    with tempfile.TemporaryDirectory(prefix="akashic-receipt-source-") as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            archive.extractall(root, filter="data")
        exclude = frozenset({".venv", "node_modules", ENVIRONMENT_FILE})
        return hashlib.sha256(encode_tree(tree_entries(root, exclude=exclude))).digest() == hashlib.sha256(
            encode_tree(tree_entries(artifact, exclude=exclude))).digest()


@dataclass(frozen=True)
class DistributionSources:
    sources: tuple[ResolvedPluginSource, ...] = ()
    ignored_installed_roots: frozenset[Path] = frozenset()
    legacy_ids: frozenset[str] = frozenset()
    root: Path | None = None


def is_distribution_owned(record: Mapping[str, object]) -> bool:
    """发行版归属由已提交选择声明，不从下一镜像的相同路径推断。"""
    if "distribution_source" not in record:
        return False
    commit = record["distribution_source"]
    plugin_id = record.get("plugin_id")
    if (record.get("source_type") != "builtin" or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(plugin_id, str) or plugin_id.count("@") != 1):
        raise ValueError("invalid distribution source descriptor")
    return True


def is_distribution_input(record: Mapping[str, object], code: Path) -> bool:
    """运行与复用输入时仍须核对实际代码的发行版身份。"""
    if not is_distribution_owned(record):
        return False
    commit = record["distribution_source"]
    provenance = json.loads((code / ".akashic-source.json").read_text())
    if provenance.get("commit") != commit:
        raise ValueError("distribution selection/source provenance mismatch")
    return True


def read_distribution_adoption(workspace: Path) -> Mapping[str, object] | None:
    """只读取当前 Root 引用的历史归属凭证，孤立凭证文件不生效。"""
    selection = PluginSelection(workspace)
    root = selection.read() if selection.path.exists() else None
    if root is None:
        return None
    record = selection.adoption()
    if record is None:
        return None
    check_distribution_adoption_format(record)
    return record


def check_distribution_adoption_format(record: Mapping[str, object]) -> None:
    """在部署清单和选择文件边界校验转换凭证的完整结构。"""
    fields = {"schema_version", "base_root_ref", "plan_sha256", "distribution_source_commit", "entries"}
    if set(record) != fields or type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("历史归属凭证格式无效")
    for key, length in (("base_root_ref", 64), ("plan_sha256", 64), ("distribution_source_commit", 40)):
        value = record[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
            raise ValueError(f"历史归属凭证身份无效: {key}")
    entries = record["entries"]
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("历史归属转换必须列出精确输入")
    seen: set[str] = set()
    fields = {"plugin_id", "component_ref", "artifact_pointer", "source_revision", "code_sha256",
              "manifest_digest", "data_dir", "source_commit", "source_path", "evidence_sha256"}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != fields:
            raise ValueError("历史归属转换条目格式无效")
        plugin_id = entry["plugin_id"]
        if (not isinstance(plugin_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*", plugin_id) is None
            or plugin_id in seen):
            raise ValueError("历史归属转换插件身份重复或无效")
        seen.add(plugin_id)
        for key in fields - {"plugin_id", "artifact_pointer", "data_dir", "source_path"}:
            value = entry[key]
            length = 40 if key in {"source_revision", "source_commit"} else 64
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
                raise ValueError(f"历史归属转换摘要无效: {plugin_id}/{key}")
        pointer = entry["artifact_pointer"]
        if not isinstance(pointer, str) or re.fullmatch(r"\.artifacts/[A-Za-z0-9][A-Za-z0-9._-]*", pointer) is None:
            raise ValueError(f"历史归属 artifact pointer 无效: {plugin_id}")
        if not isinstance(entry["data_dir"], str):
            raise ValueError(f"历史归属 data_dir 无效: {plugin_id}")
        source_path = entry["source_path"]
        if (not isinstance(source_path, str) or not source_path.startswith("plugins/")
            or "\\" in source_path or "\x00" in source_path
            or any(part in {"", ".", ".."} for part in source_path.split("/"))):
            raise ValueError(f"历史归属 source_path 无效: {plugin_id}")


def _adopted_distribution_roots(
    workspace: Path, plugins_home: Path, record: Mapping[str, object], *, preparing: bool,
    replacing_distribution: bool = False,
) -> tuple[set[Path], set[str]]:
    """核对已批准的旧输入；任何 cache、选择或数据身份漂移都停止转换。"""
    check_distribution_adoption_format(record)
    selection = PluginSelection(workspace)
    current = selection.read()
    if preparing and current != record["base_root_ref"]:
        raise SelectionConflictError("历史归属转换 Root 基线已变化")
    assert current is not None
    components = selection.components(current)
    inputs = {ref: selection.read_input(ref) for ref in components}
    selected = {cast(str, item["plugin_id"]): item for item in inputs.values()}
    base_data = [item["data_dir"] for item in inputs.values()]
    ignored: set[Path] = set()
    historical: set[str] = set()
    # 1. 逐项绑定原选择与已核验 artifact；不按插件名或 provenance 单独推断归属。
    for row in cast(tuple[Mapping[str, str], ...], record["entries"]):
        plugin_id = row["plugin_id"]
        name, marketplace = plugin_id.split("@")
        data = workspace_plugin_data_dir(workspace, name, marketplace)
        if (row["data_dir"] != data.relative_to(workspace).as_posix()
            or data.resolve() != data or not data.is_dir()):
            raise SelectionConflictError(f"历史归属数据身份不符: {plugin_id}")
        if any(item["data_dir"] == row["data_dir"] and item["plugin_id"] != plugin_id
               for item in inputs.values()):
            raise SelectionConflictError(f"历史归属数据被其它插件占用: {plugin_id}")
        old = inputs.get(row["component_ref"])
        if preparing and (old is None or old["plugin_id"] != plugin_id
                          or old["source_type"] != "installed" or old["data_dir"] != row["data_dir"]
                          or base_data.count(row["data_dir"]) != 1):
            raise SelectionConflictError(f"历史归属当前选择不符: {plugin_id}")
        base = plugins_home / "cache" / marketplace / name
        if any(path.is_symlink() for path in (plugins_home / "cache", base.parent, base)):
            raise SelectionConflictError(f"历史归属 cache 路径包含链接: {plugin_id}")
        pointers = read_pointers(base)
        if (pointers is None or pointers.stable != pointers.latest
            or pointers.stable.path != row["artifact_pointer"]):
            raise SelectionConflictError(f"历史归属 cache pointer 已变化: {plugin_id}")
        artifact = resolve_pointer(base, pointers.stable)
        assert artifact is not None
        identity = load_static_plugin_manifest(artifact)
        revision = subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(artifact), "rev-parse", "HEAD"], text=True,
        ).strip()
        code = hashlib.sha256(encode_tree(tree_entries(
            artifact, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}),
        ))).hexdigest()
        provenance = json.loads((artifact / ".akashic-source.json").read_text())
        if (identity.name != name or identity.identity_digest != row["manifest_digest"]
            or revision != row["source_revision"] or code != row["code_sha256"]
            or provenance != {"commit": row["source_commit"], "path": row["source_path"]}):
            raise SelectionConflictError(f"历史归属 cache 内容已变化: {plugin_id}")
        # 2. 提交后只能是普通发行版输入或未加载；退役/停用也不释放原数据身份。
        active = selected.get(plugin_id)
        if active is not None:
            if preparing:
                valid_source = active == old
            elif replacing_distribution:
                valid_source = is_distribution_owned(active)
            else:
                valid_source = is_distribution_input(active, Path(cast(str, active["code"])).resolve(strict=True))
            if not valid_source or active["data_dir"] != row["data_dir"]:
                raise SelectionConflictError(f"历史归属已被其它选择占用: {plugin_id}")
        ignored.add(artifact)
        historical.add(plugin_id)
    return ignored, historical


def distribution_sources(
    workspace: Path, plugins_home: Path, distribution: Path | None = None,
    *, adoption: Mapping[str, object] | None = None, replacing_distribution: bool = False,
) -> DistributionSources:
    """The immutable first receipt proves old cache ownership, never current choice."""
    if distribution is None:
        configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
        if not configured:
            return DistributionSources()
        distribution = Path(configured)
    distribution = distribution.resolve(strict=True)
    receipt_path = workspace / "runtime/distribution-install.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    ignored: set[Path] = set()
    legacy: set[str] = set()
    committed_adoption = read_distribution_adoption(workspace)
    if adoption is not None and committed_adoption is not None:
        raise SelectionConflictError("历史归属已经转换，不能再次转换")
    proof = adoption if adoption is not None else committed_adoption
    if proof is not None:
        adopted_roots, adopted_ids = _adopted_distribution_roots(
            workspace, plugins_home, proof, preparing=adoption is not None,
            replacing_distribution=replacing_distribution,
        )
        ignored.update(adopted_roots)
        legacy.update(adopted_ids)
    for row in receipt.get("installed", []):
        plugin_id = f'{row["name"]}@{row["marketplace"]}'
        base = plugins_home / "cache" / row["marketplace"] / row["name"]
        pointers = read_pointers(base)
        if pointers is None or pointers.stable.path is None:
            continue
        if pointers.stable != pointers.latest:
            raise RuntimeError(f"unsettled installed pointers: {plugin_id}")
        artifact = resolve_pointer(base, pointers.stable)
        if artifact is None or artifact != Path(row["installed_path"]):
            continue
        provenance_path = artifact / ".akashic-source.json"
        if not provenance_path.is_file():
            continue
        provenance = json.loads(provenance_path.read_text())
        revision = subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(artifact), "rev-parse", "HEAD"], text=True,
        ).strip()
        if (revision == row["source_revision"]
            and provenance.get("commit") == receipt["distribution_source_commit"]
            and isinstance(provenance.get("path"), str)
            and provenance["path"].startswith("plugins/")
            and load_static_plugin_manifest(artifact).name == row["name"]
            and _matches_receipt_tree(artifact, revision)):
            ignored.add(artifact)
            legacy.add(plugin_id)
    sources = distribution_plugin_sources(distribution)
    return DistributionSources(sources, frozenset(ignored), frozenset(legacy), distribution)


def distribution_plugin_sources(distribution: Path) -> tuple[ResolvedPluginSource, ...]:
    """读取发行版全部内置来源；迁移范围不受启停选择影响。"""
    report = json.loads((distribution / "distribution.json").read_text())
    marketplace = report["marketplace"]
    sources = []
    for row in report["plugins"]:
        name = row["name"]
        root = distribution / "sources" / name
        identity = load_static_plugin_manifest(root)
        if identity.name != name:
            raise ValueError(f"distribution source identity mismatch: {name}")
        digest = distribution / "wheels" / f"{name}.sha256"
        sources.append(ResolvedPluginSource(
            plugin_root=root, source_type="builtin", marketplace=marketplace,
            plugin_name=name, static_manifest=identity,
            wheel_tree_sha256=digest.read_text().strip() if digest.exists() else "",
            distribution_source=report["source_commit"],
        ))
    return tuple(sources)


def distribution_migration_sources(
    workspace: Path, plugins_home: Path, distribution: Path | None = None,
    *, adoption: Mapping[str, object] | None = None, replacing_distribution: bool = False,
) -> tuple[ResolvedPluginSource, ...]:
    """只把归属明确的内置数据目录交给当前发行版的 Yoyo。"""
    if distribution is None:
        configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
        if not configured:
            return ()
        distribution = Path(configured)
    available = distribution_sources(workspace, plugins_home, distribution, adoption=adoption,
                                     replacing_distribution=replacing_distribution)
    sources = distribution_plugin_sources(distribution)
    by_id = {f"{source.plugin_name}@{source.marketplace}": source for source in sources}
    legacy_codes: dict[str, str] = {}
    # 1. 同 ID 外置安装与内置共用 data root，停用也不能证明其数据归内置。
    for plugin_id, source in by_id.items():
        base = plugins_home / "cache" / source.marketplace / source.plugin_name
        pointers = read_pointers(base)
        if pointers is None or pointers.stable.path is None:
            continue
        artifact = resolve_pointer(base, pointers.stable)
        if artifact not in available.ignored_installed_roots:
            raise SelectionConflictError(f"内置与外置共用数据身份，停止迁移: {plugin_id}; 替代插件须使用独立身份")
        assert artifact is not None
        legacy_codes[plugin_id] = hashlib.sha256(encode_tree(tree_entries(
            artifact, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}),
        ))).hexdigest()
    # 2. cache 丢失也不抹掉已选外置来源的身份，旧内置只按精确来源证据接管。
    selection = PluginSelection(workspace)
    root = selection.read() if selection.path.exists() else None
    if root is not None:
        for ref in selection.components(root):
            record = selection.read_input(ref)
            plugin_id = cast(str, record["plugin_id"])
            if plugin_id not in by_id:
                continue
            owned = (is_distribution_owned(record) if replacing_distribution else
                     is_distribution_input(record, Path(cast(str, record["code"])).resolve(strict=True)))
            if (owned
                or (plugin_id in available.legacy_ids and hashlib.sha256(encode_tree(tree_entries(Path(cast(str, record["code"])), exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE})))).hexdigest() == legacy_codes.get(plugin_id))):
                continue
            raise SelectionConflictError(f"已选外置输入占用内置数据身份，停止迁移: {plugin_id}")
    return sources
