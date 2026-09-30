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
from agent.plugins.manifest import load_plugin_manifest
from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugins.selection import PluginSelection, SelectionConflictError
from agent.plugins.static_manifest import load_static_plugin_manifest
from agent.plugin_composition.archive import encode_tree, tree_entries
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


def is_distribution_input(record: Mapping[str, object], code: Path) -> bool:
    """An optional v4 source attribute is checked against the archived provenance."""
    if "distribution_source" not in record:
        return False
    commit = record["distribution_source"]
    plugin_id = record.get("plugin_id")
    if (record.get("source_type") != "builtin" or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(plugin_id, str) or plugin_id.count("@") != 1):
        raise ValueError("invalid distribution source descriptor")
    provenance = json.loads((code / ".akashic-source.json").read_text())
    if provenance.get("commit") != commit:
        raise ValueError("distribution descriptor/archive provenance mismatch")
    return True


def distribution_sources(
    workspace: Path, plugins_home: Path, distribution: Path | None = None,
) -> DistributionSources:
    """The immutable first receipt proves old cache ownership, never current choice."""
    if distribution is None:
        configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
        if not configured:
            return DistributionSources()
        distribution = Path(configured)
    distribution = distribution.resolve(strict=True)
    profile = json.loads((distribution / "profiles/default.json").read_text())
    marketplace = profile["marketplace"]
    defaults = {row["name"] for row in profile["plugins"]}
    choices = load_plugin_manifest(plugins_home)
    receipt_path = workspace / "runtime/distribution-install.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    ignored: set[Path] = set()
    historical: set[str] = set()
    legacy: set[str] = set()
    for row in receipt.get("installed", []):
        plugin_id = f'{row["name"]}@{row["marketplace"]}'
        historical.add(plugin_id)
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
    sources = tuple(source for source in distribution_plugin_sources(distribution)
                    if (f"{source.plugin_name}@{marketplace}" in choices
                        or (source.plugin_name in defaults
                            and f"{source.plugin_name}@{marketplace}" not in historical)))
    return DistributionSources(sources, frozenset(ignored), frozenset(legacy))


def distribution_plugin_sources(distribution: Path) -> tuple[ResolvedPluginSource, ...]:
    """读取发行版全部内置来源；迁移范围不受启停选择影响。"""
    report = json.loads((distribution / "distribution.json").read_text())
    profile = json.loads((distribution / "profiles/default.json").read_text())
    marketplace = profile["marketplace"]
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
) -> tuple[ResolvedPluginSource, ...]:
    """只把归属明确的内置数据目录交给当前发行版的 Yoyo。"""
    if distribution is None:
        configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
        if not configured:
            return ()
        distribution = Path(configured)
    available = distribution_sources(workspace, plugins_home, distribution)
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
    # 2. cache 丢失也不抹掉已选外置归档的身份，旧内置只按精确来源证据接管。
    selection = PluginSelection(workspace)
    root = selection.read() if selection.path.exists() else None
    if root is not None:
        for ref in cast(tuple[str, ...], selection.archive.read_descriptor(root)["components"]):
            record = selection.archive.read_descriptor(ref)
            plugin_id = cast(str, record["plugin_id"])
            if plugin_id not in by_id:
                continue
            if (is_distribution_input(record, selection.archive.open(cast(str, record["code"])))
                or (plugin_id in available.legacy_ids and record["code"] == legacy_codes.get(plugin_id))):
                continue
            raise SelectionConflictError(f"已选外置输入占用内置数据身份，停止迁移: {plugin_id}")
    return sources
