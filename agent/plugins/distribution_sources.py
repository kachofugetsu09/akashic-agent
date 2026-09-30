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

from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.manifest import load_plugin_manifest
from agent.plugins.source_resolver import ResolvedPluginSource
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
    report = json.loads((distribution / "distribution.json").read_text())
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
    sources = []
    for row in report["plugins"]:
        name = row["name"]
        plugin_id = f"{name}@{marketplace}"
        # Old opt-outs stay absent; new defaults get their first choice at deploy.
        if plugin_id not in choices and (name not in defaults or plugin_id in historical):
            continue
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
    return DistributionSources(tuple(sources), frozenset(ignored), frozenset(legacy))
