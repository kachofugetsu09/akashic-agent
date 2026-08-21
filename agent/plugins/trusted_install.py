"""在 Runtime 离线时安装 operator 已信任的 exact 插件源。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.install import install_git_plugin


@dataclass(frozen=True)
class TrustedPluginSource:
    source: str
    marketplace: str
    ref: str
    sparse: tuple[str, ...]


def load_trusted_plugin_batch(path: Path) -> tuple[TrustedPluginSource, ...]:
    """Validate one operator trust declaration and return exact plugin sources."""

    # 1. Validate the JSON envelope without accepting future unknown semantics.
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("trusted plugin batch 必须是 JSON object")
    payload = cast(dict[object, object], raw)
    if set(payload) != {"schema_version", "plugins"}:
        raise ValueError("trusted plugin batch 只接受 schema_version 和 plugins")
    if payload["schema_version"] != 1:
        raise ValueError("trusted plugin batch schema_version 必须是 1")
    plugins = payload["plugins"]
    if not isinstance(plugins, list) or not plugins:
        raise ValueError("trusted plugin batch plugins 必须是非空数组")

    # 2. Every entry names an immutable commit; branches cannot express trust.
    result: list[TrustedPluginSource] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for index, item in enumerate(cast(list[object], plugins)):
        if not isinstance(item, dict):
            raise ValueError(f"trusted plugin batch plugins[{index}] 必须是 object")
        entry = cast(dict[object, object], item)
        if not set(entry).issubset({"source", "marketplace", "ref", "sparse"}):
            raise ValueError(f"trusted plugin batch plugins[{index}] 含未知字段")
        if not {"source", "marketplace", "ref"}.issubset(entry):
            raise ValueError(f"trusted plugin batch plugins[{index}] 缺少必填字段")
        source = entry["source"]
        marketplace = entry["marketplace"]
        ref = entry["ref"]
        sparse_raw = entry.get("sparse", [])
        if (
            not isinstance(source, str)
            or not source
            or source != source.strip()
        ):
            raise ValueError(f"trusted plugin batch plugins[{index}].source 无效")
        if (
            not isinstance(marketplace, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", marketplace) is None
        ):
            raise ValueError(
                f"trusted plugin batch plugins[{index}].marketplace 无效"
            )
        if not isinstance(ref, str) or re.fullmatch(r"[0-9a-f]{40}", ref) is None:
            raise ValueError(
                f"trusted plugin batch plugins[{index}].ref 必须是完整 commit SHA"
            )
        if not isinstance(sparse_raw, list) or not all(
            isinstance(value, str) and value and value == value.strip()
            for value in sparse_raw
        ):
            raise ValueError(f"trusted plugin batch plugins[{index}].sparse 无效")
        sparse = tuple(cast(list[str], sparse_raw))
        identity = (source, marketplace, ref, sparse)
        if identity in seen:
            raise ValueError(f"trusted plugin batch plugins[{index}] 重复")
        seen.add(identity)
        result.append(TrustedPluginSource(source, marketplace, ref, sparse))
    return tuple(result)


def install_trusted_plugin_batch(
    *,
    workspace: Path,
    batch_path: Path,
    plugins_home: Path | None = None,
) -> dict[str, object]:
    """Install exact v3 sources directly as stable and report honest provenance."""

    # 1. Parse the complete operator trust declaration before changing cache state.
    sources = load_trusted_plugin_batch(batch_path)
    installed: list[dict[str, object]] = []

    # 2. Each plugin uses the existing atomic artifact installer without candidate staging.
    for index, source in enumerate(sources):
        try:
            result = install_git_plugin(
                workspace=workspace,
                source=source.source,
                marketplace=source.marketplace,
                ref_name=source.ref,
                sparse_paths=list(source.sparse),
                plugins_home=plugins_home,
                stage_candidate=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            completed = [str(item["pluginId"]) for item in installed]
            raise RuntimeError(
                "trusted plugin batch 安装失败: "
                f"index={index} completed={completed} error={exc}"
            ) from exc
        if result.source_revision != source.ref:
            raise RuntimeError(
                "trusted plugin batch 安装结果偏离 exact ref: "
                f"requested={source.ref} actual={result.source_revision}"
            )
        installed.append(
            {
                "pluginId": f"{result.plugin_name}@{result.marketplace}",
                "version": result.plugin_version,
                "sourceRevision": result.source_revision,
                "installedPath": str(result.installed_path),
                "dataPath": str(result.data_path),
            }
        )

    # 3. The receipt records trust, not a fabricated behavior-validation result.
    return {
        "mode": "operator_trusted_offline_batch",
        "programmaticValidation": "bypassed_by_operator_trust",
        "batchPath": str(batch_path.resolve()),
        "plugins": installed,
    }
