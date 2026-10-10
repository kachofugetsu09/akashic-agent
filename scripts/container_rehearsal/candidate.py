"""Build isolated rehearsal configuration and plugin state."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any, cast

from scripts.container_rehearsal.model import CopyRecord, sha256
from scripts.container_rehearsal.policy import WEBUI_ONLY_SETTINGS


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"不支持的 TOML 值: {type(value).__name__}")


def _set_toml_value(content: str, section: str, key: str, value: Any) -> str:
    """Set one plain TOML table field without rewriting unrelated content."""

    lines = content.splitlines(keepends=True)
    section_pattern = re.compile(rf"^\s*\[{re.escape(section)}\]\s*(?:#.*)?(?:\r?\n)?$")
    heading_pattern = re.compile(r"^\s*\[\[?[^]]+\]\]?\s*(?:#.*)?(?:\r?\n)?$")
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    start = next((index for index, line in enumerate(lines) if section_pattern.match(line)), None)
    assignment = f"{key} = {_toml_literal(value)}\n"
    if start is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend((f"[{section}]\n", assignment))
        return "".join(lines)
    end = next(
        (index for index in range(start + 1, len(lines)) if heading_pattern.match(lines[index])),
        len(lines),
    )
    for index in range(start + 1, end):
        if key_pattern.match(lines[index]):
            lines[index] = assignment
            return "".join(lines)
    lines.insert(end, assignment)
    return "".join(lines)


def write_candidate_config(source: Path, destination: Path, workspace: Path) -> None:
    """Preserve model registry settings while disabling external delivery surfaces."""

    source_content = source.read_text(encoding="utf-8")
    source_document = tomllib.loads(source_content)
    candidate = _set_toml_value(source_content, "runtime", "workspace", str(workspace))
    for section, key, value in WEBUI_ONLY_SETTINGS:
        candidate = _set_toml_value(candidate, section, key, value)
    candidate_document = tomllib.loads(candidate)
    if candidate_document.get("llm") != source_document.get("llm"):
        raise RuntimeError("生成候选配置时意外改变了 llm 配置")
    for section, key, expected in WEBUI_ONLY_SETTINGS:
        table: Any = candidate_document
        for part in section.split("."):
            table = table[part]
        if table[key] != expected:
            raise RuntimeError(f"候选配置字段未生效: {section}.{key}")
    _ = destination.write_text(candidate, encoding="utf-8")
    destination.chmod(0o600)


def disable_plugin_choices(source_home: Path, workspace: Path) -> tuple[CopyRecord, list[str]]:
    """演练只改副本的 patch，不让外置制品在重装前恢复启用。"""
    from agent.plugins.bundles import set_plugin_choice
    from agent.plugins.manifest import installed_plugin_ids

    disabled = sorted(installed_plugin_ids(source_home))
    for plugin_id in disabled:
        set_plugin_choice(workspace, plugin_id, enabled=False)
    destination = workspace / "bundle.patch.toml"
    if not destination.exists():
        destination.write_text("schema_version = 1\n[rows]\n")
        destination.chmod(0o600)
    return (CopyRecord(path="workspace/bundle.patch.toml", kind="rehearsal_bundle_patch",
                       size=destination.stat().st_size, sha256=sha256(destination)), disabled)


def isolate_schedules(
    workspace: Path, records: list[CopyRecord]
) -> tuple[list[CopyRecord], int]:
    """Preserve copied schedules as evidence while disabling candidate execution."""

    schedules = workspace / "schedules.json"
    if not schedules.is_file():
        return records, 0

    # 1. Preserve the exact copied schedule payload.
    raw = schedules.read_text(encoding="utf-8")
    document = json.loads(raw)
    if not isinstance(document, list):
        raise ValueError("Workspace schedules.json 必须是 JSON array")
    source_copy = workspace / "schedules.source.json"
    _ = source_copy.write_text(raw, encoding="utf-8")
    source_copy.chmod(0o600)

    # 2. Publish an empty runtime schedule set for the rehearsal.
    _ = schedules.write_text("[]\n", encoding="utf-8")
    schedules.chmod(0o600)
    filtered = [record for record in records if record.path != "schedules.json"]
    filtered.extend(
        (
            CopyRecord(
                path="schedules.json",
                kind="rehearsal_disabled_schedules",
                size=schedules.stat().st_size,
                sha256=sha256(schedules),
            ),
            CopyRecord(
                path="schedules.source.json",
                kind="rehearsal_source_schedules",
                size=source_copy.stat().st_size,
                sha256=sha256(source_copy),
            ),
        )
    )
    return filtered, len(cast(list[object], document))
