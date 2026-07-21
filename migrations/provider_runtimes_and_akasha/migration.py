from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import tomlkit

from agent.config import Config


_ROLE_NAMES = ("main", "fast", "agent", "vl")
_RUNTIME_ID_RE = re.compile(r"[^a-z0-9_]+")
_ROOT_ROLE_FIELDS = {
    "main": {
        "provider": "provider",
        "model": "model",
        "api_key": "api_key",
        "base_url": "base_url",
    },
    "fast": {
        "model": "light_model",
        "api_key": "light_api_key",
        "base_url": "light_base_url",
    },
    "agent": {
        "model": "agent_model",
        "api_key": "agent_api_key",
        "base_url": "agent_base_url",
    },
    "vl": {
        "model": "vl_model",
        "api_key": "vl_api_key",
        "base_url": "vl_base_url",
    },
}


@dataclass(frozen=True)
class MigrationContext:
    config_path: Path
    workspace: Path
    migration_commit: str
    backup_dir: Path | None


@dataclass(frozen=True)
class ConfigAssessment:
    state: Literal["absent", "current", "legacy", "blocked"]
    reason: str = ""


def _parse_args() -> tuple[str, MigrationContext]:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("assess", "apply", "verify", "revert"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--migration-commit", required=True)
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    return str(args.action), MigrationContext(
        config_path=Path(args.config).expanduser().resolve(),
        workspace=Path(args.workspace).expanduser().resolve(),
        migration_commit=str(args.migration_commit),
        backup_dir=Path(args.backup_dir).resolve() if args.backup_dir else None,
    )


def _config_assessment(config_path: Path) -> ConfigAssessment:
    if not config_path.exists():
        return ConfigAssessment("absent")
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    llm = document.get("llm")
    if not isinstance(llm, MutableMapping):
        if isinstance(document.get("provider"), str) and isinstance(
            document.get("model"), str
        ):
            return ConfigAssessment("legacy")
        return ConfigAssessment("blocked", "配置缺少可迁移的 LLM 字段")
    main = llm.get("main")
    runtimes = llm.get("runtimes")
    if isinstance(main, str):
        if not isinstance(runtimes, MutableMapping) or main not in runtimes:
            return ConfigAssessment("blocked", "llm.main 未指向有效 runtime")
        return ConfigAssessment("current")
    if not isinstance(main, MutableMapping) and not isinstance(
        document.get("model"), str
    ):
        return ConfigAssessment("blocked", "llm.main 既不是旧 table 也不是 runtime ID")
    if isinstance(runtimes, MutableMapping) and len(runtimes) > 0:
        return ConfigAssessment("blocked", "旧 llm.main 与 named runtimes 混杂")
    return ConfigAssessment("legacy")


def _runtime_id(provider: str, role: str, used: set[str]) -> str:
    normalized = _RUNTIME_ID_RE.sub("_", provider.strip().lower().replace("-", "_"))
    normalized = normalized.strip("_") or "openai"
    candidate = f"{normalized}_{role}"
    suffix = 2
    while candidate in used:
        candidate = f"{normalized}_{role}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _runtime_table(
    role: str,
    source: MutableMapping[str, Any],
    *,
    main_provider: str,
    main_runtime: MutableMapping[str, Any] | None,
    main_context_window: int,
) -> MutableMapping[str, Any]:
    runtime = cast(MutableMapping[str, Any], tomlkit.table())
    for key, value in source.items():
        runtime[key] = deepcopy(value)
    provider = str(runtime.get("provider") or (main_provider if role == "main" else "openai"))
    runtime["provider"] = provider

    # 1. 把旧上下文和多模态字段映射到 named runtime schema。
    if "context_window" not in runtime and "max_context_window" in runtime:
        runtime["context_window"] = runtime.pop("max_context_window")
    if int(runtime.get("context_window") or 0) <= 0:
        runtime["context_window"] = main_context_window
    if "input_modalities" not in runtime:
        multimodal = bool(runtime.pop("multimodal", role in {"main", "vl"}))
        runtime["input_modalities"] = ["text", "image"] if multimodal else ["text"]

    # 2. 旧角色会回退主模型连接信息；named runtime 需要把回退结果显式化。
    if role != "main" and main_runtime is not None:
        for key in ("api_key", "auth", "base_url"):
            if not runtime.get(key) and main_runtime.get(key):
                runtime[key] = deepcopy(main_runtime[key])
    return runtime


def _legacy_role_source(
    document: MutableMapping[str, Any],
    llm: MutableMapping[str, Any],
    role: str,
) -> MutableMapping[str, Any] | None:
    source = cast(MutableMapping[str, Any], tomlkit.table())
    nested = llm.get(role)
    if isinstance(nested, MutableMapping):
        for key, value in nested.items():
            source[key] = deepcopy(value)
    for target, root_key in _ROOT_ROLE_FIELDS[role].items():
        root_value = document.get(root_key)
        if target not in source and root_value is not None and root_value != "":
            source[target] = deepcopy(root_value)
    if not source.get("model"):
        return None
    return source


def _render_migrated_config(config_path: Path) -> str:
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    raw_llm = document.get("llm")
    if isinstance(raw_llm, MutableMapping):
        llm = cast(MutableMapping[str, Any], raw_llm)
    else:
        llm = cast(MutableMapping[str, Any], tomlkit.table())
        document["llm"] = llm
    main_source = _legacy_role_source(document, llm, "main")
    if main_source is None:
        raise RuntimeError("旧配置缺少 main model")
    main_provider = str(
        main_source.get("provider")
        or llm.get("provider")
        or document.get("provider")
        or "openai"
    )
    main_context_window = int(
        main_source.get("context_window")
        or main_source.get("max_context_window")
        or 128000
    )
    runtimes = cast(MutableMapping[str, Any], tomlkit.table())
    role_ids: dict[str, str] = {}
    used: set[str] = set()
    main_runtime: MutableMapping[str, Any] | None = None

    # 1. 逐个保留旧角色 table，并产生稳定的 named runtime 引用。
    for role in _ROLE_NAMES:
        source = _legacy_role_source(document, llm, role)
        if source is None:
            continue
        runtime = _runtime_table(
            role,
            source,
            main_provider=main_provider,
            main_runtime=main_runtime,
            main_context_window=main_context_window,
        )
        runtime_id = _runtime_id(str(runtime["provider"]), role, used)
        runtimes[runtime_id] = runtime
        role_ids[role] = runtime_id
        if role == "main":
            main_runtime = runtime

    # 2. 一次替换角色引用，移除只有旧 loader 消费的 provider 字段。
    llm["runtimes"] = runtimes
    for role, runtime_id in role_ids.items():
        llm[role] = runtime_id
    llm.pop("provider", None)
    for fields in _ROOT_ROLE_FIELDS.values():
        for root_key in fields.values():
            document.pop(root_key, None)
    return tomlkit.dumps(document)


def _atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _backup_file(source: Path, destination: Path) -> dict[str, str]:
    shutil.copy2(source, destination)
    os.chmod(destination, 0o600)
    return {
        "source": str(source),
        "backup": str(destination),
        "sha256": _sha256(destination),
    }


def _apply(context: MigrationContext) -> None:
    if context.backup_dir is None:
        raise RuntimeError("apply 缺少 --backup-dir")
    context.backup_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    records: list[dict[str, str]] = []
    config_state = _config_assessment(context.config_path)
    if config_state.state == "blocked":
        raise RuntimeError(config_state.reason)

    # 1. 先在临时配置上通过完整加载，再备份并原子替换旧配置。
    if config_state.state == "legacy":
        rendered = _render_migrated_config(context.config_path)
        candidate = context.config_path.with_name(
            f".{context.config_path.name}.migration-candidate-{uuid4().hex}.toml"
        )
        try:
            _atomic_write(candidate, rendered.encode("utf-8"))
            _ = Config.load(candidate, workspace=context.workspace)
            records.append(
                _backup_file(context.config_path, context.backup_dir / "config.toml")
            )
            _atomic_write(context.config_path, rendered.encode("utf-8"))
        finally:
            if candidate.exists():
                candidate.unlink()

    manifest = {
        "migrationCommit": context.migration_commit,
        "files": records,
    }
    _atomic_write(
        context.backup_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def _verify(context: MigrationContext) -> None:
    assessment = _config_assessment(context.config_path)
    if assessment.state in {"legacy", "blocked"}:
        raise RuntimeError(f"配置迁移验证失败: {assessment.reason or assessment.state}")
    if assessment.state == "current":
        _ = Config.load(context.config_path, workspace=context.workspace)


def _restore_file(backup: Path, target: Path) -> None:
    _atomic_write(target, backup.read_bytes())


def _revert(context: MigrationContext) -> None:
    if context.backup_dir is None or not context.backup_dir.is_dir():
        raise RuntimeError("revert 需要有效的 --backup-dir")
    config_backup = context.backup_dir / "config.toml"
    if config_backup.exists():
        _restore_file(config_backup, context.config_path)


def _assess(context: MigrationContext) -> dict[str, str]:
    assessment = _config_assessment(context.config_path)
    if assessment.state == "blocked":
        return {"status": "blocked", "reason": assessment.reason}
    if assessment.state == "legacy":
        return {"status": "needed"}
    return {"status": "satisfied"}


def main() -> None:
    action, context = _parse_args()
    if action == "assess":
        print(json.dumps(_assess(context), ensure_ascii=False))
        return
    if action == "apply":
        _apply(context)
        return
    if action == "verify":
        _verify(context)
        return
    _revert(context)


if __name__ == "__main__":
    main()
