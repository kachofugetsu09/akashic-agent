from __future__ import annotations

import os
import shutil
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.store import ModelRegistryStore


__depends__ = {"20260802_01_yoyo_origin"}
__transactional__ = False


def migrate_model_registry(_connection: object) -> None:
    """Move mutable model configuration into the workspace model database."""

    current = current_migration_context()
    store = ModelRegistryStore.for_workspace(current.workspace)
    existing = store.read_snapshot()
    if existing is not None:
        return

    # 1. Capture every file that this migration may replace.
    backup_dir = (
        current.workspace
        / "backups"
        / "model-registry-v1"
        / uuid4().hex
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    auth_store = CredentialStore()
    originals = _backup_inputs(
        backup_dir=backup_dir,
        config_path=current.config_path,
        registry_path=store.path,
        auth_path=auth_store.path,
    )

    try:
        store.initialize()
        if not current.config_path.is_file():
            return
        document = tomlkit.parse(current.config_path.read_text(encoding="utf-8"))
        raw_llm = document.get("llm")
        if not isinstance(raw_llm, Mapping):
            return
        runtimes = raw_llm.get("runtimes")
        if not isinstance(runtimes, Mapping) or not runtimes:
            return

        # 2. Copy every referenced credential into its Provider connection row.
        credentials: dict[str, Credential] = {}
        legacy_metadata = auth_store.metadata()
        for runtime_id, raw_runtime in runtimes.items():
            if not isinstance(runtime_id, str) or not isinstance(raw_runtime, Mapping):
                raise ValueError("llm.runtimes 必须由 named table 组成")
            api_key = str(raw_runtime.get("api_key") or "").strip()
            auth_id = str(raw_runtime.get("auth") or "").strip() or (
                f"model_{runtime_id}"
            )
            if api_key:
                credentials[auth_id] = Credential(
                    driver="api_key",
                    access_token=api_key,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                raw_runtime["auth"] = auth_id
                raw_runtime.pop("api_key", None)
            elif auth_id in legacy_metadata:
                credentials[auth_id] = auth_store.get(auth_id)

        # 3. Publish the normalized database revision, then remove only its TOML source.
        _ = store.replace_from_llm_config(raw_llm, credentials=credentials)
        replacement_llm = tomlkit.table()
        migrated_keys = {"main", "fast", "agent", "vl", "runtimes"}
        for key, value in raw_llm.items():
            if key not in migrated_keys:
                replacement_llm.add(key, value)
        replacement_llm["registry"] = "workspace"
        document["llm"] = replacement_llm
        _atomic_write(current.config_path, tomlkit.dumps(document).encode("utf-8"))

        # 4. Re-read both owners so ledger success implies a bootable result.
        migrated = store.read_snapshot()
        if migrated is None or migrated.revision <= 0:
            raise RuntimeError("模型注册库迁移后没有可用 revision")
        store.integrity_check()
        tomllib.loads(current.config_path.read_text(encoding="utf-8"))
    except BaseException:
        _restore_inputs(
            originals=originals,
            config_path=current.config_path,
            registry_path=store.path,
            auth_path=auth_store.path,
        )
        raise


def _backup_inputs(
    *,
    backup_dir: Path,
    config_path: Path,
    registry_path: Path,
    auth_path: Path,
) -> dict[str, Path | None]:
    originals: dict[str, Path | None] = {}
    for name, path in (
        ("config", config_path),
        ("registry", registry_path),
        ("auth", auth_path),
    ):
        if not path.is_file():
            originals[name] = None
            continue
        target = backup_dir / (
            "registry.before.sqlite3" if name == "registry" else f"{name}.before"
        )
        if name == "registry":
            ModelRegistryStore(path).backup_to(target)
        else:
            shutil.copy2(path, target)
            os.chmod(target, 0o600)
        originals[name] = target
    return originals


def _restore_inputs(
    *,
    originals: Mapping[str, Path | None],
    config_path: Path,
    registry_path: Path,
    auth_path: Path,
) -> None:
    for name, path in (
        ("config", config_path),
        ("registry", registry_path),
        ("auth", auth_path),
    ):
        backup = originals[name]
        if backup is None:
            if name == "registry":
                _remove_sqlite_database(path)
            elif path.exists():
                path.unlink()
            continue
        if name == "registry":
            ModelRegistryStore(path).restore_from(backup)
        else:
            _atomic_write(path, backup.read_bytes())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.migration-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _remove_sqlite_database(path: Path) -> None:
    for candidate in (
        path,
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    ):
        if candidate.exists():
            candidate.unlink()


steps = [step(migrate_model_registry)]
