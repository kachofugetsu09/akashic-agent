import hashlib
import json
import os
import sqlite3
import stat
import tomllib
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.errors import AuthenticationError
from agent.model_runtime.store import ModelRegistryStore
from agent.plugin_composition import (
    CapabilitySources,
    EmbeddingSpaceDescriptor,
    ModelCapabilities,
)

__depends__ = {"20260829_02_backfill_explicit_programmatic_effects"}
__transactional__ = False

_MIGRATION = "retire-core-model-config"
_INLINE_CONNECTION_ID = "source:legacy_memory_embedding"
_INLINE_MODEL_ID = "legacy_memory_embedding"


@dataclass(frozen=True, slots=True)
class _ConfigSnapshot:
    path: Path
    target: Path
    content: bytes
    mode: int
    symlink_target: str | None


@dataclass(frozen=True, slots=True)
class _EmbeddingHandoff:
    model_id: str
    model: str
    dimensions: int
    connection_id: str | None = None
    auth_id: str = ""
    base_url: str = ""
    credential: Credential | None = None


def _snapshot(path: Path) -> _ConfigSnapshot | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        link = os.readlink(path)
        target = path.resolve(strict=True)
        if not target.is_file():
            raise RuntimeError(f"模型配置软链接目标不是普通文件: {path}")
        return _ConfigSnapshot(
            path=path,
            target=target,
            content=target.read_bytes(),
            mode=stat.S_IMODE(target.stat().st_mode),
            symlink_target=link,
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"模型配置必须是普通文件或软链接: {path}")
    return _ConfigSnapshot(
        path=path,
        target=path,
        content=path.read_bytes(),
        mode=stat.S_IMODE(metadata.st_mode),
        symlink_target=None,
    )


def _embedding_dimensions(
    workspace: Path,
    *,
    model: str,
    configured: object,
) -> int:
    """Resolve one exact legacy space without network or model-name guesses."""

    if configured is not None:
        if (
            not isinstance(configured, int)
            or isinstance(configured, bool)
            or configured <= 0
        ):
            raise RuntimeError("memory.embedding.output_dimensionality 必须是正整数")
        expected = configured
    else:
        expected = 0

    sessions = workspace / "sessions.db"
    observed: set[int] = set()
    if sessions.is_file():
        with closing(
            sqlite3.connect(f"file:{sessions}?mode=ro", uri=True)
        ) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'message_embeddings'"
            ).fetchone()
            if table is not None:
                observed = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT dim FROM message_embeddings WHERE model = ?",
                        (model,),
                    )
                    if int(row[0]) > 0
                }
    if expected:
        if observed and observed != {expected}:
            raise RuntimeError(
                "memory.embedding 维度与既有 Session embedding 空间不一致"
            )
        return expected
    if len(observed) != 1:
        raise RuntimeError(
            "memory.embedding 缺少 output_dimensionality，且既有 Session "
            "embedding 不能确定唯一维度"
        )
    return observed.pop()


def _embedding_handoff(
    memory: object,
    *,
    workspace: Path,
    store: ModelRegistryStore,
) -> _EmbeddingHandoff | None:
    """Translate only the two legacy embedding forms accepted by old Core."""

    if not isinstance(memory, dict) or set(memory) - {"enabled", "embedding"}:
        raise RuntimeError("[memory] 尚未完成 Akasha/models handoff，拒绝删除")
    enabled = memory.get("enabled")
    if not isinstance(enabled, bool):
        raise RuntimeError("memory.enabled 尚未完成校验，拒绝删除")
    embedding = memory.get("embedding")
    if embedding is None or embedding == {}:
        if enabled:
            raise RuntimeError("memory 已启用但没有可迁移的 embedding")
        return None
    if not isinstance(embedding, dict):
        raise RuntimeError("memory.embedding 尚未完成校验，拒绝删除")

    model_ref = str(embedding.get("model_ref") or "").strip()
    if model_ref:
        stored = store.get_embedding_model(model_ref)
        if stored is None:
            raise RuntimeError(f"memory.embedding.model_ref 不存在: {model_ref}")
        dimensions = _embedding_dimensions(
            workspace,
            model=stored.model,
            configured=stored.dimensions,
        )
        return _EmbeddingHandoff(model_ref, stored.model, dimensions)

    model = str(embedding.get("model") or "").strip()
    base_url = str(embedding.get("base_url") or "").strip()
    if not model or not base_url:
        raise RuntimeError("memory.embedding 缺少 model 或 base_url")
    existing = store.get_embedding_model(_INLINE_MODEL_ID)
    configured_dimensions = embedding.get("output_dimensionality")
    if configured_dimensions is None and existing is not None:
        if existing.model != model:
            raise RuntimeError("legacy memory embedding model ID 已被占用")
        configured_dimensions = existing.dimensions
    dimensions = _embedding_dimensions(
        workspace,
        model=model,
        configured=configured_dimensions,
    )
    auth_id = str(embedding.get("auth") or _INLINE_MODEL_ID).strip()
    inline_key = str(embedding.get("api_key") or "").strip()
    if inline_key:
        credential = Credential(driver="api_key", access_token=inline_key)
    else:
        try:
            credential = CredentialStore.for_workspace(workspace).get(auth_id)
        except AuthenticationError as error:
            raise RuntimeError(
                f"memory.embedding 凭据不存在或无效: {auth_id}"
            ) from error
        if credential.driver != "api_key" or not credential.access_token:
            raise RuntimeError(f"memory.embedding 凭据不是有效 API key: {auth_id}")
    return _EmbeddingHandoff(
        model_id=_INLINE_MODEL_ID,
        model=model,
        dimensions=dimensions,
        connection_id=_INLINE_CONNECTION_ID,
        auth_id=auth_id,
        base_url=base_url,
        credential=credential,
    )


def _preflight(
    content: bytes,
    workspace: Path,
) -> tuple[dict[str, object], ModelRegistryStore, _EmbeddingHandoff | None]:
    parsed = tomllib.loads(content.decode("utf-8"))
    store = ModelRegistryStore.for_workspace(workspace)
    llm = parsed.get("llm")
    if llm is not None:
        if llm != {"registry": "workspace"}:
            raise RuntimeError("[llm] 尚未完成模型注册库 handoff，拒绝删除")
        if not store.exists():
            raise RuntimeError("[llm] 已指向 workspace，但模型注册库不存在")
        store.integrity_check()
        if store.read_snapshot() is None:
            raise RuntimeError("[llm] handoff 模型注册库为空，拒绝删除")
    memory = parsed.get("memory")
    handoff = (
        None
        if memory is None
        else _embedding_handoff(memory, workspace=workspace, store=store)
    )
    return parsed, store, handoff


def _has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    }


def _apply_embedding_handoff(
    store: ModelRegistryStore,
    handoff: _EmbeddingHandoff | None,
) -> str | None:
    """Upgrade the legacy registry and commit one selected embedding revision."""

    if not store.exists():
        if handoff is not None:
            raise RuntimeError("memory.embedding handoff 缺少模型注册库")
        return None
    auth_kind = ""
    auth_payload = ""
    if handoff is not None and handoff.credential is not None:
        auth_kind, auth_payload = CredentialStore.encode(handoff.credential)
    identity = None
    with closing(sqlite3.connect(store.path)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        changed = False
        additions = (
            (
                "model_connections",
                "driver_config_json",
                "ALTER TABLE model_connections ADD COLUMN "
                "driver_config_json TEXT NOT NULL DEFAULT '{}'",
            ),
            (
                "model_registry_meta",
                "default_embedding_model_id",
                "ALTER TABLE model_registry_meta ADD COLUMN "
                "default_embedding_model_id TEXT DEFAULT NULL",
            ),
            (
                "model_definitions",
                "capabilities_json",
                "ALTER TABLE model_definitions ADD COLUMN capabilities_json TEXT",
            ),
            (
                "embedding_models",
                "capabilities_json",
                "ALTER TABLE embedding_models ADD COLUMN capabilities_json TEXT",
            ),
        )
        for table, column, statement in additions:
            if not _has_column(connection, table, column):
                connection.execute(statement)
                changed = True
        normalized = connection.execute(
            "UPDATE model_connections SET provider = 'openai-compatible' "
            "WHERE provider IN ('openai', 'deepseek', 'qwen')"
        )
        changed = changed or normalized.rowcount > 0

        if handoff is not None and handoff.connection_id is not None:
            connection_row = connection.execute(
                "SELECT name, provider, catalog_provider_id, auth_id, base_url, "
                "auth_kind, auth_payload, enabled FROM model_connections WHERE id = ?",
                (handoff.connection_id,),
            ).fetchone()
            desired_connection = (
                "Memory Embedding",
                "openai-compatible",
                "openai",
                handoff.auth_id,
                handoff.base_url,
                auth_kind,
                auth_payload,
                1,
            )
            if connection_row is None:
                connection.execute(
                    "INSERT INTO model_connections("
                    "id, name, provider, catalog_provider_id, auth_id, base_url, "
                    "auth_kind, auth_payload, enabled"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (handoff.connection_id, *desired_connection[:-1]),
                )
                changed = True
            elif tuple(connection_row) != desired_connection:
                raise RuntimeError("legacy memory embedding connection ID 已被占用")

            model_row = connection.execute(
                "SELECT connection_id, model, dimensions, enabled "
                "FROM embedding_models WHERE id = ?",
                (handoff.model_id,),
            ).fetchone()
            desired_model = (
                handoff.connection_id,
                handoff.model,
                handoff.dimensions,
                1,
            )
            if model_row is None:
                connection.execute(
                    "INSERT INTO embedding_models("
                    "id, connection_id, model, dimensions, enabled"
                    ") VALUES (?, ?, ?, ?, 1)",
                    (handoff.model_id, *desired_model[:-1]),
                )
                changed = True
            elif tuple(model_row) != desired_model:
                raise RuntimeError("legacy memory embedding model ID 已被占用")

        if handoff is not None:
            current_default = connection.execute(
                "SELECT default_embedding_model_id FROM model_registry_meta "
                "WHERE singleton = 1"
            ).fetchone()
            if current_default is None:
                raise RuntimeError("模型注册库缺少 revision metadata")
            if current_default[0] != handoff.model_id:
                if current_default[0] not in (None, ""):
                    raise RuntimeError("模型注册库已有不同的默认 embedding")
                connection.execute(
                    "UPDATE model_registry_meta SET default_embedding_model_id = ? "
                    "WHERE singleton = 1",
                    (handoff.model_id,),
                )
                changed = True
        if changed:
            connection.execute(
                "UPDATE model_registry_meta SET revision = revision + 1 "
                "WHERE singleton = 1"
            )
        if handoff is not None:
            identity = _embedding_identity(connection, handoff)
        connection.commit()
    store.integrity_check()
    return identity


def _embedding_identity(
    connection: sqlite3.Connection,
    handoff: _EmbeddingHandoff,
) -> str:
    """Compute the stable public identity emitted by the ordinary plugin."""

    model_row = connection.execute(
        "SELECT id, connection_id, model, dimensions, capabilities_json "
        "FROM embedding_models WHERE id = ?",
        (handoff.model_id,),
    ).fetchone()
    if model_row is None:
        raise RuntimeError("默认 embedding handoff 不完整")
    connection_row = connection.execute(
        "SELECT id, provider, catalog_provider_id, auth_id, base_url, "
        "driver_config_json FROM model_connections WHERE id = ?",
        (model_row["connection_id"],),
    ).fetchone()
    if connection_row is None:
        raise RuntimeError("默认 embedding connection 不存在")
    driver_id = str(connection_row["provider"])
    if driver_id != "openai-compatible":
        raise RuntimeError(f"历史 embedding driver 不受支持: {driver_id}")

    connection_config = _json_object(
        connection_row["driver_config_json"],
        "embedding connection driver_config_json",
    )
    catalog_provider_id = str(connection_row["catalog_provider_id"] or "")
    if catalog_provider_id:
        existing = connection_config.setdefault(
            "catalog_provider_id", catalog_provider_id
        )
        if existing != catalog_provider_id:
            raise RuntimeError("embedding connection catalog provider 冲突")
    capabilities, sources, model_config = _embedding_capabilities(model_row)
    connection_fingerprint = _digest(
        {"endpoint": str(connection_row["base_url"]), "config": connection_config}
    )
    capability_digest = _digest(
        {
            "capabilities": asdict(capabilities),
            "sources": asdict(sources),
            "driver_config": model_config,
        }
    )
    return EmbeddingSpaceDescriptor(
        plugin_snapshot_id="migration",
        model_revision=0,
        model_id=str(model_row["id"]),
        connection_id=str(model_row["connection_id"]),
        driver_id=driver_id,
        driver_contract_version="1",
        auth_identity=str(connection_row["auth_id"]),
        connection_fingerprint=connection_fingerprint,
        model=str(model_row["model"]),
        dimensions=int(model_row["dimensions"]),
        normalization=capabilities.embedding_normalization or "none",
        capability_digest=capability_digest,
    ).identity


def _embedding_capabilities(
    row: sqlite3.Row,
) -> tuple[ModelCapabilities, CapabilitySources, dict[str, object]]:
    raw = row["capabilities_json"]
    if raw is None or not str(raw):
        return (
            ModelCapabilities(
                input_modalities=("text",),
                supports_tool_calls=False,
                supports_parallel_tool_calls=False,
                embedding_dimensions=int(row["dimensions"]),
                embedding_normalization="none",
            ),
            CapabilitySources(),
            {},
        )
    payload = _json_object(raw, f"embedding model {row['id']} capabilities")
    capabilities = payload.get("capabilities")
    sources = payload.get("capability_sources")
    model_config = payload.get("driver_config", {})
    if not isinstance(capabilities, dict) or not isinstance(sources, dict):
        raise RuntimeError("embedding model capability fields 已损坏")
    if not isinstance(model_config, dict):
        raise RuntimeError("embedding model driver_config 已损坏")
    try:
        decoded = ModelCapabilities(**capabilities)
        decoded_sources = CapabilitySources(**sources)
    except TypeError as error:
        raise RuntimeError("embedding model capability fields 已损坏") from error
    if decoded.embedding_dimensions != int(row["dimensions"]):
        raise RuntimeError("embedding model dimensions 冲突")
    return decoded, decoded_sources, model_config


def _json_object(raw: object, field: str) -> dict[str, object]:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{field} 不是有效 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} 必须是 JSON object")
    return value


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _retag_session_embeddings(
    workspace: Path,
    *,
    old_identity: str,
    new_identity: str,
    dimensions: int,
) -> None:
    """Rename frozen Session vectors without changing their values."""

    sessions = workspace / "sessions.db"
    if sessions.is_file():
        with closing(sqlite3.connect(sessions)) as connection, connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(message_embeddings)")
            }
            if columns:
                if not {"model", "dim"}.issubset(columns):
                    raise RuntimeError("Session embedding schema 不兼容")
                invalid = connection.execute(
                    "SELECT DISTINCT dim FROM message_embeddings "
                    "WHERE model IN (?, ?) AND dim != ?",
                    (old_identity, new_identity, dimensions),
                ).fetchall()
                if invalid:
                    raise RuntimeError("Session embedding 空间维度不一致")
                connection.execute(
                    "UPDATE message_embeddings SET model = ? WHERE model = ?",
                    (new_identity, old_identity),
                )


def _backup_database(source: Path, destination: Path) -> None:
    with (
        closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as source_db,
        closing(sqlite3.connect(destination)) as destination_db,
    ):
        source_db.backup(destination_db)
    os.chmod(destination, 0o600)
    with closing(sqlite3.connect(f"file:{destination}?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError(f"SQLite backup integrity_check 失败: {destination}")


def _restore_database(source: Path, destination: Path) -> None:
    with (
        closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as source_db,
        closing(sqlite3.connect(destination)) as destination_db,
    ):
        source_db.backup(destination_db)
    os.chmod(destination, 0o600)


def _render(content: bytes) -> bytes | None:
    parsed = tomllib.loads(content.decode("utf-8"))
    retired = [name for name in ("llm", "memory") if name in parsed]
    if not retired:
        return None
    document = tomlkit.parse(content.decode("utf-8"))
    for name in retired:
        del document[name]
    rendered = tomlkit.dumps(document).encode("utf-8")
    final = tomllib.loads(rendered.decode("utf-8"))
    if any(name in final for name in retired):
        raise RuntimeError("Core 模型配置迁移后仍含 retired table")
    return rendered


def _write_atomic(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(candidate, path)
        os.chmod(path, mode)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        candidate.unlink(missing_ok=True)


def _check_identity(snapshot: _ConfigSnapshot) -> None:
    if snapshot.symlink_target is not None and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"模型配置软链接身份改变: {snapshot.path}")


def _restore(
    snapshot: _ConfigSnapshot,
    config_backup: Path,
    store: ModelRegistryStore,
    registry_backup: Path | None,
    database_backups: tuple[tuple[Path, Path], ...],
) -> None:
    for target, backup in database_backups:
        _restore_database(backup, target)
    if registry_backup is not None:
        store.restore_from(registry_backup)
    _write_atomic(snapshot.target, config_backup.read_bytes(), snapshot.mode)
    _check_identity(snapshot)
    if snapshot.path.read_bytes() != snapshot.content:
        raise RuntimeError("Core 模型配置恢复校验失败")


def retire_core_model_config(_connection: object) -> None:
    """Hand legacy model facts to ordinary plugins, then retire Core tables."""

    _ = _connection
    current = current_migration_context()
    snapshot = _snapshot(current.config_path)
    if snapshot is None:
        return
    _parsed, store, handoff = _preflight(snapshot.content, current.workspace)
    rendered = _render(snapshot.content)
    if rendered is None:
        return

    backup_root = current.workspace / "backups" / _MIGRATION / uuid4().hex
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(backup_root, 0o700)
    config_backup = backup_root / "config.toml.before"
    _write_atomic(config_backup, snapshot.content, 0o600)
    registry_backup = None
    if store.exists():
        registry_backup = backup_root / "model-registry.before.sqlite3"
        store.backup_to(registry_backup)
    database_backups: list[tuple[Path, Path]] = []
    if handoff is not None:
        for relative, backup_name in (
            (Path("sessions.db"), "sessions.before.sqlite3"),
        ):
            source = current.workspace / relative
            if source.is_file():
                backup = backup_root / backup_name
                _backup_database(source, backup)
                database_backups.append((source, backup))

    try:
        identity = _apply_embedding_handoff(store, handoff)
        if handoff is not None:
            if identity is None:
                raise RuntimeError("embedding handoff 未生成空间身份")
            _retag_session_embeddings(
                current.workspace,
                old_identity=handoff.model,
                new_identity=identity,
                dimensions=handoff.dimensions,
            )
        _write_atomic(snapshot.target, rendered, snapshot.mode)
        _check_identity(snapshot)
        if snapshot.path.read_bytes() != rendered:
            raise RuntimeError("Core 模型配置发布校验失败")
    except BaseException as migration_error:
        try:
            _restore(
                snapshot,
                config_backup,
                store,
                registry_backup,
                tuple(database_backups),
            )
        except BaseException as restore_error:
            raise RuntimeError(
                f"Core 模型配置迁移失败且恢复失败: {migration_error}"
            ) from restore_error
        raise


steps = [step(retire_core_model_config)]
