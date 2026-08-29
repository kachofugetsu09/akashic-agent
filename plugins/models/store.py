from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import quote

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CapabilitySources,
    CreateConnectionWithModel,
    DisableConnection,
    DiscoveredModel,
    ModelCapabilities,
    ModelKind,
    RevisionConflictError,
    SetDefaultModel,
    UpdateConnection,
)

from .credentials import StoredCredentialHandle, encode_credential

MODEL_ROLES = ("default", "fast", "agent", "vision")
_LEGACY_OPENAI_DRIVER_IDS = ("openai", "deepseek", "qwen")


@dataclass(frozen=True, slots=True)
class StoredConnection:
    connection_id: str
    name: str
    driver_id: str
    endpoint: str
    auth_identity: str
    driver_config: Mapping[str, Any]
    enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "driver_config", _freeze_json(self.driver_config))


@dataclass(frozen=True, slots=True)
class StoredModel:
    model_id: str
    connection_id: str
    kind: ModelKind
    model: str
    default_reasoning_effort: str | None
    capabilities: ModelCapabilities
    capability_sources: CapabilitySources
    driver_config: Mapping[str, Any]
    discovery_owned: bool
    enabled: bool

    @classmethod
    def from_command(cls, command: AddModel) -> StoredModel:
        """Build the exact model candidate used by settings validation."""

        return cls(
            model_id=_required(command.model_id, "model_id"),
            connection_id=_required(command.connection_id, "connection_id"),
            kind=command.kind,
            model=_required(command.model, "model"),
            default_reasoning_effort=(
                command.default_reasoning_effort.strip()
                if command.default_reasoning_effort
                else None
            ),
            capabilities=command.capabilities,
            capability_sources=command.capability_sources,
            driver_config=_freeze_json(command.driver_config),
            discovery_owned=False,
            enabled=True,
        )


@dataclass(frozen=True, slots=True)
class StoredSnapshot:
    revision: int
    connections: Mapping[str, StoredConnection]
    models: Mapping[str, StoredModel]
    role_bindings: Mapping[str, str]
    role_reasoning_efforts: Mapping[str, str]
    default_embedding_model_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "connections", MappingProxyType(dict(self.connections))
        )
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
        object.__setattr__(
            self, "role_bindings", MappingProxyType(dict(self.role_bindings))
        )
        object.__setattr__(
            self,
            "role_reasoning_efforts",
            MappingProxyType(dict(self.role_reasoning_efforts)),
        )

    @classmethod
    def empty(cls) -> StoredSnapshot:
        """Represent a valid initialized registry without configured models."""

        return cls(
            revision=0,
            connections={},
            models={},
            role_bindings={},
            role_reasoning_efforts={},
            default_embedding_model_id=None,
        )


class ModelsStore:
    """Own the ordinary models plugin's durable registry and write protocol."""

    def __init__(self, path: Path, backup_dir: Path, writable: bool = True) -> None:
        self.path = path
        self.backup_dir = backup_dir
        self.writable = writable

    def initialize(self) -> None:
        """Create a new registry or expand the two approved additive columns."""

        if not self.writable:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = self._create_database_file()
        try:
            with self._connect() as connection:
                if created:
                    connection.executescript(_SCHEMA)
                    connection.execute(
                        "INSERT INTO model_registry_meta(singleton, revision) VALUES (1, 0)"
                    )
                    connection.commit()
                else:
                    connection.execute("BEGIN IMMEDIATE")
                    _require_base_schema(connection)
                    additions = _missing_additive_columns(connection)
                    legacy_driver_ids = _legacy_openai_driver_ids(connection)
                    if additions or legacy_driver_ids:
                        self._backup_locked(connection, "upgrade-schema")
                        for statement in additions:
                            connection.execute(statement)
                        if legacy_driver_ids:
                            connection.execute(
                                "UPDATE model_connections "
                                "SET provider = 'openai-compatible' "
                                f"WHERE provider IN ({','.join('?' for _ in legacy_driver_ids)})",
                                legacy_driver_ids,
                            )
                        connection.commit()
        finally:
            self._secure_files()

    def read_snapshot(self) -> StoredSnapshot | None:
        """Read connections, models, bindings, and revision from one transaction."""

        if not self.path.is_file():
            return None
        with self._connect(read_only=True) as connection:
            connection.execute("BEGIN")
            _require_base_schema(connection)
            meta_columns = _columns(connection, "model_registry_meta")
            connection_columns = _columns(connection, "model_connections")
            model_columns = _columns(connection, "model_definitions")
            embedding_columns = _columns(connection, "embedding_models")
            default_column = (
                ", default_embedding_model_id"
                if "default_embedding_model_id" in meta_columns
                else ""
            )
            meta = connection.execute(
                f"SELECT revision{default_column} FROM model_registry_meta "
                "WHERE singleton = 1"
            ).fetchone()
            if meta is None:
                raise RuntimeError("model registry is missing revision metadata")
            config_column = (
                "driver_config_json"
                if "driver_config_json" in connection_columns
                else "'{}'"
            )
            connection_rows = connection.execute(
                "SELECT id, name, provider, catalog_provider_id, base_url, auth_id, "
                f"{config_column}, enabled FROM model_connections ORDER BY created_at, id"
            ).fetchall()
            chat_rows = connection.execute(
                _SELECT_CHAT_MODELS.format(
                    capabilities_json=(
                        "capabilities_json"
                        if "capabilities_json" in model_columns
                        else "NULL"
                    )
                )
            ).fetchall()
            embedding_rows = connection.execute(
                _SELECT_EMBEDDING_MODELS.format(
                    capabilities_json=(
                        "capabilities_json"
                        if "capabilities_json" in embedding_columns
                        else "NULL"
                    )
                )
            ).fetchall()
            role_rows = connection.execute(
                "SELECT role, model_id, reasoning_effort "
                "FROM model_role_bindings ORDER BY role"
            ).fetchall()

        connections = {
            item.connection_id: item
            for item in (_connection_from_row(row) for row in connection_rows)
        }
        model_items = tuple(_chat_model_from_row(row) for row in chat_rows) + tuple(
            _embedding_model_from_row(row) for row in embedding_rows
        )
        models: dict[str, StoredModel] = {}
        for item in model_items:
            if item.model_id in models:
                raise RuntimeError(f"duplicate model id across kinds: {item.model_id}")
            models[item.model_id] = item
        roles = {str(row[0]): str(row[1]) for row in role_rows}
        efforts = {str(row[0]): str(row[2]) for row in role_rows}
        default_embedding = str(meta[1]) if len(meta) > 1 and meta[1] else None
        return StoredSnapshot(
            revision=int(meta[0]),
            connections=connections,
            models=models,
            role_bindings=roles,
            role_reasoning_efforts=efforts,
            default_embedding_model_id=default_embedding,
        )

    def add_connection(self, command: AddConnection) -> int:
        """Add one connection and credential as one revision."""

        def write(connection: sqlite3.Connection) -> None:
            _insert_connection(connection, command)

        return self._domain_write(command.expected_revision, "add-connection", write)

    def create_connection_with_model(
        self,
        command: CreateConnectionWithModel,
    ) -> int:
        """Commit one checked connection and first model in one write set."""

        def write(connection: sqlite3.Connection) -> None:
            _insert_connection(connection, command.connection)
            _insert_model(connection, command.model)

        return self._domain_write(
            command.connection.expected_revision,
            "create-connection-with-model",
            write,
        )

    def update_connection(self, command: UpdateConnection) -> int:
        """Update the mutable fields of one existing connection."""

        connection_id = _required(command.connection_id, "connection_id")
        name = _required(command.name, "name")
        endpoint = (
            None
            if command.endpoint is None
            else _required(command.endpoint, "endpoint")
        )
        auth_identity = _required(command.auth_identity, "auth_identity")
        config = (
            None
            if command.driver_config is None
            else _json_object(command.driver_config, "driver_config")
        )
        catalog_provider_id = (
            None
            if command.driver_config is None
            else _catalog_provider_id(command.driver_config)
        )
        credential = (
            None
            if command.credential is None
            else encode_credential(command.credential)
        )

        def write(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT auth_id FROM model_connections WHERE id = ?", (connection_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"connection does not exist: {connection_id}")
            if str(row[0]) != auth_identity and credential is None:
                raise ValueError(
                    "changing auth_identity requires a replacement credential"
                )
            assignments = [
                "name = ?",
                "auth_id = ?",
                "updated_at = CURRENT_TIMESTAMP",
            ]
            values: list[object] = [name, auth_identity]
            if endpoint is not None:
                assignments.insert(1, "base_url = ?")
                values.insert(1, endpoint)
            if config is not None:
                assignments.extend(
                    ("driver_config_json = ?", "catalog_provider_id = ?")
                )
                values.extend((config, catalog_provider_id))
            if credential is not None:
                assignments.extend(("auth_kind = ?", "auth_payload = ?"))
                values.extend(credential)
            values.append(connection_id)
            connection.execute(
                f"UPDATE model_connections SET {', '.join(assignments)} WHERE id = ?",
                values,
            )

        return self._domain_write(command.expected_revision, "update-connection", write)

    def disable_connection(self, command: DisableConnection) -> int:
        """Logically disable one connection while preserving all durable rows."""

        connection_id = _required(command.connection_id, "connection_id")

        def write(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                """
                UPDATE model_connections
                SET enabled = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND enabled = 1
                """,
                (connection_id,),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"connection does not exist or is already disabled: {connection_id}"
                )

        return self._domain_write(
            command.expected_revision, "disable-connection", write
        )

    def add_model(self, command: AddModel) -> int:
        """Add one chat or embedding model using the existing normalized tables."""

        def write(connection: sqlite3.Connection) -> None:
            _insert_model(connection, command)

        return self._domain_write(command.expected_revision, "add-model", write)

    def set_default(self, command: SetDefaultModel) -> int:
        """Set one chat role or the workspace default embedding model."""

        model_id = _required(command.model_id, "model_id")
        role = command.role
        role_value = (
            None
            if role is None
            else str(role.value if hasattr(role, "value") else role)
        )

        def write(connection: sqlite3.Connection) -> None:
            if role_value is None:
                row = connection.execute(
                    """
                    SELECT 1 FROM embedding_models AS m
                    JOIN model_connections AS c ON c.id = m.connection_id
                    WHERE m.id = ? AND m.enabled = 1 AND c.enabled = 1
                    """,
                    (model_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"embedding model is unavailable: {model_id}")
                connection.execute(
                    "UPDATE model_registry_meta SET default_embedding_model_id = ? "
                    "WHERE singleton = 1",
                    (model_id,),
                )
                return
            if role_value not in MODEL_ROLES:
                raise ValueError(f"unsupported model role: {role_value}")
            row = connection.execute(
                """
                SELECT m.input_modalities, m.capabilities_json
                FROM model_definitions AS m
                JOIN model_connections AS c ON c.id = m.connection_id
                WHERE m.id = ? AND m.enabled = 1 AND c.enabled = 1
                """,
                (model_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"chat model is unavailable: {model_id}")
            if role_value == "vision":
                payload = _decode_model_payload(row[1], f"model {model_id}")
                modalities = (
                    payload[0].input_modalities
                    if payload is not None
                    else tuple(
                        _decode_string_list(
                            str(row[0]),
                            f"model {model_id} modalities",
                        )
                    )
                )
                if "image" not in modalities:
                    raise ValueError("vision role requires an image-capable model")
            connection.execute(
                """
                INSERT INTO model_role_bindings(role, model_id, reasoning_effort)
                VALUES (?, ?, '')
                ON CONFLICT(role) DO UPDATE SET
                    model_id = excluded.model_id,
                    reasoning_effort = excluded.reasoning_effort,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (role_value, model_id),
            )

        return self._domain_write(command.expected_revision, "set-default", write)

    def sync_models(
        self,
        expected_revision: int,
        connection_id: str,
        discovered: tuple[DiscoveredModel, ...],
    ) -> int:
        """Persist one driver's normalized discovery evidence as one revision."""

        target_connection = _required(connection_id, "connection_id")
        items = tuple(discovered)
        if not items:
            raise ValueError("driver returned an empty model catalog")
        keys: set[tuple[ModelKind, str]] = set()
        for item in items:
            if not isinstance(item.kind, ModelKind):
                raise ValueError(f"driver returned unsupported model kind: {item.kind}")
            if not isinstance(item.capabilities, ModelCapabilities):
                raise TypeError("driver returned invalid model capabilities")
            if not isinstance(item.capability_sources, CapabilitySources):
                raise TypeError("driver returned invalid capability sources")
            model = _required(item.model, "discovered model")
            if model != item.model:
                raise ValueError("discovered model must not contain outer whitespace")
            key = (item.kind, model)
            if key in keys:
                raise ValueError(
                    f"driver returned duplicate model: {item.kind.value}/{model}"
                )
            keys.add(key)
            if item.kind is ModelKind.EMBEDDING:
                dimensions = item.capabilities.embedding_dimensions
                if dimensions is None or dimensions <= 0:
                    raise ValueError(f"embedding model lacks dimensions: {model}")

        def write(connection: sqlite3.Connection) -> bool:
            active = connection.execute(
                "SELECT enabled FROM model_connections WHERE id = ?",
                (target_connection,),
            ).fetchone()
            if active is None or not bool(active[0]):
                raise ValueError(
                    f"connection does not exist or is disabled: {target_connection}"
                )
            current = self.read_snapshot()
            if current is None:
                raise RuntimeError("model registry disappeared during catalog sync")
            if not _sync_would_change(current, target_connection, items):
                return False
            existing = _existing_model_ids(connection, target_connection)
            used = _all_model_ids(connection)
            desired = {(item.kind, item.model) for item in items}
            for key, (model_id, discovery_owned) in existing.items():
                if discovery_owned and key not in desired:
                    table = (
                        "model_definitions"
                        if key[0] is ModelKind.CHAT
                        else "embedding_models"
                    )
                    connection.execute(
                        f"UPDATE {table} SET enabled = 0, updated_at = CURRENT_TIMESTAMP "
                        "WHERE id = ? AND enabled = 1",
                        (model_id,),
                    )
            for item in items:
                key = (item.kind, item.model)
                stored = existing.get(key)
                if stored is not None and not stored[1]:
                    continue
                model_id = (
                    stored[0]
                    if stored is not None
                    else _discovered_model_id(target_connection, item.kind, item.model)
                )
                owner = used.get(model_id)
                if owner is not None and owner != (
                    target_connection,
                    item.kind,
                    item.model,
                ):
                    raise ValueError(
                        f"discovered model id conflicts with existing model: {model_id}"
                    )
                command = AddModel(
                    expected_revision=expected_revision,
                    model_id=model_id,
                    connection_id=target_connection,
                    kind=item.kind,
                    model=item.model,
                    capabilities=item.capabilities,
                    capability_sources=item.capability_sources,
                    default_reasoning_effort=item.default_reasoning_effort,
                    driver_config=item.driver_config,
                )
                if item.kind is ModelKind.CHAT:
                    connection.execute(
                        _UPSERT_CHAT_MODEL,
                        _chat_model_values(
                            command,
                            model_id,
                            target_connection,
                            item.model,
                            source="discovery",
                        ),
                    )
                else:
                    connection.execute(
                        _UPSERT_EMBEDDING_MODEL,
                        (
                            model_id,
                            target_connection,
                            item.model,
                            int(item.capabilities.embedding_dimensions or 0),
                            _model_payload(command, source="discovery"),
                        ),
                    )
                used[model_id] = (target_connection, item.kind, item.model)
            return True

        return self._domain_write(expected_revision, "sync-models", write)

    def credential_handle(
        self, connection_id: str, auth_identity: str
    ) -> StoredCredentialHandle:
        """Create a handle scoped to one exact Connection and auth identity."""

        return StoredCredentialHandle(
            self.path,
            connection_id,
            auth_identity,
            writable=self.writable,
            before_write=self._backup_locked,
        )

    def integrity_check(self) -> None:
        """Fail loudly unless SQLite and every foreign key are valid."""

        with self._connect(read_only=True) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or str(result[0]) != "ok":
                raise RuntimeError(f"model registry integrity check failed: {result}")
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise RuntimeError(
                f"model registry foreign key check failed: {foreign_keys}"
            )

    def _domain_write(
        self,
        expected_revision: int,
        operation: str,
        write: Callable[[sqlite3.Connection], bool | None],
    ) -> int:
        """CAS, back up, apply one write set, and publish one new revision."""

        if not self.writable:
            raise PermissionError("model registry is read-only")
        self.initialize()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = _revision(connection)
            if current != expected_revision:
                raise RevisionConflictError(
                    "model registry revision changed: "
                    f"expected {expected_revision}, actual {current}"
                )
            changed = write(connection)
            if changed is False:
                connection.rollback()
                return current
            self._backup_locked(connection, operation)
            connection.execute(
                "UPDATE model_registry_meta SET revision = revision + 1 "
                "WHERE singleton = 1"
            )
            connection.commit()
        return current + 1

    def _backup_locked(self, connection: sqlite3.Connection, operation: str) -> None:
        """Back up the exact pre-write database while the caller holds its lock."""

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.backup_dir, 0o700)
        revision = _revision(connection)
        target = self.backup_dir / (
            f"model-registry.before-{operation}.r{revision}.{uuid.uuid4().hex}.sqlite3"
        )
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            # Python's sqlite backup waits on a source connection that already
            # owns BEGIN IMMEDIATE. A second reader sees the same committed
            # pre-write image while the first connection keeps the CAS lock.
            with closing(sqlite3.connect(self.path)) as source:
                with closing(sqlite3.connect(target)) as destination:
                    source.backup(destination)
            os.chmod(target, 0o600)
            with closing(sqlite3.connect(target)) as backup:
                result = backup.execute("PRAGMA integrity_check").fetchone()
                if result is None or str(result[0]) != "ok":
                    raise RuntimeError(
                        f"model registry backup failed integrity check: {target}"
                    )
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    @contextmanager
    def _connect(self, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
        if read_only:
            encoded = quote(self.path.as_posix(), safe="/:")
            connection = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
        else:
            connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.row_factory = sqlite3.Row
            yield connection
        finally:
            connection.close()
            if not read_only:
                self._secure_files()

    def _create_database_file(self) -> bool:
        if self.path.exists():
            os.chmod(self.path, 0o600)
            return False
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        return True

    def _secure_files(self) -> None:
        for candidate in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            if candidate.exists():
                os.chmod(candidate, 0o600)


def _revision(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT revision FROM model_registry_meta WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("model registry is missing revision metadata")
    return int(row[0])


def _existing_model_ids(
    connection: sqlite3.Connection,
    connection_id: str,
) -> dict[tuple[ModelKind, str], tuple[str, bool]]:
    result: dict[tuple[ModelKind, str], tuple[str, bool]] = {}
    for table, kind in (
        ("model_definitions", ModelKind.CHAT),
        ("embedding_models", ModelKind.EMBEDDING),
    ):
        capabilities_column = (
            "capabilities_json"
            if "capabilities_json" in _columns(connection, table)
            else "NULL"
        )
        rows = connection.execute(
            f"SELECT id, model, {capabilities_column} FROM {table} WHERE connection_id = ?",
            (connection_id,),
        ).fetchall()
        for row in rows:
            key = (kind, str(row[1]))
            if key in result:
                raise RuntimeError(
                    f"duplicate stored model identity: {kind.value}/{row[1]}"
                )
            result[key] = (str(row[0]), _payload_source(row[2]) == "discovery")
    return result


def _all_model_ids(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, ModelKind, str]]:
    result: dict[str, tuple[str, ModelKind, str]] = {}
    for table, kind in (
        ("model_definitions", ModelKind.CHAT),
        ("embedding_models", ModelKind.EMBEDDING),
    ):
        rows = connection.execute(
            f"SELECT id, connection_id, model FROM {table}"
        ).fetchall()
        for row in rows:
            model_id = str(row[0])
            if model_id in result:
                raise RuntimeError(f"duplicate model id across kinds: {model_id}")
            result[model_id] = (str(row[1]), kind, str(row[2]))
    return result


def _discovered_model_id(connection_id: str, kind: ModelKind, model: str) -> str:
    """Build one deterministic store-owned ID for newly discovered evidence."""

    parts = (connection_id, kind.value, model)
    return "discovered:" + "".join(f"{len(part)}:{part}" for part in parts)


def _sync_would_change(
    snapshot: StoredSnapshot,
    connection_id: str,
    items: tuple[DiscoveredModel, ...],
) -> bool:
    current = {
        (model.kind, model.model): model
        for model in snapshot.models.values()
        if model.connection_id == connection_id
    }
    desired_keys = {(item.kind, item.model) for item in items}
    if any(
        model.discovery_owned and model.enabled and key not in desired_keys
        for key, model in current.items()
    ):
        return True
    for item in items:
        stored = current.get((item.kind, item.model))
        if stored is not None and not stored.discovery_owned:
            continue
        desired = StoredModel(
            model_id=(
                stored.model_id
                if stored is not None
                else _discovered_model_id(connection_id, item.kind, item.model)
            ),
            connection_id=connection_id,
            kind=item.kind,
            model=item.model,
            default_reasoning_effort=(
                item.default_reasoning_effort.strip()
                if item.default_reasoning_effort
                else None
            ),
            capabilities=item.capabilities,
            capability_sources=item.capability_sources,
            driver_config=item.driver_config,
            discovery_owned=True,
            enabled=True,
        )
        if stored != desired:
            return True
    return False


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _require_base_schema(connection: sqlite3.Connection) -> None:
    required = {
        "model_registry_meta",
        "model_connections",
        "model_definitions",
        "embedding_models",
        "model_role_bindings",
    }
    found = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = sorted(required - found)
    if missing:
        raise RuntimeError(f"model registry schema is incomplete: {', '.join(missing)}")


def _missing_additive_columns(connection: sqlite3.Connection) -> tuple[str, ...]:
    statements: list[str] = []
    if "driver_config_json" not in _columns(connection, "model_connections"):
        statements.append(
            "ALTER TABLE model_connections ADD COLUMN "
            "driver_config_json TEXT NOT NULL DEFAULT '{}'"
        )
    if "default_embedding_model_id" not in _columns(connection, "model_registry_meta"):
        statements.append(
            "ALTER TABLE model_registry_meta ADD COLUMN "
            "default_embedding_model_id TEXT DEFAULT NULL"
        )
    if "capabilities_json" not in _columns(connection, "model_definitions"):
        statements.append(
            "ALTER TABLE model_definitions ADD COLUMN capabilities_json TEXT"
        )
    if "capabilities_json" not in _columns(connection, "embedding_models"):
        statements.append(
            "ALTER TABLE embedding_models ADD COLUMN capabilities_json TEXT"
        )
    return tuple(statements)


def _legacy_openai_driver_ids(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Find retired provider IDs implemented by the OpenAI-compatible driver."""

    placeholders = ",".join("?" for _ in _LEGACY_OPENAI_DRIVER_IDS)
    rows = connection.execute(
        "SELECT DISTINCT provider FROM model_connections "
        f"WHERE provider IN ({placeholders}) ORDER BY provider",
        _LEGACY_OPENAI_DRIVER_IDS,
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _connection_from_row(row: sqlite3.Row) -> StoredConnection:
    config = _decode_json_object(str(row[6]), f"connection {row[0]} driver config")
    # Legacy rows used this separate provider hint. New rows leave it empty and
    # store every driver option in driver_config_json.
    catalog_provider_id = str(row[3])
    if catalog_provider_id:
        existing = config.setdefault("catalog_provider_id", catalog_provider_id)
        if existing != catalog_provider_id:
            raise RuntimeError(f"connection {row[0]} catalog provider 冲突")
    return StoredConnection(
        connection_id=str(row[0]),
        name=str(row[1]),
        driver_id=str(row[2]),
        endpoint=str(row[4]),
        auth_identity=str(row[5]),
        driver_config=MappingProxyType(config),
        enabled=bool(row[7]),
    )


def _chat_model_from_row(row: sqlite3.Row) -> StoredModel:
    efforts = _decode_string_list(str(row[5]), f"model {row[0]} reasoning efforts")
    modalities = _decode_string_list(str(row[8]), f"model {row[0]} modalities")
    payload = _decode_model_payload(row[16], f"model {row[0]}")
    if payload is None:
        capabilities = ModelCapabilities(
            context_window=int(row[6]) or None,
            max_output_tokens=int(row[7]) or None,
            input_modalities=tuple(modalities),
            supports_tool_calls=True,
            supports_parallel_tool_calls=bool(row[13]),
            supported_reasoning_efforts=tuple(efforts),
        )
        sources = CapabilitySources(
            context_window=str(row[10]),
            max_output_tokens=str(row[11]),
            input_modalities=str(row[12]),
            tool_calls=str(row[9]),
            parallel_tool_calls=str(row[9]),
            reasoning_efforts=str(row[9]),
        )
        driver_config: Mapping[str, Any] = MappingProxyType(
            {
                "use_responses_lite": bool(row[14]),
                "reasoning_summary": str(row[15]),
            }
        )
    else:
        capabilities, sources, driver_config, source = payload
    if payload is None:
        source = "manual"
    return StoredModel(
        model_id=str(row[0]),
        connection_id=str(row[1]),
        kind=ModelKind.CHAT,
        model=str(row[2]),
        default_reasoning_effort=str(row[4]) or None,
        capabilities=capabilities,
        capability_sources=sources,
        driver_config=driver_config,
        discovery_owned=source == "discovery",
        enabled=bool(row[3]),
    )


def _embedding_model_from_row(row: sqlite3.Row) -> StoredModel:
    payload = _decode_model_payload(row[5], f"embedding model {row[0]}")
    if payload is None:
        capabilities = ModelCapabilities(
            input_modalities=("text",),
            supports_tool_calls=False,
            supports_parallel_tool_calls=False,
            embedding_dimensions=int(row[4]),
            embedding_normalization="none",
        )
        sources = CapabilitySources()
        driver_config: Mapping[str, Any] = MappingProxyType({})
    else:
        capabilities, sources, driver_config, source = payload
        if capabilities.embedding_dimensions != int(row[4]):
            raise RuntimeError(f"embedding model {row[0]} dimensions 冲突")
    return StoredModel(
        model_id=str(row[0]),
        connection_id=str(row[1]),
        kind=ModelKind.EMBEDDING,
        model=str(row[2]),
        default_reasoning_effort=None,
        capabilities=capabilities,
        capability_sources=sources,
        driver_config=driver_config,
        discovery_owned=("manual" if payload is None else source) == "discovery",
        enabled=bool(row[3]),
    )


def _insert_connection(
    connection: sqlite3.Connection,
    command: AddConnection,
) -> None:
    connection_id = _required(command.connection_id, "connection_id")
    name = _required(command.name, "name")
    driver_id = _required(command.driver_id, "driver_id")
    endpoint = _required(command.endpoint, "endpoint")
    auth_identity = _required(command.auth_identity, "auth_identity")
    config = _json_object(command.driver_config, "driver_config")
    catalog_provider_id = _catalog_provider_id(command.driver_config)
    auth_kind, auth_payload = encode_credential(command.credential)
    connection.execute(
        """
        INSERT INTO model_connections(
            id, name, provider, catalog_provider_id, auth_id, base_url,
            auth_kind, auth_payload, driver_config_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            connection_id,
            name,
            driver_id,
            catalog_provider_id,
            auth_identity,
            endpoint,
            auth_kind,
            auth_payload,
            config,
        ),
    )


def _insert_model(connection: sqlite3.Connection, command: AddModel) -> None:
    model_id = _required(command.model_id, "model_id")
    connection_id = _required(command.connection_id, "connection_id")
    model = _required(command.model, "model")
    kind = str(command.kind.value if hasattr(command.kind, "value") else command.kind)
    if kind not in {"chat", "embedding"}:
        raise ValueError(f"unsupported model kind: {kind}")
    duplicate = connection.execute(
        "SELECT id FROM model_definitions WHERE id = ? "
        "UNION ALL SELECT id FROM embedding_models WHERE id = ? LIMIT 1",
        (model_id, model_id),
    ).fetchone()
    if duplicate is not None:
        raise ValueError(f"model already exists: {model_id}")
    active = connection.execute(
        "SELECT enabled FROM model_connections WHERE id = ?", (connection_id,)
    ).fetchone()
    if active is None or not bool(active[0]):
        raise ValueError(f"connection does not exist or is disabled: {connection_id}")
    if kind == "chat":
        connection.execute(
            _INSERT_CHAT_MODEL,
            _chat_model_values(command, model_id, connection_id, model),
        )
        return
    dimensions = command.capabilities.embedding_dimensions
    if dimensions is None or int(dimensions) <= 0:
        raise ValueError("embedding dimensions must be greater than zero")
    normalization = command.capabilities.embedding_normalization or "none"
    if not isinstance(normalization, str) or not normalization.strip():
        raise ValueError("embedding normalization must be a non-empty string")
    connection.execute(
        "INSERT INTO embedding_models("
        "id, connection_id, model, dimensions, capabilities_json"
        ") VALUES (?, ?, ?, ?, ?)",
        (
            model_id,
            connection_id,
            model,
            int(dimensions),
            _model_payload(command),
        ),
    )


def _chat_model_values(
    command: AddModel,
    model_id: str,
    connection_id: str,
    model: str,
    *,
    source: str = "manual",
) -> tuple[object, ...]:
    capabilities = command.capabilities
    sources = command.capability_sources
    context_window = int(capabilities.context_window or 0)
    max_output_tokens = int(capabilities.max_output_tokens or 0)
    if context_window < 0 or max_output_tokens < 0:
        raise ValueError("model token limits must not be negative")
    modalities = tuple(capabilities.input_modalities)
    efforts = tuple(capabilities.supported_reasoning_efforts)
    if not modalities or any(
        not isinstance(item, str) or not item for item in modalities
    ):
        raise ValueError("input modalities must be non-empty strings")
    if any(not isinstance(item, str) or not item for item in efforts):
        raise ValueError("reasoning efforts must be non-empty strings")
    return (
        model_id,
        connection_id,
        model,
        str(command.default_reasoning_effort or "").strip(),
        json.dumps(efforts, ensure_ascii=False, separators=(",", ":")),
        context_window,
        max_output_tokens,
        json.dumps(modalities, ensure_ascii=False, separators=(",", ":")),
        _required(sources.context_window, "context window source"),
        _required(sources.context_window, "context window source"),
        _required(sources.max_output_tokens, "max output tokens source"),
        _required(sources.input_modalities, "input modalities source"),
        int(bool(capabilities.supports_parallel_tool_calls)),
        int(bool(command.driver_config.get("use_responses_lite", False))),
        str(command.driver_config.get("reasoning_summary") or "none"),
        _model_payload(command, source=source),
    )


def _model_payload(command: Any, *, source: str = "manual") -> str:
    value = {
        "capabilities": asdict(command.capabilities),
        "capability_sources": asdict(command.capability_sources),
        "driver_config": command.driver_config,
        "source": source,
    }
    return _strict_json(value, "model capabilities")


def _decode_model_payload(
    raw: object,
    name: str,
) -> tuple[ModelCapabilities, CapabilitySources, Mapping[str, Any], str] | None:
    if raw is None or not str(raw):
        return None
    try:
        value: Any = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} capabilities JSON 已损坏") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} capabilities 必须是 object")
    capabilities = value.get("capabilities")
    sources = value.get("capability_sources")
    driver_config = value.get("driver_config", {})
    source = value.get("source", "manual")
    if not isinstance(capabilities, dict) or not isinstance(sources, dict):
        raise RuntimeError(f"{name} capability fields 已损坏")
    if not isinstance(driver_config, dict):
        raise RuntimeError(f"{name} driver_config 已损坏")
    if source not in {"manual", "discovery"}:
        raise RuntimeError(f"{name} source 已损坏")
    for field_name in (
        "input_modalities",
        "supported_reasoning_efforts",
    ):
        field_value = capabilities.get(field_name)
        if isinstance(field_value, list):
            capabilities[field_name] = tuple(field_value)
    try:
        return (
            ModelCapabilities(**capabilities),
            CapabilitySources(**sources),
            cast(Mapping[str, Any], _freeze_json(driver_config)),
            source,
        )
    except TypeError as exc:
        raise RuntimeError(f"{name} capability fields 已损坏") from exc


def _payload_source(raw: object) -> str:
    if raw is None or not str(raw):
        return "manual"
    try:
        value: Any = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError("model capability source JSON 已损坏") from exc
    if not isinstance(value, dict) or value.get("source", "manual") not in {
        "manual",
        "discovery",
    }:
        raise RuntimeError("model capability source 已损坏")
    return str(value.get("source", "manual"))


def _json_object(value: Mapping[str, Any], name: str) -> str:
    encoded = _strict_json(value, name)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be an object")
    return encoded


def _catalog_provider_id(config: Mapping[str, Any]) -> str:
    """Keep the legacy column equal to the public driver config."""

    value = config.get("catalog_provider_id", "")
    if not isinstance(value, str):
        raise ValueError("driver_config.catalog_provider_id must be a string")
    if value != value.strip():
        raise ValueError(
            "driver_config.catalog_provider_id must not contain outer whitespace"
        )
    return value


def _decode_json_object(encoded: str, name: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _decode_string_list(encoded: str, name: str) -> list[str]:
    try:
        value: Any = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} is invalid JSON") from exc
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RuntimeError(f"{name} must be a list of non-empty strings")
    return cast(list[str], value)


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object key must be a string")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _strict_json(value: Any, name: str) -> str:
    active: set[int] = set()

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError(f"{name} must not contain cycles")
            active.add(identity)
            try:
                result: dict[str, Any] = {}
                for key, nested in item.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{name} keys must be strings")
                    result[key] = plain(nested)
                return result
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError(f"{name} must not contain cycles")
            active.add(identity)
            try:
                return [plain(nested) for nested in item]
            finally:
                active.remove(identity)
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} numbers must be finite")
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise ValueError(f"{name} contains unsupported value {type(item).__name__}")

    return json.dumps(
        plain(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


_SELECT_CHAT_MODELS = """
SELECT
    id, connection_id, model, enabled, reasoning_effort,
    supported_reasoning_efforts, context_window, max_output_tokens,
    input_modalities, capability_source, context_window_source,
    max_output_tokens_source, input_modalities_source,
    supports_parallel_tool_calls, use_responses_lite, reasoning_summary,
    {capabilities_json}
FROM model_definitions
ORDER BY created_at, id
"""


_SELECT_EMBEDDING_MODELS = """
SELECT id, connection_id, model, enabled, dimensions, {capabilities_json}
FROM embedding_models
ORDER BY created_at, id
"""


_INSERT_CHAT_MODEL = """
INSERT INTO model_definitions(
    id, connection_id, model, reasoning_effort, supported_reasoning_efforts,
    context_window, max_output_tokens, input_modalities,
    capability_source, context_window_source, max_output_tokens_source,
    input_modalities_source, supports_parallel_tool_calls,
    use_responses_lite, reasoning_summary, capabilities_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


_UPSERT_CHAT_MODEL = _INSERT_CHAT_MODEL.rstrip() + """
ON CONFLICT(id) DO UPDATE SET
    model = excluded.model,
    enabled = 1,
    reasoning_effort = excluded.reasoning_effort,
    supported_reasoning_efforts = excluded.supported_reasoning_efforts,
    context_window = excluded.context_window,
    max_output_tokens = excluded.max_output_tokens,
    input_modalities = excluded.input_modalities,
    capability_source = excluded.capability_source,
    context_window_source = excluded.context_window_source,
    max_output_tokens_source = excluded.max_output_tokens_source,
    input_modalities_source = excluded.input_modalities_source,
    supports_parallel_tool_calls = excluded.supports_parallel_tool_calls,
    use_responses_lite = excluded.use_responses_lite,
    reasoning_summary = excluded.reasoning_summary,
    capabilities_json = excluded.capabilities_json,
    updated_at = CURRENT_TIMESTAMP
"""


_UPSERT_EMBEDDING_MODEL = """
INSERT INTO embedding_models(id, connection_id, model, dimensions, capabilities_json)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    model = excluded.model,
    enabled = 1,
    dimensions = excluded.dimensions,
    capabilities_json = excluded.capabilities_json,
    updated_at = CURRENT_TIMESTAMP
"""


_SCHEMA = """
CREATE TABLE model_registry_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    default_embedding_model_id TEXT DEFAULT NULL
);

CREATE TABLE model_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    catalog_provider_id TEXT NOT NULL DEFAULT '',
    auth_id TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    auth_kind TEXT NOT NULL DEFAULT '',
    auth_payload TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    driver_config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE model_definitions (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES model_connections(id) ON DELETE RESTRICT,
    model TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    reasoning_effort TEXT NOT NULL DEFAULT '',
    supported_reasoning_efforts TEXT NOT NULL DEFAULT '[]',
    context_window INTEGER NOT NULL DEFAULT 0 CHECK (context_window >= 0),
    max_output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (max_output_tokens >= 0),
    input_modalities TEXT NOT NULL DEFAULT '["text"]',
    capability_source TEXT NOT NULL DEFAULT 'unknown',
    context_window_source TEXT NOT NULL DEFAULT 'unknown',
    max_output_tokens_source TEXT NOT NULL DEFAULT 'unknown',
    input_modalities_source TEXT NOT NULL DEFAULT 'unknown',
    effective_context_percent REAL NOT NULL DEFAULT 0.9,
    compaction_trigger_percent REAL NOT NULL DEFAULT 0.74,
    use_responses_lite INTEGER NOT NULL DEFAULT 0 CHECK (use_responses_lite IN (0, 1)),
    supports_parallel_tool_calls INTEGER NOT NULL DEFAULT 1 CHECK (supports_parallel_tool_calls IN (0, 1)),
    reasoning_summary TEXT NOT NULL DEFAULT 'none',
    capabilities_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, model)
);

CREATE TABLE embedding_models (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES model_connections(id) ON DELETE RESTRICT,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    capabilities_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, model)
);

CREATE TABLE model_role_bindings (
    role TEXT PRIMARY KEY CHECK (role IN ('default', 'fast', 'agent', 'vision')),
    model_id TEXT NOT NULL REFERENCES model_definitions(id) ON DELETE RESTRICT,
    reasoning_effort TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


__all__ = [
    "MODEL_ROLES",
    "ModelsStore",
    "StoredConnection",
    "StoredModel",
    "StoredSnapshot",
]
