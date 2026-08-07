from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping
from urllib.parse import quote

from agent.model_runtime.auth.store import Credential, CredentialStore

MODEL_REGISTRY_FILENAME = "model-registry.sqlite3"
MODEL_ROLES = ("default", "fast", "agent", "vision")


@dataclass(frozen=True)
class StoredModelRuntime:
    runtime_id: str
    source_id: str
    source_name: str
    provider: str
    catalog_provider_id: str
    auth_id: str
    base_url: str
    model: str
    reasoning_effort: str
    supported_reasoning_efforts: tuple[str, ...]
    context_window: int
    max_output_tokens: int
    input_modalities: tuple[str, ...]
    capability_source: str
    context_window_source: str
    max_output_tokens_source: str
    input_modalities_source: str
    effective_context_percent: float
    compaction_trigger_percent: float
    use_responses_lite: bool
    supports_parallel_tool_calls: bool
    reasoning_summary: str

    def as_config_table(self, *, effort: str = "") -> dict[str, object]:
        """Render one normalized row into the existing config loader shape."""

        return {
            "provider": self.provider,
            "model": self.model,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "catalog_provider_id": self.catalog_provider_id,
            "auth": self.auth_id,
            "base_url": self.base_url,
            "reasoning_effort": effort or self.reasoning_effort,
            "supported_reasoning_efforts": list(self.supported_reasoning_efforts),
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "input_modalities": list(self.input_modalities),
            "capability_source": self.capability_source,
            "context_window_source": self.context_window_source,
            "max_output_tokens_source": self.max_output_tokens_source,
            "input_modalities_source": self.input_modalities_source,
            "effective_context_percent": self.effective_context_percent,
            "compaction_trigger_percent": self.compaction_trigger_percent,
            "use_responses_lite": self.use_responses_lite,
            "supports_parallel_tool_calls": self.supports_parallel_tool_calls,
            "reasoning_summary": self.reasoning_summary,
        }


@dataclass(frozen=True)
class StoredEmbeddingModel:
    model_id: str
    source_id: str
    source_name: str
    provider: str
    auth_id: str
    base_url: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class ModelRoleBinding:
    role: str
    runtime_id: str
    reasoning_effort: str


@dataclass(frozen=True)
class ModelRegistrySnapshot:
    revision: int
    runtimes: Mapping[str, StoredModelRuntime]
    roles: Mapping[str, ModelRoleBinding]

    def as_config_llm(self) -> dict[str, object]:
        """Expose one database revision through the legacy config parser boundary."""

        # 1. Every runtime keeps its own default effort for explicit chat selection.
        runtime_tables = {
            runtime_id: runtime.as_config_table()
            for runtime_id, runtime in self.runtimes.items()
        }

        # 2. Existing consumers use role aliases; role-specific effort is compiled
        # into a private runtime clone only when it differs from the model default.
        aliases: dict[str, str] = {}
        for role in MODEL_ROLES:
            binding = self.roles.get(role)
            if binding is None:
                if role == "default":
                    raise ValueError("模型注册库缺少 default 角色")
                aliases[role] = aliases["default"]
                continue
            runtime = self.runtimes.get(binding.runtime_id)
            if runtime is None:
                raise ValueError(
                    f"模型角色 {role} 引用了不存在的模型: {binding.runtime_id}"
                )
            alias = binding.runtime_id
            if binding.reasoning_effort and (
                binding.reasoning_effort != runtime.reasoning_effort
            ):
                alias = f"__role_{role}"
                runtime_tables[alias] = runtime.as_config_table(
                    effort=binding.reasoning_effort
                )
            aliases[role] = alias

        return {
            "main": aliases["default"],
            "fast": aliases["fast"],
            "agent": aliases["agent"],
            "vl": aliases["vision"],
            "runtimes": runtime_tables,
        }


class ModelRegistryStore:
    """Read and transactionally update workspace-owned model configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_workspace(cls, workspace: Path) -> ModelRegistryStore:
        return cls(workspace / MODEL_REGISTRY_FILENAME)

    def exists(self) -> bool:
        return self.path.is_file()

    def initialize(self) -> None:
        """Create the normalized schema without inventing any model rows."""

        # 1. Schema creation belongs to migration/onboarding owners.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_database_file()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO model_registry_meta(singleton, revision) VALUES (1, 0)"
            )
            connection.commit()

    def backup_to(self, target: Path) -> None:
        """Publish a verified SQLite backup with private permissions."""

        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.touch(mode=0o600, exist_ok=False)
        with self._connect(read_only=True) as source:
            with closing(sqlite3.connect(target)) as destination:
                source.backup(destination)
        os.chmod(target, 0o600)
        ModelRegistryStore(target).integrity_check()

    def restore_from(self, source: Path) -> None:
        """Restore a verified backup into the canonical database path."""

        ModelRegistryStore(source).integrity_check()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_database_file()
        with closing(sqlite3.connect(source)) as backup:
            with closing(sqlite3.connect(self.path)) as destination:
                backup.backup(destination)
        self._secure_database_files()

    def revision(self) -> int:
        """Return the current committed model revision."""

        if not self.exists():
            return 0
        with self._connect(read_only=True) as connection:
            row = connection.execute(
                "SELECT revision FROM model_registry_meta WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("模型注册库缺少 revision 元数据")
        return int(row[0])

    def integrity_check(self) -> None:
        """Fail loudly unless SQLite and every declared foreign key are valid."""

        with self._connect(read_only=True) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or str(result[0]) != "ok":
                raise RuntimeError(f"模型注册库完整性检查失败: {result}")
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise RuntimeError(f"模型注册库外键检查失败: {foreign_keys}")

    def read_snapshot(self) -> ModelRegistrySnapshot | None:
        """Read one consistent revision; an empty registry means onboarding."""

        if not self.exists():
            return None
        with self._connect(read_only=True) as connection:
            connection.execute("BEGIN")
            revision_row = connection.execute(
                "SELECT revision FROM model_registry_meta WHERE singleton = 1"
            ).fetchone()
            if revision_row is None:
                raise RuntimeError("模型注册库缺少 revision 元数据")
            model_rows = connection.execute(_SELECT_MODELS).fetchall()
            role_rows = connection.execute(
                "SELECT role, model_id, reasoning_effort FROM model_role_bindings"
            ).fetchall()
        if not model_rows:
            return None
        runtimes = {str(row[0]): _stored_runtime_from_row(row) for row in model_rows}
        roles = {
            str(row[0]): ModelRoleBinding(
                role=str(row[0]),
                runtime_id=str(row[1]),
                reasoning_effort=str(row[2] or ""),
            )
            for row in role_rows
        }
        if "default" not in roles:
            raise RuntimeError("模型注册库缺少 default 角色")
        return ModelRegistrySnapshot(
            revision=int(revision_row[0]),
            runtimes=runtimes,
            roles=roles,
        )

    def list_embedding_models(self) -> tuple[StoredEmbeddingModel, ...]:
        """Return every enabled embedding model with its Provider connection."""

        if not self.exists():
            return ()
        with self._connect(read_only=True) as connection:
            rows = connection.execute(_SELECT_EMBEDDING_MODELS).fetchall()
        return tuple(_stored_embedding_from_row(row) for row in rows)

    def get_embedding_model(self, model_id: str) -> StoredEmbeddingModel | None:
        """Resolve one enabled embedding model by its stable registry ID."""

        return next(
            (
                item
                for item in self.list_embedding_models()
                if item.model_id == model_id
            ),
            None,
        )

    def upsert_embedding_model(
        self,
        *,
        model_id: str,
        source_id: str,
        source_name: str,
        provider: str,
        auth_id: str,
        base_url: str,
        model: str,
        dimensions: int,
        credential: Credential | None,
        expected_revision: int | None = None,
    ) -> int:
        """Validate and publish one first-class embedding model revision."""

        # 1. Establish the registry boundary before opening the transaction.
        values = {
            "model_id": model_id.strip(),
            "source_id": source_id.strip(),
            "source_name": source_name.strip(),
            "provider": provider.strip().lower(),
            "auth_id": auth_id.strip(),
            "base_url": base_url.strip(),
            "model": model.strip(),
        }
        missing = next((name for name, value in values.items() if not value), None)
        if missing is not None:
            raise ValueError(f"Embedding {missing} 不能为空")
        if dimensions <= 0:
            raise ValueError("Embedding dimensions 必须大于 0")
        self.initialize()

        # 2. Commit the Provider connection, credential, and model as one revision.
        auth_kind, auth_payload = (
            CredentialStore.encode(credential) if credential is not None else ("", "")
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = int(
                connection.execute(
                    "SELECT revision FROM model_registry_meta WHERE singleton = 1"
                ).fetchone()[0]
            )
            if expected_revision is not None and current != expected_revision:
                raise RuntimeError("模型设置已经变化，请刷新后重试")
            connection.execute(
                _INSERT_CONNECTION,
                (
                    values["source_id"],
                    values["source_name"],
                    values["provider"],
                    values["provider"],
                    values["auth_id"],
                    values["base_url"],
                    auth_kind,
                    auth_payload,
                ),
            )
            connection.execute(
                _UPSERT_EMBEDDING_MODEL,
                (
                    values["model_id"],
                    values["source_id"],
                    values["model"],
                    dimensions,
                ),
            )
            connection.execute(
                "UPDATE model_registry_meta SET revision = revision + 1 WHERE singleton = 1"
            )
            connection.commit()
        return current + 1

    def replace_from_llm_config(
        self,
        llm: Mapping[str, object],
        *,
        source_names: Mapping[str, str] | None = None,
        credentials: Mapping[str, Credential] | None = None,
    ) -> int:
        """Import validated named runtimes as one new database revision."""

        runtimes = llm.get("runtimes")
        if not isinstance(runtimes, Mapping) or not runtimes:
            raise ValueError("llm.runtimes 必须是非空 table")
        main = llm.get("main")
        if not isinstance(main, str) or main not in runtimes:
            raise ValueError("llm.main 必须引用已配置 runtime")

        # 1. Normalize every runtime and connection before opening the write txn.
        normalized = [
            _normalize_runtime(runtime_id, raw, source_names or {})
            for runtime_id, raw in runtimes.items()
        ]
        sources: dict[str, tuple[object, ...]] = {}
        for source, _model in normalized:
            source_id = str(source[0])
            previous = sources.setdefault(source_id, source)
            if previous != source:
                raise ValueError(f"同一 source_id 的连接字段不一致: {source_id}")
        roles = {
            "default": main,
            "fast": _role_ref(llm, "fast", main, runtimes),
            "agent": _role_ref(llm, "agent", main, runtimes),
            "vision": _role_ref(llm, "vl", main, runtimes),
        }

        # 2. Replace the imported projection atomically and bump one revision.
        self.initialize()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_credentials: dict[str, tuple[str, str]] = {}
            for auth_id, auth_kind, auth_payload in connection.execute("""
                SELECT auth_id, auth_kind, auth_payload
                FROM model_connections
                WHERE auth_id != '' AND auth_kind != '' AND auth_payload != ''
                """).fetchall():
                credential_id = str(auth_id)
                encoded = (str(auth_kind), str(auth_payload))
                previous = existing_credentials.setdefault(credential_id, encoded)
                if previous != encoded:
                    raise ValueError(f"模型凭据存在冲突: {credential_id}")
            supplied_credentials = {
                credential_id: CredentialStore.encode(credential)
                for credential_id, credential in (credentials or {}).items()
            }
            connection.execute("DELETE FROM model_role_bindings")
            connection.execute("DELETE FROM model_definitions")
            inserted_sources: set[str] = set()
            for source, model in normalized:
                source_id = str(source[0])
                if source_id not in inserted_sources:
                    auth_id = str(source[4])
                    auth_kind, auth_payload = supplied_credentials.get(
                        auth_id,
                        existing_credentials.get(auth_id, ("", "")),
                    )
                    connection.execute(
                        _INSERT_CONNECTION,
                        (*source, auth_kind, auth_payload),
                    )
                    inserted_sources.add(source_id)
                connection.execute(_INSERT_MODEL, model)
            for role, runtime_id in roles.items():
                connection.execute(
                    "INSERT INTO model_role_bindings(role, model_id, reasoning_effort) VALUES (?, ?, '')",
                    (role, runtime_id),
                )
            connection.execute(
                "UPDATE model_registry_meta SET revision = revision + 1 WHERE singleton = 1"
            )
            revision = int(
                connection.execute(
                    "SELECT revision FROM model_registry_meta WHERE singleton = 1"
                ).fetchone()[0]
            )
            connection.commit()
        return revision

    def set_role(
        self,
        role: str,
        runtime_id: str,
        *,
        reasoning_effort: str = "",
        expected_revision: int | None = None,
    ) -> int:
        """Commit one role binding with optimistic concurrency."""

        if role not in MODEL_ROLES:
            raise ValueError(f"未知模型角色: {role}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = int(
                connection.execute(
                    "SELECT revision FROM model_registry_meta WHERE singleton = 1"
                ).fetchone()[0]
            )
            if expected_revision is not None and current != expected_revision:
                raise RuntimeError("模型设置已经变化，请刷新后重试")
            model = connection.execute(
                "SELECT enabled FROM model_definitions WHERE id = ?",
                (runtime_id,),
            ).fetchone()
            if model is None or not bool(model[0]):
                raise ValueError(f"模型不存在或未启用: {runtime_id}")
            connection.execute(
                """
                INSERT INTO model_role_bindings(role, model_id, reasoning_effort)
                VALUES (?, ?, ?)
                ON CONFLICT(role) DO UPDATE SET
                    model_id = excluded.model_id,
                    reasoning_effort = excluded.reasoning_effort,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (role, runtime_id, reasoning_effort.strip()),
            )
            connection.execute(
                "UPDATE model_registry_meta SET revision = revision + 1 WHERE singleton = 1"
            )
            revision = current + 1
            connection.commit()
        return revision

    @contextmanager
    def _connect(
        self,
        *,
        read_only: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        if read_only:
            encoded_path = quote(self.path.as_posix(), safe="/:")
            uri = f"file:{encoded_path}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.row_factory = sqlite3.Row
            yield connection
        finally:
            connection.close()
            if not read_only:
                self._secure_database_files()

    def _create_database_file(self) -> None:
        if self.path.exists():
            os.chmod(self.path, 0o600)
            return
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.close(descriptor)

    def _secure_database_files(self) -> None:
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            if path.exists():
                os.chmod(path, 0o600)


def _normalize_runtime(
    runtime_id: object,
    raw: object,
    source_names: Mapping[str, str],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    if not isinstance(runtime_id, str) or not runtime_id.strip():
        raise ValueError("runtime id 必须是非空字符串")
    if not isinstance(raw, Mapping):
        raise ValueError(f"llm.runtimes.{runtime_id} 必须是 table")
    provider = str(raw.get("provider") or "").strip().lower()
    model = str(raw.get("model") or "").strip()
    if not provider or not model:
        raise ValueError(f"runtime {runtime_id} 必须配置 provider 和 model")
    modalities = raw.get("input_modalities", ["text"])
    if not isinstance(modalities, list) or not all(
        isinstance(value, str) for value in modalities
    ):
        raise ValueError(f"runtime {runtime_id} 的 input_modalities 无效")
    efforts = raw.get("supported_reasoning_efforts", [])
    if not isinstance(efforts, list) or not all(
        isinstance(value, str) and value.strip() for value in efforts
    ):
        raise ValueError(f"runtime {runtime_id} 的 supported_reasoning_efforts 无效")
    source_id = str(raw.get("source_id") or f"source:{runtime_id}")
    source_name = source_names.get(runtime_id) or str(
        raw.get("source_name") or provider
    )
    source = (
        source_id,
        source_name,
        provider,
        str(raw.get("catalog_provider_id") or provider),
        str(raw.get("auth") or ""),
        str(raw.get("base_url") or ""),
    )
    model_row = (
        runtime_id,
        source_id,
        model,
        str(raw.get("reasoning_effort") or ""),
        json.dumps(efforts, ensure_ascii=False, separators=(",", ":")),
        int(raw.get("context_window") or 0),
        int(raw.get("max_output_tokens") or 0),
        json.dumps(modalities, ensure_ascii=False, separators=(",", ":")),
        str(raw.get("capability_source") or "unknown"),
        str(
            raw.get("context_window_source")
            or raw.get("capability_source")
            or "unknown"
        ),
        str(
            raw.get("max_output_tokens_source")
            or raw.get("capability_source")
            or "unknown"
        ),
        str(
            raw.get("input_modalities_source")
            or raw.get("capability_source")
            or "unknown"
        ),
        float(raw.get("effective_context_percent", 0.9)),
        float(raw.get("compaction_trigger_percent", 0.74)),
        int(bool(raw.get("use_responses_lite", False))),
        int(bool(raw.get("supports_parallel_tool_calls", True))),
        str(raw.get("reasoning_summary") or "none"),
    )
    return source, model_row


def _role_ref(
    llm: Mapping[str, object],
    role: str,
    fallback: str,
    runtimes: Mapping[object, object],
) -> str:
    value = llm.get(role, fallback)
    if not isinstance(value, str) or value not in runtimes:
        raise ValueError(f"llm.{role} 必须引用已配置 runtime")
    return value


def _stored_runtime_from_row(row: sqlite3.Row) -> StoredModelRuntime:
    raw_efforts = json.loads(str(row[9]))
    if not isinstance(raw_efforts, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_efforts
    ):
        raise RuntimeError(f"模型 {row[0]} 的 supported_reasoning_efforts 已损坏")
    raw_modalities = json.loads(str(row[11]))
    if not isinstance(raw_modalities, list) or not all(
        isinstance(item, str) for item in raw_modalities
    ):
        raise RuntimeError(f"模型 {row[0]} 的 input_modalities 已损坏")
    return StoredModelRuntime(
        runtime_id=str(row[0]),
        source_id=str(row[1]),
        source_name=str(row[2]),
        provider=str(row[3]),
        catalog_provider_id=str(row[4]),
        auth_id=str(row[5]),
        base_url=str(row[6]),
        model=str(row[7]),
        reasoning_effort=str(row[8]),
        supported_reasoning_efforts=tuple(raw_efforts),
        context_window=int(row[10]),
        input_modalities=tuple(raw_modalities),
        max_output_tokens=int(row[12]),
        capability_source=str(row[13]),
        context_window_source=str(row[14]),
        max_output_tokens_source=str(row[15]),
        input_modalities_source=str(row[16]),
        effective_context_percent=float(row[17]),
        compaction_trigger_percent=float(row[18]),
        use_responses_lite=bool(row[19]),
        supports_parallel_tool_calls=bool(row[20]),
        reasoning_summary=str(row[21]),
    )


def _stored_embedding_from_row(row: sqlite3.Row) -> StoredEmbeddingModel:
    return StoredEmbeddingModel(
        model_id=str(row[0]),
        source_id=str(row[1]),
        source_name=str(row[2]),
        provider=str(row[3]),
        auth_id=str(row[4]),
        base_url=str(row[5]),
        model=str(row[6]),
        dimensions=int(row[7]),
    )


_SELECT_MODELS = """
SELECT
    m.id,
    c.id,
    c.name,
    c.provider,
    c.catalog_provider_id,
    c.auth_id,
    c.base_url,
    m.model,
    m.reasoning_effort,
    m.supported_reasoning_efforts,
    m.context_window,
    m.input_modalities,
    m.max_output_tokens,
    m.capability_source,
    m.context_window_source,
    m.max_output_tokens_source,
    m.input_modalities_source,
    m.effective_context_percent,
    m.compaction_trigger_percent,
    m.use_responses_lite,
    m.supports_parallel_tool_calls,
    m.reasoning_summary
FROM model_definitions AS m
JOIN model_connections AS c ON c.id = m.connection_id
WHERE m.enabled = 1 AND c.enabled = 1
ORDER BY m.created_at, m.id
"""

_SELECT_EMBEDDING_MODELS = """
SELECT
    m.id,
    c.id,
    c.name,
    c.provider,
    c.auth_id,
    c.base_url,
    m.model,
    m.dimensions
FROM embedding_models AS m
JOIN model_connections AS c ON c.id = m.connection_id
WHERE m.enabled = 1 AND c.enabled = 1
ORDER BY m.created_at, m.id
"""

_INSERT_CONNECTION = """
INSERT INTO model_connections(
    id, name, provider, catalog_provider_id, auth_id, base_url,
    auth_kind, auth_payload
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    name = excluded.name,
    provider = excluded.provider,
    catalog_provider_id = excluded.catalog_provider_id,
    auth_id = excluded.auth_id,
    base_url = excluded.base_url,
    auth_kind = CASE
        WHEN excluded.auth_payload != '' THEN excluded.auth_kind
        ELSE model_connections.auth_kind
    END,
    auth_payload = CASE
        WHEN excluded.auth_payload != '' THEN excluded.auth_payload
        ELSE model_connections.auth_payload
    END,
    enabled = 1,
    updated_at = CURRENT_TIMESTAMP
"""

_INSERT_MODEL = """
INSERT INTO model_definitions(
    id,
    connection_id,
    model,
    reasoning_effort,
    supported_reasoning_efforts,
    context_window,
    max_output_tokens,
    input_modalities,
    capability_source,
    context_window_source,
    max_output_tokens_source,
    input_modalities_source,
    effective_context_percent,
    compaction_trigger_percent,
    use_responses_lite,
    supports_parallel_tool_calls,
    reasoning_summary
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPSERT_EMBEDDING_MODEL = """
INSERT INTO embedding_models(id, connection_id, model, dimensions)
VALUES (?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    connection_id = excluded.connection_id,
    model = excluded.model,
    dimensions = excluded.dimensions,
    enabled = 1,
    updated_at = CURRENT_TIMESTAMP
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_registry_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    revision INTEGER NOT NULL CHECK (revision >= 0)
);

CREATE TABLE IF NOT EXISTS model_connections (
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
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_definitions (
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
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, model)
);

CREATE TABLE IF NOT EXISTS embedding_models (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES model_connections(id) ON DELETE RESTRICT,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, model)
);

CREATE TABLE IF NOT EXISTS model_role_bindings (
    role TEXT PRIMARY KEY CHECK (role IN ('default', 'fast', 'agent', 'vision')),
    model_id TEXT NOT NULL REFERENCES model_definitions(id) ON DELETE RESTRICT,
    reasoning_effort TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


__all__ = [
    "MODEL_REGISTRY_FILENAME",
    "MODEL_ROLES",
    "ModelRegistrySnapshot",
    "ModelRegistryStore",
    "ModelRoleBinding",
    "StoredEmbeddingModel",
    "StoredModelRuntime",
]
