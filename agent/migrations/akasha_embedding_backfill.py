from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agent.config_models import Config
from agent.migrations.session_db_backup import backup_sqlite_database
from core.net.http import HttpRequester, SharedHttpResources
from memory2.embedder import Embedder
from plugins.akasha.infrastructure.sparse_index import (
    BuildConfig,
    EmbeddingAudit,
    audit_source_embeddings,
)
from session.embedding_store import MessageEmbeddingStore

_MIGRATION_NAME = "backfill-akasha-message-embeddings"


@dataclass(frozen=True)
class EmbeddingBackfillResult:
    """Report the observable work performed on canonical SessionDB history."""

    eligible_messages: int
    embedded_messages: int
    backup_path: Path | None


def backfill_akasha_message_embeddings(
    *,
    config_path: Path,
    migrated_config: bytes,
    workspace: Path,
) -> EmbeddingBackfillResult:
    """Backfill and verify every Akasha-eligible historical message."""

    sessions_path = workspace / "sessions.db"
    if not sessions_path.is_file():
        return EmbeddingBackfillResult(0, 0, None)

    # 1. Audit existing cache state without changing a complete database.
    host = _load_migrated_config(config_path, migrated_config, workspace)
    audit = _audit_if_cache_exists(sessions_path, host)
    if audit is not None and audit.complete:
        return EmbeddingBackfillResult(audit.eligible_messages, 0, None)

    # 2. Preserve the exact SessionDB before creating or repairing cache rows.
    backup_root = workspace / "backups" / _MIGRATION_NAME / uuid4().hex
    backup = backup_sqlite_database(
        sessions_path,
        backup_root,
        migration=_MIGRATION_NAME,
    )

    # 3. Resolve credentials through the normal Config boundary and repair rows.
    embedded = _run_backfill(sessions_path=sessions_path, host=host)

    # 4. Fail loudly unless replay now has a complete, coherent dense source.
    final = audit_source_embeddings(sessions_path, _build_config(host))
    if not final.complete:
        raise RuntimeError(_format_incomplete_audit(final))
    return EmbeddingBackfillResult(final.eligible_messages, embedded, backup)


def _audit_if_cache_exists(
    sessions_path: Path,
    host: Config,
) -> EmbeddingAudit | None:
    """Audit only when the cache schema already exists."""

    connection = sqlite3.connect(f"file:{sessions_path}?mode=ro", uri=True)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
        ).fetchone()
    finally:
        connection.close()
    if exists is None:
        return None
    return audit_source_embeddings(sessions_path, _build_config(host))


def _load_migrated_config(
    config_path: Path,
    migrated_config: bytes,
    workspace: Path,
) -> Config:
    """Load the post-migration contract before publishing operator config."""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.akasha-migration-",
        suffix=".toml",
        dir=config_path.parent,
    )
    candidate = Path(candidate_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            _ = stream.write(migrated_config)
            stream.flush()
            os.fsync(stream.fileno())
        return Config.load(candidate, workspace=workspace)
    finally:
        candidate.unlink(missing_ok=True)


async def _backfill(
    *,
    sessions_path: Path,
    host: Config,
    requester: HttpRequester,
) -> int:
    """Repair invalid cache rows in deterministic provider-sized batches."""

    settings = host.memory.embedding
    base_url = settings.base_url or host.light_base_url or host.base_url or ""
    api_key = settings.api_key or host.light_api_key or host.api_key
    if not base_url:
        raise ValueError("Akasha 历史 replay 缺少 embedding base_url")
    if not api_key:
        raise ValueError("Akasha 历史 replay 缺少 embedding api_key")

    store = MessageEmbeddingStore(sessions_path)
    embedder = Embedder(
        base_url=base_url,
        api_key=api_key,
        model=settings.model,
        output_dimensionality=settings.output_dimensionality,
        requester=requester,
    )
    try:
        audit = audit_source_embeddings(sessions_path, _build_config(host))
        messages = _load_issue_messages(sessions_path, audit)
        embedded = 0
        for offset in range(0, len(messages), Embedder.MAX_BATCH):
            batch = messages[offset : offset + Embedder.MAX_BATCH]
            vectors = await embedder.embed_batch([content for _, content in batch])
            for (message_id, content), vector in zip(batch, vectors, strict=True):
                store.upsert(
                    message_id=message_id,
                    content=content,
                    model=settings.model,
                    embedding=vector,
                )
                embedded += 1
        return embedded
    finally:
        store.close()
        await embedder.aclose()


async def _backfill_with_resources(*, sessions_path: Path, host: Config) -> int:
    """Own migration HTTP clients within one event-loop lifetime."""

    resources = SharedHttpResources()
    try:
        return await _backfill(
            sessions_path=sessions_path,
            host=host,
            requester=resources.external_default,
        )
    finally:
        await resources.aclose()


def _run_backfill(*, sessions_path: Path, host: Config) -> int:
    """Run the async provider boundary from either sync or async callers."""

    def run() -> int:
        return asyncio.run(
            _backfill_with_resources(
                sessions_path=sessions_path,
                host=host,
            )
        )

    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return run()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(run).result()


def _load_issue_messages(
    sessions_path: Path,
    audit: EmbeddingAudit,
) -> list[tuple[str, str]]:
    """Load exact content for messages selected by the canonical audit."""

    issue_ids = [issue.message_id for issue in audit.issues]
    if not issue_ids:
        return []
    placeholders = ",".join("?" for _ in issue_ids)
    connection = sqlite3.connect(f"file:{sessions_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f"SELECT id, content FROM messages WHERE id IN ({placeholders})",
            issue_ids,
        ).fetchall()
    finally:
        connection.close()
    contents = {str(row[0]): str(row[1] or "") for row in rows}
    missing = [message_id for message_id in issue_ids if message_id not in contents]
    if missing:
        raise RuntimeError(f"Akasha embedding 审计消息不存在: {missing[:5]}")
    return [(message_id, contents[message_id]) for message_id in issue_ids]


def _build_config(host: Config) -> BuildConfig:
    settings = host.memory.embedding
    return BuildConfig(
        embedding_model=settings.model,
        embedding_dimension=settings.output_dimensionality,
    )


def _format_incomplete_audit(audit: EmbeddingAudit) -> str:
    examples = ", ".join(
        f"{issue.message_id}:{issue.reason}" for issue in audit.issues[:5]
    )
    return (
        "Akasha 历史 replay embedding 审计未通过: "
        f"eligible={audit.eligible_messages} valid={audit.valid_messages} "
        f"issues={len(audit.issues)} examples={examples}"
    )
