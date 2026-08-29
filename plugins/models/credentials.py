from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sqlite3
from collections.abc import Mapping
from contextlib import asynccontextmanager, closing
from pathlib import Path
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, cast

from agent.plugin_composition import AuthenticationError


class CredentialError(AuthenticationError):
    """Report a missing, malformed, or out-of-scope credential."""


class StoredCredentialHandle:
    """Expose one credential identity without granting registry-wide access."""

    def __init__(
        self,
        path: Path,
        connection_id: str,
        auth_identity: str,
        *,
        writable: bool,
        before_write: Callable[[sqlite3.Connection, str], None],
    ) -> None:
        self._path = path
        self._connection_id = _required(connection_id, "connection_id")
        self._auth_identity = _required(auth_identity, "auth_identity")
        self._writable = writable
        self._before_write = before_write

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def auth_identity(self) -> str:
        return self._auth_identity

    async def read(self) -> Mapping[str, str]:
        """Read only the credential bound to this exact scoped identity."""

        _check_private_file(self._path)
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                """
                SELECT auth_kind, auth_payload
                FROM model_connections
                WHERE id = ? AND auth_id = ?
                """,
                (self._connection_id, self._auth_identity),
            ).fetchone()
        if row is None or not str(row[1]):
            raise CredentialError(
                f"credential is unavailable for connection {self._connection_id}"
            )
        return MappingProxyType(_decode_payload(str(row[0]), str(row[1])))

    async def refresh(self, payload: Mapping[str, str]) -> None:
        """Replace this identity's token payload without changing model revision."""

        if not self._writable:
            raise PermissionError("model registry is read-only")
        encoded = _encode_payload(payload)
        _check_private_file(self._path)
        with closing(sqlite3.connect(self._path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT auth_kind
                FROM model_connections
                WHERE id = ? AND auth_id = ?
                """,
                (self._connection_id, self._auth_identity),
            ).fetchone()
            if row is None:
                raise CredentialError(
                    f"credential scope no longer exists: {self._connection_id}"
                )
            auth_kind = _auth_kind(payload, fallback=str(row[0]))
            self._before_write(connection, "refresh-credential")
            cursor = connection.execute(
                """
                UPDATE model_connections
                SET auth_kind = ?, auth_payload = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND auth_id = ?
                """,
                (
                    auth_kind,
                    encoded,
                    self._connection_id,
                    self._auth_identity,
                ),
            )
            if cursor.rowcount != 1:
                raise CredentialError(
                    f"credential scope changed during refresh: {self._connection_id}"
                )
            connection.commit()
        _secure_database_files(self._path)

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Serialize one read-network-refresh sequence across processes."""

        if not self._writable:
            raise PermissionError("model registry is read-only")
        lock_path = self._path.with_name(f"{self._path.name}.credentials.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+", encoding="utf-8")
        os.chmod(lock_path, 0o600)
        acquired = False
        try:
            while not acquired:
                try:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    acquired = True
                except BlockingIOError:
                    await asyncio.sleep(0.01)
            yield
        finally:
            if acquired:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


def encode_credential(payload: Mapping[str, str]) -> tuple[str, str]:
    """Encode a driver-owned credential in the compatible registry columns."""

    return _auth_kind(payload, fallback="opaque"), _encode_payload(payload)


def _auth_kind(payload: Mapping[str, str], *, fallback: str) -> str:
    value = payload.get("driver") or payload.get("kind") or fallback
    return _required(value, "credential kind")


def _encode_payload(payload: Mapping[str, str]) -> str:
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("credential keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"credential value must be a string: {key}")
        normalized[key] = value
    if not normalized:
        raise ValueError("credential must not be empty")
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _decode_payload(auth_kind: str, encoded: str) -> dict[str, str]:
    try:
        raw: Any = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise CredentialError("credential payload is not valid JSON") from exc
    if not isinstance(raw, dict) or not raw:
        raise CredentialError("credential payload must be a non-empty object")
    result: dict[str, str] = {}
    for key, value in cast(dict[object, object], raw).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CredentialError("credential payload must contain only strings")
        result[key] = value
    declared_kind = result.get("driver") or result.get("kind")
    if declared_kind is not None and declared_kind != auth_kind:
        raise CredentialError("credential kind does not match its stored metadata")
    return result


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _check_private_file(path: Path) -> None:
    if not path.is_file():
        raise CredentialError(f"model registry does not exist: {path}")
    if path.stat().st_mode & 0o077:
        raise CredentialError("model registry permissions must be 0600")


def _secure_database_files(path: Path) -> None:
    for candidate in (
        path,
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    ):
        if candidate.exists():
            os.chmod(candidate, 0o600)


__all__ = [
    "CredentialError",
    "StoredCredentialHandle",
    "encode_credential",
]
