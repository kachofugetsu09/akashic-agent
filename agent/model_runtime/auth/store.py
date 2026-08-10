from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import shutil
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import fcntl

from agent.model_runtime.errors import AuthenticationError


@dataclass(frozen=True)
class Credential:
    driver: str
    access_token: str
    refresh_token: str = ""
    account_id: str = ""
    expires_at: str = ""
    updated_at: str = ""


class CredentialStore:
    """Read credentials from the selected JSON or workspace SQLite owner."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".akashic" / "auth.json"
        self.lock_path = self.path.with_name(f"{self.path.name}.credentials.lock")

    @classmethod
    def for_workspace(cls, workspace: Path) -> CredentialStore:
        return cls(workspace / "model-registry.sqlite3")

    @property
    def is_database(self) -> bool:
        return self.path.name == "model-registry.sqlite3"

    def get(self, credential_id: str) -> Credential:
        if self.is_database:
            return self._get_database_credential(credential_id)
        data = self._read_document()
        raw = data["credentials"].get(credential_id)
        if not isinstance(raw, dict):
            raise AuthenticationError(f"凭据不存在: {credential_id}")
        try:
            return Credential(**raw)
        except TypeError as exc:
            raise AuthenticationError(f"凭据结构无效: {credential_id}") from exc

    def api_key(self, credential_id: str) -> str:
        credential = self.get(credential_id)
        if credential.driver != "api_key" or not credential.access_token:
            raise AuthenticationError(f"凭据 {credential_id} 不是有效 API key")
        return credential.access_token

    def put(self, credential_id: str, credential: Credential) -> None:
        self.put_many({credential_id: credential})

    def put_many(self, credentials: dict[str, Credential]) -> None:
        """在一次锁和原子替换中保存一组凭据。"""
        with self.locked():
            if self.is_database:
                self._write_database_credentials(credentials)
                return
            data = self._read_document()
            for credential_id, credential in credentials.items():
                data["credentials"][credential_id] = asdict(credential)
            self._write_document(data)

    def metadata(self) -> dict[str, dict[str, str]]:
        """返回不含 token 的凭据状态，供本机设置边界展示。"""
        if self.is_database:
            return self._database_metadata()
        data = self._read_document()
        result: dict[str, dict[str, str]] = {}
        for credential_id, raw in data["credentials"].items():
            if not isinstance(raw, dict):
                raise AuthenticationError(f"凭据结构无效: {credential_id}")
            result[credential_id] = {
                "driver": str(raw.get("driver") or ""),
                "updated_at": str(raw.get("updated_at") or ""),
            }
        return result

    def replace_locked(self, credential_id: str, credential: Credential) -> None:
        """调用方持有 store 锁时替换一条凭据。"""
        if self.is_database:
            self._write_database_credentials({credential_id: credential})
            return
        data = self._read_document()
        data["credentials"][credential_id] = asdict(credential)
        self._write_document(data)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """持有跨进程独占锁，供刷新网络请求和持久化共同使用。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.is_database:
            os.chmod(self.path.parent, 0o700)
        lock_file = self.lock_path.open("a+", encoding="utf-8")
        os.chmod(self.lock_path, 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def provision_connection(
        self,
        credential_id: str,
        *,
        name: str,
        provider: str,
        base_url: str,
    ) -> None:
        """Create the provider connection needed by a pre-model login flow."""

        if not self.is_database:
            return
        from agent.model_runtime.store import ModelRegistryStore

        model_store = ModelRegistryStore(self.path)
        model_store.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                INSERT INTO model_connections(
                    id, name, provider, catalog_provider_id, auth_id, base_url
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    provider = excluded.provider,
                    catalog_provider_id = excluded.catalog_provider_id,
                    auth_id = excluded.auth_id,
                    base_url = excluded.base_url,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    credential_id,
                    name,
                    provider,
                    provider,
                    credential_id,
                    base_url,
                ),
            )
            connection.commit()
        self._secure_database_files()

    @staticmethod
    def encode(credential: Credential) -> tuple[str, str]:
        return (
            credential.driver,
            json.dumps(asdict(credential), ensure_ascii=False, separators=(",", ":")),
        )

    @staticmethod
    def decode(
        credential_id: str,
        auth_kind: str,
        auth_payload: str,
    ) -> Credential:
        try:
            raw = json.loads(auth_payload)
            credential = Credential(**raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise AuthenticationError(f"凭据结构无效: {credential_id}") from exc
        if credential.driver != auth_kind:
            raise AuthenticationError(f"凭据类型不一致: {credential_id}")
        return credential

    def _get_database_credential(self, credential_id: str) -> Credential:
        if not self.path.is_file():
            raise AuthenticationError(f"凭据不存在: {credential_id}")
        self._validate_database_permissions()
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                """
                SELECT auth_kind, auth_payload
                FROM model_connections
                WHERE auth_id = ? AND auth_kind != '' AND auth_payload != ''
                """,
                (credential_id,),
            ).fetchall()
        if not rows:
            raise AuthenticationError(f"凭据不存在: {credential_id}")
        credentials = {
            self.decode(credential_id, str(kind), str(payload))
            for kind, payload in rows
        }
        if len(credentials) != 1:
            raise AuthenticationError(f"凭据存在冲突: {credential_id}")
        return credentials.pop()

    def _write_database_credentials(
        self,
        credentials: dict[str, Credential],
    ) -> None:
        if not self.path.is_file():
            raise AuthenticationError("模型注册库尚未初始化")
        self._validate_database_permissions()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for credential_id, credential in credentials.items():
                auth_kind, auth_payload = self.encode(credential)
                cursor = connection.execute(
                    """
                    UPDATE model_connections
                    SET auth_kind = ?, auth_payload = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE auth_id = ?
                    """,
                    (auth_kind, auth_payload, credential_id),
                )
                if cursor.rowcount == 0:
                    raise AuthenticationError(
                        f"凭据没有对应的 Provider connection: {credential_id}"
                    )
            connection.commit()
        self._secure_database_files()

    def _database_metadata(self) -> dict[str, dict[str, str]]:
        if not self.path.is_file():
            return {}
        self._validate_database_permissions()
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                """
                SELECT auth_id, auth_kind, auth_payload
                FROM model_connections
                WHERE auth_id != '' AND auth_kind != '' AND auth_payload != ''
                ORDER BY auth_id
                """
            ).fetchall()
        result: dict[str, dict[str, str]] = {}
        for credential_id, auth_kind, auth_payload in rows:
            credential = self.decode(
                str(credential_id),
                str(auth_kind),
                str(auth_payload),
            )
            metadata = {
                "driver": credential.driver,
                "updated_at": credential.updated_at,
            }
            previous = result.setdefault(str(credential_id), metadata)
            if previous != metadata:
                raise AuthenticationError(f"凭据元数据存在冲突: {credential_id}")
        return result

    def _read_document(self) -> dict:
        if self.is_database:
            raise RuntimeError("SQLite CredentialStore 不能读取 JSON document")
        if not self.path.exists():
            return {"version": 1, "credentials": {}}
        self._validate_permissions()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AuthenticationError(f"凭据文件 JSON 损坏: {self.path}") from exc
        if (
            not isinstance(raw, dict)
            or raw.get("version") != 1
            or not isinstance(raw.get("credentials"), dict)
        ):
            raise AuthenticationError(f"凭据文件结构或版本无效: {self.path}")
        return raw

    def _write_document(self, data: dict) -> None:
        """fsync 后原子替换凭据文件。"""
        fd, temp_name = tempfile.mkstemp(prefix="auth-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            if self.path.exists():
                shutil.copy2(self.path, self.path.with_name("auth.json.before-write.bak"))
            os.replace(temp_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _validate_permissions(self) -> None:
        parent_mode = self.path.parent.stat().st_mode & 0o777
        file_mode = self.path.stat().st_mode & 0o777
        if parent_mode & 0o077:
            raise AuthenticationError("auth.json 父目录权限过宽，必须为 0700")
        if file_mode & 0o077:
            raise AuthenticationError("auth.json 权限过宽，必须为 0600")

    def _validate_database_permissions(self) -> None:
        file_mode = self.path.stat().st_mode & 0o777
        if file_mode & 0o077:
            raise AuthenticationError("model-registry.sqlite3 权限过宽，必须为 0600")

    def _secure_database_files(self) -> None:
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            if path.exists():
                os.chmod(path, 0o600)
