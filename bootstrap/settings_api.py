from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import socket
import tempfile
import threading
import tomllib
from collections.abc import Awaitable, Callable
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import tomlkit
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class SettingsServer(uvicorn.Server):
    """Publish one deterministic startup result across the server thread."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.startup_event = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self.startup_event.set()


class MemorySettingsPayload(BaseModel):
    enabled: bool
    embedding_model_id: str = Field(default="", max_length=128)
    expected_revision: str = Field(default="", max_length=64)


EmbeddingModelExists = Callable[[str], Awaitable[bool]]


def create_settings_app(
    config_path: Path,
    workspace: Path,
    *,
    embedding_model_exists: EmbeddingModelExists | None = None,
    on_applied: Callable[[], None] | None = None,
) -> FastAPI:
    """Serve static settings UI and the Akasha-owned memory preference only."""

    apply_lock = threading.Lock()
    static_dir = Path(__file__).resolve().parent.parent / "static" / "chat"
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, error: Exception) -> JSONResponse:
        logger.exception("[settings] unexpected failure", exc_info=error)
        return _error_response(500, "internal_error", "设置操作失败，请查看服务端日志")

    @app.middleware("http")
    async def protect_settings_boundary(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin", "")
            expected = f"http://{request.url.netloc}"
            if origin != expected or request.headers.get("x-akasic-csrf") != "1":
                return _error_response(403, "csrf_rejected", "请求来源无效")
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'self' "
            "http://127.0.0.1:5173 http://localhost:5173"
        )
        return response

    @app.get("/api/settings/memory-state")
    async def memory_state() -> dict[str, object]:
        memory = _read_memory_config(config_path)
        embedding = (
            memory.get("embedding")
            if isinstance(memory.get("embedding"), dict)
            else {}
        )
        return {
            "configured": bool(memory),
            "enabled": bool(memory.get("enabled", False)),
            "embeddingModelId": str(embedding.get("model_ref") or ""),
            "changeLocked": bool(memory) and _workspace_has_messages(workspace),
            "revision": _memory_settings_revision(config_path),
        }

    @app.post("/api/settings/memory")
    async def save_memory(payload: MemorySettingsPayload) -> dict[str, object]:
        model_id = payload.embedding_model_id.strip()
        if payload.enabled and not model_id:
            raise HTTPException(status_code=422, detail="启用语义记忆需要选择向量模型")
        if payload.enabled:
            if embedding_model_exists is None:
                raise HTTPException(status_code=503, detail="模型目录不可用")
            try:
                exists = await embedding_model_exists(model_id)
            except RuntimeError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            if not exists:
                raise HTTPException(status_code=422, detail="选择的向量模型不存在")
        if not apply_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="已有设置操作正在执行")
        try:
            current_revision = _memory_settings_revision(config_path)
            if payload.expected_revision and payload.expected_revision != current_revision:
                raise HTTPException(status_code=409, detail="记忆设置已经变化，请刷新后重试")
            current = _read_memory_config(config_path)
            current_embedding = current.get("embedding")
            current_model_id = (
                str(current_embedding.get("model_ref") or "")
                if isinstance(current_embedding, dict)
                else ""
            )
            changed = (
                bool(current.get("enabled", False)) != payload.enabled
                or current_model_id != model_id
            )
            if current and changed and _workspace_has_messages(workspace):
                raise HTTPException(
                    status_code=409,
                    detail="已有对话的记忆设置需要重建索引后才能切换",
                )
            operation_id = f"memory-settings-{uuid4().hex}"
            _apply_memory_settings(
                config_path,
                workspace,
                enabled=payload.enabled,
                embedding_model_id=model_id,
                operation_id=operation_id,
                on_applied=on_applied,
            )
            return {"status": "applied", "operationId": operation_id}
        finally:
            apply_lock.release()

    @app.get("/chat")
    @app.get("/chat/")
    @app.get("/settings")
    @app.get("/settings/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="settings-assets",
    )
    return app


def _read_memory_config(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        return {}
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise HTTPException(
            status_code=409,
            detail="config.toml 无法解析，请先修复或移走该文件",
        ) from error
    memory = raw.get("memory")
    return dict(memory) if isinstance(memory, dict) else {}


def _workspace_has_messages(workspace: Path) -> bool:
    sessions_path = workspace / "sessions.db"
    if not sessions_path.is_file():
        return False
    uri = f"file:{sessions_path.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        if table is None:
            return False
        row = connection.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
    return row is not None


def _memory_settings_revision(config_path: Path) -> str:
    payload = config_path.read_bytes() if config_path.is_file() else b""
    return hashlib.sha256(payload).hexdigest()


def _apply_memory_settings(
    config_path: Path,
    workspace: Path,
    *,
    enabled: bool,
    embedding_model_id: str,
    operation_id: str,
    on_applied: Callable[[], None] | None,
) -> None:
    """Back up and atomically publish one already-validated memory binding."""

    config_existed = config_path.is_file()
    original = (
        config_path.read_bytes()
        if config_existed
        else _new_config(workspace).encode("utf-8")
    )
    document = tomlkit.parse(original.decode("utf-8"))
    memory = tomlkit.table()
    memory["enabled"] = enabled
    embedding = tomlkit.table()
    if embedding_model_id:
        embedding["model_ref"] = embedding_model_id
    memory["embedding"] = embedding
    document["memory"] = memory
    candidate = tomlkit.dumps(document)
    _ = tomllib.loads(candidate)

    backup_dir = workspace / "backups" / "memory-settings" / operation_id
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_path = backup_dir / "config.before"
    backup_path.write_bytes(original)
    os.chmod(backup_path, 0o600)

    try:
        _atomic_write(config_path, candidate, 0o600)
        if on_applied is not None:
            on_applied()
    except BaseException:
        if config_existed:
            _atomic_write_bytes(config_path, original, 0o600)
        elif config_path.exists():
            config_path.unlink()
        raise


def _new_config(workspace: Path) -> str:
    document = tomlkit.document()
    runtime = tomlkit.table()
    runtime["workspace"] = str(workspace)
    document["runtime"] = runtime
    document["agent"] = tomlkit.table()
    channels = tomlkit.table()
    chat = tomlkit.table()
    chat["enabled"] = True
    channels["chat"] = chat
    document["channels"] = channels
    app_server = tomlkit.table()
    app_server["enabled"] = True
    document["app_server"] = app_server
    return tomlkit.dumps(document)


def _atomic_write(path: Path, content: str, mode: int) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"), mode)


def _atomic_write_bytes(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.settings-",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message},
    )


__all__ = ["EmbeddingModelExists", "SettingsServer", "create_settings_app"]
