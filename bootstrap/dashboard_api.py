from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
import logging
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from bootstrap.cleanup import run_cleanup_steps

from agent.memory import MemoryStore
from core.memory.optimizer import MemoryOptimizerBusy
from session.store import (
    InteractionDeleteRequiredError,
    SessionAdmissionConflictError,
    SessionCompactionPrepareConflictError,
    SessionStore,
)

logger = logging.getLogger(__name__)

_DASHBOARD_ACCESS_PREFIXES = ("/api/dashboard", "/assets")


def _is_dashboard_access_record(record: logging.LogRecord) -> bool:
    args = record.args
    if not isinstance(args, tuple) or len(args) < 3:
        return False
    path = args[2]
    if not isinstance(path, str):
        return False
    return path == "/" or any(
        path.startswith(prefix) for prefix in _DASHBOARD_ACCESS_PREFIXES
    )


# dashboard 会频繁轮询，访问日志只在 debug 模式保留。
class _DashboardAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not _is_dashboard_access_record(record):
            return True
        debug_enabled = logging.getLogger().isEnabledFor(
            logging.DEBUG
        ) or logging.getLogger("uvicorn.access").isEnabledFor(logging.DEBUG)
        if not debug_enabled:
            return False
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"
        return True


def _install_dashboard_access_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if any(
        isinstance(filter_, _DashboardAccessLogFilter)
        for filter_ in access_logger.filters
    ):
        return
    access_logger.addFilter(_DashboardAccessLogFilter())


class SessionUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] | None = None
    last_user_at: str | None = None
    last_proactive_at: str | None = None


class SessionBatchDeletePayload(BaseModel):
    keys: list[str]
    cascade: bool = True


class MessageUpdatePayload(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_chain: Any | None = None
    extra: dict[str, Any] | None = None
    ts: str | None = None


class MessageBatchDeletePayload(BaseModel):
    ids: list[str]


class ManualMemoryOptimizer(Protocol):
    @property
    def is_running(self) -> bool: ...

    async def optimize(self) -> None: ...


def _interaction_delete_detail(
    exc: InteractionDeleteRequiredError,
) -> dict[str, str]:
    return {
        "code": "interaction_delete_required",
        "message_id": exc.message_id,
        "control_turn_id": exc.control_turn_id,
    }


def _session_delete_detail(exc: SessionAdmissionConflictError) -> dict[str, str]:
    detail = {
        "code": "session_busy",
        "session_key": exc.session_key,
    }
    if exc.audit_id is not None:
        detail["audit_id"] = exc.audit_id
    return detail


def _compaction_prepare_detail(
    exc: SessionCompactionPrepareConflictError,
) -> dict[str, str]:
    detail = {
        "code": "session_compaction_pending",
        "session_key": exc.session_key,
        "source_ref": exc.source_ref,
    }
    if exc.audit_id is not None:
        detail["audit_id"] = exc.audit_id
    return detail


def _compaction_dashboard_dict(value: Any) -> dict[str, Any]:
    """序列化一个 ledger generation 供 dashboard 只读展示。"""

    return {
        "generation": value.generation,
        "parent_generation": value.parent_generation,
        "created_at": value.created_at,
        "trigger": value.trigger,
        "summary": value.summary,
        "source_from_seq": value.source_from_seq,
        "consolidated_through_seq": value.consolidated_through_seq,
        "source_message_count": len(value.source_message_ids),
        "source_plan_digest": value.source_plan_digest,
        "model": value.model,
        "model_runtime_id": value.model_runtime_id,
        "context_window": value.context_window,
        "threshold_tokens": value.threshold_tokens,
        "hard_input_tokens": value.hard_input_tokens,
        "keep_recent_tokens": value.keep_recent_tokens,
        "tokens_before": value.tokens_before,
        "tokens_after": value.tokens_after,
        "summary_usage": value.summary_usage,
        "invalidated_at": value.invalidated_at,
        "invalidated_reason": value.invalidated_reason,
    }


async def _close_dashboard_value(value: object) -> None:
    """关闭 dashboard 资源，并等待异步 close 完成。"""
    close = getattr(value, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


def create_dashboard_app(
    workspace: Path,
    *,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> FastAPI:
    workspace.mkdir(parents=True, exist_ok=True)
    store = SessionStore(workspace / "sessions.db")
    optimizer_task: asyncio.Task[None] | None = None
    optimizer_last_status = "idle"
    optimizer_last_error: str | None = None
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "static" / "dashboard"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await run_cleanup_steps(
                (
                    "dashboard.session_store.close",
                    lambda: _close_dashboard_value(store),
                ),
            )

    app = FastAPI(title="Akashic Dashboard API", lifespan=lifespan)
    app.state.memory_store = memory_store or MemoryStore(workspace)
    # Vite 构建产物被 gitignore，新 clone 或 CI 环境可能没有该目录。
    # 预先创建目录并在挂载时关闭目录检查，避免 app 创建依赖构建是否执行；
    # dashboard_index() 会在入口文件缺失时报告错误。
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="dashboard-assets",
    )
    # Vite 会在 /assets 下生成带内容哈希的资源 URL，因此直接原样提供 index.html；
    # 不需要手动处理缓存失效。
    @app.get("/")
    def dashboard_index() -> Response:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            return Response(
                content="Dashboard 前端尚未构建，请先运行 `npm run build`。",
                media_type="text/plain; charset=utf-8",
                status_code=503,
            )
        html = index_file.read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html")

    @app.get("/api/dashboard/sessions")
    def list_sessions(
        q: str = "",
        channel: str = "",
        updated_from: str = "",
        updated_to: str = "",
        has_proactive: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = store.list_sessions_for_dashboard(
            q=q,
            channel=channel,
            updated_from=updated_from,
            updated_to=updated_to,
            has_proactive=has_proactive,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        briefs = store.list_compaction_briefs([item["key"] for item in items])
        for item in items:
            brief = briefs.get(item["key"])
            if brief is not None:
                raw_preview = brief.pop("summary_preview")
                summary_preview = " ".join(str(raw_preview or "").split())
                item["compaction"] = {
                    **brief,
                    "summary_preview": summary_preview[:120],
                }
            else:
                item["compaction"] = None
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/sessions/{session_key:path}/compaction")
    def get_session_compaction(session_key: str) -> dict[str, Any]:
        """返回一个 session 的 ledger head、当前摘要与全部 generation 历史。"""

        if not store.session_exists(session_key):
            raise HTTPException(status_code=404, detail="session 不存在")
        head = store.get_compaction_head(session_key)
        try:
            active = store.get_active_compaction(session_key)
        except ValueError:
            # cursor 指向已失效 generation 时，只读视图保持可展示。
            active = None
        history = store.list_compactions(session_key)
        return {
            "head": {
                "parent_generation": head.parent_generation,
                "next_generation": head.next_generation,
            },
            "active": (
                _compaction_dashboard_dict(active) if active is not None else None
            ),
            "history": [_compaction_dashboard_dict(value) for value in history],
        }

    @app.get("/api/dashboard/sessions/{session_key:path}/messages")
    def list_session_messages(
        session_key: str,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        if not store.session_exists(session_key):
            raise HTTPException(status_code=404, detail="session 不存在")
        items, total = store.list_messages_for_dashboard(
            session_key=session_key,
            q=q,
            role=role,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    async def _run_memory_optimizer() -> None:
        nonlocal optimizer_last_error, optimizer_last_status
        assert manual_memory_optimizer is not None
        optimizer_last_status = "running"
        optimizer_last_error = None
        try:
            await manual_memory_optimizer.optimize()
            optimizer_last_status = "succeeded"
        except MemoryOptimizerBusy:
            optimizer_last_status = "skipped"
            logger.info("manual memory optimizer skipped because it is already running")
        except asyncio.CancelledError:
            optimizer_last_status = "failed"
            optimizer_last_error = "memory optimizer 已取消"
            raise
        except Exception as exc:
            optimizer_last_status = "failed"
            optimizer_last_error = str(exc)
            logger.exception("manual memory optimizer failed: %s", exc)

    @app.get("/api/dashboard/memory/optimizer")
    async def get_memory_optimizer_status() -> dict[str, Any]:
        running = bool(
            manual_memory_optimizer is not None
            and (
                (optimizer_task is not None and not optimizer_task.done())
                or manual_memory_optimizer.is_running
            )
        )
        return {
            "enabled": manual_memory_optimizer is not None,
            "running": running,
            "last_status": "running" if running else optimizer_last_status,
            "last_error": optimizer_last_error,
        }

    @app.post("/api/dashboard/memory/optimize", status_code=202)
    async def trigger_memory_optimizer() -> dict[str, Any]:
        nonlocal optimizer_last_error, optimizer_last_status, optimizer_task
        if manual_memory_optimizer is None:
            raise HTTPException(status_code=503, detail="memory optimizer 未启用")
        if (
            optimizer_task is not None and not optimizer_task.done()
        ) or manual_memory_optimizer.is_running:
            raise HTTPException(status_code=409, detail="memory optimizer 正在运行")
        logger.info("Manual memory optimizer triggered via dashboard")
        optimizer_last_status = "running"
        optimizer_last_error = None
        optimizer_task = asyncio.create_task(
            _run_memory_optimizer(),
            name="manual_memory_optimizer",
        )
        return {"status": "started", "message": "Memory optimizer started"}

    @app.post("/api/dashboard/sessions/batch-delete")
    def delete_sessions_batch(payload: SessionBatchDeletePayload) -> dict[str, Any]:
        try:
            deletion = store.delete_sessions_batch_with_audit(
                payload.keys,
                cascade=payload.cascade,
                action_source="dashboard.session_batch_delete",
            )
        except SessionAdmissionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(exc),
            ) from exc
        except SessionCompactionPrepareConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "session_delete_rejected",
                    "message": str(exc),
                    "audit_id": getattr(exc, "audit_id", None),
                },
            ) from exc
        return {
            "deleted_count": deletion.deleted_count,
            "audit_id": deletion.audit_id,
            "backup_path": deletion.backup_path,
            "action_source": deletion.action_source,
            "result": deletion.result,
        }

    @app.get("/api/dashboard/sessions/{session_key:path}")
    def get_session(session_key: str) -> dict[str, Any]:
        meta = store.get_session_meta(session_key)
        if meta is None:
            raise HTTPException(status_code=404, detail="session 不存在")
        meta["message_count"] = store.count_messages(session_key)
        return meta

    @app.patch("/api/dashboard/sessions/{session_key:path}")
    def update_session(
        session_key: str,
        payload: SessionUpdatePayload,
    ) -> dict[str, Any]:
        meta = store.update_session(
            session_key,
            metadata=payload.metadata,
            last_user_at=payload.last_user_at,
            last_proactive_at=payload.last_proactive_at,
        )
        if meta is None:
            raise HTTPException(status_code=404, detail="session 不存在")
        meta["message_count"] = store.count_messages(session_key)
        return meta

    @app.delete("/api/dashboard/sessions/{session_key:path}")
    def delete_session(
        session_key: str,
        cascade: bool = Query(default=True),
    ) -> dict[str, Any]:
        try:
            deletion = store.delete_session_with_audit(
                session_key,
                cascade=cascade,
                action_source="dashboard.session_delete",
            )
        except SessionAdmissionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(exc),
            ) from exc
        except SessionCompactionPrepareConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "session_delete_rejected",
                    "message": str(exc),
                    "audit_id": getattr(exc, "audit_id", None),
                },
            ) from exc
        if deletion.result != "committed":
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "session_not_found",
                    "session_key": session_key,
                    "audit_id": deletion.audit_id,
                },
            )
        return {
            "deleted": True,
            "session_key": session_key,
            "audit_id": deletion.audit_id,
            "backup_path": deletion.backup_path,
            "action_source": deletion.action_source,
            "result": deletion.result,
        }

    @app.get("/api/dashboard/messages")
    def list_messages(
        session_key: str | None = None,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = store.list_messages_for_dashboard(
            session_key=session_key,
            q=q,
            role=role,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/messages/{message_id:path}")
    def get_message(message_id: str) -> dict[str, Any]:
        message = store.get_message(message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="message 不存在")
        return message

    @app.patch("/api/dashboard/messages/{message_id:path}")
    def update_message(
        message_id: str,
        payload: MessageUpdatePayload,
    ) -> dict[str, Any]:
        try:
            message = store.update_message(
                message_id,
                role=payload.role,
                content=payload.content,
                tool_chain=payload.tool_chain,
                extra=payload.extra,
                ts=payload.ts,
                action_source="dashboard.message_edit",
            )
        except SessionAdmissionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(exc),
            ) from exc
        except SessionCompactionPrepareConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(exc),
            ) from exc
        if message is None:
            raise HTTPException(status_code=404, detail="message 不存在")
        return message

    @app.delete("/api/dashboard/messages/{message_id:path}")
    def delete_message(message_id: str) -> dict[str, Any]:
        try:
            deleted = store.delete_message(
                message_id,
                action_source="dashboard.message_delete",
            )
        except InteractionDeleteRequiredError as exc:
            raise HTTPException(
                status_code=409,
                detail=_interaction_delete_detail(exc),
            ) from exc
        except SessionAdmissionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(exc),
            ) from exc
        except SessionCompactionPrepareConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(exc),
            ) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="message 不存在")
        return {"deleted": True, "id": message_id}

    @app.post("/api/dashboard/messages/batch-delete")
    def delete_messages_batch(payload: MessageBatchDeletePayload) -> dict[str, Any]:
        try:
            deleted_count = store.delete_messages_batch(
                payload.ids,
                action_source="dashboard.message_batch_delete",
            )
        except InteractionDeleteRequiredError as exc:
            raise HTTPException(
                status_code=409,
                detail=_interaction_delete_detail(exc),
            ) from exc
        except SessionAdmissionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(exc),
            ) from exc
        except SessionCompactionPrepareConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(exc),
            ) from exc
        return {"deleted_count": deleted_count}

    if plugin_manager is not None:
        from agent.plugins.dashboard_host import (
            PluginDashboardHost,
            SnapshotDashboardMiddleware,
        )

        dashboard_host = PluginDashboardHost(
            core_routes=tuple(app.routes),
        )
        snapshot = plugin_manager.current_snapshot
        if snapshot is not None:
            dashboard_host.prepare_initial_snapshot(snapshot)
        plugin_manager.bind_dashboard_preparer(
            dashboard_host.prepare_snapshot,
            validation_releaser=dashboard_host.release_validation,
        )
        app.add_middleware(
            SnapshotDashboardMiddleware,
            snapshot_store=plugin_manager.snapshot_store,
        )

    return app


def run_dashboard_api(
    *,
    workspace: Path,
    host: str = "0.0.0.0",
    port: int = 2236,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
) -> None:
    server = uvicorn.Server(
        _build_dashboard_uvicorn_config(
            workspace=workspace,
            host=host,
            port=port,
            uds=None,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_store=memory_store,
        )
    )
    server.run()


def _build_dashboard_uvicorn_config(
    *,
    workspace: Path,
    host: str | None,
    port: int | None,
    uds: str | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> uvicorn.Config:
    config = uvicorn.Config(
        create_dashboard_app(
            workspace,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_store=memory_store,
            plugin_manager=plugin_manager,
        ),
        host=host or "127.0.0.1",
        port=port or 2236,
        uds=uds,
        log_level="info",
    )
    _install_dashboard_access_log_filter()
    return config


def build_dashboard_server(
    *,
    workspace: Path,
    host: str | None = None,
    port: int | None = None,
    uds: str | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> uvicorn.Server:
    config = _build_dashboard_uvicorn_config(
        workspace=workspace,
        host=host,
        port=port,
        uds=uds,
        manual_memory_optimizer=manual_memory_optimizer,
        memory_store=memory_store,
        plugin_manager=plugin_manager,
    )
    return uvicorn.Server(config)
