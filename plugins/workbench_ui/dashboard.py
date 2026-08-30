"""Expose the session workbench through plugin-owned Dashboard routes."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from agent.plugin_composition import DashboardContext
from session.store import (
    InteractionDeleteRequiredError,
    SessionAdmissionConflictError,
    SessionCompactionPrepareConflictError,
    SessionStore,
)


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


def _interaction_delete_detail(
    error: InteractionDeleteRequiredError,
) -> dict[str, str]:
    return {
        "code": "interaction_delete_required",
        "message_id": error.message_id,
        "control_turn_id": error.control_turn_id,
    }


def _session_delete_detail(
    error: SessionAdmissionConflictError,
) -> dict[str, str]:
    detail = {
        "code": "session_busy",
        "session_key": error.session_key,
    }
    if error.audit_id is not None:
        detail["audit_id"] = error.audit_id
    return detail


def _compaction_prepare_detail(
    error: SessionCompactionPrepareConflictError,
) -> dict[str, str]:
    detail = {
        "code": "session_compaction_pending",
        "session_key": error.session_key,
        "source_ref": error.source_ref,
    }
    if error.audit_id is not None:
        detail["audit_id"] = error.audit_id
    return detail


def _compaction_dict(value: Any) -> dict[str, Any]:
    """Serialize one compaction ledger generation for the workbench."""

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


def register(app: FastAPI, context: DashboardContext) -> SessionStore:
    """Register the workbench API and return its generation-owned store."""

    store = SessionStore(context.workspace_file("sessions.db"))

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
            if brief is None:
                item["compaction"] = None
                continue
            raw_preview = brief.pop("summary_preview")
            summary_preview = " ".join(str(raw_preview or "").split())
            item["compaction"] = {
                **brief,
                "summary_preview": summary_preview[:120],
            }
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/sessions/{session_key:path}/compaction")
    def get_session_compaction(session_key: str) -> dict[str, Any]:
        """Return the active compaction and immutable generation history."""

        if not store.session_exists(session_key):
            raise HTTPException(status_code=404, detail="session 不存在")
        head = store.get_compaction_head(session_key)
        try:
            active = store.get_active_compaction(session_key)
        except ValueError:
            # A stale cursor must not hide the remaining read-only history.
            active = None
        history = store.list_compactions(session_key)
        return {
            "head": {
                "parent_generation": head.parent_generation,
                "next_generation": head.next_generation,
            },
            "active": _compaction_dict(active) if active is not None else None,
            "history": [_compaction_dict(value) for value in history],
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

    @app.post("/api/dashboard/sessions/batch-delete")
    def delete_sessions_batch(payload: SessionBatchDeletePayload) -> dict[str, Any]:
        try:
            deletion = store.delete_sessions_batch_with_audit(
                payload.keys,
                cascade=payload.cascade,
                action_source="dashboard.session_batch_delete",
            )
        except SessionAdmissionConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(error),
            ) from error
        except SessionCompactionPrepareConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "session_delete_rejected",
                    "message": str(error),
                    "audit_id": getattr(error, "audit_id", None),
                },
            ) from error
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
        except SessionAdmissionConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(error),
            ) from error
        except SessionCompactionPrepareConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "session_delete_rejected",
                    "message": str(error),
                    "audit_id": getattr(error, "audit_id", None),
                },
            ) from error
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
        except SessionAdmissionConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(error),
            ) from error
        except SessionCompactionPrepareConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(error),
            ) from error
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
        except InteractionDeleteRequiredError as error:
            raise HTTPException(
                status_code=409,
                detail=_interaction_delete_detail(error),
            ) from error
        except SessionAdmissionConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(error),
            ) from error
        except SessionCompactionPrepareConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(error),
            ) from error
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
        except InteractionDeleteRequiredError as error:
            raise HTTPException(
                status_code=409,
                detail=_interaction_delete_detail(error),
            ) from error
        except SessionAdmissionConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_session_delete_detail(error),
            ) from error
        except SessionCompactionPrepareConflictError as error:
            raise HTTPException(
                status_code=409,
                detail=_compaction_prepare_detail(error),
            ) from error
        return {"deleted_count": deleted_count}

    return store
