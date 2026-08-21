"""Akasha v3 feedback persistence and read-only inspection surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from agent.control.context import running_turn_id
from agent.lifecycle.composition import AFTER_REASONING_PREPROCESS_EVENT
from agent.lifecycle.types import AfterReasoningCtx
from agent.plugin_composition import (
    MEMORY_RUNTIME,
    MEMORY_TURN_RUNTIME,
    UI_SLOTS,
    Context,
    MemoryTurnRuntime,
    MobileUiDefinition,
    MobileUiNavigation,
    MobileUiRpcInvalidRequest,
    ServiceView,
)
from core.memory.plugin import ActiveRecallRecord

from .config import load_akasha_config
from .inspector import AkashaInspectorReader, mobile_summary

api_version = 3
name = "akasha"
version = "3.0.0"
desc = "提供 Akasha 反馈持久化、Inspector 与移动召回视图"
inject = (MEMORY_TURN_RUNTIME, UI_SLOTS)
workspace_roots = ("memory",)
dashboard_module = "dashboard.py"

_MOBILE_RECALL_SCHEMA = "akasha.recall-card.v1"
_MOBILE_RECALL_USER_PREVIEW_CHARS = 100
_MOBILE_RECALL_ASSISTANT_PREVIEW_CHARS = 50


class _AkashaMobileQuery:
    """Serve bounded mobile projections from one exact Root runtime."""

    def __init__(
        self,
        runtime: MemoryTurnRuntime,
        *,
        memory_root: Path,
        data_root: Path,
    ) -> None:
        self._runtime = runtime
        self._reader = AkashaInspectorReader(
            memory_root=memory_root,
            config=load_akasha_config(data_root / "config.local.toml"),
        )

    def __call__(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Serve one validated mobile UI query."""

        # 1. Resolve the memory that affected one assistant response.
        if method == "recall.current":
            return self._mobile_recall(
                payload,
                session_id=session_id,
                turn_id=turn_id,
            )

        # 2. Serve the Inspector list and on-demand detail.
        if method == "inspector.recent":
            if payload:
                raise MobileUiRpcInvalidRequest("Akasha inspector.recent 不接受参数")
            items, total = self._inspector().list_turns(page=1, page_size=30)
            return {
                "items": [
                    {
                        **item,
                        "query_preview": _clip(str(item["query_text"]), 180),
                    }
                    for item in items
                ],
                "total": total,
            }
        if method == "inspector.detail":
            if set(payload) != {"query_id"}:
                raise MobileUiRpcInvalidRequest("Akasha inspector.detail 需要 query_id")
            query_id = payload["query_id"]
            if not isinstance(query_id, str) or not query_id.strip():
                raise MobileUiRpcInvalidRequest(
                    "Akasha inspector.detail 的 query_id 必须是非空字符串"
                )
            item = self._inspector().get_turn(query_id.strip())
            if item is None:
                raise MobileUiRpcInvalidRequest("Akasha 检索记录不存在")
            return mobile_summary(item)
        raise MobileUiRpcInvalidRequest(f"Akasha mobile UI 方法无效: {method}")

    def _mobile_recall(
        self,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Resolve current or persisted assistant identity to prompt lanes."""

        # 1. Validate request identity before touching active or persisted state.
        if set(payload) - {"message_id"}:
            raise MobileUiRpcInvalidRequest("Akasha recall.current 参数无效")
        if session_id is None:
            raise MobileUiRpcInvalidRequest("Akasha recall.current 需要 session_id")
        message_id = payload.get("message_id")
        if message_id is not None and not isinstance(message_id, str):
            raise MobileUiRpcInvalidRequest(
                "Akasha recall.current 的 message_id 必须是字符串"
            )

        # 2. Synthetic active messages use only the narrow exact-Root port.
        if isinstance(message_id, str) and message_id.startswith("assistant:"):
            if turn_id is None or message_id != f"assistant:{turn_id}":
                return _empty_mobile_recall()
            pending = self._runtime.wait_active_recall(session_id, turn_id)
            if pending is None:
                return _empty_mobile_recall(pending=True)
            return {
                "schema": _MOBILE_RECALL_SCHEMA,
                "query_id": pending.query_id,
                "recall_capture_available": True,
                "left": _mobile_recall_records(pending.dense),
                "right": _mobile_recall_records(pending.completion),
                "tool_left": [],
                "tool_right": [],
            }

        # 3. Persisted messages read only Akasha's deterministic sidecars.
        item = (
            self._inspector().for_assistant_message(session_id, message_id)
            if isinstance(message_id, str)
            else self._inspector().latest_for_session(session_id)
        )
        if item is None:
            return _empty_mobile_recall()
        return {
            "schema": _MOBILE_RECALL_SCHEMA,
            "query_id": item["query_id"],
            "recall_capture_available": item["recall_capture_available"],
            "left": _mobile_recall_lane(cast(list[dict[str, object]], item["left"])),
            "right": _mobile_recall_lane(cast(list[dict[str, object]], item["right"])),
            "tool_left": _mobile_recall_lane(
                cast(list[dict[str, object]], item["tool_left"])
            ),
            "tool_right": _mobile_recall_lane(
                cast(list[dict[str, object]], item["tool_right"])
            ),
        }

    def _inspector(self) -> AkashaInspectorReader:
        """Reuse one sidecar snapshot reader for this exact Root binding."""

        return self._reader


async def apply(ctx: Context, config: object) -> None:
    """Register Akasha feedback and mobile UI as exact Root Effects."""

    # 1. Static activation already proves Akasha owns the selected memory runtime.
    _ = config
    runtime = ctx.require(MEMORY_TURN_RUNTIME)
    query = _AkashaMobileQuery(
        runtime,
        memory_root=ctx.workspace_root("memory"),
        data_root=ctx.data_root,
    )

    # 2. Feedback metadata is consumed before Core builds pending user rows.
    _ = await ctx.on(
        AFTER_REASONING_PREPROCESS_EVENT,
        lambda event: _persist_feedback(event, runtime),
    )

    # 3. Mobile handlers and assets live only with this Fiber activation.
    await ctx.require(UI_SLOTS).register_mobile(
        ctx,
        MobileUiDefinition(
            module="mobile_ui.js",
            stylesheet="mobile_ui.css",
            navigation=MobileUiNavigation(
                label="Akasha Inspector",
                description="查看每轮线索、激活与模式补全",
            ),
            slots=("turn.before_reasoning",),
        ),
        query=query,
    )


def is_active(services: ServiceView) -> bool:
    runtime = services.get(MEMORY_RUNTIME)
    return runtime is not None and runtime.name == "akasha"


def _persist_feedback(
    event: AfterReasoningCtx,
    runtime: MemoryTurnRuntime,
) -> None:
    """Move selected-engine feedback into the current pending user row."""

    # 1. Candidate turns preserve topology but have no formal pending user row.
    if not runtime.formal:
        return

    # 2. Formal turns consume and merge the selected engine's metadata once.
    metadata = runtime.take_user_metadata(running_turn_id.get())
    duplicated = set(event.persist_user_metadata) & set(metadata)
    if duplicated:
        fields = ", ".join(sorted(duplicated))
        raise RuntimeError(f"Akasha user metadata 字段重复: {fields}")
    event.persist_user_metadata.update(metadata)


def _clip(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "..."


def _empty_mobile_recall(*, pending: bool = False) -> dict[str, object]:
    return {
        "schema": _MOBILE_RECALL_SCHEMA,
        "query_id": None,
        "pending": pending,
        "recall_capture_available": False,
        "left": [],
        "right": [],
        "tool_left": [],
        "tool_right": [],
    }


def _mobile_recall_lane(
    value: list[dict[str, object]],
) -> list[dict[str, object]]:
    """把语义层已选出的整条 lane 投影成移动卡片字段。"""

    projected: list[dict[str, object]] = []
    for raw in value:
        item: dict[str, object] = {
            "user_preview": _clip(cast(str, raw["user_text"]), 100),
            "assistant_preview": _clip(cast(str, raw["assistant_preview"]), 50),
            "ts": cast(str, raw["ts"]),
        }
        score = raw.get("score")
        if score is not None:
            item["score"] = score
        projected.append(item)
    return projected


def _mobile_recall_records(
    records: tuple[ActiveRecallRecord, ...],
) -> list[dict[str, object]]:
    """Project frozen runtime records through the bounded card shape."""

    return _mobile_recall_lane(
        [
            {
                "user_text": record.user_text,
                "assistant_preview": record.assistant_preview,
                "ts": record.started_at,
                "score": record.score,
            }
            for record in records
        ]
    )
