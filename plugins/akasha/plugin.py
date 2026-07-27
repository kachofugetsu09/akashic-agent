"""Plugin discovery and read-only inspection surfaces for Akasha V2."""

from agent.plugins import (
    MobileUiContribution,
    MobileUiNavigation,
    Plugin,
)
from agent.plugins.mobile_ui import MobileUiRpcInvalidRequest

from .inspector import AkashaInspectorReader, mobile_summary
from .memory_plugin import MemoryPlugin


class AkashaPlugin(Plugin):
    """Register V2 memory inspection without exposing graph mutation."""

    name = "akasha"

    def __init__(self) -> None:
        self._reader: AkashaInspectorReader | None = None

    @classmethod
    def dashboard_module(cls) -> str:
        return "dashboard.py"

    @classmethod
    def mobile_ui(cls) -> MobileUiContribution:
        return MobileUiContribution(
            module="mobile_ui.js",
            stylesheet="mobile_ui.css",
            navigation=MobileUiNavigation(
                label="Akasha Inspector",
                description="查看每轮线索、激活与模式补全",
            ),
            slots=("turn.before_reasoning",),
        )

    def mobile_ui_query(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Serve versioned read-only recall and Inspector projections."""

        # 1. Resolve the memory that affected one assistant response.
        if method == "recall.current":
            return self._mobile_recall(
                payload,
                session_id=session_id,
                turn_id=turn_id,
            )

        # 2. Serve the mobile Inspector list and on-demand detail.
        if method == "inspector.recent":
            if payload:
                raise MobileUiRpcInvalidRequest(
                    "Akasha inspector.recent 不接受参数"
                )
            items, total = self._inspector().list_turns(
                page=1,
                page_size=30,
            )
            return {
                "items": [
                    {
                        **item,
                        "query_preview": _clip(
                            str(item["query_text"]),
                            180,
                        ),
                    }
                    for item in items
                ],
                "total": total,
            }
        if method == "inspector.detail":
            if set(payload) != {"query_id"}:
                raise MobileUiRpcInvalidRequest(
                    "Akasha inspector.detail 需要 query_id"
                )
            query_id = payload["query_id"]
            if not isinstance(query_id, str) or not query_id.strip():
                raise MobileUiRpcInvalidRequest(
                    "Akasha inspector.detail 的 query_id 必须是非空字符串"
                )
            item = self._inspector().get_turn(query_id.strip())
            if item is None:
                raise MobileUiRpcInvalidRequest(
                    "Akasha 检索记录不存在"
                )
            return mobile_summary(item)
        raise MobileUiRpcInvalidRequest(
            f"Akasha mobile UI 方法无效: {method}"
        )

    def _mobile_recall(
        self,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Resolve current or persisted assistant identity to prompt lanes."""

        # 1. Validate the RPC identity before touching either sidecar.
        if set(payload) - {"message_id"}:
            raise MobileUiRpcInvalidRequest(
                "Akasha recall.current 参数无效"
            )
        if session_id is None:
            raise MobileUiRpcInvalidRequest(
                "Akasha recall.current 需要 session_id"
            )
        message_id = payload.get("message_id")
        if message_id is not None and not isinstance(message_id, str):
            raise MobileUiRpcInvalidRequest(
                "Akasha recall.current 的 message_id 必须是字符串"
            )

        # 2. Synthetic active messages use the latest committed session turn.
        if isinstance(message_id, str) and message_id.startswith("assistant:"):
            if turn_id is None or message_id != f"assistant:{turn_id}":
                return {"left": [], "right": []}
            item = self._inspector().latest_for_session(session_id)
        elif isinstance(message_id, str):
            item = self._inspector().for_assistant_message(
                session_id,
                message_id,
            )
        else:
            item = self._inspector().latest_for_session(session_id)
        if item is None:
            return {"left": [], "right": []}
        return {
            "query_id": item["query_id"],
            "left": item["left"],
            "right": item["right"],
        }

    def _inspector(self) -> AkashaInspectorReader:
        """Reuse the vector snapshot across mobile inspection requests."""

        if self._reader is None:
            workspace = self.context.workspace
            if workspace is None:
                raise RuntimeError("Akasha Inspector workspace 不存在")
            self._reader = AkashaInspectorReader(workspace)
        return self._reader


__all__ = ["AkashaPlugin", "MemoryPlugin"]


def _clip(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "..."
