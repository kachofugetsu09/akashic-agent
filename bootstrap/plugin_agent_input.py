from __future__ import annotations

from collections.abc import Mapping

from agent.control.service import ControlService

_PLUGIN_INPUT_PREFIX = "_pluginInput"
_PLUGIN_INPUT_OWNER = "_pluginInputPluginId"


class ControlAgentInput:
    """Project the plugin Agent Input seam onto the existing control owner."""

    def __init__(self, control: ControlService) -> None:
        self._control = control

    async def create_session(
        self,
        plugin_id: str,
        metadata: Mapping[str, object],
    ) -> str:
        """Create one stable Session and preserve the originating plugin identity."""

        stored_metadata = _stamp_owner(plugin_id, metadata)
        record = self._control.start_thread(stored_metadata, runtime="stable")
        session_id = record.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("ControlService 创建 Session 后没有返回 identity")
        return session_id

    async def submit(
        self,
        plugin_id: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, object],
    ) -> str:
        """Start one detached ordinary Turn and return after admission."""

        turn_metadata = {"inboundMetadata": _stamp_owner(plugin_id, metadata)}
        handle = await self._control.start_turn(
            session_id,
            content,
            turn_metadata,
            runtime="stable",
            attached=False,
        )
        return handle.id


def _stamp_owner(
    plugin_id: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    reserved = sorted(key for key in metadata if key.startswith(_PLUGIN_INPUT_PREFIX))
    if reserved:
        raise ValueError(
            "Agent Input metadata 包含 Core 保留字段: " + ", ".join(reserved)
        )
    return {**metadata, _PLUGIN_INPUT_OWNER: plugin_id}
