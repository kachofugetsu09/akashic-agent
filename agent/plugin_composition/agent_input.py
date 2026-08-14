from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import cast

from agent.plugin_composition.context import CompositionRoot, Context
from agent.plugin_composition.model import CompositionError, FiberState, ServiceKey

AgentSessionCreator = Callable[
    [str, Mapping[str, object]],
    Awaitable[str],
]
AgentInputSubmitter = Callable[
    [str, str, str, Mapping[str, object]],
    Awaitable[str],
]

_MAX_SESSION_ID_LENGTH = 512
_MAX_TURN_ID_LENGTH = 128
_MAX_INPUT_LENGTH = 1_048_576


@dataclass(frozen=True, slots=True)
class AgentSession:
    """Identify one durable Session created through the Agent Input boundary."""

    id: str


@dataclass(frozen=True, slots=True)
class AgentInputReceipt:
    """Report that one ordinary input was admitted as a new Turn."""

    session_id: str
    turn_id: str


AGENT_INPUT = ServiceKey["AgentInputService"]("core.agent_input")


class AgentInputService:
    """Admit plugin-owned ordinary inputs without exposing Core runtime owners."""

    def __init__(
        self,
        root: CompositionRoot,
        *,
        create_session: AgentSessionCreator,
        submit: AgentInputSubmitter,
    ) -> None:
        self._root = root
        self._create_session = create_session
        self._submit = submit

    async def create_session(
        self,
        ctx: Context,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> AgentSession:
        """Create one persistent Session and return its Core-owned identity."""

        plugin_id = self._require_active_owner(ctx)
        stored_metadata = _copy_json_object(metadata, "Agent Input session metadata")
        session_id = await self._create_session(plugin_id, stored_metadata)
        return AgentSession(
            _require_identity(session_id, "Session", _MAX_SESSION_ID_LENGTH)
        )

    async def submit(
        self,
        ctx: Context,
        session_id: str,
        content: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> AgentInputReceipt:
        """Admit one ordinary Turn and return after Core assigns its identity."""

        plugin_id = self._require_active_owner(ctx)
        checked_session_id = _require_identity(
            session_id,
            "Session",
            _MAX_SESSION_ID_LENGTH,
        )
        if not isinstance(content, str):
            raise TypeError("Agent Input content 必须是字符串")
        if not content or len(content) > _MAX_INPUT_LENGTH:
            raise ValueError(
                f"Agent Input content 长度必须在 1..{_MAX_INPUT_LENGTH} 之间"
            )
        turn_metadata = _copy_json_object(metadata, "Agent Input turn metadata")
        turn_id = await self._submit(
            plugin_id,
            checked_session_id,
            content,
            turn_metadata,
        )
        return AgentInputReceipt(
            session_id=checked_session_id,
            turn_id=_require_identity(turn_id, "Turn", _MAX_TURN_ID_LENGTH),
        )

    def _require_active_owner(self, ctx: Context) -> str:
        """Require the calling Context to belong to one active Fiber in this Root."""

        fiber = ctx.fiber
        if fiber.root is not self._root:
            raise CompositionError(
                "FOREIGN_AGENT_INPUT_CONTEXT",
                "Agent Input 只能由取得该 Service 的 Root Context 调用",
            )
        if fiber.state is not FiberState.ACTIVE:
            raise CompositionError(
                "INACTIVE_AGENT_INPUT_CONTEXT",
                f"{fiber.name} 在 {fiber.state.value} 状态不能提交 Agent Input",
            )
        return ctx.runtime.plugin_id


def _copy_json_object(
    value: Mapping[str, object] | None,
    field_name: str,
) -> dict[str, object]:
    """Validate and detach a plugin-supplied JSON object at the public boundary."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} 必须是字符串键 JSON object")
    raw = dict(value)
    _require_json_value(raw, field_name)
    try:
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是 lossless JSON object") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{field_name} JSON round-trip 未得到 object")
    return cast(dict[str, object], decoded)


def _require_json_value(value: object, field_name: str) -> None:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, child in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} 的所有 object key 必须是字符串")
            _require_json_value(child, field_name)
        return
    if isinstance(value, list):
        sequence = cast(list[object], value)
        for child in sequence:
            _require_json_value(child, field_name)
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{field_name} 必须是 lossless JSON object")


def _require_identity(value: object, kind: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Agent Input {kind} identity 必须是字符串")
    if not value or len(value) > max_length:
        raise ValueError(f"Agent Input {kind} identity 长度必须在 1..{max_length} 之间")
    return value
