from __future__ import annotations

import inspect
import secrets
from collections.abc import Awaitable, Callable, Mapping

from agent.control.models import TurnRequest
from agent.control.scoped_turn import ScopedTurnHandle, ScopedTurnPort, TurnScopeLease
from agent.control.turn_scope import TurnExecutionScope
from agent.plugin_composition.model import ServiceKey


class PluginScopedTurns:
    """Expose Core-owned scoped Turn admission without runtime internals."""

    def __init__(
        self,
        runtime: object | None,
        session_creator: Callable[..., object] | None,
        session_reader: Callable[[str], object] | None = None,
        scope_acquirer: Callable[[], Awaitable[TurnScopeLease]] | None = None,
    ) -> None:
        self._runtime = runtime
        self._session_creator = session_creator
        self._session_reader = session_reader
        self._scope_acquirer = scope_acquirer

    @classmethod
    def candidate_validation(cls) -> PluginScopedTurns:
        """Keep candidate topology complete while denying child admission."""

        return cls(None, None, None)

    @property
    def formal(self) -> bool:
        return self._runtime is not None and self._session_creator is not None

    async def create_session(self, *, metadata: Mapping[str, object]) -> str:
        """Create one isolated programmatic Session with detached provenance."""

        runtime, creator = self._require_formal()
        _ = runtime
        key = "programmatic:" + secrets.token_hex(16)
        result = creator(key=key, metadata=dict(metadata))
        if inspect.isawaitable(result):
            await result
        return key

    async def ensure_session(
        self,
        key: str,
        *,
        metadata: Mapping[str, object],
    ) -> str:
        """Create one plugin-named Session once and preserve an existing identity."""

        runtime, creator = self._require_formal()
        _ = runtime
        reader = self._session_reader
        if reader is None:
            raise RuntimeError("scoped Turn 缺少 programmatic session reader")
        if not key or key.strip() != key:
            raise ValueError("scoped Turn session key 必须非空且无首尾空白")
        existing = reader(key)
        if inspect.isawaitable(existing):
            existing = await existing
        if existing is not None:
            return key
        result = creator(key=key, metadata=dict(metadata))
        if inspect.isawaitable(result):
            await result
        return key

    async def start(
        self,
        session_id: str,
        content: str,
        *,
        scope: TurnExecutionScope,
        channel: str = "programmatic",
        chat_id: str | None = None,
        sender: str | None = None,
        busy_session_id: str | None = None,
    ) -> ScopedTurnHandle:
        """Admit one Turn through the exact lease bound to this invocation."""

        runtime, _ = self._require_formal()
        if not session_id:
            raise ValueError("scoped Turn session_id 不能为空")
        if not isinstance(content, str):
            raise TypeError("scoped Turn content 必须是字符串")
        from agent.plugins.snapshot import get_current_runtime_lease

        owner = get_current_runtime_lease()
        acquired_here = False
        if owner is None or not owner.active:
            if self._scope_acquirer is None:
                raise RuntimeError("scoped Turn 缺少 exact RuntimeSnapshot lease")
            owner = await self._scope_acquirer()
            acquired_here = True
        try:
            port = ScopedTurnPort(runtime, owner, execution_scope=scope)
            return await port.start(
                TurnRequest(
                    session_id,
                    content,
                    {
                        "channel": channel,
                        "chatId": chat_id or session_id,
                        "sender": sender or scope.tool_source,
                        **(
                            {"busySessionId": busy_session_id}
                            if busy_session_id is not None
                            else {}
                        ),
                    },
                )
            )
        finally:
            if acquired_here:
                await owner.release()

    def _require_formal(self) -> tuple[object, Callable[..., object]]:
        if self._runtime is None or self._session_creator is None:
            raise RuntimeError("candidate 验证期禁止创建 scoped Turn")
        return self._runtime, self._session_creator


SCOPED_TURNS = ServiceKey[PluginScopedTurns]("core.scoped_turns")
