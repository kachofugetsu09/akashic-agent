"""Request-scoped views over the capabilities consumed by the client plugin.

The channel adapter may outlive several runtime snapshots.  These small views
retain only the host scope opener; every operation resolves its provider from
the exact scope active for that operation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from typing import Any, cast

from agent.plugin_composition.commands import COMMANDS
from agent.plugin_composition.message_view import MessageDisplayReader

from .capabilities import MESSAGE_DISPLAY, MOBILE_UI, WEB_UI
from .services import MobileUiProvider, WebUiProvider


RequestScopeOpener = Callable[
    [], AbstractAsyncContextManager[Any]
]

_ACTIVE_SCOPE: ContextVar[Any | None] = ContextVar(
    "akashic_clients_active_request_scope",
    default=None,
)


@asynccontextmanager
async def open_request_scope(
    opener: RequestScopeOpener,
) -> AsyncIterator[Any]:
    """Open one exact host scope and expose it to nested synchronous readers."""

    async with opener() as scope:
        token = _ACTIVE_SCOPE.set(scope)
        try:
            yield scope
        finally:
            _ACTIVE_SCOPE.reset(token)


def active_scope() -> Any | None:
    """Return the exact scope owned by the current request task, if any."""

    return _ACTIVE_SCOPE.get()


def _require_active_scope(capability: str) -> Any:
    scope = _ACTIVE_SCOPE.get()
    if scope is None:
        raise RuntimeError(f"akashic {capability} 必须在 request scope 内读取")
    return scope


class ScopedMessageDisplay:
    """Resolve message presentation from the current exact request scope."""

    def __init__(self, opener: RequestScopeOpener) -> None:
        self._opener = opener

    async def __call__(self, page: Any, *, display_only: bool) -> list[dict[str, object]]:
        scope = active_scope()
        if scope is not None:
            reader = cast(MessageDisplayReader, scope.require(MESSAGE_DISPLAY))
            return await reader(page, display_only=display_only)
        async with open_request_scope(self._opener) as scope:
            reader = cast(MessageDisplayReader, scope.require(MESSAGE_DISPLAY))
            return await reader(page, display_only=display_only)


class ScopedMobileUiProvider:
    """Mobile UI projection that never retains a generation provider."""

    def __init__(self, opener: RequestScopeOpener) -> None:
        self._opener = opener

    def _sync_provider(self) -> MobileUiProvider:
        return cast(
            MobileUiProvider,
            _require_active_scope("mobile UI").require(MOBILE_UI),
        )

    def catalog(self) -> dict[str, object]:
        return self._sync_provider().catalog()

    def asset(
        self,
        plugin_id: str,
        plugin_revision: str,
        kind: str,
        sha256: str,
    ) -> dict[str, object]:
        return self._sync_provider().asset(plugin_id, plugin_revision, kind, sha256)

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        scope = active_scope()
        if scope is not None:
            provider = cast(MobileUiProvider, scope.require(MOBILE_UI))
            return await provider.query(
                plugin_id,
                plugin_revision,
                method,
                payload,
                session_id=session_id,
                turn_id=turn_id,
            )
        async with open_request_scope(self._opener) as scope:
            provider = cast(MobileUiProvider, scope.require(MOBILE_UI))
            return await provider.query(
                plugin_id,
                plugin_revision,
                method,
                payload,
                session_id=session_id,
                turn_id=turn_id,
            )


class ScopedWebUiProvider:
    """Web UI projection resolved separately for each HTTP operation."""

    def __init__(self, opener: RequestScopeOpener) -> None:
        self._opener = opener

    async def bootstrap(self) -> bytes:
        scope = active_scope()
        if scope is not None:
            return await cast(WebUiProvider, scope.require(WEB_UI)).bootstrap()
        async with open_request_scope(self._opener) as scope:
            return await cast(WebUiProvider, scope.require(WEB_UI)).bootstrap()

    async def state(self) -> dict[str, str]:
        scope = active_scope()
        if scope is not None:
            return await cast(WebUiProvider, scope.require(WEB_UI)).state()
        async with open_request_scope(self._opener) as scope:
            return await cast(WebUiProvider, scope.require(WEB_UI)).state()


class ScopedCommandCatalog:
    """Build the command projection only while a command request is scoped."""

    def __call__(self) -> tuple[tuple[str, str], ...]:
        commands = _require_active_scope("command catalog").require(COMMANDS)
        return tuple(
            (descriptor.name, descriptor.description)
            for descriptor in commands.freeze().descriptors
        )


__all__ = [
    "RequestScopeOpener",
    "ScopedCommandCatalog",
    "ScopedMessageDisplay",
    "ScopedMobileUiProvider",
    "ScopedWebUiProvider",
    "active_scope",
    "open_request_scope",
]
