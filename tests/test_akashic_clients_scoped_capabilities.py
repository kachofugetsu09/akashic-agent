from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agent.plugin_composition.requests import RequestContext
from plugins.akashic_clients.capabilities import MESSAGE_DISPLAY, MOBILE_UI, WEB_UI
from plugins.akashic_clients.scoped_capabilities import (
    ScopedMessageDisplay,
    ScopedMobileUiProvider,
    ScopedWebUiProvider,
    open_request_scope,
)


class _Scope:
    def __init__(self) -> None:
        self.version = 1
        self.entered = 0
        self.exited = 0

    @asynccontextmanager
    async def __call__(self):
        self.entered += 1
        version = self.version

        async def display(_page: object, *, display_only: bool) -> list[dict[str, object]]:
            return [{"version": version, "display_only": display_only}]

        class Ui:
            def catalog(self) -> dict[str, object]:
                return {"version": version}

            def asset(
                self,
                _plugin_id: str,
                _plugin_revision: str,
                _kind: str,
                _sha256: str,
            ) -> dict[str, object]:
                return {"version": version}

            async def query(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return {"version": version}

        class Web:
            async def bootstrap(self) -> bytes:
                return str(version).encode()

            async def state(self) -> dict[str, str]:
                return {"version": str(version)}

        values = {
            MESSAGE_DISPLAY: display,
            MOBILE_UI: Ui(),
            WEB_UI: Web(),
        }
        try:
            yield RequestContext(
                "akashic.clients",
                Path("/tmp/akashic-client-plugin"),
                Path("/tmp/akashic-client-data"),
                False,
                _resolve=values.__getitem__,
            )
        finally:
            self.exited += 1


@pytest.mark.asyncio
async def test_scoped_projections_resolve_each_operation() -> None:
    scope = _Scope()
    display = ScopedMessageDisplay(scope)
    mobile = ScopedMobileUiProvider(scope)
    web = ScopedWebUiProvider(scope)

    assert await display(object(), display_only=True) == [
        {"version": 1, "display_only": True}
    ]
    assert await web.bootstrap() == b"1"
    assert await web.state() == {"version": "1"}

    scope.version = 2
    assert await display(object(), display_only=False) == [
        {"version": 2, "display_only": False}
    ]
    assert await web.bootstrap() == b"2"
    assert scope.entered == scope.exited == 5

    async with open_request_scope(scope):
        assert mobile.catalog() == {"version": 2}
        assert mobile.asset("p", "r", "module", "a" * 64) == {"version": 2}
        assert await mobile.query(
            "p", "r", "method", {}, session_id=None, turn_id=None
        ) == {"version": 2}
    assert scope.entered == scope.exited == 6


def test_mobile_sync_projection_requires_an_active_scope() -> None:
    with pytest.raises(RuntimeError, match="request scope"):
        ScopedMobileUiProvider(_Scope()).catalog()
