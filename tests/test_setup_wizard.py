from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest

from bootstrap.setup_wizard import (
    WizardAnswers,
    _async_fetch_qqbot_openid,
    _render_config,
)


def test_setup_wizard_renders_web_chat_config() -> None:
    text = _render_config(WizardAnswers())

    assert "[channels.chat]" in text
    assert "enabled = true" in text
    assert 'host = "127.0.0.1"' in text
    assert "port = 6322" in text
    assert 'channel_name = "web"' in text


@pytest.mark.asyncio
async def test_qqbot_openid_fetch_times_out_without_ws_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Resp:
        def __init__(self, payload: dict[str, str]) -> None:
            self._payload = payload

        def json(self) -> dict[str, str]:
            return self._payload

    class _Client:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, *args, **kwargs) -> _Resp:
            return _Resp({"access_token": "token"})

        async def get(self, *args, **kwargs) -> _Resp:
            return _Resp({"url": "wss://example.invalid"})

    class _NeverFrames:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(3600)
            raise StopAsyncIteration

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.AsyncClient = _Client
    fake_websockets = types.ModuleType("websockets")
    fake_websockets.connect = lambda *_args, **_kwargs: _NeverFrames()
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

    result = await _async_fetch_qqbot_openid("app", "secret", 1, threading.Event())

    assert result is None
