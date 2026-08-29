from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent.tools.web_fetch import WebFetchTool


def _response(
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    content: bytes = b"",
    url: str = "https://example.com",
) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        content=content,
        request=httpx.Request("GET", url),
    )


def _decode(payload: str) -> dict[str, object]:
    return json.loads(payload)


async def test_web_fetch_converts_html_through_the_public_tool_contract() -> None:
    requester = MagicMock()
    requester.get = AsyncMock(
        return_value=_response(
            headers={"content-type": "text/html"},
            content=(
                b"<html><body><script>x</script>"
                b"<p>Hello <b>world</b></p></body></html>"
            ),
        )
    )
    tool = WebFetchTool(requester=requester)

    text = _decode(await tool.execute(url="https://example.com", format="text"))
    markdown = _decode(
        await tool.execute(url="https://example.com", format="markdown")
    )
    local = _decode(await tool.execute(url="http://127.0.0.1", format="text"))

    assert text["text"] == "Hello world"
    assert "Hello" in str(markdown["text"])
    assert local["text"] == "Hello world"


async def test_web_fetch_reports_expected_boundary_failures() -> None:
    requester = MagicMock()
    tool = WebFetchTool(requester=requester)

    requester.get = AsyncMock(return_value=_response(status=404))
    assert "HTTP 404" in str(
        _decode(await tool.execute(url="https://example.com"))["error"]
    )

    requester.get = AsyncMock(
        return_value=_response(headers={"content-type": "application/pdf"})
    )
    assert "二进制内容" in str(
        _decode(await tool.execute(url="https://example.com"))["error"]
    )

    requester.get = AsyncMock(
        return_value=_response(
            headers={"content-type": "text/plain", "content-length": "broken"},
            content=b"x",
        )
    )
    assert "Content-Length 无效" in str(
        _decode(await tool.execute(url="https://example.com"))["error"]
    )

    requester.get = AsyncMock(side_effect=httpx.TimeoutException("slow"))
    assert "请求超时" in str(
        _decode(await tool.execute(url="https://example.com"))["error"]
    )

    requester.get = AsyncMock(side_effect=httpx.ConnectError("offline"))
    assert "无法建立连接" in str(
        _decode(await tool.execute(url="https://example.com"))["error"]
    )

    requester.get = AsyncMock(side_effect=httpx.RequestError("bad request"))
    assert "请求失败" in str(
        _decode(await tool.execute(url="https://example.com"))["error"]
    )

    requester.get = AsyncMock()
    assert "http:// 或 https://" in str(
        _decode(await tool.execute(url="ftp://example.com"))["error"]
    )
    assert "URL 格式无效" in str(
        _decode(await tool.execute(url="http://[::1"))["error"]
    )
    requester.get.assert_not_awaited()


async def test_web_fetch_does_not_mask_requester_programming_errors() -> None:
    requester = MagicMock()
    requester.get = AsyncMock(side_effect=RuntimeError("requester invariant broken"))
    tool = WebFetchTool(requester=requester)

    with pytest.raises(RuntimeError, match="requester invariant broken"):
        await tool.execute(url="https://example.com")
