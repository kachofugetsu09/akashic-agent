from __future__ import annotations

import httpx
import pytest

from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.web_chat import WebChatChannel


@pytest.mark.asyncio
async def test_ephemeral_download_roundtrip_and_guards(tmp_path) -> None:
    app = create_chat_app(workspace=tmp_path, channel=WebChatChannel("akashic"))
    png_bytes = b"\x89PNG\r\n\x1a\nfake-image-payload"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        bad_type = await client.post(
            "/api/chat/downloads?filename=test.html",
            content=b"<script>alert(1)</script>",
            headers={"Content-Type": "text/html"},
        )
        assert bad_type.status_code == 415

        empty = await client.post(
            "/api/chat/downloads?filename=empty.png",
            content=b"",
            headers={"Content-Type": "image/png"},
        )
        assert empty.status_code == 400

        too_large = await client.post(
            "/api/chat/downloads?filename=huge.png",
            content=b"x",
            headers={"Content-Type": "image/png", "Content-Length": str(21 * 1024 * 1024)},
        )
        assert too_large.status_code == 413

        created = await client.post(
            "/api/chat/downloads?filename=../evil%22name.png",
            content=png_bytes,
            headers={"Content-Type": "image/png; charset=utf-8"},
        )
        assert created.status_code == 200
        body = created.json()
        assert body["url"].startswith("/api/chat/downloads/")
        assert body["filename"] == "evilname.png"

        fetched = await client.get(body["url"])
        assert fetched.status_code == 200
        assert fetched.content == png_bytes
        assert fetched.headers["content-type"] == "image/png"
        assert fetched.headers["content-disposition"].startswith("attachment;")
        assert fetched.headers["x-content-type-options"] == "nosniff"
        assert fetched.headers["content-security-policy"] == "sandbox"
        assert fetched.headers["cache-control"] == "private, no-store"

        missing = await client.get("/api/chat/downloads/bad-token")
        assert missing.status_code == 404
