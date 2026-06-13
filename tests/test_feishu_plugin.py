from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from plugins.feishu.channel import FeishuChannel, _extract_text
from plugins.feishu.config import load_feishu_config


def test_load_feishu_config_from_plugin_dir(tmp_path: Path) -> None:
    (tmp_path / "config.local.toml").write_text(
        "\n".join([
            'app_id = "cli_xxx"',
            'app_secret = "secret"',
            'allow_from = ["ou_1", "user_1"]',
            'domain = "https://open.feishu.cn"',
        ]),
        encoding="utf-8",
    )

    config = load_feishu_config(plugin_dir=tmp_path)

    assert config.app_id == "cli_xxx"
    assert config.app_secret == "secret"
    assert config.allow_from == ["ou_1", "user_1"]


def test_extract_text_from_feishu_content() -> None:
    assert _extract_text('{"text":"花月哥哥"}') == "花月哥哥"
    assert _extract_text("普通文本") == "普通文本"


@pytest.mark.asyncio
async def test_send_uses_tenant_token_and_chat_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/tenant_access_token/internal"):
            return httpx.Response(
                200,
                json={"code": 0, "tenant_access_token": "token", "expire": 7200},
            )
        return httpx.Response(200, json={"code": 0})

    channel = FeishuChannel(app_id="cli_xxx", app_secret="secret")
    channel._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await channel.send("oc_chat", "你好")
    await channel._client.aclose()

    assert requests[1].url.path == "/open-apis/im/v1/messages"
    assert requests[1].url.params["receive_id_type"] == "chat_id"
    assert requests[1].headers["authorization"] == "Bearer token"
    assert b'"receive_id":"oc_chat"' in requests[1].content


@pytest.mark.asyncio
async def test_private_text_event_publishes_inbound() -> None:
    published: list[Any] = []

    class Bus:
        async def publish_inbound(self, item: Any) -> None:
            published.append(item)

    channel = FeishuChannel(app_id="cli_xxx", app_secret="secret", allow_from=["ou_1"])
    channel._bus = Bus()  # type: ignore[assignment]
    event = SimpleNamespace(
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(open_id="ou_1", user_id="", union_id="")
            ),
            message=SimpleNamespace(
                chat_type="p2p",
                message_type="text",
                message_id="om_1",
                chat_id="oc_1",
                content='{"text":"你好"}',
            ),
        )
    )

    await channel._handle_message_event(event)

    assert len(published) == 1
    assert published[0].channel == "feishu"
    assert published[0].chat_id == "oc_1"
    assert published[0].content == "你好"
