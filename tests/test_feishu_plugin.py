from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from plugins.feishu.cards import (
    ToolLiveLine,
    build_live_card,
    build_markdown_card,
    build_summary_card,
)
from plugins.feishu.channel import FeishuChannel, _extract_post, _extract_text
from plugins.feishu.config import FeishuConfigModel


def test_config_model_normalizes_and_filters() -> None:
    config = FeishuConfigModel.model_validate(
        {
            "appId": "cli_xxx",
            "appSecret": "  secret  ",
            "allowFrom": ["ou_1", "", "user_1"],
            "domain": "https://open.feishu.cn/",
        }
    )

    assert config.app_id == "cli_xxx"
    assert config.app_secret == "secret"
    assert config.allow_from == ["ou_1", "user_1"]
    assert config.domain == "https://open.feishu.cn"


def test_config_model_drops_unresolved_env() -> None:
    config = FeishuConfigModel.model_validate({"app_id": "${FEISHU_APP_ID}", "app_secret": ""})

    assert config.app_id == ""
    assert config.app_secret == ""


def test_extract_text_from_feishu_content() -> None:
    assert _extract_text('{"text":"花月哥哥"}') == "花月哥哥"
    assert _extract_text("普通文本") == "普通文本"


def test_extract_post_collects_text_and_images() -> None:
    content = json.dumps(
        {
            "title": "标题",
            "content": [
                [{"tag": "text", "text": "第一行"}, {"tag": "img", "image_key": "img_1"}],
                [{"tag": "a", "text": "链接", "href": "https://x"}],
            ],
        }
    )

    text, images = _extract_post(content)

    assert text == "标题\n第一行\n链接"
    assert images == ["img_1"]


def test_build_markdown_card_uses_schema_2_markdown() -> None:
    card = json.loads(build_markdown_card("**hi**"))

    assert card["schema"] == "2.0"
    element = card["body"]["elements"][0]
    assert element["tag"] == "markdown"
    assert element["content"] == "**hi**"


def test_build_summary_card_folds_thinking_keeps_tools() -> None:
    card = json.loads(build_summary_card("想了想", [ToolLiveLine("c", "shell", "查询", "")]))
    elements = card["body"]["elements"]

    # 思考折叠面板（默认收起）
    panel = next(el for el in elements if el["tag"] == "collapsible_panel")
    assert panel["expanded"] is False
    assert panel["elements"][0]["content"] == "想了想"
    # 工具时间线可见
    assert any(el["tag"] == "markdown" and "shell" in el["content"] for el in elements)


def test_build_live_card_running_thinking_is_visible_block() -> None:
    card = json.loads(build_live_card("正在想", [], ""))
    elements = card["body"]["elements"]

    # 流式阶段思考是可见 markdown 块（对齐 tg live blockquote），不折叠
    assert elements[0]["tag"] == "markdown"
    assert "思考过程" in elements[0]["content"] and "正在想" in elements[0]["content"]
    assert not any(el["tag"] == "collapsible_panel" for el in elements)


@pytest.mark.asyncio
async def test_send_uses_card_and_chat_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/tenant_access_token/internal"):
            return httpx.Response(
                200,
                json={"code": 0, "tenant_access_token": "token", "expire": 7200},
            )
        return httpx.Response(200, json={"code": 0, "data": {"message_id": "om_1"}})

    channel = FeishuChannel(app_id="cli_xxx", app_secret="secret")
    channel._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await channel.send("oc_chat", "你好 **markdown**")
    await channel._client.aclose()

    send_req = requests[1]
    assert send_req.url.path == "/open-apis/im/v1/messages"
    assert send_req.url.params["receive_id_type"] == "chat_id"
    assert send_req.headers["authorization"] == "Bearer token"
    body = json.loads(send_req.content)
    assert body["receive_id"] == "oc_chat"
    assert body["msg_type"] == "interactive"
    card = json.loads(body["content"])
    assert card["schema"] == "2.0"
    assert card["body"]["elements"][0]["content"] == "你好 **markdown**"


@pytest.mark.asyncio
async def test_send_falls_back_to_text_when_card_fails() -> None:
    posts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tenant_access_token/internal"):
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "t", "expire": 7200})
        body = json.loads(request.content)
        posts.append(body)
        if body["msg_type"] == "interactive":
            # 模拟卡片渲染/大小错误
            return httpx.Response(200, json={"code": 230001, "msg": "invalid card"})
        return httpx.Response(200, json={"code": 0, "data": {"message_id": "om"}})

    channel = FeishuChannel(app_id="a", app_secret="b")
    channel._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await channel.send("oc_chat", "**hi**")
    await channel._client.aclose()

    # 先尝试卡片，失败后降级为 msg_type text
    assert posts[0]["msg_type"] == "interactive"
    assert posts[1]["msg_type"] == "text"
    assert json.loads(posts[1]["content"])["text"] == "**hi**"


@pytest.mark.asyncio
async def test_post_message_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    attempts = {"n": 0}
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tenant_access_token/internal"):
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "t", "expire": 7200})
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"code": 99991400})
        return httpx.Response(200, json={"code": 0, "data": {"message_id": "om"}})

    channel = FeishuChannel(app_id="a", app_secret="b")
    channel._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await channel.send("oc_chat", "hi")
    await channel._client.aclose()

    # 第一次 429 退避后重试成功
    assert attempts["n"] == 2
    assert slept and slept[0] >= 1.0


def test_resolve_receive_picks_id_type() -> None:
    channel = FeishuChannel(app_id="cli_xxx", app_secret="secret")

    assert channel._resolve_receive("oc_chat") == ("oc_chat", "chat_id")
    assert channel._resolve_receive("ou_user") == ("ou_user", "open_id")
    assert channel._resolve_receive("feishu:oc_chat") == ("oc_chat", "chat_id")


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
                parent_id="",
                content='{"text":"你好"}',
            ),
        )
    )

    await channel._handle_message_event(event)

    assert len(published) == 1
    assert published[0].channel == "feishu"
    assert published[0].chat_id == "oc_1"
    assert published[0].content == "你好"
    assert published[0].metadata["open_id"] == "ou_1"


@pytest.mark.asyncio
async def test_on_response_freezes_live_card_and_sends_final() -> None:
    from bus.events import OutboundMessage

    calls: list[tuple[str, str, str]] = []  # (method, path, content)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tenant_access_token/internal"):
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "t", "expire": 7200})
        content = json.loads(request.content).get("content", "")
        calls.append((request.method, request.url.path, content))
        return httpx.Response(200, json={"code": 0, "data": {"message_id": "om"}})

    channel = FeishuChannel(app_id="a", app_secret="b")
    channel._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    session_key = "feishu:oc_1"
    channel._live_messages[session_key] = "om_live"
    channel._tool_lines[session_key] = [ToolLiveLine("c", "shell", "查询", "")]
    channel._thinking_buffers[session_key] = "我的思考"

    await channel._on_response(
        OutboundMessage(channel="feishu", chat_id="oc_1", content="最终结果")
    )
    await channel._client.aclose()

    # 不撤回（无 DELETE）
    assert not any(m == "DELETE" for m, _p, _c in calls)
    # 1. 原地 PATCH 预览卡为过程卡（思考折叠 + 工具）
    patch = next(c for c in calls if c[0] == "PATCH")
    assert patch[1].endswith("/messages/om_live")
    assert "collapsible_panel" in patch[2] and "我的思考" in patch[2] and "shell" in patch[2]
    # 2. 单独发一条最终结果
    post = next(c for c in calls if c[0] == "POST" and "最终结果" in c[2])
    assert "最终结果" in post[2]


@pytest.mark.asyncio
async def test_stop_command_routes_to_interrupt() -> None:
    sent: list[tuple[str, str]] = []
    interrupts: list[str] = []

    class Interrupt:
        def request_interrupt(self, *, session_key: str, sender: str, command: str) -> Any:
            interrupts.append(session_key)
            return SimpleNamespace(message="已请求中断")

    channel = FeishuChannel(app_id="cli_xxx", app_secret="secret", allow_from=["ou_1"])
    channel._interrupt_controller = Interrupt()  # type: ignore[assignment]

    async def fake_send(chat_id: str, text: str) -> None:
        sent.append((chat_id, text))

    channel.send = fake_send  # type: ignore[method-assign]
    event = SimpleNamespace(
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(open_id="ou_1", user_id="", union_id="")
            ),
            message=SimpleNamespace(
                chat_type="p2p",
                message_type="text",
                message_id="om_2",
                chat_id="oc_1",
                parent_id="",
                content='{"text":"/stop"}',
            ),
        )
    )

    await channel._handle_message_event(event)

    assert interrupts == ["feishu:oc_1"]
    assert sent == [("oc_1", "已请求中断")]
