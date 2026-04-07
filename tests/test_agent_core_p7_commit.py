from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.context_store import DefaultContextStore, _extract_cited_ids
from agent.retrieval.protocol import RetrievalResult
from bus.events import InboundMessage


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict]:
        return self.messages[-max_messages:]

    def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if media:
            msg["media"] = list(media)
        msg.update(kwargs)
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_context_store_commit_persists_observes_schedules_and_dispatches():
    order: list[str] = []
    session = _DummySession("telegram:123")
    presence = SimpleNamespace(record_user_message=MagicMock(side_effect=lambda _key: None))
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("persist")),
    )
    writer = SimpleNamespace(
        events=[],
        emit=lambda event: order.append("observe") or writer.events.append(event),
    )
    post_turn = SimpleNamespace(
        events=[],
        schedule=lambda event: order.append("post_turn") or post_turn.events.append(event),
    )
    outbound = SimpleNamespace(
        dispatch=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("dispatch") or True)
    )
    decorator = SimpleNamespace(
        decorate=MagicMock(
            return_value=SimpleNamespace(
                content="整理好了",
                media=["/tmp/meme.png"],
                tag="shy",
            )
        )
    )
    store = DefaultContextStore(
        retrieval=SimpleNamespace(retrieve=AsyncMock(return_value=RetrievalResult(block=""))),
        context=SimpleNamespace(skills=SimpleNamespace(list_skills=MagicMock(return_value=[]))),
        session=SimpleNamespace(session_manager=session_manager, presence=presence),
        trace=SimpleNamespace(workspace=Path("."), observe_writer=writer),
        post_turn=post_turn,
        outbound=outbound,
        meme_decorator=decorator,
    )
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="123",
        content="你好",
        metadata={"req_id": "r1"},
    )
    post_turn_action = SimpleNamespace(
        run=AsyncMock(side_effect=lambda: order.append("post_turn_action"))
    )

    out = await store.commit(
        msg=msg,
        session_key="telegram:123",
        reply="<meme:shy> 整理好了",
        tools_used=["noop"],
        tool_chain=[{"text": "", "calls": []}],
        thinking="思考",
        retrieval_raw={"route": "RETRIEVE"},
        context_retry={"selected_plan": "full"},
        post_turn_actions=[post_turn_action],
    )

    assert out.content == "整理好了"
    assert out.media == ["/tmp/meme.png"]
    assert out.metadata["req_id"] == "r1"
    assert out.metadata["tools_used"] == ["noop"]
    presence.record_user_message.assert_called_once_with("telegram:123")
    session_manager.append_messages.assert_awaited_once()
    assert len(writer.events) == 2
    assert post_turn.events[0].assistant_response == "整理好了"
    outbound.dispatch.assert_awaited_once()
    post_turn_action.run.assert_awaited_once()
    assert order == ["persist", "observe", "observe", "post_turn", "post_turn_action", "dispatch"]
    assert session.messages[-1]["content"] == "整理好了"


def test_extract_cited_ids_strips_ascii_marker_only_at_end():
    clean, ids = _extract_cited_ids("答复正文\n§cited:[mem_1,mem-2]§")

    assert clean == "答复正文"
    assert ids == ["mem_1", "mem-2"]


def test_extract_cited_ids_keeps_body_text_when_marker_not_at_end():
    text = "正文里提到 §cited:[mem_1]§ 这串文本，但不是协议行。\n后面还有内容"

    clean, ids = _extract_cited_ids(text)

    assert clean == text
    assert ids == []
