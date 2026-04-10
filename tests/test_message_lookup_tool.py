import json
from pathlib import Path

import pytest

from agent.tools.message_lookup import FetchMessagesTool, SearchMessagesTool
from prompts.agent import build_agent_behavior_rules_prompt
from session.manager import SessionManager
from session.store import SessionStore


def _setup_session(store: SessionStore, key: str, n_messages: int) -> None:
    store.upsert_session(
        key,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        last_consolidated=0,
        metadata={},
    )
    roles = ["user", "assistant"]
    for seq in range(n_messages):
        store.insert_message(
            key,
            role=roles[seq % 2],
            content=f"msg-{seq}",
            ts=f"2026-01-01T00:00:{seq:02d}+00:00",
            seq=seq,
        )


@pytest.mark.asyncio
async def test_fetch_messages_returns_rows_in_input_order(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    _setup_session(store, "tg:1", 2)

    tool = FetchMessagesTool(store)
    payload = json.loads(await tool.execute(ids=["tg:1:1", "tg:1:0"]))

    assert payload["count"] == 2
    assert payload["matched_count"] == 2
    assert [m["id"] for m in payload["messages"]] == ["tg:1:1", "tg:1:0"]


@pytest.mark.asyncio
async def test_fetch_messages_with_context(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    _setup_session(store, "tg:1", 7)  # seq 0..6

    tool = FetchMessagesTool(store)
    # fetch seq=3, context=2 → expect seq 1..5
    payload = json.loads(await tool.execute(ids=["tg:1:3"], context=2))

    ids = [m["id"] for m in payload["messages"]]
    assert "tg:1:3" in ids
    assert "tg:1:1" in ids
    assert "tg:1:5" in ids
    assert "tg:1:0" not in ids
    assert "tg:1:6" not in ids
    assert payload["matched_count"] == 1
    assert payload["count"] == 5

    # in_source_ref flag: only the hit is True
    hit = next(m for m in payload["messages"] if m["id"] == "tg:1:3")
    ctx_msg = next(m for m in payload["messages"] if m["id"] == "tg:1:1")
    assert hit["in_source_ref"] is True
    assert ctx_msg["in_source_ref"] is False


@pytest.mark.asyncio
async def test_fetch_messages_context_clamps_at_seq_zero(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    _setup_session(store, "tg:1", 3)  # seq 0,1,2

    tool = FetchMessagesTool(store)
    payload = json.loads(await tool.execute(ids=["tg:1:0"], context=3))

    # context before seq 0 is clamped; should get seq 0,1,2,3 — but only 0-2 exist
    ids = [m["id"] for m in payload["messages"]]
    assert "tg:1:0" in ids
    assert "tg:1:1" in ids
    assert "tg:1:2" in ids
    assert payload["matched_count"] == 1


@pytest.mark.asyncio
async def test_fetch_messages_context_clamps_at_max_window(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    _setup_session(store, "tg:1", 30)  # seq 0..29

    tool = FetchMessagesTool(store)
    payload = json.loads(await tool.execute(ids=["tg:1:11"], context=999))

    ids = [m["id"] for m in payload["messages"]]
    assert ids[0] == "tg:1:1"
    assert ids[-1] == "tg:1:21"
    assert "tg:1:0" not in ids
    assert "tg:1:22" not in ids
    assert payload["matched_count"] == 1
    assert payload["count"] == 21


@pytest.mark.asyncio
async def test_fetch_messages_supports_window_source_ref(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    _setup_session(store, "tg:1", 6)

    tool = FetchMessagesTool(store)
    payload = json.loads(
        await tool.execute(source_ref='["tg:1:2","tg:1:3"]#profile', context=1)
    )

    assert [m["id"] for m in payload["messages"]] == [
        "tg:1:1",
        "tg:1:2",
        "tg:1:3",
        "tg:1:4",
    ]
    assert payload["matched_count"] == 2
    assert [m["in_source_ref"] for m in payload["messages"]] == [False, True, True, False]


@pytest.mark.asyncio
async def test_fetch_messages_supports_mixed_ids_and_source_refs(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    _setup_session(store, "tg:1", 5)

    tool = FetchMessagesTool(store)
    payload = json.loads(
        await tool.execute(
            ids=["tg:1:4"],
            source_refs=['["tg:1:1","tg:1:2"]#h:abc', "tg:1:4"],
        )
    )

    assert [m["id"] for m in payload["messages"]] == ["tg:1:4", "tg:1:1", "tg:1:2"]
    assert payload["matched_count"] == 3


@pytest.mark.asyncio
async def test_search_messages_returns_preview_with_source_ref(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.upsert_session(
        "tg:1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        last_consolidated=0,
        metadata={},
    )
    preview_lines = "\n".join(f"line-{i}" for i in range(55))
    store.insert_message(
        "tg:1",
        role="user",
        content=f"benchmark recall 0.62\n{preview_lines}",
        ts="2026-01-01T00:00:01+00:00",
        seq=0,
    )

    tool = SearchMessagesTool(store)
    payload = json.loads(await tool.execute(query="benchmark", session_key="tg:1"))

    assert payload["count"] == 1
    assert payload["matched_count"] == 1
    assert payload["offset"] == 0
    assert payload["limit"] == 10
    assert payload["has_more"] is False
    assert payload["next_offset"] is None

    item = payload["messages"][0]
    assert item["id"] == "tg:1:0"
    assert item["source_ref"] == "tg:1:0"
    assert item["session_key"] == "tg:1"
    assert item["role"] == "user"
    assert item["preview_line_count"] == 50
    assert item["total_line_count"] == 56
    assert item["truncated"] is True
    assert "benchmark recall 0.62" in item["preview"]
    assert "line-48" in item["preview"]
    assert "line-49" not in item["preview"]
    assert "line-50" not in item["preview"]
    assert "已截断" in item["preview"]


@pytest.mark.asyncio
async def test_search_messages_supports_filters(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.upsert_session(
        "tg:1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        last_consolidated=0,
        metadata={},
    )
    store.upsert_session(
        "tg:2",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        last_consolidated=0,
        metadata={},
    )

    store.insert_message("tg:1", role="user", content="benchmark recall 0.62", ts="2026-01-01T00:00:01+00:00", seq=0)
    store.insert_message("tg:1", role="assistant", content="benchmark done", ts="2026-01-01T00:00:02+00:00", seq=1)
    store.insert_message("tg:2", role="user", content="benchmark other", ts="2026-01-01T00:00:03+00:00", seq=0)

    tool = SearchMessagesTool(store)

    payload = json.loads(
        await tool.execute(
            query="benchmark",
            session_key="tg:1",
            role="user",
            limit=10,
        )
    )
    assert payload["count"] == 1
    assert payload["matched_count"] == 1
    assert payload["messages"][0]["session_key"] == "tg:1"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["source_ref"] == "tg:1:0"
    assert "0.62" in payload["messages"][0]["preview"]


@pytest.mark.asyncio
async def test_search_messages_supports_offset_pagination(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.upsert_session(
        "tg:1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        last_consolidated=0,
        metadata={},
    )
    for seq in range(5):
        store.insert_message(
            "tg:1",
            role="user" if seq % 2 == 0 else "assistant",
            content=f"benchmark result {seq}",
            ts=f"2026-01-01T00:00:0{seq}+00:00",
            seq=seq,
        )

    tool = SearchMessagesTool(store)

    first_page = json.loads(await tool.execute(query="benchmark", session_key="tg:1", limit=2))
    second_page = json.loads(
        await tool.execute(
            query="benchmark",
            session_key="tg:1",
            limit=2,
            offset=first_page["next_offset"],
        )
    )

    assert first_page["count"] == 2
    assert first_page["matched_count"] == 5
    assert first_page["has_more"] is True
    assert first_page["next_offset"] == 2
    assert [item["id"] for item in first_page["messages"]] == ["tg:1:4", "tg:1:3"]

    assert second_page["count"] == 2
    assert second_page["matched_count"] == 5
    assert second_page["offset"] == 2
    assert second_page["has_more"] is True
    assert second_page["next_offset"] == 4
    assert [item["id"] for item in second_page["messages"]] == ["tg:1:2", "tg:1:1"]


@pytest.mark.asyncio
async def test_search_messages_mixed_long_and_short_terms_keeps_short_only_hits(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.upsert_session(
        "tg:1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        last_consolidated=0,
        metadata={},
    )
    store.insert_message("tg:1", role="user", content="phase only", ts="2026-01-01T00:00:01+00:00", seq=0)
    store.insert_message("tg:1", role="assistant", content="只提到支付", ts="2026-01-01T00:00:02+00:00", seq=1)
    store.insert_message("tg:1", role="user", content="phase 支付 一起命中", ts="2026-01-01T00:00:03+00:00", seq=2)

    tool = SearchMessagesTool(store)
    payload = json.loads(await tool.execute(query="phase 支付", session_key="tg:1", limit=10))

    assert payload["count"] == 3
    assert payload["matched_count"] == 3
    assert {item["id"] for item in payload["messages"]} == {"tg:1:0", "tg:1:1", "tg:1:2"}


@pytest.mark.asyncio
async def test_search_messages_empty_query_returns_empty(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    tool = SearchMessagesTool(store)
    payload = json.loads(await tool.execute(query="   "))
    assert payload == {
        "count": 0,
        "matched_count": 0,
        "limit": 10,
        "offset": 0,
        "has_more": False,
        "next_offset": None,
        "messages": [],
    }


def test_next_seq_after_seq_zero_should_return_one(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:test")
    session.messages = [
        {
            "role": "assistant",
            "content": "prev",
            "timestamp": "2026-03-27T22:04:06+08:00",
        }
    ]
    manager.save(session)

    assert manager._store.next_seq("cli:test") == 1


def test_message_lookup_tools_require_fetch_for_evidence():
    assert "source_ref" in SearchMessagesTool.description
    assert "fetch_messages" in SearchMessagesTool.description
    assert "必须" in SearchMessagesTool.description
    assert "fetch_messages" in FetchMessagesTool.description
    assert "§cited:[" in FetchMessagesTool.description


def test_history_fact_guard_requires_fetch_after_search_preview():
    prompt = build_agent_behavior_rules_prompt(workspace=Path("."))
    assert "search_messages" in prompt
    assert "fetch_messages" in prompt
    assert "source_ref" in prompt
    assert "预览" in prompt
    assert "§cited:[" in prompt


def test_memory_correction_protocol_covers_soft_corrections_and_forget_memory():
    prompt = build_agent_behavior_rules_prompt(workspace=Path("."))
    assert "其实还好" in prompt
    assert "并不反感" in prompt
    assert "forget_memory" in prompt
    assert "若用户这轮是在纠正你，而你本轮没有调用 `forget_memory`" in prompt
    assert "在拿到 fetch_messages 结果前，禁止直接调用 `forget_memory`" in prompt
    assert "调用了 `forget_memory` 却没有先调用 `fetch_messages`" in prompt


def test_behavior_rules_force_fact_questions_to_answer_directly():
    prompt = build_agent_behavior_rules_prompt(workspace=Path("."))
    assert "简单问题直接回答" in prompt
    assert "时间线、日期、安排、是否记得、列事实、重新梳理" in prompt
    assert "不要追加鼓励、睡觉建议、备战计划、陪伴式抚慰" in prompt
    assert "当前这一问如果是事实整理或时间确认，也不要顺着前文继续输出情绪安慰" in prompt
    assert "事实型问题答完事实就停" in prompt
    assert "稳住就行" in prompt


def test_behavior_rules_use_evidence_threshold_not_keyword_filtering():
    prompt = build_agent_behavior_rules_prompt(workspace=Path("."))
    assert "知识截止时间" in prompt
    assert "外部世界此刻是什么样" in prompt
    assert "本轮外部证据" in prompt
    assert "这里的判断看“证据门槛”，不是看字面关键词" in prompt
    assert "如果答案取决于稳定知识" in prompt
    assert "如果答案取决于本轮外部证据" in prompt
    assert "我现在不能确认 / 我需要先查一下" in prompt
    assert "没有本轮证据就只能说记忆里的旧信息" in prompt
