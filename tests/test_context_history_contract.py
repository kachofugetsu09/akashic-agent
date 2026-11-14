from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from plugins.models.contract import (
    LLMResponse,
    ModelContinuation,
    ToolCall as ModelToolCall,
)
from plugins.content.plugin import check_text
from plugins.context.api import Materials, Reminder, material_data
from plugins.models.projection import check_facts
from plugins.ledger.embedding_store import MessageEmbeddings, EmbeddingRecords
from plugins.ledger.log import MessageLog
from plugins.ledger.contract import ContentPart, Input, Output
from tests.support.message_react import runtime
from plugins.tools.execution import Result


def assert_rows_unchanged(
    before: tuple[tuple[object, ...], ...],
    after: tuple[tuple[object, ...], ...],
    *,
    state_name: str,
) -> None:
    """断言权威状态中的既有行没有被删改。"""
    if list(after) != list(before):
        raise AssertionError(f"{state_name} 的既有行发生删改")


def assert_no_forbidden_writes(
    statements: list[str],
    *,
    tables: tuple[str, ...],
) -> None:
    """断言执行轨迹没有删改受保护表。"""
    protected = {table.casefold() for table in tables}
    violations: list[str] = []
    for statement in statements:
        normalized = " ".join(statement.casefold().split())
        if not normalized.startswith(("delete ", "update ", "replace ")):
            continue
        if any(table in normalized for table in protected):
            violations.append(statement)
    if violations:
        raise AssertionError(f"检测到受保护状态删改: {violations}")


def _seed_history(log: MessageLog) -> tuple:
    """通过真实 MessageLog writer 建立已结算历史。"""
    input_writer = log.writer(
        "s", author="user", source="conversation", body_types=(Input,),
        content={"text": check_text},
        metadata_keys=frozenset({"semantic_marker"}),
        update_metadata=lambda _body: {"semantic_marker": "history"},
    )
    output_writer = log.writer(
        "s", author="assistant", source="conversation", body_types=(Output,),
        content={"text": check_text, "model.facts": check_facts},
    )
    for index in range(3):
        input_writer.append(
            f"old-input-{index}",
            Input((ContentPart("text", f"old input {index}"),)),
        )
        output_writer.append(
            f"old-output-{index}",
            Output((ContentPart("text", f"old output {index}"),), "complete"),
        )
    return log.reader("s").snapshot()


def _text(message) -> str:
    part = message.body.parts[0]
    assert isinstance(part, ContentPart)
    assert isinstance(part.value, str)
    return part.value


def _vector_snapshot(store: EmbeddingRecords, messages) -> dict[str, tuple[float, ...]]:
    """通过 embedding owner 读取指定历史消息的向量。"""
    values: dict[str, tuple[float, ...]] = {}
    for message in messages:
        vector = store.read(message, model="gate-model", dimension=2)
        assert vector is not None
        values[message.message_id] = vector
    return values


def _message_rows(messages) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (message.message_id, message.session_id, message.seq, message.author,
         message.source, message.body)
        for message in messages
    )


def _embedding_rows(vectors: dict[str, tuple[float, ...]]) -> tuple[tuple[object, ...], ...]:
    return tuple((message_id, tuple(vector)) for message_id, vector in sorted(vectors.items()))


@pytest.mark.asyncio
async def test_real_message_reply_preserves_history_embeddings_and_restart_seq(tmp_path: Path) -> None:
    """真实回复只投影历史，不改写原消息或向量。"""
    requests: list[object] = []

    async def complete(request):
        requests.append(request)
        return LLMResponse("ok")

    async def invoke(_key, _arguments):
        pytest.fail("this history oracle has no tool effects")

    db_path = tmp_path / "sessions.db"
    with_runtime = runtime(tmp_path, complete, invoke)
    async with with_runtime as (conversation, log, _models, _run):
        before = _seed_history(log)
        before_metadata = log.reader("s").metadata()
        embedding_store = MessageEmbeddings(log, tmp_path / "sessions-derived.db")
        embeddings = embedding_store.bind(_text)
        for message in before:
            embeddings.save(
                message,
                model="gate-model",
                embedding=[float(message.seq), 1.0],
            )
        before_vectors = _vector_snapshot(embeddings, before)
        before_message_rows = _message_rows(before)
        before_embedding_rows = _embedding_rows(before_vectors)
        statements: list[str] = []
        log._connection.set_trace_callback(statements.append)  # pyright: ignore[reportPrivateUsage]
        await conversation.accept("current", Input((ContentPart("text", "continue"),)))
        task = await conversation.start(_run)
        assert task is not None
        await task.join()
        after = log.reader("s").snapshot()
        log._connection.set_trace_callback(None)  # pyright: ignore[reportPrivateUsage]

        assert_rows_unchanged(
            before_message_rows,
            _message_rows(after[: len(before)]),
            state_name="sessions.db/messages",
        )
        assert len(after) == len(before) + 2
        assert log.reader("s").metadata() == before_metadata
        assert len(requests) == 1
        assert "old input 0" in str(requests[0])
        assert "old output 2" in str(requests[0])
        assert_no_forbidden_writes(
            statements,
            tables=("messages", "message_embeddings"),
        )
        after_vectors = _vector_snapshot(embeddings, before)
        assert_rows_unchanged(
            before_embedding_rows,
            _embedding_rows(after_vectors),
            state_name="message_embeddings",
        )
        embedding_store.close()
        highwater = after[-1].seq

    reopened = MessageLog(db_path)
    try:
        assert reopened.reader("s").snapshot() == after
        writer = reopened.writer(
            "s", author="user", source="conversation", body_types=(Input,),
            content={"text": check_text},
        )
        appended = writer.append("after-reopen", Input((ContentPart("text", "after restart"),)))
        assert appended.seq == highwater + 1
        assert reopened.reader("s").head() == appended.seq
    finally:
        reopened.close()


def _request_text(request: object) -> str:
    """Read synthetic markers from the model request without relying on row positions."""
    messages = getattr(request, "messages")
    return "\n".join(
        content
        for row in messages
        if isinstance((content := row.get("content")), str)
    )


@pytest.mark.asyncio
async def test_reminder_replay_is_once_per_input_and_material_identity(tmp_path: Path) -> None:
    """A fresh projection keeps one fact per material identity and restores opaque state."""
    requests: list[object] = []
    material_values = iter(("first-a", "first-b", "first-b", "second-c"))

    async def complete(request):
        requests.append(request)
        step = len(requests)
        if step < 3:
            return LLMResponse(
                None,
                [ModelToolCall(f"tool-{step}", "example", {})],
                continuation=ModelContinuation("model", {"step": step}),
            )
        return LLMResponse(
            f"done-{step}",
            continuation=ModelContinuation("model", {"step": step}),
        )

    async def invoke(_key, _arguments):
        return Result("success", (ContentPart("text", "synthetic tool result"),))

    async def material_source(_snapshot):
        return material_data(Materials("", (Reminder("memory", next(material_values), 1),)))

    async with runtime(
        tmp_path, complete, invoke, max_steps=4, material_source=material_source,
    ) as (conversation, log, _models, run):
        await conversation.accept("first-input", Input((ContentPart("text", "first"),)))
        first = await conversation.start(run)
        assert first is not None
        await first.join()

        await conversation.accept("second-input", Input((ContentPart("text", "second"),)))
        second = await conversation.start(run)
        assert second is not None
        await second.join()

        assert len(requests) == 4
        counts = [
            tuple(_request_text(request).count(marker) for marker in ("first-a", "first-b", "second-c"))
            for request in requests
        ]
        assert counts == [(1, 0, 0), (1, 1, 0), (1, 1, 0), (1, 1, 1)]
        assert requests[3].continuation == ModelContinuation("model", {"step": 3})

        facts = [
            part.value
            for message in log.reader("s").snapshot()
            if isinstance(message.body, Output)
            for part in message.body.parts
            if isinstance(part, ContentPart) and part.kind == "model.facts"
        ]
        assert len(facts) == 4
        input_ids: list[object] = []
        for value in facts:
            assert isinstance(value, Mapping)
            input_ids.append(value["reminder_input_id"])
        assert input_ids == ["first-input"] * 3 + ["second-input"]


@pytest.mark.asyncio
async def test_legacy_reminder_facts_remain_verbatim_history(tmp_path: Path) -> None:
    """Untyped pre-upgrade facts retain their recorded replay instead of guessed deduplication."""
    requests: list[object] = []
    material_values = iter(("legacy-material", "current-material"))

    async def complete(request):
        requests.append(request)
        return LLMResponse("done")

    async def invoke(_key, _arguments):
        pytest.fail("the legacy replay fixture has no tool effects")

    async def material_source(_snapshot):
        return material_data(Materials("", (Reminder("memory", next(material_values), 1),)))

    async with runtime(
        tmp_path, complete, invoke, max_steps=2, material_source=material_source,
    ) as (conversation, log, _models, run):
        await conversation.accept("first-input", Input((ContentPart("text", "first"),)))
        first = await conversation.start(run)
        assert first is not None
        await first.join()

        first_facts = next(
            part.value
            for message in log.reader("s").snapshot()
            if isinstance(message.body, Output)
            for part in message.body.parts
            if isinstance(part, ContentPart) and part.kind == "model.facts"
        )
        assert isinstance(first_facts, Mapping)
        legacy_facts = {
            key: value
            for key, value in first_facts.items()
            if key not in {"reminder_input_id", "reminder_sha256"}
        }
        log.writer(
            "s",
            author="assistant",
            source="conversation",
            body_types=(Output,),
            content={"text": check_text, "model.facts": check_facts},
        ).append(
            "legacy-output",
            Output(
                (
                    ContentPart("text", "legacy durable output"),
                    ContentPart("model.facts", legacy_facts),
                ),
                "complete",
            ),
        )

        await conversation.accept("second-input", Input((ContentPart("text", "second"),)))
        second = await conversation.start(run)
        assert second is not None
        await second.join()

        assert len(requests) == 2
        second_request = _request_text(requests[1])
        assert second_request.count("legacy-material") == 2
        assert second_request.count("current-material") == 1


def test_idle_gap_and_idle_summary_due_respect_turn_boundaries() -> None:
    """空闲压缩只在新回合首步 Input 且前驱已结算时触发，工具循环中途或 steering 不误触。"""
    from datetime import UTC, datetime, timedelta
    from plugins.context.contract import idle_gap_minutes, idle_summary_due, summary_due
    from plugins.ledger.contract import CallRef, Message, ToolCall, ToolResult

    t0 = datetime(2026, 11, 9, 0, 0, tzinfo=UTC)
    u1 = Message("u1", "s", 0, t0, "user", "conversation", Input((ContentPart("text", "hi"),)))
    assert idle_gap_minutes(()) is None
    assert idle_gap_minutes((u1,)) is None

    # 已结算 Output 后间隔 65 分钟到达新 Input（含连续两条尾部 Input）
    a1 = Message("a1", "s", 1, t0 + timedelta(minutes=1), "assistant", "conversation",
                 Output((ContentPart("text", "done"),), "complete"))
    u2 = Message("u2", "s", 2, t0 + timedelta(minutes=66), "user", "conversation",
                 Input((ContentPart("text", "wake"),)))
    u3 = Message("u3", "s", 3, t0 + timedelta(minutes=67), "user", "conversation",
                 Input((ContentPart("text", "follow-up"),)))
    assert idle_gap_minutes((u1, a1, u2)) == pytest.approx(65.0)
    assert idle_gap_minutes((u1, a1, u2, u3)) == pytest.approx(65.0)
    assert idle_summary_due(200_000, (u1, a1, u2), idle_minutes=60, idle_tokens=200_000)
    assert not idle_summary_due(199_999, (u1, a1, u2), idle_minutes=60, idle_tokens=200_000)
    assert not idle_summary_due(250_000, (u1, a1, u2), idle_minutes=0, idle_tokens=200_000)
    assert summary_due(200_000, 1_000_000, force=False, snapshot=(u1, a1, u2),
                       idle_minutes=60, idle_tokens=200_000)

    # 工具循环中途（末尾为 ToolResult）或未完成 Output(continue) 后的插话不触发空闲压缩
    open_out = Message("a2", "s", 4, t0 + timedelta(minutes=68), "assistant", "conversation",
                       Output((ToolCall("b1", {}),), "continue"))
    steer_in = Message("u4", "s", 5, t0 + timedelta(minutes=150), "user", "conversation",
                       Input((ContentPart("text", "steer"),)))
    tool_res = Message("tr1", "s", 6, t0 + timedelta(minutes=151), "tool", "conversation",
                       ToolResult(call_ref=CallRef("a2", 0), outcome="success",
                                  parts=(ContentPart("text", "ok"),)))
    assert idle_gap_minutes((u1, a1, u2, open_out, steer_in)) is None
    assert idle_gap_minutes((u1, a1, u2, open_out, tool_res)) is None

    from plugins.compaction.plugin import Config, _idle_minutes_for
    cfg_selected = Config.model_validate({
        "idle_scope": "selected",
        "idle_sessions": ["s", "s", " other "],
        "idle_trigger_minutes": 45,
    })
    assert cfg_selected.idle_sessions == ("s", "other")
    assert _idle_minutes_for(cfg_selected, (u1, a1, u2)) == 45
    unmatched = Message("u0", "unbound", 0, t0, "user", "conversation", Input((ContentPart("text", "hi"),)))
    assert _idle_minutes_for(cfg_selected, (unmatched,)) == 0
    assert _idle_minutes_for(Config.model_validate({"idle_scope": "all", "idle_trigger_minutes": 30}), (unmatched,)) == 30
    assert _idle_minutes_for(Config.model_validate({"idle_scope": "off", "idle_sessions": ["s"]}), (u1, a1, u2)) == 0


def test_background_idle_due_and_boundary_contract() -> None:
    """后台空闲扫盘在已结算 Output 静默达标后触发，写入边界后不重复触发。"""
    from datetime import UTC, datetime, timedelta
    from plugins.context.contract import background_idle_anchor, background_idle_due
    from plugins.compaction.plugin import _check_boundary
    from plugins.ledger.contract import ContentPart, Input, Message, Output

    t0 = datetime(2026, 11, 14, 0, 0, tzinfo=UTC)
    u1 = Message("u1", "s", 0, t0, "user", "conversation", Input((ContentPart("text", "hi"),)))
    a1 = Message("a1", "s", 1, t0 + timedelta(minutes=1), "assistant", "conversation",
                 Output((ContentPart("text", "done"),), "complete"))
    b1 = Message("b1", "s", 2, t0 + timedelta(minutes=62), "compaction", "compaction",
                 Output((ContentPart("compaction.boundary", {
                     "reference": "r1",
                     "trigger": "idle",
                     "mode": "background",
                     "generation": 1,
                     "tokens_before": 300_000,
                     "tokens_after": 120_000,
                 }),), "complete"))

    assert background_idle_anchor((u1,)) is None
    assert background_idle_anchor((u1, a1)) == "a1"
    assert not background_idle_due(250_000, (u1, a1), now=t0 + timedelta(minutes=60), idle_minutes=60, idle_tokens=200_000)
    assert background_idle_due(250_000, (u1, a1), now=t0 + timedelta(minutes=61), idle_minutes=60, idle_tokens=200_000)
    assert not background_idle_due(199_999, (u1, a1), now=t0 + timedelta(minutes=90), idle_minutes=60, idle_tokens=200_000)
    assert background_idle_anchor((u1, a1, b1)) is None
    assert not background_idle_due(250_000, (u1, a1, b1), now=t0 + timedelta(minutes=180), idle_minutes=60, idle_tokens=200_000)

    u2 = Message("u2", "s", 3, t0 + timedelta(minutes=185), "user", "conversation", Input((ContentPart("text", "next"),)))
    a2 = Message("a2", "s", 4, t0 + timedelta(minutes=186), "assistant", "conversation",
                 Output((ContentPart("text", "ok"),), "complete"))
    assert background_idle_anchor((u1, a1, b1, u2)) is None
    assert background_idle_anchor((u1, a1, b1, u2, a2)) == "a2"
    assert background_idle_due(250_000, (u1, a1, b1, u2, a2), now=t0 + timedelta(minutes=246), idle_minutes=60, idle_tokens=200_000)

    _ = _check_boundary(ContentPart("compaction.boundary", {
        "reference": "r1",
        "trigger": "idle",
        "mode": "background",
        "generation": 1,
        "tokens_before": 300_000,
        "tokens_after": 120_000,
    }))
    with pytest.raises(ValueError):
        _ = _check_boundary(ContentPart("compaction.boundary", {"reference": "r1"}))


def test_background_visible_messages_hides_unfinished_foreign_tails() -> None:
    """后台扫盘隐藏其他来源未收尾的消息，保留已结算消息。"""
    from datetime import UTC, datetime, timedelta
    from plugins.compaction.plugin import _visible_messages
    from plugins.ledger.contract import ContentPart, Control, Input, Message, Output

    t0 = datetime(2026, 11, 14, 0, 0, tzinfo=UTC)
    u1 = Message("u1", "s", 0, t0, "user", "conversation", Input((ContentPart("text", "hi"),)))
    a1 = Message("a1", "s", 1, t0 + timedelta(seconds=1), "assistant", "conversation",
                 Output((ContentPart("text", "done"),), "complete"))
    sched_in = Message("s1", "s", 2, t0 + timedelta(seconds=2), "scheduler", "scheduler",
                       Input((ContentPart("text", "pending"),)))
    sched_part = Message("s2", "s", 3, t0 + timedelta(seconds=3), "scheduler", "scheduler",
                         Output((ContentPart("text", "streaming"),), "continue"))
    watch_done = Message("w1", "s", 4, t0 + timedelta(seconds=4), "watch", "watch",
                         Output((ContentPart("text", "ok"),), "complete"))

    visible = _visible_messages((u1, a1, sched_in, sched_part, watch_done), "conversation")
    assert [m.message_id for m in visible] == ["u1", "a1", "w1"]
    sched_fail = Message("s_fail", "s", 3, t0 + timedelta(seconds=3), "scheduler", "scheduler",
                         Control("failure", 2, "err"))
    assert [m.message_id for m in _visible_messages((u1, a1, sched_in, sched_fail), "conversation")] == [
        "u1", "a1", "s1", "s_fail",
    ]



