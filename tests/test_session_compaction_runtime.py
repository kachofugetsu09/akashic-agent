from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, cast

import pytest

from agent.core.passive_turn import DefaultReasoner
from agent.control.context import running_turn_id
from agent.core.runtime_support import SessionLike, ToolDiscoveryState
from agent.looping.ports import LLMConfig
from agent.plugin_composition import (
    CONTEXT_PROJECTION_FACTS,
    PROVIDER_REQUEST_PROJECTION,
    BoundModelDescriptor,
    ModelRequest,
    ModelRole,
    ProviderTurnInput,
)
from agent.plugins.manager import PluginManager
from plugins.compaction.engine import (
    CommittedContextUnit,
    ContextCompaction,
    ContextCompactionError,
    ContextCompactor,
    ContextPayloadSegments,
    SUMMARY_HEADINGS,
    canonical_source_plan,
    compaction_scope_id,
    compaction_source_ref,
    normalize_session_created_at,
    source_plan_digest,
    _selection_digest,
)
from agent.plugin_composition import LLMResponse
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from plugins.compaction.runtime import (
    CompactionProjection,
    SessionCompactionRuntime,
    _receipt_digest,
    _receipt_payload,
)
from agent.plugin_composition import SessionCompactionStorage


class _ReceiptAdapter:
    def __init__(self, markdown: object) -> None:
        self._markdown = markdown

    def read(self, source_ref: str) -> dict[str, object] | None:
        return self._markdown.read_compaction_receipt(source_ref)  # type: ignore[attr-defined,no-any-return]

    def write(
        self,
        source_ref: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return self._markdown.write_compaction_receipt(source_ref, payload)  # type: ignore[attr-defined,no-any-return]

    def list_all(self) -> tuple[dict[str, object], ...]:
        return tuple(  # type: ignore[attr-defined]
            dict(payload) for payload in self._markdown.receipts.values()
        )


def _runtime(
    manager: SessionManager,
    markdown: object,
    session_key: str = "session",
) -> SessionCompactionRuntime:
    if session_key == "session" and manager.control_store.get_session_meta(
        session_key
    ) is None:
        keys = tuple(manager._cache)
        if len(keys) != 1:
            raise RuntimeError("test runtime needs one explicit Session scope")
        session_key = keys[0]
    session = manager.get_existing(session_key)
    return SessionCompactionRuntime(
        storage=SessionCompactionStorage(manager).scope(
            session.issue_projection_grant(running_turn_id.get())
        ),
        receipts=_ReceiptAdapter(markdown),
    )
from session.manager import Session, SessionManager
from session.store import SessionCompaction
from session.store import (
    CompactionHead,
    CompactionPrepare,
    SessionCompactionPrepareConflictError,
)
from plugins.compaction.plugin import (
    _CompactionTurn,
    _DetachedSession,
    _PublishedProjection,
)
from plugins.compaction import plugin as compaction_plugin
from plugins.compaction.receipts import SqliteCompactionReceipts
from tests.model_plugin_fakes import BoundChatModelFake

SessionManagerFactory = Callable[[Path], SessionManager]


@dataclass(frozen=True)
class CompactionMarkdownDraft:
    source_ref: str
    history_entry_payloads: tuple[tuple[str, int], ...] = ()
    pending_items: str = ""


@pytest.fixture(autouse=True)
def _active_projection_turn() -> Generator[None, None, None]:
    token = running_turn_id.set("test:compaction")
    try:
        yield
    finally:
        running_turn_id.reset(token)


@pytest.fixture
def session_manager_factory() -> Generator[SessionManagerFactory, None, None]:
    """Create and close every SessionManager owned by one test."""

    managers: list[SessionManager] = []

    def factory(path: Path) -> SessionManager:
        manager = SessionManager(path)
        managers.append(manager)
        return manager

    yield factory
    for manager in managers:
        manager.close()


class _MarkdownReceiptProbe:
    def __init__(self) -> None:
        self.receipts: dict[str, dict[str, object]] = {}
        self.commit_count = 0
        self.fail_after_commit = False

    def read_compaction_receipt(self, source_ref: str):
        return self.receipts.get(source_ref)

    def write_compaction_receipt(self, source_ref: str, payload: dict[str, object]):
        self.receipts[source_ref] = dict(payload)
        return dict(payload)

    async def commit_compaction_markdown(self, draft: CompactionMarkdownDraft):
        self.commit_count += 1
        if self.fail_after_commit:
            raise RuntimeError("simulated crash after Markdown side effect")


class _MarkdownCompactionProbe(_MarkdownReceiptProbe):
    def __init__(self) -> None:
        super().__init__()
        self.prepare_count = 0

    async def prepare_compaction_markdown(
        self,
        selected_source_messages,
        *,
        source_ref: str,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> CompactionMarkdownDraft:
        self.prepare_count += 1
        assert selected_source_messages
        return CompactionMarkdownDraft(source_ref=source_ref)


class _BlockingMarkdownProbe(_MarkdownCompactionProbe):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def prepare_compaction_markdown(self, *args, **kwargs):
        self.prepare_count += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return CompactionMarkdownDraft(source_ref=str(kwargs["source_ref"]))


class _OrderedMarkdownProbe(_MarkdownCompactionProbe):
    def __init__(self) -> None:
        super().__init__()
        self.source_refs: list[str] = []

    async def prepare_compaction_markdown(self, *args, **kwargs):
        source_ref = str(kwargs["source_ref"])
        self.source_refs.append(source_ref)
        if len(self.source_refs) == 1:
            raise RuntimeError("first markdown failed")
        return CompactionMarkdownDraft(source_ref=source_ref)


def _seed_two_unit_checkpoint(
    manager: SessionManager,
    session_key: str,
    *,
    suppress_post_commit: bool = False,
) -> tuple[Session, CompactionHead, ContextCompaction, str]:
    """Create one selected unit and one retained unit backed by canonical rows."""

    session = manager.get_or_create(session_key)
    effects = {"post_commit": "suppress"} if suppress_post_commit else None
    if effects is None:
        session.add_message("user", "old user", control_turn_id="turn-old")
        session.add_message("assistant", "old reply", control_turn_id="turn-old")
    else:
        session.add_message(
            "user", "old user", control_turn_id="turn-old", effects=effects
        )
        session.add_message(
            "assistant", "old reply", control_turn_id="turn-old", effects=effects
        )
    session.add_message("user", "tail user", control_turn_id="turn-tail")
    session.add_message("assistant", "tail reply", control_turn_id="turn-tail")
    manager.save(session)
    head = manager.control_store.get_compaction_head(session.key)
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    selected_unit, retained_unit = session.history_units()
    selected = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": f"{selected_unit.source_from_seq}:"
            f"{selected_unit.consolidated_through_seq}:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(
            selected_unit.messages,
            selected_unit.message_refs,
        )
    )
    retained_tail = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": f"{retained_unit.source_from_seq}:"
            f"{retained_unit.consolidated_through_seq}:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(
            retained_unit.messages,
            retained_unit.message_refs,
        )
    )
    checkpoint = ContextCompaction(
        summary="\n".join(SUMMARY_HEADINGS),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        trigger="soft_limit",
        context_window=100,
        soft_limit_tokens=74,
        hard_input_tokens=90,
        keep_recent_tokens=20,
        estimated_tokens_before=80,
        estimated_tokens_after=40,
        source_from_seq=selected_unit.source_from_seq,
        consolidated_through_seq=selected_unit.consolidated_through_seq,
        source_message_ids=selected_unit.source_message_ids,
        retained_tail=retained_tail,
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="source-fence",
        selected_source_messages=selected,
    )
    retained_id = str(retained_unit.message_refs[0][0])
    return session, head, checkpoint, retained_id


class _CountingProvider:
    context_window: int = 1000
    runtime_id: str = "runtime"

    def __init__(
        self, *, context_window: int = 1000, runtime_id: str = "runtime"
    ) -> None:
        self.context_window = context_window
        self.runtime_id = runtime_id
        self.calls = 0

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return sum(int(message.get("tokens", 1)) for message in messages)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return sum(int(message.get("tokens", 1)) for message in messages)

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="\n".join(SUMMARY_HEADINGS))

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return BoundChatModelFake(
            self,
            model=str(getattr(self, "model", "model")),
        ).descriptor

    @property
    def max_tool_schemas(self) -> int | None:
        return None

    async def complete(self, request: ModelRequest) -> LLMResponse:
        return await BoundChatModelFake(
            self,
            model=str(getattr(self, "model", "model")),
        ).complete(request)


def _build_reasoner(
    provider: _CountingProvider,
    runtime: object,
    *,
    model: str = "model",
    max_tokens: int = 0,
    fallback_provider: _CountingProvider | None = None,
    fallback_model: str = "model",
) -> DefaultReasoner:
    reasoner = DefaultReasoner(
        llm_config=LLMConfig(max_tokens=max_tokens),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )
    reasoner._test_agent_model = BoundChatModelFake(provider, model=model)
    reasoner._test_fallback_model = BoundChatModelFake(
        fallback_provider or provider,
        model=fallback_model,
        role=ModelRole.DEFAULT,
    )
    return reasoner


def _plugin_turn(
    runtime: object,
    session: object,
    projection: CompactionProjection,
    *,
    keep_recent_tokens: int = 1,
) -> _CompactionTurn:
    async def observe(_key: object, _payload: object) -> None:
        return None

    created_at = getattr(session, "created_at")
    return _CompactionTurn(
        cast(Any, SimpleNamespace(observe=observe)),
        cast(Any, runtime),
        _DetachedSession(
            key=str(getattr(session, "key")),
            created_at=(
                created_at.isoformat()
                if isinstance(created_at, datetime)
                else str(created_at)
            ),
            units=projection.segments.committed_units,
        ),
        projection,
        keep_recent_tokens,
    )


class _ContentLengthProvider(_CountingProvider):
    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return self.estimate_context_tokens(messages, [])


class _NoopCompactionRuntime:
    """Provide the narrow runtime port for state-builder-only tests."""

    async def projection(
        self,
        session: SessionLike,
        *,
        prefix: list[dict[str, Any]],
        current_anchor: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> CompactionProjection:
        raise AssertionError(f"projection unexpectedly called for {session.key}")

    async def recover_pending(self, session: SessionLike) -> SessionCompaction | None:
        raise AssertionError(f"recovery unexpectedly called for {session.key}")

    async def commit_checkpoint(
        self,
        session: SessionLike,
        checkpoint: ContextCompaction,
        *,
        head: CompactionHead,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> SessionCompaction:
        raise AssertionError(
            f"checkpoint unexpectedly committed for {session.key}:{checkpoint.source_ref}"
        )


class _GateProvider(_CountingProvider):
    def __init__(self) -> None:
        super().__init__(context_window=250, runtime_id="gate-runtime")
        self.requests: list[dict[str, object]] = []

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        if len(messages) == 1 and str(messages[0].get("content", "")).startswith(
            "更新当前长任务"
        ):
            return 10
        total = 0
        for message in messages:
            role = message.get("role")
            if role == "system":
                total += 20
            elif role == "user":
                total += 80 if str(message.get("content", "")).startswith("old") else 10
            else:
                total += 10
        return total + len(tools)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return self.estimate_context_tokens(messages, [])

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        self.requests.append(kwargs)
        return LLMResponse(content="\n".join(SUMMARY_HEADINGS))


class _ScopedCompactionProvider(_GateProvider):
    def __init__(self) -> None:
        super().__init__()
        self.context_window = 128

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        if any(
            "<session-context-compaction>" in str(message.get("content", ""))
            for message in messages
        ):
            return 5
        return 100

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return 1

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        self.requests.append(kwargs)
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            raise AssertionError("summary request 缺少 messages")
        contents: list[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise AssertionError("summary request message 非 Mapping")
            contents.append(str(message.get("content", "")))
        content = "\n".join(contents)
        if "Closed history to consolidate" in content:
            marker = "A_SENTINEL" if "a-" in content else "B_SENTINEL"
            return LLMResponse(content="\n".join(SUMMARY_HEADINGS) + f"\n{marker}")
        marker = "A_SENTINEL" if "a-" in content else "B_SENTINEL"
        return LLMResponse(content=f"reply-{marker}")


def _checkpoint_source_mutation_digest(
    manager: SessionManager,
    session_key: str,
    checkpoint: ContextCompaction,
) -> str:
    source_ids = tuple(
        dict.fromkeys(
            [
                *checkpoint.source_message_ids,
                *(str(item["id"]) for item in checkpoint.retained_tail),
            ]
        )
    )
    return manager.control_store.source_mutation_digest(session_key, source_ids)


def _seed_receipt(
    tmp_path: Path,
    manager_factory: SessionManagerFactory,
    *,
    version: int = 4,
) -> tuple[SessionManager, _MarkdownReceiptProbe, str]:
    manager = manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "persisted")
    manager.save(session)
    probe = _MarkdownReceiptProbe()
    head = manager.control_store.get_compaction_head(session.key)
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    unit = session.history_units()[0]
    selected_source_messages = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": "0:0:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(unit.messages, unit.message_refs)
    )
    checkpoint = ContextCompaction(
        summary="\n".join(SUMMARY_HEADINGS),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        trigger="soft_limit",
        context_window=100,
        soft_limit_tokens=74,
        hard_input_tokens=90,
        keep_recent_tokens=20,
        estimated_tokens_before=80,
        estimated_tokens_after=40,
        source_from_seq=unit.source_from_seq,
        consolidated_through_seq=unit.consolidated_through_seq,
        source_message_ids=unit.source_message_ids,
        retained_tail=(),
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="selection",
        selected_source_messages=selected_source_messages,
    )
    manager.control_store.prepare_compaction(
        session_key=session.key,
        session_created_at=session.created_at.isoformat(),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        source_ref=source_ref,
        source_from_seq=checkpoint.source_from_seq,
        consolidated_through_seq=checkpoint.consolidated_through_seq,
        source_message_ids=checkpoint.source_message_ids,
        retained_tail=checkpoint.retained_tail,
    )
    receipt = _receipt_payload(
        checkpoint,
        session_key=session.key,
        head=head,
        model_runtime_id="runtime",
        model="model",
        session_created_at=normalize_session_created_at(session.created_at),
        source_mutation_digest=_checkpoint_source_mutation_digest(
            manager,
            session.key,
            checkpoint,
        ),
        scope_channel="",
        scope_chat_id="",
    )
    if version == 2:
        receipt["version"] = 2
        receipt["markdown_draft"] = {
            "source_ref": source_ref,
            "history_entry_payloads": [],
            "pending_items": "",
            "conversation": "",
            "scope_channel": "",
            "scope_chat_id": "",
        }
        receipt.pop("scope_channel")
        receipt.pop("scope_chat_id")
        receipt.pop("source_mutation_digest")
        receipt["digest"] = _receipt_digest(receipt)
    elif version == 3:
        receipt["version"] = 3
        receipt["digest"] = _receipt_digest(receipt)
    probe.receipts[source_ref] = receipt
    return manager, probe, source_ref


def test_compaction_scope_separates_session_incarnations_and_reloads_stably() -> None:
    provider = _CountingProvider()
    unit = CommittedContextUnit(
        source_from_seq=0,
        consolidated_through_seq=0,
        source_message_ids=("m0",),
        messages=({"role": "user", "content": "same"},),
        message_refs=(("m0", 0),),
    )
    first = compaction_scope_id("same-key", "2026-08-08T00:00:00+00:00")
    reloaded = compaction_scope_id("same-key", "2026-08-08T08:00:00+08:00")
    recreated = compaction_scope_id("same-key", "2026-08-09T00:00:00+00:00")
    assert first == reloaded
    assert first != recreated
    assert compaction_source_ref(first, 1) != compaction_source_ref(recreated, 1)
    digest_kwargs = {
        "provider": provider,
        "model": "model",
        "soft_limit_tokens": 74,
        "hard_input_tokens": 90,
        "keep_recent_tokens": 20,
    }
    assert _selection_digest(
        (unit,), (), scope_id=first, **digest_kwargs
    ) != _selection_digest((unit,), (), scope_id=recreated, **digest_kwargs)


def test_receipt_recovery_skips_provider_calls(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    runtime = _runtime(manager, markdown)
    session = manager.get_existing("session")
    provider_calls = 0

    recovered = asyncio.run(runtime.recover_pending(session))

    assert recovered is not None
    assert provider_calls == 0
    assert session.last_consolidated == recovered.generation
    assert markdown.commit_count == 0
    receipt = markdown.receipts[source_ref]
    raw_checkpoint = receipt["checkpoint"]
    assert isinstance(raw_checkpoint, dict)
    raw_plan = raw_checkpoint["selected_source_messages"]
    assert isinstance(raw_plan, list)
    persisted = manager.control_store.get_compaction("session", recovered.generation)
    assert persisted is not None
    assert persisted.source_plan_digest == source_plan_digest(
        canonical_source_plan(raw_plan)
    )
    assert (
        manager.control_store.get_compaction_prepare("session", source_ref=source_ref)
        is None
    )


def test_v2_receipt_recovery_defers_markdown_to_durable_fact_reader(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, _ = _seed_receipt(
        tmp_path,
        session_manager_factory,
        version=2,
    )
    runtime = _runtime(manager, markdown)

    recovered = asyncio.run(runtime.recover_pending(manager.get_existing("session")))

    assert recovered is not None
    assert markdown.commit_count == 0
    assert len(markdown.receipts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "published"), ((2, True), (3, False), (4, True))
)
async def test_formal_compaction_plugin_only_publishes_plugin_owned_receipts(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
    version: int,
    published: bool,
) -> None:
    manager, probe, source_ref = _seed_receipt(
        tmp_path,
        session_manager_factory,
        version=version,
    )
    workspace = Path(manager.control_store.db_path).parent
    receipts = SqliteCompactionReceipts(
        workspace / "memory" / "consolidation_writes.db"
    )
    receipts.write(source_ref, probe.receipts[source_ref])
    plugins = PluginManager(
        plugin_dirs=[Path(compaction_plugin.__file__).parent],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=manager,
        installed_cache_root=tmp_path / "plugin-cache",
    )
    await plugins.load_all()
    snapshot = plugins.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    root = snapshot.composition_root.context
    service = root.require(PROVIDER_REQUEST_PROJECTION)
    session = manager.get_existing("session")
    grant = session.issue_projection_grant(running_turn_id.get())
    storage = SessionCompactionStorage(manager).scope(grant)

    _ = await service.open_turn(
        ProviderTurnInput(
            session_key=session.key,
            session_created_at=session.created_at.isoformat(),
            history_units=storage.history_units(session.key),
            access_grant=grant,
        )
    )

    assert manager.control_store.get_compaction_head(session.key).parent_generation == 1
    facts = root.require(CONTEXT_PROJECTION_FACTS).list_committed(
        grant,
        session_key=session.key,
    )
    assert [(fact.source_ref, fact.generation) for fact in facts] == (
        [(source_ref, 1)] if published else []
    )
    if published:
        assert f'"version":{version}' in facts[0].checkpoint_json
    await plugins.terminate_all()


@pytest.mark.asyncio
async def test_invalidated_compaction_receipt_is_not_reemitted_as_durable_fact(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, probe, source_ref = _seed_receipt(
        tmp_path, session_manager_factory, version=2
    )
    workspace = Path(manager.control_store.db_path).parent
    receipts = SqliteCompactionReceipts(
        workspace / "memory" / "consolidation_writes.db"
    )
    receipts.write(source_ref, probe.receipts[source_ref])
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "UPDATE session_compactions SET invalidated_at = ?, invalidated_reason = ? "
            "WHERE session_key = ? AND source_ref = ?",
            (datetime.now(UTC).isoformat(), "interaction_deleted:test", "session", source_ref),
        )
        manager.control_store._conn.commit()
    plugins = PluginManager(
        plugin_dirs=[Path(compaction_plugin.__file__).parent],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=manager,
        installed_cache_root=tmp_path / "plugin-cache",
    )
    await plugins.load_all()
    snapshot = plugins.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    facts = snapshot.composition_root.context.require(CONTEXT_PROJECTION_FACTS)
    session = manager.get_existing("session")
    grant = session.issue_projection_grant(running_turn_id.get())

    assert facts.list_committed(grant, session_key=session.key) == ()
    session.revoke_projection_grant(grant)
    await plugins.terminate_all()


def test_v2_receipt_without_prepare_still_fails_loud(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(
        tmp_path,
        session_manager_factory,
        version=2,
    )
    prepare = manager.control_store.get_compaction_prepare(
        "session",
        source_ref=source_ref,
    )
    assert prepare is not None
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "DELETE FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()
    runtime = _runtime(manager, markdown)

    with pytest.raises(RuntimeError, match="durable prepare 缺失"):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))


def _seed_orphan_prepare(
    tmp_path: Path,
    manager_factory: SessionManagerFactory,
) -> tuple[SessionManager, SessionCompactionRuntime, CompactionPrepare]:
    manager = manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "prepared")
    manager.save(session)
    head = manager.control_store.get_compaction_head(session.key)
    unit = session.history_units()[0]
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    prepare = manager.control_store.prepare_compaction(
        session_key=session.key,
        session_created_at=session.created_at.isoformat(),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        source_ref=source_ref,
        source_from_seq=unit.source_from_seq,
        consolidated_through_seq=unit.consolidated_through_seq,
        source_message_ids=unit.source_message_ids,
        retained_tail=(),
    )
    runtime = _runtime(manager, _MarkdownReceiptProbe())
    return manager, runtime, prepare


def test_prepare_without_receipt_is_released_on_recovery(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, runtime, prepare = _seed_orphan_prepare(tmp_path, session_manager_factory)
    session = manager.get_existing("session")
    head = manager.control_store.get_compaction_head(session.key)

    assert asyncio.run(runtime.recover_pending(session)) is None
    assert (
        manager.control_store.get_compaction_prepare(
            session.key, source_ref=prepare.source_ref
        )
        is None
    )
    assert manager.control_store.get_compaction_head(session.key) == head


@pytest.mark.parametrize(
    ("column", "value", "match"),
    (
        ("source_message_ids_json", "[]", "source_message_ids"),
        (
            "retained_tail_json",
            '[{"id":"session:0","seq":true,"message":{},"unit_ref":"0:0:0"}]',
            "retained_tail",
        ),
        (
            "retained_tail_json",
            '[{"id":"session:0","seq":0,"message":{}}]',
            "retained_tail",
        ),
        ("session_created_at", "", "identity"),
        ("prepared_at", "", "prepared_at"),
    ),
)
def test_corrupt_prepare_without_receipt_fails_loud_and_keeps_row(
    tmp_path: Path,
    column: str,
    value: object,
    match: str,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, runtime, prepare = _seed_orphan_prepare(tmp_path, session_manager_factory)
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            f"UPDATE session_compaction_prepares SET {column} = ? "
            "WHERE session_key = ? AND generation = ?",
            (value, prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()

    with pytest.raises(ValueError, match=match):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))
    with manager.control_store._lock:
        raw = manager.control_store._conn.execute(
            "SELECT 1 FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        ).fetchone()
    assert raw is not None


def test_v3_receipt_without_prepare_is_audit_only(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    prepare = manager.control_store.get_compaction_prepare(
        "session", source_ref=source_ref
    )
    assert prepare is not None
    # Explicit SQL corruption simulates a receipt whose durable prepare vanished.
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "DELETE FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()
    runtime = _runtime(manager, markdown)

    assert asyncio.run(runtime.recover_pending(manager.get_existing("session"))) is None
    assert markdown.commit_count == 0


def test_v3_receipt_without_prepare_still_validates_identity(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    prepare = manager.control_store.get_compaction_prepare(
        "session", source_ref=source_ref
    )
    assert prepare is not None
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "DELETE FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()
    receipt = markdown.receipts[source_ref]
    receipt["session_key"] = "other-session"
    receipt["digest"] = _receipt_digest(receipt)
    runtime = _runtime(manager, markdown)

    with pytest.raises(ValueError, match="session_key 冲突"):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))


def test_v3_recovery_rejects_raw_source_mutation(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    receipt = markdown.receipts[source_ref]
    checkpoint = cast(dict[str, Any], receipt["checkpoint"])
    source_ids = cast(list[str], checkpoint["source_message_ids"])
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "UPDATE messages SET ts = ts || '-tampered' WHERE id = ?",
            (source_ids[0],),
        )
        manager.control_store._conn.commit()
    runtime = _runtime(manager, markdown)

    with pytest.raises(RuntimeError, match="source snapshot"):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))
    assert (
        manager.control_store.get_compaction_prepare("session", source_ref=source_ref)
        is not None
    )


def test_suppressed_turn_commit_advances_ledger_without_markdown(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message(
        "user",
        "excluded",
        effects={"post_commit": "suppress"},
    )
    manager.save(session)
    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)
    head = manager.control_store.get_compaction_head(session.key)
    unit = session.history_units()[0]
    message = unit.messages[0]
    message_id, message_seq = unit.message_refs[0]
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    checkpoint = ContextCompaction(
        summary="\n".join(SUMMARY_HEADINGS),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        trigger="soft_limit",
        context_window=100,
        soft_limit_tokens=74,
        hard_input_tokens=90,
        keep_recent_tokens=20,
        estimated_tokens_before=80,
        estimated_tokens_after=40,
        source_from_seq=message_seq,
        consolidated_through_seq=message_seq,
        source_message_ids=(message_id,),
        retained_tail=(),
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="selection",
        selected_source_messages=(
            {
                "id": message_id,
                "seq": message_seq,
                "unit_ref": "0:0:0",
                "message": dict(message),
            },
        ),
    )

    row = asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))

    assert row.generation == 1
    expected_digest = source_plan_digest(
        canonical_source_plan(checkpoint.selected_source_messages)
    )
    assert row.source_plan_digest == expected_digest
    reloaded_row = manager.control_store.get_compaction(session.key, row.generation)
    assert reloaded_row is not None
    assert reloaded_row.source_plan_digest == expected_digest
    assert manager.control_store.get_compaction_head(session.key).parent_generation == 1
    assert markdown.prepare_count == 0
    assert markdown.commit_count == 0
    assert (
        manager.control_store.get_compaction_prepare(session.key, source_ref=source_ref)
        is None
    )


def test_receipt_recovery_ignores_unrelated_session_metadata_change(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, _ = _seed_receipt(tmp_path, session_manager_factory)
    session = manager.get_existing("session")
    session.metadata["plugin_state"] = True
    manager.save(session)
    runtime = _runtime(manager, markdown)

    recovered = asyncio.run(runtime.recover_pending(session))

    assert recovered is not None
    assert recovered.generation == 1
    assert manager.control_store.get_compaction_head(session.key).parent_generation == 1
    assert markdown.commit_count == 0
    persisted = manager.control_store.get_compaction(session.key, recovered.generation)
    assert persisted is not None
    assert persisted.source_plan_digest == str(
        markdown.receipts[recovered.source_ref]["source_plan_digest"]
    )


def test_v3_receipt_recovery_retries_ledger_without_markdown(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    runtime = _runtime(manager, markdown)
    session = manager.get_existing("session")
    provider_calls = 0

    original_persist = manager.control_store.persist_compaction
    persist_calls = 0

    def fail_once(*args, **kwargs):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise RuntimeError("simulated crash before SQLite commit")
        return original_persist(*args, **kwargs)

    manager.control_store.persist_compaction = fail_once  # type: ignore[method-assign]
    try:
        asyncio.run(runtime.recover_pending(session))
    except RuntimeError as exc:
        assert "SQLite" in str(exc)
    else:
        raise AssertionError("expected SQLite crash")
    assert (
        manager.control_store.get_compaction_prepare(session.key, source_ref=source_ref)
        is not None
    )
    manager.control_store.persist_compaction = original_persist  # type: ignore[method-assign]
    manager.invalidate(session.key)
    resumed = manager.get_existing(session.key)
    resumed_runtime = _runtime(manager, markdown)
    recovered = asyncio.run(resumed_runtime.recover_pending(resumed))

    assert recovered is not None
    assert provider_calls == 0
    assert resumed.last_consolidated == recovered.generation
    persisted = manager.control_store.get_compaction(
        resumed.key,
        recovered.generation,
    )
    assert persisted is not None
    assert persisted.source_plan_digest == str(
        markdown.receipts[source_ref]["source_plan_digest"]
    )
    assert markdown.commit_count == 0
    assert (
        manager.control_store.get_compaction_prepare(resumed.key, source_ref=source_ref)
        is None
    )


def test_tampered_receipt_is_rejected_before_markdown_or_ledger(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    runtime = _runtime(manager, markdown)
    checkpoint = markdown.receipts[source_ref]["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["summary"] = "tampered"
    session = manager.get_existing("session")

    with pytest.raises(ValueError, match="digest"):
        asyncio.run(runtime.recover_pending(session))

    assert manager.control_store.get_compaction_head("session").parent_generation == 0
    assert markdown.commit_count == 0


def test_pending_prepare_rejects_source_deletion(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, _ = _seed_receipt(tmp_path, session_manager_factory)
    session = manager.get_existing("session")
    message_id = str(session.messages[0]["id"])
    with pytest.raises(
        SessionCompactionPrepareConflictError,
        match="pending compaction prepare",
    ):
        manager.control_store.delete_messages_batch([message_id])
    assert manager.control_store.get_compaction_head("session").parent_generation == 0
    assert markdown.commit_count == 0


def test_retained_tail_without_unit_ref_is_rejected(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, _, _ = _seed_receipt(tmp_path, session_manager_factory)
    head = manager.control_store.get_compaction_head("session")
    with pytest.raises(ValueError, match="unit_ref"):
        manager.control_store.persist_compaction(
            session_key="session",
            trigger="soft_limit",
            summary="\n".join(SUMMARY_HEADINGS),
            source_ref="bad-unit-ref",
            source_plan_digest="a" * 64,
            source_from_seq=0,
            consolidated_through_seq=0,
            source_message_ids=("session:0",),
            retained_tail=({"id": "session:0", "seq": 0, "message": {"role": "user"}},),
            model_runtime_id="runtime",
            model="model",
            context_window=100,
            threshold_tokens=74,
            hard_input_tokens=90,
            keep_recent_tokens=20,
            tokens_before=80,
            tokens_after=40,
            summary_usage={},
            parent_generation=head.parent_generation,
            generation=head.next_generation,
        )


def test_context_compactor_receipt_resume_does_not_call_summary_provider() -> None:
    provider = _CountingProvider()
    units = tuple(
        CommittedContextUnit(
            source_from_seq=index,
            consolidated_through_seq=index,
            source_message_ids=(f"m{index}",),
            messages=({"role": "user", "content": f"u{index}", "tokens": 400},),
            message_refs=((f"m{index}", index),),
        )
        for index in (0, 1)
    )
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=units,
        current_anchor=({"role": "user", "content": "query", "tokens": 1},),
    )
    first = ContextCompactor(
        provider=provider,
        scope_id="session",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
    )
    first_messages = segments.flatten()
    first_result = asyncio.run(
        first.prepare(first_messages, pending_start=3, tools=[], force=True)
    )
    assert first_result.checkpoint is not None
    assert provider.calls == 1
    head = CompactionHead(
        session_key="session",
        parent_generation=0,
        next_generation=1,
    )
    receipt = _receipt_payload(
        first_result.checkpoint,
        session_key="session",
        head=head,
        model_runtime_id="runtime",
        model="model",
        session_created_at="2026-08-08T00:00:00+00:00",
        source_mutation_digest="0" * 64,
        scope_channel="",
        scope_chat_id="",
    )
    second = ContextCompactor(
        provider=provider,
        scope_id="session",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
        receipt_loader=lambda source_ref: receipt,
    )
    second_messages = segments.flatten()
    resumed = asyncio.run(
        second.prepare(second_messages, pending_start=3, tools=[], force=True)
    )

    assert resumed.checkpoint is not None
    assert resumed.checkpoint.summary == first_result.checkpoint.summary
    assert provider.calls == 1


def test_generation_zero_windows_history_before_provider_payload() -> None:
    provider = _CountingProvider(context_window=100_000)
    runtime = _NoopCompactionRuntime()
    reasoner = _build_reasoner(
        provider,
        runtime,
    )
    units = tuple(
        CommittedContextUnit(
            source_from_seq=index,
            consolidated_through_seq=index,
            source_message_ids=(f"m{index}",),
            messages=(
                {
                    "role": "user",
                    "content": f"history-{index}",
                    "tokens": 60_000,
                },
            ),
            message_refs=((f"m{index}", index),),
        )
        for index in range(4)
    )
    projection = CompactionProjection(
        segments=ContextPayloadSegments(
            prefix=(),
            committed_units=units,
            current_anchor=(),
        ),
        active=None,
        head=CompactionHead(
            session_key="session",
            parent_generation=0,
            next_generation=1,
        ),
    )
    initial_messages = [
        {"role": "system", "content": "system", "tokens": 1},
        *[dict(unit.messages[0]) for unit in units],
        {"role": "user", "content": "query", "tokens": 1},
    ]
    session = SimpleNamespace(
        key="session",
        created_at=datetime.now(UTC),
    )

    state = reasoner._build_request_state(
        agent_model=reasoner._test_agent_model,
        fallback_model=reasoner._test_fallback_model,
        projection=_plugin_turn(runtime, session, projection),
        initial_messages=initial_messages,
        history_count=4,
        attempt_replay=[],
        prior_tool_groups=0,
        channel="",
        chat_id="",
    )

    assert "history-0" not in str(initial_messages)
    assert "history-1" not in str(initial_messages)
    result = asyncio.run(
        state.gate._compactor.prepare(
            initial_messages,
            pending_start=state.gate._compactor.pending_start,
            tools=[],
            force=True,
        )
    )
    assert result.checkpoint is not None
    assert result.checkpoint.source_from_seq == 2


def test_generation_zero_real_session_stays_append_only_then_compacts_incrementally(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session = manager.get_or_create("session")
    for index in range(4):
        session.add_message("user", f"history-{index}-" + ("x" * 60_000))
    manager.save(session)
    provider = _ContentLengthProvider(context_window=150_000)
    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)
    reasoner = _build_reasoner(
        provider,
        runtime,
    )

    def message_rows() -> list[tuple[object, ...]]:
        with manager.control_store._lock:
            rows = manager.control_store._conn.execute(
                "SELECT id, session_key, seq, role, content, tool_chain, extra, ts "
                "FROM messages WHERE session_key = ? ORDER BY seq",
                (session.key,),
            ).fetchall()
        return [tuple(row) for row in rows]

    async def compact_once() -> ContextCompaction:
        projection = await runtime.projection(
            session,
            prefix=[],
            current_anchor=[],
            pending=[],
        )
        history = [
            *projection.segments.prefix,
            *[
                message
                for unit in projection.segments.committed_units
                for message in unit.messages
            ],
        ]
        payload = [
            {"role": "system", "content": "root"},
            *history,
            {"role": "user", "content": "current query"},
        ]
        state = reasoner._build_request_state(
            agent_model=reasoner._test_agent_model,
            fallback_model=reasoner._test_fallback_model,
            projection=_plugin_turn(runtime, session, projection),
            initial_messages=payload,
            history_count=len(history),
            attempt_replay=[],
            prior_tool_groups=0,
            channel="test",
            chat_id="chat",
        )
        result = await state.gate._compactor.prepare(
            payload,
            pending_start=state.gate._compactor.pending_start,
            tools=[],
            force=True,
        )
        assert result.checkpoint is not None
        await runtime.commit_checkpoint(
            session,
            result.checkpoint,
            head=projection.head,
            scope_channel="test",
            scope_chat_id="chat",
        )
        return result.checkpoint

    async def scenario() -> tuple[ContextCompaction, ContextCompaction]:
        before_first = message_rows()
        first = await compact_once()
        assert message_rows() == before_first
        session.add_message("user", "history-4-" + ("y" * 60_000))
        session.add_message("user", "history-5-" + ("z" * 60_000))
        manager.save(session)
        before_second = message_rows()
        second = await compact_once()
        assert message_rows() == before_second
        return first, second

    first, second = asyncio.run(scenario())

    assert first.source_from_seq == 2
    assert second.generation == 2
    assert second.source_from_seq > first.source_from_seq
    assert len(message_rows()) == 6


def test_session_summary_does_not_swallow_compaction_error() -> None:
    provider = _CountingProvider()
    reasoner = _build_reasoner(
        provider,
        _NoopCompactionRuntime(),
    )

    async def fail_compaction(*args: object, **kwargs: object) -> object:
        raise ContextCompactionError("compaction failed")

    reasoner._call_provider = fail_compaction  # type: ignore[method-assign]

    with pytest.raises(ContextCompactionError, match="compaction failed"):
        asyncio.run(
            reasoner._summarize_incomplete_progress(
                [{"role": "user", "content": "query"}],
                reason="budget",
                iteration=1,
                tools_used=[],
                request_state=cast(Any, object()),
            )
        )


def test_default_reasoner_gate_commits_real_runtime_before_provider_payload(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "old one")
    session.add_message("user", "old two")
    manager.save(session)
    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)
    provider = _GateProvider()
    reasoner = _build_reasoner(provider, runtime, max_tokens=10)
    projection = asyncio.run(
        runtime.projection(session, prefix=[], current_anchor=[], pending=[])
    )
    history = [
        message
        for unit in projection.segments.committed_units
        for message in unit.messages
    ]
    render_payload = [
        {"role": "system", "content": "root"},
        *history,
        {"role": "user", "content": "current request"},
    ]
    state = reasoner._build_request_state(
        agent_model=reasoner._test_agent_model,
        fallback_model=reasoner._test_fallback_model,
        projection=_plugin_turn(runtime, session, projection),
        initial_messages=render_payload,
        history_count=len(history),
        attempt_replay=[],
        prior_tool_groups=0,
        channel="test",
        chat_id="chat",
    )
    state.gate._compactor._keep_recent_tokens = 1
    call_result = asyncio.run(
        reasoner._call_provider(
            state,
            render_payload,
            tools=[],
            max_tokens=10,
        )
    )

    assert call_result.response.content == "\n".join(SUMMARY_HEADINGS)
    assert provider.calls == 2
    assert markdown.commit_count == 0
    assert len(markdown.receipts) == 1
    assert manager.control_store.get_compaction_head(session.key).parent_generation == 1
    assert "current request" not in str(provider.requests[0]["messages"])
    assert provider.requests[1]["messages"] == render_payload
    assert any(
        message.get("role") == "system"
        and "<session-context-compaction>" in str(message.get("content"))
        for message in render_payload
    )


def test_projection_reload_does_not_duplicate_retained_tail_or_new_units(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "old user", control_turn_id="turn-old")
    session.add_message("assistant", "old reply", control_turn_id="turn-old")
    session.add_message("user", "tail user", control_turn_id="turn-tail")
    session.add_message("assistant", "tail reply", control_turn_id="turn-tail")
    manager.save(session)

    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)
    head = manager.control_store.get_compaction_head(session.key)
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    units = session.history_units()
    selected_unit, retained_unit = units[:2]
    selected = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": "0:1:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(
            selected_unit.messages,
            selected_unit.message_refs,
        )
    )
    retained_tail = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": "2:3:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(
            retained_unit.messages,
            retained_unit.message_refs,
        )
    )
    checkpoint = ContextCompaction(
        summary="\n".join(SUMMARY_HEADINGS),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        trigger="soft_limit",
        context_window=100,
        soft_limit_tokens=74,
        hard_input_tokens=90,
        keep_recent_tokens=20,
        estimated_tokens_before=80,
        estimated_tokens_after=40,
        source_from_seq=selected_unit.source_from_seq,
        consolidated_through_seq=selected_unit.consolidated_through_seq,
        source_message_ids=selected_unit.source_message_ids,
        retained_tail=retained_tail,
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="projection-reload",
        selected_source_messages=selected,
    )

    committed = asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))
    expected_digest = source_plan_digest(canonical_source_plan(selected))
    assert committed.source_plan_digest == expected_digest
    session.add_message("user", "new user", control_turn_id="turn-new")
    session.add_message("assistant", "new reply", control_turn_id="turn-new")
    manager.save(session)
    manager.invalidate(session.key)
    reloaded = manager.get_or_create(session.key)

    persisted = manager.control_store.get_compaction(session.key, committed.generation)
    assert persisted is not None
    assert persisted.source_plan_digest == expected_digest

    projection = asyncio.run(
        runtime.projection(reloaded, prefix=[], current_anchor=[], pending=[])
    )
    projected_ids = [
        message_id
        for unit in projection.segments.committed_units
        for message_id in unit.source_message_ids
    ]
    projected_content = [
        str(message.get("content"))
        for unit in projection.segments.committed_units
        for message in unit.messages
    ]

    assert projected_ids == [
        str(session.messages[2]["id"]),
        str(session.messages[3]["id"]),
        str(reloaded.messages[4]["id"]),
        str(reloaded.messages[5]["id"]),
    ]
    assert projected_content == [
        "tail user",
        "tail reply",
        "new user",
        "new reply",
    ]


def test_compaction_rejects_stale_retained_body_without_side_effects(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session, head, checkpoint, _ = _seed_two_unit_checkpoint(manager, "session:stale")
    stale_tail = tuple(
        {
            **item,
            "message": {
                **dict(item["message"]),
                "content": "stale retained body",
            },
        }
        for item in checkpoint.retained_tail
    )
    checkpoint = replace(checkpoint, retained_tail=stale_tail)
    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)

    with pytest.raises(RuntimeError, match="retained rendered message"):
        asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))

    assert markdown.prepare_count == 0
    assert markdown.commit_count == 0
    assert markdown.receipts == {}
    assert (
        manager.control_store.get_compaction_prepare(
            session.key,
            source_ref=checkpoint.source_ref,
        )
        is None
    )
    assert (
        manager.control_store.get_compaction(session.key, head.next_generation) is None
    )
    assert manager.control_store.get_compaction_head(session.key) == head


def test_included_compaction_source_edit_before_prepare_has_no_side_effects(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session, head, checkpoint, retained_id = _seed_two_unit_checkpoint(
        manager,
        "session:included-source-edit",
    )
    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)
    store = manager.control_store
    original_prepare = store.prepare_compaction

    def edit_then_prepare(**kwargs: Any) -> CompactionPrepare:
        store.update_message(retained_id, content="edited during prepare")
        return original_prepare(**kwargs)

    store.prepare_compaction = edit_then_prepare  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="source snapshot"):
        asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))

    assert markdown.commit_count == 0
    assert markdown.receipts == {}
    assert (
        store.get_compaction_prepare(session.key, source_ref=checkpoint.source_ref)
        is None
    )
    assert store.get_compaction(session.key, head.next_generation) is None
    assert store.get_compaction_head(session.key) == head


def test_excluded_compaction_source_edit_before_persist_has_no_side_effects(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session, head, checkpoint, retained_id = _seed_two_unit_checkpoint(
        manager,
        "scheduler:excluded-source-edit",
        suppress_post_commit=True,
    )
    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)
    store = manager.control_store
    original_persist = store.persist_compaction

    def edit_then_persist(**kwargs: Any) -> SessionCompaction:
        store.update_message(retained_id, content="edited during persist")
        return original_persist(**kwargs)

    store.persist_compaction = edit_then_persist  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="source snapshot"):
        asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))

    assert markdown.commit_count == 0
    assert markdown.receipts == {}
    assert (
        store.get_compaction_prepare(session.key, source_ref=checkpoint.source_ref)
        is None
    )
    assert store.get_compaction(session.key, head.next_generation) is None
    assert store.get_compaction_head(session.key) == head


def test_reasoner_builder_preserves_replay_tail_and_current_payload_order() -> None:
    provider = _GateProvider()
    reasoner = _build_reasoner(
        provider,
        _NoopCompactionRuntime(),
        model="m",
        max_tokens=100,
    )
    committed = CommittedContextUnit(
        source_from_seq=1,
        consolidated_through_seq=1,
        source_message_ids=("canonical-1",),
        messages=({"role": "assistant", "content": "history"},),
        message_refs=(("canonical-1", 1),),
    )
    replay = [
        {"role": "user", "content": "U1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        {"role": "assistant", "content": "[execution attempt interrupted]"},
    ]
    projection = CompactionProjection(
        segments=ContextPayloadSegments(
            prefix=({"role": "system", "content": "stable"},),
            committed_units=(committed,),
            current_anchor=(),
        ),
        active=None,
        head=CompactionHead(
            session_key="session", parent_generation=0, next_generation=1
        ),
    )
    render_payload = [
        {"role": "system", "content": "root"},
        {"role": "system", "content": "stable"},
        *committed.messages,
        *replay,
        {
            "role": "user",
            "content": '<system-reminder data-system-context-frame="true">memory</system-reminder>',
        },
        {"role": "user", "content": "U2"},
        {"role": "user", "content": "U3"},
    ]
    test_session = Session(
        key="session",
        created_at=datetime(2026, 8, 8, tzinfo=UTC),
    )
    state = reasoner._build_request_state(
        agent_model=reasoner._test_agent_model,
        fallback_model=reasoner._test_fallback_model,
        projection=_plugin_turn(
            _NoopCompactionRuntime(),
            test_session,
            projection,
        ),
        initial_messages=render_payload,
        history_count=2,
        attempt_replay=replay,
        prior_tool_groups=1,
        channel="",
        chat_id="",
    )
    assert state.gate._compactor._segments.flatten() == render_payload
    assert state.gate._compactor._segments.current_anchor == ()
    assert state.gate._compactor._segments.active_batches == (tuple(replay[:3]),)
    assert state.gate._compactor._segments.pending == tuple(
        [replay[3], *render_payload[7:]]
    )
    assert state.gate._compactor._current_query == (
        '{"logical_interaction_inputs":["U1","U2","U3"]}'
    )


def test_reasoner_request_state_is_call_local_per_session(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session_a = manager.get_or_create("session-a")
    session_b = manager.get_or_create("session-b")
    for session, prefix in ((session_a, "a"), (session_b, "b")):
        session.add_message("user", f"{prefix}-u1", control_turn_id=f"{prefix}-1")
        session.add_message("assistant", f"{prefix}-a1", control_turn_id=f"{prefix}-1")
        session.add_message("user", f"{prefix}-u2", control_turn_id=f"{prefix}-2")
        session.add_message("assistant", f"{prefix}-a2", control_turn_id=f"{prefix}-2")
        manager.save(session)
    markdown = _MarkdownCompactionProbe()
    runtimes = [
        _runtime(manager, markdown, session.key) for session in (session_a, session_b)
    ]
    provider = _GateProvider()
    reasoner = _build_reasoner(provider, runtimes[0], max_tokens=10)

    projections = [
        asyncio.run(
            runtime.projection(session, prefix=[], current_anchor=[], pending=[])
        )
        for runtime, session in zip(runtimes, (session_a, session_b), strict=True)
    ]
    states = []
    for runtime, session, projection in zip(
        runtimes, (session_a, session_b), projections, strict=True
    ):
        history = [
            message
            for unit in projection.segments.committed_units
            for message in unit.messages
        ]
        payload = [
            {"role": "system", "content": "root"},
            *history,
            {"role": "user", "content": f"{session.key}-current"},
        ]
        states.append(
            reasoner._build_request_state(
                agent_model=reasoner._test_agent_model,
                fallback_model=reasoner._test_fallback_model,
                projection=_plugin_turn(runtime, session, projection),
                initial_messages=payload,
                history_count=len(history),
                attempt_replay=[],
                prior_tool_groups=0,
                channel="test",
                chat_id=session.key,
            )
        )

    states[0].gate._compactor.set_pending(
        [
            *states[0].gate._compactor._segments.flatten(),
            {"role": "user", "content": "only-a"},
        ]
    )
    original_b_pending = states[1].gate._compactor._segments.pending
    assert states[0].gate._compactor._scope_id == compaction_scope_id(
        "session-a", session_a.created_at
    )
    assert states[1].gate._compactor._scope_id == compaction_scope_id(
        "session-b", session_b.created_at
    )
    assert states[0].gate._compactor._segments.pending[-1] == {
        "role": "user",
        "content": "only-a",
    }
    assert states[1].gate._compactor._segments.pending == original_b_pending


def test_reasoner_binds_configured_main_fallback_with_distinct_provenance(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "u", control_turn_id="turn-1")
    session.add_message("assistant", "a", control_turn_id="turn-1")
    session.add_message("user", "u2", control_turn_id="turn-2")
    session.add_message("assistant", "a2", control_turn_id="turn-2")
    manager.save(session)
    runtime = _runtime(manager, _MarkdownCompactionProbe())
    selected = _GateProvider()
    configured_main = _GateProvider()
    reasoner = _build_reasoner(
        selected,
        runtime,
        model="agent-model",
        max_tokens=10,
        fallback_provider=configured_main,
        fallback_model="main-model",
    )
    projection = asyncio.run(
        runtime.projection(session, prefix=[], current_anchor=[], pending=[])
    )
    history = [
        message
        for unit in projection.segments.committed_units
        for message in unit.messages
    ]
    state = reasoner._build_request_state(
        agent_model=reasoner._test_agent_model,
        fallback_model=reasoner._test_fallback_model,
        projection=_plugin_turn(runtime, session, projection),
        initial_messages=[
            {"role": "system", "content": "root"},
            *history,
            {"role": "user", "content": "current"},
        ],
        history_count=len(history),
        attempt_replay=[],
        prior_tool_groups=0,
        channel="test",
        chat_id="chat",
    )

    assert state.gate._compactor._provider is reasoner._test_agent_model
    assert state.gate._compactor._fallback_provider is reasoner._test_fallback_model
    assert state.gate._compactor._model == "agent-model"
    assert state.gate._compactor._fallback_provider.descriptor.model == "main-model"


def test_two_session_compaction_commits_are_isolated_in_sqlite(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    sessions = []
    source_message_ids: dict[str, set[str]] = {}
    for key in ("session-a", "session-b"):
        session = manager.get_or_create(key)
        prefix = key[-1]
        session.add_message("user", f"{prefix}-u1", control_turn_id=f"{prefix}-1")
        session.add_message("assistant", f"{prefix}-a1", control_turn_id=f"{prefix}-1")
        session.add_message("user", f"{prefix}-u2", control_turn_id=f"{prefix}-2")
        session.add_message("assistant", f"{prefix}-a2", control_turn_id=f"{prefix}-2")
        manager.save(session)
        source_message_ids[key] = {
            str(message["id"]) for message in session.messages if message.get("id")
        }
        sessions.append(session)
    markdown = _MarkdownCompactionProbe()
    runtimes = [_runtime(manager, markdown, session.key) for session in sessions]
    provider = _ScopedCompactionProvider()
    reasoner = _build_reasoner(
        provider,
        runtimes[0],
        max_tokens=10,
    )
    prepared = []
    for runtime, session in zip(runtimes, sessions, strict=True):
        projection = asyncio.run(
            runtime.projection(session, prefix=[], current_anchor=[], pending=[])
        )
        history = [
            message
            for unit in projection.segments.committed_units
            for message in unit.messages
        ]
        payload = [
            {"role": "system", "content": "root"},
            *history,
            {"role": "user", "content": f"{session.key}-current"},
        ]
        state = reasoner._build_request_state(
            agent_model=reasoner._test_agent_model,
            fallback_model=reasoner._test_fallback_model,
            projection=_plugin_turn(runtime, session, projection),
            initial_messages=payload,
            history_count=len(history),
            attempt_replay=[],
            prior_tool_groups=0,
            channel="test",
            chat_id=session.key,
        )
        prepared.append(
            asyncio.run(
                reasoner._call_provider(
                    state,
                    payload,
                    tools=[],
                    max_tokens=10,
                )
            )
        )

    assert all(item.prepared and item.prepared.changed for item in prepared)
    assert [
        manager.control_store.get_compaction_head(session.key).parent_generation
        for session in sessions
    ] == [1, 1]
    active_a = manager._store.get_active_compaction("session-a")
    active_b = manager._store.get_active_compaction("session-b")
    assert active_a is not None and active_b is not None
    assert active_a.generation == active_b.generation == 1
    assert active_a.source_ref != active_b.source_ref
    assert set(active_a.source_message_ids) <= source_message_ids["session-a"]
    assert set(active_b.source_message_ids) <= source_message_ids["session-b"]
    assert set(active_a.source_message_ids).isdisjoint(source_message_ids["session-b"])
    assert set(active_b.source_message_ids).isdisjoint(source_message_ids["session-a"])
    assert manager.control_store.get_session_meta("session-a")["last_consolidated"] == 1
    assert manager.control_store.get_session_meta("session-b")["last_consolidated"] == 1
    assert "A_SENTINEL" in active_a.summary
    assert "B_SENTINEL" in active_b.summary
    assert "B_SENTINEL" not in active_a.summary
    assert "A_SENTINEL" not in active_b.summary
    assert markdown.commit_count == 0
