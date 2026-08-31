from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.lifecycle.types import PromptRenderCtx
from agent.prompting.section_names import (
    LONG_TERM_PROFILE_SECTION,
    RETRIEVED_MEMORY_SECTION,
    SELF_PROFILE_SECTION,
)
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.manager import SessionManager
from plugins.markdown_memory import plugin as markdown_plugin
from plugins.markdown_memory.plugin import (
    _inject_profiles,
    _migrate_pending,
    _prepare_draft,
    _source_text,
    _validate_memory,
    _validate_preserved_bullets,
    _validate_self,
)
from plugins.markdown_memory.store import (
    DEFAULT_SELF_MD,
    MarkdownProfileStore,
    content_digest,
)


def _store(tmp_path: Path) -> MarkdownProfileStore:
    return MarkdownProfileStore(
        tmp_path / "memory/MEMORY.md",
        tmp_path / "memory/SELF.md",
        tmp_path / "memory/markdown-profile-writes.db",
    )


@pytest.mark.asyncio
async def test_prompt_profiles_are_ordinary_sections(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.memory_path.write_text("# 用户长期记忆\n\n## 用户事实\n- 花月使用 Akashic\n\n## 用户偏好\n- 简洁\n\n## 用户明确要求长期记住的关键内容\n- 无\n", encoding="utf-8")
    event = cast(
        PromptRenderCtx,
        SimpleNamespace(disabled_sections=set(), system_sections_bottom=[]),
    )

    await _inject_profiles(event, store)

    assert [section.name for section in event.system_sections_bottom] == [
        "self_model",
        "long_term_memory",
    ]
    assert DEFAULT_SELF_MD.strip() in event.system_sections_bottom[0].content
    assert "花月使用 Akashic" in event.system_sections_bottom[1].content


@pytest.mark.asyncio
async def test_prompt_profile_disable_is_source_neutral(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.memory_path.write_text("memory", encoding="utf-8")
    event = cast(
        PromptRenderCtx,
        SimpleNamespace(
            disabled_sections={"self_model", "long_term_memory"},
            system_sections_bottom=[],
        ),
    )

    await _inject_profiles(event, store)

    assert event.system_sections_bottom == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("consumer", "disabled", "visible"),
    [
        ("scheduler-subagent", {RETRIEVED_MEMORY_SECTION}, [SELF_PROFILE_SECTION, LONG_TERM_PROFILE_SECTION]),
        ("wake-evidence", {RETRIEVED_MEMORY_SECTION, LONG_TERM_PROFILE_SECTION}, [SELF_PROFILE_SECTION]),
        ("normal-wake-screen", set(), [SELF_PROFILE_SECTION, LONG_TERM_PROFILE_SECTION]),
    ],
)
async def test_consumer_scopes_project_exact_markdown_profile_visibility(
    tmp_path: Path,
    consumer: str,
    disabled: set[str],
    visible: list[str],
) -> None:
    store = _store(tmp_path / consumer)
    store.memory_path.write_text("# 用户长期记忆\n\n- 可见事实\n", encoding="utf-8")
    event = cast(
        PromptRenderCtx,
        SimpleNamespace(disabled_sections=disabled, system_sections_bottom=[]),
    )

    await _inject_profiles(event, store)

    assert [section.name for section in event.system_sections_bottom] == visible


def test_profile_draft_is_idempotent_and_recoverable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    memory = "# 用户长期记忆\n\n## 用户事实\n- x\n\n## 用户偏好\n- y\n\n## 用户明确要求长期记住的关键内容\n- z\n"
    draft: dict[str, object] = {
        "version": 1,
        "memory": memory,
        "self": DEFAULT_SELF_MD,
        "memory_before": "",
        "self_before": DEFAULT_SELF_MD,
        "memory_before_digest": content_digest(""),
        "self_before_digest": content_digest(DEFAULT_SELF_MD),
        "memory_after_digest": content_digest(memory),
        "self_after_digest": content_digest(DEFAULT_SELF_MD),
    }

    assert store.write_draft(
        "source:1", draft, session_key="session", generation=1
    ) == draft
    store.apply_draft("source:1", draft)
    store.apply_draft("source:1", draft)

    assert store.read_memory() == memory
    assert store.read_self() == DEFAULT_SELF_MD
    assert store.is_applied("source:1")
    assert store.read_backup("source:1", "memory") == ""
    assert store.read_backup("source:1", "self") == DEFAULT_SELF_MD


def test_profile_receipt_conflict_fails_loud(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = {
        "version": 1,
        "memory": "",
        "self": DEFAULT_SELF_MD,
        "memory_before": "",
        "self_before": DEFAULT_SELF_MD,
    }
    _ = store.write_draft("source:1", first, session_key="session", generation=1)

    with pytest.raises(ValueError, match="内容冲突"):
        _ = store.write_draft(
            "source:1",
            {**first, "version": 2},
            session_key="session",
            generation=1,
        )


def test_independent_document_receipts_recover_half_applied_draft(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    memory = "# 用户长期记忆\n\n## 用户事实\n- x\n\n## 用户偏好\n\n## 用户明确要求长期记住的关键内容\n"
    draft: dict[str, object] = {
        "version": 1,
        "memory": memory,
        "self": DEFAULT_SELF_MD,
        "memory_before": "",
        "self_before": DEFAULT_SELF_MD,
    }
    _ = store.write_draft(
        "source:crash", draft, session_key="session", generation=1
    )
    store._apply_document("source:crash", "memory", store.memory_path)
    assert not store.is_applied("source:crash")

    reopened = _store(tmp_path)
    assert reopened.pending_source_refs() == ("source:crash",)
    reopened.apply_pending("source:crash")

    assert reopened.is_applied("source:crash")
    assert reopened.read_memory() == memory
    assert reopened.read_self() == DEFAULT_SELF_MD


def test_pending_drafts_follow_durable_generation_not_lexical_source_ref(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for generation in (10, 2, 1):
        draft = {
            "version": 1,
            "memory": "",
            "self": DEFAULT_SELF_MD,
            "memory_before": "",
            "self_before": DEFAULT_SELF_MD,
        }
        _ = store.write_draft(
            f"session:{generation}",
            draft,
            session_key="session",
            generation=generation,
        )

    assert store.pending_source_refs() == ("session:1", "session:2", "session:10")


def test_profile_projection_rejects_implicit_fact_deletion() -> None:
    with pytest.raises(ValueError, match="不得隐式删除"):
        _validate_preserved_bullets(
            "# x\n- protected\n",
            "# x\n- replacement\n",
            document="MEMORY.md",
        )


def test_v4_source_plan_is_consumable() -> None:
    current = {
        "version": 4,
        "checkpoint": {
            "selected_source_messages": [
                {"id": "m1", "seq": 1, "message": {"role": "user", "content": "x"}}
            ]
        },
    }

    assert '"id": "m1"' in _source_text(current)


def test_profile_validators_keep_memory_and_self_contracts() -> None:
    _validate_memory("")
    _validate_memory(
        "# 用户长期记忆\n\n## 用户事实\n- x\n\n## 用户偏好\n- y\n\n"
        "## 用户明确要求长期记住的关键内容\n- z\n"
    )
    _validate_self(DEFAULT_SELF_MD)
    with pytest.raises(ValueError, match="MEMORY.md"):
        _validate_memory("# arbitrary\n- x")
    with pytest.raises(ValueError, match="SELF.md"):
        _validate_self(DEFAULT_SELF_MD + "\n## 关系演进记录\n- no\n")


def test_pending_files_are_not_created(tmp_path: Path) -> None:
    _ = _store(tmp_path)

    assert not (tmp_path / "memory/PENDING.md").exists()
    assert not (tmp_path / "memory/PENDING.snapshot.md").exists()


@pytest.mark.asyncio
async def test_markdown_candidate_and_publish_keep_formal_files_isolated(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True)
    formal_memory = "# 用户长期记忆\n\n## 用户事实\n- formal\n\n## 用户偏好\n- formal\n\n## 用户明确要求长期记住的关键内容\n- formal\n"
    (memory_dir / "MEMORY.md").write_text(formal_memory, encoding="utf-8")
    (memory_dir / "SELF.md").write_text(DEFAULT_SELF_MD, encoding="utf-8")
    (memory_dir / "SECRET.md").write_text("not granted", encoding="utf-8")
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[
            Path(markdown_plugin.__file__).parent,
            Path(__file__).parent / "fixtures/static_chat_models",
        ],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None
    generation = stable.generations["markdown_memory"]
    assert generation.instance.workspace_roots == ()
    assert generation.instance.workspace_files == markdown_plugin.workspace_files
    lease = await manager.snapshot_store.acquire()
    candidate = await manager.prepare_candidate("markdown_memory")
    assert candidate is not None and candidate.validation_workspace is not None
    assert not tuple(candidate.validation_workspace.rglob("SECRET.md"))
    candidate_snapshot = candidate.runtime_snapshot
    assert candidate_snapshot is not None
    candidate_root = candidate_snapshot.composition_root
    assert candidate_root is not None
    candidate_runtime = candidate_root.plugin_runtime("markdown_memory")
    assert candidate_runtime.workspace_file("memory/MEMORY.md").read_text(
        encoding="utf-8"
    ) == formal_memory

    publish_task = asyncio.create_task(manager.publish_prepared("markdown_memory"))
    await asyncio.sleep(0)
    assert manager.current_snapshot is stable
    assert lease.snapshot is stable
    await lease.release()
    published = await publish_task

    assert published["publication_state"] == "committed"
    assert manager.current_snapshot is not stable
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8") == formal_memory
    await manager.terminate_all()
    sessions.close()


@pytest.mark.asyncio
async def test_markdown_builtin_can_be_disabled(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[
            Path(markdown_plugin.__file__).parent,
            Path(__file__).parent / "fixtures/static_chat_models",
        ],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
        disabled_builtin_plugins=frozenset({"markdown_memory"}),
    )

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert "markdown_memory" not in snapshot.generations
    assert not (workspace / "memory/MEMORY.md").exists()
    await manager.terminate_all()
    sessions.close()


@pytest.mark.asyncio
async def test_legacy_pending_is_archived_after_one_direct_merge(tmp_path: Path) -> None:
    store = _store(tmp_path)
    pending = tmp_path / "memory/PENDING.md"
    snapshot = tmp_path / "memory/PENDING.snapshot.md"
    retired = tmp_path / "memory/PENDING.retired.md"
    lock = tmp_path / "memory/markdown-profile.lock"
    pending.write_text("- [identity] pending\n", encoding="utf-8")
    snapshot.write_text("- [preference] snapshot\n", encoding="utf-8")
    await _migrate_pending(
        store,
        lock,
        pending,
        snapshot,
        retired,
    )

    assert pending.read_text(encoding="utf-8") == ""
    assert snapshot.read_text(encoding="utf-8") == ""
    archive = retired.read_text(encoding="utf-8")
    assert '"pending": "- [identity] pending\\n"' in archive
    assert '"snapshot": "- [preference] snapshot\\n"' in archive
    memory = store.read_memory()
    assert "- [identity] pending" in memory
    assert "- [preference] snapshot" in memory

    pending.write_text("- [identity] restored later\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="出现新内容"):
        await _migrate_pending(
            store,
            lock,
            pending,
            snapshot,
            retired,
        )
    assert pending.read_text(encoding="utf-8") == "- [identity] restored later\n"


@pytest.mark.asyncio
async def test_v2_draft_rejects_inner_source_ref_mismatch(tmp_path: Path) -> None:
    receipt = {
        "version": 2,
        "markdown_draft": {
            "source_ref": "wrong",
            "pending_items": "",
            "history_entry_payloads": [],
            "conversation": "",
            "scope_channel": "",
            "scope_chat_id": "",
        },
    }
    with pytest.raises(ValueError, match="source_ref 冲突"):
        await _prepare_draft(
            json.dumps(receipt),
            "expected",
            _store(tmp_path),
            cast(Any, None),
        )


@pytest.mark.asyncio
async def test_runtime_started_migrates_legacy_pending_once(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    pending = workspace / "memory/PENDING.md"
    pending.parent.mkdir(parents=True)
    pending.write_text("- [requested_memory] keep exact\n", encoding="utf-8")
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[
            Path(markdown_plugin.__file__).parent,
            Path(__file__).parent / "fixtures/static_chat_models",
        ],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    retired = workspace / "memory/PENDING.retired.md"
    try:
        await cast(Any, manager)._start_current_runtime_snapshot()
        assert retired.exists()
        assert pending.read_text(encoding="utf-8") == ""
        assert "- [requested_memory] keep exact" in (
            workspace / "memory/MEMORY.md"
        ).read_text(encoding="utf-8")
    finally:
        await manager.terminate_all()
        sessions.close()
