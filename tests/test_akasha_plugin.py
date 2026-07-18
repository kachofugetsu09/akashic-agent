from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import tomllib
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import numpy as np
from fastapi import FastAPI

from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryQuery, MemoryQueryIntent, MemoryScope
from agent.plugins.context import PluginContext, PluginKVStore
from agent.config_models import Config, MemoryConfig, MemoryEmbeddingConfig
from plugins.akasha.config import (
    AkashaConfig,
    ensure_akasha_config_file,
    load_akasha_config,
    render_akasha_config,
)
import plugins.akasha.dashboard as akasha_dashboard
from plugins.akasha.dashboard import (
    AkashaGraphReader,
    _json_items as dashboard_json_items,
    register as register_akasha_dashboard,
)
from plugins.akasha.engine import (
    ActivationTrace,
    AkashaCard,
    AkashaCandidate,
    AkashaMemoryEngine,
    PendingActivation,
    _AkashaRetrieval,
    _compute_candidates,
    _load_turn_card,
)
from plugins.akasha.core import (
    AkashaActivationSnapshot,
    AkashaNode,
    activation_edge_updates,
    build_dense_message_index,
    dense_message_candidates,
    reinforce_boost_from_payload,
    serialize_f32,
)
import plugins.akasha.graph_snapshot as graph_snapshot
from plugins.akasha.plugin import AkashaPlugin
from plugins.akasha.replay import AkashaReplayRuntime, ReplayMessage, _turn_messages
from plugins.akasha.store import (
    ActivationEventRow,
    AkashaStore,
    EdgeUpdate,
    SourceMessage,
)
from session.embedding_store import MessageEmbeddingStore
from scripts.build_akasha_db import _iter_replay_turns, _load_embeddings_from_cache, _skip_message


QUERY_TS = datetime(2026, 1, 2, tzinfo=timezone.utc)


def test_akasha_mobile_ui_is_message_slot_only() -> None:
    contribution = AkashaPlugin.mobile_ui()

    assert contribution.navigation is None
    assert contribution.slots == ("turn.before_reasoning",)


def test_akasha_config_does_not_expose_dynamic_budget_limits(tmp_path: Path) -> None:
    (tmp_path / "config.local.toml").write_text(
        "dense_top_k = 99\nripple_top_k = 99\nactivate_limit = 99\n",
        encoding="utf-8",
    )

    config = load_akasha_config(plugin_dir=tmp_path)
    rendered = render_akasha_config(config)

    assert not hasattr(config, "dense_top_k")
    assert not hasattr(config, "ripple_top_k")
    assert not hasattr(config, "activate_limit")
    assert "top_k" not in rendered
    assert "activate_limit" not in rendered


def test_akasha_config_uses_defaults_only_for_missing_fields(tmp_path: Path) -> None:
    assert load_akasha_config(plugin_dir=tmp_path) == AkashaConfig()

    (tmp_path / "config.local.toml").write_text(
        'inject_max_chars = "7000"\nactivation_threshold = "0.3"\n',
        encoding="utf-8",
    )
    config = load_akasha_config(plugin_dir=tmp_path)

    assert config.inject_max_chars == 7000
    assert config.activation_threshold == 0.3
    assert config.assistant_preview_chars == AkashaConfig().assistant_preview_chars


def test_akasha_config_creates_workspace_data_dir(
    tmp_path: Path,
) -> None:
    target = tmp_path / "plugin-data" / "akasha-builtin"

    path = ensure_akasha_config_file(workspace=tmp_path)

    assert path == target / "config.local.toml"
    assert load_akasha_config(workspace=tmp_path) == AkashaConfig()


def test_akasha_config_rejects_symlink_data_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "plugin-data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="符号链接"):
        ensure_akasha_config_file(workspace=tmp_path)

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("db_path", "42"),
        ("inject_max_chars", '"many"'),
        ("assistant_preview_chars", "1.5"),
        ("nearby_time_seconds", "true"),
        ("activation_threshold", "nan"),
        ("dense_seed_threshold", "true"),
        ("cross_boost", "[]"),
    ],
)
def test_akasha_config_rejects_invalid_present_fields(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    (tmp_path / "config.local.toml").write_text(
        f"{field} = {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_akasha_config(plugin_dir=tmp_path)


def test_akasha_dashboard_exposes_invalid_plugin_config(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    data_dir = workspace / "plugin-data" / "akasha-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text("db_path = [", encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        register_akasha_dashboard(FastAPI(), tmp_path, workspace)


def test_graph_snapshot_distinguishes_missing_from_corruption(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "memory" / "akasha_graph_snapshot.json"

    assert graph_snapshot.load_snapshot(snapshot_path) is None
    snapshot_path.parent.mkdir()
    snapshot_path.write_text("{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="图快照读取失败"):
        graph_snapshot.load_snapshot(snapshot_path)


def test_graph_reader_surfaces_background_rebuild_failure(tmp_path: Path, monkeypatch) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    reader = AkashaGraphReader(
        store,
        akasha_db_path=store.db_path,
        sessions_db_path=tmp_path / "sessions.db",
        snapshot_path=tmp_path / "memory" / "akasha_graph_snapshot.json",
    )

    def fail_rebuild(**_: object) -> dict[str, object]:
        raise RuntimeError("rebuild boom")

    monkeypatch.setattr(graph_snapshot, "build_snapshot_to_file", fail_rebuild)
    monkeypatch.setattr(
        "plugins.akasha.dashboard.build_snapshot_to_file",
        fail_rebuild,
    )
    try:
        reader._rebuild_in_background()

        with pytest.raises(RuntimeError, match="后台重建失败"):
            reader.get_global_graph()
        with pytest.raises(RuntimeError, match="后台重建失败"):
            reader.close()
    finally:
        store.close()


def test_graph_reader_caches_signature_until_external_graph_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    reader = AkashaGraphReader(
        store,
        akasha_db_path=store.db_path,
        sessions_db_path=tmp_path / "sessions.db",
        snapshot_path=tmp_path / "memory" / "akasha_graph_snapshot.json",
    )
    calls = 0
    original = akasha_dashboard.read_graph_signature

    def counted(path: Path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(akasha_dashboard, "read_graph_signature", counted)
    try:
        first = reader._current_graph_signature()
        assert reader._current_graph_signature() == first
        assert calls == 1

        with closing(sqlite3.connect(str(store.db_path))) as db:
            db.execute(
                "INSERT INTO akasha_edges "
                "(src_key, dst_key, weight, co_count, last_used_ts) "
                "VALUES (?, ?, ?, ?, ?)",
                ("external-src", "external-dst", 1.0, 1, 1.0),
            )
            db.commit()

        second = reader._current_graph_signature()
        assert calls == 2
        assert second.edge_count == first.edge_count + 1
    finally:
        store.close()


def test_akasha_dashboard_closeables_close_reader_before_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closeables = register_akasha_dashboard(
        FastAPI(),
        tmp_path / "plugin",
        tmp_path / "workspace",
    )

    assert isinstance(closeables[0], AkashaStore)
    assert isinstance(closeables[1], AkashaGraphReader)

    close_order: list[str] = []
    for name, closeable in zip(("store", "reader"), closeables):
        original_close = closeable.close

        def close(*, _name=name, _original=original_close) -> None:
            close_order.append(_name)
            _original()

        monkeypatch.setattr(closeable, "close", close)

    for closeable in reversed(closeables):
        closeable.close()

    assert close_order == ["reader", "store"]


def test_graph_cache_rejects_orphan_message_embedding(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    embedding_store = MessageEmbeddingStore(db_path)
    store = AkashaStore(tmp_path / "akasha.db")
    embedding_store.upsert(
        message_id="orphan-message",
        content="已从 messages 删除",
        model="m",
        embedding=[1.0, 0.0],
    )
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._session_db_path = db_path
    engine._store = store
    engine._embedding_store = embedding_store
    engine._config = SimpleNamespace(
        memory=SimpleNamespace(embedding=SimpleNamespace(model="m"))
    )
    engine._graph_lock = threading.RLock()
    try:
        with pytest.raises(ValueError, match=r"count=1.*orphan-message"):
            engine._load_graph_cache()
    finally:
        embedding_store.close()
        store.close()


def test_engine_owns_event_subscription_lifetime() -> None:
    event_bus = EventBus()
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._event_bus = event_bus
    engine.closeables = []

    engine._wire_events()

    assert event_bus.handler_count() == 1
    for closeable in engine.closeables:
        closeable.close()
    assert event_bus.handler_count() == 0


def test_diagnostic_json_rejects_corruption() -> None:
    with pytest.raises(ValueError, match="JSON 损坏"):
        dashboard_json_items("{")


def _init_sessions_db(path: Path) -> None:
    with closing(sqlite3.connect(str(path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
                ("s:1", "s", 1, "assistant", "第一条助手回复会被截断展示并保留引用", "2026-01-01T00:00:01+00:00"),
                ("s:2", "s", 2, "user", "第二条用户消息只在联想块", "2026-01-01T00:00:02+00:00"),
                ("s:3", "s", 3, "assistant", "第二条助手回复也会被截断", "2026-01-01T00:00:03+00:00"),
            ],
        )
        db.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
        db.execute("INSERT INTO messages_fts(rowid, content) SELECT rowid, content FROM messages")
        db.commit()


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        _ = text
        return [1.0, 0.0]


def _candidate(key: str, score: float) -> AkashaCandidate:
    return AkashaCandidate(
        key=key,
        source="Dense",
        ripple=0.0,
        direct=score,
        state=0.0,
        edge=0.0,
        long=0.0,
        resource=1.0,
        fan=0,
        score=score,
    )


def _seed_legacy_embedding(
    store: AkashaStore,
    *,
    message: SourceMessage,
    model: str,
    embedding: list[float],
) -> None:
    now = "2026-01-01T00:00:00+00:00"
    store.db.execute(
        """
        INSERT INTO akasha_embedding_cache
            (message_id, content_hash, model, embedding, dim, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message.id,
            hashlib.sha256(message.content.encode("utf-8")).hexdigest(),
            model,
            serialize_f32(np.asarray(embedding, dtype=np.float32)),
            len(embedding),
            now,
            now,
        ),
    )
    store.db.commit()


def test_reinforce_boost_payload_uses_exact_tool_chain_call_name() -> None:
    wrong_chain = [{"calls": [{"name": "not_reinforce_memory"}]}]
    reinforce_chain = [{"calls": [{"name": "reinforce_memory"}]}]

    assert reinforce_boost_from_payload({}, wrong_chain) == 1.0
    assert reinforce_boost_from_payload({}, json.dumps(reinforce_chain)) == 3.0
    assert reinforce_boost_from_payload({"akasha_reinforce": {"boost": "4"}}, []) == 4.0


def test_reinforce_memory_tool_description_states_current_turn_contract() -> None:
    profile = AkashaMemoryEngine.__new__(AkashaMemoryEngine).tool_profile()
    reinforce = next(spec for spec in profile.tools if spec.name == "reinforce_memory")

    assert "当前轮" in reinforce.description
    assert "source_ref" in reinforce.description
    assert "fitbit_health_snapshot" in reinforce.description
    assert "sleep_report" in reinforce.description
    assert "fetch_messages(source_ref)" in reinforce.description


def test_akasha_engine_passes_embedding_dimension_to_embedder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Embedder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)

    engine = AkashaMemoryEngine(
        config=Config(
            provider="openai",
            model="chat-model",
            api_key="chat-key",
            system_prompt="system",
            memory=MemoryConfig(
                embedding=MemoryEmbeddingConfig(
                    model="embedding-model",
                    output_dimensionality=768,
                )
            ),
        ),
        akasha_config=AkashaConfig(),
        workspace=tmp_path,
        http_resources=cast(Any, SimpleNamespace(external_default=object())),
    )
    try:
        assert captured["model"] == "embedding-model"
        assert captured["output_dimensionality"] == 768
    finally:
        engine._embedding_store.close()
        engine._store.close()


def test_dense_message_candidates_vectorized_preserves_turn_ranking() -> None:
    nodes = {
        "s:0": AkashaNode(
            key="s:0",
            anchor_id="m0",
            session_key="s",
            turn_seq=0,
            first_ts_unix=QUERY_TS.timestamp(),
            salience=0.0,
            strength=0.0,
            resource=1.0,
            recall_count=0,
            last_activated_ts=0.0,
            last_strength_ts=QUERY_TS.timestamp(),
            last_resource_ts=QUERY_TS.timestamp(),
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            emb_count=1,
        ),
        "s:2": AkashaNode(
            key="s:2",
            anchor_id="m2",
            session_key="s",
            turn_seq=2,
            first_ts_unix=QUERY_TS.timestamp(),
            salience=0.0,
            strength=0.0,
            resource=1.0,
            recall_count=0,
            last_activated_ts=0.0,
            last_strength_ts=QUERY_TS.timestamp(),
            last_resource_ts=QUERY_TS.timestamp(),
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            emb_count=1,
        ),
        "s:4": AkashaNode(
            key="s:4",
            anchor_id="m4",
            session_key="s",
            turn_seq=4,
            first_ts_unix=QUERY_TS.timestamp(),
            salience=0.0,
            strength=0.0,
            resource=1.0,
            recall_count=0,
            last_activated_ts=0.0,
            last_strength_ts=QUERY_TS.timestamp(),
            last_resource_ts=QUERY_TS.timestamp(),
            embedding=np.array([0.0, 0.0], dtype=np.float32),
            emb_count=1,
        ),
    }
    message_embeddings = {
        "m0": np.array([1.0, 0.0], dtype=np.float32),
        "m2": np.array([0.8, 0.6], dtype=np.float32),
        "m3": np.array([0.9, 0.1], dtype=np.float32),
        "bad-dim": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "m4": np.array([0.0, 0.0], dtype=np.float32),
    }
    message_turn_keys = {
        "m0": "s:0",
        "m2": "s:2",
        "m3": "s:2",
        "bad-dim": "s:0",
        "m4": "s:4",
    }
    loop_result = dense_message_candidates(
        np.array([1.0, 0.0], dtype=np.float32),
        nodes,
        message_embeddings,
        message_turn_keys,
        limit=3,
    )
    indexed_result = dense_message_candidates(
        np.array([1.0, 0.0], dtype=np.float32),
        nodes,
        message_embeddings,
        message_turn_keys,
        limit=3,
        message_index=build_dense_message_index(message_embeddings),
    )

    assert [item.key for item in loop_result] == ["s:0", "s:2", "s:4"]
    assert [item.key for item in indexed_result] == [item.key for item in loop_result]
    assert [item.score for item in loop_result] == pytest.approx([
        1.0,
        0.9 / ((0.9 ** 2 + 0.1 ** 2) ** 0.5),
        0.0,
    ])
    assert [item.score for item in indexed_result] == pytest.approx(
        [item.score for item in loop_result]
    )


def test_store_merges_user_and_assistant_into_turn_node(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        store.upsert_message_node(
            SourceMessage("s:0", "s", 0, "user", "用户消息", "2026-01-01T00:00:00+00:00"),
            [1.0, 0.0],
        )
        store.upsert_message_node(
            SourceMessage("s:1", "s", 1, "assistant", "助手消息", "2026-01-01T00:00:01+00:00"),
            [0.0, 1.0],
        )

        nodes = store.list_nodes()
    finally:
        store.close()

    assert len(nodes) == 1
    assert nodes[0].key == "s:0"
    assert nodes[0].anchor_id == "s:0"
    assert nodes[0].emb_count == 2


def test_store_exposes_empty_node_embedding_corruption(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        key = store.upsert_message_node(
            SourceMessage(
                "m:0",
                "s",
                0,
                "user",
                "用户消息",
                "2026-01-01T00:00:00+00:00",
            ),
            [1.0, 0.0],
        )
        store.db.execute(
            "UPDATE akasha_nodes SET embedding = ? WHERE key = ?",
            (b"", key),
        )
        store.db.commit()

        with pytest.raises(ValueError, match=f"节点 {key} 的 embedding 为空"):
            store.list_nodes()
    finally:
        store.close()


def test_store_batch_delete_keeps_count_and_edge_cleanup_semantics(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        keys = [
            store.upsert_message_node(
                SourceMessage(
                    f"m:{seq}",
                    "s",
                    seq,
                    "user",
                    f"消息 {seq}",
                    "2026-01-01T00:00:00+00:00",
                ),
                [1.0, 0.0],
            )
            for seq in (0, 2, 4, 6)
        ]
        store.upsert_edges([
            EdgeUpdate(keys[0], keys[2], 1.0, 0),
            EdgeUpdate(keys[2], keys[1], 1.0, 0),
            EdgeUpdate(keys[2], keys[3], 1.0, 0),
        ])

        deleted = store.delete_items_batch([keys[0], keys[1], keys[0], "missing"])

        assert deleted == 2
        assert {node.key for node in store.list_nodes()} == {keys[2], keys[3]}
        assert set(store.load_edges()) == {(keys[2], keys[3])}
    finally:
        store.close()


def test_store_batch_delete_rolls_back_nodes_and_edges_on_failure(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        keys = [
            store.upsert_message_node(
                SourceMessage(
                    f"m:{seq}",
                    "s",
                    seq,
                    "user",
                    f"消息 {seq}",
                    "2026-01-01T00:00:00+00:00",
                ),
                [1.0, 0.0],
            )
            for seq in (0, 2, 4)
        ]
        store.upsert_edges([
            EdgeUpdate(keys[0], keys[2], 1.0, 0),
            EdgeUpdate(keys[1], keys[2], 1.0, 0),
        ])
        original_edges = store.load_edges()
        store.db.execute(
            f"""
            CREATE TRIGGER abort_second_edge_delete
            BEFORE DELETE ON akasha_edges
            WHEN OLD.src_key = '{keys[1]}'
            BEGIN
                SELECT RAISE(ABORT, '模拟边删除失败');
            END
            """
        )
        store.db.commit()

        with pytest.raises(sqlite3.IntegrityError, match="模拟边删除失败"):
            store.delete_items_batch(keys[:2])

        assert {node.key for node in store.list_nodes()} == set(keys)
        assert store.load_edges() == original_edges
    finally:
        store.close()


def test_reset_schema_keeps_legacy_embedding_source(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    message = SourceMessage(
        "s:0",
        "s",
        0,
        "user",
        "用户消息",
        "2026-01-01T00:00:00+00:00",
    )
    try:
        _seed_legacy_embedding(
            store,
            message=message,
            model="m",
            embedding=[1.0, 2.0],
        )
        _ = store.upsert_message_node(message, [1.0, 0.0])
        store.reset_schema()

        cached = store.list_cached_embedding_rows()
        nodes = store.list_nodes()
    finally:
        store.close()

    assert len(cached) == 1
    assert nodes == []


def test_message_embedding_store_reuses_only_matching_content(tmp_path: Path) -> None:
    store = MessageEmbeddingStore(tmp_path / "sessions.db")
    try:
        store.upsert(
            message_id="s:0",
            content="原始内容",
            model="m",
            embedding=[1.0, 2.0],
        )

        assert store.get(message_id="s:0", content="原始内容", model="m") == [1.0, 2.0]
        assert store.get(message_id="s:0", content="变更内容", model="m") is None
    finally:
        store.close()


def test_message_embedding_store_rejects_empty_vector_before_write(
    tmp_path: Path,
) -> None:
    store = MessageEmbeddingStore(tmp_path / "sessions.db")
    try:
        with pytest.raises(ValueError, match="消息向量不能为空"):
            store.upsert(
                message_id="s:0",
                content="原始内容",
                model="m",
                embedding=[],
            )

        assert store.get(message_id="s:0", content="原始内容", model="m") is None
    finally:
        store.close()


@pytest.mark.parametrize(
    ("blob", "dim"),
    [
        (b"", 2),
        (b"\x00" * 8, 3),
    ],
)
def test_message_embedding_store_rejects_corrupt_vectors(
    tmp_path: Path,
    blob: bytes,
    dim: int,
) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    store = MessageEmbeddingStore(db_path)
    try:
        content = "第一条用户消息需要完整展示"
        store.upsert(
            message_id="s:0",
            content=content,
            model="m",
            embedding=[1.0, 2.0],
        )
        store._db.execute(
            "UPDATE message_embeddings SET embedding = ?, dim = ? "
            "WHERE message_id = ? AND model = ?",
            (blob, dim, "s:0", "m"),
        )
        store._db.commit()

        with pytest.raises(ValueError, match=r"message_id=s:0 model=m"):
            store.get(message_id="s:0", content=content, model="m")
        with pytest.raises(ValueError, match=r"message_id=s:0 model=m"):
            store.list(model="m")
        with pytest.raises(ValueError, match=r"message_id=s:0 model=m"):
            store.list_until(model="m", cutoff="2026-01-01T00:00:01+00:00")
    finally:
        store.close()


def test_message_embedding_store_lists_only_messages_visible_at_cutoff(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    store = MessageEmbeddingStore(db_path)
    try:
        store.upsert(
            message_id="s:0",
            content="第一条用户消息需要完整展示",
            model="m",
            embedding=[1.0, 0.0],
        )
        store.upsert(
            message_id="s:2",
            content="第二条用户消息只在联想块",
            model="m",
            embedding=[0.0, 1.0],
        )

        visible = store.list_until(
            model="m",
            cutoff="2026-01-01T00:00:01+00:00",
        )

        assert visible == [("s:0", [1.0, 0.0])]
    finally:
        store.close()


def test_message_embedding_store_list_excludes_changed_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    store = MessageEmbeddingStore(db_path)
    try:
        store.upsert(
            message_id="s:0",
            content="第一条用户消息需要完整展示",
            model="m",
            embedding=[1.0, 0.0],
        )
        with closing(sqlite3.connect(str(db_path))) as db:
            db.execute("UPDATE messages SET content = '已编辑' WHERE id = 's:0'")
            db.commit()

        assert store.list(model="m") == []
    finally:
        store.close()


def test_legacy_akasha_embeddings_migrate_to_sessions_db(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    message = SourceMessage(
        "s:0",
        "s",
        0,
        "user",
        "第一条用户消息需要完整展示",
        "2026-01-01T00:00:00+00:00",
    )
    legacy_store = AkashaStore(tmp_path / "akasha.db")
    embedding_store = MessageEmbeddingStore(db_path)
    try:
        _seed_legacy_embedding(
            legacy_store,
            message=message,
            model="m",
            embedding=[1.0, 2.0],
        )
        _seed_legacy_embedding(
            legacy_store,
            message=SourceMessage(
                "s:1",
                "s",
                1,
                "assistant",
                "与当前消息不一致",
                "2026-01-01T00:00:01+00:00",
            ),
            model="m",
            embedding=[2.0, 1.0],
        )
        _seed_legacy_embedding(
            legacy_store,
            message=SourceMessage(
                "missing:0",
                "missing",
                0,
                "user",
                "已经删除",
                "2026-01-01T00:00:00+00:00",
            ),
            model="m",
            embedding=[3.0, 1.0],
        )

        imported = embedding_store.import_legacy_rows_once(
            legacy_store.list_cached_embedding_rows()
        )

        assert imported == 1
        assert embedding_store.get(
            message_id=message.id,
            content=message.content,
            model="m",
        ) == [1.0, 2.0]
        assert embedding_store.delete([message.id]) == 1
        assert embedding_store.import_legacy_rows_once(
            legacy_store.list_cached_embedding_rows()
        ) == 0
        assert embedding_store.get(
            message_id=message.id,
            content=message.content,
            model="m",
        ) is None
    finally:
        embedding_store.close()
        legacy_store.close()


def test_load_embeddings_from_cache_counts_hits_and_misses(
    tmp_path: Path,
) -> None:
    store = MessageEmbeddingStore(tmp_path / "sessions.db")
    messages = [
        SourceMessage("s:0", "s", 0, "user", "已缓存", "2026-01-01T00:00:00+00:00"),
        SourceMessage("s:1", "s", 1, "assistant", "新消息", "2026-01-01T00:00:01+00:00"),
    ]
    try:
        store.upsert(
            message_id=messages[0].id,
            content=messages[0].content,
            model="m",
            embedding=[1.0, 0.0],
        )

        embeddings, hits, misses = _load_embeddings_from_cache(
            store=store,
            model="m",
            messages=messages,
        )
    finally:
        store.close()

    assert hits == 1
    assert misses == 1
    assert embeddings == {"s:0": [1.0, 0.0]}


def test_replay_and_runtime_use_same_directional_stdp_edges(tmp_path: Path) -> None:
    candidate = _candidate("s:0", 0.8)
    ts = QUERY_TS.timestamp()
    expected = {
        (item.src_key, item.dst_key): 0.12 * item.strength
        for item in activation_edge_updates("s:2", [candidate], ts)
    }
    replay_store = AkashaStore(tmp_path / "replay.db")
    runtime_store = AkashaStore(tmp_path / "runtime.db")
    try:
        with closing(sqlite3.connect(":memory:")) as source_db:
            replay = AkashaReplayRuntime(
                store=replay_store,
                config=AkashaConfig(),
                source_db_path=tmp_path / "sessions.db",
                source_cursor=source_db.cursor(),
                message_embeddings={},
                message_turn_keys={},
            )
            replay.commit_turn(
                [
                    ReplayMessage(
                        SourceMessage(
                            "m2",
                            "s",
                            2,
                            "user",
                            "当前消息",
                            QUERY_TS.isoformat(),
                        ),
                        [1.0, 0.0],
                    )
                ],
                [candidate],
            )

        engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
        engine._store = runtime_store
        engine._graph_lock = threading.RLock()
        engine._edges = {}
        engine._edges_meta = {}
        engine._edges_by_src = {}
        engine._fan = {}
        engine._nodes = {}
        engine._commit_pending_activation(
            "s:2",
            PendingActivation(
                query_id="q",
                seq=2,
                ts=ts,
                items=[candidate],
                query_vec=np.array([1.0, 0.0], dtype=np.float32),
            ),
        )

        assert replay_store.load_edges() == pytest.approx(expected)
        assert runtime_store.load_edges() == pytest.approx(expected)
        assert expected[("s:0", "s:2")] > expected[("s:2", "s:0")]
    finally:
        replay_store.close()
        runtime_store.close()


def test_replay_and_runtime_reinforce_previous_activation_cluster(tmp_path: Path) -> None:
    prev = _candidate("s:0", 0.8)
    current = _candidate("s:2", 0.7)
    ts = QUERY_TS.timestamp()
    replay_store = AkashaStore(tmp_path / "replay.db")
    runtime_store = AkashaStore(tmp_path / "runtime.db")
    try:
        with closing(sqlite3.connect(":memory:")) as source_db:
            replay = AkashaReplayRuntime(
                store=replay_store,
                config=AkashaConfig(),
                source_db_path=tmp_path / "sessions.db",
                source_cursor=source_db.cursor(),
                message_embeddings={},
                message_turn_keys={},
                reinforce_boosts={"s:4": 3.0},
            )
            replay.commit_turn(
                [ReplayMessage(SourceMessage("m2", "s", 2, "user", "beta", QUERY_TS.isoformat()), [1.0, 0.0])],
                [prev],
            )
            replay.commit_turn(
                [ReplayMessage(SourceMessage("m4", "s", 4, "user", "gamma", QUERY_TS.isoformat()), [1.0, 0.0])],
                [current],
            )

        engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
        engine._store = runtime_store
        engine._graph_lock = threading.RLock()
        engine._edges = {}
        engine._edges_meta = {}
        engine._edges_by_src = {}
        engine._fan = {}
        engine._nodes = {}
        engine._prev_activation_by_session = {}
        first_key = runtime_store.upsert_message_node(
            SourceMessage("m2", "s", 2, "user", "beta", QUERY_TS.isoformat()),
            [1.0, 0.0],
        )
        first_node = runtime_store.get_node(first_key)
        assert first_node is not None
        engine._nodes[first_key] = first_node
        engine._commit_pending_activation(
            "s:2",
            PendingActivation(
                query_id="s:2",
                seq=2,
                ts=ts,
                items=[prev],
                query_vec=np.array([1.0, 0.0], dtype=np.float32),
            ),
            "s",
        )
        second_key = runtime_store.upsert_message_node(
            SourceMessage("m4", "s", 4, "user", "gamma", QUERY_TS.isoformat()),
            [1.0, 0.0],
        )
        second_node = runtime_store.get_node(second_key)
        assert second_node is not None
        engine._nodes[second_key] = second_node
        engine._commit_pending_activation(
            "s:4",
            PendingActivation(
                query_id="s:4",
                seq=4,
                ts=ts,
                items=[current],
                query_vec=np.array([1.0, 0.0], dtype=np.float32),
            ),
            "s",
            3.0,
        )

        replay_edges = replay_store.load_edges()
        runtime_edges = runtime_store.load_edges()
        assert replay_edges == pytest.approx(runtime_edges)
        assert ("s:0", "s:4") in replay_edges
        assert ("s:4", "s:0") in replay_edges
    finally:
        replay_store.close()
        runtime_store.close()


def test_replay_writes_query_log_with_activation_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    monkeypatch.setattr("plugins.akasha.core.get_jieba_keywords", lambda _: "")
    replay_store = AkashaStore(tmp_path / "replay.db")
    old_messages = [
        SourceMessage("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
        SourceMessage("s:2", "s", 2, "user", "第二条用户消息只在联想块", "2026-01-01T00:00:02+00:00"),
    ]
    try:
        replay_store.upsert_message_node(old_messages[0], [1.0, 0.0])
        replay_store.upsert_message_node(old_messages[1], [0.98, 0.02])
        with closing(sqlite3.connect(str(db_path))) as source_db:
            replay = AkashaReplayRuntime(
                store=replay_store,
                config=AkashaConfig(dense_seed_threshold=0.1, nearby_dense_threshold=0.0),
                source_db_path=db_path,
                source_cursor=source_db.cursor(),
                message_embeddings={
                    "s:0": np.array([1.0, 0.0], dtype=np.float32),
                    "s:2": np.array([0.98, 0.02], dtype=np.float32),
                },
                message_turn_keys={"s:0": "s:0", "s:2": "s:2"},
            )
            result = replay.replay_turn([
                ReplayMessage(
                    SourceMessage("s:4", "s", 4, "user", "第一条", QUERY_TS.isoformat()),
                    [1.0, 0.0],
                )
            ])

        rows, total = replay_store.list_query_logs(session_key="s", page=1, page_size=10)
        assert total == 1
        raw = replay_store.get_query_log(str(rows[0]["query_id"]))
        assert raw is not None
        activation_items = json.loads(str(raw["activation_items_json"]))
        dense_items = json.loads(str(raw["dense_items_json"]))
        ripple_items = json.loads(str(raw["ripple_items_json"]))
        assert str(rows[0]["query_id"]).startswith("s:4:context:")
        assert rows[0]["intent"] == "context"
        assert rows[0]["activated_count"] == len(result.activation_items)
        assert rows[0]["dense_count"] == len(dense_items)
        assert rows[0]["ripple_count"] == len(ripple_items)
        assert raw["text_block_preview"]
        assert activation_items
        assert dense_items
        assert all(item["happened_at"] for item in dense_items)
        assert isinstance(ripple_items, list)
        assert activation_items[0]["user_message"] in {
            "第一条用户消息需要完整展示",
            "第二条用户消息只在联想块",
        }
    finally:
        replay_store.close()


def test_replay_empty_query_commits_without_activation_or_query_log(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    replay_store = AkashaStore(tmp_path / "replay.db")
    replay_store.upsert_message_node(
        SourceMessage("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
        [1.0, 0.0],
    )
    try:
        with closing(sqlite3.connect(str(db_path))) as source_db:
            replay = AkashaReplayRuntime(
                store=replay_store,
                config=AkashaConfig(dense_seed_threshold=0.1, nearby_dense_threshold=0.0),
                source_db_path=db_path,
                source_cursor=source_db.cursor(),
                message_embeddings={"s:0": np.array([1.0, 0.0], dtype=np.float32)},
                message_turn_keys={"s:0": "s:0"},
            )
            result = replay.replay_turn([
                ReplayMessage(
                    SourceMessage("s:4", "s", 4, "user", "", QUERY_TS.isoformat()),
                    [0.0, 0.0],
                )
            ])

        rows, total = replay_store.list_query_logs(session_key="s", page=1, page_size=10)
        assert result.current_key == "s:4"
        assert result.activation_items == []
        assert total == 0
        assert rows == []
        with closing(sqlite3.connect(str(tmp_path / "replay.db"))) as db:
            assert db.execute("SELECT COUNT(*) FROM akasha_activation_events").fetchone()[0] == 0
    finally:
        replay_store.close()


def test_query_log_content_loader_allows_empty_user_message(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("m0", "s", 0, "user", "", QUERY_TS.isoformat()),
                ("m1", "s", 1, "assistant", "assistant preview", QUERY_TS.isoformat()),
            ],
        )
        db.commit()
        user_message, assistant_preview = _turn_messages(
            db.cursor(),
            "s:0",
            assistant_preview_chars=9,
        )

    assert user_message == ""
    assert assistant_preview == "assistant..."


def test_akasha_rebuild_skips_scheduler_messages() -> None:
    scheduler_user = SourceMessage(
        "scheduler:job:0",
        "scheduler:job",
        0,
        "user",
        "查询北京天气",
        "2026-01-01T00:00:00+00:00",
    )
    normal_user = SourceMessage(
        "telegram:1:0",
        "telegram:1",
        0,
        "user",
        "今天聊 Akasha",
        "2026-01-01T00:00:01+00:00",
    )

    assert _skip_message(scheduler_user, set()) is True
    assert _skip_message(normal_user, set()) is False
    assert list(_iter_replay_turns([scheduler_user, normal_user], set())) == [[normal_user]]


@pytest.mark.asyncio
async def test_runtime_skips_scheduler_turn_even_without_extra_flag(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._session_db_path = db_path
    engine._embedder = SimpleNamespace(embed_batch=AsyncMock(side_effect=AssertionError("should skip")))

    await engine._on_turn_committed(
        TurnCommitted(
            session_key="scheduler:job",
            channel="telegram",
            chat_id="1",
            input_message="查询天气",
            persisted_user_message="查询天气",
            assistant_response="天气回复",
            tools_used=[],
        )
    )

    engine._embedder.embed_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_writes_message_embeddings_to_sessions_db(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    graph_store = AkashaStore(tmp_path / "akasha.db")
    embedding_store = MessageEmbeddingStore(db_path)
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._session_db_path = db_path
    engine._store = graph_store
    engine._embedding_store = embedding_store
    engine._embedder = SimpleNamespace(
        embed_batch=AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    )
    engine._config = SimpleNamespace(
        memory=SimpleNamespace(embedding=SimpleNamespace(model="m"))
    )
    engine._pending_by_session = {}
    engine._graph_lock = threading.RLock()
    engine._nodes = {}
    engine._message_embeddings = {}
    engine._message_turn_keys = {}
    engine._message_timestamps = {}
    engine._message_index = build_dense_message_index({})
    try:
        await engine._on_turn_committed(
            TurnCommitted(
                session_key="s",
                channel="telegram",
                chat_id="1",
                input_message="第一条用户消息需要完整展示",
                persisted_user_message="第一条用户消息需要完整展示",
                assistant_response="第一条助手回复会被截断展示并保留引用",
                tools_used=[],
            )
        )

        assert embedding_store.get(
            message_id="s:0",
            content="第一条用户消息需要完整展示",
            model="m",
        ) == [1.0, 0.0]
        assert embedding_store.get(
            message_id="s:1",
            content="第一条助手回复会被截断展示并保留引用",
            model="m",
        ) == [0.0, 1.0]
        with closing(sqlite3.connect(str(graph_store.db_path))) as db:
            legacy_count = db.execute(
                "SELECT COUNT(1) FROM akasha_embedding_cache"
            ).fetchone()[0]
        assert legacy_count == 0
    finally:
        embedding_store.close()
        graph_store.close()


@pytest.mark.asyncio
async def test_runtime_rejects_partial_embedding_batch(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    graph_store = AkashaStore(tmp_path / "akasha.db")
    embedding_store = MessageEmbeddingStore(db_path)
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._session_db_path = db_path
    engine._store = graph_store
    engine._embedding_store = embedding_store
    engine._embedder = SimpleNamespace(
        embed_batch=AsyncMock(return_value=[[1.0, 0.0]])
    )
    engine._config = SimpleNamespace(
        memory=SimpleNamespace(embedding=SimpleNamespace(model="m"))
    )
    try:
        with pytest.raises(ValueError, match="数量与已提交消息数量不一致"):
            await engine._on_turn_committed(
                TurnCommitted(
                    session_key="s",
                    channel="telegram",
                    chat_id="1",
                    input_message="第一条用户消息需要完整展示",
                    persisted_user_message="第一条用户消息需要完整展示",
                    assistant_response="第一条助手回复会被截断展示并保留引用",
                    tools_used=[],
                )
            )

        assert embedding_store.get(
            message_id="s:0",
            content="第一条用户消息需要完整展示",
            model="m",
        ) is None
    finally:
        embedding_store.close()
        graph_store.close()


def test_load_turn_card_uses_full_user_and_short_assistant(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    card = _load_turn_card(
        db_path,
        "s:0",
        assistant_preview_chars=15,
        score=0.8,
        lane="dense",
        signals={},
    )

    assert card is not None
    assert card.user_message == "第一条用户消息需要完整展示"
    assert card.assistant_preview == "第一条助手回复会被截断展示并保..."
    assert card.source_ref == '["s:0", "s:1"]'
    assert card.happened_at == "2026-01-01T00:00:00+00:00"


def test_historical_snapshot_excludes_future_messages_nodes_and_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    embedding_store = MessageEmbeddingStore(db_path)
    for message_id, content, vector in (
        ("s:0", "第一条用户消息需要完整展示", [1.0, 0.0]),
        ("s:1", "第一条助手回复会被截断展示并保留引用", [0.0, 1.0]),
        ("s:2", "第二条用户消息只在联想块", [0.0, 1.0]),
    ):
        embedding_store.upsert(
            message_id=message_id,
            content=content,
            model="m",
            embedding=vector,
        )
    monkeypatch.setattr(
        embedding_store,
        "list_until",
        lambda **_: pytest.fail("historical snapshot must use the engine cache"),
    )
    cutoff = datetime.fromisoformat("2026-01-01T00:00:00.500000+00:00").timestamp()
    past_node = AkashaNode(
        key="s:0",
        anchor_id="s:0",
        session_key="s",
        turn_seq=0,
        first_ts_unix=datetime.fromisoformat("2026-01-01T00:00:00+00:00").timestamp(),
        salience=0.5,
        strength=3.0,
        resource=0.1,
        recall_count=9,
        last_activated_ts=cutoff + 10,
        last_strength_ts=cutoff + 10,
        last_resource_ts=cutoff + 10,
        embedding=np.array([0.0, 1.0], dtype=np.float32),
        emb_count=2,
    )
    future_node = replace(
        past_node,
        key="s:2",
        anchor_id="s:2",
        turn_seq=2,
        first_ts_unix=cutoff + 10,
    )
    snapshot = AkashaActivationSnapshot(
        nodes={"s:0": past_node, "s:2": future_node},
        edges={("s:0", "s:2"): 1.0},
        edges_meta={("s:0", "s:2"): cutoff + 10},
        fan={"s:0": 1, "s:2": 1},
        edges_by_src={"s:0": {"s:2": 1.0}},
        message_embeddings={
            "s:0": np.array([1.0, 0.0], dtype=np.float32),
            "s:1": np.array([0.0, 1.0], dtype=np.float32),
            "s:2": np.array([0.0, 1.0], dtype=np.float32),
        },
        message_turn_keys={"s:0": "s:0", "s:1": "s:0", "s:2": "s:2"},
        message_index=build_dense_message_index({
            "s:0": np.array([1.0, 0.0], dtype=np.float32),
            "s:1": np.array([0.0, 1.0], dtype=np.float32),
            "s:2": np.array([0.0, 1.0], dtype=np.float32),
        }),
    )
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._embedding_store = embedding_store
    engine._graph_lock = threading.RLock()
    engine._nodes = snapshot.nodes
    engine._edges = snapshot.edges
    engine._edges_meta = snapshot.edges_meta
    engine._edges_by_src = snapshot.edges_by_src
    engine._fan = snapshot.fan
    engine._message_embeddings = snapshot.message_embeddings
    engine._message_turn_keys = snapshot.message_turn_keys
    engine._message_index = snapshot.message_index
    engine._message_timestamps = {
        "s:0": datetime.fromisoformat("2026-01-01T00:00:00+00:00").timestamp(),
        "s:1": datetime.fromisoformat("2026-01-01T00:00:01+00:00").timestamp(),
        "s:2": datetime.fromisoformat("2026-01-01T00:00:02+00:00").timestamp(),
    }
    engine._config = SimpleNamespace(
        memory=SimpleNamespace(embedding=SimpleNamespace(model="m"))
    )
    try:
        visible = engine._graph_snapshot_at(cutoff)
        card = _load_turn_card(
            db_path,
            "s:0",
            assistant_preview_chars=30,
            score=0.8,
            lane="dense",
            signals={},
            cutoff="2026-01-01T00:00:00.500000+00:00",
        )

        assert set(visible.nodes) == {"s:0"}
        assert set(visible.message_embeddings) == {"s:0"}
        assert visible.edges == {}
        assert visible.nodes["s:0"].embedding.tolist() == [1.0, 0.0]
        assert visible.nodes["s:0"].recall_count == 0
        assert card is not None
        assert card.assistant_preview == ""
        assert card.source_ref == '["s:0"]'
    finally:
        embedding_store.close()


@pytest.mark.asyncio
async def test_query_places_overlap_in_dense_and_ripple_only_in_ripple(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig(assistant_preview_chars=15)
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    engine._remember_pending_activation = lambda *_, **__: None
    engine._retrieve = lambda query, query_vec, request, *, now_ts, update_state: _AkashaRetrieval(
        dense_items=[
            AkashaCandidate(
                key="s:0",
                source="Dense",
                ripple=0.0,
                direct=0.9,
                state=0.0,
                edge=0.0,
                long=0.0,
                resource=1.0,
                fan=0,
                score=0.9,
            )
        ],
        ripple_items=[
            AkashaCandidate(
                key="s:0",
                source="Dense",
                ripple=0.6,
                direct=0.9,
                state=1.0,
                edge=0.0,
                long=0.0,
                resource=1.0,
                fan=0,
                score=0.8,
            ),
            AkashaCandidate(
                key="s:2",
                source="Graph",
                ripple=0.5,
                direct=0.4,
                state=0.8,
                edge=0.2,
                long=0.0,
                resource=1.0,
                fan=1,
                score=0.7,
            ),
        ],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=2),
        seq=4,
    )

    result = await engine.query(
        MemoryQuery(
            text="用户消息",
            intent="context",
            scope=MemoryScope(session_key="s"),
            timestamp=QUERY_TS,
        )
    )

    assert "## 左脑记忆：精确回忆" in result.text_block
    assert "## 右脑联想：潜意识第一反应" in result.text_block
    assert "# Akasha memory now=" in result.text_block
    assert '- user="第一条用户消息需要完整展示" assistant=' in result.text_block
    assert " t=01-01 source_ref=" in result.text_block
    assert " score=" not in result.text_block
    dense_block, ripple_block = result.text_block.split("## 右脑联想：潜意识第一反应", 1)
    assert 'source_ref=["s:0", "s:1"]' in dense_block
    assert 'source_ref=["s:0", "s:1"]' not in ripple_block
    assert 'source_ref=["s:2", "s:3"]' in ripple_block


@pytest.mark.asyncio
async def test_context_block_sorts_injected_cards_by_time_desc(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    def candidate(key: str, score: float) -> AkashaCandidate:
        return AkashaCandidate(
            key=key,
            source="Dense",
            ripple=0.0,
            direct=score,
            state=0.0,
            edge=0.0,
            long=0.0,
            resource=1.0,
            fan=0,
            score=score,
        )

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig()
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    engine._remember_pending_activation = lambda *_, **__: None
    engine._retrieve = lambda query, query_vec, request, *, now_ts, update_state: _AkashaRetrieval(
        dense_items=[candidate("s:0", 0.9), candidate("s:2", 0.8)],
        ripple_items=[],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=2),
        seq=4,
    )

    result = await engine.query(
        MemoryQuery(
            text="用户消息",
            intent="context",
            scope=MemoryScope(session_key="s"),
            timestamp=QUERY_TS,
        )
    )

    assert result.text_block.index('source_ref=["s:2", "s:3"]') < result.text_block.index(
        'source_ref=["s:0", "s:1"]'
    )


def test_cards_from_keys_deduplicates_same_user_assistant_pair(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("s:0", "s", 0, "user", "我现在健康状态怎么样呢", "2026-01-01T00:00:00+00:00"),
                ("s:1", "s", 1, "assistant", "健康状态的话……我这边真的没有更多信息", "2026-01-01T00:00:01+00:00"),
                ("s:2", "s", 2, "user", "我现在健康状态怎么样呢", "2026-01-01T00:00:02+00:00"),
                ("s:3", "s", 3, "assistant", "健康状态的话……我这边真的没有更多信息", "2026-01-01T00:00:03+00:00"),
                ("s:4", "s", 4, "user", "另一个问题", "2026-01-01T00:00:04+00:00"),
                ("s:5", "s", 5, "assistant", "第三次回复", "2026-01-01T00:00:05+00:00"),
            ],
        )
        db.commit()

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig(assistant_preview_chars=15)
    engine._session_db_path = db_path

    cards = engine._cards_from_keys(
        [
            ("s:0", 0.9, "ripple", {}),
            ("s:2", 0.8, "ripple", {}),
            ("s:4", 0.7, "ripple", {}),
        ],
        limit=10,
    )

    assert [card.source_ref for card in cards] == [
        '["s:0", "s:1"]',
        '["s:4", "s:5"]',
    ]


@pytest.mark.asyncio
async def test_context_query_uses_akasha_top_k_over_default_query_limit(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        rows = []
        for turn in range(24):
            user_seq = turn * 2
            rows.append((f"s:{user_seq}", "s", user_seq, "user", f"用户消息{turn}", "2026-01-01T00:00:00+00:00"))
            rows.append((f"s:{user_seq + 1}", "s", user_seq + 1, "assistant", f"助手回复{turn}", "2026-01-01T00:00:01+00:00"))
        db.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)", rows)
        db.commit()

    def candidate(key: str, score: float) -> AkashaCandidate:
        return AkashaCandidate(
            key=key,
            source="Dense",
            ripple=0.0,
            direct=score,
            state=0.0,
            edge=0.0,
            long=0.0,
            resource=1.0,
            fan=0,
            score=score,
        )

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig(inject_max_chars=20000)
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    engine._remember_pending_activation = lambda *_, **__: None
    engine._retrieve = lambda query, query_vec, request, *, now_ts, update_state: _AkashaRetrieval(
        dense_items=[candidate(f"s:{turn * 2}", 1.0 - turn * 0.01) for turn in range(12)],
        ripple_items=[candidate(f"s:{24 + turn * 2}", 0.8 - turn * 0.01) for turn in range(12)],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=24),
        seq=48,
    )

    result = await engine.query(
        MemoryQuery(
            text="用户消息",
            intent="context",
            scope=MemoryScope(session_key="s"),
            limit=8,
            timestamp=QUERY_TS,
        )
    )

    assert result.trace["dense_count"] == 10
    assert result.trace["ripple_count"] == 10
    assert result.text_block.count("source_ref=") == 20


def test_compute_candidates_uses_activation_limit_for_stateful_replay(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        for seq in range(30):
            _ = store.upsert_message_node(
                SourceMessage(
                    f"s:{seq}",
                    "s",
                    seq,
                    "user",
                    f"消息 {seq}",
                    "2026-01-01T00:00:00+00:00",
                    salience=1.0,
                ),
                [1.0, 0.0],
            )
        nodes = {node.key: node for node in store.list_nodes()}
    finally:
        store.close()

    candidates, suppressed, trace = _compute_candidates(
        "消息",
        np.array([1.0, 0.0], dtype=np.float32),
        nodes,
        {},
        100,
        config=AkashaConfig(),
        fan={},
        soft_recall=False,
        return_limit=8,
    )

    assert len(candidates) == 8
    assert trace.seed_count == 30
    assert suppressed == []


def test_query_log_keeps_context_and_answer_for_same_seq(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._store = store
    engine._akasha_config = AkashaConfig()
    result = _AkashaRetrieval(
        dense_items=[],
        ripple_items=[],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=2),
        seq=10,
    )
    try:
        cases: list[tuple[MemoryQueryIntent, str]] = [("context", "注入文本"), ("answer", "")]
        for intent, text_block in cases:
            engine._write_query_log(
                request=MemoryQuery(
                    text="同一轮问题",
                    intent=intent,
                    scope=MemoryScope(session_key="s"),
                    timestamp=QUERY_TS,
                ),
                result=result,
                seq=10,
                dense_cards=[],
                ripple_cards=[],
                text_block=text_block,
            )

        items, total = store.list_query_logs(session_key="s", page=1, page_size=10)
    finally:
        store.close()

    assert total == 2
    assert {item["intent"] for item in items} == {"context", "answer"}


def test_live_query_log_rejects_invalid_internal_source_ref(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._store = store
    engine._session_db_path = tmp_path / "sessions.db"
    engine._akasha_config = AkashaConfig()
    result = _AkashaRetrieval(
        dense_items=[],
        ripple_items=[],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=1),
        seq=2,
    )
    invalid_card = AkashaCard(
        key="s:0",
        source_ref="not-json",
        user_message="历史消息",
        assistant_preview="",
        happened_at=QUERY_TS.isoformat(),
        score=0.8,
        lane="dense",
        signals={},
    )
    try:
        with pytest.raises(json.JSONDecodeError):
            engine._write_query_log(
                request=MemoryQuery(
                    text="当前问题",
                    intent="context",
                    scope=MemoryScope(session_key="s"),
                    timestamp=QUERY_TS,
                ),
                result=result,
                seq=2,
                dense_cards=[invalid_card],
                ripple_cards=[],
                text_block="",
            )
        _, total = store.list_query_logs(session_key="s", page=1, page_size=10)
        assert total == 0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_read_only_query_skips_akasha_state_effects(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig()
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    side_effects: list[str] = []
    update_state_values: list[bool] = []

    def fake_retrieve(
        query: str,
        query_vec: np.ndarray,
        request: MemoryQuery,
        *,
        now_ts: float,
        update_state: bool,
    ) -> _AkashaRetrieval:
        _ = (query, query_vec, request, now_ts)
        update_state_values.append(update_state)
        return _AkashaRetrieval(
            dense_items=[_candidate("s:0", 0.9)],
            ripple_items=[],
            activation_items=[_candidate("s:2", 0.8)],
            trace=ActivationTrace(seed_count=1, pool_count=1),
            seq=4,
        )

    engine._retrieve = fake_retrieve
    engine._remember_pending_activation = lambda *_, **__: side_effects.append("pending")
    engine._write_query_log = lambda *_, **__: side_effects.append("query_log")

    request = MemoryQuery(
        text="用户消息",
        intent="answer",
        effect="read_only",
        scope=MemoryScope(session_key="s"),
        timestamp=QUERY_TS,
    )
    result = await engine.query(request)

    assert update_state_values == [False]
    assert side_effects == []
    assert result.trace["effect"] == "read_only"
    assert result.records

    invalid_card = AkashaCard(
        key="s:0",
        source_ref="{}",
        user_message="历史消息",
        assistant_preview="",
        happened_at=QUERY_TS.isoformat(),
        score=0.9,
        lane="dense",
        signals={},
    )

    def invalid_cards(
        items: list[tuple[str, float, str, dict[str, object]]],
        **_: object,
    ) -> list[AkashaCard]:
        return [invalid_card] if items else []

    engine._cards_from_keys = invalid_cards
    with pytest.raises(ValueError, match="source_ref 必须是 JSON 数组"):
        await engine.query(request)

    assert update_state_values == [False, False]
    assert side_effects == []


def test_undo_removes_akasha_turn_state_after_session_delete(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    store = AkashaStore(tmp_path / "akasha.db")
    embedding_store = MessageEmbeddingStore(db_path)
    try:
        messages = [
            SourceMessage("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
            SourceMessage("s:1", "s", 1, "assistant", "第一条助手回复会被截断展示并保留引用", "2026-01-01T00:00:01+00:00"),
            SourceMessage("s:2", "s", 2, "user", "第二条用户消息只在联想块", "2026-01-01T00:00:02+00:00"),
        ]
        for index, message in enumerate(messages):
            embedding = [1.0, 0.0] if index < 2 else [0.0, 1.0]
            embedding_store.upsert(
                message_id=message.id,
                content=message.content,
                model="m",
                embedding=embedding,
            )
            _ = store.upsert_message_node(message, embedding)
        store.upsert_edges([
            EdgeUpdate("s:0", "s:2", 1.0, 0),
            EdgeUpdate("s:2", "s:0", 1.0, 0),
        ])
        store.insert_activation_events([
            ActivationEventRow(
                seq=0,
                query_id="s:0",
                activated_key="s:2",
                source="Dense",
                score=0.8,
                direct_score=0.8,
                state_score=0.0,
                edge_score=0.0,
                long_score=0.0,
                resource=1.0,
                fan=0,
            )
        ])
        store.insert_query_log(
            query_id="s:0:context:abc",
            session_key="s",
            seq=0,
            query_text="第一条用户消息",
            intent="context",
            ts="2026-01-01T00:00:00+00:00",
            seed_count=1,
            pool_count=2,
            activated_count=1,
            activation_threshold=0.2,
            dense_count=1,
            ripple_count=1,
            inject_chars=10,
            source_ref_count=2,
            activation_items_json="[]",
            dense_items_json="[]",
            ripple_items_json="[]",
            text_block_preview="preview",
        )

        engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
        engine._store = store
        engine._embedding_store = embedding_store
        engine._session_db_path = db_path
        engine._config = SimpleNamespace(
            memory=SimpleNamespace(embedding=SimpleNamespace(model="m"))
        )
        engine._graph_lock = threading.RLock()
        engine._nodes = {}
        engine._edges = {}
        engine._edges_by_src = {}
        engine._fan = {}
        engine._message_embeddings = {}
        engine._message_turn_keys = {}
        engine._load_graph_cache()

        dry_run = engine.undo_by_message_sources(["s:0", "s:1"], dry_run=True)
        with closing(sqlite3.connect(str(db_path))) as db:
            _ = db.execute("DELETE FROM messages WHERE id IN ('s:0', 's:1')")
            db.commit()
        result = engine.undo_by_message_sources(["s:0", "s:1"])

        assert dry_run["affected_ids"] == ["s:0"]
        assert result["affected_ids"] == ["s:0"]
        assert result["restored_ids"] == []
        assert result["rollback_source_ids"] == ["s:0", "s:1"]
        assert store.get_node("s:0") is None
        assert store.get_node("s:2") is not None
        assert store.load_edges() == {}
        assert store.list_query_logs(page=1, page_size=10)[1] == 0
        with closing(sqlite3.connect(str(store.db_path))) as db:
            event_count = db.execute("SELECT COUNT(1) FROM akasha_activation_events").fetchone()[0]
        with closing(sqlite3.connect(str(db_path))) as db:
            cache_count = db.execute("SELECT COUNT(1) FROM message_embeddings").fetchone()[0]
        assert event_count == 0
        assert cache_count == 1
        assert "s:0" not in engine._nodes
        assert ("s:0", "s:2") not in engine._edges
        assert "s:0" not in engine._message_embeddings
        assert "s:1" not in engine._message_turn_keys
    finally:
        embedding_store.close()
        store.close()


def test_akashalast_command_only_exposes_for_akasha_engine(tmp_path: Path) -> None:
    akasha = AkashaPlugin()
    akasha.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        data_dir=tmp_path / ".data",
        kv_store=PluginKVStore(tmp_path / ".akasha-kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )
    default = AkashaPlugin()
    default.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        data_dir=tmp_path / ".data",
        kv_store=PluginKVStore(tmp_path / ".default-kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="default")),
    )

    assert akasha.telegram_bot_commands() == [("akashalast", "查看上一轮 Akasha 检索诊断")]
    assert akasha.mobile_bot_commands() == [("akashalast", "查看上一轮 Akasha 检索诊断")]
    assert len(akasha.before_turn_modules()) == 1
    assert default.telegram_bot_commands() == []
    assert default.mobile_bot_commands() == []
    assert len(default.before_turn_modules()) == 1


def test_akashalast_renders_latest_query_log(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "memory" / "akasha.db")
    try:
        activation_items = json.dumps([
            {
                "user_message": "这个是他转的别人的帖子而已",
                "assistant_preview": "啊你说得对，那个是转推",
                "score": 0.501,
                "source": "Dense",
                "path_type": "direct",
            }
        ], ensure_ascii=False)
        dense_items = json.dumps([
            {
                "user_message": "这个是他转的别人的帖子而已",
                "assistant_preview": "啊你说得对，那个是转推",
                "score": 0.703,
                "source": "Dense",
            }
        ], ensure_ascii=False)
        ripple_items = json.dumps([
            {
                "user_message": "我纠正过你几次有关汪远哲这个名字",
                "assistant_preview": "花月哥哥，这个错误我真是犯过",
                "score": 0.247,
                "source": "FTS",
                "path_type": "direct",
                "direct": 0.41,
                "state": 0.18,
                "edge": 0.08,
                "resource": 1.0,
                "fan": 32,
            }
        ], ensure_ascii=False)
        store.insert_query_log(
            query_id="s:2:context:abc",
            session_key="s",
            seq=2,
            query_text="这个其实不是她的 是她转发的别人的帖子",
            intent="context",
            ts="2026-05-24T22:15:00+08:00",
            seed_count=11,
            pool_count=81,
            activated_count=4,
            activation_threshold=0.22,
            dense_count=1,
            ripple_count=1,
            inject_chars=100,
            source_ref_count=2,
            activation_items_json=activation_items,
            dense_items_json=dense_items,
            ripple_items_json=ripple_items,
            text_block_preview="preview",
        )
    finally:
        store.close()

    plugin = AkashaPlugin()
    plugin.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        data_dir=tmp_path / ".data",
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )

    reply = plugin.render_last_query("s")

    assert "🧠 Akasha 记忆检索诊断" in reply
    assert "📍 会话: `s` | seq `2`" in reply
    assert "• 种子节点 (Seeds): `11` 个" in reply
    assert "🔥 本轮图激活节点 (Activated Nodes):" in reply
    assert "🎯 左脑精确回忆 (Dense):" in reply
    assert "🌊 右脑联想记忆 (Ripple):" in reply
    assert "分: `0.501` | 源: `Dense` | 径: `direct`" in reply
    assert "得: `0.703` | 源: `Dense`" in reply
    assert "因: `dir:0.41 st:0.18 edg:0.08 res:1.00 fan:32`" in reply


@pytest.mark.asyncio
async def test_mobile_recall_binds_each_assistant_to_its_context_and_keeps_all_items(
    tmp_path: Path,
) -> None:
    _init_sessions_db(tmp_path / "sessions.db")
    store = AkashaStore(tmp_path / "memory" / "akasha.db")

    def insert_log(
        *,
        query_id: str,
        seq: int,
        intent: str,
        ts: str,
        dense_items: list[dict[str, object]],
        ripple_items: list[dict[str, object]],
    ) -> None:
        store.insert_query_log(
            query_id=query_id,
            session_key="s",
            seq=seq,
            query_text=query_id,
            intent=intent,
            ts=ts,
            seed_count=0,
            pool_count=0,
            activated_count=0,
            activation_threshold=0.0,
            dense_count=len(dense_items),
            ripple_count=len(ripple_items),
            inject_chars=0,
            source_ref_count=0,
            activation_items_json="[]",
            dense_items_json=json.dumps(dense_items, ensure_ascii=False),
            ripple_items_json=json.dumps(ripple_items, ensure_ascii=False),
            text_block_preview="",
        )

    first_dense = [
        {
            "key": "s:0",
            "user_message": "第一轮旧日志",
            "assistant_preview": "从 sessions.db 补时间",
            "score": 0.9,
        }
    ]
    second_dense = [
        {
            "key": f"memory:{index}",
            "user_message": f"左脑 {index}",
            "assistant_preview": "",
            "happened_at": f"2026-01-01T00:00:{index:02d}+00:00",
            "score": index / 10,
        }
        for index in range(8)
    ]
    second_ripple = [
        {
            "key": f"ripple:{index}",
            "user_message": f"右脑 {index}",
            "assistant_preview": "",
            "happened_at": f"2026-01-01T00:01:{index:02d}+00:00",
            "score": index / 10,
        }
        for index in range(9)
    ]
    try:
        insert_log(
            query_id="s:0:context:first",
            seq=0,
            intent="context",
            ts="2026-01-01T00:00:00+00:00",
            dense_items=first_dense,
            ripple_items=[],
        )
        insert_log(
            query_id="s:2:context:second",
            seq=2,
            intent="context",
            ts="2026-01-01T00:00:02+00:00",
            dense_items=second_dense,
            ripple_items=second_ripple,
        )
        insert_log(
            query_id="s:2:answer:later",
            seq=2,
            intent="answer",
            ts="2026-01-01T00:00:03+00:00",
            dense_items=[{"user_message": "不能覆盖 context"}],
            ripple_items=[],
        )
    finally:
        store.close()

    plugin = AkashaPlugin()
    plugin.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        data_dir=tmp_path / ".data",
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )

    first = plugin.mobile_ui_query(
        "recall.current",
        {"message_id": "s:1"},
        session_id="s",
        turn_id=None,
    )
    second = plugin.mobile_ui_query(
        "recall.current",
        {"message_id": "s:3"},
        session_id="s",
        turn_id=None,
    )
    active = plugin.mobile_ui_query(
        "recall.current",
        {"message_id": "assistant:turn-1"},
        session_id="s",
        turn_id="turn-1",
    )
    interrupted = plugin.mobile_ui_query(
        "recall.current",
        {"message_id": "assistant:turn-1"},
        session_id="s",
        turn_id=None,
    )

    assert [item["summary"] for item in cast(list[dict[str, object]], first["left"])] == [
        "第一轮旧日志"
    ]
    second_left = cast(list[dict[str, object]], second["left"])
    second_right = cast(list[dict[str, object]], second["right"])
    assert len(second_left) == 8
    assert len(second_right) == 9
    assert [item["summary"] for item in second_left] == [f"左脑 {index}" for index in reversed(range(8))]
    assert [item["summary"] for item in second_right] == [f"右脑 {index}" for index in reversed(range(9))]
    assert active == second
    assert interrupted == {"left": [], "right": []}


@pytest.mark.asyncio
async def test_mobile_recall_rejects_message_from_another_session(tmp_path: Path) -> None:
    plugin = AkashaPlugin()
    plugin.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        data_dir=tmp_path / ".data",
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )

    with pytest.raises(ValueError, match="不属于当前 session"):
        plugin.mobile_ui_query(
            "recall.current",
            {"message_id": "other:1"},
            session_id="s",
            turn_id=None,
        )


def test_mobile_inspector_lists_and_expands_existing_query_logs_read_only(
    tmp_path: Path,
) -> None:
    _init_sessions_db(tmp_path / "sessions.db")
    db_path = tmp_path / "memory" / "akasha.db"
    store = AkashaStore(db_path)
    long_query = "为什么想起这轮：" + "很长的问题原文" * 40
    try:
        for index in range(2):
            store.insert_query_log(
                query_id=f"s:{index * 2}:context:q{index}",
                session_key="s",
                seq=index * 2,
                query_text=long_query if index == 1 else "为什么想起第 0 轮",
                intent="context",
                ts=f"2026-07-17T0{index}:00:00+00:00",
                seed_count=2,
                pool_count=4,
                activated_count=3,
                activation_threshold=0.2,
                dense_count=1,
                ripple_count=1,
                inject_chars=1200 + index,
                source_ref_count=2,
                activation_items_json="[]",
                dense_items_json=json.dumps([{
                    "key": f"dense:{index}",
                    "user_message": f"精确记忆 {index}",
                    "assistant_preview": "精确回答",
                    "happened_at": f"2026-07-16T0{index}:00:00+00:00",
                    "score": 0.8,
                }], ensure_ascii=False),
                ripple_items_json=json.dumps([{
                    "key": f"ripple:{index}",
                    "user_message": f"联想记忆 {index}",
                    "assistant_preview": "联想回答",
                    "happened_at": f"2026-07-15T0{index}:00:00+00:00",
                    "score": 0.6,
                }], ensure_ascii=False),
                text_block_preview="",
            )
    finally:
        store.close()
    before_bytes = db_path.read_bytes()

    plugin = AkashaPlugin()
    plugin.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        data_dir=tmp_path / ".data",
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )

    recent = plugin.mobile_ui_query(
        "inspector.recent",
        {},
        session_id=None,
        turn_id=None,
    )
    items = cast(list[dict[str, object]], recent["items"])
    detail = plugin.mobile_ui_query(
        "inspector.detail",
        {"query_id": items[0]["query_id"]},
        session_id=None,
        turn_id=None,
    )

    assert recent["total"] == 2
    assert [item["query_preview"] for item in items] == [
        long_query[:180] + "...",
        "为什么想起第 0 轮",
    ]
    assert detail["query_text"] == long_query
    assert detail["left_count"] == 1
    assert detail["right_count"] == 1
    assert cast(list[dict[str, object]], detail["left"])[0]["summary"] == "精确记忆 1"
    assert cast(list[dict[str, object]], detail["right"])[0]["summary"] == "联想记忆 1"
    with closing(sqlite3.connect(db_path)) as db:
        assert db.total_changes == 0
        assert db.execute("SELECT COUNT(1) FROM akasha_query_log").fetchone()[0] == 2
    assert db_path.read_bytes() == before_bytes


def test_mobile_inspector_rejects_invalid_or_missing_query(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "memory" / "akasha.db")
    store.close()
    plugin = AkashaPlugin()
    plugin.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        data_dir=tmp_path / ".data",
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )

    with pytest.raises(ValueError, match="不接受参数"):
        plugin.mobile_ui_query(
            "inspector.recent",
            {"page": 2},
            session_id=None,
            turn_id=None,
        )
    with pytest.raises(ValueError, match="非空字符串"):
        plugin.mobile_ui_query(
            "inspector.detail",
            {"query_id": ""},
            session_id=None,
            turn_id=None,
        )
    with pytest.raises(ValueError, match="不存在"):
        plugin.mobile_ui_query(
            "inspector.detail",
            {"query_id": "missing"},
            session_id=None,
            turn_id=None,
        )


def test_akasha_store_read_only_mode_never_creates_or_writes_database(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "akasha.db"
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        AkashaStore(missing, read_only=True)
    assert not missing.parent.exists()

    db_path = tmp_path / "memory" / "akasha.db"
    writer = AkashaStore(db_path)
    writer.close()
    before_bytes = db_path.read_bytes()
    reader = AkashaStore(db_path, read_only=True)
    try:
        assert reader.db.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.db.execute(
                "INSERT INTO akasha_query_log "
                "(query_id, session_key, seq, query_text, intent, ts) "
                "VALUES ('q', 's', 0, 'x', 'context', '2026-07-17')"
            )
    finally:
        reader.close()
    assert db_path.read_bytes() == before_bytes
