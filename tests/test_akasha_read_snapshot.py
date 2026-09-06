import pytest
import sqlite3

from plugins.akasha.application.cycle import MemoryCycle
from plugins.akasha.application.snapshot import read_memory
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.infrastructure.lease import WriterLease
from plugins.akasha.infrastructure.persistence import logical_state_sha256
from session.message import Output
from tests.test_akasha_message_queries import memory_runtime


@pytest.mark.asyncio
async def test_read_snapshot_restores_while_live_writer_remains_owned_and_can_advance(tmp_path, monkeypatch):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        write("u1", "first memory")
        write("a1", "first answer", Output)
        assert await runtime.consume() == 1
        graph = logical_state_sha256(tmp_path / "memory.db")
        calls_before = list(calls)
        def no_replay(*args, **kwargs):
            raise AssertionError("reading a snapshot must not replay learning")
        with monkeypatch.context() as patch:
            patch.setattr(MemoryCycle, "commit", no_replay)
            async with read_memory(
                tmp_path / "memory.db", legacy_index=None, catalog=log.catalog(),
                embeddings=runtime._embeddings, bindings=runtime._bindings, config=MemoryConfig(),
            ) as (cycle, state):
                assert cycle.state_version == 1
                assert state == consumer.state
                assert cycle.turns[0].user_text == "first memory"
                assert calls == calls_before
                assert logical_state_sha256(tmp_path / "memory.db") == graph
                with pytest.raises(RuntimeError, match="already has a writer"):
                    WriterLease(tmp_path / "memory.db")
                # 正式 writer 发布下一版，不改变已经固定的读副本。
                patch.undo()
                write("u2", "second memory")
                write("a2", "second answer", Output)
                assert await runtime.consume() == 1
                assert cycle.state_version == 1
                assert len(state.applied) == 1
                assert consumer.cycle.state_version == 2
        with pytest.raises(RuntimeError, match="already has a writer"):
            WriterLease(tmp_path / "memory.db")


@pytest.mark.asyncio
async def test_read_snapshot_missing_graph_never_creates_formal_storage(tmp_path):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        missing = tmp_path / "missing.db"
        with pytest.raises(sqlite3.OperationalError):
            async with read_memory(missing, legacy_index=None, catalog=log.catalog(),
                embeddings=runtime._embeddings, bindings=runtime._bindings, config=MemoryConfig()):
                pytest.fail("missing graph was presented as a valid snapshot")
        assert not missing.exists()
        assert not missing.with_suffix(".db.lock").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_initial_material_snapshot_uses_original_cutover_and_rejects_lost_graph(tmp_path, legacy):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        write("existing", "already admitted before initialization")
        missing, index = tmp_path / "initial.db", tmp_path / "old-index.db"
        if legacy:
            index.write_bytes(b"existing legacy evidence")
            with pytest.raises(ValueError, match="旧索引仍存在"):
                async with read_memory(missing, legacy_index=index, catalog=log.catalog(),
                    embeddings=runtime._embeddings, bindings=runtime._bindings, config=MemoryConfig(),
                    allow_initial=True):
                    pytest.fail("missing legacy graph was accepted")
        else:
            async with read_memory(missing, legacy_index=index, catalog=log.catalog(),
                embeddings=runtime._embeddings, bindings=runtime._bindings, config=MemoryConfig(),
                allow_initial=True) as (cycle, state):
                assert cycle.state_version == 0
                assert state.cutover_heads == tuple(sorted(log.catalog().snapshot_heads().items()))
        assert not missing.exists()
        assert not missing.with_suffix(".db.lock").exists()
