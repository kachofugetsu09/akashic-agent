import pytest
from contextlib import asynccontextmanager
from types import SimpleNamespace

from agent.plugin_composition.models import EmbeddingResult

from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.recall_tool import RecallTool
from plugins.tools.api import InvalidArguments
from session.message import Output
from tests.test_akasha_message_queries import memory_runtime


def target(tmp_path, runtime, *, embed=None, binding=None, model_id="fixture", max_chars=12000):
    call = runtime._embed_batch if embed is None else embed
    class Model:
        descriptor = SimpleNamespace(identity="fixed")
        async def embed(self, texts):
            return EmbeddingResult(tuple(tuple(vector) for vector in await call(texts)))
    class Embeddings:
        @asynccontextmanager
        async def bind(self, *, model_id=None):
            assert model_id == "fixture"
            yield Model()
    return RecallTool(memory=tmp_path / "memory.db", legacy_index=None, config=MemoryConfig(),
        catalog=runtime._catalog, embeddings=runtime._embeddings, bindings=runtime._bindings,
        select_learning=lambda: (runtime._learning_binding if binding is None else binding, model_id),
        records=runtime._records, embedding_api=Embeddings(), max_chars=max_chars)


@pytest.mark.asyncio
async def test_recall_tool_recovers_original_query_after_graph_advances_without_embedding(tmp_path, monkeypatch):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        write("u1", "original memory")
        write("a1", "original answer", Output)
        assert await runtime.consume() == 1
        recall = target(tmp_path, runtime)
        arguments = await recall.prepare({"query": "original memory"})
        result = await recall.invoke("request", arguments)
        observed = records.read("tool:request")
        assert observed.source.kind == "program"
        assert observed.source.query == "original memory"
        assert observed.presented_message_ids == ("u1", "a1")
        assert result.parts[-1].kind == "akasha.recall"
        write("u2", "later memory")
        write("a2", "later answer", Output)
        assert await runtime.consume() == 1
        assert consumer.cycle.state_version == 2
        async def fail_embed(texts):
            pytest.fail("completed query was embedded again")
        recall = target(tmp_path, runtime, embed=fail_embed)
        def fail_selection():
            pytest.fail("query recovery selected a new model")
        recall._select_learning = fail_selection
        assert await recall.query("request") == result
        assert await recall.invoke("request", arguments) == result
        assert records.read("tool:request") == observed
        assert await recall.query("unknown") is None


@pytest.mark.asyncio
async def test_recall_rejects_invalid_user_arguments_before_query(tmp_path):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        recall = target(tmp_path, runtime)
        for arguments in ({"query": "  "}, {"query": "q", "limit": 0}, {"query": "q", "source": {}}):
            with pytest.raises(InvalidArguments):
                await recall.prepare(arguments)
        assert calls == []


@pytest.mark.asyncio
async def test_prepared_recall_keeps_learning_model_and_budget_when_defaults_change(tmp_path):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        write("u", "old memory")
        write("a", "old answer", Output)
        assert await runtime.consume() == 1
        original = target(tmp_path, runtime, max_chars=100)
        prepared = await original.prepare({"query": "recall"})
        changed = target(tmp_path, runtime, binding="unavailable-new-rule", model_id="new-default", max_chars=1)
        result = await changed.invoke("prepared", prepared)
        record = records.read("tool:prepared")
        assert record.learning_binding == runtime._learning_binding
        assert record.max_chars == 100
        assert record.presented_message_ids == ("u",)
        assert await changed.query("prepared") == result


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [[], [[1.0]], [[float("nan"), 0.0]], [[0.0, 0.0]]])
async def test_bad_embedding_never_publishes_query_or_changes_learning(tmp_path, values):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        async def invalid(texts):
            return values
        recall = target(tmp_path, runtime, embed=invalid)
        prepared = await recall.prepare({"query": "test"})
        with pytest.raises(ValueError, match="embedding"):
            await recall.invoke("invalid", prepared)
        assert records.read("tool:invalid") is None
        assert consumer.state.applied == ()
