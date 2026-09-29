"""O: background interest scoring cannot stall unrelated host work."""
import asyncio
from datetime import UTC, datetime

import pytest

from plugins.akasha.interest import SemanticInterest
from plugins.akasha.learning import LearningConfig
from tests.test_akasha_execution import WorkerGate, append_pair, memory_fixture


@pytest.mark.asyncio
async def test_interest_scoring_yields_and_drains_its_history_reader(tmp_path, monkeypatch):
    async with memory_fixture(tmp_path) as (_, _, learning, _, log, embeddings, _, _):
        append_pair(log, embeddings, learning, 0)
        gate = WorkerGate()
        original = learning.samples

        def samples(*args, **kwargs):
            gate.stop()
            return original(*args, **kwargs)

        monkeypatch.setattr(learning, "samples", samples)

        async def embed(texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def select():
            return LearningConfig(embedding_model="fixed", dimension=3, sources=("conversation",)), embed

        interest = SemanticInterest(learning, log.catalog(), embeddings, select)
        job = asyncio.create_task(interest.score(("one",), cutoff=datetime.now(UTC).isoformat()))
        try:
            await gate.wait(job)
            gate.release.set()
            assert await job == (0.999,)
        finally:
            gate.release.set()
            await asyncio.gather(job, return_exceptions=True)
