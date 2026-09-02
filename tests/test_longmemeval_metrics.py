from __future__ import annotations

import pytest

from agent.plugin_composition import ModelRequest
from eval.longmemeval.metrics import judge_answer


class _BrokenJudge:
    async def complete(self, request: ModelRequest):
        del request
        raise AssertionError("broken judge contract")


@pytest.mark.asyncio
async def test_judge_internal_failure_is_not_scored_as_wrong() -> None:
    with pytest.raises(AssertionError, match="broken judge contract"):
        await judge_answer(
            _BrokenJudge(),  # type: ignore[arg-type]
            question="question",
            gold="gold",
            predicted="predicted",
        )
