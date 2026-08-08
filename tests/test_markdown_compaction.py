from __future__ import annotations

import json
from typing import Any

import pytest

from agent.model_runtime.types import LLMResponse
from agent.provider import LLMProvider
from bus.event_bus import EventBus
from core.memory.events import ConsolidationCommitted
from core.memory.markdown import MarkdownMemoryMaintenance, MarkdownMemoryStore


class _Provider(LLMProvider):
    context_window: int = 4096

    def __init__(self) -> None:
        self.context_window = 4096
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []
        self.max_output_tokens = 0
        self.estimated_tokens: int | None = None

    def estimate_context_tokens(
        self, messages: list[dict], tools: list[dict]
    ) -> int:
        if self.estimated_tokens is not None:
            return self.estimated_tokens
        prompt = str(messages[0]["content"])
        return 1 + prompt.count("UNIT")

    async def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        prompt = str(messages[0]["content"])
        self.prompts.append(prompt)
        self.max_tokens.append(int(kwargs["max_tokens"]))
        return LLMResponse(
            content=json.dumps(
                {
                    "history_entries": [
                        {
                            "summary": f"[2026-08-08 12:00] {prompt.count('UNIT')} units",
                            "emotional_weight": 0,
                        }
                    ],
                    "pending_items": [],
                }
            )
        )


def _source_plan() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for unit_index in range(3):
        unit_ref = f"unit-{unit_index}"
        for message_index, role in enumerate(("user", "assistant")):
            message_id = f"message-{unit_index}-{message_index}"
            rows.append(
                {
                    "id": message_id,
                    "seq": unit_index * 2 + message_index,
                    "unit_ref": unit_ref,
                    "message": {
                        "id": message_id,
                        "seq": unit_index * 2 + message_index,
                        "role": role,
                        "content": f"UNIT {unit_index}",
                        "timestamp": "2026-08-08T12:00:00+00:00",
                    },
                }
            )
    return tuple(rows)


@pytest.mark.asyncio
async def test_exact_markdown_plan_pages_by_consecutive_unit_ref(tmp_path):
    provider = _Provider()
    provider.max_output_tokens = 256
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
        provider_input_budget=4,
    )

    draft = await maintenance.prepare_compaction_markdown(
        _source_plan(),
        source_ref="session:checkpoint:1",
    )

    assert len(provider.prompts) == 3
    assert all(prompt.count("UNIT") == 2 for prompt in provider.prompts)
    assert provider.max_tokens == [256, 256, 256]
    assert draft.source_ref == "session:checkpoint:1"
    assert len(draft.history_entry_payloads) == 3


@pytest.mark.asyncio
async def test_nonconsecutive_unit_ref_is_rejected(tmp_path):
    provider = _Provider()
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
        provider_input_budget=4,
    )
    plan = list(_source_plan())
    plan[-1] = {**plan[-1], "unit_ref": "unit-0"}

    with pytest.raises(ValueError, match="unit_ref 非连续"):
        await maintenance.prepare_compaction_markdown(
            tuple(plan),
            source_ref="session:checkpoint:invalid",
        )


@pytest.mark.asyncio
async def test_single_unit_at_budget_fails_without_provider_call(tmp_path):
    provider = _Provider()
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
        provider_input_budget=3,
    )

    with pytest.raises(RuntimeError, match="input_budget"):
        await maintenance.prepare_compaction_markdown(
            _source_plan()[:2],
            source_ref="session:checkpoint:oversized",
        )

    assert provider.prompts == []


def test_default_markdown_provider_budget_is_strict(tmp_path):
    provider = _Provider()
    provider.context_window = 2048
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
    )
    assert maintenance._provider_input_budget is None

    provider.context_window = 1024
    assert MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("context_window", [0, 1024])
async def test_unknown_or_too_small_window_fails_only_at_prepare(
    tmp_path,
    context_window,
):
    provider = _Provider()
    provider.context_window = context_window
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
    )

    with pytest.raises(RuntimeError, match="input_budget"):
        await maintenance.prepare_compaction_markdown(
            _source_plan()[:2],
            source_ref="session:checkpoint:unknown-window",
        )
    assert provider.prompts == []


@pytest.mark.asyncio
async def test_default_input_and_output_budgets_follow_current_provider(tmp_path):
    provider = _Provider()
    provider.context_window = 4096
    provider.max_output_tokens = 512
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
    )

    estimated = 3000
    provider.estimated_tokens = estimated
    await maintenance.prepare_compaction_markdown(
        _source_plan()[:2],
        source_ref="session:checkpoint:dynamic-1",
    )
    assert provider.max_tokens == [512]

    provider.context_window = 2000
    provider.max_output_tokens = 256
    with pytest.raises(RuntimeError, match="input_budget"):
        await maintenance.prepare_compaction_markdown(
            _source_plan()[:2],
            source_ref="session:checkpoint:dynamic-2",
        )
    assert provider.max_tokens == [512]


@pytest.mark.asyncio
async def test_markdown_commit_emits_one_combined_checkpoint_event(tmp_path):
    provider = _Provider()
    event_bus = EventBus()
    events: list[ConsolidationCommitted] = []
    event_bus.on(ConsolidationCommitted, events.append)
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=provider,
        model="memory-model",
        provider_input_budget=4,
        event_bus=event_bus,
    )
    draft = await maintenance.prepare_compaction_markdown(
        _source_plan(),
        source_ref="session:checkpoint:2",
    )

    await maintenance.commit_compaction_markdown(draft)
    await event_bus.drain()

    assert len(events) == 1
    assert events[0].source_ref == "session:checkpoint:2"
    assert len(events[0].history_entry_payloads) == 3
    await event_bus.aclose()
