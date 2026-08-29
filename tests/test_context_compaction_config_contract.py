from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent.config import load_config
from agent.config_models import ContextCompactionConfig
from agent.model_runtime.context_compaction import hard_input_limit
from tests.model_plugin_fakes import BoundChatModelFake


def test_integrations_peer_agents_is_rejected_at_config_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[agent.context.compaction]
keep_recent_tokens = 20000

[integrations]
peer_agents = {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integrations.peer_agents"):
        load_config(path, workspace=tmp_path)


def _runtime_config(*, extra: str = "") -> str:
    return f"""
[agent.context]
{extra}
[agent.context.compaction]
keep_recent_tokens = 21000
"""


class _BudgetProvider:
    """Minimal concrete provider seam for hard input boundary tests."""

    context_window: int = 0

    def __init__(self, context_window: int) -> None:
        self.context_window = context_window

    async def chat(self, **kwargs: Any):
        raise AssertionError("budget fixture must not call provider.chat")

    @property
    def descriptor(self):
        return BoundChatModelFake(self).descriptor


def test_compaction_policy_is_loaded_once_at_agent_context_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_runtime_config(), encoding="utf-8")

    config = load_config(path, workspace=tmp_path)

    assert config.context_compaction.keep_recent_tokens == 21000
    assert not hasattr(config.context_compaction, "trigger_percent")


@pytest.mark.parametrize("raw", ["true", "false", "1.5", '"20000"'])
def test_config_rejects_non_integer_compaction_tail_budget(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _runtime_config().replace(
            "keep_recent_tokens = 21000", f"keep_recent_tokens = {raw}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keep_recent_tokens.*正整数"):
        load_config(path, workspace=tmp_path)


@pytest.mark.parametrize("raw", [True, False, 1.5, "20000", 0, -1])
def test_compaction_config_rejects_invalid_direct_values(raw: object) -> None:
    with pytest.raises(ValueError, match="keep_recent_tokens.*正整数"):
        replace(ContextCompactionConfig(), keep_recent_tokens=raw)


@pytest.mark.parametrize(
    "extra",
    [
        "memory_window = 12\n",
    ],
)
def test_legacy_context_keys_fail_at_config_boundary(
    tmp_path: Path,
    extra: str,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_runtime_config(extra=extra), encoding="utf-8")

    with pytest.raises(ValueError, match="removed configuration"):
        load_config(path, workspace=tmp_path)


def test_removed_agent_compaction_trigger_fails_at_config_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    text = _runtime_config().replace(
        "[agent.context.compaction]\n",
        "[agent.context.compaction]\ntrigger_percent = 0.7\n",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="agent.context.compaction.trigger_percent"):
        load_config(path, workspace=tmp_path)


def test_model_runtime_output_edge_is_directly_bounded_by_context_window() -> None:
    provider = _BudgetProvider(1025)
    assert hard_input_limit(BoundChatModelFake(provider), 1024) == 1
    with pytest.raises(ValueError, match="max_output_tokens"):
        hard_input_limit(BoundChatModelFake(_BudgetProvider(1024)), 1024)
