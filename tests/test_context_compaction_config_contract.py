from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import load_config
from agent.config_models import ModelRuntimeConfig
from agent.model_runtime.context_compaction import hard_input_limit
from bootstrap.setup_wizard import WizardAnswers, _render_config


def test_integrations_peer_agents_is_rejected_at_config_boundary(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
provider = "openai"
model = "test-model"
api_key = "test-key"

[agent.context.compaction]
trigger_percent = 0.74
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
[llm]
main = "main"

[llm.runtimes.main]
provider = "openai"
model = "test-model"
api_key = "test-key"
base_url = "https://api.openai.com/v1"
context_window = 100000
max_output_tokens = 2000
input_modalities = ["text"]
{extra}
[agent.context.compaction]
trigger_percent = 0.71
keep_recent_tokens = 21000
"""


def test_compaction_policy_is_loaded_once_at_agent_context_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_runtime_config(), encoding="utf-8")

    config = load_config(path, workspace=tmp_path)

    assert config.context_compaction.trigger_percent == 0.71
    assert config.context_compaction.keep_recent_tokens == 21000
    assert not hasattr(config.model_runtimes["main"], "effective_context_percent")


@pytest.mark.parametrize(
    "extra",
    [
        "memory_window = 12\n",
        "effective_context_percent = 0.9\n",
        "compaction_trigger_percent = 0.7\n",
    ],
)
def test_legacy_context_keys_fail_at_config_boundary(
    tmp_path: Path,
    extra: str,
) -> None:
    path = tmp_path / "config.toml"
    text = _runtime_config(extra=extra)
    if extra.startswith("memory_window"):
        text = text.replace(
            "[llm.runtimes.main]\n",
            "[agent.context]\nmemory_window = 12\n\n[llm.runtimes.main]\n",
        )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="removed configuration"):
        load_config(path, workspace=tmp_path)


def test_model_runtime_output_edge_is_directly_bounded_by_context_window() -> None:
    provider = SimpleNamespace(context_window=1025)
    assert hard_input_limit(provider, 1024) == 1
    with pytest.raises(ValueError, match="max_output_tokens"):
        hard_input_limit(SimpleNamespace(context_window=1024), 1024)
    with pytest.raises(ValueError, match="必须小于 context_window"):
        ModelRuntimeConfig(
            runtime_id="edge",
            provider="openai",
            model="model",
            context_window=1024,
            max_output_tokens=1024,
        )


def test_setup_wizard_renders_only_new_compaction_structure() -> None:
    rendered = _render_config(
        WizardAnswers(
            provider="openai",
            model="model",
            base_url="https://api.openai.com/v1",
            context_window=64_000,
        )
    )

    assert "[agent.context.compaction]" in rendered
    assert "memory_window" not in rendered
    assert "effective_context_percent" not in rendered
