from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import load_config


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
