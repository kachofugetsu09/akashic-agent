from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import load_config


def test_legacy_top_level_peer_config_is_explicitly_unsupported(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("peer_agents = []\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported capability: peer_agents"):
        load_config(config_path, workspace=tmp_path / "workspace")


def test_legacy_integrations_peer_config_is_explicitly_unsupported(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[integrations]\npeer_agents = []\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported capability: integrations\.peer_agents",
    ):
        load_config(config_path, workspace=tmp_path / "workspace")
