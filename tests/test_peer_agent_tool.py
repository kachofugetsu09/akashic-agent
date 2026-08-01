from __future__ import annotations

import importlib
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


def test_removed_peer_modules_have_no_importable_tombstone() -> None:
    for module_name in ("agent.peer_agent", "bootstrap.toolsets.peer"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_removed_peer_paths_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "agent" / "peer_agent").exists()
    assert not (root / "bootstrap" / "toolsets" / "peer.py").exists()
