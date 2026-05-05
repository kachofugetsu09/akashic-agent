from __future__ import annotations

from pathlib import Path

from plugins.default_memory.config import (
    load_default_memory_config,
    resolve_memory_db_path,
)


def test_default_memory_config_reads_example_defaults() -> None:
    cfg = load_default_memory_config()

    assert cfg.retrieval.top_k_history == 8
    assert cfg.retrieval.thresholds.procedure == 0.66
    assert cfg.retrieval.inject.max_chars == 6000


def test_default_memory_config_local_overrides(tmp_path: Path) -> None:
    (tmp_path / "config.local.toml").write_text(
        """
db_path = "custom/memory.db"

[retrieval]
score_threshold = 0.7

[retrieval.thresholds]
event = 0.8

[retrieval.inject]
max_chars = 3000
""",
        encoding="utf-8",
    )

    cfg = load_default_memory_config(plugin_dir=tmp_path)

    assert cfg.db_path == "custom/memory.db"
    assert cfg.retrieval.top_k_history == 8
    assert cfg.retrieval.score_threshold == 0.7
    assert cfg.retrieval.thresholds.event == 0.8
    assert cfg.retrieval.inject.max_chars == 3000


def test_default_memory_db_path_resolves_under_workspace(tmp_path: Path) -> None:
    cfg = load_default_memory_config(plugin_dir=tmp_path)

    assert resolve_memory_db_path(workspace=tmp_path, default_config=cfg) == (
        tmp_path / "memory" / "memory2.db"
    )


def test_root_config_example_does_not_expose_default_memory_private_config() -> None:
    text = Path("config.example.toml").read_text(encoding="utf-8")

    assert "[memory.embedding]" in text
    assert "[memory.retrieval]" not in text
    assert "[memory.gate]" not in text
    assert "[memory.hyde]" not in text
    assert "[memory_v2]" not in text
