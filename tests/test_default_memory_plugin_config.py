from __future__ import annotations

from pathlib import Path

import pytest

from plugins.default_memory.config import (
    DefaultMemoryConfig,
    ensure_default_memory_config_file,
    load_default_memory_config,
    resolve_memory_db_path,
)


def test_default_memory_config_reads_example_defaults(tmp_path: Path) -> None:
    cfg = load_default_memory_config(workspace=tmp_path)

    assert cfg.retrieval.top_k_history == 8
    assert cfg.retrieval.thresholds.procedure == 0.66
    assert cfg.retrieval.inject.max_chars == 6000


def test_default_memory_config_creates_workspace_data_dir(
    tmp_path: Path,
) -> None:
    target = tmp_path / "plugin-data" / "default_memory-builtin"

    path = ensure_default_memory_config_file(workspace=tmp_path)

    assert path == target / "config.local.toml"
    assert load_default_memory_config(workspace=tmp_path).retrieval.top_k_history == 8


def test_default_memory_config_rejects_symlink_data_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "plugin-data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="符号链接"):
        ensure_default_memory_config_file(workspace=tmp_path)

    assert list(outside.iterdir()) == []


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


def test_default_memory_config_preserves_legacy_numeric_strings(tmp_path: Path) -> None:
    (tmp_path / "config.local.toml").write_text(
        """
[retrieval]
top_k_history = "9"
score_threshold = "0.6"
procedure_guard_enabled = false

[retrieval.inject]
max_chars = 3000.0
""",
        encoding="utf-8",
    )

    cfg = load_default_memory_config(plugin_dir=tmp_path)

    assert cfg.retrieval.top_k_history == 9
    assert cfg.retrieval.score_threshold == 0.6
    assert cfg.retrieval.procedure_guard_enabled is False
    assert cfg.retrieval.inject.max_chars == 3000
    assert cfg.retrieval.thresholds == DefaultMemoryConfig().retrieval.thresholds


@pytest.mark.parametrize(
    ("content", "field"),
    [
        ("db_path = 42\n", "db_path"),
        ('retrieval = "invalid"\n', "retrieval"),
        ('[retrieval]\nthresholds = "invalid"\n', "retrieval.thresholds"),
        ("[retrieval]\ninject = []\n", "retrieval.inject"),
        (
            '[retrieval]\nprocedure_guard_enabled = "false"\n',
            "retrieval.procedure_guard_enabled",
        ),
        ("[retrieval]\ntop_k_history = 1.5\n", "retrieval.top_k_history"),
        ("[retrieval]\nscore_threshold = true\n", "retrieval.score_threshold"),
        ("[retrieval.thresholds]\nevent = []\n", "retrieval.thresholds.event"),
        ("[retrieval.inject]\nmax_chars = true\n", "retrieval.inject.max_chars"),
    ],
)
def test_default_memory_config_rejects_invalid_present_values(
    tmp_path: Path,
    content: str,
    field: str,
) -> None:
    (tmp_path / "config.local.toml").write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        load_default_memory_config(plugin_dir=tmp_path)


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
    assert "output_dimensionality" not in text
    assert "[memory_v2]" not in text
