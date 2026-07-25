from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent.config import load_config
from agent.config_models import TaskExecutionConfig


def _base_config(extra: str = "") -> str:
    return f"""\
provider = "deepseek"
model = "deepseek-chat"
api_key = "sk-test"
system_prompt = "test"

{extra}
"""


def test_task_execution_config_rejects_unsafe_auto_risk() -> None:
    with pytest.raises(ValueError, match="read-only"):
        TaskExecutionConfig(auto_allowed_risks=["read-only", "write"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_work_tool_calls": 0}, "max_work_tool_calls must be positive"),
        (
            {"max_tool_search_calls": 2},
            "max_tool_search_calls must be exactly 1",
        ),
        ({"lease_seconds": 29}, "lease_seconds must be at least 30"),
    ],
)
def test_task_execution_config_rejects_unsafe_limits(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TaskExecutionConfig(**kwargs)


def test_task_execution_defaults_are_disabled(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_base_config(), encoding="utf-8")

    cfg = load_config(path)

    assert cfg.task_execution == TaskExecutionConfig()


def test_task_execution_loads_toml_values(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _base_config("""
[task_execution]
enabled = true
auto_allowed_risks = ["read-only"]
max_work_tool_calls = 2
max_tool_search_calls = 1
lease_seconds = 60
"""),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.task_execution == TaskExecutionConfig(
        enabled=True,
        auto_allowed_risks=["read-only"],
        max_work_tool_calls=2,
        max_tool_search_calls=1,
        lease_seconds=60,
    )
