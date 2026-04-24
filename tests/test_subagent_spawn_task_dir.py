from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.background.subagent_profiles import build_spawn_spec
from prompts.background import build_spawn_subagent_prompt


@pytest.mark.asyncio
async def test_spawn_write_file_is_scoped_to_task_dir(tmp_path: Path):
    """Scripting profile write_file tool is restricted to task_dir."""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"
    task_dir.mkdir(parents=True, exist_ok=True)

    spec = build_spawn_spec(
        workspace=workspace,
        task_dir=task_dir,
        fetch_requester=MagicMock(),
        system_prompt="test",
        profile="scripting",
    )
    write_tool = next(t for t in spec.tools if t.name == "write_file")

    result = await write_tool.execute(path="final_report.md", content="done")

    assert "已写入" in result
    assert (task_dir / "final_report.md").read_text(encoding="utf-8") == "done"
    assert not (workspace / "final_report.md").exists()


def test_spawn_prompt_mentions_isolated_task_dir(tmp_path: Path):
    """Scripting profile system prompt includes task_dir path and isolation guidance."""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"

    prompt = build_spawn_subagent_prompt(workspace, task_dir, profile="scripting")

    assert str(task_dir.resolve()) in prompt
    assert "final_report.md" in prompt
    assert "只能写入当前任务目录" in prompt


def test_research_profile_has_no_write_tools(tmp_path: Path):
    """Research profile must not include write_file or edit_file."""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"
    task_dir.mkdir(parents=True, exist_ok=True)

    spec = build_spawn_spec(
        workspace=workspace,
        task_dir=task_dir,
        fetch_requester=MagicMock(),
        system_prompt="test",
        profile="research",
    )
    tool_names = {t.name for t in spec.tools}

    assert "write_file" not in tool_names
    assert "shell" not in tool_names
    assert "read_file" in tool_names
    assert "web_fetch" in tool_names


def test_scripting_profile_has_no_web_tools(tmp_path: Path):
    """Scripting profile must not include web_fetch or web_search."""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"
    task_dir.mkdir(parents=True, exist_ok=True)

    spec = build_spawn_spec(
        workspace=workspace,
        task_dir=task_dir,
        fetch_requester=MagicMock(),
        system_prompt="test",
        profile="scripting",
    )
    tool_names = {t.name for t in spec.tools}

    assert "web_fetch" not in tool_names
    assert "web_search" not in tool_names
    assert "shell" in tool_names
    assert "write_file" in tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["research", "scripting", "general"])
async def test_spawn_read_file_respects_non_multimodal_config(
    tmp_path: Path,
    profile: str,
):
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"
    task_dir.mkdir(parents=True, exist_ok=True)
    image = workspace / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    spec = build_spawn_spec(
        workspace=workspace,
        task_dir=task_dir,
        fetch_requester=MagicMock(),
        system_prompt="test",
        profile=profile,
        multimodal=False,
    )
    read_tool = next(t for t in spec.tools if t.name == "read_file")

    result = await read_tool.execute(path="a.png")

    assert isinstance(result, str)
    assert "当前主模型不支持多模态" in result
