from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType

import pytest

from agent.lifecycle.phases.after_reasoning import default_after_reasoning_modules
from agent.lifecycle.phases.after_step import default_after_step_modules
from agent.lifecycle.phases.after_turn import default_after_turn_modules
from agent.lifecycle.phases.before_reasoning import default_before_reasoning_modules
from agent.lifecycle.phases.before_step import default_before_step_modules
from agent.lifecycle.phases.before_turn import default_before_turn_modules
from agent.lifecycle.phases.prompt_render import default_prompt_render_modules
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manifest import load_plugin_manifest


@pytest.mark.parametrize(
    "factory",
    (
        default_before_turn_modules,
        default_before_reasoning_modules,
        default_prompt_render_modules,
        default_before_step_modules,
        default_after_step_modules,
        default_after_reasoning_modules,
        default_after_turn_modules,
    ),
)
def test_core_phase_factories_have_no_plugin_module_injection(factory) -> None:
    assert "plugin_modules" not in inspect.signature(factory).parameters


def test_plugin_loader_rejects_v2_module() -> None:
    module = ModuleType("removed_api")
    module.api_version = 2
    module.name = "removed-api"
    module.version = "1.0.0"
    module.apply = lambda ctx, config: None

    with pytest.raises(ValueError, match="api_version = 3"):
        ComposablePlugin.from_module(module)


def test_plugin_manifest_rejects_removed_package_shell(tmp_path: Path) -> None:
    (tmp_path / "manifest.toml").write_text(
        '[plugins]\n\n[packages."legacy"]\nenabled = true\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不再支持 \\[packages\\]"):
        load_plugin_manifest(tmp_path)
