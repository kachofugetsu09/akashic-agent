from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import docker.debug.plugin_external_acceptance as external_acceptance
from agent.plugin_composition import ServiceKey
from docker.debug.plugin_external_acceptance import (
    _exercise_core_bootstrap,
    _ensure_empty_directory,
    _invoke_capability,
    _load_distribution,
    _write_bootstrap_config,
)


def test_external_acceptance_rejects_nonempty_runtime_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "old-state").write_text("must not be reused", encoding="utf-8")

    with pytest.raises(ValueError, match="必须为空"):
        _ensure_empty_directory(workspace, "workspace")


def test_bootstrap_workspace_seeds_legal_context_material_grants(tmp_path: Path) -> None:
    _write_bootstrap_config(tmp_path, marketplace="acceptance")

    config = (tmp_path / "plugin-data/context-acceptance/config.local.toml").read_text(
        encoding="utf-8"
    )
    assert 'default_prompt = "prompt@acceptance"' in config
    assert 'markdown_memory = "markdown_memory@acceptance"' in config
    assert 'skills = "standard_tools@acceptance"' in config
    assert 'summary_source = ["compaction", "compaction@acceptance"]' in config


def test_capability_enumeration_does_not_count_as_call() -> None:
    key = ServiceKey("message.display:model.facts")

    class Root:
        def provided_services(self, *, plugin_ids):
            _ = plugin_ids
            return {key: object()}

    spec = {
        "service": key.name,
        "entrypoint": "plugins.models.projection.display_facts",
        "input": {
            "kind": "model.facts",
            "value": {
                "call_record_id": "acceptance-call",
                "tool_ids": {},
                "thinking": None,
                "continuation": None,
            },
        },
        "expect": {"type": "dict", "value": {}},
    }

    with pytest.raises(TypeError, match="不可调用"):
        asyncio.run(_invoke_capability(root=Root(), plugin_id="models@test", spec=spec))


def test_capability_oracle_calls_declared_input_and_checks_output() -> None:
    key = ServiceKey("message.display:model.facts")
    seen = []

    def display(part):
        seen.append(part)
        return {"call_record_id": part.value["call_record_id"], "thinking": part.value["thinking"]}

    class Root:
        def provided_services(self, *, plugin_ids):
            _ = plugin_ids
            return {key: display}

    spec = {
        "service": key.name,
        "entrypoint": "plugins.models.projection.display_facts",
        "input": {
            "kind": "model.facts",
            "value": {
                "call_record_id": "acceptance-call",
                "tool_ids": {},
                "thinking": None,
                "continuation": None,
            },
        },
        "expect": {
            "type": "dict",
            "value": {"call_record_id": "acceptance-call", "thinking": None},
        },
    }

    evidence = asyncio.run(
        _invoke_capability(root=Root(), plugin_id="models@test", spec=spec)
    )
    assert evidence["call_executed"] is True
    assert evidence["status"] == "passed"
    assert len(seen) == 1
    assert seen[0].kind == "model.facts"


def test_distribution_report_requires_external_bundle_files(tmp_path: Path) -> None:
    path = tmp_path / "distribution.json"
    path.write_text(
        json.dumps(
            {
                "source_commit": "0" * 40,
                "core": {"file": "core.tar"},
                "plugins": [{"name": "example", "file": "example.bundle"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bundle 缺失"):
        _load_distribution(path)


def test_core_probe_records_real_start_and_stop_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Manager:
        current_snapshot = object()

    class Core:
        plugin_manager = Manager()

    class Runtime:
        _started = True
        _shutdown = False
        core = Core()
        channel_host = object()
        app_server = object()
        dashboard_task = None
        chat_task = None
        mobile_gateway_task = None
        plugin_watcher_task = None

        async def shutdown(self) -> None:
            calls.append("stop")
            self._shutdown = True
            self.core.plugin_manager.current_snapshot = None

    async def start(**kwargs):
        calls.append("start")
        return (
            Runtime(),
            {
                "checks": {
                    "bootstrap_start_returned": True,
                    "runtime_started": True,
                    "core_runtime_created": True,
                    "stable_snapshot_published": True,
                    "channel_host_started": True,
                    "app_server_started": True,
                    "checkout_invisible": True,
                    "core_modules_from_artifact": True,
                },
                "status": "passed",
            },
            {"AKASHIC_PLUGIN_HOME": None, "AKASHIC_WORKSPACE": None},
        )

    monkeypatch.setattr(external_acceptance, "_validate_core_root", lambda root, repo: root)
    monkeypatch.setattr(external_acceptance, "_prepare_runtime", lambda **kwargs: {})
    monkeypatch.setattr(external_acceptance, "_start_app_runtime", start)

    result = asyncio.run(
        _exercise_core_bootstrap(
            repo_root=tmp_path / "repo",
            core_root=tmp_path / "core",
            workspace=tmp_path / "workspace",
            plugins_home=tmp_path / "plugins-home",
        )
    )

    assert result["status"] == "passed"
    assert calls == ["start", "stop"]
    assert result["bootstrap"]["checks"]["stable_snapshot_drained"] is True
