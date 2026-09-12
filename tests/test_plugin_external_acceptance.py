from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.plugin_composition import ServiceKey
from docker.debug.plugin_external_acceptance import (
    _ensure_empty_directory,
    _invoke_capability,
    _load_distribution,
)


def test_external_acceptance_rejects_nonempty_runtime_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "old-state").write_text("must not be reused", encoding="utf-8")

    with pytest.raises(ValueError, match="必须为空"):
        _ensure_empty_directory(workspace, "workspace")


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
