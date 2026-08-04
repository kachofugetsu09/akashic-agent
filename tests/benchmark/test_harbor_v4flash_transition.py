import json
from pathlib import Path

from benchmark.harbor_v4flash import transition
from benchmark.harbor_v4flash.transition import (
    _controller_environment,
    _signal_old_controller,
    _terminal_task_names,
)


def test_terminal_task_names_only_accepts_campaign_terminal_events(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "events.jsonl"
    events = [
        {"event": "attempt_started", "task": "/tasks/one"},
        {"event": "accepted", "task": "/tasks/two"},
        {"event": "attempt_failed", "task": "/tasks/three"},
    ]
    ledger.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    assert _terminal_task_names(ledger) == {"two", "three"}


def test_signal_old_controller_is_idempotent_when_unit_is_inactive(
    monkeypatch,
) -> None:
    monkeypatch.setattr(transition, "_unit_active", lambda unit: False)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("inactive unit must not receive another signal")

    monkeypatch.setattr(transition.subprocess, "run", unexpected_run)

    _signal_old_controller("old.service")


def test_signal_old_controller_targets_only_main_owner(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(transition, "_unit_active", lambda unit: True)
    monkeypatch.setattr(
        transition.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    _signal_old_controller("old.service")

    assert calls == [
        (
            [
                "systemctl",
                "--user",
                "kill",
                "--kill-who=main",
                "--signal=SIGINT",
                "old.service",
            ],
            {"check": True},
        )
    ]


def test_cleanup_old_projects_uses_full_network_id(monkeypatch, tmp_path) -> None:
    project = "akasic-bench-old__env"
    network_id = "a" * 64
    commands = []
    cleaned = []
    monkeypatch.setattr(
        transition,
        "_old_source_projects",
        lambda runs_dir, source_digest: [project],
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return transition.subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{network_id}\t{project}_default\n",
        )

    monkeypatch.setattr(transition.subprocess, "run", fake_run)
    monkeypatch.setattr(
        transition,
        "stop_and_cleanup_compose_project",
        lambda project_name, *, network: cleaned.append((project_name, network)),
    )

    transition._cleanup_old_projects(tmp_path, "sha256:old")

    assert "--no-trunc" in commands[0]
    assert cleaned == [
        (
            project,
            {"id": network_id, "name": f"{project}_default"},
        )
    ]


def test_controller_environment_injects_fixed_source_and_sdk(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/python")

    environment = _controller_environment(tmp_path)

    assert environment["PYTHONPATH"].split(transition.os.pathsep) == [
        str(tmp_path),
        str(tmp_path / "sdk" / "python" / "src"),
        "/existing/python",
    ]
