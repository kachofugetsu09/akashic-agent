from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_PROBE_PATH = (
    Path(__file__).resolve().parents[1] / "docker/debug/plugin_hot_reload_probe.py"
)
_SPEC = importlib.util.spec_from_file_location("plugin_hot_reload_probe", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def test_integrity_gate_fails_when_any_check_fails() -> None:
    checks = [
        probe.CheckResult("read_only", True, {}),
        probe.CheckResult("repositories_unchanged", False, {}),
    ]

    assert probe._gate_status(checks) == "failed"


def test_controller_rejects_protected_sandbox() -> None:
    root = Path("/workspace").resolve()

    assert probe._sandbox_is_protected(root / "gate", [root])
    assert not probe._sandbox_is_protected(Path("/tmp/gate"), [root])


def test_system_gate_propagates_subgate_failure() -> None:
    baseline = {
        "build_returncode": 0,
        "integrity_returncode": 0,
        "smoke_passed": True,
        "cleanup_returncode": 0,
        "unchanged": True,
        "controller_error": "",
    }
    assert probe._controller_gate_passed(**baseline)

    for key, value in (
        ("build_returncode", 1),
        ("integrity_returncode", 1),
        ("smoke_passed", False),
        ("cleanup_returncode", 1),
        ("unchanged", False),
        ("controller_error", "boom"),
    ):
        failed = {**baseline, key: value}
        assert not probe._controller_gate_passed(**failed)


def test_mounted_tree_digest_does_not_require_worktree_common_git_dir(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /unmounted/common/worktrees/gate\n")
    source = worktree / "source.py"
    source.write_text("REVISION = 1\n")

    before = probe._mounted_tree_digest(worktree)
    source.write_text("REVISION = 2\n")
    after = probe._mounted_tree_digest(worktree)

    assert before != after


def test_smoke_config_uses_app_server_control_endpoint(tmp_path: Path) -> None:
    probe._write_smoke_config(tmp_path)

    config = (tmp_path / "config.toml").read_text()

    assert "[app_server]" in config
    assert 'listen = "/sandbox/akashic.sock"' in config
    assert "[channels]" not in config


def test_migrated_plugin_gate_uses_mobile_catalog_not_dashboard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observe_source = tmp_path / "observe.py"
    observe_source.write_text("# observe\n")
    observe_db = tmp_path / "workspace/observe/observe.db"
    observe_db.parent.mkdir(parents=True)
    observe_db.touch()
    expected_plugins = [
        "emotion@gate",
        "observe@gate",
        "proactive_feedback@gate",
        "status_commands@gate",
    ]
    monkeypatch.setattr(
        probe,
        "_control_roundtrip",
        lambda *_: {"content": "🧠 记忆整理状态：已同步"},
    )
    monkeypatch.setattr(probe, "_wait_sqlite_count", lambda *_: 1)
    monkeypatch.setattr(
        probe,
        "_read_json_object",
        lambda *_: {
            "phase_slots": [
                "meme.prompt",
                "status_commands.memory_status",
                "gate_driver.inspect_runtime_modules",
            ],
            "mobile_ui_catalog": {
                "catalog_revision": "revision",
                "items": [{"id": plugin_id} for plugin_id in expected_plugins],
            },
        },
    )
    monkeypatch.setattr(probe, "_snapshot_statuses", lambda *_: [])
    monkeypatch.setattr(
        probe,
        "_wait_snapshot_status",
        lambda *_, **__: (
            [],
            {"old_generation": "observe:old", "new_generation": "observe:new"},
        ),
    )
    monkeypatch.setattr(
        probe,
        "_dashboard_plugins",
        lambda *_: (_ for _ in ()).throw(AssertionError("不得使用桌面端目录")),
        raising=False,
    )

    result = probe._exercise_migrated_plugins("container", observe_source, tmp_path)

    assert result["passed"] is True
    assert result["mobile_ui_plugins"] == expected_plugins
