from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

GATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docker"
    / "debug"
    / "content_source_interop_gate.py"
)
SPEC = importlib.util.spec_from_file_location("content_source_interop_gate", GATE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _plugin_repo(tmp_path: Path, plugin_id: str = "fixture") -> Path:
    root = tmp_path / plugin_id
    root.mkdir()
    _ = _git(root, "init", "--quiet")
    _ = _git(root, "config", "user.email", "fixture@example.com")
    _ = _git(root, "config", "user.name", "Fixture")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f'name = "{plugin_id}"\n'
        'version = "3.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )
    (root / "plugin.py").write_text(
        "api_version = 3\n"
        f'name = "{plugin_id}"\n'
        "async def apply(ctx, config):\n"
        "    del ctx, config\n",
        encoding="utf-8",
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_plugin.py").write_text(
        "def test_fixture():\n    assert True\n",
        encoding="utf-8",
    )
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "--quiet", "-m", "test: fixture")
    return root


def test_lock_pins_real_revisions_and_resolves_feedback_interop() -> None:
    contract = gate._load_contract(gate.DEFAULT_LOCK)

    assert contract.core_contract == "9da3a988a2bf62b0f550bd4f6bb98c4eeb1f56f5"
    assert tuple(plugin.id for plugin in contract.plugins) == (
        "calendar",
        "fitbit",
        "feed",
        "steam",
        "github-watch",
        "proactive_feedback",
        "emotion",
        "observe",
    )
    assert all(len(plugin.resolved_sha) == 40 for plugin in contract.plugins)
    assert contract.pending == ()
    feedback = next(
        plugin for plugin in contract.plugins if plugin.id == "proactive_feedback"
    )
    emotion = next(plugin for plugin in contract.plugins if plugin.id == "emotion")
    assert feedback.resolved_sha == "531eae4e4ac4714aad5417b8257a724007728345"
    assert emotion.resolved_sha == "2bb332b7f51526763b444871510cb4cba866a45c"
    assert feedback.atoms == (
        "SESSION_READ",
        "UI_SLOTS",
        "proactive-feedback.history.v1",
    )
    assert "proactive-feedback.history.v1" in emotion.atoms
    assert contract.cross_repo[0].plugin_ids == ("proactive_feedback", "emotion")
    github_watch = next(
        plugin for plugin in contract.plugins if plugin.id == "github-watch"
    )
    assert github_watch.pull_request is None
    assert "content.source.v1" not in github_watch.atoms
    assert contract.retired[0]["disposition"] == "delete_zero_runtime_consumers"


def test_lock_rejects_schema_drift_and_short_revision(tmp_path: Path) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["extra"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(gate.GateError, match="根结构"):
        gate._load_contract(invalid)

    del raw["extra"]
    raw["plugins"][0]["resolved_sha"] = "abc"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(gate.GateError, match="完整 SHA"):
        gate._load_contract(invalid)

    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["cross_repo"][0]["plugin_ids"] = ["missing-plugin"]
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(gate.GateError, match="cross_repo 引用未知插件"):
        gate._load_contract(invalid)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("pending", "reason", 7, "pending 字段"),
        ("retired", "canonical_sha", "abc", "完整 SHA"),
        ("retired", "evidence", "not-a-list", "字符串数组"),
        ("retired", "evidence", ["ok", 7], "字符串数组"),
    ),
)
def test_lock_rejects_malformed_pending_and_retired_fields(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    if section == "pending":
        raw["pending"] = [{"id": "fixture", "reason": "fixture"}]
    raw[section][0][field] = value
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(gate.GateError, match=message):
        gate._load_contract(invalid)


def test_exact_plugin_verification_rejects_dirty_and_old_proactive_seams(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)
    template = gate._load_contract(gate.DEFAULT_LOCK).plugins[0]
    contract = replace(
        template,
        id="fixture",
        resolved_sha=_git(root, "rev-parse", "HEAD"),
        test_cwd="tests",
        cases=("tests/test_plugin.py",),
    )

    receipt = gate._verify_plugin(contract, root)
    assert receipt["status"] == "verified"

    (root / "plugin.py").write_text(
        (root / "plugin.py").read_text(encoding="utf-8")
        + "\nPROACTIVE_COMPONENTS = ()\n",
        encoding="utf-8",
    )
    with pytest.raises(gate.GateError, match="非 clean"):
        gate._verify_plugin(contract, root)

    _ = _git(root, "add", "plugin.py")
    _ = _git(root, "commit", "--quiet", "-m", "test: mutant")
    mutant = replace(contract, resolved_sha=_git(root, "rev-parse", "HEAD"))
    with pytest.raises(gate.GateError, match="proactive-only seam"):
        gate._verify_plugin(mutant, root)


def test_path_map_requires_exact_absolute_id_bindings(tmp_path: Path) -> None:
    assert gate._parse_path_map([f"fixture={tmp_path}"], "--plugin-root") == {
        "fixture": tmp_path
    }
    with pytest.raises(gate.GateError, match="id=/absolute/path"):
        gate._parse_path_map(["fixture"], "--plugin-root")
    with pytest.raises(gate.GateError, match="绝对路径"):
        gate._parse_path_map(["fixture=relative"], "--plugin-root")


def test_runner_replays_owner_fixture_without_copying_plugin_logic(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)

    receipt = gate._run_cases(
        Path(sys.executable),
        root / "tests",
        ("tests/test_plugin.py",),
        root,
    )

    assert receipt["returncode"] == 0
    assert "1 passed" in receipt["stdout_tail"]
    assert receipt["source_before"] == receipt["source_after"]
    assert receipt["core_before"] == receipt["core_after"]
    assert receipt["pytestInterpreter"]["realpath"] == str(
        Path(sys.executable).resolve()
    )
    assert receipt["pluginFixtureInterpreter"] is None


def test_runner_exports_distinct_artifact_python_without_using_it_for_pytest(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)
    (root / "tests" / "test_plugin.py").write_text(
        "import os\n"
        "def test_fixture_python():\n"
        "    assert os.environ['AKASHIC_PLUGIN_FIXTURE_PYTHON'] == '/usr/bin/python3'\n",
        encoding="utf-8",
    )
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "--quiet", "-m", "test: require service runtime")

    receipt = gate._run_cases(
        Path("/usr/bin/python3"),
        root / "tests",
        ("tests/test_plugin.py",),
        root,
    )

    assert receipt["returncode"] == 0
    assert receipt["pytestInterpreter"]["realpath"] == str(
        Path(sys.executable).resolve()
    )
    assert receipt["pluginFixtureInterpreter"]["requested"] == "/usr/bin/python3"


def test_owner_fixture_that_requires_service_python_fails_without_artifact_runtime(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)
    (root / "tests" / "test_plugin.py").write_text(
        "import os\n"
        "def test_fixture_python():\n"
        "    assert 'AKASHIC_PLUGIN_FIXTURE_PYTHON' in os.environ\n",
        encoding="utf-8",
    )
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "--quiet", "-m", "test: require service runtime")

    with pytest.raises(gate.GateError, match="fixture 失败"):
        gate._run_cases(
            Path(sys.executable),
            root / "tests",
            ("tests/test_plugin.py",),
            root,
        )


def test_python_probe_rejects_successful_non_python_executable() -> None:
    executable = Path("/bin/true")
    if not executable.exists():
        pytest.skip("fixture requires /bin/true")

    with pytest.raises(gate.GateError, match="不是 Python"):
        gate._python_receipt(executable)


def test_runner_rejects_passing_fixture_that_changes_tracked_source(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)
    tracked = root / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    (root / "tests" / "test_plugin.py").write_text(
        "from pathlib import Path\n"
        "def test_mutates_source():\n"
        "    Path(__file__).parents[1].joinpath('tracked.txt').write_text('after\\n')\n",
        encoding="utf-8",
    )
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "--quiet", "-m", "test: source mutant")

    with pytest.raises(gate.GateError, match="改写 source checkout"):
        gate._run_cases(
            Path(sys.executable),
            root / "tests",
            ("tests/test_plugin.py",),
            root,
        )


def test_runner_rejects_passing_external_fixture_that_changes_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _plugin_repo(tmp_path, "core")
    core_marker = core / "tracked.txt"
    core_marker.write_text("before\n", encoding="utf-8")
    _ = _git(core, "add", ".")
    _ = _git(core, "commit", "--quiet", "-m", "test: core marker")
    plugin = _plugin_repo(tmp_path, "fixture")
    (plugin / "tests" / "test_plugin.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def test_mutates_core():\n"
        "    root = Path(os.environ['AKASHIC_AGENT_ROOT'])\n"
        "    root.joinpath('tracked.txt').write_text('after\\n')\n",
        encoding="utf-8",
    )
    _ = _git(plugin, "add", ".")
    _ = _git(plugin, "commit", "--quiet", "-m", "test: core mutant")
    monkeypatch.setattr(gate, "ROOT", core)

    with pytest.raises(gate.GateError, match="改写 Core checkout"):
        gate._run_cases(
            Path(sys.executable),
            plugin / "tests",
            ("tests/test_plugin.py",),
            plugin,
        )


def test_execution_mode_requires_explicit_python_and_narrows_pending_bypass() -> None:
    with pytest.raises(gate.GateError, match="每个插件"):
        gate._validate_execution_mode(
            identity_only=False,
            allow_pending=False,
            expected_ids={"a", "b"},
            python_ids={"a"},
        )
    with pytest.raises(gate.GateError, match="只能与 --identity-only"):
        gate._validate_execution_mode(
            identity_only=False,
            allow_pending=True,
            expected_ids={"a"},
            python_ids={"a"},
        )
    gate._validate_execution_mode(
        identity_only=True,
        allow_pending=True,
        expected_ids={"a", "b"},
        python_ids=set(),
    )


@pytest.mark.asyncio
async def test_generic_coexistence_probe_keeps_non_content_plugin_out_of_mailbox(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)

    receipt = await gate._run_coexistence_probe(
        {
            "plugin_id": "fixture",
            "config_toml": "",
            "expected_content_rows": 0,
        },
        root,
    )

    assert receipt["plugin_id"] == "fixture"
    assert receipt["content_rows"] == 0
    assert receipt["content_before"] == receipt["content_after"]
    assert receipt["changed_tables"] == []
    assert receipt["source_before"] == receipt["source_after"]
    assert receipt["core_before"] == receipt["core_after"]


def test_content_logical_state_detects_empty_submission(tmp_path: Path) -> None:
    path = tmp_path / "content.sqlite3"
    store = gate.ContentStore(path)
    store.initialize()
    before = gate._content_logical_state(path)

    receipt = store.submit("fixture", "empty", ())
    after = gate._content_logical_state(path)

    assert receipt["inserted"] == []
    assert before["items"] == after["items"]
    assert before["submissions"] != after["submissions"]
