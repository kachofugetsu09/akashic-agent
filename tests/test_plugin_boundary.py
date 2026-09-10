"""插件边界门：规则判定与真实仓库一致性。

测试只验证可失败的边界规则本身，不复制 policy 表的字面内容。真实仓库的
基线数量由 `scripts/plugin_boundary.py check` 维护，本文件只断言它通过。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import scripts.plugin_boundary as boundary

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 规则纯函数 ────────────────────────────────────────────────

def _import(importer: str, module: str) -> boundary.Import:
    return boundary.Import(importer, module)


def test_core_importing_plugin_is_flagged() -> None:
    violations = boundary.check_core_imports_plugin([
        _import("bootstrap/app.py", "plugins.delivery.senders"),
        _import("agent/plugin_composition/timers.py", "session.message"),
    ])
    assert [item.key for item in violations] == [
        "bootstrap/app.py|plugins.delivery.senders"
    ]


def test_r1_exemption_applies_only_to_listed_paths() -> None:
    """迁移 payload 豁免只覆盖登记路径，其它 Core 文件仍然受 R1 约束。"""

    imports = [
        _import("agent/migrations/akasha_sidecar.py", "plugins.akasha.config"),
        _import("migrations/yoyo/20260905_04_akasha_consumption.py", "plugins.akasha.config"),
        _import("bootstrap/app.py", "plugins.delivery.senders"),
    ]
    exempt = ["agent/migrations/**", "migrations/**"]
    assert [item.key for item in boundary.check_core_imports_plugin(imports, exempt)] == [
        "bootstrap/app.py|plugins.delivery.senders"
    ]


def test_r1_exemption_rule_that_matches_nothing_fails() -> None:
    """僵尸豁免必须失败，避免留下永不命中的例外。"""

    policy = _policy({}, r1_exemptions={"paths": ["agent/nowhere/**"]})
    errors = boundary.check_r1_exemptions(policy, ["agent/migrations/akasha_sidecar.py"])
    assert len(errors) == 1 and "agent/nowhere/**" in errors[0]


def test_r1_exemptions_are_declared_in_policy() -> None:
    """真实仓库必须显式登记豁免；豁免命中数在 check 输出中可见。"""

    result = subprocess.run(
        [sys.executable, "scripts/plugin_boundary.py", "check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "R1 豁免（迁移 payload，见决策 0064）" in result.stdout
    assert "agent/migrations/**" in result.stdout


def test_plugin_deep_core_import_is_flagged() -> None:
    violations = boundary.check_plugin_deep_core([
        _import("plugins/reply/plugin.py", "session.log"),
        _import("plugins/reply/plugin.py", "agent.restart"),
    ])
    assert {item.module for item in violations} == {"session.log", "agent.restart"}


def test_plugin_sanctioned_surface_passes() -> None:
    violations = boundary.check_plugin_deep_core([
        _import("plugins/reply/plugin.py", "agent.plugin_composition"),
        _import("plugins/reply/plugin.py", "agent.plugin_composition.timers"),
        _import("plugins/reply/plugin.py", "agent.plugin_contracts"),
        _import("plugins/reply/plugin.py", "agent.plugin_contracts.message"),
        _import("plugins/reply/plugin.py", "asyncio"),
    ])
    assert violations == []


def test_prefix_lookalike_is_not_sanctioned() -> None:
    """`agent.plugin_composition_x` 不能因为前缀相同而放行。"""

    violations = boundary.check_plugin_deep_core([
        _import("plugins/reply/plugin.py", "agent.plugin_composition_extra"),
    ])
    assert [item.module for item in violations] == ["agent.plugin_composition_extra"]


def test_cross_plugin_import_is_flagged() -> None:
    violations = boundary.check_cross_plugin([
        _import("plugins/reply/plugin.py", "plugins.tools.api"),
        _import("plugins/reply/plugin.py", "plugins.reply.api"),
        _import("plugins/reply/plugin.py", "plugins.reply"),
    ])
    assert [item.module for item in violations] == ["plugins.tools.api"]


def test_relative_import_resolution() -> None:
    assert boundary._resolve_relative("plugins/reply/follow.py", 1, "api") == "plugins.reply.api"
    assert boundary._resolve_relative("plugins/reply/follow.py", 2, "tools.api") == "plugins.tools.api"
    # 越出仓库根的相对 import 无法解析，按“不是违规”处理而不是崩溃。
    assert boundary._resolve_relative("plugins/reply/follow.py", 5, "x") is None


# ── 规则 R4：角色表与代码双向一致 ─────────────────────────────

def _policy(
    capabilities: dict[str, dict[str, str]],
    phantom: dict | None = None,
    r1_exemptions: dict | None = None,
) -> dict:
    policy: dict = {"capabilities": capabilities, "phantom": phantom or {}}
    if r1_exemptions is not None:
        policy["R1_exemptions"] = r1_exemptions
    return policy


def test_capability_table_rejects_unregistered_key(monkeypatch) -> None:
    monkeypatch.setattr(
        boundary, "discover_service_keys", lambda: {"core.new_thing": "agent/x.py"}
    )
    errors = boundary.check_capability_table(_policy({}))
    assert len(errors) == 1
    assert "core.new_thing" in errors[0]


def test_capability_table_rejects_registered_but_missing_key(monkeypatch) -> None:
    monkeypatch.setattr(boundary, "discover_service_keys", lambda: {})
    errors = boundary.check_capability_table(
        _policy({"core.ghost": {"role": "core"}})
    )
    assert len(errors) == 1
    assert "core.ghost" in errors[0]


def test_capability_table_rejects_entry_without_role(monkeypatch) -> None:
    monkeypatch.setattr(
        boundary, "discover_service_keys", lambda: {"core.timers": "agent/plugin_composition/timers.py"}
    )
    errors = boundary.check_capability_table(_policy({"core.timers": {"note": "无角色"}}))
    assert any("缺少 role" in error for error in errors)


def test_phantom_names_are_absent_today() -> None:
    """SCOPED_TURNS 等文档承诺名当前必须不存在于 Python 代码。"""

    errors = boundary.check_phantom_names(boundary.load_policy())
    assert errors == []


def test_phantom_name_appearing_fails(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "fake.py"
    source.write_text("SCOPED_TURNS = object()\n", encoding="utf-8")
    monkeypatch.setattr(boundary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "tracked_python_files", lambda: ["fake.py"])
    errors = boundary.check_phantom_names(_policy({}, {"SCOPED_TURNS": {"note": "未实现"}}))
    assert len(errors) == 1
    assert "SCOPED_TURNS" in errors[0]


def test_phantom_names_ignore_tests(monkeypatch, tmp_path: Path) -> None:
    """守护测试会合法地写出幽灵名；R5 只断言实现里不存在。"""

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "guard.py").write_text("SCOPED_TURNS = 1\n", encoding="utf-8")
    monkeypatch.setattr(boundary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        boundary, "tracked_python_files", lambda: ["tests/guard.py"]
    )
    assert boundary.implementation_python_files() == []
    assert boundary.check_phantom_names(_policy({}, {"SCOPED_TURNS": {"note": "x"}})) == []


# ── 端到端：门真的会因为新增违规而失败 ────────────────────────

def _synthetic_repo(tmp_path: Path, *, plugin_import_line: str, baseline: str = "") -> None:
    (tmp_path / "bootstrap").mkdir(parents=True)
    (tmp_path / "plugins" / "reply").mkdir(parents=True)
    (tmp_path / "bootstrap" / "app.py").write_text(
        plugin_import_line, encoding="utf-8"
    )
    (tmp_path / "plugins" / "reply" / "plugin.py").write_text(
        "import asyncio\n", encoding="utf-8"
    )
    (tmp_path / "policy.toml").write_text("[capabilities]\n", encoding="utf-8")
    (tmp_path / "baseline.toml").write_text(baseline, encoding="utf-8")


def test_gate_fails_on_new_core_to_plugin_import(monkeypatch, tmp_path: Path) -> None:
    _synthetic_repo(
        tmp_path, plugin_import_line="from plugins.reply import plugin\n"
    )
    monkeypatch.setattr(boundary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "POLICY_PATH", tmp_path / "policy.toml")
    monkeypatch.setattr(boundary, "BASELINE_PATH", tmp_path / "baseline.toml")
    monkeypatch.setattr(
        boundary, "tracked_python_files", lambda: ["bootstrap/app.py", "plugins/reply/plugin.py"]
    )
    assert boundary.run_check() == 1


def test_gate_passes_when_violation_is_recorded_in_baseline(monkeypatch, tmp_path: Path) -> None:
    _synthetic_repo(
        tmp_path,
        plugin_import_line="from plugins.reply import plugin\n",
        baseline='[baseline]\nR1 = [\n    "bootstrap/app.py|plugins.reply",\n]\nR2 = []\nR3 = []\n',
    )
    monkeypatch.setattr(boundary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "POLICY_PATH", tmp_path / "policy.toml")
    monkeypatch.setattr(boundary, "BASELINE_PATH", tmp_path / "baseline.toml")
    monkeypatch.setattr(
        boundary, "tracked_python_files", lambda: ["bootstrap/app.py", "plugins/reply/plugin.py"]
    )
    assert boundary.run_check() == 0


def test_stale_baseline_entry_fails(monkeypatch, tmp_path: Path) -> None:
    """已还清的债务必须从账本删除，否则账本会掩盖真实状态。"""

    _synthetic_repo(
        tmp_path,
        plugin_import_line="import asyncio\n",
        baseline='[baseline]\nR1 = [\n    "bootstrap/app.py|plugins.gone",\n]\nR2 = []\nR3 = []\n',
    )
    monkeypatch.setattr(boundary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "POLICY_PATH", tmp_path / "policy.toml")
    monkeypatch.setattr(boundary, "BASELINE_PATH", tmp_path / "baseline.toml")
    monkeypatch.setattr(
        boundary, "tracked_python_files", lambda: ["bootstrap/app.py", "plugins/reply/plugin.py"]
    )
    assert boundary.run_check() == 1


# ── 真实仓库 ──────────────────────────────────────────────────

def test_repository_passes_boundary_gate() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/plugin_boundary.py", "check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "插件边界门通过" in result.stdout


def test_every_core_service_key_has_a_role() -> None:
    policy = boundary.load_policy()
    declared = cast("dict[str, dict[str, str]]", policy["capabilities"])
    for name in boundary.discover_service_keys():
        assert name in declared, f"{name} 未在 plugin_boundary.toml 登记角色"
        assert declared[name]["role"] in {"core", "seam", "bundle", "claim"}


@pytest.mark.parametrize("name", ["SCOPED_TURNS", "CONTINUATIONS", "BACKGROUND_JOBS"])
def test_documented_but_unimplemented_names_stay_listed(name: str) -> None:
    """实现这三个名字时必须同时更新文档与策略表，本测试强制该动作。"""

    phantoms = cast("dict[str, object]", boundary.load_policy()["phantom"])
    assert name in phantoms
