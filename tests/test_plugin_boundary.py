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
        boundary.Import("plugins/reply/plugin.py", "plugins.reply.api", absolute=False),
        boundary.Import("plugins/reply/plugin.py", "plugins.reply", absolute=False),
    ])
    assert [item.module for item in violations] == ["plugins.tools.api"]


def test_relative_import_resolution() -> None:
    assert boundary._resolve_relative("plugins/reply/follow.py", 1, "api") == "plugins.reply.api"
    assert boundary._resolve_relative("plugins/reply/follow.py", 2, "tools.api") == "plugins.tools.api"
    # 越出仓库根的相对 import 无法解析，按“不是违规”处理而不是崩溃。
    assert boundary._resolve_relative("plugins/reply/follow.py", 5, "x") is None


# ── 规则 R4：角色表与代码双向一致 ─────────────────────────────

def _policy(capabilities: dict[str, dict[str, str]], phantom: dict | None = None) -> dict:
    return {"capabilities": capabilities, "phantom": phantom or {}}


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
        plugin_import_line="import plugins.reply\n",
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


@pytest.mark.parametrize("source,rule,target", [
    ("from plugins import beta", "R3", "plugins.beta"),
    ("from .. import beta", "R3", "plugins.beta"),
    ("from .. import *", "R3", "plugins"),
    ("import plugins.alpha.helpers", "R3", "plugins.alpha.helpers"),
    ("from agent import config", "R2", "agent.config"),
    ("from prompts.completion import build", "R2", "prompts.completion"),
    ("import memory2.store", "R2", "memory2.store"),
    ("import main", "R2", "main"),
    ("from agent.plugin_composition import new_store", "R2", "agent.plugin_composition.new_store"),
])
def test_import_syntax_cannot_hide_dependencies(source: str, rule: str, target: str) -> None:
    files = ["plugins/alpha/plugin.py", "agent/plugin_composition/new_store.py"]
    imports = boundary.collect_imports(files, {files[0]: source, files[1]: ""})
    assert target in {item.module for item in boundary.import_findings(imports)[rule]}


def test_relative_helpers_and_standard_library_are_not_core_debt() -> None:
    files = ["plugins/alpha/plugin.py"]
    imports = boundary.collect_imports(files, {files[0]: "from . import helpers\nfrom types import MappingProxyType"})
    assert not any(boundary.import_findings(imports).values())


def test_migrations_are_not_a_blanket_exemption() -> None:
    assert boundary.check_core_imports_plugin([
        _import("migrations/yoyo/new.py", "plugins.alpha"),
    ])


def test_invalid_role_fails_the_cli_check(monkeypatch) -> None:
    monkeypatch.setattr(boundary, "discover_service_keys", lambda: {"x": "agent/x.py"})
    assert boundary.check_capability_table(_policy({"x": {"role": "anything"}}))


def test_service_key_alias_and_lowercase_declarations_are_checked(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent/x.py").write_text(
        'from agent.plugin_composition import ServiceKey as Key\n'
        'import agent.plugin_composition as api\n'
        'first = Key[int](name="core.first")\nsecond = api.ServiceKey(name="core.second")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(boundary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "tracked_python_files", lambda: ["agent/x.py"])
    assert set(boundary.discover_service_keys()) == {"core.first", "core.second"}


def test_base_comparison_rejects_debt_even_when_added_to_ledger(monkeypatch, tmp_path: Path) -> None:
    _synthetic_repo(tmp_path, plugin_import_line="import plugins.reply\n",
                    baseline='[baseline]\nR1 = ["bootstrap/app.py|plugins.reply"]\n')
    monkeypatch.setattr(boundary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "POLICY_PATH", tmp_path / "policy.toml")
    monkeypatch.setattr(boundary, "BASELINE_PATH", tmp_path / "baseline.toml")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "base"], cwd=tmp_path, check=True)
    assert boundary.run_check("HEAD") == 0
    (tmp_path / "bootstrap/app.py").write_text("import plugins.beta\n", encoding="utf-8")
    (tmp_path / "baseline.toml").write_text(
        '[baseline]\nR1 = ["bootstrap/app.py|plugins.beta"]\n', encoding="utf-8")
    assert boundary.run_check() == 0
    assert boundary.run_check("HEAD") == 1


@pytest.mark.parametrize("source", [
    'import importlib\nimportlib.import_module("plugins.beta")',
    'import importlib as lib\nlib.import_module(name="plugins.beta")',
    'from importlib import import_module as load\nload("plugins.beta")',
    '__import__("plugins.beta")',
    'from builtins import __import__ as load\nload("plugins.beta")',
    'from importlib import import_module\nimport_module("..beta", __package__)',
])
def test_literal_dynamic_imports_cannot_hide_plugin_dependencies(source: str) -> None:
    files = ["plugins/alpha/plugin.py"]
    imports = boundary.collect_imports(files, {files[0]: source})
    assert "plugins.beta" in {item.module for item in boundary.check_cross_plugin(imports)}


def test_dynamic_relative_helper_keeps_generation_scope() -> None:
    files = ["plugins/alpha/plugin.py"]
    imports = boundary.collect_imports(files, {files[0]:
        'from importlib import import_module\nimport_module(".helpers", package=__package__)'})
    assert not any(boundary.import_findings(imports).values())


def test_dynamic_fixed_package_cannot_escape_generation() -> None:
    files = ["plugins/alpha/plugin.py"]
    imports = boundary.collect_imports(files, {files[0]:
        'from importlib import import_module\nimport_module(".helpers", "plugins.alpha")'})
    assert boundary.check_cross_plugin(imports)


def test_all_production_python_roots_are_checked() -> None:
    """增加生产包不能让其中的跨插件依赖自动绕过门。"""
    from scripts.measure_production_sloc import PYTHON_DIRECTORY_ROOTS

    for root in set(PYTHON_DIRECTORY_ROOTS) - {"plugins"}:
        assert boundary.check_core_imports_plugin([_import(f"{root}/probe.py", "plugins.alpha")])
        assert boundary.check_plugin_deep_core([_import("plugins/alpha/plugin.py", f"{root}.probe")])


@pytest.mark.parametrize("name", ["SCOPED_TURNS", "CONTINUATIONS", "BACKGROUND_JOBS"])
def test_documented_but_unimplemented_names_stay_listed(name: str) -> None:
    """实现这三个名字时必须同时更新文档与策略表，本测试强制该动作。"""

    phantoms = cast("dict[str, object]", boundary.load_policy()["phantom"])
    assert name in phantoms
