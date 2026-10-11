"""manifest 事实缓存只省去 plugin.py 的 AST 解析；内容、校验与失败语义不变。"""
from pathlib import Path

import pytest

from agent.plugins import static_manifest
from agent.plugins.static_manifest import PluginSourceContentError, load_static_plugin_manifest

_SOURCE = 'name = "demo"\nversion = "1.0.0"\napi_version = 3\nentrypoints = {"run": "demo.cli.main"}\n'


@pytest.fixture
def plugin_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    root = tmp_path / "artifact"
    root.mkdir()
    _ = (root / "plugin.py").write_text(_SOURCE)
    return root


def _fresh_process() -> None:
    static_manifest._parse_plugin_identity.cache_clear()
    static_manifest._parse_plugin_entrypoints.cache_clear()
    static_manifest._parse_source_tree.cache_clear()


def _forbid_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object) -> object:
        raise AssertionError("cached facts should skip parsing")

    monkeypatch.setattr(static_manifest, "_parse_source_tree", fail)


def test_facts_reused_across_processes_only_for_same_source(plugin_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first = load_static_plugin_manifest(plugin_root)
    facts = sorted((tmp_path / "home" / "manifest-facts").rglob("*.json"))
    assert [path.name.split(".", 1)[1] for path in facts] == ["entrypoints.json", "identity.json"]
    assert sorted(path.name for path in plugin_root.iterdir()) == ["plugin.py"]

    _fresh_process()
    with monkeypatch.context() as patch:
        _forbid_parse(patch)
        assert load_static_plugin_manifest(plugin_root) == first

    _fresh_process()
    _ = (plugin_root / "plugin.py").write_text(_SOURCE.replace("1.0.0", "1.0.1"))
    changed = load_static_plugin_manifest(plugin_root)
    assert changed.version == "1.0.1" and changed.identity_digest != first.identity_digest


def test_invalid_source_is_never_cached(plugin_root: Path, tmp_path: Path):
    _ = (plugin_root / "plugin.py").write_text(_SOURCE.replace("api_version = 3", "api_version = 2"))
    for _ in range(2):
        _fresh_process()
        with pytest.raises(PluginSourceContentError, match="api_version"):
            _ = load_static_plugin_manifest(plugin_root)
    assert not list((tmp_path / "home").rglob("*.identity.json"))


@pytest.mark.parametrize(("kind", "payload"), [
    ("identity", '["Bad Name", "1.0.0", 3]'),
    ("identity", '["demo", "1.0.0", true]'),
    ("entrypoints", '[["run", "demo.cli.main"], ["run", "demo.cli.other"]]'),
    ("entrypoints", '[["Run", "demo.cli.main"]]'),
    ("entrypoints", '{"run": "demo.cli.main"}'),
    ("entrypoints", "not json"),
])
def test_invalid_cached_fact_is_treated_as_miss(plugin_root: Path, kind: str, payload: str):
    expected = load_static_plugin_manifest(plugin_root)
    path = static_manifest._fact_path(kind, _SOURCE)
    assert path is not None
    _ = path.write_text(payload)
    _fresh_process()
    assert load_static_plugin_manifest(plugin_root) == expected


def test_disabled_or_unwritable_cache_falls_back_to_parsing(plugin_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(static_manifest.sys, "dont_write_bytecode", True)
    expected = load_static_plugin_manifest(plugin_root)
    assert not (tmp_path / "home" / "manifest-facts").exists()

    monkeypatch.setattr(static_manifest.sys, "dont_write_bytecode", False)
    _ = (tmp_path / "home").write_text("not a directory")
    _fresh_process()
    assert load_static_plugin_manifest(plugin_root) == expected
