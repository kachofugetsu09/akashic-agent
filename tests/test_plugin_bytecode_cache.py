"""插件字节码缓存属于宿主，只在源码哈希一致时复用，不改写插件制品目录。"""
from pathlib import Path

import pytest

from agent.plugins import importer


@pytest.fixture
def plugin_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(importer.sys, "pycache_prefix", None)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    return artifact / "mod.py"


def _run(path: Path) -> object:
    importer._compile_source.cache_clear()
    namespace: dict[str, object] = {}
    exec(importer._compile_source(path.read_bytes(), str(path)), namespace)
    return namespace["VALUE"]


def test_bytecode_cache_reused_only_for_same_source(plugin_source, tmp_path, monkeypatch):
    plugin_source.write_text("VALUE = 1\n")
    assert _run(plugin_source) == 1
    cache = importer._cache_path(str(plugin_source))
    assert cache is not None and cache.is_file()
    assert cache.is_relative_to(tmp_path / "home" / "bytecode")
    assert sorted(path.name for path in plugin_source.parent.iterdir()) == ["mod.py"]

    def fail_compile(*_args):
        raise AssertionError("cached bytecode should be reused")

    monkeypatch.setattr(importer, "compile", fail_compile, raising=False)
    assert _run(plugin_source) == 1
    monkeypatch.delattr(importer, "compile")

    plugin_source.write_text("VALUE = 2\n")
    assert _run(plugin_source) == 2


def test_corrupt_or_disabled_cache_falls_back_to_compile(plugin_source, monkeypatch):
    plugin_source.write_text("VALUE = 3\n")
    cache = importer._cache_path(str(plugin_source))
    assert cache is not None
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"garbage")
    assert _run(plugin_source) == 3

    monkeypatch.setattr(importer.sys, "dont_write_bytecode", True)
    cache.unlink()
    assert _run(plugin_source) == 3
    assert not cache.exists()
