"""插件源码字节码缓存只在源码哈希一致时复用。"""
import importlib.util
from pathlib import Path

from agent.plugins import importer


def _compile(path: Path):
    importer._compile_source.cache_clear()
    return importer._compile_source(path.read_bytes(), str(path))


def test_bytecode_cache_reused_only_for_same_source(tmp_path, monkeypatch):
    source = tmp_path / "mod.py"
    source.write_text("VALUE = 1\n")
    cache = Path(importlib.util.cache_from_source(str(source)))
    namespace: dict[str, object] = {}
    exec(_compile(source), namespace)
    assert namespace["VALUE"] == 1 and cache.is_file()

    def fail_compile(*_args):
        raise AssertionError("cached bytecode should be reused")

    monkeypatch.setattr(importer, "compile", fail_compile, raising=False)
    exec(_compile(source), namespace)
    assert namespace["VALUE"] == 1
    monkeypatch.delattr(importer, "compile")

    source.write_text("VALUE = 2\n")
    exec(_compile(source), namespace)
    assert namespace["VALUE"] == 2


def test_corrupt_or_unwritable_cache_falls_back_to_compile(tmp_path, monkeypatch):
    source = tmp_path / "mod.py"
    source.write_text("VALUE = 3\n")
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir()
    cache.write_bytes(b"garbage")
    namespace: dict[str, object] = {}
    exec(_compile(source), namespace)
    assert namespace["VALUE"] == 3

    monkeypatch.setattr(importer.sys, "dont_write_bytecode", True)
    cache.unlink()
    exec(_compile(source), namespace)
    assert not cache.exists()
