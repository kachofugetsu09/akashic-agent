"""公共合同摘要按源码内容寻址缓存：变化的源码不借用旧摘要，损坏事实当未命中。"""
from pathlib import Path

import pytest

from agent.plugins import public_contracts


@pytest.fixture
def contract_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    return artifact / "contract.py"


def _digest(path: Path) -> bytes:
    return public_contracts._contract_digest(path)


def test_digest_never_serves_stale_source(contract_file):
    contract_file.write_text("VALUE = 1\n")
    first = _digest(contract_file)
    assert _digest(contract_file) == first

    contract_file.write_text("VALUE = 2\n")
    assert _digest(contract_file) != first


def test_digest_ignores_docstring_and_comment_changes(contract_file):
    contract_file.write_text("class API:\n    '''旧文档'''\n    def run(self):\n        return 1\n")
    first = _digest(contract_file)
    contract_file.write_text("# 注释\nclass API:\n    '''新文档'''\n    def run(self):\n        return 1\n")
    assert _digest(contract_file) == first


@pytest.mark.parametrize("corrupt", [b"garbage", b"\xff", b"aa" * 15 + b"  " + b"aa" * 16])
def test_corrupt_digest_fact_falls_back_to_parse(contract_file, corrupt):
    contract_file.write_text("VALUE = 3\n")
    expected = _digest(contract_file)
    fact = public_contracts._digest_fact_path(contract_file.read_bytes())
    assert fact is not None
    fact.write_bytes(corrupt)
    assert _digest(contract_file) == expected
