"""公共合同摘要按源码内容寻址缓存：未变源码不重复解析 AST,损坏事实当未命中。"""
import ast
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


def test_digest_fact_reused_only_for_same_source(contract_file):
    contract_file.write_text("VALUE = 1\n")
    first = _digest(contract_file)
    fact = public_contracts._digest_fact_path(contract_file.read_bytes())
    assert fact is not None and fact.is_file()

    def fail_parse(*_args, **_kwargs):
        raise AssertionError("相同源码应命中摘要事实,不再解析 AST")

    ast_parse = public_contracts.ast.parse
    public_contracts.ast.parse = fail_parse
    try:
        assert _digest(contract_file) == first
    finally:
        public_contracts.ast.parse = ast_parse

    contract_file.write_text("VALUE = 2\n")
    assert _digest(contract_file) != first


def test_digest_ignores_docstring_and_comment_changes(contract_file):
    contract_file.write_text("class API:\n    '''旧文档'''\n    def run(self):\n        return 1\n")
    first = _digest(contract_file)
    contract_file.write_text("# 注释\nclass API:\n    '''新文档'''\n    def run(self):\n        return 1\n")
    assert _digest(contract_file) == first


def test_corrupt_digest_fact_falls_back_to_parse(contract_file):
    contract_file.write_text("VALUE = 3\n")
    assert public_contracts._read_digest_fact(b"VALUE = 3\n") is None
    _digest(contract_file)
    fact = public_contracts._digest_fact_path(contract_file.read_bytes())
    assert fact is not None
    fact.write_text("garbage", encoding="ascii")
    assert public_contracts._read_digest_fact(contract_file.read_bytes()) is None
    assert _digest(contract_file) == public_contracts._read_digest_fact(
        contract_file.read_bytes()
    )
