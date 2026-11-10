"""web module 校验事实按内容寻址：同一源码只 tree-sitter 解析一次,失败不落事实。"""
import pytest

from plugins.ui import web

_VALID = "export function activate(ctx) {}\n"
_INVALID = "export async function activate(ctx) {}\n"


def test_validation_fact_reused_only_for_same_source(tmp_path, monkeypatch):
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
    fact = web._validation_fact_path(_VALID.encode(), tmp_path)
    assert fact is not None and fact.is_file()

    def fail_parse(*_args, **_kwargs):
        raise AssertionError("相同源码应命中校验事实,不再 tree-sitter 解析")

    monkeypatch.setattr(web.Parser, "parse", fail_parse)
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
    monkeypatch.undo()

    changed = "export function activate(ctx) { return 1 }\n"
    web._validate_javascript_module(changed, fact_root=tmp_path)
    assert web._validation_fact_path(changed.encode(), tmp_path) is not None


def test_invalid_module_never_records_fact(tmp_path):
    with pytest.raises(RuntimeError, match="同步 activate"):
        web._validate_javascript_module(_INVALID, fact_root=tmp_path)
    assert web._validation_fact_path(_INVALID.encode(), tmp_path) is not None
    assert not web._validation_fact_path(_INVALID.encode(), tmp_path).exists()
    with pytest.raises(RuntimeError, match="同步 activate"):
        web._validate_javascript_module(_INVALID, fact_root=tmp_path)


def test_corrupt_fact_falls_back_to_parse(tmp_path):
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
    fact = web._validation_fact_path(_VALID.encode(), tmp_path)
    assert fact is not None
    fact.unlink()
    fact.mkdir()
    assert not web._validation_fact_verified(_VALID.encode(), tmp_path)
    # 目录占位时照常重新校验;写事实静默放弃,清理后下一次恢复缓存。
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
    assert not web._validation_fact_verified(_VALID.encode(), tmp_path)


def test_missing_fact_root_always_parses(monkeypatch):
    calls = 0
    real_parse = web.Parser.parse

    def counting_parse(self, source):
        nonlocal calls
        calls += 1
        return real_parse(self, source)

    monkeypatch.setattr(web.Parser, "parse", counting_parse)
    web._validate_javascript_module(_VALID)
    web._validate_javascript_module(_VALID)
    assert calls == 2
    assert web._validation_fact_path(_VALID.encode(), None) is None
