"""web module 校验事实按内容寻址：缓存不能让变化或无效的源码绕过校验。"""
import pytest

from plugins.ui import web

_VALID = "export function activate(ctx) {}\n"
_INVALID = "export async function activate(ctx) {}\n"


def test_validation_fact_never_masks_changed_source(tmp_path):
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
    with pytest.raises(RuntimeError, match="同步 activate"):
        web._validate_javascript_module(_INVALID, fact_root=tmp_path)


def test_invalid_module_keeps_failing(tmp_path):
    for _ in range(2):
        with pytest.raises(RuntimeError, match="同步 activate"):
            web._validate_javascript_module(_INVALID, fact_root=tmp_path)


def test_corrupt_fact_falls_back_to_parse(tmp_path):
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
    fact = web._validation_fact_path(_VALID.encode(), tmp_path)
    assert fact is not None
    fact.unlink()
    fact.mkdir()
    # 目录占位时照常重新校验，不因写事实失败而报错。
    web._validate_javascript_module(_VALID, fact_root=tmp_path)
