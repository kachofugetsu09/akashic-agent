from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sloc_module() -> ModuleType:
    path = ROOT / "scripts" / "measure_production_sloc.py"
    spec = importlib.util.spec_from_file_location("production_sloc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_sloc_excludes_docstrings_and_comments_but_counts_real_strings() -> None:
    sloc = _sloc_module()
    source = '''
"""模块说明\n第二行说明\n"""
# 独立注释
def render():
    """函数说明\n    函数说明第二行\n    """
    value = """真实字符串 # 不是注释\n    真实字符串第二行\n    """
    return value  # 行内注释不抹掉代码
'''

    assert sloc.count_python_sloc(source) == 5


def test_python_standalone_string_in_non_docstring_block_is_counted() -> None:
    sloc = _sloc_module()
    source = "if enabled:\n    \"runtime marker\"\n"

    assert sloc.count_python_sloc(source) == 2


def test_python_code_after_single_line_docstring_is_not_removed() -> None:
    sloc = _sloc_module()

    assert sloc.count_python_sloc('"""模块说明"""; VALUE = 1\n') == 1


def test_python_parenthesised_concatenated_docstring_is_excluded() -> None:
    sloc = _sloc_module()
    source = '''
def render():
    (
        "第一段说明"
        "第二段说明"
    )
    return "value"
'''

    assert sloc.count_python_sloc(source) == 2


def test_typescript_lexer_keeps_comment_markers_inside_strings() -> None:
    sloc = _sloc_module()
    source = '''
const text = `第一行 // 仍是字符串
第二行 /* 仍是字符串 */
`;
/* 独立块注释
   第二行注释 */
const marker = "/* // 都是字符串";
// 独立行注释
return marker;
'''

    assert sloc.count_typescript_sloc(source) == 5


def test_typescript_template_interpolation_excludes_comment_only_lines() -> None:
    sloc = _sloc_module()
    source = """const text = `${
// 插值注释
value
}`;
"""

    assert sloc.count_typescript_sloc(source) == 3


def test_tracked_files_ignore_unstaged_deletions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sloc = _sloc_module()
    kept = tmp_path / "agent" / "kept.py"
    kept.parent.mkdir()
    kept.write_text("VALUE = 1\n", encoding="utf-8")
    git_output = b"agent/kept.py\0agent/deleted.py\0"
    monkeypatch.setattr(sloc, "ROOT", tmp_path)
    monkeypatch.setattr(
        sloc.subprocess,
        "run",
        lambda *_args, **_kwargs: sloc.subprocess.CompletedProcess(
            ["git", "ls-files"], 0, git_output, b""
        ),
    )

    assert sloc._tracked_files() == ["agent/kept.py"]


def test_source_set_includes_only_approved_production_extensions_and_roots() -> None:
    sloc = _sloc_module()
    included = (
        "main.py",
        "agent/core/runtime.py",
        "plugin_packages/example/plugin.py",
        "plugin_packages/example/view.tsx",
        "sdk/python/src/sdk.py",
        "frontend/chat/src/main.tsx",
    )
    excluded = (
        "tests/test_runtime.py",
        "eval/runner.py",
        "docker/debug/gate.py",
        "scripts/measure_production_sloc.py",
        "frontend/chat/src/styles.css",
        "frontend/chat/src/types.d.ts",
        "frontend/chat/dist/bundle.js",
        "plugin_packages/vendor/third_party.py",
    )

    assert all(sloc.is_production_source_path(path) for path in included)
    assert all(not sloc.is_production_source_path(path) for path in excluded)
    assert sloc.production_source_root("frontend/chat/src/main.tsx") == (
        "frontend/chat/src"
    )
    assert sloc.production_source_root("migrations/20260722_example/migration.py") == (
        "migrations"
    )


def test_measurement_report_has_stable_language_root_and_total_fields() -> None:
    sloc = _sloc_module()
    report = sloc.measure()
    by_language = report["sloc"]["byLanguage"]
    by_root = report["sloc"]["byRoot"]

    assert report["version"] == 1
    assert report["fileCount"] > 0
    assert len(report["sourceSetDigest"]) == 64
    assert set(by_language) == {"python", "typescript"}
    assert report["total"] == sum(by_language.values())
    assert report["total"] == sum(by_root.values())
