#!/usr/bin/env python3
"""计量 Git 跟踪的生产源码行数。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import subprocess
import tokenize
from pathlib import Path
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]

PYTHON_DIRECTORY_ROOTS = (
    "agent",
    "bootstrap",
    "bus",
    "core",
    "infra",
    "memory2",
    "migrations",
    "plugins",
    "prompts",
    "session",
    "utils",
)
EXCLUDED_SOURCE_PARTS = frozenset(
    {"build", "bundle", "bundles", "dist", "vendor", "vendors"}
)


class SlocCounts(TypedDict):
    byLanguage: dict[str, int]
    byRoot: dict[str, int]


class MeasurementReport(TypedDict):
    version: int
    sourceSetDigest: str
    fileCount: int
    sloc: SlocCounts
    total: int


def _normalise_path(path: str) -> str:
    """返回 Git 与摘要使用的仓库相对路径。"""

    return path.replace("\\", "/")


def _utf8_sort(values: list[str] | set[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def production_source_root(path: str) -> str | None:
    """返回路径所属的计量根；排除路径返回 None。"""

    relative = _normalise_path(path)
    parts = relative.split("/")
    name = parts[-1]
    if not relative or any(part in EXCLUDED_SOURCE_PARTS for part in parts):
        return None
    if name.endswith(".d.ts") or name.endswith(".d.tsx"):
        return None

    if name.endswith(".py"):
        if relative == "main.py":
            return "main.py"
        if parts[0] in PYTHON_DIRECTORY_ROOTS:
            return parts[0]
        if parts[:3] == ["sdk", "python", "src"]:
            return "sdk/python/src"
        return None

    if name.endswith(".ts") or name.endswith(".tsx"):
        if len(parts) >= 4 and parts[0] == "frontend" and parts[2] == "src":
            return "/".join(parts[:3])
    return None


def is_production_source_path(path: str) -> bool:
    """判断仓库相对路径是否属于生产源码。"""

    return production_source_root(path) is not None


def _tracked_files() -> list[str]:
    """返回当前工作树中仍存在的 Git 跟踪生产源码。"""

    # 1. 只从 Git 索引获取候选路径。
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    candidates = [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]

    # 2. 未暂存删除仍存在于索引，按当前工作树排除。
    return _utf8_sort(
        [
            path
            for path in candidates
            if production_source_root(path) is not None and (ROOT / path).is_file()
        ]
    )


type SourceSpan = tuple[tuple[int, int], tuple[int, int]]


def _docstring_spans(source: str) -> list[SourceSpan]:
    tree = ast.parse(source)
    spans: list[SourceSpan] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Module),
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            spans.append(
                (
                    (first.lineno, first.col_offset),
                    (cast(int, first.end_lineno), cast(int, first.end_col_offset)),
                )
            )
    return spans


def _inside_span(token: tokenize.TokenInfo, span: SourceSpan) -> bool:
    start, end = span
    return start <= token.start and token.end <= end


def count_python_sloc(source: str) -> int:
    """统计 Python 有效源码行，排除空行、纯注释和 docstring。"""

    # 1. AST 只拥有 docstring 身份，tokenize 负责物理行计数。
    docstring_spans = _docstring_spans(source)
    source_lines: set[int] = set()
    token_stream = tokenize.generate_tokens(io.StringIO(source).readline)
    ignored = {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }
    for token in token_stream:
        if token.type in ignored:
            continue
        if any(_inside_span(token, span) for span in docstring_spans):
            continue
        start_line, end_line = token.start[0], token.end[0]
        source_lines.update(range(start_line, end_line + 1))
    return len(source_lines)


def _finish_typescript_line(lines: list[bool], line_has_source: bool) -> bool:
    lines.append(line_has_source)
    return False


def count_typescript_sloc(source: str) -> int:
    """用显式状态机统计 TypeScript 有效源码行。"""

    # 1. 栈保留模板插值、嵌套字符串和注释的返回状态。
    lines: list[bool] = []
    states = ["normal"]
    expression_depths: list[int] = []
    line_has_source = False
    index = 0
    while index < len(source):
        state = states[-1]
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""

        if char == "\n":
            if state == "template_string":
                line_has_source = True
            line_has_source = _finish_typescript_line(lines, line_has_source)
            index += 1
            if state == "line_comment":
                _ = states.pop()
            continue

        if state == "line_comment":
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                _ = states.pop()
                index += 2
            else:
                index += 1
            continue
        if state in {"single_string", "double_string"}:
            line_has_source = True
            if char == "\\":
                index += 1
                if index < len(source) and source[index] != "\n":
                    index += 1
            elif (
                (state == "single_string" and char == "'")
                or (state == "double_string" and char == '"')
            ):
                _ = states.pop()
                index += 1
            else:
                index += 1
            continue
        if state == "template_string":
            line_has_source = True
            if char == "\\":
                index += 1
                if index < len(source) and source[index] != "\n":
                    index += 1
            elif char == "`":
                _ = states.pop()
                index += 1
            elif char == "$" and next_char == "{":
                states.append("template_expression")
                expression_depths.append(0)
                index += 2
            else:
                index += 1
            continue

        # 2. normal 与 template_expression 共享代码 token 规则。
        if char == "/" and next_char == "/":
            states.append("line_comment")
            index += 2
        elif char == "/" and next_char == "*":
            states.append("block_comment")
            index += 2
        elif char == "'":
            states.append("single_string")
            line_has_source = True
            index += 1
        elif char == '"':
            states.append("double_string")
            line_has_source = True
            index += 1
        elif char == "`":
            states.append("template_string")
            line_has_source = True
            index += 1
        elif state == "template_expression" and char == "{":
            expression_depths[-1] += 1
            line_has_source = True
            index += 1
        elif state == "template_expression" and char == "}":
            line_has_source = True
            index += 1
            if expression_depths[-1] == 0:
                _ = expression_depths.pop()
                _ = states.pop()
            else:
                expression_depths[-1] -= 1
        elif char.isspace():
            index += 1
        else:
            line_has_source = True
            index += 1

    if source and not source.endswith("\n"):
        lines.append(line_has_source)
    return sum(lines)


def _count_file(relative: str) -> int:
    data = (ROOT / relative).read_bytes()
    if relative.endswith(".py"):
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        source = data.decode(encoding)
        return count_python_sloc(source)
    source = data.decode("utf-8")
    return count_typescript_sloc(source)


def _source_set_digest(files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def measure() -> MeasurementReport:
    """只读计量当前 Git 工作树中的生产源码。"""

    # 1. 冻结候选路径，再按同一顺序计量内容。
    files = _tracked_files()
    roots = {root: 0 for root in _all_source_roots(files)}
    language_sloc = {"python": 0, "typescript": 0}
    for relative in files:
        sloc = _count_file(relative)
        root = production_source_root(relative)
        if root is None:
            raise RuntimeError(f"source predicate changed while measuring: {relative}")
        roots[root] += sloc
        language = "python" if relative.endswith(".py") else "typescript"
        language_sloc[language] += sloc
    # 2. 汇总稳定的语言、源码根和全局结果。
    return {
        "version": 1,
        "sourceSetDigest": _source_set_digest(files),
        "fileCount": len(files),
        "sloc": {
            "byLanguage": language_sloc,
            "byRoot": roots,
        },
        "total": sum(language_sloc.values()),
    }


def _all_source_roots(files: list[str]) -> list[str]:
    roots: set[str] = set()
    for path in files:
        root = production_source_root(path)
        if root is not None:
            roots.add(root)
    return _utf8_sort(roots)


def _print_human(report: MeasurementReport) -> None:
    by_language = report["sloc"]["byLanguage"]
    by_root = report["sloc"]["byRoot"]
    print(f"Production SLOC v{report['version']}")
    print(f"Source-set digest: {report['sourceSetDigest']}")
    print(f"Files: {report['fileCount']}")
    print("By language:")
    for language, value in by_language.items():
        print(f"  {language}: {value}")
    print("By source root:")
    for root, value in by_root.items():
        print(f"  {root}: {value}")
    print(f"Total: {report['total']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="计量 Git 跟踪生产源码 SLOC")
    _ = parser.add_argument("--json", action="store_true", help="输出稳定 JSON")
    args = parser.parse_args()
    report = measure()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
