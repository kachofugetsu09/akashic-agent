#!/usr/bin/env python3
"""验证仓库只保留获批准的 1080 项 Python 测试。"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests_scenarios" / "contracts" / "retained-test-files.txt"
EXPECTED_TESTS = 1080
EXPECTED_WEB_FILES = frozenset(
    {
        "frontend/chat/src/mobile-message-state.test.mjs",
        "frontend/chat/src/mobile-pairing.test.mjs",
        "frontend/chat/src/web-chat-transport.test.mjs",
        "tests/test_akasha_mobile_ui.mjs",
    }
)
COLLECTED = re.compile(r"(?m)^(\d+) tests? collected(?: in .*)?$")


def _load_manifest() -> tuple[str, ...]:
    """加载并验证只指向仓库 tests 目录的唯一 Python 测试文件。"""

    paths = tuple(
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not paths:
        raise ValueError("测试保留清单不能为空")
    if len(paths) != len(set(paths)):
        raise ValueError("测试保留清单包含重复路径")

    tests_root = (ROOT / "tests").resolve()
    for raw in paths:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"测试路径必须是仓库内相对路径: {raw}")
        path = (ROOT / relative).resolve()
        if path.suffix != ".py" or not path.is_relative_to(tests_root) or not path.is_file():
            raise ValueError(f"测试路径不是 tests 下的 Python 文件: {raw}")
    return paths


def _collect(paths: tuple[str, ...]) -> tuple[int, frozenset[str]]:
    """通过 pytest 的真实收集边界返回数量和测试文件。"""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"pytest 收集失败: exit={result.returncode}")
    match = COLLECTED.search(result.stdout)
    if match is None:
        raise RuntimeError("无法从 pytest 输出读取收集数量")
    files = frozenset(
        line.split("::", 1)[0]
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )
    return int(match.group(1)), files


def main() -> int:
    """拒绝超出预算、隐藏在清单外或无效的测试。"""

    try:
        paths = _load_manifest()
        test_count, collected_files = _collect(("tests/",))
        manifest_files = frozenset(paths)
        if test_count != EXPECTED_TESTS:
            raise ValueError(
                f"Python 测试必须恰好为 {EXPECTED_TESTS} 项: actual={test_count}"
            )
        if collected_files != manifest_files:
            raise ValueError(
                "pytest 收集文件必须与保留清单完全一致: "
                f"unlisted={sorted(collected_files - manifest_files)} "
                f"missing={sorted(manifest_files - collected_files)}"
            )
        web_files = {
            str(path.relative_to(ROOT))
            for parent in ("frontend", "plugins", "scripts", "tests")
            for path in (ROOT / parent).rglob("*.test.mjs")
        }
        akasha_mobile_ui = ROOT / "tests" / "test_akasha_mobile_ui.mjs"
        if akasha_mobile_ui.is_file():
            web_files.add(str(akasha_mobile_ui.relative_to(ROOT)))
        if web_files != EXPECTED_WEB_FILES:
            raise ValueError(
                "Node 测试文件必须与保留清单完全一致: "
                f"unlisted={sorted(web_files - EXPECTED_WEB_FILES)} "
                f"missing={sorted(EXPECTED_WEB_FILES - web_files)}"
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"测试预算检查失败: {error}", file=sys.stderr)
        return 1

    print(
        f"测试预算检查通过: python_files={len(paths)} "
        f"python_tests={test_count} node_files={len(web_files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
