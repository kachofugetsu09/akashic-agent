#!/usr/bin/env python3
"""验证普通 PR 回归清单完整、唯一且保持全量测试的三分之一。"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests_scenarios" / "contracts" / "pr-regression-files.txt"
COLLECTED = re.compile(r"(?m)^(\d+) tests? collected(?: in .*)?$")


def _load_manifest() -> tuple[str, ...]:
    """加载并验证只指向仓库 tests 目录的唯一 Python 测试文件。"""

    paths = tuple(
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not paths:
        raise ValueError("PR 回归清单不能为空")
    if len(paths) != len(set(paths)):
        raise ValueError("PR 回归清单包含重复路径")

    tests_root = (ROOT / "tests").resolve()
    for raw in paths:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"PR 回归路径必须是仓库内相对路径: {raw}")
        path = (ROOT / relative).resolve()
        if path.suffix != ".py" or not path.is_relative_to(tests_root) or not path.is_file():
            raise ValueError(f"PR 回归路径不是 tests 下的 Python 文件: {raw}")
    return paths


def _collect(paths: tuple[str, ...]) -> int:
    """通过 pytest 的真实收集边界计数，不从源码形状推测测试数。"""

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
    return int(match.group(1))


def main() -> int:
    """拒绝无效清单或偏离当前全量三分之一的回归预算。"""

    try:
        paths = _load_manifest()
        full_count = _collect(("tests/",))
        selected_count = _collect(paths)
        expected_count = (full_count + 2) // 3
        if selected_count != expected_count:
            raise ValueError(
                "PR 回归数量必须等于完整集向上取整后的三分之一: "
                f"selected={selected_count} expected={expected_count} full={full_count}"
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"PR 回归预算检查失败: {error}", file=sys.stderr)
        return 1

    print(
        "PR 回归预算检查通过: "
        f"files={len(paths)} selected={selected_count} full={full_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
