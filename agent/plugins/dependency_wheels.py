"""部署前在目标镜像里按 Core 锁与其他插件的版本约束求离线 wheel；只接受有 wheel 的依赖。

宿主侧经 `docker run <目标镜像> python -m agent.plugins.dependency_wheels --work DIR` 调用（联网），
Python 小版本与平台标签因此与实际安装一致；停机窗口内只用这里产出的 wheel 离线安装。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

from agent.plugins.dependency_constraints import Constraints, constraints_from_pins, core_lock, explain_failure

_NO_MATCH = re.compile(r"No matching distribution found for ([A-Za-z0-9][A-Za-z0-9._-]*)")


class WheelResolutionError(RuntimeError):
    """依赖无法解析成 wheel（冲突、只有源码包、网络），在停机前失败。"""


def _download(requirements: Path, destination: Path, constraints: Constraints | None) -> None:
    command = [sys.executable, "-I", "-m", "pip", "--disable-pip-version-check", "download", "--only-binary=:all:",
               "--no-cache-dir", "--no-input", "--dest", str(destination), "-r", str(requirements)]
    if constraints is not None:
        command += [item for file in constraints.files for item in ("-c", str(file))]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    output = result.stdout + result.stderr
    conflict = None if constraints is None else explain_failure(output, constraints)
    if conflict is not None:
        raise WheelResolutionError(str(conflict))
    missing = _NO_MATCH.search(output)
    if missing is not None:
        raise WheelResolutionError(
            f"依赖 {missing.group(1)} 没有可用的 wheel（只有源码包或版本不满足约束）；部署不现编源码包，"
            "请换用有 wheel 的版本或先在 extra 仓库里放一个 wheel")
    raise WheelResolutionError("pip download 失败: " + " ".join(output.strip().splitlines()[-6:]))


# work 目录里：spec.json（plugin、inprocess、runtimes、others、origins）与 req/ 下的 requirements 文件；
# 产出 wheels/ 与 result.json。进程内依赖带约束，子进程 runtime 的依赖各自独立（同一个 wheel 目录）。
def resolve(work: Path) -> dict[str, object]:
    spec = cast(dict[str, object], json.loads((work / "spec.json").read_text(encoding="utf-8")))
    wheels = work / "wheels"
    wheels.mkdir(exist_ok=True)
    lock, _ = core_lock()
    with tempfile.TemporaryDirectory(prefix="akashic-wheels-") as scratch:
        inprocess = cast(str | None, spec["inprocess"])
        if inprocess is not None:
            constraints = constraints_from_pins(
                str(spec["plugin"]), Path(scratch), cast(dict[str, str], spec["others"]), cast(dict[str, str], spec["origins"]))
            _download(work / "req" / inprocess, wheels, constraints)
        for name in cast(list[str], spec["runtimes"]):
            _download(work / "req" / name, wheels, None)
    return {"core_lock_sha256": None if lock is None else hashlib.sha256(lock.read_bytes()).hexdigest(),
            "wheels": sorted(item.name for item in wheels.iterdir())}


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--work", type=Path, required=True)
    work = cast(Path, parser.parse_args().work)
    result: dict[str, object]
    try:
        result = resolve(work)
        code = 0
    except (RuntimeError, ValueError, OSError) as error:
        result = {"error": str(error)}
        code = 1
    _ = (work / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
