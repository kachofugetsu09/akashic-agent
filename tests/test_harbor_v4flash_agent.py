"""Harbor benchmark 使用隔离 Prompt 包的真实恢复入口。"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark.harbor_v4flash.agent import (
    _build_gateway_command,
    _build_prompt_restore_command,
)


def test_harbor_prompt_restore_command_runs_installed_prompt(tmp_path: Path) -> None:
    """在隔离的已安装 Prompt 副本上执行 benchmark 的恢复命令。"""

    source_root = tmp_path / "installed-source"
    prompt_root = source_root / "plugins" / "prompt"
    repository_prompt = Path(__file__).resolve().parents[1] / "plugins" / "prompt"
    shutil.copytree(repository_prompt, prompt_root)
    workspace = tmp_path / "workspace"

    command = _build_prompt_restore_command(
        source_root=str(source_root),
        workspace=str(workspace),
        python_path=sys.executable,
    )
    result = subprocess.run(
        [
            "bash",
            "-ceu",
            f"mkdir -p {shlex.quote(str(workspace))} && {command}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (workspace / "memory" / "VEDA.md").read_bytes() == (
        repository_prompt / "VEDA.md"
    ).read_bytes().rstrip(b"\n") + b"\n"
    assert "main.py veda-reset" not in _build_gateway_command()
