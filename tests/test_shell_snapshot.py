from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from plugins.host_execution.shell_runtime import build_shell_snapshot, resolve_shell

# bash-completion 的同类写法：函数体里的模式只在 extglob 打开时才能解析。
_BASH_PROFILE = """shopt -s extglob
probe() { case "$1" in --!(no-*)dir*) echo extglob-ok ;; esac; }
alias greet='echo alias-ok'
"""
_ZSH_RC = """setopt kshglob
probe() { case $1 in @(a|b)) echo kshglob-ok ;; esac }
alias greet='echo alias-ok'
"""


def _run_with_snapshot(kind: str, rc_name: str, rc_body: str, home: Path, command: str) -> subprocess.CompletedProcess[str]:
    shell = resolve_shell(kind)
    _ = (home / rc_name).write_text(rc_body)
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "TERM": "dumb"}
    snapshot = build_shell_snapshot(shell, env)
    assert snapshot is not None
    argv = shell.derive_argv(command, login=True, snapshot=snapshot)
    return subprocess.run(argv, env=env, cwd=home, capture_output=True, text=True, timeout=30, check=False)


def test_bash_snapshot_restores_options_before_functions(tmp_path: Path) -> None:
    result = _run_with_snapshot("bash", ".bash_profile", _BASH_PROFILE, tmp_path, "probe --dir; greet")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["extglob-ok", "alias-ok"]


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh 未安装")
def test_zsh_snapshot_restores_options_before_functions(tmp_path: Path) -> None:
    result = _run_with_snapshot("zsh", ".zshrc", _ZSH_RC, tmp_path, "probe a; greet")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["kshglob-ok", "alias-ok"]
