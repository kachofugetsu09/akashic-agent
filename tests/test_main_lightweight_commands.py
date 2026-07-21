from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).parents[1]


def test_setup_main_does_not_import_agent_runtime(tmp_path: Path) -> None:
    """setup-main 应在完整 Agent runtime 依赖加载前完成分发。"""
    missing_config = tmp_path / "missing.toml"

    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "setup-main",
            "--config",
            str(missing_config),
            "--workspace",
            str(tmp_path / "workspace"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "配置文件不存在" in output
    assert "apscheduler" not in output


def test_init_marks_fresh_installation_at_current_head(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "init",
            "--config",
            str(config_path),
            "--workspace",
            str(workspace),
        ],
        cwd=_PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    head = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    cursor = config_path.with_name("config.toml.migration-cursor")
    assert cursor.read_text(encoding="ascii").strip() == head
