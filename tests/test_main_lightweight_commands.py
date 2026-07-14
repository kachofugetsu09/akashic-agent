from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_setup_main_does_not_import_agent_runtime(tmp_path: Path) -> None:
    """setup-main 应在完整 Agent runtime 依赖加载前完成分发。"""
    missing_config = tmp_path / "missing.toml"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "main.py"),
            "setup-main",
            "--config",
            str(missing_config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "配置文件不存在" in output
    assert "apscheduler" not in output
