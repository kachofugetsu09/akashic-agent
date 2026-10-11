"""插件 Web 纯逻辑单测：运行 plugins/*/web 与 frontend/plugins/*/src 下的 *.test.ts。"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("plugins/*/web/**/*.test.ts", "frontend/plugins/*/src/**/*.test.ts")


def _test_files() -> list[str]:
    return sorted(str(path.relative_to(ROOT)) for pattern in PATTERNS for path in ROOT.glob(pattern))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_plugin_web_units_pass() -> None:
    files = _test_files()
    assert files, "no plugin web unit tests found"
    result = subprocess.run(
        ["node", "--test", *files], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
