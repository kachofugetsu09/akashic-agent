"""人格维护由外部 Prompt 包执行，并保留恢复前的原始字节。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

from plugins.prompt.persona import VedaLoadError, read_veda_file


def test_installed_prompt_maintenance_preserves_corrupt_original(tmp_path):
    source = Path(__file__).parents[1] / "plugins" / "prompt"
    installed = tmp_path / "installed-prompt"
    shutil.copytree(source, installed, ignore=shutil.ignore_patterns("__pycache__"))
    workspace = tmp_path / "workspace"
    target = workspace / "memory" / "VEDA.md"
    target.parent.mkdir(parents=True)
    original = b"private persona\xff\x00"
    target.write_bytes(original)

    with pytest.raises(VedaLoadError, match="UTF-8"):
        read_veda_file(target)
    assert target.read_bytes() == original

    result = subprocess.run(
        [sys.executable, str(installed / "persona.py"), "--workspace", str(workspace)],
        cwd=tmp_path, env=os.environ.copy(), text=True, capture_output=True, check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["changed"] is True
    assert Path(receipt["backup_path"]).read_bytes() == original
    assert target.read_text() == (installed / "VEDA.md").read_text().strip() + "\n"

    again = subprocess.run(
        [sys.executable, str(installed / "persona.py"), "--workspace", str(workspace)],
        cwd=tmp_path, env=os.environ.copy(), text=True, capture_output=True, check=True,
    )
    assert json.loads(again.stdout)["changed"] is False
    assert Path(receipt["backup_path"]).read_bytes() == original
