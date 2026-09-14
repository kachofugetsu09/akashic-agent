"""人格维护由外部 Prompt 包执行，并保留恢复前的原始字节。"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from plugins.prompt.persona import (
    VedaLoadError,
    initialize_veda_if_missing,
    read_veda_file,
    veda_path,
)


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
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["changed"] is True
    assert Path(receipt["backup_path"]).read_bytes() == original
    assert target.read_text() == (installed / "VEDA.md").read_text().strip() + "\n"

    again = subprocess.run(
        [sys.executable, str(installed / "persona.py"), "--workspace", str(workspace)],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(again.stdout)["changed"] is False
    assert Path(receipt["backup_path"]).read_bytes() == original


def test_prompt_setup_creates_only_missing_veda(tmp_path):
    workspace = tmp_path / "workspace"

    result = initialize_veda_if_missing(workspace)

    target = veda_path(workspace)
    assert result.changed is True
    assert (
        target.read_bytes()
        == (Path(__file__).parents[1] / "plugins" / "prompt" / "VEDA.md").read_bytes()
    )
    before = target.read_bytes()
    again = initialize_veda_if_missing(workspace)
    assert again.changed is False
    assert target.read_bytes() == before


def test_prompt_read_missing_is_read_only(tmp_path):
    target = veda_path(tmp_path / "workspace")

    with pytest.raises(VedaLoadError, match="缺少 Veda"):
        read_veda_file(target)

    assert not target.exists()


def test_prompt_setup_preserves_existing_custom_bytes(tmp_path):
    workspace = tmp_path / "workspace"
    target = veda_path(workspace)
    target.parent.mkdir(parents=True)
    original = b"custom persona\nwith trailing bytes\x00"
    target.write_bytes(original)

    result = initialize_veda_if_missing(workspace)

    assert result.changed is False
    assert target.read_bytes() == original


@pytest.mark.parametrize("payload", [b"", b"broken\xff"])
def test_prompt_setup_rejects_invalid_existing_veda_without_writing(tmp_path, payload):
    workspace = tmp_path / "workspace"
    target = veda_path(workspace)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    with pytest.raises(VedaLoadError):
        initialize_veda_if_missing(workspace)

    assert target.read_bytes() == payload


def test_prompt_setup_reports_existing_io_failure_without_writing(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    target = veda_path(workspace)
    original_read_bytes = Path.read_bytes

    def fail_target(path: Path) -> bytes:
        if path == target:
            raise PermissionError("test permission failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_target)
    with pytest.raises(VedaLoadError, match="读取 Veda 失败"):
        initialize_veda_if_missing(workspace)
    assert not target.exists()


def test_prompt_setup_concurrent_create_publishes_one_complete_file(tmp_path):
    workspace = tmp_path / "workspace"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: initialize_veda_if_missing(workspace), range(2))
        )

    assert sorted(result.changed for result in results) == [False, True]
    target = veda_path(workspace)
    assert (
        target.read_bytes()
        == (Path(__file__).parents[1] / "plugins" / "prompt" / "VEDA.md").read_bytes()
    )


def test_installed_prompt_setup_entrypoint_is_standalone(tmp_path):
    source = Path(__file__).parents[1] / "plugins" / "prompt"
    installed = tmp_path / "installed-prompt"
    shutil.copytree(source, installed, ignore=shutil.ignore_patterns("__pycache__"))
    workspace = tmp_path / "workspace"
    environment = os.environ.copy()
    environment["AKASHIC_SETUP_WORKSPACE"] = str(workspace)

    result = subprocess.run(
        [sys.executable, str(installed / "configure.py")],
        cwd=installed,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    receipt = json.loads(result.stdout)
    assert receipt["changed"] is True
    assert veda_path(workspace).read_bytes() == (installed / "VEDA.md").read_bytes()


def test_installed_prompt_setup_fails_loud_on_empty_existing_veda(tmp_path):
    source = Path(__file__).parents[1] / "plugins" / "prompt"
    installed = tmp_path / "installed-prompt"
    shutil.copytree(source, installed, ignore=shutil.ignore_patterns("__pycache__"))
    workspace = tmp_path / "workspace"
    target = veda_path(workspace)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    environment = os.environ.copy()
    environment["AKASHIC_SETUP_WORKSPACE"] = str(workspace)

    result = subprocess.run(
        [sys.executable, str(installed / "configure.py")],
        cwd=installed,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Veda 内容为空" in result.stderr
    assert target.read_bytes() == b""
