import shutil
import subprocess
from pathlib import Path

import pytest

from agent.plugins.python_environment import PythonEnvironments
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    materialize_static_command,
)


def source(tmp_path):
    code = tmp_path / "source"
    code.mkdir()
    (code / "plugin.py").write_text("name = 'probe'\n")
    (code / "probe.py").write_text("import sys; print(sys.prefix)\n")
    (code / "requirements.txt").write_text("")
    (code / "akashic.plugin.toml").write_text("""schema_version = 1
name = "probe"
version = "1.0.0"
api_version = 3
entrypoint = "plugin.py"
[[python]]
requirements = "requirements.txt"
[[mcp]]
name = "probe"
command = ["python", "probe.py"]
""")
    return code, load_static_plugin_manifest(code)


def test_final_environment_survives_cache_removal_and_rejects_damage(
    tmp_path, monkeypatch
):
    code, manifest = source(tmp_path)
    store = PythonEnvironments(tmp_path / "workspace")
    ref = store.prepare(code, manifest.python[0])
    assert store.prepare(code, manifest.python[0]) == ref
    record = store.archive.read_descriptor(ref)
    archived_code = store.archive.open(record["input"]["code"])
    root = store.open(ref, archived_code, manifest.python[0])
    command = materialize_static_command(
        archived_code, manifest, manifest.mcp_servers[0], environment_root=root
    )
    shutil.rmtree(code)
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text("raise RuntimeError('host pollution')\n")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    monkeypatch.setenv("PYTHONHOME", str(poison))
    result = subprocess.run(
        command, cwd=archived_code, text=True, capture_output=True, check=True
    )
    assert Path(result.stdout.strip()) == root / ".venv"
    assert store.open(ref, archived_code, manifest.python[0]) == root
    (root / ".venv" / "unexpected.py").write_text("changed")
    with pytest.raises(RuntimeError, match="损坏"):
        store.open(ref, archived_code, manifest.python[0])
    with pytest.raises(RuntimeError, match="损坏"):
        store.prepare(archived_code, manifest.python[0])


def test_environment_rejects_different_code_and_absent_recovery_material(tmp_path):
    code, manifest = source(tmp_path)
    store = PythonEnvironments(tmp_path / "workspace")
    ref = store.prepare(code, manifest.python[0])
    root = store.open(ref, code, manifest.python[0])
    (code / "probe.py").write_text("print('new')\n")
    with pytest.raises(RuntimeError, match="安装输入"):
        store.open(ref, code, manifest.python[0])
    archived_code = store.archive.open(
        store.archive.read_descriptor(ref)["input"]["code"]
    )
    shutil.rmtree(root)
    with pytest.raises(ValueError, match="实际目录"):
        store.open(ref, archived_code, manifest.python[0])


def test_environment_keeps_real_console_script_prefix_and_local_wheel(tmp_path):
    from zipfile import ZipFile

    code, manifest = source(tmp_path)
    vendor = code / "vendor"
    vendor.mkdir()
    with ZipFile(vendor / "fixture_echo-1.0-py3-none-any.whl", "w") as wheel:
        files = {
            "fixture_echo.py": "def main():\n    print('installed fixture')\n",
            "fixture_echo-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: fixture-echo\nVersion: 1.0\n",
            "fixture_echo-1.0.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            "fixture_echo-1.0.dist-info/entry_points.txt": "[console_scripts]\nfixture-echo = fixture_echo:main\n",
        }
        for name, value in files.items():
            wheel.writestr(name, value)
        wheel.writestr(
            "fixture_echo-1.0.dist-info/RECORD",
            "".join(name + ",,\n" for name in files)
            + "fixture_echo-1.0.dist-info/RECORD,,\n",
        )
    (code / "requirements.txt").write_text(
        "--no-index\n./vendor/fixture_echo-1.0-py3-none-any.whl\n"
    )
    store = PythonEnvironments(tmp_path / "workspace")
    ref = store.prepare(code, manifest.python[0])
    root = store.open(ref, code, manifest.python[0])
    archived = store.archive.open(store.archive.read_descriptor(ref)["input"]["code"])
    shutil.rmtree(code)
    script = root / ".venv/bin/fixture-echo"
    assert (
        subprocess.run(
            [str(script)], check=True, capture_output=True, text=True
        ).stdout.strip()
        == "installed fixture"
    )
    assert store.open(ref, archived, manifest.python[0]) == root
