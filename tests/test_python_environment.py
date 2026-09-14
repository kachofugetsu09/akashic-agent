import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from agent.plugins.python_environment import PythonEnvironments
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    materialize_command,
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
    input_data = record.get("input")
    assert isinstance(input_data, Mapping)
    code_ref = input_data.get("code")
    assert isinstance(code_ref, str)
    archived_code = store.archive.open(code_ref)
    root = store.open(ref, archived_code, manifest.python[0])
    command = materialize_command(
        archived_code, manifest.python, ("python", "probe.py"), environment_root=root
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
    input_data = store.archive.read_descriptor(ref).get("input")
    assert isinstance(input_data, Mapping)
    code_ref = input_data.get("code")
    assert isinstance(code_ref, str)
    archived_code = store.archive.open(code_ref)
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
    input_data = store.archive.read_descriptor(ref).get("input")
    assert isinstance(input_data, Mapping)
    code_ref = input_data.get("code")
    assert isinstance(code_ref, str)
    archived = store.archive.open(code_ref)
    shutil.rmtree(code)
    script = root / ".venv/bin/fixture-echo"
    assert (
        subprocess.run(
            [str(script)], check=True, capture_output=True, text=True
        ).stdout.strip()
        == "installed fixture"
    )
    assert store.open(ref, archived, manifest.python[0]) == root


@pytest.mark.parametrize("command,cwd,expected", [
    (("python", "probe.py"), ".", "."),
    (("python", "mcp/server.py"), ".", "mcp"),
    (("python", "-m", "server"), "mcp", "mcp"),
    (("python", "mcp/worker/server.py"), ".", "mcp/worker"),
])
def test_discovered_runtime_binds_nearest_fixed_interpreter(tmp_path, command, cwd, expected):
    """根与嵌套 runtime 共存时，命令仍绑定所属固定环境。"""
    code, _ = source(tmp_path)
    for relative in ("mcp", "mcp/worker"):
        directory = code / relative
        directory.mkdir()
        (directory / "requirements.txt").write_text("")
        (directory / "server.py").write_text("")
    manifest = load_static_plugin_manifest(code)
    environment = tmp_path / "fixed-environment"
    interpreter = environment / expected / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("staged interpreter; never execute in this test")
    interpreter.chmod(0o755)
    assert materialize_command(
        code, manifest.python, command, cwd, environment_root=environment
    ) == (str(interpreter), "-E", "-s", "-B", *command[1:])
    with pytest.raises(RuntimeError, match="显式运行环境"):
        materialize_command(code, manifest.python, command, cwd)
    interpreter.unlink()
    with pytest.raises(RuntimeError, match="尚未完成 staging"):
        materialize_command(code, manifest.python, command, cwd, environment_root=environment)


def test_requirements_discovery_skips_dependencies_and_optional_files(tmp_path):
    """缓存内依赖和不同名的可选清单不成为必装 runtime。"""
    code, _ = source(tmp_path)
    for name in (
        ".git", ".venv", "venv", "node_modules", "cache", ".cache",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ):
        directory = code / name
        directory.mkdir()
        (directory / "requirements.txt").symlink_to(tmp_path / "absent")
    (code / "requirements-dev.txt").write_text("optional-package")
    (code / "requirements-optional.txt").write_text("optional-package")
    assert load_static_plugin_manifest(code).requirements == ("requirements.txt",)
    (code / "requirements.txt").unlink()
    assert load_static_plugin_manifest(code).python == ()


@pytest.mark.parametrize("kind", ["directory", "file", "broken", "named_directory"])
def test_requirements_discovery_rejects_symlink_runtime_paths(tmp_path, kind):
    """链接不得把制品外或别名目录变成环境输入。"""
    code, _ = source(tmp_path)
    directory = code / "mcp"
    directory.mkdir()
    (directory / "requirements.txt").write_text("")
    if kind == "directory":
        (code / "alias").symlink_to(directory, target_is_directory=True)
    elif kind == "file":
        (directory / "requirements.txt").unlink()
        (directory / "requirements.txt").symlink_to(code / "requirements.txt")
    elif kind == "broken":
        (code / "alias").symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        (directory / "requirements.txt").unlink()
        (directory / "requirements.txt").mkdir()
    with pytest.raises(ValueError, match="符号链接|必须是文件"):
        load_static_plugin_manifest(code)


def test_manifest_rejects_removed_python_declarations(tmp_path):
    code, _ = source(tmp_path)
    path = code / "akashic.plugin.toml"
    path.write_text(path.read_text() + '\n[[python]]\nrequirements = "requirements.txt"\n')
    with pytest.raises(ValueError, match="未知字段.*python"):
        load_static_plugin_manifest(code)


def test_command_binding_uses_frozen_discovery_without_installing(tmp_path, monkeypatch):
    """绑定只消费解析结果；后加清单不改变 owner，也不触发安装。"""
    code, manifest = source(tmp_path)
    nested = code / "later"
    nested.mkdir()
    (nested / "requirements.txt").write_text("must-not-install")
    (nested / "server.py").write_text("")
    environment = tmp_path / "fixed"
    interpreter = environment / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("staged interpreter")
    interpreter.chmod(0o755)

    def forbidden(*args, **kwargs):
        raise AssertionError("binding must not install or discover")

    monkeypatch.setattr(PythonEnvironments, "prepare", forbidden)
    monkeypatch.setattr("agent.plugins.static_manifest._python_runtimes", forbidden)
    assert materialize_command(
        code, manifest.python, ("python", "later/server.py"), environment_root=environment
    )[0] == str(interpreter)


@pytest.mark.asyncio
async def test_source_loading_never_prepares_even_an_empty_environment(tmp_path, monkeypatch):
    """源码装配不安装环境；只有实际 Python 命令需要已安装的引用。"""
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus

    code, _ = source(tmp_path)
    (code / "plugin.py").write_text(
        "api_version = 3\nname = 'probe'\nversion = '1.0.0'\n"
        "async def apply(ctx):\n    return None\n"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("loading cannot prepare environments")

    monkeypatch.setattr(PythonEnvironments, "prepare", forbidden)
    host = PluginManager([code], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "empty-cache")
    try:
        await host.load_all()
        generation = host.generation("probe")
        assert generation is not None
        with pytest.raises(RuntimeError, match="缺少固定 Python 环境"):
            host._resolve_runtime_command(generation, ("python", "probe.py"), ".")
    finally:
        await host.terminate_all()
