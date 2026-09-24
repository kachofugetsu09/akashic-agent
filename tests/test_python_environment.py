import shutil
import subprocess
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from zipfile import ZipFile

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

import agent.plugins.python_environment as python_environment_module
from agent.plugins.python_environment import OfflineWheels, PythonEnvironments, wheel_tree_sha256
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    materialize_command,
)


def source(tmp_path):
    code = tmp_path / "source"
    code.mkdir()
    (code / "plugin.py").write_text("name = 'probe'\nversion = '1.0.0'\napi_version = 3\n")
    (code / "probe.py").write_text("import sys; print(sys.prefix)\n")
    (code / "requirements.txt").write_text("")
    return code, load_static_plugin_manifest(code)


def write_test_wheel(directory: Path, name: str, *, payload: str = "v1") -> Path:
    """Build a real, tiny wheel with a console script and optional dependency."""

    directory.mkdir(parents=True, exist_ok=True)
    dist = name.replace("_", "-")
    path = directory / f"{name}-1.0-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0\n"
    if name == "fixture_echo":
        metadata += 'Provides-Extra: extra\nRequires-Dist: fixture-dep>=1; extra == "extra"\n'
        module = "def main():\n    import fixture_dep\n    print('installed ' + fixture_dep.VALUE)\n"
    else:
        module = f"VALUE = {payload!r}\n"
    files = {
        f"{name}.py": module,
        f"{name}-1.0.dist-info/METADATA": metadata,
        f"{name}-1.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    if name == "fixture_echo":
        files[f"{name}-1.0.dist-info/entry_points.txt"] = (
            "[console_scripts]\nfixture-echo = fixture_echo:main\n"
        )
    record = f"{name}-1.0.dist-info/RECORD"
    with ZipFile(path, "w") as wheel:
        for relative, content in files.items():
            wheel.writestr(relative, content)
        wheel.writestr(record, "".join(f"{relative},,\n" for relative in files) + f"{record},,\n")
    return path


def test_offline_wheels_install_transitive_and_keep_final_script(tmp_path: Path, monkeypatch):
    code, manifest = source(tmp_path)
    (code / "requirements.txt").write_text(
        'fixture-echo[extra]==1.0; python_version >= "3.0"\n', encoding="utf-8"
    )
    wheels = tmp_path / "wheels"
    write_test_wheel(wheels, "fixture_echo")
    write_test_wheel(wheels, "fixture_dep")
    names = sorted(wheels.iterdir(), key=lambda item: item.name.encode("utf-8"))
    exact = b"".join(
        item.name.encode() + b"\0" + hashlib.sha256(item.read_bytes()).hexdigest().encode() + b"\n"
        for item in names
    )
    digest = hashlib.sha256(exact).hexdigest()
    assert wheel_tree_sha256(wheels) == digest
    offline = OfflineWheels(wheels, digest)
    monkeypatch.setenv("PIP_INDEX_URL", "https://invalid.example.test/simple")
    store = PythonEnvironments(tmp_path / "workspace")
    ref = store.prepare(code, manifest.python[0], offline_wheels=offline)
    assert store.prepare(code, manifest.python[0], offline_wheels=offline) == ref
    record = store.archive.read_descriptor(ref)
    assert record["input"]["wheel_tree_sha256"] == digest
    root = store.open(ref, code, manifest.python[0])
    script = root / ".venv/bin/fixture-echo"
    assert f"{root}/.venv/bin/python" in script.read_text()
    assert subprocess.run([str(script)], capture_output=True, text=True, check=True).stdout.strip() == "installed v1"
    write_test_wheel(wheels, "fixture_dep", payload="v2")
    changed = OfflineWheels(wheels, wheel_tree_sha256(wheels))
    assert changed.tree_sha256 != digest
    new_ref = store.prepare(code, manifest.python[0], offline_wheels=changed)
    assert new_ref != ref
    assert subprocess.run(
        [str(store.open(new_ref, code, manifest.python[0]) / ".venv/bin/fixture-echo")],
        capture_output=True, text=True, check=True,
    ).stdout.strip() == "installed v2"
    shutil.rmtree(wheels)
    assert store.open(ref, code, manifest.python[0]) == root
    assert store.open(new_ref, code, manifest.python[0]).is_dir()


@pytest.mark.parametrize("requirements", [
    "fixture-echo==1.0\n-r other.txt\n",
    "-c constraints.txt\nfixture-echo==1.0\n",
    "--extra-index-url https://example.test\nfixture-echo==1.0\n",
    "./fixture_echo-1.0-py3-none-any.whl\n",
    "fixture-echo @ https://example.test/wheel.whl\n",
    "-e git+https://example.test/repo#egg=fixture-echo\n",
    "fixture-echo.tar.gz\n",
    "probe.py\n",
])
def test_offline_requirements_reject_unsupported_inputs(tmp_path: Path, requirements: str):
    code, manifest = source(tmp_path)
    (code / "requirements.txt").write_text(requirements, encoding="utf-8")
    wheels = tmp_path / "wheels"
    write_test_wheel(wheels, "fixture_echo")
    store = PythonEnvironments(tmp_path / "workspace")
    with pytest.raises(ValueError, match="离线 requirements"):
        store.prepare(code, manifest.python[0], offline_wheels=OfflineWheels(wheels, wheel_tree_sha256(wheels)))
    assert not store.path.exists()


def test_offline_wheel_shape_digest_and_missing_dependency_fail_closed(tmp_path: Path):
    code, manifest = source(tmp_path)
    (code / "requirements.txt").write_text("fixture-echo[extra]==1.0\n", encoding="utf-8")
    wheels = tmp_path / "wheels"
    wheel = write_test_wheel(wheels, "fixture_echo")
    store = PythonEnvironments(tmp_path / "workspace")
    digest = wheel_tree_sha256(wheels)
    with pytest.raises(subprocess.CalledProcessError):
        store.prepare(code, manifest.python[0], offline_wheels=OfflineWheels(wheels, digest))
    assert not list(store.path.glob("*.ref"))
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="摘要不匹配"):
        store.prepare(code, manifest.python[0], offline_wheels=OfflineWheels(wheels, digest))
    (wheels / "junk.txt").write_text("junk")
    with pytest.raises(ValueError, match="普通 .whl"):
        wheel_tree_sha256(wheels)
    (wheels / "junk.txt").unlink()
    (wheels / "link.whl").symlink_to(wheel)
    with pytest.raises(ValueError, match="普通 .whl"):
        wheel_tree_sha256(wheels)
    (wheels / "link.whl").unlink()
    alias = tmp_path / "alias"
    alias.symlink_to(wheels, target_is_directory=True)
    with pytest.raises(ValueError, match="普通目录"):
        wheel_tree_sha256(alias)


def test_offline_wheel_drift_after_real_pip_does_not_publish(tmp_path: Path, monkeypatch):
    code, manifest = source(tmp_path)
    (code / "requirements.txt").write_text("fixture-dep==1.0\n", encoding="utf-8")
    wheels = tmp_path / "wheels"
    wheel = write_test_wheel(wheels, "fixture_dep")
    offline = OfflineWheels(wheels, wheel_tree_sha256(wheels))
    original = python_environment_module._run

    def change_after_pip(command: list[str], cwd: Path) -> None:
        original(command, cwd)
        if "--no-index" in command:
            wheel.write_bytes(wheel.read_bytes() + b"changed")

    monkeypatch.setattr(python_environment_module, "_run", change_after_pip)
    store = PythonEnvironments(tmp_path / "workspace")
    with pytest.raises(ValueError, match="摘要不匹配"):
        store.prepare(code, manifest.python[0], offline_wheels=offline)
    assert not list(store.path.glob("*.ref"))


def test_open_checks_optional_wheel_digest_but_not_wheel_source(tmp_path: Path):
    code, manifest = source(tmp_path)
    store = PythonEnvironments(tmp_path / "workspace")
    ref = store.prepare(code, manifest.python[0])
    record = store.archive.read_descriptor(ref)
    assert "wheel_tree_sha256" not in record["input"]
    assert store.open(ref, code, manifest.python[0]).is_dir()
    broken = json.loads(json.dumps(record, default=lambda value: dict(value)))
    broken["input"]["wheel_tree_sha256"] = "bad"
    broken_ref = store.archive.save_descriptor(broken)
    with pytest.raises(ValueError, match="wheel 摘要"):
        store.open(broken_ref, code, manifest.python[0])
    broken["input"]["wheel_tree_sha256"] = None
    null_ref = store.archive.save_descriptor(broken)
    with pytest.raises(ValueError, match="wheel 摘要"):
        store.open(null_ref, code, manifest.python[0])


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


def test_python_inputs_come_from_requirements_not_old_policy(tmp_path):
    """旧 TOML 不能增减实际 requirements 输入或改变其身份摘要。"""
    code, original = source(tmp_path)
    path = code / "akashic.plugin.toml"
    path.write_text('\n[[python]]\nrequirements = "missing.txt"\n')
    assert load_static_plugin_manifest(code) == original
    (code / "requirements.txt").unlink()
    assert load_static_plugin_manifest(code).python == ()


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
    initialize_plugin_workspace(tmp_path / "workspace")
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
