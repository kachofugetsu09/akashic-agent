from __future__ import annotations

import os
import json
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

import agent.plugins.install as install_module
import agent.plugins.source_resolver as source_resolver_module
from agent.plugins.artifacts import (
    ArtifactPointer,
    read_pointers,
    resolve_pointer,
    write_pointers,
)
from agent.plugins.install import (
    finalize_uninstall_plugin,
    install_git_plugin,
    set_installed_plugin_enabled,
)
from agent.plugins.manifest import plugins_root
from agent.plugins.python_environment import ENVIRONMENT_FILE, OfflineWheels, PythonEnvironments, wheel_tree_sha256
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    materialize_command,
)
from agent.plugins.source_resolver import resolve_plugin_sources, scan_plugin_sources
from tests.test_python_environment import write_test_wheel


def test_installed_pointer_loads_code_identity_without_toml(tmp_path: Path) -> None:
    """安装制品与源码采用同一入口规则，不额外要求空 TOML。"""
    artifact = tmp_path / ".artifacts" / "probe-version"
    artifact.mkdir(parents=True)
    entry = artifact / "plugin.py"
    entry.write_text('name = "probe"\nversion = "1.0"\napi_version = 3\n')
    pointer = ArtifactPointer(".artifacts/probe-version")
    assert resolve_pointer(tmp_path, pointer) == artifact
    entry.unlink()
    with pytest.raises(ValueError, match="plugin.py 必须是普通文件"):
        resolve_pointer(tmp_path, pointer)


def test_code_identity_is_read_without_execution_or_toml(tmp_path: Path) -> None:
    """身份不依赖目录名或导入，也不要求空策略文件。"""
    from types import ModuleType
    from agent.plugins.composable import ComposablePlugin

    (tmp_path / "plugin.py").write_text(
        'name: str = "probe"\nversion = "1.2.3"\napi_version = 3\n'
        'raise AssertionError("metadata discovery executed plugin")\n'
    )
    identity = load_static_plugin_manifest(tmp_path)
    assert (identity.name, identity.version, identity.api_version) == ("probe", "1.2.3", 3)
    assert not (tmp_path / "akashic.plugin.toml").exists()
    module = ModuleType("loaded_probe")
    module.apply = lambda ctx: None
    plugin = ComposablePlugin.from_module(module, identity)
    assert (plugin.name, plugin.version, plugin.api_version) == ("probe", "1.2.3", 3)


@pytest.mark.parametrize("declaration", [
    'name = "pro" + "be"',
    'from elsewhere import name',
    'if True:\n    name = "probe"',
    'name = "probe"\nname = "again"',
    'name = other = "probe"',
])
def test_code_identity_requires_one_direct_literal(tmp_path: Path, declaration: str) -> None:
    (tmp_path / "plugin.py").write_text(
        declaration + '\nversion = "1.0.0"\napi_version = 3\n'
    )
    with pytest.raises(ValueError, match="身份"):
        load_static_plugin_manifest(tmp_path)


@pytest.mark.parametrize("api", ["True", "2", '"3"'])
def test_code_identity_rejects_unsupported_api_before_import(tmp_path: Path, api: str) -> None:
    (tmp_path / "plugin.py").write_text(
        f'name = "probe"\nversion = "1.0.0"\napi_version = {api}\n'
        'raise AssertionError("must reject before import")\n'
    )
    with pytest.raises(ValueError, match="api_version"):
        load_static_plugin_manifest(tmp_path)


@pytest.mark.parametrize("declaration", [
    'name = "other"',
    'entrypoint = "nested/custom.py"',
    '[validation]\nexclude_data_paths = ["secret.txt"]',
    'invalid TOML [',
])
def test_plugin_identity_and_discovery_ignore_old_policy_file(tmp_path: Path, declaration: str) -> None:
    """旧策略不改变代码身份、摘要或入口发现，也不会被解析。"""
    _write_v3_plugin(tmp_path, name="probe")
    identity = load_static_plugin_manifest(tmp_path)
    (tmp_path / "akashic.plugin.toml").write_text(declaration + "\n")
    assert load_static_plugin_manifest(tmp_path) == identity
    [source] = resolve_plugin_sources([tmp_path])
    assert source.plugin_root == tmp_path.resolve()
    assert source.static_manifest == identity
    (tmp_path / "plugin.py").unlink()
    assert resolve_plugin_sources([tmp_path]) == []


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory"])
def test_manifest_requires_plain_root_plugin_file(tmp_path: Path, kind: str) -> None:
    """安装和发现不能把其他 Python 文件猜作入口。"""
    repo = tmp_path / "source"
    _write_v3_plugin(repo, name="probe")
    entry = repo / "plugin.py"
    entry.rename(repo / "custom.py")
    if kind == "symlink":
        entry.symlink_to(repo / "custom.py")
    elif kind == "directory":
        entry.mkdir()
    with pytest.raises(ValueError, match="plugin.py"):
        load_static_plugin_manifest(repo)
    if kind == "symlink":
        with pytest.raises(ValueError, match="plugin.py"):
            resolve_plugin_sources([repo])
    else:
        assert resolve_plugin_sources([repo]) == []


def test_tolerant_source_scan_reports_content_without_fake_plugin_id(
    tmp_path: Path,
) -> None:
    roots = tmp_path / "plugins"
    _write_v3_plugin(roots / "healthy", name="healthy")
    broken = roots / "broken"
    broken.mkdir(parents=True)
    (broken / "plugin.py").write_text("this is not Python !!!\n", encoding="utf-8")

    scan = scan_plugin_sources([roots])

    assert [source.plugin_name for source in scan.sources] == ["healthy"]
    assert len(scan.failures) == 1
    failure = scan.failures[0]
    assert failure.source_root == broken.resolve()
    assert failure.plugin_id is None


def test_tolerant_source_scan_preserves_decode_reason(tmp_path: Path) -> None:
    roots = tmp_path / "plugins"
    broken = roots / "decode-broken"
    broken.mkdir(parents=True)
    (broken / "plugin.py").write_bytes(b'name = "broken"\n\xff\n')

    scan = scan_plugin_sources([roots])

    assert scan.sources == ()
    failure = scan.failures[0]
    assert failure.error_type == "UnicodeDecodeError"
    assert "invalid start byte" in failure.error_text
    assert failure.phase == "identity"


def test_tolerant_source_scan_keeps_shared_identity_io_fail_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "plugins"
    _write_v3_plugin(roots / "healthy", name="healthy")

    def fail_identity(_root: Path) -> object:
        raise OSError("identity read failed")

    monkeypatch.setattr(
        source_resolver_module,
        "load_static_plugin_manifest",
        fail_identity,
    )
    with pytest.raises(OSError, match="identity read failed"):
        scan_plugin_sources([roots])


def test_installed_tolerant_scan_reads_only_selected_pointer_content(
    tmp_path: Path,
) -> None:
    """A single selected artifact is tolerant of invalid source content."""
    base = tmp_path / "cache" / "lab" / "installed_snapshot"
    stable = base / ".artifacts" / "1.0.0-stable"
    stable.mkdir(parents=True)
    (stable / "plugin.py").write_text(
        'name = "installed_snapshot"\nversion = "1.0.0"\napi_version = 3\n',
        encoding="utf-8",
    )
    (base / ".pointers.json").write_text(
        json.dumps({
            "stable": ".artifacts/1.0.0-stable",
            "latest": ".artifacts/1.0.0-stable",
        }),
        encoding="utf-8",
    )

    scan = scan_plugin_sources([], installed_cache_root=tmp_path / "cache")

    assert [source.plugin_root for source in scan.sources] == [stable.resolve()]
    assert scan.failures == ()
    resolved = resolve_plugin_sources([], installed_cache_root=tmp_path / "cache")
    assert [source.plugin_root for source in resolved] == [stable.resolve()]

    (stable / "plugin.py").write_text("this is not Python either !!!\n", encoding="utf-8")
    scan = scan_plugin_sources([], installed_cache_root=tmp_path / "cache")
    assert scan.sources == ()
    assert scan.failures[0].source_root == stable.resolve()
    assert scan.failures[0].error_type == "SyntaxError"
    assert "line" in scan.failures[0].error_text
    assert scan.failures[0].plugin_id is None


def test_installed_scan_rejects_unsettled_historical_pointers(tmp_path: Path) -> None:
    """A historical pending pair needs its update owner before a new scan."""
    base = tmp_path / "cache" / "lab" / "demo"
    base.mkdir(parents=True)
    for name in ("old", "new"):
        artifact = base / ".artifacts" / name
        artifact.mkdir(parents=True)
        (artifact / "plugin.py").write_text(
            'name = "demo"\nversion = "1.0.0"\napi_version = 3\n',
        )
    (base / ".pointers.json").write_text(json.dumps({
        "stable": ".artifacts/old", "latest": ".artifacts/new",
    }))
    with pytest.raises(RuntimeError, match="历史候选指针对"):
        scan_plugin_sources([], installed_cache_root=tmp_path / "cache")


@pytest.mark.parametrize("pointer_value", [
    "../outside",
    ".artifacts/missing",
])
def test_installed_tolerant_scan_keeps_pointer_boundary_strict(
    tmp_path: Path,
    pointer_value: str,
) -> None:
    base = tmp_path / "cache" / "lab" / "installed_snapshot"
    base.mkdir(parents=True)
    (base / ".pointers.json").write_text(
        json.dumps({"stable": pointer_value, "latest": pointer_value}),
        encoding="utf-8",
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        scan_plugin_sources([], installed_cache_root=tmp_path / "cache")


def test_installed_tolerant_scan_rejects_pointer_symlink_and_identity_mismatch(
    tmp_path: Path,
) -> None:
    base = tmp_path / "cache" / "lab" / "installed_snapshot"
    artifact = base / ".artifacts" / "1.0.0"
    artifact.mkdir(parents=True)
    (artifact / "plugin.py").write_text(
        'name = "other_name"\nversion = "1.0.0"\napi_version = 3\n',
        encoding="utf-8",
    )
    (base / ".pointers.json").write_text(
        json.dumps({"stable": ".artifacts/1.0.0", "latest": ".artifacts/1.0.0"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="name 不一致"):
        scan_plugin_sources([], installed_cache_root=tmp_path / "cache")

    symlink_base = tmp_path / "cache-symlink" / "lab" / "installed_snapshot"
    symlink_base.mkdir(parents=True)
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (symlink_base / ".artifacts").symlink_to(outside, target_is_directory=True)
    (symlink_base / ".pointers.json").write_text(
        json.dumps({"stable": ".artifacts/1.0.0", "latest": ".artifacts/1.0.0"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="符号链接"):
        scan_plugin_sources([], installed_cache_root=tmp_path / "cache-symlink")


def test_installed_scan_does_not_downgrade_when_selected_artifact_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "home" / "cache" / "lab" / "installed_snapshot"
    artifact = base / ".artifacts" / "1.0.0"
    artifact.mkdir(parents=True)
    (artifact / "plugin.py").write_text(
        'name = "installed_snapshot"\nversion = "1.0.0"\napi_version = 3\n',
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0")
    write_pointers(base, stable=pointer, latest=pointer)
    real_resolve = source_resolver_module.resolve_pointer

    def resolve_then_remove(
        plugin_base: Path,
        pointer: ArtifactPointer,
        *,
        validate_content: bool = True,
    ) -> Path | None:
        target = real_resolve(
            plugin_base, pointer, validate_content=validate_content,
        )
        if target is not None:
            import shutil
            shutil.rmtree(target)
        return target

    monkeypatch.setattr(source_resolver_module, "resolve_pointer", resolve_then_remove)
    with pytest.raises(FileNotFoundError):
        scan_plugin_sources([], installed_cache_root=tmp_path / "home" / "cache")


def test_plugins_root_honors_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "isolated-plugin-home"
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(target))

    assert plugins_root() == target


def test_plugins_root_rejects_blank_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", "   ")

    with pytest.raises(ValueError, match="不能为空"):
        plugins_root()


def test_install_git_plugin_uses_static_v3_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "feed-mcp"
    _write_v3_plugin(repo, name="feed")
    _commit(repo)

    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    data_dir = workspace / "plugin-data" / "feed-lab"
    data_dir.mkdir(parents=True)
    (data_dir / "state.json").write_text('{"keep":true}\n', encoding="utf-8")

    result = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    assert result.installed_path.parent == (
        home / "cache" / "lab" / "feed" / ".artifacts"
    )
    assert result.installed_path.name.startswith("1.0.0-")
    assert result.source_revision == _git_output(repo, "rev-parse", "HEAD")
    pointer_state = result.installed_path.parents[1] / ".pointers.json"
    assert pointer_state.is_file()
    assert not (pointer_state.parent / ".stable.json").exists()
    assert not (pointer_state.parent / ".latest.json").exists()
    assert (result.installed_path / "plugin.py").exists()
    assert not (result.installed_path / "akashic.plugin.toml").exists()
    assert load_static_plugin_manifest(result.installed_path).name == "feed"
    assert (result.data_path / "state.json").exists()
    manifest = tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8"))
    assert manifest == {"plugins": {"feed@lab": {"enabled": True}}}


def test_install_git_plugin_reads_static_v3_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "citation"
    _write_v3_plugin(
        repo,
        name="citation",
        version="2.0.0",
        module_source="name = 'citation'\nversion = '2.0.0'\napi_version = 3\nraise RuntimeError('must not import during install')\n",
    )
    _commit(repo)

    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    assert result.plugin_name == "citation"
    assert result.plugin_version == "2.0.0"
    assert (result.installed_path / "plugin.py").is_file()
    assert not (result.installed_path / "akashic.plugin.toml").exists()


def test_install_git_plugin_prepares_discovered_python_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp" / "run_mcp.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    _write_v3_plugin(repo, name="feed")
    _commit(repo)
    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    store = PythonEnvironments(tmp_path / "workspace")
    environment_data = json.loads((result.installed_path / ENVIRONMENT_FILE).read_text())
    assert isinstance(environment_data, Mapping)
    ref = environment_data.get("mcp")
    assert isinstance(ref, str)
    record = store.archive.read_descriptor(ref)
    input_data = record.get("input")
    assert isinstance(input_data, Mapping)
    code_ref = input_data.get("code")
    assert isinstance(code_ref, str)
    code = store.archive.open(code_ref)
    manifest = load_static_plugin_manifest(code)
    environment = store.open(ref, code, manifest.python[0])
    command = materialize_command(
        code, manifest.python, ("python", "mcp/run_mcp.py"),
        environment_root=environment,
    )
    assert (
        subprocess.run(
            command, cwd=code, check=True, capture_output=True, text=True
        ).stdout.strip()
        == "ok"
    )
    assert not (result.installed_path / "mcp" / ".venv").exists()
    # 模拟同 revision 的旧安装：只有 cache 内 .venv，没有新环境引用。
    (result.installed_path / ENVIRONMENT_FILE).unlink()
    old_python = result.installed_path / "mcp/.venv/bin/python"
    old_python.parent.mkdir(parents=True)
    old_python.write_text("old environment must not execute")
    reinstalled = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )
    assert reinstalled.installed_path != result.installed_path
    assert (
        json.loads((reinstalled.installed_path / ENVIRONMENT_FILE).read_text())["mcp"]
        == ref
    )
    assert old_python.read_text() == "old environment must not execute"
    assert (
        subprocess.run(
            command, cwd=code, check=True, capture_output=True, text=True
        ).stdout.strip()
        == "ok"
    )
    finalize_uninstall_plugin(
        "feed@lab",
        workspace=tmp_path / "workspace",
        plugins_home=tmp_path / "plugins-home",
    )
    assert store.open(ref, code, manifest.python[0]) == environment
    assert (
        subprocess.run(
            command, cwd=code, check=True, capture_output=True, text=True
        ).stdout.strip()
        == "ok"
    )


def test_install_git_plugin_offline_wheels_keep_previous_state_on_failure(tmp_path: Path) -> None:
    """Real Git staging changes artifact refs only after exact offline install."""

    repo = tmp_path / "plugin"
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp/requirements.txt").write_text("fixture-dep==1.0\n")
    (repo / "mcp/run.py").write_text("import fixture_dep; print(fixture_dep.VALUE)\n")
    _write_v3_plugin(repo, name="probe")
    _commit(repo)
    wheels = tmp_path / "wheels"
    wheel = write_test_wheel(wheels, "fixture_dep")
    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    data = workspace / "plugin-data/probe-lab"
    data.mkdir(parents=True)
    (data / "keep.txt").write_text("keep")

    def install():
        return install_git_plugin(
            workspace=workspace, source=str(repo), marketplace="lab",
            plugins_home=home,
            offline_wheels=OfflineWheels(wheels, wheel_tree_sha256(wheels)),
        )

    first = install()
    store = PythonEnvironments(workspace)
    first_ref = json.loads((first.installed_path / ENVIRONMENT_FILE).read_text())["mcp"]
    record = store.archive.read_descriptor(first_ref)
    record_input = record["input"]
    assert isinstance(record_input, Mapping)
    code_ref = record_input["code"]
    assert isinstance(code_ref, str)
    archived = store.archive.open(code_ref)
    manifest = load_static_plugin_manifest(archived)
    env = store.open(first_ref, archived, manifest.python[0])
    command = materialize_command(archived, manifest.python, ("python", "mcp/run.py"), environment_root=env)
    assert subprocess.run(command, cwd=archived, capture_output=True, text=True, check=True).stdout.strip() == "v1"
    pointer_path = home / "cache/lab/probe/.pointers.json"
    manifest_path = home / "manifest.toml"
    old_pointer = pointer_path.read_bytes()
    old_manifest = manifest_path.read_bytes()
    old_refs = sorted(item.name for item in store.path.glob("*.ref"))

    wheel.unlink()
    with pytest.raises(ValueError, match="不能为空"):
        install()
    assert pointer_path.read_bytes() == old_pointer
    assert manifest_path.read_bytes() == old_manifest
    assert (data / "keep.txt").read_text() == "keep"
    assert sorted(item.name for item in store.path.glob("*.ref")) == old_refs

    write_test_wheel(wheels, "fixture_echo")
    with pytest.raises(subprocess.CalledProcessError):
        install()
    assert pointer_path.read_bytes() == old_pointer
    assert manifest_path.read_bytes() == old_manifest
    assert (data / "keep.txt").read_text() == "keep"
    assert sorted(item.name for item in store.path.glob("*.ref")) == old_refs
    (wheels / "fixture_echo-1.0-py3-none-any.whl").unlink()

    write_test_wheel(wheels, "fixture_dep", payload="v2")
    second = install()
    second_ref = json.loads((second.installed_path / ENVIRONMENT_FILE).read_text())["mcp"]
    assert second_ref != first_ref
    assert second.installed_path != first.installed_path
    assert pointer_path.read_bytes() != old_pointer
    assert manifest_path.read_bytes() == old_manifest
    second_record = store.archive.read_descriptor(second_ref)
    second_input = second_record["input"]
    assert isinstance(second_input, Mapping)
    second_code_ref = second_input["code"]
    assert isinstance(second_code_ref, str)
    second_code = store.archive.open(second_code_ref)
    second_env = store.open(second_ref, second_code, load_static_plugin_manifest(second_code).python[0])
    second_command = materialize_command(
        second_code, load_static_plugin_manifest(second_code).python,
        ("python", "mcp/run.py"), environment_root=second_env,
    )
    assert subprocess.run(
        second_command, cwd=second_code, capture_output=True, text=True, check=True
    ).stdout.strip() == "v2"
    assert not (workspace / "runtime/plugin-stable.json").exists()
    assert (data / "keep.txt").read_text() == "keep"


def test_retry_reuses_artifact_and_fixed_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    _write_v3_plugin(repo, name="feed", marker="v1")
    _commit(repo)

    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    _ = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    plugin_path = repo / "plugin.py"
    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("v1", "v2"),
        encoding="utf-8",
    )
    _commit(repo)
    installed = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    retried = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    assert retried.installed_path == installed.installed_path
    assert (retried.installed_path / ENVIRONMENT_FILE).read_text() == (
        installed.installed_path / ENVIRONMENT_FILE
    ).read_text()


def test_plugin_enable_disable_and_uninstall_preserve_data(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    cache = home / "cache" / "github" / "fitbit" / "1.0.0"
    data = workspace / "plugin-data" / "fitbit-github"
    _write_v3_plugin(cache, name="fitbit")
    data.mkdir(parents=True)
    state = data / "sleep-model.bin"
    state.write_bytes(b"model")
    (home / "manifest.toml").write_text(
        '[plugins."fitbit@github"]\nenabled = true\n',
        encoding="utf-8",
    )

    set_installed_plugin_enabled(
        "fitbit@github",
        enabled=False,
        plugins_home=home,
    )
    manifest = tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8"))
    assert manifest["plugins"]["fitbit@github"]["enabled"] is False

    set_installed_plugin_enabled(
        "fitbit@github",
        enabled=True,
        plugins_home=home,
    )
    disabled_before_removal = False

    def wait_until_disabled(plugin_id: str) -> None:
        nonlocal disabled_before_removal
        current = tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8"))
        disabled_before_removal = (
            plugin_id == "fitbit@github"
            and current["plugins"][plugin_id]["enabled"] is False
            and cache.parent.exists()
            and state.exists()
        )

    set_installed_plugin_enabled(
        "fitbit@github",
        enabled=False,
        plugins_home=home,
    )
    wait_until_disabled("fitbit@github")
    removed_cache, retained_data = finalize_uninstall_plugin(
        "fitbit@github",
        workspace=workspace,
        plugins_home=home,
    )

    assert disabled_before_removal
    assert removed_cache == home / "cache" / "github" / "fitbit"
    assert not removed_cache.exists()
    assert retained_data == data
    assert state.read_bytes() == b"model"
    assert tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8")) == {
        "plugins": {}
    }


def test_plugin_management_rejects_non_installed_plugin_id(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    (home / "cache" / "github" / "fitbit").mkdir(parents=True)
    (home / "manifest.toml").parent.mkdir(parents=True, exist_ok=True)
    (home / "manifest.toml").write_text("", encoding="utf-8")

    for plugin_id in (
        "fitbit",
        "../fitbit@github",
        "fitbit@../github",
        "fitbit\ncorrupt@github",
    ):
        try:
            set_installed_plugin_enabled(
                plugin_id,
                enabled=False,
                plugins_home=home,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝插件 ID: {plugin_id}")


def test_install_failure_restores_previous_cache_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "feed"
    plugin_path = repo / "plugin.py"
    _write_v3_plugin(repo, name="feed", marker="old")
    _commit(repo)
    home = tmp_path / "plugins-home"
    first = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )
    old_content = (first.installed_path / "plugin.py").read_text(encoding="utf-8")

    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("'old'", "'new'"),
        encoding="utf-8",
    )
    _commit(repo)

    def fail_prepare(
        plugin_root: Path, static_manifest: object, *, workspace: Path
    ) -> None:
        resolved = resolve_plugin_sources(
            [],
            installed_cache_root=home / "cache",
        )
        assert len(resolved) == 1
        assert resolved[0].plugin_root == first.installed_path
        assert (first.installed_path / "plugin.py").read_text(
            encoding="utf-8"
        ) == old_content
        raise RuntimeError(f"prepare failed: {plugin_root}")

    monkeypatch.setattr(install_module, "_prepare_static_python_runtimes", fail_prepare)
    with pytest.raises(RuntimeError, match="prepare failed"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )

    assert (first.installed_path / "plugin.py").read_text(
        encoding="utf-8"
    ) == old_content
    plugin_base = home / "cache" / "lab" / "feed"
    pointers = read_pointers(plugin_base)
    assert pointers is not None and pointers.stable == pointers.latest
    assert tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8")) == {
        "plugins": {"feed@lab": {"enabled": True}}
    }
    assert not any(
        child.name.startswith(".feed-install-")
        for child in (home / "cache" / "lab").iterdir()
    )

    monkeypatch.undo()

    def fail_manifest(*args, **kwargs) -> Path:
        raise OSError("manifest write failed")

    monkeypatch.setattr(install_module, "upsert_plugin_manifest", fail_manifest)
    with pytest.raises(OSError, match="manifest write failed"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )
    assert (first.installed_path / "plugin.py").read_text(
        encoding="utf-8"
    ) == old_content


def test_install_rejects_unsafe_path_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="../outside")
    _commit(repo)

    with pytest.raises(ValueError, match="安全的单一路径段"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="../outside",
            plugins_home=tmp_path / "plugins-home",
        )

    with pytest.raises(ValueError, match="静态 manifest name 无效"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )






def test_default_update_keeps_immediate_stable_compatibility(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    plugin_path = repo / "plugin.py"
    _write_v3_plugin(repo, name="feed", marker="v1")
    _commit(repo)
    home = tmp_path / "plugins-home"
    first = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )
    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("v1", "v2"),
        encoding="utf-8",
    )
    _commit(repo)

    second = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    resolved = resolve_plugin_sources([], installed_cache_root=home / "cache")
    assert resolved[0].plugin_root == second.installed_path
    assert first.installed_path.exists()


def test_install_rejects_visible_nonversion_cache_entry(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="feed")
    _commit(repo)
    home = tmp_path / "plugins-home"
    invalid_entry = home / "cache" / "lab" / "feed" / "unexpected.txt"
    invalid_entry.parent.mkdir(parents=True)
    invalid_entry.write_text("broken", encoding="utf-8")

    with pytest.raises(ValueError, match="cache 版本不是目录"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )

    assert invalid_entry.read_text(encoding="utf-8") == "broken"
    assert not (invalid_entry.parent / "1.0.0").exists()


def test_install_rejects_legacy_visible_version_directory(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="feed")
    _commit(repo)
    home = tmp_path / "plugins-home"
    legacy = home / "cache/lab/feed/1.0.0"
    legacy.mkdir(parents=True)
    (legacy / "state.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="不受支持的旧版可见目录"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )

    assert (legacy / "state.txt").read_text(encoding="utf-8") == "keep"
    assert not (legacy.parent / ".pointers.json").exists()


def test_install_allows_internal_source_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="feed")
    (repo / "helper.py").write_text("MARKER = 'inside'\n", encoding="utf-8")
    (repo / "linked_helper.py").symlink_to("helper.py")
    _commit(repo)

    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    linked_helper = result.installed_path / "linked_helper.py"
    assert linked_helper.read_text(encoding="utf-8") == "MARKER = 'inside'\n"
    assert not linked_helper.is_symlink()


def test_install_rejects_source_symlink_escape(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    outside = tmp_path / "outside.py"
    outside.write_text("MARKER = 'outside'\n", encoding="utf-8")
    _write_v3_plugin(repo, name="feed")
    (repo / "linked_helper.py").symlink_to(outside)
    _commit(repo)

    with pytest.raises(ValueError, match="符号链接越界"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )


def test_install_accepts_branch_tag_and_commit_refs(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    plugin_path = repo / "plugin.py"
    _write_v3_plugin(repo, name="feed", marker="initial")
    _commit(repo)
    initial_sha = _git_output(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "release")
    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("initial", "head"),
        encoding="utf-8",
    )
    _commit(repo)
    _git(repo, "tag", "v2")

    for ref_name, marker in (
        ("release", "initial"),
        ("v2", "head"),
        (initial_sha, "initial"),
    ):
        result = install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            ref_name=ref_name,
            plugins_home=tmp_path / f"home-{ref_name}",
        )
        assert f"marker = '{marker}'" in (
            result.installed_path / "plugin.py"
        ).read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="命令选项"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            ref_name="-bad",
            plugins_home=tmp_path / "home-option",
        )
    with pytest.raises(ValueError, match="首尾空白"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            ref_name=" release",
            plugins_home=tmp_path / "home-whitespace",
        )


def test_uninstall_converges_when_cache_is_already_missing(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    data = workspace / "plugin-data" / "feed-github"
    data.mkdir(parents=True)
    state = data / "state.db"
    state.write_bytes(b"keep")
    home.mkdir(parents=True)
    (home / "manifest.toml").write_text(
        '[plugins."feed@github"]\nenabled = true\n',
        encoding="utf-8",
    )

    set_installed_plugin_enabled(
        "feed@github",
        enabled=False,
        plugins_home=home,
    )
    cache, retained = finalize_uninstall_plugin(
        "feed@github",
        workspace=workspace,
        plugins_home=home,
    )

    assert not cache.exists()
    assert retained == data
    assert state.read_bytes() == b"keep"
    assert tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8")) == {
        "plugins": {}
    }


def _write_v3_plugin(
    root: Path,
    *,
    name: str,
    version: str = "1.0.0",
    marker: str | None = None,
    module_source: str | None = None,
) -> None:
    """Create a static v3 artifact fixture with a matching entrypoint."""

    # 1. Write an import-free entrypoint whose optional marker tracks source refs.
    root.mkdir(parents=True, exist_ok=True)
    if module_source is None:
        lines = [
            "api_version = 3",
            f"name = {name!r}",
            f"version = {version!r}",
        ]
        if marker is not None:
            lines.append(f"marker = {marker!r}")
        module_source = "\n".join(lines) + "\n"
    (root / "plugin.py").write_text(module_source, encoding="utf-8")



def _commit(repo: Path) -> None:
    for args in (
        ["init"],
        ["config", "user.name", "test"],
        ["config", "user.email", "test@example.com"],
        ["add", "."],
        ["commit", "-m", "init"],
    ):
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr


def _git(repo: Path, *args: str) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()
