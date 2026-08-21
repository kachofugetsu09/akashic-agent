from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import agent.plugins.install as install_module
import agent.plugins.manager as manager_module
from agent.plugins.generation import PluginGeneration
from agent.plugins.snapshot import RuntimeSnapshot
from agent.plugins.install import install_git_plugin
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    materialize_static_command,
    staged_python_interpreter,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _commit(root: Path) -> None:
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "fixture")


def _manifest(
    *,
    name: str = "calendar",
    version: str = "3.0.0",
    entrypoint: str = "plugin.py",
    requirements: str = "mcp/requirements.txt",
) -> str:
    return (
        "schema_version = 1\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        "api_version = 3\n"
        f'entrypoint = "{entrypoint}"\n\n'
        "[[python]]\n"
        f'requirements = "{requirements}"\n\n'
        "[validation]\n"
        'exclude_data_paths = [".env", "token.json"]\n'
    )


def test_static_manifest_is_import_free_and_exposes_runtime_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calendar"
    (root / "mcp").mkdir(parents=True)
    (root / "plugin.py").write_text(
        "raise RuntimeError('must not import during static parse')\n",
        encoding="utf-8",
    )
    (root / "mcp" / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(_manifest(), encoding="utf-8")

    manifest = load_static_plugin_manifest(root)

    assert manifest.name == "calendar"
    assert manifest.version == "3.0.0"
    assert manifest.entrypoint == "plugin.py"
    assert manifest.requirements == ("mcp/requirements.txt",)
    assert manifest.python[0].runtime_root == "mcp"
    assert manifest.exclude_data_paths == (".env", "token.json")
    assert len(manifest.identity_digest) == 64


def test_static_manifest_freezes_channel_credential_paths_before_import(
    tmp_path: Path,
) -> None:
    root = tmp_path / "feishu"
    root.mkdir()
    (root / "plugin.py").write_text(
        "raise RuntimeError('candidate must not import during static parse')\n",
        encoding="utf-8",
    )
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'feishu'\n"
        "version = '3.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[channel_credentials]\n"
        "feishu = ['app_secret', 'oauth.token']\n",
        encoding="utf-8",
    )

    manifest = load_static_plugin_manifest(root)

    assert manifest.channel_credentials == (
        ("feishu", ("app_secret", "oauth.token")),
    )


@pytest.mark.parametrize(
    "paths",
    (
        "['oauth', 'oauth.token']",
        "['bad..path']",
        "['UPPER']",
    ),
)
def test_static_manifest_rejects_ambiguous_channel_credential_paths(
    tmp_path: Path,
    paths: str,
) -> None:
    root = tmp_path / "feishu"
    root.mkdir()
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'feishu'\n"
        "version = '3.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[channel_credentials]\n"
        f"feishu = {paths}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="channel_credentials"):
        load_static_plugin_manifest(root)


def test_static_manifest_rejects_cross_channel_credential_path_overlap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "channels"
    root.mkdir()
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'channels'\n"
        "version = '3.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[channel_credentials]\n"
        "feishu = ['oauth']\n"
        "qqbot = ['oauth.token']\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="跨 channel 路径重叠"):
        load_static_plugin_manifest(root)


def test_static_manifest_validates_mcp_and_process_declarations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calendar"
    (root / "mcp").mkdir(parents=True)
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "mcp" / "run_mcp.py").write_text("", encoding="utf-8")
    (root / "mcp" / "run_server.py").write_text("", encoding="utf-8")
    (root / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        _manifest()
        + """
[[processes]]
name = "calendar_api"
command = ["python", "mcp/run_server.py"]
cwd = "mcp"
port_env = "PORT"
formal_port = 18000
readiness_path = "/health"

[[mcp]]
name = "calendar"
command = ["python", "mcp/run_mcp.py"]
cwd = "mcp"
required_tools = ["get_proactive_events"]
candidate_read_only_tools = ["get_proactive_events"]
endpoint_env = [{env = "PORT", process = "calendar_api"}]
candidate_env = {CALENDAR_BACKEND = "recording"}
""",
        encoding="utf-8",
    )

    manifest = load_static_plugin_manifest(root)

    assert manifest.managed_processes[0].name == "calendar_api"
    assert manifest.managed_processes[0].formal_port == 18000
    assert manifest.mcp_servers[0].endpoint_env == (("PORT", "calendar_api"),)
    assert manifest.mcp_servers[0].candidate_env == (("CALENDAR_BACKEND", "recording"),)
    assert manifest.mcp_servers[0].python_runtime == "mcp"
    assert manifest.managed_processes[0].python_runtime == "mcp"


def test_static_manifest_validates_relative_command_head_inside_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calendar"
    (root / "mcp").mkdir(parents=True)
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "mcp" / "run_mcp.py").write_text("", encoding="utf-8")
    runner = root / "mcp" / "runner"
    runner.write_text("", encoding="utf-8")
    (root / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        _manifest()
        + '\n[[mcp]]\nname = "calendar"\ncommand = ["mcp/runner", "mcp/run_mcp.py"]\n',
        encoding="utf-8",
    )

    manifest = load_static_plugin_manifest(root)

    assert manifest.mcp_servers[0].command == ("mcp/runner", "mcp/run_mcp.py")

    (root / "mcp" / ".venv" / "bin").mkdir(parents=True)
    interpreter = root / "mcp" / ".venv" / "bin" / "python"
    interpreter.write_text("", encoding="utf-8")
    interpreter.chmod(interpreter.stat().st_mode | 0o111)
    assert staged_python_interpreter(root, manifest.python[0]) == interpreter


def test_static_manifest_rejects_external_command_argument(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calendar"
    (root / "mcp").mkdir(parents=True)
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "mcp" / "run_mcp.py").write_text("", encoding="utf-8")
    (root / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        _manifest()
        + f'\n[[mcp]]\nname = "calendar"\ncommand = ["{sys.executable}", "/tmp/other.py"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact 外绝对路径"):
        load_static_plugin_manifest(root)


def test_static_manifest_rejects_escaped_relative_command_head(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calendar"
    (root / "mcp").mkdir(parents=True)
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        _manifest()
        + '\n[[mcp]]\nname = "calendar"\ncommand = ["../outside"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact 内的相对路径"):
        load_static_plugin_manifest(root)


def test_manager_command_containment_preserves_staged_interpreter(
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "calendar"
    plugin_root.mkdir()
    script = plugin_root / "run.py"
    script.write_text("", encoding="utf-8")

    assert manager_module._resolve_command_item(  # pyright: ignore[reportPrivateUsage]
        plugin_root,
        sys.executable,
        executable=True,
    ) == sys.executable
    assert manager_module._resolve_command_item(  # pyright: ignore[reportPrivateUsage]
        plugin_root,
        "./run.py",
        executable=False,
    ) == str(script.resolve())

    outside = tmp_path / "not-executable"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="绝对路径不允许越过 artifact"):
        manager_module._resolve_command_item(  # pyright: ignore[reportPrivateUsage]
            plugin_root,
            str(outside),
            executable=True,
        )


def test_candidate_data_inventory_excludes_manifest_paths(tmp_path: Path) -> None:
    source = tmp_path / "production-data"
    target = tmp_path / "candidate-data"
    source.mkdir()
    (source / "state.json").write_text("keep", encoding="utf-8")
    (source / ".env").write_text("SECRET=bad", encoding="utf-8")
    (source / "oauth").mkdir()
    (source / "oauth" / "token.json").write_text("secret", encoding="utf-8")

    inventory = manager_module._copy_validation_data(  # pyright: ignore[reportPrivateUsage]
        source,
        target,
        (".env", "oauth"),
    )

    assert inventory == ("state.json",)
    assert (target / "state.json").is_file()
    assert not (target / ".env").exists()
    assert not (target / "oauth").exists()


def test_candidate_data_copy_rejects_symlink_to_formal_storage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "production-data"
    target = tmp_path / "candidate-data"
    outside = tmp_path / "formal-secret"
    source.mkdir()
    outside.write_text("secret", encoding="utf-8")
    (source / "token-link").symlink_to(outside)

    with pytest.raises(RuntimeError, match="不允许复制符号链接"):
        manager_module._copy_validation_data(  # pyright: ignore[reportPrivateUsage]
            source,
            target,
            (),
        )

    assert not target.exists()


def test_candidate_data_copy_ignores_symlink_inside_excluded_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "production-data"
    target = tmp_path / "candidate-data"
    cache = source / "checkouts" / "repository" / ".venv"
    cache.mkdir(parents=True)
    (cache / "lib").mkdir()
    (cache / "lib64").symlink_to("lib", target_is_directory=True)
    (source / "state.json").write_text("keep", encoding="utf-8")

    inventory = manager_module._copy_validation_data(  # pyright: ignore[reportPrivateUsage]
        source,
        target,
        ("checkouts",),
    )

    assert inventory == ("state.json",)
    assert not (target / "checkouts").exists()


def test_static_process_declaration_must_match_c13_root_registry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calendar"
    root.mkdir()
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "run_server.py").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'calendar'\n"
        "version = '3.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[[process]]\n"
        "name = 'calendar_api'\n"
        "command = ['run_server.py']\n"
        "port_env = 'PORT'\n"
        "formal_port = 18000\n",
        encoding="utf-8",
    )
    manifest = load_static_plugin_manifest(root)
    generation = SimpleNamespace(static_manifest=manifest, plugin_dir=root)
    snapshot = SimpleNamespace(
        mcp_server_registry=None,
        managed_process_registry=None,
        composition_active_plugin_ids=frozenset({"calendar"}),
    )

    with pytest.raises(RuntimeError, match="managed process 声明"):
        manager_module._validate_static_manifest_runtime(  # pyright: ignore[reportPrivateUsage]
            cast(RuntimeSnapshot, snapshot),
            {"calendar": cast(PluginGeneration, generation)},
        )


def test_static_python_command_uses_only_staged_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "calendar"
    (root / "mcp").mkdir(parents=True)
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "mcp" / "run.py").write_text("", encoding="utf-8")
    (root / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        _manifest()
        + '\n[[mcp]]\nname = "calendar"\ncommand = ["python", "mcp/run.py"]\n',
        encoding="utf-8",
    )
    interpreter = root / "mcp" / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    interpreter.chmod(0o755)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "python").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", str(hostile))

    manifest = load_static_plugin_manifest(root)
    command = materialize_static_command(root, manifest, manifest.mcp_servers[0])

    assert command == (str(interpreter), "mcp/run.py")


@pytest.mark.parametrize("python_command", ("python", "python3.12"))
def test_static_python_command_requires_unique_declared_runtime(
    tmp_path: Path,
    python_command: str,
) -> None:
    root = tmp_path / "calendar"
    root.mkdir()
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "run.py").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'calendar'\n"
        "version = '3.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[[mcp]]\n"
        "name = 'calendar'\n"
        f"command = ['{python_command}', 'run.py']\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="唯一绑定"):
        load_static_plugin_manifest(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("entrypoint", "/plugin.py", "entrypoint"),
        ("requirements", "../requirements.txt", "requirements"),
        ("requirements", "missing.txt", "requirements"),
    ],
)
def test_static_manifest_rejects_unsafe_or_missing_runtime_paths(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    root = tmp_path / "calendar"
    (root / "mcp").mkdir(parents=True)
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    manifest_text = _manifest(
        entrypoint=value if field == "entrypoint" else "plugin.py",
        requirements=value if field == "requirements" else "mcp/requirements.txt",
    )
    (root / "akashic.plugin.toml").write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_static_plugin_manifest(root)


def test_static_manifest_rejects_manifest_symlink(tmp_path: Path) -> None:
    root = tmp_path / "calendar"
    root.mkdir()
    (root / "plugin.py").write_text("", encoding="utf-8")
    outside = tmp_path / "outside.toml"
    outside.write_text(_manifest(requirements="requirements.txt"), encoding="utf-8")
    (root / "akashic.plugin.toml").symlink_to(outside)

    with pytest.raises(ValueError, match="缺少静态 manifest"):
        load_static_plugin_manifest(root)


@pytest.mark.parametrize(
    "declaration",
    (
        "[validation]\nexclude_data_paths = ['.']\n",
        "[[process]]\nname = 'api'\ncommand = ['run.py']\n"
        "port_env = 'AKASHIC_WORKSPACE'\nformal_port = 18000\n",
        "[[process]]\nname = 'api'\ncommand = ['run.py']\n"
        "port_env = 'PORT'\nformal_port = 18000\nreadiness_path = '//evil'\n",
        "[[process]]\nname = 'api'\ncommand = ['run.py']\n"
        "port_env = 'PORT'\nformal_port = 18000\nstartup_timeout_seconds = nan\n",
        "[[process]]\nname = 'api'\ncommand = ['run.py']\n",
        "[[process]]\nname = 'api'\ncommand = ['run.py']\n"
        "port_env = 'PORT'\nformal_port = 18000\n\n"
        "[[mcp]]\nname = 'calendar'\ncommand = ['run.py']\n"
        "endpoint_env = [{env = 'AKASHIC_WORKSPACE', process = 'api'}]\n",
    ),
)
def test_static_manifest_rejects_invalid_runtime_policy_before_import(
    tmp_path: Path,
    declaration: str,
) -> None:
    root = tmp_path / "calendar"
    root.mkdir()
    marker = root / "imported"
    (root / "plugin.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('imported').write_text('bad')\n",
        encoding="utf-8",
    )
    (root / "run.py").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'calendar'\n"
        "version = '3.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        + declaration,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_static_plugin_manifest(root)
    assert not marker.exists()


def test_v3_static_install_stages_before_importing_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "calendar-source"
    (repo / "mcp").mkdir(parents=True)
    (repo / "plugin.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('imported').write_text('bad')\n"
        "raise RuntimeError('static install must not import')\n",
        encoding="utf-8",
    )
    (repo / "mcp" / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (repo / "akashic.plugin.toml").write_text(_manifest(), encoding="utf-8")
    _commit(repo)
    calls: list[tuple[str, Path]] = []

    def fake_run(args: list[str], *, cwd: Path, label: str) -> None:
        calls.append((label, cwd))
        if label.endswith("venv"):
            python_path = install_module._venv_python_path(cwd / ".venv")
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(install_module, "_run_command", fake_run)
    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    assert result.plugin_name == "calendar"
    assert result.plugin_version == "3.0.0"
    assert [label for label, _ in calls] == [
        "calendar python[0] venv",
        "calendar python[0] pip install",
    ]
    assert not (result.installed_path / "imported").exists()
    assert (result.installed_path / "akashic.plugin.toml").is_file()
    assert (result.installed_path / "mcp" / ".venv").is_dir()


def test_v3_static_install_accepts_custom_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "calendar-source"
    repo.mkdir()
    (repo / "entry.py").write_text("", encoding="utf-8")
    (repo / "requirements.txt").write_text("", encoding="utf-8")
    (repo / "akashic.plugin.toml").write_text(
        _manifest(entrypoint="entry.py", requirements="requirements.txt"),
        encoding="utf-8",
    )
    _commit(repo)

    def fake_run(args: list[str], *, cwd: Path, label: str) -> None:
        if label.endswith("venv"):
            python_path = install_module._venv_python_path(cwd / ".venv")
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(install_module, "_run_command", fake_run)
    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    assert (result.installed_path / "entry.py").is_file()
    assert not (result.installed_path / "plugin.py").exists()


def test_install_rejects_missing_static_manifest_before_plugin_import(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "legacy-source"
    repo.mkdir()
    imported = tmp_path / "imported"
    (repo / "plugin.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    _commit(repo)

    with pytest.raises(ValueError, match="akashic.plugin.toml"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )

    assert not imported.exists()
    assert not (tmp_path / "workspace" / "plugin-data").exists()


@pytest.mark.parametrize("preexisting", (False, True))
def test_v3_static_staging_failure_preserves_formal_data_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    repo = tmp_path / "calendar-source"
    repo.mkdir()
    (repo / "plugin.py").write_text("", encoding="utf-8")
    (repo / "requirements.txt").write_text("", encoding="utf-8")
    (repo / "akashic.plugin.toml").write_text(
        _manifest(requirements="requirements.txt"),
        encoding="utf-8",
    )
    _commit(repo)
    workspace = tmp_path / "workspace"
    data_path = workspace / "plugin-data" / "calendar-lab"
    if preexisting:
        data_path.mkdir(parents=True)
        (data_path / "state.json").write_bytes(b"keep")

    def fail_run(args: list[str], *, cwd: Path, label: str) -> None:
        raise RuntimeError("staging failed")

    monkeypatch.setattr(install_module, "_run_command", fail_run)
    with pytest.raises(RuntimeError, match="staging failed"):
        install_git_plugin(
            workspace=workspace,
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )

    if preexisting:
        assert (data_path / "state.json").read_bytes() == b"keep"
    else:
        assert not data_path.exists()


def test_v3_static_manifest_write_failure_removes_new_data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "calendar-source"
    repo.mkdir()
    (repo / "plugin.py").write_text("", encoding="utf-8")
    (repo / "requirements.txt").write_text("", encoding="utf-8")
    (repo / "akashic.plugin.toml").write_text(
        _manifest(requirements="requirements.txt"),
        encoding="utf-8",
    )
    _commit(repo)
    workspace = tmp_path / "workspace"

    def fake_run(args: list[str], *, cwd: Path, label: str) -> None:
        if label.endswith("venv"):
            python_path = install_module._venv_python_path(cwd / ".venv")
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

    def fail_manifest(*args: object, **kwargs: object) -> Path:
        raise OSError("manifest write failed")

    monkeypatch.setattr(install_module, "_run_command", fake_run)
    monkeypatch.setattr(install_module, "upsert_plugin_manifest", fail_manifest)
    with pytest.raises(OSError, match="manifest write failed"):
        install_git_plugin(
            workspace=workspace,
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )

    assert not (workspace / "plugin-data" / "calendar-lab").exists()
