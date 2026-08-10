from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.akashic_release import activate as activate_module
from scripts.akashic_release import prepare as prepare_module
from scripts.akashic_release import systemd as systemd_module
from scripts.akashic_release.activate import activate_release, render_environment
from scripts.akashic_release.activate import release_environment
from scripts.akashic_release.bridge import prepare_bridge_venv
from scripts.akashic_release.doctor import probe_bridge, read_environment
from scripts.akashic_release.manifest import read_json, release_lock, write_json
from scripts.akashic_release.migrate import migration_plan
from scripts.akashic_release.model import ReleasePaths
from scripts.akashic_release.prepare import prepare_generation
from scripts.akashic_release.source import resolve_target
from scripts.akashic_release.source import verify_bootstrap_checkout
from scripts.akashic_release.systemd import install_units
from scripts.akashic_release.systemd import install_operator_entrypoint
from scripts.akashic_release.systemd import verify_external_service_contract


def _repository(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "value.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=source, check=True)
    first = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    subprocess.run(["git", "tag", "kept"], cwd=source, check=True)
    (source / "value.txt").write_text("two\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "two"], cwd=source, check=True)
    second = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(source), str(remote)], check=True
    )
    return source, remote, first, second


def test_source_resolves_latest_main_and_explicit_reachable_commit(
    tmp_path: Path,
) -> None:
    _source, remote, first, second = _repository(tmp_path)

    assert resolve_target(str(remote), None, run=subprocess.run) == second
    assert resolve_target(str(remote), first, run=subprocess.run) == first
    with pytest.raises(RuntimeError, match="40 位"):
        resolve_target(str(remote), "main", run=subprocess.run)


def test_bootstrap_checkout_requires_exact_clean_origin(tmp_path: Path) -> None:
    source, remote, _first, second = _repository(tmp_path)
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)], cwd=source, check=True
    )

    verify_bootstrap_checkout(source, second, str(remote), run=subprocess.run)
    (source / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean"):
        verify_bootstrap_checkout(source, second, str(remote), run=subprocess.run)


def test_release_lock_rejects_concurrent_installer(tmp_path: Path) -> None:
    lock = tmp_path / "release.lock"
    with release_lock(lock):
        with pytest.raises(RuntimeError, match="已有"):
            with release_lock(lock):
                pass


def test_prepare_failure_removes_only_owned_partial_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ReleasePaths(tmp_path / "root")
    paths.create_layout()
    commit = "a" * 40

    def checkout(_bootstrap: Path, _commit: str, target: Path, _origin: str) -> Path:
        target.mkdir()
        return target

    def image(**kwargs: object) -> dict[str, object]:
        Path(str(kwargs["manifest"])).write_text("{}", encoding="utf-8")
        return {
            "sourceCommit": commit,
            "imageId": "sha256:" + "b" * 64,
            "hostToolchainIdentity": {"toolchainDigest": "c" * 64},
        }

    def bridge(**kwargs: object) -> Path:
        target = Path(str(kwargs["target"]))
        (target / "bin").mkdir(parents=True)
        python = target / "bin/python"
        python.write_text("", encoding="utf-8")
        return python

    monkeypatch.setattr(prepare_module, "prepare_runtime_checkout", checkout)
    monkeypatch.setattr(prepare_module, "prepare_core_image", image)
    monkeypatch.setattr(prepare_module, "prepare_bridge_venv", bridge)
    monkeypatch.setattr(
        prepare_module,
        "verify_bridge",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("doctor failed")),
    )

    with pytest.raises(RuntimeError, match="doctor failed"):
        prepare_generation(
            paths=paths,
            bootstrap_checkout=tmp_path,
            commit=commit,
            origin="origin",
            mise=tmp_path / "mise",
            run=subprocess.run,
        )

    assert not paths.source(commit).exists()
    assert not paths.bridge_venv(commit).exists()
    assert not paths.release(commit).exists()


def test_bridge_venv_uses_hashed_requirements_from_domestic_index(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    requirements = checkout / "docker/host-runtime/requirements.lock"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("", encoding="utf-8")
    target = tmp_path / "bridge-venv"
    calls: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if "venv" in arguments:
            (target / "bin").mkdir(parents=True)
            (target / "bin/python").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    python = prepare_bridge_venv(
        checkout=checkout,
        target=target,
        mise=tmp_path / "mise",
        run=run,
    )

    install = calls[-1]
    assert python == target / "bin/python"
    assert install[install.index("--default-index") + 1] == (
        "https://pypi.tuna.tsinghua.edu.cn/simple"
    )
    assert "--require-hashes" in install


def test_bridge_probe_uses_generation_python_without_secret_arguments(
    tmp_path: Path,
) -> None:
    environment_file = tmp_path / "runtime.env"
    bridge_python = tmp_path / "bridge-venv/bin/python"
    checkout = tmp_path / "runtime-source"
    environment = {
        "AKASHIC_BRIDGE_PYTHON": str(bridge_python),
        "AKASHIC_RUNTIME_CHECKOUT": str(checkout),
        "AKASHIC_HOST_BRIDGE_TOKEN": "must-not-enter-argv",
    }
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0)

    probe_bridge(environment, environment_file=environment_file, run=run)

    assert calls == [
        (
            [
                str(bridge_python),
                "-m",
                "scripts.akashic_release.doctor",
                "--bridge-probe-environment",
                str(environment_file),
            ],
            {
                "cwd": checkout,
                "check": True,
                "capture_output": True,
                "text": True,
            },
        )
    ]
    assert "must-not-enter-argv" not in " ".join(calls[0][0])


def test_activation_failure_atomically_restores_previous_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ReleasePaths(tmp_path / "root")
    paths.create_layout()
    (paths.state / "config.toml").write_text("[runtime]\n", encoding="utf-8")
    (paths.state / "workspace").mkdir()
    (paths.state / "plugin-home").mkdir()
    old = "a" * 40
    target = "b" * 40
    write_json(
        paths.activation / "active.json",
        {"schemaVersion": 1, "status": "active", "targetCommit": old},
    )
    manifest = paths.release(target)
    write_json(
        manifest,
        {
            "sourceCommit": target,
            "imageId": "sha256:" + "c" * 64,
            "hostToolchainIdentity": {"toolchainDigest": "d" * 64},
        },
    )
    environment = tmp_path / "runtime.env"
    original = "AKASHIC_RUNTIME_COMMIT=" + old + "\nOPENCODE_GO_API_KEY=secret\n"
    environment.write_text(original, encoding="utf-8")
    calls = 0

    def verify(_environment: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("candidate unhealthy")

    monkeypatch.setattr(activate_module, "verify_release", verify)
    fake_run = lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0)

    with pytest.raises(RuntimeError, match="已恢复"):
        activate_release(
            paths=paths,
            manifest_path=manifest,
            environment_file=environment,
            mise=tmp_path / "mise",
            run=fake_run,
        )

    assert environment.read_text(encoding="utf-8") == original
    assert read_json(paths.activation / "active.json")["targetCommit"] == old
    failed = list(paths.activation.glob(f"failed-{target}-*.json"))
    assert len(failed) == 1
    assert read_json(failed[0])["status"] == "rolled_back"


def test_previous_recovery_failure_records_maintenance_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ReleasePaths(tmp_path / "root")
    paths.create_layout()
    (paths.state / "config.toml").write_text("[runtime]\n", encoding="utf-8")
    (paths.state / "workspace").mkdir()
    (paths.state / "plugin-home").mkdir()
    previous = "a" * 40
    target = "b" * 40
    write_json(
        paths.activation / "active.json",
        {"schemaVersion": 1, "status": "active", "targetCommit": previous},
    )
    manifest = paths.release(target)
    write_json(
        manifest,
        {
            "sourceCommit": target,
            "imageId": "sha256:" + "c" * 64,
            "hostToolchainIdentity": {"toolchainDigest": "d" * 64},
        },
    )
    environment = tmp_path / "runtime.env"
    original = f"AKASHIC_RUNTIME_COMMIT={previous}\nOPENCODE_GO_API_KEY=secret\n"
    environment.write_text(original, encoding="utf-8")
    verify_errors = iter(("candidate unhealthy", "previous unhealthy"))
    monkeypatch.setattr(
        activate_module,
        "verify_release",
        lambda _environment: (_ for _ in ()).throw(RuntimeError(next(verify_errors))),
    )
    service_calls: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        service_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    with pytest.raises(RuntimeError, match="均验证失败.*人工恢复"):
        activate_release(
            paths=paths,
            manifest_path=manifest,
            environment_file=environment,
            mise=tmp_path / "mise",
            run=run,
        )

    assert environment.read_text(encoding="utf-8") == original
    receipt_path = next(paths.activation.glob(f"failed-{target}-*.json"))
    receipt = read_json(receipt_path)
    assert receipt["status"] == "recovery_failed"
    assert receipt["detail"] == "candidate unhealthy"
    assert receipt["recoveryDetail"] == "previous unhealthy"
    assert receipt["previousCommit"] == previous
    assert receipt["manualCommands"] == [
        "sudo systemctl stop akashic-core.service akashic-host-bridge.service",
        "sudo systemctl start akashic-host-bridge.service akashic-core.service",
        f"AKASHIC_RUNTIME_ENV={environment} akashic-release doctor",
    ]
    assert service_calls[-1] == [
        "sudo",
        "systemctl",
        "stop",
        "akashic-core.service",
        "akashic-host-bridge.service",
    ]


def test_runtime_environment_rejects_multiline_secret(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="换行"):
        render_environment({"TOKEN": "first\nsecond"})


def test_release_environment_preserves_web_bind_and_loopback_mobile_port(
    tmp_path: Path,
) -> None:
    paths = ReleasePaths(tmp_path / "root")
    paths.create_layout()
    commit = "a" * 40
    environment = release_environment(
        paths=paths,
        manifest={
            "sourceCommit": commit,
            "imageId": "sha256:" + "b" * 64,
            "hostToolchainIdentity": {"toolchainDigest": "c" * 64},
        },
        current={
            "AKASHIC_WEB_BIND_ADDRESS": "192.168.0.100",
            "OPENCODE_GO_API_KEY": "secret",
        },
        mise=tmp_path / "mise",
    )

    compose = Path("docker/host-runtime/compose.experiment.yaml").read_text()
    assert environment["AKASHIC_WEB_BIND_ADDRESS"] == "192.168.0.100"
    assert environment["AKASHIC_PUBLISHED_MOBILE_PORT"] == "6323"
    assert (
        '"${AKASHIC_WEB_BIND_ADDRESS:-127.0.0.1}:'
        '${AKASHIC_PUBLISHED_WEB_PORT:-2236}:2236"' in compose
    )
    assert '127.0.0.1:${AKASHIC_PUBLISHED_MOBILE_PORT:-6323}:6323' in compose


def test_activation_rejects_unadopted_legacy_skill_before_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ReleasePaths(tmp_path / "root")
    paths.create_layout()
    (paths.state / "config.toml").write_text("[runtime]\n", encoding="utf-8")
    workspace = paths.state / "workspace"
    skills = workspace / "skills"
    skills.mkdir(parents=True)
    plugin_home = paths.state / "plugin-home"
    target = plugin_home / "cache/plugin/skills/legacy"
    target.mkdir(parents=True)
    (skills / "legacy").symlink_to(target, target_is_directory=True)
    commit = "b" * 40
    manifest = paths.release(commit)
    write_json(
        manifest,
        {
            "sourceCommit": commit,
            "imageId": "sha256:" + "c" * 64,
            "hostToolchainIdentity": {"toolchainDigest": "d" * 64},
        },
    )
    calls: list[list[str]] = []
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-secret")

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    with pytest.raises(RuntimeError, match="legacy skill links"):
        activate_release(
            paths=paths,
            manifest_path=manifest,
            environment_file=tmp_path / "runtime.env",
            mise=tmp_path / "mise",
            run=run,
        )

    assert calls == []


def test_unit_install_backs_up_changed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        systemd_module.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(
            pw_name="operator", pw_dir="/srv/operators/operator"
        ),
    )
    monkeypatch.setattr(
        systemd_module.grp,
        "getgrgid",
        lambda _gid: SimpleNamespace(gr_name="operator"),
    )
    checkout = tmp_path / "checkout"
    source = checkout / "docker/host-runtime/systemd"
    source.mkdir(parents=True)
    unit_root = tmp_path / "units"
    unit_root.mkdir()
    for name in ("akashic-host-bridge.service", "akashic-core.service"):
        (source / name).write_text(
            f"[Unit]\nDescription=new {name}\n"
            "[Service]\nUser=huashen\nGroup=huashen\n",
            encoding="utf-8",
        )
        (unit_root / name).write_text(f"old {name}\n", encoding="utf-8")
    calls: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    assert install_units(
        checkout=checkout,
        backup_root=tmp_path / "backups",
        run=run,
        unit_root=unit_root,
    )
    backup = next((tmp_path / "backups").iterdir())
    assert (backup / "akashic-core.service").read_text().startswith("old")
    assert calls == []
    rendered = (unit_root / "akashic-core.service").read_text()
    assert "Description=new" in rendered
    assert "User=operator" in rendered
    assert "Group=operator" in rendered


def test_unit_install_accepts_canonical_huashen_service_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        systemd_module.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_name="huashen", pw_dir="/home/huashen"),
    )
    monkeypatch.setattr(
        systemd_module.grp,
        "getgrgid",
        lambda _gid: SimpleNamespace(gr_name="huashen"),
    )
    repository = Path(__file__).resolve().parents[1]
    unit_root = tmp_path / "units"
    unit_root.mkdir()
    calls: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    assert install_units(
        checkout=repository,
        backup_root=tmp_path / "backups",
        run=run,
        unit_root=unit_root,
    )

    assert calls == []
    for name in ("akashic-host-bridge.service", "akashic-core.service"):
        rendered = (unit_root / name).read_text(encoding="utf-8")
        assert "User=huashen" in rendered
        assert "Group=huashen" in rendered
        assert "%h" not in rendered
        assert (
            "EnvironmentFile=/home/huashen/.config/akashic-container/runtime.env"
            in rendered
        )


def test_isolated_unit_root_requires_and_verifies_external_contract(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    with pytest.raises(RuntimeError, match="缺少外围服务合同"):
        verify_external_service_contract(run=run, unit_root=tmp_path)

    external = tmp_path / "akashic-home-services.service"
    external.write_text("[Service]\nExecStart=/usr/bin/true\n", encoding="utf-8")
    verify_external_service_contract(run=run, unit_root=tmp_path)

    assert calls == [["systemd-analyze", "verify", str(external)]]


def test_operator_entrypoint_is_atomic_and_backed_up(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    source = checkout / "scripts/akashic-release"
    source.parent.mkdir(parents=True)
    source.write_text("#!/bin/sh\necho new\n", encoding="utf-8")
    target = tmp_path / "bin/akashic-release"
    target.parent.mkdir()
    target.write_text("#!/bin/sh\necho old\n", encoding="utf-8")

    assert install_operator_entrypoint(
        checkout=checkout,
        backup_root=tmp_path / "backups",
        target=target,
    )

    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert target.stat().st_mode & 0o777 == 0o755
    backup = next((tmp_path / "backups").iterdir())
    assert (backup / "akashic-release").read_text().endswith("echo old\n")


def test_bootstrap_pins_resolved_main_before_running_python(tmp_path: Path) -> None:
    _source, remote, _first, commit = _repository(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python3"
    python.symlink_to("/bin/echo")
    environment = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AKASHIC_INSTALL_ORIGIN": str(remote),
    }

    result = subprocess.run(
        ["sh", "scripts/install-akashic.sh", "--yes"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invoked = result.stdout.split()
    assert "--yes" in invoked
    commit_index = invoked.index("--commit")
    assert invoked[commit_index + 1] == commit


def test_bootstrap_cli_imports_without_site_packages() -> None:
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(repository / "scripts/akashic_release/cli.py"),
            "--help",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def test_migration_command_is_plan_only_and_requires_integrity(tmp_path: Path) -> None:
    manifest = tmp_path / "rehearsal.json"
    manifest.write_text(
        json.dumps(
            {
                "consistency": {"attempts": 1},
                "cleanup": {"exact_paths": [str(tmp_path / "candidate")]},
                "databases": [
                    {
                        "source_integrity_check": "ok",
                        "target_integrity_check": "ok",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = migration_plan(manifest)

    assert plan["mode"] == "plan_only"
    assert plan["automaticDataWrites"] is False
    assert isinstance(plan["phases"], list)
    assert len(plan["phases"]) == 7


def test_environment_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    environment = tmp_path / "runtime.env"
    environment.write_text("A=1\nA=2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="重复"):
        read_environment(environment)
