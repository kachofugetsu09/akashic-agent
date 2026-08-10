from __future__ import annotations

import os
import secrets
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, NoReturn

from scripts.akashic_release.doctor import read_environment, verify_release
from scripts.akashic_release.manifest import activation_receipt, atomic_write, read_json
from scripts.akashic_release.manifest import write_json
from scripts.akashic_release.model import ReleasePaths
from scripts.akashic_release.systemd import start_bridge, start_core, stop_runtime

Run = Callable[..., subprocess.CompletedProcess[str]]


def ensure_bridge_token(paths: ReleasePaths) -> str:
    token_file = paths.secrets / "host-bridge.token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise RuntimeError("Host Bridge token 文件损坏")
        return token
    token = secrets.token_urlsafe(48)
    atomic_write(token_file, token + "\n")
    return token


def release_environment(
    *,
    paths: ReleasePaths,
    manifest: Mapping[str, object],
    current: Mapping[str, str],
    mise: Path,
) -> dict[str, str]:
    """Build runtime.env by replacing only release-owned generation fields."""

    commit = str(manifest["sourceCommit"])
    host_identity = manifest["hostToolchainIdentity"]
    if not isinstance(host_identity, Mapping):
        raise RuntimeError("release manifest 缺少 host toolchain identity")
    token = ensure_bridge_token(paths)
    values = dict(current)
    values.update(
        {
            "AKASHIC_BRIDGE_PYTHON": str(paths.bridge_venv(commit) / "bin/python"),
            "AKASHIC_MISE": str(mise),
            "AKASHIC_RUNTIME_CHECKOUT": str(paths.source(commit)),
            "AKASHIC_RUNTIME_COMMIT": commit,
            "AKASHIC_HOST_TOOLCHAIN_DIGEST": str(host_identity["toolchainDigest"]),
            "AKASHIC_RELEASE_MANIFEST": str(paths.release(commit)),
            "AKASHIC_IMAGE": str(manifest["imageId"]),
            "AKASHIC_HOST_BRIDGE_SOCKET": str(paths.run / "host-bridge.sock"),
            "AKASHIC_HOST_BRIDGE_TOKEN_FILE": str(paths.secrets / "host-bridge.token"),
            "AKASHIC_HOST_BRIDGE_TOKEN": token,
            "AKASHIC_HOST_BRIDGE_DIR": str(paths.run),
            "AKASHIC_HOST_ARTIFACT_ROOT": str(paths.root / "runtime/host-executions"),
            "AKASHIC_CONFIG": str(paths.state / "config.toml"),
            "AKASHIC_WORKSPACE": str(paths.state / "workspace"),
            "AKASHIC_PLUGIN_HOME": str(paths.state / "plugin-home"),
            "AKASHIC_EXPERIMENT_ROOT": str(paths.state),
            "AKASHIC_CONTAINER_NAME": values.get(
                "AKASHIC_CONTAINER_NAME", "akashic-core"
            ),
            "AKASHIC_PUBLISHED_WEB_PORT": values.get(
                "AKASHIC_PUBLISHED_WEB_PORT", "2236"
            ),
            "AKASHIC_SERVICES_NETWORK": values.get(
                "AKASHIC_SERVICES_NETWORK", "akashic-services"
            ),
            "AKASHIC_UID": values.get("AKASHIC_UID", str(os.getuid())),
            "AKASHIC_GID": values.get("AKASHIC_GID", str(os.getgid())),
            "AKASHIC_ENVIRONMENT": values.get("AKASHIC_ENVIRONMENT", "hua-home"),
            "AKASHIC_LOG_LEVEL": values.get("AKASHIC_LOG_LEVEL", "INFO"),
        }
    )
    if not values.get("OPENCODE_GO_API_KEY"):
        inherited = os.environ.get("OPENCODE_GO_API_KEY")
        if not inherited:
            raise RuntimeError("首次安装必须通过环境提供 OPENCODE_GO_API_KEY")
        values["OPENCODE_GO_API_KEY"] = inherited
    return values


def render_environment(values: Mapping[str, str]) -> str:
    if any("\n" in value or "\x00" in value for value in values.values()):
        raise RuntimeError("runtime.env value 不得包含换行或 NUL")
    return "".join(f"{key}={values[key]}\n" for key in sorted(values))


def _manual_recovery_commands(environment_file: Path) -> list[str]:
    environment = shlex.quote(str(environment_file))
    return [
        "sudo systemctl stop akashic-core.service akashic-host-bridge.service",
        "sudo systemctl start akashic-host-bridge.service akashic-core.service",
        f"AKASHIC_RUNTIME_ENV={environment} akashic-release doctor",
    ]


def _restore_previous(
    *,
    paths: ReleasePaths,
    environment_file: Path,
    backup: Path,
    target: str,
    previous: str,
    timestamp: str,
    candidate_error: BaseException,
    run: Run,
) -> NoReturn:
    """Restore and verify the previous generation or persist maintenance evidence."""

    # 1. Restore the previous environment and perform the real service probe.
    atomic_write(environment_file, backup.read_text(encoding="utf-8"))
    try:
        start_bridge(run=run)
        start_core(run=run)
        verify_release(environment_file)
    except BaseException as recovery_error:
        maintenance_stop_detail = None
        try:
            stop_runtime(run=run)
        except BaseException as stop_error:
            maintenance_stop_detail = str(stop_error)
        receipt = activation_receipt(
            status="recovery_failed",
            target_commit=target,
            previous_commit=previous,
            detail=str(candidate_error),
        )
        receipt["recoveryDetail"] = str(recovery_error)
        receipt["manualCommands"] = _manual_recovery_commands(environment_file)
        if maintenance_stop_detail is not None:
            receipt["maintenanceStopDetail"] = maintenance_stop_detail
        write_json(paths.activation / f"failed-{target}-{timestamp}.json", receipt)
        commands = " ; ".join(_manual_recovery_commands(environment_file))
        raise RuntimeError(
            f"候选与 previous {previous} 均验证失败，停在 maintenance；人工恢复: {commands}"
        ) from recovery_error

    # 2. Record the verified rollback without claiming business data was reverted.
    write_json(
        paths.activation / f"failed-{target}-{timestamp}.json",
        activation_receipt(
            status="rolled_back",
            target_commit=target,
            previous_commit=previous,
            detail=str(candidate_error),
        ),
    )
    raise RuntimeError(f"候选激活失败，已恢复 {previous}") from candidate_error


def activate_release(
    *,
    paths: ReleasePaths,
    manifest_path: Path,
    environment_file: Path,
    mise: Path,
    run: Run,
) -> str:
    """Activate one prepared generation and restore the previous env on failure."""

    manifest = read_json(manifest_path)
    target = str(manifest["sourceCommit"])
    active_path = paths.activation / "active.json"
    previous = (
        read_json(active_path).get("targetCommit") if active_path.exists() else None
    )
    current = read_environment(environment_file) if environment_file.exists() else {}
    _verify_state_ready(paths)
    candidate = release_environment(
        paths=paths,
        manifest=manifest,
        current=current,
        mise=mise,
    )
    if previous == target:
        verify_release(environment_file)
        return "already_active"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = paths.backups / f"runtime.env.before-{target}-{timestamp}"
    if environment_file.exists():
        if backup.exists():
            raise RuntimeError(f"runtime.env backup 已存在: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(environment_file, backup)
    stop_runtime(run=run)
    atomic_write(environment_file, render_environment(candidate))
    try:
        start_bridge(run=run)
        start_core(run=run)
        verify_release(environment_file)
    except BaseException as error:
        stop_runtime(run=run)
        if previous is None or not backup.exists():
            receipt = activation_receipt(
                status="failed",
                target_commit=target,
                previous_commit=None,
                detail=str(error),
            )
            receipt["manualCommands"] = _manual_recovery_commands(environment_file)
            write_json(
                paths.activation / f"failed-{target}-{timestamp}.json",
                receipt,
            )
            raise RuntimeError("首次激活失败，已停在 maintenance") from error
        _restore_previous(
            paths=paths,
            environment_file=environment_file,
            backup=backup,
            target=target,
            previous=str(previous),
            timestamp=timestamp,
            candidate_error=error,
            run=run,
        )

    receipt = activation_receipt(
        status="active",
        target_commit=target,
        previous_commit=None if previous is None else str(previous),
    )
    write_json(paths.activation / "active.json", receipt)
    if previous is not None:
        write_json(paths.activation / "previous.json", {"targetCommit": previous})
    return "activated"


def _verify_state_ready(paths: ReleasePaths) -> None:
    config = paths.state / "config.toml"
    directories = (paths.state / "workspace", paths.state / "plugin-home")
    missing = [str(config)] if not config.is_file() else []
    missing.extend(str(path) for path in directories if not path.is_dir())
    if missing:
        raise RuntimeError(
            "正式 state 尚未准备，使用 --no-activate 后按迁移计划创建: "
            + ", ".join(missing)
        )
    ownership = paths.state / "workspace/runtime/plugin-skill-links.json"
    legacy_links = [
        item
        for directory in (
            paths.state / "workspace/skills",
            paths.state / "workspace/drift/skills",
        )
        if directory.is_dir()
        for item in directory.iterdir()
        if item.is_symlink()
    ]
    if legacy_links and not ownership.is_file():
        raise RuntimeError(
            "检测到未登记 legacy skill links；激活前备份并运行 "
            "scripts/adopt_legacy_plugin_skill_links.py: "
            + ", ".join(str(path) for path in sorted(legacy_links))
        )
