from __future__ import annotations

import os
import json
import re
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
_ROOT_REF = re.compile(r"[0-9a-f]{64}\Z")


def _stopped_upgrade(
    *, paths: ReleasePaths, candidate: Mapping[str, str], manifest: Mapping[str, object],
    backup_dir: Path, previous_commit: str, run: Run,
) -> dict[str, object]:
    """Run the target image's own upgrade code against stopped state."""

    stable = read_json(paths.state / "workspace/runtime/plugin-stable.json")
    root_ref = stable.get("root_ref")
    if stable.get("version") != 1 or not isinstance(root_ref, str) or _ROOT_REF.fullmatch(root_ref) is None:
        raise RuntimeError("release upgrade 需要已有完整 stable Root")
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,mode=1777",
        "--mount", f"type=bind,src={paths.state},dst={paths.state}",
        "--mount", f"type=bind,src={paths.backups},dst={paths.backups}",
        "--env", f"AKASHIC_CONFIG={candidate['AKASHIC_CONFIG']}",
        "--env", f"AKASHIC_WORKSPACE={candidate['AKASHIC_WORKSPACE']}",
        "--env", f"AKASHIC_PLUGIN_HOME={candidate['AKASHIC_PLUGIN_HOME']}",
        "--env", f"AKASHIC_RUNTIME_COMMIT={candidate['AKASHIC_RUNTIME_COMMIT']}",
        "--env", f"AKASHIC_RUNTIME_TREE={candidate['AKASHIC_RUNTIME_TREE']}",
        str(manifest["imageId"]), "upgrade-bundled",
        "--expected-root-ref", root_ref, "--backup-dir", str(backup_dir),
        "--previous-source-commit", previous_commit,
    ]
    result = run(command, check=True, capture_output=True, text=True)
    try:
        receipt = json.loads(result.stdout)
    except (TypeError, ValueError) as error:
        raise RuntimeError("目标 image 未返回 upgrade JSON") from error
    if not isinstance(receipt, dict) or receipt.get("status") not in {
        "selected_not_started", "partial_selected_not_started",
        "already_selected_not_started", "no_eligible_targets",
    }:
        raise RuntimeError(f"目标 image upgrade 结果无效: {receipt}")
    selected = receipt.get("new_root_ref")
    if (receipt.get("old_root_ref") != root_ref or not isinstance(selected, str)
        or _ROOT_REF.fullmatch(selected) is None
        or receipt.get("backup_dir") != str(backup_dir)
        or not (backup_dir / "manifest.json").is_file()):
        raise RuntimeError("目标 image upgrade 缺少完整 Root 或已校验恢复点")
    return receipt


def _verify_selected_runtime(
    *, candidate: Mapping[str, str], root_ref: str, run: Run,
) -> dict[str, object]:
    """Read the live runtime's exact selected Fiber identities after health."""

    result = run([
        "docker", "exec", candidate["AKASHIC_CONTAINER_NAME"],
        "/opt/venv/bin/python", "/opt/akashic/source/main.py", "plugin-status",
        "--config", candidate["AKASHIC_CONFIG"],
        "--workspace", candidate["AKASHIC_WORKSPACE"],
    ], check=True, capture_output=True, text=True)
    status = json.loads(result.stdout)
    if not isinstance(status, dict) or status.get("selection_ref") != root_ref:
        raise RuntimeError("live runtime selection 与已迁移的完整 Root 不一致")
    plugins = status.get("plugins")
    components = status.get("selection_components")
    if (not isinstance(plugins, list) or not isinstance(components, list)
        or any(not isinstance(ref, str) for ref in components)
        or len(set(components)) != len(components)):
        raise RuntimeError("live runtime 未报告 plugin owner 状态")
    active_refs: list[str] = []
    for item in plugins:
        if not isinstance(item, dict):
            raise RuntimeError("live runtime plugin 状态无效")
        selected = item.get("selected_ref")
        if selected is not None and not isinstance(selected, str):
            raise RuntimeError("live runtime selected_ref 格式无效")
        if selected is not None and (
            item.get("archive_ref") != selected or item.get("state") != "active"
            or item.get("fiber_state") != "active"
        ):
            raise RuntimeError(f"selected plugin 尚未 ACTIVE: {item.get('plugin_id')}")
        if isinstance(selected, str):
            active_refs.append(selected)
    if len(active_refs) != len(components) or set(active_refs) != set(components):
        raise RuntimeError("live runtime 未加载完整 selection components")
    return {"selection_ref": root_ref,
            "active_selected": len(active_refs),
            "optional_health": "unverified"}


def docker_socket_gid() -> int:
    """Return the host group that may open the Docker socket."""

    socket = Path("/var/run/docker.sock")
    if not socket.is_socket():
        raise RuntimeError(f"Docker socket 不存在: {socket}")
    return socket.stat().st_gid


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


def _base_python_prefix(bridge_python: Path) -> Path:
    """Resolve the host Python tree required by persisted plugin environments."""

    result = subprocess.run(
        [
            str(bridge_python),
            "-I",
            "-S",
            "-c",
            "import sys; print(sys.base_prefix)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    raw_prefix = result.stdout.strip()
    prefix = Path(raw_prefix)
    if not raw_prefix or not prefix.is_absolute():
        raise RuntimeError("Bridge base Python prefix 不是绝对路径")
    resolved = prefix.resolve(strict=True)
    if not resolved.is_dir() or resolved == Path(resolved.anchor):
        raise RuntimeError("Bridge base Python prefix 不是可挂载目录")
    return resolved


def release_environment(
    *,
    paths: ReleasePaths,
    manifest: Mapping[str, object],
    current: Mapping[str, str],
    mise: Path,
) -> dict[str, str]:
    """Build runtime.env by replacing only release-owned generation fields."""

    commit = str(manifest["sourceCommit"])
    tree = str(manifest["sourceTree"])
    host_identity = manifest["hostToolchainIdentity"]
    if not isinstance(host_identity, Mapping):
        raise RuntimeError("release manifest 缺少 host toolchain identity")
    token = ensure_bridge_token(paths)
    bridge_python = paths.bridge_venv(commit) / "bin/python"
    values = dict(current)
    values.update(
        {
            "AKASHIC_BRIDGE_PYTHON": str(bridge_python),
            "AKASHIC_HOST_PYTHON_PREFIX": str(_base_python_prefix(bridge_python)),
            "AKASHIC_MISE": str(mise),
            "AKASHIC_RUNTIME_CHECKOUT": str(paths.source(commit)),
            "AKASHIC_RUNTIME_COMMIT": commit,
            "AKASHIC_RUNTIME_TREE": tree,
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
            "AKASHIC_WORKLOAD_RUNTIME_DIR": str(paths.state / "workload-runtime"),
            "AKASHIC_CONTAINER_NAME": values.get(
                "AKASHIC_CONTAINER_NAME", "akashic-core"
            ),
            "AKASHIC_WEB_BIND_ADDRESS": values.get(
                "AKASHIC_WEB_BIND_ADDRESS", "127.0.0.1"
            ),
            "AKASHIC_PUBLISHED_WEB_PORT": values.get(
                "AKASHIC_PUBLISHED_WEB_PORT", "2236"
            ),
            "AKASHIC_PUBLISHED_MOBILE_PORT": values.get(
                "AKASHIC_PUBLISHED_MOBILE_PORT", "6323"
            ),
            "AKASHIC_SERVICES_NETWORK": values.get(
                "AKASHIC_SERVICES_NETWORK", "akashic-services"
            ),
            "AKASHIC_WORKLOAD_NETWORK": values.get(
                "AKASHIC_WORKLOAD_NETWORK", "akashic-workloads"
            ),
            "AKASHIC_UID": values.get("AKASHIC_UID", str(os.getuid())),
            "AKASHIC_GID": values.get("AKASHIC_GID", str(os.getgid())),
            "AKASHIC_DOCKER_GID": str(docker_socket_gid()),
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
    upgrade: bool = False,
) -> str:
    """Activate one prepared generation; retain stopped state after data upgrade failure."""

    manifest = read_json(manifest_path)
    target = str(manifest["sourceCommit"])
    active_path = paths.activation / "active.json"
    previous = (
        read_json(active_path).get("targetCommit") if active_path.exists() else None
    )
    current = read_environment(environment_file) if environment_file.exists() else {}
    _verify_state_ready(paths)
    _prepare_workload_dirs(paths)
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
    upgrade_result: dict[str, object] | None = None
    if upgrade and previous is not None:
        backup_dir = paths.backups / f"upgrade-{target}-{timestamp}-{os.getpid()}"
        try:
            upgrade_result = _stopped_upgrade(
                paths=paths, candidate=candidate, manifest=manifest,
                backup_dir=backup_dir, previous_commit=str(previous), run=run,
            )
        except BaseException as error:
            failed = activation_receipt(
                status="maintenance_required", target_commit=target,
                previous_commit=str(previous), detail=str(error),
            )
            failed["backupDir"] = str(backup_dir)
            failed["dataCompatibility"] = "unproved"
            write_json(paths.activation / f"failed-{target}-{timestamp}.json", failed)
            raise RuntimeError("发行升级失败；旧 runtime 保持停止，先核对数据与恢复点") from error
    atomic_write(environment_file, render_environment(candidate))
    try:
        start_bridge(run=run)
        start_core(run=run)
        verify_release(environment_file)
        live_result = None
        if upgrade_result is not None:
            live_result = _verify_selected_runtime(
                candidate=candidate, root_ref=str(upgrade_result["new_root_ref"]), run=run,
            )
    except BaseException as error:
        maintenance_stop_detail = None
        try:
            stop_runtime(run=run)
        except BaseException as stop_error:
            maintenance_stop_detail = str(stop_error)
        if upgrade_result is not None:
            failed = activation_receipt(
                status="maintenance_required", target_commit=target,
                previous_commit=str(previous), detail=str(error),
            )
            failed["upgrade"] = upgrade_result
            failed["dataCompatibility"] = "unproved"
            if maintenance_stop_detail is not None:
                failed["maintenanceStopDetail"] = maintenance_stop_detail
            write_json(paths.activation / f"failed-{target}-{timestamp}.json", failed)
            if maintenance_stop_detail is not None:
                raise RuntimeError("候选可用性失败且停机未确认；禁止启动旧 runtime") from error
            raise RuntimeError("候选启动或可用性核对失败；旧 runtime 保持停止") from error
        if maintenance_stop_detail is not None:
            raise RuntimeError("首次激活失败且停机未确认") from error
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
    if upgrade_result is not None:
        receipt["upgrade"] = upgrade_result
        receipt["runtimeCheck"] = live_result
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


def _prepare_workload_dirs(paths: ReleasePaths) -> None:
    """Create only the fixed bind roots used by the Workload Controller."""

    roots = (
        paths.state / "workspace/plugin-data",
        paths.state / "workspace/runtime/plugin-validation",
        paths.state / "workload-runtime",
        paths.state / "workload-controller",
    )
    for path in roots:
        current = paths.state
        for part in path.relative_to(paths.state).parts:
            current /= part
            if current.is_symlink():
                raise RuntimeError(f"Workload bind root 不得穿过 symlink: {current}")
            current.mkdir(exist_ok=True)
            if not current.is_dir():
                raise RuntimeError(f"Workload bind root 不是目录: {current}")
