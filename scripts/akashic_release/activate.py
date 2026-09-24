from __future__ import annotations

import os
import hashlib
import json
import re
import secrets
import shlex
import stat
import sqlite3
from contextlib import closing
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, NoReturn

from scripts.akashic_release.doctor import read_environment, verify_release
from scripts.akashic_release.manifest import activation_receipt, atomic_write, read_json
from scripts.akashic_release.manifest import write_json
from scripts.akashic_release.model import ReleasePaths
from scripts.akashic_release.systemd import start_bridge, start_core, stop_runtime
from agent.plugins.selection import PluginSelection
from agent.plugins.reload_journal import ReloadJournal
from agent.migrations.release_backup import _RUNTIME_FILES
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceMaintenanceLock

Run = Callable[..., subprocess.CompletedProcess[str]]
_ROOT_REF = re.compile(r"[0-9a-f]{64}\Z")


def _plain_external_host(root: Path, plan: Path) -> tuple[Path, Path]:
    """Keep the host mount and plan path free of link traversal."""

    root = root.absolute()
    plan = plan.absolute()
    for path, want_directory in ((root, True), (plan, False)):
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"external host 路径不能穿过符号链接: {current}")
        mode = path.lstat().st_mode
        if want_directory != stat.S_ISDIR(mode):
            raise ValueError(f"external host 路径类型不符: {path}")
    if not plan.is_relative_to(root) or plan == root:
        raise ValueError("external plan 必须位于 input root 内")
    return root, plan


def _save_environment_backup(source: Path, backup: Path) -> str:
    """Publish and read back exact runtime.env bytes before stopped mutation."""

    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"runtime.env backup 已存在: {backup}")
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    backup.parent.mkdir(parents=True, exist_ok=True)
    with backup.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(backup.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"runtime.env backup 字节核对失败: {backup}")
    return digest


def _state_entries(root: Path) -> tuple[set[str], set[str]]:
    """List every retained file/link and directory without following links."""

    files: set[str] = set()
    directories: set[str] = set()
    for current, names, leaves in os.walk(root, followlinks=False):
        folder = Path(current)
        directories.add(folder.relative_to(root).as_posix())
        for name in list(names):
            path = folder / name
            relative = path.relative_to(root)
            if path.is_symlink():
                files.add(relative.as_posix())
                names.remove(name)
        for name in leaves:
            path = folder / name
            relative = path.relative_to(root)
            if relative not in _RUNTIME_FILES:
                files.add(relative.as_posix())
    return files, directories


def _verify_full_restore(paths: ReleasePaths, failed: Mapping[str, object], environment_file: Path) -> dict[str, object]:
    """Compare the stopped state, SQLite and env to one complete saved point."""

    # 1. Fix the referenced backup and the previous environment bytes.
    backup_raw = failed.get("backupDir")
    env_raw = failed.get("environmentBackup")
    env_digest = failed.get("environmentBackupSha256")
    if (not isinstance(backup_raw, str) or not isinstance(env_raw, str)
        or not isinstance(env_digest, str) or _ROOT_REF.fullmatch(env_digest) is None):
        raise ValueError("failure 缺少完整 state/env 恢复链接")
    backup = Path(backup_raw)
    env_backup = Path(env_raw)
    if (paths.state.is_symlink() or not paths.state.is_dir()
        or environment_file.is_symlink() or not environment_file.is_file()):
        raise ValueError("restored state/runtime.env 路径无效")
    backup_root = paths.backups.resolve(strict=True)
    if (backup.is_symlink() or not backup.is_dir() or backup.resolve(strict=True) != backup
        or not backup.is_relative_to(backup_root) or not backup.name.startswith("upgrade-")
        or env_backup.is_symlink() or not env_backup.is_file()
        or not env_backup.resolve(strict=True).is_relative_to(backup_root)):
        raise ValueError("failure 恢复链接路径无效")
    manifest_path = backup / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("release backup manifest 路径无效")
    manifest = read_json(manifest_path)
    snapshot = backup / "state"
    if (manifest.get("version") != 1 or manifest.get("source") != str(paths.state)
        or manifest.get("backup") != str(snapshot) or snapshot.is_symlink()
        or not snapshot.is_dir()):
        raise ValueError("release backup manifest 与目标 state 不匹配")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("release backup file list 无效")
    expected_paths: set[str] = set()
    sqlite_count = 0
    # 2. Check every saved file against both the backup and restored state.
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("release backup file record 无效")
        relative = raw.get("path")
        kind = raw.get("kind")
        if (not isinstance(relative, str) or relative.startswith("/")
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or relative in expected_paths):
            raise ValueError("release backup path 重复或越界")
        expected_paths.add(relative)
        source = snapshot / relative
        current = paths.state / relative
        if kind == "symlink":
            if (not source.is_symlink() or not current.is_symlink()
                or os.readlink(source) != raw.get("target")
                or os.readlink(current) != raw.get("target")):
                raise RuntimeError(f"state symlink 未完整恢复: {relative}")
            continue
        digest = raw.get("sha256")
        if (kind not in {"file", "sqlite_logical"} or not isinstance(digest, str)
            or _ROOT_REF.fullmatch(digest) is None):
            raise ValueError(f"release backup 记录无效: {relative}")
        for path in (source, current):
            if (path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise RuntimeError(f"state 字节未完整恢复: {relative}")
        if kind == "sqlite_logical":
            with closing(sqlite3.connect(f"{current.as_uri()}?mode=ro&immutable=1", uri=True)) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise RuntimeError(f"restored SQLite 损坏: {relative}")
            sqlite_count += 1
    # 3. Refuse partial restores with extra or missing paths.
    source_paths, source_dirs = _state_entries(snapshot)
    current_paths, current_dirs = _state_entries(paths.state)
    if (source_paths != expected_paths or current_paths != expected_paths
        or source_dirs != current_dirs):
        raise RuntimeError(
            "state 路径集合未完整恢复: "
            f"backup_extra={sorted(source_paths - expected_paths)[:5]} "
            f"current_extra={sorted(current_paths - expected_paths)[:5]} "
            f"current_missing={sorted(expected_paths - current_paths)[:5]} "
            f"dir_extra={sorted(current_dirs - source_dirs)[:5]}"
        )
    if (hashlib.sha256(env_backup.read_bytes()).hexdigest() != env_digest
        or hashlib.sha256(environment_file.read_bytes()).hexdigest() != env_digest):
        raise RuntimeError("runtime.env 未完整恢复")
    return {"manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "fileCount": len(records), "sqliteCount": sqlite_count,
            "environmentSha256": env_digest}


def _settlement_path(paths: ReleasePaths, failed_path: Path) -> Path:
    return paths.activation / f"settled-{failed_path.name}"


def failure_settled(paths: ReleasePaths, failed_path: Path) -> bool:
    settlement_path = _settlement_path(paths, failed_path)
    if not settlement_path.is_file() or settlement_path.is_symlink():
        return False
    settlement = read_json(settlement_path)
    return (settlement.get("status") == "verified_full_restore"
            and settlement.get("failureSha256") == hashlib.sha256(failed_path.read_bytes()).hexdigest())


def _attempt_settled(paths: ReleasePaths, attempt_path: Path) -> bool:
    for failed_path in paths.activation.glob("failed-*.json"):
        if (failure_settled(paths, failed_path)
            and read_json(failed_path).get("attemptPath") == str(attempt_path)):
            settlement = read_json(_settlement_path(paths, failed_path))
            if settlement.get("attemptSha256") == hashlib.sha256(attempt_path.read_bytes()).hexdigest():
                return True
    return False


def _verified_active_attempt(
    *, paths: ReleasePaths, attempt_path: Path, attempt: Mapping[str, object],
    active: Mapping[str, object], environment_file: Path,
    current: Mapping[str, str], run: Run,
) -> tuple[str, ...] | None:
    """Bind a pending attempt to the current active receipt and live Root."""

    # 1. A missing fact cannot match another missing fact across two receipts.
    upgrade = active.get("upgrade")
    required_attempt = ("targetCommit", "imageId", "externalPlanSha256", "oldRootRef",
                        "backupDir", "environmentBackup", "environmentBackupSha256")
    if (attempt_path.is_symlink() or not attempt_path.is_file()
        or attempt_path.parent != paths.activation
        or attempt.get("status") not in {"pending", "active"}
        or any(not isinstance(attempt.get(key), str) or not attempt[key]
               for key in required_attempt)
        or active.get("status") != "active" or not isinstance(upgrade, dict)
        or active.get("attemptPath") != str(attempt_path)
        or active.get("targetCommit") != attempt.get("targetCommit")
        or active.get("imageId") != attempt.get("imageId")
        or active.get("environmentBackup") != attempt.get("environmentBackup")
        or active.get("environmentBackupSha256") != attempt.get("environmentBackupSha256")
        or upgrade.get("external_plan_sha256") != attempt.get("externalPlanSha256")
        or upgrade.get("old_root_ref") != attempt.get("oldRootRef")
        or upgrade.get("backup_dir") != attempt.get("backupDir")):
        return None
    # 2. Check current durable selection, then prove this runtime is still healthy.
    selected = PluginSelection(paths.state / "workspace")
    root_ref = selected.read()
    if root_ref is None or upgrade.get("new_root_ref") != root_ref:
        return None
    components = selected.archive.read_descriptor(root_ref).get("components")
    live_receipt = active.get("runtimeCheck")
    if (not isinstance(components, tuple)
        or upgrade.get("ordered_components") != list(components)
        or not isinstance(live_receipt, dict)
        or live_receipt.get("selection_ref") != root_ref
        or type(live_receipt.get("active_selected")) is not int
        or live_receipt["active_selected"] != len(components)):
        return None
    verify_release(environment_file)
    _verify_selected_runtime(candidate=current, root_ref=root_ref, run=run,
                             ordered_components=components)
    return components


def _completed_attempt(
    attempt: Mapping[str, object], active: Mapping[str, object],
    components: tuple[str, ...], active_path: Path,
) -> dict[str, object]:
    """Keep a historical terminal copy derived from one committed active receipt."""

    upgrade = active["upgrade"]
    if not isinstance(upgrade, dict):
        raise RuntimeError("active receipt 缺少 upgrade")
    return {**attempt, "status": "active", "newRootRef": upgrade["new_root_ref"],
            "orderedComponents": list(components), "runtimeCheck": active["runtimeCheck"],
            "activeReceiptSha256": hashlib.sha256(active_path.read_bytes()).hexdigest()}


def settle_restored_failure(
    *, paths: ReleasePaths, failed_path: Path, environment_file: Path, run: Run,
) -> dict[str, object]:
    """Settle a pre-start failure only after exact restore and owner checks."""

    # 1. Bind this operation to one pre-start failure and its durable attempt.
    if (failed_path.is_symlink() or not failed_path.is_file()
        or failed_path.parent.resolve(strict=True) != paths.activation.resolve(strict=True)
        or not failed_path.name.startswith("failed-")):
        raise ValueError("failure receipt 必须是本 release 的普通文件")
    failed = read_json(failed_path)
    phase = failed.get("phase")
    if (failed.get("status") != "maintenance_required"
        or phase not in {"stopped_upgrade", "before_target_start"}
        or (phase == "before_target_start" and failed.get("targetStarted") is not False)):
        raise ValueError("只有尚未启动目标 runtime 的 external failure 可由完整恢复结算")
    attempt_raw = failed.get("attemptPath")
    if not isinstance(attempt_raw, str):
        raise ValueError("failure 缺少 external attempt")
    attempt_path = Path(attempt_raw)
    if (attempt_path.is_symlink() or not attempt_path.is_file()
        or attempt_path.parent.resolve(strict=True) != paths.activation.resolve(strict=True)):
        raise ValueError("external attempt 路径无效")
    attempt = read_json(attempt_path)
    if (attempt.get("status") != "pending" or attempt.get("backupDir") != failed.get("backupDir")
        or attempt.get("targetCommit") != failed.get("targetCommit")):
        raise ValueError("failure 与 pending attempt 不匹配")
    for unit in ("akashic-core.service", "akashic-host-bridge.service"):
        state = run(["systemctl", "is-active", unit], check=False, capture_output=True, text=True)
        if state.stdout.strip() not in {"inactive", "failed"}:
            raise RuntimeError(f"settlement 需要确认服务已停止: {unit}")
    # 2. Check the full restored state under the stopped writer locks.
    workspace = paths.state / "workspace"
    home = paths.state / "plugin-home"
    maintenance = WorkspaceMaintenanceLock(workspace)
    maintenance.acquire()
    try:
        publication = PluginPublicationLock(home)
        publication.acquire()
        try:
            verified = _verify_full_restore(paths, failed, environment_file)
            old_root = attempt.get("oldRootRef")
            if not isinstance(old_root, str) or PluginSelection(workspace).read() != old_root:
                raise RuntimeError("restored Root 与原 attempt 不一致")
            active = read_json(paths.activation / "active.json")
            if active.get("targetCommit") != failed.get("previousCommit"):
                raise RuntimeError("旧 active release 与恢复点不一致")
            with ReloadJournal.inspect_existing(workspace) as journal:
                if journal.pending_recovery or journal.armed_updates:
                    raise RuntimeError("reload owner 尚有 pending/armed 记录")
            # 3. Preserve the original failure and add one separate settlement.
            result = {"status": "verified_full_restore", "failure": str(failed_path),
                      "failureSha256": hashlib.sha256(failed_path.read_bytes()).hexdigest(),
                      "attempt": str(attempt_path), "oldRootRef": old_root,
                      "attemptSha256": hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
                      "backupDir": failed["backupDir"], **verified}
            target = _settlement_path(paths, failed_path)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"settlement 已存在: {target}")
            write_json(target, result)
            return result
        finally:
            publication.release()
    finally:
        maintenance.release()


def _stopped_upgrade(
    *, paths: ReleasePaths, candidate: Mapping[str, str], manifest: Mapping[str, object],
    backup_dir: Path, previous_commit: str, run: Run,
    external_plan: Path | None = None, external_inputs: Path | None = None,
    preflight_only: bool = False,
    expected_plan_sha256: str | None = None,
) -> dict[str, object]:
    """Run the target image's own upgrade code against stopped state."""

    stable = read_json(paths.state / "workspace/runtime/plugin-stable.json")
    root_ref = stable.get("root_ref")
    if stable.get("version") != 1 or not isinstance(root_ref, str) or _ROOT_REF.fullmatch(root_ref) is None:
        raise RuntimeError("release upgrade 需要已有完整 stable Root")
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,mode=1777,size=4g",
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
    if external_plan is not None and external_inputs is not None:
        if (expected_plan_sha256 is None
            or hashlib.sha256(external_plan.read_bytes()).hexdigest() != expected_plan_sha256):
            raise RuntimeError("external plan 在发布期间变化")
        relative = external_plan.relative_to(external_inputs)
        mount = "/opt/akashic/external-inputs"
        command[command.index(str(manifest["imageId"])):command.index(str(manifest["imageId"]))] = [
            "--mount", f"type=bind,src={external_inputs},dst={mount},readonly",
        ]
        command.extend(["--external-plan", f"{mount}/{relative.as_posix()}",
                        "--external-inputs", mount])
    if preflight_only:
        command.append("--preflight-only")
    result = run(command, check=True, capture_output=True, text=True)
    try:
        receipt = json.loads(result.stdout)
    except (TypeError, ValueError) as error:
        raise RuntimeError("目标 image 未返回 upgrade JSON") from error
    if not isinstance(receipt, dict) or receipt.get("status") not in {
        "selected_not_started", "partial_selected_not_started",
        "already_selected_not_started", "no_eligible_targets", "preflight_ok",
    }:
        raise RuntimeError(f"目标 image upgrade 结果无效: {receipt}")
    if preflight_only:
        if (receipt.get("old_root_ref") != root_ref or external_plan is None
            or receipt.get("external_plan_sha256") != expected_plan_sha256):
            raise RuntimeError("目标 image external preflight 身份不一致")
        return receipt
    if receipt["status"] == "preflight_ok":
        raise RuntimeError("目标 image 仅返回 preflight，未执行 upgrade")
    selected = receipt.get("new_root_ref")
    if (receipt.get("old_root_ref") != root_ref or not isinstance(selected, str)
        or _ROOT_REF.fullmatch(selected) is None
        or receipt.get("backup_dir") != str(backup_dir)
        or not (backup_dir / "manifest.json").is_file()):
        raise RuntimeError("目标 image upgrade 缺少完整 Root 或已校验恢复点")
    if external_plan is not None:
        if receipt.get("external_plan_sha256") != expected_plan_sha256:
            raise RuntimeError("目标 image upgrade external plan 身份不一致")
    return receipt


def _verify_selected_runtime(
    *, candidate: Mapping[str, str], root_ref: str, run: Run,
    ordered_components: tuple[str, ...] | None = None,
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
    if ordered_components is not None and tuple(components) != ordered_components:
        raise RuntimeError("live runtime 完整 ordered selection 与发布输入不一致")
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
    external_plan: Path | None = None,
    external_inputs: Path | None = None,
) -> str:
    """Activate one prepared generation; retain stopped state after data upgrade failure."""

    manifest = read_json(manifest_path)
    target = str(manifest["sourceCommit"])
    active_path = paths.activation / "active.json"
    previous = (
        read_json(active_path).get("targetCommit") if active_path.exists() else None
    )
    current = read_environment(environment_file) if environment_file.exists() else {}
    if (external_plan is None) != (external_inputs is None):
        raise ValueError("external plan 与 input root 必须同时提供")
    for failed_path in sorted(paths.activation.glob("failed-*.json")):
        if (read_json(failed_path).get("status") == "maintenance_required"
            and not failure_settled(paths, failed_path)):
            stop_runtime(run=run)
            raise RuntimeError(f"未结算 release failure 阻止新尝试: {failed_path}")
    plan_digest: str | None = None
    active = read_json(active_path) if active_path.exists() else None
    if external_plan is not None and external_inputs is not None:
        if previous is None:
            raise ValueError("external plan 需要已有 active release 与完整 Root")
        if environment_file.is_symlink() or not environment_file.is_file():
            raise ValueError("external plan 需要已有普通 runtime.env")
        external_inputs, external_plan = _plain_external_host(external_inputs, external_plan)
        plan_digest = hashlib.sha256(external_plan.read_bytes()).hexdigest()
        plan = read_json(external_plan)
        stable = read_json(paths.state / "workspace/runtime/plugin-stable.json")
        if previous == target and active is not None:
            previous_upgrade = active.get("upgrade")
            if (active.get("imageId") == manifest.get("imageId")
                and isinstance(previous_upgrade, dict)
                and previous_upgrade.get("external_plan_sha256") == plan_digest
                and previous_upgrade.get("old_root_ref") == plan.get("expected_root_ref")):
                attempt_raw = active.get("attemptPath")
                if isinstance(attempt_raw, str):
                    replay_attempt = Path(attempt_raw)
                    if (replay_attempt.parent == paths.activation
                        and replay_attempt.is_file() and not replay_attempt.is_symlink()
                        and _verified_active_attempt(
                            paths=paths, attempt_path=replay_attempt,
                            attempt=read_json(replay_attempt), active=active,
                            environment_file=environment_file, current=current, run=run,
                        ) is not None):
                        for other in paths.activation.glob("attempt-external-*.json"):
                            if (other != replay_attempt
                                and read_json(other).get("status") == "pending"
                                and not _attempt_settled(paths, other)):
                                stop_runtime(run=run)
                                raise RuntimeError(f"其他不完整 external attempt 需人工结算: {other}")
                        return "already_active"
        if plan.get("expected_root_ref") != stable.get("root_ref"):
            raise RuntimeError("external plan expected_root_ref 与当前 selection 不一致")
    for attempt_path in sorted(paths.activation.glob("attempt-external-*.json")):
        attempt = read_json(attempt_path)
        if attempt.get("status") != "pending" or _attempt_settled(paths, attempt_path):
            continue
        try:
            components = None if active is None else _verified_active_attempt(
                paths=paths, attempt_path=attempt_path, attempt=attempt,
                active=active, environment_file=environment_file, current=current, run=run,
            )
        except BaseException as error:
            stop_runtime(run=run)
            raise RuntimeError(f"active attempt 复核失败，保持 maintenance: {attempt_path}") from error
        if components is None:
            stop_runtime(run=run)
            raise RuntimeError(f"不完整 external release attempt 需人工结算: {attempt_path}")
        # 当前 active 是成功权威；先把它的历史副本写回原 attempt，再允许下一计划覆盖 active。
        assert active is not None
        write_json(attempt_path, _completed_attempt(attempt, active, components, active_path))
    _verify_state_ready(paths)
    if external_plan is None:
        _prepare_workload_dirs(paths)
    candidate = release_environment(
        paths=paths,
        manifest=manifest,
        current=current,
        mise=mise,
    )
    if previous == target and external_plan is None:
        verify_release(environment_file)
        return "already_active"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = paths.backups / f"runtime.env.before-{target}-{timestamp}"
    environment_digest: str | None = None
    if environment_file.exists():
        environment_digest = _save_environment_backup(environment_file, backup)
    stop_runtime(run=run)
    upgrade_result: dict[str, object] | None = None
    attempt_path: Path | None = None
    if upgrade and previous is not None:
        backup_dir = paths.backups / f"upgrade-{target}-{timestamp}-{os.getpid()}"
        try:
            if external_plan is not None:
                _ = _stopped_upgrade(
                    paths=paths, candidate=candidate, manifest=manifest,
                    backup_dir=backup_dir, previous_commit=str(previous), run=run,
                    external_plan=external_plan, external_inputs=external_inputs,
                    preflight_only=True, expected_plan_sha256=plan_digest,
                )
                attempt_path = paths.activation / f"attempt-external-{target}-{timestamp}-{secrets.token_hex(4)}.json"
                write_json(attempt_path, {"status": "pending", "phase": "before_upgrade",
                                          "targetCommit": target, "imageId": manifest["imageId"],
                                          "externalPlanSha256": plan_digest,
                                          "oldRootRef": read_json(paths.state / "workspace/runtime/plugin-stable.json")["root_ref"],
                                          "backupDir": str(backup_dir),
                                          "environmentBackup": str(backup),
                                          "environmentBackupSha256": environment_digest})
            upgrade_result = _stopped_upgrade(
                paths=paths, candidate=candidate, manifest=manifest,
                backup_dir=backup_dir, previous_commit=str(previous), run=run,
                external_plan=external_plan, external_inputs=external_inputs,
                expected_plan_sha256=plan_digest,
            )
        except BaseException as error:
            if external_plan is not None and attempt_path is None:
                try:
                    start_bridge(run=run)
                    start_core(run=run)
                    verify_release(environment_file)
                except BaseException as recovery_error:
                    failed = activation_receipt(
                        status="maintenance_required", target_commit=target,
                        previous_commit=str(previous), detail=str(error),
                    )
                    failed["phase"] = "external_preflight_recovery"
                    failed["recoveryDetail"] = str(recovery_error)
                    write_json(paths.activation / f"failed-{target}-{timestamp}.json", failed)
                    raise RuntimeError("external preflight 冲突且旧 runtime 恢复失败") from recovery_error
                failed = activation_receipt(
                    status="preflight_conflict", target_commit=target,
                    previous_commit=str(previous), detail=str(error),
                )
                failed["phase"] = "external_preflight"
                write_json(paths.activation / f"failed-{target}-{timestamp}.json", failed)
                raise RuntimeError("external preflight 冲突，旧 runtime 已恢复") from error
            failed = activation_receipt(
                status="maintenance_required", target_commit=target,
                previous_commit=str(previous), detail=str(error),
            )
            failed["backupDir"] = str(backup_dir)
            failed["phase"] = "stopped_upgrade"
            failed["targetStarted"] = False
            failed["environmentBackup"] = str(backup)
            failed["environmentBackupSha256"] = environment_digest
            failed["attemptPath"] = None if attempt_path is None else str(attempt_path)
            failed["dataCompatibility"] = "unproved"
            write_json(paths.activation / f"failed-{target}-{timestamp}.json", failed)
            raise RuntimeError("发行升级失败；旧 runtime 保持停止，先核对数据与恢复点") from error
    try:
        if external_plan is not None:
            _prepare_workload_dirs(paths)
        atomic_write(environment_file, render_environment(candidate))
    except BaseException as error:
        if external_plan is None or upgrade_result is None or attempt_path is None:
            raise
        failed = activation_receipt(
            status="maintenance_required", target_commit=target,
            previous_commit=str(previous), detail=str(error),
        )
        failed.update({"phase": "before_target_start", "targetStarted": False,
                       "upgrade": upgrade_result, "backupDir": str(backup_dir),
                       "attemptPath": str(attempt_path),
                       "environmentBackup": str(backup),
                       "environmentBackupSha256": environment_digest,
                       "dataCompatibility": "unproved"})
        try:
            write_json(paths.activation / f"failed-{target}-{timestamp}.json", failed)
        except BaseException as receipt_error:
            raise RuntimeError(
                f"目标启动前写入失败: {error!r}; failure receipt 写入也失败: {receipt_error!r}"
            ) from error
        raise RuntimeError("目标启动前写入失败；旧 runtime 保持停止，先核对完整恢复点") from error
    try:
        start_bridge(run=run)
        start_core(run=run)
        verify_release(environment_file)
        live_result = None
        if upgrade_result is not None:
            expected_components = upgrade_result.get("ordered_components")
            if external_plan is not None and (not isinstance(expected_components, list)
                                              or any(not isinstance(ref, str) for ref in expected_components)):
                raise RuntimeError("upgrade 未报告完整 ordered components")
            live_result = _verify_selected_runtime(
                candidate=candidate, root_ref=str(upgrade_result["new_root_ref"]), run=run,
                ordered_components=tuple(expected_components) if isinstance(expected_components, list) else None,
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
            failed["phase"] = "target_start_or_readiness"
            failed["attemptPath"] = None if attempt_path is None else str(attempt_path)
            failed["environmentBackup"] = str(backup)
            failed["environmentBackupSha256"] = environment_digest
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
        receipt["imageId"] = manifest["imageId"]
        receipt["attemptPath"] = None if attempt_path is None else str(attempt_path)
        receipt["environmentBackup"] = str(backup)
        receipt["environmentBackupSha256"] = environment_digest
    try:
        write_json(paths.activation / "active.json", receipt)
    except BaseException as error:
        if upgrade_result is None:
            raise
        maintenance_stop_detail = None
        try:
            stop_runtime(run=run)
        except BaseException as stop_error:
            maintenance_stop_detail = str(stop_error)
        failed = activation_receipt(
            status="maintenance_required", target_commit=target,
            previous_commit=None if previous is None else str(previous), detail=str(error),
        )
        failed.update({"phase": "active_receipt", "upgrade": upgrade_result,
                       "attemptPath": None if attempt_path is None else str(attempt_path),
                       "environmentBackup": str(backup),
                       "environmentBackupSha256": environment_digest,
                       "dataCompatibility": "unproved"})
        if maintenance_stop_detail is not None:
            failed["maintenanceStopDetail"] = maintenance_stop_detail
        write_json(paths.activation / f"failed-{target}-{timestamp}.json", failed)
        raise RuntimeError("active receipt 发布失败；目标 runtime 已停在 maintenance") from error
    if attempt_path is not None:
        attempt = read_json(attempt_path)
        components = upgrade_result.get("ordered_components") if upgrade_result else None
        if not isinstance(components, list) or any(not isinstance(ref, str) for ref in components):
            raise RuntimeError("active attempt 缺少完整 ordered components")
        write_json(attempt_path, _completed_attempt(attempt, receipt, tuple(components), active_path))
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
