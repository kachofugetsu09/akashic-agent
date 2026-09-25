from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Callable, Mapping

from scripts.akashic_release.doctor import read_environment, release_health_timeout, verify_release
from scripts.akashic_release.manifest import activation_receipt, atomic_write, read_json, write_json
from scripts.akashic_release.model import ReleasePaths
from scripts.akashic_release.systemd import install_units, install_operator_entrypoint
from scripts.akashic_release.systemd import start_bridge, start_core, stop_runtime

Run = Callable[..., subprocess.CompletedProcess[str]]


def _verify_selected_runtime(
    *, candidate: Mapping[str, str], root_ref: str, run: Run,
    ordered_components: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Read the live runtime's exact selected Fiber identities after health."""

    deadline = time.monotonic() + release_health_timeout(candidate)
    while True:
        result = run([
            "docker", "exec", candidate["AKASHIC_CONTAINER_NAME"],
            "/opt/venv/bin/python", "/opt/akashic/source/main.py", "plugin-status",
            "--config", candidate["AKASHIC_CONFIG"],
            "--workspace", candidate["AKASHIC_WORKSPACE"],
        ], check=False, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            break
        if (time.monotonic() >= deadline or not any(message in result.stderr for message in (
            "[Errno 111] Connection refused", "[Errno 2] No such file or directory",
        ))):
            raise RuntimeError(f"runtime control 未就绪: {result.stderr.strip()}")
        time.sleep(1)
    status = json.loads(result.stdout)
    if not isinstance(status, dict) or status.get("selection_ref") != root_ref:
        raise RuntimeError("live runtime selection 与已发布的完整 Root 不一致")
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


def _publish(
    *, paths: ReleasePaths, candidate: Mapping[str, str], plan: Path,
    inputs: Path, digest: str, backup_dir: Path | None, preflight: bool, run: Run,
) -> dict[str, object]:
    """让目标镜像检查或执行同一份固定部署清单。"""

    # 1. 输入只读挂载；在线预检也将正式 state 设为只读。
    if hashlib.sha256(plan.read_bytes()).hexdigest() != digest:
        raise RuntimeError("部署清单在预检后改变")
    command = ["docker", "run", "--rm", "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,mode=1777,size=4g",
               "--mount", f"type=bind,src={paths.state},dst={paths.state}" + (",readonly" if preflight else ""),
               "--mount", f"type=bind,src={plan},dst=/opt/akashic/deploy-plan.json,readonly",
               "--mount", f"type=bind,src={inputs},dst=/opt/akashic/deploy-inputs,readonly"]
    if backup_dir is not None and not preflight:
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        command += ["--mount", f"type=bind,src={backup_dir.parent},dst={backup_dir.parent}"]
    for key in ("AKASHIC_CONFIG", "AKASHIC_WORKSPACE", "AKASHIC_PLUGIN_HOME",
                "AKASHIC_RUNTIME_COMMIT", "AKASHIC_RUNTIME_TREE"):
        command += ["--env", f"{key}={candidate[key]}"]
    command += [candidate["AKASHIC_IMAGE"], "publish", "--plan", "/opt/akashic/deploy-plan.json",
                "--inputs", "/opt/akashic/deploy-inputs"]
    if preflight:
        command.append("--preflight-only")
    elif backup_dir is not None:
        command += ["--backup-dir", str(backup_dir)]
    completed = run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"目标镜像发布失败（exit={completed.returncode}）: {completed.stderr.strip()}")
    result = json.loads(completed.stdout)
    # 2. 容器输出只确认本次清单；Root 仍由持久选择和运行时共同核对。
    if (not isinstance(result, dict) or result.get("plan_sha256") != digest
        or result.get("old_root_ref") != read_json(plan)["expected_root_ref"]
        or result.get("status") != ("preflight_ok" if preflight else "selected_not_started")):
        raise RuntimeError("目标镜像返回了不匹配的发布结果")
    return result


def _read_selection(*, paths: ReleasePaths, candidate: Mapping[str, str], run: Run) -> dict[str, object]:
    """在目标镜像内调用选择 owner；宿主 bootstrap 只依赖标准库。"""
    code = (
        "import json,sys; from pathlib import Path; "
        "from agent.plugins.selection import PluginSelection; "
        "s=PluginSelection(Path(sys.argv[1])); r=s.read(); "
        "print(json.dumps({'root_ref':r,'components':list(s.archive.read_descriptor(r)['components']) if r else []}))"
    )
    result = run([
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--mount", f"type=bind,src={paths.state},dst={paths.state},readonly",
        "--workdir", "/opt/akashic/source",
        "--entrypoint", "/opt/venv/bin/python", candidate["AKASHIC_IMAGE"],
        "-B", "-c", code, candidate["AKASHIC_WORKSPACE"],
    ], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("selection owner 返回无效结果")
    return value


def _check_publication(
    *, paths: ReleasePaths, candidate: Mapping[str, str], publication: Mapping[str, object], run: Run,
) -> tuple[str, tuple[str, ...]]:
    """将发布回执绑定到当前完整 Root，拒绝恢复过期尝试。"""
    selected = _read_selection(paths=paths, candidate=candidate, run=run)
    root_ref, components = selected["root_ref"], selected["components"]
    if not isinstance(root_ref, str) or root_ref != publication.get("new_root_ref"):
        raise RuntimeError("当前 Root 与本次发布结果不同；请重新核对部署清单")
    if not isinstance(components, list) or components != publication.get("ordered_components"):
        raise RuntimeError("当前 Root components 与发布结果不同")
    return root_ref, tuple(components)


def _start_and_record(
    *, paths: ReleasePaths, environment_file: Path, candidate: Mapping[str, str],
    attempt_path: Path, attempt: dict[str, object], run: Run, unit_root: Path, cli_path: Path,
) -> None:
    """启动已发布版本，核对实际消费者后才更新 active。"""

    # 1. 恢复启动也只接受已记录且仍被选择的完整 Root。
    publication = attempt["publication"]
    root_ref = None
    components = None
    if isinstance(publication, dict):
        root_ref, components = _check_publication(paths=paths, candidate=candidate, publication=publication, run=run)
    elif publication is not None:
        raise ValueError("部署回执 publication 无效")
    attempt["phase"] = "starting"
    write_json(attempt_path, attempt)
    checkout = paths.source(str(attempt["targetCommit"]))
    backup_root = paths.backups if attempt["backupDir"] is not None else None
    install_units(checkout=checkout, backup_root=backup_root, run=run,
                  unit_root=unit_root, runtime_env=environment_file)
    install_operator_entrypoint(checkout=checkout, backup_root=backup_root, target=cli_path)
    _prepare_workload_dirs(paths)
    atomic_write(environment_file, render_environment(candidate))
    start_bridge(run=run)
    start_core(run=run)
    verify_release(environment_file)
    if root_ref is None:
        # 首次安装 profile 只在首次启动建立选择；升级不会再次应用它。
        root_ref = _read_selection(paths=paths, candidate=candidate, run=run)["root_ref"]
        if not isinstance(root_ref, str):
            raise RuntimeError("首次启动尚未发布 Root")
    live = _verify_selected_runtime(candidate=candidate, root_ref=root_ref,
                                    ordered_components=components, run=run)
    # 2. active 是成功边界；写入后不再执行可能将健康服务停掉的收尾动作。
    attempt.update(status="verified", phase="verified", runtimeCheck=live)
    write_json(attempt_path, attempt)
    receipt = {**attempt, "status": "active", "phase": "active"}
    active = paths.activation / "active.json"
    if active.exists():
        write_json(paths.activation / "previous.json", read_json(active))
    write_json(active, receipt)


def _record_failure(
    *, paths: ReleasePaths, attempt_path: Path, attempt: dict[str, object],
    error: BaseException, run: Run,
) -> None:
    """保持停止并保存真实阶段，不声称数据或外部效果已经回滚。"""
    attempt.update(status="maintenance_required", detail=str(error))
    try:
        stop_runtime(run=run)
    except (subprocess.CalledProcessError, OSError) as stop_error:
        attempt["stopError"] = str(stop_error)
    write_json(attempt_path, attempt)


def activate_release(
    *, paths: ReleasePaths, manifest_path: Path, environment_file: Path,
    mise: Path, run: Run, plan: Path | None = None,
    inputs: Path | None = None, backup: bool = False,
    unit_root: Path = Path("/etc/systemd/system"), cli_path: Path = Path.home() / ".local/bin/akashic-release",
) -> str:
    """按部署者清单停机发布；默认只更新 Core/Bridge，不执行待迁移。"""

    # 1. 先固定目标与在线预检；任何预检失败都不停止现有服务。
    manifest = read_json(manifest_path)
    target = str(manifest["sourceCommit"])
    current = read_environment(environment_file) if environment_file.exists() else {}
    previous = current.get("AKASHIC_RUNTIME_COMMIT")
    _verify_state_ready(paths)
    candidate = release_environment(paths=paths, manifest=manifest, current=current, mise=mise)
    selection_path = paths.state / "workspace/runtime/plugin-stable.json"
    root_ref = (_read_selection(paths=paths, candidate=candidate, run=run)["root_ref"]
                if selection_path.exists() or previous is not None else None)
    attempt_path = paths.activation / f"deploy-{secrets.token_hex(12)}.json"
    fixed_plan = attempt_path.with_suffix(".plan.json")
    if root_ref is None:
        if previous is not None or plan is not None:
            raise RuntimeError("既有部署或显式清单必须有完整 Root；先修复正式选择")
        if backup:
            raise ValueError("首次初始化的 state 备份请由部署者在安装前完成")
    else:
        if plan is None:
            write_json(fixed_plan, {"schema_version": 1, "expected_root_ref": root_ref,
                                    "targets": [], "migrations": []})
        else:
            if plan.is_symlink() or not plan.is_file():
                raise ValueError("部署清单必须是普通文件")
            atomic_write(fixed_plan, plan.read_text(encoding="utf-8"))
        if inputs is None:
            inputs = fixed_plan.parent
        inputs = inputs.resolve(strict=True)
        if not inputs.is_dir():
            raise ValueError("--inputs 必须是目录")
        digest = hashlib.sha256(fixed_plan.read_bytes()).hexdigest()
        _publish(paths=paths, candidate=candidate, plan=fixed_plan, inputs=inputs,
                 digest=digest, backup_dir=None, preflight=True, run=run)
    backup_dir = paths.backups / attempt_path.stem if backup else None
    attempt = activation_receipt(status="pending", target_commit=target, previous_commit=previous)
    attempt.update(phase="stopping", imageId=manifest["imageId"],
                   attemptPath=str(attempt_path), publication=None,
                   backupDir=None if backup_dir is None else str(backup_dir),
                   planPath=None if root_ref is None else str(fixed_plan),
                   previousActivationSha256=(hashlib.sha256((paths.activation / "active.json").read_bytes()).hexdigest()
                                             if (paths.activation / "active.json").exists() else None))
    write_json(attempt_path, attempt)
    # 2. 从停机开始保留失败现场；安装、迁移、Root CAS 由目标镜像统一执行。
    try:
        stop_runtime(run=run)
        attempt["phase"] = "publishing"
        write_json(attempt_path, attempt)
        if root_ref is not None:
            assert inputs is not None
            if backup_dir is not None and environment_file.exists():
                atomic_write(backup_dir.with_suffix(".env"), environment_file.read_text(encoding="utf-8"))
            attempt["publication"] = _publish(
                paths=paths, candidate=candidate, plan=fixed_plan, inputs=inputs,
                digest=digest, backup_dir=backup_dir, preflight=False, run=run,
            )
            attempt["phase"] = "published"
            write_json(attempt_path, attempt)
        _start_and_record(paths=paths, environment_file=environment_file, candidate=candidate,
                          attempt_path=attempt_path, attempt=attempt, run=run, unit_root=unit_root, cli_path=cli_path)
    except BaseException as error:
        _record_failure(paths=paths, attempt_path=attempt_path, attempt=attempt, error=error, run=run)
        raise RuntimeError(f"部署未完成，保留现场: {attempt_path}; 原因: {error}") from error
    return "activated"


def resume_release(
    *, paths: ReleasePaths, attempt_path: Path, environment_file: Path, mise: Path, run: Run,
    unit_root: Path = Path("/etc/systemd/system"), cli_path: Path = Path.home() / ".local/bin/akashic-release",
) -> str:
    """只重试已发布 Root 的启动和验收，不重复备份、安装或迁移。"""

    # 1. 回执必须属于此发行目录，且有可核对的发布结果。
    if (attempt_path.is_symlink() or not attempt_path.is_file()
        or attempt_path.parent.resolve() != paths.activation.resolve()
        or not attempt_path.name.startswith("deploy-")):
        raise ValueError("--attempt 必须是本发行目录中的 deploy 回执")
    attempt = read_json(attempt_path)
    publication = attempt.get("publication")
    if not isinstance(publication, dict):
        raise RuntimeError("没有完整发布回执；核对现场后用当前 Root 编写清单重新 install")
    target = attempt["targetCommit"]
    if not isinstance(target, str) or len(target) != 40 or any(c not in "0123456789abcdef" for c in target):
        raise ValueError("部署回执 targetCommit 无效")
    manifest = read_json(paths.release(target))
    if manifest["imageId"] != attempt["imageId"] or manifest["sourceCommit"] != target:
        raise RuntimeError("已准备版本与部署回执不同")
    active_path = paths.activation / "active.json"
    active = read_json(active_path) if active_path.exists() else {}
    current = read_environment(environment_file)
    candidate = release_environment(paths=paths, manifest=manifest, current=current, mise=mise)
    _check_publication(paths=paths, candidate=candidate, publication=publication, run=run)
    active_digest = hashlib.sha256(active_path.read_bytes()).hexdigest() if active_path.exists() else None
    if active.get("attemptPath") != str(attempt_path) and active_digest != attempt["previousActivationSha256"]:
        raise RuntimeError("active 已被后续部署更新；禁止恢复过期尝试")
    if current.get("AKASHIC_RUNTIME_COMMIT") not in {attempt["previousCommit"], target}:
        raise RuntimeError("已有后续部署；禁止恢复过期尝试")
    # 2. 完成同一个启动边界，失败仍保留原发布结果。
    try:
        stop_runtime(run=run)
        _start_and_record(paths=paths, environment_file=environment_file, candidate=candidate,
                          attempt_path=attempt_path, attempt=attempt, run=run, unit_root=unit_root, cli_path=cli_path)
    except BaseException as error:
        _record_failure(paths=paths, attempt_path=attempt_path, attempt=attempt, error=error, run=run)
        raise RuntimeError(f"恢复启动失败，保留现场: {attempt_path}; 原因: {error}") from error
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
            "检测到未登记 legacy skill links；激活前按部署者的数据处理方案运行 "
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
