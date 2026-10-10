"""Shell 接单后交给独立用户 systemd service，不续跑模型回合。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts.akashic_release.doctor import read_environment
from scripts.akashic_release.manifest import read_json, write_json, release_lock
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk/python/src"))


def _request_path(root: Path, identity: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", identity) is None:
        raise ValueError("任务编号必须是 32 位小写十六进制")
    return root / "run/self-deploy" / f"{identity}.json"


def _user_environment() -> dict[str, str]:
    # Host Bridge 是系统 service；不能依赖桌面登录注入用户 bus 路径。
    return {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}


def submit(args: argparse.Namespace) -> dict[str, object]:
    """持久接纳原调用，并确认独立 worker 的 exec；不等待更新。"""
    from bootstrap.runtime_stop import StopParams

    if re.fullmatch(r"[0-9a-f]{40}", args.commit) is None:
        raise ValueError("--commit 必须是完整 40 位 SHA")
    context = json.loads(os.environ.get("AKASHIC_CALL_CONTEXT", "{}"))
    identity = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[
        :32
    ]
    stop = StopParams.model_validate(
        {**context, "request_id": identity, "timeout_s": args.timeout}
    )
    root = args.root.resolve()
    environment_file = args.runtime_env.resolve(strict=True)
    current = read_environment(environment_file)
    ready = read_json(Path(current["AKASHIC_WORKSPACE"]) / ".runtime-ready.json")
    if ready["bootId"] != stop.boot_id or ready["state"] != "ready":
        raise RuntimeError("Shell 调用不属于当前就绪 boot")
    path = _request_path(root, identity)
    unit = f"akashic-deploy-{identity}.service"
    request = {
        "targetCommit": args.commit,
        "stop": stop.model_dump(),
        "root": str(root),
        "runtimeEnv": str(environment_file),
        "backup": args.backup,
        "mise": str(args.mise.resolve()),
        "unit": unit,
    }
    with release_lock(path.with_suffix(".lock")):
        if path.exists():
            record = read_json(path)
            if record["request"] != request:
                raise ValueError("原 Shell 调用已经提交不同的部署请求")
            if record["status"] not in {"accepted", "running", "active"}:
                raise RuntimeError(
                    f"原请求状态 {record['status']}，请查询 status，不自动重放"
                )
            return {
                "requestId": identity,
                "status": record["status"],
                "unit": unit,
                "targetCommit": args.commit,
            }
        record = {"request": request, "status": "submitting"}
        write_json(path, record)
        try:
            asyncio.run(_arm_stop(request, current))
            subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    f"--unit={unit}",
                    "--property=Type=exec",
                    "--property=KillMode=control-group",
                    "--property=TimeoutStartSec=infinity",
                    f"--working-directory={Path(__file__).resolve().parents[2]}",
                    request["mise"],
                    "exec",
                    "--",
                    sys.executable,
                    "-m",
                    "scripts.akashic_release.cli",
                    "self-deploy-worker",
                    "--request",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=_user_environment(),
            )
        except Exception as error:
            detail = (
                error.stderr.strip()
                if isinstance(error, subprocess.CalledProcessError)
                else str(error)
            )
            record.update(status="failed", error=detail)
            write_json(path, record)
            raise RuntimeError(f"部署未接纳：{record['error']}") from error
        # worker 通过同一把锁等待此提交，不覆盖后来阶段。
        record["status"] = "accepted"
        write_json(path, record)
    return {
        "requestId": identity,
        "status": "accepted",
        "unit": unit,
        "targetCommit": args.commit,
    }


async def _arm_stop(request: dict[str, Any], current: dict[str, str]) -> None:
    """返回接单前保留同一调用的最终送达凭据。"""
    from akashic_sdk import AsyncAkashic
    from agent.config import resolve_app_server_endpoint
    from agent.config_models import Config

    workspace = Path(current["AKASHIC_WORKSPACE"])
    config = Config.load(current["AKASHIC_CONFIG"], workspace=workspace)
    endpoint = resolve_app_server_endpoint(config.app_server.listen, workspace)
    async with await AsyncAkashic.connect(endpoint) as client:
        await client.request(
            "runtime/prepare-stop", {**request["stop"], "arm_only": True}
        )


async def _prepare_stop(
    request: dict[str, Any], current: dict[str, str]
) -> dict[str, object]:
    """通过正式私有控制入口等待原回合与全局排空。"""
    from akashic_sdk import AsyncAkashic
    from agent.config import resolve_app_server_endpoint
    from agent.config_models import Config
    from plugins.host_execution.controller_client import UnixWorkloadController

    workspace = Path(current["AKASHIC_WORKSPACE"])
    config = Config.load(current["AKASHIC_CONFIG"], workspace=workspace)
    endpoint = resolve_app_server_endpoint(config.app_server.listen, workspace)
    async with await AsyncAkashic.connect(endpoint) as client:
        result = await client.request("runtime/prepare-stop", request["stop"])
        try:
            if not isinstance(result, dict) or result.get("state") != "drained":
                raise RuntimeError("Core 未确认正常回合和排空")
            controller = await UnixWorkloadController(
                Path(current["AKASHIC_WORKLOAD_RUNTIME_DIR"]) / "controller.sock"
            ).status()
            result["controllerId"] = controller["controller_id"]
            result["containerId"] = subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    current["AKASHIC_CONTAINER_NAME"],
                ],
                text=True,
            ).strip()
            return result
        except Exception:
            await client.request("runtime/cancel-stop", request["stop"])
            raise


def wait_for_stop(record: dict[str, Any], current: dict[str, str]) -> dict[str, object]:
    return asyncio.run(_prepare_stop(record["request"], current))


def verify_closed(ack: dict[str, object], current: dict[str, str]) -> None:
    """停止命令成功之外，核对正常关闭证据和实际容器退出。"""
    closed = read_json(
        Path(current["AKASHIC_WORKSPACE"]) / "runtime/closed" / f"{ack['bootId']}.json"
    )
    if (
        closed.get("state") != "closed"
        or closed.get("bootId") != ack["bootId"]
        or closed.get("rootIdentity") != ack["rootIdentity"]
    ):
        raise RuntimeError("旧 Core 缺少正常关闭证据")
    controller = read_json(
        Path(current["AKASHIC_EXPERIMENT_ROOT"])
        / "workload-controller"
        / "closed"
        / f"{ack['controllerId']}.json"
    )
    if controller["controller"] != {"id": ack["controllerId"], "leases_empty": True}:
        raise RuntimeError("旧 Controller 缺少精确租约清理证据")
    identity = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.Id}}", current["AKASHIC_CONTAINER_NAME"]],
        text=True,
    ).strip()
    if identity != ack["containerId"]:
        raise RuntimeError("旧 Core 容器已被替换")
    for container in (
        current["AKASHIC_CONTAINER_NAME"],
        current["AKASHIC_CONTAINER_NAME"] + "-workloads",
    ):
        state = json.loads(
            subprocess.check_output(
                ["docker", "inspect", "--format", "{{json .State}}", container],
                text=True,
            )
        )
        if state["Running"] or state["OOMKilled"] or state["ExitCode"] != 0:
            raise RuntimeError(f"旧容器未正常退出: {container}: {state}")
    for unit in ("akashic-core.service", "akashic-host-bridge.service"):
        state = subprocess.check_output(
            [
                "systemctl",
                "show",
                unit,
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
                "-p",
                "ActiveState",
            ],
            text=True,
        )
        if {"Result=success", "ExecMainStatus=0", "ActiveState=inactive"} - set(
            state.splitlines()
        ):
            raise RuntimeError(f"旧服务未正常停止: {unit}: {state}")


def _exec_target(path: Path, request: dict[str, Any]) -> None:
    """准备目标产物后用其解释器替换当前 worker，保持同一 job 和 writer。"""
    source = path.with_suffix(".source")
    root = Path(request["root"])
    origin = "https://github.com/kachofugetsu09/akashic-agent.git"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "remote", "add", "origin", origin], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "fetch",
            "--quiet",
            "--depth=1",
            "origin",
            request["targetCommit"],
        ],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", request["targetCommit"]],
        cwd=source,
        check=True,
    )
    target_cli = source / "scripts/akashic_release/cli.py"
    subprocess.run(
        [
            sys.executable,
            str(target_cli),
            "install",
            "--no-activate",
            "--yes",
            "--source-checkout",
            str(source),
            "--commit",
            request["targetCommit"],
            "--root",
            str(root),
            "--runtime-env",
            request["runtimeEnv"],
            "--mise",
            request["mise"],
        ],
        check=True,
    )
    python = root / "bridge-venvs" / request["targetCommit"] / "bin/python"
    os.execv(
        str(python),
        [
            str(python),
            str(target_cli),
            "self-deploy-worker",
            "--request",
            str(path),
            "--prepared",
        ],
    )


def worker(args: argparse.Namespace) -> dict[str, object]:
    """保留任务错误，调用既有发布器；不合并 PR、不发送新模型消息。"""
    from scripts.akashic_release.cli import install
    from bootstrap.runtime_stop import StopParams

    path = args.request.resolve(strict=True)
    with release_lock(path.with_suffix(".lock"), wait=True):
        record = read_json(path)
        expected_status = "running" if args.prepared else "accepted"
        if record["status"] != expected_status:
            raise RuntimeError("任务未接纳或已执行，不自动重放")
        record["recordPath"] = str(path)
        record["status"] = "running"
        write_json(path, record)
    try:
        request = record["request"]
        stop = StopParams.model_validate(request["stop"])
        root = Path(request["root"])
        if path != _request_path(root, stop.request_id):
            raise ValueError("部署请求不属于正式任务目录")
        source = path.with_suffix(".source")
        current = read_environment(Path(request["runtimeEnv"]))
        ready = read_json(Path(current["AKASHIC_WORKSPACE"]) / ".runtime-ready.json")
        if ready["bootId"] != request["stop"]["boot_id"]:
            raise RuntimeError("原 boot 已被替换，不停止新实例")
        origin = "https://github.com/kachofugetsu09/akashic-agent.git"
        if not args.prepared:
            _exec_target(path, request)
        result = install(
            argparse.Namespace(
                root=root,
                source_checkout=source,
                commit=request["targetCommit"],
                origin=origin,
                runtime_env=Path(request["runtimeEnv"]),
                mise=Path(request["mise"]),
                unit_root=Path("/etc/systemd/system"),
                cli_path=Path.home() / ".local/bin/akashic-release",
                yes=True,
                no_activate=False,
                plan=None,
                inputs=None,
                backup=request["backup"],
            ),
            self_deploy=record,
        )
        record.update(status="active", result=result)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(path, record)
        raise
    write_json(path, record)
    return record


def status(args: argparse.Namespace) -> dict[str, object]:
    record = read_json(_request_path(args.root.resolve(), args.request_id))
    record["service"] = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            record["request"]["unit"],
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
        ],
        text=True,
        env=_user_environment(),
    ).strip()
    return record
