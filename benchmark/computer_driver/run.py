#!/usr/bin/env python3
"""复用 Cua-Bench Basic 的原题、参考步骤和判分器，测量现有 Gateway。"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ["CUA_TELEMETRY_ENABLED"] = "false"

import aiohttp
import requests
from adapter import GatewayEnvironment
from playwright.async_api import Error as PlaywrightError

import docker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CUA_SHA = "aabb2082c170289256f0c8d9db4cce094c778578"
CSS_SHA = "02d70b1ae97aab2e87be23869b2bb5ad4ed0a3b63911c02612a028ee9e473a7b"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


async def run_tasks(args, gateway, cdp, css, container):
    """只排列原始任务变体；执行和判分交给 Cua Environment。"""
    tasks = sorted(
        (args.cua / "libs/cua-bench/datasets/cua-bench-basic").glob("*/main.py")
    )
    if args.task:
        tasks = [task for task in tasks if task.parent.name in args.task]
    if not tasks or (
        args.task and {task.parent.name for task in tasks} != set(args.task)
    ):
        raise ValueError("Task selection is empty or includes unknown upstream tasks")
    results = []
    for task in tasks:
        sample = GatewayEnvironment.load(
            task.parent, gateway, cdp, css, args.suppress_actions
        )
        count = len(sample.tasks_config_fn())
        for variant in range(min(count, args.max_variants or count)):
            label = f"{task.parent.name}-{variant}"
            env = GatewayEnvironment.load(
                task.parent, gateway, cdp, css, args.suppress_actions
            )
            env.tracing.start(label)
            row = {
                "task": task.parent.name,
                "variant": variant,
                "status": "setup_error",
            }
            case_dir = args.output / label
            case_dir.mkdir()
            try:
                # 超时的 xdotool 可能留下按下状态；只复位本次测试桌面。
                reset = container.exec_run(
                    [
                        "xdotool",
                        "mouseup",
                        "1",
                        "mouseup",
                        "2",
                        "mouseup",
                        "3",
                        "keyup",
                        "Shift_L",
                        "Shift_R",
                        "Control_L",
                        "Control_R",
                        "Alt_L",
                        "Alt_R",
                        "Super_L",
                        "Super_R",
                    ]
                )
                if reset.exit_code:
                    raise RuntimeError(f"Fixture input reset failed: {reset.output!r}")
                screenshot, cfg = await env.reset(task_id=variant)
                (case_dir / "before.png").write_bytes(screenshot)
                row["description"] = cfg.description
                row["initial_reward"] = await env.evaluate()
                row["status"] = "executing"
                start = time.perf_counter()
                try:
                    await env.solve()
                    row["status"] = "evaluated"
                except NotImplementedError as error:
                    row.update(status="unsupported", error=str(error))
                except (aiohttp.ClientError, PlaywrightError) as error:
                    row.update(status="execution_error", error=str(error))
                row["seconds"] = time.perf_counter() - start
                row["reward"] = await env.evaluate()
                row["actions"] = env.session.actions
                (case_dir / "after.png").write_bytes(await env.session.screenshot())
                env.tracing.save_to_disk(str(case_dir / "trace"), save_pngs=True)
            finally:
                await env.close()
                results.append(row)
                save(args.output / "results.json", results)
            print(label, row["status"], row.get("reward"), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="本地已有 Computer 镜像")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--cua", type=Path, default=ROOT / "benchmark/data/cua")
    parser.add_argument(
        "--css", type=Path, default=ROOT / "benchmark/data/tailwind-4.1.18.js"
    )
    parser.add_argument("--task", action="append", help="原始任务名，可重复；默认全部")
    parser.add_argument("--max-variants", type=int, help="每类最多跑几个原始变体")
    parser.add_argument(
        "--suppress-actions", action="store_true", help="负对照：不发送输入动作"
    )
    parser.add_argument("--driver", choices=("legacy", "source"), default="legacy")
    args = parser.parse_args()
    if args.driver == "source":
        from source_adapter import SourceSession

        GatewayEnvironment.session_type = SourceSession
    if args.max_variants is not None and args.max_variants < 1:
        parser.error("max-variants must be positive")
    if git(args.cua, "rev-parse", "HEAD") != CUA_SHA or git(
        args.cua, "status", "--porcelain"
    ):
        parser.error("Cua checkout must be clean at the pinned commit")
    if hashlib.sha256(args.css.read_bytes()).hexdigest() != CSS_SHA:
        parser.error("Tailwind dependency digest differs")

    # 1. 固定源码与镜像；每次使用新证据目录，不覆盖基线。
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    shutil.copytree(
        args.source / "docker/computer",
        source,
        ignore=shutil.ignore_patterns("node_modules", "target", "evidence"),
    )
    (source / "start.sh").chmod(0o755)  # 与 Computer Dockerfile 的安装模式一致。
    shutil.copy2(args.source / "agent/workloads/userns-seccomp.json", source)
    harness = args.output / "harness"
    harness.mkdir()
    for name in ("run.py", "adapter.py", "source_adapter.py", "requirements.lock"):
        shutil.copy2(HERE / name, harness / name)
    client = docker.from_env()
    image = client.images.get(args.image)
    manifest = {
        "driver": args.driver,
        "cua_commit": CUA_SHA,
        "source_commit": git(args.source, "rev-parse", "HEAD"),
        "image_id": image.id,
        "css_sha256": CSS_SHA,
        "screen": [1280, 800],
        "seed": 42,
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        },
        "adapter_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob("*.py")
        },
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "task_filter": args.task,
        "max_variants": args.max_variants,
        "mode": "suppressed-actions"
        if args.suppress_actions
        else "upstream-reference-solver",
        "cleanup": "pending",
        "completed": False,
    }
    save(args.output / "manifest.json", manifest)
    container = None
    try:
        # 2. 只发布随机 loopback 端口；全新 tmpfs 不含正式 profile。
        container = client.containers.run(
            image.id,
            detach=True,
            mem_limit="2g",
            pids_limit=512,
            shm_size="512m",
            cap_drop=["ALL"],
            security_opt=[
                "no-new-privileges",
                "seccomp=" + (source / "userns-seccomp.json").read_text(),
            ],
            tmpfs={"/data": "rw,uid=1000,gid=1000,mode=0755,size=1g"},
            ports={"8080/tcp": ("127.0.0.1", None), "9223/tcp": ("127.0.0.1", None)},
            volumes={}
            if args.driver == "source"
            else {
                str(source / name): {"bind": "/opt/computer/" + name, "mode": "ro"}
                for name in ("gateway.mjs", "start.sh")
            },
        )
        container.reload()
        ports = container.attrs["NetworkSettings"]["Ports"]
        gateway = "http://127.0.0.1:" + ports["8080/tcp"][0]["HostPort"]
        cdp = "http://127.0.0.1:" + ports["9223/tcp"][0]["HostPort"]
        container.exec_run(
            [
                "node",
                "-e",
                "const n=require('net');n.createServer(s=>{const t=n.connect(9222,'127.0.0.1');s.pipe(t).pipe(s);t.on('error',()=>s.destroy());s.on('error',()=>t.destroy())}).listen(9223,'0.0.0.0')",
            ],
            detach=True,
        )
        deadline = time.monotonic() + 60
        while True:
            try:
                requests.get(cdp + "/json/version", timeout=1).raise_for_status()
                requests.get(gateway + "/activity", timeout=1).raise_for_status()
                break
            except requests.RequestException:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        manifest["browser"] = requests.get(cdp + "/json/version", timeout=5).json()[
            "Browser"
        ]
        results = asyncio.run(
            run_tasks(args, gateway, cdp, args.css.read_text(), container)
        )
        manifest["completed"] = True
        manifest["cases"] = len(results)
        manifest["solved"] = sum(
            row["status"] == "evaluated" and row["reward"] == [1.0] for row in results
        )
        print(json.dumps(manifest, indent=2))
    finally:
        # 3. 清理由本次容器 owner 负责，失败则命令非零。
        if container is not None:
            try:
                (args.output / "container.log").write_bytes(container.logs())
            finally:
                container.remove(force=True)
                manifest["cleanup"] = "passed"
        save(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
