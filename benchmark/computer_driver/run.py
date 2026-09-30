#!/usr/bin/env python3
"""复用 Cua-Bench Basic 的原题、参考步骤和判分器，测量现有 Gateway。"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
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


@contextmanager
def computer_container(client, image, source, driver, output, record):
    """每个 owner 只启动和清理自己的容器，保留真实清理失败。"""
    container = None
    record["cleanup"] = "pending"
    try:
        # 1. 独立 tmpfs 与随机 loopback 端口，不挂载正式 profile。
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
            volumes=(
                {}
                if driver == "source"
                else {
                    str(source / name): {"bind": "/opt/computer/" + name, "mode": "ro"}
                    for name in ("gateway.mjs", "start.sh")
                }
            ),
        )
        record["container_id"] = container.id
        runtime_files = ["gateway.mjs"]
        if driver == "source":
            runtime_files += [
                str(p.relative_to(source))
                for p in sorted((source / "driver").glob("*.mjs"))
            ]
        hashes = container.exec_run(
            [
                "node",
                "-e",
                "const fs=require('fs'),c=require('crypto');const out={};for(const n of JSON.parse(process.argv[1]))out[n]=c.createHash('sha256').update(fs.readFileSync('/opt/computer/'+n)).digest('hex');console.log(JSON.stringify(out))",
                json.dumps(runtime_files),
            ]
        )
        if hashes.exit_code:
            raise RuntimeError(f"Runtime source read failed: {hashes.output!r}")
        record["runtime_sha256"] = json.loads(hashes.output)
        record["snapshot_matches_runtime"] = all(
            hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
            for name, digest in record["runtime_sha256"].items()
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
        record["browser"] = requests.get(cdp + "/json/version", timeout=5).json()[
            "Browser"
        ]
        yield container, gateway, cdp
    finally:
        # 2. 即使导出日志失败也删除自己的容器；失败不写 passed。
        if container is not None:
            record["cleanup"] = "failed"
            try:
                (output / "container.log").write_bytes(container.logs())
            finally:
                container.remove(force=True)
                record["cleanup"] = "passed"


async def run_agent_tasks(args, client, image, source):
    """每个原题变体运行独立 Agent episode，Browser 成绩默认待人工审核。"""
    from agent_adapter import run_agent

    tasks = sorted(
        (args.cua / "libs/cua-bench/datasets/cua-bench-basic").glob("*/main.py")
    )
    if args.task:
        tasks = [task for task in tasks if task.parent.name in args.task]
    if not tasks or (args.task and {t.parent.name for t in tasks} != set(args.task)):
        raise ValueError("Task selection is empty or includes unknown upstream tasks")
    results = []
    guidance = (args.output / "computer-SKILL.md").read_text()
    for task in tasks:
        sample = GatewayEnvironment.load(task.parent, "", "", "", False)
        count = len(sample.tasks_config_fn())
        for variant in range(min(count, args.max_variants or count)):
            label = f"{task.parent.name}-{variant}"
            case_dir = args.output / label
            case_dir.mkdir()
            row = {
                "task": task.parent.name,
                "variant": variant,
                "status": "setup_error",
            }
            env = None
            try:
                # 1. 原 Environment 独占准备与判分，不向 Agent 传入 solver。
                with computer_container(
                    client, image, source, args.driver, case_dir, row
                ) as (_, gateway, cdp):
                    env = GatewayEnvironment.load(
                        task.parent, gateway, cdp, args.css.read_text(), False
                    )
                    env.tracing.start(label)
                    try:
                        _, cfg = await env.reset(task_id=variant)
                        row["description"] = cfg.description
                        row["initial_reward"] = await env.evaluate()
                        started = time.perf_counter()
                        row.update(
                            await run_agent(
                                args, env, cfg.description, case_dir, guidance
                            )
                        )
                        row["seconds"] = time.perf_counter() - started
                        row["reward"] = await env.evaluate()
                        row["reward_phase"] = (
                            "settled_model_run"
                            if row["status"] == "evaluated"
                            else "snapshot_before_cleanup"
                        )
                        (case_dir / "after.image").write_bytes(
                            await env.session.screenshot()
                        )
                        env.tracing.save_to_disk(
                            str(case_dir / "trace"), save_pngs=True
                        )
                    finally:
                        # 2. endTurn 等待取消的实际 driver 排空，再关闭 host CDP。
                        if env.session is not None:
                            row["actions"] = env.session.actions
                        await env.close()
            finally:
                if "actions" not in row and env is not None and env.session is not None:
                    row["actions"] = env.session.actions
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
    parser.add_argument(
        "--agent", action="store_true", help="运行原 Cua Agent，不运行参考解法"
    )
    parser.add_argument(
        "--agent-browser", action="store_true", help="Agent 额外获得正式 Browser API"
    )
    parser.add_argument("--model", help="LiteLLM 的实际 provider/model 名称")
    parser.add_argument("--api-base", help="可选 provider endpoint")
    parser.add_argument(
        "--api-key-env", default="BENCHMARK_API_KEY", help="凭据环境变量名，不写入证据"
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--episode-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument(
        "--provider-kind", choices=("real", "simulated"), default="real"
    )
    args = parser.parse_args()
    if args.agent and (
        not args.model or args.driver != "source" or args.suppress_actions
    ):
        parser.error(
            "Agent mode requires --model, --driver source and real input actions"
        )
    if args.agent_browser and not args.agent:
        parser.error("agent-browser requires agent")
    if min(
        args.max_steps, args.max_tokens, args.episode_timeout, args.request_timeout
    ) <= 0 or not all(
        math.isfinite(value)
        for value in (args.episode_timeout, args.request_timeout, args.temperature)
    ):
        parser.error("Agent step, token and timeout limits must be positive")
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
    for name in (
        "run.py",
        "adapter.py",
        "source_adapter.py",
        "agent_adapter.py",
        "score_agent.py",
        "requirements.lock",
        "agent-requirements.lock",
    ):
        shutil.copy2(HERE / name, harness / name)
    skill = args.source / "plugins/computer/skills/computer/SKILL.md"
    if args.agent:
        shutil.copy2(skill, args.output / "computer-SKILL.md")
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
        "agent": (
            {
                "model": args.model,
                "browser": args.agent_browser,
                "provider_kind": args.provider_kind,
                "max_steps": args.max_steps,
                "episode_timeout": args.episode_timeout,
                "request_timeout": args.request_timeout,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "sdk_loop": "NativeCompletion via register_agent",
                "image_policy": "initial desktop + latest desktop + current driver-emitted images",
                "guidance_sha256": hashlib.sha256(skill.read_bytes()).hexdigest(),
            }
            if args.agent
            else None
        ),
        "mode": (
            ("agent-browser-desktop" if args.agent_browser else "agent-desktop")
            if args.agent
            else (
                "suppressed-actions"
                if args.suppress_actions
                else "upstream-reference-solver"
            )
        ),
        "cleanup": "pending",
        "completed": False,
    }
    save(args.output / "manifest.json", manifest)
    try:
        if args.agent:
            results = asyncio.run(run_agent_tasks(args, client, image, source))
            manifest["cleanup"] = (
                "passed"
                if all(row["cleanup"] == "passed" for row in results)
                else "failed"
            )
        else:
            with computer_container(
                client, image, source, args.driver, args.output, manifest
            ) as (container, gateway, cdp):
                results = asyncio.run(
                    run_tasks(args, gateway, cdp, args.css.read_text(), container)
                )
        manifest["completed"] = True
        manifest["cases"] = len(results)
        # 模拟 provider、未审核 Browser 观察或预算终止都不能成为模型成绩。
        manifest["solved"] = sum(
            row["status"] == "evaluated"
            and row["reward"] == [1.0]
            and (
                not args.agent
                or (args.provider_kind == "real" and not args.agent_browser)
            )
            for row in results
        )
        print(json.dumps(manifest, indent=2))
    finally:
        if args.agent and (args.output / "results.json").exists():
            rows = json.loads((args.output / "results.json").read_text())
            manifest["cleanup"] = (
                "passed"
                if rows and all(row["cleanup"] == "passed" for row in rows)
                else "failed"
            )
        save(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
