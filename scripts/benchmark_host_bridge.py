from __future__ import annotations

import argparse
import asyncio
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


def summarize(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "samples": len(samples),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 3),
    }


async def measure(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """在指定源码和一次性目录内测量真实进程、UDS 和清理。"""
    sys.path.insert(0, str(args.source))
    client = importlib.import_module("agent.host_bridge.client")
    unified = importlib.import_module("agent.tools.unified_exec")
    grpc = importlib.import_module("grpc")
    commit = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    source_hash = hashlib.sha256()
    sources = sorted((args.source / "agent/host_bridge").glob("*.py")) + [
        args.source / "agent/tools/unified_exec.py",
        args.source / "agent/tools/filesystem.py",
    ]
    for source in sources:
        source_hash.update(str(source.relative_to(args.source)).encode())
        source_hash.update(source.read_bytes())
    token = root / "token"
    token.write_text("benchmark-token\n")
    socket = root / "bridge.sock"
    bridge = None
    manager = None
    log = (root / "bridge.log").open("wb")
    report: dict[str, Any] = {
        "mode": args.mode,
        "source": str(args.source),
        "head": commit,
        "execution_source_sha256": source_hash.hexdigest(),
        "python": platform.python_version(),
        "grpc": grpc.__version__,
        "shell": "/bin/sh -c",
        "login": False,
        "results": [],
    }
    try:
        # 1. 候选和基线运行同一个测量程序，各自导入真实源码。
        if args.mode == "bridge":
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.source)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            bridge = await asyncio.create_subprocess_exec(
                sys.executable,
                "-B",
                "-m",
                "agent.host_bridge.server",
                "--socket",
                str(socket),
                "--token-file",
                str(token),
                "--lease-timeout",
                "60",
                "--artifact-root",
                str(root / "artifacts"),
                "--release-commit",
                commit,
                "--toolchain-digest",
                "b" * 64,
                "--runtime-checkout",
                str(args.source),
                "--bridge-python",
                sys.executable,
                cwd=args.source,
                env=env,
                stdout=log,
                stderr=log,
            )
            async with grpc.aio.insecure_channel(f"unix:{socket}") as readiness:
                await asyncio.wait_for(readiness.channel_ready(), 15)
            manager = client.HostBridgeShellProcessManager(
                socket, "benchmark-boot", "benchmark-token", commit, "b" * 64
            )
            await manager.claim_boot()
            # 冷通道和热通道均只测 Inspect，不混入命令启动。
            cold = []
            for _ in range(args.samples):
                start = time.perf_counter()
                connection = client.HostBridgeShellProcessManager(
                    socket, "benchmark-boot", "benchmark-token", commit, "b" * 64
                )
                try:
                    await connection.inspect()
                    cold.append((time.perf_counter() - start) * 1000)
                finally:
                    await connection.close_transport()
            warm = []
            for _ in range(args.samples):
                start = time.perf_counter()
                await manager.inspect()
                warm.append((time.perf_counter() - start) * 1000)
            report["inspect_cold"] = summarize(cold)
            report["inspect_warm"] = summarize(warm)
        else:
            manager = unified.ShellProcessManager(output_dir=root / "local-output")

        # 2. 固定输出、等待和预算。每个并发批次完全回收后再开始下一批。
        async def one(size: int, index: int, tty: bool = False) -> float:
            command = ":" if size == 0 else f"head -c {size} /dev/zero"
            start = time.perf_counter()
            result = await manager.exec_command(
                command=command,
                argv=["/bin/sh", "-c", command],
                cwd=root,
                env={},
                tty=tty,
                yield_time_ms=10000,
                max_output_tokens=300000,
                hard_timeout_s=30,
                owner_session_key=f"benchmark:{index}",
            )
            elapsed = (time.perf_counter() - start) * 1000
            if (
                result.exit_code != 0
                or result.execution_id is not None
                or result.output != b"\0" * size
                or result.output_omitted_bytes
            ):
                raise RuntimeError(
                    f"benchmark command result mismatch: size={size}, exit={result.exit_code}, actual={len(result.output)}"
                )
            return elapsed

        for size in args.sizes:
            for concurrency in args.concurrency:
                await one(size, 0)
                samples = []
                for _ in range(math.ceil(args.samples / concurrency)):
                    samples.extend(
                        await asyncio.gather(
                            *(one(size, i) for i in range(concurrency))
                        )
                    )
                report["results"].append(
                    {
                        "output_bytes": size,
                        "concurrency": concurrency,
                        **summarize(samples),
                    }
                )
        pty = [await one(0, 0, tty=True) for _ in range(args.samples)]
        report["pty_empty_command"] = summarize(pty)
        active = await manager.active_execution_ids()
        if active:
            raise RuntimeError(f"benchmark leaked executions: {active}")
        return report
    finally:
        # 3. 只结束本次创建的进程，清理失败使测量失败。
        try:
            if manager is not None:
                cleanup = await manager.shutdown()
                if cleanup.failures:
                    raise RuntimeError(f"benchmark cleanup failed: {cleanup.failures}")
        finally:
            if bridge is not None:
                if bridge.returncode is None:
                    bridge.terminate()
                try:
                    await asyncio.wait_for(bridge.wait(), 15)
                except TimeoutError:
                    bridge.kill()
                    await bridge.wait()
                    raise
                if bridge.returncode != 0:
                    log.flush()
                    raise RuntimeError((root / "bridge.log").read_text())
            log.close()


def measure_codec(args: argparse.Namespace) -> dict[str, Any]:
    """分别调用两版真实响应适配器，测独立编码往返和载荷字节数。"""
    sys.path.insert(0, str(args.source))
    protocol = importlib.import_module("agent.host_bridge.protocol")
    client = importlib.import_module("agent.host_bridge.client")
    server = importlib.import_module("agent.host_bridge.server")
    unified = importlib.import_module("agent.tools.unified_exec")
    results = []
    for size in args.sizes:
        raw = (bytes(range(256)) * ((size + 255) // 256))[:size]
        result = unified.ExecutionResult(
            raw, 0, (size + 3) // 4, 0, None, 0, None, "natural"
        )
        if args.wire_version == "v1":

            def roundtrip():
                wire = protocol.serialize_message(
                    protocol.encode_message(
                        {"ok": True, **server._result_payload(result)}
                    )
                )
                decoded = client._execution_result(
                    protocol.decode_message(protocol.deserialize_message(wire))
                )
                return wire, decoded

        else:
            pb = importlib.import_module("agent.host_bridge.host_bridge_pb2")

            def roundtrip():
                wire = protocol.encode_execution(result).SerializeToString()
                return wire, protocol.decode_execution(
                    pb.ExecutionReply.FromString(wire)
                )

        for _ in range(10):
            wire, decoded = roundtrip()
            if decoded != result:
                raise RuntimeError("codec roundtrip changed ExecutionResult")
        samples = []
        for _ in range(args.samples):
            start = time.perf_counter_ns()
            wire, decoded = roundtrip()
            samples.append((time.perf_counter_ns() - start) / 1000)
        ordered = sorted(samples)
        results.append(
            {
                "output_bytes": size,
                "serialized_bytes": len(wire),
                "samples": len(samples),
                "p50_us": round(statistics.median(ordered), 3),
                "p95_us": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 3),
            }
        )
    return {
        "mode": "codec",
        "wire_version": args.wire_version,
        "source": str(args.source),
        "python": platform.python_version(),
        "results": results,
    }


def main() -> None:
    """独立进程导入指定版本，stdout 只输出可比较的 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("local", "bridge", "codec"), required=True)
    parser.add_argument("--wire-version", choices=("v1", "v2"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[0, 4096, 40000, 1048576]
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    args = parser.parse_args()
    args.source = args.source.resolve(strict=True)
    if (
        args.samples <= 0
        or any(c < 1 or c > 32 for c in args.concurrency)
        or any(s < 0 or s > 1048576 for s in args.sizes)
    ):
        parser.error("samples 必须为正，concurrency 为1..32，size 为0..1MiB")
    if args.mode == "codec":
        if args.wire_version is None:
            parser.error("codec 模式必须指定 --wire-version")
        report = measure_codec(args)
    else:
        with tempfile.TemporaryDirectory(prefix="bridge-bench-") as temporary:
            report = asyncio.run(measure(args, Path(temporary)))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
