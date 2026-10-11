"""被测命令：gateway 冷启动 → exec 跑完剧本 → SIGTERM；以 wait4 记录整棵子树的精确 CPU。"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from prepare import environment, side_paths

_READY_TIMEOUT_S = 120.0


# gateway 自身在就绪时刻已用的 CPU（不含已回收子进程），单独衡量启动阶段。
def _own_cpu_s(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _children_cpu_s() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _wait_ready(gateway: subprocess.Popen[bytes], socket: Path, started: float) -> None:
    while not socket.exists():
        if gateway.poll() is not None:
            raise RuntimeError("gateway 在就绪前退出，见 gateway.log")
        if time.perf_counter() - started > _READY_TIMEOUT_S:
            raise TimeoutError("gateway 就绪超时")
        time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--root", type=Path, required=True)
    _ = parser.add_argument("--side", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    side = side_paths(Path(args.root), Path(args.side))
    out = Path(args.out)
    os.environ.update(environment(side))

    # 1. 启动 gateway 并等待控制 socket 出现。
    started = time.perf_counter()
    marks: dict[str, float | int] = {"start_epoch_ms": time.time() * 1000}
    command = [sys.executable, str(side.root / "main.py")]
    workspace_args = ["--config", str(side.state / "config.toml"), "--workspace", str(side.state / "workspace")]
    with (out / "gateway.log").open("wb") as log:
        gateway = subprocess.Popen(command + ["gateway"] + workspace_args, stdout=log, stderr=log)
        try:
            _wait_ready(gateway, side.state / "workspace/akashic.sock", started)
            marks["ready_s"] = time.perf_counter() - started
            marks["ready_cpu_s"] = _own_cpu_s(gateway.pid)

            # 2. 插件命令客户端跑完整个剧本；它的 CPU 单独记一份。
            before = _children_cpu_s()
            with (out / "exec.log").open("wb") as exec_log:
                result = subprocess.run(command + ["exec", "--new", "--final-only"] + workspace_args + [args.prompt],
                                        stdout=exec_log, stderr=exec_log)
            marks["exec_cpu_s"] = _children_cpu_s() - before
            marks["reply_done_s"] = time.perf_counter() - started
        finally:
            # 3. 与用户停服相同：SIGTERM，并要求干净退出。
            gateway.send_signal(signal.SIGTERM)
            _ = gateway.wait(timeout=60)

    own = resource.getrusage(resource.RUSAGE_SELF)
    marks.update(wall_s=time.perf_counter() - started, exec_exit=result.returncode,
                 gateway_exit=gateway.returncode, cpu_s=_children_cpu_s() + own.ru_utime + own.ru_stime)
    _ = (out / "wait4.json").write_text(json.dumps(marks, indent=2) + "\n")
    return 0 if result.returncode == 0 and gateway.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
