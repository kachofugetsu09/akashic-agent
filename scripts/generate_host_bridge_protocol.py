from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROTO = Path("agent/host_bridge/host_bridge.proto")
OUTPUTS = ("host_bridge_pb2.py", "host_bridge_pb2.pyi", "host_bridge_pb2_grpc.py")
GENERATOR_VERSION = "1.78.0"


def main() -> int:
    """使用固定生成器生成协议，或检查提交的生成物是否一致。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if version("grpcio-tools") != GENERATOR_VERSION:
        raise RuntimeError(f"需要 grpcio-tools=={GENERATOR_VERSION}")

    # 1. 临时目录生成，检查模式不改源码。
    with tempfile.TemporaryDirectory(prefix="host-bridge-proto-") as temporary:
        output = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{ROOT}",
                f"--python_out={output}",
                f"--pyi_out={output}",
                f"--grpc_python_out={output}",
                str(PROTO),
            ],
            cwd=ROOT,
            check=True,
        )
        # 2. 比较原始生成字节，不修改 generated code 的格式或版本检查。
        different = []
        for name in OUTPUTS:
            relative = PROTO.parent / name
            generated = (output / relative).read_bytes()
            target = ROOT / relative
            if args.check:
                if not target.exists() or target.read_bytes() != generated:
                    different.append(str(relative))
            else:
                target.write_bytes(generated)
        if different:
            print("协议生成物不一致: " + ", ".join(different), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
