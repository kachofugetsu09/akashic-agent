"""给同一隔离 Runtime 更换 Web Shell 源码，不重启业务插件或数据库。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not (args.workspace / "runtime/endpoints.json").is_file():
        raise RuntimeError("需要已启动的隔离 Runtime")
    spec = importlib.util.spec_from_file_location("measured_web_shell", args.source.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print(json.dumps({"sourceSha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      "port": args.port, "workspace": str(args.workspace)}), flush=True)
    uvicorn.run(module.create_web_shell_app(args.workspace), host="127.0.0.1", port=args.port,
                log_level="warning", timeout_graceful_shutdown=5)
