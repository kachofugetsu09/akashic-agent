from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from scripts.plugin_store import materialize

_IMAGE_DISTRIBUTION = "/opt/akashic/distribution"


# 把镜像里已准备好 wheels 的发行版原样放进制品库；同一提交已存在时直接返回。
# #1973 之前的镜像没有逐插件 content_commit，不放进制品库，运行时继续用镜像内发行版。
def store_image_distribution(*, image: str, store: Path, commit: str) -> Path | None:
    target = store / "distributions" / commit
    if target.exists():
        return target
    store.mkdir(parents=True, exist_ok=True)
    container = subprocess.run(["docker", "create", image], check=True, capture_output=True,
                               text=True).stdout.strip()
    try:
        with tempfile.TemporaryDirectory(prefix="akashic-store-", dir=store) as directory:
            copied = Path(directory) / "distribution"
            _ = subprocess.run(["docker", "cp", f"{container}:{_IMAGE_DISTRIBUTION}", str(copied)],
                               check=True, capture_output=True, text=True)
            report = json.loads((copied / "distribution.json").read_text())
            if any("content_commit" not in row for row in report["plugins"]):
                return None
            return materialize(copied, store, commit)
    finally:
        _ = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)


# runtime.env 指向制品库里的发行版；旧版本没有写入制品库时继续用镜像内发行版。
def runtime_distribution(store: Path, commit: str) -> str:
    target = store / "distributions" / commit
    return str(target) if target.is_dir() else _IMAGE_DISTRIBUTION
