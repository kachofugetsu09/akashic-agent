from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from scripts.plugin_store import materialize

_IMAGE_DISTRIBUTION = "/opt/akashic/distribution"


# 镜像里的发行版目录是只读的：普通用户直接 docker cp 会先建出只读目录再写不进文件。
# 改为读 tar 流，tarfile 在最后才设置目录权限。
def _copy_out(container: str, directory: Path) -> None:
    stream = subprocess.Popen(["docker", "cp", f"{container}:{_IMAGE_DISTRIBUTION}", "-"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if stream.stdout is None:
        raise RuntimeError("docker cp 没有输出流")
    with tarfile.open(fileobj=stream.stdout, mode="r|") as archive:
        archive.extractall(directory, filter="data")
    error = stream.stderr.read().decode(errors="replace") if stream.stderr is not None else ""
    if stream.wait():
        raise RuntimeError(f"docker cp 镜像发行版失败: {error.strip()}")


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
            _copy_out(container, Path(directory))
            report = json.loads((copied / "distribution.json").read_text())
            if any("content_commit" not in row for row in report["plugins"]):
                return None
            stored = materialize(copied, store, commit)
        hand_over_store(store)
        return stored
    finally:
        _ = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)


# 以 root 发布时把制品库交还给 release 根目录的属主：之后普通用户发起的发布和热更新还要写入。
# 只改属主、不改权限，制品内容保持只读。
def hand_over_store(store: Path) -> None:
    if os.geteuid() != 0 or not store.exists():
        return
    owner = store.parent.stat()
    for path in (store, *store.rglob("*")):
        os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)


# runtime.env 指向制品库里的发行版；旧版本没有写入制品库时继续用镜像内发行版。
def runtime_distribution(store: Path, commit: str) -> str:
    target = store / "distributions" / commit
    return str(target) if target.is_dir() else _IMAGE_DISTRIBUTION
