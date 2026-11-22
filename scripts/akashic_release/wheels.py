"""宿主侧（停机之前、有网络）为带第三方依赖的 extra 包求离线 wheel；只依赖标准库。

联网解析发生在目标镜像里（Python 小版本与平台标签与实际安装一致），由 Core 的 agent.plugins.dependency_wheels 执行；
这里负责准备输入目录、选择执行点（docker 或本机）、按文件名取出已解析的版本供后续包作约束，并算出清单要的树哈希。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from agent.plugins.extra_repo import PROXY_ENV

class WheelResolutionFailure(RuntimeError):
    """依赖本身无法解析成 wheel（冲突、只有源码包等）：与网络不可达区分开，调用方不要提示"检查网络"。"""


Resolver = Callable[[Path], dict[str, object]]
Run = Callable[..., subprocess.CompletedProcess[str]]
INPROCESS_FILE = "requirements.inprocess.txt"
_SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "cache", ".cache", "__pycache__"})
_ENV_ROOT = "runtime/plugin-python-environments"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# 在目标镜像里执行：联网，输出目录绑出来；用户取宿主当前用户，产物属主与宿主一致。
def docker_resolver(image: str, run: Run) -> Resolver:
    def resolve(work: Path) -> dict[str, object]:
        # 容器里取不到宿主环境：代理变量按名字透传（只在宿主设置了才传；证书文件路径在容器里没有意义，不传）。
        proxies = [item for key in PROXY_ENV if key in os.environ for item in ("--env", key)]
        command = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "--read-only", *proxies,
                   "--tmpfs", "/tmp:rw,mode=1777,size=2g", "--mount", f"type=bind,src={work},dst={work}",
                   "--workdir", "/opt/akashic/source", "--env", "HOME=/tmp", "--entrypoint", "/opt/venv/bin/python",
                   image, "-B", "-m", "agent.plugins.dependency_wheels", "--work", str(work)]
        run(command, check=False, capture_output=True, text=True)
        return _read_result(work)

    return resolve


# 不经 docker 的执行点：用本机（或容器内）的 Core 解释器与 pip；首装补装在容器里、以及本机验证用。
def local_resolver(core_root: Path) -> Resolver:
    def resolve(work: Path) -> dict[str, object]:
        subprocess.run([sys.executable, "-B", "-m", "agent.plugins.dependency_wheels", "--work", str(work)],
                       cwd=core_root, env={**os.environ, "PYTHONPATH": str(core_root)}, check=False,
                       capture_output=True, text=True)
        return _read_result(work)

    return resolve


def _read_result(work: Path) -> dict[str, object]:
    path = work / "result.json"
    if not path.exists():
        raise RuntimeError("求依赖 wheel 的命令没有产出结果（镜像里缺少 Core 的 dependency_wheels，或命令本身失败）")
    result = json.loads(path.read_text(encoding="utf-8"))
    if "error" in result:
        raise WheelResolutionFailure(str(result["error"]))
    return result


# 包里的依赖声明：根的进程内文件（非空才算）与所有子进程 runtime 的 requirements.txt（非空）。
def requirement_files(code: Path) -> tuple[Path | None, list[Path]]:
    inprocess = code / INPROCESS_FILE
    runtimes: list[Path] = []
    for current, directories, files in os.walk(code):
        directories[:] = [name for name in directories if name not in _SKIP_DIRS]
        if "requirements.txt" in files and (Path(current) / "requirements.txt").read_text(encoding="utf-8").strip():
            runtimes.append(Path(current) / "requirements.txt")
    has = inprocess.is_file() and inprocess.read_text(encoding="utf-8").strip() != ""
    return (inprocess if has else None), sorted(runtimes)


def _site_packages(environment: Path) -> list[Path]:
    return sorted((environment / ".inprocess/.venv/lib").glob("python*/site-packages"))


# 已装外置插件的进程内环境里的 name==version（读 dist-info 的 METADATA），键为规范化包名。
def installed_pins(workspace: Path) -> dict[str, tuple[str, str]]:
    selection = workspace / "runtime/plugin-stable.json"
    pins: dict[str, tuple[str, str]] = {}
    if not selection.exists():
        return pins
    document = json.loads(selection.read_text(encoding="utf-8"))
    for record in sorted(document["inputs"].values(), key=lambda item: item["plugin_id"]):
        ref = record["python_environments"].get("inprocess/")
        if record["source_type"] != "installed" or ref is None:
            continue
        for site in _site_packages(workspace / _ENV_ROOT / ref):
            for metadata in site.glob("*.dist-info/METADATA"):
                fields = dict(line.split(": ", 1) for line in metadata.read_text(encoding="utf-8").splitlines()[:12]
                              if ": " in line)
                pins.setdefault(_normalize(fields["Name"]), (fields["Version"], record["plugin_id"]))
    return pins


# wheel 文件名 name-version-...whl 里的版本，供同一次部署里后面的包当约束。
def wheel_pins(directory: Path) -> dict[str, str]:
    return {_normalize(parts[0]): parts[1] for parts in (item.name.split("-") for item in directory.glob("*.whl"))}


# 与 Core 的 wheel_tree_sha256 同一序列化：文件名、NUL、小写十六进制 SHA-256、LF；只接受普通 .whl。
def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    entries = sorted(directory.iterdir(), key=lambda item: item.name.encode("utf-8"))
    if not entries:
        raise RuntimeError("离线 wheel 目录不能为空")
    for path in entries:
        if not path.is_file() or path.is_symlink() or path.suffix != ".whl":
            raise RuntimeError(f"离线 wheel 目录只能包含普通 .whl 文件: {path.name}")
        digest.update(path.name.encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\n")
    return digest.hexdigest()


# 为一个包求 wheel：没有依赖声明返回 None；pins 是已装其他插件加本次部署里先求好的包的版本，求完后并入。
def resolve_package_wheels(
    *, code: Path, plugin_id: str, workspace: Path, pins: dict[str, tuple[str, str]], resolver: Resolver, out: Path,
) -> dict[str, object] | None:
    inprocess, runtimes = requirement_files(code)
    if inprocess is None and not runtimes:
        return None
    name = plugin_id.partition("@")[0]
    with tempfile.TemporaryDirectory(prefix="akashic-wheels-work-") as directory:
        work = Path(directory)
        work.chmod(0o777)
        (work / "req").mkdir()
        files: dict[str, Path] = {}
        if inprocess is not None:
            files["inprocess.txt"] = inprocess
        for index, path in enumerate(runtimes):
            files[f"runtime-{index}.txt"] = path
        for label, path in files.items():
            shutil.copyfile(path, work / "req" / label)
        others: Mapping[str, tuple[str, str]] = {key: value for key, value in pins.items() if value[1] != plugin_id}
        spec = {"plugin": name, "inprocess": "inprocess.txt" if inprocess is not None else None,
                "runtimes": [label for label in files if label != "inprocess.txt"],
                "others": {key: value[0] for key, value in others.items()},
                "origins": {key: value[1] for key, value in others.items()}}
        (work / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        (work / "wheels").mkdir()
        (work / "wheels").chmod(0o777)
        result = resolver(work)
        destination = out / f"extra-{name}-wheels"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(work / "wheels", destination)
    for key, version in wheel_pins(destination).items():
        pins.setdefault(key, (version, plugin_id))
    return {"offline_wheels": {"relative_path": destination.name, "tree_sha256": tree_sha256(destination)},
            "core_lock_sha256": result["core_lock_sha256"]}
