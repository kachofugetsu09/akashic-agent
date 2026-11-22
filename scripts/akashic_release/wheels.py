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
from typing import Callable, cast

from agent.plugins.extra_repo import PROXY_ENV
from agent.plugins.requirements_check import BUNDLED_WHEELS_DIR, check_bundled_wheels, check_plain_requirements

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
# 网络用宿主网络（--network host）：宿主代理常绑在 127.0.0.1，默认 bridge 容器里的同一地址指向容器自己。
# 只用于停机前的求包容器；停机窗口内的预检与安装仍是 --network none。
def docker_resolver(image: str, run: Run, cache: Path | None = None) -> Resolver:
    def resolve(work: Path) -> dict[str, object]:
        # 容器里取不到宿主环境：代理变量按名字透传（只在宿主设置了才传；证书文件路径在容器里没有意义，不传）。
        proxies = [item for key in PROXY_ENV if key in os.environ for item in ("--env", key)]
        command = ["docker", "run", "--rm", "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}", "--read-only", *proxies,
                   "--tmpfs", "/tmp:rw,mode=1777,size=2g", "--mount", f"type=bind,src={work},dst={work}",
                   *(() if cache is None else ("--mount", f"type=bind,src={cache},dst={cache}")),
                   "--workdir", "/opt/akashic/source", "--env", "HOME=/tmp", "--entrypoint", "/opt/venv/bin/python",
                   image, "-B", "-m", "agent.plugins.dependency_wheels", "--work", str(work),
                   *(() if cache is None else ("--cache", str(cache)))]
        return _read_result(work, run(command, check=False, capture_output=True, text=True))

    return resolve


# 不经 docker 的执行点：用本机（或容器内）的 Core 解释器与 pip；首装补装在容器里、以及本机验证用。
def local_resolver(core_root: Path, cache: Path | None = None) -> Resolver:
    def resolve(work: Path) -> dict[str, object]:
        done = subprocess.run([sys.executable, "-B", "-m", "agent.plugins.dependency_wheels", "--work", str(work),
                               *(() if cache is None else ("--cache", str(cache)))],
                              cwd=core_root, env={**os.environ, "PYTHONPATH": str(core_root)}, check=False,
                              capture_output=True, text=True)
        return _read_result(work, done)

    return resolve


# 目标解释器的 Python 主.次版本，惰性求值并记住：目标在镜像里就问镜像（起一次容器），否则是本机解释器。
# 显式给了 wheel_resolver 的调用方（本机验证）按本机解释器算。
def target_python(image: str | None, run: Run, resolver: Resolver | None) -> Callable[[], tuple[int, int]]:
    known: list[tuple[int, int]] = []

    def probe() -> tuple[int, int]:
        if not known:
            if image is None or resolver is not None:
                known.append((sys.version_info.major, sys.version_info.minor))
            else:
                code = "import sys;print('%d.%d' % sys.version_info[:2])"
                done = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/opt/venv/bin/python", image, "-c", code],
                           check=False, capture_output=True, text=True)
                major, _, minor = (done.stdout or "").strip().partition(".")
                if done.returncode != 0 or not (major.isdigit() and minor.isdigit()):
                    raise RuntimeError(f"无法读取目标镜像 {image} 的 Python 版本（退出码 {done.returncode}）：{(done.stderr or '').strip()[-300:]}；"
                                   "该版本用来判断已装插件的依赖环境是否需要重建，请确认镜像已拉取、docker 可用后重试（部署尚未停机）")
                known.append((int(major), int(minor)))
        return known[0]

    return probe


def _read_result(work: Path, done: subprocess.CompletedProcess[str]) -> dict[str, object]:
    path = work / "result.json"
    if not path.exists():
        # 命令自己失败（缺 pip、权限、导入错误、docker 报错）：带出真实原因，不用通用提示掩盖。
        detail = " | ".join(part for part in ((done.stderr or "").strip()[-600:], (done.stdout or "").strip()[-300:]) if part)
        raise RuntimeError(f"求依赖 wheel 的命令失败（退出码 {done.returncode}）：{detail or '无输出'}")
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


# 已装外置插件的进程内环境里的 name==version（读 dist-info 的 METADATA），键为规范化包名；exclude 是本批要替换的插件名。
def installed_pins(workspace: Path, exclude: frozenset[str] = frozenset()) -> dict[str, tuple[str, str]]:
    selection = workspace / "runtime/plugin-stable.json"
    pins: dict[str, tuple[str, str]] = {}
    if not selection.exists():
        return pins
    document = json.loads(selection.read_text(encoding="utf-8"))
    for record in sorted(document["inputs"].values(), key=lambda item: item["plugin_id"]):
        ref = record["python_environments"].get("inprocess/")
        if record["source_type"] != "installed" or ref is None or record["plugin_id"].partition("@")[0] in exclude:
            continue
        for site in _site_packages(workspace / _ENV_ROOT / ref):
            for metadata in site.glob("*.dist-info/METADATA"):
                fields = dict(line.split(": ", 1) for line in metadata.read_text(encoding="utf-8").splitlines()[:12]
                              if ": " in line)
                pins.setdefault(_normalize(fields["Name"]), (fields["Version"], record["plugin_id"]))
    return pins


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
            # 联网之前先校验（与镜像内入口同一份函数），点名文件与行。
            _ = check_plain_requirements(path.read_text(encoding="utf-8"), str(path.relative_to(code)))
            shutil.copyfile(path, work / "req" / label)
        # 插件自带的纯 Python wheel（只给进程内依赖用）：联网之前先校验，再随 work 目录交给求包容器。
        bundled = check_bundled_wheels(code / BUNDLED_WHEELS_DIR) if inprocess is not None else ()
        if bundled:
            (work / "bundled").mkdir()
            (work / "bundled").chmod(0o755)
            for wheel in bundled:
                shutil.copyfile(wheel, work / "bundled" / wheel.name)
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
    # 进程内依赖的新解析结果替换该插件旧 pin 并供本批后面的包作约束；子进程 runtime 的依赖是隔离的，不并入。
    inprocess_pins = cast(dict[str, str], result.get("inprocess_pins", {}))
    for key, version in inprocess_pins.items():
        pins[key] = (version, plugin_id)
    return {"offline_wheels": {"relative_path": destination.name, "tree_sha256": tree_sha256(destination)},
            "core_lock_sha256": result["core_lock_sha256"], **({"inprocess_pins": inprocess_pins} if inprocess is not None else {})}
