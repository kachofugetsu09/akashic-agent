"""插件资源：声明（resources.json）、取回、校验、发布；只依赖标准库，宿主脚本与镜像内共用。

资源是被 sha256 钉死的 zip（例如浏览器二进制）。Core 的代码负责下载、校验和解压，不执行插件代码，也不执行下载来的任何内容；
发布目录按内容寻址（<root>/<sha256>/），一经发布就不再改动，所以多个插件、多个版本可以共存，回退后旧版本仍能找到自己的资源。

发布目录布局：
  <root>/<sha256>/COMPLETE.json   完成标记：只描述归档内容（sha256、文件清单与大小）；没有它就不算已发布
  <root>/<sha256>/content/...     zip 的解压内容（目录 0555、文件去掉写权限，保留可执行位）
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import string
import stat
import sys
import tempfile
import time
import http.client
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, cast
from email.message import Message
from urllib.parse import quote, urljoin, urlsplit

RESOURCES_FILE = "resources.json"
MARKER = "COMPLETE.json"
CONTENT = "content"
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset({"name", "url", "sha256", "archive", "entry", "size"})
_REQUIRED = frozenset({"name", "url", "sha256", "archive", "entry"})
_MAX_RESOURCES = 8
_MAX_URL = 2048
DEFAULT_MAX_DOWNLOAD = 1 << 30
DEFAULT_MAX_EXTRACTED = 2 << 30
DEFAULT_MAX_FILES = 20000
_CHUNK = 1 << 20

Opener = Callable[[str, float], IO[bytes]]


class _ShortRead(OSError):
    """下载被截断或大小不符：网络类错误，可重试。"""


class ResourceError(RuntimeError):
    """资源声明无效，或取回、校验、解压失败；失败时不留下半成品。"""


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    sha256: str
    archive: str
    entry: str
    size: int | None = None


# 校验一个相对路径（zip 内路径或入口）：非空、不绝对、不含 ..、反斜杠、NUL 与空段。
def _safe_relative(value: str, label: str) -> PurePosixPath:
    if (not value or value.startswith("/") or "\\" in value or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ResourceError(f"{label} 不是安全的相对路径: {value!r}")
    return PurePosixPath(value)


def _resource(raw: object, label: str) -> Resource:
    if not isinstance(raw, dict):
        raise ResourceError(f"{label} 必须是对象")
    item = cast(dict[str, object], raw)
    if not _REQUIRED <= set(item) <= _FIELDS:
        raise ResourceError(f"{label} 字段不符：必须有 {sorted(_REQUIRED)}，可选 size，不允许其他字段")
    text = {key: item[key] for key in ("name", "url", "sha256", "archive", "entry")}
    if not all(isinstance(value, str) for value in text.values()):
        raise ResourceError(f"{label} 的 name、url、sha256、archive、entry 必须是字符串")
    name, url, sha256, archive, entry = (cast(str, text[key]) for key in ("name", "url", "sha256", "archive", "entry"))
    if _NAME.fullmatch(name) is None:
        raise ResourceError(f"{label} name 无效: {name!r}")
    parts = urlsplit(url)
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None
            or len(url) > _MAX_URL or parts.fragment):
        raise ResourceError(f"{label} url 必须是不带凭据的 https 地址: {url!r}")
    if _SHA256.fullmatch(sha256) is None:
        raise ResourceError(f"{label} 必须带 64 位小写十六进制 sha256")
    if archive != "zip":
        raise ResourceError(f"{label} archive 目前只支持 zip: {archive!r}")
    _ = _safe_relative(entry, f"{label} entry")
    size = item.get("size")
    if size is not None and (type(size) is not int or not 0 < size <= DEFAULT_MAX_DOWNLOAD):
        raise ResourceError(f"{label} size 必须是不超过 {DEFAULT_MAX_DOWNLOAD} 的正整数")
    return Resource(name, url, sha256, archive, entry, cast("int | None", size))


# 解析 resources.json 文本；只接受 ASCII，顶层字段与每个资源的字段都精确校验，名字不得重复。
def parse_manifest(text: str, label: str = RESOURCES_FILE) -> tuple[Resource, ...]:
    if not text.isascii() or "\x00" in text:
        raise ResourceError(f"{label} 只接受 ASCII")
    try:
        document = json.loads(text)
    except ValueError as error:
        raise ResourceError(f"{label} 不是合法 JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "resources"} or document["schema_version"] != 1:
        raise ResourceError(f"{label} 顶层必须恰好是 schema_version=1 与 resources")
    items = document["resources"]
    if not isinstance(items, list) or not 0 < len(items) <= _MAX_RESOURCES:
        raise ResourceError(f"{label} resources 必须是 1 到 {_MAX_RESOURCES} 个条目的数组")
    resources = tuple(_resource(item, f"{label} resources[{index}]") for index, item in enumerate(cast(list[object], items)))
    if len({item.name for item in resources}) != len(resources):
        raise ResourceError(f"{label} 资源名重复")
    return resources


# 插件根的声明；没有 resources.json 返回空元组。
def read_manifest(plugin_root: Path) -> tuple[Resource, ...]:
    path = plugin_root / RESOURCES_FILE
    if path.is_symlink() or not path.is_file():
        return ()
    return parse_manifest(path.read_text(encoding="utf-8"), RESOURCES_FILE)


def published_path(resource: Resource, root: Path) -> Path:
    return root / resource.sha256


# 资源入口文件的绝对路径（已发布才有意义）。
def entry_path(resource: Resource, root: Path) -> Path:
    return published_path(resource, root) / CONTENT / resource.entry


# 已发布内容是否完好：完成标记存在、sha256 一致、清单里的文件都在且大小相同、没有多出文件。标记只描述归档内容
# （按 sha256 寻址，与哪个插件、哪个入口无关）；返回 None 表示完好，否则返回原因。
# 合同：这是受控持久目录的完整性快检，只比对文件名与长度，查不出同长度的字节篡改或执行位变化，也不抵御有目录写权限的攻击者。
def content_problem(resource: Resource, root: Path) -> str | None:
    target = published_path(resource, root)
    marker = target / MARKER
    if not marker.is_file():
        return "尚未发布（没有完成标记）"
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        files = cast(dict[str, int], record["files"])
        if record["sha256"] != resource.sha256:
            return "完成标记与声明不一致"
    except (OSError, ValueError, KeyError, TypeError):
        return "完成标记损坏"
    content = target / CONTENT
    actual = {path.relative_to(content).as_posix(): path.stat().st_size
              for path in content.rglob("*") if path.is_file() and not path.is_symlink()} if content.is_dir() else {}
    return None if actual == files else "已发布的文件与清单不一致（缺文件、多文件或大小不符）"


# 调用方要用的入口：必须是已发布内容里的普通文件。入口不同不改变内容身份，所以这里只报告，绝不触发删除或重取。
def entry_problem(resource: Resource, root: Path) -> str | None:
    entry = entry_path(resource, root)
    if entry.is_symlink() or not entry.is_file():
        return f"入口 {resource.entry} 不在已发布内容里或不是普通文件"
    return None


# 资源是否可用（内容完好且入口存在）；返回 None 表示可用。
def verify_resource(resource: Resource, root: Path) -> str | None:
    return content_problem(resource, root) or entry_problem(resource, root)


# 把 path 所在文件系统的脏数据整体刷盘；没有 syncfs 时逐文件 fsync。真实写回失败直接抛出，不能发布。
def _sync(path: Path, tree: Path) -> None:
    try:
        syncfs = ctypes.CDLL(None, use_errno=True).syncfs
    except (AttributeError, OSError):
        syncfs = None
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if syncfs is not None:
            if syncfs(descriptor) != 0:
                number = ctypes.get_errno()
                raise OSError(number, f"syncfs 失败: {os.strerror(number)}", str(path))
            return
    finally:
        os.close(descriptor)
    for current, _, names in os.walk(tree, topdown=False):
        for name in names:
            with (Path(current) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        directory = os.open(current, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


# 删除目录树；发布后的目录是只读的，先放开权限。
def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for current, directories, _ in os.walk(path):
        for name in directories:
            (Path(current) / name).chmod(0o755)
    path.chmod(0o755)
    shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _locked(root: Path, sha256: str) -> Generator[None]:
    with (root / f".lock-{sha256}").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


_MAX_REDIRECTS = 5
_MAX_SECONDS = 1800.0


# 整个取回（含全部重试与重定向）共用的尽力期限：只在请求之间、块之间与成功出口检查，并把每次请求的 socket 超时收紧到剩余时间；
# 不能打断 DNS、连接或单次阻塞读（慢速滴流时可能远超期限），健康复用与解压落盘阶段也不检查。需要硬上限的调用方应在进程层面加超时。
class _Deadline:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._end = time.monotonic() + seconds

    # 剩余秒数；已超期抛 ResourceError。
    def remaining(self) -> float:
        left = self._end - time.monotonic()
        if left <= 0:
            raise ResourceError(f"下载超过总时限 {self.seconds:.0f} 秒")
        return left


# 校验一个下载地址（初始地址与每一跳重定向目标共用）：https、有主机名、不带凭据、不带片段。
def _check_url(url: str, label: str) -> None:
    parts = urlsplit(url)
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None
            or len(url) > _MAX_URL or parts.fragment):
        raise ResourceError(f"{label} 必须是不带凭据的 https 地址: {url!r}")


# 重定向在发出后续请求之前逐跳校验；不读取 30x 响应体（标准库默认实现会无界读入内存），也不转发原请求的其他头。
# Location 按标准库的做法处理：http.client 把头按 ISO-8859-1 解码，这里还原原始字节并百分号编码非 ASCII 部分。
class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, deadline: _Deadline) -> None:
        self.hops = 0
        self._deadline = deadline

    def http_error_302(self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: Message) -> IO[bytes] | None:
        # 1. 必须有目标；跳数有上限。
        location = headers.get("location") or headers.get("uri")
        self.hops += 1
        if not location or self.hops > _MAX_REDIRECTS:
            fp.close()
            raise ResourceError(f"重定向缺少目标或超过 {_MAX_REDIRECTS} 跳（第 {self.hops} 跳，来自 {req.full_url}）")
        # 2. 编码、解析相对地址，在发请求前校验；失败说明是哪一跳。
        target = urljoin(req.full_url, quote(location, encoding="iso-8859-1", safe=string.punctuation))
        fp.close()
        _check_url(target, f"重定向第 {self.hops} 跳目标")
        timeout = min(cast(float, req.timeout), self._deadline.remaining())
        return self.parent.open(urllib.request.Request(target, headers={"User-Agent": "akashic-resource-fetch"}), timeout=timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


# 默认打开方式：代理按环境变量；每次下载一个新的重定向 handler（跳数不跨下载累计），超时受同一个期限约束。
def _deadline_opener(deadline: _Deadline) -> Opener:
    def open_url(url: str, timeout: float) -> IO[bytes]:
        _check_url(url, "下载地址")
        opener = urllib.request.build_opener(_CheckedRedirects(deadline))
        request = urllib.request.Request(url, headers={"User-Agent": "akashic-resource-fetch"})
        return cast(IO[bytes], opener.open(request, timeout=min(timeout, deadline.remaining())))
    return open_url


# 下载一次：流式写入、同时计算哈希；超过上限、大小不符、哈希不符都失败。网络类错误由调用方重试。
def _download(resource: Resource, target: Path, opener: Opener, timeout: float, limit: int, deadline: _Deadline) -> int:
    if resource.size is not None and resource.size > limit:
        raise ResourceError(f"声明的 size {resource.size} 超过调用方上限 {limit}")
    cap = limit if resource.size is None else min(limit, resource.size)
    digest = hashlib.sha256()
    total = 0
    with opener(resource.url, min(timeout, deadline.remaining())) as response, target.open("wb") as stream:
        while chunk := response.read(_CHUNK):
            total += len(chunk)
            _ = deadline.remaining()
            if total > cap:
                raise ResourceError(f"下载超过上限 {cap} 字节")
            digest.update(chunk)
            _ = stream.write(chunk)
    _ = deadline.remaining()
    if resource.size is not None and total != resource.size:
        raise _ShortRead(f"下载大小 {total} 与声明 {resource.size} 不符")
    if digest.hexdigest() != resource.sha256:
        raise ResourceError(f"sha256 不符：期望 {resource.sha256}，实际 {digest.hexdigest()}")
    return total


# 下载并校验一个归档到 target（不解压）。deadline 由调用方在整个取回开始时定一次，重试不重置；
# 网络类错误（连接、超时、大小不符）最多重试 attempts 次；哈希不符、超限、地址或头部非法等内容问题不重试，统一成 ResourceError。
def download_archive(
    resource: Resource, target: Path, *, opener: Opener | None = None, timeout: float = 60.0, attempts: int = 3,
    max_download: int = DEFAULT_MAX_DOWNLOAD, deadline: _Deadline | None = None,
) -> int:
    clock = deadline or _Deadline(_MAX_SECONDS)
    for attempt in range(1, attempts + 1):
        _ = clock.remaining()
        try:
            return _download(resource, target, opener or _deadline_opener(clock), timeout, max_download, clock)
        except (UnicodeError, http.client.HTTPException) as error:
            raise ResourceError(f"下载响应不合法: {error}") from error
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            if attempt == attempts:
                raise ResourceError(f"下载失败（已重试 {attempts} 次）: {error}") from error
            time.sleep(min(2.0 * attempt, 10.0, clock.remaining()))
    raise AssertionError("attempts 必须为正")


# 安全解压：拒绝绝对路径、..、符号链接与特殊文件、重名；还原可执行位（去掉 setuid/setgid/sticky）；
# 用实际写出的字节数限制总量（zip 头里的大小可以说谎），文件数也有上限。
def _extract(archive: Path, content: Path, max_extracted: int, max_files: int) -> dict[str, int]:
    written: dict[str, int] = {}
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > max_files:
            raise ResourceError(f"压缩包条目过多（{len(infos)} > {max_files}）")
        for info in infos:
            relative = _safe_relative(info.filename.rstrip("/") if info.is_dir() else info.filename, "压缩包条目")
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ResourceError(f"压缩包含符号链接或特殊文件: {info.filename!r}")
            destination = content / relative
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if relative.as_posix() in written or destination.exists():
                raise ResourceError(f"压缩包条目重复: {info.filename!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with bundle.open(info) as source, destination.open("xb") as stream:
                while chunk := source.read(_CHUNK):
                    size += len(chunk)
                    total += len(chunk)
                    if total > max_extracted:
                        raise ResourceError(f"解压总量超过上限 {max_extracted} 字节")
                    _ = stream.write(chunk)
            executable = bool((info.external_attr >> 16) & 0o111)
            destination.chmod(0o755 if executable else 0o644)
            written[relative.as_posix()] = size
    return written


# 发布前把目录改成只读：目录 0555、文件去掉写权限（保留可执行位）。
def _freeze(stage: Path) -> None:
    for current, directories, names in os.walk(stage, topdown=False):
        for name in names:
            path = Path(current) / name
            path.chmod(path.stat().st_mode & 0o555)
        for name in directories:
            (Path(current) / name).chmod(0o555)
    stage.chmod(0o555)


# 取回并发布一个资源，返回发布目录。已发布且可用则直接复用；同一资源并发取回由文件锁串行化，后到者复用先到者的结果。
# 网络类错误（连接、超时、大小不符）最多重试 attempts 次；哈希不符、路径穿越、超限等内容问题不重试。
# timeout 是单次 socket 阻塞操作的超时；max_seconds 是整个取回（含重试与重定向）的尽力期限，只在请求之间、块之间和成功出口检查，
# 不能打断 DNS、连接或单次阻塞读；需要硬上限的调用方自行在进程层面加超时。
def fetch_resource(
    resource: Resource, root: Path, *, opener: Opener | None = None, timeout: float = 60.0, attempts: int = 3,
    max_download: int = DEFAULT_MAX_DOWNLOAD, max_extracted: int = DEFAULT_MAX_EXTRACTED, max_files: int = DEFAULT_MAX_FILES,
    max_seconds: float = _MAX_SECONDS,
) -> Path:
    deadline = _Deadline(max_seconds)
    root.mkdir(parents=True, exist_ok=True)
    target = published_path(resource, root)
    with _locked(root, resource.sha256):
        # 1. 内容完好：复用（入口不对只报错，不删除，不重取——同一归档的内容不会因换入口而变）；内容损坏：清掉重取。
        if content_problem(resource, root) is None:
            if (reason := entry_problem(resource, root)) is not None:
                raise ResourceError(reason)
            return target
        _remove_tree(target)
        for stale in root.glob(f".stage-{resource.sha256[:16]}-*"):
            _remove_tree(stale)
        stage = Path(tempfile.mkdtemp(prefix=f".stage-{resource.sha256[:16]}-", dir=root))
        try:
            # 2. 下载（网络类错误重试）、校验、解压到暂存目录。
            archive = stage / "download.zip"
            downloaded = download_archive(
                resource, archive, opener=opener, timeout=timeout, attempts=attempts, max_download=max_download, deadline=deadline)
            files = _extract(archive, stage / CONTENT, max_extracted, max_files)
            archive.unlink()
            # 3. 入口必须在解压结果里；写完成标记、刷盘、只读化，最后原子 rename 发布。
            if resource.entry not in files:
                raise ResourceError(f"入口 {resource.entry} 不在压缩包里")
            marker: dict[str, object] = {"schema_version": 1, "sha256": resource.sha256, "archive": resource.archive,
                                         "downloaded": downloaded, "files": files}
            _ = (stage / MARKER).write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
            _sync(root, stage)
            _freeze(stage)
            os.rename(stage, target)
            _sync(root, target)
        except BaseException:
            _remove_tree(stage)
            raise
    return target


def _main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.plugins.resources")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify"):
        command = commands.add_parser(name)
        _ = command.add_argument("--manifest", type=Path, required=True)
        _ = command.add_argument("--root", type=Path, required=True)
        _ = command.add_argument("--name")
    options = parser.parse_args(arguments)
    try:
        selected = [item for item in parse_manifest(cast(Path, options.manifest).read_text(encoding="utf-8"))
                    if options.name in (None, item.name)]
        if options.name is not None and not selected:
            raise ResourceError(f"声明里没有资源 {options.name}")
        report: list[dict[str, object]] = []
        for item in selected:
            root = cast(Path, options.root)
            reason = verify_resource(item, root) if options.command == "verify" else None
            if options.command == "fetch":
                _ = fetch_resource(item, root)
            report.append({"name": item.name, "sha256": item.sha256, "path": str(entry_path(item, root)),
                           "usable": reason is None, **({"reason": reason} if reason else {})})
        print(json.dumps(report))
        return 0 if all(entry["usable"] for entry in report) else 1
    except (ResourceError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
