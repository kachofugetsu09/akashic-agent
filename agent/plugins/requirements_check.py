"""插件依赖声明的统一校验：只接受普通 PEP 508 要求，其余一律拒绝；只依赖标准库（宿主脚本与镜像内共用）。

requirements 文件里可以写 pip 选项行（--no-binary、--index-url、-r、--find-links、--pre 等）和 URL/VCS/本地路径要求，
它们能让源码包的构建脚本执行或改写下载来源，所以联网解析与离线安装之前都要先过这一关。
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_SPEC = r"(?:===|==|!=|<=|>=|~=|<|>)\s*[A-Za-z0-9.*+!_-]+"
_REQUIREMENT = re.compile(
    rf"^{_NAME}(?:\s*\[\s*{_NAME}(?:\s*,\s*{_NAME})*\s*\])?"
    rf"(?:\s*\(?\s*{_SPEC}(?:\s*,\s*{_SPEC})*\s*\)?)?"
    r"(?:\s*;\s*[A-Za-z0-9_.\s\"'<>=!~()*,]+)?\s*$"
)
_ARCHIVE = re.compile(r"\.(?:tar(?:\.(?:gz|bz2|xz|zst))?|tgz|zip|whl)(?:\s|$)", re.IGNORECASE)


_CODING = re.compile(r"coding[:=]")


# 逐行校验；返回有效要求的行数。label 用于报错点名文件，行号从 1 起。
# pip 会按 BOM、注释里的编码声明或本机 locale 重新解码文件；为了让这里看到的文本与 pip 看到的不可能不同，
# 整个文件（含注释）只接受 ASCII，且不许出现反斜杠与编码声明。
def check_plain_requirements(text: str, label: str) -> int:
    if not text.isascii() or "\x00" in text:
        raise ValueError(f"{label} 含非 ASCII 字符（含注释与 BOM）；依赖声明只接受 ASCII")
    count = 0
    for number, raw in enumerate(text.splitlines(), 1):
        if "\\" in raw or _CODING.search(raw):
            raise ValueError(f"{label} 第 {number} 行含反斜杠或编码声明（会改变 pip 对文件的解释）: {raw.strip()}")
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        reason = _reject_reason(line)
        if reason is not None:
            raise ValueError(f"{label} 第 {number} 行{reason}: {line}")
        count += 1
    return count


def _reject_reason(line: str) -> str | None:
    if line.startswith("-"):
        return "是 pip 选项（--no-binary、--index-url、-r、--find-links 等不允许）"
    if "\\" in line or "://" in line or " @ " in line or line.startswith((".", "/", "~")) or "/" in line.split(";")[0]:
        return "不是普通包要求（URL、VCS、本地路径或续行不允许）"
    if _ARCHIVE.search(line):
        return "引用了本地包或源码包"
    if re.search(r"\s--?[A-Za-z]", line):
        return "同行带有 pip 选项"
    if _REQUIREMENT.fullmatch(line) is None:
        return "不是合法的 PEP 508 普通包要求"
    return None


# 插件自带的 wheel 目录：只收纯 Python wheel（py3-none-any）。自带 wheel 与插件自身代码同等受信（都是进程内任意代码），
# 限定纯 Python 是为了避免平台/ABI 问题，也让内容可以被人直接读懂；安装 wheel 不执行 setup.py，所以仍然不现编源码包。
BUNDLED_WHEELS_DIR = "wheels"
_BUNDLED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.]*-[A-Za-z0-9_.!+]+-py3-none-any\.whl$")
_MAX_BUNDLED_FILES = 32
_MAX_BUNDLED_BYTES = 256 * 1024 * 1024


# 校验并列出 wheels/ 里的文件；目录不存在返回空元组。名字不合格、不是普通文件、数量或体积超限都点名拒绝。
def check_bundled_wheels(directory: Path) -> tuple[Path, ...]:
    if directory.is_symlink():
        raise ValueError(f"{directory.name}/ 不能是符号链接")
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise ValueError(f"{directory.name} 必须是目录")
    entries = sorted(directory.iterdir(), key=lambda item: item.name)
    if len(entries) > _MAX_BUNDLED_FILES:
        raise ValueError(f"{directory.name}/ 文件过多（上限 {_MAX_BUNDLED_FILES}）")
    total = 0
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"{directory.name}/{entry.name} 不是普通文件")
        if _BUNDLED_NAME.fullmatch(entry.name) is None:
            raise ValueError(f"{directory.name}/{entry.name} 不是纯 Python wheel（只收 *-py3-none-any.whl）")
        total += entry.stat().st_size
    if total > _MAX_BUNDLED_BYTES:
        raise ValueError(f"{directory.name}/ 总体积超过上限 {_MAX_BUNDLED_BYTES // (1024 * 1024)}MB")
    for entry in entries:
        _check_pure_wheel(entry, directory.name)
    return tuple(entries)


_NATIVE_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")
_MAX_WHEEL_METADATA_BYTES = 64 * 1024


# 文件名只是声明：打开归档核对 WHEEL 元数据确为纯 Python（Root-Is-Purelib: true、标签只有 py3-none-any），且不含原生扩展成员。
def _check_pure_wheel(path: Path, label: str) -> None:
    where = f"{label}/{path.name}"
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = [name for name in names if name.count("/") == 1 and name.endswith(".dist-info/WHEEL")]
            if len(metadata) != 1:
                raise ValueError(f"{where} 不是合法的 wheel（缺少唯一的 .dist-info/WHEEL）")
            # WHEEL 元数据只有几行；先看声明大小，再对实际解压输出限长，避免小压缩包撑爆内存。
            if archive.getinfo(metadata[0]).file_size > _MAX_WHEEL_METADATA_BYTES:
                raise ValueError(f"{where} 的 WHEEL 元数据过大")
            with archive.open(metadata[0]) as stream:
                raw = stream.read(_MAX_WHEEL_METADATA_BYTES + 1)
            if len(raw) > _MAX_WHEEL_METADATA_BYTES:
                raise ValueError(f"{where} 的 WHEEL 元数据过大")
            text = raw.decode("utf-8")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise ValueError(f"{where} 不是合法的 wheel: {error}") from error
    fields = [line.partition(":") for line in text.splitlines() if ":" in line]
    values = [(key.strip().lower(), value.strip()) for key, _, value in fields]
    purelib = [value.lower() for key, value in values if key == "root-is-purelib"]
    tags = [value for key, value in values if key == "tag"]
    if purelib != ["true"] or tags != ["py3-none-any"]:
        raise ValueError(f"{where} 的 WHEEL 元数据不是纯 Python（Root-Is-Purelib={purelib}，Tag={tags}）")
    native = [name for name in names if name.lower().endswith(_NATIVE_SUFFIXES) or ".so." in name.lower()]
    if native:
        raise ValueError(f"{where} 含原生扩展成员: {native[0]}")

