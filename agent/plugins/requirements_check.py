"""插件依赖声明的统一校验：只接受普通 PEP 508 要求，其余一律拒绝；只依赖标准库（宿主脚本与镜像内共用）。

requirements 文件里可以写 pip 选项行（--no-binary、--index-url、-r、--find-links、--pre 等）和 URL/VCS/本地路径要求，
它们能让源码包的构建脚本执行或改写下载来源，所以联网解析与离线安装之前都要先过这一关。
"""
from __future__ import annotations

import re

_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_SPEC = r"(?:===|==|!=|<=|>=|~=|<|>)\s*[A-Za-z0-9.*+!_-]+"
_REQUIREMENT = re.compile(
    rf"^{_NAME}(?:\s*\[\s*{_NAME}(?:\s*,\s*{_NAME})*\s*\])?"
    rf"(?:\s*\(?\s*{_SPEC}(?:\s*,\s*{_SPEC})*\s*\)?)?"
    r"(?:\s*;\s*[A-Za-z0-9_.\s\"'<>=!~()*,]+)?\s*$"
)
_ARCHIVE = re.compile(r"\.(?:tar(?:\.(?:gz|bz2|xz|zst))?|tgz|zip|whl)(?:\s|$)", re.IGNORECASE)


# 逐行校验；返回有效要求的行数。label 用于报错点名文件，行号从 1 起。
def check_plain_requirements(text: str, label: str) -> int:
    count = 0
    for number, raw in enumerate(text.splitlines(), 1):
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
