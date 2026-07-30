from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, cast

RESOURCE_EVIDENCE_FILENAME = "container-resource.json"
RESOURCE_EVIDENCE_SCHEMA = "akasic.container-resource.v1"

_CGROUP_ROOT = "/sys/fs/cgroup"
_REQUIRED_FILES = ("memory.max", "memory.current", "memory.events")
_OPTIONAL_FILES = ("memory.peak", "memory.events.local")
_SECTION_HEADER = "@@"
_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def resource_probe_command() -> str:
    """生成只读取 cgroup v2 内存证据白名单的固定命令。"""

    # 1. required 文件缺失时非零退出，不能把未知环境伪装成无 OOM。
    required_checks = " ".join(
        f'test -r "{_CGROUP_ROOT}/{name}";' for name in _REQUIRED_FILES
    )

    # 2. 输出固定分节格式；不读取进程、环境变量、挂载或宿主日志。
    names = " ".join((*_REQUIRED_FILES, *_OPTIONAL_FILES))
    script = (
        "set -eu; "
        f"{required_checks} "
        'printf "cgroup_version=2\\n"; '
        f"for name in {names}; do "
        f'path="{_CGROUP_ROOT}/$name"; '
        'if test -r "$path"; then '
        f'printf "{_SECTION_HEADER}%s\\n" "$name"; '
        'cat "$path"; '
        "fi; "
        "done"
    )
    return f"sh -c {shlex.quote(script)}"


def parse_resource_probe_output(output: str) -> dict[str, object]:
    """校验固定探针输出并投影为脱敏数值证据。"""

    # 1. 分节必须完整且唯一，required 文件不能缺失。
    lines = output.splitlines()
    if not lines or lines[0] != "cgroup_version=2":
        raise ValueError("resource probe 缺少 cgroup v2 标识")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[1:]:
        if line.startswith(_SECTION_HEADER):
            current = line.removeprefix(_SECTION_HEADER)
            if current in sections:
                raise ValueError(f"resource probe 分节重复：{current}")
            if current not in {*_REQUIRED_FILES, *_OPTIONAL_FILES}:
                raise ValueError(f"resource probe 分节不在白名单：{current}")
            sections[current] = []
            continue
        if current is None:
            raise ValueError("resource probe 在首个分节前包含数据")
        sections[current].append(line)
    required: set[str] = set(_REQUIRED_FILES)
    missing = sorted(required - set(sections))
    if missing:
        raise ValueError(f"resource probe 缺少 required 分节：{missing}")

    # 2. 标量和事件都只接受内核数值格式。
    limit_raw = _parse_scalar(sections["memory.max"], allow_max=True)
    current_bytes = int(_parse_scalar(sections["memory.current"]))
    peak_bytes = (
        int(_parse_scalar(sections["memory.peak"]))
        if "memory.peak" in sections
        else None
    )
    events = _parse_events(sections["memory.events"])
    local_events = (
        _parse_events(sections["memory.events.local"])
        if "memory.events.local" in sections
        else None
    )
    oom_events = (
        events.get("oom", 0)
        + events.get("oom_kill", 0)
        + events.get("oom_group_kill", 0)
    )

    return {
        "schema": RESOURCE_EVIDENCE_SCHEMA,
        "status": "collected",
        "classification": "resource_limit" if oom_events > 0 else "none",
        "cgroup": {
            "version": 2,
            "memory": {
                "limit_bytes": None if limit_raw == "max" else int(limit_raw),
                "limit_raw": limit_raw,
                "current_bytes": current_bytes,
                "peak_bytes": peak_bytes,
                "events": events,
                "local_events": local_events,
            },
        },
    }


def resource_probe_failure(error: BaseException) -> dict[str, object]:
    """把探针失败保存成显式未知状态，不伪装为无资源事件。"""

    return {
        "schema": RESOURCE_EVIDENCE_SCHEMA,
        "status": "collection_failed",
        "classification": "unknown",
        "error": {
            "type": type(error).__name__,
            "message": str(error)[-2000:],
        },
    }


def load_resource_evidence(path: Path) -> dict[str, object]:
    """从 trial artifact 加载并校验资源证据的最小 envelope。"""

    if not path.is_file():
        return {
            "schema": RESOURCE_EVIDENCE_SCHEMA,
            "status": "unavailable",
            "classification": "unknown",
            "error": {
                "type": "MissingArtifact",
                "message": f"缺少 {RESOURCE_EVIDENCE_FILENAME}",
            },
        }
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("resource evidence 必须是对象")
        evidence = cast(dict[str, Any], payload)
        if evidence.get("schema") != RESOURCE_EVIDENCE_SCHEMA:
            raise ValueError("resource evidence schema 不匹配")
        if evidence.get("status") not in {"collected", "collection_failed"}:
            raise ValueError("resource evidence status 无效")
        if evidence.get("classification") not in {
            "none",
            "resource_limit",
            "unknown",
        }:
            raise ValueError("resource evidence classification 无效")
        return evidence
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return resource_probe_failure(error)


def _parse_scalar(lines: list[str], *, allow_max: bool = False) -> str:
    if len(lines) != 1:
        raise ValueError("resource probe 标量分节必须只有一行")
    value = lines[0]
    if allow_max and value == "max":
        return value
    if not value.isdecimal():
        raise ValueError(f"resource probe 标量不是非负整数：{value!r}")
    return value


def _parse_events(lines: list[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if (
            len(parts) != 2
            or not _EVENT_NAME.fullmatch(parts[0])
            or not parts[1].isdecimal()
        ):
            raise ValueError(f"resource probe event 格式无效：{line!r}")
        name, raw_value = parts
        if name in values:
            raise ValueError(f"resource probe event 重复：{name}")
        values[name] = int(raw_value)
    return values
