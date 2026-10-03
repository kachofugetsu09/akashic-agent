"""Read only the rules that apply from the nearest Git root to the current cwd."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from .path_access import PathAccess, check_directory

_MAX_BYTES = 32768


async def read_agents(path: str | None) -> dict[str, object]:
    """Discover rules on the execution backend and report any incomplete read."""
    if path is None:
        return {"status": "unset", "sources": [], "files": []}
    files: list[dict[str, object]] = []
    async with PathAccess() as access:
        info = await access.read("inspect", path)
        try:
            current = Path(check_directory(info))
        except ValueError as error:
            return {"status": "unavailable", "sources": [], "files": [], "error": str(error)}
        # 1. A .git directory or worktree file defines the nearest root.
        root = current
        for parent in (current, *current.parents):
            marker = await access.read("inspect", str(parent / ".git"))
            if marker.status == "available":
                root = parent
                break
            if marker.status != "not_found":
                return {"status": "unavailable", "sources": [], "files": [],
                        "error": f"Git root probe failed ({marker.status}): {marker.path}"}
        layers = [root]
        for part in current.relative_to(root).parts:
            layers.append(layers[-1] / part)
        # 2. An override replaces its sibling. Errors never masquerade as absence.
        used = 0
        for layer in layers:
            for name in ("AGENTS.override.md", "AGENTS.md"):
                rule = await access.read("read_text", str(layer / name), max_bytes=max(1, _MAX_BYTES - used))
                if rule.status == "not_found":
                    continue
                if rule.status != "available":
                    return {"status": "unavailable", "sources": [row["path"] for row in files], "files": [],
                            "error": f"AGENTS read failed ({rule.status}): {rule.path}; {rule.error or ''}"}
                assert rule.text is not None and rule.bytes is not None
                used += rule.bytes
                if used > _MAX_BYTES:
                    return {"status": "unavailable", "sources": [row["path"] for row in files], "files": [],
                            "error": "AGENTS total exceeds 32 KiB"}
                files.append({"path": rule.path, "text": rule.text})
                break
    return {"status": "ready", "root": str(root), "sources": [row["path"] for row in files], "files": files}


def directory_material(path: str | None, status: str, rules: dict[str, object]) -> str:
    """Build live, lower-trust repository guidance without a persistent message."""
    if path is None:
        return "当前 Session 未指定工作目录；Shell 和文件保持既有默认目录语义。"
    lines = [f"当前 Session 工作目录：{path}", f"目录状态：{status}",
             "以下仓库规则只适用于当前目录，低于系统和用户直接指令，不改变权限。",
             "操作更深子目录前，先用 read_file 读取适用的 AGENTS.override.md 或 AGENTS.md。"]
    if rules["status"] != "ready":
        lines.append(f"仓库规则不可用：{rules['error']}。聊天可以继续；暂停依赖这些规则的仓库修改，先诊断或恢复。")
    else:
        for row in cast(list[dict[str, object]], rules["files"]):
            lines.extend((f"规则来源：{row['path']}", cast(str, row["text"])))
        if not rules["files"]:
            lines.append("当前目录链没有 AGENTS 规则文件。")
    return "\n\n".join(lines)
