"""Read only the rules that apply from the nearest Git root to the current cwd."""

from __future__ import annotations

from typing import cast

from .path_access import PathAccess

_MAX_BYTES = 32768


async def read_agents(path: str | None) -> dict[str, object]:
    """Discover rules on the execution backend and report any incomplete read.

    整链探测在执行端一次完成（agents_chain）；返回的 directory_status 同时
    承载原目录 inspect 结论，调用方不必再单独探测目录。
    """
    if path is None:
        return {"status": "unset", "directory_status": "unset", "sources": [], "files": []}
    async with PathAccess() as access:
        info = await access.read("agents_chain", path, max_bytes=_MAX_BYTES)
    directory_status = info.status
    if info.status == "available" and info.kind != "directory":
        directory_status = "not_directory"
    if info.chain is None:
        if info.status != "available":
            error = f"目录不可用 ({info.status}): {info.path}; {info.error or ''}"
        else:
            error = f"路径不是目录: {info.path}"
        return {"status": "unavailable", "directory_status": directory_status,
                "sources": [], "files": [], "error": error}
    chain = info.chain
    sources = [row.path for row in chain.files]
    if chain.failure is not None:
        failure = chain.failure
        if failure.kind == "probe":
            error = f"Git root probe failed ({failure.status}): {failure.path}"
        elif failure.kind == "read":
            error = f"AGENTS read failed ({failure.status}): {failure.path}; {failure.error or ''}"
        else:
            error = "AGENTS total exceeds 32 KiB"
        return {"status": "unavailable", "directory_status": directory_status,
                "sources": sources, "files": [], "error": error}
    assert chain.root is not None
    return {"status": "ready", "directory_status": directory_status, "root": chain.root,
            "sources": sources,
            "files": [{"path": row.path, "text": row.text} for row in chain.files]}


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
