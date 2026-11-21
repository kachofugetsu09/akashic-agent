#!/usr/bin/env python3
"""导出运行中实例的能力快照，并比较两份快照；用于插件外移前后的防回归核对。"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from pathlib import Path
from typing import TextIO, cast

JsonObject = dict[str, object]

_PLUGIN_REF = re.compile(r"^([A-Za-z0-9_.-]+)@[A-Za-z0-9_.-]+$")
# 迁移时会合法变化的来源字段；--ignore-source 时整体丢弃。
_SOURCE_KEYS = frozenset({"marketplace", "builtin", "source", "source_id"})
# 前端构建产物的哈希；构建方式变化（内置构建改为外置仓库构建）时会变，--ignore-build 时丢弃。
# contract_digests 是公开的前端合同摘要，不是构建哈希，始终参与比较。
_BUILD_KEYS = frozenset({"module_sha256", "stylesheet_sha256"})
_REF_KEYS = frozenset({"id", "owner"})
# 每个列表段落及其条目身份字段。
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("plugins", "id"), ("tools", "name"), ("commands", "name"), ("ui_slots", "owner"),
    ("web_modules", "owner"), ("dashboards", "owner"), ("settings_forms", "id"),
    ("skills", "name"), ("rpc_methods", "name"), ("documents", "id"),
)


class RpcClient:
    """最小同步 JSON-RPC 客户端，只依赖标准库；Unix 端点靠文件权限，TCP 端点带 workspace token。"""

    def __init__(self, endpoint: str, token: str | None) -> None:
        if endpoint.startswith("/"):
            sock = socket.socket(socket.AF_UNIX)
            sock.connect(endpoint)
        else:
            host, _, port = endpoint.rpartition(":")
            sock = socket.create_connection((host, int(port)))
        self._sock = sock
        self._file = sock.makefile("rwb")
        self._next_id = 1
        _ = self.call("initialize", {"protocolVersion": "2.0", "workspaceToken": token,
                                     "clientInfo": {"name": "capability-snapshot", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    def _send(self, payload: JsonObject) -> None:
        _ = self._file.write(json.dumps(payload).encode("utf-8") + b"\n")
        self._file.flush()

    # 发送一个请求并读取同 id 的响应；错误一律抛出，不降级成空数据。
    def call(self, method: str, params: JsonObject | None = None) -> JsonObject:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        while line := self._file.readline():
            reply = cast(JsonObject, json.loads(line))
            if reply.get("id") != request_id:
                continue
            if "error" in reply:
                raise RuntimeError(f"{method} 失败: {reply['error']}")
            return cast(JsonObject, reply["result"])
        raise ConnectionError(f"{method} 未收到响应")

    def close(self) -> None:
        self._file.close()
        self._sock.close()


# 1. 显式端点优先；2. 否则读 workspace 已发布的唯一 JSON-RPC 端点。
def find_endpoint(workspace: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    plan = cast(JsonObject, json.loads((workspace / "runtime/endpoints.json").read_text("utf-8")))
    found = [cast(JsonObject, item) for item in cast(list[object], plan["endpoints"])]
    found = [item for item in found if item["protocol"] in ("jsonrpc+unix", "jsonrpc+tcp")]
    if len(found) != 1:
        raise ConnectionError(f"需要唯一的 Gateway 端点，当前有 {len(found)} 个；请用 --endpoint 指定")
    return str(found[0]["address"])


# TCP 端点需要 workspace token；Unix 端点不需要。
def find_token(workspace: Path, endpoint: str, explicit: str | None) -> str | None:
    if explicit or endpoint.startswith("/"):
        return explicit
    return (workspace / ".app-server-token").read_text("utf-8").strip()


# 等宿主报告就绪，避免在启动或换代途中取到半成品。
def wait_ready(client: RpcClient, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not client.call("server/status").get("ready"):
        if time.monotonic() > deadline:
            raise TimeoutError("等待实例就绪超时")
        time.sleep(1)


# 插件清单：只留身份、版本与状态；generation、input_ref、提交号随每次发布变化，不进快照。
def _plugins(status: JsonObject) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for item in cast(list[JsonObject], status["plugins"]):
        name, _, market = str(item["plugin_id"]).partition("@")
        rows.append({
            "id": item["plugin_id"], "name": name, "marketplace": market,
            "builtin": item["distribution_available"], "installed": item["installed"],
            "enabled": item["enabled"], "state": item["state"], "version": item.get("version"),
            "load_error": item["load_error"],
        })
    return rows


# 技能：内置目录的绝对路径随部署变化，只保留来源类别与插件来源。
# skills 插件缺席是合法组合，记为空列表；其他不认识的返回形态照常失败。
def _skills(result: JsonObject) -> list[JsonObject]:
    unavailable = result.get("unavailable")
    if isinstance(unavailable, dict) and cast(JsonObject, unavailable).get("code") == "skills_unavailable":
        return []
    rows: list[JsonObject] = []
    for item in cast(list[JsonObject], result["items"]):
        source_id = item["source_id"] if item["source"] == "plugin" else None
        rows.append({"name": item["name"], "source": item["source"], "source_id": source_id,
                     "available": item["available"], "description": item["description"]})
    return rows


# 按各段落的身份字段排序，保证同一实例连续导出字节相同。
def _sorted(snapshot: JsonObject) -> JsonObject:
    for section, key in _SECTIONS:
        rows = cast(list[JsonObject], snapshot[section])
        snapshot[section] = sorted(rows, key=lambda row: str(row[key]))
    return snapshot


def collect(client: RpcClient) -> JsonObject:
    status = client.call("plugin/status")
    capabilities = client.call("inspection/capabilities.get")
    snapshot: JsonObject = {
        "schema": 1,
        "plugins": _plugins(status),
        "preset": status["preset"],
        "skills": _skills(client.call("inspection/skills.list")),
        "documents": client.call("inspection/documents.list")["items"],
        "tools": capabilities["tools"], "commands": capabilities["commands"],
        "ui_slots": capabilities["ui_slots"], "web_modules": capabilities["web_modules"],
        "dashboards": capabilities["dashboards"], "settings_forms": capabilities["settings_forms"],
        "rpc_methods": capabilities["rpc_methods"],
    }
    return _sorted(snapshot)


def encode(snapshot: JsonObject) -> str:
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


# 去掉指定字段；normalize_refs 时再把 name@market 归一成 name，使两侧同序。
def strip_keys(value: object, dropped: frozenset[str], normalize_refs: bool) -> object:
    if isinstance(value, list):
        return [strip_keys(item, dropped, normalize_refs) for item in cast(list[object], value)]
    if not isinstance(value, dict):
        return value
    result: JsonObject = {}
    for key, item in cast(JsonObject, value).items():
        if key in dropped:
            continue
        match = _PLUGIN_REF.match(item) if normalize_refs and key in _REF_KEYS and isinstance(item, str) else None
        result[key] = match.group(1) if match else strip_keys(item, dropped, normalize_refs)
    return result


# 把一个段落转成 身份 -> 条目 的字典。
def _index(snapshot: JsonObject, section: str, key: str) -> dict[str, JsonObject]:
    return {str(row[key]): row for row in cast(list[JsonObject], snapshot[section])}


# 逐字段比较两个条目，返回"字段: 旧 -> 新"。
def _field_changes(old: JsonObject, new: JsonObject) -> list[str]:
    lines: list[str] = []
    for field in sorted(old.keys() | new.keys()):
        if old.get(field) != new.get(field):
            before = json.dumps(old.get(field), ensure_ascii=False, sort_keys=True)
            after = json.dumps(new.get(field), ensure_ascii=False, sort_keys=True)
            lines.append(f"    ~ {field}: {_clip(before)} -> {_clip(after)}")
    return lines


def _clip(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


# 生成人能读的差异清单；没有差异时返回空列表。
def diff_snapshots(old: JsonObject, new: JsonObject) -> list[str]:
    lines: list[str] = []
    for section, key in _SECTIONS:
        before, after = _index(old, section, key), _index(new, section, key)
        block = [f"  - 少了 {name}" for name in sorted(before.keys() - after.keys())]
        block += [f"  + 多了 {name}" for name in sorted(after.keys() - before.keys())]
        for name in sorted(before.keys() & after.keys()):
            changes = _field_changes(before[name], after[name])
            if changes:
                block += [f"  * 变了 {name}", *changes]
        if block:
            lines += [f"[{section}]", *block]
    if old.get("preset") != new.get("preset"):
        lines += ["[preset]", f"  * {old.get('preset')} -> {new.get('preset')}"]
    return lines


def _load(path: str) -> JsonObject:
    return cast(JsonObject, json.loads(Path(path).read_text("utf-8")))


def run_export(args: argparse.Namespace) -> int:
    workspace = Path(cast(str, args.workspace)).expanduser()
    endpoint = find_endpoint(workspace, cast("str | None", args.endpoint))
    client = RpcClient(endpoint, find_token(workspace, endpoint, cast("str | None", args.token)))
    try:
        wait_ready(client, float(cast(float, args.wait)))
        text = encode(collect(client))
    finally:
        client.close()
    if args.out:
        _ = Path(cast(str, args.out)).write_text(text, encoding="utf-8")
    else:
        _ = cast(TextIO, sys.stdout).write(text)
    return 0


def run_diff(args: argparse.Namespace) -> int:
    old, new = _load(cast(str, args.old)), _load(cast(str, args.new))
    dropped = (_SOURCE_KEYS if args.ignore_source else frozenset[str]()) | (_BUILD_KEYS if args.ignore_build else frozenset[str]())
    if dropped:
        old = cast(JsonObject, strip_keys(old, dropped, bool(args.ignore_source)))
        new = cast(JsonObject, strip_keys(new, dropped, bool(args.ignore_source)))
    lines = diff_snapshots(old, new)
    print("\n".join(lines) if lines else "两份快照一致")
    return 1 if lines else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="从运行中的实例导出快照")
    _ = export.add_argument("--workspace", required=True, help="实例 workspace 目录")
    _ = export.add_argument("--endpoint", help="Gateway 端点（Unix 套接字路径或 host:port）；默认读 runtime/endpoints.json")
    _ = export.add_argument("--token", help="TCP 端点的 workspace token；默认读 <workspace>/.app-server-token")
    _ = export.add_argument("--out", help="输出文件；默认写标准输出")
    _ = export.add_argument("--wait", type=float, default=60, help="等待实例就绪的秒数")
    compare = commands.add_parser("diff", help="比较两份快照；有差异时退出码为 1")
    _ = compare.add_argument("old")
    _ = compare.add_argument("new")
    _ = compare.add_argument("--ignore-source", action="store_true", help="忽略来源（市场、内置/外置、技能来源）的变化")
    _ = compare.add_argument("--ignore-build", action="store_true", help="忽略前端模块与样式的构建哈希；合同摘要仍然比较")
    args = parser.parse_args()
    return run_export(args) if args.command == "export" else run_diff(args)


if __name__ == "__main__":
    raise SystemExit(main())
