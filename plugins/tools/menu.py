from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from agent.plugin_composition.bindings import Bindings
from session.log import MessageReader
from session.message import CallRef, ContentPart, ContentReferences, Control, Output, ToolCall, ToolResult, freeze_json
from .execution import MessageReply, Result, ToolExecution
from .plugin import TOOLS

if TYPE_CHECKING:
    from .plugin import ToolCatalog


def check_menu(catalog: ToolCatalog, names: Sequence[str]) -> None:
    """在程序启动前检查显式目录，配置错误不能污染用户输入后的日志。"""
    descriptions = {cast(str, item["name"]): item for item in catalog.descriptions()}
    if set(names) - descriptions.keys():
        raise ValueError("回复配置包含未安装的工具")
    selected = [descriptions[name] for name in names]
    if any(item["requires_search"] for item in selected) and not any(item["discovery"] for item in selected):
        raise ValueError("要求搜索解锁的工具缺少发现能力")


class ToolMenu:
    """来源允许的目录与日志选择投影；不另存解锁状态或依赖运行 attempt。"""

    def __init__(
        self, catalog: ToolCatalog, bindings: Bindings, execution: ToolExecution,
        reply: Callable[[CallRef], MessageReply], *, names: Sequence[str],
        reader: MessageReader, source: str, limit: int = 8,
    ):
        if limit < 1:
            raise ValueError("工具菜单容量必须为正数")
        check_menu(catalog, names)
        descriptions = {cast(str, item["name"]): item for item in catalog.descriptions()}
        self._allowed: dict[str, Mapping[str, object]] = {name: descriptions[name] for name in names}
        self._catalog = catalog
        self._bindings = bindings
        self._execution = execution
        self._reply = reply
        self._reader = reader
        self._source = source
        self._limit = limit
        self._bound: dict[str, str] = {}
        self._selected: dict[str, str] = {}
        discovery = [name for name, item in self._allowed.items() if item["discovery"]]
        self._discovery = bool(discovery)

        # 1. 发现工具只捕获普通候选，候选不复制目录也不递归绑定发现工具。
        if discovery:
            rows: dict[str, Mapping[str, object]] = {
                name: {"binding_id": self._bind_current(name), "tool": item}
                for name, item in self._allowed.items() if not item["discovery"]
            }
            candidates = cast(Mapping[str, Mapping[str, object]], freeze_json(rows))
            for name in discovery:
                self._bound[name] = catalog.bind(name, bindings, candidates=candidates)

    def _bind_current(self, name: str) -> str:
        if name not in self._bound:
            self._bound[name] = self._catalog.bind(name, self._bindings)
        return self._bound[name]

    def _description(self, binding_id: str) -> Mapping[str, object]:
        return cast(Mapping[str, object], self._bindings.describe(binding_id, TOOLS)["tool"])

    def _refresh(self) -> None:
        """预载使用当前目录；未闭合工作中的显式选择保留原 binding。"""
        messages = [m for m in self._reader.snapshot() if m.source == self._source]
        recent: dict[str, str] = {}
        boundary = -1
        for message in messages:
            body = message.body
            if isinstance(body, Output) and body.finish != "continue":
                boundary = message.seq
            elif isinstance(body, Control) and body.action == "abandon":
                boundary = max(boundary, body.through_seq)

        # 2. 闭段只提供可预载名称；本段选择和实际调用严格按日志先后更新 LRU。
        for message in messages:
            body = message.body
            if isinstance(body, Output):
                for part in body.parts:
                    if not isinstance(part, ToolCall):
                        continue
                    name = self.name(part.binding_id)
                    item = self._allowed.get(name)
                    if item is None or item["discovery"]:
                        continue
                    if message.seq <= boundary:
                        if not item["preloadable"] or item["requires_search"]:
                            continue
                        identity = self._bind_current(name)
                    else:
                        identity = part.binding_id
                    _ = recent.pop(name, None)
                    recent[name] = identity
            elif message.seq > boundary and isinstance(body, ToolResult) and body.outcome == "success":
                request = self._reader.get(body.call_ref.message_id)
                if request is None or request.seq <= boundary:
                    continue
                for part in body.parts:
                    if part.kind != "tool.selection":
                        continue
                    refs = self.check_selection(body.call_ref, part).binding_ids
                    # 选择按相关性排序；容量不足时优先保留最相关的候选。
                    for identity in reversed(refs):
                        name = self.name(identity)
                        if name in self._allowed:
                            _ = recent.pop(name, None)
                            recent[name] = identity

        fixed = {name: self._bind_current(name) for name, item in self._allowed.items()
                 if item["always_on"] or not self._discovery}
        if self._discovery:
            remaining = max(0, self._limit - len(fixed))
            extra = [(name, identity) for name, identity in recent.items() if name not in fixed]
            fixed.update(extra[-remaining:] if remaining else ())
        self._selected = fixed

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        self._refresh()
        return tuple(
            {"type": "function", "function": {
                "name": name, "description": item["description"], "parameters": item["parameters"]
            }}
            for name, identity in self._selected.items()
            for item in (self._description(identity),)
        )

    def bind(self, name: str) -> str:
        if name not in self._selected:
            raise PermissionError(f"模型未获授工具: {name}")
        return self._selected[name]

    def name(self, binding_id: str) -> str:
        return cast(str, self._description(binding_id)["name"])

    def check_call(self, call: ToolCall) -> None:
        if call.binding_id not in self._selected.values():
            raise PermissionError("工具请求不属于本次已固定的工具集合")

    def check_selection(self, ref: CallRef, part: ContentPart) -> ContentReferences:
        """选择只能来自实际发现调用捕获的候选，普通工具不能伪造解锁。"""
        request = self._reader.get(ref.message_id)
        if request is None or not isinstance(request.body, Output):
            raise ValueError("工具选择缺少已提交调用")
        call = request.body.parts[ref.part_index]
        if not isinstance(call, ToolCall):
            raise ValueError("工具选择引用未指向调用")
        metadata = self._bindings.describe(call.binding_id, TOOLS)
        description = cast(Mapping[str, object], metadata["tool"])
        if not description["discovery"]:
            raise ValueError("普通工具不能输出工具选择")
        candidates = cast(Mapping[str, Mapping[str, object]], metadata["candidates"])
        allowed = {item["binding_id"] for item in candidates.values()}
        value = part.value
        if not isinstance(value, tuple):
            raise ValueError("工具选择必须是 binding ID 数组")
        values = cast(tuple[object, ...], value)
        if any(not isinstance(item, str) or item not in allowed for item in values):
            raise ValueError("选择不属于原调用捕获的候选")
        identities = cast(tuple[str, ...], value)
        if len(set(identities)) != len(identities):
            raise ValueError("工具选择不能重复")
        return ContentReferences(binding_ids=identities)

    async def execute(self, call: CallRef) -> Result:
        reply = self._reply(call)
        try:
            return await self._execution.execute_call(reply)
        finally:
            reply.writer.expire()
