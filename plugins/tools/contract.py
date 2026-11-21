"""工具的公共合同；注册身份、允许目录和展示各自独立于提供方实现。"""

from __future__ import annotations

from contextvars import ContextVar, Token
import json
import math
import re
from copy import deepcopy
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable, cast
import xml.etree.ElementTree as ET
from plugins.models.contract import ToolCall as ModelToolCall

from plugins.ledger.contract import Bindings
from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from plugins.ledger.contract import MessageReader, OwnerTransaction
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.tasks import ExternalRootPermit, Task
from plugins.ledger.contract import (
    CallRef,
    ContentPart,
    ContentReferences,
    Message,
    ToolCall,
)

Outcome = Literal["success", "denied", "error", "interrupted"]
ToolIcon = str | Mapping[str, object]


def durable_call_key(call_ref: CallRef) -> str:
    """Return the stable effect key already used by a submitted ToolCall."""
    if not isinstance(call_ref, CallRef):
        raise TypeError("工具调用引用无效")
    return "message:" + json.dumps(
        [call_ref.message_id, call_ref.part_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@runtime_checkable
class ResultLike(Protocol):
    """provider 返回的结构结果；tools owner 不依赖 provider 的类身份。"""

    @property
    def outcome(self) -> Outcome: ...
    @property
    def parts(self) -> tuple[ContentPart, ...]: ...


@dataclass(frozen=True, slots=True)
class CallSource:
    """实际调用的不可变消息前缀；不携带 reader 或任何写入能力。"""

    call_ref: CallRef
    messages: tuple[Message, ...]

    @property
    def effect_key(self) -> str:
        return durable_call_key(self.call_ref)


class ProviderBoundTool(Protocol):
    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ResultLike: ...

    async def query(self, key: str) -> ResultLike | None:
        """查询原调用；None 只表示无法确定，不能解释为没有效果。"""
        ...


@dataclass(frozen=True, slots=True)
class ToolRef:
    """引用当前 composition Root 中的一次真实工具注册。"""

    name: str
    description: Mapping[str, object]


def tool_key(name: str) -> ServiceKey[ToolRef]:
    """声明对单个已注册工具的依赖，随工具 owner 激活与释放。"""
    return ServiceKey[ToolRef](f"tools.ref.{name}.v1")


@dataclass(frozen=True, slots=True)
class ToolView:
    """消费者获授的一组真实工具引用。"""

    refs: tuple[ToolRef, ...]

    def __post_init__(self) -> None:
        refs = tuple(self.refs)
        names = tuple(ref.name for ref in refs)
        if len(set(names)) != len(names):
            raise ValueError("工具 view 不能包含重复名称")
        object.__setattr__(self, "refs", refs)

    def select(self, name: str) -> ToolRef:
        for ref in self.refs:
            if ref.name == name:
                return ref
        raise PermissionError(f"工具不属于获授 view: {name}")

    def without(self, names: frozenset[str]) -> ToolView:
        return ToolView(tuple(ref for ref in self.refs if ref.name not in names))

    @classmethod
    def combine(cls, *views: ToolView) -> ToolView:
        return cls(tuple(ref for view in views for ref in view.refs))


@dataclass(frozen=True, slots=True)
class Result:
    outcome: Outcome
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "denied", "error", "interrupted"}:
            raise ValueError("工具结果状态无效")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("工具结果必须是内容块")
        object.__setattr__(self, "parts", parts)


class BoundTool(Protocol):
    """tools owner 暴露给执行器的已归一化工具 facade。"""

    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result: ...

    async def query(self, key: str) -> Result | None:
        """查询原调用；None 只表示无法确定，不能解释为没有效果。"""
        ...


class ToolRelease(Protocol):
    """工具提供方释放某次调用占用的外部资源；返回失败说明，空表示已释放。"""

    async def __call__(
        self, state: Mapping[str, object], session_id: str, source: str, boundary: str | None,
    ) -> tuple[str, ...]: ...


class ToolCatalog(Protocol):
    """注册表拥有真实引用；消费者只选择本次允许的目录。"""

    async def declare_group(
        self,
        ctx: Context,
        *,
        always_on: bool = False,
        description: str = "未声明用途",
        icon: ToolIcon | None = None,
    ) -> Effect: ...
    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, object],
        open: Callable[
            [Mapping[str, object]], AbstractAsyncContextManager[ProviderBoundTool]
        ],
        capture: Callable[[Mapping[str, object]], Mapping[str, object] | Awaitable[Mapping[str, object]]] | None = None,
        public: bool = True,
        idempotent: bool = False,
        parallel: bool = False,
        exclusive_batch: bool = False,
        release: ToolRelease | None = None,
        icon: ToolIcon | None = None,
        label: str | None = None,
    ) -> ToolRef: ...
    async def register_prepare(
        self,
        ctx: Context,
        *,
        tool: ToolRef,
        name: str,
        prepare: Callable[[Mapping[str, object]], Awaitable[Mapping[str, object]]],
    ) -> Effect: ...
    async def register_authorize(
        self,
        ctx: Context,
        *,
        tool: ToolRef,
        name: str,
        authorize: Callable[[Mapping[str, object]], Awaitable[str | None]],
    ) -> Effect: ...
    def view(self, *refs: ToolRef) -> ToolView: ...
    def allows_parallel(self, name: str) -> bool:
        """调度只读注册事实；未知工具一律不能重叠。"""
        ...
    def group_description(self, ref: ToolRef) -> str: ...
    def group_always_on(self, ref: ToolRef) -> bool: ...
    async def bind(
        self,
        ref: ToolRef,
        bindings: Bindings,
        *,
        configuration: Mapping[str, object] | None = None,
    ) -> str: ...
    async def bind_scoped(
        self,
        ref: ToolRef,
        bindings: Bindings,
        *,
        configuration: Mapping[str, object] | None = None,
    ) -> str: ...
    async def bind_saved(
        self,
        metadata: Mapping[str, object],
        bindings: Bindings,
        *,
        configuration: Mapping[str, object],
    ) -> str: ...
    def open(
        self, metadata: Mapping[str, object]
    ) -> AbstractAsyncContextManager[BoundTool]: ...
    async def authorize(
        self, metadata: Mapping[str, object], arguments: Mapping[str, object]
    ) -> str | None: ...
    async def drain_calls(self, calls: tuple[CallRef, ...]) -> None: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")
ALL_TOOLS = ServiceKey[Callable[[], ToolView]]("tools.all.v1")
@dataclass(frozen=True, slots=True)
class ToolDisplay:
    """聊天展示一次调用所需的身份与可选图标、显示名：工具名、注册它的插件；没有记录 owner 的旧 binding 为 None。
    label 是给人看的动作短语（例如"读取文件"），工具没声明时聊天退回显示工具名。"""

    name: str
    owner: str | None
    icon: Mapping[str, object] | None = None
    label: str | None = None


TOOL_DISPLAY = ServiceKey[Callable[[str], ToolDisplay]]("tools.display.v2")


class DecodedCall(Protocol):
    @property
    def binding_id(self) -> str | None: ...
    @property
    def arguments(self) -> Mapping[str, object]: ...
    @property
    def rejection(self) -> Mapping[str, object] | None: ...
    @property
    def exclusive_batch(self) -> bool: ...


class CommitAfter(Protocol):
    """前驱提交完成后放行；中止时 wait 拒绝后继抢先落盘。"""

    async def wait(self) -> None: ...


# 本进程首次加载 Tools 公共合同的时刻（ADR-0100）；公共合同不随实现热更新。
PROCESS_STARTED_AT = datetime.now(timezone.utc)

# 本进程已进入 invoke、回执尚未落盘的默认消息调用；结算据此不把它们重跑或写成「未执行」。
# 与上面同理放在公共合同：实现模块热更新不能把执行事实清空。
invoked_unrecorded: set[str] = set()


class StartCheck(Protocol):
    def __call__(self, transaction: OwnerTransaction | None, /) -> None:
        """核对执行前提；新 Output 在同一事务内核对，工具启动前可在事务外核对。"""
        ...


class ToolCleanup(Protocol):
    """tools 提供的程序收尾边界：排空本程序的调用，再按注册时声明的 release 释放资源。"""

    def __call__(
        self,
        reader: MessageReader,
        source: str,
        from_seq: int,
        *,
        task: Task,
        drain: Callable[[tuple[CallRef, ...]], Awaitable[None]],
    ) -> AbstractAsyncContextManager[None]: ...


class BindSavedTool(Protocol):
    async def __call__(
        self,
        bindings: Bindings,
        binding_id: str,
        *,
        configuration: Mapping[str, object],
    ) -> str: ...


TOOL_CLEANUP = ServiceKey[ToolCleanup]("tools.cleanup.v1")
TOOL_BIND_SAVED = ServiceKey[BindSavedTool]("tools.bind-saved.v1")


class ToolPresentation(Protocol):
    """定义一次程序固定的 schema、wire 解码和系统提示词。"""

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...

    @property
    def system_prompt(self) -> str: ...

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]] | str: ...

    def configuration(self, name: str) -> Mapping[str, object] | None: ...


TOOL_LOADING_PRESENTATION = ServiceKey[
    Callable[[ToolView], tuple[ToolView, ToolPresentation]]
]("tools.loading.presentation.v1")


class ToolMenu(Protocol):
    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def names(self) -> frozenset[str]: ...
    @property
    def system_prompt(self) -> str: ...
    def name(self, binding_id: str) -> str: ...
    def parallel(self, binding_id: str) -> bool: ...
    def decode(self, call: ModelToolCall) -> DecodedCall: ...
    def check_call(self, call: ToolCall) -> None: ...
    async def execute(self, call: CallRef, *, commit_after: CommitAfter | None = None) -> Result: ...
    async def settle_abandoned(self, call: CallRef) -> Result: ...


class OrderedToolProgram(Protocol):
    async def create_menu(
        self,
        reader: MessageReader,
        source: str,
        *,
        content: Mapping[str, Callable[[ContentPart], ContentReferences]],
        check_start: StartCheck,
        authorize: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]
        ],
        view: ToolView | None = None,
        fixed_bindings: Mapping[str, str] | None = None,
        limit: int | None = None,
        presentation: ToolPresentation | None = None,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> ToolMenu: ...


TOOL_PROGRAM_V2 = ServiceKey[OrderedToolProgram]("tools.program.v2")


def normalize_tool_parameters(
    parameters: Mapping[str, Any],
    *,
    open_object: bool = False,
) -> dict[str, Any]:
    """Normalize nested object schemas while preserving explicit JSON Schema policy."""

    schema = cast(dict[str, Any], deepcopy(dict(parameters)))

    def visit(node: dict[str, Any], *, root: bool = False) -> None:
        if node.get("type") == "object" or isinstance(node.get("properties"), dict):
            if root and "additionalProperties" not in node:
                node["additionalProperties"] = open_object
            properties = node.get("properties")
            if isinstance(properties, dict):
                for child in properties.values():
                    if isinstance(child, dict):
                        visit(cast(dict[str, Any], child))
            additional = node.get("additionalProperties")
            if isinstance(additional, dict):
                visit(cast(dict[str, Any], additional))
        elif isinstance(node.get("items"), dict):
            visit(cast(dict[str, Any], node["items"]))

    visit(schema, root=True)
    return schema


_TOOL_TYPE_MAP: dict[str, type[object] | tuple[type[object], ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True, slots=True)
class NestedToolRef:
    """在当前活跃工具调用作用域内可由受信任编排工具（如 codemode）调用的获授工具描述。"""

    name: str
    owner: str
    description: Mapping[str, object]
    binding_id: str


@runtime_checkable
class NestedToolScope(Protocol):
    """当前工具调用期间获授的嵌套调用能力，严格限定于当前菜单已绑定的工具集合。"""

    @property
    def tools(self) -> Mapping[str, NestedToolRef]: ...

    async def call(self, name: str, arguments: Mapping[str, object]) -> Result: ...


_ACTIVE_NESTED_SCOPE: ContextVar[NestedToolScope | None] = ContextVar(
    "akashic_tools_nested_scope", default=None,
)


def current_nested_tool_scope() -> NestedToolScope | None:
    return _ACTIVE_NESTED_SCOPE.get()


def push_nested_tool_scope(scope: NestedToolScope) -> Token[NestedToolScope | None]:
    return _ACTIVE_NESTED_SCOPE.set(scope)


def reset_nested_tool_scope(token: Token[NestedToolScope | None]) -> None:
    _ACTIVE_NESTED_SCOPE.reset(token)


def validate_tool_parameters(
    params: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate one tool call and return user-facing parameter errors."""

    active_schema = dict(schema) if schema is not None else {}
    if active_schema.get("type", "object") != "object":
        raise ValueError(
            f"Schema 顶层类型必须为 object，当前为 {active_schema.get('type')!r}"
        )
    return _validate_tool_value(
        dict(params), {**active_schema, "type": "object"}, ""
    )


def _validate_tool_value(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
) -> list[str]:
    label = path or "参数"
    schema_type = schema.get("type")

    if schema_type in _TOOL_TYPE_MAP:
        valid_type = isinstance(value, _TOOL_TYPE_MAP[schema_type])
        if schema_type in ("integer", "number") and isinstance(value, bool):
            valid_type = False
        if not valid_type:
            return [f"{label} 应为 {schema_type} 类型"]

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label} 须为以下值之一：{schema['enum']}")

    if schema_type in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{label} 须 >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{label} 须 <= {schema['maximum']}")

    if schema_type == "string":
        string_value = cast(str, value)
        if "minLength" in schema and len(string_value) < schema["minLength"]:
            errors.append(f"{label} 最短 {schema['minLength']} 个字符")
        if "maxLength" in schema and len(string_value) > schema["maxLength"]:
            errors.append(f"{label} 最长 {schema['maxLength']} 个字符")

    if schema_type == "object":
        object_value = cast(dict[str, Any], value)
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in object_value:
                errors.append(f"缺少必填字段：{path + '.' + key if path else key}")
        for key, child_value in object_value.items():
            if key in properties:
                errors.extend(
                    _validate_tool_value(
                        child_value,
                        cast(Mapping[str, Any], properties[key]),
                        f"{path}.{key}" if path else key,
                    )
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"不允许额外字段：{path + '.' + key if path else key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(
                    _validate_tool_value(
                        child_value,
                        cast(Mapping[str, Any], schema["additionalProperties"]),
                        f"{path}.{key}" if path else key,
                    )
                )

    if schema_type == "array" and "items" in schema:
        array_value = cast(list[Any], value)
        for index, item in enumerate(array_value):
            errors.extend(
                _validate_tool_value(
                    item,
                    cast(Mapping[str, Any], schema["items"]),
                    f"{path}[{index}]" if path else f"[{index}]",
                )
            )

    return errors


_LUCIDE_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SVG_NS = "http://www.w3.org/2000/svg"
_ALLOWED_SVG_TAGS = frozenset({"path", "circle", "rect", "line", "polyline", "polygon", "ellipse"})
_ALLOWED_SVG_ATTRS = frozenset(
    {
        "d",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "points",
        "fill",
        "stroke",
        "fillRule",
        "clipRule",
        "strokeLinecap",
        "strokeLinejoin",
        "strokeWidth",
    }
)
_SVG_ATTR_ALIASES = {
    "fill-rule": "fillRule",
    "clip-rule": "clipRule",
    "stroke-linecap": "strokeLinecap",
    "stroke-linejoin": "strokeLinejoin",
    "stroke-width": "strokeWidth",
}
_SVG_MAX_BYTES = 8192
_SVG_MAX_NODES = 32
_SVG_MAX_DEPTH = 8


def _freeze_icon_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_icon_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_icon_value(item) for item in value)
    return value


def _svg_local_tag(tag: str) -> str:
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        if ns != _SVG_NS:
            raise ValueError(f"工具图标不支持的 SVG 命名空间: {ns}")
        return name
    return tag


def _validate_svg_view_box(raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError("工具图标 SVG viewBox 必须是字符串")
    parts = raw.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("工具图标 SVG viewBox 必须包含 4 个数值")
    try:
        nums = [float(item) for item in parts]
    except ValueError as error:
        raise ValueError("工具图标 SVG viewBox 数值无效") from error
    if any(not math.isfinite(val) for val in nums) or nums[2] <= 0 or nums[3] <= 0:
        raise ValueError("工具图标 SVG viewBox 宽高必须大于 0")
    return " ".join(parts)


def _normalize_svg_paint(attr: str, value: str) -> str:
    cleaned = value.strip()
    lower = cleaned.lower()
    if lower == "none":
        return "none"
    if lower == "currentcolor":
        return "currentColor"
    raise ValueError(f"工具图标 {attr} 只允许 none 或 currentColor，禁止写死颜色: {value}")


def _normalize_svg_attrs(raw_attrs: Mapping[object, object]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in raw_attrs.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("工具图标 SVG 属性键值必须是字符串")
        if key.startswith("{"):
            raise ValueError(f"工具图标 SVG 不允许带命名空间的属性: {key}")
        canonical = _SVG_ATTR_ALIASES.get(key, key)
        if canonical not in _ALLOWED_SVG_ATTRS:
            raise ValueError(f"工具图标 SVG 不允许的属性: {key}")
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 2048:
            raise ValueError(f"工具图标 SVG 属性值无效: {key}")
        if canonical in {"fill", "stroke"}:
            cleaned = _normalize_svg_paint(canonical, cleaned)
        elif canonical == "strokeWidth":
            try:
                width = float(cleaned)
            except ValueError as error:
                raise ValueError(f"工具图标 SVG strokeWidth 数值无效: {cleaned}") from error
            if not math.isfinite(width) or not (0.25 <= width <= 4.0):
                raise ValueError(f"工具图标 SVG strokeWidth 超出范围 (0.25~4): {cleaned}")
        attrs[canonical] = cleaned
    return attrs


def _collect_svg_nodes(
    element: ET.Element,
    inherited: Mapping[str, str],
    nodes: list[dict[str, object]],
    *,
    depth: int = 0,
) -> None:
    if depth > _SVG_MAX_DEPTH:
        raise ValueError(f"工具图标 SVG 嵌套深度超过上限 ({_SVG_MAX_DEPTH})")
    for child in element:
        tag = _svg_local_tag(child.tag)
        if tag == "g":
            group_attrs = {**inherited, **_normalize_svg_attrs(cast(Mapping[object, object], child.attrib))}
            _collect_svg_nodes(child, group_attrs, nodes, depth=depth + 1)
            continue
        if tag not in _ALLOWED_SVG_TAGS:
            raise ValueError(f"工具图标 SVG 不允许的元素: <{tag}>")
        if len(child):
            raise ValueError(f"工具图标 SVG <{tag}> 不允许嵌套子元素")
        if len(nodes) >= _SVG_MAX_NODES:
            raise ValueError(f"工具图标 SVG 节点数超过上限 ({_SVG_MAX_NODES})")
        merged = {**inherited, **_normalize_svg_attrs(cast(Mapping[object, object], child.attrib))}
        nodes.append({"tag": tag, "attrs": merged})


def _parse_svg_string(raw: str) -> Mapping[str, object]:
    text = raw.strip()
    if len(text.encode("utf-8")) > _SVG_MAX_BYTES:
        raise ValueError("工具图标 SVG 超过大小上限")
    if "<!" in text or "<?" in text:
        raise ValueError("工具图标 SVG 不允许包含声明或处理指令")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ValueError("工具图标 SVG 解析失败") from error
    if _svg_local_tag(root.tag) != "svg":
        raise ValueError("工具图标 SVG 根元素必须是 <svg>")
    view_box = _validate_svg_view_box(root.attrib.get("viewBox", "0 0 24 24"))
    raw_inherited = {
        key: value
        for key, value in root.attrib.items()
        if _SVG_ATTR_ALIASES.get(key, key) in _ALLOWED_SVG_ATTRS
    }
    inherited = _normalize_svg_attrs(cast(Mapping[object, object], raw_inherited))
    nodes: list[dict[str, object]] = []
    _collect_svg_nodes(root, inherited, nodes)
    if not nodes:
        raise ValueError("工具图标 SVG 至少需要一个几何绘制节点")
    return cast(
        Mapping[str, object],
        _freeze_icon_value({"kind": "svg", "viewBox": view_box, "nodes": nodes}),
    )


def normalize_tool_icon(icon: str | Mapping[str, object] | None) -> Mapping[str, object] | None:
    """校验并冻结工具图标声明；接受 Lucide 短名、受限 SVG 字符串或规范化字典。"""
    if icon is None:
        return None
    if isinstance(icon, str):
        trimmed = icon.strip()
        if not trimmed:
            raise ValueError("工具图标不能为空字符串")
        if trimmed.startswith("<"):
            return _parse_svg_string(trimmed)
        if len(trimmed) > 64 or _LUCIDE_NAME_RE.fullmatch(trimmed) is None:
            raise ValueError(f"工具图标名称无效: {icon!r}")
        return cast(Mapping[str, object], _freeze_icon_value({"kind": "lucide", "name": trimmed}))
    if isinstance(icon, Mapping):
        kind = icon.get("kind")
        if kind == "lucide":
            if set(icon) != {"kind", "name"}:
                raise ValueError("lucide 工具图标字段无效")
            name = icon.get("name")
            if not isinstance(name, str) or len(name) > 64 or _LUCIDE_NAME_RE.fullmatch(name) is None:
                raise ValueError(f"工具图标名称无效: {name!r}")
            return cast(Mapping[str, object], _freeze_icon_value({"kind": "lucide", "name": name}))
        if kind == "svg":
            if set(icon) != {"kind", "viewBox", "nodes"}:
                raise ValueError("svg 工具图标字段无效")
            view_box = _validate_svg_view_box(icon.get("viewBox"))
            raw_nodes = icon.get("nodes")
            if not isinstance(raw_nodes, (list, tuple)) or not (1 <= len(raw_nodes) <= _SVG_MAX_NODES):
                raise ValueError("工具图标 SVG nodes 数量无效")
            nodes: list[dict[str, object]] = []
            for item in raw_nodes:
                if not isinstance(item, Mapping) or set(item) != {"tag", "attrs"}:
                    raise ValueError("工具图标 SVG node 结构无效")
                tag = item.get("tag")
                attrs = item.get("attrs")
                if not isinstance(tag, str) or tag not in _ALLOWED_SVG_TAGS:
                    raise ValueError(f"工具图标 SVG 不允许的节点标签: {tag!r}")
                if not isinstance(attrs, Mapping):
                    raise ValueError("工具图标 SVG node.attrs 必须是映射")
                nodes.append({"tag": tag, "attrs": _normalize_svg_attrs(cast(Mapping[object, object], attrs))})
            encoded_size = len(json.dumps({"viewBox": view_box, "nodes": nodes}, ensure_ascii=False).encode("utf-8"))
            if encoded_size > _SVG_MAX_BYTES:
                raise ValueError("工具图标 SVG 超过大小上限")
            return cast(
                Mapping[str, object],
                _freeze_icon_value({"kind": "svg", "viewBox": view_box, "nodes": nodes}),
            )
        raise ValueError(f"未知工具图标类型: {kind!r}")
    raise TypeError("工具图标必须是字符串、映射或 None")
