from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any, Literal, TypeAlias, cast

from agent.plugin_composition.context import Context, FiberHandle, HealthHandle
from agent.plugin_composition.model import CompositionError, FiberState, ServiceKey

ToolRisk: TypeAlias = Literal["read-only", "read-write", "external-side-effect"]
PluginToolHandler: TypeAlias = Callable[[Any, Mapping[str, object]], Awaitable[object]]

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_EXPORT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:]*$")
_RISKS = frozenset({"read-only", "read-write", "external-side-effect"})
_JSON_TYPES = frozenset(
    {"object", "array", "string", "integer", "number", "boolean", "null"}
)


@dataclass(frozen=True, slots=True)
class PluginToolDefinition:
    """Declare one exact-generation Tool without retaining a handler callable."""

    name: str
    description: str
    parameters: Mapping[str, object]
    handler_export: str
    risk: ToolRisk = "read-write"
    always_on: bool = False
    preloadable: bool = True
    requires_turn_search: bool = False
    search_hint: str | None = None

    def __post_init__(self) -> None:
        _validate_text_fields(self)
        schema = _normalize_schema(self.parameters)
        object.__setattr__(self, "parameters", schema)


@dataclass(frozen=True, slots=True)
class PluginToolDescriptor:
    """Freeze Tool discovery metadata used by snapshot identity."""

    owner: str
    name: str
    description: str
    parameters: Mapping[str, object]
    handler_export: str
    risk: ToolRisk
    always_on: bool
    preloadable: bool
    requires_turn_search: bool
    search_hint: str | None


@dataclass(frozen=True, slots=True)
class PluginToolBinding:
    """Bind one Tool's exact Root-local handler to a live Fiber."""

    generation_id: str
    plugin_id: str
    descriptor: PluginToolDescriptor
    definition: PluginToolDefinition
    module: ModuleType | None
    handler: PluginToolHandler | None
    owner_fiber: FiberHandle
    activation_token: object
    required_health: HealthHandle

    def is_live(self) -> bool:
        return (
            self.owner_fiber.state is FiberState.ACTIVE
            and self.owner_fiber.activation_token is self.activation_token
            and self.required_health.healthy
        )


class PluginToolCatalog(Mapping[str, PluginToolBinding]):
    """Expose one immutable Root-local plugin Tool catalog."""

    def __init__(
        self,
        bindings: Mapping[str, PluginToolBinding],
        *,
        root_instance_token: object,
    ) -> None:
        self._root_instance_token = root_instance_token
        self._bindings = MappingProxyType(dict(sorted(bindings.items())))
        self._descriptors = tuple(
            binding.descriptor for binding in self._bindings.values()
        )
        self._identity = _digest(self._descriptors)

    @property
    def root_instance_token(self) -> object:
        return self._root_instance_token

    @property
    def descriptors(self) -> tuple[PluginToolDescriptor, ...]:
        return self._descriptors

    @property
    def identity(self) -> str:
        return self._identity

    def __getitem__(self, key: str) -> PluginToolBinding:
        return self._bindings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)


TOOL_CATALOG = ServiceKey["PluginTools"]("core.tool_catalog")


@dataclass(slots=True)
class _Registration:
    token: int
    plugin_id: str
    generation_id: str
    definition: PluginToolDefinition
    descriptor: PluginToolDescriptor
    module: ModuleType | None
    handler: PluginToolHandler | None
    owner_fiber: FiberHandle
    activation_token: object
    required_health: HealthHandle | None = None


class _ToolDeclarations:
    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _Registration] = {}
        self._names: dict[str, int] = {}
        self._frozen: PluginToolCatalog | None = None

    async def register(
        self,
        ctx: Context,
        definition: PluginToolDefinition,
        handler: PluginToolHandler | None = None,
    ) -> None:
        normalized = _normalize_definition(definition)
        module = ctx._plugin_module()  # pyright: ignore[reportPrivateUsage]
        owner_fiber = ctx.fiber
        activation_token = owner_fiber.activation_token
        if activation_token is None:
            raise CompositionError("INACTIVE_FIBER", "当前 Fiber 没有 active activation")
        registration: _Registration | None = None

        def setup() -> object:
            nonlocal registration
            registration, cleanup = self._register(
                ctx.runtime.plugin_id,
                ctx.generation_id,
                normalized,
                module,
                handler,
                owner_fiber,
                activation_token,
            )
            return cleanup

        effect = await ctx.effect(setup, label=f"plugin-tool:{normalized.name}")
        try:
            health = await ctx.health(
                f"plugin-tool:{normalized.name}",
                required=True,
            )
        except BaseException:
            await effect.aclose()
            raise
        assert registration is not None
        registration.required_health = health

    def _register(
        self,
        plugin_id: str,
        generation_id: str,
        definition: PluginToolDefinition,
        module: ModuleType | None,
        handler: PluginToolHandler | None,
        owner_fiber: FiberHandle,
        activation_token: object,
    ) -> tuple[_Registration, object]:
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_TOOLS_FROZEN",
                "插件 Tool 声明已冻结，不能新增",
            )
        if definition.name in self._names:
            raise CompositionError(
                "DUPLICATE_PLUGIN_TOOL",
                f"插件 Tool 名称重复: {definition.name}",
            )
        token = self._next_token
        self._next_token += 1
        descriptor = _descriptor(plugin_id, definition)
        registration = _Registration(
            token=token,
            plugin_id=plugin_id,
            generation_id=generation_id,
            definition=definition,
            descriptor=descriptor,
            module=module,
            handler=handler,
            owner_fiber=owner_fiber,
            activation_token=activation_token,
        )
        self._registrations[token] = registration
        self._names[definition.name] = token

        def cleanup() -> None:
            self._registrations.pop(token, None)
            if self._names.get(definition.name) == token:
                self._names.pop(definition.name, None)

        return registration, cleanup

    def freeze(
        self,
        root_instance_token: object,
        generation_ids: Mapping[str, str],
    ) -> PluginToolCatalog:
        if self._frozen is not None:
            if self._frozen.root_instance_token is not root_instance_token:
                raise RuntimeError("plugin Tool catalog 属于另一棵 Root")
            if any(
                generation_ids.get(binding.plugin_id) != binding.generation_id
                for binding in self._frozen.values()
            ):
                raise RuntimeError("plugin Tool catalog generation identity 已冻结")
            return self._frozen
        bindings: dict[str, PluginToolBinding] = {}
        for registration in sorted(
            self._registrations.values(), key=lambda item: item.token
        ):
            health = registration.required_health
            if health is None:
                raise RuntimeError("plugin Tool 缺少 required Health")
            generation_id = generation_ids.get(registration.plugin_id)
            if generation_id is None:
                raise RuntimeError(
                    f"plugin Tool owner 不属于 generations: {registration.plugin_id}"
                )
            bindings[registration.definition.name] = PluginToolBinding(
                generation_id=generation_id,
                plugin_id=registration.plugin_id,
                descriptor=registration.descriptor,
                definition=registration.definition,
                module=registration.module,
                handler=registration.handler,
                owner_fiber=registration.owner_fiber,
                activation_token=registration.activation_token,
                required_health=health,
            )
        self._frozen = PluginToolCatalog(
            bindings,
            root_instance_token=root_instance_token,
        )
        return self._frozen


class PluginTools:
    """Expose Root-token-bound Tool declarations to plugins."""

    def __init__(self, root_instance_token: object) -> None:
        self._root_instance_token = root_instance_token
        self._declarations = _ToolDeclarations()

    async def register(
        self,
        ctx: Context,
        definition: PluginToolDefinition,
        handler: PluginToolHandler | None = None,
    ) -> None:
        """Bind Root-local handlers; omission preserves stateless legacy exports."""

        if (
            ctx._root_instance_token() is not self._root_instance_token
            or ctx.require(TOOL_CATALOG) is not self
        ):
            raise CompositionError(
                "PLUGIN_TOOLS_SERVICE_ROOT_MISMATCH",
                "插件 Tool Service 不属于当前 Root",
            )
        await self._declarations.register(ctx, definition, handler)


def _freeze_plugin_tools(
    value: object,
    root_instance_token: object,
    generation_ids: Mapping[str, str],
) -> PluginToolCatalog:
    """Freeze the exact Core-created Tool declaration facade."""

    if not isinstance(value, PluginTools):
        raise RuntimeError("RuntimeSnapshot plugin Tool Service 类型无效")
    if value._root_instance_token is not root_instance_token:
        raise RuntimeError("RuntimeSnapshot plugin Tool Service 不属于 exact Root")
    return value._declarations.freeze(root_instance_token, generation_ids)


def _normalize_definition(definition: PluginToolDefinition) -> PluginToolDefinition:
    if not isinstance(definition, PluginToolDefinition):
        raise TypeError("PluginTools.register 只接受 PluginToolDefinition")
    return PluginToolDefinition(
        name=definition.name,
        description=definition.description,
        parameters=cast(Mapping[str, object], _thaw_json(definition.parameters)),
        handler_export=definition.handler_export,
        risk=definition.risk,
        always_on=definition.always_on,
        preloadable=definition.preloadable,
        requires_turn_search=definition.requires_turn_search,
        search_hint=definition.search_hint,
    )


def _descriptor(owner: str, definition: PluginToolDefinition) -> PluginToolDescriptor:
    return PluginToolDescriptor(
        owner=owner,
        name=definition.name,
        description=definition.description,
        parameters=definition.parameters,
        handler_export=definition.handler_export,
        risk=definition.risk,
        always_on=definition.always_on,
        preloadable=definition.preloadable,
        requires_turn_search=definition.requires_turn_search,
        search_hint=definition.search_hint,
    )


def _validate_text_fields(definition: PluginToolDefinition) -> None:
    if not isinstance(definition.name, str) or _NAME.fullmatch(definition.name) is None:
        raise ValueError(f"Plugin Tool name 无效: {definition.name}")
    if not isinstance(definition.description, str) or not definition.description.strip():
        raise ValueError(f"Plugin Tool description 不能为空: {definition.name}")
    if definition.description.strip() != definition.description:
        raise ValueError(f"Plugin Tool description 不能有首尾空白: {definition.name}")
    if not isinstance(definition.handler_export, str) or (
        _EXPORT.fullmatch(definition.handler_export) is None
        or ".." in definition.handler_export
        or definition.handler_export.endswith((".", ":"))
    ):
        raise ValueError(f"Plugin Tool handler_export 无效: {definition.handler_export}")
    if definition.risk not in _RISKS:
        raise ValueError(f"Plugin Tool risk 无效: {definition.risk}")
    for field_name in ("always_on", "preloadable", "requires_turn_search"):
        if not isinstance(getattr(definition, field_name), bool):
            raise TypeError(f"Plugin Tool {field_name} 必须是 bool")
    if definition.search_hint is not None and (
        not isinstance(definition.search_hint, str)
        or not definition.search_hint.strip()
        or definition.search_hint.strip() != definition.search_hint
    ):
        raise ValueError("Plugin Tool search_hint 必须是无首尾空白的非空字符串")


def _normalize_schema(value: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the supported strict JSON Schema subset and freeze a detached copy."""

    if not isinstance(value, Mapping):
        raise TypeError("Plugin Tool parameters 必须是 Mapping")
    _validate_json_object_keys(value, "parameters")
    try:
        copied = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Plugin Tool parameters 必须是 JSON value") from error
    if not isinstance(copied, dict):
        raise TypeError("Plugin Tool parameters 根必须是 object schema")
    _validate_schema_node(copied, root=True)
    return cast(Mapping[str, object], _freeze_json(copied))


def _validate_json_object_keys(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Plugin Tool {path} 的 object key 必须是字符串")
            _validate_json_object_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_object_keys(item, f"{path}[{index}]")


def _validate_schema_node(node: dict[str, object], *, root: bool = False) -> None:
    schema_type = node.get("type")
    if schema_type not in _JSON_TYPES:
        raise ValueError("Plugin Tool schema type 无效")
    if root and schema_type != "object":
        raise ValueError("Plugin Tool parameters 根 schema type 必须是 object")
    if schema_type == "object":
        properties = node.get("properties")
        required = node.get("required")
        if not isinstance(properties, dict):
            raise ValueError("Plugin Tool object schema 必须声明 properties")
        if not isinstance(required, list) or any(
            not isinstance(item, str) for item in required
        ):
            raise ValueError("Plugin Tool object schema 必须声明字符串 required")
        if len(set(required)) != len(required) or not set(required).issubset(properties):
            raise ValueError("Plugin Tool required 必须唯一且属于 properties")
        if node.get("additionalProperties") is not False:
            raise ValueError("Plugin Tool object schema 必须拒绝 additionalProperties")
        for name, child in properties.items():
            if not isinstance(name, str) or not isinstance(child, dict):
                raise ValueError("Plugin Tool properties 必须是字符串到 schema 的映射")
            _validate_schema_node(cast(dict[str, object], child))
    if schema_type == "array":
        items = node.get("items")
        if not isinstance(items, dict):
            raise ValueError("Plugin Tool array schema 必须声明 items")
        _validate_schema_node(cast(dict[str, object], items))


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _digest(descriptors: tuple[PluginToolDescriptor, ...]) -> str:
    payload = [
        {
            "owner": item.owner,
            "name": item.name,
            "description": item.description,
            "parameters": _thaw_json(item.parameters),
            "handler_export": item.handler_export,
            "risk": item.risk,
            "always_on": item.always_on,
            "preloadable": item.preloadable,
            "requires_turn_search": item.requires_turn_search,
            "search_hint": item.search_hint,
        }
        for item in descriptors
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "TOOL_CATALOG",
    "PluginToolBinding",
    "PluginToolCatalog",
    "PluginToolDefinition",
    "PluginToolDescriptor",
    "PluginToolHandler",
    "PluginTools",
    "ToolRisk",
]
