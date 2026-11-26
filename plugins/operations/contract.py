"""Self-CLI Operation Bus 公共契约。

业务插件通过可选子 Fiber 注入 `OPERATIONS_V1` 注册主操作或正交扩展选项，
彼此零 import；强父子子树也可借 `cli_node_key(path)` 复复用 CompositionRoot
的拓扑激活序与热卸载级联清理。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
from plugins.ledger.contract import ContentPart

ActorKind = Literal["agent", "user"]
AudienceKind = Literal["agent_and_user", "user_only"]
OptionValueType = Literal["string", "bool", "int"]


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """单个 CLI 参数声明（主操作裸名或正交扩展命名空间参数）。"""

    name: str
    description: str
    value_type: OptionValueType = "string"
    required: bool = False
    choices: tuple[str, ...] = ()
    default: str | bool | int | None = None
    set_once: bool = False


@dataclass(frozen=True, slots=True)
class OperationRequest:
    """规范化后的单次 Operation 调用上下文。"""

    operation_id: str
    actor: ActorKind
    session_id: str | None
    request_id: str | None
    args: Mapping[str, object]
    extensions: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Operation 执行结果：机器可读 data + 给模型/终端的简短文本与可选结构化片段。"""

    ok: bool
    summary: str
    data: Mapping[str, object] = field(default_factory=dict)
    parts: tuple[ContentPart, ...] = ()
    error_code: str | None = None


OperationHandler = Callable[[OperationRequest], Awaitable[OperationResult]]
ExtensionValidator = Callable[
    [OperationRequest, Mapping[str, object]], Awaitable[None]
]
ExtensionApplier = Callable[
    [OperationRequest, OperationResult, Mapping[str, object]],
    Awaitable[Mapping[str, object] | None],
]


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """主操作注册声明；一个 operation 只有一个 owner。"""

    operation_id: str
    path: tuple[str, ...]
    summary: str
    handler: OperationHandler
    options: tuple[OptionSpec, ...] = ()
    extension_points: tuple[str, ...] = ()
    audience: AudienceKind = "agent_and_user"


@dataclass(frozen=True, slots=True)
class ExtensionSpec:
    """正交扩展选项声明；按 target_operation 或 extension_point 挂载。"""

    namespace: str
    options: tuple[OptionSpec, ...]
    validate: ExtensionValidator
    apply: ExtensionApplier
    target_operation: str | None = None
    extension_point: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class CliNodeHandle:
    """主操作在 CompositionRoot 上发布的同构子树节点句柄。"""

    operation_id: str
    path: tuple[str, ...]


def cli_node_key(path: str) -> ServiceKey[CliNodeHandle]:
    """按点分路径返回同构子树 ServiceKey（如 `cli.node:project.create`）。"""
    return ServiceKey[CliNodeHandle](f"cli.node:{path}")


class Operations(Protocol):
    """Self-CLI 统一注册、内省与执行总线。"""

    async def register(self, ctx: Context, spec: OperationSpec) -> Effect: ...

    async def extend(self, ctx: Context, spec: ExtensionSpec) -> Effect: ...

    def describe(
        self, path: tuple[str, ...] = (), *, actor: ActorKind = "agent"
    ) -> dict[str, object]: ...

    def render_help(
        self, path: tuple[str, ...] = (), *, actor: ActorKind = "agent"
    ) -> str: ...

    async def invoke(
        self,
        argv: list[str],
        *,
        actor: ActorKind,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> OperationResult: ...

    def receipt(self, request_id: str) -> OperationResult | None: ...


OPERATIONS_V1 = ServiceKey[Operations]("operations.v1")
