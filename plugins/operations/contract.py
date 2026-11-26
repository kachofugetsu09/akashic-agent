"""Self-CLI Operation Bus 公共契约。

业务插件通过可选子 Fiber 注入 `OPERATIONS_V1` 注册主操作或正交扩展选项，
彼此零 import；强父子子树也可借 `cli_node_key(path)` 复复用 CompositionRoot
的拓扑激活序与热卸载级联清理。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
import inspect
from typing import Literal, Protocol, cast

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
from pydantic import ValidationError

from plugins.ledger.contract import ContentPart, OwnerWrite, RpcMethod

ActorKind = Literal["agent", "user"]
AudienceKind = Literal["agent_and_user", "user_only"]
OptionValueType = Literal["string", "bool", "int", "json"]


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
    # 主操作确定目标身份后调用，收集各扩展在同一联合事务里的写入；没有参与者返回空元组。
    stage: Callable[[Mapping[str, object]], Awaitable[tuple[OwnerWrite, ...]]] | None = None

    async def staged(self, target: Mapping[str, object]) -> tuple[OwnerWrite, ...]:
        return () if self.stage is None else await self.stage(target)


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
# 主操作给出目标身份（如 {"dimension": "project", "value": "p_x"}）后，扩展交出本 owner 的联合写入。
ExtensionStager = Callable[
    [OperationRequest, Mapping[str, object], Mapping[str, object]],
    Awaitable[OwnerWrite | None],
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
    # 本操作对应的用户界面入口 ID（与 scripts/check_interaction_parity.py 采集的入口同名），供一对一门禁核对。
    mirrors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtensionSpec:
    """正交扩展选项声明；按 target_operation 或 extension_point 挂载。"""

    namespace: str
    options: tuple[OptionSpec, ...]
    validate: ExtensionValidator | None = None
    apply: ExtensionApplier | None = None
    stage: ExtensionStager | None = None
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


class QueryHandler(Protocol):
    def __call__(
        self, method: str, payload: dict[str, object], *,
        session_id: str | None, turn_id: str | None,
    ) -> object | Awaitable[object]: ...


@dataclass(frozen=True, slots=True)
class QueryMirror:
    """把插件已有的一个界面查询方法原样发布为 operation：界面与 self 调用同一个领域入口。"""

    method: str
    path: tuple[str, ...]
    summary: str
    options: tuple[OptionSpec, ...] = ()
    audience: AudienceKind = "agent_and_user"
    entry: str = ""


async def register_query_mirrors(
    ctx: Context, operations: Operations, *, query: QueryHandler,
    mirrors: Sequence[QueryMirror],
) -> None:
    """逐个注册镜像 operation；参数名即界面载荷键，入口 ID 默认为 rpc:<插件 ID>/<method>。"""
    # 运行期身份带来源后缀（models@release），operation id 只用插件名，与门禁采集一致。
    plugin = ctx.runtime.plugin_id.split("@", 1)[0]
    for mirror in mirrors:
        await operations.register(ctx, OperationSpec(
            operation_id=f"{plugin}.{mirror.method}",
            path=mirror.path,
            summary=mirror.summary,
            handler=_query_handler(query, mirror),
            options=mirror.options,
            audience=mirror.audience,
            mirrors=(mirror.entry or f"rpc:{plugin}/{mirror.method}",),
        ))


def _query_handler(query: QueryHandler, mirror: QueryMirror) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        payload = {key: value for key, value in req.args.items() if value is not None}
        try:
            value = query(mirror.method, payload, session_id=req.session_id, turn_id=None)
            if inspect.isawaitable(value):
                value = await value
        except (ValueError, LookupError, PermissionError) as error:
            return OperationResult(False, str(error) or type(error).__name__, error_code="rejected")
        data = value if isinstance(value, Mapping) else {"value": value}
        return OperationResult(True, f"{' '.join(mirror.path)} 完成", data=data)
    return handle


@dataclass(frozen=True, slots=True)
class RpcMirror:
    """把插件已发布的一个 Gateway RPC 方法原样发布为 operation：终端/SDK 与 self 调同一个处理函数。"""

    method: str
    path: tuple[str, ...]
    summary: str
    options: tuple[OptionSpec, ...] = ()
    audience: AudienceKind = "agent_and_user"


async def register_rpc_mirrors(
    ctx: Context, operations: Operations, *, methods: Mapping[str, RpcMethod],
    mirrors: Sequence[RpcMirror],
) -> None:
    """参数名即 RPC 参数模型的字段名；入口 ID 为 gateway:<method>，operation id 为 <插件>.<method>。"""
    plugin = ctx.runtime.plugin_id.split("@", 1)[0]
    for mirror in mirrors:
        await operations.register(ctx, OperationSpec(
            operation_id=f"{plugin}.{mirror.method}", path=mirror.path, summary=mirror.summary,
            handler=_rpc_handler(methods[mirror.method], mirror), options=mirror.options,
            audience=mirror.audience, mirrors=(f"gateway:{mirror.method}",),
        ))


def _rpc_handler(method: RpcMethod, mirror: RpcMirror) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        try:
            params = method.params.model_validate({key: value for key, value in req.args.items() if value is not None})
        except ValidationError as error:
            # 只报字段与原因，不回显输入值。
            reasons = "; ".join(f"{'.'.join(str(p) for p in item['loc'])}: {item['msg']}" for item in error.errors())
            return OperationResult(False, f"参数无效：{reasons}", error_code="invalid_arguments")
        value = await method.invoke(params, None)
        data = value if isinstance(value, Mapping) else {"value": value}
        return OperationResult(True, f"{' '.join(mirror.path)} 完成", data=cast(Mapping[str, object], data))
    return handle
