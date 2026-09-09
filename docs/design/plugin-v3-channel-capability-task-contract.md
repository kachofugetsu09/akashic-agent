# 插件 v3 inbound/outbound channel capability 任务合同

> 2026-09-09：本文历史执行状态 unknown/UNKNOWN/uncertain 已被 [0063](../decisions/0063-execution-failures-have-terminal-results.md) 的明确失败终态和收尾规则取代；其他合同不变。


- 状态：proposed / independent review findings incorporated
- 日期：2026-08-16
- 实现起点：`19f2cca2`（只有 command catalog，C14 尚未实现）
- 清单：C14
- 首个 consumers：Feishu、QQBot

## 1. 目标与边界

Core 提供 `CHANNELS = ServiceKey("core.channels")`。v3 插件在 `apply(ctx, config)` 中登记不可变
`ChannelDefinition` blueprint；candidate Root 只冻结 descriptor，不构造 SDK client、不绑定 EventBus/
MessagePush、不占 endpoint、不收发。正式 publication 使用 C11 的 closed provisional target：旧 stable 仍是唯一
公开 `current`，新 channel 在关闭 admission 的内部 transaction 中 materialize/start，只有 stable pointer、Host binding
和新 admission 全部切换成功后才公开。

```text
candidate/formal Fiber ── Effect register blueprint ──► ChannelRegistry
                                                        │ freeze
                     public current = old stable         │
                     old/new admission = closed          ▼
closed provisional commit ──► old stop ──► service switch
        ──► formal factory/start ──► internal Host binding
                                      │ Manager finalize once
                                      ▼
                     public current = new stable ──► open admission
```

`semantic_delta: controlled`。provider parsing 与 stream presentation 保持；本任务新增稳定
`delivery_id` 与 `DELIVERED | REJECTED | UNKNOWN` 结果，禁止 provider effect 已可能发生后由 Core 盲重试。
这不能承诺远端 exactly-once：不支持 idempotency key 的 provider 返回 `UNKNOWN` 时宁可暴露未决状态，也不重复发送。

本能力分四段落地：C14a 只建立静态 blueprint/registry 与 candidate secret redaction；C14b 建立只消费
committed snapshot 的 formal Host 和 text-only `adapter.deliver(ProviderDeliveryRequest)` provider seam；C14c 才把
它接入 exact binding envelope、Bus 三态
receipt 与 drain；C14d 承接 control/turn-stream。附件持久 owner 未单独批准前不进入 C14，首批 Feishu/QQBot v3
只允许文本，附件输入返回确定性 `REJECTED`，不能把现有 uploads 路径伪装成 ref-counted lease。

本任务不发送正式渠道消息、不写 Session/message/plugin-data、不在 candidate 读取正式 secret bytes。验证只使用
recording/loopback provider 与 synthetic credential。

## 2. 公开合同与 secret 边界

```python
# akashic.plugin.toml；在 import/config copy 前由 Core 读取
[channel_credentials]
feishu = ["appId", "appSecret", "app_id", "app_secret"]
qqbot = ["appId", "clientSecret", "app_id", "client_secret"]

ChannelDefinition(
    name="feishu",
    capabilities=frozenset({
        ChannelCapability.INBOUND,
        ChannelCapability.OUTBOUND,
        ChannelCapability.CONTROL,
        ChannelCapability.TURN_STREAM,
    }),
    factory_export="build_feishu_channel",
    inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
    credential_paths=("appId", "appSecret", "app_id", "app_secret"),
)

ChannelDefinition(
    name="qqbot",
    capabilities=frozenset({
        ChannelCapability.INBOUND,
        ChannelCapability.OUTBOUND,
        ChannelCapability.CONTROL,
        ChannelCapability.TURN_STREAM,
    }),
    factory_export="build_qqbot_channel",
    inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
    credential_paths=("appId", "app_id", "clientSecret", "client_secret"),
)
```

插件 Definition 不接受 `owner`；Core 从调用 Fiber 的 `ctx.runtime.plugin_id` 写入 frozen descriptor。第一版其余
公开类型固定为：

```python
JsonValue = None | bool | int | float | str | tuple["JsonValue", ...] \
    | Mapping[str, "JsonValue"]

class ChannelCapability(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    CONTROL = "control"
    TURN_STREAM = "turn_stream"

class InboundIdentity(StrEnum):
    PROVIDER_MESSAGE_ID = "provider_message_id"

class DeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class CredentialRef:
    path: tuple[str, ...]  # opaque; plugin/candidate cannot dereference

@dataclass(frozen=True, slots=True)
class QueuedReceipt:
    delivery_id: str
    queued: bool

@dataclass(frozen=True, slots=True)
class ChannelInboundMessage:
    channel: str
    sender: str
    chat_id: str
    content: str
    timestamp: datetime
    metadata: Mapping[str, JsonValue]

class InboundOwner(StrEnum):
    INGRESS = "ingress"
    BUS = "bus"
    LANE = "lane"
    LOOP = "loop"
    CLOSED = "closed"

class InboundState(StrEnum):
    ADMITTED = "admitted"
    BUS_QUEUED = "bus_queued"
    LANE_QUEUED = "lane_queued"
    RUNNING = "running"
    TERMINAL = "terminal"

class InboundEnvelope:
    message_id: str                         # immutable public projection
    snapshot_id: str
    generation_id: str
    binding_token: str
    message: ChannelInboundMessage
    lease: ChannelBindingLease              # Core-private exact snapshot/binding owner
    state: InboundState                     # Core mutates only through methods below
    owner: InboundOwner

    def handoff(
        self,
        expected_owner: InboundOwner,
        next_owner: InboundOwner,
    ) -> "InboundEnvelope": ...

    async def close(self, expected_owner: InboundOwner) -> None: ...

    # keyword-only constructor:
    # InboundEnvelope(*, message_id, snapshot_id, generation_id,
    #                 binding_token, message, lease,
    #                 state=InboundState.ADMITTED, owner=InboundOwner.INGRESS)

@dataclass(frozen=True, slots=True)
class RawInbound:
    message_id: str
    message: ChannelInboundMessage
    provider_identity: str | None = None
    recipient: str | None = None

class ChannelBindingLease(Protocol):
    snapshot_lease: RuntimeSnapshotLease
    snapshot_id: str
    generation_id: str
    channel_name: str
    binding_token: str
    async def aclose(self) -> None: ...

@dataclass(frozen=True, slots=True)
class ChannelDefinition:
    name: str
    capabilities: frozenset[ChannelCapability]
    factory_export: str
    inbound_identity: InboundIdentity | None
    credential_paths: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ChannelDescriptor:
    owner: str
    name: str
    capabilities: tuple[ChannelCapability, ...]
    factory_export: str
    inbound_identity: InboundIdentity | None
    credential_paths: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ChannelFactoryProvenance:
    plugin_id: str
    generation_id: str
    channel_name: str
    source_revision: str
    config_revision: str
    factory_export: str

@dataclass(frozen=True, slots=True)
class ChannelRegistrySnapshot:
    descriptors: tuple[ChannelDescriptor, ...]
    factories: tuple[ChannelFactoryProvenance, ...]
    identity: str
    root_instance_token: object  # exact Root fence; excluded from identity

class ProviderClient(Protocol):
    def credential(self, ref: CredentialRef) -> str: ...
    async def aclose(self) -> None: ...

class ProviderClientFactory(Protocol):
    async def create(
        self,
        credentials: Mapping[str, CredentialRef],
    ) -> ProviderClient: ...  # Core resolves refs inside this formal-only call
    async def aclose(self) -> None: ...  # Host-owned client/secret lease scope

@dataclass(frozen=True, slots=True)
class ControlReceipt:
    accepted: bool
    reason: Literal["interrupted", "idle", "duplicate", "binding_closed"]
    response: "ChannelDeliveryReceipt | None" = None

@dataclass(frozen=True, slots=True)
class ControlResponseBodies:
    interrupted: str
    idle: str

class ChannelControlPort(Protocol):
    async def interrupt(
        self,
        raw: RawInbound,
        *,
        response_bodies: ControlResponseBodies,
    ) -> ControlReceipt: ...

class ChannelIdentityPort(Protocol):
    def resolve(self, provider_identity: str) -> str | None: ...

class TurnStreamPort(Protocol):
    def subscribe(
        self,
        callback: TurnStreamCallback,
    ) -> StreamSubscription: ...

class TurnStreamEventKind(StrEnum):
    TURN_STARTED = "turn.started"
    STREAM_DELTA = "stream.delta"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TURN_OUTPUT_COMPLETED = "turn.output.completed"

@dataclass(frozen=True, slots=True)
class TurnStartedPresentation:
    turn_id: str
    client_message_id: str  # accepted RawInbound.message_id

@dataclass(frozen=True, slots=True)
class StreamDeltaPresentation:
    turn_id: str
    sequence: int
    text_delta: str
    reasoning_delta: str

@dataclass(frozen=True, slots=True)
class ToolPresentation:
    turn_id: str
    sequence: int
    tool_call_id: str
    tool_name: str

@dataclass(frozen=True, slots=True)
class TurnOutputCompletedPresentation:
    turn_id: str
    sequence: int

@dataclass(frozen=True, slots=True)
class TurnStreamEvent:
    presentation_id: str
    kind: TurnStreamEventKind
    payload: (
        TurnStartedPresentation
        | StreamDeltaPresentation
        | ToolPresentation
        | TurnOutputCompletedPresentation
    )

@dataclass(frozen=True, slots=True)
class PresentationReceipt:
    presentation_id: str
    status: DeliveryStatus  # DELIVERED | REJECTED | UNKNOWN
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

TurnStreamCallback = Callable[
    [TurnStreamEvent],
    Awaitable[PresentationReceipt],
]

class StreamSubscription(Protocol):
    def close_admission(self) -> None: ...
    async def await_quiescence(self) -> None: ...
    async def close(self) -> None: ...

class ChannelIngress(Protocol):
    async def admit(self, raw: RawInbound) -> bool: ...  # accepted or duplicate

class BusOutboundPort(Protocol):
    async def dispatch(
        self,
        envelope: "OutboundEnvelope",
    ) -> "ChannelDeliveryReceipt": ...

class PluginChannels(Protocol):
    async def register(self, ctx: Context, definition: ChannelDefinition) -> None: ...

class ChannelRegistry(Protocol):
    def freeze(self) -> ChannelRegistrySnapshot: ...  # Core-only owner

@dataclass(frozen=True, slots=True)
class CoreChannelDefinition:
    name: str
    capabilities: tuple[ChannelCapability, ...]
    adapter_id: str

@dataclass(frozen=True, slots=True)
class LegacyChannelAdapterDescriptor:
    name: str
    generation_id: str
    adapter_id: str

@dataclass(frozen=True, slots=True)
class CommittedChannelCatalog:
    core_definitions: tuple[CoreChannelDefinition, ...]
    v3: ChannelRegistrySnapshot
    migration_v2: tuple[LegacyChannelAdapterDescriptor, ...]
    identity: str

class ChannelHost(Protocol):
    def acquire_binding(
        self,
        snapshot_lease: RuntimeSnapshotLease,
        channel_name: str,
    ) -> ChannelBindingLease: ...

@dataclass(frozen=True, slots=True)
class ChannelReady:
    binding_token: str
    subscriptions: tuple[str, ...]
    admission_open: bool = False

@dataclass(frozen=True, slots=True)
class StopReceipt:
    binding_token: str
    resources_closed: bool
    failures: tuple[ChannelCleanupFailure, ...] = ()

@dataclass(frozen=True, slots=True)
class ChannelCleanupFailure:
    stage: str
    plugin_id: str
    generation_id: str
    binding_token: str
    resource: str
    error_type: str
    message: str
    retry_action: str

@dataclass(frozen=True, slots=True)
class ChannelFactoryContext:
    snapshot_id: str
    generation_id: str
    binding_token: str
    config: Mapping[str, JsonValue | CredentialRef]
    credentials: Mapping[str, CredentialRef]
    provider_client_factory: ProviderClientFactory
    ingress: ChannelIngress | None
    identity: ChannelIdentityPort | None

@dataclass(frozen=True, slots=True)
class ChannelPresentationPorts:  # C14d
    control: ChannelControlPort
    turn_stream: TurnStreamPort

@dataclass(frozen=True, slots=True)
class ProviderDeliveryRequest:
    binding_token: str
    delivery_id: str
    recipient: str
    body: str

@dataclass(frozen=True, slots=True)
class ProviderDeliveryReceipt:
    delivery_id: str
    status: DeliveryStatus
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

class ChannelAdapter(Protocol):
    async def start(self) -> ChannelReady: ...
    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt: ...
    async def stop(self) -> StopReceipt: ...

class RuntimeChannelAdapter(ChannelAdapter, Protocol):  # C14c
    def attach_runtime(self, ports: ChannelRuntimePorts) -> None: ...

class PresentationChannelAdapter(RuntimeChannelAdapter, Protocol):  # C14d
    def attach_presentation(self, ports: ChannelPresentationPorts) -> None: ...
```

`ChannelRegistrySnapshot.identity` 是 canonical JSON SHA-256：payload 固定为按 channel name 排序的完整
`ChannelDescriptor`，以及按 `plugin_id/generation_id/channel_name/source_revision/config_revision/factory_export` 完整
六字段排序的 `ChannelFactoryProvenance`；同一 plugin/generation/channel 只能有一项，不允许 tie 依赖注册顺序。不含 callable、live
adapter、secret、client、Root token 或临时 endpoint。candidate/formal 同 generation/source/config 的 identity 必须相同，
发布校验还必须要求 `root_instance_token is runtime_snapshot.composition_root.instance_token`，防止把另一棵 Root 的 registry 与
identity 成对替换。

上述 `Context` 与 `RuntimeSnapshotLease` 是 Core 已有 exact-runtime 类型，不由插件仓复制；
`ChannelInboundMessage` 是 C14c 新建的 frozen text projection，不复用现有 mutable `bus.events.InboundMessage`。
所有插件可见 channel 类型的唯一 source 是 `agent/plugin_composition/channels.py`；
`agent.plugin_composition.__init__` 只重导出 `CHANNELS/PluginChannels/ChannelDefinition/ChannelCapability/
InboundIdentity/CredentialRef/RawInbound/ChannelAdapter` 与 factory 所需窄 Protocol。
外部 Feishu/QQBot 只能从这两个入口 import，不得复制 DTO。`ChannelRegistry` 与 freeze helper 留在同模块私有面；
committed catalog 是 exact `RuntimeSnapshot.channel_registry`；`ChannelBindingLease/ChannelGenerationHost` 的实现与
live binding 只属于 `agent/plugins/channel_generation_host.py`，不从插件 public package 导出。插件注入面只有
`PluginChannels.register()`，不能提前 freeze/unregister；inspection 也只能读取公开 stable snapshot 的 frozen catalog。

`InboundEnvelope.handoff()` 只有当前 owner 与 `expected_owner` 相同、转移符合
`INGRESS/ADMITTED→BUS/BUS_QUEUED→LANE/LANE_QUEUED→LOOP/RUNNING` 时才原位更新 owner/state并返回同一 envelope；
mismatch、terminal 后 handoff、跳级都 fail-loud。envelope 内部保存 `_closed_by`：`close(expected_owner)` 由 exact
当前 owner 首次调用时设置 `_closed_by=expected_owner`、状态 TERMINAL、owner CLOSED并释放 lease；之后只有同一
`expected_owner == _closed_by` 的重复 close 幂等，任何其他 owner 都 fail-loud。异常/取消通过 critical completion 等 lease 真正释放后
再传播。handoff 成功后旧 owner 绝不能 close。

provider adapter 必须先把原始 provider id 投影为 `RawInbound.message_id`，Core 在构造 `InboundEnvelope` 前集中校验
1～256 非空字符并执行 bounded dedupe；不得从 metadata 在后续 lane 再猜 id。现有 Feishu/QQBot
metadata 只保留兼容诊断，迁移后不再是 identity owner。

`ChannelFactoryContext.config` 在 Core boundary 递归 canonicalize：Pydantic model 先 `model_dump(mode="json")`，list
转 tuple、mapping key 必须是字符串并按 key 冻结、credential path 在 freeze 前替换为 `CredentialRef`，拒绝 set、任意
object、cycle 与非有限 float。provenance 的 `config_revision` 对该 canonical projection 做 digest；candidate/formal
Definition 必须得到完全相同的 immutable config。这样 Feishu `list[str]` 与 QQBot `list[QQBotGroupConfigModel]` 分别
变成 tuple 与 tuple-of-mapping，不把 mutable/Pydantic object 跨 Root 携带。
`ChannelStartRecord.config_revision` 记录这份脱敏投影摘要；另有只进入 Core journal 的
`raw_config_revision` 记录 `config.local.toml` 原始字节摘要。前者参与 catalog identity，后者只用于 formal
start fence，二者不能混用，raw 摘要不进入公开 registry。

`ChannelFactoryContext` 不含已构造的 HTTP/SDK client 或 secret bytes；`provider_client_factory.create(refs)` 只能由
formal adapter 的 `start()` 调用，Core-owned factory 在内部解析 raw credential 并把 client 纳入 Host binding。
candidate、factory 调用与 `start()` 之前的 credential resolution/client construction count 必须为 0；`stop()`
结束时 client 与内部 secret lease 一同归零，adapter 不能缓存 raw secret。

1. `name` 在 built-in、v2 transition 与 v3 registry 的 committed namespace 内唯一。
2. `capabilities` 是 frozen typed enum set；unknown、重复或非 canonical 顺序在 admission fail-loud，canonical
   descriptor 以 enum value 排序后进入 snapshot identity。
   含 `INBOUND` 时 `inbound_identity` 必填；不含 `INBOUND` 时必须为 `None`，outbound-only adapter 不填写无意义 identity。
3. registry 保存 `factory_export`，不保存 callable/closure。candidate 与 formal Root 分别注册 blueprint；Host 先完成
   durable binding reservation 与 raw config fence，之后才从 exact committed generation/module 解析 formal export；
   module `__getattr__` 或错误 export 不能先于 durable boundary 执行。candidate module 卸载后 stable channel 不得引用它。
4. factory 必须是同步、side-effect-free 的 `factory(ChannelFactoryContext) -> ChannelAdapter`。Context 只含
   generation identity、无副作用 provider client factory、credential view、只读非敏感 config 与 Core 窄端口；
   不暴露 PluginManager、Session repository、EventBus、MessagePush 或任意 workspace/data root。
5. `akashic.plugin.toml [channel_credentials]` 在 artifact install/admission 与 candidate workspace copy 之前由
   Core import-free 校验；每个 Definition 的 `credential_paths` 必须与对应静态 entry 完全相同。不能以
   module-level export 充当 secret 边界，因为读取它时 candidate import 和 config copy 已经发生。Feishu 使用
   `app_id/app_secret` 及兼容 `appId/appSecret`，QQBot 使用 `app_id/client_secret` 及兼容
   `appId/clientSecret`。candidate/formal `apply` 得到的 config 在这些路径上
   都是不可解引用的 `CredentialRef`；只有 formal Host start 通过 Core credential provider 从原始 config owner
   解析 secret。Definition/factory closure 不接受 secret bytes，candidate credential-provider 调用必须为 0。
   Feishu/QQBot v3 的 `ConfigModel` 必须在这些字段显式接受 `CredentialRef`，不得用 `str(value)` 把 opaque ref
   重新解释为 credential；旧 `str`-only schema 在 admission 直接失败。
   若 ConfigModel 为兼容旧配置接受 `appSecret/clientSecret` 等 physical alias，静态 manifest 与 Definition 必须列出
   每一个可接受的 raw dotted path；Core 在 ConfigModel validation / `apply` 前核对 Pydantic input alias 的完整集合并
   逐项 redaction，任一未声明 alias 都不能作为 secret field 被 v3 ConfigModel 接受。opaque credential 字段只能是
   `CredentialRef` 或 optional `CredentialRef`，不能保留 `str` 逃逸面；同一逻辑 credential 的多个 physical alias
   同时出现在原始 config 时 admission fail-loud，formal resolver 不按 alias 顺序猜值。多个 channel 可以复用同一个
   exact path，但跨 channel 的父子 path overlap fail-loud。
   manifest 是 import-free admission；首次 formal module import 仍发生在 projection 前，因此这是一条 supported API
   与 exact-source Gate，不是阻止同 UID 恶意 module-level 文件读取的安全沙箱。candidate data copy 会在 candidate
   import 前排除 `config.local.toml`，candidate `apply` 只能收到已经核验的 opaque projection。
   这是 supported API 与 exact-source Gate，不是同 UID Python 安全沙箱：formal plugin 仍持有自己的 `ctx.data_root`，
   Core 不承诺阻止恶意反射或自行读取文件。首批 exact Feishu/QQBot source 必须证明 `apply/factory` 在 Host formal
   start 前没有 raw config read，credential 只经 Core resolver 进入 provider client factory。
6. factory 构造无副作用；网络、subscription、callback 与后台 task 只能在 `start()` 获取。新 channel start 返回
   `ChannelReady(admission_open=False)`；Host 校验 binding token/订阅 ready 后才允许 finalize。`stop()` 返回
   `StopReceipt`，只有 `resources_closed=True` 且 failures 为空才算 cleanup 完成；每个 `ChannelCleanupFailure` 固定
   generation/module/binding/resource/retry identity，否则保留 cleanup-pending owner。
7. `PluginChannels.register(ctx, definition) -> None` 在内部建立调用 Fiber 的 registration Effect，不返回可由插件
   提前 dispose 的 token，也不提供 public unregister。Fiber dispose 逆序注销 descriptor；formal Host runtime binding
   独立拥有 start/stop/subscription/client cleanup。descriptor 固定
   `name/capabilities/factory_export/inbound_identity/owner/credential_paths` 并进入 snapshot identity。
8. `ChannelControlPort` 是 `/stop` 的唯一中断入口，并且是已经 attach 到 exact binding 的 per-binding facade。adapter
   只把尚未 admission 的 `RawInbound` 与 `ControlResponseBodies(interrupted, idle)` 交给 `interrupt()`；Core 先完整校验，
   再在任何 await 前按 `(provider account, binding token, raw.message_id)` claim dedupe，并从 facade 固定的 binding key
   acquire/fork lease。它不按 channel name/current snapshot 重查，也不构造或交给 Bus 普通 ingress 的
   `InboundEnvelope`。acquire 在 interrupt 前失败时释放 claim；一旦调用 interrupt（包括 idle）即保留 claim。
   Core 根据 typed interrupt outcome 选择本地化 response body，按 `(binding token, raw.message_id)` 派生唯一 outbound
   identity，经 same-binding awaited dispatch 后 terminal close。合法结果固定为：interrupted =>
   `accepted=True, reason=interrupted, response=settled`；idle => `accepted=False, reason=idle, response=settled`；duplicate =>
   `accepted=False, reason=duplicate, response=None`；binding 在 effect 前关闭 =>
   `accepted=False, reason=binding_closed, response=None`。`accepted` 只表示 interrupt effect，和 response delivery 是两项
   独立事实；`accepted=True + response=UNKNOWN` 时绝不能再次 interrupt、重试 response 或释放 claim。
   `TurnStreamPort` 同样是 per-binding facade，只投影 frozen `turn.started/stream.delta/tool.started/tool.completed/
   turn.output.completed`，不接受调用方 token，也不暴露 EventBus 或任意 Mapping payload。`turn.started.client_message_id`
   固定继承已 acceptance 的 `RawInbound.message_id`，供 QQBot input_notify 去重；`turn.output.completed` 只收口 remote
   preview，不能替代 assistant final。`TurnStreamCallback` 必须是 async；Host 在调用前递增 exact
   binding in-flight counter，并要求每次返回同 `presentation_id` 的 typed `PresentationReceipt`。普通异常一律映射
   `UNKNOWN`、转成 owner-specific Incident 并停止该 presentation 后续 patch；只有 provider classifier 能证明 effect=0
   时才返回 `REJECTED` 并允许显式 fallback。callback terminal 的 finally 才递减；caller cancellation 先完成 callback
   cleanup 再传播。`StreamSubscription.close_admission()`
   同步拒绝新 callback，`await_quiescence()` 等所有已接纳 callback terminal，`close()` 在 quiescent 后幂等 detach；
   wrong binding/close 后 callback fail-loud。subscription 与 outbound/inbound 共用 Host drain，stop 前必须关闭 admission
   并等待 callback。
   Registry/Host 必须按 descriptor capability 发放窄 port：没有 `CONTROL` 时 factory context 不可调用 control；没有
   `TURN_STREAM` 时不可 subscribe；没有 `INBOUND/OUTBOUND` 时对应 ingress/outbound port 不存在。Feishu/QQBot 的正式
   Definition 必须包含四项能力，删掉任一项都会在 factory admission/行为 Gate fail-loud。
   `presentation_id` 在同一个 remote preview artifact 的 started/delta/tool/output-completed 全程稳定，事件另带单调
   sequence；QQ input_notify 使用独立 `notify:<provider_message_id>`，live preview 使用 `preview:<turn_id>`，二者不能
   混成一个可重试效果。provider adapter callback 自己且只在 classifier 证明首个调用 effect=0 时执行一次
   provider-specific fallback，并返回 fallback 的同-presentation receipt；Host 不因 `REJECTED` 重调 callback。
   timeout/cancel/普通异常、receipt 类型或 presentation id mismatch 都先结算 `UNKNOWN`、记录 owner-specific Incident、
   关闭该 presentation，再传播内部合同错误或 caller cancellation；UNKNOWN 后禁止换 presentation id 继续 patch。
   QQBot 的 `input_notify` 不建立第三种 port：它由 exact `turn.started` presentation 驱动，payload 必须带已接受的
   provider message id并与普通 ingress 共用 dedupe。QQBot terminal stream 只收口 preview，不能代替 assistant final；
   final 始终另走 C14c awaited delivery。
   live-card/stream preview 使用独立 `presentation_id + binding_token` journal，不计为最终 logical delivery；preview patch
   after-effect failure 记 Incident、停止该 presentation 后续 patch且不重试，最终 assistant 仍以自己的
   `logical_delivery_id` 走 awaited Receipt。
   `ChannelIdentityPort` 是 provider identity 与可主动发送 recipient 映射的唯一读取面，不暴露 Session repository；
   `ChannelIngress.admit()` 根据 `RawInbound.provider_identity/recipient` 在同一 Core acceptance 临界段提交 mapping，
   写入失败时不得返回 envelope 或进入 Bus，stop/drain 后旧 binding 不得再写。Feishu 的 `ou_` 反查必须经该 port，不能让 adapter 重获
   `SessionManager` 或自行扫描 Session metadata。权威 owner 是 `sessions.db/channel_identities` 的
   `(channel, identity)` 唯一行；Session metadata 只作为旧版本非破坏迁移输入。首次 rebuild 仅在该 channel 尚无 durable
   migration marker 时按 `updated_at,key` 稳定顺序 seed，并在同一事务写入 marker；之后即使用户显式删除使权威表变空，
   包括重启在内也只读权威表，不再按重复 metadata 的扫描顺序裁决或复活 recipient。
   `remember()` 必须把目标 Session metadata 与唯一 identity 行放在同一 SQLite transaction；事务失败时不得留下新 Session
   行、cache 或 index 变化，已有 Session metadata/cache 也保持原值。并发 move 最终只能留下一个权威 recipient；历史 metadata
   不自动删除，因为普通 channel admission 无权减少既有 Session 状态。
9. 附件不属于 C14a～C14d，由独立 C23 persistence 合同拥有。C23 可以先加入 opaque DTO 与窄 Protocol，但在
   artifact intent/ready row、Session binding transaction、Host per-binding read lease、Mobile idempotency mapping 与
   backup/restore Gate 全部闭合前，production factory context 的 attachment ports 必须保持 `None`，Channel wire 与
   MessagePush 继续 text-only。首批 Feishu/QQBot v3 adapter 遇到附件输入返回 `REJECTED`，且不读取 workspace path、
   不复制 uploads、不删除旧文件。现有 v2 attachment 行为保留到 C23 与专门迁移批次完成。

Candidate 允许写集只有 `runtime/plugin-validation/<generation>/...`、candidate import/module 和 reload journal；
禁止写正式 `plugin-data`、`config.local.toml`、Session DB/messages/uploads、manifest/artifact、EventBus/MessagePush、
provider network。promotion 本身只切 endpoint/registration，不发送业务消息。

## 3. publication、Host 与 lease

1. Root settle/freeze 后 snapshot compiler 生成 `ChannelRegistrySnapshot`，并写入
   `RuntimeSnapshot.channel_registry`。它只含 frozen descriptor、formal factory identity、plugin/generation/source/
   config revision，不含 live Channel。`PluginManager.stable_channel_catalog()` 从 exact public current 返回只读值，
   是 discovery、Host 和 inspection 的唯一公开入口；candidate 不投影 live Host。recording validation 也只能由
   Core 显式 Gate 取得 candidate registry，不能成为公开 channel。
2. promotion 先 pause old/new admission，等待 old Host in-flight owner 与 stable lease 收束，seal candidate，再建立
   closed provisional transaction；其间所有 `current_snapshot`、discovery、Mobile/WebUI/inspection 仍读取 old stable。
   `PluginManager` 是 services、command catalog 与 ChannelHost 的唯一 publication coordinator：先由 SnapshotStore 建立并
   commit closed provisional，再依次驱动各 participant；全部成功后只由 Manager 一次 finalize。ChannelHost 不得直接访问
   SnapshotStore、另开 provisional transaction 或提前发布自己的 current binding。现有先 `_switch_plugin_endpoints()`、后
   `begin_publish()` 的顺序必须在 C14b 改掉；删除这一顺序约束会使 blocking participant oracle 失败。
3. transaction 顺序固定：`old admission close → old in-flight drain → old stop → old binding close → managed service
   switch → formal factory materialize → new bind/start/ready(closed) → stable pointer + Host binding finalize → new
   admission open`。
4. 任一步失败按逆序执行 `new stop/binding close → service restore old → old exact-token bind/start/ready → old
   admission open`；旧 restore 失败
   保留结构化 ownership failure、generation/module/resource，publication fail-loud，不能只恢复内存 pointer。
5. caller cancellation 由 Core critical cleanup 完整执行后再恢复 `CancelledError`；每个 stop/close 错误聚合并保留
   cleanup-pending owner，支持显式 retry，不能提前 pop generation/Host binding。
   C14b 的 channel cleanup 复用 reload journal 的 exact runtime owner，而不是只留内存错误：adapter 启动前先 durable
   记录 `binding_token/snapshot_id/catalog identity/plugin generation/artifact pointer/descriptor/factory export/target/
   boot owner/attempt`；journal 写入失败时 `start_count == 0`。`stop()` 或 rollback 失败后保留 admission-closed binding 与
   tombstone；同进程显式 retry 只按 exact token 清理 retained adapter/factory，Root dispose 后不得回调 plugin Fiber。
   tombstone 未收束前不能删除 generation/module/closeables，也不能按当前同名 channel 猜测旧 owner。跨进程时 live Python
   object 已不存在，Core 只能先由 supervised BootGuardian 收束旧 boot，再按 journal 的 exact artifact/target 重建权威
   stable Channel runtime 并写 recovery receipt；不得伪称能跨进程调用旧 adapter。
   formal Host 在解析任何 `CredentialRef`、构造 provider client 前，还必须重读原始配置的完整 revision并与 generation
   冻结的 Core-only `raw_config_revision` 完全相等；脱敏 `config_revision` 只证明 catalog/projection identity。candidate seal
   后即使只轮换 secret，也必须 fail-closed、rollback 并要求 reprepare，
   不能用新 secret 启动旧 snapshot。secret bytes 不进入 snapshot、registry 或 journal。
6. Host 为每个 committed binding 提供 admission/in-flight counter。`ChannelIngress.admit(RawInbound)` 获取 exact
   `ChannelBindingLease` 并以 keyword-only constructor 创建进程内 envelope。所有权只转移一次：
   `ChannelIngress → MessageBus queue → PassiveWorker/lane → AgentLoop terminal`；enqueue/drop/cancel/stop 任一 owner
   失败都在自己的边界 close，成功 handoff 后不得再 close/fork。AgentLoop 收到 envelope 必须绑定其 lease，禁止
   再 acquire current。普通 ingress 的旧代入队但未消费消息因此仍用 old snapshot。`/stop` 不进入此 owner 链；它由
   per-binding `ChannelControlPort` 从 attached exact key 单独 fork lease、完成 effect/response 后关闭。
7. outbound 的 Bus 与 direct MessagePush 共享唯一 `OutboundEnvelope` owner；首次 admission 生成稳定
   `delivery_id`，queue/fallback 复用同一 identity。callback 返回
   `ChannelDeliveryReceipt(DELIVERED | REJECTED | UNKNOWN, delivery_id, provider_id?)`，不得用异常表达可重试结果；
   未捕获 provider 异常由 adapter/Bus 转成 `UNKNOWN`。`UNKNOWN` 记录 Incident，并禁止 Bus retry、MessagePush retry
   与 provider 内部 fallback；只有能证明远端未接收的确定性 `REJECTED` 才可按 owner policy 发起新 attempt。
   Host drain 等待正在执行的 outbound callback 与 provider receipt。`PushToolOutboundPort` 不建立第二套
   callback/binding：它只把 direct Push 请求变成 `OutboundEnvelope`，随后委托同一个
   `BusOutboundPort.dispatch()` / `publish_outbound_awaited()`；direct Push 也必须携带 exact
   `binding_token`，并参与同一 queued/running drain。
   `ChannelHost.acquire_binding(snapshot_lease, channel)` 不消费 caller 的 snapshot lease，而是在核对 catalog/binding 后
   fork 一份 exact lease并封装成 `ChannelBindingLease`。direct Push caller 是该 binding lease 的唯一 close owner：无论
   dispatch 正常、异常或取消，都在 Receipt/critical cleanup 后 `finally await binding_lease.aclose()`；
   `PushToolOutboundPort.dispatch()` 只借用 lease，不得替 caller 关闭。
   passive turn 内的 `message_push` 必须把其既有 passive lane 身份显式传给 Bus，不能作为 non-passive send 等待自己
   结束；独立 scheduler/direct push 则使用 non-passive lane，并等待该 chat 的既有 passive turn。Bus terminal close 必须
   取消并收束已 dequeue 但仍在 lane 等待的 direct push：provider 尚未调用时返回 `REJECTED`，已经调用但没有收据时
   返回 `UNKNOWN`，随后 caller 才释放 exact binding。
   第一版边界固定为：

   ```python
   @dataclass(frozen=True, slots=True)
   class OutboundEnvelope:
       logical_delivery_id: str
       delivery_id: str
       attempt_sequence: int
       snapshot_id: str
       generation_id: str
       binding_token: str
       channel: str
       recipient: str
       body: str
       metadata: Mapping[str, JsonValue]

   @dataclass(frozen=True, slots=True)
   class ChannelDeliveryReceipt:
       delivery_id: str
       status: DeliveryStatus  # DELIVERED | REJECTED | UNKNOWN
       provider_ids: tuple[str, ...] = ()
       error: str | None = None

   ChannelOutboundCallback = Callable[[OutboundEnvelope], Awaitable[ChannelDeliveryReceipt]]
   async def publish_outbound_awaited(envelope: OutboundEnvelope) -> ChannelDeliveryReceipt: ...

   @dataclass(frozen=True, slots=True)
   class PushToolRequest:
       channel: str
       recipient: str
       body: str
       metadata: Mapping[str, JsonValue]

   class PushToolOutboundPort(Protocol):
       async def dispatch(
           self,
           request: PushToolRequest,
           binding_lease: ChannelBindingLease,
       ) -> ChannelDeliveryReceipt: ...
   ```
8. `OutboundEnvelope` 固定 `logical_delivery_id/delivery_id/attempt_sequence/snapshot_id/generation_id/binding_token/
   channel/recipient/body/metadata`；队列和订阅按 exact binding token 路由，不能只按 channel name。
   首次 attempt 的 `logical_delivery_id == delivery_id` 且 sequence=1；只有确定性 `REJECTED` 后由显式 owner policy
   发起的新 attempt 才生成新 `delivery_id`、保留 logical id 并递增 sequence。journal 分别按 logical id 与 attempt id
   计数，`UNKNOWN` 后禁止生成新 attempt。切换前已入队但未 dispatch 的消息
   仍交给 old binding，old Host 在 queued + running receipt 全归零后才关闭 provider client；new 同名 binding 不得抢取。
9. `MessageBus.publish_outbound_awaited()`、Bus callback、`BusOutboundPort.dispatch()` 与 direct Push 都返回同一个
   `ChannelDeliveryReceipt`。AgentLoop 的正常 assistant、异常/取消 terminal、provider control response 与所有其他用户可见
   输出都只走 awaited `dispatch()`；callback 尚未执行时不得完成 Turn 或把旧 enqueue 映射为 `SUCCESS`。
   `agent/looping/core.py`、`agent/turns/outbound.py` 与 `bootstrap/passive_worker.py` 的旧 fire-and-forget/入队即成功
   路径必须一并迁移。正文成功但附件/后续 part 失败的 `PARTIAL` 一律映射 `UNKNOWN`。非用户关键 telemetry 若保留
   fire-and-forget，只能返回独立 `QueuedReceipt(delivery_id, queued=True)`；后台最终 receipt 由 Bus 写结构化
   delivery journal/Incident，不参与发起 Turn 的成功判定。
   `MessagePushTool` 的 settled tool result 使用稳定 JSON 对象
   `{delivery_id, status, retryable: false, provider_ids, error}`，不得把 `UNKNOWN` 压成“消息已发送”或无身份的
   “发送失败”；上层据此不得对同一 logical delivery 自动再次调用工具。
10. inbound message id 是 1～256 字符的 provider boundary 必填字段，空白/缺失 fail-loud。Feishu 与 QQBot 在
    `/stop` 等 control 分支前共用 `MessageDeduper(max_size=500)`。dedupe scope 固定为 provider account + channel
    binding；完整校验成功后、进入 control/普通 enqueue 前 claim。enqueue/handoff 在 Bus 接收前失败时释放 claim；
    Bus 已接收后保留到 bounded window 淘汰。v1 只保证最近 500 个 accepted ids 在同进程不重复；第 501 个以后
    最老 id 和进程重启后的 provider redelivery 都允许再次进入，合同不伪称 durable/process-lifetime exactly-once。
11. inbound envelope 构造后先处于 `INGRESS/ADMITTED`，随后固定四个处理态
    `BUS_QUEUED → LANE_QUEUED → RUNNING → TERMINAL`。可执行入口固定为
    `ChannelIngress.admit(raw_message) -> bool`（内部取得 exact public stable lease；`False` 仅表示 bounded duplicate）、
    `MessageBus.enqueue_inbound(envelope)`（成功后取得 BUS_QUEUED owner）、
    `PassiveMessageWorker.accept(envelope)`（取得 LANE_QUEUED owner）与
    `AgentLoop.react_envelope(envelope)`（取得 RUNNING owner并绑定 envelope lease）。每次通过
    `envelope.handoff(expected_owner, next_owner)` 转移唯一 close owner。`MessageBus.aclose()` 逐条 close
    `BUS_QUEUED`；`PassiveMessageWorker.stop()` 取消新 admission 后逐条 close `LANE_QUEUED`，等待 RUNNING terminal；
    AgentLoop finally close RUNNING。queue drop/caller cancel 由当时 owner close。shutdown 顺序固定为 channel ingress
    close admission → Bus stop accepting new envelopes并逐条 close BUS_QUEUED → PassiveWorker stop accepting并逐条
    close LANE_QUEUED → 等待 RUNNING terminal → Host outbound drain/stop。测试结束 lease/in-flight 全零，不能只
    清空 `_lane_queues`。
12. `/stop` 等 provider control 命令把尚未普通 admission 的 `RawInbound` 交给 per-binding Control facade；Core 按
    attached exact key fork lease、dedupe、执行 typed interrupt，并通过同一 binding 的 `BusOutboundPort.dispatch()`
    发送所选响应；禁止插件直接 `self.send()`。旧 binding admission 已关闭且 effect 尚未开始时返回
    `binding_closed/response=None`，provider 调用后失败仍结算 response `UNKNOWN`。
    QQBot 的 `input_notify` 与两者的 live preview 都属于 presentation effect，必须带 exact binding token并参与
    Host drain；它们不是最终 delivery，也不得触发 final message fallback。interactive→text、stream→ordinary 只有
    provider-specific classifier 能证明首个调用尚未产生远端 effect 时才返回 `REJECTED` 并允许显式 fallback；超时、
    连接中断、响应解析失败及任何 after-effect failure 一律 `UNKNOWN`，同一 logical delivery 不得重试。
13. 普通 assistant、error 与 cancel 从发起 Turn 的 `InboundEnvelope` 派生 outbound identity；control response 则从
    `(attached binding token, raw.message_id)` 派生并持有 Control facade fork 的 lease。两者的
    `snapshot_id/generation_id/binding_token` 都不能在 terminal 时按 channel name 或 current snapshot 重查。无 inbound 的
    proactive/direct Push 必须从其调用时持有的 exact snapshot lease，经
    `ChannelHost.acquire_binding(snapshot_lease, channel)` 解析一次 `ChannelBindingLease`；排队和 dispatch 全程持有该
    lease，caller 在 terminal finally 关闭，不能只传裸 RuntimeSnapshotLease、让 port 隐式关闭或按 channel name 重查。

Bus/Host 的可执行 drain API 固定为：

```python
binding = bus.bind_outbound(binding_token, channel_name, callback)
binding.close_admission()
await binding.await_quiescence()  # queued + running receipts == 0
await channel.stop()              # critical completion; provider resources settle
await binding.close()             # idempotent detach after provider stop
```

`publish_outbound[_awaited]` 必须按 envelope binding token 找 exact binding；找不到不能 fallback 到同名新 generation。
ChannelHost swap 先 `close_admission → await_quiescence → channel.stop → binding.close`，再切 service 并建立新代。
旧 queue 已在 `await_quiescence()` 明确归零；rollback 只能用同一 old token、old adapter identity 重新 bind/start/open，
不能用同名新 generation fallback。`channel.stop()` 或 `binding.close()` 的异常/取消必须在 critical cleanup 中继续
settle；任一无法确认完成时保留 old Host/binding/generation 为 `cleanup_pending`，公开 stable 仍为 old 但 admission
保持关闭并报告 degraded，禁止继续切 service。显式 `retry_channel_cleanup(token)` 成功后才允许恢复 old
start/open 或重试 publication。新代 start/finalize 失败时按 `new stop → new binding close → service restore old →
old exact-token bind → old start/ready → old admission open`；任何恢复失败同样保留 owner，不得 pop closeables。

迁移期 `RuntimeSnapshot.channels` 继续承载 v2 live object；新增 v3 `channel_registry` 独立存在。v2 live channel
只能来自已经 committed 的旧 generation 或 formal stable boot；candidate admission 不得调用 `Plugin.channels()`、
读取原始 credential 或构造 SDK/HTTP client。修改了 v2 channel plugin 的 candidate 在 contribution compile 前
fail-loud，要求先迁到 v3 blueprint；其他 candidate 只复用已 committed v2 generation 的冻结 contribution。
最后一个 external v2 channel 迁走并通过 E3 后，只能删除 Feishu/QQBot 的 `Plugin.channels()` consumer；
Core Telegram/QQ/Web/Mobile 仍是旧 Channel/MessageBus adapter。只有 C14c 将这些 Core adapter 逐一接入
`CommittedChannelCatalog`、并由 zero-consumer Gate 证明旧 bootstrap/Bus 路径为空后，Host 才只消费 committed registry，删除
`Plugin.channels()`、
`PluginContributions.channels`、Manager mutable `_channels` 与 bootstrap `plugin_channels` 参数。

Core 内建 Telegram/QQ/Web/mobile adapter 由 `CoreChannelDefinition` 投影；Host 的唯一输入是
`CommittedChannelCatalog(core_definitions + v3 ChannelRegistrySnapshot + migration-only v2 adapters)`。Core name 优先保留，
v3/v2 collision fail-loud，不允许覆盖。迁移期 v2 callback 用薄 adapter 转成 `ChannelDeliveryReceipt`：正常返回只能在
provider 调用已明确成功时标 `DELIVERED`；旧 `FAILED` 只有 adapter 能证明 provider 尚未被调用/远端未接收时映射
`REJECTED`，其余 `FAILED`、异常与 `PARTIAL` 都映射 `UNKNOWN`，且不再由 Bus 自动重试。E3 后删除 v2 merge lane。

## 4. command catalog 边界

- ChannelDefinition 不注册 command、不拥有 handler、不修改 `CommandRegistry`；命令只由 C11 `COMMANDS` 注册。
- channel 只能取得按 provider 类型明确投影的 committed 只读 catalog；Feishu/QQBot 默认不消费 bot command catalog。
- `telegram_bot_commands/mobile_bot_commands` 的删除依据是最后一个 v2 command consumer，而不是最后一个 v2 channel。
- C11 external command refresh 与 C14 channel swap 可共享 publication transaction，但两套 catalog/rollback owner
  保持独立，不能把 Proactive quiescer 当 channel drain。

## 5. 验证与停止条件

### C14 unit/Manager oracle

- registry：非法/重复/builtin collision、non-callable export、错误 adapter、freeze、Effect cleanup、descriptor identity；
- snapshot：candidate/formal `ChannelRegistrySnapshot` identity 等价；stable accessor 只见 public current，
  RuntimeSnapshot 不再从 live v2 object 推导 v3 catalog；
- credential：manifest 在 import-free admission/candidate copy 前声明 credential path；ConfigModel validation 与
  `apply` 前核对完整 physical alias 并完成 redaction；candidate config copy 不含正式 config，apply 只见
  CredentialRef；candidate provider 调用为 0；正式 config/plugin-data digest 不变；
- factory：candidate、factory 与 formal `start()` 前 credential resolution/provider client construction=0；只有 start
  可调用 `provider_client_factory.create(refs)`；ChannelReady 必须 closed，StopReceipt 必须资源全关；
- candidate：install/latest/discard 对正式 Host、bus、push、provider、port 的调用为 0，写集只在 validation root；
- legacy candidate：变更 v2 channel plugin 时 `Plugin.channels()` 调用、raw credential read、SDK/client construction
  全为 0 并 fail-loud；未变化 v2 channel 只复用 old committed contribution；
- stable：真实 Manager committed snapshot → Host materialize/start；candidate module 卸载后 formal factory仍可用；
- provisional：remote/start await 期间公开 current/catalog/inspection 全为 old；成功后三者一起切换；
- swap：new factory/start、service、owner callback、caller cancel、old restore failure 都核对 Host/snapshot/endpoint/
  admission/cleanup-pending；阻塞 inbound/outbound adapter 时 stop 必须等待 drain；
- ingress：old message 入队、worker 阻塞、hot switch 后仍绑定 old snapshot；bounded dedupe window 内相同 message id
  只产生一个 Turn；该 Turn 在 swap 后恢复生成的 normal/error/cancel response 仍携带 old binding token；
- identity：legacy Session metadata 只 seed 一次并留下 durable marker；同 identity 从 old recipient move 到 new 后重启仍解析 new，权威表恰一行；
  并发 move 仍只有一个 durable owner；新 recipient 与已有 Session 两种 SQLite failure 都断言 identity row、Session row/cache、
  in-memory index 保持提交前状态；move old→new 后显式删除 new，重启不得从 old metadata 复活；identity write 阻塞时
  close/drain 必须等待 exact binding，不得在 stop 后写入；
- dedupe：同 account/binding 最近 500 个 id 去重；enqueue-before-handoff failure 释放 claim；第 501 个淘汰最老项的
  bounded 行为显式可见；重复 `/stop` 在窗口内只有一次 control/outbound；
- queue：old outbound 在切换前入队但阻塞 dispatch，swap 后仍由 old binding token 发送；worker/Bus stop 逐条释放
  queued inbound lease；分别在 BUS_QUEUED、LANE_QUEUED、RUNNING 阶段 shutdown，断言 exact lease/owner 收束；
- outbound：稳定 delivery id；Bus 与 direct Push 都覆盖 provider 先记录 effect 再抛错 → `UNKNOWN`、effect count=1、
  无自动第二次调用；Feishu interactive→text 与 QQBot stream→ordinary 只有确定性 pre-effect `REJECTED` 才 fallback；
- attempts：确定性 `REJECTED` 后显式新 attempt 保留 logical id、生成新 delivery id 并递增 sequence；`UNKNOWN`
  后 attempt count 不变；
- error terminal：真实 `agent/looping/core.py` 异常分支等待 Receipt，`UNKNOWN` 不得被 Turn/Tool 标成发送成功；
- entry matrix：`passive_turn`、`after_turn`、turn orchestrator/outbound、passive worker、Telegram/QQ/Web/Mobile adapter
  的 normal/error/cancel/proactive/control 输出均产生 awaited Receipt；删除任一路旧 callback 会使 Gate 失败；
- push result：direct MessagePush 的 ToolResult 保留 delivery id、三态 status 与 `retryable=false`，相同 id 的
  logical attempt count 始终为 1；
- control/stream：`/stop` 只调 ChannelControlPort；preview 只经 TurnStreamPort，subscription 纳入 drain；preview
  after-effect failure 不重试且不伪造 final delivery；
- text-only：首批 adapter 收到 provider attachment 或 push attachment 请求时确定性 `REJECTED`，不读 workspace path、
  不写 uploads，也不创建临时 attachment owner；
- config：Feishu list 与 QQBot nested Pydantic model 经 canonical projection 后 candidate/formal digest 相等，mutable source
  变化不反向改变 frozen config；非法 object/cycle/non-finite fail-loud；
- finalize：stable pointer 与 Host binding 之间注入同步失败/取消，二者、endpoint、catalog、admission 全回 old，
  candidate task/subscription/module 全零；
- terminate：listener、MessagePush registration、callback、HTTP/WS/gateway/task/module reference 全零。
- transition：一个 Core built-in/legacy callback 经过真实 Bus adapter，正常返回统一 Receipt，异常与旧 `PARTIAL`
  映射 `UNKNOWN`；旧 `FAILED` 分别覆盖可证明 pre-effect 的 `REJECTED` 与不可证明的 `UNKNOWN`，Bus retry/fallback
  次数为 0；证明迁移期不会绕过新终态合同。
- cleanup：old `stop()` 普通错误、partial stop、caller 二次取消与 `binding.close()` 失败均保留 cleanup-pending
  owner；显式 retry 前不能开放/替换，retry 后资源和 closeable 全零。

### E3 一次性组合 Gate

E3 把 Feishu `071278d518aea0ac80bcc76d9346e5bb02d93df1` 与 QQBot
`d9d105515db9e63f3639968fd488904f230be95b` 记录为 v2 base heads；两仓 v3 PR 完成后必须换成迁移后的 exact
40-SHA，base 不能作为完成证据。Gate 记录以下事件：

```text
candidate.register
candidate.factory_calls=0 credential_calls=0 provider_client_construction=0 provider_effects=0
legacy_channels_calls=0 sdk_client_construction=0 raw_credential_reads=0
candidate_write_set=validation_root_only

old_admission_closed → old_inflight_drained → old.stop → old_binding.close → service.switch
→ new.factory(formal) → new.bind → new.start.ready(closed)
→ stable_and_host_finalize → new_admission_open

inbound(message_id, snapshot_id, generation_id) → one terminal Turn
duplicate_inbound(message_id) → zero additional Turn
duplicate_stop(message_id) → one control effect + one awaited response
old_inbound.block → channel.swap → old_inbound.resume/respond
→ outbound.binding_token=old → old provider receipt
outbound(delivery_id, body)
→ logical_attempt_count(logical_delivery_id)=1
→ raw_effect(part_index, chunk, same_delivery_id)*
→ terminal receipt.delivery_id == envelope.delivery_id

preview(presentation_id, old_binding_token) → patch_effect_count=1
→ patch_after_effect_failure → incident + no_retry

rejected_attempt(delivery_id=1, logical_id=1, sequence=1)
→ explicit_new_attempt(delivery_id=2, logical_id=1, sequence=2)
unknown_attempt(delivery_id=3) → no_new_attempt

normal_terminal | error_terminal | cancelled_terminal
→ awaited ChannelDeliveryReceipt before Turn terminal

failure: new.start.fail → new.stop → new_binding.close → service.restore_old
→ old_exact_token.bind → old.start
→ stable/catalog/host=old → candidate resources=zero

failure: stable_and_host_finalize.fail_or_cancel
→ public current/catalog/host/endpoint/admission=old → candidate resources=zero

shutdown_at(BUS_QUEUED | LANE_QUEUED | RUNNING)
→ exact inbound lease closed once → no dropped owner

direct_push(old_binding_token) during swap
→ old queued/running owner → one terminal receipt → old drain may complete

control_after_effect_failure(/stop)
→ UNKNOWN → provider effect count=1 → no direct send/retry
```

Gate 使用 synthetic credential 与 recording HTTP/WS provider，不 monkeypatch正式网络，不读取正式凭据；核对
workspace config/plugin-data、Session DB、artifact/pointer 的 before/after digest。一个 E3 同时覆盖 candidate discard、
promote/reload、正向收发、provider-after-effect failure 与 terminate，不为每个 channel 启动独立完整 E2E。

一个 logical delivery 可产生多个 chunk raw effects，但全部共享 delivery id；任一已可能提交的 part 失败，
terminal 为 `UNKNOWN` 且不得开始第二个 logical attempt。Feishu live-card patch、interactive→text、QQBot
stream→ordinary 的 after-effect failure 都纳入该 oracle。

任何 candidate 读取 secret/访问正式 provider、stable 使用 candidate factory、provisional 对公开 consumer 可见、
inbound 丢失 exact lease、`UNKNOWN` 被盲重试、rollback 只改 pointer、stop 后仍有 owner，均停止交付。

## 6. 实现与删除顺序

1. C14a：typed Definition/Registry、manifest credential declaration/redaction、Root provider、snapshot identity 与 collision；
   不创建 adapter、不改 Bus、不承诺 attachment lease。
2. C14b：`CommittedChannelCatalog`、Manager-owned closed provisional、ChannelHost formal binding、formal-only credential
   resolution、text-only provider delivery、durable cleanup tombstone 与 critical retry；成功前不能移除 owner/closeables，
   且不引用 C14c 的 `OutboundEnvelope`/MessageBus DTO。
3. C14c：binding-token inbound/outbound envelope、Bus queued/running lease、三态 awaited receipt、Core built-in channel
   adapter 迁移与 exact drain。
4. C14d：control/turn-stream 窄 port；附件留给单独 persistence 合同。
5. Feishu、QQBot canonical v3 PR：第一批只做文本，领域单测 + Manager formal-generation test，不做插件级完整 E2E。
6. exact pair E3 后删除两个 external v2 shell；Core adapter zero-consumer Gate 后才物理删除 v2 channel public ABI、
   fixed contribution 与 live snapshot path。

Core 真实入口包括 `agent/plugins/manager.py`、`agent/plugins/snapshot.py`、`bootstrap/app.py`、
`bootstrap/channel_host.py`、`bootstrap/channels.py`、`agent/tools/message_push.py`、`bus/queue.py`、`bus/events.py`、
`infra/channels/contract.py`、`infra/channels/delivery.py`、`agent/looping/core.py`、`agent/turns/outbound.py` 与
`bootstrap/passive_worker.py`；还必须迁移 `agent/core/passive_turn.py`、`agent/lifecycle/phases/after_turn.py`、
`agent/turns/orchestrator.py` 的 normal/error/cancel/proactive outbound，以及
`infra/channels/telegram_channel.py`、`qq_channel.py`、`web_chat_channel.py`、`infra/mobile_realtime/channel.py` 的
Core adapter。zero-consumer scan 未覆盖这些入口前不得删除旧 channel/MessagePush callback。

V2 删除 inventory 同时覆盖 `RuntimeSnapshot.channels`、Host `_plugin_channels/ChannelSwap`、app endpoint switcher、
Manager endpoint signatures、bootstrap `plugin_channels`、旧 MessagePush channel registration、fire-and-forget
`publish_outbound`/入队即 SUCCESS 路径与 external `Plugin.channels()` contract/tests。

## 7. 回滚

Core 恢复点为 `19f2cca2`。插件各自保留 exact base。验证只使用一次性 workspace 与 recording/loopback channel；
不写 hua-home、不加载正式 Feishu/QQBot credential。
