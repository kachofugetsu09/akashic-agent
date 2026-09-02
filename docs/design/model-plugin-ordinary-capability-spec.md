# 模型普通插件与 Provider 组合规格

- 状态：accepted / implementing
- 日期：2026-08-29
- 关联需求：RUN-005～RUN-012、ONB-001、CTX-001、PLG-003、PLG-006、PLG-010、PLG-014、PLG-016、SEC-005、SEC-007、TST-001～TST-006
- 关联决策：[0027](../decisions/0027-runtime-models-use-generation-leases.md)、[0028](../decisions/0028-model-credentials-live-with-workspace-connections.md)、[0039](../decisions/0039-react-core-atoms-keep-sources-unprivileged.md)、[0050](../decisions/0050-model-revision-lives-in-ordinary-plugin.md)
- 现行实现设计：[运行时模型注册表与 Onboarding](runtime-model-registry-and-onboarding.md)
- 持久化边界：[持久化状态地图](persistence-state-map.md)

## 1. 结论

模型能力拆成一个普通 `models` 基础插件和若干普通 Provider 插件。Core 只提供插件组合、生命周期、快照、租约、路径和诊断原子，不认识 OpenAI、Codex、OpenCode Go、聊天模型、embedding、上下文长度或 reasoning effort。

`models` 插件是模型领域唯一 owner。它只有一份私有状态，却向不同调用者投影五个窄 Service facade：

```text
CHAT_MODELS       为一个执行冻结整组聊天模型，并按 role 取得绑定模型
EMBEDDINGS        执行 embedding
MODEL_CATALOG     只读查看 connection、model、默认值和可用状态
MODEL_SETTINGS    在用户授权的控制面请求中修改 connection、model 和 workspace 默认值
MODEL_DRIVERS     Provider 插件注册协议实现
```

这五个名字不是五套 manager，也不能把同一个 Python 对象原样 provide 五次。它们是共享一个私有 `ModelsState` 的五个不同 facade value：正常调用路径中，Akasha 持有的对象没有聊天方法，Onboarding 持有的对象没有写方法，Provider 持有的对象也没有其他 Connection。Provider 插件只通过 `MODEL_DRIVERS` 注册 driver，不提供第二套模型注册表，也不要求 Core 按 Provider 名称分支。仓库内置调用者只从公开插件 API 注入上述 Service，不 import `agent.provider`、`agent.model_runtime.*`、`bootstrap.providers` 或其他插件源码。

`builtin` 只表示随发行版安装和默认启用。每个模型插件必须通过外置源码、正式 install、冷启动和真实调用 Gate；不能外置安装的实现不属于普通插件。

[0050](../decisions/0050-model-revision-lives-in-ordinary-plugin.md) 已接受 owner 变化：Core 只拥有 exact runtime snapshot lease；`models` 插件拥有持久 revision，并在该 lease 内复制一次不可变执行绑定。revision 仍用于 SQLite CAS 和恢复，但不再拥有 lease、retired generation、manager 或第二套发布生命周期。

## 2. 用户意图与设计准则

用户希望 Akashic 最终由多个平等、非特权插件组合而成。模型插件化是第一条高价值纵向切片：普通 OpenAI-compatible 模型、Codex 登录和 OpenCode Go 登录可以独立增加或移除；ReAct、Akasha、Onboarding、Scheduler、Subagent 和 Wake 只依赖自己真正需要的能力。

本设计同时遵守两组约束：

1. 正交性：一个变化轴不迫使无关轴变化。增加 Provider 不修改 Core；切换默认模型不修改 ReAct；替换 embedding 不修改聊天模型；修改设置 UI 不修改 transport。
2. 概念完整性：所有扩展都使用同一套 `Service → inject → Effect → snapshot/lease` 语言。不得为模型另建第二套 generation、lease、依赖图、热更新或权限系统。

常用任务必须直接：

```text
Provider 作者：注册一个 driver
Turn/ReAct 调用者：在 admission 时冻结一组模型，然后按 role 调用
Akasha：绑定一个 embedding 模型，然后批量编码
Onboarding：读取 catalog，通过 settings 提交修改
```

如果一个普通任务必须同时操作 PluginManager、模型数据库、credential store、provider factory 和 generation，设计即未通过直接性检查。

## 3. 当前真实状态

### 3.1 当前启动顺序

`bootstrap/tools.py:build_core_runtime()` 先调用 `build_model_registry()`，取得 `default`、`fast`、`agent`、`vision` role provider，再构造工具和 `AgentLoop`；之后才创建 `PluginManager`。因此模型实现当前位于插件组合根之外，不能由普通插件提供或替换。

```text
Config
  ↓
bootstrap/providers.py
  ↓
ModelRegistry + RoleBoundProvider
  ↓
Tools + AgentLoop
  ↓
PluginManager
```

### 3.2 当前 Provider 耦合

`agent/provider.py:LLMProvider` 根据 `provider_name == "codex"` 选择 `CodexResponsesTransport`，否则选择 `ChatCompletionsRuntime`。同一实现还选择 DeepSeek、DashScope 和 OpenCode Go message/tool strategy。Core 因此同时知道模型领域、认证方式、wire transport 和厂商差异。

### 3.3 当前模型代际

`agent/model_runtime/registry.py:ModelRegistry` 已经提供有价值的现行语义：

- workspace 模型注册库 revision；
- immutable `ModelGeneration`；
- `ModelExecutionBinding`；
- default、fast、agent、vision role；
- Session 显式模型和 reasoning effort；
- active execution 保持旧 generation；
- 新执行读取新 revision。

本设计保留这些可观察语义，但把 owner 从 Core 实现迁到 `models` 普通插件。

### 3.4 当前内置插件使用方式

| 调用者 | 当前方式 | 问题 |
|---|---|---|
| 被动 ReAct / Turn | bootstrap 直接把 `RoleBoundProvider` 注入 `AgentLoop` | 不是插件 Service；启动顺序硬连线 |
| Scheduler | 注入 `SCOPED_TURNS`，由 Core Turn 间接使用模型 | 依赖方向正确；无需直接拿模型 |
| Subagent | 注入 `SCOPED_TURNS`，由 Core Turn 间接使用模型 | 依赖方向正确；无需直接拿模型 |
| Wake | 注入 `SCOPED_TURNS`，由 Core Turn 间接使用模型 | 依赖方向正确；无需直接拿模型 |
| Akasha | 注入 `TEXT_EMBEDDING_SETTINGS`，再自行构造 `memory2.Embedder` | 只注入配置，没有注入 embedding 执行资源；Akasha知道 URL、key 和 wire format |
| 设置与 Onboarding | `bootstrap/settings_api.py` 直接操作模型 store、认证和 provider probe | UI/API owner 与模型领域混合 |

除表中入口外，当前 `bootstrap/chat_api.py` 的模型目录、`agent/plugins/generation_job_host.py` 的 job provider/model scope、视觉工具、compaction、memory consolidation 和若干 bootstrap tool wiring 也直接捕获 `ModelRegistry`、role provider 或 `LLMProvider`。迁移不能只替换被动 ReAct；第 10.6 节列出完整目标映射。

### 3.5 已确认事实、推断与未知

已确认事实：

- v3 插件已经使用 `ServiceKey`、`inject`、`Context.provide()`、`Effect`、Root/Fiber、candidate/stable snapshot 和 lease。
- `CHANNELS`、`SCOPED_TURNS`、`UI_SLOTS`、`TOOL_CATALOG` 等已经证明“Core 提供 registry，插件注册 contribution”的组合形状可行。
- Scheduler、Subagent 和 Wake 不需要直接选择模型；它们通过 `SCOPED_TURNS` 使用同一 Turn 执行路径。
- Akasha 当前只获得 embedding 设置，实际 HTTP embedding 仍由 Akasha内部构造。

设计推断：

- 模型 driver 可以使用与 Channel contribution 相同的 candidate-local 注册、freeze、publish、drain 结构。
- 当前 `ModelRegistry` 的可观察冻结语义可以由 exact plugin snapshot lease 内的一次不可变数据复制承载；不需要迁入第二套 revision lease。

未知边界：

- 第一阶段是否保留现有 workspace `model-registry.sqlite3` 路径，还是迁到 plugin-data。本文选择保留现有权威路径，避免把数据迁移和代码插件化绑成一次变更；未来搬迁需要独立审批和恢复合同。
- Provider 网络访问是否立即改走通用 Core Network Service。当前插件没有完整网络沙箱；本文不为模型单独发明网络权限系统。
- `fast`、`vision` role 是否在后续产品设计中保留。第一阶段为行为等价继续保留，不借插件化删除角色。
- 当前 `UI_SLOTS` 只实现 Mobile UI slot，不支持 2236 顶部导航或完整模型页面。Web 插件化是独立设计；不得让模型执行切片等待它，也不得把当前 Mobile slot 描述成已经具备该能力。

## 4. 最少概念

### 4.1 组合概念

模型插件不得增加独立插件框架，只使用现有四个组合概念：

| 概念 | 唯一含义 |
|---|---|
| `Service` | 一个可注入的 typed capability |
| `inject` | 插件激活所需的 Service 集合 |
| `Effect` | 当前 Fiber/generation 拥有、卸载时撤销的注册或资源 |
| `snapshot/lease` | 一次执行观察的不可变插件世界及其存活保证 |

权限、health、diagnostics 和路径是 Context 提供的属性或窄句柄，不建立第二套模型组合语言。

### 4.2 模型领域概念

模型领域只保留四个持久事实：

| 对象 | 含义 |
|---|---|
| `Connection` | 一套具名 driver 连接、认证引用、endpoint 和已校验 driver config |
| `Model` | Connection 下一个可选聊天或 embedding 模型及能力快照 |
| `Binding` | workspace role/default 或 Session 对 Model 的选择 |
| `Revision` | 一次成功模型配置事务产生的单调 CAS/恢复版本；不是运行时 lease |

`Connection` 复用现有 `provider` 列作为稳定 `driver_id`，并只在首个真实 driver 需要公共字段无法表达的配置时增加一个不含 secret 的 `driver_config_json`。driver contract version 属于已安装 artifact/诊断；config format version 放在该 JSON 内，不各建一列或一张表。`Provider` 是插件/driver 类型，不是第五个持久对象。`Runtime` 是现行兼容名，迁移完成后公共 API 使用稳定 `model_id`，不把显示名当身份。

### 4.3 执行概念

一次执行只增加一个临时对象：

```text
ModelExecution
```

`ModelExecution` 表示 exact plugin snapshot 内一次复制出来的完整 chat role set 与模型配置。`BoundChatModel` 是它投影出的 typed operation view，不是独立 owner、lease 或持久对象。Embedding 通过独立的 `EMBEDDINGS` facade 绑定；若当前 task 已有 `ModelExecution`，它复用同一份 frozen snapshot，而不把 embedding 塞进 chat API。

## 5. 目标结构

```text
                           Core
┌──────────────────────────────────────────────────────────────┐
│ Service · inject · Effect · Root/Fiber · snapshot/lease      │
│ plugin paths · health · diagnostics · generic permissions    │
└──────────────────────────────┬───────────────────────────────┘
                               │ ordinary v3 plugin
                               ▼
                         models plugin
┌──────────────────────────────────────────────────────────────┐
│ 一个私有 ModelsState · Connection · Model · Binding · Revision │
│ CHAT_MODELS · EMBEDDINGS                                     │
│ MODEL_CATALOG · MODEL_SETTINGS · MODEL_DRIVERS               │
└──────────────▲───────────────────────────────┬───────────────┘
               │ driver registration          │ bound resource
               │                              │
     ┌─────────┴─────────┐          ┌─────────┴────────────────┐
     │ Provider plugins   │          │ Consumers                │
     │ openai-compatible  │          │ Turn runtime / ReAct     │
     │ codex              │          │ Akasha                   │
     │ opencode-go        │          │ Onboarding / Model UI    │
     └────────────────────┘          └──────────────────────────┘
```

所有插件处于同一 Root/Fiber 模型。Core 不把 `models` 或任何 Provider ID 放进特判 allowlist。产品发行清单可以把 `models` 标记为默认安装或聊天 readiness 所需，但这只是安装策略，不增加权限。

## 6. Core 最小原子与禁止事项

### 6.1 模型切片只需要 Core 的两个原子

1. `CompositionContext`：Fiber 身份/config、typed `provide/require/inject`、可撤销 `Effect`、声明式 workspace file、窄访问模式与 diagnostics 投影。`PluginRuntime` 只是 `ctx.runtime` 的数据，不是第三个原子。
2. `RuntimeSnapshotLease`：candidate/stable publication、exact snapshot 存活保证，以及从该 snapshot 的 Root 读取 Service。它们是一个子系统，不拆成三个模型原子。

实现只补两个通用组合不变量，不增加模型原子：candidate 重建沿明确 `inject` 与冻结 topology 取得完整双向连通 component，避免只重建 provider 或 consumer 的半个注册表；Root 在全部插件 mount/readiness 完成、snapshot compile 前发送一次通用 `SNAPSHOT_SEALING` 串行事件，让 contribution owner 冻结私有 registry。`models` 使用该事件冻结 driver，不要求 `RuntimeSnapshotCompiler` 识别模型。`Context.get()` 仍是即时可选查询，不声明 activation 或热更新依赖；需要随 Service 安装、升级重建的插件必须显式 `inject`。

每个无父 lease 的 HTTP、Mobile 和设置 request boundary 先通过现有 `RuntimeSnapshotStore.acquire()` 取得 current generic lease，绑定 owner task 后才从 exact Root 读取 Service。有父 lease 的 `CHAT_MODELS.execution()` 直接通过通用 `agent.plugins.snapshot.lease_current_runtime_snapshot()` fork 当前 exact lease；模型契约不重导出第二个 lease API。没有当前 task binding 时，调用者必须按该 operation 的错误语义 fail-loud，不能自行读取 current。普通插件自行创建的 timer/worker 若要调用其他插件 Service，只能用 Core 的 `ctx.runtime_scope()` 给一次短操作绑定该插件所在的 exact Root；不得让长期 task 持有 Root lease。事件转交给异步 worker 时，在同步 listener 内 fork source lease，并由 worker 在 `finally` 中释放。三条路径都复用同一 snapshot lease，不新增 model lease、model acquire helper 或 `lease.require()`。

Core 不再为 plugin snapshot 与 model revision 增加共同 fence 或 ordered operation。删掉它成立的前提是更简单、也更严格的 driver 演进合同：

1. Connection 的首轮公共字段只有既有 `provider`（语义上就是 `driver_id`）、endpoint、auth identity、credential payload 和可选非敏感 `driver_config_json`。
2. 新 driver artifact 必须读取该 `provider` 过去所有已 committed connection config 和 credential payload formats；不能兼容时不得 promotion。
3. 两类格式都只能 expand：先发布同时读旧/新格式的 driver，再以显式可备份事务迁移。本规格不允许停止读取任何曾经 committed 的 config 或 credential format；删除旧 reader 需要另立能排空旧 settings/auth writer 的迁移协议。
4. settings 操作持有其 exact snapshot lease 完成 probe，再以现有 expected revision 做 SQLite CAS；promotion 后旧 operation 仍可提交旧格式，因为新 driver 已承诺可读。
5. admission 先租 exact plugin snapshot，再用一个 SQLite read transaction 复制完整 revision。它不需要把两者锁成同一个 generation；driver 兼容合同保证组合可读。

Provider 完整卸载与 settings 竞态最多使 Connection 进入 `driver unavailable`；它保留数据并 fail-loud，不产生错误 transport。真实 probe、OAuth 等待和其他网络 I/O 因此也不进入任何全局发布锁。

用户写操作继续由 authenticated settings/control host 拥有认证、同源/CSRF 和请求生命周期。`MODEL_SETTINGS` 不再接收一个改名后的 grant；同进程 Service facade 是 API/拓扑边界，不伪装成恶意插件 sandbox。未来若隔离不可信插件，应另立通用组合权限设计。Web navigation/data/action contribution 属于独立 UI 规格，不是本切片前置。

### 6.2 Core 不提供

1. 不提供 `LLMProvider` 单例或全局 provider locator。
2. 不保存模型能力对照表。
3. 不识别 Provider 名称、认证类型或 Base URL。
4. 不选择默认聊天或 embedding 模型。
5. 不解析 reasoning effort、tool call、vision 或 embedding dimension。
6. 不刷新 Codex/OpenCode token。
7. 不为某个 Provider 添加 bootstrap 分支。
8. 不把模型设置写进 `config.toml` 作为运行时传播手段。
9. 不创建 models 预启动 Root、模型专用 PluginManager、模型专用 publication lock、model revision lease 或按 plugin ID 选择 Root。

## 7. `models` 插件的一份状态与五个操作 facade

公共 key 和协议首先放入现有 `agent.plugin_composition` facade；它们是稳定扩展合同，不让 Core 获得模型 owner。外部插件只 import 该 facade，具体实现留在可外置的 `models` artifact 中。五个 ServiceKey 由五个小 facade 对象提供；它们共享一个私有 `ModelsState`，不复制数据库、registry、锁或 lifecycle。未来把协议移到独立 SDK distribution 只是包管理问题，不改变 Service 合同。

```text
ModelsState（仅 models artifact 内可见）
├─ ChatModelsView
├─ EmbeddingsView
├─ ModelCatalogView
├─ ModelSettingsView
└─ ModelDriversView
```

验收必须检查 facade 的实际 Python 表面：`EmbeddingsView` 没有 `execution`，`ModelCatalogView` 没有 `apply`，`ModelDriversView` 没有 catalog、settings 或执行方法。这些 facade 让正常调用者只持有所需操作，不是同进程安全 sandbox。现有 `Context.get()` 允许插件显式查询未列入 `inject` 的可选 Service；`inject` 只拥有 activation 语义，不是权限声明。恶意或主动绕过自身声明的插件隔离属于独立通用 sandbox 问题，不能靠模型 API 伪装解决。

### 7.1 `CHAT_MODELS`

职责：在 Turn/job admission 时，根据 workspace role、显式 model ID 和 reasoning effort，从 exact snapshot 内一次复制完整 chat role set，再投影不可变聊天模型 operation view。

```python
CHAT_MODELS = ServiceKey[ChatModels]("models.chat.v1")

class ChatModels(Protocol):
    def execution(
        self,
        *,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncContextManager[ModelExecution]: ...

class ModelExecution(Protocol):
    def chat(self, role: ModelRole) -> BoundChatModel: ...

class BoundChatModel(Protocol):
    @property
    def descriptor(self) -> BoundModelDescriptor: ...

    async def complete(self, request: ModelRequest) -> LLMResponse: ...
```

`execution()` 在 admission 时租住 exact stable snapshot，并在一个 SQLite read transaction 中复制 revision、default、fast、agent 和 vision 的完整映射。显式 Session 选择只按现行规则覆盖 default/agent，不改变其他 role。`fallback` 继续是 provider/role policy，不在没有独立调用者时升格为第五个公开 role。退出前只持有 plugin snapshot lease；revision 是 binding descriptor 上的值，不计数、不 retire。

同一个 Turn、job 或 scoped work 只能建立一个 `ModelExecution`。compaction、vision、summary 和 ReAct 的所有请求都从它按 role 取得模型；不同 role 合法，不建立嵌套 generation。嵌套执行只有复用同一个 execution object 时允许；尝试在同一执行中重新读取 current 或改变 selection 必须 fail-loud。

`ModelRequest`/`LLMResponse` 不是新建的第二套 DTO：迁移现有 `agent.model_runtime.types` 合同到公开 facade，并让它成为唯一 provider-neutral request/response vocabulary。公开 `ModelRequest` 不再允许调用者传 model、Base URL、API Key、provider 名、transport flavor 或 provider `extra_body`；这些由 bound model 与 driver 拥有。Adapter 独自负责公共 DTO 与 wire payload 的转换。

跨请求状态只通过同一个 `ModelContinuation(binding_id, payload)` 在 response 与下一次 request 之间原样透传。`binding_id` 必须等于接收请求的 `BoundModelDescriptor.binding_id`；错配由 bound driver 在任何外部 I/O 前以 `ModelUnavailableError` fail-loud。Core 不读取或改写 payload，payload 必须是无循环且不含非有限浮点的严格 JSON 数据。

现有 ReAct 真正需要的 token 估算留在 bound operation：`estimate_context_tokens()`、批量 `estimate_appended_message_tokens()` 与 `max_tool_schemas`。它们由 driver 的实际 wire tokenizer/profile 提供，不作为 workspace capability snapshot 新增长期字段。`ModelRequest.disable_reasoning` 只抑制一次调用，不改变 execution 的 reasoning binding。

### 7.2 `EMBEDDINGS`

职责：根据显式 model ID 或 workspace 默认值创建不可变 embedding 绑定。

```python
EMBEDDINGS = ServiceKey[Embeddings]("models.embeddings.v1")

class Embeddings(Protocol):
    def describe(
        self,
        *,
        model_id: str | None = None,
    ) -> EmbeddingSpaceDescriptor: ...

    def bind(
        self,
        *,
        model_id: str | None = None,
    ) -> AsyncContextManager[BoundEmbeddingModel]: ...

class BoundEmbeddingModel(Protocol):
    @property
    def descriptor(self) -> EmbeddingSpaceDescriptor: ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...
```

`describe()` 只读取当前 models revision 的配置和已封印 driver 定义，不读取 credential、不打开网络，也不拥有 lease 或第二份 identity 算法。它让 Akasha 在构造 kernel 前审计既有 sparse index。`bind()` 是 embedding 的唯一执行入口：当前 task 有 `ModelExecution` 时复用其 frozen snapshot；没有时必须已处于 exact runtime scope，再建立短命 embedding binding。Akasha 只取得 `EMBEDDINGS`，不取得完整 chat execution；Turn、post-commit worker 和 Wake maintenance 都沿用各自已有或短命的 generic runtime scope。ContextVar 只是 models 插件内部的 snapshot 传播，不是第二个公共参数；继承到子 task 不构成授权，owner task 不同必须 fail-loud。

`EmbeddingSpaceDescriptor` 至少包含 driver identity、model ID、dimensions、normalization 和 schema version；这些字段共同决定 embedding space identity，不另造一个 owner 类型。默认 embedding 改变时产生新 space；不得把新旧向量静默写入同一索引空间。

首版继续把 dimensions 写入现有列，同时把完整 capability/source snapshot 写入 additive JSON；normalization 因此可以原样持久化。space identity 还包含 connection fingerprint 与 capability digest，因此 endpoint、driver config、normalization 或维度证据变化不会复用旧索引。

### 7.3 `MODEL_CATALOG`

职责：只读查询，不返回 secret。

```python
MODEL_CATALOG = ServiceKey[ModelCatalog]("models.catalog.v1")

class ModelCatalog(Protocol):
    def snapshot(self) -> ModelCatalogSnapshot: ...
    def validate_chat_selection(
        self, selection: ChatModelSelection
    ) -> ChatModelSelection: ...
```

Snapshot 包含 revision、连接、模型、默认 binding、capability source 和 availability，并自行提供同一 revision 内的 lookup。Service 不再提供 `connection(id)`/`model(id)` 便利方法，避免一个请求混读不同 revision。Snapshot 不返回 API Key、access token、refresh token 或 credential payload。Session/Turn owner 使用 `validate_chat_selection()` 做纯校验，随后仍由 Session owner 写 `sessions.metadata`；`models` 不取得 Session write surface。

### 7.4 `MODEL_SETTINGS`

职责：执行模型领域写事务。它与 `MODEL_CATALOG` 分离，使只读 UI、Dashboard 和 Onboarding 的正常调用路径只持有查询方法面，不直接耦合模型库写入口。

```python
MODEL_SETTINGS = ServiceKey[ModelSettings]("models.settings.v1")

class ModelSettings(Protocol):
    async def apply(self, command: ModelChange) -> SettingsReceipt: ...
```

`ModelChange` 是 `AddConnection | UpdateConnection | DisableConnection | AddModel | CreateConnectionWithModel | SetDefaultModel | SyncModels | StartConnectionAuth | FinishConnectionAuth | CancelConnectionAuth` 的闭合 typed union；持久写命令携带 expected revision。`CreateConnectionWithModel` 只表达“新连接及其首个手工模型”这一项必须共同成功的用户动作：它在 SQLite transaction 外完成 connection probe、driver open 和 model/embedding probe，再用一次 revision CAS 与 transaction 同时写入两者，避免失败留下孤立 credential Connection；它不是通用 batch DSL。`SyncModels` 只给出 Connection ID：`models` 在 SQLite transaction 外调用该 Connection driver 的 `discover`，再在同一个 `BEGIN IMMEDIATE` 中校验 expected revision、比较整批标准化证据并写入。Driver 不选择持久 model ID；`models` 保留已有 ID，并为新证据生成无歧义的稳定 ID。Capability JSON 中的 store-owned `source=discovery` 只区分目录拥有的行与手工行：刷新可更新 discovery 行并禁用本轮消失的 discovery 行，但绝不覆盖或禁用同 wire identity 的手工行；规范化结果没有变化时不备份、不增加 revision。三种 auth 命令覆盖 Codex/OpenCode 的 begin → poll/complete → credential commit，但不把 Provider wire 字段暴露给调用者：driver definition 提供 handler，settings facade 返回 provider-neutral attempt/challenge/result。短命 auth attempt 可在重启后明确失败，不新增第六个 Service。

auth attempt ID 由 `models` 生成。Driver 的私有 attempt state 只留在内存；公开 receipt 只投影其 `challenge`。完成结果使用统一 connection 字段（name、endpoint、auth identity、credential、driver config），由 `models` 在同一个 expected-revision CAS 中新增或更新 Connection；credential 不经 UI 回传。

网络登录、长轮询和 probe 在 SQLite write transaction 外运行；最终 credential/Connection 写入使用 expected revision CAS。认证、同源与 CSRF 由调用它的 control host 负责。`models` 插件集中完成结构校验、领域规则、真实 probe、operation backup、SQLite transaction 和 revision commit。UI 与 Onboarding 不直接写数据库。

Session 模型选择不属于 `MODEL_SETTINGS`。Chat/Session control owner 先调用 `MODEL_CATALOG.validate_chat_selection()`，再按现有 Session admission 与持久化合同写入或清除 `sessions.metadata.model_selection`。

Connection 的物理删除不包含在首版日常 API；禁用与显式独立删除保持不同命令。

### 7.5 `MODEL_DRIVERS`

职责：让 Provider 插件在 candidate Root 内注册 driver definition。

```python
MODEL_DRIVERS = ServiceKey[ModelDrivers]("models.drivers.v1")

class ModelDrivers(Protocol):
    async def register(
        self,
        ctx: Context,
        definition: ModelDriverDefinition,
    ) -> Effect: ...
```

`ModelDriverDefinition` 包含：

- stable `driver_id` 和 driver contract version；plugin artifact/generation 版本仍由现有 snapshot 拥有；
- `open(connection, credential_handle)`：校验不含 secret payload 的 Connection snapshot，并返回该 driver 的 chat/embedding operation；
- 可选的 provider-neutral `discover`、`probe`、`start_auth`、`finish_auth`、`cancel_auth` handler。`discover` 只返回 `DiscoveredModel`（kind、wire model、capability/source、default reasoning 与 model-level driver config），不返回 store-owned model ID、Connection ID、availability 或 revision。

不再平行声明“支持能力子集”、connection/credential schema、format range、capability mapper、health probe 或设置 UI metadata。handler 是否存在就是 capability；`open()`/`probe()` 是 config 与健康边界；discover/probe 返回已经标准化且携带 source 的 capability evidence。OpenAI-compatible 默认用 `/models` 做真实 connection probe；确实没有目录端点的网关只有在用户显式设置 `allow_unverified_manual=true` 后才允许 config/credential-shape admission，并把“首次模型调用才验证 endpoint/key”作为可见降级，不能静默假装 probe 成功。credential payload 只通过窄 `CredentialHandle` 进入 handler；Connection/Model driver config 使用明确 allowlist，不提供任意 header 或 extra body secret surface。Web contribution 属于独立规格。

Registry 在 candidate apply 阶段开放注册；candidate 完成前冻结。重复 driver ID、缺失声明能力，或者 candidate 仍声明某个 `driver_id` 却无法读取该 driver 的 enabled Connection contract/config schema 时拒绝发布。完整缺失某个 `driver_id` 是正常 Provider 卸载：candidate 可以发布，`models` 把其 Connection/Model 投影为 unavailable、禁止新 execution，并保留持久数据与旧 lease。不得使用全局可变 registry 或运行中后补注册。

`register()` 必须像现有 Channel/MCP registrar 一样绑定当前 Root、当前 Fiber 和返回的 Effect，并在 freeze 后保存 owner provenance。跨 Root Context、重复 driver ID、Effect 已释放后的 replace，以及 freeze 后首次注册都 fail-loud。

Driver upgrade 不在 promotion 中改写 Connection。Candidate 对 enabled Connection 调用 `open()` 做纯读取兼容验证；不能读取时拒绝。需要升级持久 config 时，由显式、可备份的 settings migration transaction 完成，不能把 plugin promotion 与数据库迁移伪装成一个可回滚动作。

## 8. Model Driver 合同

### 8.1 Driver 不拥有的事实

Provider driver 不拥有：

- workspace 默认模型；
- Session 模型选择；
- model registry revision；
- Turn 和 Session；
- Akasha embedding index；
- PluginManager 或 snapshot publication；
- 其他 Provider 的 connection 或 credential。

### 8.2 Driver 拥有的行为

Provider driver 拥有：

- 外部认证和 token refresh；
- endpoint/wire transport；
- 请求转换、stream 解码和响应转换；
- provider 内部 retry 边界；
- provider 错误到公共错误类型的窄映射；
- model discovery 和真实 probe；
- provider-owned capability evidence；
- usage 原始字段解析。

`models` 插件拥有统一请求/响应 DTO、selection、binding、normalized capability schema、usage 汇总和 registry revision。Core 不参与这两者的领域分工。

### 8.3 Credential 边界

Credential 随 Connection 由 `models` 插件保存，延续 0028。Driver factory 获得只绑定一个 Connection/auth identity 的 `CredentialHandle`，只能读取或原位刷新该 identity 的 credential payload。它不能枚举其他 Connection，也不能取得 credential store。

需要 token refresh 的 Driver 必须把完整的 `read → outbound refresh → refresh` 放在 `CredentialHandle.exclusive()` 内；这个锁按 registry 跨进程排他，等待期间可取消，并在取消或异常时释放。这样并发进程不会用同一 refresh token 重复刷新，也不需要 Core 认识任何认证协议。

`ModelExecution` 冻结 connection ID、auth identity、driver artifact、model、capability 和 role binding，不复制一个永远不变的 token payload。Credential refresh 是同一 Connection/auth identity 的 operational update，不产生 credential generation，也不进入 model binding identity；同一 execution 的下一次 outbound request 从原 handle 读取刷新后的 payload。改变 endpoint、auth identity 或 API Key 选择的用户设置事务增加 model revision；同 auth identity 的 token refresh 不改变当前模型绑定。刷新失败保留明确认证错误，不切换 identity 或 Provider。

日志、health、catalog、receipt 和 exception 不得包含 secret。候选 probe 使用临时 credential handle；失败候选不得把 token 或 connection 写入正式库。

### 8.4 三个首批 Provider 插件

| 插件 | 提供 | 需要 |
|---|---|---|
| `openai-compatible` | API Key、Base URL、Chat Completions、embedding、stream、tool call、usage、`/models` | `MODEL_DRIVERS` |
| `codex` | ChatGPT/Codex 登录、refresh、Responses transport、Codex catalog、headers、reasoning/usage | `MODEL_DRIVERS` |
| `opencode-go` | 本机登录或 API Key、catalog、reasoning effort、OpenCode Go message/tool profile | `MODEL_DRIVERS`；只有形成第二个真实消费者时才抽取窄 transport Service |

Codex 不能只做 auth 插件而把 Codex transport 留在 Core。否则 Core 仍然需要识别 Codex。OpenCode Go 不得 import `openai-compatible` 插件源码；若两者确有稳定共享边界，后续通过公开窄 Service 组合。

## 9. 能力快照归属

模型能力字段属于 `models` 插件的公共领域 schema：

```text
context_window
max_output_tokens
input_modalities
tool_calls
parallel_tool_calls
reasoning_efforts
embedding_dimensions
normalization
```

字段值来源优先级延续现行设计：用户显式覆盖、Provider 权威目录/probe、固定版本本地 registry、unknown。Provider 插件提交原始证据和来源，`models` 插件统一、校验并保存 capability snapshot。

Unknown 保持 unknown。未知不等于零、不支持或 false。Core 不保存这些字段，也不依据字段选择执行分支。

## 10. 仓库内置插件的引用方式

### 10.1 公共 import 边界

允许：

```python
from agent.plugin_composition import CHAT_MODELS, EMBEDDINGS
from agent.plugin_composition import MODEL_CATALOG, MODEL_SETTINGS, MODEL_DRIVERS
```

禁止：

```python
from agent.provider import LLMProvider
from agent.model_runtime.registry import ModelRegistry
from bootstrap.providers import build_model_registry
from plugins.models import ModelsRuntime
from plugins.openai_compatible import OpenAIClient
```

普通插件可以 import 主机正式发布的公共 plugin API；不能 import 主机或兄弟插件实现。

### 10.2 Turn runtime / ReAct

Turn runtime 注入 `CHAT_MODELS`，在 admission 时根据 inbound/Session selection 建立一个完整 `ModelExecution`，再把 agent role 的 `BoundChatModel` 作为参数交给 ReAct；compaction、vision 和 summary 从同一 execution 取得各自 role，fallback 留在已有 policy：

```text
Turn admission
  → CHAT_MODELS.execution(selection)
  → execution.chat(agent)
  → react(messages, bound_agent_model, granted_tools, limits)
  → compaction/vision 使用同一 execution 的其他 role
  → terminal/cleanup
```

ReAct 不读取模型数据库、不选择 default、不持有全局 model service。Scheduler、Subagent 和 Wake 继续只注入 `SCOPED_TURNS`；它们不得复制选择或 binding 逻辑。

### 10.3 Akasha

Akasha 把 `TEXT_EMBEDDING_SETTINGS` 替换为 `EMBEDDINGS`。Prompt、Tool、post-commit 和 Wake semantic scoring 都由 Akasha 的窄 adapter 在调用边界建立 embedding binding；已有 Turn 复用当前 generic runtime scope，detached post-commit 携带 source scope，Wake timer 用 `ctx.runtime_scope()` 建立一次短 scope：

```python
inject = (COMMANDS, TOOL_CATALOG, UI_SLOTS, EMBEDDINGS, INTERACTION_UNDO)

embeddings = ctx.require(EMBEDDINGS)
async with embeddings.bind(model_id=pinned_descriptor.model_id) as bound_embedding:
    result = await bound_embedding.embed(texts)
```

Akasha 只保存从 `EmbeddingSpaceDescriptor` 导出的 space identity 和向量，不读取 Base URL、API Key 或 provider 名，不自行创建通用 HTTP Embedder。默认空间、connection 或 driver 改变后，现有 kernel 在任何新读写前进入 optional health degradation，Prompt/Wake 不因此打断聊天，也不继续写旧空间。

旧 sidecar 或新默认空间的恢复由 artifact 自己拥有：`/akasha_reindex confirm` 只原子记录当前 descriptor 的显式请求；下一次 `runtime.started` 只创建 artifact-owned worker，formal Root 成为 current 且可租用后再开始修复。worker 先备份 `sessions.db` 和两份 sidecar，每个远程 embedding batch 单独取得并释放一次 exact-Root scope，再审计完整性、构建候选 sidecar 并发布。请求只在最终 kernel 重新通过 identity 检查后删除；失败或 Root 退役保留请求与备份并维持可观察降级。已经发布的历史 Yoyo ID 和既有 Akasha-import helper 继续可重放；helper 只可为已删除的 Core Config 做窄兼容读取改写，不得增加新的 import edge 或执行入口。fresh workspace 与新修复统一走 artifact-owned repair。

### 10.4 Scheduler、Subagent 与 Wake

三者不直接注入 `CHAT_MODELS`：

```text
Scheduler ─┐
Subagent  ─┼─▶ SCOPED_TURNS ─▶ Turn runtime ─▶ CHAT_MODELS
Wake      ─┘
```

它们只拥有各自的 Timer、admission、Tool grant、Session policy、delivery 和 settlement。通用 Turn owner 只请求公开、稳定的 consumer role，并持有 exact plugin snapshot lease；role 到 model 的 binding 和完整 `ModelExecution` 由 `models` 插件冻结。

### 10.5 Onboarding 与模型 UI

Onboarding 注入 `MODEL_CATALOG` 判断是否具备可用默认聊天模型和所需 embedding。用户配置仍通过现有 authenticated settings control host 调用 `MODEL_SETTINGS.apply()`；Onboarding 不获得授权 action 的创建能力。

2236 顶部模型页、Provider 面板动态注册和未来 Dashboard 平凡化属于独立通用 Web contribution 规格。本模型切片只保证 catalog/settings 是来源无关 API，因此将来页面无需修改模型运行时；它不把尚不存在的 Web 原语伪装成模型插件前置。

本轮只要求 server/control 的 catalog/settings API 不按 Provider ID 分支，并让已有 Connection/Model 在 driver 缺失时显示 `driver unavailable`。Provider 安装状态驱动的入口增减、可创建连接类型和无连接时的动态认证面板，等待通用 Web contribution 规格定义来源无关投影后验收；当前前端可以继续显示既有三个入口，但不能把它描述成已完成的动态插件 UI。

### 10.6 当前直接消费者的目标映射

| 当前消费者 | 目标引用 | 解析时机 |
|---|---|---|
| `AgentLoop` / passive Turn / control execution | exact snapshot 的 `CHAT_MODELS.execution()` | Turn admission |
| `bootstrap/chat_api.py` 模型列表与 Chat picker | exact request snapshot 的 `MODEL_CATALOG`；Session 写仍由 Chat/Session owner | 每次请求 |
| `bootstrap/settings_api.py` | 现有 authenticated control boundary → `MODEL_SETTINGS.apply()` | 每个用户控制请求 |
| `BackgroundJobActivityAdapter` 和 plugin job | exact job snapshot 的 `CHAT_MODELS` | job 真正开始时，不在 host 构造时 |
| compaction / Markdown profile projection | exact execution snapshot 的 `CHAT_MODELS`，显式 role | 各自执行单元开始时 |
| vision/read-image 工具 | exact Turn snapshot 的 `CHAT_MODELS`，显式 vision role | 工具调用开始且继承父 Turn lease |
| Akasha online/rebuild | `EMBEDDINGS` | 每个 embedding batch/完整 rebuild scope 开始时 |
| Scheduler / Subagent / Wake | 继续只用 `SCOPED_TURNS` | 由 Turn runtime 间接解析 |
| setup wizard / 无模型壳 | 通用 Plugin Installer；模型配置暂沿用现有 settings surface | 不创建临时 Core provider |
| `bootstrap/app.py` Mobile binding | 不再接收 registry；Mobile handler 每请求从 exact UI/control Service view 读取 catalog | Mobile command admission |
| `infra/mobile_realtime/channel.py` model catalog | exact request snapshot 的 `MODEL_CATALOG` | list/refresh command 开始时 |
| `agent/config.py` | 静态 Config 不读取模型库、不派生 LLM runtime；只保留非模型启动配置 | Config load |
| `main.py` / `bootstrap/app.py` model reload | `MODEL_SETTINGS` receipt；删除 `reload_model_config()` 直达 registry | 用户设置事务 |

这些 host 只持有 generic exact snapshot Service view 或公开 Service protocol，不持有 `ModelRegistry`、`LLMProvider`、provider factory 或 plugin ID。

### 10.7 无模型启动与唯一 Root

目标启动顺序解决当前构造环：

```text
Supervisor 2236 管理壳 + generic Plugin Installer
  ↓
Session/ConversationRuntime 骨架（尚不捕获 provider）
  ↓
PluginManager 绑定该 runtime
  ↓
构建一个 candidate Root
  ├─ Core generic services
  ├─ models / Provider plugins
  └─ SCOPED_TURNS（引用 runtime 骨架）
  ↓
原子发布 committed Root
  ↓
Turn/job admission 在 snapshot publication critical section 内
  ├─ 捕获 current exact snapshot lease
  └─ 通过 CHAT_MODELS 读取一个 SQLite snapshot 并建立 ModelExecution
  ↓
executor 只使用这组已经共同冻结的 services/models
```

`ConversationRuntime` 可以在插件之前构造，因为它只拥有 Turn admission/terminal，不在构造时需要模型。`SCOPED_TURNS` 可以引用该 runtime，因为任何执行只允许在 Root committed 后开始。不得创建 models 预启动 Root，或让 bootstrap 按 `models` ID 取实例。

没有安装 `models` 时，2236 管理壳仍可启动，聊天 readiness 报告缺少 `models.chat.v1`。未来 Web contribution 完成后，模型页面是否出现只由相应 contribution 决定；本规格不要求 Core 按 `models` plugin ID 分支。

## 11. 代际、并发与取消

### 11.1 组合身份

一次 bound model 记录：

```text
plugin_snapshot_id
model_revision
driver_id + driver_contract_version
connection_id + auth_id
model_id
capability_digest
```

同 auth identity 的 token refresh 不产生代际，也不改变 binding。

Plugin install/upgrade 改变 plugin snapshot；模型设置事务改变 model revision。这是两个独立变化轴。`ModelExecution` 必须记录精确组合，但只租住 plugin snapshot：snapshot 已唯一指向 driver owner generation 集，revision 只是被复制的持久版本。

### 11.2 原子可见性

- Provider candidate 先在隔离 Root 注册并完成 probe；发布前 stable 调用者不可见。
- 模型设置在一个 SQLite 事务中提交完整 Connection/Model/Binding/revision；调用者不可见半事务。
- active binding 在一次 ReAct 的所有 provider request、retry、tool batch、summary 和 compaction 中保持不变。
- 下一次执行读取最新 committed plugin snapshot 与 model revision。

### 11.3 跨 snapshot/revision 的兼容合同

Plugin snapshot 和 model revision 是两个正交变化轴，不强行合成一个 generation，也不增加共同锁：

1. execution 先租 exact plugin snapshot，再在一个 SQLite read transaction 中复制完整 revision；之后不再读取 current。
2. settings operation 持有 exact snapshot lease 完成 probe，再用 expected revision CAS 提交完整事务。即使随后 promotion，提交格式仍被新 driver 的向后兼容合同覆盖。
3. candidate 必须证明自己能读取所有 enabled Connection 的已 committed 公共字段、历史 config formats 和 credential payload formats；完整缺失的 driver 只投影 unavailable。
4. 旧 snapshot 中已经开始的 settings/auth operation 可能在 promotion 后才 CAS commit，因此 config 与 credential 都只允许“兼容 reader → 显式数据迁移”，不允许删除旧 reader。未来若要 contract，必须先有独立 writer-quiescence 规格。
5. embedding batch/rebuild 建立自己的短命 binding；Turn 内 embedding 复用父 execution 的 frozen snapshot。

并发 oracle 必须覆盖 settings update 与兼容 Provider promotion、settings update 与 uninstall、execution admission 与 promotion，以及进程在 probe、SQLite commit、pointer swap 三个边界崩溃。允许结果只有旧 binding、完整新 binding、CAS conflict 或 `driver unavailable`；不得出现半 revision、错误 transport 或 silent fallback。

### 11.4 取消

取消从 Turn/execution scope 传播到 bound model，再传播到 driver transport。Driver 必须重新抛出 `CancelledError`，关闭本次 stream/response，并释放自身 Effect 资源。取消不刷新选择、不回滚已经提交的 credential refresh，也不把未确认的外部请求记成成功。

### 11.5 Driver 升级与卸载

当前 execution 持有旧 plugin snapshot lease。升级或卸载只阻止新 binding 选择旧 driver；旧调用排空后才执行 driver Effect cleanup。新 artifact 必须向后读取所有历史 committed Connection config 与 credential payload formats；本规格不允许 contract。不能按 driver name 在每次请求时重查 current registry。

## 12. 失败语义

| 失败 | Owner | 可观察结果 |
|---|---|---|
| `models` Service 缺失 | composition Root | 依赖插件 pending；聊天 readiness 明确缺失 Service；管理壳仍可用于安装 |
| driver 缺失 | models | Model 标记 unavailable；新 binding fail-loud；其他 driver 不受影响 |
| credential 缺失/失效 | driver + models connection | typed auth error；不尝试其他账号 |
| catalog 不可用 | driver | 保留手工 Model 路径或明确 discovery error；不伪造空目录成功 |
| capability unknown | models | 保存 unknown；调用保持 provider 原始边界 |
| provider timeout/rate limit | driver | 保持可区分 typed error 和 retryable 属性 |
| context 超限 | driver/models public error | 不转为空回复，不自动切换模型 |
| embedding dimension 不符 | embeddings/models | 拒绝写入目标 space；Akasha index 不变 |
| candidate probe 失败 | PluginManager + models | candidate 拒绝；stable 保持可用 |
| model revision 冲突 | models settings | 拒绝 stale writer并返回 current revision |

不得使用默认 provider、旧 credential、同名新 driver、空向量、假 usage 或 mock success 静默降级。

## 13. 持久状态

第一阶段保留现有 workspace 模型注册库与 Session selection 路径。代码 owner 迁移不授权数据移动。

首轮 schema 只增加四个已经由真实合同证明的字段：`model_connections.driver_config_json`、`model_registry_meta.default_embedding_model_id`，以及 chat/embedding 表各自的 `capabilities_json`。后两个 JSON 保存完整、严格 JSON 的 capability、source 与 model-level driver config；现有 capability 列继续双写供旧 reader 使用，新 reader 对没有 JSON 的历史行按旧列完整投影。这个 envelope 代替为每个新 capability 加列，也避免 capability 写后重启变成 unknown。现有 `provider` 即 driver identity，`model_role_bindings` 继续只承载聊天 role；不新建 driver、contract、digest 或通用 Binding 表。

| 状态 | 正常增加 | 允许原位更新 | 逻辑失效 | 物理减少 | Owner | 恢复证据 |
|---|---|---|---|---|---|---|
| Connection | 用户/Onboarding 成功 probe 后新增；包含 driver/config identity | 名称、endpoint、driver config、auth identity；用户设置事务增加 revision；同 auth ID token refresh 不增加 revision | `enabled=false` | 仅独立显式删除；存在 Model/Binding 引用时拒绝 | models plugin | operation backup、SQLite integrity check、revision |
| Model | discovery 或用户确认后新增 | capability snapshot、显示信息；事务增加 revision | `enabled=false` 或 driver unavailable | 仅独立显式删除；存在 default/Session/index 引用时拒绝 | models plugin | backup、catalog digest、probe receipt |
| workspace Binding | 首次设置 default chat/default embedding/role | 显式切换并增加 revision | 指向 disabled Model 时 unavailable，不自动改指 | 清除显式 binding 后回到已定义 fallback 规则 | models plugin | transaction receipt、旧 revision |
| Session model selection | 用户首次固定 model/effort | Session/Turn admission owner 在 catalog 纯校验后切换 | 清除后跟随 workspace default | 只删除该 metadata key；不改变 Message | Session owner；models 仅校验 | sessions.db backup、message digest |
| Credential payload | Connection 创建/登录成功 | 同 auth ID refresh/token rotate | Connection disabled | 只随独立 Connection 删除流程 | models plugin；driver 仅持有窄 handle | secret-mode backup、auth probe、无日志泄漏检查 |
| Embedding space/index | 首次使用一个完整 identity | 同 space 只追加合法向量和索引状态 | default 改变后旧 space retired/read-only | 仅显式 reindex/删除操作；不得随 Provider 卸载自动删除 | consumer plugin，如 Akasha | space identity、source message digest、index audit |
| Plugin artifact/cache | install candidate/stable publication | upgrade/revert | uninstall/retire generation | 正式 uninstall/cache GC 协议 | PluginManager | install receipt、snapshot identity、revert target |

`sessions.db/messages` 继续只追加。模型切换、Provider 卸载、embedding 重建和插件迁移都无权 UPDATE 或 DELETE 已有消息正文。

## 14. 普通插件证明

每个 `models`/Provider builtin 必须通过同一个外置安装 Gate：

1. 从仓库复制插件源码到独立临时 source repository，并构建正式安装 artifact。
2. 在待测 checkout 中移走或改名原插件源码，确认 import path 不再指向仓库副本。
3. 使用一次性 plugin home、workspace 和正式 `plugin install` 安装 artifact。
4. 清空旧 cache 后冷启动；不得注入额外 `PYTHONPATH` 或 repo-relative path。
5. 静态依赖图只允许 Python 标准库、插件自身依赖和公开 `agent.plugin_composition` API；运行后收集模块 `__file__` provenance，确认没有从主仓库插件源码或旧 cache 加载实现。
6. PluginManager 不含待测 plugin ID、provider ID、目录名或 factory 特判。
7. 验证 candidate、promotion、stable、upgrade、revert、uninstall 和 reinstall。
8. 用真实或受控本地 provider 完成 discovery/probe、chat、stream/tool call、usage 和 embedding。
9. 卸载后贡献全部消失、lease/in-flight 归零；权威 Connection、Model、Session 和 embedding data 默认保留。
10. 重新安装兼容版本后从保留数据恢复可用，不依赖旧源码目录。
11. 联合场景同时移走 `models`、`openai-compatible`、`codex`、`opencode-go` 四个源码目录，再只通过正式 artifact 安装并完成端到端调用。

仅有 manifest、`plugin-doctor healthy`、安装成功或 import 成功都不足以证明普通插件。

## 15. 正交性与概念完整性 Gate

### 15.1 变化轴矩阵

| 变化 | 允许改变 | 不得迫使改变 |
|---|---|---|
| 新增 Provider | 新 Provider 插件、driver catalog | Core、models 实现、ReAct、Akasha |
| Provider transport 升级 | 对应 driver generation | model revision、Session selection、其他 Provider |
| 切换默认 chat model | model Binding/revision | Provider 插件、ReAct 算法、embedding default |
| Session 选择模型 | Session selection | workspace default、embedding、其他 Session |
| 切换默认 embedding | embedding Binding/revision、新 space | chat selection、ReAct、Provider lifecycle |
| 修改能力识别 | Provider evidence mapper 或 models normalization | Core、Session/Turn schema |
| 修改模型设置 UI | UI contribution | transport、credential schema、ReAct |
| 卸载 Akasha | Akasha contribution/runtime | models、Provider、Session messages |
| 卸载某 Provider | 该 driver contribution与 availability | 其他 Provider、models data、Core |

任一矩阵行出现不在“允许改变”列的代码修改，评审者必须要求解释或拒绝方案。

### 15.2 概念数量检查

- 不新增 `ModelManager`、`ProviderManager`、`DriverHost`、`ModelRuntimeHost` 等只转发的平行 owner。
- `models` 插件唯一拥有模型 registry/revision；五个 Service facade 共用一个私有 state 和 store。`MODEL_DRIVERS` 是 contribution facade，不是第二个模型 registry。
- `Bound*Model` 只是 `ModelExecution` 的 typed operation view，不拥有 lease、revision manager 或持久状态。
- Provider driver 只拥有协议行为，不复制 Model/Connection/Binding。
- Core 使用同一套 Root/Fiber/Effect/snapshot，不建立 model generation manager、revision lease 或公开 publication fence 特例。

五个 ServiceKey 不能继续合并，理由不是实现分层或同进程安全隔离，而是正常调用路径的方法面、激活依赖和消费者集合不同：

| 如果合并 | 立即增加的无关能力 |
|---|---|
| `CHAT_MODELS + EMBEDDINGS` | Akasha 的正常向量调用对象直接暴露无关 chat/tool-call 方法面 |
| `MODEL_CATALOG + MODEL_SETTINGS` | Onboarding、Dashboard、picker 的正常只读对象直接暴露模型库写方法 |
| `MODEL_DRIVERS + 任一消费 view` | Provider 的正常注册对象直接暴露执行其他 Provider 或改 workspace default 的方法 |

因此最小实现是“一份私有状态，五个 capability facade”，不是“一个万能 `MODELS` Service”，也不是“五个 manager”。这比 DSH 的单一 `ctx.llm` 多出的分离只来自 Akashic 已存在的 embedding consumer、持久 workspace settings 和非特权插件边界；adapter registry、调用准备和 effect 撤销仍采用 DSH 的同一种组合哲学。

### 15.3 本轮删减账本

| 已删除或折叠 | 最小替代 |
|---|---|
| model generation manager、retired generations、revision lease | exact plugin snapshot lease + 一次 SQLite snapshot copy |
| 独立 `publication fence` / ordered operation | driver expand-contract + SQLite revision CAS + exact snapshot lease |
| 模型专用 `OperatorActionGrant` / renamed action token | authenticated HTTP/control boundary；不伪装成同进程 sandbox |
| Web navigation/data/action 前置 | 独立 UI contribution 规格；模型只暴露 catalog/settings |
| `ModelExecution.embedding()` 与 `EMBEDDINGS.bind(execution=...)` 双入口 | 所有场景只用 `EMBEDDINGS.bind()`；插件内部复用当前 frozen snapshot |
| `MODEL_CATALOG.connection()` / `model()` | 一个 immutable catalog snapshot 内 lookup |
| 五个 settings 方法 | `MODEL_SETTINGS.apply(ModelChange)` |
| `driver_owner_generation_ids` | `plugin_snapshot_id` 的 topology 投影 |
| 新 `ChatRequest`/`ChatResponse` DTO | 迁移并收紧现有 `ModelRequest`/`LLMResponse` 为唯一公共 vocabulary |
| `ModelExecutionIdentity` | execution object identity + bound model descriptor |
| Driver 的能力/schema/health/UI 平行 metadata | `open()` 与可选 discover/probe/auth handlers |
| 公开 fallback role | 现有 provider/role policy |
| 旧 bootstrap → 新 Service 临时 adapter | 直接交付普通插件纵向切片 |

### 15.4 直接性检查

下面四个调用必须各自只出现一个领域入口：

```text
注册 Provider        MODEL_DRIVERS.register(...)
运行聊天模型          CHAT_MODELS.execution(...) → execution.chat(role).complete(...)
Turn 外 embedding     EMBEDDINGS.bind(...) → embed(...)
Turn 内 embedding     EMBEDDINGS.bind() → embed(...)
修改模型设置          authenticated route → MODEL_SETTINGS.apply(command)
```

如果调用者还必须手动刷新 generation、读取 credential、选择 transport、操作数据库或通知 PluginManager，设计失败。

## 16. 一次功能切换

没有临时 adapter 时，不存在“models 已接管，但某个现有 Provider、Akasha 或 control consumer 尚未迁移”的可发布中间态。实现可以拆成未启用的独立提交，产品切换必须一次完成：

1. **基线 Gate，不是运行状态**：固定当前 model role、Session selection、usage、stream、tool call、retry、error、embedding 和 active execution 语义；建立差分 normalization allowlist，默认 `semantic_delta: none`；记录全部直接消费者。
2. **决策前置**：接受对 0027/0039 的 owner 勘误；没有 accepted 决策不得切换 owner。
3. **公共合同与 state owner**：增加五个 ServiceKey/Protocol/DTO 和五个 facade；把 registry、selection、capability normalization、settings transaction 与 credential handle 移入普通 `models` artifact。
4. **最小 schema**：保留现有模型库和 `model_connections.provider`，把它解释为 `driver_id`；增加 connection config、default embedding 与两张 model 表的 capability JSON envelope。不增加 driver/version/contract/digest 表或通用 Binding 表。
5. **三个 Provider 同时迁移**：`openai-compatible`、Codex 和 OpenCode Go 分别成为外置 artifact；auth、refresh、transport、catalog 和 profile 全部随各自 driver 迁移。
6. **全部消费者同时切换**：bootstrap 在 committed snapshot 内解析 Service；Akasha 改为 `EMBEDDINGS`；chat/settings/Mobile API、job、compaction、vision 和 memory consumer 按第 10.6 节迁移。
7. **一次 legacy handoff**：Yoyo 把旧 `[llm]` 与 `[memory.embedding]` 的最终事实写入 models registry，并只把 Session 中同一维度的既有向量改为最终 space identity；消息正文和 Akasha sidecar 不变。Akasha 先明确降级，再由 `/akasha_reindex confirm` 沿 artifact-owned repair 重建派生文件，不重复请求已经完整的向量。
8. **删除旧链**：证明零消费者后删除 `agent/provider.py` 的 Provider 分支、`ModelRegistry`/`ModelGeneration`/`RoleBoundProvider`、DB-to-Config 投影和 bootstrap builders；不保留旧 bootstrap → 新 Service adapter。
9. **联合普通插件 Gate**：四个源码目录同时移出仓库，仅通过正式 artifact install，完成 chat + embedding + auth + control API + uninstall/reinstall 数据恢复。

2236 顶部导航、模型页面、Onboarding 动态 Provider 面板另立 UI contribution 规格；它们消费本规格的 catalog/settings，但不阻塞这次运行时切换。不得把模型数据库搬迁、Session schema 改名、role 删除或整个 ReAct 插件化混入本次切换。

## 17. 回滚

- 发布回滚使用切换前 artifact/commit；新旧 owner 不在同一进程共同写模型库。
- additive schema 由旧 reader 忽略；回滚读取同一模型库，不进行逆向数据迁移。
- Provider candidate 失败只 discard candidate；stable snapshot 和 active lease 保持。
- Provider stable 出现运行故障时使用正式 plugin revert 发布旧 artifact；不能修改 current pointer 或 cache 文件伪造回滚。
- settings 事务失败回滚 SQLite transaction；已经成功提交的事务通过 operation backup 和显式后续事务恢复，不改写 revision 历史。
- embedding default 回滚只恢复 default binding；已经创建的新 space 保留，不自动删除向量。

## 18. 验收

### 18.1 结构验收

- Core 和 bootstrap 对 `openai`、`codex`、`opencode-go`、DeepSeek、DashScope 零名称分支。
- bootstrap 在 PluginManager 之前不构造 ModelRegistry/LLMProvider。
- 只有一个 committed plugin Root；不存在 models 预启动 Root 或 plugin ID bootstrap lookup。
- Core 启动路径对 `ModelRegistryStore` 零读取；启用 Mobile 且没有安装 models 时仍能冷启动到 2236 管理壳。
- 仓库内置普通插件不 import模型实现或兄弟插件源码。
- `models`、三个 Provider 均通过第 14 节外置安装 Gate。
- Service topology 能显示 provider plugin → `MODEL_DRIVERS`，consumer → 对应窄 Service。
- 五个 facade 是五个不同 Service value：Embeddings 没有 chat execution，Catalog 没有 apply，Drivers 没有 catalog/settings/execute；它们只共享私有 state。

### 18.2 Chat 验收

- 配置至少两个 Connection 和三个聊天 Model；workspace default 与 Session override 分别正确解析。
- 一个 ReAct 在两次 provider request 之间修改 model revision，当前两次仍使用旧 model，下一 ReAct 使用新 model。
- 一个 ReAct 在两次 request 之间升级 Provider plugin，当前两次仍使用旧 driver generation，下一 ReAct 使用新 driver。
- 同一 ReAct 两次 request 之间刷新同一 auth identity：descriptor 与 revision 不变，第二次通过原 `CredentialHandle` 使用新 payload；刷新失败返回明确 auth failure，不切换 identity。
- 同一 Turn 的 agent、vision、compaction 和 summary 全部来自一个 `ModelExecution`；role 可以不同，snapshot/revision descriptor 必须相同；fallback 仍由同一绑定内的 policy 执行。
- settings update、兼容 Provider upgrade/uninstall 和 bind 的并发 race oracle 证明每个执行只观察一个 exact plugin snapshot 与一个完整 SQLite revision；结果满足 expand-contract，或明确 `driver unavailable`。
- 旧 snapshot 的 settings/auth 开始、兼容 driver promotion、旧 operation CAS commit 的顺序只能得到新 driver 可读的历史 config/credential format、CAS conflict 或卸载后的 unavailable；不得发布删除任一旧 reader 的 candidate。
- stream、tool call、parallel tool call、reasoning、usage、取消和错误分类与基线等价。
- 缺失 driver、credential 错误和 capability unknown 分别可观察，不发生静默 fallback。

### 18.3 Embedding 验收

- Akasha 唯一直接 inject 的模型能力是 `EMBEDDINGS`；运行配置和日志无 secret。
- 默认 embedding 切换产生新的 space descriptor/identity，旧索引不混写；在显式 reindex 完成前 Akasha 可观察降级。
- 一个 Turn 中途修改默认 embedding 或升级其 Provider，Turn 内后续 embedding 仍使用该 Turn view 的旧组合；下一独立 embedding execution 使用新组合。
- dimension mismatch 在写入前失败；Session message 和旧索引 digest 不变。
- Provider 卸载后旧 embedding 数据保留；重装兼容 driver 后可继续使用。
- 显式 reindex 留下 SessionDB/sidecar backup、请求与完成 manifest；失败或取消不发布不完整候选，重启可重试。

### 18.4 内置插件验收

- Passive Turn 通过通用 Turn owner取得 `CHAT_MODELS`。
- Scheduler、Subagent、Wake 的 manifest/inject 不增加模型 Service，仍经 `SCOPED_TURNS` 执行。
- Akasha inject 从 `TEXT_EMBEDDING_SETTINGS` 改成 `EMBEDDINGS`，不再构造通用 HTTP Embedder。
- server/control 的 Model API 不含 Provider ID 分支；现有 Onboarding 前端与完整动态 Web 面板由后续 UI contribution 规格验收。
- 无模型时 2236 管理壳可打开；聊天 readiness 明确不可用而不是进程崩溃。

### 18.5 状态与恢复验收

- 每个设置命令核对 before/after revision、行级 write set、backup 和 integrity check。
- Provider install/uninstall/revert 不减少 Connection、Model、Session selection、Message 或 embedding space。
- active execution、plugin lease、driver in-flight 在完成/失败/取消后全部归零；不存在 model revision lease 计数。
- crash 后从 committed plugin snapshot、模型库 revision 和各 consumer 自有状态恢复；不从内存 current pointer 推断外部效果。

### 18.6 独立评审通过条件

评审者必须逐项回答：

1. 每个 Service 是否拥有独立操作轴、激活依赖或消费者集合；能否删除或合并而不增加无关耦合。
2. Core 是否仍含模型或 Provider 业务语义。
3. models 与 Provider 是否有重复 registry、generation、credential 或 capability owner。
4. 常用任务是否通过一个直接入口完成。
5. 增加第四个 Provider 是否只增加一个普通插件。
6. Akasha、ReAct、Onboarding、Wake/Scheduler/Subagent 是否只依赖所需能力。
7. plugin snapshot 与 model revision 是否在一次绑定中同时冻结。
8. 外置安装 Gate 是否能发现仓库路径、私有 import、Core 特判和残留 cache 造成的假普通插件。
9. 持久状态的增加、更新、失效、减少、owner 和恢复证据是否完整。
10. 是否存在一个“很好但不服从 Service/inject/Effect/snapshot 哲学”的额外机制。
11. 启动是否只有一个 Root，且 Turn runtime 能在不预先捕获模型的情况下从 exact snapshot 解析 Service。
12. driver 是否通过 expand-contract 保证任一 committed revision 都能被当前 artifact 读取，从而无需第二个 fence Service。
13. Session selection 是否仍由 Session owner 写入，模型插件只做纯校验。
14. 当前全部直接模型消费者是否已有明确迁移落点。
15. Provider 完整缺失是否能正常发布 unavailable，而同名 driver 的不兼容升级是否 fail-loud。
16. Mobile、Config load 和 reload 路径是否已经停止在 bootstrap 捕获模型实现。

任何一项为否，规格不得进入实现。
