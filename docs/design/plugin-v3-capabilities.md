# 插件 V3 能力手册

本文记录当前插件 V3 的公开能力和最短用法。代码真源是
`agent/plugin_composition/__init__.py`、`agent/plugins/composable.py` 以及各能力模块；未从公开包
导出的 Core 对象不属于插件 API。

## 1. 最小插件

```python
from agent.plugin_composition import Context

api_version = 3
name = "example"
version = "1.0.0"
inject = ()


async def apply(ctx: Context, config: object) -> None:
    pass
```

Core 只接受精确的 `apply(ctx, config)`。`api_version != 3`、V2 `Plugin` 子类、固定 lifecycle
方法和 phase module 注入都不会被加载，也没有自动包装或兼容 fallback。插件不能直接接入
`EventBus`；V3 事件由明确 owner 通过 typed key 发布。

| 模块声明 | 用途 |
|---|---|
| `api_version`、`name`、`version`、`apply` | 必需的身份和唯一入口 |
| `Config` | 可选配置模型；Core 校验后传给 `apply` |
| `inject` | 根 Fiber 激活所需的 `ServiceKey` |
| `is_active(services)` | 根据冻结的静态 Service view 决定是否发布静态贡献 |
| `static_semantic_checks()` | 返回安装或 generation 的静态语义检查 |
| `skill_roots`、`drift_skill_roots` | 发布普通 Skill 和 Drift Skill |
| `workspace_roots`、`workspace_files` | 声明被授权的 workspace 路径；只授予真正的数据 owner |
| `dashboard_module` | 发布 Dashboard HTTP/面板模块 |
| `web_module`、`web_requires`、`web_provides`、`web_contract_digests` | 发布 Web 模块及版本化组合合同 |

## 2. 组合原子能力

每次 `apply` 都属于一个 generation-bound Fiber。下列注册和任务归该 Fiber 所有，并在失活、
重启或卸载时逆序清理。

| 原子能力 | 最短用法 | 语义 |
|---|---|---|
| 硬依赖 | 模块级 `inject = (KEY,)` | 全部 Service 可用时根 Fiber 才激活 |
| 可选依赖 | `await ctx.inject((KEY,), child)` | 子 Fiber 随依赖出现和消失，不阻塞 Root readiness |
| 子 Fiber | `await ctx.mount(child, name="worker")` | 分开生命周期、Health、Effect 和依赖 |
| 提供 Service | `await ctx.provide(KEY, value)` | 当前 Fiber 成为该 key 的活动 provider |
| 读取 Service | `ctx.require(KEY)` / `ctx.get(KEY)` | 必需读取 fail-loud；可选读取返回 `None` |
| Effect | `await ctx.effect(setup, label="client")` | `setup` 返回 cleanup；Fiber 逆序调用 |
| 后台任务 | `await ctx.spawn(run(), name="poll")` | 失败进入 Fiber 状态，卸载时取消并等待 |
| Health | `health = await ctx.health("upstream")` | `degrade(reason)` / `recover()`；required 项参与 readiness |
| Incident | `ctx.report_incident("fetch", "timeout")` | 记录历史失败，不隐式改变 Health |
| 数据根 | `ctx.data_root` | Core 为 formal 或 candidate 分配的独立数据根；插件可正常读写 |
| Workspace 路径 | `ctx.workspace_root("memory")` | 返回模块预先声明的原生 `Path`；Core 校验路径归属，但不拦截写入 |
| 运行身份 | `ctx.runtime`、`ctx.generation_id` | plugin、artifact、generation 和目录身份 |
| 短运行作用域 | `async with ctx.runtime_scope(): ...` | 后台操作绑定 exact Root lease |
| 跨 task 作用域 | `scope = ctx.capture_runtime_scope()` | 显式 fork 当前 lease；调用者负责关闭 |
| 诊断 | `ctx.diagnostics.operation(...)` | 记录 generation-bound 边界和有限指标 |

跨插件 Service 使用本地、版本化结构合同：

```python
from typing import Protocol
from agent.plugin_composition import Context, ServiceKey

class Greeter(Protocol):
    def greet(self, name: str) -> str: ...

GREETER = ServiceKey[Greeter]("example.greeter.v1")

async def apply(ctx: Context, config: object) -> None:
    await ctx.provide(GREETER, MyGreeter())
```

双方各自声明同名、同结构的 key，通过 `inject` 和 `ctx.require()` 连接，不能 import 对方源码。

服务若依赖动态注册者，使用 `await ctx.provide(KEY, value, binding_contributors=read_contexts)`
声明归档依赖。`read_contexts()` 同步返回当前实际注册者的 `tuple[Context, ...]`，只读原注册状态；
固定 binding 时，这些 Context 与静态 `inject` 一起进入同一依赖闭包。伪造或其他 Root 的 Context
会被拒绝，声明随该 Service 的 Effect 清理。普通服务不传此参数。固定 binding 时已知目标子集的
目录（如工具）继续传已有的 `contributors`，只归档该目标；恢复后才选择目标的服务（如材料）
声明其可用注册者，具体调用仍由原 `bind` 合同选择。

## 3. Typed events

插件通过明确的 typed key 注册 listener；同一事件名只能绑定一种 dispatch 合同，注册顺序就是 listener
顺序。事件 payload、返回值和失败处理由 key 的声明者定义，listener 的生命周期由其 Fiber owner 管理。

| Key / API | 调度语义 | 顺序与失败 |
|---|---|---|
| `EmitEventKey[P]` / `ctx.emit(key, payload)` | 同步调用 listener | 按注册顺序；listener 必须同步，异常立即向调用者传播；返回 awaitable 是错误 |
| `SerialEventKey[P, R]` / `await ctx.serial(...)` | 逐个等待 listener | 按注册顺序；`None` 继续，类型正确的 `Bail[R]` 立即短路；异常记录 owner failure 后传播 |
| `ParallelEventKey[P]` / `await ctx.parallel(...)` | 为全部 listener 建立异步任务并等待 settle | listener 必须返回 awaitable；并发执行，异常聚合为 `BaseExceptionGroup`；调用取消会取消并 drain 子任务 |
| `TransformEventKey[P]` / `await ctx.transform(...)` | 将当前 payload 依注册顺序交给下一 listener | 每步必须返回声明的同类型 payload；`None`、`Bail`、错误类型和异常都失败 |
| `ObserveEventKey[P]` / `await ctx.observe(...)` | 调用全部 observer，再等待异步结果 | 全部 observer 都会被调用；普通 listener 失败隔离为 owner Incident，调用者继续；调用取消仍取消并 drain 未完成 observer |

Runtime lifecycle signal 使用同一 typed event 基础：`RUNTIME_STARTING` 在正式接纳开放前准备资源，
`RUNTIME_STARTED` 在外部服务就绪后启动工作，`RUNTIME_STOPPING` 在服务停止前收束工作，
`SNAPSHOT_SEALING` 在 candidate catalog 冻结前完成 seal。`SOURCE_CHANGED`、`EVENTMAIL_CHANGED` 和
`DRIFT_CHANGED` 等来源 signal 由各自插件声明和发布；它们不组成 Core 业务事件表。新插件定义自己的
typed key 或窄 `ServiceKey`，不依赖已退役的 Core 业务事件名。

## 4. Runtime Service 与插件能力

Runtime Service 通过 `inject` 和 `ctx.require(KEY)` 连接；插件能力由拥有该 key 的插件注册，声明型注册本身是 Effect。

### 4.1 人与 Agent 的入口

| Key | 主要方法 | 用途 |
|---|---|---|
| `COMMANDS` | `register(ctx, CommandDefinition(...))` | 人类命令、alias 和 handler |
| `TOOLS` | `register(...)`、`bind(...)`、`open(...)` | `plugins.tools` 的工具描述、参数准备、exact binding 与执行入口 |
| `UI_SLOTS` | `register_mobile(ctx, definition, query=...)` | Mobile 页面、查询和导航 |
| `CHANNELS` / `CHANNEL_INPUT` | 注册 blueprint，按绑定调用入站入口 | inbound/outbound Channel 适配 |
| `DELIVERY` / `DELIVERY_READ` | 打开发送 admission 或只读历史 | Delivery 发送、恢复和查询 |

旧 Core `TOOL_CATALOG` 及其注册、冻结和快照装配已删除。旧 `DELIVERIES`、
`DURABLE_DELIVERIES` ServiceKey 和注入也已退役。Core 保留中立的持久投递记录、
恢复与只读升级检查，保护旧 `prepared/provider_started` 状态；移除接口不会删除记录或重发旧效果。
工具消费者通过 `tools.v1` ServiceKey 和本地结构接口协作，
不能 import `plugins.tools.api` 或其他兄弟插件实现。工具结果提供 `outcome` 与 `parts`；
Tools owner 在入口校验。ToolResult Message 是对话调用的持久结果正文。

需要共用参数 schema 校验或旧工具值表示的 adapter 可以使用公开的
`agent.tool_catalog`。该模块是纯结果值与 schema 函数的实际 owner，没有插件注册器、
快照或执行生命周期；它与普通 Tools 插件的 `outcome/parts` 结果合同是不同的协议边界。

### 4.2 Message、Session 与上下文

当前生产组合中与 Message、Context 和 Turn projection 直接相关的能力如下：

| Key | owner | 用途 |
|---|---|---|
| `MESSAGE_CATALOG` | Core Message owner | 读取已提交 Message |
| `MESSAGE_WRITERS` | Core Message owner | 按绑定权限追加 Input、Output 或 ToolResult |
| `SESSION_ADMISSION` | Core Message owner | 首次创建具有固定 `SessionAttributes` 的 Session；不提供 Message 或 Control 写权 |
| `MESSAGE_EMBEDDINGS` | Core Message owner | 读取和追加 Message embedding |
| `CONTEXT` | `plugins.context` | 用已选 Message 与材料组装 provider request |
| `MATERIALS` | `plugins.context` | 注册并按来源选择 Prompt、摘要与其他 Context 材料 |
| `COMPACTION_SUMMARIES` | `plugins.compaction` | 读取已发布摘要记录和父链 |
| `TURN_PROJECTION` | `plugins.turn_projection` | 从 Message 日志读取无状态 Turn 投影 |

`SESSION_COMPACTION_STORAGE`、旧语义兴趣评分和 `SESSION_READ` 已退役；当前
Context/Compaction 直接消费 Message 与普通材料能力。外部反馈插件使用
`MESSAGE_CATALOG.follow()` 与 Turn projection，不依赖旧 SessionManager。

Turn 是 `plugins.turn_projection` 从 Message 日志得到的无状态读投影，不是 Core Service 中的可变执行对象。
消费者自行保存 cursor 和学习状态；投影不能授权消息写入、工具执行或外部发送。

### 4.3 调度与外部运行

| Key | 主要方法 | 用途 |
|---|---|---|
| `TIMERS` | `schedule(deadline)` | Core-owned timer |
| `MCP_SERVERS` | `register(ctx, McpServerDefinition(...))` | generation-bound MCP server |
| `MANAGED_PROCESSES` | `register(ctx, ManagedProcessDefinition(...))` | Core 监督的进程 |
| `WORKLOADS` | `register(ctx, Workload(...))` | 窄 Controller 管理的容器 workload |
| `EXECUTOR_SERVICE` | `parallel_sync(jobs)` | 有界纯同步工作；worker 不取得 Context/Fiber |

Skill 和 Drift Skill 使用模块级 `skill_roots` / `drift_skill_roots`，由安装、candidate readiness
和 generation catalog 原子发布。MCP、process 和 workload 有两份职责不同但必须一致的声明：
`akashic.plugin.toml` 提供 import-free admission identity，`apply` 再通过上表 Service 建立
Fiber-owned registration；candidate readiness 会逐字段核对，不一致时 fail-loud。

### 4.4 模型

| Key | 主要方法 | 用途 |
|---|---|---|
| `CHAT_MODELS` | `execution()`、`independent_execution()` | 在执行作用域内通过 `chat(role)` 取得模型 |
| `EMBEDDINGS` | `describe()`、`bind()`、`save_binding()` | 固定向量空间，在绑定作用域内调用 `embed(texts)` |
| `MODEL_CATALOG` | `snapshot()`、`validate_chat_selection()` | 模型和 connection 目录 |
| Models 内部 `MODEL_SETTINGS` | `discover()`、`apply(ModelChange)` | Models 自己的设置事务，外部控制调用 `models/command` RPC |
| `MODEL_DRIVERS` | `register(ctx, ModelDriverDefinition(...))` | Provider 注册模型 driver |

Provider 返回结构化 `ModelUsage` 和公开错误类型；未知能力保持 unknown，不用默认值伪装。

设置命令与 `MODEL_SETTINGS` 不由 Core 导出；消费者不能 import Models 的命令类型。
模型选择能力 `models.selection.v1` 由 owner 和消费者分别声明本地窄 key。角色是字符串，
当前 Models 的四个预设及 fallback 由插件解释，Core 不维护角色枚举。

`DriverConnection` 可提供异步 `close`。Models 在 chat/embedding scope 结束、取消或部分绑定失败时调用 `aclose()`；嵌套的同一次 chat execution 共用连接，设置探测使用的临时连接在检查后关闭。Bound model 只能在取得它的 scope 内使用。没有资源的旧 driver 可省略 `close`。内置 HTTP driver 延迟创建客户端，在同一连接内复用 socket，每次请求仍读取凭据并独立生成请求头。

### 4.5 插件间声明

插件可以提供自己的 `ServiceKey`，例如 `memory.recall.v1`、`eventmail.wake.v1` 或
`drift.proposals.v1`。这些不是 Core 能力总表：owner 定义结构合同，consumer 只通过 key 连接。
`EMBEDDING_MEMORY_PLUGIN` 是当前 embedding-memory owner claim，同一 Root 只允许一个 owner。

## 5. Dashboard 与 Web

`dashboard_module = "dashboard.py"` 让 Core 用 `DashboardContext` 加载模块。Dashboard 只能通过
`workspace_root()`、`workspace_file()` 和 `workload_url()` 取得已声明资源。

`web_module` 指向随 artifact 发布的浏览器模块。`web_requires` / `web_provides` 声明组合合同，
`web_contract_digests` 固定合同内容。缺少 provider、digest 不一致或越界资源在 publication Gate
fail-loud。

## 6. Generation 与 candidate

```text
source + config
      │
      ▼
isolated candidate Root ── settle / Health / Incident / semantic checks
      │ pass
      ▼
committed snapshot ── stable/latest pointer ── request lease
      │
      └─ old request keeps old Root until lease drain
```

- Candidate 使用隔离 Root、plugin-data 副本、workspace 投影、端口和外部效果策略，不能
  作为 stable Root 执行，也不能复用 stable 的执行或数据效果。owner 不变时可以复用已批准的不可变 catalog，
  但不能借此取得 stable 的运行状态或写入权。
- Root 只生成能力，不能自行晋升。artifact、journal、stable/latest、parent Turn 授权和恢复由 Core
  publication plane 拥有。
- Workspace path 是显式授予正式数据 owner 的高权限能力，不应替代窄 Service；candidate
  只得到声明路径在 attempt workspace 内的副本。
- 普通卸载删除代码、manifest 和派生投影，默认保留 plugin-data。`manifest.toml` 只接受
  独立 `[plugins."<id>"]` 条目；旧 `[packages]` 分组不会展开、保留或静默忽略。

## 7. 选择能力

1. 同一插件内部拆生命周期：`ctx.mount()`。
2. 插件之间共享行为：版本化 `ServiceKey` + `provide/require`。
3. 已提交事实的变更通知：由事实 owner 定义窄 typed signal。
4. 对人暴露动作：`COMMANDS`；对模型暴露动作：`TOOLS`。
5. Message、Context、Turn projection、Model、Tool、Content 和 Delivery 通过各自插件 Service 组合。可选插件标签写入普通 Message metadata，使用 [SES-009](../projectneed.md#ses-009-插件附加信息使用普通-message-metadata) 与[消息附加信息合同](0902-reviewed-v4.md#34-message-metadata-的实现与迁移)，不注册新的内容块。
6. 长时或可恢复工作：`TASKS`、`TIMERS`。
7. 外部进程、MCP、容器：`MANAGED_PROCESSES`、`MCP_SERVERS`、`WORKLOADS`。
8. 找不到匹配能力时先定义窄 Service，不给 Manager 增加新的固定插件方法。


### 归档接口版本

组件归档的 `runtime.binding_api` 当前为 2。Core 在打开任何组件源码前核对完整
闭包的接口版本和 Python tag；ABI 1 明确不兼容，不能混用新接口或从当前插件补齐。
原 descriptor、源码树、binding 引用和已开始效果的回执保持原位，旧归档需要原 Core
版本及其安装环境恢复。该接口版本与 Python environment descriptor 的版本独立。
新版本创建的归档仍能在原安装移除后，按原配置与 generation 闭包恢复。
