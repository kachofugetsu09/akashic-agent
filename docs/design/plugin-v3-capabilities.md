# 插件 V3 能力手册

本文记录当前插件 V3 的公开能力和最短用法。代码真源是
`agent/plugin_composition/__init__.py`、`agent/plugins/composable.py` 以及各能力模块；未从公开包
导出的 Core 对象不属于插件 API。

## 1. 最小插件

所有插件制品只从根目录普通文件 `plugin.py` 加载，再调用 `apply(ctx)`。
不接受 TOML `entrypoint`，也不寻找其他文件名；缺失或符号链接入口在导入前拒绝。
Akasha、Wake、Scheduler、Compaction 和 Markdown Memory 的旧 `message_plugin.py`
已重命名，Python 导入者须改为对应的 `.plugin`，没有兼容别名。

```text
┌──────────────────┐    ┌────────────────────┐    ┌────────────┐
│ 安装/发现固定制品 │ ─▶ │ 根目录 plugin.py   │ ─▶ │ apply(ctx) │
└──────────────────┘    └────────────────────┘    └────────────┘
```

source revision、实际模块文件路径和代码树仍保留精确来源。Generation 与 source resolver
不再复制可选入口字段，`code_dir` 从实际导入文件的父目录取得。
新组件归档 descriptor 使用 v4，不再保存入口选择；旧 v2/v3 在导入前明确拒绝。
本层不迁移或改写旧归档、binding、插件数据和固定 Python 环境；恢复旧记录须保留基线 Core
及原归档，采用新入口须从已更新源码显式重装。身份仅由 `plugin.py` 顶层的 `name`、`version`、`api_version` 字面量赋值提供。

```python
from agent.plugin_composition import Context

api_version = 3
name = "example"
version = "1.0.0"
inject = ()


async def apply(ctx: Context) -> None:
    pass
```

Core 用一个位置参数调用 `apply(ctx)`，不限制参数名字或默认值；无法调用时由实际装配报告原错误。
`api_version != 3`、V2 `Plugin` 子类、固定 lifecycle
方法和 phase module 注入都不会被加载，也没有自动包装或兼容 fallback。插件不能直接接入
`EventBus`；V3 事件由明确 owner 通过 typed key 发布。

| 模块声明 | 用途 |
|---|---|
| `api_version`、`name`、`version`、`apply` | 必需的身份和唯一入口 |
| `inject` | 根 Fiber 激活所需的 `ServiceKey` |
| `workspace_roots`、`workspace_files` | 声明被授权的 workspace 路径；只授予真正的数据 owner |

配置从 `ctx.config` 读取，是当前组合固定输入的插件本地副本，不跟随全局文件变化。
插件自行选择解析方式，例如 `config = Config.model_validate(ctx.config)`；`Config` 只是插件内部普通类，
Core 不读取它。无配置时输入为空对象。候选只取得授权允许的输入，凭据仍是不可直接解析的引用。
启用条件写在普通 `apply` 分支中；所有贡献走同一注册路径，没有另一个 `is_active` 协议。

可选的根目录 `configure.py` 是插件自己的配置程序，不是普通辅助模块名称。
只有显式运行 `main.py setup` 才会从已启用的已安装 stable 制品发现并执行它，使用根目录固定 Python 环境。
正常加载、候选装配和换代不运行配置程序。旧制品的 `[setup]` 等已删字段必须通过显式重装或格式转换更新，
普通启动不改写旧制品或正式数据。

### Python 安装输入（0071 过渡层）

制品根目录和嵌套目录中的 `requirements.txt` 是 Python runtime 的唯一文件约定，
其父目录拥有该环境。TOML 不再接受 `python` 或 `[[python]]`；`StaticPythonRuntime`
暂时保留为环境 owner 的输入，由解析制品时一次发现，命令绑定不再扫描文件。

精确名称 `requirements.txt` 表示安装必需输入，包括空文件；`requirements-dev.txt`、
`requirements-optional.txt` 等其他名称不自动安装，除非被必需文件显式引用。
发布者不能把无关示例或可选依赖也命名为 `requirements.txt` 留在制品中。
Computer 的空 requirements 文件已删除；容器内命令不需要 Core 的 Python 环境。
扫描跳过 `.git`、`.venv`、`venv`、`node_modules`、`cache`、`.cache`、`__pycache__`、
`.pytest_cache`、`.mypy_cache` 和 `.ruff_cache`，不进入目录链接；其余目录链接、失效链接
或 requirements 文件链接直接拒绝，以免隐藏环境输入。

```text
┌─────────────────────────┐     ┌─────────────────────────┐
│ 固定制品 requirements   │ ──▶ │ 安装器准备固定环境引用  │
└─────────────────────────┘     └────────────┬────────────┘
                                            ▼
                               ┌─────────────────────────┐
                               │ 命令绑定最近 runtime    │
                               │ 使用其 exact interpreter│
                               └─────────────────────────┘
```

根 runtime 与嵌套 runtime 共存时，命令按既有脚本路径/cwd 解析结果选择最近的父 runtime。
缺少已 staging 的显式环境时失败，不借用 PATH 或制品中的 `.venv`。
环境只由安装器创建，加载或候选不准备环境，包括空 requirements 文件。
源码插件的纯进程内能力可以直接装配；实际 Python 命令缺少固定环境时明确失败。

### 身份读取（0071）

安装和每次加载前，loader 只用 AST 读取 `plugin.py` 顶层三个单次字面量赋值：
`name`、`version`、`api_version`。支持普通赋值和带类型注解的赋值，不接受计算表达式、
导入值、条件分支或重复直接赋值作为身份声明；不运行模块，不读取能力或配置 schema。
API 必须为整数 `3`。身份由 loader 固定后交给 Composable；运行中的模块属性变化不改变
安装名、展示版本或 API，也不触发第二次 expected/actual 比对。

`akashic.plugin.toml` 已退役，loader 不再读取或解释它，也不创建该文件。
凭据通过固定配置中的 `CredentialRef` 与独立正式授权解析，不再声明凭据路径。
历史制品中的旧 TOML 字节保留在原代码归档，不因此取得运行语义。安装清单的
`manifest.toml` 与插件自己的业务 `config.toml` 不是插件协议文件，不随本次删除。
代码树摘要、source revision、实际导入文件路径和环境引用继续固定原始来源。

v4 组件记录只新增；旧记录和正式数据不改写、不自动迁移或删除。旧格式需用原 Core 和完整
恢复材料读取，采用新格式须从已更新源码显式重装；旧 binding 不能被解释成新代码身份。
本层仅编写测试与静态查看，运行验收尚未执行。

## 2. 组合原子能力

初始化约束在 `apply` 或对应 provider 的实际注册中检查并抛出错误。底座不调用另一个
`static_semantic_checks` 自测入口，也不把报告存入运行 generation；诊断报告只记录真实装配步骤。

每次 `apply` 都属于一个 generation-bound Fiber。下列注册和任务归该 Fiber 所有，
编译后组合冻结；换代或卸载关闭整个 Root，依赖者先于 provider 退出，不原位重启 Fiber。

| 原子能力 | 最短用法 | 语义 |
|---|---|---|
| 硬依赖 | 模块级 `inject = (KEY,)` | 全部 Service 可用时根 Fiber 才激活 |
| 可选依赖 | `await ctx.inject((KEY,), child)` | 初始化期间按依赖选择子 Fiber，不阻塞 Root readiness；编译后不重绑 |
| 子 Fiber | `await ctx.mount(child, name="worker")` | 分开生命周期、Health、Effect 和依赖 |
| 提供 Service | `await ctx.provide(KEY, value)` | 当前 Fiber 成为该 key 的活动 provider |
| 读取 Service | `ctx.require(KEY)` / `ctx.get(KEY)` | 必需读取 fail-loud；可选读取返回 `None` |
| Effect | `await ctx.effect(setup, label="client")` | `setup`（可异步）只返回一个 cleanup 或 `None`；不解释 iterable 或生成器。Fiber 逆序关闭，成功才解除 owner；失败保留句柄与依赖供显式重试 |
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

async def apply(ctx: Context) -> None:
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
| `COMMANDS` | `register(ctx, CommandDefinition(...))` | 显式 `commands` provider 拥有人类命令、alias、封存与执行；消费者声明硬依赖 |
| `TOOLS` | `register(...)`、`bind(...)`、`open(...)` | `plugins.tools` 的工具描述、参数准备、exact binding 与执行入口 |
| `UI_SLOTS` | `register_mobile(ctx, definition, query=...)` | Mobile 页面、查询和导航 |
| `CHANNELS` / `CHANNEL_INPUT` | 注册实际 factory，按绑定调用入站入口 | 显式 `channels` provider 拥有连接、接纳、原绑定发送与恢复；来源插件拥有输入消费 |
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
| `MCP_SERVERS` | `register(ctx, definition)`、`open(ctx, name)` | 普通 `mcp` provider；每次调用独立会话 |
| `MANAGED_PROCESSES` | `register(ctx, definition)` | 普通 `managed_processes` provider；返回实际进程句柄 |
| `WORKLOADS` | `register(ctx, definition)` | 普通 `workloads` provider；取得窄 Controller 管理的容器句柄 |
| `EXECUTOR_SERVICE` | `parallel_sync(jobs)` | 有界纯同步工作；worker 不取得 Context/Fiber |

资产通过普通 `assets` 插件提供的 `INSTALLED_ASSETS` 注册。贡献方在 `inject` 中声明依赖，
在 `apply(ctx)` 中调用 `await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")`。
类别只是贡献方与读取方之间的约定；Core 不解释 Skill、Drift Skill 或类别表。provider 必须在
安装清单或 profile 中显式选择，缺少它时拒绝装配，Manager 不补入隐藏 provider。

```text
┌────────────────────┐    register(ctx, category, relative_path)
│ 贡献插件 apply(ctx) │ ──────────────────────────┐
└────────────────────┘                           ▼
                                  ┌────────────────────────┐
                                  │ assets：本 Root 注册表 │
                                  └───────────┬────────────┘
                                              │ callable：精确 scope 只读
                                              ▼
                                  ┌────────────────────────┐
                                  │ standard_tools：解析   │
                                  └────────────────────────┘
```

provider 从实际 Context 取得 owner 与固定代码制品根；拒绝跨 Root、越界路径与跨制品资源链接。
注册 Effect 随贡献方关闭，只移除内存记录。读取必须持有同一 Root 的实际 runtime scope，
返回代码归档中的原目录，不再复制临时资产树。服务 binding 收集注册者 Context，保持资源与
贡献代码闭包固定；standard_tools 仍独立保存技能工具的资源归档及其相对路径，不弱化解析。
代码制品、工具归档和用户 workspace 数据没有新的更新、逻辑失效或物理减少协议；关闭不会
删除它们，恢复仍依赖原代码归档、binding 与各数据 owner 的备份。

MCP、process 和 Workload 由显式选择的普通 provider 提供，Manager 不补入隐式依赖。
资源在 `apply` 中取得，Scope 在外部等待前登记关闭责任；失败保留同一资源句柄。
MCP 的端口引用直接使用 Workload/Process 返回的句柄，provider 检查 owner，Snapshot 不再列举
三类注册表或解释它们的依赖。Python 命令仍由宿主绑定固定制品环境；候选不解析正式凭据。
公开协议、每调用 MCP 的关闭语义与未知 Controller 请求限制见[普通资源 provider](plugin-resource-providers.md)。

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

Web/UI 由显式选择的普通 `ui` 插件提供；Core 不读取 UI 模块常量。
贡献方声明 `inject = (UI,)`，在自己的 `apply(ctx)` 中注册：

```python
from importlib import import_module
from agent.plugin_composition.ui import UI

inject = (UI,)

async def apply(ctx):
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__),
        requires=("shell.pages.v1",),
    )
```

`web` 和 `dashboard` 至少选一个。Web 路径必须位于贡献 Context 的固定代码制品；
Dashboard loader 必须定义在该制品中，返回的模块也必须属于同一制品。
延迟 loader 保留原包的 Python 类型身份，并让 provider 处理导入失败和资源取得。
`requires`、`provides` 和 `contract_digests` 是这次注册的领域参数。
provider 在 `SNAPSHOT_SEALING` 校验并封存目录；重复 provider、合同 digest 不匹配、
越界资源和无效 JS/CSS 都显式失败。未挂载的浏览器 mount 合同仍允许 consumer 自己等待，
不把缺少可选 mount 误判为缺少 Python UI 服务。

Dashboard 继续使用 `DashboardContext` 的 `require()`、`workspace_root()`、
`workspace_file()` 和 `workload_url()`；请求必须属于注册 Context 的实际 Root。
注册 Effect 拥有路由资源，关闭失败保留句柄供原 Effect 重试，不重放初始化。
初次导入失败仅允许 dashboard-only 插件暂不可用；配套 Web/API 不能半发布。
Web bootstrap 和 DashboardHost 从所选 Root 的 typed service 读取目录；
`RuntimeSnapshot` 不复制 Web/UI 字段，Core compiler 不解释 UI 合同。

```text
贡献插件 apply(ctx) ── UI.register ──┐
                                   ▼
                        本 Root 的 UI provider
                        ├── seal：Web 目录
                        └── Effect：Dashboard 资源
                                   │
              实际请求租约 ────────┘
```

### Mobile UI

同一个显式选择的 `ui` 插件提供 `UI_SLOTS`（保留字符串 `core.ui_slots`）。
SDK 的 `UiSlots`、`MobileUiRegistry` 是窄 Protocol，具体注册表和资源校验在 provider 内。
贡献插件仍在 `apply(ctx)` 调用 `ctx.require(UI_SLOTS).register_mobile(...)`，
使用原 `MobileUiDefinition`、navigation、slots 和同步 query/available 合同。

```text
贡献 Context ── register_mobile ── UI provider 的注册 Effect
                                      │ SNAPSHOT_SEALING
                                      ▼
                             本 Root 的封存目录
                                      │
                   Mobile HTTP/RPC 域消费者按实际 Root 读取
```

provider 校验贡献方属于同一 Root 和服务，资源路径仍固定在该 Context 的代码制品中。
目录与服务均带实际 Root token；域消费者拒绝借用另一 Root 的服务或目录。
Core compiler 不再读取、冻结或复制 Mobile 目录，RuntimeSnapshot 不含 Mobile UI 字段。
注册 Effect 关闭只解除内存归属，不删除代码、plugin-data、消息或历史记录。

`PluginMobileUiProvider` 继续承担已有 RPC 线程池、容量、超时和请求租约；
MobileHTTP/RPC 的 revision、摘要、slot、授权和响应格式不变。
Manager 的既有 `core.mobile_ui.v1` 请求 adapter 接线仍保留，但不再检测
`inject(UI_SLOTS)` 或创建业务注册表。仓库内 Akasha 的真实安装组合已显式选择 `ui`。

## 6. Generation 与 candidate

```text
┌────────────────────┐      ┌────────────────────┐
│ 固定代码与配置归档 │ ───▶ │ 独立候选 Root 检查 │
└────────────────────┘      └──────────┬─────────┘
                                       │ 调用程序授权晋升
                                       ▼
┌────────────────────┐      ┌────────────────────┐
│ 新正式 Root 初始化 │ ◀─── │ 候选退出，旧组排空 │
│ 完成前保持关闭接纳 │      │ 并成功释放旧 Root │
└──────────┬─────────┘      └────────────────────┘
           ▼
┌────────────────────┐      ┌────────────────────┐
│ 完整 stable 提交   │ ───▶ │ 开放新请求的 lease │
└────────────────────┘      └────────────────────┘
```

- Candidate 与正式 Root 使用同一组精确归档，但模块、Scope 和 generation 都重新创建，
  不把候选实例或目录改作正式实例。候选从独立空数据环境开始，底座不复制正式业务库或目录。
- Root 不能自行晋升。调用程序拥有业务验证与正常终态/未撤销授权；底座检查候选和基线，
  只在初始化成功后提交完整 stable。重启只读取该记录，不追随尚未晋升的源码或安装指针。
- 整组换代先等待旧请求结束，再释放旧资源；不是逐插件无停顿替换。写入结果不确定时保留
  实际 owner 并关闭接纳，不能自动重放外部启动或声称已回滚。代码恢复不回滚插件数据。
- Workspace path 是显式授予正式数据 owner 的高权限能力，不应替代窄 Service；candidate
  只得到声明路径在独立 workspace 内的位置，所需数据由插件自行准备，不是正式目录的副本。
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

组件归档的 `runtime.binding_api` 当前为 3。Core 在打开任何组件源码前核对完整
闭包的接口版本和 Python tag；ABI 1、2 明确不兼容，不能混用新接口或从当前插件补齐。
原 descriptor、源码树、binding 引用和已开始效果的回执保持原位，旧归档需要原 Core
版本及其安装环境恢复。该接口版本与 Python environment descriptor 的版本独立。
新版本创建的归档仍能在原安装移除后，按原配置与 generation 闭包恢复。

### 固定配置输入与凭据（0071）

显式配置程序使用 `agent.plugin_composition.config_input` 的 `save_config(data_dir, mapping)`。
文件为 `plugin-data/<owner>/config.input.json`，使用既有 tagged JSON codec 保存普通值和
`CredentialRef`。Core 只加载、固定和传递映射，不解析插件字段；配置程序和 `apply(ctx)` 各自
校验自己边界的输入。不要把明文放进普通映射，公共 writer 不猜测字段名。

```text
┌────────────────────────────────────┐
│ configure：解释字段、接收明文       │
└────────────────┬───────────────────┘
                 ▼
┌────────────────────────────────────┐
│ save_credential → 不可变引用        │
│ save_config → 无明文的固定映射      │
└────────────────┬───────────────────┘
                 ▼
┌────────────────────────────────────┐
│ Core 固定输入 → 插件请求凭据短租约  │
│ candidate 没有正式凭据解析权        │
└────────────────────────────────────┘
```

`save_credential(data_dir, value)` 只增加私有版本，返回现有 `CredentialRef`；调用程序再把引用
放入自己的配置字段。凭据保存在 `<workspace>/.plugin-credentials/<owner>/`，目录 0700、文件
0600；引用固定随机 ID 和内容版本。Core factory 只接受当前 owner 的固定输入中出现的引用，
请求别名不解释为配置路径。普通插件用 `CREDENTIALS.open(ctx, refs)`，Channel adapter 用同一
factory 合同；Channel host 不再提取配置字段或维护第二份凭据路径表。

每次打开租约先核对配置版本、凭据内容版本和撤销标记；已打开的租约保留其取得的值，关闭时
清空。`revoke_credential(data_dir, ref)` 只增加撤销标记，不删除历史版本。新配置原子替换前把
旧输入保存在私有 `config-history/`。凭据、撤销标记、配置历史没有自动 GC；恢复必须一起保留
私有目录和对应配置输入。Models 的连接凭据与刷新协议保持自己的 owner，不使用这份存储。

普通读取与写入只识别准确的旧入口 `config.local.toml`，存在时明确要求升级。
缺少固定输入时返回空映射，与业务目录是否存在或含哪些数据无关；安装不写空配置占位文件。
普通路径不递归扫描业务目录，不按备份文件名判断兼容性。只有显式升级工具收集命名备份。
Telegram Channel 和两个 Sender 的 `configure.py --upgrade` 由插件解释旧 TOML；其他明确不含
秘密的配置可离线运行 `python -m scripts.upgrade_plugin_config --data-dir <path> --no-secrets`。
执行插件配置程序时沿正式安装环境提供 `AKASHIC_PLUGIN_DATA_DIR`，Channel 程序同时使用
`AKASHIC_SETUP_CONFIG_PATH=<data-dir>/config.input.json`。日常 setup 不自动选择升级模式。
配置向导使用安装解释器和依赖，并追加宿主 Python 导入路径、预载共享 writer，再执行制品内
`configure.py`；不依赖工作目录或 `PYTHONPATH` 提供宿主 SDK。手工调用配置程序也须使用能
导入宿主 SDK 及插件依赖的环境。

升级先在私有 `upgrades/<id>/original/` 保存旧配置与命名备份，再发布固定输入，最后把原件
移入同一恢复点的 `retired/`。中断后保留全部材料；若新输入与旧入口同时存在，继续拒绝启动，
操作者须核对恢复点后完成移动或恢复原配置，不能重复覆盖；仅剩命名备份不构成运行栅栏。旧 artifact、归档和 binding 不改写；
含已删除 TOML 字段的旧安装必须显式重装。历史 Yoyo 脚本保持原字节，若它产生旧配置，随后仍须
经过显式配置升级，不能把旧输出直接作为新输入。

候选不复制私有凭据根，workspace root/file 授权也不能授予它。新格式不是任意 plugin-data 的
“无秘密证明”：旧的任意命名备份、模型自有存储和其他业务私有数据仍需要其 owner 的隔离/排除
协议。不能把未知旧目录写一个空输入就声称验收通过。同进程 Python 插件仍属于受信任代码；
这些窄接口和复制限制不是操作系统文件沙箱。本层只完成代码与静态 diff 检查，行为验证另行执行。

日常向导、发布 profile、旧渠道升级命令、Docker 调试辅助写入器、共享 fixture 和原先列出的
11 个非秘密测试输入已迁移。迁移历史与备份合同中的旧 TOML 样本保留。当前仍存在的 Core
候选复制职责须由整体换代层删除；本层不以文件名扫描充当数据 owner 授权，因而不保证旧命名
备份不会被既有复制路径带入候选。候选 broker 禁止正式凭据解析的边界仍独立生效。

SDK 导入路径静态链路：向导从自身 `__file__` 定位宿主源码根，将该根及父进程依赖路径作为
参数传给安装解释器；`-I -B -c` 启动代码显式加入这些路径，先导入共享 writer，再执行制品内
配置程序。该链路不依赖 `PYTHONPATH` 或空的 `sys.path` 项；依赖版本的实际导入仍未运行验证。
