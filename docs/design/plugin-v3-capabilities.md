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
Core 不读取它。无配置时输入为空对象。固定输入中的凭据仍使用引用；Models 自有连接
按 PLG-001 由模型 owner 提供给当前调用，不受此配置存储协议解释。
启用条件写在普通 `apply` 分支中；所有贡献走同一注册路径，没有另一个 `is_active` 协议。

首次业务配置使用插件自己的 Web 页面/API，并可向普通 onboarding provider 注册步骤。`main.py setup` 仅初始化 Core，不再发现或执行 `configure.py`。自身配置应用端口、首次 `initial_config.json` 与回执合同见[引导设计](plugin-onboarding-projection.md#11-实现决断与交付边界)。已有固定输入优先，新安装缺省与旧安装语义不能混用。

旧 Python 转发模块已删除：归档使用 `agent.plugin_composition.archive`，进程使用
`agent.process_runtime`，Shell 选择使用 `agent.plugin_composition.shell_runtime`，Host Bridge
文件能力使用 `agent.host_bridge.filesystem`。旧 `agent.tools.*` 转发入口和组合包下的进程、
工具上下文别名不再存在。无当前消费者的旧 Tool 基类、全局工具事件、ToolExecutionContext、
ToolGrant 和 TurnExecutionScope 也已退役；工具调用由 `CallSource`、`ToolRef` 和 Task 表达。
这些删除不改变进程 owner、当前 scope 或持久归档内容。

### 当前服务合同

同一能力只发布当前设计，不为旧插件保留导出、别名或同步适配器。插件必须与目标 Core
一起准备并验证；缺少依赖时拒绝激活，旧归档不会把新服务降级。ABI 的 `api_version = 3`
与各服务名称中的版本分别描述入口形状和业务合同，不能按数字统一改名。

| 能力 | 当前入口 |
|---|---|
| Channel / Source | `channel.input.v2`、`sources.v5`、`source.session.v4`、`source.changed.v3` |
| 来源控制 / 完成 | `source.check.v2`、`source.interrupt.v2`、`conversation.complete.v2` |
| 回复 / 执行 | `reply.execute.v4`、`reply.program.v3`、`react.ordered-start.v2`、`tools.program.v2` |
| 上下文材料 | `context.materials.v4`；注册必填 `kind`，仅按 `exclude_kinds` 选择 |
| Delivery | `delivery.guarded-start.v1`；持久准备可等待，`start_guard` 保护首次持久开始 |
| Drift | `drift.proposals.v2`、`drift.wake.v2`、`drift.delivery.v2`，全部可等待 |

```text
┌────────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│ 当前插件 apply(ctx)│ ─▶ │ 当前服务契约       │ ─▶ │ 原状态 owner     │
└────────────────────┘    └────────────────────┘    └──────────────────┘
                          缺失即拒绝激活              消息与回执保留
```

退役运行接口不授权改写历史事实。消息、业务回执、选择记录和归档不因 API 清理而减少；
读取边界保留已存在事实的格式解释，显式数据迁移仍由对应 owner 负责。新消息照常追加，
领取、配置和回执仅沿已有 owner 协议更新；本次没有新增逻辑失效或物理删除路径。
代码可由 Git 恢复，正式状态恢复仍需对应 workspace 的一致备份。旧 Core 与当前插件
不能任意混装，源码检查通过也不代表正式 Root 已切换。

### Python 安装输入

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
环境只由安装器创建，加载和换代不准备环境，包括空 requirements 文件。
源码插件的纯进程内能力可以直接装配；实际 Python 命令缺少固定环境时明确失败。

### 身份读取

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
恢复材料读取，采用新格式须从已更新源码显式重装；binding 的来源证据保持原身份；业务选择由当前兼容实现解释，不兼容明确失败。

## 2. 组合原子能力

初始化约束在 `apply` 或对应 provider 的实际注册中检查并抛出错误。底座不调用另一个
`static_semantic_checks` 自测入口，也不把报告存入运行 generation；诊断报告只记录真实装配步骤。

每次 `apply` 都属于一个 generation-bound Fiber。下列注册和任务归该 Fiber 所有，
正式进程只有一个 live Root。局部换代排空受影响的依赖者和 provider，再安装新 Fiber；无关分支保持运行。

| 原子能力 | 最短用法 | 语义 |
|---|---|---|
| 硬依赖 | 模块级 `inject = (KEY,)` | 全部 Service 可用时根 Fiber 才激活 |
| 可选依赖 | `await ctx.inject((KEY,), child)` | 独立子 Fiber 随依赖出现、消失而激活或退出，不阻塞父级计算 |
| 子 Fiber | `await ctx.mount(child, name="worker")` | 分开生命周期、Health、Effect 和依赖；名字在同一父级唯一，path 标识完整层级 |
| 提供 Service | `await ctx.provide(KEY, value)` | 当前 Fiber 成为该 key 的活动 provider |
| 读取 Service | `ctx.require(KEY)` / `ctx.get(KEY)` | 只允许声明依赖或自身提供的 key；未声明读取失败，已声明但缺席的 get 返回 None |
| 有界借用 | `with ctx.borrow(KEY) as service` | 按调用选择可选 provider 并保护其寿命；缺席返回 None，不得越过 scope 使用服务 |
| 执行入口 | `ctx.entrypoint(handler)` | 同步/异步调用均由框架接纳实际 provider；不包装生成器 |
| Effect | `await ctx.effect(setup, label="client")` | `setup`（可异步）只返回一个 cleanup 或 `None`；不解释 iterable 或生成器。Fiber 逆序关闭，成功才解除 owner；失败保留句柄与依赖供显式重试 |
| 后台任务 | `await ctx.spawn(run(), name="poll")` | 任务绑定实际 owner；失败进入 Fiber 状态，卸载先取消并等待任务，再排空外部调用 |
| Health | `health = await ctx.health("upstream")` | `degrade(reason)` / `recover()`；required 项参与 readiness |
| Incident | `ctx.report_incident("fetch", "timeout")` | 记录历史失败，不隐式改变 Health |
| 数据根 | `ctx.data_root` | 插件 owner 的正式数据根；换代不回滚或自动减少数据 |
| Workspace 路径 | `ctx.workspace_root("memory")` | 返回模块预先声明的原生 `Path`；Core 校验路径归属，但不拦截写入 |
| 运行身份 | `ctx.runtime`、`ctx.generation_id` | plugin、artifact、generation 和目录身份 |
| 短运行作用域 | `async with ctx.runtime_scope(): ...` | 保护实际 Fiber owner；同步入口使用 entrypoint |
| 跨 task 作用域 | `scope = ctx.capture_runtime_scope()` | 显式捕获 owner scope；调用者负责移交与关闭 |
| 诊断 | `ctx.diagnostics.operation(...)` | 记录 generation-bound 边界和有限指标 |

跨插件 Service 使用公共模块中的版本化结构合同（下例声明位于公共合同模块）：

```python
from typing import Protocol
from agent.plugin_composition import Context, ServiceKey

class Greeter(Protocol):
    def greet(self, name: str) -> str: ...

GREETER = ServiceKey[Greeter]("example.greeter.v1")

async def apply(ctx: Context) -> None:
    await ctx.provide(GREETER, MyGreeter())
```

双方从同一公共模块导入 key 与 Protocol，通过 `inject` 和 `ctx.require()` 连接，不能 import 对方实现。
`ServiceKey` 的值类型不变型，提供者必须符合明确的合同。`scripts/plugin_boundary.py check` 拒绝重复声明；
`python scripts/plugin_boundary.py catalog` 输出能力、提供与消费的静态位置，动态激活以运行时组合图为准。

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
`RUNTIME_STARTED` 在相应 activation 具备启动条件后启动工作，`RUNTIME_STOPPING` 在其停止前收束工作；
这些信号不是全局 readiness 或第二个提交点。`SOURCE_CHANGED`、`EVENTMAIL_CHANGED` 和
`DRIFT_CHANGED` 等来源 signal 由各自插件声明和发布；它们不组成 Core 业务事件表。新插件定义自己的
typed key 或窄 `ServiceKey`，不依赖已退役的 Core 业务事件名。

## 4. Runtime Service 与插件能力

Runtime Service 通过 `inject` 和 `ctx.require(KEY)` 连接；插件能力由拥有该 key 的插件注册，声明型注册本身是 Effect。

### 4.1 人与 Agent 的入口

| Key | 主要方法 | 用途 |
|---|---|---|
| `COMMANDS` | `register(ctx, CommandDefinition(...))` | 显式 `commands` provider 拥有人类命令、alias、注册与执行；消费者声明硬依赖 |
| `TOOLS` | `register(...)`、`bind(...)`、`open(...)` | `plugins.tools` 的工具描述、参数准备、exact binding 与执行入口 |
| `UI_SLOTS` | `register_plugin_ui(ctx, definition, query=...)` | Web 插件界面的资源、查询和导航 |
| `CHANNELS` / `CHANNEL_INPUT` | 注册实际 factory，按绑定调用入站入口 | 显式 `channels` provider 拥有连接、接纳、原绑定发送与恢复；来源插件拥有输入消费 |
| `DELIVERY` / `DELIVERY_READ` | 打开发送 admission 或只读历史 | Delivery 发送、恢复和查询 |

旧 Core `TOOL_CATALOG` 及其注册、冻结和快照装配已删除。旧 `DELIVERIES`、
`DURABLE_DELIVERIES` ServiceKey 和注入也已退役。旧持久投递记录与恢复实现保留，
Manager 不扫描这些业务记录来判断业务兼容性；移除检查不会删除记录、结算或重发旧效果。
工具消费者通过公共 `agent.plugin_contracts.tools` 中的 key 和结构接口协作，
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
MCP 的端口引用直接使用 Workload/Process 返回的句柄，provider 检查 owner。
Python 命令由宿主绑定固定制品环境；CredentialRef broker 校验 owner 与固定输入授权，
Models 自有连接沿其独立 owner 协议接续。
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
模型选择能力 `models.selection.v1` 由 owner 和消费者共同导入公共合同 key。角色是字符串，
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
provider 在注册时校验资源和同 Root 归属；重复 provider、合同 digest 不匹配、
越界资源和无效 JS/CSS 都显式失败。未挂载的浏览器 mount 合同仍允许 consumer 自己等待，
不把缺少可选 mount 误判为缺少 Python UI 服务。

Dashboard 继续使用 `DashboardContext` 的 `require()`、`workspace_root()`、
`workspace_file()` 和 `workload_url()`；请求必须属于注册 Context 的实际 Root。
注册 Effect 拥有路由资源，关闭失败保留句柄供原 Effect 重试，不重放初始化。
初次导入失败仅允许 dashboard-only 插件暂不可用；配套 Web/API 不能半发布。
Web bootstrap 和 DashboardHost 从所选 Root 的 typed service 读取目录；
宿主投影不复制 Web/UI 注册状态，Core compiler 不解释 UI 合同。
目录只包含已初始化且 ACTIVE 的贡献方；无关安装或卸载不阻止 bootstrap。
浏览器按 snapshot 与 catalog identity 判断目录是否变化，manager 的 `updating`
只描述操作进行中，不使未变的 UI 失效。请求继续核对贡献方 generation 与权限。
Channel 激活只声明启动实际需要的硬依赖；例如回复状态在请求中通过已声明的
可选能力借用，缺失时该请求明确失败，设置与其他只读入口继续服务。

```text
贡献插件 apply(ctx) ── UI.register ──┐
                                   ▼
                        本 Root 的 UI provider
                        ├── Effect：Web 活动目录
                        └── Effect：Dashboard 资源
                                   │
              实际请求租约 ────────┘
```

### 插件 UI

同一个显式选择的 `ui` 插件提供 `UI_SLOTS`（保留字符串 `core.ui_slots`）。
SDK 的 `UiSlots` 是窄 Protocol，具体注册表和资源校验在 provider 内。
贡献插件在 `apply(ctx)` 调用 `ctx.require(UI_SLOTS).register_plugin_ui(...)`，
使用 `PluginUiDefinition`、navigation、slots 和同步 query/available 合同。

```text
贡献 Context ── register_plugin_ui ── UI provider 的注册 Effect
                                      │ 注册与释放
                                      ▼
                             本 Root 的活动目录
                                      │
                   Web HTTP/RPC 持实际 provider scope 读取
```

provider 校验贡献方属于同一 Root 和服务，资源路径仍固定在该 Context 的代码制品中。
目录与服务均带实际 Root token；域消费者拒绝借用另一 Root 的服务或目录。
Core compiler 不读取、冻结或复制插件 UI 目录，宿主只持请求 adapter，不拥有插件 UI 注册状态。
注册 Effect 关闭只解除内存归属，不删除代码、plugin-data、消息或历史记录。

`LivePluginUiProvider` 承担 RPC 线程池、容量、超时和请求租约；
Web HTTP/RPC 使用 revision、摘要、slot 和有界结果。
宿主装配模块提供 `core.plugin_ui.v1` 请求 adapter，但不检测
`inject(UI_SLOTS)` 或创建业务注册表。仓库内 Akasha 的真实安装组合已显式选择 `ui`。

## 6. Generation 与单 Root

```text
┌────────────────────┐      ┌────────────────────────┐
│ 安装固定制品和环境 │ ───▶ │ 原子提交 PluginSelection │
└────────────────────┘      └────────────┬───────────┘
                                        ▼
┌────────────────────┐      ┌────────────────────────┐
│ 原 Root 局部应用   │ ◀─── │ 受影响 owner 排空与释放  │
│ 无关分支持续运行   │      │ 清理失败保留 owner       │
└────────────────────┘      └────────────────────────┘
```

- `PluginSelection` 是持久选择的唯一提交点；accepted 不等于 active。重启使用已提交选择，
  不跟随未提交的安装输入。没有候选 Root 的晋升或 stable/latest 双视图。
- Generation 记录实际代码、配置和环境来源。局部排空由组合内核拥有，安装与应用恢复由 Manager 拥有；
  `agent.plugins.host` 用明确宿主端口装配消息、客户端投影与安装接口，不接收 Manager 实例。
- `binding` 固定业务选择及来源证据，恢复调用当前兼容 provider；缺失或不兼容明确失败。
  正在运行的调用持有实际 owner，排空前不会释放其资源；历史归档不启动第二个执行图。
- 外部效果未知时保留原状态和回执，不自动重放，不把内存恢复说成外部回滚。
  Workspace 路径只授予已声明的数据 owner；换代不复制或回滚正式数据。
- 普通卸载保留 plugin-data；归档、消息与历史记录没有自动 GC。安装清单只接受
  独立 `[plugins."<id>"]` 条目；旧 `[packages]` 分组不再解释。

## 7. 选择能力

1. 同一插件内部拆生命周期：`ctx.mount()`。
2. 插件之间共享行为：版本化 `ServiceKey` + `provide/require`。
3. 已提交事实的变更通知：由事实 owner 定义窄 typed signal。
4. 对人暴露动作：`COMMANDS`；对模型暴露动作：`TOOLS`。
5. Message、Context、Turn projection、Model、Tool、Content 和 Delivery 通过各自插件 Service 组合。可选插件标签写入普通 Message metadata，使用 [SES-009](../projectneed.md#ses-009-插件附加信息使用普通-message-metadata) 与[消息附加信息合同](0902-reviewed-v4.md#34-message-metadata-的实现与迁移)，不注册新的内容块。
6. 长时或可恢复工作：`TASKS`、`TIMERS`。
7. 外部进程、MCP、容器：`MANAGED_PROCESSES`、`MCP_SERVERS`、`WORKLOADS`。
8. 找不到匹配能力时先定义窄 Service，不给 Manager 增加新的固定插件方法。


### 工具绑定与异步资产读取

`await tools.bind(...)`、`await tools.bind_scoped(...)` 与 `await tools.bind_saved(...)`
在返回 binding ID 前完成准备。`capture` 可返回 JSON mapping 或可等待的 mapping；
普通同步回调仍在调用者任务内执行，文件密集型回调自行异步等待文件工作。
调用期间保留工具、参数准备和授权贡献者的作用域；准备失败或取消不提交该 binding。
原先同步调用 `bind` / `bind_saved` 的插件必须随提供方一起更新为 `await`。
绑定的持久 metadata 格式保持不变。

`async with installed_assets.open(ctx, category="skills") as assets` 固定当前目录与
资产贡献者的作用域；目录只能在作用域内使用。后台文件工作必须实际结束后再退出，
不能把等待取消当作物理工作结束。技能归档按既有合同追加并校验，不新增自动删除。

`core.common.file_io.run_file_io` 是公开的窄文件工作入口，复用 HostBridge 原有
实现：每个事件循环最多四项文件工作并行，取消后等待实际线程结束，保留文件错误。
它只拥有并发上限与取消排空，不拥有 Context、权限、注册或 binding；这些作用域仍由
调用方持有。传入函数不得访问 Context/Fiber，绑定提交也不得移入文件线程。

### 归档接口版本

组件归档的 `runtime.binding_api` 当前为 3。历史格式、Python 环境及源码树只用于校验来源证据；
读取旧格式需要原工具及完整恢复材料，不自动改写。业务 binding 恢复不从归档启动历史 Root，
而是将原业务选择交给当前 provider 的 bind 合同。当前 provider 缺席或不能解释选择时明确失败。
该接口版本与 Python environment descriptor 的版本独立，原始记录与外部效果回执保持原位。

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
私有目录和对应配置输入。Models 的连接凭据与刷新协议保持自己的 owner，不使用这份存储；
当前调用默认复用已有模型设置和凭据，不要求独立账号，见 PLG-001 与 0071。

普通读取与写入只识别准确的旧入口 `config.local.toml`，存在时明确要求升级。
缺少固定输入时返回空映射，与业务目录是否存在或含哪些数据无关；安装不写空配置占位文件。
普通路径不递归扫描业务目录，不按备份文件名判断兼容性。只有显式升级工具收集命名备份。
四个 Telegram/QQ channel、sender 的 `upgrade_config.py --data-dir PATH` 由插件解释旧 TOML；其他明确不含秘密的配置可离线运行 `python -m scripts.upgrade_plugin_config --data-dir PATH --no-secrets`。升级使用可导入宿主 SDK 与插件依赖的安装环境，必须显式指定准确的私有数据目录；setup 和 onboarding 均不自动升级。

升级先在私有 `upgrades/<id>/original/` 保存旧配置与命名备份，再发布固定输入，最后把原件
移入同一恢复点的 `retired/`。中断后保留全部材料；若新输入与旧入口同时存在，继续拒绝启动，
操作者须核对恢复点后完成移动或恢复原配置，不能重复覆盖；仅剩命名备份不构成运行栅栏。旧 artifact、归档和 binding 不改写；
含已删除 TOML 字段的旧安装必须显式重装。历史 Yoyo 脚本保持原字节，若它产生旧配置，随后仍须
经过显式配置升级，不能把旧输出直接作为新输入。

私有凭据根由 broker 拥有，workspace root/file 授权不能授予它。新格式不解释
任意 plugin-data、模型自有存储或旧备份；这些数据由各自 owner 使用和接续。
不能把未知旧目录写一个空输入就声称验收通过。同进程 Python 插件仍属于受信任代码；
这些窄接口不是操作系统文件沙箱。本地验证不代表正式环境的安装与发布验收。

日常向导、发布 profile、旧渠道升级命令、Docker 调试辅助写入器、共享 fixture 和原先列出的
11 个非秘密测试输入已迁移。迁移历史与备份合同中的旧 TOML 样本保留。Core 的正式数据
复制职责现已删除；调用程序另行提供样本时仍须取得数据 owner 授权，不能根据文件名猜测
它不含秘密。凭据解析仍核对实际 owner 和固定配置授权。

SDK 导入路径静态链路：向导从自身 `__file__` 定位宿主源码根，将该根及父进程依赖路径作为
参数传给安装解释器；`-I -B -c` 启动代码显式加入这些路径，先导入共享 writer，再执行制品内
配置程序。该链路不依赖 `PYTHONPATH` 或空的 `sys.path` 项；依赖版本的实际导入仍未运行验证。

## 模型内容投影

`CONTENT_VIEWS`（`models.content-views.v1`）由普通 Models provider 提供。注册纯
`prepare(messages, source, tools, seen)`，返回按 `(Message, part_index)` 取 `RenderedContent | None`
的函数；None 表示交给基础 renderer，同一位置两个贡献者处理时拒绝。`complete=True` 只用于
完整表达该内容块，节选和脱敏视图必须为 False。`seen` 来自同 source 已提交的真实模型响应，
不是调用次数或最近消息推断。注册返回 owner Effect，bind 固定活贡献者并持有 scope。

能力没有消息 writer、模型调用或工具执行口。`content_view` 是其普通消费者，仅另依赖 CONTENT
与 TOOLS。完整行为、范围和恢复约束见 [0081](../decisions/0081-content-views-keep-original-messages.md)。
