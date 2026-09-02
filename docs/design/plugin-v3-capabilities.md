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

## 3. Typed event

注册统一使用 `await ctx.on(KEY, listener)`。

| Key | 发布 | 失败与顺序 |
|---|---|---|
| `EmitEventKey[T]` | `ctx.emit(KEY, payload)` | 同步、按注册顺序、首个失败立即传播 |
| `SerialEventKey[T, R]` | `await ctx.serial(KEY, payload)` | 逐个等待；只有显式 `Bail(value)` 短路 |
| `ParallelEventKey[T]` | `await ctx.parallel(KEY, payload)` | 仅 async listener；全部 settle 后聚合失败 |
| `TransformEventKey[T]` | `await ctx.transform(KEY, payload)` | 按顺序把同类型不可变值传给下一 listener |
| `ObserveEventKey[T]` | `await ctx.observe(KEY, payload)` | 全部 settle；普通失败隔离为 Incident |

Key 的结构与调度实现归 Core；事实只能由下列领域 owner 发布：

| Key | 时机 |
|---|---|
| `CONTEXT_PREPARED_EVENT` | Session 与上下文准备后 |
| `PROMPT_RENDER_EVENT` | Prompt 渲染前 |
| `AFTER_REASONING_PREPROCESS_EVENT` | 推理结果形成后、持久化前 |
| `AFTER_REASONING_CLEANUP_EVENT` | Core 清理阶段 |
| `AFTER_TURN_COMMITTED` | user/assistant 已原子提交 |
| `RUNTIME_STARTED`、`RUNTIME_STOPPING` | committed snapshot 启停 |
| `SNAPSHOT_SEALING` | candidate catalog 冻结前 |
| `RETRIEVAL_COMPLETED` | Akasha 插件检索完成；仅 Akasha 活动时发布 |
| `CONTEXT_PROJECTION_COMMITTED` | Compaction 插件完成上下文投影；仅该插件活动时发布 |

`MEMORY_WRITTEN` key 仍是公开结构合同，但当前 pure-V3 Core 没有生产者；Memory2 退役后不得把
“能注册 listener”误报成“线上会发布事件”。新增生产者必须由领域 owner 明确发布，不能恢复
EventBus 类型猜测桥。

## 4. Core Service 原子能力

所有 Service 先写入 `inject`，再用 `service = ctx.require(KEY)`。声明型注册本身是 Effect。

### 4.1 人与 Agent 的入口

| Key | 主要方法 | 用途 |
|---|---|---|
| `COMMANDS` | `register(ctx, CommandDefinition(...))` | 人类命令、alias 和 handler |
| `TOOL_CATALOG` | `register(ctx, PluginToolDefinition(...), handler)` | 模型 Tool |
| `UI_SLOTS` | `register_mobile(ctx, definition, query=...)` | Mobile 页面、查询和导航 |
| `CHANNELS` | `register(ctx, ChannelDefinition(...))` | inbound/outbound Channel blueprint |
| `DELIVERIES` | `send(...)` | 当前 Turn 内的一次投递 |
| `DURABLE_DELIVERIES` | `submit()`、`lookup()`、`resume()` | 三态、可恢复的外部投递 |

Tool 还支持纯 V3 的命名 handler：省略直接 handler，使用 `handler_export` 指向模块内
`async (context, arguments)`。Core 在 snapshot 编译时解析并校验 exact generation；它用于需要稳定
导出身份的已安装插件，不是 V2 fallback。

### 4.2 Turn、Session 与上下文

| Key | 主要方法 | 用途 |
|---|---|---|
| `SCOPED_TURNS` | `create_session()`、`ensure_session()`、`start()`、`read()` | generation-bound programmatic Turn |
| `CONTINUATIONS` | `submit(...)` | 向已有 Turn 提交继续输入 |
| `SESSION_READ` | `read(session_key)` | 只读既有 Session 投影 |
| `SESSION_COMPACTION_STORAGE` | `history_units()`、`prepare()`、`persist()` | Compaction 专用窄持久化边界 |
| `PROVIDER_REQUEST_PROJECTION` | `open_turn(...)` | provider request 的冻结投影和 retry gate |
| `CONTEXT_PROJECTION_FACTS` | `list_committed()`、`get_committed()` | 读取已提交上下文投影事实 |
| `INTERACTION_UNDO` | `bind_source_fence()`、`undo_latest()` | 显式撤销最近 interaction |
| `CONVERSATION_SEMANTIC_INTEREST` | `score(texts, cutoff=...)` | 统一语义兴趣评分 |

### 4.3 调度与外部运行

| Key | 主要方法 | 用途 |
|---|---|---|
| `TIMERS` | `schedule(deadline)` | Core-owned timer |
| `BACKGROUND_JOBS` | `register(ctx, BackgroundJobDefinition(...))` | trigger job 与命名 handler export |
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
| `CHAT_MODELS` | `execution()`、`independent_execution()`、`describe()`、`bind()` | 按冻结 revision 取得 chat model |
| `EMBEDDINGS` | `embed(texts)` | 统一 embedding space |
| `MODEL_CATALOG` | `snapshot()`、`validate_chat_selection()` | 模型和 connection 目录 |
| `MODEL_SETTINGS` | `discover()`、`apply(ModelChange)` | 原子修改模型设置 |
| `MODEL_DRIVERS` | `register(ctx, ModelDriverDefinition(...))` | Provider 注册模型 driver |

Provider 返回结构化 `ModelUsage` 和公开错误类型；未知能力保持 unknown，不用默认值伪装。

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
  复用 stable Root 或正式 plugin-data 宣称通过。
- Root 只生成能力，不能自行晋升。artifact、journal、stable/latest、parent Turn 授权和恢复由 Core
  publication plane 拥有。
- Workspace path 是显式授予正式数据 owner 的高权限能力，不应替代窄 Service；candidate
  只得到声明路径在 attempt workspace 内的副本。
- 普通卸载删除代码、manifest 和派生投影，默认保留 plugin-data。`manifest.toml` 只接受
  独立 `[plugins."<id>"]` 条目；旧 `[packages]` 分组不会展开、保留或静默忽略。

## 7. 选择能力

1. 同一插件内部拆生命周期：`ctx.mount()`。
2. 插件之间共享行为：版本化 `ServiceKey` + `provide/require`。
3. 已结算事实的一对多通知：`ObserveEventKey`。
4. 顺序策略：`SerialEventKey`；同类型改写链：`TransformEventKey`。
5. 对人暴露动作：`COMMANDS`；对模型暴露动作：`TOOL_CATALOG`。
6. 长时或可恢复工作：`BACKGROUND_JOBS`、`SCOPED_TURNS`、`DURABLE_DELIVERIES`。
7. 外部进程、MCP、容器：`MANAGED_PROCESSES`、`MCP_SERVERS`、`WORKLOADS`。
8. 找不到匹配能力时先定义窄 Service，不给 Manager 增加新的固定插件方法。
