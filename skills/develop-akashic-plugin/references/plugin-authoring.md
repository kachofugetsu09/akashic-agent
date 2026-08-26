# Akashic v3 插件编写参考

本文只描述当前 v3 authoring contract。字段和类型以同一 Core checkout 的 `agent/plugins/static_manifest.py`、`agent/plugins/composable.py` 与 `agent/plugin_composition/` 为准；可对照 `docker/debug/plugins/replay_debug/` 以及外部 `calendar`、`feishu`、`huayue-skills`、`citation`、`observe` 和 `proactive_feedback` source。

## 1. Source 与静态 manifest

外部安装包的根目录至少包含：

```text
plugin-repo/
├── akashic.plugin.toml
├── plugin.py
└── ...
```

最小 manifest：

```toml
schema_version = 1
name = "example"
version = "1.0.0"
api_version = 3
entrypoint = "plugin.py"
```

字段规则：

| 字段 | 规则 |
| --- | --- |
| `schema_version` | 整数 `1`。 |
| `name` | 小写字母开头，只能含小写字母、数字、`_`、`-`，最长 64。 |
| `version` | 非空版本字符串；应与 module 导出一致。 |
| `api_version` | 整数 `3`。 |
| `entrypoint` | artifact 内存在的相对 Python 文件；不能是 symlink、绝对路径或含 `..`。 |
| `[[python]]` | 可选。每项只写一个存在的 `requirements` 相对路径；Core 会在该目录准备独立 Python runtime。无依赖就省略。 |
| `[validation]` | 可选，唯一字段为 `exclude_data_paths`；只列 candidate 验证不应纳入比较的 artifact 相对数据路径。 |
| `[[processes]]` | 可选的 managed process 声明；`[[process]]` 与 `[[managed_processes]]` 是读取层支持的别名，新增 source 统一使用 `[[processes]]`。 |
| `[[mcp]]` | 可选的 MCP 声明；`[[mcp_servers]]` 是读取层支持的别名，新增 source 统一使用 `[[mcp]]`。 |
| `[channel_credentials]` | 可选。按 channel 名映射 config dotted paths，用于精确脱敏与凭据边界。 |

`[[processes]]` 字段：

```toml
[[processes]]
name = "example_api"
command = ["python", "mcp/run_server.py"]
cwd = "."
env = {HOST = "127.0.0.1"}
port_env = "PORT"
formal_port = 18000
readiness_path = "/health"
startup_timeout_seconds = 15.0
```

`name`、`command`、`cwd` 必须指向 artifact 内的安全相对内容；`port_env` 是大写环境变量名，不能覆盖 Core 保留变量；`formal_port` 为 `1..65535`；`readiness_path` 只能是本地绝对 URL path；超时范围是 `0 < seconds <= 300`。

`[[mcp]]` 字段：

```toml
[[mcp]]
name = "example"
command = ["python", "mcp/run.py"]
cwd = "."
required_tools = ["fetch_events", "ack_events"]
candidate_read_only_tools = ["fetch_events"]
endpoint_env = [{env = "PORT", process = "example_api"}]
candidate_env = {BACKEND = "recording"}
```

`env`、`candidate_env` 只能是字符串映射；`endpoint_env.process` 必须指向同一 manifest 中的 process；candidate 的非只读 MCP 不会因为声明存在就自动获准执行。

channel 凭据示例：

```toml
[channel_credentials]
feishu = ["appId", "appSecret", "app_id", "app_secret"]
```

每个 path 只能是安全的 dotted config path，不能与同 channel 或其他 channel 的前缀重叠。静态 manifest 在 import 前校验；未知字段、symlink、越界路径、依赖或端点声明错误都会 fail-loud。

## 2. Module namespace

`entrypoint` 指向的 module namespace 至少导出：

```python
from agent.plugin_composition import Context

api_version = 3
name = "example"
version = "1.0.0"
desc = "Example capability"
author = "Example Team"
inject = ()
skill_roots = ("skills",)
drift_skill_roots = ()
workspace_roots = ()
dashboard_module = None


async def apply(ctx: Context, config: object) -> None:
    """Register this generation's typed capabilities."""

    _ = ctx, config
```

`api_version/name/version/apply` 是安装身份和入口的硬合同；`desc`、`author`、`inject`、`skill_roots`、`drift_skill_roots`、`workspace_roots`、`dashboard_module` 按需提供。`apply` 必须精确接受两个无默认值的位置参数 `ctx, config`，不得增加 keyword-only、`*args`、`**kwargs` 或重排参数；返回同步值或 awaitable 均可。

`inject` 只列 apply 需要的 `ServiceKey`，且不得重复。可选能力在使用点 `ctx.get()`；必须存在的服务用 `ctx.require()`，Core 会在服务不活跃时明确失败。`workspace_roots` 只能是插件自有的单层目录名，不能声明 `plugin-data` 或 `runtime`。

## 3. Typed capability services

先在 namespace 中声明所需 key，再在 `apply` 中取得对应 service。常用公开 key 与定义类型如下：

| ServiceKey | 用途 | 典型 definition |
| --- | --- | --- |
| `TOOL_CATALOG` | 注册 Tool 的 schema、风险和 handler export | `PluginToolDefinition` |
| `COMMANDS` | 注册命令及结果类型 | `CommandDefinition` |
| `CHANNELS` | 注册 Channel descriptor，不在 apply 中开正式 ingress | `ChannelDefinition` |
| `MCP_SERVERS` | 注册 MCP command、工具和 candidate endpoint | `McpServerDefinition` |
| `MANAGED_PROCESSES` | 注册进程、端口和 readiness | `ManagedProcessDefinition` |
| `BACKGROUND_JOBS` | 注册 interval 或 programmatic Turn job | `BackgroundJobDefinition` |
| `UI_SLOTS` | 注册移动 UI 资源和查询 handler | `MobileUiDefinition` |
| `SESSION_READ` | 读取既有 Session 的脱离快照 | `SessionReadService` |
| `TEXT_EMBEDDING_SETTINGS` | 读取来源无关的文本向量端点配置 | `TextEmbeddingSettings` |
| `CONVERSATION_SEMANTIC_INTEREST` | 提供或消费会话语义兴趣评分 | `ConversationSemanticInterest` |

向量记忆实现不再读取 Core memory runtime。它们像其他插件一样组合 Prompt lifecycle、
`AFTER_TURN_COMMITTED`、`TOOL_CATALOG` 与 `UI_SLOTS`；需要互斥角色时可优先 `provide`
一个纯 marker ServiceKey，让 Composition Root 在任何存储副作用前 fail-loud。

`ctx` 还提供生命周期边界：`require/get/provide/effect/on/emit/serial/parallel/transform/observe/spawn`、`diagnostics`、`data_root` 和已声明的 `workspace_root(name)`。写入、监听、后台 task 与外部效果都应由当前 Fiber 持有的 Effect 或 service owner 管理；不要取得 Core repository、任意 SQL 或可变全局集合。

插件内部观测只使用身份已绑定的 `ctx.diagnostics`：

```python
with ctx.diagnostics.operation("calendar.refresh"):
    items = await refresh_calendar()
    ctx.diagnostics.measure("calendar.items", len(items))
```

Core 自动记录正式插件接入点，插件只补充自己拥有含义的内部阶段和有限数值。名称必须是稳定的
小写标识；measurement 只接受有限数字与固定 unit，不接受动态 label mapping。跨显式队列 handoff
时可以在生产者内 `captured = ctx.diagnostics.capture()`，再由同一 Fiber 在消费者内使用
`with ctx.diagnostics.resume(captured)`；不得把 capture token 持久化、跨 generation 或交给其他插件。
不要用 `ctx.observe` 上报诊断：ObserveEventKey 是 Core 向插件分发已结算领域事实的反向合同。
不要构造或导入 Core 内部 diagnostics concrete；受支持的身份只能由当前 `ctx` 发放。
插件不得直接连接 Loki、Prometheus 或 Grafana，也不得记录正文、Prompt、凭据或工具参数。

### 3.1 Tool

Tool 以不可变 `PluginToolDefinition` 注册，schema 使用严格 JSON object subset：

```python
from agent.plugin_composition import (
    Context,
    PluginToolDefinition,
    TOOL_CATALOG,
)

inject = (TOOL_CATALOG,)


async def inspect_repository(context: object, arguments: object) -> str:
    """Return a read-only inspection result."""

    _ = context
    return str(arguments)


async def apply(ctx: Context, config: object) -> None:
    _ = config
    await ctx.require(TOOL_CATALOG).register(
        ctx,
        PluginToolDefinition(
            name="inspect_repository",
            description="Inspect one repository without changing it.",
            parameters={
                "type": "object",
                "properties": {"repository": {"type": "string"}},
                "required": ["repository"],
                "additionalProperties": False,
            },
            handler_export="inspect_repository",
            risk="read-only",
            search_hint="when a repository inspection is requested",
        ),
    )
```

`name`、`description`、`handler_export`、risk 和 JSON schema 会在注册时校验；`risk` 只能是 `read-only`、`read-write` 或 `external-side-effect`。handler export 是 source-relative 的名字，不把闭包或跨 generation callable 放进 descriptor。schema、真实参数、返回值、异常和副作用都必须由 source test 与 attached child oracle 覆盖。

### 3.2 MCP 与 managed process

典型接线是一个 `apply` 同时注册 typed process 和 MCP：

```python
from agent.plugin_composition import (
    Context,
    EndpointEnv,
    MANAGED_PROCESSES,
    MCP_SERVERS,
    ManagedProcessDefinition,
    McpServerDefinition,
)

inject = (MANAGED_PROCESSES, MCP_SERVERS)


async def apply(ctx: Context, config: object) -> None:
    _ = config
    await ctx.require(MANAGED_PROCESSES).register(
        ctx,
        ManagedProcessDefinition(
            name="example_api",
            command=("python", "mcp/run_server.py"),
            port_env="PORT",
            formal_port=18000,
            readiness_path="/health",
            startup_timeout_seconds=15.0,
        ),
    )
    await ctx.require(MCP_SERVERS).register(
        ctx,
        McpServerDefinition(
            name="example",
            command=("python", "mcp/run.py"),
            required_tools=("fetch_events", "ack_events"),
            candidate_read_only_tools=("fetch_events",),
            endpoint_env=(EndpointEnv("PORT", "example_api"),),
            candidate_env={"BACKEND": "recording"},
        ),
    )
```

`apply` 只登记声明；Core 负责 Fiber readiness、隔离端口、MCP 启停、快照和 cleanup。`managed process` 的服务进程和 MCP 必须真正读取 `port_env`；忽略注入端口或 readiness 失败不能降级为成功。

### 3.3 Channel、事件和 UI

Channel 通过 `ChannelDefinition` 声明 `capabilities`、`factory_export`、可选 inbound identity 与 `credential_paths`；factory 必须从同一 source generation 解析，candidate 不持有正式 token 或 ingress ownership。

事件使用 typed event key：`ctx.on(key, listener)` 返回由当前 Fiber 持有的 Effect；同步传播使用 `emit`，有序结果使用 `serial`，并行使用 `parallel`，类型变换使用 `transform`，只读观察使用 `observe`。不要自行建立 priority、listener DAG 或第二个事件总线。

移动 UI 使用 `UI_SLOTS.register_mobile(ctx, MobileUiDefinition(...), query=...)`；资源路径必须在 source 内，RPC 边界校验 method、payload、session identity 和返回体大小。查询结果是 generation 投影，不是服务端权威事实。

### 3.4 Skill 与 workspace data

```text
skills/<skill-name>/
├── SKILL.md
├── references/   可选
├── scripts/      可选
└── assets/       可选
```

在 namespace 中声明 `skill_roots = ("skills",)` 或 `drift_skill_roots = ("drift/skills",)`。每个 Skill 目录必须有 `SKILL.md`，frontmatter 至少有 `name` 和 `description`；同一 generation 内名称不能冲突。Core 会从 source snapshot 构建 catalog，workspace 软链接只是可重建投影，不是 canonical source。

需要持久状态时使用 `ctx.data_root` 或已声明的 `ctx.workspace_root("<name>")`；candidate validation 不得写正式 Session、memory、plugin-data 或外部服务。卸载代码默认保留 plugin-data，删除数据需要另一个明确且可恢复的用户操作。

## 4. 主动能力也组合普通服务

来源轮询使用 `TIMERS` 加来源私有 store，离散事实提交 `CONTENT`，当前状态通过插件私有
cache 暴露给 Wake，候选行动提交 `DRIFT`，需要完整推理时由 `BACKGROUND_JOBS` 创建普通
programmatic Turn。Core 不提供 proactive catalog、私有 lifecycle family 或 MCP 聚合桥。

## 5. Source 验证清单

```text
manifest parse → module identity → apply(ctx, config) signature
      → typed declaration/schema/readiness
      → Skill/MCP/Tool source behavior
      → candidate attached child oracle
      → write set、cleanup、generation identity
```

测试应从 source commit 运行，至少确认：未知 manifest 字段、路径越界、缺失 root、重复 service/name、错误 schema、错误 apply 参数、服务未就绪和 Fiber cleanup 都真实失败；不能只断言文件存在或返回字符串。
