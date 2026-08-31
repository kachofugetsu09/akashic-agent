# 插件 v3 MCP / managed process capability 任务合同

- 状态：Core capability complete / workspace MCP compatibility removed / Calendar consumer Gate pending
- 日期：2026-08-16
- 实现提交：`8653bab0`
- 清单：C12、C13、C22
- 首个真实 consumer：Calendar MCP
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-014、WSP-001～WSP-005、TST-001～TST-008
- 独立评审：Root-local registry、Host/route、journal、取消与 shutdown recovery 经 Terra xhigh 终审无 P0/P1；Calendar 插件迁移尚未开始

## 1. 目标与边界

Core 提供两个 generation-scoped 接入点：

```text
MCP_SERVERS       = ServiceKey("core.mcp_servers")
MANAGED_PROCESSES = ServiceKey("core.managed_processes")
```

插件在 `apply(ctx, config)` 中注册不可变声明；注册由调用 Fiber 的 Effect 拥有。Core 在 candidate
Root settle 后冻结 descriptor，再用现有 MCP JSON-RPC host 与 managed-process host materialize
进程、端口、readiness、工具目录和清理。插件拥有领域命令、配置、required/read-only tool 选择及
自身数据 schema；Core 拥有进程组、候选隔离、端口、发布、lease、drain 和失败恢复。

```text
plugin Fiber ── Effect register ──► candidate-local declarations
                                            │ freeze descriptors
                                            ▼
Core validation host ── start/handshake/ready ──► candidate snapshot
                                            │ seal + formal rebuild
                                            ▼
Core production host ── exact descriptor ──► stable snapshot / tool routes
```

本任务不迁移 proactive source，不执行 Calendar 旧数据搬运，不调用真实 OAuth/远端写操作，也不暴露任意
`create_subprocess_exec` 给插件。独立 workspace MCP watcher/admin 兼容岛已删除；所有运行时 MCP 必须经静态
manifest、exact Root registry 和 generation host。

Core seam 的 `semantic_delta: none`；Calendar 迁移的 data/port 行为另行核对。旧 v2 声明在迁移期仍由
adapter 冻结。同一 Root/registry 内的 v2/v3 server/process 名称冲突 fail-loud；stable 与 candidate
跨 generation 使用相同 public name 是正常替换，运行 owner 使用 `name@generation_id` 区分。

## 2. 公开声明

### 2.1 MCP

```python
MCP_SERVERS = ServiceKey[PluginMcpServers]("core.mcp_servers")

async def apply(ctx, config):
    registry = ctx.require(MCP_SERVERS)
    await registry.register(
        ctx,
        McpServerDefinition(...),
    )

McpServerDefinition(
    name="calendar",
    command=("python", "mcp/run_mcp.py"),
    cwd=".",
    env={},
    required_tools=("list_events",),
    candidate_read_only_tools=("list_events",),
    endpoint_env=(EndpointEnv("PORT", process="calendar_api"),),
    candidate_env={"CALENDAR_BACKEND": "recording"},
)
```

`await PluginMcpServers.register(ctx, definition) -> None` 只接受调用 Fiber 自己的 Context；它在内部
创建 registration Effect 与 required `HealthHandle`，不返回可由插件提前关闭的 Effect，也不提供公共
`unregister`。`freeze() -> McpServerRegistry` 返回 immutable `definitions/descriptors/catalog_digest`；
freeze 后插件侧 register/mutation fail-loud，Root dispose 仍通过内部 Effect 反向注销 descriptor/health，且不重新
开放 frozen registry。Core 只从 frozen registry 取得 materialization input，插件不取得
`McpClient/McpGeneration/McpRoute`。Core 内部 route factory 是唯一运行调用面：

```python
async with mcp_routes.route_for(
    snapshot_lease,
    plugin_id,
    generation_id,
    public_name,
) as route:
    result = await route.call(tool, arguments)
```

factory fork exact active lease，核对 snapshot 内 `(plugin_id, generation_id, public_name)` binding，再返回
single-call route；不得按 current/public name 二次 lookup。route close/terminal 后释放 fork。`McpCallResult` 只承载
`success | tool_error`；transport/handshake failure 抛 `McpTransportError`，timeout 抛 `McpTimeoutError`，caller
cancellation 完成 call/process cleanup 后恢复 `CancelledError`，不能折叠为空 success或隐藏重试。

- `name` 在 generation 内唯一；command 非空；`cwd` 与相对 command 文件只允许解析到 immutable
  plugin artifact 内。
- plugin env 不得覆盖 `AKA_PLUGIN_DATA_DIR`、`AKASHIC_WORKSPACE` 等 Core reserved values。
- `EndpointEnv(env, process)` 必须引用同 generation 的 `ManagedProcessDefinition`；Core 先启动该
  process，再把 candidate/formal 实际端口以十进制字符注入 MCP env。缺少引用、跨 generation
  引用、env 重名或试图覆盖 Core reserved value 都在 materialize 前 fail-loud。
- `candidate_env` 作为 source-relative declaration 同时存在于 candidate/formal descriptor 并进入 identity，
  所以两棵 Root digest 相同；只有 runtime env projection 在 validation owner 注入，formal 不应用。它是首个
  consumer 的 controlled-backend seam，不是安全沙箱。
- Core 完成 initialize、`tools/list`、重复 tool name、required tool、schema 和 fatal-recovery 检查。
- candidate 默认不能调用工具；只允许显式 `candidate_read_only_tools`，且该集合必须属于实际 tools。
  allowlist 由 candidate Core-owned `McpRoute.call()` 强制，不能只靠 AgentLoop 隐藏 Tool，也不能让 semantic
  check 直接取得无保护的 `McpToolWrapper`。
- MCP stdio 继续由协议专属 `McpClient/McpGenerationHost` 拥有。它不经 C13 二次包裹，避免两个
  recovery/terminate owner；两者只共用底层 process-group primitive。

### 2.2 Managed process

```python
MANAGED_PROCESSES = ServiceKey[PluginManagedProcesses]("core.managed_processes")

async def apply(ctx, config):
    registry = ctx.require(MANAGED_PROCESSES)
    await registry.register(
        ctx,
        ManagedProcessDefinition(...),
    )

ManagedProcessDefinition(
    name="calendar_api",
    command=("python", "mcp/run_server.py"),
    cwd="mcp",
    env={},
    port_env="PORT",
    formal_port=18000,
    readiness_path="/health",
    startup_timeout_seconds=15.0,
)
```

`await PluginManagedProcesses.register(ctx, definition) -> None` 与
`freeze() -> ManagedProcessRegistry(definitions, descriptors, catalog_digest)` 遵循同样的内部
Effect/Health owner 与 freeze/digest 合同。materialized endpoint/process handle 只由
Core host 持有；插件通过自己声明的本地 endpoint/更高层 capability 消费，不取得任意 process 控制权。

- 第一版只支持 Core 可验证的 loopback HTTP readiness。candidate 总是分配临时端口并通过
  `port_env` 注入；formal 使用经过校验的 `formal_port`。
- readiness URL 由 Core 从 `127.0.0.1 + exact port + readiness_path` 生成，插件不得传任意 URL。
- Core 拥有 start/ready/cancel/terminate/recovery 和 bounded log ring；stdout/stderr 不得继续丢到
  `DEVNULL`，也不得把无限日志留在内存或 plugin-data。
- 同一 generation/name 同时只能有一个 current epoch；dispose/cancel 后旧 recovery task 不得复活。

两个 registry 的 descriptor 只包含 source-relative declaration，不包含 candidate 临时目录、端口、PID、
进程 epoch 或 module clone name。candidate/formal 两棵 Root 的 descriptor digest 必须相同；运行态进入
Health/Incident/inspection，不进入 immutable identity。

## 3. Materialization 与发布

1. Root provider 在挂载插件前创建，插件只能登记声明，不能直接取得 host/client/process。
2. Root settle 后 Core freeze 两个 registry；freeze 后插件侧 register/mutation/公共 unregister fail-loud；Root
   dispose 的内部 Effect cleanup 始终允许执行，且不重新开放 frozen registry。
3. snapshot registry 以 `(plugin_id, generation_id, public_name)` 路由。candidate 只 materialize 已变更
   generation 的新声明；未变 generation 复用 exact generation host，已有 stable generation 继续使用
   自己的旧 host。stable boot batch materialize 批次中全部 generation。Calendar 同名 candidate process
   使用临时端口；formal fixed-port 交接必须先停 admission、排空旧 endpoint lease，停旧后启新。
4. candidate 的 `AKA_PLUGIN_DATA_DIR`、workspace、端口和 cwd 都来自 validation runtime；正式数据不得写。
5. MCP handshake、required tool、managed readiness 与 semantic check 全部成功后才允许 `latest_ready`。
6. promotion 先 pause/drain/seal candidate，再停止 validation host、formal rebuild Root 与 declarations，
   核对 descriptor digest 后启动正式 host；stable pointer/snapshot/route 不得提前提交。formal process 的批准
   write-set 或 endpoint stop/start 可能已发生；rollback 成功时恢复旧 stable，rollback/cleanup 失败时必须进入
   显式 degraded/cleanup-failure，保留 owner 与证据，不能报告 promote/drain 成功。
7. stable snapshot 只保存 committed MCP tool/catalog 与 managed endpoint 的不可变 projection；candidate
   registry 不公开。旧 generation 的 host 在 snapshot lease 归零后才 drain。
8. runtime resource 的唯一 owner 是 generation host；Fiber registration Effect 只拥有 descriptor/Health。generation
   scope 只注册一个 `host.release_generation(generation_id)` cleanup，不再同时注册 client/process cleanup。
   stop/disconnect 失败时 Host 不在 `finally` pop entry，而把纯 Core process/client cleanup handle 保留为
   `cleanup_failed`。Root/Effect/Scope 可完成 dispose，plugin module 可卸载；Manager 保留 generation/artifact identity、
   runtime handle、错误和 retry tombstone，普通 stop/scope cleanup 遇到 tombstone 必须拒绝隐式重试；只有
   `retry_generation_cleanup(generation_id)` 成功后才删除 Host entry/tombstone 并允许 artifact GC。retry 不得依赖
   已卸载 plugin callable。
9. `ReloadPhase` 新增非终态 `cleanup_failed` 与 `degraded`，`RecoveryActionName` 新增
   `retry_generation_cleanup` 与 `retry_runtime_recovery`：
   - formal side effect 已恢复、仅新/旧 runtime cleanup 未完成 → `cleanup_failed`；回滚到 base 的 candidate cleanup
     retry 成功进入 `aborted`，已经 committed、只待旧 generation drain 的 cleanup retry 成功进入 `recovered`；
   - endpoint/old runtime restore 未完成、外部状态不确定 → `degraded`，阻止该 endpoint 后续 publication；
     `retry_runtime_recovery` 成功后进入 `recovered`，再执行 cleanup；
   - journal 的 `reload_transactions` 固定保存 `base_snapshot_id`/`base_generation_id`（old stable）、
     `candidate_snapshot_id`/`generation_id`（new snapshot/attempt generation）、`formal_effects_json`、
     `failure_resource`、`error`、`recovery_action` 与单调 `attempt_count`；`reload_events` 追加同一 attempt 的
     old/new identity、resource、error 和 retry evidence。重启后重新打开 SQLite 仍必须得到相同的失败 owner 与
     action，不能用当前 stable 或新 generation 猜测替代。
   - `pending_recovery()` 必须返回两种非终态及其完整 evidence。failure state 按
     `normal < cleanup_failed < degraded` 单调 join：同一 transaction 只允许 `cleanup_failed -> degraded` 这一条
     强化边，且只能追加更强的 unresolved runtime evidence、把 action 升级为 `retry_runtime_recovery`；不得反向降级，
     不得覆盖首次固定的 recovery target。其余公开 `advance()` 只能保留当前 failure phase，不能直接进入
     `complete`、`promoted`、`drained`、`aborted` 或 `recovered`；终态只能由匹配 action/attempt 的
     `finish_recovery()` 在全部 Host retry receipt 已持久化后原子写入，失败记录的 error 不得被 recovery receipt
     覆盖。
   - 首次启动 candidate/formal runtime 前，Manager 必须把当前 `runtime_owner_boot_id`、base/candidate artifact
     pointer 与 old/new generation identity 写入 journal，且 boot owner 不可覆盖。同一 boot 只能调用保留在
     generation host 中的 exact retry handle；不得借启动恢复杀掉仍可能存活的同 boot 子进程。
   - 新 boot 只有在 supervisor 明确提供 `AKASHIC_SUPERVISED=1`、旧 boot identity 非空且不同于当前 boot 时，
     才能先按旧 boot ID 调 Boot Guardian 回收子进程，再根据 journal 的 exact `recovery_target` 与 pointer 恢复
     base 或 candidate。随后必须重建 fresh stable Root/formal Host，并把 cleanup、pointer、snapshot、generation、
     old/current boot receipt 一起写入 journal 后才收束；缺 boot identity、同 boot 或 unsupervised 启动均 fail-loud。
   - stable boot batch、candidate partial start、snapshot drain 与已经 committed 的 runtime watchdog 使用同一个 durable
     failure owner。inner MCP/process host 一旦保留 tombstone，必须通过 Core-only callback 把 exact generation failure
     立即交给 Manager；不能只更新 Root Health/Incident。没有活动 reload transaction 时 Manager 新建 runtime recovery
     transaction，普通 shutdown/dispose 不得把它降成只存在于内存的 `_cleanup_failures`。
   - 同一插件已有 prepared/latest-ready candidate 时，stable watchdog failure 必须加入该 candidate transaction；该记录
     已同时持有 candidate generation 与 base generation identity，不能再创建第二条 nonterminal transaction。Manager
     同步撤销 candidate admission，`publish/switch_ready` 在 formal handoff 与 pointer commit 前都检查 failure latch；显式
     recovery 以 `target=base` 回收 candidate Root/runtime/scope、恢复 stable runtime 与 exact pointer 后，才允许 journal
     收束为 `recovered`。
   - 显式 recovery 从 exact Host retry 开始，到 stable rebuild、endpoint/skill 恢复、pointer normalize、candidate drain、
     snapshot resume 与 journal terminal 为止，是一个不可被调用方取消截断的 critical handoff。取消只能延迟到完整交接和
     terminal receipt 后再返回；snapshot drain 自身失败必须记录为 `runtime-snapshot-drain` cleanup owner，并由同一个
     Manager recovery 入口完成 `retry_drains()`，不能预先写成 `aborted`。
   - Root/Scope 开始 dispose 时，Composition Host 必须先 detach 该 generation 的 Health/Incident observation bridge；
     后续 Core-only stop/retry 仍清理 retained process/MCP owner 并上报 durable failure，但不得再次调用已失效 Fiber binding。
     一个 child Host 已有 tombstone 时不得隐式 retry 它，同时仍要停止同 generation 中没有 tombstone 的 sibling。
10. v3 artifact 根使用静态 `akashic.plugin.toml`，installer 在任何 Python import 前解析：

    ```toml
    schema_version = 1
    name = "calendar"
    version = "3.0.0"
    api_version = 3
    entrypoint = "plugin.py"

    [[python]]
    requirements = "mcp/requirements.txt"

    [validation]
    exclude_data_paths = [".env", ".gcp-saved-tokens.json", "token.json", "oauth.json"]
    ```

    installer 用 `tomllib` 校验静态 name/version/api/entrypoint、requirements 与 validation paths，拒绝绝对路径、
    symlink、重复和 artifact/data 越界，在 requirements 父目录构建 `.venv`。MCP/process Python command 若落在
    该 root 下，static admission 必须唯一绑定 runtime，Manager 将 `mcp:<name>` / `process:<name>` 的 argv[0]
    冻结为该 artifact 已 staging interpreter；C12b/C13b Host 只能消费这份 generation 投影，禁止再按 PATH 解析
    manifest 中的 `python*` token。安装事务不 import/执行 `plugin.py`，不执行
    `apply()`、不启动进程；真实 runtime 从 immutable artifact 首次导入并再次核对 module export 与 manifest。
    迁移期旧 v2 installer 仍走 class import，但 pure-v3 删除 Gate 要求所有 v3 artifact 都有静态 manifest，届时删除
    v3 install-import fallback。
11. register 时 Core 为每个 MCP/process 建立 required Health handle，host adapter 在 exact Root 的
    lifecycle loop 中 degrade/recover，并通过同 owner 的 `report_incident()` 记录 handshake/readiness/
    recovery/cleanup 失败。Root dispose 后 handle 不可写；不允许降级成只有 Manager 字符串 receipt。

## 4. Calendar 数据边界

Calendar 旧 `activate()` 会把 workspace `mcp/calendar-mcp` 的 token、`.env`、SQLite 和 proactive config
复制到 plugin-data。该动作不是 C12/C13 的普通 `apply()` 行为，必须在 Calendar 迁移 PR 中单独实现一次性、
可恢复的数据迁移：

- candidate 可以创建/迁移/写自己的 validation SQLite/data copy，但绝不写正式 plugin-data；
- `akashic.plugin.toml[validation].exclude_data_paths` 必须至少排除 Calendar `.env`、
  `.gcp-saved-tokens.json` 与其他 token/OAuth 文件；Core 构造
  validation data 时只复制其余明确文件，并记录 inventory。candidate 目录不得出现 secret path/hash/content；
- formal 迁移前生成明确 backup，按文件逐项 `source missing / target exists / copied` 记录 receipt；
- 已存在 target 不覆盖；部分失败恢复到迁移前 tree；
- candidate Gate 使用 fake/controlled Calendar backend。canonical `list_events` 会更新 validation SQLite，
  且真实 credentials 可能触发 OAuth refresh，因此不能称为纯只读外部 probe；真实 OAuth
  refresh 和远端写入禁止。Calendar canonical source 必须先提供 `CALENDAR_BACKEND=recording`
  provider factory；入口必须先选择 recording backend，再 lazy import live auth/bridge，不能以顶层 import 读取
  `.env`。该 backend 不读 credential、不发网络/OAuth，返回固定 fixture，但仍让真实
  `list_events` 路径写 candidate SQLite，用于验证数据 owner。Core 只在 candidate 注入该 env。
- Calendar 当前 `calendar_mcp.log`/server `FileHandler` 必须在迁移 PR 中改为 stdout/stderr，
  交给 Core bounded generation log ring；不保留无限 append 的 plugin-data 日志。

## 5. 验证与停止条件

### Core 定向 oracle

- declaration：非法 name/path/env/port/readiness、duplicate、freeze 后修改、v2/v3 collision；
- candidate：controlled MCP initialize/tools/list、required/read-only set；非 allowlist route call 拒绝；HTTP 临时
  端口 readiness；Calendar MCP 的 `PORT` 必须等于同 generation `calendar_api` 的临时端口，
  正式 `18000` 零连接；validation data 可写但正式 data hash 不变；discard 后 process group、socket、task、Effect、
  module、validation workspace 全清；
- promotion：candidate/formal descriptor 等价，正式 data/port 生效；drift、readiness、handshake、cancel、
  cleanup failure 均保持旧 stable；
- lease/drain：旧 turn 继续使用旧 MCP catalog；新 admission 只见新 catalog；旧 process 在 lease=0 后退出；
- recovery：stale epoch 不复活，fatal failure 进入 exact Root generation Health/Incident 与 runtime owner；
- logs：stdout/stderr 进入 bounded generation log view，停止后无 reader task 残留。
- ownership：disconnect/terminate 注入失败后 Host entry、generation/artifact tombstone 与 journal
  `cleanup_failed` 仍可查询，Root/Scope/module 可完成释放；调 `retry_generation_cleanup()` 成功后 runtime entry/
  tombstone 才消失，且每个 runtime resource 只有一个 Host cleanup owner。rollback/old restore 失败进入
  `degraded`，`pending_recovery()` 返回准确 action；
- journal：`cleanup_failed/degraded` 不能通过 `advance()` 直接收束；base rollback cleanup retry 后为 `aborted`，
  committed candidate 的 old-generation cleanup retry 后为 `recovered`；legacy SQLite schema 原位补列且保留旧记录；
- mixed failure：cleanup failure 后再发生 watchdog degraded 时只保留一条 transaction，phase 单调升级、resource 合并、
  attempt 增加且 target 不变；反向到达的 cleanup evidence 不得把 degraded 降级；
- cancellation/drain：在 Host retry 或 endpoint resume 中取消调用方，recovery 仍完成 exact Host、pointer、snapshot 与
  journal terminal 后才恢复 `CancelledError`；candidate snapshot drain 首次失败时 journal 保留
  `runtime-snapshot-drain/retry_generation_cleanup`，显式 retry 完成 drain 后才进入 target 对应终态；
- boot recovery：同 boot、unsupervised、缺 old boot identity 均在 pointer/snapshot/process 变化前拒绝；不同 supervised
  boot 只清 exact old boot 子进程，按 base/candidate target 恢复 exact pointer，fresh stable Root/Host 通过后写完整
  receipt；
- watchdog/stable boot：真实 committed process 意外退出并耗尽 recovery 后立即形成 durable `degraded`，显式 retry
  同时清理 MCP/process sibling host 并重建 exact stable formal runtime；candidate partial start 或 stable batch cleanup 注入
  terminate 失败后 journal 保留 `cleanup_failed`，进程仍活且可查询，retry 成功后端口与 tombstone 同时消失；
- concurrent candidate：prepared/latest-ready candidate 存在时杀掉 old stable process，journal 只能保留 candidate tx 一条
  `degraded(target=base)`；candidate admission/promotion 被拒，retry 后 candidate 临时端口、Root、scope 与 latest pointer
  全部回收，stable formal endpoint/tool 恢复；
- install：静态 manifest 在 Python import 前给出 identity/runtime；v3 Calendar immutable artifact 的 MCP/server
  runtime 已 staging，candidate 不在首次启动临时装依赖；validation inventory 明确不含 `.env`、
  `.gcp-saved-tokens.json` 或其他 token/OAuth 文件。

### 首个 consumer Gate

Calendar exact source + exact Core：真实 Manager install（`akashic.plugin.toml` 在任何 Python import 前完成
identity/runtime/validation staging）→ publish candidate 为 `latest_ready/committed` → acquire exact latest snapshot lease
→ controlled MCP route call（route 绑定 candidate generation，MCP `PORT` 命中 candidate `calendar_api`）→ route/fork
lease 释放 → pause/drain/seal → promote → formal local
readiness → reload/discard。Gate 使用一次性 workspace、loopback 端口和 fake/controlled Calendar data；不读取正式
credential，不执行 ack/OAuth/远端写入。单插件 Gate 完成后才迁 Feed/Fitbit/Steam/Computer Use，并把它们合并到
最终 E2 Tool/MCP/Process 一次验收。

下列任一情况停止交付：candidate 写正式 data、占正式 port、公开 tool route；descriptor 不含完整声明；进程或
recovery task 残留；结果不健康仍 promote；旧 generation 在 lease 前退出；日志无限增长或被静默丢弃。

## 6. 实现拆分与 v2 删除前置

C12/C13 Core capability 已在 `8653bab0` 收成 generation-scoped runtime owner。C12/C13/Journal focused `44 passed`，
Manager `63 passed`、Hot Reload `147 passed`、loader `59 passed`；相关 Basedpyright `0 errors`、compileall 与
diff-check 通过。真实 probe 覆盖 candidate/formal route、watchdog、retained tombstone、跨 boot、caller cancellation、
snapshot drain 与 Root observer detach。Terra xhigh 终审无 P0/P1。Calendar exact artifact/consumer Gate 未完成，因此
C12/C13 只能标为 `CANDIDATE`，不能标为 `READY`。

1. C13a（完成）：`ManagedProcessDefinition/Registry`、Root provider、descriptor 与 exact Root fence。
2. C13b（Core 完成）：现有 process host adapter、bounded logs、retained cleanup owner/retry/journal；首个 consumer Gate 待办。
3. C12b/C22（Core 完成）：MCP host adapter、exact-lease route、full catalog fence、静态 manifest 与 artifact runtime staging；
   首个 consumer Gate 待办。
4. Calendar canonical source owner 是 `/mnt/data/coding/akashic-plugin/calendar-mcp`。独立插件 PR 负责 pure-v3
   declarations、`akashic.plugin.toml`、recording/lazy-auth seam、stdout/stderr 日志与一次性数据迁移；Core PR 不直接
   修改 cache，只消费该 PR 的 immutable exact artifact 并运行 Gate。两仓提交、base 与回滚点分别记录。
5. 其余 MCP/process consumers 并行迁移。
6. 最后删除 `Plugin.mcp_servers()/managed_services()`、公开 v2 spec、`PluginContributions` 对应字段、Manager
   固定收集/validation adapter、Snapshot v2 固定 projection 与 v2 Gate。底层 `McpClient`、process-group、
   readiness/recovery implementation 保留为 Core 内部实现。

workspace MCP watcher/admin 兼容岛已删除；迁移后的 MCP 只能由插件静态 manifest 声明并由 Core generation host
拥有运行时 client、route 和 cleanup。
Fitbit monitor 当前没有通用 candidate port injection；canonical source 增加受 Core 控制的端口入口前不得标记
C13 candidate-isolated，也不能借固定 `18765` 通过 Gate。

## 7. 回滚

Core 恢复点为 `4e890b75`；外部 Calendar 使用自己的独立 source backup。C12/C13 Core PR 不写正式 runtime data，
回滚只撤销源码、测试、合同和一次性测试 workspace。
