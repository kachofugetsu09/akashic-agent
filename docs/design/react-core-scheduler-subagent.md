# React Core 原子能力与 Scheduler/Subagent 插件设计

- 状态：confirmed design；S0～S4 implemented and validated
- 日期：2026-08-22
- 决策：[0039](../decisions/0039-react-core-atoms-keep-sources-unprivileged.md)
- 任务合同：[React Core、Scheduler 与 Subagent 分阶段任务合同](react-core-scheduler-subagent-task-contract.md)

## 1. 目标与本轮边界

目标是让不同来源用同一套积木执行 Agent 工作，同时保持各自的生命周期、Prompt、工具、记忆和持久状态语义：

- Core 提供最小、正交、与来源无关的原子能力；
- Scheduler 与 Subagent 分别成为仓库内置但非特权的 v3 插件；
- Proactive/Wake 只用于证明积木足够，不在本轮迁移或改写；
- 执行语义迁移保持等价；唯一 breaking delta 是旧 `spawn_enabled` 配置 fail-loud 退役，统一改用 `agent.plugins.disabled_builtin`。

若基线回放发现当前实现与 `projectneed` 或决策 0034 冲突，实施必须停止，把“现状”“合同目标”和“拟议修复”拆成单独审批，不能让候选实现顺便改变 oracle。

## 2. 当前真实链路

### 2.1 被动链路

当前 `AgentLoop._process_with_runtime_admission()` 获取或复用 exact `RuntimeSnapshot` lease，`AgentLoop._process()` 建立 turn/session context，随后 `AgentLoop._react()` 把普通消息交给 `PassiveTurnPipeline.run()`。`PassiveTurnPipeline` 内部运行 `before_turn`、reasoning、commit、`after_turn` 等阶段。

这条链路当前同时拥有普通被动语义与一部分 programmatic/scheduler 参数，例如 `stateless`、`disabled_tools`、`skip_post_memory`。它证明核心能力已经存在，但调用参数仍把来源策略投影进通用入口。

### 2.2 Scheduler 迁移前基线

S0 基线中，`SchedulerService` 同时拥有 `JobStore`、每秒 tick、misfire、任务执行和重排程：

```text
schedules.json
    │ JobStore.load / fail-loud
    ▼
SchedulerService._tick
    ├─ instant → message_push
    └─ soft → AgentLoop.process_direct(stateless=True)
                → content
                → message_push
    ▼
成功后 run_count + 1；one-shot disabled；every 计算下一次 fire_at
```

这条路径的业务 owner 是 Scheduler。现有插件 background job 合同明确禁止把它迁成 `BACKGROUND_JOBS`；用户调度状态不能被误当成 generation-scoped job。

### 2.3 Subagent 迁移前基线

S0 基线中，`SubagentManager.spawn()` 获取容量 lease、创建 `subagent-runs/<job-id>/`、租用 exact snapshot，再让 `AgentBackgroundJobRunner` 执行独立 `SubAgent` 循环。完成或取消经 `SpawnCompletionItem` 回到原 chat，并追加 `memory/spawn_trace.jsonl`。

它已经具备容量、快照、取消和完成协议，但推理循环、Prompt 和工具 profile 与被动 `react` 是平行实现。迁移目标是复用 Turn 执行，不改变 spawn 业务事实。

### 2.4 S4 完成后的真实链路

```text
普通消息 ───────────────────────────────┐
spawn Tool ─→ Subagent 私有 profile ───┤
Timer 到时 ─→ Scheduler 私有 job ──────┤
                                       ▼
                              exact scoped Turn
                                       │
                                     react
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          父 Tool result / Continuation            Delivery + settle
```

每棵 `CompositionRoot` 的 `PluginToolBinding` 保存自己的 bound handler。Scheduler/Subagent 的 Tool、lifecycle listener 和 cleanup 捕获同一个插件私有 runtime；即使未变化的 generation 同时挂在旧、新 Root，也不会通过 Python module 全局变量串线。Core 只拥有 exact binding、Turn、Timer、Delivery 与 Root drain，不认识 `scheduler`、`subagent`、cron 或 profile。

## 3. 领域词与原子能力

系统继续只用五个基础词解释主链路：`Message` 组成 `Turn`，`Turn` 归入 `Session`，`Loop` 表达 `Message → react → Message`。下面的对象只在独占边界时存在，不形成第二套领域模型。

| 原子能力 | 唯一职责 | 不拥有 |
|---|---|---|
| scoped Turn port | 在 exact runtime scope 中开始一个 Turn，返回可等待/取消的 handle | cron、spawn、hazard、投递目标 |
| Turn handle | 暴露 accepted identity、terminal、cancel 和 cleanup 完成 | 业务重试、任务列表、持久化策略 |
| one-shot Timer | 在指定 deadline 唤醒一次并可取消/回收 | interval、cron、misfire、数据库、回调重试 |
| Tool grant | 不可变地描述本 Turn 可见和可执行的工具集合，由 executor 强制 | 全局 registry 修改、来源字符串判断 |
| typed receipts | 暴露已有 phase、Turn、工具、投递和 cleanup 的 settled facts | 成为第二套状态、改变控制流、写记忆 |

`TurnRequest`、`TurnScope` 等名字只在实现确有独立不变量时保留。若只是字段转发，应内联到 scoped port 或现有 Turn 输入，不为图形对称创造类。

`Skip` 不是 Core 原子，也不是一个待实现的结果类。它只是一句人话：“这个插件拥有的这轮工作到这里结束”。代码通常就是插件先记录自己的领域事实，再从自己拥有的函数返回。调试 trace 从 typed receipt 和插件记录投影，不再单独成为能改变行为的原子。

## 4. Scoped Turn 怎样保护生命周期语义

来源不能在每次调用时传一串 `skip_before_turn=True`。插件从 exact committed Root 获得一个已经绑定的 Turn port；这个 port 由 Core 根据调用者 Fiber、generation、公开 capability 和 mandatory kernel 形成 scope。

```text
exact RuntimeSnapshot
    ├─ mandatory kernel
    │    admission identity / lease / reasoner / executor /
    │    terminal / cleanup / trace
    ├─ caller Fiber exports
    │    lifecycle modules / Prompt contributions
    └─ immutable grants
         memory-read / memory-write / tools / outbound / files
                │
                ▼
         scoped Turn port
```

规则如下：

1. mandatory kernel 不能被插件移除；取消、terminal、lease release 和 cleanup 总会收束。
2. `before_turn`、`before_reasoning`、`after_reasoning`、`after_turn` 是有序接入点，不表示其中每个模块对所有来源都生效。
3. lifecycle module 只有在当前 scope 可见且依赖 slot 满足时运行。passive-only module 不会因为“先跑 before_turn 再判断来源”而触发。
4. Prompt 注入和替换继续使用现有 typed context transform、section export 与稳定 slot；S0 先冻结每个 handler 的实际可写字段、顺序和冲突结果，不新增字符串后处理或更宽的改写权。
5. scope 在 Turn admission 时冻结。插件热重载只服务新 Turn，旧 Turn 继续使用原 generation。
6. 插件是否调用这个 port，由插件自己拥有的外层循环决定；未调用就没有 Core Turn，也没有伪造的 Turn terminal。

首个 Core 批次只提取和锁定现有语义，不立即重排 phase。某个现有 module 是否属于 mandatory kernel，必须由真实 consumer、调用路径和差分回放证明。

### 4.1 `return` 的所有权边界

“记录然后 `return`”只有在当前函数拥有被结束的循环时才成立：

```text
plugin-owned tick / spawn admission
        │
        ├─ condition false → plugin records fact → return
        │                                      （没有 Core Turn）
        └─ condition true  → scoped Turn port → react → terminal/cleanup

Core-owned Turn lifecycle
        │
        ├─ listener return None → 仅结束这个 listener，继续既有 phase
        ├─ event-specific Bail  → 只执行该 event 声明的领域动作
        └─ exception            → 走既有 fail-loud / error settlement
```

当前代码大体证明这条界线：composition lifecycle serial 明确拒绝 `Bail`；`before_turn` / `before_reasoning` 的 abort 是既有 Phase 状态，不是 v3 插件可随意推广的控制权；普通 listener 里的裸 `return` 也只返回 dispatcher。`tool.execution.authorize` 的公开合同只把 `Bail(reason)` 结算为 tool `denied`，Reasoner 原则上仍可继续。

S0 已删除这一非正交残留：passive 与旧 subagent Reasoner 不再识别 deny 文本的 `tool_loop_guard:` 前缀，当前 fleet/composition lock 与 Gate 也不再安装或期待该插件。普通 deny 只结算当前工具调用；专属字符串暗号没有兼容壳。hua-home 旧 `workspace/plugin-data/tool_loop_guard-github` 仍按 PLG-010 原样保留，没有被代码重构当成可删除状态。

因此，本轮不会给所有 lifecycle 时间点增加一个万能短路协议。未来若要实现“某插件在 tool 触发前结束整个 Turn”，必须单独回答：它结算成什么 Turn terminal、是否保留已经写入的消息和工具结果、哪些 after/finally 仍运行、多个 listener 谁先赢、重试与取消怎样区分。没有这些答案时，用异常、特殊字符串、共享 flag 或私有 import 穿透 Core 都属于特权后门。

## 5. 记忆、Session 与 Prompt

记忆不是“某种 React 默认都会做”的隐式行为，而是两项独立 grant：

| 使用场景 | Session | memory read | memory write | Prompt |
|---|---|---:|---:|---|
| 普通被动 Turn | 既有 durable Session | 保持现状 | 保持现状 | 保持现有完整 passive scope |
| Scheduler SOFT | Scheduler 自己的 stateless/ephemeral Session | 否 | 否 | S3/S4 精确保留基线实际运行的 Prompt 与 hook；后续裁掉 passive-only 内容需单独批准 |
| Subagent child | 每个 child 的 ephemeral Session | 首版否 | 否 | Subagent profile + 父任务；不继承未声明的父 Prompt |
| 未来 Wake tick | 由未来合同决定 | 显式 grant | 显式 grant | Wake scope；本设计不批准具体内容 |

`skip_post_memory` 在迁移期仍可作为现有持久投影，但目标接口不靠来源传布尔开关。scope 没有 memory-write grant 时，executor 和事件 owner 都不能产生记忆写；不能只在最后一个 hook 中跳过。

这里 scope 与私有 early return 是两条正交轴：scope 决定某个 lifecycle/Prompt/memory contribution 是否存在；插件私有 gate 决定是否调用 Turn port。不能让不适用的 hook 先运行并写入状态，再靠 return 补救。

Subagent 的“递归”表示 child 通过同一 scoped Turn port 进入同一 `react`，不是复制父 Turn 的整个上下文。首版只继承显式父子 lineage、任务文本、profile 映射和已授予工具；不实现 Codex 式 fork 历史、durable child Session 或跨重启续跑。

## 6. 非特权插件边界

Scheduler/Subagent 即使位于仓库内，也必须通过真实 v3 loader 和普通 `apply(ctx)` 进入 committed Root：

- 只消费公开 `ServiceKey`；
- resource、timer、task 和 listener 全部登记为当前 Fiber 的 Effect，逆序清理；
- 不 import `PluginManager`、`AgentLoop` 实例、私有 provider、全局 ToolRegistry 或 bootstrap singleton；
- 不接收任意 Session repository、任意 SQL、任意 workspace root 或任意删除能力；
- Core 和 bootstrap 不按插件 ID、包路径或 builtin 标志分支；
- 所需 Service 缺失时 candidate/formal admission fail-loud，不在首次触发时静默 fallback；
- candidate 只建立隔离资源和回执，不启动正式 timer、不读取正式 schedule、不创建 child Turn、不发送消息。

Core 可以为所有插件提供同一公开 capability，但不能为两个内置插件暴露私有捷径。

## 7. Scheduler 插件组合

Scheduler 插件保留当前所有业务积木：

| 积木 | 能力与 owner |
|---|---|
| JobStore | 严格加载并原子保存完整 `ScheduledJob` 候选；损坏不解释为空 |
| schedule calculator | `at` / `after` / `every`、cron、时区、P90 lead、misfire |
| fire loop | 选择 due job、避免同 ID in-flight、调用 one-shot Timer |
| scoped Turn port | 只执行一次 SOFT Turn；无调度知识 |
| delivery port | 将完整逻辑消息提交给目标 channel，保留 ChatLane 被动优先 |
| settle | 成功后增加 `run_count`；one-shot 禁用；every 推进；失败不增加成功计数 |

Core Timer 不循环。插件每次 settle 完成后读取当前持久 job，再登记下一次 one-shot wait。插件卸载时 Effect 取消所有 wait 和 in-flight task；取消不把已提交 delivery 冒充回滚。

Scheduler 自己拥有 fire callback，所以 misfire、job 已禁用或 generation 已过期时，可以在插件内记录对应事实后直接返回，不调用 Turn port。这不是 Core skip；是否推进、禁用或保留 job 仍按 Scheduler 的既有持久合同结算。

插件私有记录必须闭环而不是只打一条日志：一次 fire 的消费事实与确定性的下一次 wait identity 由 Scheduler 自己原子提交或幂等 reconcile。这样在提交前后崩溃都能恢复成恰好一个 next wait、零重复 Turn。Core Timer 仍只发布 fire/cancel/dispose receipt，不理解 job、no-work 或 next schedule。

## 8. Subagent 插件组合

Subagent 插件保留下列业务积木：

| 积木 | 能力与 owner |
|---|---|
| spawn admission | workspace 全局并发上限、父子 lineage 与 owner lease；首版不开放 child 再 spawn |
| profile mapper | `research`、`scripting` 等 profile 到 Prompt 与 Tool grant 的映射 |
| task directory | 创建 `subagent-runs/<job-id>/`，只授予该 child 所需 root |
| scoped Turn port | 在 ephemeral Session 中递归执行同一 `react` |
| completion | 同步返回父 Tool result，或后台经 `PluginContinuations` 发布普通 `InboundMessage` |
| trace | 追加 started/completed/cancelled 与 parent/child identity |

Core Turn handle 表达 child cancel、cleanup 完成与 lease release。插件取消路径先提交一次 cancelled Continuation，再请求 child interrupt，并通过 task cleanup 阻止迟到 success；后台取消只能产生一个 cancelled completion，已完成 delivery 不因父 Turn 后续失败被撤销。

Tool grant 在真正的 executor 边界执行。`research` child 没有 shell/file-write grant 时，即使模型或插件伪造工具名也必须被拒绝；`scripting` 只获得 task directory 和明确宿主执行能力，不获得任意 Core 数据库或 plugin control plane。

Subagent 自己拥有 spawn admission，所以容量不足、profile 不存在等合法拒绝可由插件形成既有 tool result、记录 admission receipt 后返回，不创建 child Turn。child 一旦通过 scoped Turn port 被接受，插件不能再靠一个私有 flag 或 listener `return` 越权跳过 Core terminal/cleanup。

scoped Turn port 的 accepted receipt 是责任交接点。在它之前，Subagent 插件负责释放 provisional capacity、lineage reservation 和 task directory；在它之后，Core handle 唯一拥有 child cancel、terminal 与 cleanup，插件只组合 completion 的既有可观察顺序。普通 `return` 既不能冒充 accepted，也不能冒充 cleanup complete。

## 9. Proactive/Wake 结构验算

本设计不修改 `proactive.db`、`wake_proactive.db`、`drift.db`、现有 source/module ABI 或运行服务。只检查未来 Wake 风格能否这样组合：

```text
one-shot Timer fires
        │
        ▼
collect observation ──► gate
                         ├─ no work → record reason + next wait + return
                         └─ run
                              │ scoped Wake Turn
                              ▼
                            react
                              │
                              ▼
                 delivery + domain state settle
```

hazard、reservoir、ack、quota、dedupe 和 next wake 都由未来插件拥有。gate 不调用 scoped Turn port 时，Core 只看见 Timer callback 完成；调用后才看见一个 Turn 和它的 terminal。若实现 Wake 必须给 Core 增加 `if proactive`、通用 `Skip` 类型、私有 hook 过滤或数据库知识，说明插件边界设计失败，应先回到规格而不是扩展特判。

## 10. 持久状态与减少协议

本设计不改变现有保留规则：

| 对象 | 正常增加 | 允许原位更新/逻辑失效 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| `schedules.json` | 新建 job | `fire_at`、`run_count`、`enabled` 等由完整候选原子替换；禁用是逻辑失效 | 只有显式 cancel 删除对应 job | Scheduler plugin；文件备份、完整 schema、restart smoke |
| `sessions.db/messages` | 既有 passive/主动 Turn 按合同追加 | 既有 Turn/Attempt 状态机 | 只按用户显式撤销/删除 | Session owner；完整 rows 与 write set |
| `memory/spawn_trace.jsonl` | started/completed/cancelled 追加 | 不原位改写 | 当前无自动减少协议 | Subagent plugin；trace digest 与 scenario receipt |
| `subagent-runs/<job-id>/` | child 运行产物增加 | task 自己可在授权 root 内更新 | 当前无自动减少协议 | Subagent plugin；目录 manifest 与恢复副本 |
| proactive/Wake/Drift DB | 本轮不写 | 本轮不改 | 本轮不减 | 既有 runtime owner；before/after digest |

测试和差分回放只使用一次性 workspace、隔离 plugin home、fixed clock、scripted provider 和 recording adapters。不得连接 hua-home 正式 workspace、正式 channel credential 或真实外部 API。

## 11. 调试与差分回执

实施阶段在 `docker/debug/` 增加公开入口的确定性 scenario runner。runner 不 import 私有实现，也不提供 mock success；它用真实 loader、真实 scoped port、可控 provider、fixed clock 和 recording adapter 运行完整边界。

每侧回执至少包含：

- identity：base/head、runtime snapshot、plugin generation、scenario hash；
- turn：parent/child、admission、phase/slot、Prompt section digest、provider request、Tool call/result、terminal；
- lifecycle：prepare、activate、lease、cancel、dispose、残留 timer/task/process；
- state：逻辑 before/after、INSERT/UPDATE/DELETE、文件 write set；
- effects：channel、HTTP、shell、process 等调用 envelope 与终态；
- verdict：exact difference、允许归一化字段和 mutant 结果。

只允许预登记的时间戳、UUID、PID、端口和临时路径归一化。Prompt 顺序、hook 顺序、工具 schema/可见性、错误类型、持久 write set、外部调用和 cleanup 残留不能被 normalizer 隐藏。

## 12. 历史代表场景

历史运行只提供真实形状，不作为可复制秘密或正式 workspace fixture。实现时将下列场景脱敏为确定性输入：

1. `scheduler-weather-d494`：周期 SOFT 天气任务已经多次成功（调查时 `run_count=27`）。固定时钟推进一次，期望一次模型执行、一次完整 delivery、成功计数加一和下一次 `fire_at`。
2. `subagent-scripting-391611fd`：scripting child 产生 task-directory 产物并成功完成；期望父 tool result、child trace、受限文件写和零额外记忆写。
3. `subagent-research-3f5c`：research child 成功返回调查摘要；期望无 shell/file-write 权限仍能完成。
4. `subagent-cancel-41b`：运行中 child 被取消；冻结 completion、worker cancel、cleanup 与 lease release 的当前顺序，只出现一次 cancelled completion，task directory 与 trace 保留。child-first 顺序作为单独待批准 delta 验算，不混入 S1/S2。
5. `wake-proactive-structure`：用已调查的 Wake/proactive tick 形状做设计验算，分别覆盖“插件记录后返回，不调用 Turn port”与“调用一次 Turn port”；本轮不运行真实 proactive runtime。
6. `tool-loop-guard-removal`：固定正式 manifest/cache/runtime topology 无 consumer 的证据，再用普通 deny reason 运行 passive 与 subagent 对照，证明 deny 只结算当前工具；mutation 恢复 `tool_loop_guard:` 字符串暗号时必须失败。旧 plugin-data 保持逐项不变。

scenario fixture 不复制用户正文、credential、真实 chat ID 或完整历史数据库。需要复核时从只读证据提取最小字段，并把来源 identity 写入私有运行报告，不提交敏感 payload。

## 13. 实施与验收阶段

每一阶段独立建任务合同、commit、回滚点和 Gate；后一阶段只在前一阶段验收通过的基线上开始。fixture runner 的骨架先建立，但每种领域 fixture 在对应实现完成后才成为切换 oracle。

1. **S0 Core 基建（已完成）**：已建立 disposable workspace、fixed clock、scripted executor、recording scope adapter 与 receipt comparator；`ScopedTurnPort/Handle` 绑定 accepted identity、terminal、interrupt 与 exact scope cleanup，既有 background-job programmatic Turn 已真实复用该实现；`ToolGrant` 在 executor 的插件 hook 之前强制执行；one-shot Timer 仅冻结 `schedule/result/cancel/cleanup` 合同，没有运行实现。`tool_loop_guard:` 残留与旧 Gate 已删除。runner 双跑归一化后零差异，provider input mutant 精确报错；公开 Gate 与 3998 项全量测试通过。
2. **S1 Subagent 实现（已完成）**：仓库内置非特权 v3 插件已通过公开 `SCOPED_TURNS` Service 组合 profile、task directory、ephemeral programmatic Session、exact-snapshot scoped Turn 与 shadow trace，并在隔离 fixture 中递归取得同一 `ConversationRuntime` terminal。candidate 在创建 Session 前拒绝，插件未登记生产 Tool；旧路径仍是正式 owner 与 shadow oracle。
3. **S2 Subagent fixture 验收与切换（已完成）**：正式 Tool binding 已切到 v3 插件，profile 使用 Turn-local Tool 实例保持 task root/网络边界；fixture 覆盖 sync/background、research/scripting/general、capacity、success/cancel、completion 恰好一次、cancel 后无迟到 success 和 lease 归零。旧独立 `SubAgent` 循环与 manager/runner/Tool adapter 已物理删除。
4. **S3 Timer 与 Scheduler 实现（已完成）**：来源无关的 one-shot Timer、完整逻辑消息 Delivery 与声明式 workspace file 由 Core Service 提供；非特权 Scheduler v3 shadow 组合 JobStore、calculator、Timer、scoped Turn、delivery 与 settlement。正式插件只挂载 dormant runtime，不读取 schedule、不登记 wait，旧 Scheduler binding 仍是唯一生产 owner 与差分 oracle。
5. **S4 Timer/Scheduler fixture 验收与切换（已完成）**：生产 `schedule`/`list_schedules`/`cancel_schedule` binding 已由内置非特权 v3 插件提供，旧 `SchedulerService` 与 Tool adapter 已物理删除。fixture 覆盖 fire/cancel/dispose、instant/SOFT、at/after/every/cron、misfire、restart、delivery failure、capacity、no-work 与真实 stable Root 热重载；Root 切换使用 exact snapshot lease，先停止旧 wait，再启动新 wait，不重复 fire。Scheduler SOFT 继续使用 stateless、memory-read/write=false 的 scoped Turn，完整 delivery 成功后才增加 `run_count`。

Proactive 不在 S0～S4 的实现、删除或迁移范围内。S4 结束时只用 `Timer → private gate → optional scoped Turn → delivery/settle` 做结构验算；若需要新增 Proactive 专用 Core API，则验收失败。

## 14. 验收矩阵

### Core

- passive baseline 的 Message、Prompt、phase/slot、provider payload、tools、Session rows、events、delivery 和 cleanup 无差异；
- 未调用 scoped Turn port 时不产生 provider/tool/Turn terminal；插件私有记录与异常可区分；
- lifecycle listener 的普通 return 不改变外层 Turn；event-specific `Bail` 不能越过该事件已声明的领域边界；
- 普通 tool deny 只产生 denied result；`tool_loop_guard:` 专属字符串分支零 consumer 后删除且 mutation 不能恢复；
- child 使用 exact parent-selected generation 或明确的新 child scope，热重载不改变在途 Turn；
- Tool grant 的拒绝发生在 executor，绕过 catalog/UI 仍失败；
- Timer fire/cancel/dispose 无残留，且不知道 cron、job、source 或 delivery；
- 调试投影关闭或 observer 失败不改变业务结果，核心故障仍 fail-loud。

### Scheduler

- instant、SOFT、at/after/every/cron、P90 lead、misfire grace/expired、restart recovery 全覆盖；
- 只有完整 delivery 成功才增加 `run_count`；one-shot 无论成功失败都按既有合同终结为 disabled；
- 同 ID 不并发，插件 unload/reload 后旧 timer/task 为零，不重复 fire；
- SOFT 没有 memory read/write 与 `message_push` 工具，最终 outbound 仍走 ChatLane；其余 passive lifecycle module 以 S0 实际回执为准，任何裁减必须登记 `declared_delta`。

### Subagent

- sync/background、research/scripting、capacity、parent/child lineage、success/error/cancel、parent shutdown 全覆盖；
- child 与 parent 的 Prompt、Tool grant、session state、file root 和 trace 不串值；
- background completion 恰好一次，取消后没有迟到 success；现有 completion/cleanup 顺序保持，child-first barrier 另行批准；
- generation unload 等待 child lease 或显式取消并完成 cleanup；task directory 不被普通 cleanup 删除。

### 非特权与结构完整性

- 静态 import Gate 证明插件没有私有 Core/全局 registry/repository 依赖；
- mutation 删除一个 scope/grant/cleanup 检查时，代表场景至少有一个失败；
- Core public API 删除插件名后仍可解释并测试；Wake 结构验算无需新增 API；
- 全量 Gate 通过，未运行的真实模型/真实渠道验证明确分层报告，不以 fixture 冒充生产行为。

## 15. 停止条件

出现以下任一情况停止当前切片：

- 需要修改 SessionDB schema、消息保留、proactive/Wake/Drift 数据或正式 workspace；
- 需要按插件 ID/source 特判 Core lifecycle、Prompt、memory 或 Tool；
- 发现 `tool_loop_guard` 仍有 active/canonical consumer，导致“失败实现直接删除”的前提不成立；
- 差分只能通过放宽 oracle、跳过场景或扩大 normalizer 获得全绿；
- 已取得 Turn/delivery handle 后结果不确定，却准备自动重试；
- 旧路径仍有真实 consumer，却准备删除或保留兼容壳掩盖双 owner。
