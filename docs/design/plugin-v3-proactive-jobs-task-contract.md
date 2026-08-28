# 插件 v3 Proactive / background job capability 任务合同

- 状态：implemented / C15、C21 candidate；模型 lease 部分由 [0050](../decisions/0050-model-revision-lives-in-ordinary-plugin.md) 勘误，其余 external consumer 迁移继续
- 日期：2026-08-16
- 实现起点：`19f2cca2`（只有 legacy prepared catalogs，C15/C21 public registry 尚未实现）
- 清单：C15、C21、C20 的前置
- 首个 consumers：Calendar MCP、Daynight Gate、Emotion
- 独立评审：C21 十轮 owner、transaction、lease、DB/Markdown crash recovery review 后无 P0/P1；
  C15 首个 Calendar consumer 的声明、recording、数据迁移和真实 Host Gate 复审后无 P0/P1

## 1. 目标与拆分

旧 `proactive_sources()/proactive_modules()/jobs()` 把不同事实塞进 `PluginManager` 固定贡献面。本任务拆为：

```text
PROACTIVE_COMPONENTS = ServiceKey("core.proactive_components")
BACKGROUND_JOBS      = ServiceKey("core.background_jobs")
```

- `PROACTIVE_COMPONENTS` 只登记主动数据 source 与主动 DAG module；Core proactive runtime 从 committed
  snapshot catalog 读取，插件不能直接 enqueue/send Turn。
- `BACKGROUND_JOBS` 登记 interval/Core-event job；Core runtime 取得 exact snapshot lease 后执行 handler。
- generation-bound LLM 只存在于一次 `BackgroundJobContext` invocation，不是可在 `apply()` 中取得或保存的
  Root Service；插件不能持有全局 provider/registry。

Default/Wake Proactive 的 flow/runtime/state machine 暂不经这些公共声明重写。它们最终进入 C20 私有兼容岛，
只允许 Core 内建 module identity；普通 external plugin 不可取得旧 `proactive_*` ABI。

`semantic_delta: compatible`。本任务不改变主动发送策略、ack/cursor/dedupe、proactive DB/Markdown schema，
不在 candidate 调模型、拉远端 source 或发送消息。

插件注入面只有两个 facade：

```python
class PluginProactiveComponents(Protocol):
    async def register(
        self,
        ctx: Context,
        definition: ProactiveSourceDefinition | ProactiveModuleDefinition,
    ) -> None: ...

class PluginBackgroundJobs(Protocol):
    async def register(
        self,
        ctx: Context,
        definition: BackgroundJobDefinition,
    ) -> None: ...
```

两者只返回 `None` 并由调用 Fiber 的 internal Effect + required Health 拥有。插件 facade 不暴露 freeze/unregister；
Core-private registry 在全部 Root settle 后分别产生 `ProactiveCatalog` 与 `BackgroundJobCatalog`。每个 frozen binding
保存 owner Fiber、opaque activation token、required Health、generation/source identity 与 export name；`is_live()` 仅在
ACTIVE + exact token + healthy 成立。

## 2. Proactive committed catalog

### 2.1 Source definition

```python
ProactiveSourceDefinition(
    name="upcoming_events",
    channels=("alert",),
    mcp_server="calendar",
    fetch_tool="get_proactive_events",
    ack_tool="acknowledge_events",
    fetch_page_size=0,
)
```

typed source 结果固定为：

```python
FetchResult = FetchItems(items, cursor) | FetchEmpty(cursor) \
    | FetchSkip(reason, retry_at) | FetchFailure(error, retryable)
AckResult = AckCommitted(ids) | AckSkipped(reason) | AckFailure(error, retryable)
```

- source 引用 exact generation 的 committed MCP descriptor/tool contract；缺 server/tool、重复 name、非法
  channel 或把 ack tool 放进 candidate allowlist 均 fail-loud。
- candidate 只编译引用，不调用 fetch/ack。stable proactive runtime 在自己的 snapshot lease 内取得 MCP route。
- fetch 与 ack 是两个明确阶段；fetch empty、skip、failure 与 ack failure 不得折叠成 success。

### 2.2 DAG module definition

```python
ProactiveModuleDefinition(
    slot="proactive.gate.daynight",
    lifecycle_id="default.proactive.frame.v1",
    requires=(),
    produces=(
        "proactive:gate:pass_probability",
        "proactive:gate:reason",
        "proactive:effect:daynight_gate",
    ),
    collects=(),
    handler_export="run_daynight_gate",
    domain_effect="emotion.state",  # 仅需领域写入的 module
    domain_effect_lookup_export="lookup_emotion_domain_effect_v3",
)
```

- 保留现有 `lifecycle_id/slot/requires/produces/collects` DAG 语义以保证 Daynight/Emotion 行为等价；
  `lifecycle_id` 只能引用 Core 已注册 proactive frame lifecycle，不能提供任意 callable/runtime factory。Core 编译依赖和
  duplicate producer；descriptor 只保存 `handler_export` 与可选的 exact `domain_effect_lookup_export`，candidate/formal 从
  exact Root 各自绑定 handler/lookup，callable 不进 hash。
- descriptor 进入 snapshot identity；candidate/formal handler 与 lookup 分别来自各自 Root，candidate 不执行任何 handler、lookup
  或 transaction。
- module handler 默认只修改本次 `ProactiveFrame` 投影，不持有 Session repository、push tool 或任意 MCP client。
  需要领域状态写入的首个例外是 Emotion：descriptor 显式声明 `domain_effect="emotion.state"`，handler 只能把
  transaction closure 交给 invocation-scoped `ProactiveDomainEffects.run()`；Core 等待该 effect terminal 并记录
  Health/Incident，插件仍拥有 SQLite schema/transaction 实现。未声明 effect 的 module 直接写文件/DB fail-loud。

Core proactive runtime 仍是 turn/delivery owner：它决定 tick、presence、busy、source 拉取、model、持久化、
enqueue 和 send。公共插件没有 `turn_enqueue()`/`send()` capability；因此 candidate 无发送面，普通插件也不能
绕过主动策略。

## 3. Background job / LLM

```python
BackgroundJobDefinition(
    name="merge_proactive_pending",
    triggers=(CoreEventTrigger(CoreEvent.DRIFT_FINISHED),),
    handler_export="merge_pending",
    debounce_seconds=0,
    coalesce=True,
)
```

`ProactiveModuleContext` 提供 invocation-scoped `domain_effects`，不是 Root/public ServiceKey。每次 formal module tick 都由 Core
按 exact `snapshot_id + generation_id + module slot + tick_id` 新建 `DomainEffectContext` 与 `ProactiveDomainEffects`；
`invocation_id` 与 `idempotency_key` 使用 `semantic module id + tick_id`，从而同一 tick 在 Core 进程崩溃后可重入而不重复
plugin transaction。lookup export 只在 formal invocation 的 `run()` 中调用，且必须返回该 invocation 的 durable committed receipt；
普通失败、取消和 handler cleanup 都关闭该 view。bootstrap 不向 adapter 注入共享 effects singleton。

```python
class ProactiveDomainEffects(Protocol):
    async def run(
        self,
        effect_id: str,
        transaction: DomainTransaction,
    ) -> DomainEffectReceipt: ...

class ProactiveDocuments(Protocol):
    async def prepare_pair(
        self,
        expected: ProactiveDocumentDigests,
        content: ProactiveDocumentPair,
    ) -> ProactiveDocumentIntent: ...
    async def commit_after(
        self,
        intent: ProactiveDocumentIntent,
        effect_receipt: DomainEffectReceipt,
    ) -> ProactiveDocumentReceipt: ...
    async def abort_prepared(
        self,
        intent: ProactiveDocumentIntent,
    ) -> None: ...

class ProactiveModuleContext:
    domain_effects: ProactiveDomainEffects

class BackgroundJobContext:
    llm: BoundChatModel
    documents: ProactiveDocuments | None
    def spawn_child(self, awaitable: Awaitable[None], *, name: str) -> None: ...
```

module export 的精确签名是
`async handler(ctx: ProactiveModuleContext, frame: ProactiveFrame) -> ProactiveModuleOutcome`。
两个 proxy 在 Core 构造时已绑定不可伪造 invocation token，插件不读取或传递 token；`run()` 成功返回
`DomainEffectReceipt(effect_id, idempotency_key, state, result_digest)`，失败/取消先结算 transaction，再以带 owner 的
异常使本次 module/job 失败并记录一次 Incident。documents proxy 同理从内部取 invocation identity；
`prepare_pair()` 必须先把两份文档的 `old state(absent | bytes + digest)`、完整 new bytes、idempotency key 与目标路径
写入 `runtime/proactive-documents/intents/<invocation-id>/`，逐文件 fsync 后再 fsync intent 目录，才返回 opaque intent。
`abort_prepared()` 是 DB effect 尚未提交时唯一的中止 owner：它确认目标文档仍等于 expected digest，写入 aborted
terminal receipt 并删除 staging bytes；若目标已经偏移则 fail-loud，不能覆盖第三方修改。
`commit_after(intent, effect_receipt)` 是首个 consumer 的唯一正常提交入口。Core 在 invocation 内记录由
`ProactiveDomainEffects.run()` 真正签发的 receipt object/token；该方法拒绝插件自行构造、其他 invocation/effect 或旧 attempt
的 receipt，向 domain effect adapter 查询 durable DB receipt，原子推进 ledger 到 `phase=documents`，再在内部签发
`DocumentCommitPermit(intent/invocation/DB receipt digest/ledger revision)` 并调用私有 commit。permit 不进入插件 API。
正常提交在同一个 Core pair lock 下、第一次 replace 前重新读取两份目标：必须逐字节等于 intent 的 old bytes/absent marker，
任一偏移即进入 degraded，保留 DB receipt 与 intent，且不写正文。

只有 descriptor 同时声明 `domain_effect` 与 `domain_effect_lookup_export` 的 module 能调用 `run()`；Core allowlist 同时固定直接 source identity `emotion` 与 installed identity `emotion@github`，且只有 `merge_proactive_pending`（保留测试期 `merge_pending`）得到 documents port，
其他插件为 `None`。Core 只拥有 effect terminal/journal 与两份文档原子恢复，不理解 Emotion DB/schema/content。

- 第一版 triggers 只支持正整数 interval 与 `CoreEvent` typed enum。plugin-defined event class/字符串不作为跨 clone
  稳定合同；需要时先在 Core enum 增加事件并定义 payload。
- registration 是 Fiber Effect；candidate 只 freeze catalog，不订阅 EventBus、不启动 timer、不调模型。
- stable runtime 每个 request 在入队时取得 exact snapshot lease；queued/running/cancelled 期间 lease 不提前释放。
- handler 必须 async，返回后才结算 job；异常进入 Incident/structured job failure，不伪装成功。
- `BackgroundJobContext` 只提供 event/reason/time、generation identity 与 invocation-scoped `llm` view。view
  校验 exact snapshot/generation token，只在 handler 执行区间有效；handler 返回、取消或 lease 释放后调用 fail-loud。
  插件在 `apply()` 时从
  `ctx.data_root/workspace_root()` 冻结自己的窄路径，不再取得 v2 `PluginContext`。
- Job execution 在已有 exact snapshot lease 内通过 `CHAT_MODELS.execution()` 建立一个 `ModelExecution`，再把选定 role 的 `BoundChatModel` 交给 handler。调用返回 text、usage 与 binding descriptor；candidate、discarded 或已释放 snapshot 均不可调用。
- stop/cancel 等待 queue task、handler 与 model request 的 owner 结算；caller cancellation 不截断 cleanup。

冻结后的 `BackgroundJobBinding` 至少包含 `generation_id/plugin_id/name/owner_fiber/activation_token/
required_health/handler_export`；Host materialize 时再加入 exact `snapshot_id`。queue、debounce、
coalesce、interval、subscription 与运行中 invocation
使用包含 `snapshot_id + generation_id + activation_token + job name` 的 binding key，禁止退回 `plugin_id:job_id`。
`is_live()` 只在 owner Fiber ACTIVE、token 未变化且 required Health healthy 时成立；旧 snapshot lease 已接纳的请求可完成，
Fiber dispose/restart 后该 binding 不再接收新请求。descriptor digest 进入 snapshot identity，handler 与 token 不进 hash。

每次 job invocation 只持有已有 exact `RuntimeSnapshotLease`、一个 `ModelExecution` 与 invocation token。
`_execute_request` 在实际 handler task 开始时先用现有 `bind_runtime_snapshot()` 绑定 request 的 lease，
并在 `finally` 中 `reset_runtime_snapshot()`；外层现有 cleanup 随后才 release lease。这样
`CHAT_MODELS.execution()` 能在同一 owner task fork exact lease，子 task 不能意外继承可用 binding。
handler 获得的 `BoundChatModel` 核对 snapshot、plugin generation、model binding descriptor 和 invocation identity；
handler 只能用 `BackgroundJobContext.spawn_child()` 进入 Core-owned job scope，scope 在 handler
terminal 前 drain。handler 返回/取消、snapshot lease 释放或 invocation terminal 后，保存的 bound view 和 child
调用均 fail-loud。model revision 变化不改变已复制的 `ModelExecution`；旧 invocation 可继续使用该 exact binding。
cancel/stop 必须等待 handler、provider request 与 snapshot/model execution cleanup；一个失败只产生一个结构化 terminal/retry
记录和 Incident，不能只写 logger。

Core-owned `JobOutcomeLedger` 位于一次性/正式 workspace 的 `runtime/plugin-jobs/outcomes.sqlite`，是 queued/running/
cancelled/succeeded/failed/retry_pending 的唯一 durable owner；`retry_pending` 的 `phase` 固定为 `handler | provider |
documents`，不另造与主状态并列的 `documents_pending`。记录固定 semantic job id（`plugin_id + job name`）、invocation id、
event id/interval bucket、exact snapshot/plugin generation、model binding descriptor、artifact identity/source revision、handler export、
lifecycle/API revision、attempt、state/phase、error、timestamps与 terminal result digest。`cancel_queued` 在 handler 前结算
cancelled；`cancel_running` 取消 handler/child/provider 并等待 cleanup；
`phase=handler|provider` 只有 descriptor 明示 retry policy 且失败发生在可证明零领域 effect 前才进入 retry_pending；
`phase=documents` 是 Core-owned forward recovery，不依赖插件 retry policy，也绝不重跑 handler/LLM/DB effect。restart 只恢复 ledger 记录的
exact artifact/binding。activation token 不持久化；Core 从 ledger 固定的 artifact/source/export/lifecycle identity 重建
exact Root 后生成新 token并再次核对 descriptor digest。找不到旧 artifact/binding 时保持 retry_pending/degraded，不得
fallback 到 current generation。`ActivityHost.recover_pending_documents(invocation_id)` 是 `phase=documents` 的唯一恢复
入口，只读取 DB 幂等 receipt 与 Core pair journal并向前完成/恢复文档，不能重跑 handler、DB effect 或模型调用。
启动扫描发现 durable intent 时，Core 必须先读取 DB receipt：没有 receipt 的 intent 执行 `abort_prepared()`；已有 receipt
的 intent 只允许 forward recovery。`recover_pending_documents()` 由 ActivityHost 从 durable ledger/DB receipt 构造同一内部
permit 并调用私有 commit，不需要已卸载插件 handler；它也必须在同一 pair lock 下执行相同 old-state fence。ordered replace
若部分失败，只有当前文件仍逐字节等于本 intent 刚写入的 new bytes 时，才可用 intent 内的 old bytes/absent marker恢复；
若任一文件出现第三方偏移则进入 degraded、保留 intent/receipt，不能覆盖该内容。digest 不能被当作可恢复内容。
只有 commit/abort terminal receipt 与目录 fsync 完成后才可清理 intent。
`abort_prepared()` 必须在同一 Core critical section 内查询 exact invocation 的 domain effect durable receipt 与 ledger phase；
一旦 receipt 已存在或 phase 已是 documents，就拒绝 abort 并转交 forward recovery，不能写 aborted receipt 或删除 intent。
本轮 crash recovery 只覆盖两类真实 owner：同进程 handler/provider/callback 失败或取消，以及进程崩溃后的启动恢复；
不为断电、磁盘中途失效或任意指令断点另造状态机。

Emotion 保留自己的 SQLite owner，但必须声明 `workspace_roots=("emotion",)` 并只从该窄 root 打开现有
`emotion/emotion.db`。prompt projection 与 DB commit 拆成两阶段：projection 只形成 frame update/domain transaction；
只有 formal invocation 的 `emotion.state` domain effect 才在单个 SQLite transaction 内 commit，candidate 与纯 projection
均零写。Core 不理解或迁移 Emotion schema。

`PROACTIVE_CONTEXT.md` 与 `proactive_pending.md` 继续是 Core proactive 文档，不暴露任意 workspace 给 Emotion。
新增窄 `ProactiveDocuments` port，只接受这两个文档的 expected digest、新内容与 invocation token，以同目录 staging、
fsync、recoverable journal 和 ordered replace 提交；第二个 replace/cancel 失败时依据 journal 恢复两份原始 bytes。
Skill 仍拥有 pending append；ActivityHost 通过该窄 port 拥有一次授权 merge 的发布与恢复，Emotion job 只形成 merge
内容而不取得文件句柄或通用写能力。`PROACTIVE_CONTEXT.md` 的其他编辑仍只属于用户或获授权文件工具。

`CoreEvent.DRIFT_FINISHED` 使用固定 `DriftFinishedEvent(event_id, session_key, skill_name, status, briefing,
message_result, timestamp)`。`event_id` 由一次 drift run 的 Core owner 生成并跨 retry 保持；迁移后只有
`record_commit_result()` 在结果 durable 后发出 typed event，`finish_drift()` 不再发第二份 legacy event。Job host 以
`event_id + semantic job id` 去重；binding key 只选择第一次 admission 已冻结的 exact handler，不参与幂等 key。
重复 delivery 只复用既有 terminal/queued 事实，不重复 LLM merge。插件不得直接订阅
legacy `DriftFinished`。

## 4. 发布与生命周期

1. Root provider 在插件 mount 前创建；register 是 Effect，freeze 后 mutation fail-loud且不能因 cleanup 重新开放。
   frozen binding 记录 owning Fiber active token；owner dispose/restart 后新 admission 隐藏该 source/module/job，旧 lease
   可完成，必须经新 snapshot/formal Root 才能再次公开，不能调用 stale handler。
2. snapshot compiler 分别冻结 source/module/job descriptors 与 exact handler binding；v2/v3 collision 阻止 publish。
3. candidate catalog 不进入 active proactive loop/job runtime；discard 后 handler、subscription、timer、task 引用归零。
4. snapshot identity 分别拼入两个 canonical digest：source descriptor 固定
   `owner/name/channels/mcp_server/fetch_tool/ack_tool/fetch_page_size`；module 固定
   `owner/lifecycle_id/slot/requires/produces/collects/handler_export/domain_effect/domain_effect_lookup_export`；job 固定
   `owner/name/triggers/debounce/coalesce/handler_export/retry_policy/documents_scope/model_role`。tuple 按 owner/name/slot
   排序，enum/trigger 用 canonical value；handler callable、Fiber/token/Health/Root 临时身份不进 hash。
5. snapshot compiler 计算 `activity_changed`，覆盖上述 descriptor digest 与 exact handler binding revision。任何
   activity-only 变化也必须 promotion gated，不能走直接 `promote_latest=True`。
6. promotion 先 seal/formal rebuild并核对 descriptor，再进入 closed provisional target；公开 current 仍为 old，candidate
   catalog 不可见。job timer/subscription 与 proactive kernel 的新 binding 在 closed 状态 materialize。
7. Core 只有一个 `ActivityHost` publication/lease/drain owner。新的 `BackgroundJobActivityAdapter` 与 C20
   `PrivateProactiveHost` 只是
   ActivityHost 的 child adapter，不能自行切 stable/admission。Manager 只绑定一个 `bind_activity_host(...)`：

   ```python
   tx = activity.prepare_transaction(old_binding, target_catalog, target_lease)
   await activity.pause(tx)
   await activity.drain(tx)
   await activity.materialize_closed(tx)
   await activity.finalize(tx)   # 与 stable pointer transaction 同 commit boundary
   await activity.open(tx)
   await activity.rollback(tx, cause)       # critical completion
   await activity.retry_recovery(tx.id)
   ```

   `prepare_transaction()` 只能校验 immutable descriptor、分配纯内存 transaction id，timer/subscription/process/model
   request/client 构造计数必须为 0；只有 old admission 已 pause 且 drain 完成后，`materialize_closed()` 才能创建新资源。

   三类 child adapter 共享同一 Core-private `ActivityChildAdapter[Plan, Binding]` 协议，不复用旧
   `agent.plugins.jobs.PluginJobRuntime`：

   ```python
   plan = child.prepare_components(tx.id, tx.target_lease, tx.target_catalog)
   await child.stop_components(tx.id, tx.old_binding)       # ActivityHost.drain 内
   new_binding = await child.materialize_closed(tx.id, plan)  # old drain 后
   await child.restore_components(tx.id, tx.old_binding)    # rollback only
   await child.close_components(tx.id, new_binding)         # cleanup/retry
   ```

   `prepare_components()` 只返回不可变计划，不创建 timer、subscription、task、model lease 或 snapshot lease；
   计划固定 ActivityHost 传入的 exact target lease/catalog identity。`stop_components()` 只在 admission pause、in-flight
   归零后的 `ActivityHost.drain()` 内调用；全部 old child stop 完成后才能按 plan 调 `materialize_closed()` 并返回 binding。
   后续方法只处理传入 transaction/binding，不允许查询 Manager current snapshot、全局 current job或自行取得/释放
   publication lease；所有错误、cleanup 与 rollback 结果回到 ActivityHost journal。

   bootstrap 只构造/启动一个 ActivityHost，再把 job adapter、public proactive adapter 与 C20 private adapter 注册为子组件。
   proactive kernel swap 继续持有 old lease：`old admission close → old in-flight drain/stop → new build/start(closed)
   → stable pointer + host binding finalize → new admission open`；失败逆序清 new 并恢复 old。旧 job request 使用旧 handler
   直到完成，新 admission 只用最终 committed catalog。start/rollback/cleanup failure 必须进入 reload journal 的
   `degraded/cleanup_failed` 并保留可查询 owner/retry，不能只恢复 pointer。
8. source MCP route、module handler、job handler 与 LLM binding 必须来自同一 snapshot identity；不得从 Manager
   mutable list 或 current global instance 拼接。

## 5. 验证与停止条件

- registry unit：非法 name/trigger/channel/slot/dependency/handler、duplicate、freeze、Effect cleanup、identity；
- candidate：proactive source runtime 的 fetch/ack invocation 为 0、module 零 invocation、job 零 subscription/timer/model、
  stable catalog 不变；C12/C13 semantic readiness 可经 Core-owned controlled recording route 调一次 allowlisted fetch，
  该调用不进入 proactive admission，`source_invocations=0`、`validation_route_calls=1`，并禁止 ack/OAuth/远端写；
- promotion：old kernel、old job 与 candidate binding 同时存在时只有 ActivityHost 能改变 admission；exact formal handler
  生效，old lease/handler 可完成；drift/启动失败恢复旧 kernel/catalog，cleanup/recovery owner 可查询/retry；
- identity：source/module/job canonical field 任一变化改变 snapshot id；candidate/formal descriptor digest 相等，Root/token
  不影响 digest；handler binding drift 触发 activity_changed；
- job：event/interval、跨 generation 同名 job、debounce/coalesce、queue、Fiber token/Health、exact lease、invocation LLM
  token、provider 阻塞 cancel/drain、handler 返回后 child LLM 拒绝、单一 failure Incident、restart；
- proactive：fixed clock + recording MCP/sink，覆盖 empty/skip/fetch failure/model failure/delivery/ack failure；
- proactive domain effect：Manager → ActivityHost formal tick 覆盖成功、普通 transaction failure、caller cancellation 后 view 清理、
  plugin receipt 已提交而 Core 进程崩溃后的同 tick lookup/re-entry；candidate module invocation、lookup 和 transaction 均为 0；
- event：dedupe key 固定为 `event_id + semantic job id`，不含 generation/token；第一次 admission 把 exact binding 写入
  JobOutcomeLedger。事件投递、promotion、worker restart 任意交错，同一 key 最多一次 LLM merge；后续 generation 不得
  重跑，除非原 ledger 明确 retry_pending 且仍可恢复 exact 原 binding。不同 skill/status 保持旧过滤行为；
- data：Emotion frame projection 零 DB 写；formal domain effect 只改变测试副本 SQLite。SQLite 与 Markdown 是两个独立
  durable owner，不伪称跨库原子 transaction：Emotion transaction 以 semantic job id + event id 写幂等 receipt；
  `ProactiveDocuments.prepare_pair()` 必须在 DB effect 前以同 key fsync 含完整 old/new bytes（或 old absent marker）的
  recoverable intent；DB effect 未提交即失败/取消时调用 `abort_prepared()`，并由无 DB receipt 的启动恢复执行同一动作。
  DB commit 后 handler 把 `run()` 返回的 exact effect receipt 传给 `documents.commit_after(intent, receipt)`；Core 核对
  durable DB receipt、原子推进 `phase=documents`，再通过内部 permit commit。pre-effect failure 三者 digest 不变；DB 已
  commit 而 documents 失败/崩溃时，重启通过 DB receipt + durable intent 把 running/retry_pending 统一恢复为
  `retry_pending(phase=documents)`，DB receipt 阻止重复 effect，文档恢复原 bytes 或向前完成，restart
  后最终两份文档一致且同一事件的 DB effect count=1；
- full-fleet 最终只跑 E3 一次，不为每个 source/module/job 各启完整服务。

`cancel_running` 只适用于 handler/provider 尚未形成 committed DB effect 的阶段；进入 `phase=documents` 后，取消/停止
必须等待或保留 Core forward recovery，ledger 不得转成 cancelled。该授权恢复写不是插件“cancel 后继续写”。

出现 candidate 调模型/拉 source/发送、handler 脱离 lease、old kernel 恢复失败却恢复 pointer、job cancel 后继续写
（不含已 committed DB receipt 的 Core documents forward recovery）、
fetch/ack 结果混淆或 private proactive ABI 对 external plugin 可见时停止交付。

## 6. 实现顺序与删除

1. C21a：job registry + snapshot handler + generation-bound LLM，迁 Emotion event job。
2. C15a：source/module registry + proactive kernel snapshot binding，迁 Calendar source 与 Daynight/Emotion modules。
3. 迁 Feed/Fitbit/Steam sources；固定时钟/recording sink 做族群 Gate。
4. C20：Default/Wake 六个内建实现进入 Core-private allowlist/registry，保留领域 runtime/state，不保留通用 v2 ABI。
5. 公共 v2 删除批次与 C20 的 F/H/I/full-fleet 前置共用。zero-consumer 后删除：
   `agent/plugins/base.py` 的 `proactive_*()/jobs()`；`agent/plugins/jobs.py` 的旧 `PluginJob*`/`PluginJobRuntime`
   （新 `BackgroundJobActivityAdapter` 是不同 Core-private 类型，不保留旧 current-job lookup ABI）；
   `agent/plugins/activity_host.py` 的旧 prepared hosts；`agent/plugins/generation.py`、`snapshot.py`、`manager.py` 的旧
   fields/collectors；`bootstrap/app.py` 与 `bootstrap/proactive.py` 的旧 lists/factories 注入；对应 Manager/hot-reload
   v2 tests、manifest/discovery adapter 与旧 Gate；`agent/plugins/specs.py` 的 `ProactiveSourceSpec/
   RegisteredProactiveSource` 与 `proactive_v2/mcp_sources.py/runtime_scope.py` 的旧 source DTO imports 也在 public source
   迁移后删除。C20 私有岛及 `proactive_v2` 领域实现保留到自己的 E3。C15/C21 只移除 `PluginContext` 的 proactive/job
   注入路径；整个 DTO 必须等所有 v2 family 清零并满足 C20/F 才能删除。

不得删除或迁成 `BACKGROUND_JOBS`：`agent/scheduler.py`、`agent/tools/schedule.py`、bootstrap 的 Core Scheduler binding、
`schedules.json` 及其 turn delivery owner。它们是用户持久化 Scheduler，不是 generation-scoped plugin job。

没有 locked consumer 的通用 plugin timer/直接 turn enqueue 不在第一版预建；以后必须由真实 consumer 与独立合同
拉动。Core proactive runtime 自己的 tick/turn enqueue 仍是内部 owner，不是插件 public capability。

## 7. 回滚

Core 恢复点为 `19f2cca2`。所有验证使用一次性 workspace、fixed clock、fake model/MCP 与 recording sink；不写
hua-home、不使用正式渠道 credential。外部插件各自保留 exact base 与独立回滚点。
