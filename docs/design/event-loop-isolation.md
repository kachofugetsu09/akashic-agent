# Event-loop execution boundaries

Related issues: #827 (stack), #828 (storage), #829 (interest), #836 (materials).

Long history reads use a private read-only SQLite transaction and connection. Nested readers of the same MessageLog in that synchronous call share its read snapshot; they do not acquire the writer's connection or decoded-message cache. Reads inside an existing write transaction still see that transaction's uncommitted rows. Owner transactions remain synchronous and atomic, and listeners wake only after commit.

MessageLog uses file-backed WAL mode so a pinned read does not delay a writer's commit. This changes the runtime journal mode, not the schema. Backups must use SQLite backup or include the SQLite sidecars; copying only the live main database file is not a snapshot. Existing databases keep their schema and data; an unsupported journal mode is rejected explicitly. Short synchronous writes can still wait on SQLite file-level contention; this change does not claim that all storage I/O is asynchronous.

Reply preparation captures the source head and full message head before awaiting history. Async warmup decodes only that fixed prefix, drains the worker on cancellation, and installs the existing incremental cache only if the original connection's data version is unchanged. An external edit during the read prevents cache reuse; the caller still receives that one consistent SQLite snapshot without retrying or blocking the event loop. Later appends are read as a tail; external edits invalidate the cache on the next read.

Interest scoring keeps model selection and candidate embedding in the original async owner. Historical sample and prototype construction run in a drained worker without changing the formula, sample order or cutoff.

Independent context material owners prepare in a TaskGroup. Their results merge in the fixed source order with the existing output sorting and conflict checks. An ordinary failure drains all started siblings and is then reported in frozen source order. Cancellation cancels and drains siblings. Owner scopes close before either reaches the caller. Ordered transform events and the unique summary reducer remain sequential.

Joined material tasks use the generic child context boundary. It removes task-owned handles and lets their owners copy only frozen request data. Models transfers the parent settings snapshot and explicit model choices; each child opens and closes its own Models and driver scopes. A settings update cannot change that reply's model revision. Raw child tasks still cannot borrow a parent's model execution, and independent Tasks start with fresh choices. This boundary is covered with real ModelsState, concurrent settings updates, nested children, and cancellation (#847).

```text
fixed input / head
       │
       ├─ independent read or compute job
       │       └─ await actual completion, including on cancellation
       ▼
ordered merge / state-dependent transition
       ▼
durable commit → notify → next dependent operation
```

Validation is tied to real MessageLog and CompositionRoot boundaries. The new regressions fail before their fixes; the stack retains the existing per-graph publication and ordered tool-result tests.

Akasha keeps one notification event and worker per routed graph. Notifications coalesce while that graph is busy. A graph access lock orders startup, consumption and live material preparation; the MessageMemory lock still guards each transition and publication. Explicit recall reads the published graph without the graph access lock; rebuild publishes its candidate by atomic file replacement. A starting graph reserves its embedding space before awaiting, so a concurrent graph cannot silently bind a changed space. Startup schedules these consumers without waiting for the default graph's backlog. Explicit rebuild closes graph admission, drains existing users, then closes writers and rebuilds; admission reopens even on failure. Fiber shutdown closes call admission, cancels and drains background tasks, then waits for accepted calls before emitting RUNTIME_STOPPING and closing memories.

Delivery policy fixes selections and its Session cursor before sending. Each `(Session, sink)` worker reads bounded pages of that durable prefix in order; a slow sink owns only its own send wait. Notifications coalesce and new message backlogs remain in MessageLog instead of an unbounded in-memory queue. Restart recovery keeps the original prepared destinations and receipts. Failures still settle through Delivery, and shutdown drains its owned tasks.

Shell spawn and owner cleanup share only that owner's gate. Process capacity and spawn registration still use the existing short global gate; capacity eviction can still delay another spawn. Whole-manager shutdown closes owner admission, drains active owner operations, then confirms process cleanup. Failed cleanup keeps the existing owner quarantine and execution evidence.

Workload start/adopt/stop serialize by plugin, since workloads of the same plugin can share writable mounts. Different plugins' Docker I/O can overlap. Exact lease checks and atomic lease/stop-receipt files remain authoritative. Candidate cleanup and Core-owner recovery still close all admission and drain active effects before workspace-wide cleanup. Controller shutdown still drains accepted requests through the existing effect wrapper. The regression controls Docker responses and verifies durable receipts; it is not a real Docker acceptance run.

A due Alert is selected before taking the Content maintenance lock or calling semantic interest. Its request keeps the real unscored Content snapshot and reports scoring as deferred; it does not persist fake zero scores. The due loop requires the chat model but does not require semantic-interest readiness for an already due Alert. Content/Drift admission still uses the original scoring rules. This bypass covers Alerts due at admission; it does not preempt an already accepted Wake flow or a Content check already awaiting its score.

Plugin UI queries share eight physical workers with a per-plugin limit of four running and twelve admitted queries; the existing global admission limit remains 24. A single slow plugin cannot occupy every worker or every admission slot. Timeout/caller cancellation withdraws work waiting for a plugin slot or an executor start. A running thread cannot be stopped: it keeps its original captured scope and quota until it physically finishes, and provider shutdown waits for that completion. These limits do not promise capacity when several different plugins saturate the shared pool.

## 子任务终态与回传读取

子任务的启动、结算、取消、同步工具回查和后台完成回传使用既有
`MessageReader.snapshot_async` 读取固定前缀，避免主会话或子会话的完整历史解码
阻塞事件循环。读取取消时仍排空 worker 后退出，不提前关闭其数据库资源。

取消命令在读取前固定子来源 head，以同一 head 条件追加 pause；如果读取期间子任务
有新消息，提交冲突后重新检查终态。刚完成的任务不会被旧快照改成 cancelled，
已经保存的 Control/Output 和所有历史正文不变。来源规则继续由 subagent 拥有，
Core 不增加子任务状态或来源专属查询。

`tests/test_storage_read_execution.py` 的子任务回归用真实 MessageLog 和受控解码屏障
守护 O/C3/C4：历史读取期间对等来源和 pause 都能提交，旧快照不吸收后来消息。
已有存储层回归不能证明实际子任务消费者使用异步入口，所以增加该消费者回归。
其他来源的同步历史入口仍需独立定位与验证；本变更不宣称全系统 I/O 已异步化。

## EventMail / Drift 的读写事务（#879）

插件 apply 在发布能力前，通过 `run_file_io` 完成建库、原有迁移和完整性检查。
取消会排空已开始的初始化，之后才释放 generation。运行期只读入口要求已初始化，
使用独立 `mode=ro` 连接、`query_only` 和普通读事务，不再申请 `BEGIN IMMEDIATE`。
写事务和原有 state_version 条件提交保持不变；只读快照不能升级为写事务。

这保留了 SQLite 的单写者约束，释放了读者不必要占用的写锁，符合
[SQLite WAL](https://www.sqlite.org/wal.html) 的读写并行模型。
快照仍是同步 API；较大的读取及运行期写入隔离是后续工作，不宣称全部 I/O 已异步化。
建库与 schema 迁移只能由已有写路径执行；只读入口不再隐式创建或升级数据库。
现有 EventMail v0/v1/v2/current 与 Drift v0/v1/current 迁移和 schema identity 校验不变。
消息正文、mail envelopes、提案和回执的增加、更新及删除权限均不变。

`PYTHONPATH=. .venv/bin/python docker/debug/store_read_isolation.py` 使用临时数据库，
持有未提交写事务时验证已提交快照可读，提交后新快照可见，并检查数据库完整性。
旧代码在读取时报告 database is locked；无需靠 sleep 调度。

## 插件源码准备（#879）

PluginManager 保留唯一操作 owner。变化检测的源码哈希，以及完整的检查、复制、
编译、归档事务交给 `run_file_io`；模块导入、Scope 创建、正式 selection 条件提交、
挂载与事件仍在原事件循环执行。取消观察者会撤销提交许可，但 Manager 在物理工作
完成前仍拒绝新操作。未发布归档沿用原有保留规则，不把取消伪装成文件回滚。

`test_cancelled_source_preparation_drains_before_manager_reuse` 守护 O/C1：
真实 Manager 热更新在归档屏障处仍能处理取消，排空前拒绝重用，旧 generation 和
selection 不变，排空后可正常重试。旧的编译失败与局部更新测试不能覆盖后台线程
尚未退出时的 owner 生命周期；该回归在同步实现上因 loop 无法接收释放信号而失败。
