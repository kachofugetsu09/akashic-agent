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

## Markdown 档案 I/O（#879）

更新锁继续串行化“读档案 → 模型 → 提交”，两文件短锁保护“恢复/读取”或“安装/receipt”。
完整存储操作交给 `run_file_io`，并在物理结束后才释放锁；模型、binding、summary lookup
和 Scope 留在原 Task。两处历史解码也通过取消排空，避免 provider 先于读线程关闭。
文件锁的非阻塞轮询不变，未把等待文件锁的线程塞进共享池。

before-image、草稿、两份文件及其独立 receipt 的顺序不变；已有退休 PENDING 的协议
也不变，只在原双锁内一次完成恢复和迁移。没有新增数据减少入口或乐观重试模型。
`docker/debug/markdown_io_isolation.py` 在第一份文件写完后持有屏障，验证取消后读者
仍等双文件完成，并检查两份 applied receipt 和原文备份。旧代码阻塞 loop 而失败。

## 远程附件流式落盘（#879）

网络仍由异步 HTTP owner 控制，校验每块实际长度后，通过 `run_file_io` 按序写入和
更新 hash；写完一块才读取下一块，不建立额外队列。每块文件句柄在同一同步工作内
打开并关闭，以增加一次 open/close 的代价避免跨取消点转移句柄所有权。
文件 fsync、rename、目录 fsync 作为完整工作执行，取消后排空，再清理未交付文件。
总超时停止接纳网络数据，但不能中止已经运行的内核磁盘操作，因此排空可能超过时限。
异常路径两个 unlink 仍是同步小操作，本层不宣称消除了全部系统调用阻塞。

URL/DNS 校验、长度上限、hash、最终路径与成功返回时机不变。只删除本次失败下载
尚未交付的随机路径，不触碰已有附件。`docker/debug/remote_media_io_isolation.py`
验证真实字节/hash，并在 fsync 屏障内取消，确认文件只在物理工作结束后清理。

## Scheduler 文件事务（#879）

JobStore 的写入方法改为异步等待文件锁：线程以 `LOCK_NB` 尝试，冲突就立即归还槽位，
回 loop 等待后重试。抢锁、重读、校验、修改、序列化、原子保存和释放是一个物理工作。
锁只活在物理事务内；提交结束即释放，不依赖 loop 再次调度。
不同 generation 仍共享原 `.lock` 文件；取消排队写入
不会进入事务，取消已开始的写入可能留下成功 receipt，工具 query 可读取原结果。

Scheduler 的运行与工具读取也交给 worker；Task 接纳、消息追加和 Delivery 留在原
Task。诊断公开读取的异步合同见下节。无消费者的 `save(jobs)` 全量覆盖入口已移除，修改只通过具名领域操作。文件 schema、操作和触发回执不变，
任务只沿显式取消减少，过期和完成仍按原规则逻辑失效。

`docker/debug/scheduler_io_isolation.py` 使用两个 JobStore 与真实 ScheduleTool，
控制第一代原子保存、取消调用方后让第二代提交，验证任务和操作回执均未丢失。
本层保留 runtime 的同步 stat 检测，未增加版本表或文件 CAS 协议。

## Scheduler 诊断请求（#879）

Web jobs 请求直接经 ScopedRpcRuntimeInspection → RpcMethod → runtime_inspection →
SchedulerReader，原链路没有 UI executor。v3 将两项公开读取显式改成 async，在实际
Scheduler provider 中 offload load，RPC 逐层 await。请求通过 borrow 保护当前 owner
直到读取排空；不再保存另一份 scheduler 指针或用可选子 Fiber 维护这份指针。

公开合同只保留 `SCHEDULER_INSPECTION_V3` 和异步 SchedulerReader，相关插件统一
使用 v3；旧同步插件输入需要重新准备，不提供兼容层。HTTP/RPC 的值格式与既有
缺能力错误保持，存储内容与恢复回执不变。
`docker/debug/scheduler_inspection_isolation.py` 经过真实 Root、客户端 adapter 和
RPC，验证缺能力、正常列表/详情、慢读不冻结 loop、取消排空前 owner 不卸载。

## 流式读取与 Turn 引用（#879）

MessageReader.scan 在一个短读快照内逐页交给同步消费者，页大小 64；迭代器离开
回调即关闭，不能跨 await 或把连接交给插件。完整 snapshot 仍明确返回全部正文。
TurnProjection 接收有序流，待闭合状态只保存 seq、消息 ID 和调用引用，不保留正文。
显式 after_seq 只能来自已提交闭合 Turn；重读尾部时忽略已消费边界内的 abandon。
这是投影消费起点，不是新的执行事实；原始 Message、schema 和删除权限保持不变。

能力 owner 为 MessageLog 的只读快照和 turn_projection 的分段算法；消费者为 Reply
与外部 Observe。Core 仅增加中立 scan，无插件名称或专属状态。普通客户端无法独立
实现同一 SQLite 快照下的流式读取，因此需要存储 owner 提供窄接口。
验证见 docker/debug/turn_streaming.py：真实 SQLite 分页、abandon 后仍开放的输入、
迟到工具结果、从闭合边界重读等价，以及大开放 Turn 的正文对象可回收。
恢复点为 pre-memory-stack.bundle；不迁移、删除或重写正式消息和插件数据。
