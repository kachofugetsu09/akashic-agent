# Event-loop execution boundaries

Related issues: #827 (stack), #828 (storage), #829 (interest), #836 (materials).

Long history reads use a private read-only SQLite transaction and connection. Nested readers of the same MessageLog in that synchronous call share its read snapshot; they do not acquire the writer's connection or transaction lock. Reads inside an existing write transaction still see that transaction's uncommitted rows. Owner transaction callbacks remain synchronous and atomic, including when pure SQL runs in a drained worker. Listeners wake only after commit.

同一 MessageLog 的连接共享弱引用解码缓存，只在完整数据库行相同时复用仍被调用者持有的不可变 Message。每次读取仍执行原 SQL，先由该连接的事务确定可见行；外部修改、撤销和回滚不能通过缓存隐藏。缓存使用独立短锁，不持有 writer 锁，也不延长 Message 的生命周期。
活跃 ReAct Loop 持有当前历史直到下一份真实快照替换；不在轮间主动丢弃仍供下一轮使用的事实。
这使超过 reader 有界缓存的长循环也能复用仍存活的 Message。每次读取继续查询原 SQL，
按完整行区分外部修改；没有扩大全局缓存或跳过失效检查。Loop 结束、取消或失败后释放历史。

OwnerRecord 也只在实际读取的版本和 JSON 正文完全相同时复用解码结果。弱引用不保留无人使用的大请求；SQL 读取、版本 CAS、固定快照与损坏记录报错保持不变。

会话目录的 heads 快照只在同一只读连接的 SQLite `data_version` 未变时复用；不同连接的版本不能相互比较。每个连接只保留最近一份目录，外部或本地 writer 提交后重新读取；当前 writer 的未提交事务不使用此缓存。固定只读事务仍观察原快照，归还连接后下一次读取再观察新提交。

消息订阅只在事务实际新增消息、接纳会话或修改会话管理属性后唤醒。单独更新 owner 记录或 embedding 不产生消息变化，不广播空唤醒；同事务追加消息仍在提交后通知，失败回滚不通知。关闭唤醒与显式轮询保持不变，通知本身不替代数据库事实。

普通 `MessageReader.snapshot_async` 与 `OwnerStore.snapshot` 的只读准入不等待另一个线程
持有的 writer 锁。当前线程能重入的写事务仍使用原连接，读取自己的未提交行；其他读取
直接打开独立只读事务。准入与 close 使用短锁：close 拒绝后来读者，已取得的只读连接
仍由原同步读取关闭。来源提交和首次效果的异步顺序见下面的 #879 说明；权威消息仍只追加。

MessageLog 最多保留四个空闲只读连接。一次读取独占借用的连接，嵌套读取沿用该快照；归还前结束事务，下次借用重新开始事务。连接仍为 `mode=ro` 和 `query_only`，不能升级为 writer。close 拒绝新读取并关闭空闲连接，已经准入的读取完成后关闭自己的连接。复用连接不复用事务，也不延长旧快照。

```text
┌───────────────────────┐       ┌────────────────────────┐
│ writer：未提交事务       │       │ reader：独立只读快照     │
│ 继续等待或提交           │       │ 只看已提交的固定前缀      │
└───────────────────────┘       └────────────────────────┘
```

短 Reader/OwnerStore/Catalog/binding 读取使用同一 private RO 准入，自己的写事务内读取仍重入原连接。
listener 登记和释放只持有其现有注册表的短锁，不等待 writer 的磁盘工作。
增量 reader 的同步调用、剩余 Core 写入和关闭仍有同步路径；不能据此声称全部 I/O 已异步化。

MessageLog uses file-backed WAL mode so a pinned read does not delay a writer's commit. This changes the runtime journal mode, not the schema. Backups must use SQLite backup or include the SQLite sidecars; copying only the live main database file is not a snapshot. Existing databases keep their schema and data; an unsupported journal mode is rejected explicitly. Short synchronous writes can still wait on SQLite file-level contention; this change does not claim that all storage I/O is asynchronous.

回复准备在等待历史前固定来源 head 和完整消息 head。回复范围内的增量 reader 持有
已提交消息前缀，生命周期与当前回复一致；同步 scan/snapshot 和异步 snapshot 共用同一读取路径。
正常追加只补读尾部，来源筛选与旧前缀查询不覆盖完整视图。没有全局永久历史缓存。

已迁移的库在同一只读事务中读取 `message_prefix_revision`、实际 head 和新增消息。
正常追加不改变旧前缀；消息 UPDATE、DELETE、旧位置的 INSERT 和 REPLACE 由 SQLite
触发器在原事务推进标记，包括其他连接的写入。标记不同就完整重读；回滚也回滚标记。
因此内部 writer 忙于正常提交时，reader 仍可复用本次快照中有效的旧前缀。
已有事务直接读取原事务，不借用缓存或发布可能回滚的消息。异步取消仍排空物理读取。

```text
┌───────────────────┐    ┌─────────────────────────────┐
│ 当前只读事务       │───▶│ 前缀标记 → head → 新增消息   │
└───────────────────┘    └─────────────────────────────┘
                         标记相同：复用旧前缀
                         标记不同：重读当前快照
```

标记只帮助判断内存视图是否有效，不是消息事实，也不授予编辑或删除权。
新库由 MessageLog 初始化；旧库通过 Yoyo 只增加表、初始行和触发器，不改写原消息。
迁移不完整或初始行缺失时启动失败。尚未迁移的已知旧库仍使用 writer 的
`data_version` 检查；无法检查时完整重读，不降低原读取保证。

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

## 显式重试的读取范围（#869）

SourceSession 在同 key 的异步准入内接纳 resume 与显式 abandon。纯历史判定通过
`MessageReader.read_async` 在独立只读快照中完成；resume 只查最后同来源 Input、Control，并分页
扫描该 Input 后的同来源消息，abandon 分页核对同来源终态。worker 只返回小判定或
Control/head，不把历史正文带回 loop。已提交 resume 的同 ID 重放只核对原 through_seq
内最后 Input，不解码更早的关闭正文，也不改用当前 head、活动 handle 或重启状态。

Task、handle、重启闸门和提交通知留在 loop；同 key 准入一直持有到物理读取与原提交结束。
读取期间的新 Output 要么已在固定快照中参与判定，要么使原来源 head CAS 失败；显式
head 不重试、不重选。取消先排空读取再释放准入，服务关闭仍等待排空。最新输入、已有
控制、完成/abandon 拒绝与条件追加规则保持；没有把存储事务跨 await，也不删改历史。

`docker/debug/resume_history.py` 在一次性 SQLite 上量测大闭合历史的首次重试与
同 ID 重放，并核对关闭、活动任务和来源边界。它也量测已异步化的子任务 outcome，
把解码时间与事件循环回调延迟分开；本机样本不代表生产 p99。
`docker/debug/source_control_read_isolation.py` 通过真实 MessageLog、Tasks 和大尾部
解码屏障核对对等来源可提交、同 key Input 不能插队、读取中完成触发 CAS 冲突、重复
取消与服务关闭排空、同 ID 重放及来源身份验证；逐页正文可释放，原历史摘要和数据库
完整性不变。它不启动真实 provider 或投递。开放尾部其他读取与同步通知的隔离
沿用下文 #933 的统一窄读取入口和 pending 归约，不再保留单独的控制读取 helper。

## EventMail / Drift 的读写事务（#879）

插件 apply 在发布能力前，通过 `run_file_io` 完成建库、原有迁移和完整性检查。
取消会排空已开始的初始化，之后才释放 generation。运行期只读入口要求已初始化，
使用独立 `mode=ro` 连接、`query_only` 和普通读事务，不再申请 `BEGIN IMMEDIATE`。
写事务和原有 state_version 条件提交保持不变；只读快照不能升级为写事务。

这保留了 SQLite 的单写者约束，释放了读者不必要占用的写锁，符合
[SQLite WAL](https://www.sqlite.org/wal.html) 的读写并行模型。
EventMail 快照与写入仍是同步 API；Drift 的 Wake 消费路径使用下面的 v2 异步能力。
建库与 schema 迁移只能由已有写路径执行；只读入口不再隐式创建或升级数据库。
现有 EventMail v0/v1/v2/current 与 Drift v0/v1/current 迁移和 schema identity 校验不变。
消息正文、mail envelopes、提案和回执的增加、更新及删除权限均不变。

`PYTHONPATH=. .venv/bin/python docker/debug/store_read_isolation.py` 使用临时数据库，
持有未提交写事务时验证已提交快照可读，提交后新快照可见，并检查数据库完整性。
旧代码在读取时报告 database is locked；无需靠 sleep 调度。

## Drift 的 Wake 存储调用（#879）

Wake 使用 `drift.wake.v2` 和 `drift.delivery.v2` 等待快照、领取、状态流转和送达结算。
Drift provider 把完整同步操作交给已有四名额 `run_file_io`；连接、事务与关闭在同一
worker 完成，调用方取消仍等待物理退出。Context、Task、程序执行、Message、通知与
最终 flow pointer 继续由 Wake 在原 Task 操作。没有新增队列、表、状态副本或 executor。

```text
┌─────────────────────────────┐    ┌────────────────────────────────┐
│ Wake：固定请求与原领取身份     │───▶│ Drift：worker 内读快照或完整事务 │
│ await 回执，再继续原流程      │◀───│ 原 state_version / token 校验   │
└─────────────────────────────┘    └────────────────────────────────┘
```

同版本竞争仍由原 SQL CAS 决定；取消不能撤销已提交的领取或结算。Wake 恢复读取原
accepted Input 与 selection token：准备完成后才发送，真实 Delivery 回执后才结算，
结算提交后才关闭 flow。提交后响应丢失使用原领取/settlement 重放，不创建第二份提案。

Drift 只发布 v2 异步能力；旧消费者必须迁移后重新准备，旧服务名不再解析。
新 Wake 明确依赖 v2，旧 provider 缺少 v2 时由既有依赖解析阻止激活，不猜测或降级。
来源上报使用 `drift.proposals.v2` 可等待接口；存储事实和领取身份仍由同一 DriftStore 拥有。
EventMail/Alert 已按下文迁入 v2 与共享准入段，保证原版本检查覆盖 Delivery started 提交。

持久化 schema 不变：来源仍按原身份增加 proposal；领取、流转与结算只更新同一行的
状态、版本和原回执字段；终态是逻辑变化，没有新增物理减少权限。恢复点是原数据库
备份与原 proposal/selection/settlement，代码回滚不改写这些事实。

`docker/debug/drift_io_isolation.py` 用真实临时 Root、SQLite 写锁、提交屏障和 Wake
任务验证计时器推进、同版本竞争、失败回滚、重复/冲突结算、取消后 owner 排空及重开。
旧同步基线保存在 Git 历史；当前场景只运行实际发布的 v2 能力。
Wake 的来源拒绝分支保留原 Input 并完成唯一 flow；未使用的 binding 仅作持久引用夹具，
不代表真实模型、发送、正式插件换代或生产延迟验收。正式 workspace 未写入。

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
Message ID 唯一性由消息库主键拥有，投影不再另建全历史 seen 集合。
显式 after_seq 只能来自已提交闭合 Turn；重读尾部时忽略已消费边界内的 abandon。
这是投影消费起点，不是新的执行事实；原始 Message、schema 和删除权限保持不变。

能力 owner 为 MessageLog 的只读快照和 turn_projection 的分段算法；消费者为 Reply
与外部 Observe。Core 仅增加中立 scan，无插件名称或专属状态。普通客户端无法独立
实现同一 SQLite 快照下的流式读取，因此需要存储 owner 提供窄接口。
验证见 docker/debug/turn_streaming.py：真实 SQLite 分页、abandon 后仍开放的输入、
迟到工具结果、从闭合边界重读等价，以及大开放 Turn 的正文对象可回收。
恢复点为 pre-memory-stack.bundle；不迁移、删除或重写正式消息和插件数据。

## Reply 的阶段内存（#879）

启动选模用 scan 分段，并用 include_closed=False 不保留已闭合 Turn 的引用，
只在同一快照内读取开放 Turn 的 Input/Output，选模与工具
起点计算后释放这些正文。React 工具结算从流式扫描取得调用引用，不持有完整前缀；
完成一轮输出后释放该轮快照和材料，再进入可能很慢的工具执行。
增量 reader 只保留最多 256 行且序列化正文/metadata 不超过 4 MiB 的小前缀，大会话
按需读取；显式 after_seq 尾部请求不为缓存补读前面的历史。字节门槛不是 Python RSS
上限。消息 schema、请求冻结、CAS 与失败恢复不变；全部数据仍留在原数据库。

这是减少重复表示与引用寿命，不是改变模型上下文合同：构建请求和容量拒绝缩减期间
仍使用同一完整快照，模型可见窗口、摘要与工具回放不被截短。大历史再次解码存在
CPU/I/O 代价；完整请求快照与恢复快照通过有界 I/O worker 解码，取消后排空。
后续若改为数据库支持的上下文窗口，应另验摘要和任意材料插件合同。
docker/debug/reply_memory.py 验证小前缀复用、大正文同步/异步释放和固定范围读取。


## Source 提交与首次效果（#879）

Source 的 Input/Control 接纳使用 `Tasks.admit_async`：同 key 的接纳、启动与控制不能插队，
SQL transaction 本身仍是同步回调。关闭拒绝后来操作，等待已接纳操作及实际 worker 排空。
既有 `Tasks.admit` 仍拒绝异步回调；活动句柄和长任务不占接纳锁。

```text
┌──────────────────────────────────────────┐
│ loop：固定当前 owner 的内容校验与 metadata │
│ 固定内容引用、授权 metadata 和纯投影      │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ worker：先读取同 ID 收据，再核对 grant/head │
│ 引用与 CAS，原子追加，不调用 Context       │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ loop：通知一次，再撤权；取消也先交付收据   │
│ caller/Scope 在物理排空后才退出            │
└──────────────────────────────────────────┘
```

Tool started、command intent 和 generation claim 把 Source 前提检查放进原 Core owner transaction。
判断来自该 Task 已接纳的输入边界或冻结读取前缀，不能在等待后用最新 head 替换原前提。
若新 Input/Control 先提交，旧首次 intent 回滚；若 started 先提交，原 owner 如实结算已开始效果，
取消不伪称效果回滚。ToolResult 与 done 仍在原事务共同提交。

生成请求的正文、材料与启动身份在同一次 owner transaction 内保存，来源检查在该事务内完成。
不再先写冻结请求，再读写整份记录领取生成。模型 I/O 仍在提交之后；旧的已冻结但未领取记录
在恢复时复用原内容，并通过来源检查补齐启动身份。`started_attempts` 只记录该准备已通过首次启动检查，
不复制 Models 的调用事实。Models 仍拥有独立数据库和发送状态，按同 key 前向恢复；
Core claim 不能证明远端有或没有效果。旧 v2/v3 准备和消息表示保留，不做 schema 迁移或历史改写。
准备的纯 SQL 也离开 loop；Context 和模型句柄的读取保持在原 scope。

能力必须按完整组选择：`sources.v5`、`source.session.v4`、`source.check.v2`、
`source.changed.v3`、`channel.input.v2`、`source.interrupt.v2`、`tools.program.v2`、
`react.ordered-start.v2`、`conversation.complete.v2`、`reply.program.v3`、`reply.execute.v4`。
旧公共常量及旧二参数完成回调的结构合同保持原值，新 provider 不提供旧 alias。
来源注册表也换 key，因为其 open 返回的 session 带新的完成回调合同。旧 actor 与新 provider、
或新 actor 与旧 provider 不能静默混用；不能只迁移直接 factory 而漏掉注册表消费者。

Manager 的逐插件局部更新不能跨越整组能力版本：它会明确拒绝旧依赖 PENDING，并恢复旧组。
跨版本启用应在新 Root 中选择完整组；正式选择、安装和启用验收属于发布流程。
完整旧组仍可恢复。保持同一能力版本的局部替换仍在同 Root 完成，并先等待已接纳 Source worker 排空。
原候选 `bac7877` 的本地 scenario 验证过拒绝、恢复和同版本替换；该证据不自动覆盖后续能力组。
当前源码的原概念测试、来源路由与材料场景分别核对实际 Root 和当前 key；新的完整跨版本
Manager 发布/旧组恢复仍需独立验收。没有正式 workspace、账户或付费 provider 写入。

回归 `test_source_commit_drains_before_cancel_and_rejects_late_start` 在 `32b0aaf2` 的真实通知边界失败，
候选的普通取消及服务关闭场景均通过。它守护 C3/C4/C5 的收据、终态和效果顺序，不改变既有消息正文。
剩余同步消息/owner 写入，以及 EventMail/Drift/Alert 的多库顺序继续独立处理。

现行源码的回复、图片、费用、Source 控制及性能场景显式使用 `channel.input.v2`，不会向当前
provider 请求已退役的旧键。`content_view_scenario.py` 另核对全新进程中的完成/失败同 ID 重放；
`--directory-page` 使用实际目录页验证完整显示与回读。`scripts/check_list_dir.py` 在真实目录
handler 完成后丢一次 RPC 响应，核对无自动重发、断连重接后的显式读取及全部 manager 排空。
这些是临时状态和受控 provider 的验收，不是生产性能或模型推理质量证据。

内部 pause 等待磁盘时，活动回复仍可能追加 Output。来源 head 的 CAS 失败不提交任何事实；
内部 pause 重新读取前缀再尝试。显式 control 的 expected head 不重试，身份、权限或引用错误也不重试。
停止完成前后已提交的 Input/Output 都保留，不能用撤销正文消除这类竞态。

直接调用回复程序的 Scheduler、Subagent 和 Wake 也由各自来源固定实际 Input 的 seq，
不能依赖只在 SourceSession 设置的默认 Task 边界。Scheduler 使用 append 的原收据，
Subagent 核对原请求的同来源 Input，Wake 按每个实际阶段 Input 固定边界；重放不吸收
后来 Input/Control。共享回复程序仍拒绝被替代的边界，不把负边界放宽为当前 head。
`docker/debug/source_reply_boundaries.py` 在真实来源、Task、MessageLog、回复程序、ReAct
和 Models 账本上验证首次/恢复调用、后来输入/控制拒绝和 Wake 同 Task 多阶段。
模型 driver 与发送端为本地夹具，不代表真实 provider 或正式投递验收。


### 跨来源完成回传的控制前提

后台 Subagent 回传由父 Conversation 的空闲准入控制，但新 Output 属于该子任务的独立来源。
只检查输出来源会漏掉父 Conversation 已提交、尚未发送取消通知的新 Input/Control。
SourceSession.complete 在准入时固定原 reader/source/seq，并以 SourceGuard 闭包交给完成回调；
它沿 ConversationComplete、ReplyProgram 和 ReplyExecute 的显式新合同传递，没有新 Core 状态表、
Task 来源字段或全局查找表。改变输出来源不改变这份控制前提。

回复入口先核对原控制前提，材料阶段及 Tool/Models 首次效果再同时核对控制前提与输出前提。
新 Output 在原 owner transaction 内也执行同一检查，不能把一次 started 许可当作永久提交权。
已存在 Output 的同 ID 读取仍复用原事实；已开始 Tool 的真实成功 ToolResult 与 done 仍按原协议
共同提交，已保存 Models response 仍保留，取消不冒充外部效果回滚。

`docker/debug/completion_source_ordering.py` 运行真实 Subagent 回传、SourceSession、Task、
回复程序、ReAct、ToolExecution、Models/Message 存储。真实 SQL 提交后延迟 loop 撤权，分别在
入口、生成 claim、工具 started、实际本地 fsync 效果后及模型成功后的 Output 前设置屏障，
核对 Input/Control 两种替代、成功路径、原历史和完整性。脚本使用本地 model driver 与发送端；
不是正式安装、跨进程恢复、远端 provider、真实发送或生产延迟证据。

## Source 的开放尾部读取（#869）

已闭合历史跳过后，最新 Input 或工具结果仍可很大。Source 的待回复、恢复、停止、控制、
启动、材料完成与失败判定使用 `MessageReader.read_async`：同一个 worker 私有只读快照
完成 SQL、正文解码和规则归约，只返回短命判定或调用者确需的材料。该回调必须同步；
不能传入 Context、Task、SQL transaction 或写入能力。原四个 worker 名额及取消排空保持。

```text
┌─────────────────────────────────────────┐
│ loop：同来源准入锁，固定 Task/hint        │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ worker：私有只读快照，归约 head/pending   │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ loop：重查 Task/head；原 writer 条件提交   │
│ 同步通知 pending、活动占位和撤权          │
└─────────────────────────────────────────┘
```

待回复规则仍由 Source 拥有，判定结果不形成缓存、cursor 或另一份持久事实。显式 control/resume
绑定读取的来源 head；后来输入导致 CAS 冲突，不把旧前提换成新 head。start/complete 在等待后
重查 Task 与 head，原首次效果 fence 保持。材料等待通过 `follow_heads` 订阅提交序号，
不先解码全部历史；订阅退出立即释放 listener。同 ID Input/Control 的早期收据读取也离开 loop。

提交回调向 `source.changed.v3` 传入该提交前缀的 pending，Reply 同步占用或释放原活动计数。
启动恢复等待读取后重查 head 和来源身份，避免旧读取释放新输入的占位。Sources 的异步
needs_reply 属于 `sources.v5`，工厂属于 `source.session.v4`；旧公共类型和键保留归档含义，
当前 provider 不提供旧 alias。ChannelInput、SourceCheck 和 SourceInterrupt 的已有签名保持。
完整旧归档使用旧组，新旧半组保持 PENDING；正式跨版本启用仍走新 Root 的完整选择。

Conversation 命令材料、Programmatic 结果读取与提交帧结算也使用纯读取 worker，Context、
命令首次效果和连接帧状态留在原 loop。同步 transport frame resolver 及其他 owner 写入仍属
#879 的独立范围；本层不宣称全部 CPU/I/O 或生产延迟已经解决。

Programmatic 的结果查询只返回带 `through_seq` 的快照，不回收连接。后台提交订阅是终态
通道清理的唯一发起方；异步判定返回后重查来源 head，前缀改变就等待订阅重新读取，
不能用旧 pause/failure 结果释放同一 Input 恢复后的通道。FrameBook 仍拥有连接、claim 和
writer drain。连接 reservation 早于 Input 的实际提交，后台只处理快照中已经存在的
programmatic Input；暂时没有正文时等待原提交，不把合法准入窗口解释为损坏。

`source_read_cohort.py` 使用实际安装的 Programmatic、Source 和 FrameBook，确定性延迟
结算与结果查询的旧快照，再从真实 resume 入口恢复到新连接。场景验证两条路径都保留新
通道、最终 Output 对应的受控 writer future 完成、稳定终态仍回收通道；同时核对 reservation
先于 Input 的窗口、原消息不变与 SQLite 完整性。传输 future 是本地受控边界，不是实际网络送达。

`docker/debug/source_read_isolation.py` 使用原 writer 创建 1 MiB Input 与 12 MiB ToolResult，
记录解码线程、首次 peer callback 和读取全程的最大 loop 心跳间隔；首次让出不代表持续响应
上界，线程中的 CPU/GIL 竞争仍可能造成停顿。另以确定性解码屏障验证其他来源 append/pause、同来源新输入
冲突、取消及 Tasks.close 的物理排空。原行逐字段相等，integrity/FK 检查保持；未运行付费
provider、正式 workspace 或生产 p99。只追加原协议允许的 Input/Control，不迁移或减少历史。

`docker/debug/source_read_cohort.py --previous-source <旧源码>` 使用仍提供 v4/v3 的完整旧源码，
经临时 Manager 验证旧归档执行、半组拒绝和失败局部更新后的恢复；当前安装的 Reply 在真实
Input ACK 前取得活动占位，同 ID 重放不重复通知或执行。启动慢读后的 head/来源重查属于本层
实现，独立概念 Gate 和正式启动验收仍需分别记录，不能由这些安装夹具代替。

## 回复入口统一执行前提（#925 追加修复）

`reply.execute.v4` 要求每次调用显式交入一个固定的 `check_admission`。普通来源在入口
用原接纳边界构造检查；不能在异步准备之后重读 head 作为新的授权。Scheduler、Subagent
和 Wake 仍各自固定原 Input 或阶段 Input，回复程序不再读取 Task 的边界来猜测来源权限。

后台结果回传的 Reply 入口一次性组合原 conversation 控制前提和输出来源的固定前提。
控制来源与输出来源仍是独立事实，但通用回复程序只接收同一个检查，不分普通/回传两条分支。
`reply_program` 不再依赖 `source.check` provider；来源入口负责构造检查，
Tool/Command/React 仍在自己的启动事务内核对，Core 只拥有原事务和收据。

本轮不增加持久状态、来源队列或执行框架。原输入接纳、停止、重试、跨来源输出提交与取消
排空合同不变。旧 `reply.execute.v3` 常量保持原合同，新 provider 只发布 v4；
外置调用者须在完整新组合里显式传入执行前提。

### 2026-10-01 实际回传适配器验收

`docker/debug/completion_source_ordering.py` 现挂载真实 Sources、Conversation、Reply、ReplyProgram provider，经 Subagents 的回传入口消费 `reply.execute.v4`，不在场景里复写 report 或直接绕过适配器调用 run_reply。16 个受控场景分别令控制来源或输出来源的 Input/Control 先落盘，并延迟 loop 通知，核对入口、generation claim、Tool started、真实本地效果之后与最终 Output 的行为。调用账、原 Message、SQLite 完整性和清理一同核对；模型 driver、材料与投递仍是本地受控边界，不代表正式 provider 或真实送达验收。

命令没有新增第二份 started 状态：原不可变 `CommandIntent` 就是同事务首次准入事实；不能把 durable claim 到 handler 的物理时间间隔解释为缺少另一份准入状态。最终 Output 仍由同来源 head CAS 拒绝过期提交，未知外部效果由固定命令 owner 恢复。

## ReAct Output 的事务（#879）

ReAct 在原插件 scope 内准备 Output 的内容引用、metadata 和 Session 投影，
通过 `MessageWriter.prepare_async` 固定一次追加。`OwnerTransaction.append_prepared`
只提交该 writer 的固定身份，不授予额外消息类型或跨库权限。来源检查、竞争消息检查、
身份重放、附件引用和追加仍在同一 SQL 事务；事务整体进入有界文件 worker。
无 OwnerStore 的 ReAct 路径使用相同准备边界和 `append_async` 的来源 head CAS。

writer 撤权只持短 grant 锁，不等待 SQL 或 commit。事务取得 SQL 写权后核对 grant：
先撤权则拒绝新追加；先通过 grant 的在途追加可以完成，取消等待实际事务结束，
不能把取消当成已提交 Output 的回滚。新 Input/Control 与 Output 仍由同库事务排序，
Source 前提检查防止旧回复越过已接纳的新边界。正常消息只追加，无 schema 或历史迁移。

`docker/debug/react_output_io.py` 通过真实 Source、ReAct、Models 与 MessageLog，
分别阻塞独立 SQLite 写锁及 INSERT 后的提交，检查 Timer、只读请求、重复取消、
撤权、实际排空、完整旧消息和 SQLite 完整性。模型驱动只返回本地固定响应；
不是付费模型、正式 workspace 或生产延迟验收。ToolResult、Delivery、Wake 来源
与 EventMail 写入仍须分别迁移，不由 Output 这一层自动覆盖。

## Tools 的结果与接纳（#879）

Tools 在同 key 的异步接纳中固定 requested 回执，再启动既有 Task；取消接纳
可以留下没有外部效果的 requested，恢复仍使用原 key、binding、参数和结果身份。
prepared 与 ToolResult/done 通过原 OwnerStore 的有界事务提交，内容校验仍在调用者
scope。放弃结算同样持有该 key 的异步接纳，并在真实提交回到 loop 后取消原 Task。

放弃与真实结果竞争时，done 的首次提交胜出；迟到结果只读已提交的正文指针。
requested/prepared 的等待期间发生放弃，不得用旧版本覆盖终态。错误回执可能先唤醒
订阅者，等待路径仍取回原 Task 的错误，不能把真实执行异常伪装成正常返回。
普通 Source 停止仍排空已有工具；已通过 started 前提的工具按原协议完成，不改变
并行工具按模型顺序提交结果的规则。无 schema、旧消息或正式 workspace 改写。

`docker/debug/tool_result_io.py` 使用真实 Source/ReAct/Tools、SQLite 和本地文件效果，
覆盖四个持久阶段的正常/停止八场景，以及放弃先提交而成功结果迟到的竞争。
逐项核对消息与 done 指针、外部文件效果、旧消息及数据库完整性；不使用付费模型。

## Delivery 的异步准备能力（#879）

`delivery.v2` 为正文与目的地发布、已有消息选路、按序消费和新增目的地提供显式
异步入口。消息内容仍在原 scope 校验；固定正文、selection、prepared 和 cursor
由原 owner 在同一 SQL 事务提交，取消后排空，不拆成可能留下半笔状态的多个提交。
消息推送、调度、更新通知、子任务、Wake 和自动投递消费 v2；缺少 v2 不能回退到
同步写入。当前只发布 `delivery.guarded-start.v1`，保留原 recovery owner；
旧 v1/v2 入口及同步方法退役，所有消费者使用同一异步合同。

`docker/debug/delivery_prepare_io.py` 挂载真实 Delivery provider，覆盖四类准备的正常与
重复取消八场景，核对原消息、目的地、cursor、数据库完整性及原 loop 上的内容校验。
发送开始、终态回执和 Alert 的版本顺序仍由后续独立层处理。

## Delivery 的开始与终态（#879）

无同步 `before_start` 的发送使用原 Task key 的异步接纳，started 与显式 prepared
撤回串行；终态回执、确认时间索引和显式重试也通过原 owner 的有界事务完成。
取消在 started 提交后仍先取回确切版本，再记录真实失败；送达回执提交中取消则
保留已完成的确认事实。物理工作结束前不释放 Task、sender scope 或领域保护。

`delivery.guarded-start.v1` 明确提供 `start_guard`：领域 owner 的 async context
覆盖前提读取与首次 started 的完整提交，返回拒绝原因则保存 rejected。
started 提交后释放保护，再进入网络发送；恢复旧 started 不重新判定为从未发生。
该能力不执行 Context 回调的线程迁移。旧 `before_start` 仍保留原同步语义，Wake
会在 EventMail 写入迁移时改用新能力，两种检查不得混用。

`docker/debug/delivery_receipt_io.py` 使用实际 MessageLog、Delivery、EventMailStore
和本地 sender 文件效果，覆盖慢 started/终态、重复取消、prepared 撤回、重试、
版本先更新则拒绝，以及 started 先提交后更新无需等待网络结束的十二场景。
既有投递集成的观测点改为真实提交完成，不把 sender 返回或事务内 INSERT 当成回执。

### EventMail 来源与 Wake v2

EventMail 只发布 `eventmail.content_source.v2`、`eventmail.alert_source.v2`、
`eventmail.context_source.v2`、`eventmail.wake.v2` 和 `eventmail.delivery.v2`。
来源提交、查询和结算均须 await；Feed、Fitbit、Calendar、Steam 与 Wake 必须按这组能力一起发布。
同步 v1 来源不能与异步告警开始保护混用，否则换版可能越过首次发送的持久开始。
同一 EventMail owner 的写锁覆盖告警版本检查到 `delivery.guarded-start.v1` 的首次 started 提交；
随后释放锁再进入网络发送。取消排空已开始的 SQL，已提交来源仍在原 loop 发出 changed。
原 envelope、选择、结算与 ACK 的身份和保留规则不变，没有新增持久表或删除路径。

SessionAdmission 的异步准入在原 scope 校验维度，然后排空完整 create-once 事务。
Conversation、Programmatic、Scheduler 与 Wake 使用它；固定 Session 属性仍不可改写。
Wake 的请求与 flow 指针继续同事务追加，阶段 Input、quiet Output、失败 Control 和结算指针均等待实际提交。
取消若发生在 quiet 已提交但指针未推进时，重读原消息即可补齐结算；不重写旧正文。

Subagent 的容量检查、Session 准入与 Input/恢复指针提交共用同一 Tasks 准入段；
并发第四个请求明确拒绝，不留下空 Session。pause 提交后的回调先撤销原 Task，
再排空其效果；取消等待者也不跳过这一步。结算与诊断文件写入使用有界 worker。


## 其他运行期 Owner 与来源写入

摘要 prepare/reduce 的 head/父链读取、发布时的原前缀检查和 summary/head 事务，
Computer started/ended/failed 回执及收尾完整投影，插件更新的持久通知意图，均由
原 async 调用者等待完整 worker 操作。Context 能力先在原 scope 取得；真实网络或
安装效果仍在意图提交之后开始。摘要只增加记录并推进 head，不改写原消息；Computer
与安装意图沿用原版本、身份和恢复入口，不增加删除权限。

Drift 来源使用 `drift.proposals.v2` 时先复制请求内容，再在线程中提交原 proposal。
changed 事件仍在原 loop 发出，已提交后取消不丢通知，同身份重放不重复通知。
同步 v1 契约与 provider 已退役。[fleet 当前外置集合](hua-home-plugin-runtime-source-of-truth.md#1-维护范围只认-fleet运行事实只认-hua-home)
中没有 Drift proposal 消费者；非 fleet 的历史仓库不构成保留兼容接口或新增迁移的理由。

`owner_write_io.py` 的真实 SQLite/本地 socket 场景覆盖摘要 CAS、Computer 回执、
取消排空和原消息完整性；同一提交屏障下旧基线约 1 秒，改后小于 1 毫秒。
`drift_proposal_io.py` 验证真实 Root 的提案提交、请求快照、取消、重放与 loop 通知。
这些是隔离组件证据，不是生产 P99。外置 Observe、status_commands、proactive_feedback、
github-watch 与四个 EventMail 来源的源码修复已经交付；正式安装链、选择新 Root 和
生产完整进程恢复仍未验，#879 在该层验收前保持开放。Akasha 已在线程中的 Recall/Scope
操作及 settings/auth 等未证明的次级线索不扩入本次修改。

### Source startup reads only reply facts

Source pending checks use the last Input sequence, the last finished Output sequence after that input, and ordered Control bodies in the same frozen source prefix. `MessageReader.latest_input_seq` and `latest_finished_output_seq` return positions without loading content or metadata. `scan_controls` pages Control bodies in sequence order inside one read snapshot; its consumer is synchronous and must not keep the iterator. Ordinary message/context readers still return complete messages.

The source body-kind/finish index serves these narrow reads. The Sources owner retains pause/resume/abandon rules and the full source head for CAS and task boundaries. Earlier controls before the latest input remain outside that decision, matching the existing bounded-tail algorithm. A terminal Output follows every earlier Control through_seq because a Control cannot refer to a future prefix; therefore selecting the last terminal Output and applying ordered controls yields the same boundary.

Reply startup shares one Session iterator across four TaskGroup workers, matching the existing file I/O capacity. Each worker keeps one Session's sources in order, checks the source head before and after the awaited predicate, and skips a registration that has been removed. A changed head retries before publishing pending state. All workers finish before runtime readiness; failure or cancellation cancels siblings and drains their actual reads before the parent returns.

```text
Session iterator → four owned workers → fixed source read → head CAS → pending hold
                         └──────── all joined before ready ───────────────┘
```

`reply.prepare.timing` records outer wall time, Session/check/retry counts, and head-read, awaited-predicate and changed-callback time. Awaited phase totals overlap across workers and must not be added to outer wall time. Records contain no message content or Session IDs. Parallelism changes scheduling of independent Sessions; source admission, predicate rules and per-source CAS remain unchanged.

## Models 调用账本连接

ModelsStore 持有一条串行写连接，初始化时使用 WAL，关闭时先等待当前事务结束，
再关闭连接和释放宿主锁。读范围仍使用独立只读连接，不缓存查询结果。
调用方显式提交；退出写范围时回滚剩余事务。FULL 同步、请求准入、首段记账、
响应结算、配置 CAS 与写前备份保持原顺序。连接复用不增加消息或调用记录的删除路径。

Models 在一次 `complete` 内计算一次固定请求摘要，活调用合并与持久准入共用该值。
同 key 的恢复检查与退避判断共用一次账本读取；只有结算孤儿改变了记录时才重读。
写事务仍重新核对真实状态、binding、预算和允许时间，因此并发准入不依赖旧快照。


## 固定模型输入的派生计算

原生工具菜单首次读取 schema 时深冻结固定 binding 的描述，后续请求复用相同值；
自定义 presentation 仍按原接口读取，不替插件缓存动态 schema。OpenAI-compatible driver
只复用同一个深冻结消息对象的字符和图片成本，普通可变输入先取得独立冻结值。
历史缩短后只保留本次输入的计数，driver scope 结束后释放。字符先合计再除以三，
图片成本、空输入、工具 schema 成本与原容量公式一致；不改变压缩水位或请求正文。

MessageLog 的单条查询使用 SQLite 隐式读取快照；查询在归还连接前取完全部结果。
分页、组合读取、显式 `read_snapshot` 和嵌套 writer 读取仍使用原事务，不能跨查询混读。
只读连接创建时固定 row factory 和 query-only 配置，借用时不重复设置。

文件 I/O 入口直接等待线程 Future，并在提交时复制调用者的 ContextVar；不再创建转发用的 asyncio Task。
四个磁盘名额、排队取消、已启动工作排空以及取消与物理失败的联合传播保持不变。

消息重放先查询实际身份。没有旧行时，普通新正文只在 INSERT 时编码；
已有身份仍比较原编码和 metadata，旧 unknown 工具结果仍禁止作为新消息写入。

绑定模型复用当前深冻结消息和工具 schema 的 JSON 编码来计算请求摘要。
摘要仍使用原字段、排序、UTF-8 和分隔符；同 key 的内容冲突与恢复判断不变。
每次只保留当前请求引用的编码，历史缩短、模型 scope 结束后释放旧引用。
