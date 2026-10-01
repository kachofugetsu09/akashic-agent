# Event-loop execution boundaries

Related issues: #827 (stack), #828 (storage), #829 (interest), #836 (materials).

Long history reads use a private read-only SQLite transaction and connection. Nested readers of the same MessageLog in that synchronous call share its read snapshot; they do not acquire the writer's connection or decoded-message cache. Reads inside an existing write transaction still see that transaction's uncommitted rows. Owner transaction callbacks remain synchronous and atomic, including when pure SQL runs in a drained worker. Listeners wake only after commit.

普通 `MessageReader.snapshot_async` 与 `OwnerStore.snapshot` 的只读准入不等待另一个线程
持有的 writer 锁。当前线程能重入的写事务仍使用原连接，读取自己的未提交行；其他读取
直接打开独立只读事务。准入与 close 使用短锁：close 拒绝后来读者，已取得的只读连接
仍由原同步读取关闭。来源提交和首次效果的异步顺序见下面的 #879 说明；权威消息仍只追加。

```text
┌───────────────────────┐       ┌────────────────────────┐
│ writer：未提交事务       │       │ reader：独立只读快照     │
│ 继续等待或提交           │       │ 只看已提交的固定前缀      │
└───────────────────────┘       └────────────────────────┘
```

短 Reader/OwnerStore/Catalog/binding 读取使用同一 private RO 准入，自己的写事务内读取仍重入原连接。
listener 登记和释放只持有其现有注册表的短锁，不等待 writer 的磁盘工作。
增量 reader 的同步 snapshot、剩余 Core 写入和关闭仍有各自的同步路径；不能据此声称全部 I/O 已异步化。

MessageLog uses file-backed WAL mode so a pinned read does not delay a writer's commit. This changes the runtime journal mode, not the schema. Backups must use SQLite backup or include the SQLite sidecars; copying only the live main database file is not a snapshot. Existing databases keep their schema and data; an unsupported journal mode is rejected explicitly. Short synchronous writes can still wait on SQLite file-level contention; this change does not claim that all storage I/O is asynchronous.

回复准备在等待历史前固定来源 head 和完整消息 head。增量异步快照的解码、原连接 data_version 核对和缓存大小 SQL 都在同一个文件 worker 完成，取消等待 worker 实际退出。writer 忙时跳过可选缓存预热，private RO 仍可读取已提交前缀；只有原连接版本未变且 writer 空闲时才发布原有有界缓存。外部编辑使缓存不能复用，不重跑或推翻本次固定快照。后续追加仍按尾部补读，外部编辑仍在下次同步读取时使旧缓存失效。已准入的 private reader 可跨 close 完成，关闭后跳过缓存发布。

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

SourceSession 在同 key 的异步准入内接纳 resume；loop 上的只读检查仍只查询固定前缀的最后同来源
Input、Control，并扫描该 Input 后的同来源消息。已提交 resume 的同 ID 重放，只
核对原 through_seq 内最后 Input，不解码更早的关闭正文。最新输入、已有控制、
完成/abandon 拒绝与条件追加规则保持；没有把存储事务跨 await，也不删改历史。

`docker/debug/resume_history.py` 在一次性 SQLite 上量测大闭合历史的首次重试与
同 ID 重放，并核对关闭、活动任务和来源边界。它也量测已异步化的子任务 outcome，
把解码时间与事件循环回调延迟分开；本机样本不代表生产 p99。
当前未闭合 Input 后仍可能有大正文；其他同步历史消费者继续独立追踪。

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
│ loop：同 ID 收据重放，或当前 owner 内容校验 │
│ 固定内容引用、授权 metadata 和纯投影      │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ worker：Core SQL 重新核对身份、grant、head │
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

生成请求继续使用既有准备记录和稳定 request key。`started_attempts` 只记录该准备已通过首次启动检查，
不复制 Models 的调用事实。Models 仍拥有独立数据库和发送状态，按同 key 前向恢复；
Core claim 不能证明远端有或没有效果。旧 v2/v3 准备和消息表示保留，不做 schema 迁移或历史改写。
准备的纯 SQL 也离开 loop；Context 和模型句柄的读取保持在原 scope。

能力必须按完整组选择：`sources.v4`、`source.session.v3`、`source.check.v2`、
`source.changed.v2`、`channel.input.v2`、`source.interrupt.v2`、`tools.program.v2`、
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
