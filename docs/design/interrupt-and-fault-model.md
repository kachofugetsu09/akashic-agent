# 执行进展、中断与故障恢复设计

- 状态：已评审设计稿，独立 Devin 两轮复核结论"可接受"；用户已授权按本文实施，实现以 stacked draft PR 交付（main ← 本设计 PR ← 实现 PR），不修改现行需求、不做正式数据迁移。
- 日期：2026-09-20。
- 源码基线：`69a67e7f3d464d1bd749d738d2d31402229fde60`，与补充红测 worktree 的 HEAD 相同。
- 前案：本文即 [PR #628](https://github.com/kachofugetsu09/akashic-agent/pull/628) 的设计正文（旧 head `c4274cf383b663c019ec095119889f909c445f9e` 的概念稿已由本文取代）；实现见 stacked draft PR（base 为 `feature/interrupt-fault-model`）。
- 上游：[projectneed](../projectneed.md) 的 STA、CAP、ERR、SES、RUN-007～009、OUT、PLG；[0063](../decisions/0063-execution-failures-have-terminal-results.md)、[0070](../decisions/0070-plugins-own-persisted-data.md)、[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)、[持久化状态地图](persistence-state-map.md)。

## 1. 用户目标与本次范围

用户希望参考 Codex 和本地 pi-mono，用正交设计提高整体鲁棒性：已经接纳的工作不因丢通知消失；单项故障不破坏无关会话；取消和已提交结果不被清理卡住；恢复不盲目重复模型或工具调用。

本文是设计合同；实现与测试在独立的 stacked draft PR 中交付，正式 Akashic workspace、长期需求与数据迁移仍不在授权范围。文中“应当”为目标合同；实现 PR 的测试、Gate 与独立概念评审证据以该 PR 说明为准，本文不声称已实现或已迁移。

设计保留 `Message → Turn → Session` 与 `Loop: 输入 → react → 输出`。不增加全局持久 Run/Attempt/Lane 状态机。“来源”指既有 `(session_id, source)`，不是新增领域实体。

推荐方案把四个独立问题交还各自 owner：

| 问题 | 唯一依据与 owner |
|---|---|
| 工作是否结束 | Message、Control 与各 owner 的终态回执；Turn 只投影 |
| 谁还能开始新效果、提交 Output | 来源持有的短命提交许可 |
| 一次外部调用产生什么结果 | Models、Tools、Delivery 各自的回执 |
| 任务及资源是否真正退出 | Task、Scope、资源 provider 与宿主 |

## 2. 已核对事实与证据边界

### 2.1 当前真实链路

```text
┌──────────────────────────────┐
│ SourceSession 接纳 Input/Control│
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ MessageLog 提交，再通知 follow │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ reply.follow → SourceSession.start│
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ react：结算工具 → 模型 → Output│
└───────────┬───────────┬──────┘
            ▼           ▼
      Models 回执    Tools 回执

Task / Scope 持有上述活动所需资源、租约与真实关闭责任
```

| 已确认事实 | 基线代码与符号 | 影响 |
|---|---|---|
| 提交后通知异常继续上抛；一个 listener 可阻断后续通知 | `session/log.py: MessageLog._write` | 已提交 Input 被报告失败，且失去唤醒 |
| follow 空闲后只等进程内 event | `MessageCatalog.follow`、`MessageReader.follow` | 最后一次通知丢失时没有后继 |
| start 异常逃出 per-source drive；join 异常后无条件再驱动 | `plugins/reply/follow.py: follow/drive` | 故障扩大，或在失败 Control 未提交时反复付费 |
| SourceSession.start 等旧 Task.join，slot.current 直到 task.done 才消失 | `plugins/sources/session.py`、`agent/plugin_composition/tasks.py` | 业务完成和清理共用存活性 |
| ToolResult 提前返回仅限 abandoned | `plugins/tools/execution.py: _wait_result` | 普通成功结果仍等待物理清理 |
| 模型账保存摘要、usage、错误，没有完整 response | `plugins/models/state.py: _BoundChat.complete`、`plugins/models/store.py` | 收到响应后到 Output 前崩溃会再次请求 |
| Output 使用整个 source head CAS | `plugins/react/plugin.py: react` | 无关旧段结算可令新提交失败 |
| abandon 的工具协议和结果被投影跳过；可读正文仍保留 | `plugins/models/projection.py: MessageProjection.render` | 模型缺少中断及效果因果；PR 的“整个 Output 消失”描述需修正 |
| abandon consumer 只捕获一个 legacy identity 异常 | `plugins/tools/abandon.py: follow_abandon` | 单个 item 故障终止全部消费者 |
| driver 已拥有模型重试循环 | `plugins/openai_compatible/driver.py: _stream_chat` | 外层再次重试会放大实际请求数 |

PR 中主要执行 owner 的旧路径已变化，当前以 `plugins/sources/session.py` 为准。

补充红测文件在 `/mnt/data/coding/akasic-agent-red-liveness-tests-20260920/tests/test_reply_liveness_red.py`，SHA-256 为 `19d7c125f0e14c4f87894e4ef9cc8795c31fa51f9d3799c81876c7c4173d2408`。本会话前序调查已运行这 10 条测试，全部失败；包括实际重复模型调用 `2 != 1`、缺少 response 字段、TaskGroup 传播和等待超时。测试不是实现验收，进程内 Crash 模拟也不替代真实进程强杀。

### 2.2 参考实现与取舍

Codex 基线 `/mnt/data/source-code/codex@553df1c691fe8bf7747e50da22f1342984495ae0`：

- `codex-rs/core/src/tasks/mod.rs: handle_task_abort`：取消 token、有限宽限期、abort、中断 marker、终态通知，说明执行身份和取消顺序必须明确。
- `codex-rs/core/src/session/turn.rs` 与 `responses_retry.rs`：请求局部重试、错误分类和延迟。该版本还存在可选无界连接重试，不把“所有 Codex 重试都有界”当事实。
- 不照搬 Rust abort；其 flush 失败有告警后继续路径，也不是 Akashic 的持久成功保证。

pi 基线 `/mnt/data/source-code/pi-mono@4a6ed01945c7f6a2a350996fb439148149ab65ee`，指该 checkout 的 durable harness，不概括上游所有 Pi 版本：

- `packages/agent/src/harness/runtime/drive.ts`：每次推进必须改变状态或返回明确等待/结果。
- `runtime/drive/generation.ts`、`recovery.ts`：效果前保存 intent；孤儿请求从已保存材料结算，不再调用 provider。
- `execution/effect-gate.ts`：效果准入与取消通知分开。
- `runtime/harness.ts: fault` 会封闭整个 harness；恢复占位使用 ZERO_USAGE。Akashic 不照搬这两点，故障按实际 owner 隔离，未知 usage 保持未知。

当前参考只证明这些机制存在，不证明它们具备 Akashic 所需全部持久、插件和多会话合同。

## 3. 目标结构与最短路径

```text
┌────────────────────────────────┐
│ Message / owner records：持久事实│
└───────────────┬────────────────┘
                │ 通知 + 可恢复核对
                ▼
┌────────────────────────────────┐
│ 来源：选择工作，授予/撤销提交许可 │
└───────────────┬────────────────┘
                ▼
┌────────────────────────────────┐
│ ReAct：固定生成准备，提交 Output │
└───────────┬─────────────┬───────┘
            ▼             ▼
┌─────────────────┐ ┌─────────────────┐
│ Models 请求/响应 │ │ Tools 调用/结果  │
└─────────────────┘ └─────────────────┘

┌────────────────────────────────┐
│ Task / Scope：运行资源及真实清理 │
│ 业务停止等待，不解除其清理责任   │
└────────────────────────────────┘
```

正常：Input 提交 → 来源许可 → 生成准备 → 模型回执 → Output 提交 → 释放业务等待；清理独立完成。

取消：Control 提交 → 撤销旧程序提交许可 → 各效果 owner 结算原调用 → 后续来源按控制语义继续；旧 Task 保留资源直到真实退出。

恢复：重新发现持久待办 → 检查现有 Message/回执 → 恢复缺失的本地阶段，或结算无法恢复的旧调用；不会从 react 顶部盲目重新付费。

整体换代：停止接纳 → 等待所有真实租约 → 关闭旧组合 → 启动新组合 → 提交 stable。新设计不改变 0071；业务终态不能冒充租约排空。

## 4. 提交、通知与持久待办

### 4.1 提交结果只由事务决定

MessageLog 在提交成功后返回原提交结果。通知每个 listener 时单独处理通知通道已关闭等可识别错误，报告 incident 并继续其他 listener；不把已提交事实改报为未提交。未知内部通知故障也必须报告，但不能触发原业务重放。

确认 event loop 已关闭的 listener 在同一 owner 锁内移除，注销幂等；不能移除未经确认的活订阅。无法确认死亡时保留订阅，按 listener 合并并限频报告 incident，避免每次提交产生同一告警。日志关闭、订阅退出、通知之间以同一 owner 的锁保护。

来源同步接纳中的执行撤权不是可选 observer：Input/Control 提交后必须执行对应撤权，observer 失败不能跳过它。写入与效果开始仍核对持久来源边界，不能只靠取消通知及时到达。若撤权本身失败，停止对应许可的后续使用并报告控制故障；原 Message 身份依然可查。

### 4.2 follow 以持久状态为准

订阅先登记、再取快照；事件只降低延迟。空闲时以可配置但有界的核对周期重新读取 head；重启和重建订阅均执行首次核对。不新增持久通知队列，因为原日志已保存待办依据。

扫描只发现工作，不授予外部重试。相同来源仍由原子准入合并并发扫描；失败项不能因整体 cursor 前移而丢失。Tools abandon、Delivery 等消费者各自保存未完成项或能重新查询的条件，成功项通过原回执幂等复用。

Tools abandon 选择现有 session cursor 保守保留、按回执幂等重扫，不新建逐 item 待办表。失败 session 即使 head 未变也进入有界退避核对；分页扫描保留本轮位置，坏 item 不阻止同 session 后续 item，整轮结束后仍保留失败范围。来源故障修复与容量等待也须进入独立于 head 变化的核对集合，但只执行各自允许的动作（§5.3、§7.2）。

对大型历史，首次扫描分页；后续读 head 与 owner 未完成索引，不周期性加载全部正文。扫描周期与额度在实现阶段通过隔离负载确定，不把红测的 0.2/1 秒当生产 SLA。

## 5. 提交许可与物理任务分离

### 5.1 两个 owner，不改变 join 的含义

来源只登记当前获准决策的 handle；Task/Scope 登记全部尚未退出的任务。释放当前 handle 用身份比较，旧任务完成 callback 不得删除新 handle。

“提交许可”只是现有 `slot.current`、`check_source` 与 `writer.expire` 的行为合同，不新增许可表或第三个 owner。改变的是当前 handle 的释放条件；全部活 Task 的登记仍归 Tasks。

`Task.join()` 仍等待真实退出。Core 不增加 `turn_finished` 或识别 Output 的逻辑；来源根据日志决定许可释放，Tools 根据回执提供结果等待。可复用的 Core 改动只处理通用 key 准入、撤权及保留未退出任务。

许可撤销同步禁止旧 Output writer、新模型请求和新工具启动。已经开始的效果保留独立的原调用结算权限；它只能更新自己的回执和允许的 ToolResult，不能再发起新业务。这份权限不因父来源撤权而丢失。

来源许可检查与效果 owner 的 started 写入在同一同步准入段内排序；许可撤销不能插入检查和 started 之间。真实 provider I/O 紧随其后，started 已提交但尚未调用就崩溃仍保守视为不确定窗口。

### 5.2 不同控制的可观察合同

| 事件 | 业务等待与后继 | 旧任务与效果 |
|---|---|---|
| 最终 Output complete/quiet 已提交 | 关闭本次等待，可接纳后继 | 无新决策权，继续清理 |
| 普通工具结果已提交 | 结果等待者读取回执继续 | 工具 Scope 可仍在关闭 |
| abandon 已提交 | 关闭旧前缀，后续工作不等 provider 退出 | 未 started 为 denied，started 未结算为 interrupted；不覆盖已有结果 |
| pause 或普通 Input 抢占 | 保留现行“已开始工作先结算”的语义 | 不隐式转换为 abandon |
| 结果提交失败 | 不声称业务终态已保存 | 保留回执/intent，进入故障恢复 |

abandon 后，新模型请求之前先确认模型上下文依赖的旧调用已经由 Tools 保存终态。该结算操作不等待 provider 退出；结算存储或 binding 故障必须可见，不能靠虚构 ToolResult 强行继续。没有工具调用的旧模型请求可直接释放来源许可。

明确执行者是 ReAct：构造请求前直接调用 Tools 的窄结算入口，覆盖 abandoned 前缀，而非只调用当前会跳过该前缀的 `_pending_calls`。该入口复用 `abandon_call` 的 denied/interrupted 规则与 `finish` 的事务幂等，不启动工具、不打开外部 provider、不等待旧 Task。`follow_abandon` 是后备清扫；两个入口竞争同一调用回执，先保存的终态获胜。结算可等待本地存储操作，但不得跨 await 持有来源准入锁；返回后重新检查来源依据，再固定请求。未能结算就明确阻断该请求，不依赖 watcher 先运行。

业务终态提交与撤权之间的进程内窗口必须由同一来源准入排序；重启只按持久终态判断。进程不崩溃但通知丢失时，核对路径也要释放已关闭来源的业务等待，不能继续等旧 join。

### 5.3 清理超时与背压

Scope 保留尚未退出任务、依赖、连接、lease 和 external permit。业务完成不触发这些句柄的伪释放。Tools 的结果等待 API 同时覆盖消息调用和独立 programmatic 调用，避免只修 Message 路径。

宿主对残留任务数量和资源设有界准入额度。额度满时拒绝尚未接纳的新工作；已提交 Input 保留为待处理并显示原因，不删除、不重复接纳。清理完成后重新核对待办。

Tasks 拥有活动与残留 Task 的计数，在 `slot.start` 的同步准入段原子核对额度。来源接纳前检查只能是提示，不能替代启动处检查。容量不足返回明确等待原因，不走无进展故障或 failure Control；Task 退出事件及周期核对唤醒容量等待，即使 source head 未变也能推进。

超时表示“停止继续等待并报告”，不表示线程或外部效果已被杀死。进程 controller 可以按既有授权终止其子进程；同进程不合作任务保留 owner 并阻止冲突的整体换代。阻塞事件循环的同步插件只能由进程外宿主处理。本提案不启用 crash auto-restart。

## 6. 模型生成的持久接续

### 6.1 必须补充的两类事实

ReAct 的准备记录使用既有 plugin owner records。它在付费请求前固定：生成 ID、目标 Output ID、来源依据、模型 binding 引用、Models 请求 key、模型请求材料及解码需要的不可变引用/值。它不保存独立的 Turn 状态或复制最终回复。

Models 以调用者命名空间与请求 key 唯一标识一次请求；`model_calls` 的每个实际 attempt 保存请求摘要、binding、宿主身份、attempt 序号、started、完整规范化响应或结构化错误、usage 和已知时间。完整响应含工具调用、continuation 和恢复所需协议事实。Models 不解释 Session、Turn 或 Output。

ReAct 保存调用者所需的冻结请求材料，Models 保存请求摘要与实际响应；不各存一套可独立更新的 prompt。Models 在首次调用时验证请求摘要，后续以原 key 读取回执；同 key 参数漂移明确拒绝。

准备记录按当前来源输入/控制边界与已提交输出前驱唯一选取。重复扫描先查原准备，不因重新获取时间、reminder 或新随机 ID 创建第二次生成。来源已经被替代时，旧准备只保留审计，不重新提交。相同准备的 retry 使用同一请求 key 与递增 attempt；显式 resume 只有在原生成已失败且仍获来源授权时才创建新的准备身份。

ReAct 对“哪项生成属于当前工作”负责，Models 对“这个请求实际尝试了几次”负责。这两份记录不互相复制状态机。

### 6.2 提交与恢复顺序

```text
┌───────────────────────┐
│ ReAct 保存生成准备与固定 ID│
└────────────┬──────────┘
             ▼
┌───────────────────────┐
│ Models 保存 attempt started│
└────────────┬──────────┘
             ▼
         provider I/O
             ▼
┌───────────────────────┐
│ Models 原子保存响应/usage │
└────────────┬──────────┘
             ▼
┌───────────────────────┐
│ ReAct 解码，提交固定 Output│
└───────────────────────┘
```

跨库没有假定的事务。固定 key、响应回执和 Message 幂等共同闭合窗口；已提交 Output 是唯一发布事实，不再维护第二份 published 布尔位。

| 崩溃位置/恢复事实 | 处理 |
|---|---|
| 准备保存前 | 没有该生成的外部调用，可重新准备 |
| 准备已保存，无 model attempt | 当前来源授权仍有效时首次调用 |
| started，原 owner 仍活跃 | 仅恢复同一生成准备时等待该 attempt；abandon 后的新准备不等无关旧调用，不当孤儿、不重发 |
| started，确认原 owner 死亡 | 默认结算 error，保留中断原因和未知 usage；仅 provider 确实提供查询合同才查询原 key，不假设兼容驱动具有此能力 |
| response 已保存，无 Output | 用原响应和冻结材料继续解码与提交；不请求模型 |
| Output 已存在 | 复用原 Message；不得再次触发工具 |
| 原来源已被替代/abandon | 保留响应与账目，拒绝旧 Output；不把旧成功升级成当前成功 |

规范化响应落盘失败时不把响应交给 ReAct，也不重新请求 provider。当前进程可在保留原响应的前提下重试该次写入；进程死亡后按 started 孤儿处理。成功响应之后的 content decode 失败只恢复解码阶段，不自动重新生成。

保存响应的确认丢失时，先按 attempt key 读回：已保存相同响应及 usage 则复用；仍 started 才重试同一次保存；已有其他终态不覆盖，相异成功回执视为契约违反。不得把重复 UPDATE 的零行数直接当作首次写入失败。

恢复调用 content/tool decode 时只允许无外部效果的转换；具有外部效果的扩展不能挂在可重放 decode 中。冻结材料包括 reminder、引用解析输入、摘要引用和工具解释所需 binding。当前插件不理解旧材料时明确失败，遵守 0070，不以旧 generation hash 自动加载或拒绝恢复。

本设计不承诺 provider exactly-once。provider 已处理但本地没有响应的窗口只能查询、依据真实幂等合同恢复，或明确失败。响应“已收到但尚未耐久保存”不等于“已可恢复”。

### 6.3 孤儿与重试

孤儿判断由真实运行 owner/宿主身份提供证据：独占冷启动可以证明旧宿主失效；插件 apply、记录年龄、heartbeat 超时或新 generation 本身不能证明死亡。candidate 的调用账在其独立环境，由该环境 owner 处理，不扫正式账。

持久 owner 身份固定为进程启动实例 ID、Root 启动实例 ID、attempt ID；Root 实例不是可复用的代码 generation/hash。Models 在原子请求准入下先登记活 attempt，再保存 started；真实 Task 退出时才注销。等待必须查到该确切活 attempt，不能仅因进程存活而永久等待。旧 Root 已真实关闭或同 Root 的 Task 已确认退出，才证明其执行 owner 不再存在；这不证明远端效果未发生。

Models 退出时尝试将未结算调用保存为 error（原因 interrupted，usage 未知）；写不下则保留 started 和 incident，由恢复 owner 在上述死亡证据成立后结算。旧 Task 不合作时仍阻止 0071 换代，不能先启动新 Root 再声称旧 owner 死亡。候选隔离是实施前置验收：核对实际 `workspace_file("model-registry.sqlite3")` 路径、Root 实例和写权限；不能只凭 candidate 名称推断。不能证明隔离的候选不得扫描或结算正式账。

模型请求的自动尝试预算由 Models 独占。移除 driver 的内层重试循环，禁用 SDK 自动重试；driver 每次调用至多发起一次 provider 尝试，负责协议错误分类和 Retry-After。不能关闭隐式重试的适配器不满足该合同，不以合并记账放行。现有 `max_retries` 的配置含义迁移到 Models，不能两个 owner 同时生效。first-token、usage、错误和退避均关联具体 attempt。ReAct 只拥有上下文缩减和后续推理步骤，不再包第二层 transport retry。

每个可重试失败先耐久保存错误及下一次允许时间，再尝试下一次请求。attempt 序号和限制从回执核对，重启不能重置。退避可取消，每次外部开始重新检查来源许可与实际 binding。空响应等已完成但不可用的响应可按显式策略消耗下一次尝试，不伪造 quiet。

结构化错误分别记录低基数原因、可重试建议和已知诊断；可重试建议不授予无限预算。存储失败、回调异常、解码缺陷和提交冲突不能被分类成 provider 网络错误。工具/发送的恢复权限仍归各自 provider 的原 key 合同。

## 7. 故障隔离与进展

### 7.1 故障范围由受损事实决定

| 失败 | 默认处理范围 | 恢复动作 |
|---|---|---|
| 单个不可解释 binding/receipt | 对应调用及依赖它的来源 | 当前插件修复/显式处理，不能重跑未知效果 |
| 来源驱动违反进展不变量 | 对应来源 | incident + 停止自动执行；修复后重新读取原记录 |
| 共享存储不可写或 schema 损坏 | 依赖该存储的所有效果入口 | 关闭新效果准入，修复实际存储操作 |
| follow 通道暂时失效 | 该消费者订阅 | 有界退避后重新订阅并扫持久待办 |
| watcher 内部程序缺陷 | 对应服务 | 报告能力不可用，停止无界自启；修复或现行显式重启 |

来源故障不是业务成功。存储可写时，来源复用结构化 failure Control 保存领域失败或执行故障（区分原因），作为持久停摆依据；不另建一份 owner-record fault 状态。Control 绑定失败工作的输入边界，不能误暂停后来新 Input。已经被替代的旧工作只保留其回执与诊断，不给新工作追加旧 failure。

failure Control 也写不下时，来源 owner 保留内存中的待保存故障及准确工作身份，关闭对应新效果入口并报告 health；共享库故障则封闭所有依赖入口。此时没有持久停摆回执，必须如实报告，不声称健康或已经暂停成功。

每个 item 的异常边界覆盖 reader、open、start、执行和收尾；处理动作是封闭该 item 的启动权、保留恢复位置并报告。父任务取消继续传播。整库故障不能被伪装成互不相关的单项错误无限扫描。

### 7.2 进展与解封

一次 drive 必须：产生相关持久进展、到达终态，或等待一个明确 owner/条件。同一工作在相同恢复位置异常退出后不得无条件再执行；第一次发现无进展就停，不以多次付费失败作为熔断阈值。

进展包括该工作自己的准备、attempt 或 Message 变化，不包括无关 source 的 head 增长。周期扫描只允许重读和恢复原阶段；不能绕过故障状态或请求预算。

待保存故障由来源 owner 的退避核对驱动，即使没有新 head 也只重试原 failure Control 的保存，不重新进入 react、不请求模型。成功保存后进入现行等待显式 resume 的状态；若已被新输入取代，核对身份后结束旧故障修复，新的执行仍须通过真实存储恢复检查。已持久停摆不会被周期扫描自动解封。

failure 尚未保存时，resume 返回明确的“执行故障尚未保存/存储不可用”状态，不误报“输入没有等待恢复的失败或暂停”，也不绕过准入。修复保存成功后使用现行 resume 合同；不放宽为任意无终态 Input 均可 resume。这些故障分类与公共错误返回是需批准的语义变更。

新的 head/Input 不能证明存储已修复。恢复先由故障 owner 验证原操作或其可安全恢复步骤，再解除对应准入；`SELECT 1` 不能证明失败写入会成功。显式 resume 表达用户继续意图，也不能绕过损坏 schema 或未知外部效果。

重启后的内存 fault 消失不授权重发：已 started 的模型/工具记录仍阻断未知请求；已保存响应只重放本地后续步骤；准备前失败没有外部付费效果。程序缺陷在恢复后再出现，继续停止该来源，不做无人看管的进程重启循环。

保证边界：数据库拒绝全部写入且进程随后死亡时，无法保证未保存的故障跨重启仍被记住；在同一坏库加 fault 表也无法解决。重启仅按已保存准备/回执恢复，可能再次执行尚无持久故障证据的本地阶段；不承诺该阶段跨任意人工重启只执行一次。没有响应的 started 不自动重发，已保存响应不重新付费；若产品要求任意本地故障也跨重启永久停摆，须另行批准独立耐久故障介质，而非在本文隐藏引入。

## 8. 提交前提与中断投影

### 8.1 原子提交检查

日志 seq 继续只表示事实追加顺序。来源插件定义本次 Output 的提交前提：当前许可、输入/控制边界、输出前驱，以及本次决策依赖的工具结果身份。

ReAct 只携带准备时冻结的依据。来源通过窄提交入口，在现有 Message 事务里检查前提并追加 Output；Core 不解释 pause/abandon 或 provider 名称。事务只向该入口提供同来源必要读视图和 Output 写入权，不暴露任意 SQL、删除或其他 owner 状态。

复用 `OwnerStore.transact` 的事务载体，补充同事务窄来源读视图，由来源插件执行固定的提交检查；不得在事务回调内开启嵌套读事务。不是另建事务框架，也不向任意调用者开放自定义提交谓词。公共 MessageWriter 保持通用严格 CAS，不把输入边界、pause 等来源规则搬进 Core `_append`。

先保留严格 CAS 作为安全默认。发生冲突时，由来源检查新增事实是否改变声明的前提；只有证明不变才复用同一响应和 Output ID 重试本地提交。检查与追加必须同事务，不能“事务外检查，再换新 head”。新 Input/Control、竞争 Output 或相关结果变化使旧准备失效，作为被替代处理，不给新工作追加旧 failure。

不采用“所有 ToolResult 都无害”的规则。新模型请求之前闭合其依赖的 abandoned 调用，减少结算竞争；仍有旧段或不相关结果时，只有当前来源合同证明它不影响读集才可放行。普通消息追加 API 的严格 CAS 不被全局放宽。

### 8.2 模型可见中断

projection 从真实 Control 派生中断说明，不新增用户 Input。已提交的调用与结果保留关联：denied 表明未执行，success/error 使用原结果，interrupted 明确可能已有外部效果。

没有持久 ToolResult 时不能合成“已经成功/已经无效果”的观察；先由 Tools 结算，或只给出来源中断说明并阻断依赖该结果的新决策。provider 要求的 call/result 配对在投影中保证，不通过删除原 Message 达成。

Control 标记、工具观察和旧正文与来源读集使用同一边界解释。改变哪些结果进入模型上下文时，必须一起检查提交前提；不得一面恢复旧结果的决策作用，一面宣称它永远不影响决策。

## 9. 持久状态、迁移与恢复点

| 对象与 owner | 正常增加 | 原位更新/逻辑失效 | 物理减少与恢复证据 |
|---|---|---|---|
| MessageLog 的 Message | 追加 Input、Output、Control、ToolResult | 原正文不改；来源由 Control 关闭/暂停；Turn 仍投影 | 本设计不授权减少；保留全库备份与消息身份/seq/body 摘要 |
| ReAct 生成准备 | 首次生成前只增固定准备；同来源依据唯一 | 准备不可变，是否已输出/已被替代由 Message 引用推导 | 当前不得自动减少；与 owner records、binding 同库恢复 |
| Models model_calls | 按请求 key/attempt 增加真实调用；保存可恢复响应 | started → success/error；成功响应原子保存后不可覆盖；迟到响应不得覆盖终态 | 当前不得自动减少；Models 自有 schema 迁移和 SQLite 备份 |
| Tools/Delivery 回执 | 沿既有效果身份增加 | 沿既有阶段结算；已有终态不可被旧任务覆盖 | 沿现行合同，本设计无新删除权；原回执与 binding 保留 |
| 活动许可、残留任务、通知 event | 当前进程创建 | 许可撤销、任务退出、event 合并 | 只回收真实结束的临时对象；重启从持久记录重建，不伪造旧任务仍活着 |

Models 新 schema 需要请求 key、attempt、宿主身份、规范化 response 和错误信息。准确 DDL、唯一索引与版本号由 Models 实施合同固定，不能沿用已有版本号表示不同表形状。

旧 model_calls 没有生成 key/response，不按 request_digest 猜测映射，也不把缺失响应填成空成功。保留旧账、旧终态和未知值。切换前在独占维护边界检查未完成旧调用：无法证明恢复关系的来源需明确暂停/失败并保留原记录，不能由新扫描器自动重发。该切换协议是实施批准项，不是本次文档操作。

迁移前创建 owner 数据库一致备份和 manifest，固定 schema lineage、数量、受保护字段摘要、FK 与 integrity_check；迁移只增字段/记录及已批准终态更新。旧版本能否读新 schema 必须实际验证。接纳新数据后优先向前修复，不以旧备份覆盖新增消息；离线恢复需要同时核对生成准备、模型账和 Output 引用，不从不同时间点随意拼库。

不新增响应缓存 TTL、准备记录 GC 或孤儿数据删除。若将来需要保留期，另行确定 owner、引用与删除协议。

## 10. 分阶段实施与行为验收

本节是实施顺序，用户已授权在 stacked draft 实现 PR 中执行。每阶段单独审查 diff、公共 API 及持久语义，不靠修改红测获得通过。

| 阶段 | 范围 | 最小退出证据 |
|---|---|---|
| A | Message 通知、follow 核对、item 隔离、无进展停摆 | 丢通知仍处理 Input；坏 item 不杀好 item；失败回执写不下时不再次调用模型 |
| B | 来源许可与 Tasks 残留登记、结果等待、背压 | 终态/abandon 后新工作推进；旧写入被拒；真实清理和租约仍可观察；programmatic Tools 同样成立 |
| C | ReAct 准备、Models 响应回执、孤儿处理及单一重试预算 | 在每个崩溃窗口强杀后重开；响应已保存时 provider 次数不增；started 孤儿不盲目请求；重启不重置预算 |
| D | 来源提交前提、中断投影、其他后台消费者对照 | 无害竞争复用响应，真实抢占拒绝旧提交；模型看见效果证据；Delivery/scheduler/subagent 保留各自 owner |

### 10.1 十条红测逐项处置

以下名称均来自已固定 hash 的 `test_reply_liveness_red.py`。

| 测试 | 目标与需修订之处 |
|---|---|
| `test_committed_input_recovers_when_its_wakeup_is_lost` | 保留无新写入也恢复的断言；目标 API 应返回已提交成功，故不再要求 stale listener 异常污染 accept；另验证 incident |
| `test_one_lane_admission_fault_does_not_kill_reply_follower` | 保留隔离目标，补坏来源可见故障与无重复启动 |
| `test_reply_follower_faults_when_drive_makes_no_durable_progress` | 保留 starts==1；将“整个 watcher 抛错退出”改为“该来源停摆，好来源继续”，修订先独立评审 |
| `test_terminal_output_releases_lane_before_cleanup_finishes` | 保留；补旧 writer 失效、新 handle 不被旧 cleanup 删除 |
| `test_successful_model_response_is_durable_before_return` | 保留；扩展 tool calls/continuation/usage 和读回 schema |
| `test_received_model_response_survives_crash_before_output` | 保留 calls==1；补固定 Output 身份、冻结材料和真实子进程强杀 |
| `test_orphaned_started_model_call_reconciles_without_replay` | 保留不重发；去掉吞任意异常的宽松验收，明确断言 error 原因、未知 usage 和来源停止 |
| `test_tool_result_releases_waiter_before_cleanup_finishes` | 保留；补 programmatic 调用、成功结果不被清理失败改写 |
| `test_abandon_releases_lane_when_model_ignores_cancel` | 保留；补迟到成功不提交、旧 Scope 仍有 owner |
| `test_abandon_item_fault_isolated_and_later_calls_continue` | 保留；补故障项仍可发现，cursor 不吞待办 |

### 10.2 必需反例

1. 通知丢失、扫描、用户重复输入并发：只有一条 Input、一个被接纳生成，无新消息也能恢复。
2. 模型响应保存后解码失败：不重新请求；输出幂等；工具只在 Output 提交后执行一次。
3. abandon、旧结果、新 Input 和新 Output 交错：旧回执至多一个终态，旧任务不能串写新来源。
4. 模型流式输出后 retry：旧 preview 明确撤销，新 attempt 身份可见；callback 或账目失败不当网络错误。
5. 存储不可写：停止相关外部效果；恢复只继续缺失阶段，不先发新模型请求“试试看”。
6. 插件 apply/换代时存在真实活调用：不扫成孤儿，不释放租约；候选/正式账互不误处理。
7. 残留任务达到额度：明确背压；已接纳 Input 不丢，资源退出后恢复；热更新不得假报成功。
8. 持久 write-set 验收：旧 Message/seq/body 不变，schema 迁移无猜测回填、无自动删除、unknown usage 不变成零。
9. 无关 ToolResult 抬高 head：复用同一响应完成原子提交，不升级为 failure Control；相关结果改变则拒绝旧准备。
10. 同进程旧 Root 已排空但有 started 残留，以及同 Root 的 attempt 已退出：明确结算而非永久等待；仍活的旧 Task 不被误判，新 abandon 后工作不等它。
11. `follow_abandon` 完全不运行：ReAct 直接完成所需幂等结算，或明确阻断且 provider 次数为零；与后备清扫竞态只产生一个结果。
12. failure Control 首次保存失败、head 不变：修复核对能保存停摆，期间 provider 次数不增；显式 resume 才继续。补保存前强杀，验证上述耐久保证边界而非假定故障标记存在。
13. 死 listener 被移除一次，迟到注销无异常；不确定通道故障保留订阅且告警有界。响应保存确认丢失按回读复用，不产生第二次模型请求。

并发使用 Event/barrier 或可控时钟协调。真实进程恢复使用独立一次性数据库和 stub provider 请求账，按提交屏障强杀子进程；请求计数与持久结果一起验收。运行时行为验证、迁移验证、CI/Gate、真实客户端证据分别报告。

## 11. 方案权衡、批准边界与评审记录

不采用只增加 timeout/catch/retry 的补丁路线：它不能证明业务终态、原调用身份或清理责任。也不照搬完整 durable harness：Message、Control 和工具回执已经承担大量执行事实，第二套 Turn/Operation 表会重复 owner。

选择局部扩展的代价是需要两个明确的新合同：可恢复的生成准备，以及来源许可与物理任务分离。它们分别解决跨崩溃关联和不合作清理，不能相互替代。删除任一项都会重新出现对应红测的失败路径。

设计批准后才可更新现行条款/决策并建立实施合同，重点确认：Models 调用账升级为响应恢复依据、自动重试的唯一 owner、来源窄提交入口，以及终态提前释放等待的公共 API 行为。pause 和普通新输入不采用 abandon 语义，0070/0071、append-only 和未知 usage 不变。

Devin 首轮独立评审：hgt `T-77bba0`，回执“修订后可接受”，验收命令通过（原稿及红测 hash、`git diff --check`），不是运行时验收。回执标题称四项必修订，正文实际为 F1～F5 五项。

同一位 `devin-1` 复核：hgt `T-f06538`，结论“可接受”，F1～F5 必修订全部闭合，无新增必修订。审阅版本 SHA-256 为 `ed770d56d394e80599baf8fbd4535661df96860b0544a48cfce362f0e0a32301`；作者核对完整 `hgt_show` 回执、验收退出成功以及设计和索引 hash 未变。本工具未提供独立 scope 判定，范围核对由作者检查 worktree 完成，不把 `accept=ok` 当作范围或语义证明。此后只更新本文交付状态、评审记录和以下实施注意事项。

| 意见 | 处理 |
|---|---|
| F1 结算无后继 owner | 采纳：ReAct 直接请求 Tools 结算，后备 watcher 共用同一回执；不跨 await 持来源锁 |
| F2 宿主粒度 | 采纳：进程实例、Root 实例、attempt；真实退出证明，不以 generation 或进程存活替代 |
| F3 停摆与恢复 | 采纳问题，调整方案：复用 failure Control 与原故障保存重试，不建重复 fault 表、不放宽任意 resume；明确坏库加崩溃的保证边界 |
| F4 重试预算及保存歧义 | 采纳：driver 单次尝试、SDK 无隐式重试、attempt 级计账、保存后读回 |
| F5 死订阅 | 采纳：确认死亡才移除，注销幂等，未知故障告警限频 |
| F6 许可实体 | 采纳：只是既有 handle/边界/writer 合同，不建新许可实体 |
| F7 提交 API | 采纳复用事务与 CAS 回归；不采纳把来源语义放进 Core，复用 OwnerStore 的窄事务能力 |
| F8/F9/F10 | 采纳：保守 cursor 幂等重扫、Tasks 原子容量等待、补充决定性反例 |

用户本轮指定一个 Devin，不将此评审冒称仓库实施阶段要求的 Terra xhigh 概念 Gate。实现阶段仍须执行对应正式 Gate。

复核留下的非阻断实施注意事项：

- R1：确定性坏 item 的无效果重扫仍可能长期消耗本地资源。实施时区分可恢复存储失败与不可解释回执；后者沿 §7.1 由人工处理，不用任意 N 次失败阈值决定事实是否损坏。停止自动处理不得丢失该项，也不得阻止同 session 后续 item。具体隔离和恢复入口在实施合同中验收。
- R2：关闭顺序明确为先禁止新调用、排空相关 Tasks 并允许原调用保存响应，再由 Models 结算剩余 started，最后关闭 store。Task 尚未退出不能提前扫为孤儿或关闭其存储；不合作任务仍阻止整体换代。实施时验证资源依赖确实满足此顺序。
- R3：provider 查询是可选能力、候选账物理隔离是前置验收，均非已经完成的运行验证。进程内活 attempt 表也不能单独证明另一个进程死亡；共享调用库必须先验证独占宿主约束，否则须另行设计跨进程所有权协议，不能直接套用进程内恢复。

本设计 PR 的交付范围仅 `docs/design/interrupt-and-fault-model.md` 和 `docs/INDEX.md`；实现、红测修订与验证证据在 stacked 实现 PR 中。文档差异检查通过；设计评审阶段未运行实现测试、CI、正式 Gate 或真实 provider。补充红测的既有失败结果见 §2.1。

恢复证据位于本机 `/tmp/akashic-liveness-design-20260920-PzWjTc`：原索引 `INDEX.before.md`、基线归档 `source-69a67e7f.tar`、首稿 `design-review-v1.md`、已审修订稿 `design-reviewed-v2.md` 和两轮 `devin-review-v*-reply.md`。临时目录不是长期归档，后续提交时应把必要证据随实施合同保存。首轮后出现的空 `.done` 已移入该目录保留，没有删除用户数据。未 commit、push 或修改远端 PR。
