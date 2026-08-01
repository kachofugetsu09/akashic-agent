# Companion 安全边界与 Edge Case 实施设计

- 状态：implemented / public Gate
- 日期：2026-08-01
- 设计 owner：主 Agent
- 合同决策：[0017](../decisions/0017-one-person-companion-security-boundary.md)
- 长期条款：SEC-001～SEC-010
- Gate：G1～G9

## 1. 目标与威胁模型

Akashic 只服务一个人。Telegram、QQ、Mobile、Web Chat、设备和 session 是渠道，不是身份、租户或权限边界。所有进入渠道的消息都按服务对象本人处理；本设计不增加认证、Origin、per-channel ACL、per-device session isolation 或跨渠道记忆隔离。

本轮只处理四种仍然真实存在的问题：外部输入影响宿主或权威状态、模型参数覆盖 runtime provenance、长期运行资源无界增长、畸形数据使局部流程进入错误终态。Peer 是用户明确要求删除的能力，不再建立 Peer 信任模型。

## 2. 共同失败语义

| 分类 | 当前操作 | Runtime | 典型 edge case |
|---|---|---|---|
| `operation_rejected` | 当前请求返回明确错误 | 继续 | 未知工具字段、blocked URL、第 11 个 Schedule、容量已满 |
| `item_quarantined` | 单条数据记录原因/source/identity，不提交正常状态 | 同 batch/tick 继续 | MCP score/timestamp 非法 |
| `degraded_continuation` | 结果标记缺失项，合法部分继续 | 继续 | QQ 一张图片失败但文本和其他附件合法 |
| `unit_failed` | 当前 job/tick/query 有失败终态和诊断 | 其他 unit 继续 | 单个 plugin query、Schedule 执行或持久提交失败 |
| `cleanup_degraded` | 保留 owner、resource identity 和可重试诊断 | 继续；必要时隔离同 owner 新工作 | shell log、spill、receipt、durable handoff、lease 清理失败 |
| `runtime_fatal` | readiness/主循环明确失败 | 退出或拒绝启动 | canonical store 损坏、无法建立 owner、核心内部不变量违反 |

可恢复失败不得被静默记录成空成功；权威状态损坏不得被解释成空集合；cleanup 失败不得倒推已经提交的 turn 或外部效果失败。

## 3. Runtime provenance 与显式 target（G1 / D1）

```text
ToolExecutionContext(origin_channel, origin_chat_id, origin_session_key, turn_id)
                         │ runtime 注入，模型不可覆盖
                         ▼
                   Tool implementation
                         ▲
                         │ schema 校验后的模型参数
              target_channel / target_chat_id
```

Registry 统一拒绝未知字段。`memorize`、`recall_memory` 等普通工具不公开 origin 字段；`message_push` 的参数命名为 `target_channel`、`target_chat_id`，允许 Mobile turn 显式发送到 Telegram/QQ。origin 仍写入 provenance，target 只决定外部效果。当前调用被拒绝时，Agent 可修正后继续同一 turn。

Mutants：模型参数覆盖 origin、未知字段被忽略、message_push target 被强制为 origin。

## 4. 有界外部 I/O 与附件（G2 / D2）

`web_fetch` 允许单人本地运行访问 localhost、私网和内网 HTTP 服务；每一次 redirect hop 仍校验 HTTP URL 结构并受 hop 上限约束，且不读取环境代理。其他外部 HTTP consumer 继续逐次执行公开地址策略。响应在分配内存前受传输和磁盘绝对上限约束。合法的大响应不拒绝：超过内联阈值后流式写入 execution-owned 私有临时目录，并返回 Agent 可分页读取的文件引用。turn 结束或显式 release 后清理；cleanup 未确认时保留 owner/path 和诊断，不报告已删除。

上传、附件和 QQ media 在读取/分配前验证单项与总量字节上限。`message_push` 使用单个 `ChannelMessage` 和结构化 `DeliveryReceipt` 提交完整逻辑消息；Mobile 只有在目标设备仍存在时，才把附件记录与全部设备 inbox 行原子提交。临时文件或正式附件原子提交失败时，不发布半完成引用；提交后的临时快照清理失败只记录 `cleanup_degraded`，不把已经完成的外部效果改写成失败。QQ 单个 media 失败只产生缺失诊断，合法文本和其他附件继续。

Mutants：redirect URL 结构或 hop 上限被跳过、读完才检查上限、大响应被错误拒绝、spill 失去 execution owner、上传先读完整 body。

## 5. Peer 全面退役（G3 / D3）

删除 Peer config schema、route/worker、tool catalog、Prompt 注入、任务状态和协议处理。遗留配置必须在边界返回 `unsupported` 或 `unknown capability`，不能静默启用或解释为空能力。全局记忆继续属于同一位用户，不按 channel 拆分。

Mutants：Peer route 仍注册、遗留配置被当作有效、删除后仍注入 Prompt。

## 6. MCP 摄入与 Wake reservoir（G4 / D4）

单条 score/timestamp/schema 非法时 quarantine 并记录 source/cursor/item identity；同批合法记录和后续 tick 继续。material candidate window 固定 100。新事件可 kick Wake；旧 reservoir 只提供衰减后的聚合 wake mass，不因仍在池中就自动成为本轮素材。

记录满足最小驻留期且分数低于 decay floor 后，协议 owner 先完成 ack/cursor 提交，再在同一可恢复事务中删除 payload。事务失败时 cursor 不前移，payload 不提前删除；低分事件在到达阈值前仍可贡献少量唤醒质量，但不是素材。

Mutants：坏 item 使 batch 失败、旧池未经衰减成为素材、material window 无界、未 ack 即删除、低于阈值但未满足驻留期就删除。

## 7. Schedule 容量（G5 / D5）

workspace 全局最多 10 个 active job。add 构造 candidate 并原子保存；第 11 个 add 返回 `schedule_capacity_reached`，不改变既有任务。Agent 应询问用户移除哪个已不用的任务。频率、due time、外发预算和自动降频不在本轮限制内；只有显式 cancel 才能减少任务。损坏 `schedules.json` 仍按 SCH-001 fail-loud，不能变成空任务集。

Mutants：无限 add、达到上限时删除旧任务、按 channel 分开计数、把频率限制混入容量合同。

## 8. Receipt 与 Plugin query lease（G6 / D6）

completed receipt 从 `completed_at` 起保留 7 天，每设备高水位为 10,000 条或 64 MiB。新命令先清理过期 completed；仍满则当前命令返回 `mobile_command_receipt_capacity_reached`，有效 receipt 不得删除，runtime 继续。相同 request 重放返回原结果；同 ID 不同 request 继续 conflict。

processing receipt 不按 TTL 盲删。重启或 reconciliation 必须判断真实外部效果：已完成则补交 completed，明确未执行且无副作用才允许重试，无法判断则持久化 `outcome_unknown` 并阻止自动重放。receipt 状态提交失败不得报告 accepted/terminal success。

Mobile `message.send` 在返回 accepted 前，先把完整 inbound handoff 持久化到 `sessions.db/inbound_handoffs`，再进入 MessageBus。worker 消费不释放 durable handoff owner；只有 turn 收束且 handoff 删除确认后才释放。进程崩溃时按有限页恢复；canonical user 已存在时删除 handoff并补 receipt 对账，不再创建重复 turn。handoff 删除失败保留 row、强引用 owner 和 `cleanup_degraded` 诊断，由 bus-owned cleanup retry 收束。

Plugin query timeout 只结束当前 query 观察；真实 worker 结束前继续占用 quota 和 generation lease，超时后的 handler 结果进入明确终态，lease drain 完成后才释放。

Mutants：有效 receipt 被高水位删除、processing 被盲删重放、handoff 删除失败后丢失 owner、timeout 提前释放 quota/lease。

## 9. Shell、Subagent 与 MessageBus（G7 / D7、G8 / D8）

Shell retained log 由 execution owner 管理，达到 cap 只拒绝当前 execution。terminal cleanup 失败保留 execution/log owner 和诊断，并隔离同 owner 新 spawn；已经提交的 turn 不改回失败。同步 subagent 和后台 subagent 使用同一个 admission owner，不能由同步路径绕过容量。

MessageBus 不设置独立全局 backpressure 或容量拒绝；它负责 lane 顺序，Mobile accepted 另由 durable handoff 保证崩溃恢复，直到 handoff 删除确认。控制 admission 只统计 queued/running turn 的数量、字节和 live runtime objects，不统计历史 programmatic thread/channel。

Mutants：cleanup 丢失 owner、sync spawn 绕过 admission、已接纳 Mobile handoff 静默丢失、历史 thread 阻止新 turn。

## 10. Control replay（G8 / D8）

运行中每 turn replay ring 最多 256 events/4 MiB，全局最多 32 MiB；eviction 只影响晚到 replay 请求，当前 live subscriber 继续获得新事件。晚到请求的起点已被淘汰时返回 `replay_truncated` 与当前 snapshot。terminal replay 最多保留 5 分钟；过期返回 `replay_expired`，并从 SessionStore 读取权威最终状态。

replay 回收不得删除 `sessions.db/messages` 或覆盖既有 terminal result。单一 runtime-owned reaper 按最早 expiry 唤醒，空闲时也必须在 wall-clock grace 后清除 replay/runtime objects；shutdown 取消并收束 reaper。history、sequence、global index 或字节计数不一致属于内部契约损坏，必须 `runtime_fatal`，不能降级成 `cleanup_degraded`。

Mutants：ring 无界、淘汰导致 live subscriber 丢事件、过期返回静默空流、历史 replay 回收删除 SessionDB 消息、terminal object 永不回收。

## 11. Fitbit 字段安全（G9 / D9）

`efficiency` 只接受有限范围数字；数字字符串可以规范化，NaN、Infinity、负数、越界或 HTML 字符串显示 `--` 并写字段级诊断。渲染使用数字节点、`textContent` 或等价安全 sink，不能把 provider 原值拼接到 `innerHTML`。其他睡眠字段和其他天继续展示，不把单字段非法升级为整批 snapshot 失败。

Mutant：`efficiency` 通过 `innerHTML` 进入 DOM。

## 12. 持久化状态与恢复证据

| 对象 | 正常增加 | 原位更新/逻辑终态 | 物理减少条件 | owner/恢复证据 |
|---|---|---|---|---|
| Mobile receipt | command admission 新建记录 | processing → completed / outcome_unknown | completed 超过 7 天；processing/unknown 不按 TTL 删除 | Mobile receipt owner；request hash、external effect count、reconciliation report |
| Mobile inbound handoff | accepted 前持久化完整消息与 dedupe identity | pending → worker owned；重启分页恢复；canonical user 对账 | worker terminal 且 DELETE 确认；失败保留 row/owner | MessageBus/Passive worker；row、cleanup retry、canonical message |
| MCP reservoir | source event、quarantine 记录 | score、ack、cursor、consumed/decayed 状态机 | 最小驻留期 + decay floor + ack/cursor 与删除同一可恢复事务 | Wake/MCP owner；cursor、accepted/quarantine 快照、提交证据 |
| Schedule | 用户明确 add | reschedule/due metadata；完成或过期 one-shot → `enabled=false` | 用户明确 cancel | JobStore；candidate/commit 快照 |
| Execution spill/log | execution 输出追加 | active → terminal / cleanup_degraded | execution 结束且删除确认；失败保留 owner | execution owner；registry、path/size、cleanup report |
| Control replay | turn event 追加 | active bounded ring → terminal grace | ring 高水位或 terminal 超 5 分钟；不减 SessionStore | Control owner；SessionDB 不变、truncated/expired result |

`sessions.db/messages` 正常路径继续 append-only；本设计的容量和清理协议无权 UPDATE/DELETE 对话正文。所有物理减少都必须从 DB、文件、事件和诊断观察并可恢复。

## 13. Gate 映射与实施波次

| Gate | Stable contract | 主要 owner |
|---|---|---|
| G1 | SEC-001 | Tool Registry、memory adapter、message_push |
| G2 | SEC-002 | HTTP requester、web_fetch、upload/QQ media |
| G3 | SEC-003 | Peer config、route、tool、Prompt、protocol |
| G4 | SEC-004 | MCP ingestion、Wake reservoir |
| G5 | SEC-005 | Schedule admission/JobStore |
| G6 | SEC-006 | Mobile receipt、plugin query lease |
| G7 | SEC-007 | Shell、subagent admission |
| G8 | SEC-007～SEC-008 | MessageBus、control replay |
| G9 | SEC-009 | Fitbit DTO 与 Dashboard renderer |

实现已按依赖关系拆入 stacked PR；共享合同、oracle 和 baseline 由本层唯一拥有，产品 PR 只修改各自 owner 路径和 focused tests。累计 head 的公开 Gate 覆盖 selected scenarios、正例、known-wrong mutant、受保护 write set 和资源清理。生产路径与合同同时变化时执行完整公开场景。

G9 的公开 command 运行语义 oracle 和 Dashboard focused tests；私有 provider 的本地 native 测试可以作为维护辅助证据，但不再构成额外的合并 Gate。

`coverage-baseline.json` 的 `purpose` 是 `approved_contract_mapping`：`coveredP0` 只记录批准的场景映射、冻结 base 和 catalog digest，不表示这些场景或 mutant 已经运行通过。只有实现 head 上的公开 Gate 报告才能提供运行证据。

## 14. 非目标与移除的扫描语义

本设计明确不修：认证、Origin、per-channel/per-device/session ACL、跨用户泄露假设、跨渠道全局记忆隔离、Peer trust 修补、额外 Schedule 频率/due/外发预算、receipt 7 天内清理、programmatic 历史 thread quota、整批 Fitbit snapshot 失败、自动 post-response correction scope/低置信度 skip，以及没有真实可达违反路径的重复防御检查。

旧扫描编号 #1、#3、#4、#6、#7、#8、#11、#12、#13、#14、#15、#17、#18、#20、#21、#22、#23、#24、#27、#30、#33、#35、#37、#40、#43、#45、#46、#48 从可执行待办和 Gate 中移除；#10/#19 通过 D3 删除 Peer 表面而不是修补信任模型。
