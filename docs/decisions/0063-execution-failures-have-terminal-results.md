# 0063：执行失败有明确终态，恢复依据原回执

- 状态：accepted
- 日期：2026-09-09
- 范围：Tools、Delivery、Wake、scheduler、subagent、Mobile command、Models
- 取代：旧执行合同中的 `unknown`、`outcome_unknown`、`delivery_unknown`、`uncertain` 状态

## 问题与选择

`unknown` 混合了执行失败、取消、缺少回执和是否产生外部效果。Tools 已把它保存为结果，
Delivery 却继续把它当待恢复状态；Wake 不关闭领域领取，scheduler 和子任务也不结算，
于是失败会无限占用后续工作的入口。

结果只说明本次执行如何结束；回执记录已经确认的效果；是否能重试由实际 provider 的
同 key 幂等和查询合同决定。三个问题各有 owner，不再由一个“不确定”状态代答。

| 边界 | 终态 | 可观察含义 |
|---|---|---|
| Tools | `success` | 工具正常返回成功结果；后台作业的成功只证明已接纳并返回作业身份 |
| Tools | `denied` | 尚未执行时被明确拒绝 |
| Tools | `error` | 工具返回失败，或中断后不能找回结果；不能据此推断效果为零 |
| Tools | `interrupted` | 本地执行或等待被取消，或恢复时授权撤回；不表示外部效果已撤销 |
| Delivery、Channel | `delivered` | provider 或原持久回执确认完整送达 |
| Delivery、Channel | `rejected` | 有明确未发送证据；只有这一失败允许现有显式 retry 重新准备 |
| Delivery、Channel | `failed` | 本次发送失败；保留错误及已确认的 provider IDs，不自动重发 |
| Models | `success / error` | 本次调用正常返回或失败；usage 和耗时只记录实际已知值 |
| Mobile command | `completed` | 已保存正常成功或错误回复；相同 command ID 复用该回复 |

`started`、`prepared`、`processing` 是 owner 正在处理的阶段，不是新的失败桶。
插件 apply 和 channel start 也用于热重载，不能据此认定旧 generation 的调用已经死亡。

## 自愈、跳过、重试与中断

```text
┌──────────────┐
│ 读取原持久回执 │
└──────┬───────┘
       ├── 已有终态 ────────────▶ 返回原结果，关闭本次业务等待
       ├── 尚未 started ────────▶ 原 owner 按当前授权首次执行
       └── 已 started ──▶ 查询原 key
                          ├── 找到结果 ─────▶ 保存并返回
                          ├── 同 key 幂等 ──▶ 按 provider 合同恢复
                          └── 无法安全重发 ─▶ 保存失败并关闭等待
```

| 情况 | 行为与 owner |
|---|---|
| 重启后已有 ToolResult、发送回执或 Message | 原 owner 读取并复用；不重跑模型或副作用 |
| started 后缺少本地结果 | Tools/Delivery 先 query 原 key；只有 provider 声明幂等才可用相同 key 重发 |
| 调用参数、工具名或模型展示协议错误 | 保存可反馈原因，模型在原步数上限内纠正；不得伪造成功工具调用 |
| 正常业务失败、网络超时、部分送达 | adapter 返回 error/failed；模型可检查现场再决定下一步，后台 owner 关闭本次工作 |
| 明确未发送的 rejected | 调用者显式 retry；同一消息、原地址、原 binding 和效果 key 不变 |
| 原工作已结算、版本失效或明确取消 | 可以不再执行；保留回执、控制或诊断原因。静默跳过不是吞掉任意异常 |
| 数据库、binding、schema、不变量或意外程序异常 | 保留已有结果/错误诊断并传播原异常；立即中断当前执行，不能包装成业务成功 |
| 用户取消或停机 | 排空自己拥有的 Task 并保存取消结果；未启动的持久工作仍由原 owner 恢复 |

可恢复的工具失败继续反馈模型；内部错误仍 fail-loud。这与本地 Codex 的
`RespondToModel` / `Fatal` 分工一致，但 Akashic 保留自己的持久效果身份和业务收尾合同。
不新增全局重试次数、超时重发策略或 effect certainty 布尔位。

## 后台来源收尾

- Wake 发送失败：关闭原 Pointer；Content 仅把实际引用成员标记 failed，其他候选仍可选择；
  Drift 保留原领取身份并标记 failed；Alert 结束原领取为 skipped，发送回执和 attempt 明确保留失败。
  这些操作不产生 delivered 或来源 ACK，也不自动重新选择同一失败 revision。
- scheduler：一次 fire 的全部发送完成后保存 delivered 或 failed；失败不永久保留 pending。
- subagent：结果通知返回终态后关闭该次回传；失败留下发送回执与诊断，不再次生成或发送原结果。
- Mobile：只在原 command 的恢复路径确认没有活跃 owner 后生成 `command_interrupted` 错误。
  只有 Message 和 durable handoff 都不存在且没有已执行证据时，才提示可以安全重试。

## 持久化与迁移

`20260909_02_execution_failures` 在正式 workspace 独占锁下迁移四个 owner 的旧表：
Mobile command、Models 调用账、Wake attempt v8→v9、旧 Core Delivery v1→v2。
每个数据库先验证已知 schema 和完整性，再用 SQLite backup 保存受保护的恢复目录、数据库与 manifest。
表重建只解释旧失败终态，逐行核对其余字段；不删除行，不补造送达、usage 或时间测量。
迁移保留 `started/processing`，不推断跨 generation 的存活事实；重复运行不生成第二份迁移。

Message 正文正常路径仍只追加。旧 ToolResult `unknown` 和 Message owner 回执只在读取边界
解释为 error/failed，磁盘原值和原正文不改写；新公共类型不再接受 unknown。
失败发送不减少 Message、附件、领域输入或回执。Mobile completed 仍采用原 7 天保留合同，
processing 和尚未完成的 handoff 不因此次变更获得 TTL 删除权限。

本轮只交付 Core 代码和一次性测试库演练；正式 workspace 未迁移、PR 未合并、服务未部署。
外部插件适配按后续明确任务进行，不能通过运行 cache 修改或静默状态别名绕过合同。
