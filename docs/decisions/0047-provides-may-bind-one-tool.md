# 0047 · 一个 provide 可以绑定一个 Tool

- 状态：accepted / implementing
- 日期：2026-08-28
- 关联条款：PLG-001～PLG-014、PRO-001～PRO-002
- supersedes：0041 中“互斥角色声明都不提供可调用服务”的局部约束
- superseded by：无

## 背景

插件已经能用 `provide` 表达互斥能力，用全局 Tool catalog 表达可调用工具，但消费者只能按
工具名字耦合具体 provider。Memory 只是第一个暴露问题的例子：Wake 需要“当前 Memory
provider 选择暴露的召回 Tool”，不应 import Akasha，也不应让 Core 新建 memory registry。

## 决定

一个插件可以 provide 多个独立 Service；每个 Service 最多显式绑定一个由同一插件注册的
Tool。没有绑定 Tool 的 provide 仍只是占位和互斥声明。消费者 inject Service，并在运行时
从当前 exact-generation Tool catalog 解析该 Service 绑定的 Tool 名字。

```text
provider plugin
├─ provide A ── bind ──▶ tool_a
├─ provide B ── bind ──▶ tool_b
└─ provide C             空占位

consumer plugin ── inject A ──▶ tools.from_provide(A) ──▶ tool_a
```

绑定只增加在现有 `register(..., provided_for=...)` 上，不新增插件 manifest 字段或 memory
专用接口。重复绑定、跨 owner 绑定和绑定不存在的 Service 都在 catalog freeze 时 fail-loud；
消费者从空占位取 Tool 时以 `PROVIDED_TOOL_NOT_BOUND` fail-loud。热更新时解析当前租约保护的
完整 RuntimeSnapshot，不能只看候选 Root 的局部声明。

Akasha 继续用 `plugin.claim.embedding_memory` 作为空的互斥占位，并另行 provide
`memory.recall.v1`，只把 `recall_memory` 绑定给它。`remember`、`forget` 和诊断 Tool 仍在普通
catalog 中，不因 Wake 依赖 recall 而一并授权。

Wake 将来源事实分成三个正交入口：Feed 等待选文章仍进入 Content；Calendar 与 Fitbit 健康
事件上报 Alert；Fitbit 睡眠与 Steam presence 上报会过期的 Context。Alert 绕过兴趣初筛，
Context 只注入 Alert 和 Content 第二轮，不进入第一轮。

Content 使用两个内存 Turn：第一轮读取 MEMORY.md、主动偏好规则、时间和候选，必须选择
一到八条；第二轮不再注入整份 MEMORY.md，只接收初筛结果、主动偏好、时间、ContextEvent，
并获得 recall、web fetch、share、skip 四类 Tool。调查总预算由变量拼入 Prompt；预算用尽且
尚无终态时，Core 通用 scoped ReAct 只再给一轮终态 Tool。只有 share 或 skip 拥有决定语义。

## 理由

- Service 仍只表达一项能力；一个插件提供多个能力时使用多个 provide，不把多个 Tool 塞进一个宽接口。
- Tool 的发现、generation、授权和执行继续由现有全局 catalog 拥有，Service 只提供精确索引。
- 空占位有明确意义，不需要 nullable Tool、异常捕获惯例或新的 inject 字段。
- Wake 只依赖能力合同，不知道 Akasha、Calendar、Fitbit 或 Steam 的实现。

## 影响

- 插件作者若要共享 Tool，必须显式提供一个独立 Service 并绑定它；不绑定就是有意不可调用。
- 每个 provide 只能绑定一个 Tool；需要多个 Tool 时拆成多个可独立依赖的 provide。
- Wake Dashboard 从自己的 durable projection 展示看见数量、初筛选择、时间、最终决定和理由，不读取推理临时历史。
- Alert/Context 来源先上报 Wake 自有状态，再推进各自 cursor 或 ACK；崩溃重放使用稳定来源身份。

## 验收

- [ ] 冷启动与“稳定 provider + 候选 consumer”热更新都能解析同一绑定 Tool。
- [ ] 空 provide、重复绑定、跨 owner 绑定全部 fail-loud。
- [ ] Akasha 只有 `memory.recall.v1` 绑定 `recall_memory`，embedding claim 保持空占位。
- [ ] Content 第一轮没有 ContextEvent 和外部 Tool；第二轮没有完整 MEMORY.md。
- [ ] 第二轮预算耗尽后只多一轮 share/skip，成功终态立即结束。
- [ ] Calendar/Fitbit Alert 和 Fitbit/Steam Context 在重放、过期与终态 ACK 测试中保持唯一 owner。
- [ ] Dashboard 源码通过构建、lint 与 AI slop 扫描。
- [ ] hua-home exact Core/plugin generation 与真实 Wake Turn 留下运行证据。
