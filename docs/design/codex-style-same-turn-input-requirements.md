# Codex 式同 Turn 输入需求合同

- 状态：accepted / implemented
- 日期：2026-08-06
- 决策：[0025](../decisions/0025-codex-style-same-turn-input.md)
- 设计：[Codex 式同 Turn 输入设计与任务合同](codex-style-same-turn-input.md)
- 关联条款：CTX-003、SES-001～SES-002、SES-005、SES-007～SES-008、MEM-009～MEM-010、RUN-002～RUN-003、RUN-007～RUN-008、OUT-001、OUT-005

## 1. 用户可见目标

普通用户输入在 session 没有 active execution attempt 时创建新 attempt；若上一个 attempt 被中止且尚无最终 A，则自动续接同一个 logical interaction。用户不选择 `steer`、`follow-up` 或 `next prompt`。

一次 logical interaction 可以按顺序经历 `U1 → stop → U2 → stop → U3 → A_final`。每次 stop 只结束 execution attempt；下一 attempt 必须看到此前 U 和所有已闭合工具事实。Mobile active 时尾部动作始终是中止，草稿保留但不能发送；中止收束后才恢复发送。Telegram、QQ 等 channel 的 `/stop` 使用相同语义。

## 2. 需求

### STI-001 普通输入自动选择 logical interaction

同 session 没有 active attempt 时，普通 U 创建 attempt；最新 interaction 没有最终 A 时，新 attempt 沿用其 identity。active attempt 期间 Mobile 不接收普通发送。core 根据 durable terminal 状态选择 interaction，客户端不暴露模式选择器。

### STI-002 同一 turn 支持多个 U

一个 turn 的输入是有序非空集合，而不是单条 user message。每个 U 拥有稳定 item ID、ordinal、原始正文、附件引用和客户端身份；任何 consumer 不得用相邻角色推断 turn 归属。

### STI-003 新 U 在 Attempt 边界生效

新 U 只能在前一 attempt 已中止收束后进入下一 attempt。active attempt 上的普通 `turn/start` 必须返回 busy；控制协议不提供 `turn/steer`。已完成的工具调用和结果按既有证据保留规则继续持久化并进入下一 attempt；正在执行且没有 result 的工具以 interrupted 闭合，但不进入下一次模型 replay。

### STI-004 最终 A 才结束 Logical Interaction

interrupted、cancelled 或 failed attempt 都不关闭 logical interaction。只有一次 attempt 成功提交 terminal assistant 时，该回复才成为 `A_final`，interaction completed；此后的普通 U 才创建新的 interaction。

### STI-005 控制面只提供精确中断

active attempt 期间唯一改变执行状态的用户动作是携带精确 thread/turn identity 的 `turn/interrupt`。普通输入只走 `turn/start`：active 时明确 busy，terminal 后由 core 根据 durable predecessor 创建下一 attempt。adapter 不得提供 steer、follow-up 或 next-prompt 分支。

### STI-006 Attempt checkpoint 可恢复和重放

每个 attempt 的初始 U、工具 started/completed 和状态实时写入 `turns`。下一 attempt 从前驱链恢复全部 U，并把每个已完成 tool call/result 投影回模型历史；未闭合工具和 partial assistant delta 不重放。SessionDB `messages` 仍只在 interaction completed 时提交完整 transcript batch。

### STI-007 Completed transcript 是多个 U 加一个最终 A

completed turn 在一个 SessionDB 事务中按 ordinal 追加全部 U，随后追加唯一 `A_final`。每条消息携带相同 `control_turn_id`；user 携带 `turn_input_ordinal`，assistant 携带 terminal 标志和输入数量。历史窗口或 consolidation 起点若落在该 turn 中间，replay 必须退回 U1，允许实际投影量超过消息数预算。现有消息正文保持只追加。

### STI-008 硬终止保持独立

Mobile 中止动作和 `/stop` 只调用 `turn/interrupt`，把 active attempt 终结为 interrupted。它们不注入 user message、不伪装成 steer，也不自动启动下一 attempt。Mobile active 时无论草稿是否为空都只提供中止；草稿保留，中止收束后才允许发送。

### STI-009 Akasha 使用显式多输入投影

Akasha 对一个 completed logical interaction 建立一个学习样本：有序聚合全部 U，输出为 `A_final`。attempt checkpoint 不触发在线学习，也不进入离线 rebuild；在线与离线 builder 只从最终 transcript 的 `control_turn_id`、ordinal 和 terminal 标志重建。旧消息继续走明确的 legacy pair 兼容路径。

### STI-010 失败和竞态不得丢输入或串 turn

输入 admission、hard interrupt 和 turn completion 共用同一个 session/turn owner。active 时到达的普通 U 明确拒绝，不得塞回旧 attempt；客户端在 interrupt terminal 后重发，core 才创建下一 attempt。重复或过期控制请求不得影响当前 turn。

### STI-011 历史预算按 Logical Interaction 计数

Session compaction、Markdown consolidation 的保留尾部、分页切点和 prompt history 都按不可拆分的逻辑历史单元计数。一个 completed `U1..Un+A_final` 是一个单元；一次已送达 proactive assistant 是一个独立单元。窗口和游标不得落入显式 `control_turn_id` 中间。

0030 已取代按消息数配置的窗口：迁移删除 `memory_window`、`keep_count` 和
`consolidation_min_new_messages` 语义，改由每次完整 provider payload、当前 frozen model
capability 和固定 74% soft watermark 触发；完整逻辑单元仍不可拆分。

### STI-012 连环中断的工具前缀可压缩但不可丢证据

下一 attempt 初始 prompt 中的前驱工具调用/结果由 0030 session Context Gate 处理，不再
进入独立 query-local compactor。Gate 只淘汰已闭合工具组，显式把本 interaction 的全部
U 作为 current anchor，并保留最近原始工具后缀。无合法切点、摘要无效或节省不足时
fail-loud；不得改写或删除权威 turn/message 证据。

## 3. 验收序列

1. U1 创建 interaction I1 / attempt E1，Agent 完成一个工具批次。
2. 用户中止 E1；U2 创建 E2，模型看到 U1 和 E1 的完整工具调用/结果。
3. 用户再次中止 E2；U3 创建 E3，模型看到 U1/U2 和 E1/E2 的完整已闭合工具事实。
4. 只有 A_final 输出后 I1 completed；E1/E2/E3 各有独立 attempt ID。
5. SessionDB 完成 transcript 为 `U1、U2、U3、A_final`，四条消息携带同一 interaction `control_turn_id`。
6. Akasha 只建立一个 I1 节点，输入由 U1/U2/U3 确定性聚合，输出来自 A_final。
7. 下一轮 U4 把 I1 当作一个历史单元；热窗口可见 I1 的全部 U、压缩摘要/最近工具后缀和 A_final，超出热窗口后由 Markdown consolidation 与 Akasha recall 提供派生语境，SessionDB 仍保留权威 transcript。

## 4. 非目标

- 不恢复模型隐藏思维或 provider 私有流状态。
- 不在任意 token delta 中间注入 U。
- 不让 `/stop` 自动变成继续任务的 user message。
- 不修改或删除既有 message 正文。
- 不在客户端复制 active-turn admission 规则。
- 不在本功能中新增通用 tool-result archive/read 协议；单个 logical interaction 仍可能很大，这是已接受风险。
