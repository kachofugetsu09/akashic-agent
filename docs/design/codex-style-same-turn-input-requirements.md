# Codex 式同 Turn 输入需求合同

- 状态：accepted / implemented
- 日期：2026-08-06
- 决策：[0025](../decisions/0025-codex-style-same-turn-input.md)
- 设计：[Codex 式同 Turn 输入设计与任务合同](codex-style-same-turn-input.md)
- 关联条款：CTX-003、SES-001～SES-002、SES-005、SES-007～SES-008、MEM-009～MEM-010、RUN-002～RUN-003、RUN-007～RUN-008、OUT-001、OUT-005

## 1. 用户可见目标

普通用户输入在 session 没有 active turn 时创建新 turn；active regular turn 存在时自动注入该 turn。用户不选择 `steer`、`follow-up` 或 `next prompt`，也不需要发送特殊命令。

一次 turn 可以按顺序接收 `U1、U2、U3`，中间执行模型采样和完整工具批次，最终只在 Agent 真正停止时提交 `A_final` 并结束 turn。Mobile 输入区只有一个自适应尾部动作：active 且没有草稿时是中止，存在文字或附件草稿时是发送；Telegram、QQ 等 channel 的 `/stop` 仍是独立硬终止，不属于普通输入注入。

## 2. 需求

### STI-001 普通输入自动选择 active turn

同 session 没有 active turn 时，普通 U 创建 turn；存在可接收输入的 active regular turn 时，普通 U 自动追加到该 turn。core 根据 session lane 和 active turn 状态作出选择，客户端不暴露模式选择器。

### STI-002 同一 turn 支持多个 U

一个 turn 的输入是有序非空集合，而不是单条 user message。每个 U 拥有稳定 item ID、ordinal、原始正文、附件引用和客户端身份；任何 consumer 不得用相邻角色推断 turn 归属。

### STI-003 新 U 在安全边界生效

active turn 中的新 U 先持久进入 turn-local pending input，再进入模型上下文。runtime 只能在一次 provider response 已结束或完整 tool batch 已闭合后 drain；不得切开 tool call/result，也不得粗暴取消已经发生的外部效果。

### STI-004 最终 A 才结束 turn

当前采样得到无 tool call 的回复时，runtime 必须先原子检查 pending input。仍有 U 时，该回复只是本 turn 的中间模型输出，新增 U 进入下一次采样；只有 pending input 为空并成功 seal 后，回复才成为 `A_final`，turn 才能 completed。

### STI-005 控制命令精确栅栏

显式程序化输入携带 `expected_turn_id`；不匹配、已封口、非 regular 或已终态时明确拒绝。普通 channel 消息使用同一个 core admission owner 自动解析 active turn，不能在 adapter 中各自猜测。

### STI-006 In-flight checkpoint 可恢复

初始 U、追加 U、工具 started/completed 和 turn 状态实时写入 `turns` 的稳定 item 记录。进程内 pending queue 可以从这些事实核对；SessionDB `messages` 仍只在 completed turn 时提交完整 transcript batch。

### STI-007 Completed transcript 是多个 U 加一个最终 A

completed turn 在一个 SessionDB 事务中按 ordinal 追加全部 U，随后追加唯一 `A_final`。每条消息携带相同 `control_turn_id`；user 携带 `turn_input_ordinal`，assistant 携带 terminal 标志和输入数量。历史窗口或 consolidation 起点若落在该 turn 中间，replay 必须退回 U1，允许实际投影量超过消息数预算。现有消息正文保持只追加。

### STI-008 硬终止保持独立

Mobile 空草稿时显示的中止动作和 `/stop` 只调用 `turn/interrupt`，把 active turn 终结为 interrupted。它们不注入 user message，不伪装成 steer，也不自动启动下一 turn。Mobile 存在草稿时同一位置切换为发送，普通 U 到达才执行 STI-001；两个动作不得同时出现。

### STI-009 Akasha 使用显式多输入投影

Akasha 对一个 completed turn 建立一个学习样本：有序聚合全部 U，输出为 `A_final`。在线与离线 builder 从 `control_turn_id`、ordinal 和 terminal 标志重建，禁止扫描相邻 `user → assistant` 作为新格式的归属协议。旧消息继续走明确的 legacy pair 兼容路径。

### STI-010 失败和竞态不得丢输入或串 turn

输入 admission、final seal、hard interrupt 和 turn completion 共用同一个 session/turn owner。seal 后到达的普通 U 不得塞回旧 turn；它等待旧 turn terminal 后创建新 turn。重复或过期控制请求不得影响当前 turn。

## 3. 验收序列

1. U1 创建 T1，Agent 完成一个工具批次。
2. U2 在 T1 active 时到达并自动进入 T1。
3. Agent 根据 U2 继续；U3 再次进入 T1。
4. 只有最后 A 输出后 T1 completed。
5. SessionDB 的完成 transcript 顺序为 `U1、U2、U3、A_final`，四条消息携带同一 `control_turn_id`。
6. Akasha 只建立一个 T1 节点，输入文本和向量由 U1/U2/U3 确定性聚合，输出来自 A_final。

## 4. 非目标

- 不恢复模型隐藏思维或 provider 私有流状态。
- 不在任意 token delta 中间注入 U。
- 不让 `/stop` 自动变成继续任务的 user message。
- 不修改或删除既有 message 正文。
- 不在客户端复制 active-turn admission 规则。
