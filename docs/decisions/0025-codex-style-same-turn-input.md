# 0025 · 普通输入采用 Codex 式 Logical Interaction continuity

- 状态：accepted
- 日期：2026-08-06
- 关联条款：SES-007～SES-008、MEM-010、RUN-008、OUT-005
- supersedes：无
- superseded by：上下文压缩部分由 [0030](0030-session-context-compaction-ledger.md) 取代

## 背景

当前 runtime 需要在用户中断后保留尚未完成任务的连续性。用户确认的目标是：U1 启动 attempt，用户中断后再发送 U2/U3；每条新 U 创建新 attempt，但都属于同一个 logical interaction，直到唯一 A_final 才结束。

Codex `2b5bdcf67547860f2e5c5a605009a70026796b2b` 的普通 user admission 会先尝试 active-turn steer；pending input 在下一次模型采样前 drain，检测到 pending input 时同一 user-visible turn 继续。显式 `turn/interrupt` 则独立终结 turn。

## 决定

普通输入不暴露 steer/follow-up 选择。core 在没有 active attempt 时创建；active 时 `turn/start` 明确返回 busy，唯一可用动作是精确中断。中断 terminal 后的下一条 U 沿 durable predecessor 创建新 attempt。最终回复候选必须先 seal input source，成功后才提交 completed interaction。

Mobile 输入区在 active 时只显示 hard interrupt，草稿保留但不能发送；各 channel `/stop` 使用相同 hard interrupt。控制协议不再提供 `turn/steer`。Akasha 对 completed turn 聚合全部有序 U 和唯一 final A，不再用相邻角色推断新格式。

## 2026-08-06 勘误

维护者以真实 Mobile 场景重新确认后，以上 active-with-draft steer 交互不再成立。当前接受语义如下：

- Mobile active 时无论是否存在草稿，尾部动作都只能是中止；草稿保留但不能发送。
- hard interrupt 只终结当前 execution attempt，不终结 logical interaction。
- 中止收束后发送的下一条 U 创建新 attempt，沿用同一 interaction identity，并重放此前全部 U 和已经闭合的工具调用/结果。
- `U1 → stop → U2 → stop → U3 → A_final` 是一个 completed logical interaction、三个 execution attempts；只有 `A_final` 关闭 interaction。
- SessionDB 最终只追加 `U1、U2、U3、A_final` 的 canonical transcript。Akasha 在线与离线都只从该 completed transcript 建立一个样本；attempt checkpoint 不进入学习。
- Session compaction、Markdown consolidation 和 prompt history 都把这四条消息视为一个不可拆分逻辑历史单元；proactive assistant 单独占一个单元。
- 当前 attempt 的闭合工具组可以进入临时 compaction；提交后的整个 interaction 不再拆分。摘要输入显式包含 U1/U2/U3，最近工具后缀保留原文，权威工具证据不因压缩被改写。

## 理由

这个模型直接匹配 Mobile 操作：执行中只有中止，终止后发送就是补充尚未完成的任务。attempt identity 和 terminal fencing 避免迟到输入串到错误执行；闭合工具组压缩控制模型热上下文，同时让 durable ledger 保持可审计。

## 影响

- 正面影响：连环中止后补充要求仍保留同一 interaction、完整工具事实和同一最终回答；Mobile 没有 send/stop 模式选择。
- 兼容性：`turn/start` active 行为、移除 `turn/steer`、completed transcript、history budget 和 Akasha projection 发生 breaking 变化。
- 数据和迁移：不改旧正文；新消息在 extra 中携带显式 turn 归属。Akasha builder 对旧数据保留 legacy pair。
- 失败与回滚：输入先写 turn checkpoint 再进入内存；代码可回滚，已追加 message 不删除。

## 验收

- [x] 两次中止产生三个 attempt ID，但沿用同一 interaction ID，只有一个 terminal A。
- [x] SessionDB 和 Akasha 都把全部 U 归入同一 completed turn。
- [x] `/stop` 仍只产生 interrupted terminal，不注入 user input。
- [x] Mobile active 时只显示中止；草稿不能把动作切换回发送。
- [x] 同一 interaction 的前驱工具组可进入临时 compaction，摘要锚定全部 U。
- [x] Session compaction、Markdown consolidation 与 proactive 使用一致的逻辑历史单元。

## 接受风险

一个 logical interaction 可以包含很多 U 和工具结果，因此 SessionDB/control ledger 会按既有只追加和结果边界持续增长。当前接受这一风险：活动 attempt 只压缩已闭合的旧工具组，提交后整个 interaction 不可拆分；若单个 query anchor、completed interaction 或最近不可拆工具后缀本身超过 provider 边界，则明确失败。本决定不引入 Maka 式通用 ArchiveRead 协议，也不自动删除权威证据。

## 未决问题

- 无。
