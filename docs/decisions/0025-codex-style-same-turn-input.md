# 0025 · 普通输入采用 Codex 式同 Turn admission

- 状态：accepted
- 日期：2026-08-06
- 关联条款：SES-007～SES-008、MEM-010、RUN-008、OUT-005
- supersedes：无
- superseded by：无

## 背景

当前 runtime 把同 session 的第二条普通消息等待到上一 turn 完成后再创建新 turn，导致用户在 Agent 工作期间补充要求时必须先硬终止。用户确认的目标是 Codex 模型：U1 启动 turn，U2/U3 自动注入同一个 active turn，最后 A 才结束 turn。

Codex `2b5bdcf67547860f2e5c5a605009a70026796b2b` 的普通 user admission 会先尝试 active-turn steer；pending input 在下一次模型采样前 drain，检测到 pending input 时同一 user-visible turn 继续。显式 `turn/interrupt` 则独立终结 turn。

## 决定

普通输入不暴露 steer/follow-up 选择。core 自动执行：没有 active turn 时创建，有可接收输入的 active regular turn 时追加。输入只在完整 provider response 或 tool batch 边界生效；最终回复候选必须先 seal input source，成功后才提交 completed。

Mobile 中止按钮与各 channel `/stop` 保留为独立 hard interrupt。Akasha 对 completed turn 聚合全部有序 U 和唯一 final A，不再用相邻角色推断新格式。

## 理由

这个模型直接匹配用户操作：发送消息就是补充当前任务，不需要理解 runtime 术语。安全边界避免切开工具效果；turn ID fencing 和 seal 避免迟到输入串到错误 turn。保留 hard interrupt 使失控任务仍可终止，但不会把“补充要求”和“放弃任务”混成一个动作。

## 影响

- 正面影响：连环补充要求保持同一 turn 和同一最终回答；channel 行为一致。
- 兼容性：`turn/start` active 行为、控制能力、completed transcript 和 Akasha projection 发生 breaking 变化。
- 数据和迁移：不改旧正文；新消息在 extra 中携带显式 turn 归属。Akasha builder 对旧数据保留 legacy pair。
- 失败与回滚：输入先写 turn checkpoint 再进入内存；代码可回滚，已追加 message 不删除。

## 验收

- [x] 两次 active-turn 普通输入仍返回同一 turn ID，只有一个 terminal A。
- [x] SessionDB 和 Akasha 都把全部 U 归入同一 completed turn。
- [x] `/stop` 仍只产生 interrupted terminal，不注入 user input。

## 未决问题

- 无。
