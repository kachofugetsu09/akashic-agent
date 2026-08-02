# Mobile 长消息投递设计

- 状态：phase 1 implemented
- 日期：2026-08-02
- 决策：[0019](../decisions/0019-mobile-long-messages-use-bounded-events.md)
- 关联条款：MOB-003、MOB-005、MOB-007、SES-001

## 1. 问题和用户意图

模型应该能充分表达长回答，Mobile 不应因为服务端投影复制内部轨迹或把整段正文塞进一个 final 而报告“消息发送失败”。单条 frame 保持有界，同时完整正文仍由 SessionDB 保存并能在客户端精确重建。

## 2. 当前调用链和 owner

```text
┌──────────────────────┐
│ Provider stream      │  thinking/content delta
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Core lifecycle       │  完成后 INSERT 完整 assistant message
└──────┬───────────────┘
       │ stream events                  │ canonical outbound
       ▼                                ▼
┌──────────────────────┐       ┌────────────────────────┐
│ Mobile answer.delta  │       │ Mobile message.final   │
│ 有界、排序、可重放   │       │ 身份/附件/终态/纠正    │
└──────────┬───────────┘       └────────────┬───────────┘
           └──────────────┬─────────────────┘
                          ▼
                 ┌──────────────────┐
                 │ Android projection│  追加 delta，final 提交
                 └──────────────────┘
```

- 权威正文 owner：SessionDB assistant message，只追加，不因投递优化改写。
- 传输与提交 owner：Core Mobile 中立协议，负责 event sequence、durable inbox、ACK/replay 和 final。
- 本地投影 owner：Android Room/WebUI，可以重建，不反向决定权威正文。
- 内部轨迹 owner：SessionDB 与诊断链路；Mobile 工具 UI 只消费独立 `react.tool.*` 投影。

## 3. 已确认事实、推断和未知

- **F：** Mobile JSON frame 上限是 256 KiB；`message.final` 原样复制 outbound metadata。
- **F：** 事故回复正文约 3.8 KiB、thinking 约 10.1 KiB、55 项 `tool_chain` 约 347.7 KiB，final 超限；正文此前已完成并持久化。
- **F：** Android 先追加 `answer.delta`；final 正文为空时保留已经累积的正文。
- **F：** Codex/OpenAI WebSocket 使用 typed delta 和完成事件；OpenAI completed 可以包含完整 Response，Akashic 不能在 256 KiB 应用帧和 durable inbox 下照搬该 payload。
- **C：** 第一阶段只处理可证明为前缀兼容的正文；分歧 final 保留纠正语义。
- **U：** 超大分歧纠正和历史页中的超大单条消息需要第二阶段 range/chunk 数据面。

## 4. 第一阶段协议

1. 每个 provider 正文增量同时记录在 turn 内存状态，并按 UTF-8 字节边界拆成有界 `answer.delta`。
2. canonical final 到达后先刷新待发送 delta。
3. 若已发送正文是 canonical 正文前缀，只发送缺少后缀，随后发送 `content: ""` 的 final。
4. 若尚未发送正文且 canonical 正文超过 delta 预算，把全文合成为若干 delta，随后发送空正文 final。
5. 若两者分歧，final 保留完整 canonical 正文，让客户端现有 finalize 路径覆盖草稿。
6. final metadata 使用显式 allowlist；`tool_chain` 等内部字段不得进入协议。

### 事故前后示例

```text
以前：55 个工具结果 ─┐
      3.8 KiB 正文 ──┼─> 一个约 361 KiB message.final ─> 超限失败
      10.1 KiB 思考 ─┘

现在：55 个工具结果 ───> react.tool.*（独立事件）
      3.8 KiB 正文 ────> answer.delta
      消息身份/附件 ────> 紧凑 message.final ────────> 成功提交

1 MiB 非流式正文 ─────> 约 256 个 4 KiB answer.delta
                       └> 一个紧凑 message.final
```

## 5. 失败、恢复和回滚

- delta 使用既有 durable event sequence；断线后由 inbox 与 ACK 重放，重复事件仍由现有序列语义处理。
- final 发布前失败不伪造完成；SessionDB 权威正文仍在，可由历史恢复。
- 前缀分歧时不追加后缀；小型 final 明确覆盖。大型纠正在第二阶段完成前保持可见失败。
- 回滚代码 commit 即恢复原 final 行为；不需要数据库、workspace 或 Android schema 回滚。

## 6. 第二阶段

引入以 `message_id + byte_offset` 标识的正文 chunk/range 数据面，manifest 携带总 UTF-8 字节数与内容摘要。Android 只持久化已验证连续 offset，重连查询缺口；final 仍是提交标记，不拆成多个伪终态。历史页遇到大消息时返回 manifest，而不是截断正文或抬高 WebSocket 单帧上限。

## 7. 验收

- 大内部元数据、1 MiB Unicode 正文、流式前缀补齐和分歧纠正都有定向测试。
- 每个生成的正文事件小于协议单帧上限，重组结果逐字节等于 canonical 正文。
- final 不包含内部工具轨迹；SessionDB、附件和正式 workspace 无写入变化。
