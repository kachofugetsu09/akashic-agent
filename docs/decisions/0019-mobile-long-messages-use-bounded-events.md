# 0019 · Mobile 长消息使用有界正文事件和紧凑终态

- 状态：accepted
- 日期：2026-08-02
- 关联条款：MOB-001、MOB-003、MOB-005、MOB-007、SES-001

## 背景

Mobile 协议把单条 JSON frame 限制为 256 KiB，但被动回复的 `message.final` 曾复制完整 `tool_chain` 和最终正文。一次包含 55 个工具事件的回复产生约 347.7 KiB 内部轨迹，正文只有约 3.8 KiB，结果仍因 final 超限而投递失败；SessionDB 已经保存了正确回复，手机只看到通用失败文案。

WebSocket 可以把一个消息拆成多个传输 frame，但库会在应用收到 JSON 前重新组装它。重组后的 `message.final` 仍然超过协议上限，且传输分片不提供正文 offset、幂等、重放或提交语义。

## 决定

1. Core Mobile 协议用既有 `answer.delta` 承载可追加正文，按 UTF-8 字节边界限制单个事件；`message.final` 只提交未被增量表达的纠正正文、附件、稳定消息身份和显式 Mobile 元数据。
2. 已发送正文是权威终稿的前缀时，Core 只补发缺少的后缀，final 不再重复正文。没有流式增量但正文超过事件预算时，Core 先合成有界 delta，再发送空正文 final。
3. 增量与权威终稿不满足前缀关系时，第一阶段保留完整 final 作为显式纠正。若纠正本身超过单帧上限则 fail-loud，交由第二阶段 range/chunk 恢复协议处理，不能把两份正文错误拼接。
4. `tool_chain`、`tools_used`、provider retry 等内部字段继续保存在 SessionDB 和诊断链路，不投影到 Mobile final。final 元数据只允许协议明确消费的字段。
5. 不提高 256 KiB 单帧上限，不修改 SessionDB 正文，不要求第一阶段同步修改 Android schema。

## 理由

正文增长和消息提交是两个不同动作。有界正文事件可以复用现有 event sequence、durable inbox、ACK 和 replay；紧凑 final 保留原子终态。OpenAI Responses WebSocket 同样用 typed [`response.output_text.delta`](https://developers.openai.com/api/reference/resources/responses/websocket-events#response.output_text.delta) 传递新增文本，再用 [`response.completed`](https://developers.openai.com/api/reference/resources/responses/websocket-events#response.completed) 结束响应；其 completed 事件可以包含完整 Response 对象，而 Akashic 因 256 KiB 应用帧与 durable inbox 预算选择紧凑终态。Codex 的 WebSocket consumer 逐事件解析 delta，并把 `response.completed` 作为正常结束条件。

把应用层片段命名成 `message.final.part` 也能实现传输，但它必须重新定义 part index、总长、摘要、幂等、ACK、重放和客户端组装，此时已经不是一个原子的 final。将正文作为 chunk、将 final 保持为提交标记，状态机更清楚且兼容现有客户端。

## 影响

- 旧客户端继续把 `answer.delta` 追加到当前消息；空正文 final 使用已累积正文完成提交。
- 非流式长回复也会在 final 前收到若干 `answer.delta`，用户不再因为正文可分片而看到通用失败。
- 内部工具轨迹不会占用 Mobile final 带宽；工具展示继续来自独立 `react.tool.*` 事件。
- 第二阶段增加内容摘要、offset/range 恢复与历史大消息数据面，任务记录在 `NOW.md`。

## 验收

- 超过 256 KiB 的内部 `tool_chain` 不进入 final，显式 Mobile 元数据仍保留。
- 至少 1 MiB、包含多字节 Unicode 的非流式正文能由多个小于单帧上限的 delta 精确还原，final 正文为空。
- 已流式前缀只补缺失后缀；分歧流仍用 final 明确纠正。
- SessionDB 的完整正文、完整工具轨迹和既有消息行不被更新或删除。
