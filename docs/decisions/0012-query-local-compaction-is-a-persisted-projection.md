# 0012 · Query 内压缩是可持久重放的非破坏性投影

- 状态：accepted
- 日期：2026-07-31
- 关联条款：CTX-001～CTX-007、SES-001、SES-005、CAP-001、ERR-001

## 背景

同一个 user query 可以执行数十到数百轮 ReAct。当前实现把每轮 assistant tool call 和 tool result 持续追加到本轮 `messages`，下一次模型请求会重新携带全部前缀；回合完成后，Session 又会从 assistant message 的完整 `tool_chain` 重建这些消息。只压缩当前函数里的临时列表可以让本轮继续，却不能阻止下一次 query 从 SessionDB 再次展开旧前缀。

pi-mono 把 compaction 建模为一等 session entry：原始 session entry 保留，模型上下文只使用最新摘要和未压缩后缀。Akashic 的 SessionDB 按完整被动 Turn 原子提交 user 与 assistant，工具步骤聚合在 assistant message 的 `tool_chain`，不能直接照搬逐条 session entry 的写入时机。

## 决定

core model runtime 拥有当前 query 的压缩计算。它只接收当前 prompt view、工具 schema、模型预算和完整工具批次边界，不获得 SessionManager 或存储写接口。

```text
┌───────────────────────────────┐
│ Current query full tool_chain │  完整执行事实
└───────────────┬───────────────┘
                │ read-only projection
                ▼
┌───────────────────────────────┐
│ Active model view             │  当前 query 可替换
│ compact pair + recent suffix  │
└───────────────┬───────────────┘
                │ turn commit metadata
                ▼
┌───────────────────────────────┐
│ New assistant message         │  单次 INSERT
│ tool_chain + react_compaction │
└───────────────┬───────────────┘
                │ next query replay
                ▼
┌───────────────────────────────┐
│ compact pair + suffix + reply │
└───────────────────────────────┘
```

默认压缩水位是模型 `context_window` 的 `74%`，并且必须低于 runtime 的硬输入边界。每次模型调用前优先使用 provider 上一次返回的准确 input usage 加新增消息估算；没有准确 usage、工具 schema 改变或压缩替换前缀后，重新估算完整 provider input。

压缩只在完整 tool batch 后执行。压缩请求使用同一 provider 和模型、禁用工具，并以有界序列化输入生成结构化摘要。活动模型视图把摘要投影成内部 compact call/result；这对消息不是可执行工具，不进入 `tool_chain`。重复压缩用上一份摘要和新淘汰步骤更新同一个边界。

回合完成时，reasoner 只把压缩元数据交给 AfterReasoning。Session owner 在既有 user + assistant 原子提交中，把 `react_compaction` 写入新 assistant message 的 `extra`；完整 `tool_chain` 仍按既有有界持久化规则保存。Session 重放识别该字段，跳过已压缩工具组并投影摘要与未压缩后缀。既有消息不 UPDATE、不 DELETE。

## 理由

- 当前 query 的 prompt view 与完整执行事实生命周期不同，分开保存可以同时控制模型窗口和保留诊断证据。
- 把 compact pair 仅作为 provider 投影，可以建立清楚的因果边界，又不会污染真实工具权限和执行统计。
- 延续完整 Turn 的原子提交，不会把中途摘要伪装成已经完成的对话历史。
- 按模型窗口比例触发可以服务不同上下文大小；与 `max_output_tokens` 解耦后，关闭输出上限不会关闭上下文保护。

## 影响

- `react_compaction` 成为 assistant message `extra` 的版本化字段，SessionStore 在反序列化边界校验。
- 重放压缩过的 assistant turn 时，模型看到摘要和后缀；用户可见正文、消息身份和完整工具证据保持不变。
- 当前 query 中途崩溃或取消时不单独提交 compaction；既有 Turn 状态继续说明该次执行未完成。
- Session-wide 历史归档、自动删除、摘要展开工具和跨模型重新总结不属于本决定。

## 2026-08-06 补充：中断 Attempt 仍属于当前 Query

同一 logical interaction 经历 `U1/interrupt/U2/interrupt/U3` 时，前驱 attempt 的闭合工具组必须加入当前 QueryCompactor，不能作为永久不可压缩的 base prefix。current-query anchor 显式合并全部 U；interrupt marker 和最近闭合工具后缀保留在热视图。最终 `react_compaction.compacted_tool_groups` 相对于所有 attempt 聚合后的完整 `tool_chain` 计数，SessionDB 和 control checkpoint 仍保持既有只追加/状态机写入。

## 验收

- 低于软水位时 provider payload 与基线等价，不产生 `react_compaction`。
- 达到软水位后只在完整工具批次边界压缩，当前 user query 和最近后缀仍存在。
- 多次压缩后活动 payload 只有一个 compact pair。
- 完整回合提交后，SessionDB 保留完整 `tool_chain` 和版本化 `react_compaction`；write set 没有既有 message UPDATE/DELETE。
- 关闭并重新加载 Session 后，下一次 query 不展开已压缩工具组。
- summary 错误或压缩后仍达到硬输入边界时明确失败；没有可压缩前缀时不切开消息，provider overflow 保留原错误。

## 2026-08-01 紧急勘误：摘要语义无效使用有界退避

摘要请求关闭 thinking。provider 返回成功响应但正文为空、空白或携带工具调用时，按 `2s → 4s → 8s` 最多重试三次并累计 usage；耗尽后仍按原决定明确失败，且不提交候选投影。传输、限流和服务端异常继续由 provider 重试层拥有，本勘误不引入无差别异常重放。
