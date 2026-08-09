# 0032 · 大型工具结果首次消费后使用 archive-backed prompt projection

- 状态：accepted / implemented
- 日期：2026-08-09
- 关联条款：CTX-001～CTX-008、SES-001～SES-005、MIG-001、BAK-001
- 设计：[Tool result artifact projection](../design/tool-result-artifact-projection.md)

## 背景

长网页、日志和文件读取可能一次产生数万字符。ReAct 的下一次调用需要完整 observation 才能
形成新的 action 或最终回答，但后续调用继续重传同一正文会同时占用上下文、降低注意力密度并
增加费用。直接在结果产生时只给引用会让模型从未见过 observation；下一轮再改写整个历史则会
改变缓存前缀。已有 Session compaction 又只应在容量水位触发，不能替代轮内结果管理。

## 决定

1. 只处理主 Agent 被动 session turn 中成功、纯文本且不少于 8192 字符的工具结果。完整正文在
   下一次 provider 请求前先写入 SessionDB immutable artifact；错误、拒绝、跳过、多模态和
   subagent 输出保持现状。
2. 最新闭合工具 batch 保留原文一次。产生更新的闭合 batch 后，更旧 batch 的已归档结果才在
   provider-only 副本中变成 `{"tool_result_ref":"<artifact_id>"}`；进入 committed history 的
   已归档结果直接使用占位符。未完成 attempt 的 replay 始终保留原文。
3. raw runtime messages、SessionDB messages/tool_chain 与 compaction source 保持完整。
   Context Gate 先按 raw-equivalent payload 估算并决定是否 compact，最后才生成 provider-only
   placeholder 投影。
4. `read_tool_result` 始终可见，仅凭当前 tool execution context 读取同 session artifact。
   `offset` 按 Unicode 字符定位，默认 4000、单次上限 6000；成功返回与 read evidence 在同一
   SQLite 写事务提交。artifact ID 不能作为跨 session capability。
5. artifact/read evidence 不设 TTL 或自动 GC。只有既有用户显式 session cascade 删除协议可在
   verified backup 后删除两表；代码回滚保留数据。

```text
tool result >= 8192 chars
          │
          ▼
┌──────────────────────┐
│ SQLite immutable body│
└──────────┬───────────┘
           │ latest closed batch
           ├──────────────────────► provider sees full body once
           │ older/committed
           ▼
  {"tool_result_ref":"id"}
           │ model needs details
           ▼
   read_tool_result ───────────────► bounded slice + read evidence
```

## 理由

这个时序保留 `observation → next action/output` 的完整第一次消费，同时让已经产生后继推理的旧
observation 退出模型窗口。某个旧结果只发生一次 full-to-placeholder 前缀变化，之后引用稳定；
不承诺 provider 必然命中缓存，但避免每次重写或动态摘要。SQLite 与现有 session 删除备份共用
owner，避免引入文件目录、索引清单和第二套生命周期。

## 影响与回滚

- 新增两张 append-only SessionDB 表和一个 always-on 只读工具。
- provider token 统计使用实际投影输入；compaction 水位仍使用 raw-equivalent 估算。
- migration 只做 verified backup 后的 additive DDL，不回填旧结果。回滚代码不删除新表或正文。

## 验收

- 8192 字符边界、失败/多模态豁免和 archive-before-reference 可从测试观察。
- 第一个大结果在下一次 provider 请求中完整；完成第二个 batch 后，第一个只剩稳定 ref，第二个
  仍完整；原消息对象与 SQLite 正文未改写。
- committed history 使用 ref；attempt replay 保留原文。
- 同 session 分页读取成功并追加一次 evidence；跨 session、越界或不存在均失败且不计数。
- session 非 cascade 删除被 artifact/read 阻断；cascade 前备份同时含正文与读取证据。
