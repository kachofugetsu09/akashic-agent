# 记忆系统——Markdown 文件层

本文档保留 Markdown 记忆层的当前入口；旧的滑动近期摘要实现已退役，不能作为运行时状态清单。

## 文件

| 文件 | 用途 |
|------|------|
| `MEMORY.md` | 长期用户档案与稳定事实 |
| `SELF.md` | Akashic 自我认知 |
| `PENDING.md` | consolidation 提取出的待归档候选 |
| `HISTORY.md` | 由 `ConsolidationCommitted` 订阅者追加的事件日志 |

## Consolidation

被动请求在调用模型前读取 session compaction ledger。超过模型 context window 水位时，compactor 从 `last_consolidate` cursor 之后选择完整逻辑单元，生成版本化摘要并提交 ledger；提交成功后 Markdown saga 才追加 `HISTORY.md`/`PENDING.md`，随后推进 cursor。

```
provider payload gate
  → session_compactions (summary + cursor + retained tail)
  → MarkdownMemoryMaintenance.commit_compaction_markdown()
  → ConsolidationCommitted
```

所有写入按 `source_ref` 幂等。消息正文只追加，压缩不 UPDATE/DELETE 既有消息。

旧版本的滑动摘要文件、按消息数裁切和后台 turn 刷新均不属于当前设计；迁移脚本负责备份并清理遗留文件。
