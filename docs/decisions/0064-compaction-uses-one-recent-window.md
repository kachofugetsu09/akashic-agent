# 0064 · Compaction 每代只摘要一个近期窗口

- 状态：accepted
- 日期：2026-09-11
- 关联条款：CTX-001～CTX-007、MEM-011～MEM-012、SES-003～SES-005
- supersedes：0030 中“已有 generation 后追平有效 cursor 到当前全部增量”的选择
- superseded by：无

## 背景

现行插件先从上一摘要覆盖末尾开始取得全部可压缩完整单元，再按摘要模型容量把它们拆成多个
请求。Session 很久没有触发 compaction 时，一次业务调用会串行执行多次摘要；每批又携带上一批
摘要，延迟和失败窗口随积压长度增长。

模型窗口只需要一份可继续工作的近期投影。为追平旧 cursor 而反复解释所有中间消息，并不是
保留权威事实的必要条件；权威 Message 已由 Session 日志完整保存。

## 决定

每次 compaction 先从最新消息向前保留至少 20,000 token 的完整 raw tail。在 tail 之前取得本代
可以退出 Prompt 的完整旧前缀，再从该前缀末尾向前选择摘要模型单次请求可容纳的最大连续窗口。
只用“上一份摘要 + 该近期窗口”生成一次新摘要，不再把更早积压拆成串行批次。

```text
上一摘要覆盖末尾
        │
        ├── 本代明确省略：退出 Prompt，不摘要、不学习
        │
        ├── 单次摘要窗口：从末尾向前取最大完整窗口
        │
        └── raw tail：保留原文
                           │
                           ▼
                新摘要 + raw tail
```

新记录分别保存三项事实：

1. `source_message_ids` 是摘要替代的连续 Prompt 覆盖范围，保持 Context cutoff 和父链可重放。
2. `summary_message_ids` 是本代真正交给摘要模型的消息。
3. `omitted_message_ids` 是本代退出 Prompt、但没有进入摘要模型的消息。

后两项必须无交集，并完整分区本代新增覆盖。Markdown Memory 沿父链只读取各代的
`summary_message_ids`；省略消息不能因为累计覆盖而被误学。旧 version 0/1 记录没有该区分，继续按
原父链累计差值解释；新记录写 version 2，不改写已有记录。

摘要 Prompt 把 `author=user` 消息作为用户目标、要求、偏好和关系的核心证据。assistant 消息只
表示助手的判断、计划和执行记录；助手转述或猜测不能单独建立用户事实，也不能补全用户没有表达
的上下文。

## 理由

- 单次摘要延迟不再随旧积压批次数线性增长。
- “退出 Prompt”与“已摘要、可学习”成为两个可审计事实，不用 cursor 掩盖有损投影。
- Message 日志仍是完整权威事实，模型窗口的有损选择不会取得删除权。
- 最近窗口比最老未处理前缀更接近当前工作，能在固定预算内保留更相关的连续证据。

## 影响与回滚

- 被省略消息不再自动进入后续摘要或 Markdown Memory，但仍可从 Session 历史显式读取。
- 主模型的可恢复失败仍可使用已固定 fallback；fallback 对同一候选前缀重新执行单窗口选择，不恢复串行追平。
- 回滚代码时 version 2 记录不能由旧代码读取，因此回滚前应先恢复升级前的代码与 owner-state
  备份；不得删除 version 2 记录或原始 Message 来伪造兼容。
- Compaction、Context 和 Markdown 仍是原有普通插件 owner，不增加 Core 特权或第二套 cursor。

## 验收

- 六个以上可压缩批次积压时，每代只产生一个成功摘要模型调用，且输入来自最近完整窗口。
- version 2 记录的新增覆盖被 `summary_message_ids` 与 `omitted_message_ids` 精确分区。
- 业务 Prompt 使用新摘要与 raw tail，省略消息不出现；Session Message 快照逐字不变。
- child 被直接使用时，Markdown 学习未应用祖先的真实摘要输入，但不读取任一代省略消息。
- 旧 version 0/1 摘要仍可读取、解析父链，并可成为 version 2 子代。
