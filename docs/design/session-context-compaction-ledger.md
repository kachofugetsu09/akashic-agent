# Session Context Compaction Ledger

- 状态：accepted / implementation
- 日期：2026-08-07
- 决策：[0027](../decisions/0027-session-context-compaction-ledger.md)

## 1. 目标与 owner

Core 在 provider 调用边界拥有完整输入，因此由 Core `ContextCompactor` 读取只读
Session snapshot、当前 tool batch、system/memory sections 和动态工具 schema。SessionStore
只拥有 messages、`session_compactions` 和 `sessions.last_consolidated`；Markdown owner
只拥有 PENDING、consolidation event 和其幂等写入索引；Akasha 只消费 completed transcript。

```text
session snapshot + assembled provider payload
                 │
                 ▼
          Model Call Gate
       ┌─────────┴──────────┐
       │                    │
   < 74%                 >= 74% / hard edge
       │                    │
       ▼                    ▼
 provider.chat       ContextCompactor
                         ├─ temporary turn view
                         └─ committed checkpoint
                              │ summary + tail + provenance
                              ▼
                         next provider.chat
```

## 2. 持久化合同

`session_compactions` 使用 `(session_key, generation)` 主键和唯一 `source_ref`。每行
保存 `parent_generation`、summary format、source seq/message IDs、retained tail 引用、
runtime/model/capacity、threshold、before/after token、summary usage，以及
`invalidated_at/invalidated_reason`。generation 只递增、不复用；`last_consolidated=0`
表示无有效 checkpoint，非零值必须指向同 session 的有效 generation。

checkpoint 提交顺序为：读取 provenance → 生成并校验 summary/Markdown draft → 以
source_ref 幂等提交 PENDING 与 `ConsolidationCommitted` → 在一个 SessionDB 事务中
INSERT generation 并推进 cursor。跨文件阶段失败不推进 cursor；重试复用 source_ref。

messages、tool_chain、embeddings、Akasha 输入和长期 Markdown 状态均不可因压缩更新或
删除。显式 interaction 删除先备份并验证 SessionDB，再失效命中的 checkpoint 和
descendants；session 删除 cascade ledger。

## 3. 模型预算与切点

- 检查完整实际 payload，包含 system prompt、memory、persistent/prompt history、
  multimodal budget、动态 tools 和 provider overhead。
- soft limit = `floor(context_window * 0.74)`。
- hard input limit = `context_window - current request max_output_tokens`。
- 输出上限为 0 时不额外预留；hard limit 不得被 `effective_context_percent` 覆盖。
- 只在完整 tool-call/result batch 闭合后选择切点；保留当前 user anchor 和活动效果。
- raw tail 从后向前累积至少 20k token，跨越完整 logical unit 可以略超 20k；重建后
  仍需同时低于 soft/hard，不能满足则阻断。

## 4. Summary 与 fallback

summary 仅接受以下 Pi 格式标题：Goal、Constraints & Preferences、Progress（Done /
In Progress / Blocked）、Key Decisions、Next Steps、Critical Context。上一 generation
summary 与新淘汰证据共同输入下一次 summary。当前模型失败后使用 main/default fallback；
两者失败、正文空白、包含 tool call 或格式无效时，业务调用返回可区分的阻断错误。

## 5. 主动流程与退役路径

`RECENT_CONTEXT.md` 不再由 workspace 初始化，不再从 prompt block、proactive、Wake
或 Drift 读取。主动流程保留 VEDA、SELF、MEMORY、PROACTIVE_CONTEXT、Akasha recall、
`get_recent_chat` 和自己的送达历史。手动 consolidate、`context_compact` 工具、
`react_compaction` 生产/读取和 cursor mutation API 删除。

## 6. 迁移与恢复

唯一 Yoyo migration 在 workspace lock 下先对 config、sessions.db、RECENT 文件做可读
备份及 `integrity_check`，再创建 ledger、reset legacy cursor、写入新 compaction config
默认值并归档删除 RECENT。任一步失败恢复备份、拒绝 Yoyo success receipt、阻止 runtime
启动；fresh install 直接创建新 schema 且不生成 RECENT。旧 session 不在迁移阶段调用
LLM，首次真实请求 lazy rebuild。
