# citation 插件

记忆引用追踪协议。在系统 prompt 中注入引用规范，从 LLM 回复里提取引用的记忆 ID，并清理协议标签。

---

## 接入点

| 接入方式 | 阶段 |
|---|---|
| `prompt_render_modules_bottom()` | PromptRender 阶段末尾——注入引用协议文本 |
| `after_reasoning_modules_before_emit()` | AfterReasoning emit 之前——提取 cited ID |
| `after_reasoning_modules_before_persist()` | AfterReasoning persist 之前——清理残留协议标签 |

---

## 运作逻辑

### 1. 注入引用协议（CitationPromptModule）

每轮推理前，在系统 prompt 底部追加一段隐藏指令（`_CITATION_PROTOCOL`），要求 LLM 在用到记忆条目时，在回复末尾输出 `§cited:[id1,id2]§` 格式的引用行，且不向用户暴露这行的存在。

### 2. 提取 cited ID（CitationAfterReasoningModule）

推理完成后，用正则扫描 `reply` 尾部，匹配 `§cited:[...]§` 标签：

- 若匹配成功，提取 ID 列表，写入 `persist:assistant:cited_memory_ids` slot，并把标签从 reply 中剥除。
- 若 reply 里没有引用行，fallback 到工具调用链：扫描 `recall_memory` 工具的返回结果，从 JSON 里取出 `cited_item_ids` 或 `items[].id`，作为本轮引用 ID。

提取到的 ID 由下游持久化模块写入数据库，用于更新记忆条目的被引用计数和时间戳。

### 3. 清理协议标签（ProtocolTagCleanupModule）

在 persist 之前再做一次扫描，用正则清除 reply 末尾所有残留的 `<tag:value>` 形式协议标签（包括其他插件可能留下的），保证对外输出的文本干净。
