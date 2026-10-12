# 记忆系统——Markdown 文件层

本文档保留 Markdown 记忆层的当前入口；旧的滑动近期摘要实现已退役，不能作为运行时状态清单。

## 文件

| 路径（`memory/` 下） | 用途 |
|------|------|
| `SELF.md` | Akashic 自我认知（人格与形象 / 对用户的理解 / 关系定义），常驻 |
| `MEMORY.md` | 用户长期记忆（用户事实 / 用户偏好 / 明确要求记住的内容），常驻；三个用户章节合计默认 3000 字配额，其他手写章节不计 |
| `topics/<名>.md` | 全局主题；frontmatter 含 `description`（可选 `pinned`） |
| `topics/<文件夹>/_scope.md` | 文件夹归属：`scope: project=p_…` 与 `description`；文件夹名由模型自取 |
| `topics/<文件夹>/<名>.md` | 该 scope（会话宽键，如项目）专属主题 |

非空的 `PENDING.md` / `PENDING.snapshot.md`、未完成的旧版 draft 会让插件拒绝启动，需先人工处理。
旧版 `MEMORY.md` 可以直接沿用；其中旧的"助手操作上下文"等章节照常注入，后台提炼不再写入它们，可按需手动移到 `topics/`。

## 后台提炼

Output 引用压缩摘要后，follower 读取尚未学习的完整 Turn，把当前 `SELF.md`、`MEMORY.md`、全局主题与本 scope
文件夹全文交给模型，模型只返回操作：`add`、`supersede`、`retract`、`create`、`create_folder`。

- 证据：每个操作引用本次来源的真实 message_id；`MEMORY.md`、`SELF.md` 中对用户与关系的判断及 `retract` 只认用户 Input。
- 新条目为不超过 120 字的单行 `- ` 文本；配额只在内容变长时检查，既有手写内容不阻断提炼。不合格草稿修正一次。
- 只能写全局主题或当前 scope 的文件夹；后台不设置 `pinned`。

```
compaction summary used by Output
  → markdown_ops_v1（操作 draft，按 source_ref 幂等）
  → 文件锁内叠加到当前文件（幂等收敛：已存在跳过、旧条目不在则 noop、同 scope 已有文件夹则重定向）
  → markdown_before_v1（首次叠加前内容）+ markdown_ops_applied_v1
```

叠加作用于锁内读到的当前文件；替换前再比对一次读到的内容，变化则重读重叠加，3 次仍变化就保留 draft 待下次重试。
重放时前提已失效的操作（scope 文件夹不在、supersede 新旧条目都不在、主题文件已删除）跳过，写入 applied receipt 的 `conflicts` 并记日志。
操作 draft 与会话进度同一事务提交；缺少进度的提炼 draft 拒绝应用。崩溃后启动时重放未确认的 draft。

已知限制：原生 `write_file` / `edit_file`、shell 与外部编辑器不参与后台的文件锁。比对与替换之间、以及原生工具自身读写之间仍有
很小的竞争窗口，极少数并发下可能静默丢失一条记忆；为此提示 agent 对已有记忆文件只用 `edit_file`。
消息正文只追加，压缩不 UPDATE/DELETE 既有消息。

## 注入与 agent 写入

system 材料按"会话 + 最新压缩摘要"冻结：同一压缩代内前缀不变，agent 或后台写入在下一次压缩或新会话中可见。内容依次为：
使用规则（记忆目录绝对路径、读写方式、易变信息先核实）→ `SELF.md` → `MEMORY.md` → 当前项目文件夹的 pinned 全文与索引 →
全局主题的 pinned 全文与索引 → 其他项目文件夹清单（不加载，需要时 `rg`）。索引由 frontmatter `description` 自动生成；
pinned 每组 4000 字，超出截断并提示。不属于项目的会话没有项目段。

agent 用原生 `read_file` / `write_file` / `edit_file` / `shell rg` 读写记忆，不经过证据校验，也不与 Akasha 召回争用同一通道。

## 每轮浮现

每条新的用户 Input 首次准备材料时，用 BM25（ASCII 词 + 中文二元组）对当前项目文件夹与全局主题中**未 pinned** 的文件排序，
得分不低于 `surface_min_score`（默认 2.0）的前 `surface_limit`（默认 3，0 关闭）个以可回放 reminder 附在本轮，
标注"N 天前更新"，单文件最多 `surface_chars` 字。同一 Input 的后续工具步骤复用结果；同一压缩代内同一主题只浮现一次。不改变 system 前缀。

## 整理

follower 每次唤醒检查：距上次整理不少于 `consolidate_hours`（默认 24）且其后新应用的提炼 draft 不少于
`consolidate_min_drafts`（默认 5）时，对全局（MEMORY、SELF、全局主题）和这些 draft 涉及的 scope 文件夹各调用一次模型。
整理只能使用 `merge`（把同一文件中 ≥2 条现有条目逐字替换为一条），不新增事实；提炼阶段不能使用 `merge`。
重放时只有全部旧条目仍在才合并；旧条目全不在且合并结果已在视为已应用；其余（部分变化、或旧新都不在）记为冲突跳过。
矛盾条目只在一方带更新日期或明确是更正时合并，不按文件中的先后位置判断。
结果走同一 `markdown_ops_v1` → 幂等叠加 → `markdown_ops_applied_v1` 路径，完成后写 `markdown_consolidated_v1`；
两次不合格的 scope 跳过，模型暂时失败留到下次唤醒。`consolidate: false` 关闭。
