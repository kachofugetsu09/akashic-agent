# 0052 · Compaction 与 Markdown 记忆是普通插件

- 状态：accepted
- 日期：2026-08-31
- 关联条款：CTX-007、MEM-001～MEM-011、PLG-001～PLG-014、SES-003～SES-005
- supersedes：0030 的 Runtime-owned Markdown/PENDING 后台阶段；0041 的 Markdown 特权通道暂留结论
- superseded by：无

## 背景

Session compaction 已按真实模型上下文窗口触发。一次成功 checkpoint 已经提供完整、稳定、
带 `source_ref` 的长期记忆输入，因此按消息数量周期写入 `PENDING.md`，再由 optimizer 合并
`MEMORY.md` 与 `SELF.md`，形成了没有独立必要性的第二套调度、队列和恢复语义。

现有 Core 还直接构造 Markdown runtime，并在 provider 调用中直接调用 compaction runtime。
这使两个可替换能力获得 bootstrap、Session、Prompt 和关闭流程的专用接线。仓库内置身份也
因此变成了权限来源，与 PLG-003、PLG-006 和 0039 的普通插件边界冲突。

## 决定

Compaction 与 Markdown 记忆各自成为 v3 普通插件。Core 只补充来源无关的原子能力：

1. 完整 provider 请求组装后发布 request prepare，并在 context overflow 后发布一次有界的
   retry decision；事件只携带模型能力、完整 messages/tools、调用身份和可替换结果。
2. Session owner 提供 compaction 窄端口。端口只允许读取脱离投影、写 prepare、提交 immutable
   ledger generation、推进 cursor 和执行既有确定性恢复；不暴露任意 SQL、删除或 Session repository。
3. Core 发布已提交 checkpoint 的 typed fact。事实包含 immutable source plan、scope、generation
   与 `source_ref`，不包含 Markdown 路径或写入策略。

Compaction 插件消费前两项，拥有 74% 软水位、硬输入边界、完整 logical unit、20k raw tail、
六段摘要、fallback、overflow retry 和 checkpoint 提交。没有该普通插件时，Core 不补回私有
compaction；provider 保持自身 context error。

Markdown 记忆插件声明精确 workspace 文件，消费 committed checkpoint，并按同一 session 的
generation 有序更新 `MEMORY.md` 与 `SELF.md`。每种文档以 `source_ref + kind` 独立幂等；写前
备份，写入经过 schema 校验、fsync 和 atomic replace。失败不回滚 Session ledger，不伪装成功，
并保留可观察 receipt。它不提供 Markdown 专用 read Service。需要 Prompt 内容的插件消费
通用 ordered prompt event；确实需要文件的插件像其他普通插件一样声明精确
`workspace_files`，不获得 store、写入其他文件、删除或 workspace root。

`PENDING.md`、周期 optimizer、消息数 trigger 和它们的 Dashboard 操作退役。升级时若旧
`PENDING.md` 非空，迁移先建立恢复备份，再只执行一次受 receipt 保护的直接合并；成功后归档
旧文件，不静默删除未消费内容。历史 `consolidation_writes.db` receipt 保留为审计和旧 saga
恢复证据；新 Markdown draft、before-image 与 applied receipt 由插件声明的独立精确文件
`memory/markdown-profile-writes.db` 管理，不与 compaction 共享写 owner。

插件化后的 compaction receipt 使用 v4，作为新 Markdown profile 投影的明确协议边界。升级前
的 v3 receipt 仍可恢复 Session ledger，但不重新发布给新插件；它们已经属于旧
PENDING/optimizer 管线，重放会重复解释历史来源。v2 receipt 保留其确定性 legacy draft 恢复。

```text
Core atoms                         ordinary plugins
┌──────────────────────────┐       ┌────────────────────┐
│ complete request events  │──────▶│ compaction         │
│ Session compaction port  │       │ project + summarize│
└──────────────────────────┘       └─────────┬──────────┘
                                             │ committed fact
                                             ▼
                                   ┌────────────────────┐
                                   │ markdown-memory    │
                                   │ MEMORY + SELF      │
                                   └─────────┬──────────┘
                                             │ prompt event / exact file grant
                                  Wake / Subagent; existing host inspection
```

## 非特权判定

- Core Prompt 与 compaction 源码不出现插件 ID、`MEMORY.md`、`SELF.md`、`PENDING.md`
  或 Markdown Service key；现有 host inspection 的文档目录不是插件权限来源。
- 两个插件走与外部 v3 插件相同的 loader、generation、candidate isolation、Effect cleanup 和
  disabled builtin 机制；Core 没有按内置来源放宽能力。
- Compaction 端口的 Session 写集合固定；Markdown 插件只获得自己声明的精确文件。
- 这是 Python 进程内架构权限，不声称抵御任意恶意 Python 代码；正式权限由公开 API、静态
  admission、candidate root 和 write-set Gate 共同证明。

## 影响与回滚

- `sessions.db/messages` 继续只追加；compaction 仍只是不可破坏的持久投影。
- included checkpoint 触发 Markdown 更新；post-commit suppressed checkpoint 只推进 ledger。
- 禁用 compaction 插件会失去自动压缩，不触发 Core fallback。禁用 Markdown 插件不影响
  Session ledger，只停止长期 Markdown 投影和它提供的 ordered prompt sections。
- 回滚代码不删除 ledger、prepare、旧 receipt、Markdown applied receipt 或备份。迁移前的
  Git bundle 与一次性 workspace 备份是恢复点。

## 验收

- 两个 builtin 都能被普通 external v3 source shadow，并通过 candidate/stable/hot reload Gate。
- base/candidate 对相同请求的 provider payload、Session rows、checkpoint provenance、摘要、
  included/excluded 分支和错误分类相同；登记的语义变化只有 PENDING/optimizer 退役与 direct write。
- `source_ref` 重放不重复写；同 ref 内容漂移 fail-loud；崩溃点不会丢 Session 消息或覆盖
  MEMORY/SELF 的最近成功版本。
- Wake、Subagent、QQ、Mobile inspection 与 Akasha 逐项通过迁移清单中的行为 oracle。
