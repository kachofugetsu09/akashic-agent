# Tool Result Artifact Projection

- 状态：implemented
- 日期：2026-08-09
- 决策：[0032](../decisions/0032-tool-results-use-archive-backed-prompt-projection.md)
- 关联条款：CTX-001～CTX-008、SES-001～SES-005、MIG-001、BAK-001

## 1. Owner 与数据流

`DefaultReasoner` 判断主 Agent 工具结果是否达到归档门槛；`SessionStore` 独占正文、读取权限和
审计写入；`ContextCompactor` 只拥有 provider prompt 投影，不持有任意 SQL 或删除权限。

```text
ToolRuntime success
       │ normalize text
       ▼
DefaultReasoner ── >=8192 chars ──► SessionCompactionPort
       │                                  │
       │                                  ▼
       │                         SessionStore INSERT artifact
       │                                  │ committed ref
       ▼                                  ▼
raw messages ◄──────────────── ContextCompactor
       │ raw Gate/compaction               │ provider-only copy
       │                                   ▼
       └────────────────────────► full latest / placeholder old
```

只有 `execution_status == success`、没有 `content_blocks` 且 `len(text) >= 8192` 才归档。写入以
`session_key + call_id` 幂等；相同身份但 turn、tool name 或正文漂移时 fail-loud。占位符形成前
artifact INSERT 必须已经提交。

## 2. 可见性时序

`ContextCompactor` 已拥有 committed units、current anchor、completed batches 和 pending 的明确
分段，因此不从角色邻接猜测“第几轮”。投影集合为：

- 所有 committed units 中已有 artifact 的 tool result；
- `completed_batches[:-1]` 中已有 artifact 的 tool result；
- 减去当前 attempt replay 明确保护的 call IDs。

因此一个新结果的最小时序是：

```text
request N     : model emits tool call A
request N+1   : A result is latest batch, model sees full A
request N+2   : after batch B closes, model sees ref(A) + full B
next user turn: A/B are committed history, archived results use refs
```

若 N+1 直接给出最终回答，没有必要在同一 turn 再请求模型；下一 user turn 才看到 ref。该规则不
依赖“模型已经理解”的不可观察推断，而依赖一次真实 provider 请求是否已把该 batch 作为最新输入。

## 3. 读取与寻址

placeholder 是固定紧凑 JSON：

```json
{"tool_result_ref":"<artifact_id>"}
```

模型通过 always-on `read_tool_result(artifact_id, offset=0, limit=4000)` 恢复正文。Store 在一个
`BEGIN IMMEDIATE` 事务中完成：

1. 校验 reader turn 属于当前 session；
2. 查 artifact，并拒绝跨 session ID；
3. 校验 `0 <= offset <= total_chars` 与 `1 <= limit <= 6000`；
4. 计算字符切片，INSERT read evidence，commit 后返回正文、`next_offset`、`total_chars`、`eof`。

失败事务 rollback，不产生“读过”的假证据。读次数可直接按 artifact 或 session 聚合
`tool_result_reads`；正文不进入 read evidence，避免复制膨胀。

## 4. 缓存、容量与 compaction

raw-equivalent payload 先进入现有 Context Gate。达到 soft/hard 水位时，compaction 仍能读取
完整结果并保留关键事实；Gate 成功后才对发往 provider 的副本遮蔽旧正文。provider usage 不能
反向校准 raw token meter，因为两种视图长度不同。

投影会在结果从“最新 batch”变为“旧 batch”时改变一次历史前缀，这次之后 placeholder 内容
稳定。设计不声称零缓存损失；它用一次前缀失效交换后续每次请求少传大正文。新工具结果仍只在
尾部追加，且不会因动态摘要反复改变旧引用。

## 5. 持久化与减少协议

`tool_result_artifacts` 和 `tool_result_reads` 都只追加。fresh workspace 由 `SessionStore` 创建；
existing workspace 通过 Yoyo migration 在 verified SessionDB backup 后执行 additive DDL，不
回填旧 tool chain。普通运行没有 TTL、容量 GC、UPDATE 或 DELETE。

用户显式 session cascade 时，既有删除 owner 先创建并验证完整 SessionDB backup，再在同一事务
按 reads → artifacts → messages/turns/ledger → session 的顺序删除。非 cascade 检查把两表计入
仍有子状态的判定。代码回滚保留未知新表，恢复使用删除审计返回的 backup path。
