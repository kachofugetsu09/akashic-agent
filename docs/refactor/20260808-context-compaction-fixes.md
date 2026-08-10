# Context Compaction Stack 修复任务（PR #338/#339/#340）

- 日期：2026-08-08
- 类型：review 结论 → 已批准修复任务
- 目标分支：`codex/context-compaction-03-legacy-removal`（stack 3/3）
- 权威语义：[`0030`](../decisions/0030-session-context-compaction-ledger.md)、
  [`session-context-compaction-ledger`](../design/session-context-compaction-ledger.md)

## 目标与完成标准

修复 session context compaction stack 的调用前 Gate、首次窗口化、异步 Markdown saga、
遗留死代码和交付声明，使模型窗口只由当前模型 capability 决定，同时保持 SessionDB 原始
消息与既有 ledger/receipt 不变。

完成时必须同时满足：F1～F6 落地；targeted tests、三套 Pyright 和累计 Public Gate 通过；
PR 描述与真实验证层一致；改动拆成可独立阅读的小 commit。

## 已批准边界

- subagent 使用内存态 Pi 风格 Gate，不持久化；所有四个 `provider.chat` 入口统一检查。
- 软水位为 `floor(context_window * 0.74)`；raw tail 约 20k token，可因完整 logical unit
  略宽。无合法切点或重建后仍越界时 fail-loud。
- session 业务调用与 subagent 必 gate；`plugins/jobs.py`、`policies/history_route.py`、
  `tools/vision.py` 豁免，超窗自然失败。
- Markdown 生成与写入是 Runtime 拥有的后台任务，不复用 scheduler。ledger 提交不等待它；
  失败不回滚、不重试、启动不补跑。
- Runtime 对后台任务持强引用，同一 session 按 generation 串行；done callback 必须消费异常，
  并记录 session、source_ref、generation。后续 generation 不因前一次失败而停止。
- 优雅关闭采用维护者选择的 A：取消全部未完成 Markdown task，并等待取消收束；不等待 LLM
  自然完成。进程崩溃与取消都不补跑 Markdown。
- `semantic_delta` 为 `compatible`；旧 config key 的 fail-loud 防御保留，升级兼容只由 Yoyo
  迁移承担。
- 不提供手动 compact/consolidation API，不恢复 `RECENT_CONTEXT.md` 或内部 query compactor。

## F1 · subagent 内存态 token Gate

修改 `agent/subagent.py` 及直接测试：

1. 主循环、incomplete summary、forced final summary、mandatory exit 四个 provider 调用统一
   经过一个内存态 Gate helper；各入口保留原 tools 语义。
2. Gate 用 subagent 冻结模型的 `context_window` 估算完整 messages、tools 和协议开销。
3. 达到 74% 时按完整 logical unit 选择旧前缀，以六段格式折叠为内存摘要，保留约 20k raw
   tail；摘要调用有自己的硬输入边界、关闭 tools/thinking，并且不递归经过业务 Gate。
4. `ContextCompactionError` 不得被 summary fallback 捕获为普通 fallback 文本；其他 provider
   失败维持当前明确的降级语义。
5. 删除 `_trim_tool_results`，不产生 DB、Markdown 或其他持久化写入。

验收至少证明四个入口均经过 Gate、超窗先 compact、无合法切点 fail-loud、没有持久写入，
并且 `git grep _trim_tool_results` 无生产或测试引用。

## F2 · receipt v3 与异步 Markdown

修改 `session/compaction_runtime.py`、`core/memory/markdown.py`、必要的调用 owner 与直接测试。

### v3 提交顺序

```text
prepare
   │
   ▼
immutable receipt v3
   │  canonical checkpoint/source plan/digest；不预生成 Markdown draft
   ▼
ledger INSERT + cursor advance + clear prepare（同一 SessionDB 事务）
   │
   ├──► 主业务路径返回
   └──► per-session ordered background Markdown task
```

included checkpoint 的主路径只执行 summary 与 ledger saga。v3 receipt 保存重建 Markdown
输入所需的 canonical source plan、checkpoint、runtime/model 和 digest，但不要求在 ledger 前
调用 Markdown LLM。excluded checkpoint 继续 ledger-only，不创建 Markdown receipt/task。

### 恢复矩阵

| 状态 | 恢复动作 |
|---|---|
| prepare，无 receipt | 证明仍在 pre-effect window，清除 orphan prepare |
| v3 receipt + prepare | 校验 source plan/digest/incarnation 后完成 ledger/cursor/clear prepare；不生成 Markdown |
| v3 receipt，无 prepare | 完整校验 schema/digest/incarnation/source snapshot 后视为已提交 ledger 的审计状态；不补跑 |
| v2 receipt + prepare | 保留旧版确定性恢复：按 receipt draft 幂等完成 Markdown，再提交 ledger |
| schema、digest、source plan 或 incarnation 损坏/不一致 | fail-loud |

后台任务失败必须可观察，但不能伪装成 ledger 失败；receipt 保留为审计和人工恢复凭据，不增加
自动扫描或重试 owner。

## F3 · generation 0 首次窗口化

修改 compaction selection/projection 与直接测试：

1. ledger 尚无 generation 时，在最终 provider payload、source plan 和 summary 输入形成前，
   从最新历史向前按完整 logical unit 回扫至约 74% 窗口。
2. 选中的窗口仍需为约 20k raw tail 和 summary provider 硬输入边界留出空间；完整 unit 可跨过
   目标。如果不存在合法且可执行的窗口，明确 fail-loud。
3. 窗口外更早 message IDs 不进入首次 provider payload、receipt source plan 或摘要；
   `sessions.db/messages` 原始行保持完整，只追加。
4. 首次 ledger 的 `source_from_seq > 0`；已有 generation 后继续按 cursor 到当前做增量。

测试至少覆盖 200k token 历史、旧 ID 不进入 provider payload/receipt、首次成功、第二次增量，
以及 SessionDB 行数和正文逐项不变。

## F4 · 契约与描述

- 修订 `projectneed.md` CTX-007、0030 和 session design：明确业务/subagent Gate 范围、豁免
  清单、generation 0 窗口化和 v2/v3 saga 恢复矩阵。
- PR #339/#340 的“所有 LLM 调用”改为 session 业务调用与 subagent 范围。
- PR #340 `change_intent.semantic_delta` 改为 `compatible`，并说明 Yoyo 是唯一升级兼容 owner。

## F5 · 死代码清理

删除已确认无生产 caller 的代码，并同步直接测试与当前契约性文档：

1. `agent/prompting/budget.py` 及 `agent/prompting/__init__.py` 中的导出。
2. `plugins/default_memory/engine.py` 的 `_keep_count`。
3. `agent/subagent.py` 的 `_trim_tool_results`。

`project-workbook-and-semantic-safety.md` 的当前契约表应标注 `ContextTrimPlan` 已退役；历史事故
叙述可以保留原名并明确它是历史实现，不做全仓机械删除。

“零残留”指生产 runtime 不再读写 `RECENT_CONTEXT`、MemoryWindow、旧 cursor/query compactor；
迁移和迁移测试中为识别或拒绝旧状态而保留的名称不属于生产兼容壳。

## F6 · CI 声明

- #338 记录完整远端 CI 的真实结果。
- #339/#340 只声明各自实际出现的 stacked checks，不把部分检查写成完整 main CI。
- #340 记录累计 Public Gate 的 `sourceDigest`、`planDigest` 和报告路径；重新执行后以新报告
  替换旧证据。

## 受保护状态与允许副作用

- `sessions.db/messages` 正文只追加；compact 不得 UPDATE/DELETE 既有消息。
- 既有 `session_compactions`、prepare、v2/v3 receipt 和 migration backup 不删除、不改写。
- 允许新增 ledger generation、推进 session-local cursor、按协议清除 matching prepare、追加 v3
  receipt，以及由后台任务追加 PENDING/history/event。
- 窗口外历史只从临时 provider projection 排除，权威 SessionDB 事实保持可恢复。
- 不修改正式 Akashic workspace，不运行破坏性迁移，不触碰 main 工作区的其他 dirty 文件。

## 验证与回滚

- 每项运行 targeted tests；随后运行 production/tests/SDK 三套 Pyright。
- 累计执行 `python docker/debug/gate.py run --base origin/main` 并保存 digest/report。
- 只读 Review 逐项核对调用入口、receipt lineage、write set、取消语义、SessionDB 不变和死代码。
- 代码恢复点：`backup/context-compaction-fixes-pre-impl-20260808`。
- 用户文档副本：`/mnt/data/coding/akasic-agent-backups/context-compaction-fixes-pre-impl-20260808/`。
- 若 v3 实现需要回滚，解析严格按 receipt `version` 分流；不得猜测格式或删除已写 receipt。

## 2026-08-08 复查记录（修复推送后）

修复 commit 直接推送到 PR #340（head 分支 codex/context-compaction-03-legacy-removal）：
cbde35a7(docs合同) → 55335a75(异步Markdown+窗口化) → 78f99c2d(subagent gate) → a11e9ddb(死代码)
→ 53bbadf0(测试对账) → da941709(receipt v3 加固)。六项修复全部核实：

1. subagent gate：复用 ContextCompactor（内存态，generation 0 temporary，不写 ledger）；
   四个入口统一走 `_provider_chat`→gate.prepare；`prepare` 在 compact 成功后以
   `messages[:] = rebuilt` 就地替换（context_compaction.py:624），投影对后续请求生效；
   `_trim_tool_results` 删除。
2. 异步 Markdown：commit_checkpoint 同步 prepare→v3 receipt→ledger（清 prepare）→
   `_schedule_markdown`；task 同 session 按 generation 链式有序；从持久化 receipt 重建
   draft；shutdown 取消+等待（bootstrap/tools.py 挂载 compaction.shutdown）。
3. receipt v3：新增 source_mutation_digest/scope；恢复语义——prepare 缺失=ledger 已提交
   →return None（乐观不补跑）；prepare 存在→确定性重放 ledger 不生成 Markdown；
   v2 旧格式兼容重放。
4. 窗口化：`window_initial_context_units` 从后向前累计到 floor(74%) 取完整逻辑单元；
   仅 generation-0（active 空且 next==1）触发，就地替换 initial_messages。
5. 死代码：budget.py + __init__ 导出 + _keep_count + _trim_tool_results（含旧测试）全删。
6. 文档：projectneed CTX-007 写明 subagent 四入口/内存态/豁免清单（jobs、history route、
   视觉短调用超窗暴露既有错误或 fail-open）；PR #340 semantic_delta=compatible。

本地验证：test_context_compaction_contract + test_tool_loop_guard 44 passed；
test_session_compaction_runtime + config_contract 52 passed。subagent 投影经深拷贝快照
验证（call 3 收到 7 条投影，含 SUMMARY 块 + 单个 BIG 结果）。

## 复查发现的讨论点

- 窗口化截断实际语义：generation-0 的 session **每个 turn** 都把 payload 截到 74% 窗口
  （即使未发生 compact）；窗口外历史不进 ledger、不进摘要；存量 MEMORY.md 在旧架构下
  已持续 consolidate，所以大部分早期事实由旧 MEMORY 承载；但"升级后新产生且未 compact
  的超窗历史"既不可见也无记忆替代（窗口化不是 compact，不触发 MEMORY 更新）。
- generation-0 + 极长中断 attempt 超过窗口时，replay 定位错位 → fail-loud raise。
- subagent 收尾 summary 调用（tools=[]）也走 gate：超窗会先 compact 再收尾（多一次摘要，
  无死循环）。
- test_subagent_compacts_in_memory_before_the_next_provider_call 断言基于 kwargs 列表
  引用（就地替换使最终状态=每次调用状态），语义正确但断言弱。
