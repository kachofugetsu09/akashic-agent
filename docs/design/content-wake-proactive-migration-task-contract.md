# Content / Wake / Proactive 分层任务合同

- 状态：implementation in progress
- 日期：2026-08-23
- 目标分支：`origin/main`
- 设计基线：`9586a931fda5d1266d0449e44bcd569d9103d6fa`
- 总 owner：当前 Codex 主任务
- 恢复点：`/mnt/data/coding/backups/akasic-agent-content-wake-design-20260823/origin-main-e89f75dd.bundle`
- 关联设计：[Content / Wake 现有原子能力与第一阶段](content-wake-existing-atoms-first-stage.md)
- 关联决策：[0039](../decisions/0039-react-core-atoms-keep-sources-unprivileged.md)、[0040](../decisions/0040-wake-duty-gate-lives-in-scoped-react.md)

## 1. 最终结果

旧 `proactive_v2`、default proactive island、Wake 私有 proactive loop 和 proactive MCP 聚合桥全部退出运行代码。Wake、Drift、Content 与来源插件变成普通 v3 插件，只组合同一套 Core 原子：

```text
Source plugin ── TIMERS + fetch ──▶ Content.submit
                                           │
                                           ▼
Wake plugin ── TIMERS + lifecycle ──▶ SCOPED_TURNS ──▶ react
                                           │                │
Drift plugin ───────── narrow proposal ────┘                ▼
                                                    durable delivery
                                                           │
                                                           ▼
Source plugin ◀──────── Content unsettled + ACK ───── provider receipt
```

Core 不认识 Fitbit、Feed、Calendar、Steam、GitHub Watch、Content、Wake 或 Drift。每个来源只拥有自己的外部协议、cursor、Timer 和 ACK；Content 只拥有邮箱；Wake 只拥有 admission 与 Content→Drift 串行 duty 选择；React、Turn、Session 与 delivery 保持通用。

### 2026-08-25 · 大重构前行为等价修订

首次正式候选暴露出一个测试盲区：普通 `final_response` 被误当成主动正文，导致“过滤、不推送”的内部判断进入目标 Session。修复不能只拦一类字符串，激活前必须同时保持大重构前的用户体验和记录语义：

```text
new Content revisions
        │
        ▼
legacy hazard admission ── reject ──▶ zero Turn / zero delivery
        │ accept
        ▼
freeze ≤100 candidates in Content-owned selection ledger
        │
        ▼
target Session scoped react ── share(1..5) ──▶ one durable delivery
        │                          │
        └─ skip ──▶ release all    └─▶ one proactive Session projection + cited ACK
```

- Wake 只能从 durable `share_content(message, items)` 取得用户正文；普通 `final_response` 永不投递。
- Content 的固定分数池只在新条目到达时检查 threshold；一次接受只创建一个批次 Turn，最多看 100 个候选并聚合 1～5 个，不能按单条快速排空。
- 新条目按稳定 identity 只计算一次初始分并逐项记为已检查；同一 snapshot 中尚未到期的条目继续保留未来 deadline，不能被 watermark 顺带吃掉。
- 候选语义兴趣必须同时影响 admission 与冻结页排序；prototype 只来自最近 256 个完整非 proactive Turn，主动推送不参与。
- v1 已经 `ready_for_delivery` 的单条 selection 迁移时标记为 `legacy_single`，只允许其既有 `share_content(message)` Turn 完成一次 provider/Session/settlement 链；新 Turn 仍严格要求 1～5 个 candidate id。
- skip、低分未准入和未引用候选都保持用户侧静默；skip 不 ACK，候选仍 pending。
- Drift 使用同一 typed share/skip 和 durable delivery 链；旧 schema 中已丢失正文且无法证明 provider effect 的 `ready_for_delivery` orphan 只逻辑 invalidated，禁止猜测重发。
- 配置 target 后 Wake 读取目标 Session 历史与记忆，但 `memory_write=false`，临时 input/reasoning 不写 messages；仅 provider delivered 的 proactive assistant 追加到目标 Session，并携带 `message_push`、evidence 与 source refs 投影。
- fixture 必须覆盖 20 条未回复 proactive 后的 `u → a`：Session 保留全部 22 条消息，Akasha 只把 `u → a` 形成一个普通 interaction，20 条 proactive 不被并入。

## 2. Change intent

```yaml
change_type: migration
semantic_delta: compatible_then_breaking_cleanup
capability_owner: mixed
consumer_scope:
  - core generic delivery and turn atoms
  - built-in content wake drift plugins
  - installed proactive information source plugins
runtime_patch: required_only_for_proven_generic_delivery_settlement
runtime_patch_reason: provider delivered and Session projection are not currently one recoverable logical delivery
authoritative_state_owner: Content owns inbox; each source owns upstream cursor and ACK; Core delivery owns provider receipt and projection settlement
client_only_alternative: not_applicable
concept_gate: required_per_pr_and_cumulative
invariants:
  - one fact has one owner
  - source submit commits before cursor advances
  - hints are lossy and never recovery truth
  - Wake reuses ordinary scoped react and existing lifecycle
  - Session messages remain append-only
  - delivered content is not redelivered merely because ACK failed
protected_state:
  - existing Session message bodies and ordering
  - existing plugin data until an explicit migration reads and supersedes it
  - provider credentials and formal hua-home workspace
  - scheduler and subagent v3 behavior
allowed_effects:
  - isolated workspaces and recording external boundaries
  - DeepSeek V4 Flash request in an isolated workspace
  - Git commits, pushes, stacked pull requests
forbidden_effects:
  - production activation without a separate activation gate
  - editing installed plugin cache instead of canonical source
  - deleting old runtime data as part of code cleanup
  - mocks or silent fallback that turn a real failure green
rollback: close the affected stacked PR and return to its parent commit; formal runtime remains on its prior activation receipt
```

## 3. 工作纪律

- [x] 进入仓库先读 `INDEX`、`WORKFLOW`、持久化地图、相关需求、决策和真实实现。
- [x] 使用独立 Git worktree，核对 dirty state，并在首次写入前建立可恢复 bundle。
- [x] 用 hua-home 只读日志确认旧 island、Wake phase、来源插件与真实 ACK 历史。
- [x] 用 ADHD 发散后收敛到 source-owned Timer、durable fact + lossy hint、source-owned ACK。
- [x] 用两名只读 agent 分别证明 lifecycle/abort 语义与现有 fixture 复用入口。
- [ ] 每张架构 PR 写入前和最终 HEAD 都由独立 Terra xhigh 检查正交性与 Conceptual Integrity。
- [ ] 每张 PR 只改变一根设计轴；相邻 diff 验收后才进入下一层。
- [ ] 每个已知缺口使用固定 oracle 先红后绿，不使用 `xfail`、skip 或 mock success。
- [ ] 每次持久文件写入都有 Git commit 或外部 repo 自己的备份/commit 可恢复。

## 4. Stacked PR 清单

### PR-A · 能力地图、决策与任务合同

目标：只固定已有原子、owner、真实 lifecycle 语义、阶段顺序和验收 oracle，不修改生产代码。

- [x] 盘点 `TIMERS`、`SCOPED_TURNS`、`DELIVERIES`、`CONTINUATIONS`、typed lifecycle、Service 与 exact Root。
- [x] 决定 source 复用 `TIMERS`，Content 不拥有 poll Timer。
- [x] 决定 Wake 使用 `channel="wake"` 分流，不新增 Core origin。
- [x] 决定 quiet abort 的双重事实：Session message/outbound/after hook 为零；Control Turn/items 仍可诊断。
- [x] 建立 Content、Wake、Drift、source、delivery 与 ACK 的唯一 owner 表。
- [x] 固定 delivery crash probe 在修复前必须真实非零，修复后同一 oracle 变绿。
- [x] Terra xhigh 对架构 HEAD 给出 APPROVE，must-fix 为零。
- [x] `git diff --check`、文档合同测试与 change-impact Gate 通过。
- [x] 推送并创建草稿 PR-A #481；远端公开 checks 全部通过。

### PR-B · Scoped Turn durable recovery 原子

目标：只补已经被 fixture 证明缺失的通用 Turn 能力，不出现 Wake/Content/Drift 名词分支。

- [ ] `BeforeTurnCtx` 投影当前 durable `turn_id`，让 lifecycle 能绑定已经接受的 Turn。
- [ ] `SCOPED_TURNS.read(accepted_receipt)` 返回 immutable Turn view；内部薄代理 Core durable Turn owner，不暴露 SessionStore/ControlService。
- [ ] Control 启动先把遗留 queued/in-progress 收敛为 cancelled/interrupted，再启动插件 reconciliation。
- [ ] scoped start 提供来源无关的 fresh-interaction admission；在新 Turn 创建事务中保存 append-only supersession edge，原子关闭旧 recoverable interaction 后再发布新 identity。
- [ ] 重启 continuation 不续接被 supersede 的 interaction；普通 passive failed/interrupted 自动续接完全不变。
- [ ] fixture 覆盖 active、completed、failed retryable/nonretryable、cancelled、interrupted、missing receipt 与进程重启。
- [ ] fixed session 下两个并发 admission 最多一个 Turn；随机 session 不能作为并发绕过方案。
- [ ] Terra xhigh、targeted tests、pyright、相邻 change-impact Gate 通过。
- [ ] 提交、推送并创建 PR-B，base 指向 PR-A head。

### PR-C · Content 普通插件与真实组合 fixture

目标：建立没有时钟、没有模型、没有 delivery 的 durable Content 邮箱，并用一个普通模拟来源插件证明现有原子足够。

- [ ] 实现窄 capability：submit、Wake read/transition、source-bound unsettled/ack。
- [ ] 实现 `source_id + item_id + revision` 幂等、冻结 high-watermark snapshot 与 CAS selection。
- [ ] 实现 defer/await-change/invalidated，保证 decline 不 hot-loop。
- [ ] 实现 selected token：只有 delivery settlement 才消费；已知失败按 terminal receipt retry/defer/invalidated，unknown 保持可恢复 selected。
- [ ] 创建普通 v3 `content_clock_source` fixture 插件，只 inject `TIMERS` 与 Content submit/ACK。
- [ ] 固定顺序：poll → submit commit → cursor/next_due persist → re-arm。
- [ ] 验证 duplicate、submit 后崩溃重启、cursor-before-submit mutant、丢 hint 恢复、source-bound ACK 隔离。
- [ ] 验证 candidate Root 零 timer/poll/write，old Root drain 后只由 new Root 恢复。
- [ ] Terra xhigh、targeted tests、pyright、相邻 change-impact Gate 通过。
- [ ] 提交、推送并创建 PR-C，base 指向 PR-B head。

### PR-D · Wake / Drift 普通插件与生命周期 fixture

目标：Wake 只用自己的 Timer、Content/Drift capability 与普通 scoped Turn；Drift 保持独立插件。

- [ ] Wake startup 从 durable due 恢复，hint 只加速，Content/Drift deadline 取最早值。
- [ ] 外层 due 不命中直接 re-arm，不创建 Turn。
- [ ] `channel="wake"` 的 `turn.context_prepared` 内固定执行 ContentGate→DriftGate。
- [ ] Content 命中后 Drift 不运行；两者 decline 时提交领域 transition 后 quiet abort。
- [ ] 验证 quiet case：provider/Tool/delivery/Session messages/after hooks 为零。
- [ ] 同时验证 Control Turn completed、输入 item、空 assistant item 与 TurnStarted 诊断仍存在。
- [ ] 验证模型/Tool 失败、取消、进程崩溃与 delivery rejected/unknown：不消费、不 ACK、不创建第二个并发 Turn。
- [ ] 启动恢复只使用 PR-B 的 durable Turn view 与 settlement forward-complete，不使用超时猜测。
- [ ] 验证普通 passive、Scheduler、Subagent 和其他 channel 不运行 Wake duty。
- [ ] 用 recording channel 跑 selected/declined/duplicate/new-item-after-snapshot/reload 全场景。
- [ ] 运行未修饰 oracle 的 delivery crash probe，按预期真实非零并记录 known gap，不计为通过。
- [ ] Terra xhigh、targeted tests、pyright、相邻 Gate 通过。
- [ ] 提交、推送并创建 PR-D，base 指向 PR-C head。

### PR-E · 来源无关的 durable delivery settlement

目标：只修通用 delivery 的跨崩溃窗口，不在 Core 引入任何 proactive/source 名词。

- [x] 固定 stable logical delivery id 与 prepared→provider_started→delivered→projected→settled 状态，并以 rejected/uncertain 收束不可前进结果。
- [x] provider receipt durable 后，重启只补 Session projection/领域通知，不再次 send；caller cancellation 也先完成 Core-owned forward step 再恢复取消。
- [x] provider 结果未知且不可幂等时进入可观察 `uncertain`，不伪装成功或盲目重发。
- [x] 通用 settlement 持久化 stable target service 与 logical/settlement ref；Content 以该 ref 幂等提交 delivered/settled 并 forward-complete，不宣称跨库原子。
- [x] pending settlement 不捕获退役 Root closure；candidate promotion 只读证明 `prepared/delivered/projected` 的 target service 仍能由 candidate topology 解析，否则不发布候选。
- [x] ACK 首次失败只重试 source ACK；Content settle 后本地崩溃只重放稳定 receipt 并补 Core confirm。
- [ ] 不修改 PR-D crash probe/oracle，让同一命令由非零变为零并进入 required Gate。
- [x] 验证 Session messages 仍只追加，普通 passive delivery 与 `message_push` 既有入口不变。
- [ ] Terra xhigh、targeted tests、pyright、相邻 Gate 通过。
- [ ] 提交、推送并创建 PR-E，base 指向 PR-D head。

### PR-F · 真实来源插件兼容迁移

目标：让真实插件只组合与其领域事实匹配的普通 v3 原子，并在 Core 保存可重放的跨仓
互操作证据。不是所有主动信息都进入 Content：离散待处理事实使用 Content，可覆盖的当前状态
使用插件私有 cache + Wake context，候选建议使用 Drift proposal。

- [ ] fetch 每个 canonical repo 的最新远端，读取各自 AGENTS/INDEX/发布规则并建立恢复点。
- [ ] Calendar：其 Timer 拥有 calendar poll/cursor；submit 后推进 cursor；ACK 使用 calendar provider 语义。
- [x] Feed：Feed 插件组合 `TIMERS` 的 source runtime 是唯一外部 poll owner；旧 MCP lifespan `FeedPoller` 物理退场，source runtime 直接调用共享 Feed domain library 后 submit/ACK，不复制第二份 poll loop。
- [ ] Fitbit：保留 monitor 作为采集 owner；adapter Timer 调 monitor HTTP 读 snapshot/event 与 ACK，不要求 Core MCP call。
- [ ] Steam：presence/current games 由 Steam 私有 current cache 原位覆盖；只有 fresh `channel="wake"` 追加 context hint，不进入 Content，也不伪造 ACK。
- [ ] GitHub Watch：保留 programmatic Turn producer；验证它不被 Wake 重复消费，并固定 reaction ACK 失败不重跑 Turn。
- [x] Emotion：Timer 刷新当前 context，普通 Drift proposal 表达候选行动，普通 Tool 提交结果；不保留旧 background job/domain effect 或 proactive documents 特权链。
- [x] Emotion Drift skill：只形成普通 proposal，并由 `emotion_commit_preference_context` Tool 把完整结果写入 Emotion 自有账本、覆盖 current context；不直接写 `proactive_pending.md`。
- [x] Proactive Feedback：保留 committed event observer 与独立 DB/outbox，并以 immutable history page Service 供可选 consumer 拉取；不进入 Content。
- [x] Observe：迁到普通 Turn/React trace，不再消费 `ProactiveFinished`，然后才删除该 Core event。
- [x] Daynight：正式 manifest 不含该插件、cache 无安装项、plugin-data 目录为空、当前 journal 无事件；已从最终 fleet/debug owner 删除，不造 no-op。
- [ ] 每个插件分别覆盖正常、重复、重启、ACK/无 ACK、reload cleanup。
- [ ] 用正式安装链构建 isolated plugin home，禁止直接修改 cache。
- [x] 完成跨仓库协议 commit、PR 与固定 source/runtime SHA 报告。
- [ ] Terra xhigh、各 repo tests、Core interoperability Gate 通过。
- [x] 各 canonical repo 提交、推送并合并兼容 PR；Core 固定互操作 revision 与回执。

PR-F 的保留规则按事实类型而不是插件名字决定：

| 事实类型 | 正常写法 | 物理减少规则 |
|---|---|---|
| 纯诊断 file log | 只记录运行诊断 | 固定文件大小与固定代数轮转 |
| delivery、ACK、cursor、observation、proposal、result、Session 历史 | 唯一领域 owner 追加或按状态机推进 | 没有名称明确的数据管理协议时全量保留 |
| current singleton/cache | 同一身份的当前值 | owner 可以原位覆盖；它不是历史账本 |
| empty/no-change poll | 不产生新领域事实 | 零持久历史；不能用“心跳成功”伪造业务记录 |
| 没有既有 file log 的插件 | 不新增日志文件 | 使用已有 Health/Incident/fixture receipt |

当前跨仓 revision 账本由 `docker/debug/content-source-interop.lock.json` 唯一维护；文档只解释
这些 revision 的语义，不另抄一份可漂移的 SHA 表。GitHub Watch 已确认正式 exact
`b9266ab3ca9932c074a6d91cf48ab69691bcf1ce` 本身就是普通 `BACKGROUND_JOBS` programmatic
Turn producer，无需迁移 PR：它不进 Content/Wake，reaction ACK 失败只调用一次且不重跑 Turn，
uncertain/cancelled 进入 `manual_reconcile`，candidate 不创建 client 或 data。

Proactive Feedback → Emotion 的 compatibility BLOCK 已由普通原语组合解除。Proactive Feedback
拥有 immutable history page Service；Emotion 是可选 consumer，用自己的 Timer 拉 page，并在
同一 Emotion 事务中提交结果与推进本地 cursor。它没有 Content settlement，因此不造 ACK；
Core 也没有 PF typed event、业务分支或新特权路径。

真实双插件 `CompositionRoot` fixture 保留旧红色 oracle 并把它变成时序断言：两种 mount 顺序下，
普通 Wake delivery 的 follow-up 先令 PF accepted history 恰为 1，而同一 committed Turn 的
Emotion PF-import/cursor 仍为 0；下一个普通 Timer tick 后 cursor 恰为 1。explicit quote 同时
存在时，Emotion 保存直接事实和“已由直接事实应用”的 import terminal 共 2 个 event，但只生成
1 个 feedback sample 和 1 次非零 delta，不重复改变状态。普通非 quote follow-up 由 fixture
提供的确定性 embedding 边界执行真实 PF scoring，同轮 Emotion event/sample 都为 0，Timer 后
各为 1。

远端回执也已闭合：PF PR #8 的
[contract](https://github.com/akashic-plugins/proactive_feedback/actions/runs/32639835703/job/97195110938)
与 [plugin-tests](https://github.com/akashic-plugins/proactive_feedback/actions/runs/32639835703/job/97195110846)
均通过；Emotion PR #6 已合并为 `9c8d94bdb13cfc2602409ba23556a61e26a3f031`，其
[contract](https://github.com/akashic-plugins/emotion/actions/runs/32640486430/job/97196710259)
与 [plugin-tests](https://github.com/akashic-plugins/emotion/actions/runs/32640486430/job/97196710357)
均通过。

### PR-G · Wake 真 provider 与全插件兼容 E2E

目标：在不删除旧链的前提下，证明新组合链能在隔离 workspace 用 DeepSeek V4 Flash 完成真实 Turn，并冻结所有兼容证据。

- [x] 参考 provider 合同创建无正式数据的隔离 workspace；credential/endpoint 只从环境进入内存，报告不含 secret、endpoint、prompt 或正文。
- [ ] 正式 `PluginManager → CompositionRoot → ConversationRuntime → execute_control_turn → AgentLoop.react` 链使用普通 recording Channel receipt；DeepSeek V4 Flash selected case 固定只允许一次 logical provider request、一次 delivery、一次 Session projection、一次 ACK settlement。唯一获授权真实 attempt 已证明一次 logical/HTTP request、formal workspace unchanged，并以 `Turn FAILED/nonretryable → Content invalidated → zero delivery` 失败；其 runner 手工 provider 丢失正式 runtime profile 语义，不能作为相同配置成功证据，也禁止自行重试。
- [x] selected provider 改走正式 `load_config → build_providers/from_runtime`，保留 `context_window=1_000_000`、`reasoning_effort=max`、`enable_thinking=true`、`max_output_tokens=0` 和 caller-composed system 优先语义；manual-provider mutant 与 loopback 200/400/503 fixture 已冻结装配差异和 completed/invalidated/deferred 分层结果。
- [x] quiet/empty-poll、settlement crash/restart 与 ACK-retry 使用确定性边界 fixture；所有唯一性 oracle 来自 SQLite ledger、Channel receipt、Session projection、Content 和 source ACK，不把模型随机性放进 oracle。
- [x] 不调用 `init_workspace`；测试前后只读对账 formal workspace 的 Session/旧 island 目标文件 digest、SQLite integrity/row counts 与旧 island archive hash/size。
- [x] 缺 secret、provider 非 2xx、identity mismatch 和 unsettled 均生成固定 `failure_stage/failure_code` 脱敏失败报告并返回非零。
- [x] formal live state 使用双 baseline + after 分离报告；并发变化只记录 path/type/count 并标 `formal_concurrent_change`，不伪装成 E2E 写入，严格 unchanged 留给 deployment Gate。
- [x] selected 失败仍在隔离 root 回收前累计 logical/HTTP/provider terminal/Control Turn/delivery/Channel/Session/Content/ACK count 与 identity digest，并在 finally 执行 formal-after；Turn error type 只保留 digest，baseline 与 after changes 各自保留 phase-local path/type/count。
- [x] 对 Calendar、Feed、Fitbit、Steam、GitHub Watch、Emotion、Proactive Feedback 跑组合 E2E，并固定 source/runtime SHA；Daynight 已按零消费者协议退出，不再作为运行组合成员。
- [ ] 验证旧链仍存在时新链没有双 poll、双 Wake、双 delivery 或双 ACK。
- [ ] Terra xhigh、累计 tests、pyright、公开 Gate 与 isolated E2E 通过。
- [ ] 提交、推送并创建 PR-G，base 指向 PR-F。

### PR-H · 旧 proactive island 与兼容藩篱清空

目标：只在 PR-G 的独立兼容证据通过后删除旧实现，不让同一 diff 中新增的行为替自己证明可替代。

- [x] 建立旧 island producer/consumer 矩阵：canonical source、installed cache、动态 loader、正式日志四类证据逐项齐全。
- [x] 为 Drift state、`proactive.db`、Wake/Drift DB、Markdown、配置/setup/dashboard/event consumers 逐项指定迁移、只读归档或零消费者删除结论。
- [ ] 删除 `ProactiveDocuments` 前证明旧 intent 目录无未决项、Emotion migration receipt 已提交、两份 Markdown 最终 digest 已记录、Drift skill 已无直接 writer。
- [x] 先把仍被普通被动 Turn/optimizer 使用的 presence 与 memory optimizer 移到各自通用 owner，再删除 `proactive_v2/` 包。
- [x] 删除 `bootstrap/proactive.py`、`proactive_v2/`、`plugins/default_proactive/` 与 Wake 私有 loop 的运行入口。
- [x] 删除 proactive MCP 聚合/轮询/ACK 桥；模型主动选择的普通 MCP tool 保留。
- [x] 删除旧配置、启动参数、health/readiness、reload、日志和测试藩篱；不保留 deprecated alias 或空壳。
- [x] 旧 DB/Markdown/plugin-data 不自动物理删除；迁移只读并 supersede，恢复证据写入报告。
- [ ] 全仓 `rg`、动态 loader、真实启动日志证明旧运行入口与 island 事件为零。
- [ ] 对栈顶相对 `origin/main` 运行累计 tests、pyright、公开 Gate、PR-G 冻结 E2E replay 与 Terra xhigh Gate。
- [ ] 提交、推送并创建 PR-H，base 指向 PR-G；建立 umbrella PR/issue 展示整个 stack。

### PR-H5 · 删除后的确定性组合证据

目标：删除旧 island 后，用一个不拥有业务语义的薄 runner 组合安装、跨仓 revision 和既有
fixture 回执；不为 E2E 增加 Core 接口或第二套 runtime。

- [x] Feed revision 账本指向 PR #7 branch `codex/feed-fixture-python-env` exact `bc26736d16dd34420d1097ff14ea707c79f2f117`，并运行其 legacy handoff/cutover fixture；Fitbit PR #7 与 Steam PR #4 同样固定测试专用 service interpreter 合同。
- [x] runner 使用正式 trusted batch CLI 安装 exact 批次，插件 root 只取回执 `installedPath`。
- [x] 同一次 run 固定 Core head/tree/dirty、lock hash、installed revisions、Content interop、Scheduler/Subagent/MCP、Wake/Drift/H2 和 provider loopback 报告 hash/status。
- [x] 一次性 root 明确分开 workspace、plugin-home、reports 与 HOME；隔离的非空 protected fixture 含 Session/旧 island 文件与 SQLite 行，按 path/inode/hash/size/quick-check/row counts 只读前后对账，生成证据不提交。
- [ ] DeepSeek V4 Flash 真实请求保持 `PENDING`；只有新的明确授权才运行 manifest 记录的命令。

H5 runner 只顺序调用各 owner，并校验进程退出码、回执身份和报告状态。Core dev Python 运行
owner pytest；只有具有 service runtime 的 artifact 才通过测试专用
`AKASHIC_PLUGIN_FIXTURE_PYTHON` 收到回执中的解释器路径。一次性 root 内固定版本的 pytest
fixture layer 只做 artifact 隔离探针，不进入 owner fixture；runtime 依赖继续来自 artifact，
Core site-packages 不进入 service 路径。它不读取业务 payload、
不解释 Wake/Drift/Content 状态，也不把 operator trust 写成 programmatic validation。确定性 fixture
冻结 selected/declined、quiet empty poll、settlement crash/restart、ACK retry attempt 2、provider
200/400/503 以及不重复模型、投递和 Session projection 的现有 oracle。

H3 已把 Core 运行态收成普通 `BACKGROUND_JOBS` Activity：启动、快照、reload、drain 和
outcome ledger 只认识 interval/programmatic/LLM job，不再认识 proactive catalog、私有 family、
`DriftFinished`、domain effect 或 paired documents。旧 `proactive_v2` runtime、Default/Wake 私有
插件和 Dashboard 路由已经从代码树删除。H4 又删除了旧配置类型/parser、setup 向导、Prompt、
Dashboard 前端和 Mobile `proactive-context` 投影；任意空或非空 `[proactive]` 都在打开 workspace
store 前明确失败。Session 的 `last_proactive_at` 与 H2 历史迁移入口继续保留；Mobile
`message.proactive` decoder/event 已由 0045 删除，客户端只按 Session seq 同步。
`init_workspace` 不再创建 `proactive.db` 或 `PROACTIVE_CONTEXT.md`，但 H2 inventory/history 继续
只读已有文件，任何代码升级都不删除 workspace 数据；`force` 初始化也保持既有文件 inode 与
digest 不变。

这只证明 isolated Core 已没有旧 island，并不等于正式 activation READY。Observe #5、Emotion
#6 与 Feed #7 已合并；Daynight 则以四类零消费者证据退出最终 fleet，并与 PR-G 固定的
Content/Wake/Drift/source interoperability E2E 一起验证；H2 对未交接 continuity、quota、pending
documents 或 Wake archive consumer 的 `BLOCK` 也必须清零。上述条件未满足时保持旧正式 runtime，
不修改 cache、不用 no-op compatibility shell 掩盖依赖。

PR-H 按 owner 迁移与删除分层，当前 H2 只建立 active-state handoff，不删除旧 writer：

```text
legacy source row ── inventory ──▶ source/target owner adapter
       │                              │ plan: read-only
       │                              │ apply: target first
       │                              │ verify: read-only
       │                              ▼
       └──────────────────────▶ Core lineage marker
```

| legacy active fact | 目标 owner / receipt | source marker | H2 不能处理时 |
|---|---|---|---|
| Wake unread Content，保留 legacy source/event locator | 来源插件先从自己的 provider DB 恢复 revision，再通过既有 source-bound Content view 提交；receipt 指向 target source/item/revision | Core sidecar 只追加 locator、source digest、receipt id/digest、target identity | source identity 缺失、重复冲突、provider revision 缺失或 adapter 缺失时 `BLOCK` |
| Wake pending ACK | 对应 source owner 的 ACK/settlement receipt | 同上 | source row 无 owner、孤儿 ACK 或未知 action 时 `BLOCK` |
| Wake 私有连续性：quarantine、tombstones、hazard、context、context reevaluate、drift | 当前没有逐表接收其下一轮 ingress/decision/timer 语义的 v3 owner；Core 每表只保存 row count + ordered digest，不解码字段或复制 row | 无 | 非空时 `wake_continuity_owner_unavailable`；未知 Wake 表 `unknown_wake_table` |
| Wake 历史：runs、observations、hazard monitor | 旧库只读 historical decoder；`hazard_monitor` 只被 Dashboard/只读 load 消费，runtime 下一轮只读取 `hazard_state` | 无 | 不迁移、不阻止 activation inventory；原库不自动物理删除 |
| `PROACTIVE_CONTEXT.md` | Wake 私有 exact-bytes archive + versioned receipt；仅 Wake `BeforeTurn` 读取并注入 | 同上 | archive/receipt 不成对或 digest 不符时 `BLOCK` |
| Drift paused/staged | 没有可恢复 proposal payload，H2 不伪造 proposal | 无 | `proposal_payload_unrecoverable` |
| `proactive.db` continuity 表：deliveries、session、context、rejection、seen、kv | 当前没有逐项接收其连续性语义的 v3 owner；Core 每表只保存 row count + ordered digest 的阻塞摘要，不复制 row | 无 | `proactive_continuity_owner_unavailable`；原库由 history decoder 只读，tick/step/semantic 同样保留但不阻止 |
| `proactive_quota.json` | 当前窗口计数仍会改变下一次动作，没有接收 owner 时不能解释为空 | 无 | `proactive_quota_owner_unavailable`；按 exact bytes 备份并保留原文件 |
| proactive documents active intent / nonempty pending | 需要成对的领域 owner handoff，H2 不拆成单 owner | 无 | `paired_target_handoff_unavailable` / `pending_document_owner_unavailable` |
| generic `BACKGROUND_JOBS` rows | 继续由 v3 background-job ledger 拥有；只进入历史投影 | 无 | 不盘点、不迁移、不阻止；GitHub Watch running 也不属于旧 island active state |

Core 只拥有 inventory 顺序、preflight 和 append-only lineage，不解释 Feed、Emotion、Drift
或 Documents payload。adapter 的 `plan` 必须零写：未有 lineage 时只读 source/provider，不能为了
取得 Content binding 而 mount 插件、创建 schema/WAL 或目录。已有 lineage 的 `verify` 只能使用
已经存在的 target service 或严格只读 target view。`apply(fact, plan)` 必须返回与 plan 完全相同的
`target_identity`，Core 验证 target receipt 后才写 source marker；target 后、marker 前崩溃由目标
owner 的幂等 identity 重放收口。

H2 handoff 是 offline maintenance，不是 live migration API。`proactive.db`、Wake、Drift 和 job
历史都经过同一个 immutable reader；任一 legacy DB 有非空 WAL 时明确停止，避免漏读尚未
checkpoint 的事实。Content 的 exact receipt/revision
view 使用不初始化 schema/WAL 的 immutable read；若存在未 checkpoint 的 WAL 就明确失败，不能把
“immutable 看不到刚提交的 row”误报成 missing receipt。正式 activation 另行证明 runtime 已
quiesce；本 PR 不增加进程锁、隔离 marker 或新的 maintenance manager。

CLI 默认只执行 plan。显式 apply 要求 absolute workspace 与独立 `--backup-root`，先用 SQLite
online backup 和完整 Markdown/quota bytes/digest 建立恢复点；backup 与 workspace 不能互相包含。
备份完成后重新盘点；active facts 的 locator/digest/owner identity 与 blocks 必须逐项相同，变化时
不调用 target adapter，也不写 lineage。tick/step/semantic、Wake consumed 和 generic job outcomes
只是保留历史，不是本次 handoff 输入，因此不会把 Core 变成锁住所有 owner 的全局快照器。
不要求额外 marker 文件，也不写正式 hua-home workspace。historical decoder 只用
严格只读 SQLite 和文件读取形成 Dashboard/CLI 投影，不实例化 legacy writer、event bus 或
proactive runtime；`proactive.db` 九张已知表、Wake runs/observations/hazard monitor 与 generic job
ledger 都只投影、不复制第二份历史。Wake quarantine/tombstones/hazard/context/reevaluate/drift
仍会改变下一轮 ingress、decision 或 timer，在目标 owner 缺失时按表阻止 activation。

正式 cutover 另有一个显式 operator retirement yoyo；默认 H2 行为不变。yoyo 要求维护窗口内
冻结的完整 inventory digest、独立 backup root 和可验证 manifest，只允许把
`proactive_continuity_owner_unavailable`、`wake_continuity_owner_unavailable`、
`proposal_payload_unrecoverable`、`proactive_quota_owner_unavailable` 四类已审批 block 标记为
`operator_approved_pre_cutover_supersession`。receipt 逐项绑定 locator/reason/source digest，并在
每次 plan 时重新验证 backup manifest 和每个归档文件；出现新 block、digest 漂移、备份丢失或
未知类别都继续 fail-loud。旧 DB/Markdown 原路径和 bytes 不被 yoyo 删除或改写。

Feed 的 canonical yoyo 还会在写入前只读冻结新 source 当下会提交的完整 provider backlog，要求
operator 同时固定 item count 与 identity digest，再把整批 provider revision 原子写成
`cutover_superseded` ACK 和 batch receipt。随后 H2 才为 Wake unread rows 写逐项 source receipt；
两者都不向 Content 提交旧条目。这样旧 Wake 中已经 consumed、但新 Feed 尚未 ACK 的 provider
条目也不会在首次启动时重新出现；切换后新增或 revision 改变的条目仍按普通 source 链处理。

## 5. 旧 island producer / consumer 迁移表

本表在 PR-G 固定真实 revision 与行为，在 PR-H 才允许删除旧路径。`unknown` 不能自动解释为零消费者。

| 对象 | 当前职责/消费 | 目标 owner | PR-G 证据 | PR-H 处置 |
|---|---|---|---|---|
| Calendar/Feed/Fitbit | old source poll + ACK | 各 source Timer/adapter + Content | canonical/cache/loader/log/E2E | 删除 old source registration |
| Steam | latest context | Steam current cache + Wake-only fresh hint | passive/stale 零 hint、history 保留、reload | 删除 old context bridge |
| GitHub Watch | programmatic Turn + reaction ACK | 既有 job/Turn owner | 不双消费、ACK failure replay | 保持独立，不强塞 Content |
| Emotion | current context、Drift proposal、result history | Timer + context listener + Drift proposal + ordinary Tool | proposal replay/revision、Tool receipt、Wake/passive context | 删除 proactive module/documents bridge |
| Proactive Feedback | delivered Message feedback DB/outbox/event | v3 committed event consumer | exactly-once event/outbox replay | 删除旧 feedback hook/registry |
| Observe | `ProactiveFinished` 线上消费者 | 普通 `TurnCommitted` / react trace | Wake Turn observation equivalence | 删除 proactive-only event seam |
| Daynight | 历史 Wake phase gate；formal 当前 manifest/cache 均无安装项，仅残留空 plugin-data 目录 | 无运行 owner | 当前 manifest/cache 无条目、plugin-data 为空、当前 journal 无事件；旧日志只记录 disabled | 已删除 fleet/debug/install owner；canonical 仓库保留为历史源码，不造 no-op |
| Wake reservoir/quarantine/hazard | Wake selection continuity | Content + Wake plugin-data | copied-state migration/restart | 旧库只读归档，不自动物理删 |
| Drift DB/cursor | Drift due/proposal continuity | Drift plugin | state equivalence/restart | supersede，旧库保留 |
| `PROACTIVE_CONTEXT.md`/pending docs | prompt/context 与 Emotion merge | 明确的 Content/Emotion/Drift owner | write-set 与 prompt snapshot | 迁移或只读归档 |
| `ProactiveDocuments` intents | 两份 Markdown 的跨 owner intent/recovery | Emotion-owned append + 明确 prompt projection | pending intents=0、receipt、final digests | 迁移完成后删除特权 document host |
| presence | proactive 与普通被动 Turn 的 session activity | 通用 session activity owner | passive/proactive differential replay | 先移出包，再删除旧 import |
| memory optimizer | app/dashboard/optimizer runtime | memory owner | optimizer startup/job/dashboard regression | 先移出包，再删除旧 import |
| setup/config/dashboard/health/log event | 启停与观察旧 island | 普通 plugin projection/inspection | startup API/log diff | 删除旧开关与事件名 |
| proactive MCP bridge | Core 聚合 source poll/ACK | source adapter direct protocol | tool catalog + adapter E2E | 删除聚合桥；普通 MCP tools 保留 |

## 6. Fixture 规则与场景

特殊模拟插件只模拟外部世界：clock、外部 feed、provider 和 channel。下面这些产品内部必须使用真实实现：v3 loader、Root/Fiber、Timer service、Content store、ConversationRuntime/react、SessionStore、reload/drain 与 settlement 顺序。

| case | 必须观察的结果 | 必须杀死的错误实现 |
|---|---|---|
| selected | 一个 timer、一个 item、一个 Turn、一个 delivery、一个 projection、一个 ACK | 每阶段各自造 loop |
| declined | 领域 transition 存在；provider/delivery/Session message 为零 | listener `return` 被当成 Turn skip |
| duplicate | mailbox 一条、一次 Wake | hint 或重复 poll 被当新事实 |
| source crash | submit 已存在，cursor 未推进；重启重 poll 后仍一条 | cursor 先于 submit |
| selected failure | 已知失败 release/defer，unknown 保持 selected；零 ACK/零并发 Turn | CAS 后永远卡住或超时重选 |
| ACK retry | delivery/projection 一次，第二次只 ACK | ACK 失败触发重做 Turn |
| provider ACK/local crash | 不再远端 ACK，只补 Content settle | 未保存 provider receipt |
| root reload | old cleanup、new recovery、一个 timer/Turn | module global 跨 Root 泄漏 |
| new item after snapshot | 新 item 留给下一 Turn | 运行中 snapshot 漂移 |
| delivery crash | send 一次，重启只补 projection/settle | provider delivered 后重发 |

统一 debug receipt 至少包含：source poll id、submit receipt/snapshot seq、Wake timer receipt、Turn id/selected refs、logical delivery id/provider receipt、Content settlement ref、source ACK receipt、final state、Root/revision identity。

## 7. 正交性与 Conceptual Integrity 门检

每张 PR 写入前和最终 HEAD 都回答：

1. 新名词是否独占一种状态、控制流、生命周期或边界；否则删掉或并回 Message/Turn/Session/Loop/react。
2. 每项事实是否只有一个 writer；Core 是否出现来源名、插件 ID 或专属暗号。
3. 改一个轴是否迫使无关插件变化；若传播，必须来自真实信任/持久化边界。
4. 正常、失败、重启、reload 是否沿同一条状态机，而不是另造例外路径。
5. 新检查是否有真实可达违反路径，且当前 owner 能明确恢复；没有就不加。

Terra xhigh 的 `must-fix` 未清零时，该 PR 不进入下一层。测试全绿不能替代此 Gate。

## 8. 停止与交付

遇到 provider credential 缺失、canonical plugin source 无发布权限、当前 main 改变合同、或 fixture 证明必须改变已批准持久语义时，保留完整证据并只停止受阻的外部步骤；继续完成不依赖该阻塞的层。

全部 `[ ]` 变成 `[x]` 后，最终报告用六岁小孩能理解的积木比喻解释：每块积木是谁、保存什么、什么时候动作；同时列出每张 PR、commit、测试、Gate、真实 E2E、方向性修订、遗留数据、未激活生产的边界与恢复点。
