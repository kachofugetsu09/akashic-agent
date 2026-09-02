# 插件 v3 生产替代清单

> 历史执行清单：其中 E1～E4 表格记录 2026-08 的迁移计划，不再是当前 CI 或发布命令。2026-09-02 的 Gate 去留与代码演进依据见[测试与 Gate 清理账本](../refactor/test-gate-cleanup-ledger.md)；当前候选入口以 [`docs/WORKFLOW.md`](../WORKFLOW.md) 为准。

本文是 Issue [#394](https://github.com/kachofugetsu09/akashic-agent/issues/394) 的唯一执行清单。
[插件 v3 最终迁移地图](plugin-v3-final-migration-map.md)负责解释目标架构、现有 PR DAG 和删除顺序；
本文只记录每项能力是否已经具备可替代生产的证据。状态必须由实际 commit、测试和 Gate 推进，
不能用实现者自述或单个单元测试把项目标成完成。

## 1. 任务合同

### 1.1 目标与完成标准

目标是让 Akashic 的通用插件平台只接受 v3 namespace，以
`Context / Service / Fiber / Effect / typed event` 组合能力，同时保留 Akashic 的 candidate
验证、promotion、snapshot lease 和旧 generation drain。除 Default/Wake Proactive 两族的内部
领域实现外，生产目标 fleet 中的插件都迁入 v3；两族只保留私有 proactive 运行实现，不再取得
通用 v2 插件 ABI。Context Pressure 与 Computer Use Linux 已由维护者移出目标 fleet，不再迁移；
最终 consumer scan 和 E4 必须证明正式替代清单不加载这两个插件，不能为它们保留 v2 兼容面。

只有下列条件全部成立，才能向维护者报告“可替代线上 Akashic”：

- [x] Core 基线清单全部为 `READY`；
- [x] 插件清单全部为 `READY` 或已批准的 `PRIVATE_LEGACY_RUNTIME`；
- [x] `api_version = 2`、`Plugin`、`PluginContext`、`ToolHook` 和 Manager 固定贡献面不再构成
      production 插件兼容 API；
- [x] v2 删除批次 A～J 全部完成，或只剩明确属于 Proactive 私有实现、无法被普通插件调用的代码；
- [x] 精确锁定的跨仓组合、集中 E2E 和复制 workspace 数据安全演练全部通过；
- [x] hua-home 正式 workspace、服务、渠道、凭据和端口在本任务中未被修改；正式替换另行批准。

### 1.2 Change intent

```yaml
change_type: migration
semantic_delta: breaking
capability_owner: mixed
consumer_scope:
  - akashic core plugin runtime
  - 20 locked external plugins
  - 8 in-tree plugin implementations
  - later admitted GitHub Watcher canonical source
runtime_patch: required
runtime_patch_reason: >-
  stable/candidate publication、generation lease、能力冲突、进程和渠道提交边界由 Core
  拥有；插件只拥有领域实现和自身 data schema。
authoritative_state_owner: >-
  SessionStore、Memory store、plugin data owner、Proactive state store、plugin artifact
  publisher 和外部 channel/process owner 各自保持不变。
invariants:
  - PLG-001 through PLG-014
  - SES-001 through SES-008
  - OUT-001 through OUT-005
  - WSP-001 through WSP-005
  - PRO-001 through PRO-003
  - CTRL-003
  - MOB-006
  - BAK-001
  - TST-001 through TST-008
protected_state:
  - sessions.db messages and attachments
  - consolidation_writes.db and compaction receipts
  - memory2.db and Markdown memory archives
  - akasha.db and deterministic sidecars
  - plugin-data
  - proactive.db, wake_proactive.db, drift.db, PROACTIVE_CONTEXT.md, and proactive_pending.md
  - schedules, quota, plugin reload journal, rollout fact, artifacts, manifest, pointers, and credentials
allowed_effects:
  - source, tests, docs, CI, and disposable test workspace changes
  - isolated local processes, loopback ports, and controlled external read-only probes
forbidden_effects:
  - writes to hua-home or its formal workspace
  - production channel delivery
  - deletion or rewrite of existing authoritative messages, memories, plugin-data, or proactive state,
    except the separately tested explicit Plugin Undo operation on a disposable copy
  - direct edits to installed plugin cache
rollback: >-
  Core recovery point is backup/plugin-v3-full-migration-base-20260816 at
  501dad1c86cfe2cf4c62982d4dde92e831110251. Each external plugin uses its own base commit and
  independent worktree; data tests use disposable copies and never become recovery owners.
```

### 1.3 状态定义

| 状态 | 含义 |
|---|---|
| `OPEN` | 还没有可审阅实现，或能力合同尚未闭合 |
| `IMPLEMENTED` | 有实现和定向测试，但尚未通过独立 review/组合 Gate |
| `CANDIDATE` | 精确 commit 已通过独立 review 与本族行为 Gate，尚未进入最终全量组合 |
| `READY` | 在最终集成 head 上通过适用的集中 E2E、数据 write-set 和 cleanup 验收 |
| `PRIVATE_LEGACY_RUNTIME` | 仅允许 Default/Wake Proactive 私有领域实现使用；不暴露通用 v2 插件入口 |
| `BLOCKED` | 缺 canonical source、凭据、外部环境或必须由维护者决定的语义 |

状态从 `CANDIDATE` 进入 `READY` 时必须记录 Core commit、插件 commit、scenario catalog 摘要和
Gate 报告。分支名、PR 号和浮动 ref 不能代替 commit。

## 2. Core 基线能力

### 2.1 已有候选底座复核

| ID | 能力 | 当前状态 | 验收证据 |
|---|---|---|---|
| C01 | Context / Service / Fiber / Effect 与逆序、抗取消清理 | `READY` | 最终集成 head 的 kernel/cleanup 回归 |
| C02 | serial / parallel / transform / observe typed events | `READY` | C16 四项 admission 收口后累计回归 |
| C03 | immutable topology、parent edge 与 composition revision | `READY` | candidate/formal drift mutant 保持通过 |
| C04 | isolated candidate Root、atomic stable batch、promotion/lease/drain | `READY` | full-fleet reload/discard/promote Gate |
| C05 | Validation / Health / Incident 分离与 runtime inspection 数据模型 | `READY` | inspection 查询面与 full-fleet health 场景 |
| C06 | generation `data_root`、workspace roots 与 candidate 隔离 | `READY` | 复制 workspace write-set 验证 |
| C07 | typed Tool 六段链与 exactly-once result | `READY` | Tool 组合 Gate 纳入最终 exact lock |
| C08 | prepared context、Memory capability 与原子 assistant metadata commit | `READY` | Akasha/Observe/Emotion 组合 Gate |
| C09 | Skill / Drift skill / Dashboard generation 投影 | `READY` | 全量插件 collision、dispose、artifact immutability |
| C10 | passive WebUI stable snapshot E2E | `READY` | 最终全量 WebUI-only E2E |

### 2.2 必须补齐的 Core seam

| ID | 能力 owner | 状态 | 验收 oracle | 首个真实 consumer |
|---|---|---|---|---|
| C11 | committed channel command catalog | `READY` | command/provisional 独立复核 25 tests、累计 command/kernel/loader/Manager/hot-reload 339 tests、Basedpyright/compileall/diff-check 已通过；Status Commands `eb245ad` 已以真实 Manager committed registry 执行 `/memorystatus`；Core `PluginManager`/bootstrap/Telegram/Mobile 已只读 committed catalog provider，旧 `telegram_bot_commands/mobile_bot_commands` 聚合与 list fallback 已删除，legacy claim fail-loud；待 E3 全量命令目录 | Status Commands |
| C12 | scoped MCP capability | `READY` | `8653bab0` 已接入 Root-local declaration、candidate/formal MCP catalog fence、exact snapshot route、跨 boot durable recovery；`b18e876e` 已物理删除 workspace MCP 的第二套 Manager/snapshot/admin/watcher owner，插件 MCP 只走 static manifest → exact Root → McpGenerationHost。Calendar `654d078d` 的真实 Manager Gate 完成 stdio handshake、完整 14-tool catalog 与零 CallTool/Google 调用；待最终 E2 exact lock | Calendar MCP |
| C13 | managed process capability | `READY` | `8653bab0` 已接入 generation-scoped start/readiness/port/log、sibling drain、retained tombstone 与不可取消 recovery；Calendar `654d078d` 的正式 `calendar_api` readiness `/health=200`，terminate 后端口、进程、task、Root Effect/listener 全零；待最终 E2 process 族 Gate | Calendar MCP |
| C14 | inbound/outbound channel capability | `READY` | `5e58d38d` 已在 `fc1a2a76` 基础上补齐 exact-binding control、typed turn presentation、EventBus owner-task bridge、provider identity 与同步 snapshot claim；`4d6459ff` 已把 Core built-in Channel 发布到 committed catalog/Host，并物理删除 MessagePush/Bus/Channel legacy delivery fallback，299 个组合回归与 BasedPyright 通过。待旧测试 oracle 对账与最终 E3 才进入 `READY` | Feishu / QQBot |
| C15 | timer / proactive source / turn enqueue capability | `READY` | Core `78e50d4d` 已修复 candidate/formal Root 重建时 proactive/private/background catalog 的 exact Root payload 替换；Calendar `654d078d`、Feed `b4a8626`、Steam `2c492d7`、Fitbit `f3fd6ee` 已证明 exact committed source/MCP binding、candidate recording 零凭证/远端写入、typed empty/items/failure 与 ActivityHost cleanup。待最终 fixed-clock E3 | Calendar、Feed、Steam、Fitbit |
| C16 | v3 admission/lifecycle 收口 | `READY` | `4ba266ad` 已通过独立 review；non-callable listener、spawn coroutine、apply signature、wrong-task lifecycle 全部 fail-loud，malformed admission 零 data-dir 写入 | Core |
| C17 | mobile UI/query capability | `READY` | `2c6e4f71` + `b173f551` 已通过独立 review；activation token、strict JSON、candidate 不发布与 exact lease 已由 372 个集成回归、Basedpyright/compileall/diff-check 验证，待 Akasha/Observe 迁移后进入 E1/E4 | Akasha / Observe |
| C18 | Core-private v3 generation metadata | `READY` | `2d9fb408` 已通过独立 review；v3 stable load、candidate clone 与 formal rebuild 不再构造或读取 `PluginContext`，59 loader + 208 Manager/hot-reload 回归通过 | Core |
| C19 | full-fleet Health/Incident/Topology inspection | `READY` | stable lease 按插件投影 current Fiber/Health、累计与 bounded Incident、Topology；active/inactive v3 inspection 与 kernel/protocol 回归通过，独立 review 的 inactive projection P1 已关闭 | 全量 runtime |
| C20 | Proactive 私有兼容岛 | `READY` | Core `1968e503` 已把 Default/Wake 六个内建 module 收进 Core-private catalog/Host；跨 publication tx reload/rollback、lexical symlink admission 与 kernel start failure 的 exact ownership 已闭合。独立复审无 P0/P1，主任务复跑 4 个关键 mutant，累计 private/Manager/hot-reload/ActivityHost 回归与 Pyright 0 通过，待 E3 fixed-clock/recording Gate | Default/Wake Proactive |
| C21 | generation-scoped background job / LLM capability | `READY` | Core `de08b698` + `467a4c93` 已完成 committed catalog、trigger/interval、exact LLM lease、cancel/drain、Emotion domain receipt 与 paired-document recovery；Emotion `201ff1e` 把 `emotion_state`、projection 与 durable domain receipt 收进同一 SQLite transaction，去掉 external production 对 `proactive_v2` 的依赖，并以真实 Manager 覆盖 precommit rollback、commit 后取消重入和 Core 进程崩溃重入。Core/插件定向回归、Pyright/contract/compileall/diff-check 通过，待 E1/E3 | Emotion |
| C22 | static v3 artifact manifest / install staging | `READY` | 集成 head `3e1f5c10`（独立复审 head `b6967d13`）已在 import 前校验 identity/runtime/validation、custom entrypoint 与 C12/C13 descriptor，不成功 staging 不创建正式 data/artifact/pointer；Calendar `654d078d` 已由真实 Host 消费 exact staged `.venv/bin/python`，manifest/apply/runtime identity 一致且候选排除 credential/receipt；待全部 external v3 与最终 artifact Gate | Calendar MCP / 全部 external v3 |
| C23 | Core-owned Channel attachment artifact / Session binding | `READY` | `0bd2d928` 已完成 immutable opaque artifact、fixed-ID resumable Mobile import、SessionDB 原子 message binding、exact fd read lease、无自动 GC 与目录/SQLite 可恢复备份；Mobile 114、Bus/lifecycle/Host 124、最终 ownership 聚焦 201 tests、Basedpyright 0，独立 review 无 P0/P1；待 Feishu/QQ 与复制 workspace E3/E4 进入 `READY` | Feishu / QQBot / Core channels |
| C24 | read-only existing Session projection | `READY` | Core `cb2011b4` formal 只经 `get_existing` 返回 detached snapshot，candidate 同名 Service 调用即 fail-loud；Status Commands `eb245ad` 的真实 active compaction ledger 同时驱动 committed command 与 Mobile query，查询前后 `sessions.db*` 摘要不变。Core 38、插件 6、面板 7 tests、Pyright/contract/compileall/diff-check 通过；待 E3 复制 workspace | Status Commands |
| C25 | explicit interaction undo coordinator | `READY` | Core `b58b7905` + Plugin Undo `7b0e4cd` 已通过独立复审：destructive owner、latest interaction/active/pending-compaction fence、SQLite backup、Default Memory durable receipt、Akasha source gate、进程内取消与 Core 重启重放均闭合；`/undo` 已回显可恢复 backup 与强制存在的 compaction cursor receipt。Core 6、插件 Manager 4、既有 Memory/Akasha/SessionStore 回归与 Pyright/contract/compileall/diff-check 通过，待 E1/E3 copied-workspace Gate | Plugin Undo |
| C26 | exact programmatic Turn / v3 Tool catalog | `READY` | Core `7d68020a` + `2f1f304a` 已建立 Root-local Tool catalog、exact generation handler、invocation-scoped programmatic Turn port 与主服务/stdio 启动前 owner binding；整张 candidate snapshot 均不取得 Turn port且不发布 Tool，durable Session 只可由同一 plugin/job 跨 invocation 复用，`submitting/admitted` receipt 与 typed pre-admission/uncertain failure 防止失败、取消及进程崩溃后重复 Turn；post-persist start failure 会把 durable Turn 收束为 failed、释放 active owner 并保留 manual-reconcile receipt。GitHub Watcher `aea802c` 已纯 v3，exact Core Gate 完成初始正式 job 准入 Turn、candidate 零外部效果与晋升后 exact generation job 再准入 Turn；312 个 Core 回归、45 个插件测试、Pyright/compileall/diff-check 通过，待 E3 controlled repository Gate | GitHub Watcher |

实现原则：C11～C17、C21～C22 只由表中的首个真实 consumer 拉动，不提前复制
`commands()/mcp_servers()/managed_services()/channels()/jobs()/proactive_*()/mobile_ui()` 旧方法。

以上表格末尾保留的“待 E1/E2/E3/E4”是候选阶段的进入条件和追溯说明；最终 `READY`
由 6.4 的同一 clean head 报告统一闭合，不再表示未完成事项。

## 3. 每个插件的完成定义

每行插件只有同时满足以下检查项才能进入 `CANDIDATE`：

- [x] canonical source、base commit 和 candidate commit 已固定；
- [x] external v3 artifact 有静态 `akashic.plugin.toml`，安装期不 import/执行 plugin；
- [x] 模块只暴露 `api_version = 3` 与精确 `apply(ctx, config)`；
- [x] capability、Service 依赖、listener 顺序、静态投影和 data/workspace roots 声明完整；
- [x] `apply()` 在 candidate 环境不写正式 workspace、不发送、不占正式 endpoint；
- [x] dispose/reload/cancel 后 task、Effect、listener、process、port 和 module 均清理；
- [x] v2 与 v3 的正常、空、拒绝、错误和取消行为等价；批准的差异单独写 `semantic_delta`；
- [x] 真实 `PluginManager` install → snapshot lease → consumer 行为链通过；
- [x] 插件仓 CI 与 Core contract 固定 exact Core/protocol commit；
- [x] 对应 v2 owner 已加入可删除 inventory，且没有未列出的 consumer。

单元测试和契约测试负责每个插件自身行为；不能为每个插件单独启动完整 Docker E2E。

## 4. 插件迁移账本

### 4.1 已有 v3 候选

| 插件 | v3 能力 | 状态 | 最终组合批次 |
|---|---|---|---|
| Citation | prompt protocol、assistant metadata | `READY` | E1 |
| Meme | required citation service、prompt/media、Skill、Dashboard | `READY` | E1 |
| Shell Restore | `tool.input.prepare` | `READY` | E2 |
| Shell Safety | `tool.execution.authorize` | `READY` | E2 |
| Tool Loop Guard | typed authorization、per-generation state | `READY` | E2 |
| Default Memory | static Memory capability、result observer、Dashboard | `READY` | E1 |
| Calendar MCP | MCP、managed process、proactive source | `READY` | E2/E3 |

### 4.2 External v3 candidates

| 插件 | 当前 v2 能力 | 依赖 Core seam | 状态 | 最终组合批次 |
|---|---|---|---|---|
| Daynight Gate | proactive module / prompt gate | C15 | `READY` | plugin `07c2bfe`；external production 已无 `proactive_v2` import，真实 Manager/ActivityHost exact lease、配置行为、dispose、contract 与 Pyright 已通过，待 E3 |
| Emotion | Dashboard、mobile、Drift Skill、proactive module、job/LLM | C09/C15/C17/C21 | `READY` | plugin `201ff1e`；external production 已无 `proactive_v2` import，candidate 不写正式数据，formal exact Root 冻结 UI/Skill/proactive/job，领域写集与 receipt 原子提交；真实 Manager 的进程内失败、取消与 Core 进程崩溃重入通过，待 E1/E3 |
| Plugin Undo | command、显式 interaction 撤销 | C11/C25 | `READY` | Core `b58b7905` + plugin `7b0e4cd` 已把 destructive owner 留在 Core；candidate 调用拒绝、formal `/undo`、backup/事务 fence、Memory/Akasha 恢复与 Core 重启重放均通过，用户响应可直接审阅 backup 与 cursor，待 E1/E3 copied-workspace Gate |
| Observe | Dashboard、mobile、committed event observers | C02/C17 | `READY` | plugin `bac337f` 已完成 pure-v3 typed committed observers、generation-owned workspace、Dashboard/Mobile exact binding、candidate discard/formal rebuild 与全局错误 hook 逆序清理；按插件 CI 入口复跑 15 tests 通过，独立实现复审无 P0/P1。待 E1 exact lock 与复制 workspace write-set Gate |
| Setup Helper | command | C11 | `READY` | Core `78e50d4d` + plugin `65770db`；`/chatid` 与 `/myid` 走 committed registry，在 Session/模型 admission 前短路，installed candidate→formal 晋升与 Root/validation cleanup 已通过，待 E3 |
| Status Commands | mobile、command、只读 Session projection | C11/C17/C24 | `READY` | Core `cb2011b4` + plugin `eb245ad`；真实 Manager command/Mobile/ledger oracle 通过，待 E3 复制 workspace |
| Feed MCP | Skill、MCP、proactive source | C09/C12/C15 | `READY` | Core `78e50d4d` + plugin `29919dc`；在 `b4a8626` 行为迁移上补齐 SQLite 首次 WAL 并发初始化，真实 Manager/stdio、exact source lease、typed empty fetch、candidate data 排除、进程内回滚与进程崩溃后重启迁移均通过，待 E2/E3 |
| Feishu | channel | C14 | `READY` | plugin `b693404` 已完成 pure-v3 exact channel binding、credential redaction、provider identity、reply-aware inbound、control、preview/final、受限 provider host、流式附件总量和 UNKNOWN cleanup；32 个 Feishu tests、Pyright/contract/compileall/diff-check 与独立复审通过，待 E3 recording adapter 与受控 Feishu provider Gate |
| Fitbit MCP | MCP、managed process、proactive source、mobile | C12/C13/C15/C17 | `READY` | Core `78e50d4d` + plugin `f3fd6ee`；真实 formal monitor/MCP 与 candidate recording route 已验证只读 typed empty、写工具拒绝、exact Root 重建、敏感数据排除、显式 v2 数据迁移的进程内回滚与 Core 进程崩溃后重启恢复；40 个 Python、12 个面板测试、contract、Pyright、compileall、diff-check 与独立 review 通过，待 E2/E3 |
| Steam MCP | Skill、MCP、proactive source | C09/C12/C15 | `READY` | Core `78e50d4d` + plugin `2c492d7`；真实 stdio formal/candidate→promote、recording 零凭证/网络/DB、exact proactive catalog、显式 v2 数据迁移与 cleanup 已通过，待 E2/E3 |
| QQBot | channel | C14 | `READY` | plugin `d4bf1ed` 已完成 pure-v3 exact channel binding、provider identity、control、input-notify/preview/final 分离、附件 fail-closed、受限 media host 与取消后 UNKNOWN；36 个 QQ tests、Pyright/contract/compileall/diff-check 与独立复审通过。待 E3 recording adapter 与受控 QQ provider Gate |
| Proactive Feedback | Dashboard、mobile、committed event observers | C02/C17 | `READY` | plugin `83d6eb7` 已完成 pure-v3 committed input/outbox、candidate read/write 拒绝、ordered user IDs、重启重放与 session 公平轮转，并物理移除最后一个 v2 固定查询方法名；30 个 Python、5 个 Node、Pyright/API v3 contract/compileall/diff-check 通过，主任务复核 3 个公平性/取消 oracle 通过。待 E1 exact lock 与进程崩溃 grouped Gate |
| Huayue Skills | Skill roots | C09 | `READY` | plugin `1171904`；pure-v3 module-level skill roots、contract 与 Pyright 已通过，待 E3 |

### 4.3 In-tree plugins 与保留族群

| 插件实现 | 目标 | 依赖 Core seam | 状态 | 最终组合批次 |
|---|---|---|---|---|
| Akasha | pure v3，保持 Memory engine、Dashboard 与 mobile recall | C08/C17/C18 | `READY` | Core `713c6d9d` 已完成 tool-chain sidecar、pure-v3 lifecycle/Mobile/Dashboard、统一 memory root、symlink fail-loud 与 bounded Mobile Inspector detail；独立复审无 P0/P1，48 个 Memory/Inspector/Akasha 回归、5 个 Mobile Node tests、Pyright 0 errors、compileall/diff-check 通过。待 E1/E4 copied-workspace Gate |
| Default Proactive | v3 薄入口 + 原内部 runtime | C15/C20 | `PRIVATE_LEGACY_RUNTIME` | Core `1968e503`；只允许 default family 的 exact Core-private catalog/Host 使用，外部同名/re-export/symlink fail-loud，待 E3/E4 |
| Proactive Flow | Default 族私有实现 | C20 | `PRIVATE_LEGACY_RUNTIME` | Core `1968e503`；仅作为 default family 私有 member，待 E3/E4 |
| Drift Flow | Default 族私有实现 | C20 | `PRIVATE_LEGACY_RUNTIME` | Core `1968e503`；仅作为 default family 私有 member，待 E3/E4 |
| Wake Proactive | v3 薄入口 + 原内部 runtime | C15/C20 | `PRIVATE_LEGACY_RUNTIME` | Core `1968e503`；只允许 wake family 的 exact Core-private catalog/Host 使用，外部同名/re-export/symlink fail-loud，待 E3/E4 |
| Wake Proactive Flow | Wake 族私有实现 | C20 | `PRIVATE_LEGACY_RUNTIME` | Core `1968e503`；仅作为 wake family 私有 member，待 E3/E4 |
| Wake Drift Flow | Wake 族私有实现 | C20 | `PRIVATE_LEGACY_RUNTIME` | Core `1968e503`；仅作为 wake family 私有 member，待 E3/E4 |

Default Memory 是第 8 个 in-tree 实现，已列在 4.1；本节列出其余七个。六个 Proactive
实现最终不得继续继承通用 `Plugin`、声明 `api_version = 2` 或让
`PluginManager` 保留 `proactive_*()` 固定聚合。允许保留的是领域 runtime、状态机、prompt、
dedupe、ack、cursor、hazard 和原数据库协议。

### 4.4 GitHub Watcher

| 项目 | 状态 | 解除阻塞条件 |
|---|---|---|
| canonical source 与公开凭据审计 | `READY` | canonical repo 已确认为 `kachofugetsu09/github-watch`，fleet exact head `7334ae5f7a8a7ad3642b4d42256de65cac8a7eec`；tracked inventory 未发现 PEM/private key 或私有 artifact，待最终 E3 固定报告 |
| v3 迁移 | `READY` | 纯 v3 `BACKGROUND_JOBS + TOOL_CATALOG + AFTER_TURN_COMMITTED`；candidate 不读 PEM/不建账本，formal invocation 才取得 programmatic Turn port 与插件数据 |
| 行为 Gate | `READY` | exact Core `00b13940` clean Gate 已证明初始/晋升后正式 job 各准入一个 Turn、candidate 零 external effect、Tool/listener/catalog 与 Root cleanup；待 E3 专用远端测试仓库的只读/受控写 probe |

GitHub Watcher 只以上述 canonical source exact head 进入最终 fleet lock，不允许通过复制 cache 制造候选。

## 5. v2 删除账本

删除顺序沿用[最终迁移地图](plugin-v3-final-migration-map.md#7-v2-物理删除清单)的 A～J。
每批删除 PR 必须先执行 consumer scan，再记录删前 owner、最后 consumer、替代能力和 Gate。

| 批次 | 对象 | 状态 |
|---|---|---|
| A | Default Memory legacy data name | `READY`（Core `3f25f767` 已删除正式/candidate 对 `workspace/observe/recall_inspector.jsonl` 的读取与 hard-link；旧文件不删除，exact generation data root 与 discard write-set 由 51 个定向回归锁定，待 E1/E4） |
| B | legacy assistant metadata slots | `READY`（Core `695f35ac` 已删除 legacy assistant metadata slot 出口，待 E1/E4） |
| C | legacy Dashboard ABI | `READY`（Core `92ac1713` 已删除 v2 backend/import 路径，待 E1/E3/E4） |
| D | ToolHook ABI、catalog 与 traces | `READY`（Core `0940e9e7` 已把 execution contract/Executor 收归 typed Tool owner并物理删除 `agent.tool_hooks`；最终 Tool Gate 待跑） |
| E | v2 static-active / stable-health exception | `READY`（stable-health exemption 已删除，正式 Root 始终必须 ready；`static_active/is_active` 是 Akasha/Default Memory 二选一所需的 v3 static projection，不属于删除对象） |
| F | `PluginContext` | `READY`（通用 `PluginContext` 文件与 production consumer 已物理删除；memory-engine factory 不是该 ABI） |
| G | v2 doctor / class discovery | `READY`（doctor、Manager 与 installer 都只接受静态 manifest + v3 namespace，旧 class discovery 已删除） |
| H | `Plugin` base、registry、Manager 固定能力方法 | `READY`（`Plugin` base/registry、fixed contribution consumer、workspace MCP owner与 Channel fallback 已物理删除；只保留 Default/Wake exact builtin 私有 proactive bridge） |
| I | RuntimeSnapshot v2 固定字段 | `READY`（snapshot 只保存 generation、Root/topology 与 typed capability catalog；workspace MCP generation 与 health exemption 已删除，待最终零 consumer scan） |
| J | v2 lock、Gate 和 runtime 双路径 | `READY`（v2 lock/Gate/CI 已删除，static fleet 增加 legacy class/fixed-method AST mutant；待同一 clean head 的 E1～E4 报告） |

最终 production scan 必须证明：普通插件无法通过 import、动态 discovery、manifest 或 cache
重新进入 v2；测试 fixture 若保留历史格式，必须位于明确的 migration-test namespace。

## 6. 验证分层与克制的 E2E

### 6.1 每个 PR 都运行，但不启动完整服务

1. 纯函数和领域行为等价测试；
2. 事件、Service、Effect、candidate/write-set 的 kernel oracle；
3. 真实 `PluginManager` install/lease/dispose 测试；
4. 修改文件的静态检查与仓库 contract；
5. 受影响族群的 exact-commit Gate。

### 6.2 只保留四个集中 E2E

| 批次 | 一次覆盖的组合 | 主要 oracle | 运行时机 |
|---|---|---|---|
| E1 Passive/Data/Mobile | Akasha、Default Memory、Citation、Meme、Emotion、Observe、Proactive Feedback、Plugin Undo | prompt/recall/metadata/media、bounded mobile query/lease、SessionDB 普通 append-only；显式 `/undo` 按 `control_turn_id` 原子删除完整 interaction、embedding/reference 协调与恢复；Akasha/plugin-data write-set | 被动与数据族全部 `CANDIDATE` 后一次 |
| E2 Tool/MCP/Process | Restore、Safety、Loop Guard、Calendar/Feed/Fitbit/Steam | transform→authorize→invoke、readiness、端口、取消、process cleanup、受控外部只读调用 | MCP/process 族全部 `CANDIDATE` 后一次 |
| E3 Fleet/Channel/Proactive | Commands、Feishu/QQBot recording adapters、Daynight、Emotion、Calendar/Feed/Fitbit/Steam sources、Huayue Skills、Default/Wake 薄入口 | full boot、catalog、candidate discard/promote、reload；loopback channel 正向收发；固定时钟/模型/sink 的 enabled proactive empty/skip/source/model/delivery/restart | 全插件接线完成后一次 |
| E4 Production Rehearsal | E1～E3 的 exact heads + 复制的真实 workspace，WebUI-only | DB integrity、完整 write-set、artifact/pointer、restart、stop cleanup、恢复证据 | 删除 v2 后最终一次 |

E1～E3 使用一次性 workspace 和受控端点。E4 只能使用经过校验的副本；正式 hua-home workspace、
正式 channel credential、正式 proactive sender 和正式端口均不进入本任务。

### 6.3 最终集中证据

最终交付只接受同一 clean Core head/tree 和同一 fleet lock 的下列报告；E4 会读取并复核前三组
报告与 Passive WebUI 报告的 exact identity，不接受旧 head、blocked scenario 或 cleanup 残留：

- `fleet.json`：20 个 external pure-v3 exact source、静态 manifest、v2 consumer 清零和退役插件排除；
- `mobile.json`：Mobile catalog、query、exact inbound、附件 handoff 与重启恢复；
- `webui.json`：Citation/Meme、WebSocket final、opaque artifact、Session history 与停止清理；
- `e1.json`：Passive/Data/Mobile、普通 append-only、显式 Undo、进程内失败与 Core 进程崩溃；
- `e2.json`：Tool/MCP/managed process、readiness、端口、取消和进程崩溃恢复；
- `e3.json`：full fleet、Channel/MessagePush、Command、Proactive 与 GitHub Watch controlled remote；
- `e4.json`：复制真实 workspace/config/plugin-home 后的 SQLite integrity、write-set、artifact/pointer、
  restart 与 stop cleanup；源 workspace 在演练前后保持一致。

报告存放在 CI artifact 或本次交付的临时证据目录，不提交其中的 workspace、凭据、数据库或日志。
hua-home 正式切换不是本清单的一部分，必须另行获得维护者授权。

### 6.4 数据安全 oracle

| 状态 | 正常允许变化 | 本迁移禁止变化 | 恢复/验收证据 |
|---|---|---|---|
| `sessions.db/messages` | E1/E4 测试 session 只追加测试 user/assistant rows | 既有正文 UPDATE/DELETE、跨 session 混写 | SQLite backup、integrity、row/write-set、session identity |
| Plugin Undo interaction | 只在 disposable copy 上由显式 `/undo` 和精确 `control_turn_id` 操作；删除前生成不可覆盖的完整 SessionDB backup；同一事务删除完整 user+assistant interaction 及匹配 `message_embeddings` 并回滚 cursor；Memory2 通过 superseded/active 与 `memory_replacements` 留替代证据；Akasha/pending 旧引用失效或重建 | 普通 turn 删除、仅删一侧 message、错误身份/cascade、覆盖旧 backup、硬删 Memory2 事实、遗留 Akasha/pending 引用、部分失败伪装成功；非目标 message、附件与 `seq` high-water 不得变化 | backup 路径与完整性、目标/保留 message IDs、embedding 差集、旧/新 cursor、非目标 rows/附件/`seq`、memory replacement、Akasha/pending 结果、audit 与失败恢复 |
| `memory2.db` / Markdown | 仅显式测试策略允许的新增 | 既有事实覆盖、删除、自动清理 | 前后摘要、表计数、档案备份 |
| `consolidation_writes.db` | 测试 compaction source 追加幂等 receipt | 既有 receipt 删除、同 key 内容漂移、跨库失配 | integrity、source_ref/kind、payload 与 source-plan digest |
| Akasha sidecars | disposable copy 可按固定输入重建 | 正式 sidecar 写入或残缺图替代成功 | 输入 hash、embedding coverage、parity |
| plugin-data | 测试 generation 在自己 root 内增加 | candidate 写 stable root、卸载删数据 | 路径归属、tree digest、candidate discard |
| proactive/wake/drift DB 与 Markdown | E3/E4 在显式启用时按原状态机更新测试副本；获授权 job 原子更新测试 `PROACTIVE_CONTEXT.md` | schema 偷迁、覆盖规则面板、重复发送、ack/cursor 丢失、提前清 pending | schema/file identity、continuity rows、recording sink、restart parity |
| plugin artifacts/pointers/runtime journal | 安装事务增加 immutable candidate、journal/rollout fact，提交后改 pointer/manifest | plugin/runtime 改 artifact bytes、失败后残留 pointer、删审计 journal | artifact digest、manifest、journal、rollout fact、stable/latest identity |
| credentials/config | 一次性测试配置可创建 | 输出、提交、复制或改写正式 secret | 文件 inventory、权限、脱敏报告 |

## 7. 多 Agent 分工规则

- 主 agent 是本清单、Core capability 合同、集成分支和最终 Gate 的唯一 writer。
- 并行 agent 只在 Core seam 稳定后接收按仓库隔离的插件迁移；一个 agent 一个
  repository/worktree/branch，必须先记录 base commit 和 allowed paths。
- 适合并行的单位是彼此不共享权威文件的插件仓库，例如 lifecycle 插件组与 MCP 插件组。
- Core seam、同一外部插件、跨仓 lock、迁移清单和最终 E2E 不能由多个 writer 并发修改。
- 每个 agent 必须提交 clean handoff commit；主 agent 复核 diff、测试和真实行为后才更新本表状态。
- 如果多个插件等待同一个未完成 Core seam，则保持等待，不让 agent 在插件仓复制临时兼容层。

## 8. 实施波次

- [x] W0：提交本清单，冻结状态定义、数据边界和 E2E 批次；
- [x] W1：完成 C16、C18、C19，并复核 C01～C10；
- [x] W2：以真实 consumer 依次完成 C11～C17、C21～C22；
- [x] W3：并行迁移 lifecycle/metadata/command 插件；
- [x] W4：并行迁移 MCP/process/channel 插件；
- [x] W5：迁移 Akasha、Proactive Feedback，并建立 C20；
- [x] W6：执行 A～I 删除批次，运行 production v2 consumer scan；
- [x] W7：集中运行 E1～E3，关闭所有行为差异；
- [x] W8：执行 J、运行 E4 与独立只读 review；
- [x] W9：对账 exact heads、CI、报告与文档，只在全部为 `READY` 后汇报。
