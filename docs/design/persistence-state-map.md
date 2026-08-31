# Akashic Agent 持久化状态地图

- 状态：accepted target / implementation
- 核对基线：`origin/main@31b976d82cbd5766e6450d7e287ceda71d9b7573`
- 核对日期：2026-08-07
- 目标读者：维护者、coding agent、迁移与备份实现者、评审者
- 关联条款：STA-001～STA-003、CTX-001、SES-001～SES-006、MEM-001～MEM-009、PLG-001～PLG-013、WSP-001～WSP-004、SCH-001～SCH-002、PRO-001～PRO-002、BAK-001

## 1. 这份地图怎样使用

这份文件不只回答“落了哪些文件”，还回答每类数据怎样增加、怎样原位更新、怎样逻辑失效、什么条件才允许物理减少。它先陈述代码事实，再提出设计意图推断。两者不能混用：

- **F（fact）**：能从当前代码、schema 或已存在的仓库说明直接确认。
- **I（inference）**：根据代码结构推断的产品意图，等待维护者确认。
- **G（gap）**：当前实现或文档没有给出完整答案，不能自行补齐。
- **T（target）**：已经由 accepted 决策确认、但当前代码尚未实现的目标状态；不得写成现状证据。

本文件不是删除白名单。一个对象被标为派生或诊断数据，不等于普通 refactor 可以删除；删除、重建、迁移和保留期仍需明确 owner、备份、完整性检查和用户授权。

## 2. Workspace 到底是什么

`<workspace>` 是一个显式选中的 **Akashic 运行实例工作区**，也是该实例主要的持久数据根。启动时按 `--workspace PATH`、`AKASHIC_WORKSPACE`、`config.toml:[runtime].workspace` 的优先级选出一个目录。此后，Akashic 在这个目录里继续同一批会话、记忆、调度和自主流程状态。

它不是源码仓库，也不是 Git checkout 或 Git worktree：

```text
Git repository / worktree
└── 代码、测试、项目工作手册、Git 历史

Akashic <workspace>
├── 用户与 Agent 的对话：sessions.db、uploads/
├── 记忆：memory/*.md、memory2.db、akasha.db
├── 旧主动岛保留数据：proactive.db、wake_proactive.db、drift/drift.db
├── 操作状态：schedules.json、proactive_quota.json
├── 插件运行数据：plugin-data/
├── 插件能力投影：skills/、drift/skills/ 软链接
├── 待迁移兼容路径：mcp/servers/、手工 skill 目录
├── 诊断证据：observe/、trace、subagent-runs/
└── 当前进程控制：lock、PID、readiness、socket、token
```

切 Git 分支、删除代码 worktree 或做代码 refactor，不应改变 Akashic workspace。反过来，迁移或恢复 workspace 也不等于迁移源码和 Git 历史。后文出现裸词 `workspace` 时，都指这个运行数据根。

workspace 仍不是完整运行环境的全部。模型 Provider credential 已随 connection 进入 workspace；显式主配置、旧或非模型全局凭据、全局插件安装清单、插件代码缓存和外部插件 canonical source 仍可以位于它之外。这些对象要通过 companion manifest 单独列出，不能靠猜测 HOME 路径补齐。

## 3. 用“增、改、减”阅读持久状态

本文固定使用四种变化，避免把所有写入都含糊地叫“更新”：

- **增加**：INSERT 新行、追加记录或创建新文件；旧事实仍然存在。
- **原位更新**：同一身份的记录改变字段、状态或当前值；要列出允许变化的字段和状态机。
- **逻辑减少**：旧记录被 supersede、消费、终结或标成 inactive，但物理内容仍在，可继续审计。
- **物理减少**：DELETE 行、删除文件、截断日志或用较少内容覆盖旧内容；必须有明确 owner、触发条件和恢复办法。

代码里存在 `DELETE` 方法，只证明存储层具备能力，不证明普通运行、重构或后台清理拥有调用授权。没有写明物理减少协议的对象，默认不得自动减少。

### 3.1 对话与附件

| 对象 | 正常增加 | 允许的原位或逻辑变化 | 允许物理减少的条件 |
|---|---|---|---|
| `sessions.db/messages` | 每次持久化一批新消息时 INSERT；同一 session 的 `seq` 单调增加且不复用 | 正常收发不改旧正文；当前代码存在显式 `update_message`，但它是否属于获授权产品语义仍待确认 | 只有用户主动撤销消息或删除会话/线程，管理命令才能 DELETE；带 `control_turn_id` 的显式 interaction 只能整组原子撤销，并声明目标、cascade、备份和审计 |
| `sessions.db/sessions` | 新 session INSERT；已有 session 的新消息仍追加到 `messages` | 允许更新名称、时间、高水位、当前 compaction generation 和主动流程时间等 session metadata；`last_consolidated` 只能由 checkpoint 提交事务推进 | 只有用户主动删除 session/thread 时，由 session 管理边界级联删除 |
| `sessions.db/channel_identities` | 首次 v3 channel identity admission 为 `(channel, identity)` INSERT 唯一 recipient；旧 Session metadata 只在 `channel_identity_migrations` 尚无对应 marker 时按稳定顺序非破坏 seed | provider identity move 只原位替换同一唯一行的 `chat_id/updated_at`，并与目标 Session metadata 在同一 SQLite transaction 提交；重启与主动发送只读该表，不再扫描重复 metadata 裁决 | 只有用户显式删除对应 `channel:chat_id` Session 时，Session 删除审计事务才级联删除指向该 recipient 的行；删除前整库 backup 同时保存 identity，普通 turn、插件卸载和 candidate discard 无权减少 |
| `sessions.db/channel_identity_migrations` | 每个 channel 首次 legacy seed 或首次正式 identity write 时 INSERT 一条 durable marker | marker 不原位更新；它证明旧 Session metadata 已永久失去路由裁决权，即使当前 identity 表为空也不得重新 seed | 普通 Session 删除、插件卸载和 workspace 维护不得删除；只随用户明确删除/恢复整个 workspace 而减少，整库 backup 是恢复证据 |
| `sessions.db/attachments` + `attachment_imports` | Core Channel artifact import 先固定 intent，再以不可覆盖文件发布和 ready row 增加 immutable artifact | import 只按 `prepared → file_published → artifact_committed` 推进；ready metadata、hash、size 与 storage key 不原位改写 | 当前没有普通自动减少协议；Session/插件删除只减少 binding，不删除 artifact；物理减少必须是用户明确的数据管理操作并先做引用扫描、备份与 hash 验证 |
| `sessions.db/message_attachments` | user/assistant message 与 ordered artifact IDs 在同一 SessionDB transaction 增加 | binding 不原位改写；`extra.attachment_ids` 必须与表完全一致 | 仅随用户明确删除对应 message/session 的既有级联减少；不得连带删除 `attachments` row 或 artifact bytes |
| `sessions.db/session_compactions` | 每次 committed compaction INSERT 新 generation，保存 lineage、source_ref、canonical `source_plan_digest`、tail、summary、usage 和模型容量；included 必须与 receipt digest 相等，excluded 仍写 session-local digest ledger | 只允许设置 `invalidated_at/invalidated_reason` 逻辑失效字段；generation、provenance、source_plan_digest、summary 和 tail 不原位改写；缺列非空旧 schema 不回填 | session 删除 cascade；用户删除 interaction 时失效命中 generation 及 descendants，物理删除没有普通运行协议 |
| `sessions.db/session_compaction_prepares` | Included compaction 在跨文件 receipt/Markdown effect 前 INSERT 一条 incarnation-scoped durable prepare，固定 source seq/message IDs 与 retained tail | 同一 source_ref/generation 只能幂等复用完全相同的 fence identity；prepared_at 不改变 source identity | 无 receipt 的 pre-effect orphan 只由 compaction recovery 清除；pending 时 message/interaction/session destructive mutation 必须阻断并返回带 audit identity 的 409；session 管理删除可按同一 SessionDB 事务 cascade prepare |
| `sessions.db/turns` | 新 turn 先 INSERT 为 queued | 按状态机更新 items、usage、error、final response 和终态；这是同一 turn 的进展，不是改写对话正文 | 当前只有显式 thread/session 删除路径可以减少；是否另设 retention 仍待确认 |
| FTS 与 `message_embeddings` | 由新消息触发建索引或计算向量 | FTS 可以从正文重建；embedding 迁移属于独立流程。Akasha 确定性重建必须复用 sessions 中已存向量 | 用户撤销/删除原始消息时同步减少，或由独立索引维护流程重建；上下文裁切无权删除 |
| `uploads/` | 每个新附件写入新的 UUID 文件 | 当前没有生产代码原位改写附件；消息引用决定附件仍然有效 | 消息仍引用时必须保留；当前没有引用计数、级联删除或 GC 协议，因此不得按年龄、当前 prompt 是否使用或代码清理自动删除 |
| `uploads/artifacts/` | Core Channel import 以固定 artifact ID 创建不可覆盖 regular file，并 fsync 文件与目录 | 已发布文件不可原位覆盖；SessionDB ready row 是可见性 owner，orphan 只由 audit 报告 | 当前没有普通自动删除协议；candidate discard、Turn 失败、Session 删除、插件卸载均无权删除；未来 GC 需单独合同、完整引用扫描与可恢复备份 |
| `mobile.db/mobile_attachment_imports` | Mobile message 首次引用 finalized upload 时按 `(device, session, client_message_id, ordinal)` 固定 mobile attachment 与 Core artifact ID | 只按 `prepared → artifact_committed → message_bound` 推进；跨进程重放必须解析到同一 artifact，失败保留非终态 owner | 普通消息失败、插件重载和 artifact import 不删除；Mobile owner 的明确设备/数据删除协议或整个 workspace 恢复才可减少，且不得反向删除 Session artifact |
| `backups/interaction-deletions/sessions-<uuid>.db` | 每次 interaction 撤销前通过 SQLite online backup 创建完整 SessionDB 快照，并以 `integrity_check` 验证 | 已发布快照不可原位更新；路径随删除响应与审计日志返回 | 当前没有自动 retention；只有名称明确、目标精确的备份管理操作可以删除，不能由普通清理或下一次撤销覆盖 |

这里所说的“`sessions.db` 默认 append-only”，精确含义是：**数据库中的完整对话正文 `messages` 在正常运行中只追加，只有用户主动撤销消息或删除会话才允许减少。** SQLite 文件整体并非字面只追加，因为 `sessions` 元数据、`turns` 状态和派生索引都有受约束的 UPDATE/重建路径。当前 dashboard 已暴露旧消息编辑接口；是否保留这项 UPDATE 例外，是需要维护者明确回答的实现与意图差异。

### 3.2 记忆

| 对象 | 正常增加 | 允许的原位或逻辑变化 | 允许物理减少的条件 |
|---|---|---|---|
| `MEMORY.md`、`SELF.md` | Markdown memory 普通插件消费 committed compaction fact，按文档发布下一版 | 每个文档以独立 draft、before-image、atomic replace 和 applied receipt 收敛；在线投影不能隐式删除既有事实 | 没有普通自动减少协议；未来移除事实需要显式 tombstone、来源、理由和独立管理合同 |
| `VEDA.md` | 新 workspace 初始化或旧 workspace 一次性迁移只在缺失时创建默认人格 | Main Agent 仅在用户明确要求时原子更新；`main.py veda-reset` 先备份原始字节再原子恢复版本化默认 | 正常运行没有删除协议；migration revert 仅可删除该 migration 创建且此后未修改的文件 |
| `PENDING.md` / `PENDING.snapshot.md` | 在线路径不再增加 | Markdown plugin 启动迁移先把两份原始文本和 digest 写入 immutable receipt，再确定性合入 MEMORY | 合入和 `PENDING.retired.md` 发布成功后才清空旧文件；任一步失败由 receipt 重启收敛 |
| `RECENT_CONTEXT.md` | 旧版本曾由近期会话生成投影；新安装不创建 | 新语义不读取、不原位更新 | 仅由 DAG 最后阶段 R06 在备份、完整性检查和 config 归档成功后删除；失败恢复原文件 |
| `consolidation_writes.db` | compaction plugin 为 `session_compaction_receipt` INSERT immutable crash-recovery receipt | v4 保存 source-plan digest、实际 runtime/model/usage 并发布新 profile fact；旧 v3 只恢复 ledger 和保留审计；同 key 内容漂移 fail-loud | receipt 是恢复与审计证据，当前没有自动删除或跨库 cascade |
| `markdown-profile-writes.db` | Markdown plugin 按 `source_ref + memory/self kind` INSERT model、draft、before-image 和 applied receipt | 每个文档独立推进；文件或 receipt 单侧领先时重启确定性收敛 | 当前没有自动删除协议；它是 MEMORY/SELF 和旧 PENDING 的恢复证据 |
| `memory2.db/*` | 无当前 writer；经典记忆退出前曾写入结构化记忆和替换关系 | runtime 不再读取、导入或更新 | 只作为历史归档备份，不自动删除 |
| `akasha.db` 与 `akasha-v2-index.db` | 固定算法读取 `sessions.db/messages` 和已有 `message_embeddings`，增加图、激活和查询记录 | 可以用同一组输入确定性重建；用户整组撤销 interaction 后由 Akasha owner 串行全量替换；只读 Inspector 从既有表派生视图，不新增状态；重建不调用 LLM，也不重新解释历史 | 只能由显式 sidecar rebuild/maintenance 或 interaction 撤销协调流程替换；embedding 缺失或模型不匹配时完整重建必须失败，不能跳过后声称成功 |

### 3.3 自主运行、扩展与控制状态

| 对象 | 正常增加 | 允许的原位或逻辑变化 | 允许物理减少的条件 |
|---|---|---|---|
| `proactive.db` | H3 后当前 runtime 不再写；H2 historical decoder 只读旧 tick、step、delivery 与 continuity | 只有名称明确的 owner handoff 可以推进连续性；当前无写 owner | 不随代码升级清理；delivery dedupe、cooldown 和连续性状态在交接前阻止 activation |
| `wake_proactive.db` | H3 后当前 runtime 不再写；H2 historical decoder 只读 runs、observations、hazard monitor，inventory 读取 active continuity | reservoir/ACK 只由目标 source adapter 交接；其余 continuity 当前无写 owner | 不随代码升级清理；未交接 ACK、tombstone、quarantine、消费、hazard/context/timer 状态阻止 activation |
| `drift/drift.db` | run、step、journal 持续追加 | continuum、cursor、global note 和 self state 原位更新 | 日志可以另定 retention；cursor、journal 和下一轮选择所需状态必须恢复，不能按临时 trace 清空 |
| `schedules.json` | 获授权的 add 创建 job | reschedule 更新同一 job；one-shot 执行完成或错过 grace 后保留为 `enabled=false` 逻辑终态；整份 JSON 以 candidate 原子替换 | 只有明确 cancel 操作可以移除 job；损坏文件不能解释成用户取消了全部任务 |
| `proactive_quota.json` | 动作增加当前窗口计数 | 新窗口滚动时重置计数并更新当前状态 | 这是当前计数器的状态迁移，不是用户历史删除；损坏不得静默重置 |
| `PROACTIVE_CONTEXT.md` | H3 后初始化与 runtime 均不再创建或写入；H2 按 exact bytes/digest 只读归档 | 用户或获授权文件工具仍可修改既有文件；Wake archive consumer 未接通前 activation `BLOCK` | 当前没有自动清空或删除协议；代码升级不得用默认模板覆盖已有内容 |
| `proactive_pending.md` | H3 后旧 paired-documents writer 已删除；H2 只盘点既有非空内容和 intents | 只有后续名称明确的目标领域 owner handoff 可以改变 | 非空 pending 或 active intent 在 owner 缺失时阻止 activation；代码升级不减少正文或 intent |
| `plugin-data/` | 已激活插件在自己的 opaque 目录增加数据 | 由插件 schema 和 owner 决定 | 普通卸载只删除代码和能力投影，保留数据；永久删除必须使用名称不同的用户操作、影响预览、备份和再次确认 |
| `runtime/plugin-reloads.sqlite3` | 每次热重载增加 transaction 与阶段事件；首次启动 candidate/formal runtime 前固定 old/new snapshot、old/new generation、base/candidate artifact pointer 与 `runtime_owner_boot_id`；已有 candidate 时 stable watchdog 加入同一 transaction，不另建 owner | 同一 transaction 按状态机更新 phase、failure resource、recovery target/action、单调 attempt 和 retry receipt；failure 只允许 `cleanup_failed → degraded` 单调强化且 target 不可覆盖，终态只能由 exact Host retry 成功后收束；同 boot 不做进程清理，新 supervised boot 只按已记录 old boot ID 恢复 | 当前没有自动 retention；恢复和事故审计仍依赖的记录不得自动删除，artifact/Host tombstone 只在对应 retry receipt 成功后减少 |
| `runtime/plugin-jobs/outcomes.sqlite` | generation-scoped plugin job 首次 admission INSERT semantic job/event/interval identity、exact snapshot/plugin/model generation、artifact/source/handler/lifecycle identity 与 queued 状态 | 同一 invocation 只按 queued→running→terminal/retry_pending 状态机更新 attempt、phase（handler/provider/documents）、error 与 result digest；跨 generation redelivery 复用同一 semantic key，不新建第二次 effect；documents phase 只由 ActivityHost forward recovery | 当前没有自动 retention；这是 event dedupe、取消与 crash recovery 证据，普通插件卸载、重载或日志清理不得删除。workspace 备份应以 SQLite online backup + integrity_check 保存；只有后续名称明确的 retention/插件数据管理操作可减少 |
| `runtime/deliveries/settlements.sqlite` | Core 为每个 accepted Turn INSERT 一条 immutable delivery envelope 与 stable logical id | 只按 `prepared → provider_started → delivered → projected → settled` 前向更新 exact binding、provider receipt、Session message 与 opaque domain receipt；provider 调用中断进入 `uncertain`，明确拒绝进入 `rejected`；候选只读检查 `prepared/delivered/projected` 的 target service 仍可解析 | 当前没有 DELETE 或自动 retention；Core ledger 是 provider effect、Session projection 与领域 settle 的恢复证据。备份应覆盖数据库、WAL/SHM 并使用 SQLite online backup + `integrity_check`；只有后续名称明确的 delivery retention 操作可以减少 |
| `runtime/proactive-documents/intents/<invocation-id>/` | `ProactiveDocuments.prepare_pair()` 在 DB effect 前创建，保存两份 old state（bytes 或 absent marker）、完整 new bytes、expected digest、idempotency key 与 fsync receipt | 无 DB receipt 时只允许 abort 并保持正文原状态；有 DB receipt 时只允许 ordered replace/forward recovery；partial replace 依据 old bytes 恢复两份原始状态 | commit/abort terminal receipt、目标 digest 与目录 fsync 均完成后才能删除该 intent；启动恢复不得按年龄猜测 orphan。workspace 备份必须与 outcomes.sqlite、两份 Markdown 一起覆盖该目录 |
| `runtime/plugin-rollout-fact.json` | turn 后 install/uninstall 产生一条待反馈事实 | 新结果原子替换尚未消费的旧事实 | 下一次非 programmatic 用户 turn 注入后删除；它是可重建反馈，不是会话或长期记忆 |
| `runtime/plugin-skill-links.json` | legacy adoption 或首次插件 Skill/Drift skill 投影时创建 ownership registry；每次链接切换先原子写入含 old/new 的 pending journal | 目录项切换后原子提交 `links` 并清除对应 pending；进程重启只在实际链接仍等于 old 或 new 时收敛，用户文件、未登记软链接和第三种状态 fail-loud | 只有插件 disable/uninstall 或 generation 切换的 linker owner 可以删除已登记且 target 匹配的投影链接并移除对应 ownership；不得删除用户文件、普通目录、未登记链接或外部 canonical source。registry 是重建与恢复证据，当前没有整文件自动删除协议 |

H4 后 Core 配置、Setup、Prompt、Dashboard 与 Mobile Runtime Inspection 均不再读取旧主动岛状态。
任意空或非空 `[proactive]` 在打开 workspace-backed Model Registry 前失败；这只删除代码入口，
不授权修改上述旧数据库、Markdown、quota 或 plugin-data。
| `migrations.sqlite3` | Yoyo 在 migration step 成功后记录唯一 migration ID | 已应用回执保持不变；新增迁移只追加新的成功回执 | runtime 没有删除或回滚回执权限；只随用户明确删除整个 workspace 而减少，恢复依赖 workspace 备份与 SQLite 完整性检查 |
| `model-registry.sqlite3` | onboarding 或设置事务增加含 credential payload 的 connection、model 和 role binding，并增加单调 revision；`model_definitions.context_window`/`max_output_tokens` 与各自 source 保存模型 capability snapshot | connection 的 key/token、Base URL、模型字段和角色绑定可原位更新；Codex token refresh 不增加模型 revision，其余成功模型事务增加 revision，旧 execution generation 只在 lease 归零后失效。预算 owner 只读取当前 generation 的 `context_window`、`max_output_tokens` 及字段来源；遗留 `effective_context_percent`/`compaction_trigger_percent` 列仅为 v1 schema identity 保留，完全惰性，不是配置或 capability source | 只有独立模型/来源删除操作可以减少；被 role 或 session 引用时必须拒绝，普通模型切换不得 cascade；数据库、WAL/SHM 与备份均按 secret 使用 `0600` |
| `data/mobile/master-keys.json` | 文件型密钥 provider 初始化或轮换时追加随机 master key；离线迁移可按既有 ID 导入同一密钥 | 完整集合以 `0600` 原子替换发布；同 ID 同内容导入幂等，不同内容 fail-loud；旧 key 继续支持历史 keyset 回滚 | 当前没有自动删除协议；只能由名称明确的移动身份重置或密钥退役操作在备份、引用扫描和恢复验证后减少；Mobile key store owner 与 keyset manifest 提供恢复证据 |
| `sessions.metadata.model_selection` | 会话首次固定 model ref/effort 时增加版本化对象 | 用户切换 model/effort 时仅更新该对象；旧字符串 override 在下一次显式选择时升级 | 用户选择“跟随默认”时只移除该 metadata 键；不得改写或减少 messages |
| 插件贡献的 Skill/Drift skill | 插件 source 持有 skill 正文；安装把版本化副本发布到 cache，generation 从 `skill_roots` 建 catalog | workspace `skills/` 和 `drift/skills/` 软链接随 active generation 重建 | 禁用/卸载插件可以移除已安装副本、catalog 和软链接；外部 canonical source 不归 workspace 或卸载流程所有 |
| 插件贡献的 MCP | 插件安装读取 `mcp_servers()` 并准备 runtime，generation readiness 通过后发布 MCP catalog | 插件升级或热重载按 generation 原子替换，旧代随 lease 排空 | 禁用/卸载插件移除 MCP catalog 和 runtime；plugin-data 不级联删除 |
| `mcp/servers/*.toml` 与手工 skill 目录 | 当前代码仍允许绕过插件直接声明或放置能力 | watcher/loader 可以热加载这些兼容内容 | 目标架构不再扩展这条路径；应迁移成插件并删除第二套 owner，迁移完成前不得把兼容目录写成 canonical 产品资产 |
| `memes/manifest.json` | workspace 初始化时创建空 manifest；Meme 插件和管理 Skill 按显式用户操作增加类别与素材 | Meme 插件按 manifest mtime 重载；Dashboard/Skill 可原位更新 manifest 和类别目录 | 仅明确的 Meme 管理动作可在备份后减少；插件卸载、candidate discard 和 Core 清理不得删除正式素材根 |
| 诊断 JSONL、`subagent-runs/` | 运行和调查持续追加产物 | 通常不原位改写 | 当前缺少统一 retention；没有策略前不得假装它们会永久存在，也不得擅自 prune 事故证据 |
| lock、PID、readiness、socket | 进程启动时创建 | 随当前 boot 更新 | 由进程生命周期 owner 在停止或重启时移除；它们不是业务事实；`.app-server-token` 作为持久 secret 单独处理 |

### 3.4 Workspace 之外的 companion state

| 对象 | 正常增加 | 允许的原位或逻辑变化 | 允许物理减少的条件 |
|---|---|---|---|
| 显式 `config.toml` | 用户增加 channel、plugin、memory 和进程配置；模型迁移后只保留 workspace registry 标记 | 由配置管理动作修改静态当前值；它可能位于源码目录或任意 `--config` 路径 | 只能由明确配置管理动作删除；workspace 迁移不能假设它一定随目录存在 |
| `~/.akashic/auth.json` | 旧安装、非模型配置或其他 workspace 可以增加 credential ID；0026 后本 workspace 的模型不再以它为 owner | 兼容 owner 可以继续更新自己的 credential；模型 Yoyo 只复制被当前 workspace 引用的值，不原位修改旧文件 | 当前 store 没有通用删除 API；模型迁移不得因当前 workspace 已复制就删除可能被其他消费者引用的 credential |
| `~/.akashic-plugin/manifest.toml` | 安装时增加 plugin/package identity；运行时加载该插件后取得 Skill/MCP 声明 | enable/disable 更新 entry | 明确卸载时移除对应 entry；这只减少安装清单和能力，不删除 workspace 内 plugin-data |
| `~/.akashic-plugin/cache/` | 插件安装在 staging 校验后发布插件代码、Skill 和 MCP runtime | 更新版本时原子替换并可回滚当前安装事务 | 明确卸载可以删除代码与能力 cache；它不是外部 canonical source，也不授权删除 plugin-data |
| 外部插件 canonical source | 用户在独立源码仓库创建和提交 | 通过该仓库自己的 Git 工作流演进 | 只受该源码仓库的用户操作管理；Akashic workspace 备份、插件卸载和 cache 清理都不拥有它 |

### 3.5 Companion 安全边界涉及的临时与可衰减状态

本节把 [0017](../decisions/0017-one-person-companion-security-boundary.md) 采用的 receipt、durable inbound handoff、MCP reservoir 和 control replay 规则落到状态地图。它们不是 `sessions.db/messages` 的删除授权，也不改变同一位用户跨渠道的连续性。

| 对象 | 正常增加 | 允许的原位更新/逻辑终态 | 物理减少条件 | owner 与恢复证据 |
|---|---|---|---|---|
| Mobile completed receipt | 有副作用或持久结果的 command admission 增加 request hash、device、状态和结果引用；只读当前快照的启动查询不增加 | `processing → completed`；无法判断真实外部效果时进入 `outcome_unknown`；request hash 不可改写 | 仅 `completed_at` 超过 7 天且清理事务成功；processing/unknown 不按 TTL 删除 | Mobile receipt store；同 ID 重放结果、external effect count、reconciliation report |
| `sessions.db/inbound_handoffs` | Mobile 消息在进入内存队列前 INSERT 完整 handoff 与 `session + client_message_id` 去重身份 | pending 期间由 MessageBus 持有 durable handoff/lane owner；canonical user 已存在时只对账、不重开 turn；永久超过单请求容量时由 Control Runtime 持久化真实 failed turn；restart 排空期间 worker 保持 accepted owner 并在同一 coroutine 等待 admission 恢复，完整进程退出则保留 durable row，由下一 boot 按有限页恢复 | completed、failed、interrupted、cancelled 都必须先把带权威 turn/client identity 的 terminal 提交到 Mobile durable inbox，再由 worker 确认 handoff DELETE；渠道、socket 前的 durable 提交或 DELETE 失败均不得减少，DELETE 的暂态 `OSError` 保留 owner 并进入 `cleanup_degraded` 重试 | MessageBus + PassiveMessageWorker + Control Runtime；handoff row、terminal turn、Mobile inbox event、三元身份 milestone、recovery report |
| MCP reservoir event | source event、cursor、score、timestamp、payload 增加；坏 item 进入 quarantine 记录 | score/ack/cursor/consumed/decayed 按状态机更新；旧池只作为衰减 wake mass | 最小驻留期已过、分数低于 decay floor，且 ack/cursor 提交与 payload 删除处在同一可恢复事务 | Wake/MCP owner；source cursor、accepted/quarantine 快照、ack/delete 提交证据 |
| Control replay ring | 每个 live turn 追加 replay event | 每 turn ring 最多 256 events/4 MiB；terminal 进入最多 5 分钟 grace；runtime reaper 按 wall clock 回收；live subscriber 不受 eviction 影响 | 每 turn或全局高水位回收临时 replay；terminal 超 5 分钟后回收；不得减少 SessionStore；索引不变量损坏必须 runtime fatal | Control owner；`replay_truncated`/`replay_expired`、snapshot、SessionDB unchanged |
| Execution spill/log | 当前 execution 追加输出或 spill 文件 | active → terminal；cleanup 未确认时保持 `cleanup_degraded` 和 owner | execution 结束且删除确认；cleanup 失败保留 path/identity，不报告已回收 | Execution owner；registry、path/size/lifetime、cleanup report |

上述状态的容量拒绝、quarantine 和 cleanup 失败只影响当前 operation/item/unit。权威 schema 损坏、owner 无法建立或提交结果不可判定时，按 `ERR-001` 与 `SEC-010` fail-loud；不得写入空成功或静默丢弃。

### 3.6 移动 WebUI 发布与客户端缓存

[0022](../decisions/0022-mobile-webui-uses-server-selected-generations.md) 把服务端当前 `ReleaseView` 和其可达 generation 定义为 deployment 权威状态，把设备上的 verified generation 定义为按服务端隔离的派生 UI 缓存。两者都不得借用 SessionDB、Mobile Realtime DB 或 plugin-data 的删除和恢复协议。

| 对象 | 正常增加 | 允许的原位更新/逻辑终态 | 物理减少条件 | owner 与恢复证据 |
|---|---|---|---|---|
| `<workspace>/mobile-webui/publication.sqlite3` | 显式 build/import 增加 immutable generation/file 引用；每次 publish/clear/promote/rollback/restore 追加 journal；rollback pin 显式增加 | 单 writer 事务原子替换 Stable/Preview 指针并递增审计 sequence；generation、manifest 和 journal 既有内容不改写 | 指针不以 DELETE 代替更新；journal retention 未另立合同前不得自动减少；只有显式 unpin/GC 可减少 rollback eligibility，用户删除整个 workspace 属于其既有范围 | Core WebUI publisher；SQLite integrity、当前 `ReleaseView`、selection digest、journal、pin 与 source provenance |
| `<workspace>/mobile-webui/blobs/` | 候选校验后以 SHA-256 创建不可变文件；相同内容复用 | 只改变 generation、指针和 pin 的可达性，不原位改 blob | 显式 publication GC 在写锁内重新读取引用，只删除 Stable/Preview、每 channel 最近 4 个选择、显式 pin、候选和进行中备份 source set 均不可达的对象 | Core WebUI publisher；manifest/file digest、blob bytes、引用扫描与 GC report |
| `<workspace>/mobile-webui/staging/` | build/import 为当前候选创建临时对象 | 成功提交后变为 immutable CAS 引用；崩溃遗留保持未提交 | 启动恢复或显式 GC 只能删除能证明未被 publication DB 引用的 staging | Core WebUI publisher；候选 marker、publication transaction 与 orphan report |
| 用户指定的 WebUI backup artifact | `backup` 在临时目录写 SQLite online snapshot、自包含 CAS、source manifest 与 artifact digest，完整校验后原子发布；已存在目标不覆盖 | 已发布 artifact 内容不可原位更新；它独立保留快照时的 lineage、ReleaseView、journal 和全部声明资源 | publisher 不自动删除；只有名称明确、目标精确的 backup retention/delete 操作可减少，且不能把删除备份当成 live GC | Core WebUI backup/restore owner；artifact manifest/digest、SQLite `integrity_check`、server/epoch/selection/journal 与全部 reachable member hash |
| 移动端 app-private WebUI store | `Ensure` 在单 server staging 写入 manifest/blob，完整校验后增加 verified generation | desired/serving/fallback/attempt/reject marker 按 `Resolve/Ensure/Present` owner 更新；`WaitFor(space)` 是 coordinator 持有的进程内协调事实，不写业务 Room 表 | 安全 GC 只删除未 pinned 对象，或用户明确“重置此服务端 UI 缓存”；必须先取消并等待该 server owner，物理文件删除成功后才能删 metadata/reference，且不得删除其他 server 或业务状态 | Android/future iOS native store；embedded baseline、per-server manifest/hash、verified/attempt marker、删除失败后仍在的 owner、业务 write-set 对比 |

发布仓的 `release_epoch` 是 store 初始化时生成并持久化的 lineage UUID；从备份恢复到历史 `ReleaseView` 后保持备份中的 lineage 与当前选择。客户端不使用 epoch、sequence 或时间排序，因此恢复不会要求伪造更大的版本号。正式备份必须在同一 source snapshot 中列出 `publication.sqlite3`、当时所有数据库声明的 generation/blob、rollback pin 和 artifact digest；只复制数据库或只复制目录都不能证明可恢复。backup source set 在快照完成前 pin，避免与 live GC 竞态；恢复先在隔离目录验证 SQLite `integrity_check`、server identity、epoch、ReleaseView/selection、journal 连续性及每个 manifest/member digest，再原子替换 live publication root，替换前另建可恢复备份，替换后重复全部校验。

## 4. 再看上层所有权

```text
启动参数 / AKASHIC_WORKSPACE / config.toml
                    │
                    ▼
             ┌──────────────┐
             │ 显式 workspace│
             └──────┬───────┘
                    │
      ┌─────────────┼──────────────┬────────────────┐
      ▼             ▼              ▼                ▼
┌──────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐
│Session   │  │Memory    │  │Proactive   │  │Plugin/MCP  │
│Manager   │  │Runtime   │  │Runtime     │  │Runtime     │
└────┬─────┘  └────┬─────┘  └─────┬──────┘  └─────┬──────┘
     │             │              │               │
     ▼             ▼              ▼               ▼
 sessions.db   memory/*.md    proactive.db     plugin-data/
 migrations.sqlite3
 uploads/      memory2.db     wake*.db         插件 Skill/MCP catalog
               akasha.db      drift/drift.db   workspace 能力投影
```

workspace 之外还有两组明确的全局状态：

```text
~/.akashic/auth.json              旧模型迁移输入与非模型兼容凭据
~/.akashic-plugin/manifest.toml   已安装/启用插件目录
~/.akashic-plugin/cache/          已安装插件代码缓存
```

因此，“整个 workspace 已备份”可以覆盖已迁移模型及其凭据，但仍不能推出“系统已完整备份”。显式 `config.toml`、旧或非模型全局凭据、全局插件清单和插件 canonical source 仍在 workspace 之外。

## 5. 状态根与选择规则

| 对象 | 当前代码事实 | 上层 owner | 当前含义 |
|---|---|---|---|
| `--workspace PATH` | 最高优先级；`main.py::_workspace_from_args` 解析并写入 `AKASHIC_WORKSPACE` | `main.py` | 本次进程使用的状态根 |
| `AKASHIC_WORKSPACE` | 未给 CLI 参数时使用 | `main.py` | 显式环境级 workspace 选择 |
| `config.toml:[runtime].workspace` | CLI 和环境变量都为空时使用 | `main.py`、`agent.config` | 默认 workspace 选择 |
| 显式 `--config` | 可把主配置放在任意路径 | `main.py`、setup | 运行配置根，不保证位于 workspace |
| `AKASHIC_PLUGIN_HOME` | 未设置时回退 `~/.akashic-plugin` | `agent.plugins.manifest` | 全局插件安装根 |
| `~/.akashic/auth.json` | 旧配置或显式 JSON store 使用；已迁移模型不再回退读取 | `agent.model_runtime.auth` 兼容边界 | 迁移输入、恢复证据与非模型兼容凭据 |

**F-001：** runtime 的大部分可写状态已经从显式 workspace 派生。模型 credential 属于 workspace connection；旧或非模型全局凭据与插件安装状态是有意保留的例外，而不是 workspace 内的隐式目录。

**G-001：** 当前没有一个单独 manifest 同时声明主配置、workspace、旧或非模型全局凭据、插件清单和外部 plugin source 的完整恢复集合。

## 6. Workspace 当前文件结构

下面的树只列当前生产代码会创建、读取或写入的核心对象。插件仍可在自己的 `plugin-data` 中保存额外文件，因此它不是穷尽所有第三方数据的固定 schema。

```text
<workspace>/
├── sessions.db
├── migrations.sqlite3                 Yoyo 迁移成功回执
├── mobile-webui/
│   ├── publication.sqlite3             WebUI generation、ReleaseView 与 journal
│   ├── blobs/sha256/<prefix>/<digest>  不可变静态资源
│   └── staging/                         未提交候选；可证明 orphan 后才清理
├── sessions/                         目前只创建目录，未找到生产写入者
├── schedules.json
├── PROACTIVE_CONTEXT.md              旧安装可保留；新 init 不创建
├── proactive_pending.md              旧安装可保留；H2 只读盘点
├── proactive.db                      旧安装可保留；H2 只读
├── wake_proactive.db                 旧安装可保留；H2 只读/交接
├── proactive_quota.json              旧安装可保留；无目标 owner 时 BLOCK
├── uploads/
├── plugin-data/
│   ├── default_memory-builtin/
│   │   ├── config.local.toml
│   │   └── recall_inspector.jsonl     v3 inspector 追加式诊断证据
│   ├── akasha-builtin/
│   │   └── config.local.toml
│   └── <plugin>-<marketplace>/
│       ├── config.local.toml          可选
│       ├── .kv.json                   可选
│       └── <plugin-owned files>
├── memory/
│   ├── MEMORY.md
│   ├── SELF.md
│   ├── veda.md
│   ├── PENDING.md                     仅升级遗留输入；在线不再创建
│   ├── PENDING.snapshot.md            仅升级遗留输入
│   ├── PENDING.retired.md             一次迁移的原始边界归档
│   ├── consolidation_writes.db
│   ├── markdown-profile-writes.db
│   ├── markdown-profile.lock
│   ├── memory2.db                     退役经典记忆归档
│   ├── akasha.db                      akasha engine
│   ├── MEMORY.bak.md / SELF.bak.md
│   ├── backups/
│   ├── veda-backups/<UTC时间>/VEDA.md
│   ├── spawn_trace.jsonl
│   ├── proactive_config_trace.jsonl
│   └── proactive_rate_trace.jsonl
├── drift/
│   ├── drift.db
│   └── skills/                        插件 Drift skill 软链接投影；仍兼容手工目录
├── skills/                            插件 skill 软链接投影；仍兼容手工目录
├── mcp/
│   ├── servers/*.toml                 现有直装兼容路径，目标是迁入插件
│   ├── backups/<server>/*.toml        直装兼容路径的声明备份
│   └── <legacy server files>          不再作为新增能力的 canonical 安装位置
├── observe/
│   └── recall_inspector.jsonl         v2 兼容名字；迁移后与 plugin-data 文件同 inode
├── subagent-runs/<job-id>/            子任务产物
├── runtime/
│   ├── plugin-reloads.sqlite3         插件热重载事务与恢复阶段
│   ├── plugin-jobs/outcomes.sqlite    插件 background job 幂等与恢复状态
│   ├── deliveries/settlements.sqlite  通用 delivery、provider、Session projection 与领域 settle
│   ├── proactive-documents/
│   │   └── intents/<invocation-id>/   paired Markdown old/new bytes 与恢复回执
│   ├── plugin-rollout-fact.json       下一用户 turn 消费的一条派生结果
│   └── plugin-validation/<generation>/ 候选隔离 plugin-data 副本
├── memes/manifest.json
├── .app-server-token
├── .instance.lock
├── .supervisor.lock
├── .supervisor.pid
├── .runtime-ready.json
└── akashic.sock                       Unix 控制面启用时
```

`bootstrap/init_workspace.py` 只预创建基础 Markdown（包括缺失时的 `memory/VEDA.md`）、`schedules.json`、`memes/manifest.json`、目录、`sessions.db`、`consolidation_writes.db` 和当前 memory engine 声明的存储。新安装不创建 `memory/RECENT_CONTEXT.md`、`PROACTIVE_CONTEXT.md` 或 `proactive.db`；已有这些旧文件即使在 `init --force` 下也不覆盖或删除。附件、诊断记录和插件私有文件按对应普通能力首次使用时创建。

## 7. 会话、消息与附件

### 7.1 `sessions.db`

| 表 | 写入 owner | 上层使用者 | 代码事实 |
|---|---|---|---|
| `sessions` | `session.store.SessionStore`，由 `SessionManager` 协调 | channel、AgentLoop、`session.activity.PresenceStore`、dashboard | session metadata、时间、高水位和当前 compaction generation |
| `channel_identities` | `SessionStore` 原子事务，由 `SessionManager`/Core Channel Host 协调 | v3 channel inbound 与 proactive recipient resolve | `(channel, identity)` 唯一 durable recipient；legacy Session metadata 只作一次性迁移输入，显式 Session 删除由同一审计事务级联并可从整库 backup 恢复 |
| `channel_identity_migrations` | `SessionStore` | v3 channel identity rebuild | 每个 channel 的一次性 migration marker；identity 表删除到空也禁止重新扫描 legacy metadata |
| `session_compactions` | `session.store.SessionStore`，由 Core checkpoint owner 请求 | prompt replay、Markdown reconciliation、删除恢复 | append-only generation lineage、source provenance、retained tail、summary、usage 和失效状态 |
| `interaction_memory_reconciliations` | 已退役 | 无当前 consumer；turn-effects Yoyo 备份 SessionDB 后删除旧表 | 不保留兼容读写路径 |
| `messages` | `SessionStore` | prompt 历史、dashboard、Akasha、检索工具 | 原始 user/assistant/tool 消息和单调 `seq` |
| `attachments` / `attachment_imports` | `SessionStore` + Core `ChannelAttachmentArtifactStore` | v3 Channel、Mobile adapter、Session read projection | immutable ready artifact metadata 与 crash-resumable import phase；不提供普通 delete/GC |
| `message_attachments` | `SessionStore` message append transaction | prompt/read adapter、Channel history | ordered message→artifact binding，与 `extra.attachment_ids` 同事务一致 |
| `turns` | control/runtime 持久化路径 | 控制面、恢复和审计 | turn 输入、items、usage、error、final response 与终态 |
| `messages_fts` + triggers | `SessionStore` 自动维护 | 消息全文搜索 | 可由 `messages` rebuild 的 FTS5 索引 |
| `message_embeddings` | `MessageEmbeddingStore` | Akasha、Wake 等语义检索 | 按消息和模型保存的向量；Akasha 重建必须复用这些已存向量，不得临时重算 |
| `message_embedding_migrations` | `MessageEmbeddingStore` | embedding 迁移 | 已导入 source 的幂等记录 |

**F-002：** `plugins.akasha.engine` 明确把 `sessions.db/messages` 标为 truth。正常会话持久化路径只 INSERT 新消息；现有 update/delete/session cascade 方法属于用户数据管理边界，不能由上下文裁切、重构、检索或保留期猜测调用。

**F-002A：** 含 `control_turn_id` 的 transcript 是一个不可拆分的用户数据管理单元。单消息和 generic batch 删除入口必须返回 `interaction_delete_required` 及 `control_turn_id`，由客户端显式转调 interaction 撤销；撤销前创建并验证不可覆盖的完整 SessionDB 快照，再在一个 SessionDB 事务中删除整组 U+A 与对应 `message_embeddings`。若 `last_consolidated` 位于该组内或组后，回退到组前消息边界；响应与日志报告旧/新游标及备份路径，其他消息、`seq` 高水位和附件保持不变。

**F-003：** FTS 在缺失或旧 tokenizer 时会从 `messages` rebuild。独立的 embedding 迁移可以显式生成新模型向量，但这不属于 Akasha 重建；Akasha 只能读取 sessions 中与目标模型匹配的已存向量，缺失或错配必须失败。`message_embeddings` 与完整历史一起受 `CTX-001` 保护，不能在 prompt 裁切中顺带删除。

### 7.2 `uploads/`

`AttachmentStore` 由 Telegram、QQ 和 Web Chat 共用，把收到或上传的原始字节写到 `<workspace>/uploads/`。消息的 `media` 字段保存文件路径，读取 prompt 时图片会再次从路径加载。

**F-004：** 当前没有找到 production garbage collector、引用计数或 session 删除时的附件级 cascade 实现。

**F-004A：** 产品语义已经确认：消息仍引用附件时，附件必须保持可读。当前实现尚未提供引用计数、孤儿判定和安全 GC；这些能力完成前禁止自动清理。

## 8. Markdown 记忆与语义记忆

### 8.1 当前活动 Markdown 与旧队列

| 文件 | 写入 owner | 当前用途 | 状态性质 |
|---|---|---|---|
| `memory/MEMORY.md` | ordinary `markdown_memory` plugin | 稳定用户档案，通过 ordered prompt event 进入 prompt | 人类可读长期事实 |
| `memory/SELF.md` | ordinary `markdown_memory` plugin | Akashic 自我认知，通过 ordered prompt event 进入 prompt | 人类可读长期事实 |
| `memory/VEDA.md` | Main Agent 仅响应用户明确指令；`main.py veda-reset` 是独立恢复 owner | React 链路与已安装插件按各自生命周期读取的人格真源 | 用户可维护的权威人格状态 |
| `memory/PENDING.md` / `PENDING.snapshot.md` | 仅 `markdown_memory` legacy migration 读取 | 升级前未迁移事实；在线不再写入 | 迁移完成前的历史输入 |
| `memory/RECENT_CONTEXT.md` | 旧安装遗留文件；新运行时无 writer/reader | 不再进入 prompt、proactive、Wake 或 Drift | 只由最后阶段 R06 带备份、校验并归档删除 |

Markdown plugin 还维护：

- `markdown-profile-writes.db`：每个文档独立保存 draft、before-image 和 applied receipt。
- `PENDING.retired.md`：保留 legacy PENDING 与 snapshot 的原始文本和文件边界。
- `consolidation_writes.db` 现在只由 compaction plugin 写 checkpoint receipt；Markdown 不共享写 owner。

**F-005：** 当前主线不创建或写入 `memory/HISTORY.md` 和 `memory/journal/`。compaction 只发布来源无关的 durable committed fact；Markdown plugin 只更新 MEMORY/SELF，Akasha 继续独立消费 completed transcript。旧 `_handbook/memory-markdown.md` 对五文件模型的说明已经过时，入口处已加警告。

### 8.2 `memory/memory2.db`（退役归档）

| 表 | 作用 | 当前性质 |
|---|---|---|
| `memory_items` | 结构化长期记忆、类型、摘要、source_ref、状态和 embedding | 该存储中的 canonical 事实 |
| `consolidation_events` | 同一 source_ref 最多摄入一次 | 幂等提交记录 |
| `memory_replacements` | supersede 前后的完整条目和关系 | 勘误与 undo 证据 |
| `vec_items` | sqlite-vec 加速 | 可由 `memory_items.embedding` 重建的索引 |

经典记忆插件已经删除；runtime 不再打开、导入或更新 `memory/memory2.db`。已有文件原样保留，
只作为可恢复历史归档，不参与 Akasha replay。

**F-006：** `memory_items` 不只是从原始消息确定性计算出的缓存。它包含模型提取、显式记忆、强化、supersede 和人工管理结果；只保留 `sessions.db` 不能证明能无损重建同一份 `memory2.db`。

### 8.3 `memory/akasha.db`

Akasha V2 保存 turn 指针、稀疏特征、engram hub、有向关系、activation/plasticity 事件和因果上下文。宿主 adapter 与重建 CLI 都只从 `sessions.db` 读取原始正文；Akasha sidecar 不充当事实来源。完整调用链见 [Akasha V2 在线与确定性重放设计](akasha-v2-runtime-migration.md)。

**F-007：** `akasha.db` 与 `akasha-v2-index.db` 是由 `sessions.db/messages`、`sessions.db/message_embeddings`、固定算法和固定配置得到的派生索引与图。标准 rebuild 复用已有 embedding，不调用 LLM，也不让模型重新解释历史。旧 `akasha_graph_snapshot.json` 和私有图 Dashboard 已退出 V2 运行时接口；新 Inspector 只读查询这两个 sidecar，不生成第三份快照，也不拥有保留或删除权限。

**F-007A：** 同一份 messages、匹配的 message embeddings、算法和配置必须得到可复现的图。算法与配置要作为重建输入固定；改变它们属于显式图迁移，不是同输入重建。

**F-007B：** 当前 `build_akasha_db.py` 在备份和目标数据库写入前审计全部合法对话 embedding。缺失、内容 hash 不匹配、模型/维度不匹配、非有限或零向量会写出确定性缺口报告并 fail-loud；声明 `effects.post_commit=suppress` 的 Turn 和双方都为空的纯媒体 Turn 不属于学习输入。

**F-007C：** Akasha 启用时，interaction 撤销由 Akasha owner 先以 source-event gate 排空已开始的 `TurnCommitted` embedding + staging，再封住在线 query/commit，调用只允许删除目标 interaction 的 SessionStore 回调，递增 source generation、清除所有基于旧图节点生成的 pending ticket，并从剩余 canonical source 生成完整 sidecar 候选。候选按 index→memory 发布；两文件之间的崩溃窗口在下次启动通过 source/index 或 index/memory 身份失配触发确定性重建。删除已提交但重建失败时，运行时保持 fail-loud，不得继续提供旧 turn 节点；等待删除期间才开始的 source event 必须因 generation 失配而失效，不能重新写回 embedding。

## 9. 主动流程、Wake、Drift 与调度

### 9.1 `proactive.db`

H3 前的 `ProactiveStateStore` 由 `bootstrap/proactive.py` 构造并保存以下表；H3 后生产 runtime 不再实例化 writer，H2 只读 decoder/inventory 继续解释旧库：

- `deliveries`：按 session 和 delivery key 去重的已发送时间。
- `session_state`：last tick、delivery、drift、context-only 等 session 级时间状态。
- `context_only_timestamps`：频率限制窗口。
- `tick_log`：一次 proactive tick 的 gate、候选、选择、发送和 effect 摘要。
- `tick_step_log`：tick 内阶段级输入输出和错误审计。

**F-008：** 这个库同时含“防止重复外部发送的运行连续性”和“诊断审计”。把整个库当临时日志清空，会改变 dedupe、cooldown 和下一次主动行为。

### 9.2 `wake_proactive.db`

H3 前的 `WakeStateStore` 保存以下表；H3 后旧 writer 已删除，H2 只读 decoder 与 owner handoff 继续处理已有库：

- `wake_runs`、`wake_observations`：一次 wake 的调查、消息和输入证据；只用于历史读取。
- `reservoir_events`：未读/已消费 source event 与 embedding。
- `reservoir_quarantine`、`reservoir_tombstones`：无效输入的有界诊断连续性，以及成功 expire ACK 后防止相同 identity 重收的连续性。
- `hazard_state`：下一轮唤醒计算读取的压力与最近 wake 状态。
- `hazard_monitor`：Dashboard 和只读诊断消费的最近计算观测；runtime 下一轮不把它作为决策输入。
- `context_state`、`context_reevaluate_state`：外部上下文及重评节流。
- `drift_state`：每个 session 的 drift 计时与重复指纹。
- `pending_acknowledgements`：尚未成功回写外部 source 的 ack 队列。

**F-009：** 该库含 pending ack、tombstone、quarantine、消费状态、hazard/context 和计时器。
这些连续性表不是只影响 dashboard 的可丢弃缓存；丢失可能造成重复摄入、重复消费、漏 ack 或
行为时间线重置。`wake_runs`、`wake_observations` 与 `hazard_monitor` 可以走只读历史投影，但这不
授权物理删除原表或把其余表解释为历史。

### 9.3 `drift/drift.db`

`DriftStateStore` 保存：

- `runs`、`run_steps`：执行和工具步骤记录。
- `skill_continuum`：run count、last status、scratchpad 和 cursor。
- `skill_journal`：按 skill、entry type 和 key 记录的连续性日志。
- `global_note`、`self_state`：全局 note 和当前自我选择状态。

**F-010：** Drift runtime 依赖这些表在下一轮继续工作。对启用 Drift 的 workspace，它是工作流连续性状态，不是单纯可重建 trace。

### 9.4 `schedules.json` 与 `proactive_quota.json`

- `schedules.json` 由 `JobStore` 原子保存完整 `ScheduledJob` 集合。只有文件缺失能解释为空；JSON 或 schema 损坏会 fail-loud。
- `proactive_quota.json` 由 `QuotaStore` 保存每日窗口、已用次数和最后动作时间。损坏不会静默重置。

**F-011：** 两者都会决定重启后的外部行为。它们是体积很小但不能随意丢弃的操作状态。

### 9.5 EventMail、Wake attempt 与通用 delivery

- `plugin-data/eventmail-builtin/eventmail.sqlite3` 由 EventMail 插件拥有。`mail_envelopes` 与 `mail_transitions` 正常只追加；Content、Alert、Context 各有独立 source capability 和生命周期。三类查询 projection 允许由 EventMail 原位更新，但每次 Content 状态变化也会追加完整的 `projected` transition；四张 Content 查询表与 Alert/Context projection 都能从不可变流水确定性重建，不能反向改写 envelope。
- 安装迁移 `20260828_01_migrate_eventmail_state` 先在线备份旧 Content 与 Wake DB，再把旧 Content、Alert、Context 写入 EventMail 并核对 schema、状态和 SQLite integrity。发布成功后，旧 `content-builtin` 目录移入 migration backup，Wake DB 删除已迁移的 Alert/Context 表并升级为 v7；运行路径不再读取两套真源。恢复证据是 migration receipt、retired Content 完整目录和 Wake SQLite backup。
- `plugin-data/wake-builtin/wake.sqlite3` 保存 admission、seen Content、不可变 Content 初始质量、decision run 与 Timer attempt。`20260828_02_add_wake_content_scores` 先用 SQLite backup API 备份 v7，再移除失去语义的旧冷却时间并新增 v8 `content_scores(initial_score, semantic_interest, scored_at)`；现有 pending revision 由 runtime 首次维护时走普通一次评分路径补齐。每次真实 Timer fire 先追加 attempt，再读取 EventMail watermark；水位读取前可以是 `NULL`。`no_due`、`content_insufficient`、历史 `admission_rejected`、`shared`、`model_skip`、`deferred`、`cancelled_after_fire`、`delivery_unknown`、`failed` 都分别闭合并可由 Dashboard 查询。`cancelled_after_fire` 只表示 Timer 已触发、runtime 在职责检查前关闭，不产生 scoped Turn 或外部发送。新进程在领域恢复前把上一进程遗留的 `checking` 保守闭合为 `delivery_unknown`，因为崩溃前可能已经越过 provider I/O。attempt 是运行诊断，不拥有三类 mail 正文。
- `runtime/deliveries/settlements.sqlite` 由 Core 拥有。Core 不读取 EventMail DB，也不持有 EventMail closure；它只保存通用 envelope、exact channel attempt/receipt、Session projection message id 和 opaque EventMail receipt。
- Wake 是普通组合 owner：先让 Core 到 `projected`，再让 EventMail settle，最后把 EventMail 的稳定 receipt 交回 Core confirm。两库之间不宣称原子事务；任一崩溃窗口都只向前补尚未完成的一步。
- `rejected` 与 `uncertain` 是可观察终态，不自动重发、不投影 Session、不消费 Content。source-bound ACK 失败只保留 Content `delivered`，由该 source 的 `unsettled/ack` 窄口重试，不重做 Turn 或 provider delivery。

**F-012：** EventMail mail、Wake attempt、provider effect、Session message 和 delivery settlement 各有唯一 owner。候选 generation 不能移除仍被 `prepared/delivered/projected` ledger row 引用的 target service；`settled/rejected/uncertain` 不形成该发布 fence。

## 10. 插件数据与 Skill/MCP 能力发布

### 10.1 `plugin-data/`

每个插件的数据目录固定为 `<workspace>/plugin-data/<plugin>-<marketplace>/`。插件可以保存：

- `config.local.toml`：插件本地配置。
- `.kv.json`：`PluginKVStore` 的原子 JSON 状态。
- 插件自定义数据库、附件、游标或其他文件。

候选插件 generation 只能修改内存中的 KV staging；校验失败不写正式 `.kv.json`，commit
时只落盘实际发生过的修改。正式发布后，同一个 store 才转为受 generation fencing 约束的
直接写入。卸载流程删除全局 cache 和 manifest entry，但只返回 workspace data path，
没有删除该目录。

**F-012：** plugin-data 与插件代码生命周期分离，当前卸载会保留数据。备份系统不能只列出已知内置文件；整个目录对 core 来说必须按 opaque plugin-owned state 处理。

### 10.2 全局插件状态

| 路径 | owner | 性质 |
|---|---|---|
| `~/.akashic-plugin/manifest.toml` | `agent.plugins.manifest` | 已安装/启用插件和 package 的全局目录 |
| `~/.akashic-plugin/cache/` | install/source resolver | 可通过安装源重新获取的插件代码缓存 |
| 外部插件 canonical source | 用户选择的源码仓库 | 开发资产，不等同于 cache，也不由 workspace 备份拥有 |

**F-013：** 修改外部插件必须定位 canonical source；直接备份或编辑 cache 不能替代源码仓库和安装清单。

插件包同时是 Skill 和 MCP 的能力交付单元：

1. 插件类通过 `skill_roots()`、`drift_skill_roots()` 和 `mcp_servers()` 声明能力。
2. `plugin-install` 在 staging 中复制插件代码并准备 MCP runtime，完成后原子发布到全局 cache，再更新 manifest。
3. `PluginManager` 为候选 generation 准备 Skill/MCP catalog；readiness 失败时拒绝候选，旧 generation 继续服务。
4. generation 发布后，`PluginSkillLinker` 才把 active plugin 的 skill 同步成 workspace 软链接。

因此，Skill/MCP 的 canonical code 和声明属于插件 source；cache 是已安装版本，manifest 记录安装身份，workspace 只保存 plugin-data 和必要的运行投影。

#### 10.2.1 0024 stable/latest 实现

**F-014：** [0024](../decisions/0024-plugin-self-validation-uses-stable-and-latest.md) 与 [0026](../decisions/0026-plugin-rollout-is-owned-by-the-parent-turn.md) 要求插件安装 artifact 按 source revision/tree digest 不可变保存；同一版本号的新 commit 不能覆盖 stable runtime 仍引用的代码。插件目录内的原子 `.pointers.json` 拥有 stable/latest artifact descriptor；`<workspace>/runtime/plugin-reloads.sqlite3` 拥有单一未决 candidate phase、install provenance、turn lineage 与 append-only phase journal。普通 turn 只读取 stable；只有 owner parent turn 创建的 attached programmatic child 自动读取匹配 latest。候选独占服务使用 `runtime/plugin-validation/<generation>/` 的 plugin-data 副本和临时端口，提交或丢弃后删除。

该目标的状态变化固定为：

| 对象 | 正常增加 | 允许原位更新/逻辑终态 | 物理减少条件 | owner 与恢复证据 |
|---|---|---|---|---|
| 不可变 plugin artifact | 每个新的 source revision/tree digest 增加独立目录 | artifact 内容不原位更新；stable/latest 只改变引用 | 不再被 stable、latest、active lease、rollback/recovery source set 引用后，显式 GC 才可减少 | Plugin install owner；source commit/tree digest、artifact digest、manifest |
| stable/latest descriptor | install 追加 candidate/journal；首次初始化建立 stable | 单 writer 事务更新 pointer 与 candidate phase；既有 journal 不改写 | pointer 不以 DELETE 表达；journal retention 未定义前不得自动减少 | Runtime snapshot publisher；SQLite integrity、pointer identity、phase journal、lease set |
| programmatic validation session | 新 thread/turn/messages/tool items 正常 INSERT | turn 按控制状态机进入 terminal；session metadata 固定 memory policy | 只按既有用户 thread/session 删除协议减少 | Control + SessionStore；thread/turn read、tool items、memory write-set |

`plugin-install` 由当前 Gateway 的 runtime owner staged publish，并等待 `latest_ready`；`RuntimeSnapshotStore` 只允许显式 selector 租用 latest，普通 turn 默认 stable。promote/discard 通过 pointer、journal 和 snapshot lease 收敛。安装成功只证明候选 ready，仍必须用 programmatic child 的 snapshot identity、SessionDB/tool trace 和领域 oracle 证明行为有效。

### 10.3 MCP 的插件路径与现有直装路径

- 目标路径：插件类的 `mcp_servers()` 声明 MCP；安装器准备 runtime；插件 generation 发布 MCP catalog。
- 旧 `mcp/servers/*.toml`、`WorkspaceMcpAdmin` 和 `WorkspaceMcpWatcher` 已删除；MCP 声明只由插件静态 manifest 进入 Root registry 与 generation host。
- 两条路径发生同名冲突时，当前启动流程 fail-loud。这证明它们确实是两套并列 owner，而不是同一安装流程的不同界面。

产品意图已经确认：新增和保留的 MCP 应通过插件安装。直装声明需要迁移成插件贡献；完成迁移前保留兼容读取和恢复能力，但不再把它定义为长期 canonical 资产，也不新增依赖这条路径的功能。

### 10.4 `skills/` 与 `drift/skills/`

当前实现允许两种对象并存：

1. `PluginSkillLinker` 根据 active plugin generation 创建的软链接，这是目标路径，也是可重建投影。
2. 手工创建的真实目录仍会被 loader 读取，这是需要迁移的兼容路径，不再承担新的 canonical 能力所有权。

**F-014：** 新 Skill 和 Drift skill 必须装进插件，由插件 source 持有正文。备份和迁移应记录插件 manifest/source，并在恢复后重建 workspace 软链接，不能把链接目标复制成 workspace 内的普通目录。

**G-004A：** 仓库仍有手工 skill 创建与加载路径。后续迁移需要先把现存真实目录封装成插件，再收窄 loader 和写入工具；直接删除目录会丢失尚未迁移的能力。

### 10.5 `memes/manifest.json`

`init_workspace` 会创建 `{"categories": {}}`。canonical `meme` 插件把
`<workspace>/memes` 作为产品级共享素材根：运行时按 manifest mtime 重载类别，Skill 与
Dashboard 可按显式用户操作增改 manifest、类别目录和图片。它不是插件私有 data root，
也不能因卸载插件而删除。

v3 candidate 通过插件声明的 `workspace_roots = ("memes",)` 获得隔离副本；candidate
listener 与 Dashboard 读写同一副本，discard 不改正式素材，promotion 后 formal Root 才解析
正式目录。正常增改由 Meme 插件和 `meme-manage` Skill 拥有；物理删除只允许明确的用户
管理动作，并以目录备份、manifest/content hash 与恢复读取为证据。

## 11. 配置、凭据与控制面

### 11.1 主配置和凭据

- `config.toml` 通常位于仓库根并被 `.gitignore` 排除，也可通过 `--config` 指向其他位置。模型迁移后它保存 runtime、channel、memory、proactive 等静态设置；动态 connection/model/role 由 workspace `model-registry.sqlite3` 保存。
- `model-registry.sqlite3` 保存 Provider connection 的 Base URL 与 credential payload、模型能力快照、角色绑定和 revision。它及其备份属于 secret；普通模型选择只改该库，不改 `config.toml`；每个完整执行在入口读取 revision 并冻结整组角色。
- `sessions.db/sessions.metadata.model_selection` 保存单个会话固定的 model ref 与 reasoning effort；它跨重启保留但不复制 Provider 凭据或能力，实际执行仍从开始时取得的 registry generation 解析。
- `~/.akashic/auth.json` 继续由 JSON 兼容 owner 以 0600 权限维护。模型 Yoyo 从中复制当前 workspace 引用的 credential，但不删除旧值；迁移后模型设置、Codex refresh 和 runtime 请求只读写 workspace 模型库。

两者都是恢复运行所需配置，但不能直接提交到 Git 或写入普通诊断文档。

### 11.2 控制面文件

| 路径 | 代码事实 | 恢复意义 |
|---|---|---|
| `.app-server-token` | workspace loopback TCP 控制认证，0600 | secret；可重新生成会改变现有控制客户端凭据 |
| `.instance.lock` | 文件可常驻，内核 flock 才是真 owner | 不应把文件存在当运行中，也不应从快照恢复 owner |
| `.supervisor.lock` | supervisor flock 与诊断 PID | 临时运行控制 |
| `.supervisor.pid` | 当前 supervisor PID，停止时清理 | 临时运行控制 |
| `.runtime-ready.json` | 当前 boot ID、PID 和 ready 状态 | 只属于当前启动，停止时清理 |
| `akashic.sock` | Unix 控制 socket | 进程端点，不能从备份恢复 |

**F-015：** lock、PID、readiness 和 socket 即使落在磁盘，也不属于可恢复业务状态。`.app-server-token` 是例外：它是持久 secret，但恢复策略要与控制客户端配对设计。

## 12. 诊断、审计和临时产物

| 路径 | writer | 当前用途 |
|---|---|---|
| `plugin-data/default_memory-builtin/*` | 退役经典插件遗留归档 | runtime 和 Dashboard 均不再读取 | 不自动删除 operator 历史文件 |
| `memory/spawn_trace.jsonl` | subagent manager | spawn 决策与完成 trace |
| `memory/proactive_*_trace.jsonl` | proactive loop | 配置和频率决策 trace |
| `subagent-runs/<job-id>/` | background subagent | 隔离的子任务报告和脚本产物 |
| `/tmp/akashic-llm-payloads/` | provider，可配置启用 | 完整 LLM 请求快照；跨进程互斥写入，按最旧优先轮转，最多保留 16 份且合计不超过 64 MiB；单份超过 64 MiB 时明确报警并拒绝落盘 |
| `/tmp/akashic-exec-*.log` | shell execution manager | 完整命令输出诊断日志；无截断终态、显式 stop、容量回收或 runtime shutdown 时删除，工具结果发生省略时保留并返回路径；不进入 SessionDB，也没有跨 runtime 恢复语义 |

这些文件可能包含用户原文、工具参数、检索记忆、路径或模型 payload。它们对事故取证有价值；LLM 请求快照已由 provider 执行数量和容量轮转，`/tmp/akashic-last-llm-payload.json` 与最新快照共享 inode，不重复占用空间。其他诊断文件仍没有统一 retention、脱敏、容量或备份策略。

**G-005：** 诊断证据的最低保留期、隐私边界和容量上限尚未形成项目级合同。确认前不能把它们提升为永久记忆，也不能在事故调查中默认它们一定存在。

## 13. 当前恢复机制与缺口

### 13.1 已存在的局部机制

- Markdown plugin 在 `markdown-profile-writes.db` 中按文档保存 before-image、draft 和 applied receipt，并在启动时前向恢复。
- PENDING 与 PENDING.snapshot 只由一次 legacy migration 读取；原始 bytes/digest 先进入 immutable receipt，并归档到 PENDING.retired 后才清空旧文件。
- MCP 声明修改前创建历史备份，发布失败自动回滚。
- 凭据覆盖前保留一个固定名称备份。
- 插件 cache 激活使用 staging、backup 和目录原子替换，失败回滚当前安装事务。
- Akasha rebuild 脚本会备份旧 sidecar。
- `scripts/rolling_backup.py` 支持普通文件复制、SQLite online backup、`integrity_check`、manifest、临时目录原子发布和成功后 prune。

### 13.2 已确认缺口

1. 仓库只有通用 `rolling_backup.example.toml`，没有 Akashic workspace 的正式 source manifest。
2. 通用脚本现在支持普通文件、SQLite online backup 和拒绝 symlink 的目录树快照，并以 relative path、mode、size、SHA-256 manifest 支持隔离恢复；但仍没有 Akashic workspace 的正式 source manifest，也没有把插件 manifest/source 作为 companion 安装状态恢复。
3. 当前快照 manifest 不记录 workspace 选择、应用 commit、schema/插件版本、主配置位置或全局状态位置。
4. 没有一条仓库内工作流证明同一份快照能在隔离 workspace 恢复并通过应用级只读 smoke。
5. SQLite 分别 backup 时，每个文件内部一致，但多个数据库与普通文件之间没有全局事务时点。
6. 备份范围没有明确包含或排除 `.app-server-token`、diagnostic traces、旧或非模型全局凭据和全局插件 manifest。
7. `mcp/servers/*.toml` 与 workspace 手工 skill 目录仍绕过插件安装系统，形成第二套能力 owner；现存内容尚未迁移。
8. 通用 rolling backup 尚不能把 WebUI publication DB 与它引用的 blobs 作为同一一致性 source；首版 publisher 必须至少导出可审阅 reachable manifest，并在发布正式资源前完成隔离恢复 smoke。

这些缺口正是 `BAK-001` 和 `NOW.md` 中恢复演练事项尚未完成的部分。

## 14. 设计意图确认记录与剩余推断

INT-001～INT-008 和 INT-011 已由花月哥哥确认，其中长期语义已经提升为 projectneed 条款。其余条目仍是 inference；未确认前不能写入删除器、迁移器或自动备份排除规则。

### INT-001 Workspace 是主要运行工作区 — 已确认

确认内容：显式 workspace 保存 Akashic 的主要运行文件，是会话、记忆、附件、调度、自主流程、模型 connection/credential 和 plugin-data 的主要备份/迁移单元。主配置、旧或非模型全局凭据、插件 manifest/cache 和外部 plugin source 是明确的 companion state。

已提升条款：WSP-001、WSP-004。迁移工具以 workspace 为主体，同时生成 global companion manifest，不再散落猜路径。

### INT-002 `sessions.db/messages` 是对话唯一原始事实源 — 已确认

确认内容：prompt history、runtime history view、FTS、embedding、Akasha 和 dashboard 都只能读取或投影消息。正常收发只追加原始消息；只有用户主动撤销消息或删除会话时，独立数据管理命令才能减少原始消息。

已提升条款：SES-003、SES-005。任何 context/refactor 的 protected write set 永久禁止 `DELETE/UPDATE messages`；用户撤销/删除操作必须单独声明 change intent、目标、cascade、备份和审计。

剩余未知：当前 `update_message` 是否是要保留的用户编辑语义；如果保留，它是只追加合同的另一个显式例外，还是应该改成追加 correction/revision。`turns` 是否也属于用户必须长期保留的审计事实，还是允许独立 retention 的运行记录？这些问题都不影响正常收发只追加的合同。

### INT-003 Markdown 四文件有不同耐久等级 — 已确认

确认内容：`MEMORY.md` 与 `SELF.md` 是人类可读长期档案；0052 已把 `PENDING.md` 改为只读的升级遗留输入，迁移 receipt 与 `PENDING.retired.md` 发布前必须保留原始内容；旧的近期摘要投影已退役。

已提升条款：MEM-001～MEM-004、MEM-008。备份和 oracle 分别验证档案事实、legacy 队列恰好一次与 recent projection 可重建性，不再把这些文件统一叫“memory cache”。

### INT-004 `memory2.db` 是独立长期记忆库 — 已确认

确认内容：`memory2.db` 包含模型提取、显式记忆、强化、替换历史和人工管理结果，是不可由 Markdown 或 `sessions.db` 无损替代的长期状态。

已提升条款：MEM-005、MEM-008。`memory2.db` 进入最高备份级别；`vec_items` 可以重建，但整库不能按“向量缓存”排除。

### INT-005 Akasha 两个 V2 sidecar 是确定性派生状态 — 已确认

确认内容：Akasha 使用固定算法读取 `sessions.db/messages` 和已有 `message_embeddings` 重建，不引入 LLM 重新解释历史。算法和配置固定时，同一输入必须产生可复现的图。

已提升条款：MEM-009。常规备份可以选择重建 Akasha，但必须检测全部 embedding 命中并验证 parity；缺失时不能生成残缺图后报告成功。

### INT-006 主动、Wake 和 Drift 数据属于运行连续性 — 已确认

确认内容：delivery dedupe、pending ack、reservoir consumption、hazard timer、Drift cursor 和 journal 都影响下一次真实外部行为，启用相关功能时必须随 workspace 恢复。

已提升条款：PRO-002。不能把 `proactive.db`、`wake_proactive.db` 和 `drift.db` 全部按日志清理；审计表与行为状态表分别制定 retention。

### INT-007 plugin-data 默认保留并跨卸载复用 — 已确认

确认内容：普通卸载只删除代码 cache、manifest entry 和能力投影，保留 data path；这是用户数据保护语义。

已提升条款：PLG-010。卸载 UI/CLI 明确区分“卸载代码”和“连同数据永久删除”；后者需要单独确认和备份。

### INT-008 附件是会话引用的用户数据 — 已确认

确认内容：只要消息仍引用附件，附件就必须可读；清理从完整引用关系和显式 retention 出发，不能按目录年龄或 prompt 是否使用判断。

已提升条款：SES-006。后续补引用扫描、孤儿判定、dry-run、备份和 cascade 语义；在此之前不做自动 GC。

### INT-009 调度和 quota 是小体积 canonical 操作状态

推断：`schedules.json` 与 `proactive_quota.json` 决定重启后会不会执行或发送，损坏和缺失必须与空状态区分。

确认后的影响：二者进入默认备份；恢复 smoke 要验证任务和 quota 窗口，而不是只检查文件存在。

### INT-010 配置与 secret 属于恢复集合，但不属于 Git 文档

推断：`config.toml`、plugin config、MCP env、auth store 和 app-server token 要有受保护备份渠道；项目工作手册只记录位置和 schema，不复制值。

确认后的影响：备份 manifest 支持 secret 分类、权限保持和脱敏清单，文档检查禁止泄露值。

### INT-011 Skill 和 MCP 通过插件安装 — 已确认

确认内容：Skill、Drift skill 和 MCP 都由插件包声明和安装。插件 source 持有能力正文，cache 保存已安装版本与 MCP runtime，manifest 记录安装身份；workspace skill 软链接只是 active generation 的投影，plugin-data 继续留在主要 workspace。

已提升条款：PLG-009。当前 `mcp/servers/*.toml` 直装通道和 workspace 手工 skill 目录是待迁移兼容路径；恢复应根据插件 manifest/source 重装能力并重建投影，不复制链接目标。

### INT-012 诊断数据需要有界保留，不自动进入长期记忆

推断：recall、spawn、proactive 和 payload trace 用于问责，但不是产品事实源；应该按隐私和事故窗口设置 retention，而不是永久注入 prompt。

确认后的影响：新增统一 trace policy、容量/时间上限和事故冻结机制；不会从 trace 自动写 MEMORY。

### INT-013 控制文件不进入业务恢复

推断：PID、lock、readiness 和 socket 在新环境必须重建；`.app-server-token` 作为 secret 单独处理，不能和这些临时文件一起粗暴排除。

确认后的影响：restore 明确删除陈旧控制端点，再决定恢复或轮换 token。

### INT-014 项目需要一份可执行的完整备份合同

推断：当前通用脚本是机制，不是 Akashic 的备份范围。项目最终需要一份版本化 source classification、外部 secret 配置、目录快照能力和隔离恢复 smoke。

确认后的影响：下一阶段先把本地图转换成机器可读 manifest，再扩展目录一致性快照和恢复验证；不能只往示例配置里手工多写几个路径。

## 15. 当前备份与恢复分层

| 标签与级别 | 对象 | 当前合同 |
|---|---|---|
| **C · P0 原始事实与固定重建输入** | `sessions.db`（含 messages 与 message_embeddings）、`memory2.db`、MEMORY/SELF/PENDING、plugin-data、uploads | 不允许因缓存、裁切或重构丢失；Akasha 重建复用已存向量 |
| **I · 待确认默认恢复集** | `schedules.json`、主配置、auth store、app-server token | INT-009/INT-010 尚未确认；当前只能在任务合同中显式选择并保护，不能写入默认 machine manifest |
| **C · P1 行为连续性** | `proactive.db`、`wake_proactive.db`、`drift.db`、PROACTIVE_CONTEXT、consolidation idempotency | 启用对应能力时，恢复后不重复发送、不漏 ack、不丢工作流游标 |
| **I · quota 恢复策略** | `proactive_quota.json` | INT-009 尚未确认；是否进入默认恢复集、是否只恢复当前窗口仍是 U |
| **C · P2 可重建派生** | Akasha V2 sidecar、FTS、plugin Skill/MCP catalog、workspace skill links | 允许从固定输入或插件安装清单显式重建，但要有版本与 parity 证据 |
| **C · Companion 安装状态** | 插件 manifest、安装源与版本；cache 可从 source 重装 | 恢复后重新准备 Skill/MCP runtime 并发布同一能力集合 |
| **I · P3 诊断证据** | recall/spawn/proactive traces、subagent-runs | INT-012 尚未确认；保留期、事故冻结、隐私与容量策略均不能从本表推导 |
| **F + U · 运行控制** | PID、locks、readiness、socket、临时 shell/payload 文件；app-server token 除外 | 前五类由新进程重建是代码事实；token 是持久 secret，其恢复或轮换仍按 INT-013 等待确认 |

这里的分层是恢复优先级，不是任意删除权限。特别是 FTS 和 embeddings 位于 `sessions.db` 内，SQLite 备份会自然包含它们；没有必要为了减小备份而在线拆库或删表。

## 16. 请维护者确认的最小问题集

1. 当前 dashboard 的 `update_message` 是否应继续原位改写旧消息，还是改为追加 correction/revision，保持物理 append-only？
2. `sessions.db/turns` 是否和 messages 一样永久保留，还是允许单独 retention？
3. `schedules.json` 与 quota 是否都进入默认恢复集；quota 是否允许只恢复当前窗口？
4. `config.toml`、auth store 和 app-server token 是否进入加密/受权限保护的 companion backup，而不是普通 workspace 快照？
5. 诊断 traces 希望保留多久；发生事故时是否需要冻结防 prune？
6. `memes/manifest.json` 是待恢复功能还是可以正式废弃的旧接口？
7. 是否同意下一步把本地图转成机器可读 manifest，并补目录快照与隔离 restore smoke？

确认后的结论再进入 `projectneed.md` 或新的 accepted 决策。不同意的条目保留在本文件并标记 rejected/更正理由，不用删除历史推理。
