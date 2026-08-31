# Akashic 未来方向与 GitHub Issue 拆分草案

- 状态：proposed；第 5 节已由 [0044](../decisions/0044-akashic-channel-uses-web-and-mobile-adapters.md) 和 [正式规格](akashic-channel-client-adapters.md)取代，其余方向仍不是 accepted 合同或已实现事实
- 日期：2026-08-12
- 目标父 Issue：[GitHub Issue #367 · Akashic Roadmap](https://github.com/kachofugetsu09/akashic-agent/issues/367)
- 详细 Issue：[Web/Mobile Canonical Session #368](https://github.com/kachofugetsu09/akashic-agent/issues/368)、[Project Session #369](https://github.com/kachofugetsu09/akashic-agent/issues/369)、[Project Akasha #370](https://github.com/kachofugetsu09/akashic-agent/issues/370)、[Tool Result Artifact #371](https://github.com/kachofugetsu09/akashic-agent/issues/371)
- 占位 Issue：[Plugin permission #372](https://github.com/kachofugetsu09/akashic-agent/issues/372)、[Burst self-improvement #373](https://github.com/kachofugetsu09/akashic-agent/issues/373)、[Continuous Onboarding #374](https://github.com/kachofugetsu09/akashic-agent/issues/374)
- 关联条款：OBJ-001～OBJ-003、STA-001～STA-003、CAP-001～CAP-002、CTX-001～CTX-007、SES-001～SES-008、MEM-001～MEM-011、RUN-003、RUN-007、OUT-001～OUT-005、MIG-001～MIG-002、WSP-001～WSP-004、CTRL-003、WEBUI-001～WEBUI-007、AKC-001～AKC-003
- 关联决策：[0002](../decisions/0002-context-reduction-is-a-nondestructive-projection.md)、[0006](../decisions/0006-akasha-v2-is-the-canonical-explicit-memory-engine.md)、[0021](../decisions/0021-yoyo-workspace-ledger-defines-migration-origin.md)、[0023](../decisions/0023-akashic-tokens-own-material-3-semantics.md)、[0026](../decisions/0026-plugin-rollout-is-owned-by-the-parent-turn.md)、[0030](../decisions/0030-session-context-compaction-ledger.md)、[0044](../decisions/0044-akashic-channel-uses-web-and-mobile-adapters.md)

## 1. 本文边界

本文把一次产品方向讨论按语义拆成七个可独立跟踪的 GitHub Issue。Issue 1 已由 0044 和
正式规格取代；Issue 2～4 仍是可评审草案；Issue 5～7 只保留位置。

除已提升的 Issue 1 外，本文与 #367～#374 只记录讨论结果，尚未成为现行实现合同：

- Issue 2～7 不修改 `projectneed.md` 的现行长期合同。
- Issue 2～7 不写入 `NOW.md`，也不启动实现。
- 不修改正式 Akashic workspace、数据库、配置、调度、主动流程或客户端状态。

GitHub 已创建七个子 Issue 并更新 #367。与当前 `main` 冲突的 PR #348、#349 已分别由 #371、#374 取代并关闭；旧 PR 的评审结论保留为实现输入，不能把旧分支直接当成新合同或可合并实现。

## 2. 产品北极星

Akashic 的目标是个人 Companion：它持续理解用户，能完成实际工作，也让用户看到能力随长期使用而成长。

```text
┌──────────────────────── Personal Companion ────────────────────────┐
│                                                                    │
│  Akasha                                                            │
│  理解用户、会话和项目                                              │
│         │                                                          │
│         ├──────────────┐                                           │
│         ▼              ▼                                           │
│  Companion Session   Project Session                               │
│  陪伴与跨渠道连续性   通用 Agent 在项目上下文中承担 coding work     │
│         │              │                                           │
│         └──────┬───────┘                                           │
│                ▼                                                   │
│  Plugin system                                                     │
│  提供、验证和演进可执行能力                                        │
└────────────────────────────────────────────────────────────────────┘
```

Coding Agent 不是第二个 Agent，也不是终端产品的复制。它是同一个 Akashic 在 Project Session 中获得明确工作根、项目指令和项目记忆后的工作形态。当前上下文压缩已经由 Session compaction ledger 拥有；本路线不另造一套 terminal resume、父子 Session 或 workflow engine。

容器化与 hua-home 是当前部署背景，不是本轮新增的第八个路线 Issue。正式主机迁移继续服从现有 Host Bridge、隔离实验和独立批准边界。

## 3. 当前事实与参考实践

### 3.1 当前 Akashic

当前实现仍把对话身份与渠道绑定：

- `SessionStore` 以自由字符串 `sessions.key` 为会话主键，`messages`、`turns`、handoff 和 compaction 都保存 `session_key`。
- Web 前端自行生成 `web:<uuid>`，Web API 只列出 Web 渠道会话。
- Mobile 只接受 `mobile:<uuid>`，只列出 Mobile 会话，且客户端本地生成 ID。
- Telegram 和 QQ 的默认身份仍分别是 `telegram:<chat_id>`、`qq:<user_id>` 或 `qq:gqq:<group_id>`。
- Scheduler 用 `scheduler:<job_id>` 执行无状态内部推理，再按 `channel + chat_id` 投递；Proactive 配置也以 `channel + chat_id` 定位目标。
- `message_push` 是显式渠道投递工具，不拥有目标 Session 的执行权，也不会把任意调用方的工具效果伪装成目标 Session 对话。
- Akasha 当前使用一组 `memory/akasha.db` 与 `memory/akasha-v2-index.db`，图中保存 `session_key`，burst 连续性按 Session 区分。
- Yoyo 已以 `<workspace>/migrations.sqlite3` 作为唯一迁移账本；新增迁移必须使用新 ID，离线、持锁执行。

因此，把前缀从 `mobile:` 或 `web:` 换成 `akashic:` 本身不会改写消息正文，但也不能只改 UI 字符串。所有保存或引用 `session_key` 的权威状态、运行连续性状态、派生 sidecar、配置和客户端投影都要在同一次维护迁移中对账。

### 3.2 Codex 与 Pi 可借鉴的部分

本设计参考 `/mnt/data/source-code/codex` 与 `/mnt/data/source-code/pi-mono`，只采用适合 Akashic 的边界：

| 实践 | 可采用 | 不照搬 |
|---|---|---|
| Codex ThreadStore | 权威追加历史与可重建查询投影分离；活动 turn 具有唯一 owner；cwd 是恢复时可见的明确上下文 | 不复制本地 JSONL 目录、terminal resume、fork lineage 或后台进程模型 |
| Codex compaction | checkpoint 是持久投影，不能改写旧事实 | Akashic 已有 0030，不再新增第二套 compaction |
| Pi Session | Session 持久绑定 cwd；恢复时按 cwd 重建项目资源；项目资源先经过 trust | 不采用 parent session、树形 Session 产品模型或只靠 cwd 字符串充当权限边界 |
| Pi Skills/Extensions | 专业能力按需加载，不把所有项目能力塞入核心 Prompt | 第一版只自动信任 `AGENTS.md`，不自动加载项目 Skill、MCP、插件或 extension |

## 4. 语义拆分与依赖

```text
详细合同

Issue 1  Canonical Session
   │
   ▼
Issue 2  Project Session / coding context
   │
   ▼
Issue 3  Project-scoped Akasha multi-graph

Issue 4  Large tool result artifact ── 可独立实现
   └─ Project 跨 Session 读取在 Issue 2 落地后启用

延后占位

Issue 5  Plugin rollout permission closure
Issue 6  Burst-driven capability completion
Issue 7  Continuous onboarding
```

Issue 1→2→3 是身份、项目上下文和项目记忆的明确依赖。Issue 4 的原 Session 归档与回读不依赖 Project；同项目跨 Session 访问需要 Issue 2 提供稳定 `project_id`。Issue 5～7 本轮不冻结依赖关系。

---

## 5. Issue 1 草案：Web/Mobile 共享 Canonical Session

> **已取代：** 本节保留早期推理历史，不再是实现输入。Web/Mobile 的现行确认合同见
> [0044](../decisions/0044-akashic-channel-uses-web-and-mobile-adapters.md) 与
> [Akashic Channel 与 Web/Mobile Adapter 规格](akashic-channel-client-adapters.md)。特别是
> 现行设计保留 `channel/chat_id`、不新增 `target_session_id`，并对旧客户端 Session 投影
> 执行 breaking reset，而不是迁移本地 outbox/draft 映射。

- GitHub：[Issue #368](https://github.com/kachofugetsu09/akashic-agent/issues/368)

### 建议标题

`[Session] 统一 Web 与 Mobile 的 akashic canonical session`

### 用户可见结果

Web 和已配对 Mobile 能看到、打开并继续同一条对话。渠道只表示从哪里投递和展示，不再拥有 Web/Mobile 对话身份。

每条旧 `web:*` 或 `mobile:*` 会话迁移成一条不同的 `akashic:*` 会话；迁移不合并历史。例如 `mobile:1` 与 `web:1` 即使尾部相同，也必须得到两个不同 ID。

### 目标结构

```text
┌──────────┐       ┌─────────────────────────────┐       ┌──────────┐
│ Web      │◄─────►│ akashic:<canonical uuid>    │◄─────►│ Mobile   │
│ delivery │       │ messages / turns / artifacts│       │ delivery │
└──────────┘       └──────────────┬──────────────┘       └──────────┘
                                  │
                         single active-turn writer
                                  │
                       other devices read / subscribe

Telegram / QQ ──继续使用 channel-bound session identity
```

### 已确认合同

1. 新 Web/Mobile Session 由服务端生成 `akashic:<uuidv7>`。客户端不能自行声明任意 canonical ID。
2. 迁移函数使用固定 namespace 和完整旧 ID 生成 UUIDv5：`M(old_id) = akashic:uuidv5(namespace, old_id)`。同一备份重复迁移得到相同映射，不同旧 ID 永不合并。
3. Yoyo 只迁移 `web:*` 与 `mobile:*`。Telegram、QQ、scheduler 和 programmatic ID 不改名。
4. 迁移后的 Session 增加 first-class `session_kind`；所有旧 Web/Mobile Session 默认为 `companion`。
5. 所有已认证 Web 与已配对 Mobile 都能列出并打开全部 `akashic:*` Session。当前选中项仍由每台设备本地保存，不建立账号级 active pointer。
6. 同一 Session 同时只允许一个 active turn writer。其他设备保持只读订阅，发送按钮禁用；草稿留在本地，不进入服务端队列。精确 stop 仍可用。
7. 写锁只在 turn durable terminal 后释放。进程重启把遗留 `in_progress` 收敛为 `interrupted` 后才能再次写入。
8. 不同 Session 继续并发；本 Issue 不引入 workspace 全局串行。
9. 成功送达的 Proactive 或 Schedule 结果只向指定 canonical Session 追加一次 assistant 历史。通用 `message_push` 仍只记录调用 Session 的工具事实，不自动改写任意目标 Session。
10. 外部送达和 Session 追加不升级成分布式两阶段事务。若渠道已成功而进程在历史追加前崩溃，允许历史缺少该条，但必须保留可诊断 delivery identity，不能伪造原子保证。
11. 第一版只有自动标题，不做 rename、archive 或 delete UI。既有显式 destructive Session 管理合同不因此改变。
12. 删除被 Schedule 或 Proactive 引用的 Session 必须拒绝，并列出引用；用户先 retarget 或 cancel 后才能删除。

### 一次性迁移范围

维护窗口必须停止 Runtime、admission、turn、handoff 恢复、compaction prepare、Scheduler、Proactive 与 Mobile receipt 写入，取得 workspace 锁并创建可验证备份，再执行 Yoyo。

| 状态面 | 迁移动作 | 受保护状态 |
|---|---|---|
| `sessions.db` | 改写 `sessions.key` 及 messages、turns、admissions、handoffs、compaction、audit 等全部 Session 外键/引用 | message 正文、seq、message ID、turn ID、interaction、compaction 内容与来源不变 |
| Akasha sidecar | 以迁移后的 canonical history 重建 Companion sidecar，或执行能证明等价的确定性重绑 | 用户/assistant 文本、embedding bytes、固定算法参数不变 |
| `proactive.db`、`wake_proactive.db`、Drift/continuity state | 精确重绑目标 Session 与按 Session 保存的连续性键 | dedupe、cooldown、pending ack、cursor 与终态不丢失 |
| `schedules.json`、Proactive config | 从 `{channel, chat_id}` 派生并写入 `target_session_id`；保留独立 delivery channel/chat | fire time、prompt、message、enabled、run count 与投递目标不变 |
| Mobile realtime state | 重绑 claim、attachment、inbox、receipt、outbox 与历史投影中的 Session 引用 | delivery/request identity、附件、终态与已确认回执不变 |
| Android app-private state | Room/DataStore、选中项、Incoming Share 与 outbox 使用迁移映射 | 消息缓存、未发送草稿和服务端身份不被跨服务器合并 |

配置目标从单一投递地址拆成两个角色：

```text
旧：{ channel: "mobile", chat_id: "1" }

新：{
  target_session_id: M("mobile:1"),
  delivery_channel: "mobile",
  delivery_chat_id: "1"
}
```

迁移不扫描正文猜目标，不用尾部 UUID 自动匹配，不允许半新半旧运行。服务端与新 APK 必须协调发布；旧 APK 连接新协议时明确拒绝升级，不能继续创建 `mobile:*`。

### 持久化增改减合同

| 对象 | 正常增加 | 允许原位或逻辑变化 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| Canonical Session | 服务端创建 `akashic:<uuidv7>` | 标题、updated_at、模型选择和 turn 状态按既有状态机更新 | 只接受 SES-003 的显式删除 | Session owner；workspace backup、Yoyo receipt、映射清单、SQLite integrity |
| `messages` | 正常 turn 只 INSERT | 本 Issue 不授权正文 UPDATE | 只接受显式 interaction/session 删除 | SessionStore；迁移前后规范化 message 快照 |
| Legacy mapping | 维护迁移一次性增加 old→new 映射审计 | 成功回执后不可改写 | 当前不得自动减少 | Migration owner；固定 namespace、映射 digest、Yoyo receipt |
| 设备选择和缓存 | 每设备按服务端 Session 建派生投影 | 选择、已读和缓存按设备更新 | 只删除可重建缓存；不得级联服务端正文 | 原生客户端；服务端 history 与客户端重拉对账 |

### 验收

- 对含 Web、Mobile、Proactive、Schedule、compaction、附件、receipt 和 Akasha 数据的一次性 workspace 执行迁移，所有旧 ID 一对一映射且无悬空引用。
- 迁移前后 message/turn/interaction/compaction 的规范化内容一致；Akasha 固定召回 oracle 等价。
- Web 创建 Session 后 Mobile 能打开并继续；Mobile 创建后 Web 同样可见。
- 两台设备同时发送时只有一台取得 writer；另一台可读、可 stop、不可排队发送，terminal 后可继续。
- Schedule/Proactive 精确写入目标 canonical Session；通用 `message_push` 不越权写入目标历史。
- 旧 APK 被明确拒绝；新 APK 完成真实升级、冷启动、历史打开和发送验证。
- 迁移失败不写 Yoyo 成功回执，Runtime 不启动；从备份恢复后旧系统数据完整可读。

### 非目标

- 不把 Telegram/QQ 变成可切换多 Session UI。
- 不合并历史相似的旧 Session。
- 不做跨用户、跨 workspace 或跨服务器 Session。
- 不新增消息队列、父子 Session、rename/archive/delete UI。

---

## 6. Issue 2 草案：Project Session 与 Coding Context

- GitHub：[Issue #369](https://github.com/kachofugetsu09/akashic-agent/issues/369)

### 建议标题

`[Project] 为通用 Akashic 增加项目 Session 与 coding context`

### 用户可见结果

用户新建 Session 时可以显式选择一个服务器路径。该 Session 从此持续以这个工作根、仓库身份、适用 `AGENTS.md` 和项目记忆完成 coding work；同一仓库可以有多个平级 Session。

### 已确认合同

1. Session 使用 `session_kind = companion | project`。Project Session 必须持久保存 `project_id`、不可变 `working_root` 和创建时的 project identity snapshot。
2. 第一版不建 `projects` 表。每个 Project Session 重复保存同一个确定性 `project_id`，UI 按 ID 分组或着色。
3. 有 Git remote 时，`project_id` 从去凭据、规范化后的 origin identity 派生；没有 remote 时从 canonical `git-common-dir` 派生。算法带版本。相同 remote 的 clone/worktree 属于同一 Project，fork remote 属于不同 Project。
4. `working_root` 是用户选择的、服务端已存在的绝对路径，可以是仓库子目录。它是默认 cwd 和上下文根，不是硬 sandbox；Agent 可以按既有工具权限跳出根目录，但不会因此重新绑定项目。
5. 创建后不允许修改、清除或重新指定工作根。需要其他根时创建新 Session。
6. 用户可以从最近 Git 根列表选择，也可以手工输入绝对服务端路径。只有路径存在、位于 Git repository/worktree 且 identity 可确定时才能创建。
7. 同一 Project 可有多个平级 Session，不建立用户可见 parent/child。内部 programmatic 子任务可以继承 project scope，但不进入普通抽屉。
8. 同一 repository/worktree 的不同 Session 可以同时修改文件；第一版不加跨 Session 文件锁或警告。UI 显示 project、root、branch、dirty 和同项目 active Session 数量，帮助用户判断并发。
9. 抽屉保持扁平。Companion 使用 primary 语义色，Project 使用 tertiary，路径不可用使用 neutral；error 色只表示真实错误。颜色必须同时有图标或文字标签，不能成为唯一信息载体。
10. Project 自动标题为 `repo · 首条用户消息`；第一版仍只支持标题搜索。
11. 工作根暂时不可访问时，Session 从普通列表隐藏并进入“不可用项目”，权威数据永久保留；路径恢复后自动回到列表。

### `AGENTS.md` Prompt 合同

- 每个新 turn 从 Git root 到 `working_root` 重新发现适用 `AGENTS.md`；更深层文件只在工具将要接触对应路径时按需加载。
- 离开工作根执行操作时，按目标路径重新检查适用 `AGENTS.md`，但 Session 的 `project_id` 和 `working_root` 不变。
- 更深层 `AGENTS.md` 对其目录树更具体；direct system/developer/user 指令仍高于仓库指令。
- 总注入上限 32 KiB，来源路径和截断必须可见。文件变化在下一 turn 替换旧 Prompt 内容，不把旧版本留成并列真相。
- 第一版只自动信任 `AGENTS.md`。项目内 Skill、MCP、插件、extension、prompt template 和其他约定文件都不自动加载。

### 项目记忆与 programmatic 规则

- Companion Session 继续读取全局 `MEMORY.md` / `SELF.md` 和 Companion Akasha。
- Project Session 以全局 `MEMORY.md` / `SELF.md` 作为人物与通用偏好基线；项目事实写入 Project Akasha，不复制到全局 Companion 图。
- ordinary Markdown memory plugin 可以判断编码偏好是否跨 Project 通用，并从已提交的精确来源直接更新 profile；本 Issue 不恢复 `PENDING.md` 或 optimizer，也不新增审批状态机。
- `scheduler:*` 始终排除记忆。
- `programmatic:*` 缺少 `skip_post_memory` 时默认排除；只有创建时显式 `persist_memory=true` / `skip_post_memory=false` 才学习。现存未标记 programmatic Session 在下一次派生重建时按新默认排除。
- Project 内部子任务只有在显式继承 `project_id` 且显式开启记忆时，才能进入对应 Project 图。

### 持久化增改减合同

| 对象 | 正常增加 | 允许原位或逻辑变化 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| Project Session fields | Session 创建时一次写入 kind、project_id、root、identity snapshot | 标题、updated_at、模型选择可按现行规则更新；project/root 不可改 | 跟随显式 Session 删除 | Session owner；SessionDB backup、identity 重算报告 |
| AGENTS Prompt block | 每 turn 从当前文件生成临时块 | 文件变化使下一 turn 的块替换 | Prompt 结束即释放；不删除仓库文件 | Prompt owner；source list、digest、截断报告 |
| Project preference candidate | 真实 completed turn 可产生已提交的精确来源 | ordinary Markdown plugin 以独立 document receipt 更新 profile | 只按 MEM-002～MEM-004 | Markdown memory owner；backup、document receipt、source_ref |
| Project unavailable state | 路径探测派生，不新增权威删除标记 | 可用/不可用随主机状态变化 | 仅派生投影可重建 | Session query/UI owner；Server path probe 与 SessionDB unchanged |

### 验收

- 同一 Git remote 的两个 worktree 得到同一 `project_id`，fork remote 不同；无 remote 的同一 `git-common-dir` 稳定一致。
- 一个 Project 建立多个 Session 后，共享 Project 身份但保持独立对话、turn 和 compaction。
- 每个 turn 重新读取适用 `AGENTS.md`；嵌套覆盖、32 KiB 截断、来源展示和跳出根目录行为有真实仓库测试。
- `working_root` 创建后不可重绑；路径离线时隐藏但数据不减少，恢复后重新可见。
- Web/Mobile 显示相同 Project Session，语义颜色在 light/dark/warm-paper 下均可区分且满足可访问性。
- scheduler 与默认 programmatic 不进入 Markdown、Memory2 或 Akasha；显式开启的 Project child 只进入对应 Project scope。

### 非目标

- 不增加 Projects 表、父子 Session 或 Project workflow engine。
- 不把 working root 当安全 sandbox，也不承诺阻止 Agent 访问外部路径。
- 不自动信任仓库里的 Skill、MCP、插件或 extension。
- 不重做 Session compaction，也不要求 terminal 风格 resume。

---

## 7. Issue 3 草案：Project-scoped Akasha 多图记忆

- GitHub：[Issue #370](https://github.com/kachofugetsu09/akashic-agent/issues/370)

### 建议标题

`[Memory] 为 Companion 与各 Project 建立独立 Akasha 图`

### 用户可见结果

Companion 对话继续形成一张长期关系图；每个 `project_id` 形成自己的图。同一 Project 的多个 Session 延续项目记忆，不同 Project 不互相污染。

### 目标结构

```text
                         Host scope registry
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
       Companion scope       Project A scope   Project B scope
       akasha.db + index     akasha.db+index   akasha.db+index
                │                 │                 │
       all companion turns   A 的多个 Session   B 的多个 Session

每个 Session 的 burst continuity 仍只在该 Session 内推进
```

### 已确认合同

1. Companion 使用一张图；每个稳定 `project_id` 使用一张互相独立的图。
2. 保留现有 `memory/akasha.db` 与 `memory/akasha-v2-index.db` 作为 Companion scope。Project scope 放在 `memory/akasha-projects/<sha256(project_id)>/`，manifest 保存 hash、规范化 identity、算法版本与创建时间。
3. Scope registry、路径选择和 lifecycle 由宿主拥有，不修改 upstream `MemoryCycle`、burst 算法、权重、阈值或其他算法参数。
4. 每个 scope 都有独立 graph SQLite 与 sparse index。配置只允许选择存储路径和非算法输出边界，不允许为 Project 调参。
5. Project runtime 在首次 retrieval 或 commit 时惰性打开；打开后一直保留到 Core shutdown，不做 LRU、自动 close 或 Dashboard residency 指标。
6. 同一 Project 的并发 turn 共享一个 runtime 和单写者 commit gate；不同 Project runtime 可以并行。
7. 每个 scope 保留当前确定性因果顺序：`started_at + session_key + user_seq + turn_id`。迟到的更早 turn 触发该 scope 的局部派生重建。
8. burst 连续性仍以 `session_key` 隔离。共享 Project 图不把两个 Session 误判成同一 burst。
9. 新 Project 冷启动时，当前 Session history、全局 `MEMORY.md` / `SELF.md`、适用 `AGENTS.md` 和仓库内容进入各自既有 Prompt 通道；Project Akasha 在没有已提交样本时明确为空，不回退查询 Companion 图，也不把仓库全文自动摄入图。
10. Session Inspector 只显示当前 scope；Dashboard 可以显式选择 Companion 或某个 Project scope。
11. Project 路径不可用时保留图和 manifest，不自动删除。图损坏只使对应 scope fail-loud，其他 scope 继续服务。
12. 第一版不提供自动 repair、正式修复 CLI 或 UI 按钮；必须写人工备份恢复和确定性重建 runbook。
13. 第一版只适配 Akasha。退役 `memory2.db` 归档不参与 project scoping，也不在本 Issue
    导入、改写或删除。

### Companion 图迁移

Canonical Session 迁移改变 `session_key`，不改变消息正文或 embedding。因为 `session_key` 仍参与 causal order、burst continuity 与 sidecar identity，迁移不能假设文件字节不受影响。

维护迁移从迁移后的 canonical history 确定性重建 Companion sidecar，并在发布前比较：

- 规范化合法 turn 集与排除集。
- embedding identity 与 bytes。
- canonical logical graph、burst membership 和固定 recall oracle。
- 新默认下被排除的历史 programmatic Session 清单。

预期差异只有 Session identity 重绑和已确认的 programmatic 默认排除。其他图结构、权重、burst 或召回差异必须中止发布并保留旧 sidecar 与备份。

### 持久化增改减合同

| 对象 | 正常增加 | 允许原位或逻辑变化 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| Scope manifest | 首次打开 Project scope 时原子创建 | immutable identity；运行状态不写入 manifest | 当前不得自动减少 | Host scope registry；manifest digest、project_id mapping |
| Graph SQLite | completed、允许学习的 turn 通过 `MemoryCycle.commit` 增加/演进派生图 | 算法按固定状态机更新；不是人工 SQL 管理面 | 只有显式 Session/Interaction 删除后的协调重建，或用户明确删除整个 workspace | Akasha scope owner；source history、embedding audit、backup、logical snapshot |
| Sparse index | 从固定 source 构建或增量推进 | 可确定性 supersede/rebuild | 只作为派生索引随受控重建替换 | Akasha index owner；source digest、index validation |
| In-memory runtime | scope 首次访问时创建 | commit/retrieve 更新内存视图 | Core shutdown 释放 | Runtime owner；持久 sidecar 与重开 smoke |

### 验收

- Companion、Project A、Project B 使用三组不同文件和 runtime；向 A 提交不会改变 Companion/B 的规范化快照。
- A 的两个 Session 可召回同一项目事实，但 burst membership 保持各自 Session 内连续。
- Project 首次访问惰性创建；空图不回退 Companion，Core shutdown 后可从 sidecar 重开。
- 同 Project 并发 commit 保留确定性顺序；不同 Project 可并行且互不持锁。
- 单图损坏只阻断对应 scope；错误包含 scope/project/path，其他图仍能查询和提交。
- Companion 重建 Gate 能发现额外 turn 缺失、embedding 漂移、burst 漂移和 recall 漂移；失败不切换正式 sidecar。
- 算法 upstream identity 与参数在改动前后逐项相同。

### 非目标

- 不建立跨 Project 统一 Akasha 图。
- 不做 Companion fallback、自动图合并、自动 GC、自动 repair 或参数自调。
- 不把 Memory2、Markdown memory 或仓库全文复制成 Project graph。

---

## 8. Issue 4 草案：大型 Tool Result 归档与按需回读

- GitHub：[Issue #371](https://github.com/kachofugetsu09/akashic-agent/issues/371)

### 建议标题

`[Context] 归档超大 tool result 并提供稳定引用回读`

### 用户可见结果

工具返回超大正文时，Session 仍保存完整结果；模型先看到有界摘要和稳定引用，确有需要时按范围回读，不因为 provider 输入预算而静默丢失关键工具事实。

### 已确认合同

1. 只有超过当前 provider-safe projection budget 的 tool result 才归档；正常小结果继续使用现有路径。
2. Artifact owner 保存完整 raw body 和必要 transport status。发给模型的是有界 projection、长度/类型信息和稳定 `artifact_ref`。
3. 超大结果归档失败时，当前 turn fail-loud。不得把失败伪装成截断成功，也不得继续生成一个无法解析的 ref。
4. Artifact 是 Session 工具事实的一部分，但物理存储与 `messages.content` 分离；compaction 只投影 ref，不能删除 raw body。
5. Companion artifact 只允许原 Session 回读。Project artifact 在 Issue 2 提供稳定 `project_id` 后，可以由同 Project 的其他 Session 只读访问。
6. 稳定 ref 必须在 hard-input-overflow 路径仍可见；不能先把正文裁掉，再让模型永远拿不到回读入口。
7. raw token meter、projection token 与实际 readback token 分开记录；不能把 masked projection 的估算值宣传成 raw exact usage。
8. transport status、正文前缀和 Inspector/API 展示使用同一字段合同，不让归档正文与工具 trace 对同一结果给出不同状态。
9. PR #348 只作为历史实现与评审输入。新实现从当前 `main` 重做最小可评审 diff，不直接合并冲突分支。

### 目标结构

```text
Tool execution
      │ full result
      ▼
┌──────────────────────┐
│ Artifact owner       │─── immutable raw body + metadata
└──────────┬───────────┘
           │ stable ref + bounded projection
           ▼
      Prompt history
           │
           └──── read_tool_result(ref, range) ────► bounded readback
```

### 持久化增改减合同

| 对象 | 正常增加 | 允许原位或逻辑变化 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| Artifact metadata/body | 超阈值工具完成时先完整持久化，再把 ref 交给 turn | immutable body；可追加诊断/访问事件，不能改写原结果 | 只随 SES-003 的显式 Session/Interaction 删除；普通 compaction/retention 不得减少 | Session artifact owner；digest、length、content type、source turn/tool identity、backup |
| Prompt projection | 每次请求从 artifact 派生 bounded block | 随 provider budget 重建 | 请求结束释放 | Context owner；ref 始终可解析、payload capture |
| Readback event | 每次有界读取追加 tool trace/usage | terminal status 按执行状态机更新 | 跟随所属 Session 的显式删除 | Tool runtime；range、bytes、tokens、artifact digest |

### 验收

- 阈值上下边界、小结果、文本、JSON、二进制描述和 transport status fixture 行为确定。
- 超大结果成功时 raw digest/length 与工具原始输出一致，模型首轮只看到有界 projection 和可用 ref。
- 归档 I/O、事务或完整性失败使 turn 明确失败，Session 中没有悬空 ref 或半个 artifact。
- hard input overflow 仍能让模型获得 ref 并按页读取；readback 本身继续受输入预算约束。
- compaction 前后 artifact body/digest 不变，Session 重开后仍可回读。
- Companion 跨 Session 访问被拒绝；Project 同 scope 访问在 Issue 2 落地后按 `project_id` 授权。
- 显式 Session/Interaction 删除先备份，再原子减少对应 artifact；其他 Session artifact 逐项不变。

### 非目标

- 不归档所有工具结果。
- 不用 artifact 取代 canonical message、turn 或完整外部效果 trace。
- 不提供跨 Project、跨 workspace 或匿名 ref 读取。

---

## 9. 延后占位 Issue

以下三项只保留路线位置。本轮不记录现状问题、不选择技术方案、不定义依赖、优先级或验收。

### Issue 5

- GitHub：[Issue #372](https://github.com/kachofugetsu09/akashic-agent/issues/372)

`[Plugin] 收敛插件自验证与晋升权限链`

留位：未来统一插件自验证、授权和正式晋升边界。

### Issue 6

- GitHub：[Issue #373](https://github.com/kachofugetsu09/akashic-agent/issues/373)

`[Self-improvement] 结合 Akasha burst 完成自主能力补全`

留位：未来探索 Akasha 理解与插件能力成长的闭环。

### Issue 7

- GitHub：[Issue #374](https://github.com/kachofugetsu09/akashic-agent/issues/374)

`[Onboarding] 建立连续、易理解的首次配置体验`

留位：未来继续收敛首次使用和配置引导。

## 10. 发布与收口状态

2026-08-12 已完成以下外部协调：

1. Issue 1～4 分别发布为 #368～#371，保留目标、合同、持久化语义、验收和非目标。
2. Issue 5～7 发布为 #372～#374，只包含标题、一行方向和“不展开讨论”的边界。
3. #367 已增加七个子 Issue checklist，并区分详细项与占位项。
4. PR #348、#349 已留下 superseded 说明并关闭，未标记为已交付。

后续只有在长期语义再次经维护者确认后，才把对应合同提升到 `projectneed.md` 和 accepted decision；被接受且可立即接手的实施项才进入 `NOW.md`。

## 11. 本地设计验收

- Issue 1～4 能独立复制成 GitHub Issue，不依赖本轮聊天记录才能理解。
- Issue 5～7 除标题与一行方向外没有技术讨论。
- Canonical Session、Project、Akasha scope 与 artifact 的 owner 没有互相替代。
- 每类持久状态都说明正常增加、允许更新、物理减少、owner 和恢复证据。
- 现有 compaction、Telegram/QQ 身份、Host Bridge 和正式 workspace 边界未被暗中扩张。
- 代码分支只修改项目文档；外部写入限于 #367～#374、PR #348/#349 的路线收口，以及承载本文档的 Draft PR。没有数据库、配置、正式 workspace、服务或消息投递副作用。
