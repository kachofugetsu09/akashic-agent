# 插件 install/uninstall/revert 与 turn 边界发布设计合同

- 状态：implemented；Core 与 Skill 已实现，真实 Fitbit source 适配和扩展 crash matrix 记录在 `NOW.md`
- 日期：2026-08-08
- 基线：`origin/main@31b976d82cbd5766e6450d7e287ceda71d9b7573`
- 关联条款：OBJ-002、STA-001～STA-003、CAP-001～CAP-002、ERR-001、RUN-003、PLG-001～PLG-013、BAK-001、TST-001～TST-006
- 关联决策：[0008](../decisions/0008-plugin-runtime-publishes-only-committed-snapshots.md)、[0015](../decisions/0015-cleanup-does-not-own-turn-or-restart-finality.md)、[0024](../decisions/0024-plugin-self-validation-uses-stable-and-latest.md)

## 1. 用户可见目标

Agent 修改插件后只使用三个直观的插件管理动作：

```text
plugin-install    安装或更新，创建当前 turn 的待提交候选
plugin-uninstall  创建当前 turn 的待提交卸载
plugin-revert     撤销当前 turn 最近一次尚未提交的 install/uninstall
```

Agent 不再执行 `plugin-status`、`plugin-promote`、`plugin-discard` 或手工 restart。Core 保留 stable/latest、准备、验证绑定、提交、丢弃、排空、端点切换和恢复等内部机制，但不要求 Agent 理解或编排这些阶段。

`plugin-install` 成功后，当前 turn 自己仍绑定原 stable；由该 turn 启动的 attached programmatic child 自动绑定刚安装的候选。Agent 根据真实 child 结果和工具轨迹决定：符合目标就正常结束 turn，不符合就先 `plugin-revert`，修改源码后继续递归。

当前 turn 正常结束且候选已完成 programmatic 验证时，Core 才在旧 lease 释放后自动切换。下一 turn 自动绑定新 stable。

## 2. 当前事实与有意语义变化

当前实现已经具备不可变 artifact、stable/latest pointer、snapshot lease、reload journal、managed service 切换、Channel 切换和失败恢复骨架。普通卸载也已经由 runtime owner 异步等待旧 lease 排空。

当前 PLG-013、决策 0024 和递归自验证设计要求 Agent 在 `plugin-install` 后显式选择 `runtime=latest`，再根据验证结果执行 promote/discard。改变独占 managed service 或 Channel 的 candidate 被 `endpoint_coexistence` Gate 拒绝，因为同进程 stable/latest 不能安全拥有同一个进程级资源。

2026-08-07 Fitbit 事故中，调用者为完成这组内部操作而绕过 coexistence Gate，随后形成 stable snapshot、全局 endpoint 和 admission 分裂。本设计保留 Gate 保护和 programmatic 自验证，改变的是 Agent 可见操作面与最终提交 owner：

1. latest 继续存在，但只由发起 install 的 turn 所创建的 programmatic child 因果继承。
2. 验证失败由 Agent 表达为 `revert`；Core 内部完成 candidate discard。
3. 验证通过不再需要 promote；当前 turn 正常完成就是对尚未 revert 候选的提交授权。
4. turn 后的 lease 排空、endpoint 切换、stable/manifest 提交和恢复全部由 Core 拥有。
5. implementation 必须新建 accepted 决策，明确勘误 0024 的公开 promote/discard 流程，不能抹除历史。

## 3. 能力与状态 owner

```text
Agent / Shell
└─ install / uninstall / revert
                 │
                 ▼
┌────────────────────────────────────────┐
│ Core plugin rollout owner              │
│ turn-local pending、programmatic 绑定、 │
│ lease barrier、提交、恢复与返回说明     │
└──────────────┬─────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
managed service host  ChannelHost
进程/listener owner    外部入口生命周期 owner
```

- `capability_owner`：Core plugin runtime。
- `consumer_scope`：所有 Akashic 插件；Fitbit 只是事故复现和真实验收对象，不获得特判。
- `runtime_patch`：required。turn lineage、snapshot、manifest、journal、managed service 和 Channel 的提交与恢复都由 Core 拥有。
- `authoritative_state_owner`：RuntimeSnapshotStore/PluginManager 拥有 active generation；reload journal 拥有 pending operation 与恢复证据；installer/manifest owner 拥有已安装代码；插件拥有 plugin-data。
- `client_only_alternative`：不存在。Skill 可以减少误操作，不能原子协调 lease、endpoint、pointer、manifest 和 crash recovery。

## 4. Turn-local 操作合同

### 4.1 `plugin-install`

命令返回成功前，Core 必须完成 artifact 身份、manifest、依赖、Skill/MCP/tool catalog、managed service/Channel 声明、名称冲突、受保护 write set 和恢复源检查，并把 pending install 绑定到当前 turn。

成功返回表示候选已经可以由当前 turn 的 programmatic child 验证，不表示正式 endpoint 已经切换：

```text
Fitbit 候选版本安装成功。

当前 turn 仍使用原版本。
从现在开始，本 turn 启动的 programmatic 验证会自动使用新版本。

请执行 programmatic 验证：
- 如果行为和工具轨迹正确，正常结束当前 turn；
- 如果不正确，执行 plugin-revert，然后继续修改。

验证通过并结束当前 turn 后，系统会自动重启 Fitbit 服务。
下一 turn 使用新版本。你不需要 promote、discard 或 restart。
```

失败必须非零退出，并说明失败阶段、具体对象、实际影响和 Agent 唯一下一步。未成功持久登记 pending install 时，不得留下 turn 后隐藏切换。

### 4.2 programmatic 自动绑定

候选按因果来源隔离：

```text
Turn T 执行 install S1
├─ T 本身                         → 继续使用 S0
├─ T 创建的 attached child        → 自动绑定 S1
├─ 其他 session/turn 的 child     → 继续使用各自 stable
└─ detached child                 → 不得作为本次验证
```

Core 记录 `owner_turn_id + candidate_generation_id + source_revision`。child 在创建时冻结候选身份；T 后续安装其他 revision 不得让已经运行的 child 半途换代。

至少一个绑定当前候选的 attached child 必须正常完成，当前 turn 才能授权提交。child 失败、取消、超时、身份不一致或根本没有运行时，Core 在 turn 结束时取消 pending install，不发布候选。

Core 只核对 child 确实绑定候选、真实 terminal/tool trace 存在且没有越过 write-set 边界。Fitbit 领域结果与轨迹是否符合修改目标由 Agent 判断；不符合时 Agent 必须在结束 turn 前执行 `plugin-revert`。

### 4.3 `plugin-uninstall`

`plugin-uninstall` 只登记当前 turn 的 pending uninstall、阻止新的非当前请求取得目标插件，并把最终停止和删除交给 Core。返回时不得同步等待当前 turn 自己持有的 lease，也不得提前删除代码。

```text
Fitbit 卸载已确认。

当前 turn 仍可完成正在进行的操作。
本轮结束后，系统会自动停止 Fitbit 服务并删除已安装代码。
plugin-data 将保留；下一 turn 不再加载 Fitbit。

如果要取消本次卸载，请在当前 turn 结束前执行 plugin-revert。
```

### 4.4 `plugin-revert`

`plugin-revert` 不接受历史版本，也不跨 turn 回滚。它只撤销当前 turn 最近一次尚未提交的成功 `install` 或 `uninstall`：

```text
install S1 → revert    取消候选 S1；stable S0 始终不变
uninstall  → revert    取消卸载；代码、能力和 stable 始终不变
```

`revert` 成功后，Core 追加撤销证据并清理本次 candidate/pending operation；不会删除 plugin-data。当前 turn 可以继续修改并再次 install。

下列情况 fail-loud：

- 当前 turn 没有尚未提交的 install/uninstall；
- 最近操作已经被 revert；
- 操作属于其他 turn；
- 当前 turn 已进入 terminal 封口；
- 调用者试图用 revert 回滚上一 turn 已提交的版本。

建议错误：

```text
无法执行 revert：当前 turn 没有尚未提交的插件操作。
revert 只能撤销本 turn 最近一次 install 或 uninstall，不能回滚上一 turn。
```

## 5. Turn 边界与最终结果

```text
当前 turn 内                         当前 turn 正常结束后
────────────                         ──────────────────
install/uninstall 仍可 revert        pending operation 封口
父 turn 继续绑定旧 snapshot          等待全部旧 generation lease 归零
programmatic child 验证候选           切换 endpoint 与 snapshot
旧代码和恢复源必须保留                成功后清理不再引用的旧代码
```

只有以下条件同时满足，pending install 才进入 turn 后切换：

```text
install 前置检查成功
AND 当前 candidate 完成真实 attached programmatic 验证
AND 没有 revert
AND parent turn 正常完成
```

parent turn interrupted/failed、验证 child 非正常终结或候选身份变化时，Core 自动取消候选，旧 stable 不变。pending uninstall 只有在 parent turn 正常完成且没有 revert 时才执行。

Core 不主动创建 turn、不主动发送用户消息，也不向 SessionDB 追加伪造对话。用户下一次主动发起 turn 时，runtime 从 journal 与当前 snapshot 派生最近一次相关操作的事实，供 Agent 自然语言回答，不要求 status 或轮询。

## 6. Core 通用安全检查

Core 不判断 Fitbit 算法和业务结果，只执行四类通用判断：

1. **身份一致**：source revision、artifact digest、manifest/pointer、snapshot、进程和 current/previous identity 一致。
2. **运行一致**：Skill、MCP、tool catalog、managed service、listener 和 Channel 属于同一 generation；启动/readiness 条件在期限内通过。
3. **受保护状态不受伤害**：SessionDB、memory、plugin-data、旧 stable 和未授权外部效果保持合同要求。
4. **恢复真实可用**：previous 在提交前保留；失败时重新启动并验证旧 endpoint，而不是只恢复 pointer。

## 7. Fitbit managed service 与 Channel

### 7.1 独占 managed service

候选 S1 不能在 programmatic 验证期间接管 S0 的正式端口。Core 必须为候选提供通用的隔离 service host/临时 endpoint；不能通过 `active is None` 绕过 coexistence Gate。

```text
programmatic child → 隔离 endpoint 上的 S1
正式请求           → S0

turn 结束后：暂停 admission → 排空 S0 → 停 S0 → 启并验证 S1
              ├─ 成功：共同提交 snapshot/pointer/manifest
              └─ 失败：恢复并验证 S0，再恢复 admission
```

正常路径只重启该插件 managed service，不重启整个 Akashic runtime。Core 崩溃恢复必须在 service/channel host 绑定后执行。

### 7.2 Channel

不新增 Agent 可见 Channel 状态。现有生命周期合同必须加强：

- `stop()` 返回前停止新 ingress、交回或完成在途工作，并释放外部连接/ownership；
- `start()` 返回前取得外部入口并通过 readiness；
- 不能证明释放或就绪时，按 Channel 名称、资源 identity 和阶段 fail-loud。

服务与 Channel 作为同一 generation 整体提交：

```text
old Channel.stop()
        │
managed service S0 → S1
        │
new Channel.start()
        │
        ├─ 成功：共同提交
        └─ 失败：恢复 service S0 + old Channel，并重新验证
```

唯一 bot token/webhook 等资源不能在隔离验证中复制时，programmatic 只验证不接管正式 ownership 的部分；正式 ownership 只能在 turn 后维护切换中确认。不能证明安全局部切换时拒绝发布，不自动扩大为整 runtime 重启。

## 8. 持久化增减与代码清理

| 对象 | install | uninstall | revert | turn 后物理减少 |
|---|---|---|---|---|
| canonical source | 只读固定 revision | 不修改 | 不修改 | 永不由本流程删除 |
| immutable installed artifact | 增加 candidate；旧 stable 保留 | 当前 turn 内保持 | install 候选可清理；uninstall 保持原代码 | 提交、稳定检查、旧 lease 归零且不再被恢复引用后，旧 artifact 可删除 |
| manifest/pointer/catalog | 当前 turn 内登记 pending，不改父 turn binding | 登记 pending，不提前移除 | 恢复操作前视图 | commit 时原子切换或移除 active entry |
| plugin-data | 候选默认不得污染；激活后由插件写入 | 始终保留 | 始终保留 | 本流程永不删除 |
| reload journal | 追加 operation/阶段 | 追加 operation/阶段 | 追加撤销事实 | 本任务不定义 retention，不删除历史 |
| SessionDB/messages、memory | 只发生正常 turn 追加 | 同左 | 同左 | 不得因 rollout 改写或删除 |

`install` 不在当前 turn 内删除旧代码。新代切换、readiness、identity、journal commit 和旧 lease 排空全部完成后，旧 artifact 才失去恢复职责并允许清理。

`uninstall` 在当前 turn 内不删除代码，使 revert 始终可撤销。turn 后成功停止 endpoint、排空 scope 并提交 manifest/catalog 移除后，删除已安装 artifact、venv 和能力投影，但保留 plugin-data、journal 和 canonical source。

清理失败不能反向改写已经合法提交的新 stable 或 turn terminal：

- install 的旧 artifact 清理失败：新版本仍成功，下一 turn 明确报告残留路径与清理错误；
- uninstall 的代码删除失败：插件保持停用且不再发布能力，下一 turn说明卸载未完整清理；再次 uninstall 只重试残留清理，不假报插件不存在。

## 9. 并发、取消与崩溃

- 同一时间只允许一个全局未决 candidate；冲突 install/uninstall fail-loud，不隐式排队。
- `revert` 只操作调用 turn 拥有的最近 pending operation。
- rollout 等待全部旧 generation lease，不只等待发起 turn；持有 lease 的调用栈不得同步等待自己。
- admission 暂停后的每条分支必须 resume 或明确 degraded，不能无限等待。
- prepared、等待 lease、旧 endpoint 已停、新 endpoint 已启未提交和恢复途中崩溃，都从 journal、pointer、manifest 和真实进程/Channel identity 恢复。
- 重复安装相同 revision 返回幂等结果，不因 installer 自己创建的 runtime symlink 误判 artifact。
- operation 在 turn terminal 前被取消或 revert 时不得产生后置 endpoint 效果。

## 10. 实现任务合同

### Goal

Agent 只用 `install/uninstall/revert` 管理插件；install 后的 programmatic child 自动验证当前候选，失败可在同一 turn revert 并继续递归，成功则由 Core 在 turn 后安全提交；uninstall 与 revert 保持代码、plugin-data 和错误语义明确。

### Success criteria

- [x] Agent 可见插件管理动作只有 install、uninstall、revert。
- [x] 当前 turn 自己保持旧 snapshot；其 attached programmatic child 自动、精确绑定当前 candidate。
- [x] 没有真实 programmatic 验证、child 非正常结束、parent 非正常结束或已经 revert 时，candidate 不得发布。
- [x] revert 只撤销同一 turn 最近 pending install/uninstall，不能跨 turn 回滚。
- [x] install/uninstall 返回说明已发生、未发生、后续动作和 Agent 下一步。
- [x] 独占 service/Channel 不绕过 coexistence Gate，在隔离验证或 turn 后维护切换中处理。
- [x] endpoint、snapshot、pointer、manifest 和 admission 共同提交或共同恢复。
- [x] install 成功后允许清理无引用旧 artifact；uninstall 清理已安装代码但保留 plugin-data、journal 和 source。
- [x] 下一次用户主动 turn能获知最终切换、恢复或清理失败事实，不需要 status/polling。
- [ ] targeted tests、semantic mutants、change-impact Gate 和一次性 workspace 真实插件场景通过。

### Change intent

```yaml
change_type: feature
semantic_delta: breaking
capability_owner: core
consumer_scope:
  - agent plugin development flow
  - all installed plugins
  - managed services
  - plugin channels
runtime_patch: required
runtime_patch_reason: "Core 独占 turn lineage、snapshot、manifest、journal、managed service 和 Channel 的提交与恢复。"
authoritative_state_owner: "Plugin rollout owner + RuntimeSnapshotStore + installer/manifest owner；plugin-data 仍由插件拥有。"
client_only_alternative: "只改 Skill 无法关闭 Agent 手工 promote/discard 与独占 endpoint 分裂路径。"
invariants:
  - "父 turn 从 admission 到 terminal 始终绑定旧 snapshot。"
  - "attached programmatic child 按 owner turn 精确绑定候选。"
  - "未验证、已 revert 或非正常终结的候选不得发布。"
  - "独占 endpoint 先停 admission，再等待全部旧 lease 归零。"
  - "endpoint、snapshot、pointer 和 manifest 共同提交或恢复。"
  - "plugin-data、SessionDB/messages 和 memory 不因插件管理减少。"
protected_state:
  - "正式 Akashic workspace 中既有 session、memory、附件、调度和凭据。"
  - "插件 canonical source。"
  - "旧 stable artifact，直到新代提交和恢复检查完成。"
  - "workspace/plugin-data。"
allowed_paths:
  - "main.py"
  - "agent/control/**"
  - "agent/plugins/**"
  - "agent/looping/**"
  - "agent/lifecycle/**"
  - "bootstrap/app.py"
  - "bootstrap/channel_host.py"
  - "infra/channels/**"
  - "skills/develop-akashic-plugin/**"
  - "skills/plugin-system/**"
  - "tests/test_plugin_*.py"
  - "tests/test_channel_host.py"
  - "tests/test_builtin_*plugin*.py"
  - "tests/control/**"
  - "tests/semantic/**"
  - "tests_scenarios/contracts/**"
  - "docker/debug/plugin_hot_reload_probe.py"
  - "docs/INDEX.md"
  - "docs/projectneed.md"
  - "docs/NOW.md"
  - "docs/decisions/**"
  - "docs/design/plugin-install-uninstall-turn-boundary-rollout.md"
  - "docs/design/recursive-plugin-self-validation.md"
  - "docs/design/persistence-state-map.md"
forbidden_paths:
  - "frontend/**"
  - "migrations/**"
  - "正式 Akashic workspace/**"
  - "~/.akashic-plugin/cache/**"
  - "~/.akashic-plugin/manifest.toml"
  - "外部 Fitbit canonical source"
allowed_effects:
  - "在隔离测试目录创建一次性 workspace、plugin home、artifact 和 journal。"
  - "在一次性 endpoint 启停 fixture managed service/Channel。"
forbidden_effects:
  - "修改或重启正式 runtime。"
  - "安装、卸载或改写正式 Fitbit 插件。"
  - "删除正式 plugin-data、SessionDB、memory、附件、凭据或外部源码。"
  - "发送真实 Channel 消息或调用不可逆外部 API。"
validation:
  - "plugin install/uninstall/revert、hot reload、service host、ChannelHost 与 control targeted tests"
  - "CLI/Skill 合同测试，证明 Agent 可见管理面只有三个动作"
  - "parent/child lineage、candidate 冻结、无验证、revert、取消和超时测试"
  - "故障注入：每个 crash point、admission resume、恢复失败、相同 revision 重装和 cleanup residue"
  - "semantic mutants：coexistence 绕过、提前提交、pointer-only rollback、跨 turn revert、错误 child 继承"
  - "python docker/debug/gate.py run --base origin/main"
  - "一次性 workspace 中真实 fixture；Fitbit 外部 Gate 单独固定依赖与 revision"
rollback: "实现前基线 bundle；实现提交可整体 revert。运行事务在 turn commit 前取消 pending，在切换失败时恢复并重新验证 previous。"
worktree_writer: "Codex /root"
handoff_head: "PR branch; exact head recorded at delivery"
external_revisions: []
schema_lineages:
  - "runtime/plugin-reloads.sqlite3 当前 schema；如需迁移先枚举已发布 lineage，未知形状 fail-loud。"
```

### Autonomy and stop rules

- 批准后可自主修改 allowed paths、创建隔离测试状态并运行非破坏性验证。
- 正式 runtime/plugin 操作、Fitbit source 变更、数据库 schema 迁移、外部依赖或扩大路径需再次确认。
- 如果 turn lineage 无法可靠传给 programmatic child，停止并带调用链证据退回设计，不以全局 latest 默认替代。
- 如果独占 Channel 无法证明 ownership 释放/取得，停止局部发布，不自动重启整个 runtime。
- 如果实现仍要求 Agent 调用 promote/discard/status/restart，视为合同失败。

## 11. 最小验收矩阵

| 场景 | 必须观察到的结果 |
|---|---|
| install 后父 turn | 父继续使用 S0；返回明确要求 programmatic 验证 |
| attached programmatic child | 自动绑定 S1，真实 tool/Skill/trace 来自同一 candidate identity |
| programmatic 失败后 revert | S1不发布，S0始终不变；修复后可 install S2继续递归 |
| 没有 programmatic 就结束 | pending install自动取消，下一 turn仍是 S0 |
| 验证通过且正常结束 | turn 后排空 S0并提交 S1；下一 turn自动使用 S1 |
| pending uninstall 后 revert | manifest、代码、能力和 plugin-data均保持原样 |
| uninstall 正常提交 | turn 后停止 endpoint、移除能力和代码；plugin-data保留 |
| 跨 turn revert | 明确失败，不改变当前 stable；需要修复后 install或明确 uninstall |
| Fitbit 类独占 service | programmatic使用隔离 endpoint；正式端口只在 turn 后切换 |
| service + Channel 失败 | 恢复并验证整套 old generation，不出现混合代际 |
| install 旧代码清理失败 | 新代保持 committed；下一 turn报告残留和清理错误 |
| uninstall 代码删除失败 | 插件保持未发布；报告残留；再次 uninstall重试清理 |
| 任一 crash point | host 绑定后从 journal继续提交或恢复，admission最终恢复或明确 degraded |

## 12. 非目标

- 不判断 Fitbit 数据或算法的领域正确性。
- 不把 Shell 变成恶意代码安全沙箱。
- 不实现跨 turn版本回滚、历史版本选择、多候选队列或 redo。
- 不新增 Agent 可见 rollout/Channel 状态机或 Fitbit 专用 Core 检查。
- 不在本任务修改 Fitbit source、正式 cache、manifest、workspace 或运行进程。
- 不用整 runtime 重启代替可证明的局部事务；确需整 runtime 维护时另行批准。

## 13. 实施批准门

维护者已在 2026-08-08 明确批准按本合同实施、拆分提交并创建大 PR。该批准不包含正式 Fitbit 安装、正式 runtime 重启或外部插件 source 修改。
