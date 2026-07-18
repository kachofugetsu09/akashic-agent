# Akashic Agent 开发工作流

`WORKFLOW.md` 是一份从接手任务到提交评审的执行手册。长期产品语义由 [`projectneed.md`](projectneed.md) 负责，当前未完成事项由 [`NOW.md`](NOW.md) 负责。

## 1. 固定顺序

```text
┌──────────────┐
│ 1. Read      │  INDEX → 需求/NOW/决策 → 真实代码和数据
└──────┬───────┘
       ▼
┌──────────────┐
│ 2. Ownership │  能力 owner、消费者范围、是否需要 runtime patch
└──────┬───────┘
       ▼
┌──────────────┐
│ 3. Isolate   │  目标分支、独立 worktree、基线、备份
└──────┬───────┘
       ▼
┌──────────────┐
│ 4. Contract  │  目标、semantic delta、受保护状态、副作用
└──────┬───────┘
       ▼
┌──────────────┐
│ 5. Implement │  按真实证据做最小改动
└──────┬───────┘
       ▼
┌──────────────┐
│ 6. Verify    │  targeted tests → static/build → Gate
└──────┬───────┘
       ▼
┌──────────────┐
│ 7. Review    │  相邻 diff、累计行为、owner、write set、证据
└──────┬───────┘
       ▼
┌──────────────┐
│ 8. Reconcile │  目标分支、完整 diff、文档、报告
└──────┬───────┘
       ▼
┌──────────────┐
│ 9. Deliver   │  commit/PR、验证状态、阻塞、回滚点
└──────────────┘
```

任务只有在当前阶段退出条件满足后才进入下一阶段。

## 2. 阶段出口

| 阶段 | 必须完成的动作 | 退出证据 |
|---|---|---|
| Read | 每个新会话先读 [`INDEX.md`](INDEX.md)，再按索引读相关需求、NOW、决策、设计和真实实现 | 已确认事实、未知项和文档冲突已经列出 |
| Ownership | 对跨仓库、客户端、插件和协议任务声明 `capability_owner`、`consumer_scope`、`runtime_patch`、`runtime_patch_reason`、`authoritative_state_owner` 与 `client_only_alternative` | 核心改动能引用已批准语义；“未来可能复用”没有被当作 owner 证据 |
| Isolate | 核对目标分支、base commit、worktree、唯一 writer、用户未提交改动和恢复点 | 改动不会写进用户当前 checkout、其他 agent 的 worktree 或正式 Akashic workspace |
| Contract | 声明目标、成功标准、`change_type`、`semantic_delta`、受保护状态、允许副作用、验证和回滚 | 高风险歧义已获确认，或任务停止等待确认 |
| Implement | 只改合同允许的路径和行为；持久化语义从数据库、文件、事件或外部边界观察 | Diff 没有新增未声明副作用 |
| Verify | 运行相关测试、类型或前端检查，再运行 change-impact Gate | 测试与报告来自当前源码；未运行项有明确状态 |
| Review | 按基线审查完整 diff；stacked PR 逐层检查相邻 `base..head`，再在最终 head 检查累计行为 | Findings 带严重度、文件位置、触发路径和证据；需要维护者决定的语义已停止确认 |
| Reconcile | 获取目标分支最新状态，核对完整 diff、工作手册变化和报告摘要 | 目标分支的新变化没有使任务合同失效 |
| Deliver | 使用 PR 模板写明改动、证据、private 状态、阻塞和回滚方式 | 另一位维护者可以独立评审并继续处理 |

历史会话、自动记忆和 `_handbook/` 只提供调查线索。当前工作手册和真实实现负责确认事实。

## 3. Worktree 与测试数据

已有专用 worktree 的任务先核对分支、基线和未提交改动。没有专用 worktree 的非简单任务从最新目标分支迁出：

```bash
git fetch origin main
git worktree add -b feature/<task> ../akasic-agent-worktrees/<task> origin/main
```

每个 worktree 同一时刻只有一个 writer。并行 subagent 默认只读审查各自的 commit/diff；需要修复时，为每个 writer 分配独立 worktree 和分支，并按 Review 合同记录 `repository + worktree + branch + owner + base_head + allowed_paths + status`。产生修改的 writer 必须先提交允许范围内的修改，再记录 `handoff_head + dirty_state + next owner`；没有产生修改时只能交接已经核对的 clean HEAD。不得用 reset、checkout 或清理未跟踪文件制造 clean 状态。旧 writer 完成或被明确中断前不得转移 owner，交接后的旧后台任务不得继续写入或提交。

Git worktree 保存源码、测试和项目文档。Akashic `<workspace>` 保存会话、记忆、附件、调度和 plugin-data。测试使用一次性 workspace、plugin home、config 和 HOME。修改持久化文件前创建名称清楚的备份。

## 4. 最小任务合同

普通局部修改用一段开工说明记录。复杂或高风险任务使用 [`templates/agent-task-contract.md`](templates/agent-task-contract.md)；独立评审或 stacked PR 使用 [`templates/review-contract.md`](templates/review-contract.md)。开工说明至少包含：

- 用户可见目标和可判断的成功标准。
- `change_type` 与 `semantic_delta`。
- 允许变化、受保护状态和关联不变量。
- 文件、数据库、进程、网络和消息副作用。
- 验证命令、停止条件和回滚点。
- 跨仓库或客户端任务的能力 owner、消费者范围、runtime patch 理由和客户端替代方案。

涉及裁切、压缩、清理、迁移、同步、重建、替换或恢复的任务须写清数据怎样增加、更新、逻辑失效和物理删除。不同解释会改变数据或外部行为的任务等待维护者确认。

[`templates/change-intent.yaml`](templates/change-intent.yaml) 当前只提供填写字段。自动 checker 完成前，PR 模板保存这些字段，不提交临时 YAML。

## 5. Gate

完成相关测试和静态检查后运行：

```bash
python docker/debug/gate.py run --base origin/main
```

Gate 根据 Git diff 选择场景，并把报告写入 `docker/debug/reports/change-gate/<run-id>/`。报告的 `sourceDigest`、`planDigest` 和当前源码必须匹配。源码在计划生成后发生变化会使原计划失效，此时重新运行 Gate。

| 公开计划结果 | PR 状态 |
|---|---|
| `privateGateRequired=false` | `private-contract-gate: not_required` |
| `privateGateRequired=true`，贡献者没有私有插件 | `private-contract-gate: pending_maintainer` |
| 维护者已用同一份 `plan.json` 取得私有通过报告 | `private-contract-gate: passed` |

公开结果不能替代待运行的私有检查。测试失败先归因为实现、环境或契约冲突；修改断言、跳过场景和缩减 Gate 需要独立理由与授权。

## 6. Review 模式

纯评审任务走 `Read → Ownership → Review → Deliver`，默认只读，不创建实现分支、不修改候选代码，也不把发现自动写入 GitHub。用户要求修复、发表评论或更新工作手册时，重新建立相应写入合同。

Stacked PR 先确认依赖链，每张只审查自己的相邻 `base..head`；最后对栈顶 head 做累计协议、持久化、构建和用户行为审查。数据库评审不能只比较 `user_version`，还要列出所有已知 schema lineage、每条迁移路径和最终 schema identity；未知的同版本异构 schema 必须 fail-loud。

跨仓库协议和插件报告分别固定协议 source commit/path/hash、实际 runtime commit/tree、provider `requested_ref/resolved_sha/change_source_pr_head` 和 scenario profile/hash。手工构建、Docker、隔离互操作和 Pixel/ADB 记录与远端 CI checks 分开报告，没有 check 就写未验证，不能用 PR 描述中的成功声明代替。

Pixel/ADB Gate 只从干净 source commit/tree 构建，同一 Android worktree 同时只允许一个 Gate。构建完成后、首次 ADB 调用前必须再次核对 worktree clean、HEAD 和 tree 与起始值相同；任一漂移以 `failed_setup` 结束且不得读取或写入设备。安装前从 app/test APK 读取真实 application ID 与 instrumentation target，为本次 run 使用唯一的 run-specific application ID，再用 `pm list packages -u` 检查 app/test package collision。任一 collision 都必须 blocked，禁止安装、clear 或 uninstall；安装禁止 replace，签名一致和 `adb install -r` 不构成覆盖许可。只有本进程成功安装的 package 才归本进程清理。

测试按声明的阶段执行，需要验证进程恢复时在阶段间显式 force-stop；instrumentation 还要核对实际执行数量、指定方法、开始/成功状态和失败标记，不能把 shell 退出码 0 或 0 test 当作通过。测试成功不等于 Gate 成功：只有 cleanup 完成后才能写唯一 `gate_result=passed`；清理失败必须返回非零、写 `gate_result=failed_cleanup` 并列出残留 package。结束后核对正式 package 状态未变，并证明临时 app/test package、ADB reverse、容器和测试 workspace 已 cleanup。涉及实时 Gateway 时，另外记录 Mobile Lab 的 core SHA、run ID 和配对材料来源；设备 package 隔离不能替代服务端 workspace 隔离证明。

评审移动端或其他客户端时按 MOB-001 区分平台实现、产品体验、中立协议和核心权威状态。核心 runtime patch 缺少既有不变量、权威 owner 或客户端替代分析时停止并询问维护者。

## 7. 文档维护

| 发生的情况 | 动作 |
|---|---|
| 维护者确认长期产品语义 | 更新 `projectneed.md` |
| 形成长期影响 owner、接口、数据或迁移的选择 | 新建或勘误 `decisions/` 记录 |
| 单个问题需要调用路径、方案、迁移和验收说明 | 更新相关 `design/` 文档并标明状态 |
| 当前任务结束后仍有已接受、可接手的未完成工作 | 写入 `NOW.md` |
| `NOW.md` 事项已经实现并通过约定验证 | 在同一交付中删除该事项 |
| 只完成当前任务的中间步骤 | 留在任务状态或 handoff，不写 `NOW.md` |
| 工作手册文件新增、移动或删除 | 更新 `INDEX.md` 和所有入站链接 |

`NOW.md` 不记录当前会话的逐步计划，也不保存已经完成的工作。Git、commit、PR 和决策记录负责历史。

## 8. 完成定义

- 用户可见目标和成功标准已经满足。
- 实际 diff 没有超出声明范围。
- 受保护状态和禁止副作用经过独立核对。
- 跨仓库或客户端变化已经完成能力 owner 与 runtime patch 归属检查。
- 相关测试、静态检查和 Gate 已通过；未运行项有明确状态。
- private Gate 准确写成 `not_required`、`pending_maintainer` 或 `passed`。
- 文档、代码和当前接手点一致，完成事项已从 `NOW.md` 删除。
- [PR 模板](../.github/pull_request_template.md) 已写明报告摘要、真实阻塞和回滚方式。
