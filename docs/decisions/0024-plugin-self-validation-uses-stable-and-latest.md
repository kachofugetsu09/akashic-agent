# 0024 · 插件自验证使用 stable/latest 与 session 级并发

- 状态：superseded by [0026](0026-plugin-rollout-is-owned-by-the-parent-turn.md)；内部 stable/latest 与 session 并发机制保留，Agent 显式选择/promote/discard 流程废止
- 日期：2026-08-05
- refines：[0008](0008-plugin-runtime-publishes-only-committed-snapshots.md)、[0014](0014-shell-uses-unified-execution.md)、[0015](0015-cleanup-does-not-own-turn-or-restart-finality.md)
- 关联条款：RUN-007、OUT-004、PLG-013、CTRL-003、SH-001、TST-001～TST-006

## 背景

插件开发 turn 绑定开始时的 runtime snapshot。当前候选在 commit 前不可租用，control runtime 与 AgentLoop 又分别持有全局整轮锁。父 turn 即使先完成 `plugin-install`，它启动的 programmatic 子 turn 也只能等父 turn 结束；新插件在下一轮生效后，原父 turn 已不能观察、判断和继续修复。

单纯删除全局锁会让 scheduler、不同 session、共享文件、Prompt 状态和外部投递失去 owner；让候选直接成为 current 又会把未验证能力短暂暴露给普通请求。

## 决定

1. 同一 session 的 turn 串行，不同 session 可以并发。全局 active turn、字节和 runtime object 边界使用有界 admission，不使用跨 session 的整轮互斥。
2. runtime 对外提供两个选择：
   - `stable` 是普通 turn 默认使用、已经通过行为验证的 snapshot。
   - `latest` 是最新完成静态与 readiness Gate 的候选，只能由显式 `runtime=latest` 的验证 session 租用。
3. 同时只允许一个未决 latest；没有候选时 `latest is stable`。第二次 install 在候选未 promote/discard 前失败。
4. `plugin-install` 的成功终态必须表示 latest 已可租用；调用方先等待 install，再启动 programmatic 验证，不传额外 revision/epoch。
5. programmatic 验证创建独立 session，默认不沉淀新语义记忆但允许检索；SessionDB thread、messages、tool items 和 terminal 正常持久化。显式参数才允许长期记忆写入。
6. Shell 作为异步可观察的 control CLI 载体。验证调用默认 attached；CLI 进程或连接结束会取消服务端子 turn。显式 detached 不用于插件自验证。
7. 验证通过后原子执行 `stable=latest`；失败时原子执行 `latest=stable`。旧 snapshot 按既有 lease drain 语义回收。
8. `message_push` 只取得短 ChatLane send owner，不等待目标 session lane。推送不注入父 Prompt或目标 session history；真实 delivery result 保存在调用者的工具 trace。
9. latest 默认只验证只读行为。共享状态写入、不可撤销外部效果和独占 endpoint 必须使用插件事务/dry-run、隔离环境或显式副作用授权；pointer rollback 不冒充效果回滚。
10. Tool、Skill、Prompt 与文件读取来自 turn 绑定的不可变 snapshot/TurnFrame。并发实现前，现有共享可变字段必须归入 session、turn、runtime service 或具体 repository owner。

## 理由

- 两个 pointer 满足父 turn 同时保留旧行为和显式执行新行为，不引入每 session candidate namespace、reservation token 或多版本选择协议。
- session lane 与 Actor/Codex 的状态归属一致：历史相关行为串行，无关 session 保持吞吐。
- programmatic child 使用真实 Gateway 和真实候选 catalog，比 import/doctor 或模型自述更接近生产行为。
- 默认 memory read-only 既保留自验证所需的背景检索，又避免测试对话进入长期人格与语义记忆。
- attached Shell 把父取消传播到子 turn，避免父查询结束后留下孤儿验证任务。
- 0008 对普通读者仍成立：未行为验证候选不进入 stable/default publication。0024 只增加一个显式验证 reader。

## 被拒绝方案

### 保留全局锁，等下一轮验证

不能形成同一修改者接收反馈后继续改写的递归闭环。

### install 后立即把候选设为 current

普通 session 会短暂获得未验证工具；失败回滚期间还可能产生不可撤销外部效果。

### 为每次调用传 revision/epoch/token

单一未决 latest 已提供足够身份；额外 token 增加协议、恢复和泄漏面，首版没有收益。

### 为每个候选启动完整隔离 runtime

适合独占 endpoint 和写型副作用，但作为普通 tool/skill 插件的默认路径成本过高，也没有利用现有 immutable snapshot/lease runtime。

### 让子 turn 内容直接注入父 turn

会破坏父 Prompt 冻结、消息顺序和 SessionDB 事实边界。父 turn只读取结构化 terminal/trace。

## 影响

- control、channel 与 direct admission 需要统一到 session lane。
- snapshot store、安装 cache 与 reload journal 需要持久 stable/latest descriptor 和不可变 artifact identity。
- `main.py exec`、control thread metadata 和 disconnect owner 需要新增严格参数与 attached 取消。
- Skill/tool catalog 必须从当前绑定 snapshot 解析。
- `message_push` 要从直接 bypass ChatLane 改为 passive-send 提交。
- programmatic 新 session 的默认记忆语义发生有意变化；旧显式 thread 的已存 metadata 不自动改写。
- 当前 `develop-akashic-plugin` Skill 在实现完成前必须报告 runtime self-validation unavailable，不得假报完整通过。

## 验收

完整 oracle 见[插件递归自验证运行时设计](../design/recursive-plugin-self-validation.md#14-独立验收)。最低通过条件是：父 turn 持有 stable lease 时，独立 programmatic session 能在严格超时内租用 latest、实际调用候选工具、返回结构化终态，并在 promote/discard 后保持普通 session 的 pointer 与持久 write set 正确。
