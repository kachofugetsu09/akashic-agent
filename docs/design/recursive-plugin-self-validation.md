# 插件递归自验证运行时设计

- 状态：historical mechanism；Agent 显式 latest/promote/discard 流程已由 2026-08-08 turn-boundary 合同取代
- 确认日期：2026-08-05
- 决策：[0024](../decisions/0024-plugin-self-validation-uses-stable-and-latest.md)、[0026](../decisions/0026-plugin-rollout-is-owned-by-the-parent-turn.md)
- 关联条款：RUN-007、OUT-004、PLG-013、CTRL-003、SH-001、TST-001～TST-006
- 当前实现基线：`feat/programmatic-latest-validation`

## 1. 结论

> 2026-08-08 勘误：本文保留 stable/latest、session lane、attached cancellation 和真实 child trace 的机制说明，但不再定义 Agent 操作。当前合同见 [插件 install/uninstall/revert 与 turn 边界发布](plugin-install-uninstall-turn-boundary-rollout.md)：父 turn 只执行 install/uninstall/revert，attached child 自动继承候选，正常 terminal 后由 Core 提交。本文后续出现的显式 `--runtime latest`、promote/discard 仅是历史设计，不得用于当前 Agent 流程。

原实现不能在写完插件的同一 turn 内证明插件可用，不是因为缺少一条等待命令，而是因为候选能力、执行机会和验收反馈没有形成闭环：当前 turn 绑定旧 snapshot，程序化子 turn 又被两层全局执行锁挡在父 turn 后面。实现已移除跨 session 整轮互斥，并接通 stable/latest、runtime selector、attached cancellation 和候选管理接口；父 turn 现在能在自己结束前取得隔离 latest 的真实行为结果，再决定 promote 或 discard。

目标只引入两个公开运行时选择和一个长时序列化维度：

```text
runtime selector
├── stable  普通 turn 的默认快照，已经通过行为验证
└── latest  最新已准备候选，只接受显式程序化验证

execution lane
└── session_key
    ├── 同一 session 的 turn 串行
    └── 不同 session 的 turn 可并发
```

父 turn `T` 安装插件后，通过统一 Shell 启动一个新 programmatic session `V`，显式选择 `latest`。`V` 在同一 Gateway 中真实加载候选工具与 Skill，默认可检索既有记忆但不沉淀新语义记忆，并把消息、工具调用和终态写入自己的 SessionDB 记录。`T` 通过 Shell 的增量 JSON 输出观察 `V`；通过则晋升 `latest → stable`，失败则丢弃 `latest`、修复源码并再次迭代。

这条路径不重启 Agent，不让普通会话短暂看到候选，也不把 Shell 子进程误当成第二个 runtime。

## 2. 问题抽象：为什么现在永远差一轮

一次可递归自改进至少需要下面的闭环：

```text
┌─────────────┐   写候选   ┌─────────────┐
│ Modifier T  ├───────────►│ Candidate C │
└──────┬──────┘            └──────┬──────┘
       │                           │ 在真实环境执行
       │                           ▼
       │                    ┌─────────────┐
       │      反馈           │ Evaluator V │
       └────────────────────┤ + Oracle    │
                            └──────┬──────┘
                                   │
                         pass ─────┴───── fail
                           │                │
                        promote         revise
                           │                │
                           └────── recurse ─┘
```

Phase 1 之前的 Akashic 已经有 `Modifier`、候选准备、SessionDB、Shell 和 control turn，但三处边界把环切断：

1. `ConversationRuntime._admission` 在所有 control thread 之间持有全局整轮锁。
2. `AgentLoop._passive_runtime_lock` 又在 channel、scheduler 和 programmatic 路径之间持有一层全局整轮锁。
3. `RuntimeSnapshotStore.begin_publish()` 把候选标为 `validating` 且 `accepting_leases=False`；只有 commit 后才可被 turn 租用。

因此实际时序是：

```text
父 Turn T：绑定 stable=S0
  ├── 修改并 install 插件 C1
  ├── watcher 最多只能异步构造/发布新 snapshot
  └── 启动 programmatic V
        └── V 等待全局锁，而全局锁仍由 T 持有  ← 自死锁/等待到 T 结束

T 结束
  └── 新 turn 才可能看到 C1，但 T 已不能依据结果继续修复
```

把 watcher 轮询得更快、让 `plugin-install` 多等几秒或在当前 turn 重查 `tool_search` 都不改变这个可达性问题。当前 turn 的工具目录来自冻结的 S0；安装完成只证明磁盘与 manifest 已更新，也不证明候选已经成为一个可执行 runtime snapshot。

## 3. 理论依据与不可缺失特征

本设计不把“模型自评一句可用”称作自进化。经典与近期工作共同要求修改、执行、反馈、选择和继续迭代形成真实闭环：

| 工作 | 对本设计的启示 |
|---|---|
| [Gödel Machines](https://arxiv.org/abs/cs/0309048) | 自修改必须相对于明确效用函数证明有益；工程系统无法证明一般插件改动时，至少要保留独立、可执行的验收器与保守提交规则。 |
| [Self-Refine](https://papers.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html) | 生成、反馈、改写可以由同一个模型循环完成，但反馈必须重新进入下一次改写，而不是停在日志里。 |
| [Reflexion](https://proceedings.nips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf) | 环境反馈和失败原因需要成为下一轮可访问的经验；这对应子 session 的终态、工具 trace 与可检索历史。 |
| [Voyager](https://voyager.minedojo.org/assets/documents/voyager.pdf) | 可执行 Skill 库、环境错误和 self-verification 必须结合；只生成代码而不在目标环境执行，不会形成能力积累。 |
| [Automated Design of Agentic Systems](https://proceedings.iclr.cc/paper_files/paper/2025/hash/36b7acf6f6010652b3f2a433774a66fe-Abstract-Conference.html) | Agent 可以用代码表示并由 meta-agent 搜索，但必须保存候选与评估结果，不能让一次失败覆盖已工作的系统。 |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | 形式证明不可行时，可用真实 benchmark 经验验证自修改；保留稳定候选和失败分支比只追一条最新链更可靠。Akashic 首版只保留一个未决候选，不引入完整 archive。 |

并发模型采用 Actor 的最小结论：状态由 owner 串行处理，独立 owner 可以并发。1973 年的 [Actor formalism](https://www.ijcai.org/Proceedings/1973) 提供理论来源；Codex 当前也把“一次最多一个活动任务”放在每个 Session 内，而不是跨全部 Session。Claude Code 的子 Agent 则拥有独立 messages、read-file state、AbortController、权限与 sidechain transcript。两者共同说明：共享 runtime 不等于共享 turn-local 可变状态。

快照生命周期接近 [RCU](https://www.kernel.org/doc/html/v6.2/RCU/whatisRCU.html)：读者绑定不可变版本，更新者发布新指针，旧版本只在既有 reader 离开后回收。

据此，递归插件系统缺一不可的特征是：

1. **可修改表示**：插件 canonical source 和不可变安装 artifact。
2. **候选隔离**：未通过行为验证的能力不进入普通请求。
3. **真实执行**：候选必须能在目标 Gateway、目标工具路由和目标 Prompt 中运行。
4. **独立 oracle**：加载成功、工具出现和模型自述都不能代替行为断言。
5. **提交与回滚**：验证成功才改变默认指针；失败保持旧 stable。
6. **反馈可回读**：父 turn 能取得终态、工具 trace、错误与 session 身份。
7. **递归控制与停止条件**：反馈能触发下一次修复，同时受尝试预算、取消和真实阻塞约束。

当前系统具备 1、部分 2、部分 4 和快照回滚，但缺少 3 的可达路径，也就没有 6、7 所需的同 turn 反馈闭环。

## 4. 目标调用链

```text
用户 Turn T（session=A，绑定 stable=S0）
  │
  ├─ 修改插件 canonical source
  ├─ 测试、commit；需要远端 source 时再 push
  ├─ plugin-install
  │    ├─ 构造不可变 artifact G1
  │    ├─ 准备 snapshot S1
  │    ├─ static/readiness Gate 通过
  │    └─ 原子设置 latest=S1；stable 仍为 S0
  │
  ├─ shell: python main.py exec --new --runtime latest --json "验证提示"
  │    ├─ Shell 短等待后返回 execution_id
  │    ├─ CLI 连接当前 Gateway，不启动第二个 runtime
  │    └─ programmatic Turn V（session=B）
  │         ├─ 获取 session B lane
  │         ├─ 租用 latest=S1
  │         ├─ 读取 S1 的 tools / skills / prompt modules
  │         ├─ 默认 recall allowed + semantic writes disabled
  │         ├─ 消息与工具链追加到 session B
  │         └─ 产生 terminal TurnResult
  │
  ├─ write_stdin(execution_id) 读取 JSONL 直至 terminal
  ├─ 根据独立 oracle 判断
  │    ├─ pass → plugin-promote；stable=S1，latest=S1
  │    └─ fail → plugin-discard；latest=stable=S0；修复后递归
  │
  └─ 读取最终 pointer/journal 后向用户报告
```

这里不存在“如何保证 latest 至少包含 T 刚完成的 install”的额外分布式协议：调用者先 `await plugin-install`，再启动 programmatic turn，因果顺序已经确定。需要加强的是 `plugin-install` 的完成定义——它必须等到 S1 已经可通过 `latest` 租用，而不能只表示 cache/manifest 写入完成、等待 watcher 将来发现。

## 5. 两个 pointer，而不是版本参数网络

### 5.1 语义

| 名称 | 默认可见性 | 可租用者 | 含义 |
|---|---|---|---|
| `stable` | 所有普通 turn | 普通 channel、scheduler、programmatic | 最近一次行为验证通过的完整 runtime snapshot |
| `latest` | 仅显式选择 | `runtime=latest` 的验证 session | 最新一次完成 static/readiness Gate、尚待行为判断的完整 snapshot |

不传选择时永远使用 `stable`。没有未决候选时 `latest is stable`。首版全局只允许一个未决 `latest`；它存在时再次 install 必须 fail-loud，不能覆盖正在验证的候选。

`latest` 不是“磁盘上最后修改的目录”，而是可租用、不可变、具有完整 tools/skills/MCP/prompt catalog 的 snapshot。调用方不传 revision、epoch 或候选 token；单一未决候选和因果顺序已经给出足够简单的身份模型。

### 5.2 持久与内存状态

内存 snapshot 不能单独承担 crash recovery。目标状态分两层：

```text
~/.akashic-plugin/cache/
└── immutable artifacts keyed by source revision/tree digest

<workspace>/runtime/plugin-reloads.sqlite3
├── stable snapshot descriptor
├── latest snapshot descriptor
├── candidate phase + install provenance
└── append-only phase journal
```

同一 plugin version 的不同 commit 也必须生成不同 artifact；未决候选存在时不得覆盖 stable 仍引用的代码目录。Gateway 重启后先重建 stable，再重建未决 latest；普通 admission 始终从 stable 开始。若 latest 重建失败，记录 candidate failure 并保持 stable，不把损坏候选提升为默认。

### 5.3 状态机

```text
none
  │ install + prepare gates pass
  ▼
latest_ready
  ├─ behavior pass ──► promoting ──► stable=latest ──► none
  ├─ behavior fail ──► discarding ─► latest=stable ──► none
  └─ process crash ──► recover latest_ready or fail candidate; stable unchanged
```

promotion 只切换 pointer，不重新构造已经验证的 snapshot。旧 stable 立即停止接受新 lease，但仍服务已绑定 turn；lease 归零后逆序清理。discard 同理等待 latest 的验证 lease 释放后清理。

## 6. 并发模型：只保留一个长时锁

### 6.1 Session lane

唯一覆盖整轮的序列化 owner 是 `session_key`：

```text
SessionLaneRegistry
├── session A：T0 → T1 → T2
├── session B：V0 → V1
└── session C：S0 → S1

A、B、C 可并发；每一行内部严格串行。
```

control 的 `_active_by_thread`、channel inbound 和内部 `process_direct` 必须汇入同一个 session-lane owner，不能各自再套一把整轮全局锁。全局资源边界保留 semaphore/counter，例如 active turn 数、请求字节和 runtime object 数；容量控制不是互斥执行。

因此删除或降级为短临界区的是：

- `ConversationRuntime._admission` 的全局整轮互斥。
- `AgentLoop._passive_runtime_lock` 的全局整轮互斥。

保留的是：

- 同 session 一次只有一个 active turn。
- snapshot pointer 切换的短事务锁。
- SessionDB、文件和外部 endpoint 各自 owner 的短提交锁。
- ChatLane 的出站提交顺序。

### 6.2 为什么不会让定时任务插进用户回复

计算与投递是两个维度。scheduler 可以在自己的 session 并发推理，但它向同一 chat 的非被动投递仍经过 ChatLane：只要用户被动 turn 或其最终 send 尚未结束，scheduler 的主动消息就等待。因此用户不会看到定时任务正文插进正在生成的被动回复。

这不要求把所有 session 的模型推理重新串行化。

## 7. `message_push` 的特殊语义

`message_push` 是外部投递工具，不是向目标 session 注入一个 inbound turn。它不能等待目标 session 的执行 lane，否则下面的合法调用会自死锁：父 T 等子 V，子 V 又等父 T 释放目标 session。

目标规则是：

1. `message_push` 不获取目标 session lane。
2. 它使用受信任的 origin turn role 进入 ChatLane，只串行实际 adapter send；不能绕过正在进行的另一次 send。
3. 普通 scheduler/proactive 投递仍是 non-passive，等待同 chat 的 passive turn 优先完成。
4. 正在执行的 passive turn（包括 programmatic 验证 turn）发起的 `message_push` 使用 passive-send 路径，可以在父 turn 尚未结束时提交，但实际 send 之间仍串行。
5. 推送正文不注入父 T 的冻结 Prompt，也不追加到目标 session 的 `messages`；父 T 不会“突然知道自己已经说过这句话”。
6. 调用参数、结构化 delivery receipt 和工具终态保存在调用者 V 的工具链；需要复核时读取 V 的 control result 或 SessionDB。

```text
父 Turn T（session=A，仍在运行）
  └─ 等待 Shell / 子 Turn V（session=B）
       └─ message_push(target chat=A)
            ├─ 不等待 session A lane
            ├─ 获取短 ChatLane send owner
            ├─ adapter 返回 DeliveryReceipt
            └─ result 写入 session B tool trace

父 T 的 Prompt：不可见 push 正文
外部用户：看见真实投递
父 T：通过 V 的 result 判断成功/失败
```

pointer 回滚不能撤销已经发送的消息。验证提示若会触发外部发送，必须在任务合同中获得授权，并使用明确测试目标；没有授权时只允许验证到发送前的确定性边界。

## 8. Turn-local 状态、工具可见性和共享文件

跨 session 并发之前必须先完成状态归属审计。不是给每个字段加锁，而是把状态放回唯一 owner：

| 状态 | owner | 并发规则 |
|---|---|---|
| runtime tools、skills、prompt modules | 不可变 `RuntimeSnapshot` | turn admission 时租用；整轮不变 |
| messages、read-file state、tool trace、disabled tools | `TurnFrame` / task-local context | 不得写入 `AgentLoop` 模块级可变字段 |
| session history、metadata、active turn | session lane + `SessionStore` | 同 session 串行；事务短且不得跨 `await` |
| current turn/session/snapshot identity | `ContextVar` 或显式参数 | 子 task 只继承明确声明的值；新 programmatic session 获取自己的值 |
| plugin-data、配置和普通文件 | 具体 repository/file owner | 同一 canonical target 串行、原子替换；不同 target 可并行 |
| provider client、HTTP pool、模型配置 | runtime service owner | client 可共享；一次 request 的消息、用量和取消信号 task-local |
| Shell execution registry | `ShellProcessManager` | 按 owner session 与 execution_id 隔离 |

Tool/Skill 可见性严格来自绑定 snapshot：T 继续看到 S0；V 看到 S1。Skill 正文或其他文件一旦读入 Prompt，就成为 V 本轮的值副本；磁盘后续变化不会反向修改已发给模型的上下文。

当前 `AgentLoop`、reasoner、event subscription 和工具执行链中的可变字段必须逐项证明属于上述某个 owner。无法归属、靠“通常只有一轮”成立的字段，是并发实现的阻塞项；不能用更多局部锁掩盖。

### 8.1 Owner 审计结果

审计覆盖生产 `PassiveMessageWorker → AgentLoop → DefaultReasoner → ToolRegistry` 路径，不把 legacy 串行入口误当成生产并发模型。结果如下：

| 对象 | 实际 owner | 审计结论 |
|---|---|---|
| `_active_tasks`、`_active_turn_states` | `AgentLoop`，key 为 `session_key` | 修改只发生在同一 event loop；每个 key 受 session lane 约束，不存在跨 session 共用 turn frame。中断和续接由 `ConversationRuntime` 独占，不在 Loop 复制状态。 |
| Prompt messages、tool trace、compactor 与 provider request | 当前 `TurnFrame` / reasoner 调用栈 | 每次请求局部创建；共享 provider 只持连接池与配置，请求消息、usage 和取消信号不回写 service 字段。 |
| reasoner phase snapshot cache | reasoner service cache | 命中只返回完整不可变 phase；并发替换最多造成重复构建，不会把 A turn 的局部状态返回给 B turn。 |
| `ToolRegistry` current execution/search scope | `ContextVar` | 每个 task 独立；snapshot 内 registry 只提供冻结 catalog，candidate write 工具按 lease policy 禁用。 |
| `ToolDiscoveryState` | runtime service 内按 session key 的同步 LRU | 读写之间没有 `await`，一次更新原子完成；不同 session 只共享容量，不共享选中工具值。 |
| Session cache、history 与 SQLite | session lane + `SessionManager` / store lock | 同 session 整轮串行，跨 session 的持久提交由 store 短锁拥有；messages 正常路径仍只追加。 |
| plugin module/tool 实例与 plugin-data | generation + 插件资源 owner | generation 可被多个 session 租用，因此插件实例必须可重入；一次调用状态放 frame/局部变量，共享文件由插件 repository 或原子替换拥有。 |
| Shell execution registry | `ShellProcessManager` | `execution_id` 与 owner session 共同隔离；wait/write 不持有 session lane 或 snapshot pointer 锁。 |
| semantic memory writer | memory engine repository owner | validation session 保留 recall，但 `skip_post_memory` 与 candidate write-tool policy 让 semantic write set 保持为空。 |

审计发现并修复一个真实可达的串值：`ContextBuilder` 曾把 `last_debug_breakdown` 和 `last_assembled_contexts` 存在共享实例字段。A turn 完成 render 后等待 provider，B turn 可以覆盖字段，导致 A 的 after-turn budget/debug 读取 B 的投影。现在两项诊断投影由 `ContextVar` 按 task 保存，并用两个并发 render 在交错后回读各自 marker 的测试锁定。该问题不会改写权威消息，但会污染诊断与预算证据，不能靠“只是 debug”忽略。

### 8.2 写型插件的边界

默认 latest 验证 session：

- 可读取 SessionDB、长期记忆和 plugin-data。
- 不沉淀新的语义记忆。
- 默认禁用记忆写工具，以及 candidate generation 中所有非 read-only 的 Tool/MCP。
- 仍把验证 session 的消息和工具 trace 追加到 SessionDB。

要验证写型插件，必须满足下列至少一项：

1. 插件提供真实的事务/dry-run 边界，且 dry-run 与正式路径共享同一领域校验；或
2. 使用隔离 workspace/plugin-data 与受控外部 test endpoint；或
3. 用户明确授权真实副作用，并有幂等键、before/after oracle 和不可撤销效果说明。

stable/latest 只回滚代码和 runtime pointer，不回滚任意文件、数据库、消息和外部 API。没有上述边界时，在线自验证只能验证读路径和纯逻辑，必须明确报告写路径未验证。

## 9. Programmatic + Shell 合同

目标 CLI：

```bash
python main.py exec --new --runtime latest --json "验证新插件的目标行为"
```

语义：

- `--runtime` 只接受 `stable|latest`；默认 `stable`。
- `--new` 创建独立 programmatic session；Control 默认让它的 Turn 声明 `effects.post_commit=suppress`。
- 默认仍允许 `recall_memory`、`search_messages` 和 `fetch_messages`。
- 显式 `--persist-memory` 才在 thread metadata 写入 `effects.post_commit=allow`，允许该 session 沉淀长期语义记忆；该参数只能在创建 thread 时使用。
- SessionDB 中的 thread、turn、user/assistant message 和 tool items 始终正常持久化，便于审计和回读。
- `--json` 输出当前 control event 流；terminal 至少包含 `threadId`、`turn id`、`status`、`finalResponse`、`items`、`usage` 和 `error`。
- Control 两端共享显式 2 MiB 单帧上限；超过 asyncio 默认 64 KiB 的合法 terminal 必须完整送达，超过协议上限则 fail-loud。

2026-08-29 起，新 thread 会把默认 `suppress` 或显式 `allow` 写成 session
事实。旧版 `--persist-memory` 与旧版默认 thread 都可能没有 effect，历史数据无法
安全区分；迁移只修复已经持久化明确 `suppress` 的 Turn，不猜测空 metadata。

Shell 是异步可观察传输：命令在初始窗口没有结束时返回 `execution_id`，父 T 用 `write_stdin` 读取增量输出。它不创建第二个 Akashic runtime；CLI 仍连接当前 Gateway，所以能租用内存中的 latest。

### 9.1 attached 生命周期

验证调用默认 attached：

```text
父 Turn T cleanup / task_stop / Shell hard timeout
  → 终止 exec CLI process group
  → control socket 关闭
  → Gateway 取消该连接拥有的 attached Turn V
  → V 写 cancelled/interrupted terminal，释放 latest lease
```

当前实现完成整条 attached 取消链。显式 `--detach` 才允许调用方离开后 V 继续，且 CLI 先返回 thread/turn handle；插件自验证不得使用 detached。

## 10. 安装、晋升与丢弃接口

首版最小接口：

```text
plugin-install ...   安装并等待 latest_ready；存在未决 latest 时拒绝
plugin-status        读取 stable/latest identity、phase、provenance 和 error
plugin-promote ID    原子 stable=latest；没有未决候选时拒绝
plugin-discard ID    原子 latest=stable；没有未决候选时拒绝
```

这是单用户、单 runtime owner 下的乐观模型：不增加 reservation token 或每 session candidate namespace。其他 session 只有显式请求 `runtime=latest` 才能读取候选；普通请求永远拿 stable。任何 session 都不能靠普通 turn 自动 promote/discard，管理动作仍通过受认证 control/CLI 边界。

安装首次新增插件时，stable snapshot 表示“插件不存在”，latest 表示“插件存在”；更新、禁用和卸载同样以完整 snapshot 差异表达，不能只给单个工具打补丁。

## 11. 独占 endpoint 例外

如果候选改变 `managed_services()` 或 `channels()`，旧 stable 与 latest 可能竞争同一 TCP/Unix socket、bot long-poll、webhook/token 或 singleton daemon。此时同进程双快照无法完整启动候选。

分类规则必须比较真实资源 identity，而不是笼统地把任意贡献差异都当冲突：

```text
纯 tool / skill / prompt / non-conflicting MCP
└── 走 stable + latest 在线验证

固定端口 / bot ownership / singleton service 冲突
├── 走隔离 runtime + 隔离 endpoint/workspace 验证
└── 或在父 turn 结束后 quiesce/switch，再由后续 turn 验收
```

持有 S0 lease 的 T 不能发起等待 S0 自己归零的 endpoint 切换。这个例外不应把普通插件重新拖回全局串行。

## 12. 递归循环与停止条件

Agent 的开发 Skill 使用下面的受控循环：

```text
inspect → modify → source tests → commit/install → latest behavior test
   ▲                                                   │
   └──────────── actionable failure + remaining budget ┘

pass      → promote → re-read final pointers → report complete
blocked   → preserve stable → report exact missing capability/evidence
cancelled → discard candidate when owned and safe → report terminal
```

禁止：

- 相同失败不改代码就重复 install/exec。
- 仅凭模型 finalResponse 自述“工具可用”通过。
- latest 失败后覆盖 stable 或清理 stable artifact。
- 为得到全绿修改 oracle、跳过真实外部边界或伪造 tool result。
- 在未验证写副作用、独占 endpoint 或 memory policy 时宣布全部完成。

## 13. 迁移顺序

### Phase 1：证明跨 session 并发安全

1. [完成] 建立 `SessionLaneRegistry`，让 channel/direct 与 control executor 汇入同一 session owner。
2. [完成] 移除两层全局整轮互斥，保留全局有界 admission。
3. [完成] 对其余 `AgentLoop`、reasoner、ContextBuilder、ToolRegistry、SessionManager、provider、插件实例和 Shell owner 完成审计；将共享 ContextBuilder 诊断投影迁回 task-local owner。
4. [完成] 验证不同 session 并发、同 session 串行和 passive `message_push` 的 ChatLane 顺序。

### Phase 2：引入 stable/latest

1. [完成] cache 改成 generation-addressed artifact identity，旧 stable 不被候选更新覆盖；现有安装命令保持 immediate stable 兼容，runtime owner 接通后再显式 staged install。
2. [完成] snapshot store 持有 stable/latest 与单一 candidate transaction。
3. [完成] reload journal 持久化 `latest_ready` / `promoting` 阶段，并按 durable pointer 恢复 stable 或候选。
4. [完成] `plugin-install` 由 runtime owner 完成 latest_ready 终态。
5. [完成] PluginManager 的 promote/discard/status 已接通 control RPC 与 CLI。

当前 cache 布局如下。两个逻辑 pointer 存在同一个原子状态文件中，避免进程崩溃留下跨文件撕裂状态；pointer 只引用插件目录内的安全相对路径。旧 artifact 在显式卸载前保留，不由 watcher 自动清理：

```text
cache/<marketplace>/<plugin>/
├─ .pointers.json          # {stable, latest}
└─ .artifacts/
   ├─ <version>-<git-sha-prefix>/
   └─ ...
```

### Phase 3：接通程序化验证

1. [完成] control thread/turn 接受严格枚举的 runtime selector。
2. [完成] programmatic 新 session 默认 `skip_post_memory=true`，保留 recall。
3. [完成] `exec` 增加 `--runtime`、`--persist-memory` 和 attached disconnect cancellation。
4. [完成] Tool/Skill loader 从当前绑定 snapshot 取 catalog。
5. [完成] `message_push` 使用 passive-send ChatLane，不等待目标 session lane。

### Phase 4：交付 Agent Skill

[完成] 启用 `develop-akashic-plugin` 的完整递归路径。Skill 仍先 feature-detect；运行旧版本或能力不完整时明确报告 runtime self-validation unavailable，不能把静态验证写成完整成功。

## 14. 独立验收

### 14.1 并发与死锁

1. T 持有 session A 和 stable S0 时，V 使用 session B + latest S1 完成；测试有严格超时并证明不是 T 先结束。
2. 同一 session 的第二个 turn 仍 queued/rejected，不发生历史交错。
3. 第三个普通 session C 在 V 运行时仍获得 S0。
4. 全局 active turn/bytes/runtime objects 容量拒绝只影响新请求，不终止既有 turn。

### 14.2 可见性与记忆

1. S1 新增的 tool 与 Skill 只在 `runtime=latest` 可见；T/S0 与 C/S0 不可见。
2. V 能 recall 已有记忆；V 的内容不进入 Memory2、Markdown consolidation 或 Akasha。
3. V 的 messages、tool items、final response 和 terminal status 可从 SessionDB/control 重新读取。
4. `--persist-memory` 只在显式创建时改变 policy，非法组合在边界失败。

### 14.3 pointer 和恢复

1. install 返回时 latest 已可租用；立即 exec 不需 sleep/revision 参数。
2. 未决 latest 存在时第二次 install fail-loud，stable/latest 均不变。
3. promote 后新默认 turn 获得已验证的同一 S1；T 的 S0 lease 不变，随后正常 drain。
4. discard 后新默认 turn 仍为 S0，S1 在验证 lease 归零后清理。
5. 在 latest_ready、promoting 和 discarding 三个 crash point 重启，stable 始终可恢复且不会误晋升。

### 14.4 Shell、取消和 `message_push`

1. Shell 返回 execution_id，`write_stdin` 只读新增 JSONL 并最终观察 terminal。
2. 父 terminal 即使包含超过 64 KiB 的工具轨迹，也必须在 2 MiB 协议上限内完整送达调用方。
3. 杀死 attached exec CLI 后服务端 V 进入 cancelled/interrupted，并释放 latest lease。
4. V 的 `message_push` 在 T 未结束时可取得短 send owner；不会等待 session A lane。
5. 两次实际 channel send 不重叠；scheduler 的 non-passive send 仍排在用户被动回复之后。
6. push 正文不进入 T 的 Prompt/目标 session history；V 的工具 trace 包含真实 DeliveryReceipt 终态。

### 14.5 行为 oracle

对一个测试插件新增 `candidate_only_tool` 和 `candidate-only-skill`：

- known-good candidate：V 必须实际调用新工具，并断言结构化 tool item、参数、result 和领域状态。
- mutant 1：工具注册缺失；验收必须失败，不能接受 finalResponse 自述。
- mutant 2：工具返回假 success 但领域状态未变；before/after oracle 必须失败。
- mutant 3：把 V 错绑 stable；tool visibility oracle 必须失败。
- mutant 4：恢复全局锁；死锁超时 oracle 必须失败。
- mutant 5：让 programmatic session 写入 semantic memory；memory write-set oracle 必须失败。

### 14.6 自动化证据

| 合同 | 直接证据 |
|---|---|
| latest/stable、install 完成定义、candidate 单 owner | `tests/test_plugin_runtime_control.py`、`tests/test_plugin_hot_reload.py` 的 selector、promotion、KV write 与 crash recovery 用例 |
| candidate 诊断入口 | `tests/test_plugin_doctor.py::test_plugin_doctor_reads_latest_artifact_candidate` 证明 doctor 按 pointer 读取 `.artifacts` 下的 latest |
| 跨 session 并发、同 session 串行 | `tests/test_turn_pipelines.py::test_process_direct_runs_concurrently_with_another_session`、`test_process_direct_waits_for_the_same_session_lane` |
| programmatic runtime、长 terminal、SessionDB 与默认 memory policy | `tests/control/test_exec_cli.py::test_exec_new_defaults_to_read_only_memory_and_selects_runtime`、`test_control_client_reads_terminal_larger_than_asyncio_default`、`tests/control/test_protocol.py::test_thread_runtime_selector_is_strict_and_inherited_by_turn` |
| `message_push` 不等父 session 且实际 send 串行 | `tests/test_support_modules.py::test_message_push_passive_role_does_not_wait_for_passive_lane`、`test_message_push_passive_role_serializes_actual_same_chat_send` |
| turn-local 调试投影 | `tests/test_support_modules.py::test_context_builder_debug_projection_is_turn_local` |
| 生产轨迹 oracle | `tests/semantic/test_recursive_plugin_self_validation_trajectory.py` 通过真实 `PluginManager` install、`ConversationRuntime` latest child、`DefaultReasoner`、候选工具、`message_push`、SessionDB、reload journal 与 promote 取证；stable misbinding、假 tool success、假领域结果从真实执行 seam 注入并必须被拒绝 |
| 聚合合同 oracle 与已知错误 | `tests/semantic/test_recursive_plugin_self_validation_contract.py` 对跨场景 observation 做稳定性自测：global lock、parent terminal overflow、semantic write、blocking push、crash promotion 等 mutant；它不替代生产轨迹测试 |

上述证据注册为 P0 `recursive_plugin_validation` group、`plugin_runtime_selection` state contract 与 `recursive_plugin_self_validation_contract` scenario。Gate 的主通过证据由生产组件生成，不接受手工 observation：它观察 pointer/journal、真实 tool item、SessionDB、semantic write set、ChatLane timer 和 promote；独立 startup 用例覆盖 crash recovery。coverage baseline 只记录批准后的合同映射，不充当测试通过报告。

### 14.7 真实模型闭环证据

2026-08-06 在一次性 workspace、独立 plugin home、关闭 channel/proactive 且 `memory.enabled=false`、`engine=""` 的 Gateway 中完成真实验收：

```text
parent stable T
├─ 自行编写 prompt-only Git 插件 helloworld_rule@local
├─ source tests 首跑暴露测试契约错误，修复后 5 passed
├─ install: stable 335c0bba4d86f063 → latest 12aec517458d6252
├─ child latest V: 普通问题 → 精确 helloworld，status=completed
├─ 回读 SessionDB、reload journal 与 pointer
└─ promote: stable == latest == 12aec517458d6252
```

- 父 session/turn：`programmatic:e8971595-33b5-4062-b6a5-0e244030c546` / `turn:6d59a0e4-7e6b-4cbd-aa25-3f304f67201e`，`21:58:48.912912Z` 开始，`22:07:25.139531Z` 完成。
- 子 session/turn：`programmatic:d3850b5b-b004-473b-9106-c95b28395a0f` / `turn:ca68e274-33b5-4be0-acd8-1d7b898825e8`，`22:06:06.976456Z` 开始，`22:06:10.879357Z` 完成；因此 V 在 T 释放前约 494 秒完成，不存在跨 session 整轮锁。
- 子输入是普通问题 `What is the capital of France?`，不含注入规则；SessionDB 的 user/assistant 两条消息和 turn final 均证明最终回答严格为 `helloworld`，session metadata 为 `runtime=latest`、`skip_post_memory=true`。
- reload transaction `abf4b6346c6e43dd9d9509de3530af54` 最终 `phase=complete`、`error=''`；source commit 为 `a5f7d54b13ac88c9dfa5a36ba08f64960c984dd7`，stable/latest pointer 均为 `.artifacts/0.1.0-a5f7d54b13ac88c9`。
- 父 terminal 的 `items_json` 为 769,056 bytes；CLI 完整收到 `completed`，证明显式 2 MiB control frame 合同修复了旧 64 KiB 断连。
- 本次父 Agent 用了 81 次模型迭代、100 次工具调用，功能成立但效率不可接受。主要浪费来自成功路径预读 diagnostics、重复枚举已给定运行时信息和反向考古 CLI/EventBus；builtin Skill 已把 prompt-only 模板、单向快路径与按失败诊断写成明确约束。
- 现场还发现 `plugin-doctor` 仍按旧可见版本目录查找、误报 `.artifacts` 候选不存在；现已让 doctor 通过原子 pointer 读取 latest，并以定向测试覆盖。

## 15. 非目标与仍需单独设计的边界

- 首版不实现多候选 archive、分支搜索或自动选择多个 parent。
- 不让 programmatic child 的内容直接注入父 Prompt；父只消费结构化结果。
- 不把 Shell 当安全沙箱；不可信插件仍需要容器/namespace/最小权限。
- 不承诺在线回滚任意外部效果或 plugin-owned 文件写入。
- 不用全局锁保护无法归属的共享可变字段；这些字段必须先迁回明确 owner。
- 不让独占 endpoint 候选绕过资源冲突证明。

## 16. 当前实现对账入口

- control thread owner 与容量短临界区：`agent/control/runtime.py::ConversationRuntime._control_admission_lock`
- session 整轮 owner：`agent/looping/session_lane.py::SessionLaneRegistry`
- channel 入站分发：`bootstrap/passive_worker.py::PassiveMessageWorker`
- passive/direct 统一准入：`agent/looping/core.py::AgentLoop._process_with_runtime_admission`
- programmatic CLI：`main.py::run_exec`
- control 执行桥：`bootstrap/control_execution.py::execute_control_turn`
- snapshot 事务：`agent/plugins/snapshot.py::RuntimeSnapshotStore`
- plugin 安装：`agent/plugins/install.py::install_git_plugin`
- 消息投递 lane：`bus/queue.py::ChatLane`
- `message_push`：`agent/tools/message_push.py::MessagePushTool`
- session 级记忆排除：[程序化调用的 session 级记忆排除](programmatic-session-memory-exclusion.md)
- Shell 生命周期：[Unified Shell Execution](unified-shell-execution.md)
