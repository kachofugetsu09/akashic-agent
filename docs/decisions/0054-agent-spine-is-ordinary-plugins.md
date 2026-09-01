# 0054 · Agent 内骨架由七个普通插件组成

- 状态：accepted
- 日期：2026-09-01
- 关联条款：STA-001～STA-003、CAP-001～CAP-002、RUN-001～RUN-012、OUT-001～OUT-005、PLG-001～PLG-018、TST-001～TST-006
- supersedes：0039 中“React 原子能力留在 Core”的 owner 结论
- superseded by：无

## 背景

当前被动回复由 bootstrap 先构造 `SessionManager`、`ToolRegistry` 和 `AgentLoop`，再构造
`ConversationRuntime`，最后加载普通插件。`AgentLoop._process()` 和
`PassiveTurnPipeline` 因此同时拥有 ReAct 算法、Session 提交、模型选择、Prompt phase、
`tool_search`、`message_push` 媒体、命令短路、插件 rollout 提示和 Shell cleanup 等不同变化轴。
普通插件虽已拥有 Context、Service、Inject、Fiber、Effect 和 exact generation lease，完整
被动链仍要先经过一条 Core 私有控制流。

DeepSeek Harness 的可复用部分不是七个特权组件，而是一条很小的依赖骨架：agent loop 只组合
agents、sessions、LLM、tools、prompt 和 session view。Akashic 已经拥有比它更强的
持久化、候选隔离和 snapshot publication；缺口是让 snapshot 内普通 Service 成为正式调用入口，
不是再复制一套插件系统。

## 决定

Akashic 的 Agent 内骨架由七个经同一 v3 loader 装载的普通插件组成：

1. `sessions`：唯一拥有 Session、Message、Turn 和附件等权威持久事实及窄事务端口；
2. `models`：唯一拥有 provider、模型目录、角色选择和一次 Turn 的冻结执行绑定；
3. `tools`：唯一拥有工具注册、每 Turn 可见集合、执行、授权结果和结构化工具事实；
4. `prompt`：唯一拥有有序 `PromptSection` 组合，不拥有 Session 或模型调用；
5. `session-view`：唯一拥有从 Session 快照构造 model-facing 临时只读 view，
   不建立第二份权威 history，也不拥有展示、保存或提交后事实；
6. `agents`：唯一拥有 agent registry、Turn admission 规则、取消/terminal 领域语义和 runner slot；
7. `agent-loop`：实现一次直接 ReAct 算法，注入前六项并向 `agents` 注册 runner。

“七个”只描述一次 Agent 工作的内骨架，不是整个系统只能安装七个插件。Channel、Command、
Scheduler、Wake、Subagent、Compaction、Markdown memory、Tool Search、Shell 和其他产品能力仍是
普通插件；它们注入上述 Service，不能要求 Core 增加插件 ID、工具名或来源特判。

七项能力同权但不要求没有依赖。依赖方向固定且无环：基础 registry 先提供 Service，
`agents` 只依赖 `sessions`，`agent-loop` 注入其余六项并向 `agents` 的 slot 注册 runner。
Root 在 `snapshot.sealing` 时要求正式拓扑恰有一个默认 runner；缺失或冲突在发布前 fail-loud。

```text
sessions ───────────────► agents ◄──────────── agent-loop
                            ▲                      ▲
models ─────────────────────┼──────────────────────┤
tools ──────────────────────┼──────────────────────┤
prompt ─────────────────────┼──────────────────────┤
session-view ───────────────┴──────────────────────┘

Channel / Command / Scheduler / Wake / Subagent
                    │ public Service
                    └──────────────► agents
```

Core publication plane 继续唯一拥有 artifact、generation、candidate isolation、Root readiness、
stable/latest、lease、原子 publication、drain 和恢复日志。它补齐三个领域中性的执行原子：

1. 一个绑定单一 `ServiceKey`
的泛型 `ServiceCall`：取得构造时已经固定的 exact snapshot lease、绑定当前 task、
取已绑定 Service、等待调用结束、逆序释放。公开 `call(action)` 没有 selector、snapshot ID 或
plugin ID 参数；普通 host 的 `ServiceCall` 永远固定 stable，插件不能创建 `ServiceCall`。attached validation
child 只使用 Core 根据父 Turn 与 candidate identity 铸造的一次性 exact lease，host 和插件都不能
选择 latest。该边界不知道 `Message`、`Turn`、`Session`、ReAct 或工具名，也不能替任何缺失
Service fallback；
2. 每个 Fiber 都能使用但不能选择目标的 `RootScope`。它只取得自身 exact Root lease；
   Root 退休后新 acquire 返回 `RootRetired`，不改投新 stable；
3. process-wide `TaskControl`。它只按 opaque scope key/task key 原子 claim，保存
   exact lease、task 和 cancel callback，并在 terminal 后 release；它不知道 Turn、Session、Agent
   或来源。窄 `TaskCancel` 可以按已知 task key 取消旧 generation 的仍活 task，使 reload
   期间不丢 `/stop`，但不能创建工作、读取业务状态或选择 snapshot。

因此不保留“特权 Agent 插件”。snapshot lease 包住完整 ReAct Turn 并不授予其中某个插件
特权；它只是保证该 Turn 看到同一整盒积木。普通 Channel 回调已经处于 exact snapshot 时直接
注入 `agents`；Control 等 snapshot 外入口通过泛型调用边界进入。`agents` 若启动长于调用栈的
task，只能从自己实例绑定的 `RootScope` 取得 lease，再交给 `TaskControl` 保持到
terminal，不能自行选择 stable/latest。每次 admission 返回 opaque task key；Control
从 accepted receipt 或 durable active-attempt fact 取得该 key，通过 `TaskControl` 保存的旧 cancel
callback 通知旧 task。新 generation 不接管旧插件的业务状态，也不能 claim 同一 opaque scope。

## 非特权判定

- 七个插件使用相同的 manifest、loader、Context、Fiber、Effect、PluginRuntime、generation 和
  disabled builtin 规则；“内置”只表示默认发行。
- 七个实现只能 import 版本化 public Plugin API、结构合同和自身包；不得用旧 Core class、
  `PluginManager`、Session 私有 store 或兄弟插件源码作隐藏后门。
- Core 与 bootstrap 不按七个插件的 ID、`tool_search`、`message_push`、`shell`、模型名、记忆文件
  或 Channel 来源分支，也不制造这些领域的专用 Service。
- `RootScope` 与 `TaskControl` 只处理 Root identity、opaque key、task、lease 和 cancel
  callback；它们的接口和测试不得出现 Agent/Turn/Session/Scheduler 等领域字段。
- 插件只获得声明的 `workspace_files`、plugin-data 和公开窄 Service；`sessions` 是
  `sessions.db` 的唯一正式 writer，其他插件不能获得任意 SQL 或全功能 repository。
- 缺少任一 required Service、默认 runner 或唯一 writer 时，Root 不发布；运行时没有
  `try plugin else legacy`、旧新双写、双 sender 或静默默认值。
- 一个内置插件移到独立仓库后，只要声明和版本化公共合同相同，就不需要 Core patch。

## 行为归属

ReAct 的 provider/tool 迭代、工具 batch 结算、max-iteration、stream 与取消检查是一个直接算法，
留在 `agent-loop` 内，不为每个步骤各造插件。独立变化轴通过领域 Service 组合：模型选择由
`models`，工具发现由普通 Tool Search 插件调用 `tools` 的 catalog/grant，Prompt facts 由
`prompt`，model-facing history view 由 `session-view`，提交由 `sessions`，发送由 Channel/Delivery，
Shell owner cleanup 由 Shell 插件自己的 Turn terminal listener 负责。

现有 `before_turn`、`before_reasoning`、`before_step`、`after_step`、`after_reasoning` 和
`after_turn` 总 phase 不是目标公共模型。迁移后只保留 owner 明确的领域接入点，例如
`PromptSection`、request prepare、tool check/result、`TurnSaved` 和 outbound view；
不能用一个可任意改写总状态的 hook 代替 Service 所有权。外部 consumer 只影响迁移顺序，不决定
接口是否保留：immutable 且 owner 明确的事实才可以 `keep`；只把 dataclass 标成 frozen，而内部仍
装有 mutable list/dict 或多个 owner 的总 payload，仍必须 `move`。mutable ctx、metadata bag 和内部
编码也必须 `move` 或 `remove`。Core 内部切换后，旧 public surface 只允许作为有 exact consumer 和删除阶段的
migration block；外部源码迁完的同一批必须物理删除。

## 迁移与回滚

迁移按 owner 串行切换，不进行灰度、流量分组、运行时 shadow、双执行或双写。isolated candidate
只做拓扑、权限和无正式副作用的验证，不接生产流量，也不把比较结果写入正式状态。行为等价通过
同一固定输入依次运行旧基线 artifact 与新实现，并比较记录结果；两条实现不会同时服务一次请求。

每批先给即将退役的 owner 明确 `deprecated` 标记和零新增 consumer 约束，再把正式调用者一次
切到唯一新 owner。关键测试通过、两个独立 Terra xhigh review 清除 P0/P1，且独立 Terra xhigh
name review 得到 `NAME PASS` 后，立即物理删除该批
旧实现、分支、配置和测试替身。外部源码 consumer 阻塞的旧 public surface 必须显式登记为
`DEPRECATED(EXTERNAL)`，且 Core 内部零 consumer；它不属于目标设计。跨仓收尾后不保留 alias、
adapter、feature flag、legacy mode、fallback 或 migration block。

回滚只选择上一个完整 Git commit、不可变 generation 和执行前备份。已经提交的 Session 行、
已发送消息、远程调用或文件写入不因代码回滚而伪装撤销；需要恢复的独占 writer 先停 admission、
排空 lease、关闭新 owner，再由旧完整 artifact 重新取得同一路径。

## 影响

- `sessions.db/messages` 继续正常只追加；本决策不改变 schema、Turn/Attempt/Interaction 身份、
  compaction ledger、附件或删除权限。
- passive、control、scheduler、wake 与 subagent 最终使用同一 `agents` Service，但各来源仍拥有
  自己的准入前规则、领域状态和 delivery settle。
- reload 后新 `agents` 可以通过 opaque `TaskCancel` 取消旧 Root 内尚未终结的工作；旧 owner
  保持自己的 lease 与 terminal，结束后才从 `TaskControl` 移除，不迁移内存业务对象。
- 禁用一个基础插件会使依赖拓扑不可发布，不触发 Core 私有实现。
- 0036 的 publication owner 与 0046 的增量候选隔离保持有效；stable/latest 是发布状态，不是
  灰度或双正式 writer。

## 验收

- [ ] snapshot 外入口只能通过泛型 Service 调用边界进入；该边界源代码与测试无 Agent 领域词。
- [ ] root-bound task 在 reload 前后只运行自己的 exact Root；opaque task 可跨代取消，且同一
  scope 不会由两代同时 claim。
- [ ] 七个基础插件全部由正式 v3 loader 挂载，依赖图无环，Root sealing 恰有一个默认 runner。
- [ ] passive 完整路径只经过 snapshot 内普通 Service；禁用任一依赖会在 admission 前 fail-loud。
- [ ] Core 零插件 ID/工具名/来源特判，零旧 `AgentLoop`/`PassiveTurnPipeline` consumer，零兼容壳。
- [ ] 固定场景依次比较迁移前后 Session write set、事件、provider/tool trace、stream、delivery、
  error/cancel/interrupt 和附件结果；除批准字段外相同。
- [ ] 每个迁移批次都有两个独立 Terra xhigh review 和一个独立 name review；最终 Concept Gate、
  全量 Gate 与 `NAME PASS` 全部通过。
- [ ] 外部插件源码、正式 Akashic workspace、协议 schema 和客户端不在本次 Core 迁移中改变。
