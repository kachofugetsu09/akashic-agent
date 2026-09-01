# 0054 · Agent 由普通插件组成

- 状态：accepted
- 日期：2026-09-01
- 关联条款：STA-001～STA-003、CAP-001～CAP-002、RUN-001～RUN-012、OUT-001～OUT-005、PLG-001～PLG-018、TST-001～TST-006
- supersedes：0039 中“React 原子能力留在 Core”的 owner 结论
- superseded by：无

## 背景

当前被动回复由 bootstrap 先构造 `SessionManager`、`ToolRegistry` 和 `AgentLoop`，再构造
`ConversationRuntime`，最后加载普通插件。`AgentLoop._process()` 和 `PassiveTurnPipeline` 因此同时
拥有 ReAct、Session 提交、模型选择、Prompt phase、`tool_search`、`message_push`、命令短路、rollout 提示
和 Shell cleanup 等不同变化轴。普通插件虽已拥有 Context、Service、Fiber、Effect 和 exact lease，
完整被动链仍要先经过一条 Core 私有控制流。

DeepSeek Harness（DSH）以 commit `dd6322d604e00eec1ba5e0c8541159906a21094a` 为本决策的参考快照。
它的原则是“所有产品部件都是插件，没有特权 Core”（`docs/architecture.md:9-13`）。默认
`AgentLoop` 注入 `agents`、`sessions`、`llm`、`tools`、`systemPrompt` 和 `sessionProjections`，加上自己
恰好是当前默认装配的七个 Service（`packages/core/agent-loop/src/index.ts:351-354`）。但数量不是
架构律：每块是否存在，只能由它是否拥有独立事实、不变量、控制流或真实边界决定。

DSH 还给出两个必须保留的边界：`Session` 自己从 append-only log 派生模型历史
（`packages/core/session/src/index.ts:567-653,699-745`）；`session-projection` 只对已提交事件做可回放 fold
（`packages/session/session-projection/src/index.ts:34-85,169-211`）。两者不能变成重复的 history owner。

## 决定

Akashic 的 Agent 骨架是一张由同一 v3 loader 装载的最小无环能力图。默认装配包含：

1. `sessions`：唯一拥有 Session、Message、Turn 权威事实、事务与模型历史派生；
2. `models`：唯一拥有 provider/model registry、执行绑定与流式调用；选择作为输入传入，不读写
   Session metadata；
3. `tools`：唯一拥有工具注册、作用域可见集合、运行与结构化结果；
4. `system-prompt`：唯一拥有有序 system section 注册与组装，不拥有 history、Session 或模型调用；
5. `model-input`：为一次 Agent Turn 打开一只私有输入状态，并在**每次**模型尝试前把权威 history、
   当前 Turn transcript、system text、tool schemas 和 model limit 变成不可变 provider input；
   basic provider 直接组合，compaction provider 按已提交 ledger 投影，Root 只允许一个 provider；
6. `agents`：唯一拥有公开 Agent 合同、live registry、source 归属和 factory slot；
7. `agent-loop`：提供默认具体 Agent，拥有 inbox、Turn/Step/ReAct、取消与 terminal，并向 `agents` 注册
   factory。

`model-input` 是 Akashic 相对 DSH 的有意差异。DSH compaction 通过 `agent/pre-step` 改变进入 Step 的 Message，
并向 Session log 追加 surface replacement（`packages/compaction/compaction-basic/src/index.ts:127-225`）。Akashic 的
`sessions.db/messages` 不得因上下文裁切而 UPDATE/DELETE，且已有独立 compaction ledger 与 request projection。
因此这条“权威历史与本 Turn 进展 → 每次有限模型输入”是真实边界，但它不是 before-step hook，
不拥有 history，也不允许任意中间件。定义、provider 和 consumer 必须齐全：公共结构合同定义
`open`、每次 `build` 和每次 `settle`；basic/compaction 普通插件二选一提供，`agent-loop` 唯一消费。

`MODEL_INPUT.open(TurnInput) -> InputState` 每个 Turn 恰好一次。随后每次 provider attempt 都必须调用
`InputState.build(InputCall) -> ProviderInput`。`InputCall` 冻结 `call_id`、call/try 序号、`normal` 或
`too_long` 原因、当前完整 Turn transcript、tool schemas、model/input/output limit 和此前已结算 usage；
`ProviderInput` 返回完整 provider payload、`InputSize` 与一只 opaque `InputReceipt`。该 attempt 无论成功、
上下文过长、失败或取消，都必须恰好一次调用 `settle(CallResult)`；只有返回的 `InputRetry` 允许同一
逻辑 call 再 build 一次。basic provider 永不要求 overflow retry；compaction provider 可以在第二次 build
收缩输入。具体 loop 只传这些 typed value，不识别 compaction，不读写 provider 的 turn-local state。

compaction provider 可在 `InputState` 内私有保存 ledger head、已闭合 tool batch、token meter 和待发布
fact。它从下一次 `InputCall.transcript` 识别新闭合 batch；成功 settle 记录 usage，并在单次运行只发布
一次已提交 fact；崩溃恢复按 `source_ref` 幂等补发。
失败或取消不能把已经提交的 checkpoint 伪装成回滚，下一次 `open` 必须从 ledger/receipt 重放并补发。
这保留了当前每次调用 gate、overflow retry、tool batch、usage 与 recovery 行为，却删除 mutable
`ProviderRequestBinding` 和 Core 私有 gate。

`build`/`settle` 不是一对公共 phase hook：两者不能分开注册，没有 listener 顺序，不接受任意 Agent
状态，也不能改写别的 capability。它们只属于同一个 exclusive provider，并由同一只 receipt 强制配对。

`session-view` 只有在证明需要“对已提交 Session 事实做通用可回放 fold”时才是额外普通
插件。它不参与模型 history 构造，不允许 I/O 或回写 Session。若真实 consumer 可以直接读已提交
事实，就不安装它。数量由能力决定，不由“七块”反向决定能力。

Channel、Command、Scheduler、Wake、Subagent、Compaction、Markdown memory、Tool Search、Shell 和其他产品能力
仍是普通插件。它们注入上述 Service，不能要求 Core 增加插件 ID、工具名或来源特判。

```text
sessions ─────────────┐
models ──────────────┤
tools ───────────────┤
system-prompt ───────┤
model-input ────────┤
                         ▼
                     agent-loop ── factory ──► agents

session-view? ── committed Session facts ──► observers (只有真实 fold consumer 时)

Channel / Command / Scheduler / Wake / Subagent
                    │ public Service
                    └──────────────► agents
```

安装、权限和生命周期上同权不等于角色对称。Registry、provider 和 driver 本来就有不同依赖位置。
`agent-loop` 位于图顶端不是特权；它仍以同样的 manifest 装载，可禁用、卸载或被另一个 Agent
factory 替换。Root 在 `snapshot.sealing` 时要求正式拓扑恰有一个默认 factory；缺失或冲突在发布前
fail-loud。DSH 也通过 `agents.setFactory()` 把 registry 与具体 driver 分开
（`packages/core/agent/src/index.ts:352-422`）。

## Core 边界

Core publication plane 继续唯一拥有 artifact、generation、candidate isolation、Root readiness、stable/latest、
lease、原子 publication、drain 和恢复日志。它保留三个领域中性执行原子：

1. `ServiceCall`：绑定一个 `ServiceKey`、取得构造时已固定的 exact lease、绑定当前 task、等待调用完成并
   逆序释放。公开 `call(action)` 没有 selector、snapshot ID 或 plugin ID；
2. `RootScope`：只取得自身 exact Root lease；Root 退休后返回 `RootRetired`，不改投新 stable；
3. `TaskControl`：只按 opaque scope key/task key 原子 claim，保存 exact lease、task 和 cancel callback，terminal 后
   release。

三者不识别 Message、Turn、Session、Agent、ReAct、工具名或来源，不为缺失 Service fallback。snapshot lease
包住完整 ReAct Turn 不授予某个插件特权；它只保证同一项工作始终使用同一盒插件。

## 非特权判定

- 所有骨架插件使用与外部插件相同的 manifest、loader、Context、Fiber、Effect、PluginRuntime、generation 和
  disabled builtin 规则；“内置”只表示默认发行。
- 实现只能 import 版本化 public Plugin API、结构合同和自身包；不得 import 旧 Core class、Session 私有 store
  或兄弟插件源码。
- Core 与 bootstrap 不按插件 ID、`tool_search`、`message_push`、Shell、模型、记忆文件或 Channel 来源分支。
- `sessions` 是 `sessions.db` 的唯一正式 writer；其他插件只获得窄 Service，不获得任意 SQL 或全功能 repository。
- 缺少 required Service、默认 factory 或唯一 writer 时 Root 不发布；运行时没有 `try plugin else legacy`、
  旧新双写、双 sender 或静默默认值。
- 一个内置插件移到独立仓库后，只要声明和版本化公共合同相同，就不需要 Core patch。

## 行为归属

ReAct 的 provider/tool 迭代、工具 batch 结算、max-iteration、stream、inbox、取消与 terminal 是一个具体
Agent 算法，留在 `agent-loop` 内，不为每个步骤各造插件。模型选择从 Agent request 传入 `models`；
Tool Search 只使用 `tools` 的 scoped view；system sections 由 `system-prompt` 拥有；model-facing 权威
history 由 `sessions` 派生，有限 provider input 由 `model-input` 构造；提交由 `sessions` 拥有；发送由
Channel/Delivery 拥有；Shell cleanup 由 Shell 插件消费 terminal fact。

现有 `before_turn`、`before_reasoning`、`before_step`、`after_step`、`after_reasoning` 和 `after_turn` 总 phase 全部退役。
DSH 虽有只返回 `reject | enter(messages)` 的窄 `agent/pre-step`
（`packages/core/agent/src/runtime-types.ts:55-63,226-238`），但没有对称的 after-step 套件；Akashic 当前也没有
非测试 `before_step` consumer。因此本迁移不预建 `before-step`、`step-decision` 或对称 phase API。真实 consumer
未来必须用独立需求证明新 seam。

迁移后只保留 owner 明确的领域接入点，例如 system section、model request、tool view/run、`TurnSaved` 和
outbound view。不新增 `ReplyEdit`；它仍是缩小的 after-reasoning 有序改写链。模型指引归
`system-prompt`，已提交的插件事实归插件自己的 ledger，仅 Channel 显示变化归 outbound view。若必须
改变持久 assistant 正文，需要独立的 Message 合同决策，不能伪装成 loop hook。

## 迁移与回滚

迁移按 owner 串行切换，不进行灰度、流量分组、运行时 shadow、双执行或双写。isolated candidate 只做拓扑、权限和
无正式副作用的验证。每个实现批次先标记待退役 owner，再把正式调用者一次切到唯一新 owner。关键测试、两个独立
Terra xhigh review 和一个独立 name review 通过后，同批物理删除旧实现。外部 consumer 只能以明确
`DEPRECATED(EXTERNAL)` migration block 阻塞公共面删除；跨仓收尾后不保留 alias、adapter、flag、fallback 或 block。

回滚只选择上一个完整 Git commit、不可变 generation 和执行前备份。已提交的 Session 行、已发送消息、远程调用
或文件写入不因代码回滚而伪装撤销。

## 影响

- `sessions.db/messages` 继续正常只追加；本决策不改 schema、Turn/Attempt/Interaction 身份、compaction ledger、
  附件或删除权限。
- passive、control、scheduler、wake 与 subagent 最终使用同一 `agents` Service，但各来源仍拥有自己的准入前
  规则、领域状态和 delivery settle。
- reload 后可以通过 opaque `TaskCancel` 取消旧 Root 内尚未终结的工作；旧 owner 持有自己的 lease 到
  terminal，不迁移内存业务对象。
- 禁用任一 required plugin 使依赖拓扑不可发布，不触发 Core 私有实现。

## 验收

- [ ] snapshot 外入口只能通过泛型 Service 调用边界进入；该边界源码与测试无 Agent 领域词。
- [ ] root-bound task 在 reload 前后只运行自己的 exact Root；opaque task 可跨代取消，同一 scope 不由两代
  同时 claim。
- [ ] 最小骨架全部由正式 v3 loader 挂载，依赖图无环，Root sealing 恰有一个默认 factory。
- [ ] `sessions` 唯一派生权威 model history；`model-input` 每 Turn 只 open 一次、每个 provider attempt
  恰好 build/settle 一次，overflow retry、tool batch、usage 与 recovery 等价；`session-view` 若存在只
  fold 已提交事实，且有 exact consumer 证据。
- [ ] `agents` 只拥有合同/registry/factory；默认具体 Agent 的 inbox、Turn/Step、cancel/terminal 只属于 `agent-loop`。
- [ ] 没有 `before-step`、`after-step`、`ReplyEdit`、总 mutable ctx 或对称 phase 套件。
- [ ] Core 零插件 ID/工具名/来源特判，零旧 `AgentLoop`/`PassiveTurnPipeline` consumer，零兼容壳。
- [ ] 固定场景依次比较迁移前后 Session write set、事件、provider/tool trace、stream、delivery、error/cancel/
  interrupt 和附件结果；除批准字段外相同。
- [ ] M0 由一个 Terra xhigh concept review 和一个独立 name review 批准；M1～M9 每个实现批次有
  两个独立 Terra xhigh implementation review 和一个独立 name review；最终全量 Gate 通过。
