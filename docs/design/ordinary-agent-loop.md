# 普通 Agent Loop 规格

- 状态：proposed
- 日期：2026-09-02
- 上游：[0055](../decisions/0055-agent-loop-is-an-ordinary-plugin.md)、[0056](../decisions/0056-no-revert-promotes-candidate.md)、[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[0039](../decisions/0039-react-core-atoms-keep-sources-unprivileged.md)
- 参考：DeepSeek Harness `dd6322d604e00eec1ba5e0c8541159906a21094a`

## 1. 结论

最终不是“整个程序都是一个插件”，也不是“七个 phase 各做一个插件”。边界是：

```text
┌──────────────────────── Core ────────────────────────┐
│ 收到 Message · Turn 排队/取消 · exact snapshot       │
│ Session/delivery 提交 · terminal · plugin publication│
└──────────────────────────┬───────────────────────────┘
                           │ require(REACT)
                           ▼
┌──────────── ordinary agent-loop plugin ──────────────┐
│ 读上下文 → 组 Prompt → 调模型/工具 → 形成 Message    │
│ 顺序和循环属于插件；依赖来自普通 Service              │
└───────┬─────────────┬─────────────┬──────────────────┘
        │             │             │
   Prompt plugins  Tool plugins  Memory plugins ...
```

把它当积木盒：Core 是桌子和锁扣，保证一局游戏只有一个 Turn、用同一盒积木、最后只提交一次；
`agent-loop` 是普通玩法说明书；Tool、Prompt、Memory 是它从盒子里拿的平等积木。换玩法说明书不需要
改桌子。

## 2. 现状

### 2.1 当前真实调用链

```text
Inbound / Control
  → ConversationRuntime
  → bootstrap.app._execute_control_request
  → execute_control_turn(self.agent_loop, event_bus, request)
  → AgentLoop.process_direct_message
  → AgentLoop._process_with_runtime_admission
  → AgentLoop._react
  → PassiveTurnPipeline.run
```

- `bootstrap/app.py` 的 closure 固定捕获一个全局 `AgentLoop`。
- `ConversationRuntime` 只保存一个全局 `TurnExecutor` callable。
- 普通 inbound 到 `AgentLoop` 内才取得 stable snapshot；candidate child 当前也只带 selector/identity，仍由全局 loop 延后取得 lease。
- `AgentLoop` 固定构造 `DefaultReasoner` 和 `PassiveTurnPipeline`。
- `PassiveTurnPipeline` 固定组装 before-turn、reasoning、step、after-reasoning 和 after-turn 模块。

这条链能工作，但 Core 同时拥有“Turn 怎样安全结束”和“默认 Agent 怎样思考”两个独立变化轴。

### 2.2 已有可复用资产

| 已有积木 | 当前 owner | 目标用法 |
|---|---|---|
| Context / Service / Inject | composition Core | 选择能力和声明硬依赖 |
| Fiber / Effect | composition Core | 插件生命周期与逆序清理 |
| RuntimeSnapshot lease | plugin runtime | 一个完整 Turn 固定同一 Root |
| `SCOPED_TURNS` | control Core | Scheduler、Subagent、Wake 进入同一 Turn |
| `CHAT_MODELS` | model plugins | exact snapshot 中选择模型 |
| Tool catalog 和 Turn scope | Tool owner | 列出、授权和执行当前 Turn Tool |
| typed events | domain owners | Prompt、Tool、committed Turn 等贡献 |
| Session read/compaction | Session / plugins | 普通插件读取与投影 |
| delivery services | channel Core | 外部发送与 durable finality |
| stable/latest 和 journal | PluginManager | generation 发布与恢复 |

Scheduler 和 Subagent 已经是普通 v3 插件；Compaction、Markdown memory、model revision 和请求投影
也已经走普通插件能力。本设计不重做它们。

### 2.3 仍写死的特殊功能

| 特殊点 | 现状 | 目标 |
|---|---|---|
| 默认 loop | bootstrap 固定全局 `AgentLoop` | 一个普通插件提供 `REACT` |
| phase 列表 | Core 固定 pipeline 模块顺序 | agent-loop 插件私有算法 |
| Tool search | Reasoner 检查精确名字 `tool_search` 并解析返回文本 | 普通 Tool 改 Turn scope；下一 step 从 scope 重建可见 Tool |
| Tool search cache | Core `ToolDiscoveryState` | Tool search 插件自己的普通状态，或证明无用后删除 |
| host 提示 | pipeline 读取 Host Bridge 环境并拼 Prompt | 普通 system-prompt 插件贡献 |
| 人格和身份 | `ContextBuilder` 固定 Veda/identity/behavior | 普通 system-prompt 插件贡献 |
| Skill catalog | `ContextBuilder` 固定拼装 | 普通 Skill 与 `PROMPT_RENDER_EVENT` 贡献 |
| candidate skill 名 | pipeline 写 `_activeSkillNames`，未发现读取者 | 证明零 consumer 后删除 |
| rollout 业务 Gate | Core 检查 child/parent completion | 只记录 `plugin-revert`；没有就晋升 |

“迁出去”不等于每行代码一个插件。人格、host 提示和 Skill catalog 都是 system prompt 的普通贡献；
Tool search 是一个普通 Tool；循环顺序仍由一个 agent-loop 插件拥有。

## 3. DSH 怎样组装

DSH 没有把 loop 的每个步骤升级成 Cordis 原语。它只有两层：

1. Cordis 提供 Context、Service、Inject、Fiber 和 Effect。
2. 普通 `@deepseek-ai/dsh-agent-loop` 插件注入 agents、sessions、llm、tools、systemPrompt 和
   sessionProjections，并在自己内部拥有 `ReactLoopAgent` 算法。

`packages/bundle/base/cordis.patch.yml` 只把 agent-loop 当一行普通配置装入。DSH 当前也没有
Akashic 这种特殊 `tool_search` loop 分支；搜索词只在模型 catalog 文件中偶然出现。

Akashic 不照抄 DSH 的 TypeScript 类型，也不删除已有 generation lease。学习的是所有权：组合内核
不知道 agent loop 的业务阶段，agent-loop 插件也不拥有发布指针。

## 4. 目标合同

### 4.1 唯一新选择点

```python
REACT = ServiceKey[TurnExecutor]("core.react")
```

名字使用已有动词 `react`，callable 复用当前 `TurnExecutor`：输入 `TurnRequest`，返回
`ControlExecutionResult`。实施 PR 必须先把 `_controlTurnInputSource`、`_controlItemEvent` 等 callable
移出 request metadata，留在 Core task-local 边界；传给普通插件的 request 只含公开、可验证的
Turn 输入。不能因此新建一份平行 Turn DTO。

每个可运行 Root 恰有一个 `REACT` provider。默认 provider 来自仓库内置 `agent-loop`，但它通过
与外部插件相同的 loader、Context、Fiber 和 Effect 运行。Core 不按插件 ID 自动挂载。

`REACT` 进入 Core 的 required-service 集合。snapshot install/publish 在开放 admission 前按
ServiceKey 检查 zero provider；duplicate provider 继续由通用 composition 冲突检查拒绝。这个检查
只认识能力，不认识默认插件名字。

### 4.2 exact snapshot

`ConversationRuntime` 的固定顺序是：

```text
admit Turn
  → acquire exact snapshot
  → bind lease to Turn context
  → snapshot.context.require(REACT)
  → await react(request)
  → receive ControlExecutionResult
  → settle terminal / ACK and build TurnResult
  → release lease
```

当前 candidate child 只带 `runtime=latest` 和 generation/source metadata，真正的 lease 仍由全局
loop 延后取得，这是一个待修缺口。目标是在 child capability 消费时解析并核验 exact snapshot ID，
立即用现有 `RuntimeSnapshotStore.acquire(snapshot_id)` 取得 lease，再把 lease 交给 runtime。该
snapshot 已退役或身份漂移时 child fail-loud，绝不改选 stable。这个失败不成为 promotion Gate：
没有 revert 仍按 0056 晋升。

普通 inbound 由 `ConversationRuntime` 取得 current stable。两条来源只在“snapshot 从哪里来”不同，
之后走同一个 `REACT`。plugin snapshot 覆盖完整 react Turn 正是 lease 的用途，不需要特权插件。

### 4.3 依赖

`agent-loop` 不接收当前宽 `AgentLoopDeps`。迁移时逐项做以下判断：

1. 已有公开 Service 能表达：直接复用。
2. 事实 owner 已有但接口是私有对象：只导出最窄行为协议，普通名称优先使用 `SESSION`、`TOOLS`。
3. 只是转发、缓存或重复字段：删除。
4. 没有真实 consumer：不预建接口。

公共 Service 只提供完成 Turn 所需的操作，不提供任意 SQL、全局 PluginManager 或 bootstrap 对象。
它们仍由 Core 实现，因此 Session append、Tool grant 和 delivery finality 的不变量不搬进插件。

### 4.4 Tool search

Tool search 继续可以叫 `tool_search`，但名字只属于 Tool 声明，不进入 loop 分支：

```text
agent-loop asks TOOLS for visible schemas
  → model calls any visible Tool
  → tool_search changes current Turn tool scope
  → next step asks TOOLS again
  → newly granted Tool appears
```

loop 不解析 Tool 返回文本，不调用 `unlock_names_from_result`，也不维护第二份 visible list。grant 必须
带 Turn、attempt 和 snapshot identity；执行 owner 继续拒绝未授权 Tool。

### 4.5 Prompt

默认人格、行为、host 提示和 Skill catalog 由普通 system-prompt 插件贡献。当前可复用原子是
`PROMPT_RENDER_EVENT`，PR 2 先把它从私有 lifecycle 路径公开到稳定插件入口；只有它无法表达真实
consumer 时才新增窄 Service。agent-loop 只请求“本 Turn 的 system prompt”，不认识各块名字。固定安全指令只有在它是所有
agent-loop provider 都必须遵守的 Core 不变量时才留在 Core；产品人格不是这类不变量。

### 4.6 提交

插件可以调用 Session/delivery Service，或发出公开 typed event。各自的 Core owner 仍保证：

- Session Message 正常路径只追加；
- Turn identity、提交顺序和重复调用防护唯一；
- external delivery 有明确 receipt 和失败语义；
- terminal、ACK、lease release 由 runtime 最终结算。

不创建一个包含 messages、effects 和 state 的 `Turn Plan`。这种大 DTO 会复制 Context 和提交状态机。

## 5. 非特权证明

| 问题 | 合格答案 |
|---|---|
| loader 是否相同 | built-in 与 external 都走 pure-v3 `apply(ctx, config)` |
| Root 权利是否相同 | 都只能 provide/require 公共 Service |
| snapshot 是否相同 | 都被 exact Root 的 Fiber/Effect 生命周期约束 |
| Core 是否认识插件名 | 不认识；只 require `REACT` |
| 缺失是否 fallback | 不 fallback；Root readiness 失败 |
| 私有 import 是否允许 | 不允许；共享 contract checker 阻止 |
| 卸载后是否有壳 | 没有；没有 `REACT` 的 Root 不可发布 |

## 6. 自更新流程

```text
old agent-loop parent
  → plugin-install(candidate)
  → attached child gets exact candidate Root
  → child uses candidate Tool / Skill / loop normally
  → wrong: parent calls plugin-revert
  → no revert: parent seals, Core promotes candidate
```

Core 只检查候选 Root 在结构上可运行，以及 child 真正绑定 exact generation/source。Core 不检查 child
调用了哪类能力，也不把 terminal status 当业务投票。Agent 会在发现问题时调用 revert；设计中没有
“Agent 忘了”的世界。

## 7. 外部插件

2026-09-02 对 hua-home 正式 artifact 的只读审查发现：16 个启用 external 插件中，9 个可达插件
仍有 18 个私有 Core import，主要指向 timer、lifecycle、prompt、tool events 和 turn events。

迁移规则：

- 以 hua-home production artifact commit 为基线；本机 16/16 checkout HEAD 均与生产 artifact
  不同，不能直接拿本机 HEAD 当生产源码。
- 复用 `/mnt/data/coding/akashic-plugin/plugin-contracts`；它被 21 个 CI workflow 使用。升级现有
  checker 支持 pure-v3 和禁止私有 import，不创建第二个 checker。
- 接入点存在不代表设计干净。能由更小公开 Service/事件表达的旧入口，标出替换关系，最后与外部
  插件同批删除。
- Feishu/QQbot 的 candidate prepare/discard/publish 集成测试改用正式 testkit，不让外部测试依赖
  `PluginManager` 私有构造。

## 8. Stacked PR 顺序

每个 PR 都是一次完整 owner 切换；不运行 gray 或 shadow。`deprecated` 只作为同一 stack 中待删除
标记，最终 PR 前必须清零。

### PR 0 · 设计（本 PR）

- 记录现状、目标、owner、外部消费者和验收。
- 决定 `REACT` 与 no-revert 规则。
- 不改 runtime，不写正式 workspace，不改外部插件，不部署。

### PR 1 · 回退规则

- 删除 `child_checked`、child completion Gate 和 parent status Gate。
- 保留 exact candidate binding、结构 readiness 和发布事务。
- 崩溃矩阵按是否存在 revert 事实恢复。

### PR 2 · 服务边界

- 盘点 `AgentLoopDeps` 每项消费者。
- 复用或公开最窄 Session、Tool、Model 和 delivery Service；把现有 event（包括 `PROMPT_RENDER_EVENT`）移到稳定公开入口。
- 升级现有跨仓 contract checker。
- 当前生产链改用这些 Service，行为不变；不新增 `REACT` wrapper。

### PR 3 · 工具和提示

- Tool search 只通过 Turn scope 生效，删除 Reasoner 的名字和文本解析分支。
- 人格、host 提示和 Skill catalog 迁入普通 prompt 插件。
- 删除 `_activeSkillNames` 等零 consumer 状态。
- 每项切换后删除原 owner，不保留双写。

### PR 4 · Agent Loop

- 默认 `agent-loop` 通过普通 loader provide `REACT`。
- `ConversationRuntime` 在 exact snapshot 中 resolve 并调用它。
- 直接删除 bootstrap 全局 `AgentLoop` 和旧被动链；不先包一层 legacy adapter。
- Scheduler、Subagent、Wake 和 passive 全部证明走同一 Service。

### PR 5 · 外部插件

- 从 production artifact commits 迁移 9 个可达外部插件。
- 删除散落的私有 Core import 入口和 legacy host 分支。
- 升级后的共享 checker 在所有相关插件 CI 通过。

### PR 6 · 最终删除

- 删除所有 deprecated 标记、兼容壳、旧测试和死文档入口。
- 全仓、正式 artifact 和 cache 扫描证明零私有 consumer。
- 完成最终累计 diff、能力等价和恢复演练后停止；部署另行批准。

## 9. 等价验收

不运行双生产链。每个实施 PR 在隔离 fixture 中分别运行 base commit 与 candidate commit，比较：

- exact input Message、Turn/Session identity；
- system prompt blocks 和顺序；
- 每 step 可见 Tool、调用参数、结果与 grant；
- provider request、usage、重试和错误分类；
- SessionDB write set 与只追加约束；
- lifecycle / typed event 顺序和 payload；
- attachment、memory、compaction 和 request projection；
- outbound、delivery receipt、ACK、取消和 shutdown；
- snapshot/generation identity、Effect cleanup 和旧 lease 排空。

只允许登记 UUID、时间、端口和进程号等非语义差异。代表场景至少包含 passive、同 Turn 输入、
Tool search、Skill、MCP、附件、provider retry、取消、Scheduler、Subagent、Wake、candidate child、
plugin-revert、无 revert 晋升和进程崩溃。

## 10. 明确不做

- 不创建七个 phase 插件。
- 不创建第二套 Turn、Session、Context、scope、executor 或 publication 模型。
- 不让普通插件直接写 SQL、切 stable、结算 terminal 或伪造 delivery success。
- 不用 shadow、gray、runtime flag 或长期 compatibility shell。
- 不直接修改 plugin cache，不把设计 PR 部署到 hua-home。
- 不因为某个接入点已有消费者就永久保留不干净的接口。

## 11. 风险与停止条件

| 风险 | 处理 |
|---|---|
| 同 Turn 混用两代 Service | capability 消费时 acquire exact snapshot ID；任何 selector fallback 或 live registry 回读都阻塞迁移 |
| 提交后崩溃导致重复发送 | 先证明 Session、delivery、ACK 的现有 owner 和幂等边界，不在插件复制 |
| 公共 Service 变成 God object | 每个接口只保留当前真实调用；无 consumer 不添加 |
| 外部源码身份错位 | 只从 production artifact commit 建迁移分支 |
| base/candidate 差分缺口 | 缺少 write set、event 或外部调用证据时不删除旧 owner |
| 最终仍有私有入口 | Core、外部源码、正式 artifact、cache 四层零命中才完成 |

如果实现发现 `REACT` 之外还必须新增控制原语，先停止并回到本规格说明哪个不可替代事实没有 owner；
不能在实施 PR 中顺手增加。
