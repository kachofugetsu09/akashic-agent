# 插件 Agent Input 组合能力任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-14
- 目标分支：`codex/plugin-tools` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-agent-input-before-20260814@9365a6ea`
- 上游：[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件元能力底座与测试先行合同](plugin-meta-capability-task-contract.md)、[Codex 式同 Turn 输入设计](codex-style-same-turn-input.md)
- 参考实现：`/mnt/data/source-code/deepseek-harness@47f943859bef60e4160492346772ded9b24f765a`

## Goal

给 v3 插件提供一个创建持久 Session、向既有 Session 准入普通 Turn 的窄入口。Core 继续独占 Session/Turn identity、busy admission、持久化、执行 snapshot 与候选晋升；插件只决定何时产生领域输入及其内容，不接触 `MessageBus`、`SessionManager`、`ConversationRuntime` 或完整 `ControlService`。

```text
┌──────────────────┐  inject core.agent_input  ┌────────────────────┐
│ stable v3 Fiber  │ ─────────────────────────▶ │ AgentInputService  │
│ owns domain wake │                            │ validates boundary │
└────────┬─────────┘                            └─────────┬──────────┘
         │ create Session / submit input                   │ stable Root lease
         ▼                                                 ▼
┌──────────────────┐                            ┌────────────────────┐
│ plugin domain DB │                            │ ControlService     │
│ owns dedupe/retry│                            │ Session + Turn     │
└──────────────────┘                            └─────────┬──────────┘
                                                       │ ordinary start
                                                       ▼
                                             ┌────────────────────┐
                                             │ ConversationRuntime │
                                             │ busy or admitted    │
                                             └────────────────────┘
```

## Ownership and invariants

- `AgentInputService` 只做插件边界校验和 Core admission，不拥有消息队列、重试、Delivery 或领域幂等状态。
- 创建 Session 与提交 Turn 是两个独立提交点；创建成功后即使后续提交失败，Core 不伪装成已回滚。插件必须先保存 Session identity，再按自己的账本提交输入。
- Session 与 Turn identity 只由 Core 产生；插件身份来自 `ctx.runtime.plugin_id`，并由 Core 写入保留 metadata，插件不能伪造。
- 输入只接受普通 `turn/start`；active Session 原样暴露 busy，不排队、不转成 steer、follow-up 或 next-step injection。
- Service 不返回模型结果、不提供 interrupt/delete/list，也不承担输出发送；后续消费者按 typed event、自己的领域账本或独立 Delivery 能力观察完成。
- 每次调用必须来自取得该 Service 的 active Fiber；跨 Root、LOADING、UNLOADING 或 disposed Context 立即失败。
- 只有当前 stable 且开放 lease admission 的 composition Root 可以产生输入。latest candidate、retired Root 或暂停中的 Root 会被 Core 拒绝；即使插件捕获异常，拒绝尝试仍进入 composition 验证回执。
- Core 在调用后端期间持有该 stable snapshot 的 lease；成功只表示 Session 已创建或 Turn 已准入，不表示模型完成或外部效果已提交。
- metadata 在插件边界完成 lossless JSON 拷贝；Core 保留字段由 adapter 再次拒绝并写入真实插件 owner。

## Public seam

- `AGENT_INPUT = ServiceKey("core.agent_input")`
- `AgentInputService.create_session(ctx, *, metadata=None) -> AgentSession`
- `AgentInputService.submit(ctx, session_id, content, *, metadata=None) -> AgentInputReceipt`
- `AgentSession.id` 是已持久化 Session identity。
- `AgentInputReceipt(session_id, turn_id)` 只证明普通 Turn 已通过准入并取得 identity。
- `create_session.metadata` 写入 Session metadata；`submit.metadata` 只投影到该输入的 `inboundMetadata`，插件不能借此设置 `runtime`、`channel` 或其他控制面字段。

第一版不提供 `send` 模式参数、队列、结果等待、取消、Session 查询/删除、active-step injection 或 Channel envelope。GitHub Watcher 后续组合 Timer、外部客户端、自己的 durable ledger 与本能力；本 PR 只包含隔离实验 consumer，不修改其 canonical source。

## Selective DSH translation

吸收 DSH `AgentRegistry` 的窄 service port、明确 Session identity、调用方拥有领域生命周期与 Core 拥有实际 agent factory 的思想；不转译 `Agent.followup()`、`steer()`、`inject()`、live inbox 或 AgentHandle 自动删除。Akashic 的 SES-007、RUN-008 与 OUT-005 已确定 active attempt 只接受精确中断，普通输入必须在 idle 时创建下一 attempt。

## Persistence and effects

```yaml
change_type: additive
semantic_delta: none for existing plugins
capability_owner: "Core owns Session and Turn admission; plugin owns domain wake, dedupe and retry."
consumer_scope:
  - v3 composition plugins
protected_state:
  - formal workspace, sessions and plugin-data
  - stable and latest plugin generations
allowed_effects:
  - temporary SessionStore and deterministic fake backend in tests
  - isolated namespace plugin and composition receipts
forbidden_effects:
  - formal plugin migration or promotion
  - real model, channel, GitHub or external API calls
rollback: "Revert this adjacent PR or return to backup/plugin-agent-input-before-20260814; v2 jobs and control protocol remain unchanged."
```

## Verification

- service unit tests覆盖 JSON 拷贝、identity/content bounds、跨 Root 与非 active Context 拒绝；
- real namespace fixture 在 Fiber active 后创建 Session 并准入 Turn，证明实验 consumer 只取得窄能力；
- manager fixture 证明未绑定 Core backend 时 fail-loud，绑定后 stable Root 成功；
- candidate/retired Root 即使捕获拒绝，也在 receipt 中留下 external-effect evidence；
- `switch_ready` 在最终晋升提交前重新读取 latest Root receipt，拒绝已被异步行为污染的候选；
- busy、缺失 Session 与保留 metadata 错误从现有 Control owner 原样传播；
- disposer mutant 与相同 fixture 比较，证明 Context owner oracle 能发现失效 Fiber 仍可提交的错误实现；
- public plugin generation Gate 绑定以上结果与 exact source digest。
