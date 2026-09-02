# 0055 · Agent Loop 是普通插件

- 状态：proposed
- 日期：2026-09-02
- 关联条款：PLG-001～PLG-017、RUN-001～RUN-012、SES-001～SES-008、OUT-001～OUT-004、CTRL-003、TST-001～TST-008
- refines：[0039](0039-react-core-atoms-keep-sources-unprivileged.md)
- supersedes：0039 中“Core 自己实现 `react` 算法”的局部归属

## 背景

0039 已让 Scheduler 和 Subagent 作为普通 v3 插件调用同一条 `react`，但默认被动回复的
`AgentLoop`、`DefaultReasoner` 和 `PassiveTurnPipeline` 仍由 bootstrap 固定组装。Core 因此仍
认识 Prompt、Tool search、phase 顺序和默认回答策略。

DeepSeek Harness 在 revision `dd6322d604e00eec1ba5e0c8541159906a21094a` 中把完整
agent loop 放在普通 `@deepseek-ai/dsh-agent-loop` Service 插件。它依赖 agents、sessions、
llm、tools、systemPrompt 和 sessionProjections，但 Cordis 不为它增加特殊加载权。

Akashic 已经有 Context、Service、Inject、Fiber、Effect、exact RuntimeSnapshot、Turn admission、
Session、Tool、Model、事件和 delivery。继续为被动回复增加 Core phase 会重复这些资产。

## 决定

1. 默认 `agent-loop` 是普通 pure-v3 插件。它通过 `REACT` 提供现有 `TurnExecutor`，并拥有
   `Message → react → Message` 的完整算法、Prompt 顺序、模型与工具循环。
2. Core 在 Turn admission 后先取得并绑定 exact RuntimeSnapshot，再从该 snapshot 的 Context
   选择 `REACT`。切换 stable 只影响之后的新 Turn。
3. `REACT` 是 Core 按 ServiceKey 检查的唯一 required service。每个可运行 Root 必须恰有一个
   provider；缺失、重复或依赖不就绪时在 snapshot 发布前 fail-loud。Core 没有默认实现和 fallback。
4. `agent-loop` 只使用普通插件可注入的公开 Service。内置发行不授予私有 import、全局 registry、
   任意 SQL、PluginManager 或 bootstrap 对象。
5. Core 继续唯一拥有 Turn admission、取消和 terminal，snapshot lease，Session 与 delivery 的
   提交不变量，以及 generation publication。插件调用这些 owner 提供的窄 Service，不复制 owner。
6. 不新增 phase 插件类型、`Turn Plan`、`React Kit`、`Turn Capsule` 或另一份总 Context。
   一个普通插件拥有算法；其他插件通过现有 Service 和 typed event 贡献能力。
7. 最终删除 bootstrap 固定 `AgentLoop`、Core 的被动 phase 组装和功能名特判。最终状态不保留
   deprecated alias、兼容 wrapper 或双链路。

```text
Transport
   │ Message
   ▼
ConversationRuntime: admit → bind exact snapshot → require(REACT)
                                      │
                                      ▼
                         ordinary agent-loop plugin
                         Message → model/tools → Message
                                      │
                                      ▼
Core Services: Session commit · delivery · terminal · release
```

## 理由

Turn、snapshot 和提交是权威边界；回答方法是可替换策略。把前者留在 Core、后者交给普通插件，
每个变化轴只有一个 owner。完整算法放在一个插件内也与 DSH 一致，不会把七个执行阶段误做成七个
相互知道顺序的特权插件。

## 影响

- Scheduler、Subagent、Wake 和被动消息继续复用同一个 `REACT`，来源不进入 Core 分支。
- Tool search、人格、Skill catalog 和 host 提示必须变成普通 Tool/Prompt 能力或删除，不能随算法
  搬进新插件后继续硬编码。
- 迁移分批进行，但每批只有一条生产路径；测试使用 base/candidate 离线差分，不运行 shadow。
- 外部插件先迁到稳定公开入口，最后删除其私有 Core import。

## 验收

- [ ] 默认 agent loop 由正式 v3 loader 装载，并与外部插件使用同一 `apply(ctx, config)` 权限。
- [ ] Core 不 import 默认 agent-loop 实现，不出现 phase、tool_search、Veda 或插件 ID 分支。
- [ ] attached child 在 capability 消费时按 snapshot ID 取得 exact lease；同 Turn 不串用 stable 与 candidate Service，也不 fallback。
- [ ] 缺失或重复 `REACT` 在 Root readiness 阶段失败，没有 runtime fallback。
- [ ] `REACT` 返回 `ControlExecutionResult`；只有 `ConversationRuntime` 生成 terminal `TurnResult`。
- [ ] Session write set、事件顺序、Tool 调用、错误分类、delivery 和取消与批准基线等价。
- [ ] 旧链、legacy host 和 deprecated 标记全部删除。

## 关联设计

- [普通 Agent Loop 规格](../design/ordinary-agent-loop.md)
- [React Core、Scheduler 与 Subagent](../design/react-core-scheduler-subagent.md)
