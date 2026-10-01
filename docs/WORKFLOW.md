# Akashic Agent 开发工作流

流程只保留理解边界、实现验证、交付三步。长期语义见 [projectneed.md](projectneed.md)，理由见 [0083](decisions/0083-short-workflow-preserves-design-intent.md)。

```text
┌──────────────────────────┐
│ 读设计原则和相关决定      │  说明本次可以改变什么
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 修改并验证实际结果        │  同时核对受保护状态
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 提交小而清楚的 PR         │  由维护者评审
└──────────────────────────┘
```

## 1. 理解本次边界

先按 [INDEX.md](INDEX.md) 读取设计原则、相关条款、有效决策与真实实现。普通任务用一段说明写目标和完成标准。涉及持久数据、owner、生命周期、公共接口或外部效果时，在动手前补充：**改变什么、保持什么、依据哪条决定、怎样验证**。不必填写 YAML 或另建任务文档；复杂任务可用 [简短合同](templates/agent-task-contract.md) 交接。

例如：依据 ADR-0002，本次只缩小发给模型的历史窗口；既有消息正文、身份和顺序保持不变，上下文路径不获得消息 writer。验证窗口缩小后重开会话，原历史仍可读取。只写“语义不变”或“已读文档”不足以说明这个边界。

与现行合同一致且已获授权就继续。需要改变数据保留、权限、公共行为或现行决定时，先指出当前与拟议语义、理由及影响，由维护者决定；不能用新实现反向修改验收标准。

## 2. 修改并验证

非简单修改使用从最新目标分支建立的独立 worktree，记录基线，保护原 checkout 的未提交内容。每个 worktree 同时只有一个 writer；交接按 GOV-005 记录分支、HEAD、dirty state 和下一位 owner，旧 writer 先结束写入，不用 reset 或清理制造 clean 状态。

```bash
git fetch origin main
git worktree add -b codex/<task> ../<task> origin/main
```

需要代码索引时，为该 worktree 单独执行 `codegraph init <worktree>` 并用 `codegraph status <worktree>` 核对路径；不共享 `.codegraph/`。需要 Python 时先核对环境；依赖未变可软链接原 `.venv`，依赖变化则使用独立环境，不向共享 venv 安装分支依赖。纯文档任务无需初始化代码索引或语言环境。

按 TST-001～TST-006 **禁止编写单元测试**，也不把函数级 mock/断言换名放进 scenario 或 E2E。选择能直接观察本次行为的最小验证：真实 CLI/API、浏览器操作、真实组件集成、E2E 或隔离实例中的重放；E2E 是选项，不是所有任务的固定步骤。

验证既要证明目标达成，也要核对相关受保护结果。持久化改动从真实入口触发，比较既有记录的完整内容、身份和顺序，按风险检查写入尝试、失败和重新加载；新增合法消息不要求整个数据库文件字节不变。只返回成功、行数一致或测试全绿不能证明没有越权改写。使用一次性 workspace、plugin home、config 和 HOME，不能借验证操作正式状态。

本地只运行相关静态检查；完整既有检查由 CI 执行，不要求每次重复整套。常用入口：

| 改动范围 | 本地检查入口 |
|---|---|
| Python 类型 | `.venv/bin/pyright --level error`；SDK 用其项目范围 |
| 插件依赖边界 | `python scripts/plugin_boundary.py check --base <base>` |
| Yoyo 迁移 | `python scripts/check_yoyo_migrations.py --base <base>` |
| Control / Host Bridge 协议 | 对应 `scripts/generate_control_schema.py --check` / `scripts/generate_host_bridge_protocol.py --check` |
| 前端类型 | `npm run typecheck` |
| 文档 | 相对链接、规则一致性、代表任务阅读路径、`git diff --check` |

前端视觉验证按 [frontend/design.md](../frontend/design.md)；实际设备隔离按 TST-008。已有测试与 CI 不因本流程自动删除或降低断言；它们的结果与功能、部署、真实设备证据分开报告。没有实际运行的层次写明未验证。

源码和文档以 Git 为恢复点，保留未提交工作，不额外打包备份。正式数据按 BAK-001 判断本次实际写入及可重建性，只在必要范围内备份；部署详见 [操作手册](design/operator-deployment.md)。

## 3. 交付小而清楚的 PR

交付前检查实际完整 diff 与目标分支新变化，确认仍符合本次边界。架构与概念评审由维护者亲自完成，不设强制 agent reviewer、模型或概念 Gate。执行者仍负责实现符合已确认设计和报告真实问题。

每张 PR 默认新增不超过约 **500 行**，删除量不限。按 PR 实际 `base...head` 的新增行数计算，不用删除抵消新增，不以格式压缩规避。较大任务优先 stacked PR：每层一个完整、可独立理解和验证的变化，base 指向前一层；保留依赖顺序和栈顶累计验证。无法合理拆分的超额改动须先说明原因，由维护者决定。

[PR 模板](../.github/pull_request_template.md) 提示三件事：问题与结果、方案与理由、验证。正文面向没读过会话的评审者；bug 写触发条件和影响，方案写关键取舍，验证写实际操作与观察。小改动两三段即可，不强制标题或不适用字段。数据、兼容性、外部效果与部署风险存在时再补充受保护合同、相关决定及恢复方式；stacked PR 加前置 PR 链接。

可复用的设计选择按 [writing-rules.md](writing-rules.md) 沉淀到 decisions：为什么选、为什么不选、什么前提改变后值得重议。普通修复理由留在 PR，已有决定直接引用。只更新被改变的事实和入口，完成事项从 NOW 删除，不记录工作流水账。

用户要求只读评审时，按实际 base/head 检查相关 diff 与证据，报告具体失败路径；[评审模板](templates/review-contract.md) 可选。提交、推送、修改 GitHub、合并和部署各需对应授权，开 PR 不自动授权合并或部署。
