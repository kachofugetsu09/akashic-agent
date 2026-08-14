# 0038 · 人类命令不是模型 Tool

- 状态：accepted
- 日期：2026-08-14
- 关联条款：PLG-001～PLG-014、SES-001～SES-008、OUT-001～OUT-005、TST-001～TST-007
- 上游：[0037](0037-plugin-services-name-capabilities-not-categories.md)
- superseded by：无

## 背景

已批准的迁移设计曾把 `setup_helper`、`status_commands` 与 MCP bridge 一起排在 Tools 之后。真实代码表明这三者不是同一种能力：MCP bridge 向模型注册可调用 Tool，而 `setup_helper` 和 `status_commands` 在 `BeforeTurn` 识别人类输入的斜杠命令，直接返回结果，不建立模型调用。

DeepSeek Harness 也把两者分开。`@deepseek-ai/dsh-commands` 用 `ctx.commands.register()` 收集插件拥有的命令定义，由 UI adapter 在模型之前执行；Tool registry 不参与命令准入。它还把 `command/run` 和 `command/done` 写入 Session log，但 Akashic 当前两个插件的已确认基线是命令 short-circuit 且不写持久状态。直接复制该日志协议会改变 Session 语义。

## 决定

增加独立能力 `core.commands`：

```text
┌──────────────────┐  register Effect  ┌──────────────────┐
│ v3 plugin Fiber  │ ────────────────▶ │ core.commands    │
│ owns behavior    │                   │ owns name registry│
└──────────────────┘                   └─────────┬────────┘
                                                │ frozen snapshot
                                                ▼
                                      ┌────────────────────┐
                                      │ passive admission  │
                                      │ before Session/LLM │
                                      └────────────────────┘
```

- Core 拥有命令名称排他、不可变目录、稳定 snapshot 绑定、直达准入位置和 handler 结果边界。
- 插件拥有命令描述、兼容 alias、参数解释、领域查询与用户可见文本。
- 已知命令在 `BeforeTurn` 的 Session acquisition 之前执行；未知命令原样进入既有 lifecycle 与模型路径。
- 第一版保持 Akashic 现有大小写、`@botname` 后缀和 alias 兼容；alias 不进入发现目录。
- 第一版不写 `command/run`、`command/done` 或其他 Session 事件，也不创建缺失 Session。若以后需要可重放命令状态，必须先单独批准事件 schema、持久 owner、投影与迁移。
- 第一版只把稳定 v3 目录接到现有 Telegram 命令发现 adapter；Mobile 目录仍由它自己的 UI 合同迁移。命令执行本身保持 channel-neutral。

`core.commands` 不是旧 `before_turn_modules()` 的同名翻译。它成立是因为它独占跨插件命令 namespace 与“命中后不进入模型”的准入控制流，符合 0037 的 Service 保留条件。

## 影响

- `setup_helper` 成为第一个正式 consumer，以 v2/v3 相同输入证明输出、目录、零模型调用和零持久写入等价。旧 v2 会在命令模块前先取得内存 Session；v3 不再执行这次无业务用途的 acquisition，这项 compatible delta 必须在迁移 PR 明示。
- `status_commands` 后续组合 Commands、只读 Session 查询接入点与 Mobile UI Slots；Commands 不向插件暴露 `SessionManager` 或数据库。
- MCP bridge 仍使用 Tools；模型 Tool 与人类命令不会共享 registry 或执行事件。
- v2 legacy host 在对应插件去壳 PR 合入前保持原样，不增加 deprecated 适配层。

## 验收

- 同一 Root 的 canonical name 与 alias 冲突在候选发布前 fail-loud。
- 每个注册由 Fiber Effect 精确撤销；泄漏 disposer mutant 必须被同一 fixture 杀死。
- stable snapshot 的已知命令 short-circuit，并证明 Session、Context、Reasoner 和持久写入口均未调用。
- 无效语法和未知名称进入原有 passive turn；handler 失败按现有命令阶段错误路径显式暴露。
- teardown 后 command effect、service 与目录归零。
