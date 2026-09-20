# Commands provider 合同

- 状态：按 [0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md) 实施；本层只静态阅读，行为验证尚未运行。
- 范围：显式 Commands provider、SDK 协议、命令消费者与安装组合。
- 本文替代旧 C11 的 Core 注入和 Snapshot 目录编译说明；不沿用旧版“命令不创建 Session”的执行模型。

## 归属与选择

`plugins/commands` 是普通插件；只有组合显式选入才提供 `COMMANDS`。安装名可以不同，
消费者在 `inject` 声明同一服务键，不依赖 provider 的名字。缺少 provider 时依赖不满足，
Core 不补装，也不创建空实现。SDK 只公开 `Commands`、`CommandCatalog` 协议、DTO 和恢复异常；
`PluginCommands` 与 `CommandRegistry` 的真实实现、语法校验、碰撞检查、执行和恢复均在 provider。
保留服务键 `core.commands` 只为已有 Binding 的服务身份，不表示 Core 拥有实现。

```text
┌──────────────────────────────┐
│ 显式选择 Commands provider   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 贡献方 inject → register(ctx) │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ provider 监听封存事件固定目录 │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 实际 Root / Context → 目录   │
│ Binding → handler → Message  │
└──────────────────────────────┘
```

provider 校验贡献 Context 属于同一 Root 且实际选择自身。注册 owner、generation、Fiber
全部来自该 Context。命令名与 alias 共享命名空间，`stop` 继续归会话控制；未知命令仍交回普通回复。
provider 监听 `SNAPSHOT_SEALING` 封存自身目录；之后注册明确拒绝。已经发布的 Root 绑定仍不可改。
`RuntimeSnapshot` 不再保存命令目录或拼接命令 digest；宿主从目标 Root 取得实际服务，
渠道和会话消费者从实际 Context 取得服务。目录不是 Core 发布载荷中的独立权威事实。

## Binding、消息与恢复

匹配命令后，目录用 handler 的原贡献 Context 创建 Binding；通用 Binding owner 同时纳入
实际 Commands provider、handler owner 和它们的依赖闭包。普通运行或恢复都使用调用方实际选中的
Root，不修改服务绑定，也不因这次迁移复活历史 Root。保存的 handler 名称继续与原 Input 核对。

`plugins/conversation/commands.py` 保持原输入身份、来源授权、abandon 和消息追加协议：
先固定 intent，执行前后校验来源，结果以原 `command.result` 与 Output 提交。
副作用恢复只调用 handler 的 `recover`；只读命令允许按原合同重读。缺领域回执时抛出
`CommandRecoveryRequired`，保留未知结果，不自动重跑外部效果，不把恢复内存解释成效果回滚。

| 状态 | owner 与正常增加 | 更新、失效与减少 | 恢复证据 |
|---|---|---|---|
| 命令目录 | provider 持有当前 Root 的注册 | 封存后只读；退出清理临时注册 | 固定插件制品与组合 |
| 命令 intent | Conversation 的 owner state 只增加 | 不原位改写；abandon 停止后续执行，不删除 intent | 原消息库与 Binding |
| 命令结果 | Message writer 追加 Output | 不覆盖已有正文或重复提交结果；无自动减少 | 原消息库 |
| handler 效果回执 | 各 handler 的领域 owner | 沿原 owner 协议；本层不新增更新或删除权 | 各领域回执 |
| Binding 与归档 | 原 Binding/Archive owner 只增加闭包 | 不改写已有描述，不自动 GC | 原消息库和完整代码归档 |

本层不操作正式 workspace、迁移数据或修改 slash command 授权及回执格式。

## 静态交付与后续验证

默认 profile、正式安装夹具、会话和 Akasha 相关子集显式选择 Commands provider；
Akasha 和 Akashic Clients 已有 `COMMANDS` 依赖，Conversation 增加显式依赖。
已有消息回归保留身份、未知命令、效果后失败、恢复、abandon 和输入顺序检查。
新增回归覆盖不隐式提供、异名 provider、跨 Root 注册拒绝、封存后注册拒绝及 Binding 闭包。

本轮严格不运行 tests、Gate、CI、build、lint、AST 或项目 runtime；仅静态阅读及 `git diff --check`。
历史 C11 的测试数量和评审结论不作为本层证据。恢复点为 `69716690` 和
`/tmp/akasic-command-provider-before-69716690.tar`；只需恢复源码，本层没有正式数据变更。
