# 0037 · 插件 Service 表达能力而不是插件类别

- 状态：accepted
- 日期：2026-08-14
- 关联条款：PLG-001～PLG-014、GOV-001～GOV-005、TST-001～TST-007
- supersedes：[0036](0036-plugin-composition-keeps-promotion-owner.md) 中“旧 Job、Channel、MCP 等类别各自成为领域 Service”的映射与先建基建再补验收的顺序
- superseded by：无

## 背景

[0036](0036-plugin-composition-keeps-promotion-owner.md)已经确定使用 Context、Service、Inject、Fiber、Effect 和 typed event 替换固定插件能力总表，同时保留 Akashic 的候选隔离、自验证、generation lease 与晋升 owner。后续设计仍把 Job、Channel、MCP、Proactive 等旧 `PluginManager` 类别逐项写成待建 Service。这会把旧总表换成同名 Service 总表，插件仍只能沿 Core 预先划定的类别扩展。

DeepSeek Harness 没有这样映射。它的 `jobs` 只承载可观察、可取消、可等待并向模型报告完成的长任务协议；普通周期工作使用随 Fiber 释放的 timer。Session reminder 不公开 Schedule Service，而是组合 Session event、timer owner 与普通 agent follow-up。MCP client 本身是插件，它注入 tool registry，并把外部工具注册成普通 Tool；MCP 不是 Core Service。

## 决定

组合内核只提供挂载、依赖、事件和资源生命周期。公开 Service 必须表达一项稳定能力，不能只表达“这是一类插件”。新增 Service 前至少证明以下一项：

1. 它独占一项运行状态、不变量、名称排他或提交协议；
2. 它有可替换 provider 与不依赖具体实现的 consumer；
3. consumer 必须直接调用这项能力，typed event、Effect、Fiber-owned task 或已有 Service 无法准确承载。

旧 `jobs()`、`channels()`、`mcp_servers()` 和 `proactive_sources()` 不自动产生 `JobService`、`ChannelService`、`McpService` 或 `ProactiveService`。迁移按最小能力组合：

```text
┌──────────────── composition plane ────────────────┐
│ Context / Inject / typed event / Fiber / Effect   │
└───────────────────────┬────────────────────────────┘
                        ▼
┌──────────────── proven capability seams ──────────┐
│ Timer / Tools / Skills / Agent Input / Delivery   │
│ UI Slots / 其他由真实 owner 与 consumer 证明的能力 │
└───────────────────────┬────────────────────────────┘
                        ▼
 MCP bridge / Channel adapter / Watcher / Proactive policy
```

- GitHub Watcher 组合 Timer、外部客户端和 Agent Input；普通周期工作不是 Job。
- MCP bridge 组合进程或 HTTP、凭据、Timer 与 Tools；它不发布 MCP Service。
- Channel adapter 从 Agent Input 接收入口，向 Delivery 注册发送实现，并用 Effect 拥有连接。
- Proactive 插件监听 typed event，组合 Timer、Agent Input、Delivery 与自己的持久状态；“主动”是策略，不是 Core Service。
- 只有未来出现可列出、可取消、可等待并有独立 owner fence 的长任务协议时，才按该协议决定是否增加 Jobs 能力。

事件系统属于 Context，不增加 `LifecycleEvents` Service。每个公开事件由拥有该阶段的模块声明，事件合同固定名称、payload、dispatch mode、scope、精确发生位置和失败语义。内部 phase slot DAG 保持 Core 私有；只在真实插件需要的稳定边界发布 typed event，不把所有内部 slot 变成插件 API。

Core 可以在内部把 Skill、UI 等注册汇入同一个 candidate generation 收集器，但插件公开面使用按能力命名的注册表。当前 `PLUGIN_ASSETS` 只作为过渡实现，不再扩展为通用插件 SDK；继续迁移前要由 `Skills`、`UI Slots` 等窄能力面取代其公开用法。

测试先于新的能力 seam。先建立独立 conformance testkit，覆盖真实加载、依赖波动、scope、reload/dispose、generation 回执和 mutant；之后每项能力基建必须同时提供自身 invariant、真实装配测试和第一个实验 consumer。实验 fixture 不算正式插件迁移。

## 理由

Service 数量少不是目标，职责可证明才是目标。Tool registry、Delivery 或 Memory provider 仍然可以是领域 Service，因为它们拥有目录、路由、提交或互斥 provider 等独立事实。MCP、Channel、Watcher 与 Proactive 则首先是这些能力的组合方式；提前为它们建 Service 会制造第二套固定插件分类。

先建立 conformance testkit 可以让后续抽象由失败场景驱动。它也把“插件能加载”与“依赖、资源、顺序、持久写集和用户结果等价”分开，避免迁移完成后再进行第二次公共 API 重构。

## 影响

- 现有 Context、ServiceKey、Inject、Fiber、Effect、typed event、generation 与晋升实现继续有效。
- 当前 Citation/Meme 候选不获得发布批准；先补完整回执与 mutant，并把过渡能力面收敛后再验收。
- 后续基建不按旧插件基类方法列表批量建设；Timer、Tools、Agent Input 等能力各自使用独立小 PR。
- v2 legacy host 在全部消费者完成等价迁移前保持不变，不增加长期适配器。
- 本决定不修改正式 workspace、plugin-data、SessionDB、渠道、MCP 进程或远程 API。

## 验收

- [ ] Conformance testkit 能杀死依赖、顺序和 disposer mutant。
- [ ] 新能力 seam 都能指出权威 owner、独立 invariant 与第一个 consumer。
- [ ] GitHub Watcher 不依赖 Job 类别接口即可随 generation 启停且无 task 泄漏。
- [ ] MCP 插件只通过窄宿主能力注册普通 Tool，并在 dispose/reload 后资源归零。
- [ ] Channel 与 Proactive 迁移不要求 Core 恢复同名固定贡献表。
