# 插件元能力底座与测试先行任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-14
- 目标分支：`codex/plugin-v3-pilot-reconciliation` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-meta-contract-before-20260814@e28674de6fb32281c7684bf66d3b861770938989`
- 上游：[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[Cordis 插件迁移能力等价验收](cordis-plugin-capability-parity.md)

## Goal

先建立能够独立发现依赖、顺序、资源和行为漂移的插件 conformance testkit，再按真实能力逐项建设组合基建，最后把现有插件从 v2 legacy host 等价转译到 v3；不把旧 Job、Channel、MCP、Proactive 类别翻译成同名 Service。

## Success criteria

- Conformance testkit 覆盖真实 namespace 加载、inject 等待与重激活、event mode、scope、reload/dispose、generation 回执和故意错误 mutant。
- 每项新增 Service 都能列出独立 owner、invariant、consumer 与已有能力不能承载的理由。
- Timer、Tools、Agent Input、Skills、UI Slots 与 Delivery 以独立小 PR 建设；没有 consumer 的 seam 不进入 Core。
- 每个正式插件迁移使用相同输入运行 v2/v3，并比较 catalog、Prompt、事件、持久 write set、外部效果、清理和用户结果。
- 最后一个消费者迁移并通过累计 Gate 前，v2 legacy host 与正式 runtime 保持不变。

## Change intent

```yaml
change_type: migration
semantic_delta: none
capability_owner: mixed
consumer_scope:
  - core plugin composition and publication
  - installed Akashic plugins
runtime_patch: required
runtime_patch_reason: "PLG-001～PLG-014 要求候选隔离、可逆组合、依赖 fail-loud 与 Core-owned promotion；插件侧无法独自提供 generation lease 和原子晋升。"
authoritative_state_owner: "Core owns artifact/generation/promotion; each plugin owns its domain state and external behavior."
client_only_alternative: "不适用；这是服务端插件 runtime，插件内适配无法替换 Core publication owner。"
invariants:
  - PLG-001～PLG-014
  - GOV-001～GOV-005
  - TST-001～TST-007
protected_state:
  - formal workspace and plugin-data
  - sessions, memory, schedule, proactive and delivery state
  - current stable plugin generation
allowed_paths:
  - agent/plugin_composition/**
  - agent/plugins/**
  - owning lifecycle and capability modules required by one approved seam
  - tests/**
  - tests_scenarios/contracts/**
  - docs/**
forbidden_paths:
  - installed plugin cache
  - formal Akashic workspace
  - unrelated runtime and client modules
allowed_effects:
  - isolated temporary workspaces and plugin homes
  - loopback test processes and sockets with verified cleanup
  - Draft branches and pull requests
forbidden_effects:
  - production deployment or runtime switch
  - formal manifest, plugin-data or database writes
  - real channel, GitHub, browser account or external API writes
validation:
  - targeted pytest for each adjacent PR
  - ruff and pyright for changed Python surfaces
  - change-impact Gate bound to the exact source
  - old/new capability receipt comparison for migrated plugins
  - at least one killed mutant per P0 oracle
rollback: "Keep the v2 legacy host and previous stable generation; revert the adjacent PR or return to the named backup ref."
```

## PR boundaries

```text
PR-A  decision/design correction only
PR-B  reusable conformance testkit and mutants
PR-C  typed event catalog and exact stage contracts
PR-D  lifecycle-owned Timer
PR-E  Tools registry and execution events
PR-F  Agent Input
PR-G  Skills and UI Slots public seams
PR-H  Delivery, only with the first Channel consumer
```

每张基建 PR 包含一个实验 fixture，但不迁移正式插件。只有依赖的能力 seam 与 conformance oracle 全部通过后，才开始正式插件迁移。

## Persistence and side effects

本阶段不增加也不减少权威运行数据。测试只使用临时目录和 loopback 资源；清理证据必须证明没有残留 task、listener、进程、socket、注册项或临时 workspace。generation 指针回滚不宣称撤销文件、数据库、消息或远程效果；正式插件迁移继续保留各领域已有的提交与恢复协议。

## Stop rules

- 如果 Service 只能用旧 `Plugin` 方法名证明必要性，停止实现并退回设计。
- 如果插件迁移需要尚未批准的行为、schema、write set 或错误语义变化，停止迁移。
- 如果 mutant 不能和正确候选运行同一 fixture，停止验收并修复 oracle。
- 如果测试需要真实用户账号或正式 workspace 效果且没有单独授权，停止测试。
