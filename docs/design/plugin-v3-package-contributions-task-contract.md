# 插件 v3 包级 contribution 任务合同

> UI 部分已由 ADR0071 和 [V3 能力手册](plugin-v3-capabilities.md#5-dashboard-与-web) 的普通
> provider 注册替代。下文包级 UI 常量只记录旧迁移背景，不是当前接入合同。


- 状态：implemented candidate
- 日期：2026-08-15
- 关联条款：PLG-001、PLG-004、PLG-009、PLG-014
- 上游：[插件 v3 generation loader 任务合同](plugin-v3-loader-task-contract.md)

## 1. 目标

让 v3 namespace 插件以无副作用的模块常量声明安装期必须读取的包级能力：`skill_roots`、`drift_skill_roots` 与 `dashboard_module`。Loader 在插件 `apply()` 前冻结并校验这些声明，复用现有 generation contribution、Skill catalog 和 Dashboard snapshot 发布链。后续[被动回复组合接入点合同](plugin-v3-passive-response-seams-task-contract.md)在同一冻结 namespace 上增加 `workspace_roots`，但其 candidate 复制与运行路径由该后续合同拥有。

这不是新的固定 runtime lifecycle 表。Tool、Prompt、Turn、Job、Channel、MCP 和外部效果仍通过 Context、Service、typed event 与 Effect 组合；本 PR 不为它们增加 namespace 特判。

## 2. Change intent

```yaml
change_type: feature
semantic_delta: compatible
capability_owner: core
consumer_scope:
  - composition plugin api v3
runtime_patch: required
runtime_patch_reason: "Skill source 和 Dashboard module 必须在 apply 前完成路径校验，并进入 generation 的原子 catalog/snapshot 发布。"
authoritative_state_owner: "插件包拥有源文件；Core generation 拥有已校验路径、catalog 与 snapshot publication。"
client_only_alternative: "不适用；客户端无法拥有服务端插件包路径。"
```

## 3. 合同与边界

```text
plugin.py namespace constants
          │ import / shape validation
          ▼
  ComposablePlugin frozen declaration
          │ plugin-root path containment
          ▼
  PluginContributions
     ├─ Skill catalog / links
     └─ Dashboard generation binding
          │
          ▼
  stable/latest snapshot publication
```

- 缺省值是空 roots 与无 Dashboard；现有 v3 插件行为不变。
- roots 只接受非空字符串序列；namespace 拒绝重复字符串，Manager 再拒绝解析后指向同一路径的别名；Dashboard 只接受非空字符串或 `None`。
- Manager 继续集中执行路径 containment、目录/文件存在性和 `.py` 校验；声明无效时在发布前 fail-loud。
- 声明只读取插件包，不写 workspace。Skill 投影与 Dashboard binding 沿用现有 generation owner、candidate isolation、snapshot lease 和逆序清理。
- candidate Dashboard 只取得 `generation.validation_workspace`，其 module、route 与 closeable 归 candidate child scope；discard 会连同 validation root 清理，promotion/formal rebuild 会先关闭 candidate binding，再用正式 workspace 建立新 binding。candidate binding 不得被缓存或复制进 stable snapshot。
- 不增加 `jobs()`、phase module、ToolHook 或其他 v2 contribution adapter。

## 4. 验证与回滚

- 真实 `PluginManager.load_all()` 发布 v3 roots 与 Dashboard path，并在 `active_plugins` 与 generation contribution 中保持同一绝对路径。
- 非序列、空值、重复值和空 Dashboard 声明在 namespace 边界失败。
- 解析后重复或越界的 root 不进入 generation；candidate Dashboard 写入只出现在 validation workspace，discard 后零残留，promotion 后才出现正式 binding。
- 相关 loader、Skill、Dashboard、candidate、snapshot 与 composition 回归通过；Pyright、compileall、change-impact Gate 通过。
- 回滚点：`backup/plugin-v3-static-contributions-before-20260815`。
