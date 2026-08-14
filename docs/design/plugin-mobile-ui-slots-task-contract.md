# 插件 Mobile UI 组合能力任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-15
- 目标分支：`codex/plugin-mobile-ui-binding` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-mobile-ui-slots-api-before-20260815@6214af1c`
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件 UI Slots 组合能力合同](plugin-ui-slots-service-task-contract.md)

## Goal

让 v3 插件通过 `core.ui_slots` 登记自己的 Mobile UI 静态资产、动态可用性与只读 query handler。插件继续拥有业务 projection；Core 继续拥有资产边界、catalog、generation lease、有界执行、timeout 和协议错误映射。现有 v2 Mobile UI API 与客户端协议保持不变。

```text
┌────────────────────┐  register_mobile  ┌────────────────────┐
│ v3 plugin Fiber    │ ────────────────▶ │ core.ui_slots      │
│ asset + projection │                   │ validate + collect │
└─────────┬──────────┘                   └─────────┬──────────┘
          │ Effect owns declaration                │ freeze
          ▼                                         ▼
┌────────────────────┐                   ┌────────────────────┐
│ generation contrib │ ◀──────────────── │ committed Root     │
│ handler + content  │                   │ promotion owner    │
└─────────┬──────────┘                   └────────────────────┘
          │ snapshot lease + bounded worker
          ▼
┌────────────────────┐
│ existing catalog / │
│ asset / query host │
└────────────────────┘
```

## Ownership and public seam

- 插件 canonical source 拥有 JS/CSS、导航 metadata、slot 选择、`available` 判断与 query projection。
- `PluginUiSlots` 只在调用 Fiber 的 `plugin_dir` 内解析资产，固化内容和摘要，并把登记绑定为该 Fiber 的 Effect。重复登记、绝对路径、symlink 越界、无效 slot、超限 metadata 或总资产超过 240 KiB 均 fail-loud。
- `PluginMobileUiProvider` 继续拥有当前 snapshot 选择、plugin revision、catalog、asset digest、线程池容量、20 秒 timeout、192 KiB query 结果上限和错误分类。query 超时或调用方取消后，generation lease 持有到工作线程真实退出。
- `MobileUiRpcInvalidRequest` 属于能力边界，由组合 API 公开；旧 `agent.plugins.mobile_ui` 导入路径继续指向同一异常类型。
- v2 与 v3 都编译成同一个 generation contribution。v2 class method 仍由 legacy host 收集；v3 通过显式登记提供 handler，provider 不再依赖把 generation instance 强制转换成旧 `Plugin`。
- Cordis 风格只用于 Root/Fiber 内的显式 service 注册和 Effect 回收。Mobile UI 不是新的 `MobilePluginManager` 类别，也不把 Android、WebView 或某个业务 DTO 放进 Core。

```python
from agent.plugin_composition import (
    UI_SLOTS,
    MobileUiDefinition,
    MobileUiNavigation,
)

inject = (UI_SLOTS,)

async def apply(ctx, config):
    await ctx.require(UI_SLOTS).register_mobile(
        ctx,
        MobileUiDefinition(
            module="mobile_panel.js",
            stylesheet="mobile_panel.css",
            navigation=MobileUiNavigation(
                label="Status",
                description="查看当前会话的只读状态",
            ),
            slots=("drawer.panel",),
        ),
        query=build_query(ctx),
    )
```

query handler 是同步只读函数，由 Core 在线程池中异步等待。它不是 typed event，也不新增 waterfall、listener DAG 或第二套调度语义。

## Persistence and effects

```yaml
change_type: feature
semantic_delta: compatible
capability_owner: "Plugins own UI source and projections; Core owns publication and query execution boundaries."
consumer_scope:
  - v3 plugins with Mobile UI assets and read-only queries
  - first planned consumer: status_commands
runtime_patch: required
runtime_patch_reason: "Generation selection, revision identity, leases, admission and protocol limits are Core-owned cross-plugin invariants."
authoritative_state_owner: "Existing plugin generation snapshot and each plugin's own business store remain authoritative."
client_only_alternative: "A client cannot validate plugin source paths, select the committed generation or hold its server-side lease."
protected_state:
  - existing Mobile UI catalog, asset and query protocol
  - v2 plugin behavior and stable/latest generation pointers
  - formal workspace, plugin-data and SessionDB
allowed_effects:
  - Root-local UI registrations and exact Effect cleanup
  - bounded worker execution under the existing snapshot lease
  - temporary plugin roots and workspaces in tests
forbidden_effects:
  - formal plugin migration or runtime switch
  - canonical plugin source, installed cache or formal workspace mutation
  - Session, plugin-data, channel or external API writes
rollback: "Revert this adjacent PR or return to backup/plugin-mobile-ui-slots-api-before-20260815; the binding PR, v2 provider and previous stable generation remain usable."
```

本能力不增加、更新、逻辑失效或物理减少任何权威持久记录。静态内容只存在于 generation 内存贡献与既有传输响应中；Root dispose 后登记归零。

## Verification

- 真实 v3 namespace loader 经 `core.ui_slots` 生成 catalog、内容寻址 asset 和 query 结果，并验证动态 `available`；
- v2 Mobile UI catalog、worker offload、queue bound、timeout lease、错误隔离和结果大小回归保持不变；
- symlink escape、重复 Mobile UI 和 freeze 后登记均拒绝；
- leaked-disposer mutant 与正确实现运行同一 fixture，证明 Fiber dispose 能发现残留登记；
- public change-impact Gate 绑定 exact source digest，并回归 Mobile WebUI、plugin generation、热重载和 publication plane。
