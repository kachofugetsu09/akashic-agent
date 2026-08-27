# 插件 candidate Root 隔离任务合同（R2b）

- 状态：superseded by [0046](../decisions/0046-plugin-candidate-validation-is-incremental.md)
- 日期：2026-08-15
- 实现基线：`4eac51991cc7c8f6eff2f196b787a4172ee234b4`
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[R2a stable 原子组装](plugin-stable-atomic-assembly-task-contract.md)

## 1. 目标与边界

> 2026-08-27 勘误：本文“重建全部 stable v3 participants”的实现会重新启动并复制无关 stateful 插件，违反 PLG-001。保留本文作为历史验收记录；当前合同以 0046 为准。

任何 validation/latest candidate 都不得复用 stable `CompositionRoot`，包括只修改 v2 插件、v3 generation 集合表面未变的 candidate。candidate snapshot 可以复用不可变源码与配置事实，但不能复用 stable Root、Fiber、Effect、v3 module instance、可写 data root 或 workspace。

```text
stable snapshot                           candidate snapshot
┌──────────────────────┐                 ┌──────────────────────────┐
│ stable Root          │                 │ candidate Root (new)     │
│ stable v3 instances  │                 │ changed v3 clone         │
│ formal workspace/data│                 │ unchanged v3 clones      │
└──────────┬───────────┘                 │ attempt workspace/data   │
           │ stable leases               └────────────┬─────────────┘
           │                                          │ latest leases
           ▼                                          ▼
     独立 drain/dispose                         独立 drain/dispose
```

`semantic_delta: compatible`：候选看到同一插件声明、Service 拓扑、配置和初始 plugin-data；允许变化是它们运行在 candidate-owned module/runtime/path 中，stable module 全局状态和正式数据不再被候选 `apply` 污染。

## 2. 隔离 owner

- 每次 candidate snapshot compile 创建新的 Root object；`_resolve_composition_root()` 的 stable Root 复用分支只允许 `candidate_owner is None`。
- candidate generation 拥有 `validation_workspace`，其 Scope 在 discard、失败或 promotion restore 后删除整个 validation root。
- 每次 composition rebuild 在 validation root 下创建独立 attempt workspace，避免 prepared Root 与 publish-rebase Root 共用可变数据。
- 所有 v3 参与者从各自 canonical `plugin.py` 重新 import；配置从 attempt plugin-data 重新按 clone `ConfigModel` 校验。
- clone module、Fiber、Effect 和 attempt data 归 candidate Root；Root dispose 时卸载 module tree 并删除 attempt root。
- stable generation、stable Root、stable module instance、正式 workspace/plugin-data 始终不归 candidate cleanup。

Core 提供隔离路径和生命周期 owner，不解释插件领域数据，也不预建通用 HTTP client。Python 代码仍不是安全沙箱；配置内绝对路径、import-time I/O 或插件自行访问全局对象必须由插件合同、review 与跨仓 Gate 约束。

## 3. promotion 与 identity

- validation 阶段的 clone 只承载 composition apply；generation 自己继续拥有 static/readiness Gate、catalog 和 publication identity。
- 所有 candidate Gate 和 post-publish invariant 先在隔离 Root 完成；随后排空 candidate lease、dispose candidate Root，再用正式 generation/runtime 重建 production Root。
- production rebuild 一旦开始，后续 projection、pointer 或 owner commit 失败都终止并销毁该 candidate，不允许把已经 formalized 的资源恢复成可租用 latest。
- candidate 与 production 的逻辑 topology identity 必须相同；Root object、module path、workspace 和 data path 必须不同。
- v2-only candidate 验证完成后，candidate Root drain/dispose；stable snapshot 继续持有原 stable Root，不把 candidate Health/Incident 或 Effect 带回 stable。
- R3 将进一步把 mutable state 从 topology identity 移除，并用 composition revision 封存验证后的结构变化；本 PR 不预建该模型。

## 4. 验证与停止条件

- installed v3 candidate 的 Root 与 stable Root 不同，candidate Fiber runtime 位于 `plugin-validation`；promotion 后 clone module 和 attempt root 已删除，production apply 使用正式 workspace。
- v2-only candidate 也重建全部 stable v3 participants；stable v3 module global observation 不增加，candidate data write 只出现在 attempt data root。
- candidate rebuild 可以产生多个 attempt，但旧 Root dispose 后旧 clone module/data 必须为零。
- Fiber 挂载中途取消也必须回收已完成的 clone module、registry instance、Effect 和 attempt data，不得触碰 stable Root。
- discard candidate 后 stable snapshot/Root identity 与 object 都保持不变。
- direct candidate invariant 失败不得执行 formal apply；下一次 fresh attempt 可以独立成功。
- installed promotion 在 Skill projection 或 owner commit 失败后，latest、production Root 和正式候选 owner 必须全部清除，stable pointer/Root/data 保持原值。
- v2-only candidate promotion 产生新 stable snapshot，但继续复用未变化的旧 stable Root；candidate clone module 与 attempt data 不进入 stable。
- targeted：`tests/test_plugin_composition_loader.py`、`tests/test_plugin_hot_reload.py`。
- cumulative：manager/runtime control/composition kernel 与公开 change Gate。
- 本地证据：composition loader/kernel/hot reload `188 passed`；manager/runtime control/reload journal/turn rollout/skill links/source/install 与 composition events/executor/lifecycle/experiment `159 passed`；Basedpyright `0 errors`，`git diff --check` 通过。
- Terra xhigh 只读复审无 P0；其 promotion 失败、partial mount cancellation 和 registry cleanup findings 已转成上述 oracle。
- 停止条件：candidate 与 stable Root 是同一对象、clone 读取/写入正式 data root、clone module 在 Root drain 后残留、promotion 复用 candidate Effect、candidate/production topology identity 不等。
- 回滚点：Git tag `backup/plugin-candidate-root-isolation-r2b-before-20260815`。

## 5. v2 清理关联

R2b 不新增 v2 API。v2-only candidate oracle 只是证明兼容期不会借 stable v3 Root 执行候选 Turn；当 v2 插件全部迁移后，该场景测试、`Plugin` class candidate 分支和 legacy contribution 编译路径一起进入最终物理删除 Gate。
