# Review 合同模板

> 评审默认只读。发表评论、修改代码、更新工作手册或改变 PR 状态需要单独授权。

## Target

```yaml
repository: ""
pull_requests: []
stack:
  target_branch: ""
  layers: []
final_head: ""
```

## Semantic intent

```yaml
goal: ""
semantic_delta: none|compatible|breaking
capability_owner: core|protocol|mobile|plugin|mixed|not_applicable
consumer_scope: []
runtime_patch: none|required
runtime_patch_reason: ""
authoritative_state_owner: ""
client_only_alternative: ""
invariants: []
protected_state: []
```

## Evidence

- 必读需求、决策和设计：
- 每层 `base..head`：
- 最终累计 diff：
- 持久化增加、更新、逻辑失效和物理删除：
- 外部仓库、协议快照与固定 commit：
- 本地验证：
- 远端 checks：
- 未验证项：

## Review order

1. 确认 stacked PR 的目标分支、相邻 base/head 和依赖顺序。
2. 逐层检查本 PR 新增的语义、owner、write set、权限和错误路径。
3. 在最终 head 检查累计协议、数据库迁移、恢复、构建和用户可见行为。
4. 区分客户端平台实现、产品体验、中立协议和核心权威状态。
5. Findings 按严重度排序，写明文件位置、触发条件、影响和最小修复方向。

## Stop rules

- 需要改变长期语义或能力 owner 时停止并询问维护者。
- PR 描述、实现和协议真源冲突时停止，不替作者选择最方便的版本。
- 没有远端 check 时报告未验证，不把手工构建声明写成 CI 通过。
- 用户没有授权时不提交 review、不修改代码、不更新 PR 状态。
