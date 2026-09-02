# 0056 · 没有 Revert 就晋升

- 状态：accepted / not implemented
- 日期：2026-09-02
- 关联条款：PLG-010、PLG-012、PLG-013、RUN-007、CTRL-003、ERR-001、TST-001～TST-006
- supersedes：[0026](0026-plugin-rollout-is-owned-by-the-parent-turn.md) 第 4 条中的 child/parent 成功 Gate

## 背景

0026 让 Agent 在旧 stable parent 中安装候选，并用绑定 exact candidate 的 attached child 做业务
检查。当前 Core 还要求 child 和 parent 都以 `COMPLETED` 结束，等于 Core 又判断了一次 Agent 的
检查是否合格。

维护者确认的 Agent 合同更简单：Agent 如果发现候选有问题，一定在 parent 封口前调用
`plugin-revert`；不考虑 Agent 忘记 revert 的分支。

## 决定

1. `plugin-install` 成功建立候选后，默认结果是晋升。
2. attached child 只负责让 Agent 在 exact candidate 世界中试用能力，不向 Core提交“检查通过”证明。
3. parent 封口前调用同一操作的 `plugin-revert`，候选丢弃；没有 revert，parent 封口后晋升。
4. Core 不根据 child/parent 的 `COMPLETED`、`FAILED`、`CANCELLED` 或业务输出猜测是否晋升。
5. artifact、schema、依赖、Root readiness、generation identity 和 source revision 仍由 Core 在候选
   建立时校验。这些是结构有效性，不是 Agent 的业务判断。
6. 不新增确认、超时猜测、忘记提醒或人工批准状态。

```text
install candidate
       │
       ▼
Agent uses exact candidate child
       │
       ├─ wrong ─▶ plugin-revert ─▶ discard
       │
       └─ no revert ─▶ parent seals ─▶ promote
```

## 理由

Agent 拥有“这个插件能不能完成我的工作”的业务判断；Core 只拥有候选结构和发布事务。默认晋升加
一个明确的反向动作，比多个成功 Gate 更直接，也不会把插件类别或验证方法写进 Core。

## 影响

- 现有 `child_checked`、child terminal success Gate 和 parent status Gate 是待删除实现。
- attached child 的 exact generation/source binding 仍保留，因为它保证 Agent 试用的是候选本身。
- 进程崩溃恢复必须按“是否已经记录 revert”继续同一决定，不能重新猜业务结果。

## 验收

- [ ] 没有 `plugin-revert` 时，不论 child/parent terminal status，候选都进入既有晋升事务。
- [ ] 有 `plugin-revert` 时候选不晋升，stable 不变。
- [ ] wrong generation/source 的 child 不能冒充 exact candidate，但也不成为新的 promotion Gate。
- [ ] Core 代码和文档没有 forgotten-revert 分支。
