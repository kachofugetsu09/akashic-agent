# 语义 Oracle 实施清单

## 契约

- [ ] 有稳定 invariant ID。
- [ ] 说明触发条件、受保护状态、允许变化和禁止副作用。
- [ ] 指定状态 owner 和恢复 owner。
- [ ] 说明 `semantic_delta: none` 时必须保持的外部行为。

## 观察

- [ ] 从系统边界触发，不只调用内部纯函数。
- [ ] observer 不复用被测 repository mock。
- [ ] 核对完整规范化内容，不只核对数量。
- [ ] 记录实际 write set 和违规尝试。
- [ ] 覆盖错误、取消、再次加载和继续写入。

## 独立性

- [ ] oracle 位于受保护路径。
- [ ] 普通实现改动不能同时修改预期结果。
- [ ] 规格变化有独立决策和用户确认。

## 有效性

- [ ] 至少一个已知错误 mutant。
- [ ] mutant 的失败原因指向对应 invariant。
- [ ] 当前正确实现成功。
- [ ] 测试失败时输出 before/after、write trace 和关键标识。

## 交付

- [ ] CI 有独立 job 和稳定名称。
- [ ] PR 模板引用 invariant、oracle 和 mutant。
- [ ] `NOW.md` 中对应事项在完成后删除。
