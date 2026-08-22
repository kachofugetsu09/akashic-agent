## 问题与结果

- 解决的问题：
- 用户可见结果：
- 本 PR 不处理：

## Change intent

- `change_type`：`fix | feature | refactor | migration | docs`
- `semantic_delta`：`none | compatible | breaking`
- `capability_owner`：`core | protocol | mobile | plugin | mixed | not_applicable`
- `consumer_scope`：
- `runtime_patch`：`none | required`
- `runtime_patch_reason`：
- `authoritative_state_owner`：
- `client_only_alternative`：
- `concept_gate`：`required | not_applicable`
- `concept_gate_reason`：
- 关联不变量：
- `protected_state`：
- 允许的副作用：
- 禁止的副作用：

## 改动范围

- 主要文件：
- 配置或迁移影响：
- 已知 schema lineage 与最终 schema identity：
- 协议 source、runtime、provider 与 scenario 的不可变 revision：
- 回滚方式：

## 验证

- [ ] 相关 targeted tests 已通过。
- [ ] Python 类型检查或前端 typecheck/lint 已通过；不适用项已说明。
- [ ] `python docker/debug/gate.py run --base origin/main` 已运行。
- `sourceDigest`：
- `planDigest`：
- 真实设备证据（设备/API、debug application ID、源码/APK 身份；不适用时说明）：
- 未运行项与原因：

## 正交性与概念完整性

- 是否属于架构性 PR 或大改动：`yes | no`
- 独立 reviewer / model / reasoning：
- 审查 head：
- 新增概念及其唯一 owner；没有时写 `none`：
- 无关设计轴的变化传播；没有时写 `none`：
- 最短正常/失败/热更新链路：
- legacy/source-specific 残留扫描：
- must-fix 及处置证据：
- 最终结论：`pass | fail | not_applicable`

## 工作手册

- [ ] 已核对 `projectneed.md`、相关决策和 `NOW.md`。
- [ ] 长期语义变化已先获得确认，或本次没有长期语义变化。
- [ ] 跨仓库或客户端改动已按 MOB-001 核对能力 owner；不适用时已标明。
- [ ] 已完成事项已从 `NOW.md` 删除；当前没有对应事项时标记不适用。
