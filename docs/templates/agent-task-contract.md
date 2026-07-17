# Agent 任务合同模板

> 只保留会改变当前任务行为的字段。简单任务不需要机械填满全部内容。

## Role

- 负责范围：
- 当前阶段：research / design / implementation / review

## Goal

[一句话写用户最终能看到的结果。]

## Success criteria

- [ ] [可以独立判断的结果 1]
- [ ] [可以独立判断的结果 2]
- [ ] 相关验证已运行，未运行项和原因已说明。

## Evidence

- 必须先读取：
- 已核对事实：
- 未确认事实：
- 关键假设：

## Change intent

```yaml
change_type: fix|feature|refactor|migration|docs
semantic_delta: none|compatible|breaking
capability_owner: core|protocol|mobile|plugin|mixed|not_applicable
consumer_scope: []
runtime_patch: none|required
runtime_patch_reason: ""
authoritative_state_owner: ""
client_only_alternative: ""
invariants: []
protected_state: []
allowed_paths: []
forbidden_paths: []
allowed_effects: []
forbidden_effects: []
validation: []
rollback: ""
worktree_writer: ""
handoff_head: ""
external_revisions: []
schema_lineages: []
```

`runtime_patch: required` 必须引用既有或已批准的不变量，并说明客户端实现为什么会复制、猜测或破坏权威语义。“未来可能复用”不是充分理由。

## Autonomy

- 可自主执行：
- 执行前需确认：

## Tools

| 工具 | 使用时机 | 关键结果 | 空/失败如何处理 |
|---|---|---|---|
| | | | |

## Output

- 交付文件或字段：
- 格式和长度：
- 必须附带的证据：

## Stop rules

- 满足全部成功标准后停止。
- 缺少下列事实时提出最小问题：
- 最多尝试下列 fallback：
- 出现下列状态时停止并报告阻塞：
