# Implementation Plans

由 improve skill 于 2026-07-11 生成。按依赖顺序执行；每一步必须通过计划内 Gate，失败时停止并保留上一可运行状态。

## Execution order & status

| Plan | Title | Priority | Effort | Depends on | Status |
|---|---|---|---|---|---|
| 001 | 以验证门禁完成全插件热重载 | P1 | L | — | DONE |

Status values: TODO | IN PROGRESS | DONE | BLOCKED | REJECTED

## Dependency notes

```text
001
└─ G-1 沙盒完整性
   └─ G0 基线
      └─ G1 候选验证
         └─ G3 插件语义验证
            └─ G2 资源预热
               └─ G4 原子发布
                  └─ G5 旧代际排空
                     └─ G6 完整 Runtime 验收
```

任何 Gate 未通过时，不得继续后续步骤或删除旧实现。

## Findings considered and rejected

- 只增加文件 Watcher：无法注销旧事件、任务和运行时快照，会制造重复执行。
- 使用 `importlib.reload()`：外部旧对象引用不会自动更新，无法保证同一 turn 一致。
- 为每类能力分别实现 reload：会形成多套生命周期和回滚语义。
- 先卸载旧插件再加载新插件：候选失败时失去 last-known-good，且 Channel、MCP 和主动链路产生不必要空窗。
