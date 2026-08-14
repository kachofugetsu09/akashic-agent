# 插件 assistant 持久 metadata 接入点任务合同

- 状态：implemented / verified
- 日期：2026-08-14
- 实现基线：`66fc16c666de14a13c93208d301684ec8e2c9217`
- 关联条款：PLG-002～PLG-004、PLG-008、PLG-014、STA-001～STA-003
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[v3 loader 合同](plugin-v3-loader-task-contract.md)

## 1. 目标与边界

本 PR 只给 `AfterReasoningCtx` 增加 `persist_assistant_metadata`：v3 listener 可以在 preprocess/cleanup 生命周期里声明应随 assistant 消息持久化的领域 metadata，真正写 Session 的 owner 仍是 `_PersistAssistantMessageModule`。

`semantic_delta: compatible`。legacy `persist:assistant:*` slots 继续工作；本 PR 不解析 citation 协议、不实现 Meme、不改变 Session schema、不写正式 workspace 或数据库。

```text
v3 listener ──► ctx.persist_assistant_metadata
                         │
legacy module ──► persist:assistant:* slots
                         │
                         ▼
           persistence boundary validation
                         │
                         ▼
             Session.add_message(assistant)
```

## 2. 失败语义

- 固定字段仍由 persistence phase 独占；v3 metadata 覆盖 `tools_used / tool_chain / reasoning_content / model_state` 时 fail-loud。
- 已退役字段继续拒绝。
- 同一个 key 同时由 v3 context 和 legacy slot 生产时视为迁移错误，不定义隐式优先级。
- listener 只修改当前 Turn 的内存 context；Session 写入、事务、消息追加和删除权限没有转移给插件。

## 3. 验证与回滚

- targeted：composition listener 写入、legacy slot 共存、assistant message 实际持久化、固定/重复 key 拒绝。
- cumulative：lifecycle、composition、PluginManager 与 hot reload 回归。
- static：compileall、Basedpyright error-level、`git diff --check`。
- Gate：`python docker/debug/gate.py run --base origin/main`。
- 回滚：`/mnt/data/coding/akasic-agent/.backups/20260814-pre-plugin-assistant-metadata-seam-66fc16c6.bundle`。
