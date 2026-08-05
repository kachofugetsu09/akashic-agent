# NOW

这份文件只保存 Akashic Agent 当前仍未完成的工作。事项完成后删除，不保留“已完成”记录。

## P0 · 插件递归自验证

- 完成 turn-local/shared-state owner 审计，并用 pointer、memory write set、真实 tool item、timer 投递顺序和 crash recovery mutants 验收完整合同。

## P0 · 上下文与持久历史隔离

- 从 `DefaultReasoner` 移除 `SessionManager` 依赖；动态区块退化不改 session，history window 退化只改运行时视图，不写 store。
- 清理 `trim_history_async` 等含糊 API；需要独立 cache owner 时再引入运行时视图类型，不新增与 `PromptRenderInput` 同义的抽象。

## P0 · 独立语义验收

- 将 CTX-001 当前的 trace、完整状态快照和 fixture `DELETE` pilot 升级为 SQLite authorizer 与一次性候选真实 retry seam mutant；导入失败、fixture 失败或超时不得计为 mutant kill。
- 建立受保护路径 policy：`semantic_delta: none` 的普通实现改动不能同时修改 P0 oracle、mutant 或 coverage baseline 来获得全绿。
- 建立轻量 `change-intent` 校验，检查实际 diff、允许路径、受保护状态和副作用是否超出声明。

## P1 · 工作流扩展

- 把 `projectneed.md` 中其他 P0 不变量逐步迁入可执行契约，优先处理 MEM-001、MEM-002、OUT-001、PLG-001、PLG-004、WSP-001 和 BAK-001。
- 为高风险 refactor 增加 base/candidate 差分回放，核对持久 write set、事件、外部调用和错误分类。
- 由维护者继续确认 [`design/persistence-state-map.md`](design/persistence-state-map.md) 的 INT-009、INT-010、INT-012～INT-014，以及旧消息编辑和 turns retention；INT-001～INT-008、INT-011 已提升为 projectneed 条款。
- 把 `mcp/servers/*.toml` 直装声明和 workspace 手工 skill 目录迁移成插件贡献；迁移现存能力后收窄 `WorkspaceMcpAdmin`、watcher 和 loader，Skill/MCP 只保留插件安装、readiness 与 generation 发布这一个 owner。
- 把已确认的持久化状态地图转成机器可读备份 manifest，补齐目录快照、global companion state 和隔离恢复演练；确认 snapshot 能启动只读 runtime，并读取会话、记忆、调度、插件数据和主动流程连续性。
