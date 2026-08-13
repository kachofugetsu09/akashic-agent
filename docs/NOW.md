# NOW

这份文件只保存 Akashic Agent 当前仍未完成的工作。事项完成后删除，不保留“已完成”记录。

## P1 · 移动端主题 token 边界

[`移动端投影审计 D2`](design/mobile-projection-audit.md) 已确认原生壳 Compose 色板与 Core WebUI CSS token 是两个渲染层的表示，不是重复 owner。仍需决定色值一致性由构建期产物还是显式 token 边界保证。

- 移动端用户 checkout 存在未提交 Theme diff（Theme.kt 等 5 个文件）；D2 决策（原生壳与 WebUI token 边界）完成前不得合入。

## P0 · 插件递归自验证

- 在独立 Fitbit canonical source 变更中让 monitor 与 MCP 读取同一个 `validation_port_env`，再以一次性 workspace 验收真实隔离 listener、child tool trace、正式切换和旧 listener 恢复；不得在 Core 中添加 Fitbit 特判。
- 补充 turn-boundary rollout 的进程崩溃注入矩阵，覆盖 terminal 封口后、候选服务停止后、正式 endpoint 切换后和 pointer 提交前；恢复失败必须保持 degraded 可见，不能只恢复 pointer。

## P0 · 独立语义验收

- 将 CTX-001 当前的 trace、完整状态快照和 fixture `DELETE` pilot 升级为 SQLite authorizer 与一次性候选真实 retry seam mutant；导入失败、fixture 失败或超时不得计为 mutant kill。
- 建立受保护路径 policy：`semantic_delta: none` 的普通实现改动不能同时修改 P0 oracle、mutant 或 coverage baseline 来获得全绿。
- 建立轻量 `change-intent` 校验，检查实际 diff、允许路径、受保护状态和副作用是否超出声明。

## P1 · Agent Harness 抽象收敛

- 目标骨架只使用 `Message`、`Turn`、`Session`：Message 组成 Turn，Turn 归入 Session；`Loop` 表达“输入 Message → 内部 `react` → 输出 Message”。当前从 `AgentLoop._react → PassiveTurnPipeline` 继续向内审查；只有独占权威状态、不变量、控制流、生命周期或真实边界的层才保留。纯转发、重复结果包装、字段复制、内部重复校验和平行模型分批内联、合并或删除；命名使用普通英语和 Python 风格，不再引入 `Unit` 一类没有独立事实的概念。
- Turn 的待审目标是：多次 user 输入可以跨越被中断的执行尝试，最后与唯一 terminal assistant 构成一个完整 Turn；主动投喂、scheduler、spawn 和不依赖 user query 的消息可以各自成为独立 Turn，再由 Turn 组合时间线与 Akasha 节点。当前 SES-007、SES-008、RUN-008、OUT-001、OUT-004 和 OUT-005 的 `logical interaction / execution attempt`、主动送达与 `message_push` 合同仍是权威语义；改变名称、数据库身份或归属前，必须先用 SessionDB 与 runtime 日志证明真实路径，再单独批准规格、数据和迁移，不能借普通 refactor 偷改。
- 接手顺序固定为从内向外的小批次：`ReasonerResult` metadata dict 已类型化、`AfterReasoningResult` 已内联进 `TurnSnapshot`（less-is-more PR62/PR63）；剩余先审查重复 input DTO（`BeforeReasoningInput`/`AfterReasoningInput`/`PromptRenderInput` 与 GATE ctx 的平行字段）是否重述同一事实，审查结论写进账本；再完整画出 Message、Turn、Session、interaction 和 attempt 的 owner/写入链，确认非唯一 attempt 的真实频率与恢复用途；最后才评估 proactive、scheduler、spawn 和 `message_push` 怎样由同一组原子能力拼接。每个 PR 只处理一个冗余组，不为未来预建总框架，也不把现有独立实现直接包进新的总抽象。
- Compaction 是不可替代能力，现有插件也是受保护边界。插件 lifecycle、phase/hook 顺序、slot、context、错误传播、generation、scope 和已有能力不得改变；疑似冗余的插件桥只做标记，删除前必须同时扫描插件源代码、已安装 cache、测试和真实运行证据。不得为了收敛 Core 提前改变 proactive、scheduler、spawn、`message_push` 或持久化行为。
- 每个删除候选都要回答四个问题：它拥有哪项独立事实、谁在生产或插件中消费、日志或数据库中多常触发、删除后由谁承接职责。无独立事实且无真实消费者时直接删除，不保留 deprecated alias、兼容壳或占位抽象；竞态和防御分支只有存在具体可达路径且当前位置拥有正确恢复动作时才进入主链，极低频风险只记录。每批以 base/candidate 差分回放、完整 write set/事件/外部调用、插件 Gate 和全量回归证明 `semantic_delta: none`。

## P1 · 工作流扩展

- 按 [`容器与 Host Bridge 非迁移实验合同`](design/akashic-container-host-bridge-experiment-contract.md) 完成 mise/锁文件前置、本机 Local/Bridge/容器分层验证和 hua-home 隔离候选运行时；在 capability matrix、Supervisor 故障注入、OpenCode V4 Flash High 与正式状态零写入证据齐全前，不启动正式 workspace 迁移。
- 把 `projectneed.md` 中其他 P0 不变量逐步迁入可执行契约，优先处理 MEM-001、MEM-002、OUT-001、PLG-001、PLG-004、WSP-001 和 BAK-001。
- 为高风险 refactor 增加 base/candidate 差分回放，核对持久 write set、事件、外部调用和错误分类。
- 由维护者继续确认 [`design/persistence-state-map.md`](design/persistence-state-map.md) 的 INT-009、INT-010、INT-012～INT-014，以及旧消息编辑和 turns retention；INT-001～INT-008、INT-011 已提升为 projectneed 条款。
- 把 `mcp/servers/*.toml` 直装声明和 workspace 手工 skill 目录迁移成插件贡献；迁移现存能力后收窄 `WorkspaceMcpAdmin`、watcher 和 loader，Skill/MCP 只保留插件安装、readiness 与 generation 发布这一个 owner。
- 把已确认的持久化状态地图转成机器可读备份 manifest，补齐目录快照、global companion state 和隔离恢复演练；确认 snapshot 能启动只读 runtime，并读取会话、记忆、调度、插件数据和主动流程连续性。
