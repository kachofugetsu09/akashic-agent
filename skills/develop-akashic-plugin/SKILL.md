---
name: develop-akashic-plugin
description: 创建、编写、修改并验证 Akashic 插件及插件内 Skill/MCP。用户要求创建 Akashic 插件、编写插件、把 skill 收入插件、验证插件、验证 skill、热重载后自测或递归自验证插件时使用。
---

# 开发并验证 Akashic 插件

只在 canonical source 中修改插件。先取得可恢复备份和明确 write set；不要直接编辑 `~/.akashic-plugin/cache`、workspace skill 软链接或正式 plugin-data。

## 1. 读取真实合同

1. 进入目标仓库后先读它的 `AGENTS.md`、文档索引和工作流。
2. 创建或修改插件前，完整读取 [references/plugin-authoring.md](references/plugin-authoring.md)。
3. 安装和行为验证前，完整读取 [references/self-validation.md](references/self-validation.md)。
4. 只有子 turn 排队、超时、结果错误或插件行为不明时，才完整读取 [references/runtime-diagnostics.md](references/runtime-diagnostics.md)，从 reload journal、SessionDB 和真实日志重建执行轨迹；成功前不要预先做全量诊断考古。
5. 从当前 `agent.plugins.Plugin`、装饰器、spec 和相邻插件核对 API；参考文件只提供路由，代码是当前事实。

已给出 workspace、config、Gateway cwd/解释器或插件根时直接使用，不再枚举所有进程、环境、配置和数据库 schema。prompt-only 插件走 [authoring 的 Prompt 注入模板](references/plugin-authoring.md#3-prompt-注入)：只核对模板直接引用的公开类型；在 import/source test 失败前，不搜索 manager、EventBus、phase、control 或安装器内部实现。

## 2. 实现最小插件

保持一个清楚的能力 owner：

```text
plugin source/
├── plugin.py
├── tests/
├── skills/<skill-name>/SKILL.md   可选
├── mcp/                            可选
└── requirements.txt                仅在确有依赖时
```

- 根目录必须有 `plugin.py`，Plugin 子类声明 `name` 和 `version`。
- Tool 使用 `@tool` 声明稳定名称、真实 risk 和可搜索提示；handler 失败要暴露。
- Skill 放入插件 source 的 `skills/`，由 `skill_roots()` 声明；不要先复制到 workspace。
- MCP、channel、managed service 和 proactive source 只在能力确实需要时声明。
- 没有第三方依赖时不要创建空 `requirements.txt`。
- plugin-data 写入必须由插件 owner 管理；候选验证不得假设 snapshot rollback 能撤销文件或外部效果。
- 不添加 mock success、宽泛异常、空 fallback 或只为通过 doctor 的假能力。

## 3. 先验证 source

运行插件自己的最小测试，至少覆盖：

- `plugin.py` 可从干净 checkout 导入。
- Tool schema、risk、参数与真实返回值。
- Skill frontmatter、目录名、引用文件和触发描述。
- MCP/readiness、配置和生命周期（存在时）。
- 失败、取消与 cleanup 的真实终态。

安装只读取已提交 Git HEAD。source 测试通过后提交；使用远程 source 时还要确认远端包含该 commit。未获授权时不要自行发布、开 PR 或改变外部服务。

## 4. 安装并验证候选

先按 [references/self-validation.md](references/self-validation.md) 检查当前 runtime 处于哪一级：完整的 stable/latest 候选隔离，或只有 session-lane + current snapshot 的隔离环境自验证。

每次 programmatic 验证都先保存 `execution_id`、`thread_id`、`turn_id`、`plugin_id` 和 reload `tx_id`。命令超时、子 turn 停在 `queued`、final response 不符合 oracle 或工具没有执行时，不要直接搜索源码或重复安装；按 [references/runtime-diagnostics.md](references/runtime-diagnostics.md) 查询真实状态和内容，先定位失败层。

正常快路径按 `source test → commit → install → plugin-status/doctor → latest child → 定向 SessionDB/journal 查询 → promote` 单向推进。命令成功时不要为确认其实现而反向阅读 CLI、socket、pointer 或 EventBus 源码；正式输出和数据库行就是该边界的证据。

支持时使用下面的闭环：

```text
install → latest ready → programmatic child on latest → behavioral oracle
   ▲                                                  │
   └──────── actionable failure → discard → fix ─────┘

pass → promote → re-read stable/latest → report
```

programmatic child 必须：

- 创建新 session 并显式选择 `latest`。
- 默认不沉淀语义记忆，但允许检索已有记忆。
- 实际加载或触发新增 Skill，并实际调用新增 Tool；不能只问“你能否看到”。
- 返回结构化 terminal、tool items 和领域 oracle。
- 通过 Shell 的 `execution_id` / `write_stdin` 被父 turn 观察，保持 attached。

验证 Skill 时同时证明：

1. latest catalog 能发现 Skill，source 为 plugin。
2. Skill 正文和引用资源可以加载。
3. 一个真实触发提示会遵循 Skill 的关键步骤。
4. 预期 Tool/文件/领域状态确实出现，而不只是 final response 自述成功。

如果 runtime 尚未实现 `--runtime latest`、promote/discard 或 attached cancellation，但已经按 session lane 允许不同 session 并发，则可在隔离 workspace/runtime 中验证 current snapshot：等待 reload journal 出现 `committed` 事件，再用 Shell 启动新的 programmatic session，父 turn 同步读取子 turn 终态。事务行此时可能已经是 `draining`；它表示旧快照仍有 lease，不表示新快照不可用，也不能把它当成失败。

这种路径证明的是 `current-snapshot self-validation`，不是候选隔离：新入站 turn 也可能看到未验证插件，且失败后没有原子 promote/discard。非隔离正式 runtime 未获用户明确授权时不要使用这条路径；报告 `safe candidate self-validation unavailable`。无论哪一级，都要读取 reload journal、子 turn 的 SessionDB 轨迹、final response、items/tool trace 和可用 runtime log。不要用 sleep、当前 turn 的 `tool_search`、手改 cache 或新启动第二个 Gateway 冒充验证。

## 5. 处理副作用与独占 endpoint

- read-only Tool 可以直接在 latest child 中验证。
- candidate generation 的非 read-only Tool/MCP 默认禁用；只有真实事务/dry-run、隔离 workspace/test endpoint 或用户明确授权时才能另行验证。
- `message_push` 的成功以真实 delivery receipt 和子 session tool trace 为准；push 不会注入父 Prompt或目标 session history。
- 固定端口、bot ownership、channel 或 singleton service 与 stable 冲突时，使用隔离 runtime/endpoint；不要让父 turn 等待自己的 stable lease 排空。

## 6. 收口

完整 stable/latest 路径只有以下事实同时成立，才告诉用户任务完成：

- canonical source 已按授权保存，安装所需 commit 可回源。
- source tests 和结构/readiness 检查通过。
- latest child 的真实行为 oracle 通过；若目标包含 Skill，Skill 发现、加载和行为均通过。
- latest 已晋升 stable，最终 pointer/journal 已重新读取。
- 未授权的记忆写入、plugin-data 写入和外部发送为零。

current-snapshot 隔离路径可以报告“插件行为验证完成”，但必须同时说明没有 candidate/stable 隔离、promote/discard 和 attached cancellation，不能升级成“安全自进化闭环完成”。最终简洁报告 source commit、测试、candidate/stable identity（若存在）、验证 session/turn、关键 tool evidence，以及任何未验证边界。
