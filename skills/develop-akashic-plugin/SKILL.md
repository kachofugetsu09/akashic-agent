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

完整读取 [references/self-validation.md](references/self-validation.md)，只使用 `plugin-install`、`plugin-uninstall` 和 `plugin-revert`。stable/latest、排空、提交和恢复是 Core 内部机制，不要求 Agent 查询或编排。

每次 programmatic 验证都先保存 `execution_id`、`thread_id`、`turn_id`、`plugin_id` 和 reload `tx_id`。命令超时、子 turn 停在 `queued`、final response 不符合 oracle 或工具没有执行时，不要直接搜索源码或重复安装；按 [references/runtime-diagnostics.md](references/runtime-diagnostics.md) 查询真实状态和内容，先定位失败层。

正常快路径按 `source test → commit → install → attached child → 行为 oracle → 正常结束 turn` 单向推进。命令成功时不要为确认其实现而反向阅读 CLI、socket、pointer 或 EventBus 源码；正式输出、child trace 和下一 turn 的 Core 运行事实就是边界证据。

支持时使用下面的闭环：

```text
install → attached programmatic child → behavioral oracle
   ▲                                  │
   └──── failure → revert → fix ──────┘

pass → 正常结束父 turn → Core 自动切换 → 下一 turn 生效
```

programmatic child 必须：

- 创建 attached 新 session；不要指定 `--runtime`，Core 自动绑定当前候选。
- 默认不沉淀语义记忆，但允许检索已有记忆。
- 实际加载或触发新增 Skill，并实际调用新增 Tool；不能只问“你能否看到”。
- 返回结构化 terminal、tool items 和领域 oracle。
- 通过 Shell 的 `execution_id` / `write_stdin` 被父 turn 观察，保持 attached。

验证 Skill 时同时证明：

1. latest catalog 能发现 Skill，source 为 plugin。
2. Skill 正文和引用资源可以加载。
3. 一个真实触发提示会遵循 Skill 的关键步骤。
4. 预期 Tool/文件/领域状态确实出现，而不只是 final response 自述成功。

若 CLI 返回 Core 不支持 turn lineage、attached candidate 或 revert，停止正式安装并报告 `safe candidate self-validation unavailable`。只能在一次性 workspace/runtime 中做隔离验证，不能让正式新请求看到未验证插件。无论哪一级，都要读取子 turn 的 SessionDB 轨迹、final response、items/tool trace 和可用 runtime log；不要用 sleep、当前 turn 的 `tool_search`、手改 cache 或第二个 Gateway 冒充验证。

## 5. 处理副作用与独占 endpoint

- read-only Tool 可以直接在 latest child 中验证。
- candidate generation 的非 read-only Tool/MCP 默认禁用；只有真实事务/dry-run、隔离 workspace/test endpoint 或用户明确授权时才能另行验证。
- `message_push` 的成功以真实 delivery receipt 和子 session tool trace 为准；push 不会注入父 Prompt或目标 session history。
- 固定端口服务必须声明 `ManagedServiceSpec.validation_port_env`，并让服务与 MCP 读取同名环境变量；Core 分配隔离 endpoint 和 plugin-data 副本。Channel 的正式 ownership 只在父 turn 结束后切换。

## 6. 收口

完整 stable/latest 路径只有以下事实同时成立，才告诉用户任务完成：

- canonical source 已按授权保存，安装所需 commit 可回源。
- source tests 和结构/readiness 检查通过。
- attached child 的真实行为 oracle 通过；若目标包含 Skill，Skill 发现、加载和行为均通过。
- 父 turn 正常结束，下一用户 turn 的 Core 事实确认已提交或明确报告失败。
- 未授权的记忆写入、plugin-data 写入和外部发送为零。

一次性 current-snapshot 路径只能报告“隔离环境行为验证完成”，不能升级成“正式安全自进化闭环完成”。最终简洁报告 source commit、测试、验证 session/turn、关键 tool evidence、Core 的 turn 后结果，以及任何未验证边界。
