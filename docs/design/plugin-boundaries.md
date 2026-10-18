# 插件边界评估与收口计划（2026-10）

基线：main `c713f815`。目标：每个插件都是平凡、非特权的；宿主（聊天前端核心、`akashic_clients`、`ui`、
`context`、`reply`）只提供通用扩展点，不认识任何具体插件。插件 A 只能通过已声明依赖的
`plugins/B/contract.py` 使用 B，B 未安装时 A 的功能自然缺席，而不是由宿主替 B 留着一块 UI 或接口。

## 判定规则

1. **宿主不点名。** 宿主代码里不出现插件 ID、插件专属方法名、插件专属内容类型或插件专属路由。
2. **功能跟着 owner 走。** 一个功能的后端、查询方法与前端模块都由同一个插件提供；跨插件流程由
   依赖方通过对方 contract 完成，而不是宿主串起两个插件。
3. **扩展点有抽象度。** 宿主提供的是“输入框建议源”“侧栏分区”“状态行”“通知条”“内容类型显示策略”
   这类通用槽位，由宿主统一呈现视觉；插件只提供数据或挂载内容。
4. **一个插件一件事。** 互不相关的能力不放在同一个插件里互相拖着装卸。

## 现状：违反点（按耦合程度）

| # | 功能 | owner | 现在在哪里 | 规模 |
|---|---|---|---|---|
| 1 | 项目导航、新建项目、目录浏览、项目置顶；新建项目时顺带设 akasha 记忆策略 | projects（+akasha） | `web-projects.ts` 写死 `"projects"`/`"akasha"`；`project-navigation.tsx`、`project-directory-dialog.tsx`、侧栏与控制器约 150 行胶水；后端 `akashic_clients/navigation.py` 按前缀 `"projects"` 找插件 | ≈1000 行 |
| 2 | 模型胶囊、模型切换 | models（provider 图标还认识 codex/deepseek/opencode/openrouter） | `model-capsule-picker.tsx`、`model-capsule-data.ts`；`/api/chat/models` 在 `chat_api.py`；`CLIENT_CAPABILITIES` 把模型列为聊天启动的硬依赖 | ≈700 行 |
| 3 | 调用统计行、重试/尝试状态、`model.facts` | models | `composer-stats-line.tsx`、`model-call-stats.ts`、`message-view.tsx` 的 RetryStatus；`/api/chat/model-calls/{id}` 在 `chat_api.py` | ≈300 行 |
| 4 | 后台任务列表与停止 | jobs | `job-list-action.tsx`；控制器 WebSocket 分发里解析 `session.jobs`；`akashic_clients/channel.py`、`web_chat.py` 推帧 | ≈260 行 |
| 5 | 技能提及菜单 | standard_tools（技能） | `skill-mention.tsx` 读宿主的 `/api/chat/runtime/capabilities`；该接口在宿主里把技能、MCP、运行时目录拼在一起 | ≈275 行 |
| 6 | 会话工作目录与 AGENTS 规则 | standard_tools（工作目录） | `session-directory.tsx` 写死 `installed("standard_tools")` 与 `directory.current`；插件自己的 `plugin_ui.js` 是空壳 | ≈125 行 |
| 7 | 宿主执行桥告警 | host_execution | `host-bridge-notice.tsx` 每 5 秒轮询插件自有路由 `/api/runtime/host-bridge` | ≈50 行 |
| 8 | 隐藏的内容类型 | conversation、models、compaction、akasha | `message-timeline.ts:307` 中央名单；`context.notice`、`history.transcript` 也在核心特判 | ≈80 行 |
| 9 | 定时任务消息标记 | scheduler | `message-view.tsx:648` 判断 `author === "scheduler"` | 1 行 |
| 10 | MCP 详情、运行时文档、定时任务查看 | mcp、runtime_inspection | `chat_api.py:524-563` 代理并在宿主里渲染 MCP Markdown；`capabilities.py:45-49` 复制 RPC 键字符串 | ≈100 行 |
| 11 | 核心迁移里的频道资源名 | channels | `agent/plugins/manager.py:121-127` | 迁移代码 |

已经做对的：akasha、models（回合后统计）、workspace_changes、sites、visuals 通过 `UI_SLOTS`
贡献 `message_ui.js`；工作台文件意图由宿主广播、插件认领；Shell 结果卡按形状识别而非工具名。

### 混杂的插件

- **standard_tools**：工作目录、AGENTS 规则、文件工具、Shell、技能、视觉六件事。`sites`、`terminal`、
  `projects`、`workspace_changes` 只为拿 cwd 就依赖整个 standard_tools；视觉工具是它依赖 `CHAT_MODELS` 的唯一原因。
- **runtime_inspection**：文档、定时任务、技能三类检视混在一组 RPC。
- **projects**：带了一个通用的文件系统浏览器 `directory.browse`。
- **akashic_clients/chat_api.py**：通用聊天路由之外，还承载模型、调用统计、运行时文档、任务、技能+MCP 能力、项目置顶。

### 边界门的盲区

`scripts/plugin_boundary.py` 只看 Python import 与 Core。它没有检查：宿主插件与聊天前端里的插件 ID
（`installed("x")`、`queryHostPlugin("x", …)`、`*_PLUGIN = "x"`）；`RpcMethod.key(...)` 字符串重复声明；
R9 的领域词表与扫描范围都偏窄。

本轮补上 **R12（UI 宿主不点名插件）**，现有 22 处记入基线：
- `plugin_boundary.toml [ui_hosts]` 列出宿主插件（`akashic_clients`、`ui`）和它们可以依赖的基础设施合同；导入其他插件合同即违规（绝对、相对、普通 `import` 都算）。宿主名单只决定扫描范围，不豁免宿主之间的依赖。
- 宿主里拿插件身份比较即违规：`plugin_id == "x"`、`id.split("@")[0] == "x"`、`source == "x"`、`plugin_id.startswith("x")`；身份包括目录名与 manifest 名。普通文件名或 kind 比较不算。
- 聊天前端里 `queryHostPlugin(…)`、`.installed(…)` 传入静态 ID 或 `*_PLUGIN` 常量、定义 `*_PLUGIN = "x"`，以及访问 `host_routes` 以外的 `/api/…` 路由（按完整路径段比较），即违规；注释不算（按 JS 词法跳过字符串、模板与正则字面量里的注释标记）。
- **只许减少**：带 `--base` 时用同一规则扫描 base 快照，新增点名即使写进账本也失败；放宽 `infrastructure`、`host_routes`，或缩小 `plugins`、`frontend` 的扫描范围也失败。
- 不覆盖：动态拼出的插件 ID 或路由、`RpcMethod.key(...)` 字符串在宿主里重复声明（`capabilities.py:45-49`，随第 8 步由 runtime_inspection 各 owner 发布合同时收掉）。

## 目标扩展点

聊天前端补一组通用槽位，与现有 `turn.*`、`message.attachment`、`drawer.panel` 同一套目录、资产与查询协议：

| 扩展点 | 形态 | 宿主负责 | 首个使用者 |
|---|---|---|---|
| `composer.suggest` | 数据源：`triggers`、`items(query)`、`insert(item)`、`decorate(text)` | 菜单视觉、键盘、IME、可访问性、提及高亮 | 技能 |
| `composer.status` | 挂载 | 输入框下方状态行的布局与收起 | 工作目录、调用统计 |
| `composer.toolbar` | 挂载 | 输入框工具条位置 | 模型胶囊、后台任务 |
| `chat.notice` | 挂载 | 通知条位置与层级 | 宿主执行桥 |
| `navigation.section` | 挂载 + 置顶引用 `(plugin_id, ref)` | 侧栏分区顺序、折叠、置顶存储 | 项目 |
| 内容类型显示策略 | owner 为自己的内部内容注册 `message.display:<kind>` = `hidden_part` | 时间线隐藏 owner 声明的内部材料；展示方缺席的仍显示“无法展示此内容” | 所有内容类型 |
| 插件推送 | `ctx.subscribe(topic)`，经现有 WebSocket 复用 | 连接、重连、背压 | 后台任务 |

插件前端模块通过页面 import map 共享宿主的 React 与 `@akashic/chat-ui-v1`（类型包 + 少量基础组件与令牌，
与 Dashboard 的 `@akashic/web-ui-v1` 同一做法），保证视觉一致，也避免每个插件各打一份 React。

## 收口顺序（每个 PR 新增 ≤500 行，删除不计）

1. 本文档 + 边界门 R12（宿主与聊天前端不点名插件）带基线，只许减少。
2. 内容类型显示策略：owner 声明，删掉中央隐藏名单与 `context.notice`/`history.*`/`scheduler` 特判。
3. `chat.notice` + `composer.status` + import map 与 `@akashic/chat-ui-v1`；迁走宿主执行桥告警、工作目录。
4. `composer.suggest`；技能列表改由技能 owner 的插件查询提供，迁走 `skill-mention.tsx`。
5. 插件推送；迁走后台任务（含 `akashic_clients` 的 jobs 帧）。
6. 模型胶囊与调用统计迁入 models 前端模块（分两个 PR）；provider 图标由各 provider 插件声明；`/api/chat/models`、`model-calls` 移出 chat_api。
7. `navigation.section` 与通用置顶；项目导航迁入 projects（分两到三个 PR）；记忆策略改由 projects 通过 akasha contract 完成或由 akasha 贡献对话框片段。
8. 后端拆分：standard_tools → working_directory / files / shell / skills / vision；runtime_inspection 各 owner 自带检视；chat_api 只留通用路由。插件 ID 变化需要迁移，单独评估。
