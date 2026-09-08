# 0062 · 工具通过 provider view 流向消费者

- 状态：accepted / implemented
- 日期：2026-09-08
- 关联条款：CTX-004、CTX-007、PLG-003、PLG-008、PLG-009、PLG-014、PLG-016
- supersedes：0047 中“Service 绑定一个工具名并从全局目录解析”的选择
- superseded by：无

## 背景

现行 `ToolCatalog` 同时保存所有注册并允许任何持有者按字符串绑定。消费者的名称白名单在
调用边界再次检查，但依赖关系本身没有交付实际工具。搜索又把 schema 展示、选择、授权和
LRU 混在日志投影中。于是工具名字可以绕过 provider 依赖，搜索状态也承担了不属于展示层的
权限含义。

## 决定

工具注册返回当前 Root 的实际 `ToolRef`。引用只携带冻结的公开描述；工具池独占 open、capture
以及按 exact ref 关联的 prepare/authorize。provider 通过普通 `ServiceKey` 提供由引用组成的
`ToolView`；工具池只绑定当前有效引用。需要完整池的管理插件显式 inject `ALL_TOOLS`。当前引用
与已经持久化的归档 binding 保持两个清楚的生命周期。

搜索只在获授 view 内读取完整元数据。它返回整组 schema，并用固定的间接调用展示把 wire call
解码成唯一真实 ToolCall；不创建发现授权、loaded 状态或 compaction 撤权。模型展示使用一个小
接口，通用 ReAct 不识别具体搜索工具或来源。

## 理由

- 依赖直接交付能力，工具名字不再成为第二条授权路径。
- 引用就是实际注册，不需要 token、grant 表或另一套 generation 身份。
- 搜索、顶层 schema 和持久执行各有一个 owner；安装变化不改写旧 Message 与 binding。
- 内置与外部插件、直接工具与 MCP 工具使用同一消费者边界。

## 影响

`register()`、prepare/authorize 贡献、按名字配置和搜索选择协议是 breaking API。全部真实
消费者迁移到 view；旧 `tool.selection` 只作为历史内容读取。数据库 schema 不变。workspace
迁移只把既有 `context.prompt_sources.skills = "skills"` 精确改为 `"standard_tools"`，并在同目录
保留 `config.before-tool-provider-views.toml` 恢复点；其他值不猜测、不改写。

完整调用链、来源范围、reminder/replay 合同和验收见
[插件工具引用与模型展示设计](../design/plugin-tool-view.md)。
