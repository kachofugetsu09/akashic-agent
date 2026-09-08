# 插件工具引用与模型展示设计

- 状态：implemented
- 日期：2026-09-08
- 关联条款：CTX-004、CTX-007、RUN-003、PLG-003、PLG-008、PLG-009、PLG-014、PLG-016
- 决策：[0062](../decisions/0062-tools-flow-through-provider-views.md)

## 1. 目标

插件注册工具后得到当前 Root 中的实际引用。能力提供者通过 `ServiceKey` 把引用组成的
`ToolView` 交给消费者；消费者不能只拿到通用工具池，再按全局字符串选择任意工具。

```text
provider ── register ──▶ ToolRef ── provide ToolView ──▶ consumer
                              │                              │
                              └──────── pool.bind(ref) ◀─────┘
```

搜索只帮助模型理解已获授 view。它不新增授权，也不保存 loaded、grant、LRU、TTL、epoch
或 compaction 撤权状态。知道已获授工具的名称和 schema 时，模型可以直接调用。

## 2. 引用、view 与持久 binding

- `ToolRef` 直接指向一次真实注册，只携带名称和冻结的公开描述。工具池保存实际 open、capture
  以及 exact ref 对应的 prepare/authorize，在 view、binding 和贡献注册边界核对引用仍属于当前
  Root 的同一个注册。工具卸载后，同名新注册不会继承旧贡献。
- `ToolView` 只是一组引用。工具池用引用描述中的实际 provider plugin ID 分组，并保存 provider
  声明的唯一组级 `always_on`。组没有独立简介；目录只使用插件 ID 和工具描述摘要。
- `ALL_TOOLS` 是显式能力。确实需要完整池的插件必须 inject 它；持有 `TOOLS` 本身不能列出、
  按名称绑定或取得完整池。
- `Bindings` 继续保存实际归档闭包。当前 `ToolRef` 只用于新绑定；Message 中已经提交的
  `binding_id` 继续打开原归档实现。安装、卸载和重启不能让旧调用重新选择当前注册。

这次是 breaking 注册 API：工具 provider 必须保存 `register()` 返回的引用并 provide view；
prepare/authorize 贡献也必须取得该 exact ref；按全局名称配置工具的消费者改为依赖 provider
view。没有新的数据库或业务状态；旧 binding 和旧 Message 原样可读、可恢复。唯一 workspace
迁移把 `context.prompt_sources.skills` 的准确旧 owner `skills` 改为 `standard_tools`，并保存原
TOML；缺字段、不同 owner 或结构错误分别保持不变或 fail-loud。
旧 `reply.tools` 等配置字段，以及 `disabled` 中的旧 `skills`、`agent_restart` 插件 ID 已明确
拒绝；升级者必须审阅并修改这些配置。自动迁移不推测这些字段的意图。

## 3. 模型展示与搜索

`ToolPresentation` 只拥有三件事：本次固定顶层 schema、把 provider 返回的 wire tool call
解码为一个真实 `ToolRef + arguments`，以及本次程序需要的目录 reminder。`react` 只调用这个
接口，不识别 `tool_search`、间接调用工具或来源名称。

Native presentation 把获授 view 中的工具直接放入固定 schema。Search presentation 固定展示
conversation 基础组中 `always_on` 的真实工具、`tool_search` 和 tool-search 插件拥有的间接
调用 schema。

搜索命中一个插件组时，在唯一一次 `ToolResult` 文本中返回该组全部工具的完整 schema。
间接调用只可在 Search presentation 已获授的 view 中按名字解析；模型的 wrapper call 作为
`model.facts` 的 wire replay 材料保存，Message 日志只提交解码后的真实 `ToolCall`，因此
prepare、authorize、invoke 和 receipt 都只执行一次。

目录通过当前 Search presentation 交给现有 `system-reminder`。Wake、Scheduler、Subagent
和其他 native presentation 不得到该提醒。顶层 schemas 在搜索前后严格相同。

`model.facts` 兼容旧字段；新成功 Output 额外保存本次实际请求末尾的 reminder replay 和原
wire tool calls。下一次同一摘要下重放这些请求事实，避免把后来生成的末尾 reminder 插入旧
provider 前缀。失败或取消不保存成功 replay。新 Summary 正常开启新上下文，旧搜索结果只是
普通 ToolResult；不专门删除 raw tail，也不改变调用权限。

## 4. 现有来源的工具范围

| 来源 | 新工具范围 | 保留的固定事实 |
|---|---|---|
| conversation/reply | `ALL_TOOLS` 中全部公开 provider view，加 tool-search 私有 view；只有 provider 声明的 always-on 组直接展示，其余组通过搜索展示 | 每轮当前 Root 引用；旧调用按 binding 恢复 |
| Wake | 自己的私有决定工具、Memory recall provider、Web provider | `Request.tools` 保存原 binding；缺硬依赖不激活 |
| Scheduler | 当前全池减 message-push 与 memory 写入/召回工具 | 原允许范围不扩大；外部工具仍可用 |
| Subagent | `research`、`scripting`、`general` 三个既有 profile | prepared 时保存配置化 binding；无搜索能力 |
| Plugin validation | 请求显式 `validation_tools`，未声明时为候选 Root 的完整 view | 只在候选隔离 Session 执行 |
| Programmatic | 作为普通 Source 进入 reply，复用同一 presentation | 不新建独立工具池或来源特判 |

conversation 的 always-on provider 是 `standard_tools`、`standard_web`、`tool_search` 和
`message_push`。`standard_tools` 拥有文件、编辑、shell、`write_stdin` 与 `load_skill`；
`standard_web` 拥有 `web_fetch/web_search`。`agent_restart` 是 `message_push` 在 supervised
runtime 且完整送达依赖就绪时挂载的可选 child，因此沿用 `message_push` provider 组，不建立
第二个插件 ID。Akasha 和外部公开 provider 默认只在搜索结果中展示。

MCP `tools/list` 得到的工具必须注册进同一引用池并归到声明它的实际插件组；正式 MCP route
仍拥有进程、allowlist、恢复和 transport 错误。Akasha 的直接 Tool 与 MCP 工具对消费者都只
通过 view 暴露，不能绕回 legacy `ToolRegistry` 获得额外权限。

## 5. 验收

- 未获 view 的消费者即使猜对名称也不能 bind；获授 view 内可不经搜索直接调用。
- provider 消失导致硬依赖 consumer 不激活；已提交 binding 仍打开原归档闭包。
- 搜索返回整组完整 schema，下一轮目录截断正确，顶层 schema 搜索前后相同。
- 间接调用只生成一个真实 ToolCall 和一个最终 ToolResult；wrapper wire call 可重放。
- compaction 不修改 Message、不撤销工具权限，也不把旧搜索结果当成新的授权事实。
- Scheduler、Wake、Subagent、Plugin validation 和 Programmatic 保留上述范围。
- `agent_restart` 不再要求搜索，仍只在 supervisor runtime、成功送达后按原 receipt 执行。
- a1b386b7 生成并归档的 Wake Request 在候选代码中恢复后，实际产生
  `Input → Input → Output(ToolCall) → ToolResult(success) → Output(quiet)`，证明旧
  `tool_names`、旧 ReAct wire 和原 binding 仍跨升级执行；兼容入口只接受与 fixed bindings
  完全一致的旧名称集合。
