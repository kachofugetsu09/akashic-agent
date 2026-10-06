# 插件工具引用与模型展示设计

- 状态：implemented
- 日期：2026-09-09
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

按插件加载只帮助模型取得已获授 view 的 schema。它不新增授权，也不保存 loaded、grant、LRU、TTL、
epoch 或 compaction 撤权状态。知道已获授工具的名称和 schema 时，模型可以直接调用。

## 2. 引用、view 与持久 binding

- `ToolRef` 直接指向一次真实注册，只携带名称和冻结的公开描述。工具池保存实际 open、capture
  以及 exact ref 对应的 prepare/authorize，在 view、binding 和贡献注册边界核对引用仍属于当前
  Root 的同一个注册。工具卸载后，同名新注册不会继承旧贡献。
- `ToolView` 只是一组引用。工具池用引用描述中的实际 provider plugin ID 分组，并保存 provider
  声明的组级 `always_on` 与用途。内置 provider 复用模块声明的 `desc`；未声明用途时明确显示
  “未声明用途”，不从工具名称猜测插件用途。
- `ALL_TOOLS` 是显式能力。确实需要完整池的插件必须 inject 它；持有 `TOOLS` 本身不能列出、
  按名称绑定或取得完整池。
- `Bindings` 保存归档来源证据和固定 metadata。当前 `ToolRef` 只用于新绑定；Message 中已经提交的
  `binding_id` 由当前 selected Root 的同一服务打开，并由当前 provider 核对固定描述，不能重选
  另一个工具。归档 v2 描述中的 `risk` 和 `search_hint` 只在恢复比较时移除；其余字段仍须精确匹配
  当前注册。

这次是 breaking 注册 API：工具 provider 必须保存 `register()` 返回的引用并 provide view；
prepare/authorize 贡献也必须取得该 exact ref；按全局名称配置工具的消费者改为依赖 provider
view。没有新的数据库或业务状态；旧 Message 原样可读，旧 binding 只在当前 provider 接受其固定描述时
可恢复。唯一 workspace
迁移把 `context.prompt_sources.skills` 的准确旧 owner `skills` 改为 `standard_tools`，并保存原
TOML；缺字段、不同 owner 或结构错误分别保持不变或 fail-loud。
旧 `reply.tools` 等配置字段，以及 `disabled` 中的旧 `skills`、`agent_restart` 插件 ID 已明确
拒绝；升级者必须审阅并修改这些配置。自动迁移不推测这些字段的意图。

## 3. 模型展示与按插件加载

`ToolPresentation` 只拥有三件事：本次固定顶层 schema、把 provider 返回的 wire tool call
解码为一个真实 `ToolRef + arguments`，以及本次程序固定的 system 工具目录。`react` 只调用这个
接口，不识别 `load_tools`、间接调用工具或来源名称。

Native presentation 把获授 view 中的工具直接放入固定 schema。Tool loading presentation 固定展示
conversation 基础组中 `always_on` 的真实工具、`load_tools` 和 tool-search 插件拥有的间接
调用 schema。

`load_tools` 只接收 system 目录中的准确插件 ID。在唯一一次 `ToolResult` 文本中，它返回该 ID
在冻结获授 view 内的全部工具 schema；错误 ID 明确失败且不泄漏其他组。间接调用只可在 Tool
loading presentation 已获授的 view 中按名字解析；模型的 wrapper call 作为 `model.facts` 的 wire
replay 材料保存，Message 日志只提交解码后的真实 `ToolCall`，因此 prepare、authorize、invoke
和 receipt 都只执行一次。

`tool_search` 的 v2 查询注册已由 `load_tools` 替换，不能把旧 `query/top_k/allowed_risk` wire call
转交给新 loader。更新 selected Root 前，旧 selection 中每个 `tool_search` 的已提交 `ToolCall` 必须已有
最终 `ToolResult`；之后历史仅按已提交的真实 ToolCall/ToolResult 读取，不重新执行旧查询。未结算调用
必须在旧 selection 内完成或按既有 abandon 规则结算，不能依赖新 selection 猜测旧参数。

目录通过当前 Tool loading presentation 加入 system，按插件 ID 稳定排序，每行只显示“插件 ID ·
声明用途 · 工具数量”。固定工具已有顶层 schema，不在目录重复列出工具名。Wake、Scheduler、
Subagent 和其他 native presentation 不得到该目录。顶层 schemas 在加载前后严格相同。加载返回
一整组，不提供关键词、排序、风险过滤、截断或分页。

`model.facts` 兼容旧字段；新成功 Output 额外保存本次实际使用的 reminder replay、它所属
Input 与内容 SHA-256 摘要，以及原 wire tool calls。同一 Input 的同一材料只重放一次，当前请求复用
首次使用位置；新材料在请求末尾加入。不同材料、不同 Input 和旧的无身份事实不会因文本相同被折叠。失败或取消不保存成功
replay。新 Summary 正常开启新上下文，旧搜索结果只是
普通 ToolResult；保留原文中的 schema 继续可见，只有摘要覆盖它后才不再提供原 schema。
不专门删除 raw tail，也不改变调用权限。工具目录不写入新的 reminder replay。

模型调用格式或名称不属于当前展示时，展示层抛出 `InvalidToolCall`；ReAct 将原 wire 请求和
拒绝原因作为 `model.tool_rejection` 内容随同一个成功模型 Output 提交，不生成 binding、
真实 ToolCall、ToolResult 或 effect。Model 投影重放原请求与“调用未执行”的协议反馈。
Output 仍为 continue 并计入原步数上限，所以取消、重启或新输入不能重置纠错预算。混合响应
中的有效调用继续走唯一的 Tools 执行链。展示实现返回未获授 binding 等内部契约错误仍直接失败。

```text
┌───────────────┐     ┌───────────────────┐
│ 模型 wire 调用 │ ──▶ │ 展示协议校验与解码 │
└───────────────┘     └──────┬─────┬──────┘
                       有效 │     │ 拒绝
                 ┌─────────▼──┐ ┌▼────────────────────┐
                 │ ToolCall   │ │ model.tool_rejection │
                 │ → Tools    │ │ 原请求 + 未执行原因  │
                 │ → 回执     │ └──────────┬───────────┘
                 └─────┬──────┘            │
                       └────────┬──────────┘
                         ┌──────▼────────┐
                         │ Model 历史投影 │
                         │ → 下一次纠正   │
                         └───────────────┘
```

失败反馈、持久恢复和后台收尾统一遵循 [0063](../decisions/0063-execution-failures-have-terminal-results.md)，不再产生 unknown 工具状态。

## 4. 现有来源的工具范围

| 来源 | 新工具范围 | 保留的固定事实 |
|---|---|---|
| conversation/reply | `ALL_TOOLS` 中全部公开 provider view，加 tool-search 私有 view；只有 provider 声明的 always-on 组直接展示，其余组通过准确插件 ID 加载 | 每轮当前 Root 引用；旧调用按 binding 恢复 |
| Wake | 自己的私有决定工具、Memory recall provider、Web provider | `Request.tools` 保存原 binding；缺硬依赖不激活 |
| Scheduler | 当前全池减 message-push 与 memory 写入/召回工具 | 原允许范围不扩大；外部工具仍可用 |
| Subagent | `research`、`scripting`、`general` 三个既有 profile | prepared 时保存配置化 binding；无搜索能力 |
| Plugin validation | 请求显式 `validation_tools`，未声明时为候选 Root 的完整 view | 只在候选隔离 Session 执行 |
| Programmatic | 作为普通 Source 进入 reply，复用同一 presentation | 不新建独立工具池或来源特判 |

conversation 的 always-on provider 是 `standard_tools`、`standard_web`、`tool_search` 和
`message_push`；`tool_search` 提供 `load_tools`。`standard_tools` 拥有文件、编辑、shell、`write_stdin` 与 `load_skill`；
`standard_web` 拥有 `web_fetch/web_search`。`agent_restart` 是 `message_push` 在 supervised
runtime 且完整送达依赖就绪时挂载的可选 child，因此沿用 `message_push` provider 组，不建立
第二个插件 ID。Akasha 和外部公开 provider 默认只在按插件加载的结果中展示。

MCP `tools/list` 得到的工具必须注册进同一引用池并归到声明它的实际插件组；正式 MCP route
仍拥有进程、allowlist、恢复和 transport 错误。Akasha 的直接 Tool 与 MCP 工具对消费者都只
通过 view 暴露，不能绕回 legacy `ToolRegistry` 获得额外权限。

## 5. 验收

- 未获 view 的消费者即使猜对名称也不能 bind；获授 view 内可不经搜索直接调用。
- provider 消失导致硬依赖 consumer 不激活；已提交 binding 只在当前 provider 接受固定描述时打开，
  不得改选同名或其他工具。
- `load_tools` 以准确插件 ID 返回获授 view 内整组完整 schema；未知或未获授 ID 明确失败，system 目录与顶层 schema 加载前后相同。
- 间接调用只生成一个真实 ToolCall 和一个最终 ToolResult；wrapper wire call 可重放。
- compaction 不修改 Message、不撤销工具权限，也不把旧加载结果当成新的授权事实。
- Scheduler、Wake、Subagent、Plugin validation 和 Programmatic 保留上述范围。
- `agent_restart` 不再要求搜索，仍只在 supervisor runtime、成功送达后按原 receipt 执行。
- a1b386b7 生成并归档的 Wake Request 在候选代码中恢复后，实际产生
  `Input → Input → Output(ToolCall) → ToolResult(success) → Output(quiet)`，证明旧
  `tool_names`、旧 ReAct wire 和原 binding 仍跨升级执行；兼容入口只接受与 fixed bindings
  完全一致的旧名称集合。

## 6. 本轮交付边界

外部 provider 必须通过 `declare_group(description=...)` 声明用途并交付真实 view。
Calendar、Feed、Fitbit、Steam 和 GitHub Watch 在各自源码仓库迁移；Shell Restore 与
Shell Safety 按 `tool_key("shell")` 发布的精确 ToolRef 挂载检查；工具释放时先排空这些贡献者。
Observe 在 JSON 持久化边界递归
转换冻结参数，不能只复制最外层字典。正式交付须逐一固定这些仓库的 commit，再走安装链验证。

Core 的 `register()` 合同和使用它的外部 provider 必须作为同一 selected Root 更新：先固定全部外部
archive，再一次提交完整 components selection。不能先发布删除属性的 Core、再让旧 provider 在新 Root
调用旧签名；任一 provider 无法加载即阻断该 selection，不能用宽松参数兼容绕过。旧 `tool_search`
调用的排空也在这次 selection 切换之前完成。

线上配套恢复保持原 Message 与 receipt：Markdown 仅对可重试的模型失败保留游标，释放
lease 后延时重订阅，即使没有新消息也会再读原消息；其他错误继续显式失败。Wake 筛选工具
拒绝截短或未知的候选 ID，并允许最多三次模型输出供纠错。Skill doctor 验证链接所指归档
的完整性和技能内容，不把正常的不可变归档路径误判为缓存链接偏移。

固定顶层 schemas 与工具目录消除了旧 LRU 展示顺序对 provider 前缀的影响，对应 issue #551。
本地测试、独立评审、正式安装与线上观察分别记录证据，不能互相替代。
