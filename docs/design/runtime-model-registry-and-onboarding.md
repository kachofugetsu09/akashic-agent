# 运行时模型注册表与 Onboarding

- 状态：现行实现与历史验收基线；Core `ModelRegistry` owner 已由 [0050](../decisions/0050-model-revision-lives-in-ordinary-plugin.md) 勘误，不再是目标实现
- 日期：2026-08-06
- 决策：[0027](../decisions/0027-runtime-models-use-generation-leases.md)、[0028](../decisions/0028-model-credentials-live-with-workspace-connections.md)；目标 owner 见 [0050](../decisions/0050-model-revision-lives-in-ordinary-plugin.md)
- 需求：RUN-009～RUN-012、ONB-001、CTX-001

## 1. 目标与当前差距

用户可以配置多个 Provider connection，在每个 connection 下启用一个或多个模型，把模型绑定到 `default`、`fast`、`agent` 和 `vision`，并在 Gateway 运行期间修改。数据库始终保存最新 revision；已经开始的执行保持旧快照，下一执行读取新 revision。首次设置先登录 Codex、登录 OpenCode，或填写 Base URL、API Key 和 Model Name；随后明确选择启用记忆、使用哪种引擎，或暂不启用。

实施前 `bootstrap/providers.py` 在启动时一次性构造 provider，`bootstrap/tools.py` 随后把实例注入所有消费者。`bootstrap/settings_api.py` 写配置后调用 Supervisor restart bridge；Supervisor 向 Gateway 发送 `SIGUSR2`，`main.py` 排空全局 Turn 后退出。因此原实现没有运行时模型 owner。

## 2. 已实现的旧结构

```text
浏览器只访问 http://127.0.0.1:2236
               │
               ▼
┌─────────────────────────────────────┐
│ Supervisor Web Shell                │
│ / Chat · /settings 模型配置         │
│ 无配置/Gateway 重载或退出时仍存活   │
└──────────────┬──────────────────────┘
               │ workspace Unix socket
               ▼
┌────────────────────────────┐
│ workspace model registry DB│
│ connection/model/role/rev  │
└──────────────┬─────────────┘
               │ each execution start
               ▼
┌────────────────────────────┐
│ Gateway ModelRegistry      │
│ immutable leased generation│
└────────┬───────────┬───────┘
         │ lease      │ role proxy
         ▼            ▼
┌────────────────┐  ┌────────────────────┐
│ExecutionBinding│  │default/fast/agent/ │
│generation +    │  │vision provider view│
│session override│  └────────────────────┘
└───────┬────────┘
        ▼
 Turn / Proactive / Schedule / Plugin / Memory
```

本设计实施时，`ModelRegistry` 是 generation、角色和 runtime lookup 的 Core owner。`RoleBoundProvider` 保持现有 `LLMProvider` 调用形状：属性读取和 `chat()` 都委托给当前 execution binding 中对应 runtime。没有 execution scope 的单次内部调用在调用开始时租用 current，并在返回后释放。0050 已替换这一 owner；本节只用于迁移差分和回归基线，不得作为新实现路线。

## 3. 执行与模型解析

执行开始时先读取模型库 revision，再创建不可变 `ExecutionBinding`：

- `generation_id` 与配置摘要；
- generation 内全部 role → runtime 映射；
- 会话可选的显式 model ref 与 reasoning effort；
- 每个 runtime 的 provider、model、credential 引用和能力快照。

对话选择解析优先级为：本次 inbound 显式 model ref/effort、session metadata、generation 的 default。显式选择只替换 default/agent 主推理，不把 fast、vision 等内部角色偷偷改成同一模型；effort 也只覆盖这个显式模型。Plugin job、proactive tick 和 memory run 不读取某个 Web session 的选择，按自身角色解析。

Turn、job 或 tick 内的所有 retry、工具 batch、summary 和 fallback 都继承 context-local binding。passive ReAct、proactive ReAct、schedule SOFT、compaction、Markdown profile projection 和 plugin job 都在各自入口建立 scope。没有外层 scope 的单次 LLM 调用在调用前读取最新 revision。数据库新 revision 只改变后续 scope；旧代进入 retiring，最后一个 lease 释放后移出 registry 引用并由 Python 回收。

## 4. 设置提交和进程边界

Supervisor 在启动任何 Gateway 之前先取得 `2236`，并在整个进程生命周期内持有该监听。它直接提供统一 Dashboard 静态壳层、Chat/设置静态页面与 settings transaction；Gateway 的 Chat/Dashboard API 只绑定 workspace 内的 owner-scoped Unix socket，由 Web Shell 同源转发。系统不再监听 `6321` 或 `6322`。

无配置时 `/` 仍直接渲染统一 Dashboard 壳层并默认选中内嵌 Chat，地址栏不跳转。Chat 在读取 Web Shell readiness 后不连接 Gateway WebSocket，不读取会话，也不制造“连接失败”；空状态显示“连接模型”，发送区禁用。`/settings` 使用同一前端构建和同源设置 API。首次合法保存会在候选 `Config.load` 后初始化 VEDA、SessionDB 等缺失的 workspace 基线，再由 Supervisor 启动 Gateway；运行中修改模型不重复执行首次初始化。readiness 成功后原页面重新探测并解锁聊天。Gateway ready 后：

1. 设置服务读取模型库 revision 并拒绝陈旧写入。
2. 使用临时候选凭据执行真实 model probe。
3. 创建 operation backup，用一个 SQLite 事务提交含 credential payload 的 connection、model、role 和新 revision；数据库与备份均按 secret 使用 `0600`。
4. 新执行开始时读取新 revision、构造完整 candidate generation 并原子 publish。
5. candidate 构建失败时保留 current，并让本次执行明确失败；修复后的下一执行再次读取数据库。
6. 首次 onboarding 保存后，设置服务仍通知 Supervisor 启动第一代 Gateway；普通 role binding 修改不触发进程重启。

`SIGUSR2` 和 restart commit 协议继续只用于正式 Gateway 换代。模型设置不得调用 quiesce、退出码或 Guardian cleanup。

并发设置由设置 owner 串行，并使用 expected revision 拒绝 stale writer。调用方在响应丢失时重新读取 state；revision 已增加表示提交已经持久化，不能盲目重复写入凭据。

## 5. 模型能力注册表

固定 `litellm==1.95.0` 只作为本地能力 registry，不接管请求 transport、重试、fallback 或错误类型。设置进程从 wheel 内置 `model_cost` 读取 max input、output、vision/modalities、reasoning effort、tool call 和 parallel tool call；内部总上下文由 max input 与 max output 相加，避免把输入上限误当总预算。`LITELLM_LOCAL_MODEL_COST_MAP=True` 禁止运行时联网刷新。版本更新必须作为依赖变更评审，避免上游变化静默进入运行配置。

LiteLLM 曾发生 `1.82.7`、`1.82.8` PyPI 供应链事故，因此这两个版本明确禁止。当前依赖精确固定到带 GitHub tag 的 `1.95.0`；升级 Gate 必须同时核对 release tag、wheel 版本、registry contract tests 和依赖安全报告，不能改成范围版本。

字段级来源优先级：

1. 用户打开高级设置后显式覆盖；
2. Codex/OpenCode 或目标 provider 的权威目录；
3. 固定版本 LiteLLM 本地注册表；
4. unknown。

通用 Base URL 先由 `genai-prices` 的 provider API 正则识别，随后用 LiteLLM 的 provider/model key 解析能力。无法确定 provider/model 时保持 unknown。Unknown runtime 的 `context_window=0`、`max_output_tokens=0`、文本输入基线只代表当前请求形状，不声明模型没有其他能力；UI 显示“能力未知”，Core 关闭主动压缩和硬预算，让 provider 错误保持原义。任意自建网关可隐藏或改名上游模型，不能仅凭 Base URL 保证识别；这种情况不得猜测。

模型注册表中的 `context_window`、`max_output_tokens` 和各自字段级 source 是运行时预算
owner 的 capability snapshot。`model_definitions` 遗留的
`effective_context_percent`/`compaction_trigger_percent` 列只为 v1 schema identity 保留，
读写完全惰性，不参与配置、能力解析、generation 或 session compaction policy；新的
model/config flow 不得把它们当作有效能力。

Custom API 的 transport provider 继续是 `openai` 兼容协议，注册表另存 `catalog_provider_id`。已知 Base URL 可把它解析成 `deepseek`、`openrouter` 等 usage/catalog 身份，但不得借此更换 wire transport 或重试策略。

## 6. Usage 归一化

Chat Completions 与 Responses transport 把完整 response payload 交给统一 extractor。固定 `genai-prices==0.0.71` 根据 provider id/API URL 和 `chat`/`responses` flavor 提取 input、cache read、cache write 和 output。Reasoning output 使用响应中的明确 provider detail 窄映射补齐。

`ModelUsage` 增加 `cache_write_input_tokens`，并保留 request/covered request/coverage。聚合规则只对已知字段求和；任一请求未覆盖时总 coverage 至少为 partial。插件 `generate()` 返回正文与 usage 的结构化结果；`generate_text()` 作为兼容便捷方法调用它，不再丢弃 usage。ReAct 的 `cache_prompt_tokens/cache_hit_tokens` 展示统计只从 normalized usage 派生，不再由模型响应保存第二份事实。

Extractor 找不到 provider、flavor 或字段时只捕获明确的 `LookupError`、`TypeError` 或 `ValueError`，记录 normalizer unavailable，再使用已声明的本地窄映射；仍无数据则返回 unavailable。解析失败不应让已经成功的模型回复变成失败，也不得生成假零。

## 7. Onboarding 与 Chat UI

设置页按具名 Provider connection 展示多套账号或 API Key，并提供 Codex、OpenCode、DeepSeek 和自定义 API 四个互斥入口。入口决定后续认证语义：Codex 只显示订阅登录，OpenCode 只显示本机登录或其 API Key，DeepSeek 只显示预填官方地址的 API 表单，自定义 API 显示全空表单；进入表单后不再二次切换认证类型。API 连接先填写连接名称、Provider ID、Base URL 和 API Key，再通过 provider `/models` 发现 model；目录不可用时才手工填写 Model Name。Codex 与 OpenCode 保存后从权威目录自动同步模型，无需用户先选模型。模型识别后展示 effort 等能力，unknown 字段保持未知。

Chat composer 上方只保留一个等宽向上展开胶囊，显示“model：来源”并按 Provider connection 分组滚动。模型列表底部固定一行“思考强度”；点击后在同一宽度和高度内切换到当前模型支持的 effort 二级列表，返回按钮恢复模型列表。选择模型或 effort 时面板保持展开，点击外部或 Escape 收起；不兼容的旧 effort 按“模型默认 → medium → 第一项”选择可用值。切换在发送下一条消息时随 inbound frame 提交；服务端校验后更新 `sessions.metadata.model_selection = {schema_version, model_ref, reasoning_effort}`。当前 active Turn 不受影响。重新打开 session 时从服务端读取选择；选择“跟随默认”删除该对象。旧 `model_runtime_override` 字符串继续只读兼容，并在下一次显式选择时转成新结构。

Web 导航与路由固定为：

```text
2236 /（统一 Dashboard 壳层，默认选中 Chat，地址不跳转）
├── /chat?embedded=1                 壳层内 Chat
├── /chat?embedded=1&surface=runtime 壳层内知识与运行
├── /settings?embedded=1             壳层内模型与认证
└── /dashboard                       旧链接兼容，同样返回统一壳层
```

页面之间只使用同源路径。前端不拼接端口，不使用 iframe 端口探测；设置成功回到根路径 `/` 即可在统一壳层中对话。

### 7.1 首次使用的完整交互

首次使用只建立两个连续决定，不把模型角色、能力字段、Embedding 和 Akasha Inspector 同时堆到第一屏：

```text
┌──────────────────────┐
│ Chat：尚未连接模型   │
│ 唯一主操作：连接模型 │
└──────────┬───────────┘
           ▼
┌──────────────────────────────┐
│ 选择连接方式                 │
│ Codex / OpenCode / API 服务  │
└──────────┬───────────────────┘
           │ 保存并同步模型
           ▼
┌──────────────────────────────┐
│ 选择记忆方式                 │
│ 启用 Akasha / 暂不启用       │
└───────┬──────────────┬───────┘
        │启用          │关闭
        ▼              ▼
┌────────────────┐  ┌────────────────────┐
│ 添加并验证     │  │ 隐藏 Embedding 与  │
│ 向量模型       │  │ Akasha 专属 UI     │
└───────┬────────┘  └─────────┬──────────┘
        └──────────┬──────────┘
                   ▼
             ┌──────────┐
             │ 进入对话 │
             └──────────┘
```

| 状态 | 页面重点 | 可见操作 | 失败语义 |
|---|---|---|---|
| 无模型连接 | 一组连接方式 | 选择 Codex、OpenCode 或 API 服务 | 不连接 Chat WebSocket，不显示原始网络错误 |
| 登录或验证中 | 当前连接表单 | 完成授权、检测模型、保存 | 错误留在当前模态框，可修正后重试 |
| 已有模型、未决定记忆 | Akasha 状态 | 启用 Akasha、暂不启用 | 关闭记忆不要求向量模型 |
| 已选择记忆、无向量模型 | 向量模型必填区 | 添加、验证并选择向量模型 | 完成按钮可点击；点击后把焦点移到缺失字段并明确报错 |
| 向量模型验证失败 | 添加向量模型对话框 | 修正 Base URL、API Key 或模型名 | 错误显示在遮罩上方的对话框内，不丢失输入 |
| 记忆关闭 | 普通聊天 | 之后可从设置重新启用 | 不渲染 Embedding、Akasha Inspector 或每轮 Akasha 召回入口 |
| Akasha 启用 | 普通聊天与 Akasha 能力入口 | 查看当前召回与 Inspector | 插件只在当前 memory owner 为 Akasha 时声明 UI 可用 |
| Gateway 启动或重载 | Chat 的准备状态 | 查看设置、等待自动恢复 | 2236 保持可用，不要求换端口或反复刷新 |
| 配置需修复 | 独立修复状态 | 阅读具体错误并修复 | 不用默认值、空目录或假成功进入聊天 |

页面只保留能改变当前任务的表面：连接卡片表示一套可选择、可编辑的凭据来源，因此保留；记忆方式是互斥决定，因此使用选择行；聊天模型胶囊表示当前会话下一轮的模型与 effort，因此保留。说明文字、装饰徽章和嵌套卡片不取得独立表面。主色只表示主操作和当前选择，绿色只表示已验证或已连接，错误色只表示失败或缺失必填项。

### 7.2 多凭据与来源身份

Memoh 的可复用部分是 owner 模型，不是整页复制：一套登录或 API Key 对应一个具名 Provider 实例，模型以 `provider_id + model_id` 唯一。它的数据层允许相同 `client_type` 的多个实例，只要求实例名称不同；现有预设画廊会隐藏已经配置过的模板，因此“重复添加同一预设”在 UI 上仍不够直接。Akashic 采用前者并补齐入口：同一 Provider 可以继续添加主账号、备用账号或不同网关，不把多个 secret 塞进同一 runtime。

设置 API 需要从当前“按 Provider 类型派生固定 runtime ID”升级成具名 runtime 的 create/update 操作。每项至少持有稳定 `runtime_id`、用户可读 `source_name`、transport/provider、credential 引用、Base URL、model name、默认 reasoning effort 与能力快照。列表和 Chat 都以 `model name：source name` 显示，身份和提交仍使用稳定 runtime ID，不能用显示文字做主键。

API Key 是 write-only：读取状态只返回掩码和凭据状态，空值保存表示保留原 secret；替换 key 必须显式输入新值。Codex 和 OpenCode 使用同一来源实例外壳，但表单切换成登录动作，不伪装成 API Key 字段。新增来源成功后显示非阻塞 Toast；表单使用 Portal 到 `document.body` 的模态层，支持 Escape、焦点圈定和焦点恢复。遮罩只表达模态边界，不用模糊或装饰性渐变制造层级。

### 7.3 五个等权交互候选

这五版只用于选择交互方向，当前无推荐顺序：

| 编号 | Chat 选模 | 设置页 | 主要取舍 |
|---|---|---|---|
| 01 悬浮坞 | 输入框上方胶囊，向上展开带搜索列表 | 来源卡片 + 居中表单 | 模型上下文最完整，占用纵向空间较多 |
| 02 双段胶囊 | 模型与 reasoning effort 分段显示 | 行式清单 + 右侧表单 | 强度更直达，胶囊信息密度较高 |
| 03 卡片叠层 | 候选按卡片逐层展开 | 来源索引 + 主从详情 | 来源关系明显，大量模型时需搜索辅助 |
| 04 指令面板 | 搜索优先的 command menu | 可搜索清单 + 紧凑表单 | 键盘效率高，对首次用户提示要求更高 |
| 05 等宽上展胶囊（用户选择） | 输入框上方唯一胶囊，固定下沿和左右边界，只增加高度向上展开；内部按供应商分组纵向滑动 | 单列来源 + 分步表单 | 只保留一个模型入口，来源与同名模型仍可区分 |

模型体验候选预览已退出独立 showcase；模型选择与认证改在正式 Settings 与生产 Chat composer 验收。对话壳层方向预览改走 `/?preview=chat-product`。

05 修订版删除 composer 内部的第二个模型胶囊。唯一胶囊固定在输入框上方：桌面高 44px、宽 320～420px；展开前后宽度完全一致，只把高度增加到约 420px，稳定显示 5～6 个模型行。移动端宽度与输入框一致，展开高度不超过约 `62vh`。展开层的下沿和左右边界保持同一锚点，滚动只发生在模型列表，使用 sticky Provider 标题和 overscroll containment；点击模型或 effort 后更新胶囊但保持展开，点击外部或 Escape 才关闭，当前 Turn 语义不变。

Provider/模型 Logo 候选使用 MIT 的 `@lobehub/icons`，已覆盖 Codex、DeepSeek、OpenAI、OpenCode、OpenRouter 等。生产构建应固定 npm 版本并本地打包，不依赖 CDN；无法识别的来源回退到稳定首字母标识。Memoh 仓库与其内置图标整体是 AGPL-3.0，本任务不复制其 SVG。

## 8. 持久状态

| 对象 | 增加 | 原位更新 | 逻辑失效 | 物理减少 | 恢复证据 |
|---|---|---|---|---|---|
| model connection/model/role | 用户保存新来源或模型 | 修改来源、能力快照或角色绑定并增加 revision | 旧 revision 被新 revision supersede | 仅独立删除操作；外键和引用存在时拒绝 | operation backup + SQLite integrity check |
| model connection credential | 新登录或新 API Key 随 connection 增加 | 同 auth id 原位刷新 token/key；Codex refresh 不增加模型 revision | 新 payload supersede 旧 payload | 只随以后独立来源删除；本任务不删除 | SQLite operation backup + auth probe + `0600` mode |
| session model selection | 首次固定 model ref/effort | 切换 model 或 effort | 清除 selection 后跟随 default | 只删除 metadata 单键；消息不变 | sessions.db 完整消息快照 |
| turn binding/usage | 新 Turn 提交时追加 | terminal metadata 按既有协议更新 | 后续 Turn 使用新代 | 不自动删除 | turn/message join + Observe DB |
| messages | 新 Turn 原子 INSERT | 不允许 | 不适用 | 仅用户撤销/删除会话 | SQLite backup + full snapshot |
| capability snapshot | 保存 runtime 时从固定 wheel 派生 | 用户换模型时更新 | 新 runtime generation supersede 旧快照 | 仅随独立 runtime 删除；本任务不自动删除 | config backup + dependency pin + registry test |
| 首次 workspace 基线 | 首次合法模型配置后创建缺失的 VEDA、数据库与目录 | 本任务不覆盖既有基线 | 不适用 | 本任务不减少 | init summary + 文件/DB 完整性检查 |

## 9. Edge cases

- A 已开始、设置改成 B：A 的后续 tool loop 仍是 A，下一执行是 B。
- 请求已进入队列但未取得 execution lease：开始时使用 B。
- 同时修改 provider、key、model 和角色：作为一个 candidate generation 提交，不允许混合观察。
- candidate probe 成功但 publish 前崩溃：模型库 revision 是真源；重启或下一执行重新读取，调用方通过 revision 确认结果。
- candidate 构建失败：旧 current 继续服务，设置写入恢复 backup。
- 新模型窗口更小：只重新计算临时 Prompt；历史消息不改写。
- unknown/目录消失：已有显式覆盖继续；没有覆盖则保持 unknown，不沿用另一个模型的能力。
- 流式响应没有 usage：正文成功，coverage unavailable。
- 一次 Turn 多次请求且部分无 usage：request_count 全量，covered_request_count 只计已覆盖请求，总 coverage partial。
- Session override 指向后来删除的 runtime：设置删除必须先拒绝仍被引用的 runtime；本任务不做 cascade。
- 会话选择的 effort 不在模型支持集合：发送边界明确失败，不降级到默认 effort；fast/vision 等内部角色不继承会话 effort。
- Codex token 刷新原位更新 workspace connection；OpenCode 本机登录只作为 onboarding 导入源。迁移后的模型请求只从 workspace 数据库读取，不把 token 内容写入 session、设置状态或 Observe。
- 首次设置耗时超过启动脚本旧等待窗口：Supervisor 保持存活并明确输出设置 URL。
- 无 `config.toml`：2236 返回 Chat 壳层和 `needs_setup`，不创建假 session、不连接不存在的 WebSocket。
- config 已写但 Gateway 仍在启动：Chat 显示“正在启动”，保留设置入口并按有界间隔探测 readiness。
- Gateway 换代或异常退出：2236 不掉线；现有页面显示“正在重新连接”，恢复后重新拉取 session/model state。
- 遗留配置仍包含 `channels.chat.host/port`：加载可以兼容，但字段不再产生监听；新配置不再写入 6322。

## 10. 实施与验证

1. 引入 registry、execution scope、role proxy 和 generation reload 回执。
2. 把 Turn、plugin job、proactive tick 和 memory run 接入 scope。
3. 用固定 LiteLLM registry 加入能力解析和简化设置 API/UI。
4. 加入 Chat runtime selector 与 session metadata。
5. 用 `genai-prices` 统一 transport usage，并升级插件结果契约。
6. 运行针对性单测、前端检查、静态检查和 Change Gate。
7. 从 canonical Observe 仓库固定 revision 安装到一次性 plugin home，在一次性 workspace 产生真实 committed Turn；从 lifecycle event 核对 model binding，并从 Observe SQLite 核对稳定 message identity 和 usage。

验证不得连接正式 workspace、正式插件 cache 或正式 2236。隔离容器只映射一个随机 host port 到容器内 2236，并断言容器内没有 6321/6322 listener。Observe 报告同时固定 Core tree、插件 requested ref/resolved commit、安装目录摘要、scenario profile 和报告摘要。

### 10.1 隔离实测结果

2026-08-07 使用随机端口 `22367`、一次性 workspace 和一次性 plugin home 验证；正式 workspace、正式插件 cache 和正式 `2236` 未写入。Observe 从 canonical 仓库固定到 commit `4d85b9dc64ef0d8d96c5a635586ca17dd94b59cd` 安装。

- 一个已取得 generation 2 lease 的 ReAct 在两次 provider 请求之间把数据库 `agent` role 从 gate-a 更新为 gate-b；该 ReAct 的请求序列仍为 gate-a、gate-a，下一 ReAct 为 gate-b。
- 第一 ReAct 的 terminal metadata 与 Observe SQLite 都记录 input 21、cache hit 7、output 5、request count 2；第二 ReAct记录 input 13、cache hit 6、output 4、request count 1，coverage 均为 exact。
- 浏览器从所有来源中把会话固定到 gate-a 后实际发送成功；模型网关收到 gate-a。刷新 Chat、重新进入原会话后，服务端 `sessions.metadata.model_selection` 和胶囊都恢复 gate-a；新会话仍跟随当前 default gate-b。
- 随机端口首次暴露 settings iframe 的 `frame-ancestors` 硬编码问题；CSP 已改为同源 `'self'`，保留 `5173` 本地开发壳层，随机端口 Dashboard 复验通过。
- Change Gate 通过，报告目录为 `docker/debug/reports/change-gate/20260807-015529-0c00821a`。

### 10.2 SQLite 凭据与真实 GUI 回归

2026-08-07 另用全新容器、一次性 HOME/workspace 和随机 host 端口 `22368` 回归本轮 0028 与 UI；容器内仍只监听 `2236`，正式 workspace、正式插件 cache 和正式 `2236` 均未写入。API Key 取自维护者明确授权的源码配置，只经浏览器表单提交；Codex 复用现有登录凭据作为隔离测试夹具，模型发现、保存、选择和对话均从 GUI 完成。

- 无 `config.toml` 时根路径直接显示统一 Chat 空状态和“连接模型”，不会先挂载未就绪的 Dashboard runtime iframe，也不再弹出 503 错误框。
- GUI 保存一个 OpenCode Go connection 下的 `deepseek-v4-flash`、`deepseek-v4-pro`，以及 DeepSeek API 的 `deepseek-v4-flash`；容器没有 `opencode` CLI 时明确记录目录降级，并从固定 LiteLLM 本地注册表恢复三个模型支持的 `low/medium/high`，不猜测远端状态。
- Codex GUI 目录发现 `gpt-5.6-luna` 及其 `low/medium/high/xhigh/max`，保存后 API state 只返回 `codexConfigured=true`，不返回 token。
- 同一 Web session 依次从胶囊选择并完成四个真实 Turn：OpenCode Flash/high、OpenCode Pro/low、DeepSeek/medium、Codex Luna/high。胶囊 trigger 为 `412px`、展开层为 `418px`（3px shell inset），模型和 effort 选择后保持展开，Escape 收起并恢复焦点；重新打开该 session 后恢复 Luna/high。
- SessionDB 四个 terminal Turn 均为 `coverage=exact`：input 分别为 `12452/12388/12814/11999`，cache hit 为 `4992/4992/4992/0`，output 为 `80/32/11/21`，reasoning output 为 `70/23/0/8`。DeepSeek 的独立 reasoning token 为 `0`，表示该响应没有返回可归一化的独立字段，不影响 `medium` effort 已随 inbound metadata 提交。
- Observe 固定安装并晋升 commit `4d85b9dc64ef0d8d96c5a635586ca17dd94b59cd`；从该 stable snapshot 冷启动后记录同四个 session/turn，output、prompt 与 cache hit 和 SessionDB 完全一致。聚合为 `turn_count=4`、`tracked_turn_count=4`、`prompt_tokens=49653`、`hit_tokens=14976`。
- `model-registry.sqlite3` 与 credential lock 均为 `0600`；三类 connection 的 secret 只在 SQLite JSON payload 中，隔离 `config.toml` 不含 API Key、token、Provider URL 或模型表。
