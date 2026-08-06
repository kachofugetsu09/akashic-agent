# 运行时模型注册表与 Onboarding

- 状态：implemented and verified；定向测试、正式 UI 浏览器验证、Observe 隔离实测与 Change Gate 已通过
- 日期：2026-08-06
- 决策：[0025](../decisions/0025-runtime-models-use-generation-leases.md)
- 需求：RUN-008～RUN-011、ONB-001、CTX-001

## 1. 目标与当前差距

用户可以配置多个 Provider connection，在每个 connection 下启用一个或多个模型，把模型绑定到 `default`、`fast`、`agent` 和 `vision`，并在 Gateway 运行期间修改。数据库始终保存最新 revision；已经开始的执行保持旧快照，下一执行读取新 revision。首次设置只要求登录 Codex、登录 OpenCode，或填写 Base URL、API Key 和 Model Name。

实施前 `bootstrap/providers.py` 在启动时一次性构造 provider，`bootstrap/tools.py` 随后把实例注入所有消费者。`bootstrap/settings_api.py` 写配置后调用 Supervisor restart bridge；Supervisor 向 Gateway 发送 `SIGUSR2`，`main.py` 排空全局 Turn 后退出。因此原实现没有运行时模型 owner。

## 2. 目标结构

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

`ModelRegistry` 是 generation、角色和 runtime lookup 的 Core owner。`RoleBoundProvider` 保持现有 `LLMProvider` 调用形状：属性读取和 `chat()` 都委托给当前 execution binding 中对应 runtime。没有 execution scope 的单次内部调用在调用开始时租用 current，并在返回后释放。

## 3. 执行与模型解析

执行开始时先读取模型库 revision，再创建不可变 `ExecutionBinding`：

- `generation_id` 与配置摘要；
- generation 内全部 role → runtime 映射；
- 会话可选的显式 model ref 与 reasoning effort；
- 每个 runtime 的 provider、model、credential 引用和能力快照。

对话选择解析优先级为：本次 inbound 显式 model ref/effort、session metadata、generation 的 default。显式选择只替换 default/agent 主推理，不把 fast、vision 等内部角色偷偷改成同一模型；effort 也只覆盖这个显式模型。Plugin job、proactive tick 和 memory run 不读取某个 Web session 的选择，按自身角色解析。

Turn、job 或 tick 内的所有 retry、工具 batch、summary 和 fallback 都继承 context-local binding。passive ReAct、proactive ReAct、schedule SOFT、Memory Optimizer、consolidation、plugin job 和 post-response memory job 都在各自入口建立 scope。没有外层 scope 的单次 LLM 调用在调用前读取最新 revision。数据库新 revision 只改变后续 scope；旧代进入 retiring，最后一个 lease 释放后移出 registry 引用并由 Python 回收。

## 4. 设置提交和进程边界

Supervisor 在启动任何 Gateway 之前先取得 `2236`，并在整个进程生命周期内持有该监听。它直接提供统一 Dashboard 静态壳层、Chat/设置静态页面与 settings transaction；Gateway 的 Chat/Dashboard API 只绑定 workspace 内的 owner-scoped Unix socket，由 Web Shell 同源转发。系统不再监听 `6321` 或 `6322`。

无配置时 `/` 仍直接渲染统一 Dashboard 壳层并默认选中内嵌 Chat，地址栏不跳转。Chat 在读取 Web Shell readiness 后不连接 Gateway WebSocket，不读取会话，也不制造“连接失败”；空状态显示“连接模型”，发送区禁用。`/settings` 使用同一前端构建和同源设置 API。首次合法保存会在候选 `Config.load` 后初始化 VEDA、SessionDB 等缺失的 workspace 基线，再由 Supervisor 启动 Gateway；运行中修改模型不重复执行首次初始化。readiness 成功后原页面重新探测并解锁聊天。Gateway ready 后：

1. 设置服务读取模型库 revision 并拒绝陈旧写入。
2. 使用临时候选凭据执行真实 model probe。
3. 创建 operation backup，用一个 SQLite 事务提交 connection/model/role 和新 revision；secret 只写 CredentialStore。
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

Custom API 的 transport provider 继续是 `openai` 兼容协议，注册表另存 `catalog_provider_id`。已知 Base URL 可把它解析成 `deepseek`、`openrouter` 等 usage/catalog 身份，但不得借此更换 wire transport 或重试策略。

## 6. Usage 归一化

Chat Completions 与 Responses transport 把完整 response payload 交给统一 extractor。固定 `genai-prices==0.0.71` 根据 provider id/API URL 和 `chat`/`responses` flavor 提取 input、cache read、cache write 和 output。Reasoning output 使用响应中的明确 provider detail 窄映射补齐。

`ModelUsage` 增加 `cache_write_input_tokens`，并保留 request/covered request/coverage。聚合规则只对已知字段求和；任一请求未覆盖时总 coverage 至少为 partial。插件 `generate()` 返回正文与 usage 的结构化结果；`generate_text()` 作为兼容便捷方法调用它，不再丢弃 usage。旧 `cache_prompt_tokens/cache_hit_tokens` 只能由 normalized usage 派生，逐步移除私有 tool-call 字段。

Extractor 找不到 provider、flavor 或字段时只捕获明确的 `LookupError`、`TypeError` 或 `ValueError`，记录 normalizer unavailable，再使用已声明的本地窄映射；仍无数据则返回 unavailable。解析失败不应让已经成功的模型回复变成失败，也不得生成假零。

## 7. Onboarding 与 Chat UI

设置页按具名 Provider connection 展示多套账号或 API Key，并提供 Codex、OpenCode、DeepSeek、OpenRouter 模板。API 连接先填写连接名称、Provider ID、Base URL 和 API Key，再通过 provider `/models`、Codex/OpenCode 权威目录发现 model；目录不可用时才手工填写 Model Name。模型识别后展示 effort 等能力，unknown 字段保持未知。

Chat composer 上方只保留一个等宽向上展开胶囊，显示“model：来源”并按 Provider connection 分组滚动。切换在发送下一条消息时随 inbound frame 提交；服务端校验后更新 `sessions.metadata.model_selection = {schema_version, model_ref, reasoning_effort}`。当前 active Turn 不受影响。重新打开 session 时从服务端读取选择；选择“跟随默认”删除该对象。旧 `model_runtime_override` 字符串继续只读兼容，并在下一次显式选择时转成新结构。

Web 导航与路由固定为：

```text
2236 /（统一 Dashboard 壳层，默认选中 Chat，地址不跳转）
├── /chat?embedded=1                 壳层内 Chat
├── /chat?embedded=1&surface=runtime 壳层内知识与运行
├── /settings?embedded=1             壳层内模型与认证
└── /dashboard                       旧链接兼容，同样返回统一壳层
```

页面之间只使用同源路径。前端不拼接端口，不使用 iframe 端口探测；设置成功回到根路径 `/` 即可在统一壳层中对话。

### 7.1 多凭据与来源身份

Memoh 的可复用部分是 owner 模型，不是整页复制：一套登录或 API Key 对应一个具名 Provider 实例，模型以 `provider_id + model_id` 唯一。它的数据层允许相同 `client_type` 的多个实例，只要求实例名称不同；现有预设画廊会隐藏已经配置过的模板，因此“重复添加同一预设”在 UI 上仍不够直接。Akashic 采用前者并补齐入口：同一 Provider 可以继续添加主账号、备用账号或不同网关，不把多个 secret 塞进同一 runtime。

设置 API 需要从当前“按 Provider 类型派生固定 runtime ID”升级成具名 runtime 的 create/update 操作。每项至少持有稳定 `runtime_id`、用户可读 `source_name`、transport/provider、credential 引用、Base URL、model name、默认 reasoning effort 与能力快照。列表和 Chat 都以 `model name：source name` 显示，身份和提交仍使用稳定 runtime ID，不能用显示文字做主键。

API Key 是 write-only：读取状态只返回掩码和凭据状态，空值保存表示保留原 secret；替换 key 必须显式输入新值。Codex 和 OpenCode 使用同一来源实例外壳，但表单切换成登录动作，不伪装成 API Key 字段。新增来源成功后显示非阻塞 Toast；表单用带 backdrop blur 的模态层或侧栏，Portal 到 `document.body`，支持 Escape、焦点圈定和焦点恢复。

### 7.2 五个等权交互候选

这五版只用于选择交互方向，当前无推荐顺序：

| 编号 | Chat 选模 | 设置页 | 主要取舍 |
|---|---|---|---|
| 01 悬浮坞 | 输入框上方胶囊，向上展开带搜索列表 | 来源卡片 + 居中表单 | 模型上下文最完整，占用纵向空间较多 |
| 02 双段胶囊 | 模型与 reasoning effort 分段显示 | 行式清单 + 右侧表单 | 强度更直达，胶囊信息密度较高 |
| 03 卡片叠层 | 候选按卡片逐层展开 | 来源索引 + 主从详情 | 来源关系明显，大量模型时需搜索辅助 |
| 04 指令面板 | 搜索优先的 command menu | 可搜索清单 + 紧凑表单 | 键盘效率高，对首次用户提示要求更高 |
| 05 等宽上展胶囊（用户选择） | 输入框上方唯一胶囊，固定下沿和左右边界，只增加高度向上展开；内部按供应商分组纵向滑动 | 单列来源 + 分步表单 | 只保留一个模型入口，来源与同名模型仍可区分 |

独立预览入口为 `/chat?preview=model-experience`，只使用演示数据，不调用设置 API、不保存或发送凭据。五版都使用 brief spring 入场、选择反馈和 `prefers-reduced-motion`；高频切换不使用持续或循环动画。

05 修订版删除 composer 内部的第二个模型胶囊。唯一胶囊固定在输入框上方：桌面高 44px、宽 320～420px；展开前后宽度完全一致，只把高度增加到约 420px，稳定显示 5～6 个模型行。移动端宽度与输入框一致，展开高度不超过约 `62vh`。展开层的下沿和左右边界保持同一锚点，滚动只发生在模型列表，使用 sticky Provider 标题和 overscroll containment；点击模型后更新胶囊并收回，Escape 只关闭展开层，当前 Turn 语义不变。

Provider/模型 Logo 候选使用 MIT 的 `@lobehub/icons`，已覆盖 Codex、DeepSeek、OpenAI、OpenCode、OpenRouter 等。生产构建应固定 npm 版本并本地打包，不依赖 CDN；无法识别的来源回退到稳定首字母标识。Memoh 仓库与其内置图标整体是 AGPL-3.0，本任务不复制其 SVG。

## 8. 持久状态

| 对象 | 增加 | 原位更新 | 逻辑失效 | 物理减少 | 恢复证据 |
|---|---|---|---|---|---|
| model connection/model/role | 用户保存新来源或模型 | 修改来源、能力快照或角色绑定并增加 revision | 旧 revision 被新 revision supersede | 仅独立删除操作；外键和引用存在时拒绝 | operation backup + SQLite integrity check |
| CredentialStore | 新登录或新 API Key | 同 auth id 刷新 token/key | 旧 credential generation supersede | 本任务不删除 | credential backup + auth probe |
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
- Codex token 刷新或 OpenCode auth 文件变化：下一个 generation/请求从各自 credential owner 读取；不把 token 内容写入模型目录或 session。
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
