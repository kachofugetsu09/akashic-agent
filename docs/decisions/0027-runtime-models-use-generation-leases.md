# 0027 · 运行时模型切换使用 execution generation lease

- 状态：accepted
- 日期：2026-08-06
- 部分勘误：[0028](0028-model-credentials-live-with-workspace-connections.md) 将模型凭据从全局 JSON 迁入 workspace connection
- 关联条款：RUN-009～RUN-012、ONB-001、CTX-001、PLG-003

## 背景

当前 Gateway 启动时从 `config.toml` 构造 main、fast、agent 和 vision provider，并把实例注入 Turn、主动流程、调度、记忆和插件。频繁变化的模型状态和端口、渠道等进程配置混在同一文件；设置服务修改模型后还会通过 Supervisor 让整个 Gateway 排空并重启。因此一次普通模型切换会阻止新 Turn，且首次没有配置时启动脚本可能在用户完成设置前结束 Supervisor。

模型能力和 usage 还存在两个平行问题：设置要求用户手填上下文窗口、多模态和输出上限；插件与部分主动流程消费私有缓存字段，没有统一的 coverage 语义。

## 决定

workspace 新增 `model-registry.sqlite3`，以 connection、model 和 role binding 三层保存模型配置；`config.toml` 只保留进程、渠道、记忆和其他静态配置。根据 0028 的勘误，模型 CredentialStore 也以同一 workspace 数据库为 backend，connection 行拥有 credential payload。

Core 的 `ModelRegistry` 在每个新执行单元开始时读取模型库最新 revision，把 `default`、`fast`、`agent` 和 `vision` 整组绑定编译为不可变 generation，再租用这一代。passive ReAct、proactive ReAct、schedule SOFT、Memory Optimizer、consolidation、plugin job 和回复后记忆任务各自是完整执行单元。执行期间出现新 revision 不改变当前租约；没有外层执行单元的单次模型调用在调用前读取最新 revision。旧代在 lease 归零后释放。模型凭据的最终 owner 由 0028 勘误为同一 workspace connection。

Supervisor 继续拥有进程和 boot 代际，但不拥有模型 generation。新增连接时可以通过独立 reload 信号要求 Gateway 立即验证并发布；普通 role binding 修改由下一执行直接读取数据库 revision。两条路径都不触发 Gateway 退出、全局 quiesce 或 Guardian 换代。首次没有配置时 Supervisor 仍拥有 bootstrap 写入，合法配置产生后再启动第一代 Gateway。

Supervisor 同时在整个生命周期内独占 `2236` Web Shell。根路径 `/` 直接提供统一 Dashboard 壳层并默认选中 Chat，不做 `/dashboard` 跳转；壳层内部 Chat 使用 `/chat`，模型与认证使用 `/settings`。Gateway 的 Chat/Dashboard API 只通过 workspace Unix socket 向 Web Shell 提供能力。`6321`、`6322` 不再监听，Web 页面不得用端口号承担路由或能力探测语义。这样 onboarding、Gateway reload 和异常退出都不会让用户入口消失。

模型能力使用“显式覆盖 → provider 权威目录 → 固定版本 LiteLLM 本地注册表 → unknown”的字段级优先级。只读取 wheel 内置快照，不让设置与启动依赖公共网络。Unknown 不猜测。Usage 使用固定版本 `genai-prices` 的 provider extractor，再补充当前统一类型仍需要的窄字段映射；解析缺口显式降级 coverage。

## 理由

- 数据库 revision 加 execution lease 直接表达“库中始终是最新状态，当前执行用旧快照，下一执行用新快照”，不依赖消息发送方猜测是否需要重启。
- Core 是 Turn、后台任务和插件共同模型语义的唯一交点；只在 Web 客户端切换会遗漏其他消费者。
- LiteLLM 只拥有 provider/model 能力归一化；保留现有传输层，避免引入其 router 的重试、fallback 和路由语义。
- 固定 wheel 的内置目录让正常启动不依赖公共网络，又能直接复用成熟 registry API。
- Supervisor 只传递 reload admission 和回执，继续保持 RUN-004 的进程所有权。

## 数据影响

- `model-registry.sqlite3`：增加 connection、model、role binding 和单调 revision。设置事务原位更新当前绑定或增加来源/模型；删除必须是独立操作，且受外键和 session 引用检查约束。
- `config.toml`：迁移前保留具名备份，迁移后删除动态模型表并写入 `registry = "workspace"` 标记；后续模型切换不得再改写该文件。
- workspace 模型 CredentialStore：新 API Key 随 connection 增加或更新；迁移把旧 inline key 或被引用的旧 JSON credential 复制入数据库，并在成功校验后从 TOML 删除动态模型配置。
- `sessions.metadata`：允许原位更新版本化 `model_selection`（model ref + reasoning effort）；清除只删除这一键，不影响消息正文。旧 `model_runtime_override` 只读兼容至下一次显式选择。
- turn 元数据与 usage：随新 Turn 追加实际 runtime/generation 和规范化 usage；既有 Turn 不回填。
- `sessions.db/messages`：保持只追加；模型切换、能力变化和上下文缩小都不得 UPDATE 或 DELETE。
- 能力快照：解析结果随 runtime 配置持久化并标记 `litellm` 来源；依赖版本固定后才允许更新，不是用户输入真源。

## 失败与回滚

候选先完成边界校验和真实 provider probe，再用一个数据库事务提交模型行和 revision。Gateway 在新执行开始前完整构造 candidate generation；失败继续使用 current。若数据库已提交但调用方未收到回执，重试读取 revision 判断是否已经提交。进程崩溃后从模型库重建，不通过内存指针推断外部效果已回滚。

源码回滚使用任务前 Git stash 恢复点或回退本分支提交。运行配置恢复使用对应 operation backup，并再次走同一数据库提交协议；不得直接修改内存 current 指针冒充持久配置已经恢复。

## 验收

- A Turn 运行中发布 B，A 的多次请求仍使用旧代，下一 Turn 使用 B。
- 排队但未开始、主动 tick、plugin job 和记忆任务在开始时取得最新代。
- 候选失败不改变配置摘要和 current generation。
- 更小窗口只改变临时 Prompt 投影，不改变完整 session 历史。
- 无配置的正式入口持续提供 onboarding，配置成功后启动 Gateway。
- 只出现一个 `2236` TCP listener；无配置时 `/` 仍是 Chat，且不存在 `6321/6322` listener。
- 使用固定 Observe 插件 revision 在一次性 workspace 观察真实 committed Turn、model binding 和 usage。

## 未选择的方案

- 每次切换重启 Gateway：会排空无关执行，不能表达 execution-local 冻结。
- 由 Web 客户端比较上一轮模型后决定重启：遗漏后台消费者，并把 Core 权威语义交给单一客户端。
- 采用完整 LiteLLM/Pydantic AI transport：会改变现有协议、重试和错误边界，超出本次目标；本决定只采用 LiteLLM 的本地元数据 registry。
