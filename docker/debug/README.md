# Docker 调试沙盒

## 统一变更影响 Gate

实现者只需运行一个公开入口：

```bash
python docker/debug/gate.py run --base origin/main
```

Gate 先用 `tests_scenarios/contracts/impact.toml` 解释 Git diff，再运行所选公开语义场景。每个场景都使用新的 `/tmp/akashic-change-gate-*` sandbox，容器只读挂载候选源码，只允许写本次 `/sandbox` 与 tmpfs `/tmp`。`workspace`、`plugin-home`、`HOME` 和 config 都从空目录建立；Gate 不接收正式运行路径。

```text
Git diff
   │
   ▼
公开 capability/state/scenario catalog
   │
   ├── 公开 Docker 场景（所有贡献者可运行）
   └── plan.json：group + digest，不含 provider 身份
```

常用维护命令：

```bash
python docker/debug/gate.py audit
python docker/debug/gate.py plan --base origin/main
```

如果同一 diff 同时包含生产 source set 与 protected contract/policy paths，`plan` 和 `run` 会扩大为完整公开场景，同时仍分别列出两组路径。未知可执行改动和触及 baseline gap 仍以非零退出。`migrations/**` 本身不在该 protected 集合内；已注册的 `migrations/yoyo/*.py` 由精简的 append-only 检查保护。

`init` 只用于仓库第一次建立 coverage baseline。baseline 已存在时再次执行会失败，不能覆盖人工合同。新增未映射可执行文件会先运行全量公开语义场景，最终仍以 `unmapped_change` 失败。报告位于 `docker/debug/reports/change-gate/<run-id>/`。

公开 Gate 不安装也不枚举私有插件，不依赖外部私有验证或 provider 身份；公开报告是当前仓库的合并依据。

## Citation + Meme 纯 v3 WebUI Gate

Gate 用 exact commits 和 installed stable artifact
布局启动 supervised Gateway，只保留 WebUI channel，经公开 WebSocket 完成一轮回复，再从
公开 HTTP 读取消息、媒体、Dashboard 与 capability。它同时核对模型 prompt 中
Citation→Meme 顺序、SessionDB、artifact 前后摘要以及 Compose 零残留。
`plugin_passive_composition_v3_gate.py` 仅保留为该 runner 的 exact source、装配和摘要 helper，
不再作为独立 Gate 执行。

```bash
python docker/debug/plugin_passive_webui_v3_e2e.py --require-clean-core
```

证据写入 `docker/debug/reports/plugin-passive-webui-v3/gate.json`；模型响应由 Compose 私网内
的 deterministic model-gate 提供，不调用外部生产 provider。

### Shell execution 固定 Runtime 场景

`shell_execution_contract` 在 change-gate 的只读 Arch Linux runtime 中运行，不读取
宿主 `HOME` 或正式 workspace。普通路径覆盖短命令、非零退出、显式/默认 shell、
login 开关、长命令增量输出和多 execution 隔离；edge case 覆盖 PTY 多次输入、等待
取消、stop 与 initial/poll 竞态、执行进程组清理、输出 head/tail、recent-8 LRU、owner
隔离、截止点退出、主 ReAct/SubAgent active execution pin、turn owner 回收和 Drift
owner 隔离。

只运行该固定场景可使用：

```bash
python docker/debug/gate.py run --base <仅含合同的前置提交>
```

Gate 根据 runtime source set 自动选择该场景，并验证候选源码只读、临时 workspace、
Compose cleanup 和无残留容器/网络/卷。完整 diff 同时修改受保护合同与生产源码时必须
执行完整公开场景，不能用 focused 场景替代。

## 程序化控制面验收门

`programmatic_control_probe.py` 拥有独立 Compose project、隔离 sandbox、证据收集、
源码 digest 审计和强制 cleanup。`/app` 与 model-gate 源码均只读挂载，运行时只允许写
`/sandbox` 和 tmpfs `/tmp`。

```bash
python docker/debug/programmatic_control_probe.py --gate smoke
python docker/debug/programmatic_control_probe.py --gate failure-matrix
python docker/debug/programmatic_control_probe.py --gate soak
```

当前基建实现 `smoke`、`failure-matrix` 和 `soak`。每周 `Runtime Lifecycle Weekly`
顺序运行 `failure-matrix`、`soak` 与 restart soak；它们不再是普通 PR 的重复 Gate。`smoke`
覆盖 UDS/stdio、基本 turn，以及 streaming/tool/usage 的事件与 DB 一致性；
`failure-matrix` 覆盖双连接隔离、同 thread active-start busy、精确中断、断线恢复、慢客户端背压、
provider 分类、非法协议、Web channel parity、workspace lock、SIGTERM 和 crash/restart。
`soak` 执行 10 次预热与 100 次混合 turn，包含 10 次 reconnect、interrupt 和 provider
failure，并检查 RSS、fd、线程与 DB 非终态阈值。
每次运行的证据位于
`docker/debug/reports/programmatic-control/<run-id>/`。

### 程序化控制 API 的边界

调试实例的程序化来源通过现有控制连接提供固定方法表：

```text
programmatic/session/admit
programmatic/message/send
programmatic/message/pause
programmatic/message/resume
programmatic/message/result
```

调用方先保存自己生成的 `session_id`；相同身份和准入属性可以幂等恢复，改变已创建属性必须报告冲突。
`send`、`pause` 和 `resume` 的 ACK 只表示 Input/Control 已由 Message writer 提交，不表示回复已经完成或已经送达。
`result` 从同一 Message 日志快照读取 `complete`、`pause`、`failure` 或 `open`，持久消息按 `seq` 追赶；回复活动与持久完成事实分别读取，不能用空闲状态代替完成回执。

Input、Control 和模型原始 Output 都进入同一 append-only Message 日志；投影只计算状态，不保存第二份正文或完成标记。
Session 的准入属性创建后不自动改写或减少；显式 `persist_memory=true` 取得 `learning=eligible`，省略或传 `false` 取得 `learning=excluded`。调试 API 不绕过普通来源的读取、回复和学习边界。

## Akasha Message 在线学习与重启恢复 Gate

Gate 通过公开 programmatic Message API 在隔离 workspace 中验证普通 Akasha 插件的在线学习与重启恢复。
模型回复由 scripted model-gate 控制，embedding 由本地 HTTP fixture 或显式配置的外部 provider 提供。
第一轮写入 Message 与 embedding，第二轮先验证自动 Akasha 上下文和只读 `recall_memory`，再提交新的学习图；
随后重启同一个 Gateway，读取已发布的 Akasha 学习图并核对运行代际变化。该重启检查不宣称新 Message
offline rebuild；源 `messages` 和 `message_embeddings` 行必须保持不变。local fixture 还会核对重启后没有新的
embedding 调用；本次外部 provider 未实测，也不记录其请求计数。

本地可观察的 embedding 边界使用：

```bash
python docker/debug/akasha_v2_runtime_probe.py --local-fixture
```

正式 provider 模式要求在启动前提供以下三个环境变量：

```text
AKASHIC_E2E_EMBEDDING_API_KEY
AKASHIC_E2E_EMBEDDING_BASE_URL
AKASHIC_E2E_EMBEDDING_MODEL
```

```bash
AKASHIC_E2E_EMBEDDING_API_KEY=... \
AKASHIC_E2E_EMBEDDING_BASE_URL=https://provider.example/v1 \
AKASHIC_E2E_EMBEDDING_MODEL=text-embedding-v4 \
python docker/debug/akasha_v2_runtime_probe.py \
  --formal-workspace /path/to/formal-workspace
```

`--formal-workspace` 是可选保护检查；local Gate 不需要正式 workspace。`config.toml` 按
`0600` 写入 sandbox；凭据由 `add_openai_models` 通过 HTTP 写入普通模型插件的 credential
storage，不写入 `config.toml`。配置与凭据只写入一次性 sandbox，报告不输出 API key，清理失败会使
Gate 失败。证据位于
`docker/debug/reports/akasha-v2-runtime/<run-id>/`。

`scripts/build_akasha_db.py` 仍是旧 schema 的离线 builder，只能用于它已有的 legacy
重建合同；本 Gate 不把它当作新 Message 链路的 offline rebuild 验证。

## Yoyo 迁移检查

迁移使用普通 pytest 覆盖执行与失败重试，CI 另以精简检查保护已注册 migration 不被改写：

```bash
python -m pytest tests/test_migration_runner.py tests/test_yoyo_migration_append_only.py
python scripts/check_yoyo_migrations.py --base origin/main
```

新增 migration 前按 [Yoyo 迁移维护手册](../../docs/design/git-migration-authoring.md)补齐
真实状态变换与相应 case。不再构造 Git cursor、固定 baseline、repair 清单或专用容器 Gate。

## v3 MCP / managed-process 验收门

MCP 与 managed process 只能由插件静态 manifest 声明，经过 exact Root-local
`McpServerRegistry` / `ManagedProcessRegistry` 冻结，再由对应 generation host 物化。
workspace 手工 TOML、watcher/admin 和独立热重载路径已删除；Gate 不读取正式 workspace。

```bash
python docker/debug/plugin_v3_fleet_gate.py
python docker/debug/restart_probe.py --soak
```

fleet 报告固定来源和 v3-only 静态合同；真实 MCP handshake/readiness、进程/stdio cleanup
由保留的行为回归和每周 restart soak 负责。已删除的 composition Gate 只是按固定插件排列
重复内部 snapshot，不再作为独立证据。

## Content / Wake / Drift 真实插件互操作 Gate

`content_source_interop_gate.py` 不复制 Calendar、Fitbit、Feed、Steam、Emotion 或 Observe
实现。它先把调用方提供的 canonical checkout 与 exact lock 对账，再运行 Core 已有的
Content/Wake/Drift/Session 组合 fixture，最后在每个插件自己的目录运行其原样 fixture。
因此修改一个插件的业务模型只需要更新该插件与 exact revision，不会给 Core 增加来源分支。

```text
exact lock ──► checkout SHA/manifest/no old seam
                         │
                         ├─► Core owned composition fixtures
                         └─► plugin owned domain/reload/ACK fixtures
```

每个 root 必须显式绑定。完整行为 Gate 还要求每个插件显式绑定它实际安装依赖的
测试 Python；Gate 会运行受控探针并在 receipt 记录 executable realpath 与 Python 版本，
不会悄悄借用 Core Python：

```bash
python docker/debug/content_source_interop_gate.py \
  --plugin-root calendar=/absolute/calendar-checkout \
  --plugin-root fitbit=/absolute/fitbit-checkout \
  --plugin-root feed=/absolute/feed-checkout \
  --plugin-root steam=/absolute/steam-checkout \
  --plugin-root github-watch=/absolute/github-watch-checkout \
  --plugin-root proactive_feedback=/absolute/proactive-feedback-checkout \
  --plugin-root emotion=/absolute/emotion-checkout \
  --plugin-root observe=/absolute/observe-checkout \
  --plugin-python calendar=/absolute/calendar-python \
  --plugin-python fitbit=/absolute/fitbit-python \
  --plugin-python feed=/absolute/feed-python \
  --plugin-python steam=/absolute/steam-python \
  --plugin-python github-watch=/absolute/github-watch-python \
  --plugin-python proactive_feedback=/absolute/proactive-feedback-python \
  --plugin-python emotion=/absolute/emotion-python \
  --plugin-python observe=/absolute/observe-python
```

默认 pending 调查会让 Gate 非零；开发期间只核对已完成 revision 可使用
`--identity-only --allow-pending`，`--allow-pending` 不能放宽完整行为 Gate。GitHub Watch
在这个 Core Gate 中只证明真实插件可以 mount/activate 且 Content mailbox 完整逻辑状态零变化；
BACKGROUND_JOBS 的 dispatch、ledger 和失败不重放由 GitHub Watch 自己的 exact fixture 证明。
`cross_repo` 套件只声明参与的 exact plugin ids、fixture 和显式 Python；runner 不包含插件
业务分支。当前 PF→Emotion fixture 用真实 `CompositionRoot` 按两种 mount 顺序分别执行
普通 follow-up 与 explicit quote，证明 accepted history 在提交当轮不被 Emotion 抢读，只由
下一个普通 Timer tick 拉取一次；普通 scoring 仅把 embedding provider 换成确定性测试边界。
报告写入
`docker/debug/reports/content-source-interop/gate.json`，不读取或写入正式 workspace。

## Content / Wake H5 组合证据

`content_wake_h5_e2e.py` 不实现插件或 runtime 行为。它只在一个显式的一次性 root 中，按顺序
调用正式 `plugin-install-trusted-batch`、Content source interoperability Gate 和 manifest
列出的既有 pytest fixture，并把各 owner 报告的路径、SHA-256 与状态组合为
`reports/h5-index.json`：

```bash
python docker/debug/content_wake_h5_e2e.py \
  --run-root /absolute/new/h5-run \
  --protected-workspace /absolute/empty/protected-fixture \
  --seed-protected-fixture
```

`run-root` 必须尚不存在；runner 在其中创建 `workspace/`、`plugin-home/`、`reports/` 和
`home/`。插件 root 只取 trusted batch 回执中的 `installedPath`，revision 只取
`content-source-interop.lock.json`。显式 `--seed-protected-fixture` 只接受空的隔离目录，并写入
带一行数据的 `sessions.db`、旧 proactive/wake/drift DB 与历史 Markdown/JSON；不复制正式数据。
不使用 seed 时，调用者也必须提供至少包含非空 Session、旧 proactive DB 和 archive 的 fixture，
空 protected target 会在安装前失败。runner 复用 Wake provider 的快照，对账 path、inode、hash、
size、SQLite integrity/quick_check/row counts，前后不相等时本次组合失败。owner pytest 由
Core dev Python 运行，实际插件 service 只通过 `AKASHIC_PLUGIN_FIXTURE_PYTHON` 使用回执中的
artifact runtime；一次性 root 内固定版本的 pytest layer 仅验证 artifact 隔离，不进入 owner
fixture，Core site-packages 不会暴露给插件 service 解释器。
真实 DeepSeek 命令只作为 `PENDING` 项进入 index；没有单独授权时 runner 不调用外部 provider。
生成的证据留在一次性 root，不提交进 Git。

## Wake v3 真 provider E2E

`wake_v3_provider_e2e.py` 只把临时目录当作测试 data root，并通过正式
`PluginManager → CompositionRoot → ConversationRuntime → execute_control_turn → AgentLoop.react`
安装链运行 Content、Drift、Wake、普通 Content source fixture 和普通 recording Channel。
它不调用 `init_workspace`，也不把 production workspace 复制进沙盒。

```text
fixture source ── Timer ──▶ eventmail.content_source.v1 submit
                                  │
                                  ▼
                         Wake scoped Turn ──▶ AgentLoop.react
                                  │
                                  ▼
 recording Channel ◀── durable delivery ◀── provider response
          │                       │
          └── receipt ──▶ Session projection ──▶ Content settle ──▶ source ACK
```

真实 selected case 固定使用 `deepseek-v4-flash`。runner 从外置副本加载普通 `models` 和
`openai-compatible` 插件，经 `RuntimeModelControl` 写连接、模型和默认 role，再由 exact
snapshot 的 `CHAT_MODELS` 执行；Core 不读取 provider 配置。credential 只从进程环境读取，
不写入临时 TOML 或报告。运行前先完成确定性的 settlement crash/restart、
ACK retry、quiet 和 empty-poll 检查，之后才允许一次真实 logical provider request：

```bash
PR_G_DEEPSEEK_API_KEY='...' \
python docker/debug/wake_v3_provider_e2e.py \
  --protected-workspace /absolute/formal/workspace \
  --report /tmp/wake-v3-provider-e2e.json
```

可选自定义 provider 地址使用 `PR_G_DEEPSEEK_BASE_URL`。报告不会包含 prompt、response
正文、secret 或 endpoint；只保留计数、状态和 identity digest。缺少 secret、provider
非 2xx、identity 不一致或未 settled 都会返回非零。
protected workspace 检查包括 `sessions.db` 与旧 proactive island 目标文件的 hash/size、
SQLite integrity/row counts 和旧 island archive hash/size；整个检查只读。

正式 runtime 可能在 E2E 期间自行推进旧 island。runner 因此先连续读取两份 baseline，
随后分别报告 isolated 产品链与 formal deployment evidence：若 formal target 在窗口内变化，
结果标为 `formal_concurrent_change`，只列 changed path/type/count，不把外部并发写误判成
E2E 写入，也不宣称 formal unchanged。严格 digest 只在 baseline 稳定且 after 完全相等时
设置 `deployment_gate_verified=true`。失败报告只增加固定 `failure_stage/failure_code`，仍不
包含异常正文、prompt、response、credential 或 endpoint。即使 provider 或 selected 链失败，
runner 也会在临时 data root 删除前读取 logical/HTTP/provider terminal/Control Turn/
delivery/Channel/Session/Content/ACK 计数和 identity digest，再执行 formal-after 快照。
Control Turn 只报告 status、retryable 分类和 error type digest，不保留 error message。
非流式 provider 的 HTTP attempt 复用既有
`nonstream.start` 与结构化 retry 记录计数，handler 不保存 warning 中的 endpoint 或正文。
loopback 200/400/503 fixture 分别冻结 completed+settled、nonretryable+invalidated 与
retryable+deferred 三条边界，不访问外部 provider。

确定性模型 sidecar 的控制协议：

- `PUT /control/script`：装载一个脚本对象或脚本数组。`mode` 支持 `complete`、
  `stream`、`error`、`timeout`、`truncate`；可提供 `content`、`deltas`、`tool_calls`、`usage`、
  `status` 和 `body`。
- `PUT /control/barriers/{name}`：创建 barrier。将 `"barrier":"{name}"` 放入脚本后，
  对应模型请求到达 provider sidecar 时会精确阻塞。
- `GET /control/barriers/{name}/wait?timeout=30`：服务端长等待请求到达，不靠 controller
  固定 sleep 猜竞态。
- `POST /control/barriers/{name}/release`：释放已到达的模型请求。
- `GET /control/requests`：读取完整 payload、关联 header、脚本和请求状态证据。

示例脚本：

```json
{
  "mode": "stream",
  "barrier": "turn-entered-provider",
  "deltas": ["hello ", "world"],
  "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
}
```

这个目录用于临时调试真实入口，例如 Telegram 图片、多模态链路、独立 bot 配置。调试容器基于 Arch Linux，沙盒不会挂载宿主机 `HOME`，也不会挂载正式 `~/.akashic/workspace`。

```
host
  |
  +-- akashic-agent
      |
      +-- docker/debug
          |
          +-- Dockerfile
          +-- docker-compose.yml
          +-- entrypoint.sh
          +-- profiles
              |
              +-- default
                  |
                  +-- config.toml
                  +-- workspace
                  +-- home
                  +-- akashic.sock

container
  |
  +-- /app                 -> 当前代码
  +-- /sandbox/config.toml -> 调试 bot 配置
  +-- /sandbox/workspace   -> 调试 workspace
  +-- /sandbox/home        -> 容器 HOME
```

## 安全边界

- 默认调试配置只在 `docker/debug/profiles/default/config.toml`。
- 默认调试 workspace 只在 `docker/debug/profiles/default/workspace`。
- 容器内 `HOME` 是 `/sandbox/home`，不是宿主机 HOME。
- 启动脚本会拒绝 `/sandbox` 外的 config/workspace 路径。
- `profiles/` 已加入 `.gitignore`，不要提交调试 bot token 和测试记忆。

## 插件变更 Gate

pure-v3 候选证据由 fleet、Mobile 和公共 WebUI 三个边界组成。所有 Gate
使用 exact commit 锁、一次性 workspace/plugin-home/HOME 与受控端点，不读写正式
Akashic workspace、正式凭据或 hua-home 服务。

这些是插件候选与发布 Gate，不是普通 Core Pull Request 的固定矩阵。普通 Pull Request
运行全部保留回归和按 diff 选场景的统一变更影响 Gate；当改动进入插件候选时，
`Plugin v3 Candidate Gates` 手动 workflow 只运行 fleet completeness、Mobile 和公共 WebUI。

```text
精确能力 lock
      │
      ├── fleet ─────── 全插件来源、v3-only 与 retired exclusion
      ├── Mobile ────── Python catalog / JS ABI / plugin tests
      └── WebUI ─────── Citation / Meme / public WebSocket
```

候选 workflow 保留的独立边界：

```bash
python docker/debug/plugin_v3_fleet_gate.py \
  --require-clean-core --require-full-core-history
python docker/debug/plugin_v3_mobile_gate.py --require-clean-core
python docker/debug/plugin_passive_webui_v3_e2e.py --require-clean-core
```

被删除的 E1/E2 固定了 2026-08 的组合 API；Core 在 2026-09-02 删除 v2 兼容后，
锁定的 Emotion 和 Calendar 插件分别仍导入已删除的 `CoreEvent` 与
`PROACTIVE_COMPONENTS`，所以两条 Gate 只会阻止当前合法演进。E4 又依赖 E1 和仓库中
不存在的 E3 runner，无法形成可执行发布合同，也一并删除。正式 workspace 的发布验收
应在拥有真实部署输入的发布流程中重建，不能由仓库内永远 blocked 的脚本冒充。

## 第一次启动

全新的 profile 可以直接启动。入口会初始化只含 Core 和 workspace 的空白本地实例；随后打开
`http://127.0.0.1:2236`，在“模型”页连接订阅或 API。模型和已知能力都从页面同步，不要求先在终端填写。

```bash
docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

只有需要同时配置专用 Telegram bot 时才运行交互向导：

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug setup
```

这里填写专用 Telegram bot、模型 key 和多模态配置。不要填正式 bot。

## 启动调试 Agent

```bash
docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

此时向调试 Telegram bot 发消息或图片，所有会话和记忆都会进入 `docker/debug/profiles/default/workspace`。

调试容器通过固定 Supervisor 启动每个 boot 唯一的 Guardian，再由 Guardian 启动 Gateway。
Supervisor 只会在当前 boot 已通过私有事件 ready、`agent_restart` 的最终回复已经实际送达、
Gateway 提交一次匹配证据、以 75 退出且 Guardian 证明旧 boot 已空时拉起下一代。普通退出、
崩溃、伪造 75、断线和送达超时都不会触发重启。`RestartWatcher` 只消费成功 ToolResult，先调用
`RestartGate.prepare` 关闭新 Root 接纳，再等待所属 Turn complete；programmatic 来源随后等待
`FrameBook` 的同连接完整 frame drain，普通来源等待 `FinalOutputDelivery` 的实际送达。两条分支
都由 `RestartGate.commit` 等待 permit 排空并向 supervisor 私有 lifecycle pipe 提交 opaque request ID；
commit 成功后 programmatic claim 才由 watcher consume。连接断线时 FrameBook 移除 active route、保留
原始 `ConnectionError` 到 claim consume/abort，不能被另一连接接管。

本机若仍由忽略版本控制的 `start.sh` 启动，应让它调用正式默认入口：

```bash
python main.py --config /absolute/config.toml --workspace /absolute/workspace
```

`supervise` 子命令在 Linux 保留为兼容别名。只有需要让调试器直接附着未托管 child 时才显式使用
`python main.py gateway`；该入口不注册 `agent_restart`。

不要在外层脚本再做 `while`、`pgrep` 或“任意非零退出就重启”；进程唯一性、信号转发、
重启授权和 boot 代际由 Supervisor 持有，Guardian 只拥有当前 boot 的进程清理。

## 多套调试配置

不同功能可以用不同 profile 保存配置和 workspace：

```bash
AKASHIC_DEBUG_PROFILE=multimodal docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug setup
AKASHIC_DEBUG_PROFILE=multimodal docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

对应目录是 `docker/debug/profiles/multimodal/`。

## 调用调试实例

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug exec --new "测试消息"
```

app-server socket 固定为 `/sandbox/akashic.sock`，不会连接正式实例。

## 打开调试 Dashboard

```bash
docker compose -f docker/debug/docker-compose.yml run --rm --service-ports akashic-debug dashboard
```

宿主机访问 `http://127.0.0.1:2237`。

## 停止调试环境

```bash
docker compose -f docker/debug/docker-compose.yml down
```

这只会停止容器，不会删除当前 profile 目录。

## 清空调试 workspace

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug reset-workspace
```

这个命令只删除并重建当前 profile 下的 `workspace`，会保留当前 profile 下的 `config.toml`。

## 上下文连续性探针

`context_probe.py` 用于复现一段固定纯聊天场景，自动记录用户输入、LLM 回复、工具调用、compaction ledger 和 Akasha Inspector 事件。

```
context probe
  |
  +-- profile
  |     |
  |     +-- config.toml
  |     +-- workspace
  |
  +-- phase1 chat
  |
  +-- phase2 chat
  |
  +-- final question
        |
        +-- markdown report
        +-- json report
```

从已启动的沙盒运行：

```bash
python docker/debug/context_probe.py \
  --profile default \
  --messages docker/debug/scenarios/sleepy_study_plan.json
```

自动重置、启动、运行并停止：

```bash
python docker/debug/context_probe.py \
  --profile v4flash-memory-window \
  --messages docker/debug/scenarios/sleepy_study_plan.json \
  --reset-workspace \
  --start-agent \
  --stop-agent \
  --quiet-agent \
  --disable-qq
```

`--disable-qq` 会在运行期间临时给当前 profile 的 `[channels.qq]` 加 `enabled = false`，结束后恢复原配置，适合只测 CLI 但该 profile 配了 QQ 的情况。

默认报告写到：

```text
docker/debug/profiles/<profile>/workspace/context-probe-<profile>.md
docker/debug/profiles/<profile>/workspace/context-probe-<profile>.json
```

自定义场景 JSON 格式：

```json
{
  "name": "sleepy_study_plan",
  "turns": [
    {
      "role": "user",
      "content": "前置闲聊"
    },
    {
      "role": "user",
      "content": "后续闲聊"
    },
    {
      "role": "user",
      "content": "最后问题",
      "final": true
    }
  ]
}
```

场景 JSON 只描述连续输入和流程，不写语义结果要求。探针遇到主流程的通用失败回复时会立即失败，正常回复则只记录 observe 结果，不主观判断内容质量。

内置样例在：

```text
docker/debug/scenarios/sleepy_study_plan.json
```

公开场景和 schema 都放在：

```text
docker/debug/scenarios/
```

这里的文件是稳定输入，可以提交；`docker/debug/profiles/<profile>/workspace/` 里的报告 JSON / Markdown 是运行产物，默认不提交。

兼容旧格式：

```json
{
  "phase1": ["第一段闲聊"],
  "phase2": ["第二段闲聊"],
  "final_question": "最后问题"
}
```

## 真实 Runtime 时间回放基础

`replay_controller.py` 只维护隔离 profile 下的模拟时钟、历史事件和捕获消息，不读取或挂载正式 workspace。`docker-compose.yml` 会让真实 `main.py` 加载调试插件目录；`replay_debug` 插件注册 `replay` 渠道，把 outbound 原样写入 profile。

```text
┌─ replay_controller
│  ├─ clock.json             模拟当前时间
│  ├─ events.jsonl           历史事件输入
│  └─ outbox.jsonl           捕获的 outbound
│
└─ Docker profile
   ├─ python main.py         正式启动入口
   ├─ SystemClock            线上默认时钟
   ├─ ReplayClock            调试文件时钟
   └─ CaptureChannel         channel = replay
```

初始化独立 profile 的回放状态：

```bash
python docker/debug/replay_controller.py \
  --profile wake-replay init \
  --start-at 2026-05-01T00:00:00+08:00
```

该 profile 仍需要自己的 `config.toml`。可以运行 `setup`，或复制另一份专用调试配置。启动前应关闭 Telegram、QQ 等外部渠道，并将待测发送目标设为 `channel = "replay"`。

```bash
AKASHIC_DEBUG_PROFILE=wake-replay \
docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

注入单条历史事件：

```bash
python docker/debug/replay_controller.py --profile wake-replay inject \
  --event-id feed-001 \
  --kind content \
  --source-id rss-example \
  --title "历史候选标题" \
  --content "历史候选摘要" \
  --published-at 2026-05-01T08:30:00+08:00
```

批量输入支持 JSON 数组、`{"events": [...]}` 或 JSONL：

```bash
python docker/debug/replay_controller.py \
  --profile wake-replay import-events /path/to/history.jsonl
```

推进时间并查看当前可见事件和捕获结果：

```bash
python docker/debug/replay_controller.py \
  --profile wake-replay advance --seconds 3600
python docker/debug/replay_controller.py \
  --profile wake-replay status
```

`events.jsonl` 只由 replay controller 保存为调试输入；当前 Core 没有隐式消费者，推进时钟不会自动触发 Turn。需要验证 Content/Wake 时使用上面的普通插件互操作 Gate。

## 完全清理

```bash
docker compose -f docker/debug/docker-compose.yml down --remove-orphans
rm -rf docker/debug/profiles/default
```

完全清理后，下次需要重新运行 `setup`。
