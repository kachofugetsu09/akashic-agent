# Plan 001: 以验证门禁完成全插件热重载

> **Executor instructions**: 严格按步骤施工。每一步必须运行对应 Gate，只有预期结果全部满足后才能进入下一步。任何 Gate 失败时保留当前可运行实现，停止扩展范围并报告证据。完成后更新 `plans/README.md` 状态。
>
> **Drift check (run first)**: `git diff --stat d2957df..HEAD -- agent/plugins bus agent/tools agent/core agent/looping proactive_v2 bootstrap frontend/dashboard tests tests_scenarios docker/debug plugins`
> 若范围内代码已变化，先逐项核对“当前状态”；关键入口不一致时停止并更新计划。

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: none
- **Category**: migration
- **Planned at**: commit `d2957df`, 2026-07-11
- **State**: DONE
- **Completed at**: 2026-07-11

实现已收敛到 `RuntimeSnapshot` 单一发布点。安装、删除、清单启停、源码和私有配置变化由 Watcher 触发候选验证；被动执行、主动 tick、Job、Tool、MCP、Skill、Lifecycle、Channel、Dashboard 与托管服务统一按代际租约切换和排空。全部外部插件已迁移，Fitbit 的睡眠模型、数据目录和预测接口保持不变。

最终真实 Runtime 证据：`all-plugins` Gate 对 19 个非 Fitbit 外部插件逐个完成 load → reload → disable；`fitbit` Gate 单独完成监控服务 reload → disable，并确认进程数、健康接口和持久数据哈希不变。G-1 与 G6 的失败传播另有纯单元测试覆盖。

## Why this matters

当前插件能力在启动时直接追加到多个 Runtime，插件实例、事件订阅、任务、MCP 连接和 Channel 没有统一所有权。只增加文件 Watcher 会让旧实例残留、回调重复或同一 turn 混用两代能力。

目标是让安装、升级、启停、源码修改和配置修改统一进入 `reconcile(plugin_id)`，并通过可机器验证的 Gate 判断候选代际是否可以发布。中心层只理解插件代际、能力类别、状态和验证结果，不出现具体插件名或业务字符串。

## Current state

- `agent/plugins/base.py:14-82`：插件公开 skills、MCP、七类 passive module、proactive、jobs、channels 和 initialize/terminate。
- `agent/plugins/manager.py:91-107`：各能力保存在平铺列表中，没有 generation owner。
- `agent/plugins/manager.py:254-260`：已加载模块直接跳过，没有 reload 入口。
- `agent/plugins/manager.py:329-369`：加载时立即注册副作用，初始化失败只能回滚部分列表，EventBus handler 无法解绑。
- `bus/event_bus.py:24-31`：`on()` 不返回 subscription，也没有 unsubscribe。
- `bootstrap/tools.py:112-165`：启动时一次性把插件能力追加到 AgentLoop、Skills 和 MCP。
- `agent/core/passive_turn.py:283-309`：phase module 只能 extend 后重建，不能按插件替换。
- `proactive_v2/loop.py:65-110`：proactive contributions 在构造时保存并一次性构建 kernel。
- `agent/plugins/jobs.py:106-180`：JobRuntime 构造时固定 jobs，并创建无法解绑的 EventBus handler。
- `bootstrap/channel_host.py:11-39`：ChannelHost 只有 add、start_all、stop_all。
- `agent/mcp/registry.py:80-111`：MCP 只按 server name 增删，同名配置改变不会重连。
- `agent/skills.py:51-125`：SkillLoader 每次调用都会扫描，天然适合 snapshot/invalidation。
- `bootstrap/dashboard_api.py:602-642`：Dashboard 插件直接向全局 FastAPI app 注册，无法可靠卸载路由。
- `bootstrap/wiring.py:79-121`：Memory Engine 使用第二套插件加载路径。
- `/mnt/data/coding/akashic-plugin/proactive_feedback/plugin.py:34-50`、`observe/plugin.py:30-52`：插件直接创建 task 和 EventBus subscription。
- `/mnt/data/coding/akashic-plugin/fitbit-mcp/plugin.py:61-86`：Fitbit 直接管理 sleep ML monitor 子进程，迁移必须保持模型、数据和行为不变。
- `docker/debug/docker-compose.yml`：现有 `akashic-debug` 把仓库挂载到 `/app`，默认可写；它适合开发调试，不满足“Gate 无法损伤项目”的隔离要求。
- `docker/debug/docker-compose.yml`：现有宿主插件缓存挂载到 `/sandbox/home/.akashic-plugin/cache:ro`；热安装、更新、卸载测试不能在这里运行。
- `docker/debug/Dockerfile` 与 `.dockerignore`：镜像只包含依赖和 entrypoint，真实源码必须通过 `/app` 提供，因此 Gate 应把 `/app` 挂成只读，而不是另造一份 Runtime。
- `docker/debug/context_probe.py`：已有真实进程启动、Unix socket readiness、超时和 `finally` 清理模式。
- `docker/debug/runtime_race_probe.py`：已有 `build_core_runtime()` 的结构化场景输出，可复用真实 wiring 与确定性 provider 的组合。
- `docker/debug/proactive_sandbox.py`：已有真实 PluginManager、MCP 子进程和 proactive kernel 的沙盒路径约束，可复用主动链路验证方式。

仓库约定：Python 使用四空格、针对性类型标注、避免宽泛异常；测试使用 pytest/pytest-asyncio；前端只修改 `frontend/**/src`；提交信息使用简洁 Conventional Commit。

## Gate model

所有 Gate 返回结构化结果，不以日志文本作为唯一判据：

```text
┌─ GateResult
│  ├─ gate_id
│  ├─ plugin_id
│  ├─ candidate_revision
│  ├─ status = passed／failed
│  ├─ checks[]
│  │  ├─ check_id
│  │  ├─ status
│  │  └─ evidence
│  └─ failure_reason
```

运行时只负责通用 Gate；业务语义由插件提供无副作用检查：

```text
┌─ G-1 Sandbox Integrity Gate
│  └─ 仓库只读、HOME／插件缓存／workspace 隔离，运行前后宿主仓库状态一致
├─ G0 Baseline Gate
│  └─ 现有行为和资源计数已记录
├─ G1 Candidate Gate
│  └─ 新代际可导入、配置可验证、能力图可编译
├─ G2 Readiness Gate
│  └─ MCP／Service／Channel／Dashboard 候选可启动或预热
├─ G3 Semantic Gate
│  └─ 插件自己的只读验证通过
├─ G4 Publication Gate
│  └─ 新执行只见新快照，旧执行保持旧快照
├─ G5 Drain Gate
│  └─ 旧代际无 subscription／task／process／connection 泄漏
└─ G6 System Gate
   └─ 完整测试、Docker Runtime 和真实插件行为通过
```

硬规则：G-1 失败不得在 Docker 内执行后续 Gate；G1-G3 任一失败不得发布；G4 发布失败时只为后续执行重新发布旧 snapshot，已经持有失败代际 lease 的执行继续完成并在 drain 后清理；G5 失败不得删除兼容适配器；G6 失败不得完成迁移。

验证按同一条逐步升级链路执行，不用轻量探针冒充真实 Runtime：

```text
┌─ 宿主保护层
│  ├─ /app 只读
│  ├─ canonical plugin repo 不可写
│  └─ /sandbox 独占可写
├─ 纯单元 Gate
│  └─ generation／scope／snapshot 契约
├─ 确定性真实 Runtime Gate
│  ├─ 真实 AppRuntime／PluginManager／MCP／Job／Proactive
│  └─ 测试 provider 与测试 outbound，禁止外部副作用
└─ 最终真实 Profile Gate
   ├─ 由 main.py 启动完整 Runtime
   ├─ 外部 Channel 关闭
   └─ 只对测试账户或匿名 fixture 做业务契约验证
```

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Plugin tests | `pytest -q tests/test_plugin_manager.py tests/test_plugin_hot_reload.py` | exit 0, all pass |
| Proactive tests | `pytest -q tests/proactive_v2 tests/test_plugin_hot_reload.py` | exit 0, all pass |
| Full tests | `pytest -q tests/` | exit 0, all pass |
| Python types | `pyright <changed-python-files>` | exit 0, no errors |
| Frontend types | `npm run typecheck` | exit 0 |
| Frontend lint | `npm run lint` | exit 0 |
| Frontend build | `npm run build` | exit 0 |
| Docker build | `python docker/debug/plugin_hot_reload_probe.py --scenario sandbox-integrity` | controller builds current Gate image before validation |
| Sandbox integrity | `python docker/debug/plugin_hot_reload_probe.py --scenario sandbox-integrity` | host controller creates an external sandbox; exit 0, host state unchanged |
| Docker runtime | `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase <phase>` | separate controller and Runtime container; exit 0, structured report passed |
| Docker cleanup | controller `finally` | unique Compose project is removed and cleanup result enters GateResult |

## Scope

**In scope**:

- `agent/plugins/**`
- `agent/tools/**`
- `agent/core/passive_turn.py`
- `agent/looping/core.py`
- `bus/event_bus.py`
- `proactive_v2/**`
- `bootstrap/tools.py`
- `bootstrap/app.py`
- `bootstrap/channel_host.py`
- `bootstrap/dashboard_api.py`
- `bootstrap/memory.py`
- `bootstrap/wiring.py`
- `frontend/dashboard/src/pluginRuntime.ts`
- `tests/test_plugin_manager.py`
- `tests/test_plugin_hot_reload.py`（新增）
- `tests_scenarios/test_plugin_hot_reload_runtime.py`（新增）
- `docker/debug/docker-compose.plugin-gate.yml`（新增）
- `docker/debug/README.md`
- `docker/debug/plugin_hot_reload_probe.py`（新增）
- `_handbook/plugins-tutorial.md`
- `/mnt/data/coding/akashic-plugin/*` 的插件源码迁移

**Out of scope**:

- 修改插件业务功能或产品行为。
- 修改 Fitbit 模型、训练数据、预测阈值和持久数据格式。
- 用进程隔离重写全部 Python 插件。
- 给公共协调层加入插件名、业务路径或 payload 字符串特判。
- 永久保留两套 Runtime 链路；阶段内允许单向兼容适配器，Step 7 必须统一删除。
- 让 Watcher 直接修改 Runtime registry。
- 热替换 CPython、原生动态库或核心 Runtime ABI；这类变化仍是进程重启边界。

## Git workflow

- 在独立 feature branch 施工。
- 每个通过 Gate 的步骤形成一个可独立回退的 commit。
- Commit 示例：`feat(plugin): 引入插件代际与资源作用域`。
- 核心仓库与 `/mnt/data/coding/akashic-plugin/*` 下的各插件仓库分别提交，不混淆历史。
- 未经操作者要求，不 push 或创建 PR。

## Steps

### Step 0A: 先建立不可损伤宿主的 Docker Gate

新增独立 `docker/debug/docker-compose.plugin-gate.yml`，不向普通 `docker-compose.yml` 加 Gate service 或必填变量。Gate service 复用同一个 Dockerfile 和 entrypoint，但必须满足：

- `/app` 只读挂载，容器不能写源码、测试、构建产物或 Git 元数据。
- `/sandbox` 是唯一持久可写挂载；其宿主源必须由 host-side controller 在仓库外创建，禁止使用仓库内 `docker/debug/profiles/`。HOME、workspace、socket、配置和测试插件缓存全部位于其下。
- 不挂载宿主 `~/.akashic-plugin/cache`；Gate 在 `/sandbox/home/.akashic-plugin/cache` 安装和修改一次性测试插件。
- canonical plugin repo 若用于 fixture，只能只读挂载；Probe 先复制到 sandbox cache 再执行 reload。
- root filesystem 只读，`/tmp` 使用 tmpfs；不挂载 Docker socket，不发布端口，不启用 Telegram、QQ、Chat 等外部 Channel。
- Controller 从临时目录名生成带 nonce 的独立 Compose project，清理命令不能影响普通 debug 容器或并行 Gate。

新增 host-side controller `plugin_hot_reload_probe.py`。Controller 使用系统临时目录创建唯一 sandbox，规范化路径并拒绝位于核心仓库、任一插件仓库或宿主插件缓存内；随后启动独立 Runtime 容器。`/mnt/data/coding/akashic-plugin` 是多个独立插件仓库的父目录，不是单一 Git 仓库；Controller 在运行前后分别记录核心仓库和每个插件仓库状态。容器内 probe 只检查 mount 与沙盒路径。工作树允许原本就是 dirty，但前后状态必须完全一致；Git 审计只是二次证据，真正保护依赖只读 mount。

```text
┌─ Host repositories
│  ├─ akasic-agent ───────── read-only ──┐
│  └─ akashic-plugin/* repos read-only ──┤
│                                       ▼
├─ akashic-plugin-gate container
│  ├─ /app              read-only
│  ├─ /fixtures/plugins read-only
│  ├─ /tmp              tmpfs
│  └─ /sandbox          writable
│     ├─ home/.akashic-plugin/cache
│     ├─ workspace
│     ├─ config.toml
│     └─ reports
└─ Cleanup
   └─ only compose project akashic-plugin-gate-<nonce>
```

**G-1 Verify**:

- 容器内 mount 信息证明 `/app` 和 fixture repo 为只读，`/sandbox` 可写。
- Compose 未提供 controller 生成的 `AKASHIC_GATE_SANDBOX` 时必须拒绝启动。
- sandbox 的宿主规范路径不位于核心仓库、插件父目录或宿主插件缓存中。
- `HOME`、workspace、socket 和 installed plugin cache 的规范路径都位于 `/sandbox`。
- 启动、安装测试插件、修改 sandbox 中的插件、关闭容器后，核心仓库和全部插件仓库的前后状态完全一致。
- Probe 失败时保留 `/sandbox/reports` 和 profile 内容，`finally` 只停止自己的 Runtime 和 Compose project。

### Step 0B: 固化行为基线与全能力测试插件

新增 `tests/test_plugin_hot_reload.py`，动态生成一个最终覆盖全部核心能力的测试插件。Step 0 只建立 fixture builder、v1、v2、invalid revision 与现有能力基线；各 capability descriptor 随对应 Host 实现逐步加入，不提前要求尚不存在的 managed service API。

为每一代写入可观测 generation marker，测试必须能判断一次执行到底使用了哪一代，而不是只检查“没有报错”。记录 reload 前后的 handler、task、process、MCP client、channel 和 module namespace 数量。

在 canonical plugin repo 为 Fitbit 增加只读行为基线：固定匿名 fixture 输入、预测输出契约、monitor 单实例、数据目录 hash 不变。不得复制 Token 或真实健康数据进入测试。

**G0 Verify**:

- `pytest -q tests/test_plugin_manager.py tests/test_plugin_hot_reload.py` → 新基线测试通过。
- Fitbit 契约测试在迁移前通过，并输出不包含隐私数据的结果摘要。
- `git diff -- tests/` 与 `git -C /mnt/data/coding/akashic-plugin/fitbit-mcp diff -- .` → 只有测试与 fixture 变更，没有生产行为变更。
- 在 G-1 容器中运行 baseline scenario，证明真实 `build_core_runtime()` 能从隔离 cache 加载测试插件并正常关闭。

### Step 1: 以加法方式建立 PluginScope

在 `agent/plugins/` 中引入 plugin-owned scope。EventBus subscription、asyncio task、subprocess、closeable 和 deferred cleanup 都必须挂到 scope，并按逆序释放。

修改 `EventBus.on()` 返回可释放 subscription；插件通过 scoped EventBus 注册。`terminate_all()` 改为关闭 scope，而不是依赖插件手工记住所有资源。

先让 PluginManager 新加载的插件使用 scope。旧插件入口暂时经过单向适配器注册到 scope，保证每个阶段都能启动真实 Runtime；全部插件源码迁移与适配器删除留到 Step 7。长驻进程或循环最终必须成为 Runtime 可追踪的 managed service。

**Foundation Verify**:

- 加载并关闭测试插件 20 次后，EventBus handler 数、活动 task 数、子进程数和 scope child 数回到初始值。
- 插件 `close()` 抛错时，scope 仍完成其余强制回收，并返回 failed cleanup check。
- `pytest -q tests/test_plugin_manager.py tests/test_plugin_hot_reload.py` → all pass。
- `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase scope` → 真实 `main.py` 启动、加载插件、触发 subscription/task、关闭 Runtime 后资源恢复基线。

### Step 2: 建立 PluginGeneration 与 Candidate Gate

每次候选加载使用唯一模块命名空间，不使用 `importlib.reload()`。Generation 持有 plugin instance、scope、source/config revision、贡献集合、状态和 lease count。

加载过程必须先收集完整贡献，再做任何全局发布。Candidate Gate 验证本阶段已有声明的 API version、ConfigModel、路径边界、重复工具名、重复 source/job/channel ID、插件 phase graph、proactive lifecycle 结构、MCP spec 和 proactive source 引用。Dashboard 在 Step 5 成为正式 contribution 时接入同一 Gate，不让 Step 2 预先理解文件名旁路。

同一 plugin_id 已有 active generation 时，Step 2 只产生 `prepared` candidate，不能在 RuntimeSnapshot 建立前替换旧代。candidate context 不开放 EventBus、ToolRegistry、LLM、Session 或 Memory Engine，KV 只读；路径只属于可信插件的无副作用声明契约，不构成恶意 Python 的安全沙盒，宿主仓库隔离由 Docker Gate 保证。initialize 失败前不得向全局 ToolRegistry、EventBus 或 capability lists 发布。插件 phase graph 在这里与其他 active plugin 合并验证；builtin graph 与 runtime factory 需要的完整依赖由 Step 4 的 Snapshot compiler 验证。

插件可提供两级只读 semantic checks：Step 2 执行不依赖外部服务的 static checks，Step 3 在候选 Host ready 后执行 readiness checks。协调器只执行并汇总 `GateResult`，不得理解检查内容；两级适用检查都通过后才允许发布。

**G1/G3-static Verify**:

- invalid Python、invalid config、重复 tool、phase cycle、无效 MCP spec／source 引用、失败 semantic check 都返回 failed GateResult。
- 每种失败后 active generation id、工具集合、EventBus handler 和 Runtime 行为保持不变。
- v1 → invalid → v2 后 active 始终为 v1，invalid 被回收，v2 只进入 prepared；Step 4 才允许原子发布 v2。
- `pytest -q tests/test_plugin_hot_reload.py` → all pass。
- `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase candidate` → host 修改隔离 cache 并向真实 `main.py` 发送 SIGHUP，验证同 ID v1 → invalid → v2 时 active 始终为 v1、invalid 被拒绝、v2 进入 prepared；同时验证 initialize 失败的候选没有泄漏 tool／handler。

### Step 3: 接入 Skills、MCP、Jobs 与 Proactive Host

Skill Host 能编译 workspace、builtin 和候选 plugin roots，不再把插件软链接视为运行时真相源；本阶段只建立候选 catalog 和失效能力，不切换在途 turn。

MCP client 使用内部 generation identity。候选先完成 initialize、capability negotiation 和 tools/list，再把 wrapper 放入 candidate ToolCatalog。公开 server/tool 名不带 generation；Host ready 后执行依赖候选服务的 readiness semantic checks。本阶段验证候选可预热、检查和关闭，不实现 live routing。

JobHost 使用稳定 `<plugin_id>:<job_id>` 保存调度状态，Proactive source 使用稳定 source ID 保存 poll/ACK 状态。本阶段只建立 candidate host、readiness 和 close；新旧 handler 路由与 tick 边界在 Step 4 的 snapshot lease 上实现。

**G2/G3-readiness Verify**:

- MCP initialize/tools/list 失败：旧 client 和旧 tool catalog 不变。
- MCP spec 声明的远端 tool 缺失时 readiness 失败，候选 client 关闭。
- Job、proactive source 和 MCP client 都能预热并由 scope 完整关闭。
- `pytest -q tests/test_plugin_hot_reload.py tests/proactive_v2 tests/test_plugin_manager.py` → all pass。
- `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase capability-hosts` → 真实 `main.py` 内完成 MCP、Job、tick readiness 与关闭；本阶段不提前要求跨代际发布。

### Step 4: 建立 RuntimeSnapshot、执行 lease 与安全回滚

RuntimeSnapshot 一次性包含 lifecycle graphs、event handlers、tool hooks、tool catalog、skill catalog、generation-bound job catalog、proactive declarations、Channel sender 和 Dashboard/Memory dispatcher binding。Channel outbound 与 Dashboard request 从已持有的 snapshot 选择 endpoint，不读取独立全局 slot。发布和回滚都只替换一个 snapshot 引用；Job queue envelope 固化入队时的 snapshot identity 和 handler 引用，不能在出队时重新查 current。

Passive turn、proactive tick、job、tool execution、EventBus queue envelope 和 Dashboard request 都在入口获取 snapshot lease，在完成时释放。同一次执行禁止重新读取 current snapshot。

替换 `add_*_plugin_modules()` 的永久追加语义，改为从 snapshot 构建或选择 phase graph。Tool schema、tool search 和 tool execute 必须来自同一 snapshot。发布使用固定状态机：`PREPARED → PUBLISHED_PENDING → COMMITTED／ABORTED`。进入 pending 时原子发布 v2，同时保留 v1 rollback hold；Reconciler 在 5 秒内执行无业务副作用的 post-publish invariants，包括 active snapshot identity、snapshot 中 generation-keyed endpoint binding 可用、候选 Host alive 和资源计数。全部通过即 COMMITTED 并释放 v1 hold；失败或超时即 ABORTED，先恢复 v1 active hold并重新发布 v1。已经持有 v2 lease 的执行继续完成，v2 等待 lease 清零后回收。

**G4 Verify**:

- 阻塞 v1 turn，在中途发布 v2：该 turn 所有 marker 均为 v1，下一 turn 所有 marker 均为 v2。
- 阻塞 v1 tool call、EventBus queued event 和 proactive tick，重复同样断言。
- Candidate snapshot 只要任一 graph/index 构建失败，current snapshot identity 不变。
- v2 发布后触发 rollback：后续执行回到 v1，已经持有 v2 lease 的执行安全完成，最后 v2 才清理。
- v1 rollback hold 在发布验收完成前始终保留其 MCP、Channel 和 Service；验收通过后才允许 v1 drain。
- Job 在 v1 入队、v2 发布后出队时仍调用 v1 envelope 中的 handler；新入队任务使用 v2 job catalog。
- 测试 Job 与 Proactive source 必须写出 generation marker；Controller 等待 v1 已入队／tick 已开始的结构化事件，再发布 v2、释放 barrier 并断言旧执行为 v1、新执行为 v2。
- `pytest -q tests/test_plugin_hot_reload.py tests/proactive_v2` → all pass。
- `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase snapshot` → 真实 `main.py` 中阻塞 v1 turn，中途更新隔离 cache，当前 turn 保持 v1，下一 turn 使用 v2。

### Step 5: 收敛 Channels、Dashboard、Memory Engine 与 Plugin Services

Channel contribution 声明通用 `ingress_mode`。`stable_adapter` 由 ChannelHost 持有唯一外部连接，入站消息在执行入口获取 RuntimeSnapshot，再路由到 generation handler；候选只验证 handler/sender，不启动第二个 ingress。无法分离连接的 `exclusive` channel 使用与 exclusive service 相同的 quiesce → execution lease drain → stop old → start new → publish 流程。RuntimeSnapshot 的 channel binding 选择 sender，不存在独立 current slot，也不允许候选提前双开 ingress。

Dashboard 使用 generation-keyed ASGI sub-app registry，请求 lease 从 RuntimeSnapshot binding 选择 sub-app，禁止插件直接修改全局 FastAPI route table或替换独立 current dispatcher。前端插件模块返回 dispose，更新时卸载 React root、formatter 和样式后再加载 revision URL。

Memory Engine 移入主插件 contribution，删除第二套动态 loader。核心通过稳定代理和当前 snapshot 选择 engine；同一 turn 的检索、工具和写入固定在同一 engine generation。

Fitbit monitor 声明为 exclusive service，禁止 standby 双开。候选阶段只验证模型、配置、命令和端口，不启动进程。切换先进入 quiescing，阻止新的相关 snapshot lease，只等待现有 v1 execution/service-use lease 退出，rollback hold 不参与 drain；随后停止 v1 monitor、启动 v2 并等待 ready，再发布含 v2 binding 的 pending snapshot。入口保持 quiesced，直到 post-publish invariants 通过并 COMMITTED 后才恢复。若失败则先停止 v2、重启 v1 monitor、发布 v1，确认恢复后再开放入口，因此不会破坏已经放行的 v2 执行。整个过程 monitor 数量不超过一个，data 目录不参与 generation 清理。

**G2/G3/G5 Verify**:

- Channel v2 启动失败时 v1 恢复收发；成功时旧 Channel 不再接收或发送。
- Dashboard 更新后旧 route、React root、CSS 和 formatter 数量不增长。
- Memory Engine reload 中的旧 turn 完整使用旧 engine，新 turn 使用新 engine，旧写入完成后才 close。
- Fitbit reload 前后数据目录 hash 不变、预测契约通过、monitor 进程始终不超过一个。
- `pytest -q tests/test_channel_host.py tests/test_dashboard_api.py tests/test_memory_engine_contract.py tests/test_plugin_hot_reload.py` → all pass。
- `npm run typecheck && npm run lint && npm run build` → exit 0。
- `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase endpoints` → 真实 `main.py` 内触发 Channel、Dashboard、Memory 与 managed service 的切换和失败恢复。

### Step 6: 建立 PluginControlPlane、Watcher 与 Reconciler

`manifest.toml` 只保存 plugin enabled 状态。Control Plane 从 manifest、已发现 source revision 和插件私有 config revision 计算 desired state；Runtime status 单独保存，不回写能力声明。

Watcher 只发送 invalidation hint。Reconciler 必须在启动、Watcher 事件和低频一致性检查时比较 desired/current；漏掉 Watcher 事件也能恢复。事件按 plugin_id 合并，reconcile 运行中出现更新时只保留最新 revision。

变化路由：manifest → enable/disable；Python/config/普通资源 → full generation；skill root → SkillCatalog；dashboard asset → Dashboard revision。源码暂时缺失或写入未完成时保留 last-known-good 并标记 degraded；明确 disable/uninstall 才卸载。

**G1-G5 Verify**:

- enable、disable、install、update、uninstall 都只通过 reconcile 改变 Runtime。
- 快速写入 v2、v3、v4，最终只激活 v4，不出现 v2/v3 副作用。
- 模拟丢失 Watcher 事件后，一致性检查仍发现 revision 变化。
- 写入半成品 plugin.py 时旧代际继续工作；文件稳定有效后自动切换。
- `pytest -q tests/test_plugin_hot_reload.py tests/test_plugin_manager.py` → all pass。
- `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase reconcile` → 由 host controller 修改外部 sandbox cache，Runtime 容器内 Watcher 真实触发 reload、reconfigure 和 enable/disable。

### Step 7: 迁移全部插件并删除旧链路

迁移 core bundled plugins 和 `/mnt/data/coding/akashic-plugin` 全部插件。把隐藏的 dashboard、bot commands、memory engine、background task/process 纳入正式 contribution 或 PluginScope。

删除：平铺 manager capability lists、启动时永久 add、无法解绑的 direct subscriptions、skills 软链接真相源、Dashboard 文件名旁路、Memory Engine 第二加载器，以及任何旧 API 兼容壳。

每个插件必须提供至少一个结构检查；拥有外部进程、模型或持久状态的插件还必须提供只读 semantic check。

**G3/G5 Verify**:

- 所有插件通过 `plugin-doctor`，报告 active revision、能力清单、semantic checks 和 resource ownership。
- 对每个插件执行 load → reload → disable，旧 generation resource count 最终为零。
- 公共协调层不包含具体插件名：`rg -n "fitbit|feed-mcp|calendar-mcp|emotion|proactive_feedback" agent/plugins bus proactive_v2 bootstrap` → 除测试 fixture/文档外无匹配。
- `pytest -q tests/` → all pass。
- 对全部 changed Python files 运行 `pyright` → exit 0。
- `python docker/debug/plugin_hot_reload_probe.py --scenario full-runtime --phase all-plugins` → 在真实 `main.py` 中逐个 load → reload → disable 19 个非 Fitbit 插件；Fitbit 使用独立匿名 fixture Gate。

### Step 8: Docker 完整 Runtime 验收

扩展 host-side `docker/debug/plugin_hot_reload_probe.py`。Controller 启动独立 Runtime container，其 PID 1 由 entrypoint 执行 `python main.py`；Controller 通过 Unix socket 和共享的外部 sandbox 驱动 Runtime，不能让 probe 子进程冒充 Runtime。Controller 在隔离 cache 安装全能力测试插件，制造 active turn、MCP call、job、proactive tick 和 Dashboard request，再修改 sandbox 内源码、config 和 manifest。

Controller 复用 `context_probe.py` 的进程 readiness/cleanup；Runtime 保持正式 Provider wiring，通过同一 Compose network 的 OpenAI-compatible mock sidecar 和 `base_url` 获得确定性响应，不 monkeypatch `AppRuntime.start()`。新增 `docker/debug/model_gate.py`：`POST /control/scripts` 装载脚本化文本、tool-call 或 stream 响应；`/v1/chat/completions` 在命名 barrier 记录 request 后暂停；`POST /control/barriers/{id}/release` 释放；`GET /control/events` 返回结构化请求顺序。这样 Snapshot Gate 能确定性制造“v1 已进入、更新 v2、释放 v1”。触发入口分别为：IPC 驱动 passive turn、容器内 HTTP 驱动 Dashboard、sandbox 文件驱动 Watcher、短周期配置驱动 Job/Proactive、结构化 status API 读取 generation/Gate。

使用独立 `plugin-reload-gate` profile，绝不挂载正式 workspace、正式 HOME 或宿主插件缓存。Fitbit 从只读 canonical source 复制到 sandbox cache 后安装；验证使用匿名 fixture 和测试凭据，没有可用测试凭据时只运行本地模型与 monitor 契约，不访问真实账户。

**G6 Verify**:

- host controller 使用独立 `docker-compose.plugin-gate.yml` 构建当前镜像 → exit 0。
- 先运行 `sandbox-integrity`，再运行 `full-runtime`；所有 scenario 返回 passed GateResult。
- `full-runtime` 报告必须证明 `main.py`、AppRuntime、Watcher、Reconciler 和 capability hosts 全部实际启动。
- `pytest -q tests/`、`npm run typecheck`、`npm run lint`、`npm run build` → 全部 exit 0。
- 连续 20 次 reload 后，handler/task/process/connection/module generation 数量无单调增长。
- Runtime 不重启，下一 turn/tick/job 确实使用新 generation。
- 核心仓库和全部插件仓库的运行前后状态完全一致；所有 Runtime 写入只存在于 `/sandbox`。

## Test plan

- `tests/test_plugin_hot_reload.py`：代际、scope、snapshot、rollback、rapid revision、资源泄漏、各 capability host。
- `tests/test_plugin_manager.py`：保留 discovery、manifest、config 和程序化能力声明的既有契约。
- `tests/test_channel_host.py`：Channel 成功切换与失败恢复。
- `tests/test_dashboard_api.py`：ASGI sub-app 替换和旧请求排空。
- `tests/test_memory_engine_contract.py`：Memory Engine snapshot consistency。
- `tests/proactive_v2/`：tick boundary、source state、MCP client generation。
- `tests_scenarios/test_plugin_hot_reload_runtime.py`：真实 provider 下下一 turn 生效。
- `/mnt/data/coding/akashic-plugin/fitbit-mcp`：匿名模型 fixture、monitor 单实例和数据不变。
- `docker/debug/plugin_hot_reload_probe.py`：完整 Runtime 综合 Gate。
- `docker/debug/docker-compose.plugin-gate.yml`：独立只读源码 Gate service，不改变普通 debug service。
- `docker/debug/model_gate.py`：可控 OpenAI-compatible sidecar 与并发 barrier。

## Done criteria

- [x] G-1-G6 都有结构化结果和失败测试。
- [x] 任一候选验证失败时，旧 generation 保持可用。
- [x] 同一 turn/tick/job/request 不跨 generation。
- [x] 旧 generation drain 后没有 subscription、task、process、MCP client、Channel 或 Dashboard resource 残留。
- [x] 所有现有插件迁移完成，无旧 API 兼容壳。
- [x] Fitbit 模型、持久数据和预测契约保持不变。
- [x] `pytest -q tests/` 通过。
- [x] changed Python files 的 `pyright` 通过。
- [x] `npm run typecheck && npm run lint && npm run build` 通过。
- [x] Docker 完整 Runtime probe 通过。
- [x] Docker Gate 无法写核心仓库、任一插件仓库或宿主插件缓存，运行前后状态完全一致。
- [x] `_handbook/plugins-tutorial.md` 说明代际、Gate、热重载语义和进程重启边界。
- [x] 除已知 `private_runtime` 状态、核心仓库和本次迁移的插件仓库外，没有无关文件变更。

## STOP conditions

停止并报告，不要临时绕过 Gate：

- 当前代码与本计划的关键入口已发生结构性变化。
- G-1 发现 `/app`、canonical plugin repo、宿主 HOME 或宿主插件缓存可写／被错误挂载。
- Gate 需要复用正式 workspace、正式账户或正式 Channel 才能完成验证。
- 为完成热重载必须在公共协调层硬编码插件名或业务 payload。
- 某能力无法归属 PluginGeneration 或 PluginScope。
- Candidate Gate 需要执行有用户副作用的业务调用才能判定通过。
- 新 snapshot 无法在一次指针发布中保持 lifecycle、tool 和 event 一致。
- Channel 或 Fitbit monitor 无法失败恢复到旧实例。
- Fitbit 行为基线、数据 hash 或预测契约发生变化。
- 同一 Gate 连续两轮针对性修复后仍失败。
- 完成步骤需要改动 Out of scope 内容。

## Maintenance notes

- 新增插件能力时，必须先确定它属于 Snapshot Capability、Managed Resource 或 Exclusive Endpoint，并补充对应 Gate；不要在 Reconciler 中加业务分支。
- PluginScope 的资源计数和 generation lease 是以后排查“热重载后重复执行”的第一证据，必须暴露给 doctor/status。
- Semantic Gate 只能验证插件自己声明的无副作用契约；它不能替代插件仓库测试和真实 Runtime scenario。
- 更新插件依赖中的原生动态库或核心 ABI 不属于插件热重载，必须明确提示进程重启。
