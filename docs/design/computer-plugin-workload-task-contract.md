# Computer 插件与 Workload 原子能力任务合同

- 状态：accepted / implementation authorized
- 日期：2026-08-31
- 基线：`origin/main@322f71a464eee3da99b018914bf4644f0b7338c5`
- 目标分支：`feature/computer-workload-plugin`
- 实现 worktree：`/mnt/data/coding/akasic-agent-worktrees/computer-workload-plugin`
- 关联条款：RUN-013～RUN-016、PLG-009～PLG-017、WSP-001～WSP-006、WEBUI-001～WEBUI-008
- 关联决策：[0052](../decisions/0052-plugins-declare-managed-workloads.md)、[0051](../decisions/0051-web-ui-composes-ordinary-plugin-modules.md)、[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)

## 1. 目标

Akashic 默认安装一个普通 `computer` 插件。插件启用后，Akashic 自动启动一台单用户、持久化的
Computer；禁用或卸载插件后停止它，并同时撤下 Tool、Skill 和 Chat UI，但保留插件数据。

Computer 内运行同一个 Chromium、同一个 profile、OpenCLI、结构化 Browser 和视觉 Computer Use。
OpenCLI 必须继续自动刷新登录状态，让用户只登录一次便能长期复用登录态。

```text
docker compose up -d
        │
        ├── akashic-core
        └── workload-controller
                    │
                    │ exact plugin generation
                    ▼
             computer workload
             ├── Chromium + one profile
             ├── OpenCLI
             ├── screen + mouse + keyboard
             └── Computer Gateway
```

## 2. 用户可见成功标准

- [ ] 默认安装的 `computer` 插件启用后自动启动 Computer，无需用户逐个启动 Browser 或 OpenCLI。
- [ ] 禁用插件会停止 Computer，并移除对应 Tool、Skill 和 Chat 标签；再次启用后登录态仍在。
- [ ] 普通卸载保留 `plugin-data`；永久删除 Computer 数据仍是名称不同、先备份并再次确认的操作。
- [ ] Agent 可以观察、点击、移动、拖动、滚动、输入、按键和等待。
- [ ] OpenCLI、结构化 Browser、视觉输入和人工接管操作同一个 Chromium/profile。
- [ ] OpenCLI 的登录刷新路径通过真实登录态持久化测试，不以进程健康替代。
- [ ] Chat 最右侧有可展开的通用工具区；多个普通插件可各自登记一个顶部标签。
- [ ] Computer 使用时可以自动提示或打开自己的标签；用户关闭后保持用户选择，除非发生新的明确请求。
- [ ] 本地单元、集成、Docker E2E、CDP、Playwright 和真实模型行为验证均有可审阅证据。

## 3. Change intent

```yaml
change_type: feature
semantic_delta: compatible
capability_owner: mixed
consumer_scope: [all_v3_plugins, builtin_computer_plugin, conversation_ui]
runtime_patch: required
runtime_patch_reason: >-
  插件 generation 当前只能声明 Core 子进程，无法声明由插件拥有、需要容器权限隔离的运行单元。
  把 Docker 命令放入 Computer 插件会复制 generation、readiness、cleanup 和 rollback owner。
authoritative_state_owner: >-
  Core owns plugin generation and desired workload membership; Workload Controller owns actual
  container effects; computer plugin-data owns the Chromium profile; Computer Gateway owns input control.
client_only_alternative: >-
  Chat-only implementation cannot start or stop the Computer, bind it to plugin generation, or protect
  the single writable profile.
concept_gate: required
concept_gate_reason: new public plugin atom, privileged controller boundary, lifecycle and Web mount
invariants:
  - builtin and external plugins use the same public API
  - Core never receives the Docker socket
  - one exact workload belongs to one exact plugin generation
  - candidate workload never mounts formal writable plugin-data
  - only one formal Computer writes the persistent Chromium profile
  - workload readiness settles before its plugin snapshot becomes usable
  - workload cleanup failure remains visible and owned
  - plugin disable or uninstall stops the workload but does not delete plugin-data
  - all Computer mutations pass through one input owner gate
  - OpenCLI login refresh keeps using the one persistent Chromium profile
protected_state:
  - formal Akashic workspace
  - sessions.db and all messages
  - existing plugin-data outside computer
  - current hua-home Chromium profile until an explicit backed-up migration
  - /home/huashen/.cloakbrowser
allowed_paths:
  - agent/plugin_composition/**
  - agent/plugins/**
  - agent/workloads/**
  - bootstrap/**
  - docker/**
  - plugins/computer/**
  - plugins/conversation_ui/**
  - frontend/chat/src/**
  - tests/**
  - docs/**
  - scripts/**
  - external huayue-skills source in its own worktree and PR
forbidden_paths:
  - plugin cache
  - formal workspace
allowed_effects:
  - generated Web plugin bundles produced by scripts/build-web-plugins.mjs
  - isolated test containers, networks, volumes and workspaces
  - read-only inspection of Memoh and grokbot
  - backed-up hua-home test configuration after local gates pass
forbidden_effects:
  - public Docker API or public VNC/CDP
  - direct Core access to docker.sock
  - two writers for one Chromium profile
  - editing installed plugin cache
validation:
  - targeted pytest for declaration, controller, generation, candidate and rollback
  - plugin install/disable/uninstall E2E with isolated plugin-data
  - frontend unit/build and Playwright keyboard/layout tests
  - CDP verification against an isolated copy of /home/huashen/.cloakbrowser
  - real model turn using hua-home provider after local gates pass
  - python docker/debug/gate.py run --base origin/main
rollback: >-
  Stop and remove only containers labeled with the test run ID, restore the prior plugin generation,
  keep plugin-data, and reset deployment pointers to the recorded base without rewriting formal data.
worktree_writer: /root in /mnt/data/coding/akasic-agent-worktrees/computer-workload-plugin
external_revisions:
  - /mnt/data/source-code/Memoh@8eb3b667f7d4021206ea44fa8d933a13e801e746
  - /mnt/data/source-code/grokbot@a9f633e09d49a85829b8236331b9e21f7e612634
```

## 4. 最少领域词

只新增下面三个公共词：

| 词 | 唯一含义 | 不拥有 |
|---|---|---|
| `Workload` | 插件声明、Core 随 generation 管理的一个外部运行单元 | Docker 权限、业务协议、持久数据删除 |
| `WorkloadPort` | Workload 对同 generation 消费者开放的一个命名端口 | 公网发布、业务 route、独立生命周期 |
| `ConversationTab` | conversation-ui 右侧工具区中的一个可撤销标签 | Computer 状态、全局导航、独立 generation |

`WorkloadData`、`WorkloadHealth`、`WorkloadLimits` 和 `WorkloadEnv` 是不可变字段值，不是独立产品概念。
`Registration`、`Definition`、`Binding`、`Registry` 和 `GenerationHost` 继续是现有组合实现词。`Computer` 是
第一个 Workload 消费者，不进入 Core 分支。

首版不增加 `Pool`、`Machine`、`Instance`、`RuntimeProvider`、`DriverRegistry`、`ComputerId`、租户、休眠池、
任意 Compose DSL 或 Kubernetes 对象。

## 5. 公共插件能力

### 5.1 Workload 声明

插件通过当前 Root 的 `WORKLOADS` Service 登记不可变声明：

```python
await ctx.require(WORKLOADS).register(
    ctx,
    Workload(
        name="computer",
        image="ghcr.io/.../akashic-computer@sha256:...",
        command=("computer-gateway",),
        ports=(WorkloadPort(name="gateway", number=8080),),
        data=(WorkloadData(name="state", target="/data"),),
        health=WorkloadHealth(port="gateway", path="/health"),
        limits=WorkloadLimits(memory_mb=2048, cpu_count=2.0, pids=512),
    ),
)
```

名字使用普通名词：`Workload`、`Port`、`Data`、`Health`、`Limits`、`start`、`stop`。不使用
`orchestrate`、`materialize`、`reconcile` 或 Computer 专属名字描述公共 API。

`image` 首版必须使用 digest，`command` 必须非空且固定，不隐式继承镜像默认命令。插件不能声明 privileged、host network、device、capability、Docker socket、
宿主任意路径或公开端口。`WorkloadData.name` 只能映射当前插件 data root 下的受控子目录。

同一份声明必须先出现在不导入 Python 的 `akashic.plugin.toml` 中。静态 manifest 将 image digest、
command、命名端口、data name/target、health 和 limits 编入 artifact identity digest。Root 冻结后，Core
对静态声明与 `WORKLOADS.register()` 逐字段对账；缺失、多出或漂移全部拒绝。静态声明
只存 data name，不存主机路径。

v1 health 只支持对命名端口的 HTTP GET；任何 `2xx` 为 ready，总 deadline 由声明给出并有固定上限。
超时、取消和最后一次状态码或连接错误进入诊断回执，不增加可编程 health DSL。

### 5.2 端点交给 MCP

现有 MCP 继续由 Core 拥有 stdio client。插件可声明一个轻量 MCP adapter，并用新的明确字段取得
Workload 端口：

```python
McpServer(
    name="computer",
    command=("python", "mcp_server.py"),
    workload_env=(
        WorkloadEnv(env="COMPUTER_URL", workload="computer", port="gateway"),
    ),
)
```

不把 `process` 与 `workload` 塞进一组可空字段。现有 `EndpointEnv(process=...)` 保持原义；新的
`WorkloadEnv` 只表达 Workload 端口。它也进入静态 manifest 和 identity digest。Controller 返回已验证的
完整端点回执，Core 只注入该回执的值，不猜 scheme 或 authority。MCP、Workload 与 generation
必须同 owner；v1 拒绝跨插件引用。未来共享能力由提供者另行发布 Service，不放宽此绑定。

### 5.3 Chat 工具区

`conversation-ui` 的 Web module 声明 `conversation.tools.v1`，cardinality 为 `list`。子 entry 合同只有：

```ts
interface ConversationTab {
  id: string;
  label: string;
  order: number;
  render(host: HTMLElement, view: ConversationTabView): void | (() => void);
}

interface ConversationTabView {
  active: boolean;
  requestAttention(noticeId: string): void;
}
```

顶部是标签，不叫 `tag`；用户界面使用“标签”，entry 使用浏览器常用的 `ConversationTab`。
`ConversationTabView` 是 generation-bound 父视图；它不暴露 `open()` 或可写父状态。conversation-ui 唯一拥有
展开、关闭、当前标签、已处理 notice ID、键盘导航和窄屏行为。一个新的、未处理的 `noticeId`
是“新的明确请求”；重放同一 ID 不能重新打开。Computer 只发自己的 notice，不直接改父 UI。

```text
┌──────────────────────────────────────────────────────────────┐
│ Sessions │                 Chat                 │  工具 ▸    │
├──────────┼──────────────────────────────────────┼────────────┤
│          │                                      │ Browser X  │
│          │                                      ├────────────┤
│          │                                      │ 同一屏幕   │
│          │                                      │            │
└──────────┴──────────────────────────────────────┴────────────┘
```

多个插件只增加标签：

```text
conversation.tools.v1
├── computer: Browser
├── files: Preview
└── call: Transcript
```

父 entry 被撤销时，现有 Web mount disposer 递归释放全部标签。没有消费者时，右侧工具按钮不出现。

## 6. Core 与 Controller 的责任

```text
插件声明 desired state
        │
        ▼
Core WorkloadGenerationHost
├── exact generation / candidate / formal
├── readiness / health / Incident
├── endpoint binding
└── cleanup tombstone / rollback
        │ narrow authenticated Unix socket
        ▼
Workload Controller
├── validate request again
├── Docker create/start/inspect/logs/stop/remove
├── labels + resource limits + private network
└── actual container ownership
        │
        ▼
Docker socket
```

- Core 是插件 generation 与 desired membership owner，但没有 Docker socket。
- Controller 是实际容器副作用 owner，但不能发布插件 snapshot、读取 SessionDB 或删除 plugin-data。
- Controller 只接受固定 schema，不接受命令行、Compose YAML、原始 Docker JSON 或 caller-supplied 主机路径。
- Controller 只管理带 Akashic owner、workspace、plugin、workload 和当前 lease labels 的容器。
- formal 稳定键是 `(workspace_id, plugin_id, workload_name, mode=formal)`，不使用进程内 generation 序号。
  candidate 键额外包含可恢复的 transaction ID。
- Core 对冻结声明做固定编码并传递 `spec_digest`。Controller 把稳定键、mode、data names、
  `spec_digest` 和 container ID 写入 Docker labels/受控状态。已有 identity 的 image、ports、limits、network、
  mount 或 mode 任一不同都 fail-loud。
- Controller 的 data 协议只接受 `(workspace_id, plugin_id, transaction_id, workload_name, data_name)`。
  Controller 从自己的受控 workspace root 推导路径，逐段 no-follow 检查并拒绝 symlink、越界和 owner 不匹配。
- 部署层把同一组非 root 数值 UID:GID 交给 Core 和 Controller；Controller 同时核对两个固定 data root 的
  owner，并把新建 data 子目录 chown 后重新 stat。插件不能指定运行用户；这样持久 data 可直接读写，
  也不为单个镜像增加新的身份配置轴。
- 主 Compose 创建一条专用 Workload bridge network，并让 Core 加入；Controller 只把动态 Workload 加入
  这条网络。它与可选的外部服务网络分开，因此单独 `docker compose up` 也能解析 Workload endpoint。
- Controller socket 只通过共享 runtime 目录挂给 Core，校验 Unix peer credential，不对宿主或网络公开。
  当前 Python 插件是安装时信任代码；Controller 隔离 Docker 权限和误用面，不宣称对同进程恶意代码构成 sandbox。
- Core 重启时先请求 `inspect/adopt` formal 稳定键。只有 spec 相同且容器真实 ready 时才接管；
  spec 不同时先得到强 stop 回执，才允许新 writer 启动。
- `adopt` 是 Controller 内的原子 formal owner 交接。它校验稳定键、spec digest、container ID 和
  实际 Docker image、command、user、ports、mounts、limits、network、security 和 running state 后，把当前
  Core generation 记为唯一 stop lease。回执包含旧 generation、新 generation、
  container ID 和 spec digest。交接失败时，Core 不得向新 generation 注入 endpoint 或发布 snapshot。
- create 后先持久化 cleanup lease 再 start；start 失败时 Controller 自己完成强 stop。若 cleanup 也失败，
  lease 继续留在 Controller state，后续请求只能恢复或清理，不能把孤儿容器当成无 owner 资源。
- UDS timeout、断连或调用方取消属于“副作用未知”，不是普通失败。Core 保留 pending request，cleanup 用同一请求
  幂等取回 lease 后再 stop；Controller 等 Docker effect 结束后才恢复 cancellation。已完成 stop 的 lease 与 mount
  证据会持久化，响应丢失后的相同 stop 可重验并返回同一强回执。
- stop 只接受 formal 稳定键加最新 adopt receipt，或 candidate 的 transaction identity；未知、
  过期 lease 或标签不匹配的容器 fail-loud。
- stop/remove 成功回执必须同时证明 container ID/spec 匹配、`inspect=absent` 且受管 mount 已释放。
  Workload 合同不伪造通用“应用锁已释放”字段；单 writer data 声明和仅受管容器可挂载该目录构成 Core
  的 writer 栅栏，Computer Gateway 另在 readiness 中验证 Chromium 自己的 profile lock。
- Controller 在 Docker delete 前先 fsync `lease + mount sources + complete=false`；delete 后重验并写
  `complete=true`。即使 Controller 在两步之间崩溃，重启仍用原 source 集合完成释放证明。
- 一次 cleanup 中已取得强 stop 回执的 entry 立即退出待清理集合；重试只处理仍由 generation 持有的 entry，
  不重复使用已经失效的 stop lease。

Core 复用 `ManagedProcessGenerationHost` 已证明的启动、readiness、watch、cleanup tombstone 和 recovery
语义，但 `WorkloadGenerationHost` 独立拥有容器特有的 image、port、data 和 limits，不向
`ManagedProcessDefinition` 增加大量可空字段。

Workload 是 0036 中的窄同步参与者：只因 `WorkloadData` 可含 WSP-006 单 writer 状态而参与
admission close → lease drain → stop receipt → new ready → publish/restore。这不改变 MCP 和 managed process
默认的非同步语义，也不给其他 runtime 扩权。

同一插件可以让多个 Workload 引用同名 data，但同一个 data name 最多一个 `writable=true` 声明；其他引用
必须只读。Controller 单实例锁、串行 effect lock 和最新 stop lease 共同保证控制面只有一个 owner。

## 7. generation 顺序

候选顺序固定为：

```text
freeze Root
  → start candidate Workload with isolated data
  → wait Workload health
  → inject candidate Workload URL
  → start candidate MCP and read tools/list
  → attached child behavior check
  → parent Turn terminal barrier
```

正式切换固定为：

```text
drain old snapshot leases
  → stop old formal Workload
  → prove container absent and managed mounts released
  → start new formal Workload with formal plugin-data
  → wait health and MCP handshake
  → publish new stable
```

`stop old formal Workload` 包含 remove；后续 writer 证明来自 container absent、managed mount released 和
单 writer data owner，不再做第二次 remove。

若新正式启动失败，Core 停止它并用旧 image/spec 重新启动旧 formal。旧服务真实 readiness 通过前，
rollout 保持 degraded，不把 pointer 恢复冒充服务已恢复。
在回滚中，新 formal 也必须先返回上节的强 stop 回执；未证明 container/mount 全部
释放时，禁止重启旧 formal，并保留唯一可重试 failure owner。

候选永远使用 candidate data root。即使插件声明 `candidate_data_mode = "shared_read"`，包含可写 profile 的
WorkloadData 也不能挂正式目录；Computer candidate 只使用隔离复制或空目录完成协议验证。

## 8. Computer 插件责任

内置 `computer` 与外部插件走同一 loader、manifest、Root、Service、Effect、generation 和卸载路径。
插件包负责：

- Workload 声明与固定 image digest；
- Gateway：`/health`、`/activity`、`/screenshot`、`/input`、`/opencli`；
- 同一 Chromium/profile 的启动与监督；
- OpenCLI daemon/extension 配对和登录自动刷新；
- `browser`、`computer_observe` 与 `computer_action` MCP Tool；
- `computer` Skill 及选择顺序说明；
- `conversation.tools.v1` 中的 Browser 标签；
- Agent 与用户输入都通过同一个 Gateway 校验；
- profile 与登录刷新状态的插件数据 schema。

Core 不出现 `computer`、`browser`、`opencli`、`chromium` 或 `human takeover` 分支。

Computer Gateway 的 formal readiness 必须同时证明 Chromium CDP、OpenCLI daemon、extension 和
connectivity 可用。登录态是各站自己的业务状态，不混入进程 health；自动 refresh 成功与失败写入明确日志，
失败后 15 分钟重试，成功后每 12 小时刷新。

## 9. Agent 能力

视觉动作首版只有：`observe`、`move`、`click`、`double_click`、`scroll`、`type`、`key` 和 `wait`。
Gateway 只接受这组固定动作和有界参数。Agent 通过 MCP 调用；用户在 Chat 工具区点画面、发送文字或发送
Tab、Shift+Tab、Enter、Escape。两条路径不建立第二套浏览器或 profile。

能力选择顺序：

```text
API/CLI → OpenCLI adapter → OpenCLI Browser → visual Computer → 用户完成登录
```

## 10. 数据与迁移

| 数据 | 正常增加/更新 | 普通卸载 | 物理删除 owner |
|---|---|---|---|
| Computer image/container | Controller 创建、替换 | 停止并移除 | Controller |
| `plugin-data/computer/state/profile` | Chromium 原位更新 | 保留 | 独立永久删除操作 |
| `plugin-data/computer/state/state` | Gateway 与 OpenCLI 原位更新 | 保留 | 独立永久删除操作 |
| candidate data | candidate 写入 | remove 强回执后删除 | Core rollout cleanup journal |
| Skill symlink | generation 投影 | 移除 | PluginSkillLinker |
| Chat tab | browser catalog 内存 | disposer 撤销 | BrowserCatalogSession |

hua-home 现有 profile 迁移前必须停止唯一 writer，创建可校验恢复点，再移动或绑定到 Computer plugin-data。
本 PR 的代码和隔离 E2E 不自动迁移正式 profile；正式迁移需要单独的执行前清单和维护窗口。

candidate root、container ID、spec digest 和 transaction ID 一同进入现有可恢复 reload journal。只有
Controller remove 强回执后才能删 candidate root；删除或回执失败保留 retry owner。formal plugin-data
绝不进入这条清理路径。

`huayue-skills` 中 `opencli` 只能在 Computer artifact 已可安装且名字所有权切换不产生重复时删除。Core 的
跨插件 Skill 重名 Gate 保持 fail-loud，不增加临时双 owner 或静默覆盖。

## 11. 验证矩阵

### 11.1 Core

- 不导入代码的静态 Workload/WorkloadEnv admission 与 Root 逐字段对账；
- 声明 schema、digest 和路径逃逸；
- 同 Root 重名、跨 generation identity；
- candidate 独立 data、端口和容器名；
- start cancel、health timeout、watch failure、stop timeout；
- cleanup tombstone 与 retry；
- formal 切换失败恢复旧服务；
- Core 崩溃后 inspect/adopt 同 spec formal，不产生第二个 profile writer；
- Controller 身份、label、socket auth 和禁止字段；
- Core 进程无 Docker socket。

### 11.2 Computer

- 同一 profile 重启后 cookie/local storage 保留；
- OpenCLI daemon、extension、connectivity 与真实登录刷新；
- screenshot 尺寸、体积边界与不落盘；
- 全部输入动作、参数边界和崩溃后 activity 收束；
- disable/uninstall 停容器但保留数据；
- MCP tools/list 与一次真实 Tool 调用。

### 11.3 Chat

- 无 ConversationTab 时不显示工具按钮；
- 一个和多个标签的排序、选择、关闭、卸载；
- 点击按钮展开，Escape 关闭，方向键切标签，焦点可见；
- 用户可通过画面点击或键盘按键与隐藏文字输入完成登录；
- 对话宽度、composer、滚动锚点和窄屏无回归；
- module dispose 清理 iframe、listener、timer 和请求；
- Playwright 截图与 Memoh 参考只比较布局目的，不复制其多 Bot 产品结构。

### 11.4 E2E

1. 使用一次性 workspace/plugin-home/data root 启动 Core、Controller 和 Computer。
2. 用 CDP 连接 Computer 内 Chromium，写入测试 cookie，重启插件并证明 cookie 仍在。
3. Playwright 打开 Chat，展开 Browser 标签，验证画面、多个测试标签、键盘与响应式布局。
4. 让模型通过 Skill 选择 OpenCLI，再执行一次 visual fallback，并核对 tool trace。
5. 禁用插件，证明容器、Tool、Skill 和 UI 消失而 data checksum 不变。
6. 清理仅带本次 run label 的容器、网络、临时数据和进程。

## 12. 分批交付与停止条件

1. 本合同与 0052 经两个独立 Terra High 只读评审，must-fix 清零。
2. Core Workload 原子先通过硬 Gate：一个主仓库不认识名字的外置 fixture，仅通过静态 manifest
   和公共 API 完成 candidate/formal、MCP `WorkloadEnv`、disable/uninstall 和 recovery。然后分别做正交性
   与真实代码逻辑评审；must-fix 清零前不创建 `plugins/computer`。
3. Computer 与 Chat 完成后运行全部本地 Gate，再使用 hua-home 模型做真实行为验收。
4. 最终由 Terra XHigh 审查完整 diff；另一 reviewer 按 `reclaim-code-entropy` 只判断能否用更少概念和代码。
5. 所有 review 后的架构修改重跑对应审查。
6. 只有验证、外部 Skill owner 迁移和文档对账完成后才提交可合并 PR。

停止并报告：Docker/Controller 权限要求 Core 挂载 socket；候选必须写正式 profile 才能通过；无法可靠恢复旧
formal；huayue-skills 无法形成无重复/无空窗的 owner 切换；或正式主机测试需要扩大公网、数据删除或未授权部署。
