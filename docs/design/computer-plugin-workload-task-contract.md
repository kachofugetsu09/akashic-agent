# Computer 插件与 Workload 原子能力任务合同

- 状态：implemented / verified
- 日期：2026-08-31
- 基线：`origin/main@322f71a464eee3da99b018914bf4644f0b7338c5`
- 目标分支：`feature/computer-workload-plugin`
- 实现 worktree：`/mnt/data/coding/akasic-agent-worktrees/computer-workload-plugin`
- 关联条款：RUN-013～RUN-016、PLG-009～PLG-017、WSP-001～WSP-006、WEBUI-001～WEBUI-008
- 关联决策：[0053](../decisions/0053-plugins-declare-managed-workloads.md)、[0051](../decisions/0051-web-ui-composes-ordinary-plugin-modules.md)、[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)

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
             ├── Xvnc desktop + window manager
             ├── RFB screen + full pointer + keyboard + clipboard
             └── Computer Gateway
```

## 2. 用户可见成功标准

- [x] 默认安装的 `computer` 插件启用后自动启动 Computer，无需用户逐个启动 Browser 或 OpenCLI。
- [x] 禁用插件会停止 Computer，并移除对应 Tool、Skill 和 Chat 标签；再次启用后登录态仍在。
- [x] 普通卸载保留 `plugin-data`；永久删除 Computer 数据仍是名称不同、先备份并再次确认的操作。
- [x] Agent 可以观察、点击、移动、拖动、滚动、输入、按键和等待。
- [x] OpenCLI、结构化 Browser、视觉输入和人工接管操作同一个 Chromium/profile。
- [x] Chat 中的 Computer 是真实远程桌面，不是定时截图：用户可以完成移动、单击、双击、右键、
      中键、拖动、滚动、组合键、连续文字输入、剪贴板收发和窗口操作。
- [x] 远程桌面断线后自动重连；关闭再展开不丢失桌面、标签页、焦点以外的运行状态或登录态。
- [x] OpenCLI 的登录刷新路径通过真实登录态持久化测试，不以进程健康替代。
- [x] Chat 右上角有可展开的通用工具入口；展开后是可调整宽度的右侧分栏，多个普通插件可各自登记一个顶部标签。
- [x] Computer 使用时可以自动提示或打开自己的标签；用户关闭后保持用户选择，除非发生新的明确请求。
- [x] 本地单元、集成、Docker E2E、CDP、Playwright 和真实模型行为验证均有可审阅证据。

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
  container effects; computer plugin-data owns the Chromium profile; Xvnc owns the one visible desktop
  and RFB input stream; Computer Gateway owns bounded Agent actions.
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
  - human RFB input and Agent actions target the same Xvnc desktop and never a second browser or profile
  - a WebSocket is generation-bound and is closed before its snapshot can drain
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
  - frontend/dashboard/src/**
  - packages/akashic-web-ui-v1/**
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
| `WorkloadPort` | Workload 对同 generation 消费者开放的一个命名端口；可选声明正式代际的本机回环端口 | 公网发布、业务 route、独立生命周期 |
| `ConversationTab` | conversation-ui 右侧工具区中的一个可撤销标签 | Computer 状态、全局导航、独立 generation |

`WorkloadData`、`WorkloadHealth`、`WorkloadLimits`、`WorkloadEnv` 和 `user_namespaces` 是不可变字段值，
不是独立产品概念。
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
        ports=(
            WorkloadPort(name="gateway", number=8080),
            WorkloadPort(name="opencli", number=19826, loopback=19825),
        ),
        data=(WorkloadData(name="state", target="/data"),),
        health=WorkloadHealth(port="gateway", path="/health"),
        limits=WorkloadLimits(memory_mb=0, cpu_count=0.0, pids=0),
        user_namespaces=True,
    ),
)
```

名字使用普通名词：`Workload`、`Port`、`Data`、`Health`、`Limits`、`start`、`stop`。不使用
`orchestrate`、`materialize`、`reconcile` 或 Computer 专属名字描述公共 API。

`WorkloadLimits` 的单项值 `0` 表示不限制；Computer 默认不限制 CPU、内存与 PID，与 Memoh workspace 的默认资源语义一致。
`image` 首版必须使用 digest，`command` 必须非空且固定，不隐式继承镜像默认命令。插件不能声明 privileged、host network、device、capability、Docker socket、
宿主任意路径或公开端口。`WorkloadPort.loopback` 只能发布到宿主 `127.0.0.1`，只对 formal generation 生效；
candidate 不发布，避免与当前正式代际争抢端口。`WorkloadData.name` 只能映射当前插件 data root 下的受控子目录。
`user_namespaces` 默认关闭；打开时只选择 Core 固定并审阅过的 seccomp profile，让非 root 进程创建自己的
Linux user namespace。插件不能提交 profile、syscall 或其他 Docker security option。

同一份声明必须先出现在不导入 Python 的 `akashic.plugin.toml` 中。静态 manifest 将 image digest、
command、命名端口、data name/target、health、limits 和 `user_namespaces` 编入 artifact identity digest。Root 冻结后，Core
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
  onActiveChange(listener: (active: boolean) => void): () => void;
  requestAttention(noticeId: string): void;
}
```

顶部是标签，不叫 `tag`；用户界面使用“标签”，entry 使用浏览器常用的 `ConversationTab`。
`ConversationTabView` 是 generation-bound 父视图；它不暴露 `open()` 或可写父状态。conversation-ui 唯一拥有
展开、关闭、当前标签、分栏宽度、已处理 notice ID、键盘导航和窄屏行为。`onActiveChange` 只报告父级
拥有的可见事实，让子标签暂停昂贵画面或恢复焦点；它不授予子标签写父状态的能力。一个新的、未处理的 `noticeId`
是“新的明确请求”；重放同一 ID 不能重新打开。Computer 只发自己的 notice，不直接改父 UI。

```text
关闭：

┌──────────────────────────────────────────────────────────────┐
│ Sessions │                 Chat                     Computer │
└──────────────────────────────────────────────────────────────┘

展开：

┌──────────────────────────────────────────────────────────────┐
│ Sessions │          Chat           ║ Computer              × │
├──────────┼─────────────────────────║─────────────────────────┤
│          │                         ║ 同一台实时桌面           │
│          │                         ║ 鼠标、键盘和剪贴板直达   │
└──────────┴─────────────────────────║─────────────────────────┘
```

多个插件只增加标签：

```text
conversation.tools.v1
├── computer: Computer
├── files: Preview
└── call: Transcript
```

父 entry 被撤销时，现有 Web mount disposer 递归释放全部标签。没有消费者时，右侧工具按钮不出现。

### 5.4 Dashboard 实时通道

Web Host 在现有窄 HTTP client 上增加一个普通原子：

```ts
interface WebHostHttp {
  request(path: string, init?: RequestInit): Promise<Response>;
  webSocketUrl(path: string): string;
}
```

`webSocketUrl()` 只接受当前 origin 下的 `/api/dashboard/` 路径，并把 exact snapshot、catalog、module 和
generation 身份绑定进 URL。Dashboard Host 只接受同 origin 的 WebSocket 握手，按同一身份选择当前插件 route；
插件不能访问兄弟 route。snapshot 停止接纳 lease 时，Host 以 service-restart close code 关闭连接并释放 lease，
客户端按新 catalog 重建连接。身份值不是长期 credential，不进入日志正文、插件数据或 URL 以外的持久状态。

该能力不认识 RFB、Computer 或浏览器。Computer 只是第一个消费者：它把 Workload 内部 RFB WebSocket 通过
自己的 dashboard route 透传给自己的 Web module。VNC、CDP 和 Workload 私网端口仍不向宿主或公网发布。

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
- 普通 Workload 继续使用 Docker 默认 seccomp；声明 `user_namespaces=true` 时，Controller 只换用 Core
  固定的默认策略派生 profile，仍保持非 root、`cap-drop=ALL` 和 `no-new-privileges`。profile 不进入插件输入。
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
  这条网络，并固定提供 `host.docker.internal:host-gateway`。它与可选的外部服务网络分开，因此单独
  `docker compose up` 也能解析 Workload endpoint，Workload 也能用稳定名称访问 Docker 宿主网关。
- 没有配置 Controller socket 的旧式本地部署不会启用内置 Workload 插件，并记录明确 warning；用户安装并
  启用的外部 Workload 插件仍然 fail-loud。这个 deployment admission 按静态 Workload 声明判断，不认识
  Computer 名字。
- Controller socket 只通过共享 runtime 目录挂给 Core，校验 Unix peer credential，不对宿主或网络公开。
  当前 Python 插件是安装时信任代码；Controller 隔离 Docker 权限和误用面，不宣称对同进程恶意代码构成 sandbox。
- 正式 Compose 把 Controller 绑定到明确的 Core container owner。Controller 一旦观察过 owner running，随后
  看到 absent/stopped 就用持久 exact lease 强 stop 全部 Workload；Controller 先启动但 Core 从未 running 时，
  只给固定启动宽限期。Docker inspect 自身失败不等于 owner 停止，清理失败保留 lease 并按同一路径重试。
- Core 启动时仍请求 `inspect/adopt` formal 稳定键，恢复上次 stop 尚未收束的中间态。只有 spec 相同且容器
  真实 ready 时才接管；spec 不同时先得到强 stop 回执，才允许新 writer 启动。
- Controller 仅在其持久化的 exact old lease 与现有容器 ID/owner/spec labels 一致时，把 spec 变化解释为
  插件升级；它先完成旧 lease 的强 stop，再创建新容器。没有 lease 或真实 owner 漂移时 fail-loud。
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
- supervised 新 boot 由 Controller 先收束旧 Core owner 的 Workload，再装配当前 release。installed 插件仍必须通过 exact
  stable/latest pointer 证明恢复目标；builtin 没有该 pointer，只有 `candidate`（插件仍属于当前 release）
  可以跨 boot 恢复，并必须在新 snapshot 中重建为普通 builtin generation 后才封口 journal。

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

候选永远使用 candidate data root。旧 `candidate_data_mode = "shared_read"` 已删除；包含可写 profile 的
WorkloadData 也不能挂正式目录；Computer candidate 只使用隔离复制或空目录完成协议验证。

## 8. Computer 插件责任

内置 `computer` 与外部插件走同一 loader、manifest、Root、Service、Effect、generation 和卸载路径。
插件包负责：

- Workload 声明与固定 image digest；
- Gateway：`/health`、`/activity`、`/screenshot`、`/input` 和结构化 Browser route；
- 同一 Xvnc 桌面、轻量窗口管理器、Chromium/profile 与 RFB WebSocket bridge 的启动和监督；
- OpenCLI daemon/extension 配对和登录自动刷新；
- 直接通过 CDP 实现的 `browser_observe`、`browser_action`，以及视觉
  `computer_observe`、`computer_action` MCP Tool；
- `opencli` Skill；Skill 只教 Agent 通过普通 `shell` 调用 CLI，不把 CLI 参数包装成 Browser Tool；
- `conversation.tools.v1` 中的 Computer 标签；标签使用标准 noVNC RFB client，不自行模拟鼠标或键盘；
- 人工输入经 generation-bound dashboard WebSocket 到 RFB；Agent 视觉输入经 Gateway 到同一个 Xvnc display；
- profile 与登录刷新状态的插件数据 schema。

Core 不出现 `computer`、`browser`、`opencli`、`chromium` 或 `human takeover` 分支。

普通 Shell 中的 OpenCLI 客户端使用固定的 `127.0.0.1:19825`；formal Workload 将它转发到 Computer 的
`opencli:19826`，再由 Computer 转发到容器内 daemon。这样不修改 OpenCLI 的标准端口，也不让 Core 识别
OpenCLI 协议。插件停用时容器和回环端口一起消失，profile 仍保留。该端口只是通用
`WorkloadPort.loopback` 的第一个消费者，Core 不识别 OpenCLI 协议。Gateway 不再提供接收 argv 的
`/opencli` Browser route。

Computer Gateway 的 formal readiness 必须同时证明 Xvnc、窗口管理器、RFB WebSocket bridge、Chromium CDP、
OpenCLI daemon、extension 和 connectivity 可用。登录态是各站自己的业务状态，不混入进程 health；自动 refresh 成功与失败写入明确日志，
失败后 15 分钟重试，成功后每 12 小时刷新。

## 9. Agent 能力

Browser Use 分成只读 `browser_observe` 和写入 `browser_action`。前者首版提供 `snapshot`、`get_content`、
`get_url`、`get_title`、`screenshot` 和 `tab_list`；每次 snapshot 返回不透明 `snapshot_id`，ref 只在该快照、
标签页和文档内有效，导航或 DOM 节点失效后必须明确报 stale。后者提供 `navigate`、基于 snapshot ref 的 `click`、
`fill`、`type`、`press`、`scroll`、`wait`、前进后退、刷新和标签页操作。Browser Tool 直接使用 CDP，
点击和文字输入使用 CDP 原生 Input 事件；不得转发 OpenCLI argv，也不得让模型猜 CLI 语法。
截图结果原子写入 Computer plugin-data 下的有界文件集合并返回绝对路径；文字模型随后用
`read_image_vision` 读取，不能把 base64 图片正文塞回模型上下文。每次写入只保留最近 32 张截图。

视觉 Computer Use 首版只有：`observe`、`move`、`click`、`double_click`、`drag`、`scroll`、`type`、`key`
和 `wait`。Gateway 只接受这组固定动作和有界参数。Agent 通过 MCP 调用；用户通过完整 RFB client 直接使用
全部鼠标按钮、移动、拖动、滚轮、普通键、组合键、连续输入和剪贴板。两条路径不建立第二套桌面、浏览器或 profile。

`move`、`click` 和 `double_click` 使用 1280×800 画面中的 `x`、`y`；`drag` 额外使用同一边界内的
`to_x`、`to_y`。

能力选择顺序：

```text
OpenCLI Skill + shell → Browser Use → visual Computer Use → 用户完成登录
```

## 10. 数据与迁移

| 数据 | 正常增加/更新 | 普通卸载 | 物理删除 owner |
|---|---|---|---|
| Computer image/container | Controller 创建、替换 | 停止并移除 | Controller |
| `plugin-data/computer/state/profile` | Chromium 原位更新 | 保留 | 独立永久删除操作 |
| `plugin-data/computer/state/state` | Gateway 与 OpenCLI 原位更新 | 保留 | 独立永久删除操作 |
| `plugin-data/computer/screenshots` | MCP 原子写入并只保留最近 32 张 | 保留有界集合 | Computer MCP retention |
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
- Core 正常停止或异常退出后 Controller 强 stop/remove Workload 并保留 plugin-data；新 Core 只用
  inspect/adopt 收束尚未完成的中间态，不产生第二个 profile writer；
- Controller 身份、label、socket auth 和禁止字段；
- `user_namespaces` 默认关闭并进入 identity/spec；开启后只增加受限 user namespace 与 Chromium 所需
  `chroot`，不能变成 `seccomp=unconfined`，adopt 会核对实际 security option；
- Core 进程无 Docker socket。
- WebSocket route 与 HTTP route 分开查重；握手必须具有同 origin、完整 exact Web identity 和相同插件 owner；
- stale generation 不能建立新连接，已建立连接在 snapshot drain 时关闭并释放 lease。

### 11.2 Computer

- 同一 profile 重启后 cookie/local storage 保留；
- OpenCLI daemon、extension、connectivity 与真实登录刷新；
- Xvnc/RFB handshake、WebSocket bridge、screenshot 尺寸、体积边界、文件可读性与 32 张 retention；
- 全部输入动作、参数边界和崩溃后 activity 收束；
- noVNC 与 Agent 操作同一个 display；右键、中键、拖动、滚轮、组合键、剪贴板和重连可用；
- Chromium renderer 位于不同于容器 PID 1 的 user namespace，同时保持零 capability、
  `NoNewPrivs=1` 和 seccomp filter；
- disable/uninstall 停容器但保留数据；
- MCP tools/list 与一次真实 Tool 调用。

### 11.3 Chat

- 无 ConversationTab 时不显示工具按钮；
- 一个和多个标签的排序、选择、关闭、卸载；
- 关闭态入口只出现在 Chat 右上角，不永久占用整条右侧轨道；
- 点击入口展开，Escape 关闭，方向键切标签，拖动或键盘调整分栏，焦点可见且可恢复；
- 用户可直接在桌面完成登录，不出现独立的方向按钮或隐藏文字输入表单；
- 对话宽度、composer、滚动锚点和窄屏无回归；
- module dispose 清理 iframe、listener、WebSocket、RFB client、timer 和请求；
- Playwright 以 Cursor 的右上入口与分栏人体工学、Memoh 的真实远程桌面控制链为参考，不复制多 Bot 产品结构。

### 11.4 E2E

1. 使用一次性 workspace/plugin-home/data root 启动 Core、Controller 和 Computer。
2. 用 CDP 连接 Computer 内 Chromium，写入测试 cookie，重启插件并证明 cookie 仍在。
3. Playwright 打开 Chat，从右上角展开 Computer，验证可调分栏、多个测试标签、窄屏、关闭再展开和自动重连。
4. 通过 RFB 在真实页面完成左/右/中键、拖动、滚轮、组合键、文字输入和双向剪贴板，并证明第二次 Agent 操作
   会更新同一个持续画面而无需刷新 Chat。
5. 让模型加载 `opencli` Skill，通过普通 `shell` 执行 OpenCLI，再执行一次 Browser Use 和 visual fallback，
   并证明 Tool catalog 中没有转发 argv 的假 Browser Tool。
6. 禁用插件，证明现有 RFB/WebSocket 被关闭，容器、Tool、Skill 和 UI 消失而 data checksum 不变。
7. 清理仅带本次 run label 的容器、网络、临时数据和进程。

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

## 13. 2026-08-31 验收记录

- 最终候选为 `9d23dbbabeb69b3360a9b9c95c4a46d8b01ab1ad`；Computer image 固定为
  `ghcr.io/kachofugetsu09/akashic-computer@sha256:6fd3c605380a3daef5ddebb34f2905ee992d2b4e1490fbfb78dcce9f06a3dadb`。
  [GitHub Actions image build 33365143243](https://github.com/kachofugetsu09/akashic-agent/actions/runs/33365143243)
  通过，运行容器的 image ID 与 revision 已和该产物对账。
- 隔离 compose 部署只固定启动 Core 与 Workload Controller；默认 `computer` generation 自动创建正式
  Workload。真实 disable、uninstall、enable 和 Core restart/adopt 路径均通过；容器、Tool、Skill 与 UI
  随 generation 撤下和恢复，Computer data checksum 与登录态保持不变。
- 正式 Workload 中 Chromium renderer 与 PID 1 使用不同 user namespace；运行身份为 `1000:1000`，
  `CapEff=0`、`NoNewPrivs=1`、`Seccomp=2`，Docker 配置为 `cap-drop=ALL`，CPU、内存与 PID 不限制。
- Agent 的结构化 Browser、视觉动作和人工 noVNC 输入均操作同一 display/profile。真实页面验收覆盖
  move、单击、双击、右键、中键、拖动、滚轮、Enter、连续输入与窗口内导航；Agent 输入后人工继续输入，
  DOM 最终值为 `Agent started here | Human took over`。
- 双向剪贴板分别得到 `chat-to-computer-20260831` 与 `computer-to-chat-20260831`；全屏时工具区为
  1920×1080；关闭后立即重开和等待 31.5 秒重开都恢复为 `已连接`。
- Chat 工具区在 1920px 视口从 806px 经键盘调整到 830px、经鼠标拖动到 951px；关闭重开后宽度保持。
  900px 视口切换为 900px 全宽工作区，页面无横向溢出。新的 Agent Computer activity 会重新打开标签，
  重放旧 notice 不会覆盖用户关闭选择。
- `opencli auth status --site github -f json` 返回 `logged_in=true`；daemon 只监听 `127.0.0.1:19825`，
  动态 Skill 来自 `plugins/computer/skills/opencli`。真实 refresh、容器替换和插件重启后登录态仍可复用。
- 真实模型 turn `local-ed1518555a5f0bc1` 先动态加载 `mcp_computer__browser_action` 与
  `mcp_computer__browser_observe`，再导航 `https://example.com`、读取标题并回复“标题是 Example Domain。”；
  两次插件操作均记录 exact `computer` generation 与 success outcome，右侧 Computer 同时自动打开。
- 最终源码验证为 209 个相关 Python 测试、3 个 Computer Web module 测试、46 个桌面导航测试、
  TypeScript 类型检查和生产构建全部通过。全量 Change Gate 通过，报告为
  `docker/debug/reports/change-gate/20260831-150822-1edeb96d`，27 个公开合同场景无失败或残留 Docker 资源。
- `huayue-skills` 的旧 OpenCLI owner 已在独立
  [PR #6](https://github.com/akashic-plugins/huayue-skills/pull/6) 删除；该 PR 与本 PR 需要按“先让 Computer
  artifact 可用、再撤旧 owner”的顺序合并，不在本任务中越权合并。
