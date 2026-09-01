# 0053 · 插件声明受管 Workload

- 状态：accepted / implemented
- 日期：2026-08-31
- 关联条款：RUN-016、PLG-017、WEBUI-008、WSP-006
- 关联设计：[Computer 插件与 Workload 原子能力任务合同](../design/computer-plugin-workload-task-contract.md)

## 背景

v3 插件已经可以声明 Skill、Tool、MCP、Web module 和 Core 子进程，但不能声明需要容器隔离的长期运行
单元。让 Computer 插件直接执行 Docker 命令，会把 generation、readiness、candidate isolation、cleanup
和 rollback 复制到插件里，也迫使 Core 或插件取得 Docker socket。

Computer 又必须与插件共同启停，同时让 Chromium profile 和 OpenCLI 登录刷新状态在普通卸载后保留。

## 决定

Core 增加普通插件可消费的 `Workload` 声明能力。`Workload` 随 exact plugin generation 启动、检查、发布、
排空和停止；持久数据继续属于 plugin-data，不随 Workload 或普通卸载删除。

Docker 副作用由固定、窄权限的 Workload Controller 独占。Core 通过认证 Unix socket 发送固定 schema，
不取得 Docker socket；Controller 不读取插件代码、SessionDB 或任意 workspace 文件，也不决定插件晋升。

Workload 同时出现在静态 manifest 和冻结 Root 中，逐字段对账并纳入 artifact identity digest。
Controller 不接收主机路径；只按 workspace/plugin/data name 从自己的受控根目录推导挂载。
同一插件的同名 data 最多一个 Workload writer；Controller adopt 必须逐项核对 Docker inspect 的真实配置，
不能只相信 labels。create 后先持久化 cleanup lease，再启动容器。
传输失败或取消后，Core 以同一 request 恢复 lease 再清理；Controller 持久化 completed stop 证据，使相同
stop 在响应丢失后仍可安全重试。

首版 Workload 使用 digest 固定的 OCI image 和非空固定 command，只允许命名端口、当前插件 data root 子目录、明确资源上限和
HTTP health。它不接受 Compose、任意 Docker JSON、宿主路径、privileged、host network、device、capability
或公开端口。

Controller 为所有 Workload 固定提供 `host.docker.internal:host-gateway`。这只是 Core 拥有的网络拓扑名称，
不是插件声明字段，也不会把 Workload 改成 host network；插件仍不能改变 alias、目标或网络模式。

Workload 可选择一个普通布尔字段 `user_namespaces`。默认值为 false；true 只让 Controller 使用 Core 固定的
user namespace seccomp profile，仍保持非 root、`cap-drop=ALL` 和 `no-new-privileges`。插件不能传入任意
seccomp profile、syscall 列表或 security option。该字段进入静态 manifest、artifact identity、spec digest
和 Docker inspect 漂移核对。

部署层给 Core 与 Controller 配置同一组非 root 数值 UID:GID，Controller 核对固定 data root owner 后让
Workload 使用该身份。插件不能选择主机用户或 root；运行身份不是插件声明的新变化轴。
Controller 新建 data 子目录后 chown 并重新 stat；stop 在 Docker delete 前持久化 mount source 证据。

`computer` 是默认安装的普通插件和第一个消费者，不获得专属 Core API。它自己拥有 Xvnc Linux 桌面、
RFB 通道、Chromium、OpenCLI、Computer Gateway、输入控制权、Skill、Tool 和 UI。人工 RFB 输入与 Agent
动作只操作同一个 display 和 Chromium profile。

`conversation-ui` 作为普通插件声明 `conversation.tools.v1` 多 entry mount。Computer 和未来其他插件各自
登记一个顶部标签；Core Web Host 不认识 Browser 或 Computer。

Workload 是 0036 的窄同步例外：只当 data 包含单 writer 状态时，它参与现有
admission-close → lease-drain → stop-receipt → new-ready → publish/restore 事务。这不改变 MCP
或 managed process 的默认切换语义。

formal 容器使用稳定的 workspace/plugin/workload key、不可变 spec digest 和真实 container ID。正式部署把
Controller 绑定到 Core 容器 owner；Core 从 running 变为 absent/stopped 后，Controller 使用持久 exact lease
强 stop 全部 Workload，但不删除 plugin-data。Core 没有机会清理、清理响应丢失或 Docker 暂时不可用时，
Controller 继续保留并重试该 owner，不能把容器留给插件或人工巡检。

新 Core 启动仍以 inspect/adopt 作为恢复异常中间态的幂等路径；spec 不同则必须先取得容器已不存在、mount 与
受管 mount 已释放的强 stop 回执。通用合同不声称理解应用内部锁；同一 plugin data name 最多一个 writer，
Computer 再用自己的 readiness 验证 Chromium profile lock。没有强 stop 回执，正向切换和失败回滚都不得
启动第二个 writer。adopt 同时原子把 stop lease 从旧 Core generation 交给新 generation；未取得包含新旧
generation、container ID 和 spec digest 的 adopt receipt 时，新 generation 不得取得 endpoint 或发布。
若 Controller 持久化的 exact lease 与新声明 spec 不同，`start` 先用旧 lease 完成强 stop，再创建新容器；
它不把未知容器或真实配置漂移解释成升级。supervised 跨 boot 恢复 installed 插件继续依赖 durable artifact
pointer；内置插件没有第二套 pointer，恢复目标只能是当前不可变 release 中仍存在的 builtin generation。

## 理由

Workload 独占一个真实变化轴：插件 generation 所拥有的外部运行生命周期。它不是进程的改名，也不把
容器字段塞进 `ManagedProcessDefinition`。Controller 独占另一个真实边界：Docker 权限和实际容器副作用。

三个 owner 保持分离：Core 决定 desired generation，Controller 决定实际容器效果，插件领域服务决定
Computer 行为。改变 Computer、Docker backend 或 Chat 工具内容时，不迫使另外两个概念变化。

## 影响

- v3 public API 增加 Workload 声明、可选 `user_namespaces` 字段和 Workload-to-MCP 端点绑定。
- runtime snapshot 和 generation host 增加 Workload registry/facade，但不增加第二套 plugin generation。
- Compose 固定启动 Controller；具体 Computer 容器由插件声明动态创建。
- 未配置 Controller 的旧式本地部署不启用内置 Workload 插件并记录 warning；不按插件名字分支。
- 内置与外部插件使用同一 Workload API。
- 普通插件卸载继续保留 plugin-data。
- huayue-skills 的 OpenCLI Skill 在 Computer artifact 可用后移交给 Computer 插件。

## 验收

- [x] 外置测试插件无需主仓库名称分支即可启动、更新、停止一个 Workload。
- [x] Core 容器没有 Docker socket；Controller 不能访问 SessionDB 和任意插件数据。
- [x] candidate 与 formal 的数据、端口、容器 identity 分离。
- [x] stop/cleanup/restore 失败保持可见 owner，不假报完成。
- [x] 禁用或卸载 Computer 会停止容器并撤下能力，但不删除登录态。
- [x] 正式 Core 正常停止或异常退出后，Controller 会停止并移除 Workload，但保留登录态目录。
- [x] Chat 多标签工具区只依赖 `conversation.tools.v1`，没有 Computer 特判。
- [x] Computer renderer 使用独立 user namespace；容器仍为非 root、零 capability、
      `no-new-privileges` 和受限 seccomp，不使用 `--no-sandbox` 或 `seccomp=unconfined`。
