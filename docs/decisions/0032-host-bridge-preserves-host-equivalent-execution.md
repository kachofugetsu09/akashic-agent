# 0032 · Host Bridge 保留宿主等价执行能力

- 状态：accepted
- 日期：2026-08-10
- 关联条款：RUN-013、RUN-014、WSP-005、SH-001～SH-003
- 设计：[Akashic 容器与 Linux 主机运行适配设计](../design/akashic-container-cloud-runtime-adaptation.md)
- 实验：[容器与 Host Bridge 非迁移实验合同](../design/akashic-container-host-bridge-experiment-contract.md)

## 背景

Akashic 的 Agent-facing Shell 与 File 不只是开发辅助：真实 Turn 会通过它们操作宿主源码、插件源码、Git、OpenCLI、长进程和媒体文件。只把现有 Core 放进普通容器会让这些工具看到容器文件系统和工具链，行为不再等同当前开发机。给 Core `privileged`、Docker socket 或宿主根目录 bind 虽然恢复能力，却同时把容器变成不透明的宿主控制面，并没有建立可诊断的运行边界。

Memoh 证明了 Agent control plane 与 workspace target 可以分离；Akashic 可以借用其 target、文件和双向 PTY 语义，但不能直接搬用 Memoh 的 Go runtime，因为 Akashic 的插件 generation、MCP、managed service、programmatic control 和 Supervisor 已有自己的 owner。

## 决定

1. 原生开发运行继续使用 Local backend，保持当前开发方式。
2. 正式容器运行使用宿主上的 Python Host Bridge。Bridge 以 `huashen` 用户身份执行 Agent-facing Shell、File 和 Process 请求，使其能力语义与该用户 SSH 登录宿主后一致。
3. Core 与 Bridge 使用同版本的 gRPC Unix socket 协议；V1 是带 `protocolMajor` 的 JSON envelope，
   由 Protobuf `BytesValue` 承载。正式容器不得在 Bridge 失败时回退到容器 Local backend。
4. 主 Turn、programmatic Turn、subagent 和 Drift 共享同一种 backend 注入；programmatic admission、SessionDB、插件 generation、MCP/managed service、Supervisor 与 restart 事务继续留在 Core。
5. 每个 bridged execution 绑定 Core boot lease，保留统一执行句柄、流式输出、PTY、取消、超时和进程组回收语义。旧 boot 的宿主 job 未清空前不得启动新代。
6. 正式镜像携带不可变 runtime identity 和只读当前源码。Agent 从精确运行 commit 创建独立 worktree 调试、提交和发起 PR；运行中源码不被原地修改。
7. 正常 `agent_restart` 优先保持当前 Supervisor 的进程级事务。只有进程恢复证据失败时，外层 systemd lifecycle owner 才重启同一 image digest；部署新版本是另一条需要维护者批准的流程。

## 理由

- Agent 获得的是清晰的“我 SSH 登录这台 Linux 主机能做什么”语义，不必理解容器内外路径差异。
- Core 的依赖、版本和生命周期可复现，同时不牺牲宿主 Git、SSH、浏览器和文件工具能力。
- Python Bridge 可复用 Akashic 现有 ShellProcessManager 语义，改动比引入另一套 Go Agent runtime 更集中。
- Bridge 只接管模型可调用的通用主机工作台，不破坏现有 plugin/MCP generation owner。

## 未选择的方案

- **容器内执行全部工具**：无法获得宿主文件、systemd、pacman、SSH 身份和真实浏览器环境。
- **`privileged`、Docker socket 或宿主根目录 bind**：能力强但边界不可管理，也不能解决路径、身份和 boot job ownership。
- **SSH 作为本机默认 transport**：远端 target 可继续使用 SSH，但本机多一层认证、转义和会话环境差异；Unix socket 更直接。
- **直接采用 Memoh runtime**：语言、插件生命周期和持久化 owner 不同，改动面反而更大。
- **自研独立 Go executor**：首版没有足够收益抵消第二语言实现与发布成本。

## 影响

- 必须先统一 mise 工具链、完整依赖锁和 host capability profile。
- Agent-facing Shell/File 的所有注册入口都要注入 backend，包括 programmatic child、subagent 与 Drift。
- 图片和附件必须支持原始字节跨 Bridge，不能只传宿主路径。
- 宿主 Bridge、Core image、Supervisor 适配和 systemd unit 形成同一 release，但仍可分阶段实现和验证。
- OpenCLI 使用独立持久 Chromium 边车；登录态自然过期时检测并通知人工重登，不承诺任意网站永不过期。

## 验收

先按非迁移实验合同完成本机协议/容器实验，再在 hua-home 使用隔离 workspace 拉起完整候选运行时。只有 Local 与 Bridge 后端的能力矩阵、boot cleanup、Supervisor 故障注入、插件/MCP owner、OpenCLI 持久 profile 和 runtime identity 全部有真实证据后，才能另行批准正式数据迁移。
