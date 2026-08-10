# Akashic 容器与 Linux 主机运行适配设计

- 状态：accepted target / experiment pending
- 日期：2026-08-10
- 决策：[0032 Host Bridge 保留宿主等价执行能力](../decisions/0032-host-bridge-preserves-host-equivalent-execution.md)
- 实验合同：[容器适配非迁移实验合同](akashic-container-host-bridge-experiment-contract.md)
- 范围：让 Akashic Core 在 hua-home 或具有完整 Linux 主机权限的云 VM 中以固定容器运行，同时保持当前开发机的原生运行方式和 Agent 的宿主操作能力
- 非目标：本设计不批准迁移正式 Workspace、切换移动端身份、发布公网入口或让 Agent 自行部署新 Runtime

## 1. 用户意图

容器只负责固定 Core 的代码、依赖和生命周期，不得把 Agent 变成只能操作容器内部文件的受限版本。
正式环境中的 Agent 默认看到真实 Linux 宿主：用户通过 SSH 登录正式主机后能完成的普通用户操作，
Agent 也应通过同名 Shell、File、Process、Git、SSH 和 OpenCLI 能力完成。

Agent 不修改或替换正在运行的 Core。它必须能查看准确的 Runtime 身份和同版本源码，在该版本上
复现问题；需要修复时，从当前运行 commit 创建宿主 Git worktree、提交分支并提出 PR。用户负责
最终合并和手动部署。

## 2. 已核对事实

### 2.1 真实工作负载

对正式 `sessions.db` 最新 150 个 turn 的只读统计覆盖 2026-08-07 至 2026-08-09：

- 150 个 turn 中 124 completed、25 interrupted、1 failed；99 个 turn 使用工具。
- 共 1035 次工具调用：Shell 673、Read 170、`write_stdin` 69、Edit 45，其余包括插件、MCP、
  web、memory、vision、restart 和 spawn。
- 42 个 programmatic turn 自身使用 Shell 308 次、Read 101 次、`write_stdin` 11 次。
- OpenCLI 真实任务反复形成 `shell → write_stdin`，证明执行句柄、流式输出和交互输入是硬合同。
- 样本覆盖插件源码修改、candidate validation、GitHub review、MCP、图片读取和 Runtime restart；
  未覆盖 Drift、PTY、`task_stop`、表情包外发和冷启动，不能把缺样本解释成无需支持。

因此 Bridge 只接主 Agent 会真实降级；主 Agent、programmatic turn、subagent 和 Drift 必须共享
同一执行后端合同。

### 2.2 当前执行与生命周期 owner

- `ShellTool`、文件工具、subagent profiles 和 Drift 当前直接使用本地 `Path` 与
  `ShellProcessManager`。
- Programmatic calling 的 thread、turn、admission、snapshot、attached cancellation、event replay
  和 terminal 由 Core Control/App Server 拥有；`main.py exec` 只是 JSON-RPC 客户端。
- PluginManager、candidate generation、MCP stdio、managed services、SessionDB、Supervisor 和
  RestartCoordinator 都是 Core 内部 owner，不能机械代理到宿主。
- 当前 Supervisor 已拥有唯一 caller、准入冻结、turn terminal、真实 delivery、boot identity、
  私有 nonce、优雅清理、exit 75 和下一代 readiness 合同。

### 2.3 SSH 与 systemd 环境

真实 SSH 登录和 systemd 服务环境不同。当前机器的 Node/OpenCLI 来自交互配置，systemd PATH
缺少这些目录；hua-home 的非交互 SSH 甚至无法从默认 PATH 解析 `hostname`。同一 Unix 用户不等于
同一工具链环境。长期方案必须由 mise 固定工具版本并生成 SSH 与 Bridge 共用的 capability
environment，而不是复制一次 PATH 或对所有命令 `source ~/.zshrc`。

## 3. 目标结构

```text
GitHub release
├── Core image@sha256
├── 同 commit 的 akashic-cli.pyz
└── 协议 major 兼容的 akashic-host-bridge.pyz
             │
             ▼
┌──────────────────── Linux 宿主 ────────────────────┐
│ systemd: Akashic Python Host Bridge                │
│ User=huashen · mise toolchain · Git/SSH/OpenCLI    │
│ Shell/PTY/File/Process · boot lease · gRPC UDS     │
│                                                    │
│ 独立 akashic-home-services 私有仓库                │
│ RSS/browser workers · 持久 Chromium/OpenCLI        │
└────────────────────────┬───────────────────────────┘
                         │ Unix Socket
┌──────────────── Core 容器▼─────────────────────────┐
│ Supervisor → Boot Guardian → Gateway               │
│ Session/Memory/Plugin/MCP/Control                   │
│ 正式 profile 只注册 BridgeBackend                  │
│ Bridge 失效即 fail-loud，不回退 LocalBackend       │
└────────────────────────────────────────────────────┘
```

开发机使用同一代码库的 `LocalBackend`；正式镜像的不可变 profile 只注册 `BridgeBackend`。正式
镜像可以包含 LocalBackend 源码以保持同源测试，但运行时不能通过配置启用它。

外围容器的 canonical source、image pins、Compose、systemd unit 和 release manifest 属于独立私有仓库
[`kachofugetsu09/akashic-home-services`](https://github.com/kachofugetsu09/akashic-home-services)。本仓库只保留
外部 `akashic-services` 网络和 `akashic-home-services.service` 的消费合同；Core release 不构建、校验或
重启外围容器。

## 4. Host Bridge 合同

### 4.1 传输和身份

- V1 使用版本化 JSON envelope，经 Protobuf `BytesValue` 编码并由 `grpc.aio` over Unix Socket
  传输；只参考 Memoh 的能力边界和失败语义，不复制其 AGPL proto 或 Go 实现。字段级 typed
  Protobuf message 属于后续协议 major 的独立演进，不得把 V1 描述成已实现 typed schema。
- Bridge 由 system-level systemd unit 以 `User=huashen` 运行，使用真实 UID、GID、补充组、HOME、
  SSH配置和宿主文件权限。
- 连接同时受 Socket权限与每 boot 一次性 capability token/lease约束。请求带 boot、session、turn
  和 execution owner；旧 lease 断开后拒绝新任务并清理该 boot 的全部进程组。
- 协议使用 `protocolMajor + capability set`。major 不兼容直接失败；新版 Bridge 必须先兼容当前
  旧 Core，再允许部署新 Core。

### 4.2 执行能力

首版必须保持现有工具 schema 和可观察语义：

```text
Exec
├── argv / shell script / cwd / env profile
├── 非PTY stdout、stderr和真实exit code
├── PTY、stdin、resize和signal
├── execution_id、增量读取、timeout和task_stop
└── owner/boot cleanup与进程组空集证明

File
├── stat / list / mkdir / rename / delete
├── 分页text read / atomic write / exact edit
└── raw byte stream，供图片、附件和表情包使用
```

Bridge 不增加全局 Shell 互斥。不同 session/turn 可并发执行；每个 execution 保留独立 owner 和
输出游标。Core 继续拥有工具参数、风险、restricted directory 和结果预算校验，Bridge 在 RPC
边界验证协议、路径类型、owner 和 boot lease。

### 4.3 Shell环境

- 普通 argv 使用 mise 固定的确定性 capability environment。
- 非交互脚本使用 `zsh -lc` 加载同一轻量环境，不依赖完整 `.zshrc`。
- 交互命令使用真实 PTY 与 `zsh -lic`，支持输入和窗口大小变化。
- mise拥有Python、Node、uv和OpenCLI等工具版本；应用凭据放在应用配置、Secret Service或权限受控
  的专用环境文件中，不把完整env写入Bridge审计。

### 4.4 审计

Bridge保存 boot/turn/execution、cwd、命令摘要、resolved executable、PID、时间、退出码、字节数
和清理结果。完整输出继续由 SessionDB/tool trace 拥有；Bridge不永久复制完整PTY内容、环境变量、
token或私钥。

## 5. Core与Bridge边界

### 5.1 经过Bridge

- 主 Agent、programmatic turn、三类subagent和Drift的Shell/File/Process。
- 由Skill经Shell调用的Git、gh、SSH、OpenCLI、yt-dlp、ffmpeg、drawio、测试和构建。
- 工作区之外文件的raw bytes读取，以及宿主生成文件进入附件/图片链的物化。

### 5.2 留在Core

- Programmatic Control/App Server、thread/turn、snapshot、replay、terminal和取消。
- PluginManager、install/revert/uninstall、candidate、stable/latest和turn-boundary提交。
- MCP stdio、managed services、plugin-data、manifest、cache和skill projection。
- SessionDB、memory、migration、Supervisor、RestartCoordinator和Gateway readiness。
- web、MCP、plugin tools、memory tools、vision模型和spawn编排；其中只有实际宿主文件/进程访问
  进入Bridge。

Host CLI是Core Control的版本化客户端，统一提供 `exec`、插件管理、`runtime info` 和人工
`core restart`；CLI与Core必须来自同一release并握手验证commit和协议。Agent通过Bridge启动CLI
时，Bridge以不可由命令文本伪造的元数据传递parent turn和plugin rollout lineage。

## 6. Runtime身份与源码

构建 owner 从用户批准的完整40位commit创建detached clean worktree，使用该Git tree生成构建
上下文。禁止从会漂移或dirty的main checkout构建。

每个release至少固定：

- source commit与tree；
- source archive摘要；
- base image digest；
- 完整依赖lock摘要；
- Python/Node版本；
- image digest、构建时间、migration集合摘要和SBOM。

路径分层：

```text
/opt/akashic/source                         镜像内运行源码，只读
/srv/akashic/runtime-sources/<commit>       宿主同版本只读参考
/srv/akashic/repos/akasic-agent             canonical Git repo
/srv/akashic/worktrees/<task>               Agent修复目录
```

`runtime-info`报告并交叉校验以上身份和路径。Agent从运行commit创建worktree，不从最新main猜测
诊断基线，也不把源码修改解释成已更新Runtime。

## 7. Supervisor与重启

Supervisor保留在Core容器内并继续拥有restart事务：caller必须是唯一active turn，冻结新准入，
等待terminal持久化和原回复delivery，验证当前boot身份，完成AppRuntime shutdown和boot清理。

```text
正常agent_restart
  → 同容器内Gateway process restart

process cleanup/protocol/new-ready失败
  → Supervisor失败退出
  → systemd作为唯一外层owner重启同一image digest的Core容器
  → Bridge清空旧boot宿主jobs
  → 新boot通过in-band readiness
```

Agent不选择restart scope。系统始终process-first，仅根据清理、协议、readiness、Supervisor退出、
workspace lock和boot job空集等真实证据升级容器重启。systemd负责有限backoff和StartLimit；Compose
不得再启用另一套无限restart loop。

## 8. 插件与外围能力

- 普通插件、Calendar/Fitbit/Feed/Steam MCP、managed services和GitHub Watch留在Core。
- canonical插件源码由Agent经Bridge修改；标准源码根以相同绝对路径只读挂入Core，PluginManager
  只读取并生成immutable artifact。
- `hypruse`和`computer-use-linux`不进入hua-home正式manifest或cache；旧plugin-data不迁入正式
  Workspace，旧工作站暂时保留到迁移验收结束。
- OpenCLI在语义上保持当前电脑方式：宿主命令连接长期有头Chromium、扩展、daemon和固定profile。
  服务器使用虚拟显示；Core/Bridge重启不关闭浏览器。OpenCLI失效只让该能力fail-loud，不阻止
  Core ready。

## 9. 数据与路径

- 权威状态物理存储在宿主显式目录，不使用Docker命名卷；Core以与宿主一致的逻辑绝对路径挂载。
- `/srv/akashic/workspace`可以bind到 `/home/huashen/.akashic/workspace`，保持现有配置、媒体和插件
  路径语义。
- 正式迁移只恢复完整Workspace及当前 `workspace/plugin-data`。旧 `~/.akashic-plugin/data` 不merge，
  不迁入正式运行目录。
- cache、MCP venv和skill symlink从canonical source与manifest重建。重建skill链接前必须检查同名
  普通文件/目录冲突。
- OAuth token先原样迁移并真实只读验证；供应商拒绝时再授权。移动端使用新server identity并重新
  配对一次。hua-home Chromium使用新持久profile人工登录一次。

## 10. 构建、部署与迁移

### 10.1 日志与可观测性边界

Core和Host Bridge通过`python-json-logger`输出一行一个JSON事件；库负责标准`LogRecord`、异常和JSON序列化，
Akashic只拥有字段白名单、关联上下文与脱敏策略。事件至少包含UTC时间、level、service、logger、event、pid；
存在时附加boot、release、session、turn、request、execution、phase、duration、outcome和错误指纹。
日志只保存诊断证据，不成为SessionDB或外部效果的权威事实。命令正文、消息正文、token、cookie、
authorization、完整env和工具输出不得进入日志；原命令和消息只能记录字节数与不可逆短指纹。

```text
Core container local log ─┐
peripheral local logs ────┼─> Alloy ─> Loki ─> Grafana(loopback)
Host Bridge journal ──────┘      │
                                 └─ read-only Docker API proxy
```

Grafana、Loki、Alloy和Docker API proxy由独立私有外围仓库拥有，不能并入Core image或Core发布合同。
Core不持有Docker socket，也不依赖采集链存活：Loki或Alloy中断时，本地`local`日志driver继续限额写入，
外围恢复后从持久position补采。`environment`、`service`、`source`和`level`可作为低基数label；turn、
request和execution只能作为structured metadata，避免索引基数失控。外围栈默认保留30天，实际保留期、
资源上限、镜像digest和loopback端口由外围release固定。

一次release同时产生Core image、同commit CLI和协议兼容Bridge。GitHub Actions只构建、生成SBOM并
发布不可变artifact，不自动部署hua-home。

唯一正式入口：

```text
mise run deploy <release>
  → 验证commit/tree/artifacts
  → 必要时先升级向后兼容Bridge并验证旧Core
  → 取得Workspace恢复点
  → 启动候选Core并核对identity/readiness
  → 切换正式入口
```

迁移采用单写者：旧工作站停止新写入并取得一致性备份，hua-home恢复并完整验收后才切换域名与
手机入口。旧工作站保留为恢复源。现有全局备份继续作为备份owner，正式迁移前必须验证覆盖范围、
最近成功时间和抽样恢复；当前实验不复制正式Workspace。

## 11. 分阶段实施

1. mise与完整依赖lock。
2. Runtime identity、生产Dockerfile和可复现release。
3. `ExecutionBackend` seam与LocalBackend回归。
4. Python Bridge、versioned Host CLI和宿主doctor。
5. Core容器、Supervisor/boot lease适配和systemd生命周期。
6. OpenCLI/Chromium外围服务。
7. 本机与hua-home非迁移实验及真实工作流验收。
8. 硬件就绪后的数据、身份迁移和正式入口切换。

实现拆成可独立评审的PR，并在同一集成分支完成端到端验收；开发机Local与容器Bridge两条路径
必须同时通过。

## 12. 验收

正式迁移前至少证明：

- LocalBackend保持开发机现有行为；正式镜像没有Local fallback。
- 主Agent、programmatic、subagent和Drift真实使用Bridge完成Shell/File。
- exec、PTY、stdin、resize、增量输出、timeout、task_stop和boot cleanup符合统一执行合同。
- Git/worktree/PR和本地插件install→programmatic child→turn后提交闭环成立。
- MCP、managed services、SessionDB和插件业务状态仍由Core拥有。
- runtime identity从build到readiness一致，当前运行commit源码可查看但不可原位修改。
- OpenCLI使用持久服务器profile完成真实只读站点操作，Core重启后仍可用。
- 图片、附件和表情包bytes跨Bridge后能被Core与客户端读取。
- 正常restart只重启Gateway；证据失败自动升级同digest容器重启，旧boot宿主jobs为空。
- Bridge失效时Core fail-loud退出，systemd恢复Bridge后才能重新启动Core。
- 非迁移实验没有写入、复制或删除正式Workspace及其身份。

## 13. 明确禁止

- Core挂Docker Socket、`privileged`、宿主根目录rw bind或通用免密sudo。
- Bridge失效时静默使用容器LocalBackend。
- 从dirty/main checkout构建，使用`:latest`部署，或让Core自动pull/update。
- 修改镜像内参考源码并称为Runtime已更新。
- 把programmatic control、PluginManager、MCP host或Supervisor整体搬进Bridge。
- 将旧插件数据merge覆盖当前 `workspace/plugin-data`。
- 在非迁移实验中复制正式sessions、memory、plugin-data、移动身份或浏览器profile。
