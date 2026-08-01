# Linux Supervisor 安全自重启提议

- 状态：accepted（本分支已实现）
- 日期：2026-08-01
- 核对基线：`origin/main@e49f2a737c44`
- 能力 owner：Akashic Core runtime
- 关联条款：RUN-001～RUN-004、STA-001～STA-003、WSP-001～WSP-004
- 实现对账：RUN-004 已按 Linux Supervisor、非 Linux unmanaged 语义勘误

## 1. 结论

Supervisor 应继续存在，但只承担一个明确目标：**让 Agent 在修改 Gateway/Core 代码后，能够在当前回复已持久化且已实际送达的前提下，安全地结束旧 boot、装载新代码并确认新 boot 可用。**

第一版改造只支持 Linux。Linux 默认入口保留完整 Supervisor 与 `agent_restart`；非 Linux 默认进入明确标记的 unmanaged gateway，不注册 `agent_restart`，也不提供设置应用、重启、跨进程清理和 readiness 等配套 Supervisor 能力。显式执行 `supervise` 在非 Linux 上直接报告不支持。

Linux 实现采用每个 boot 一个轻量 `Boot Guardian`，不引入通用进程编排框架：

```text
┌────────────┐  lease + lifecycle pipe  ┌───────────────┐
│ Supervisor │─────────────────────────▶│ Boot Guardian │
└─────┬──────┘                          └───────┬───────┘
      │ settings / restart                      │ spawn + reap
      │                                         ▼
      │                                  ┌─────────────┐
      └─────────────────────────────────▶│   Gateway   │
                                         └──────┬──────┘
                                                │ existing groups / boot identity
                           ┌────────────────────┼────────────────────┐
                           ▼                    ▼                    ▼
                         MCP              managed service          peer
```

Guardian 是 Gateway 的父进程并设置为 Linux child subreaper。它持有当前 boot 的清理责任；Supervisor 同样注册为 subreaper，只在 Guardian 异常退出时收割转交给自己的孤儿进程。现有 MCP、managed service 和 peer 的进程组管理继续保留。Supervisor 只管理 boot 代际、设置事务和重启授权，不轮询 readiness 文件，也不充当通用子进程 RPC 服务。

## 2. 用户意图与能力边界

最初引入 Supervisor 的原因不是“所有平台都需要一个守护进程”，而是 Python 进程不能可靠地用自身替换自身并同时证明旧进程树已经清空。没有外部 owner 时，Gateway 可以请求退出，却无法同时完成下面的闭环：

1. 等待发起重启的正式 turn 成功持久化。
2. 等待该 turn 的最终回复被 transport 实际送达。
3. 拒绝普通退出、崩溃或伪造退出码触发重启。
4. 尽力结束旧 boot 的 Gateway 及其残留后代；无法清理的目标进入结构化诊断。
5. 从磁盘重新加载修改后的代码，启动下一代并确认 ready。

因此，去掉 Supervisor 会失去可靠的自重启能力；但把它扩展成跨平台通用进程管理器也不是目标。本提议只保留实现上述闭环所需的最小机制。

### 2.1 本次目标

- Linux 上保留 Agent 自重启和设置应用后的受控重启。
- Supervisor、Guardian 或 Gateway 单点异常时，旧 boot 的后代被尽力清理，残留可从日志定位。
- 启动等待改为事件驱动，并能指出停在哪个阶段。
- 未知端口占用者、未知 PID 和不属于当前 boot 的进程始终 fail-loud，不得误杀。

### 2.2 非目标

- 不支持 Windows、macOS 或其他非 Linux 平台的完整 Supervisor。
- 不提供 crash auto-restart、常驻服务安装、高可用或多实例编排。
- 不让运行中的 Supervisor 自更新；修改 Supervisor 后仍需人工完整重启。
- 不新增每服务 guardian、通用 spawn RPC、插件并发启动或 channel 并发启动。
- 不以 cgroup、systemd、容器或 root 权限作为直接 `python main.py` 的正确性前提。

## 3. 改造前真实链路

当前默认入口和显式 `supervise` 都进入 `main.py -> run_supervisor()`；Supervisor 再以固定参数启动 `main.py gateway`。Gateway 完成 Core、插件服务、channels、HTTP 服务和主动流程等串行初始化后，才写 `.runtime-ready.json`。

```text
python main.py
    │
    ├─ 启动迁移检查
    ▼
run_supervisor()
    ├─ workspace supervisor lock / PID
    ├─ settings HTTP thread
    ├─ Popen(main.py gateway, new session)
    └─ 每 20ms：poll child + 读 commit pipe + 读 readiness JSON
                         │
                         ▼
                 main.py gateway
                    ├─ 再做一次启动迁移检查
                    ├─ Core / plugins / MCP
                    ├─ managed services
                    ├─ channels / web / mobile / proactive
                    └─ 写 runtime-ready.json
```

已经具备并应保留的机制：

- `RestartCoordinator` 只在正式 turn 完成且回复实际送达后写私有 commit。
- commit 绑定当前 `boot_id`、私有 FD、nonce 和 request ID。
- 只有合法 commit、ready 和退出码 75 同时成立，Supervisor 才启动下一代。
- MCP 和 managed service 已使用独立进程组并继承 `AKASHIC_BOOT_ID`；peer 默认继承 boot identity，但当前没有进入独立进程组。
- workspace instance lock、Supervisor lock、设置候选配置与回滚路径已经存在。

## 4. 改造前问题

### 4.1 启动路径存在可达死锁风险

`run_supervisor()` 先启动 settings server 线程，随后 `Popen` 使用 `preexec_fn` 在 child 的 fork 与 exec 之间恢复 signal mask。Python 官方文档明确警告：存在多线程时，child 可能在 exec 前因锁状态而死锁。这个调用顺序是当前代码中的真实可达路径，也能解释“偶发启动超时但缺少 Gateway 日志”的现象。

改造必须移除 `preexec_fn`。信号屏蔽和子进程启动只使用 `Popen`/OS 已定义的参数与父进程时序，不在 fork 后执行 Python 回调。

### 4.2 常驻 20ms 轮询浪费资源并混合多种状态

`_wait_child()` 在 Gateway 整个生命周期内每 20ms 执行一次 child poll、pipe 读取、readiness JSON 读取和 settings event 检查。设置服务启动也使用同样的 20ms 轮询。这带来无业务价值的持续唤醒，并把进程退出、启动完成、重启提交和设置请求混进同一个循环。

改造后使用 pidfd、pipe、signal 和线程 wake FD 等可等待事件；`.runtime-ready.json` 只保留为诊断投影，不再是私有控制协议。

### 4.3 固定 15 秒把完整启动误当成单一步骤

当前 `AKASHIC_READINESS_TIMEOUT_S` 默认 15 秒，从 child 创建开始覆盖完整 Gateway 初始化。插件加载、MCP、managed services、channels、Web/Mobile 和主动流程均可能在 ready 前串行执行，各自还可能有局部超时。2026-08-01 调查期间的一次已部署容器现场观测，从容器开始到 readiness 约 8.25 秒，从 settings 开始监听到 readiness 约 6.2 秒；该记录没有作为当前分支的可复现 benchmark，只能说明固定 15 秒可能余量有限。

改造不能简单把 15 改成更大的拍脑袋数字。Gateway 应发送阶段事件，Supervisor 使用一个不可续期的总体硬 deadline，并在失败时报告最后阶段与耗时。默认值由改造前后的同机冷启动/热启动 profile 确定，而不是由阶段事件不断续命。

### 4.4 生命周期清理依赖 Supervisor 存活

当前清理会按 `AKASHIC_BOOT_ID` 扫描 `/proc/*/environ`，对找到的进程组先 TERM、再 KILL，并验证目标消失。这能处理 Supervisor 存活时的 Gateway 异常，但 Supervisor 被 SIGKILL 后没有仍存活的清理 owner；MCP 或 managed service 可能继续占用端口。

扫描 `/proc` 也只能作为同一 boot 的发现与兜底证据，不能承担稳定 PID 身份。PID 复用与权限变化要求 owner 优先持有从创建时取得的 pidfd/父子关系，且绝不能仅按端口杀进程。

### 4.5 迁移与设置边界不够收敛

当前 Supervisor 启动前和其创建的 Gateway 都会经过 `_prepare_startup_migrations()`，导致同一正式启动做两次迁移检查。迁移已有 Git cursor 幂等合同，但正式启动只应由一个 owner 检查一次。

settings server 默认监听 loopback，但 `AKASHIC_SETTINGS_HOST` 可以扩大到非 loopback。当前 Origin 与 CSRF 检查不能把本地设置接口变成安全的远程管理面。本提议固定只允许 loopback，非 loopback 配置在边界直接拒绝。

## 5. 成熟实践与取舍

| 实践 | 借鉴内容 | 本提议不照搬的部分 |
|---|---|---|
| systemd control-group kill | 一个服务单元停止时清理整个控制组，不只杀主 PID | 直接 Python 启动不能假设可写 cgroup 或 systemd 单元 |
| Linux pidfd | 从进程创建时持有稳定身份，避免只凭可复用 PID 操作 | pidfd 不提供整个后代树所有权，仍需父子关系、subreaper 与进程组 |
| Kubernetes startup/readiness probes | 启动完成与存活分开；启动慢不应被普通存活判断误杀 | 本地单进程无需完整 probe 框架或周期性健康控制器 |
| Supervisor/s6 状态与 readiness notification | 明确启动、运行、退出状态；用通知而非猜测“进程存在即 ready” | 不引入配置 DSL、服务依赖图或通用 supervision suite |

当前环境可使用 cgroup v2，但当前直接启动上下文没有可写的独立 cgroup 控制权。因此 cgroup 可作为未来部署层增强，不能成为此方案正确性的必要条件。

## 6. 最小目标架构

### 6.1 Supervisor

Supervisor 只拥有：

- workspace 级唯一实例锁。
- boot 代际与 boot token 创建。
- settings 本地事务入口。
- 读取 `stage`、`ready`、`commit` 生命周期事件。
- 判断是否允许启动下一代。
- Guardian 异常时使用同一 boot token 做一次兜底清理并收割被内核转交的孤儿进程；残留写日志，不撤销已有合法重启提交。

它不拥有插件、MCP 或 channel 的创建细节，不提供任意进程管理 API，也不在 Guardian 缺失时继续运行 Gateway。

### 6.2 Boot Guardian

每个 boot 创建一个 Guardian。它只拥有：

- 成为 Gateway 的直接父进程，并在 Linux 上设置 child subreaper。
- 将 `AKASHIC_BOOT_ID` 传给 Gateway；现有子进程继续继承该身份。
- 持有 Supervisor lease；lease EOF 表示 Supervisor 已消失。
- 等待 Gateway、收割被托管树中成为其后代的孤儿。
- 在 Gateway 退出或 lease 断开后，对当前 boot 执行一次有总预算的 drain/TERM/KILL/验证为空。
- 无论 cleanup 是否完全成功，都转交 Gateway 退出码；失败另外输出结构化诊断。

Guardian 不参与业务 readiness、设置、回复送达或重启授权。它负责尽力清空 boot、持续回收 adopted zombie，并明确报告仍存活的目标。

### 6.3 Gateway 与现有子进程 owner

Gateway 继续拥有业务启动、正常 shutdown 与 readiness 判定；MCP、managed service 和 peer 继续由现有业务 owner 管理。MCP 与 managed service 复用现有进程组实现；peer 使用同一个现有 helper 补齐独立进程组，不增加新的进程抽象。所有启动路径保留 boot identity，并把真实启动阶段报告给生命周期 pipe。

不新增每服务 Guardian。现有 owner 先做正常 drain；Boot Guardian 只在整个 boot 结束时提供最后的边界收束。

## 7. 最小协议与状态机

Supervisor 与当前 boot 只需要一个单向、小帧、boot-scoped lifecycle pipe：

- `stage`：`bootId`、阶段名、单调时钟耗时；只用于进度与超时诊断。
- `ready`：`bootId`、Gateway PID；表示所有对外入口已进入可服务状态。
- `commit`：`bootId`、nonce、request ID；表示正式重启提交已经完成。

不增加 heartbeat、双向 RPC 或可无限重置的超时。生命周期 pipe 的写端只传给 Gateway，Guardian 不转发业务帧；Supervisor 通过 Guardian 退出状态取得 Gateway 结果和 cleanup 成败。帧必须小于 `PIPE_BUF` 并一次写入；`stage` 可以按单调顺序出现多次，未知帧、重复 `ready/commit` 或跨 boot 帧使当前 boot 失败，不做静默兼容。

正常自重启顺序：

```text
Agent 修改代码
    │
    ▼
agent_restart 冻结新 turn 准入
    │
    ├─ 当前 turn 持久化成功
    ├─ 最终回复实际送达
    ▼
commit 写入私有 lifecycle pipe
    │
    ▼
Gateway 正常 shutdown 并以 75 退出
    │
    ▼
Guardian drain → TERM → KILL → verify empty
    │
    ▼
Supervisor 验证 ready + commit + 75，启动下一 boot
```

任一条件缺失都不得拉起下一代：

| 观察结果 | Supervisor 行为 |
|---|---|
| ready + 合法 commit + exit 75 | 记录 cleanup 结果并启动下一 boot |
| bare exit 75 或伪造/跨 boot commit | 失败退出，不重启 |
| 普通退出、异常、OOM 等价退出 | 清理后返回原始失败，不自动重启 |
| 回复未送达或 turn 未完成 | 不 commit；恢复准入或失败退出 |
| Guardian 先退出 | Supervisor 兜底清理当前 boot，随后失败退出 |
| Supervisor lease EOF | Guardian 清理当前 boot后退出 |
| Gateway 退出 | Guardian 清理当前 boot；由 Supervisor 按授权条件决定是否下一代 |

本提议保证 Supervisor、Guardian、Gateway 三者中一个 lifecycle owner 单点故障时的收束。不承诺 Supervisor 与 Guardian 同时 SIGKILL、内核崩溃或断电后的在线清理；下一次启动必须检测 stale lock/端口占用并 fail-loud，未知 owner 不得被杀。

## 8. 启动、停止与性能规则

### 8.1 启动

- 正式启动迁移只执行一次，完成后才创建当前 boot。
- 启动只有一个总体硬 deadline；最后 `stage` 只解释失败位置，不延长 deadline。
- MCP 和 managed service 保留各自局部启动 timeout；局部失败保留最后阶段，Gateway 失败退出并触发 boot cleanup。
- `.runtime-ready.json` 在收到/发出私有 `ready` 后原子写入，只供人和诊断工具观察。
- 第一版不并行化插件、服务和 channels。先移除死锁与轮询，再用阶段 profile 决定是否值得优化串行路径。

### 8.2 停止

一次 boot cleanup 只使用一个总宽限预算，而不是每发现一批后代就重新获得完整超时：

```text
业务 drain（预算内） → SIGTERM（剩余预算） → SIGKILL → verify empty
```

TERM-ignore、double-fork、`setsid` 和 wrapper 进程必须包含在故障注入中。Guardian 的 subreaper/父子收割是主路径，boot identity 扫描是兜底；端口只能作为“仍未释放”的失败证据，不能作为 kill 授权。

### 8.3 性能基线

实现前先记录冷启动、热启动每阶段单调耗时和失败阶段；实现后比较：

- ready 前总耗时及各阶段耗时。
- Supervisor/Guardian 空闲 CPU 唤醒。
- 20 次合法重启后的 RSS、FD、线程、zombie 与端口占用。
- migration 实际调用次数。

验收目标是消除常驻 20ms polling 和同次启动的重复迁移检查。启动 deadline 的具体默认值由 profile 给出，并保留显式配置入口。

## 9. 平台分流

平台判断发生在 `main.py` CLI 边界，进入 Supervisor 代码之前：

| 平台与入口 | 行为 |
|---|---|
| Linux，默认 `python main.py` | 完整 Supervisor → Guardian → Gateway；注册 `agent_restart` |
| Linux，显式 `supervise` | 与默认入口相同，保留兼容别名 |
| Linux，显式 `gateway` | unmanaged 调试；不注册 `agent_restart` |
| 非 Linux，默认 `python main.py` | 打印明确 warning，直接进入 unmanaged gateway；不暴露任何 Supervisor 配套能力 |
| 非 Linux，显式 `supervise` | 非零退出并报告仅支持 Linux |
| 非 Linux，显式 `gateway` | unmanaged gateway；不注册 `agent_restart` |

“不暴露配套能力”包括：无 `agent_restart`、无 Supervisor settings server、无 Supervisor readiness/commit、无 boot 进程树清理。Gateway 自身的 workspace instance lock、迁移、正常 signal shutdown 和普通业务能力仍保留。

这与现行 RUN-004 的“默认入口必须由 Supervisor 托管”存在文字冲突。若维护者接受本提议，后续实施合同必须先把 RUN-004 收窄为 Linux 正式入口，并为非 Linux unmanaged 行为补充可观察需求与 semantic test；在此之前本文保持 proposed，不能被当成现行行为。

## 10. 安全与权限边界

- settings server 固定 loopback。任何非 loopback host 在配置边界失败；本提议不设计远程设置模式。
- restart commit 继续使用继承的私有 FD、boot ID、nonce 和单次 request ID，不改成 workspace 文件或网络端口。
- `ready` 与 `stage` 使用同一私有继承通道；workspace JSON 不是授权来源。
- Supervisor 与 Guardian 只操作创建时获得的进程身份、其后代和匹配当前 boot token 的兜底目标。
- 发现未知端口 owner、身份不匹配、无法读取/验证进程归属或清理后仍存活时，失败并输出证据。
- 不捕获并吞掉 spawn、IPC、配置、清理或权限错误；失败必须通过非零退出码和明确日志暴露。

## 11. 持久状态与恢复

本提议不迁移或重写业务持久状态。实施时必须逐项保持以下合同：

| 对象 | 正常增加/更新 | 逻辑或物理减少 | Owner 与恢复证据 |
|---|---|---|---|
| sessions、turns、messages、附件 | 沿现有业务路径增加或按既有状态机更新 | 本提议无减少权限 | 各现有 repository；重启前后行数、正文与附件引用一致 |
| memory、Akasha、主动流程、调度、plugin-data | 沿现有 owner 写入 | 本提议无减少权限 | 既有 runtime；重启前后文件/DB 完整性与连续性状态一致 |
| `config.toml` | 仅现有 settings candidate/原子提交可更新 | 回滚只恢复该事务的备份，不删除其他配置 | settings owner；候选、备份、应用结果和回滚结果可核对 |
| migration cursor | 正式启动时按既有 Git cursor 合同推进一次 | 只允许既有显式 revert 协议 | migration owner；cursor、commit 列表与前后检查结果 |
| lock、PID、readiness、socket | 每个进程生命周期创建/替换 | 对应 boot 停止后由生命周期 owner 删除 | 当前 boot ID、持有锁的 FD、PID/pidfd 与端口核对 |
| lifecycle pipe、lease、pidfd、nonce | 仅内存/内核态创建 | boot 结束即关闭 | 不持久化；关闭和后代为空是恢复证据 |

旧 boot 未清空时必须记录残留目标，但不阻止具备合法 commit 的新 boot。Supervisor 或 Guardian 故障不能用删除 workspace 控制文件伪装成清理成功，更不能修改任何权威业务状态来恢复启动。

## 12. 分阶段实施（已完成）

### 阶段 A：可观测启动与安全 spawn

- 增加 `stage/ready/commit` 私有事件编码与严格解析。
- 移除 `preexec_fn` 和 readiness JSON polling。
- 把 migrations 收敛到正式启动唯一 owner。
- 记录真实阶段 profile，确定总体 startup deadline 默认值。

### 阶段 B：单个 Boot Guardian

- 新增 Linux-only Guardian，建立 lease、subreaper、pidfd/父子等待和总预算 cleanup。
- 复用现有 boot ID 与进程组，不改插件/MCP/service 的业务 owner。
- 覆盖 Supervisor、Guardian、Gateway 与后代进程的单点故障。

### 阶段 C：平台与设置边界

- 在 CLI 边界实现 Linux/非 Linux 分流。
- 非 Linux 移除完整 Supervisor 能力并输出明确状态。
- 固定 settings loopback，拒绝环境变量扩大监听边界。
- 若提议正式接受，同步勘误 RUN-004 和 semantic tests。

每个阶段都必须能独立回滚到阶段开始的源码提交。不得用跳过故障注入、放宽清理 oracle 或保留新旧两套控制协议来获得通过。

## 13. 验收矩阵

最低验收必须从真实进程和端口边界观察：

1. 合法 commit：回复持久化且实际送达后启动下一代；旧 boot 清理失败时存在结构化诊断，新 boot 仍 ready 且新代码已加载。
2. 非法重启：bare 75、伪造 nonce、旧 boot commit、未送达回复和失败 turn 均不拉起下一代。
3. owner 故障：分别 SIGKILL Supervisor、Guardian、Gateway，证明单点故障下 boot 被清空或明确 fail-loud。
4. 后代故障：SIGKILL/OOM 等价、TERM-ignore、double-fork、`setsid`、wrapper 和 MCP/service leader 异常不留下 zombie 或监听端口。
5. 身份安全：未知端口/PID 永不被杀；boot identity 不可验证时启动失败。
6. 启动诊断：每个超时报告最后阶段和单调耗时；过期 readiness JSON 不能使 boot ready。
7. 性能：Supervisor/Guardian 等待无 20ms polling；一次正式启动只检查一次 migration。
8. 设置：non-loopback settings host 被拒绝；候选失败恢复旧配置与旧可用 boot。
9. 平台：Linux 三种入口和非 Linux 三种入口符合第 9 节矩阵。
10. soak：连续 20 次合法重启后，FD、线程、RSS、zombie、端口和非终态 turn 无单调泄漏。

## 14. 已确认事实、推断与未知边界

### 已确认事实

- 当前 Supervisor 在启动 settings 线程后使用 `Popen(preexec_fn=..., start_new_session=True)`。
- `_wait_child()` 和 settings 启动等待都以 20ms 周期轮询。
- 当前 readiness 默认总超时为 15 秒，ready 在完整 Gateway 启动后发布。
- 默认入口和 Gateway 都会经过启动迁移选择逻辑。
- 现有 restart commit 已绑定 turn 完成、实际送达、boot ID、nonce 与退出码 75。
- 当前 Linux 清理按 boot ID 和进程组执行，但 owner 是 Supervisor。
- 2026-08-01 调查环境具备 pidfd 与 cgroup v2；直接启动上下文没有可依赖的可写 cgroup。

### 设计推断

- `preexec_fn` 的多线程风险是偶发启动超时的高概率根因，但没有单次超时现场栈可证明所有超时都由它造成。
- Boot Guardian + subreaper + 现有进程组是满足直接 Python 启动和单 owner 故障清理的最小组合。
- 串行启动在移除轮询、重复迁移与死锁风险后可能已经足够；没有 profile 前不应并行化。

### 后续仍需测量

- 冷/热启动各阶段分布，以及总体 deadline 的合理默认值。
- Guardian 故障后 Supervisor 兜底扫描在目标部署权限下的覆盖率。
- 各插件包装器是否存在无法被 subreaper/boot identity 覆盖的 daemonize 行为；发现时应修复具体 owner，不扩张为通用进程管理框架。

## 15. 参考资料

- [Python `subprocess`：多线程环境不要使用 `preexec_fn`](https://docs.python.org/3/library/subprocess.html)
- [Linux `pidfd_send_signal(2)`：以 pidfd 指向稳定进程身份](https://www.man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)
- [Linux cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
- [systemd.kill：按 control group 管理服务进程](https://man7.org/linux/man-pages/man5/systemd.kill.5.html)
- [Kubernetes Pod lifecycle：startup 与 readiness 分离](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Supervisor subprocess states](https://supervisord.org/subprocess.html)
- [s6 readiness notification](https://www.skarnet.org/software/s6/)
