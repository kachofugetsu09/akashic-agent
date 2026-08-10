# Akashic Core 与 HostBridge 一键安装设计

- 日期：2026-08-10
- 状态：accepted；安装器已实现，真实目标机部署证据仍按验收节记录
- 上游栈：#353 → #354 → #355 → #352

## 1. 目标与边界

用户通过一条命令安装或升级同一 Git commit 的 Akashic Core 与 Python HostBridge。未指定
`--commit` 时解析远端 `main` 的最新完整 SHA；指定时只接受远端可达的 40 位 SHA。安装器展示
current/target identity 并等待确认，只有 `--yes` 允许无人值守。

```bash
curl -fsSL https://raw.githubusercontent.com/kachofugetsu09/akashic-agent/main/scripts/install-akashic.sh | sh

curl -fsSL https://raw.githubusercontent.com/kachofugetsu09/akashic-agent/main/scripts/install-akashic.sh \
  | sh -s -- --commit <40-character-sha>
```

Core 与 Bridge 保持同仓库、同 commit、同 release manifest，但继续运行在不同权限域。Core 不获得
Docker socket、宿主根目录、`privileged` 或 systemd 更新权限；HostBridge 不拥有 turn、SessionDB、
plugin generation、MCP control plane 或部署事务。外围服务仍由私有 `akashic-home-services` 仓库拥有，
本安装器只验证其 systemd unit、网络和端点合同，不修改外围仓库或容器。

本设计不授权 Core 自更新。安装、升级、回滚和正式迁移只能由通过 SSH 登录宿主的 operator 发起。

### 1.1 任务合同

- `change_type`：HostBridge 修复、MCP 故障语义修复、持久化文档修复、安装能力新增。
- `semantic_delta`：不同 session 的 HostBridge Shell 从错误串行恢复为并发；非关键 MCP 恢复耗尽从
  `runtime_fatal` 改为 generation-local degraded；安装由人工拼装收口为 operator-owned 事务。
- capability owner：release manager 拥有软件 generation 准备与激活；systemd 拥有进程生命周期；
  HostBridge 拥有宿主执行；Core 拥有 Agent 业务状态。
- authoritative state owner：`<workspace>` 及 companion state 的既有 owner 不变；release manifest 和
  activation receipt 只描述软件代际，不反向定义业务数据。
- 受保护状态：正式 Workspace、plugin-data、模型凭据、浏览器 profile、外围服务数据和旧 generation。
- 禁止副作用：安装代码不得自动删除旧 generation、迁移正式数据、切换正式 ingress、修改外围服务、
  push Git 或替用户合并 PR。

## 2. 选定方案

第一版采用目标机按精确 commit 构建的 source-resolved local release。它复用现有 exact Git archive、
Core image identity、toolchain digest、Bridge doctor 和 systemd 合同，不依赖 GitHub Release，也不要求
每个 `main` commit 已有可下载的 CI artifact。

未采用的方案：

1. CI 同时发布 Core image 与 Bridge wheel：安装更快，但 `main` head 与 CI 成功产物可能短暂不一致。
2. 从 Core 镜像提取 Bridge：类似 Memoh，但会混淆容器 Python 与宿主 Mise/OpenCLI/Git 运行边界。

后续可以在不改变 CLI 和 manifest schema 的前提下，把 preparation backend 换成经过签名和 digest
验证的预构建产物；第一版不增加这条未验证路径。

## 3. 组件与目录

Shell bootstrap 只检查基础命令、取得精确源码并用系统 Python 进入仓库内 release manager；这段
bootstrap import graph 只能依赖 Python 标准库。准备完成后，Bridge RPC doctor 必须由该 generation
锁定的 `AKASHIC_BRIDGE_PYTHON` 执行，不能要求宿主系统 Python 预装 `grpcio`。稳定 operator CLI 同样
从 runtime.env 解析并进入当前 generation 的 Bridge Python。事务、状态机和验证逻辑不堆在 Shell 中。

```text
scripts/install-akashic.sh
  └─ scripts/akashic_release/
     ├─ cli.py       参数、确认与稳定退出码
     ├─ source.py    main SHA 解析、远端可达性、shallow checkout
     ├─ image.py     Core 构建和 exact image ID 验证
     ├─ bridge.py    Bridge venv、toolchain identity 与 doctor
     ├─ manifest.py  release manifest、receipt 与原子 JSON
     ├─ systemd.py   unit 安装、启停和状态观察
     ├─ activate.py  激活、真实健康检查和上一代恢复
     └─ migrate.py   正式 Workspace 迁移编排与停止边界
```

非平凡模块保持单一职责；命令执行和文件持久化复用仓库现有 runner、原子 JSON 与 release identity
实现，不另造通用 subprocess、锁文件或 JSON storage 框架。

```text
/srv/data/services/akashic/
├─ runtime-sources/<commit>/       一代只读 shallow checkout
├─ bridge-venvs/<commit>/          一代 Bridge Python 环境
├─ releases/<commit>.json          不可变 release manifest
├─ activation/                     active/previous/failed receipts
├─ run/                             release lock、UDS、readiness
├─ secrets/                         Bridge token 等 0600 secret
├─ state/                           正式 Workspace/config/plugin-data
└─ backups/                         runtime.env、unit 与迁移恢复点
```

systemd unit 是稳定入口，通常只在模板摘要变化时更新。固定 EnvironmentFile 由安装器通过同目录临时
文件、fsync 和原子替换切换 generation；secret 不进入 release manifest 或日志。

## 4. HostBridge 并发与生命周期

当前 `_manager_operation` 在整个 RPC 期间持有 `operation_lock`。一个 Core 只有一个 Shell manager，
因此不同 session 的 `exec` 在首次完成或 yield 前被错误串行。这违反 SH-003，不是产品取舍。

修复后，manager lease 使用 admission 与在途计数：

```text
普通 RPC
  → 在 manager state lock 下确认 active boot 且 accepting
  → active_operations += 1
  → 释放 state lock
  → 并发调用 ShellProcessManager
  → finally: active_operations -= 1 并通知 drain waiter

boot takeover / lease reaping / shutdown
  → 在 state lock 下设置 accepting=false、reaping=true
  → 等待 active_operations == 0
  → shutdown manager 并证明 execution table 为空
  → cleanup 成功后释放 manager ownership
```

不同 `ownerSessionKey` 的 Shell 同时执行，不互相等待。相同 execution 的 stdin、stop 和输出顺序仍由
`ShellProcessManager` 的 execution ownership 约束。只有 boot takeover、lease reaping 和 shutdown
关闭新 admission 并等待已进入的 manager operation；它们不会在新旧 boot 间并发复用 manager。

协议 V1 准确描述为 versioned JSON envelope inside protobuf `BytesValue` over gRPC UDS。#353 不再宣称
已经具有自定义 typed Protobuf schema。typed `.proto`、generated stub、metadata interceptor 和可重连
streaming Exec 属于后续独立 Protocol V2，不塞入本次安装 PR。

## 5. 安装与升级事务

### 5.1 准备

整个流程持有 `/srv/data/services/akashic/run/release.lock`，第二个 installer 明确失败。准备阶段不停止
当前 Akashic：

1. 检查 Docker、Compose v2、systemd、Git、Mise、架构、磁盘、目录权限和外围合同。
2. 未指定 commit 时用远端 Git ref 解析 `origin/main`，不从本地 remote-tracking ref 推断。
3. 指定 commit 时校验完整 SHA、远端可达性和 commit object。
4. 展示 current/target SHA、提交标题与 generation 状态；交互确认或验证 `--yes`。
5. 在 `.staging-<run-id>` 准备 shallow checkout、Core image、Bridge venv 和 manifest。
6. 核对 commit/tree、source inventory、dependency locks、image ID、toolchain digest 和 Bridge doctor。
7. 全部通过后把 staging 原子发布成不可变 generation。

hua-home 构建使用清华、科大 Arch package cache、清华 PyPI 和 npmmirror；Arch 数据库仍固定到
`AKASHIC_ARCH_SNAPSHOT`，Python `--require-hashes` 与 npm lock integrity 继续拥有制品身份校验。

正式 hua-home 的入口分成两个不同信任边界：

```text
┌──────────── LAN 192.168.0.0/24 ────────────┐
│ 浏览器 ──> 192.168.0.100:2236 ──> Core WebUI │
└─────────────────────────────────────────────┘

Internet ──> Cloudflare Tunnel ──> 127.0.0.1:6323 ──> Mobile WSS
```

WebUI 默认仍绑定 `127.0.0.1`；仅由 operator 在 `runtime.env` 显式设置
`AKASHIC_WEB_BIND_ADDRESS=192.168.0.100` 后进入局域网，并由宿主 `DOCKER-USER` 链只放行
`192.168.0.0/24` 到 2236。Mobile 6323 固定发布到 loopback，只允许 Cloudflare Tunnel 回源，不能因
WebUI 的局域网需求一并暴露。

同 commit 的完整 generation 摘要一致时复用；存在同名但摘要不一致、缺 manifest 或身份漂移时
fail-loud，不覆盖目录。staging 失败只清理本轮 manifest 明确拥有的对象。

### 5.2 激活

候选完整后进入短维护窗口：

```text
记录 previous generation
  → 备份 runtime.env 与发生变化的 unit
  → stop Core
  → stop Bridge
  → 原子切换 runtime.env
  → start Bridge
  → systemd/UDS/commit/toolchain/Shell/File probe
  → start Core
  → Docker health/readiness identity/HTTP/DB/plugin-MCP smoke
  → 写 completed activation receipt
```

unit 内容未变化时不重写、不执行无意义的 daemon-reload。模板变化时由普通用户执行 installer，仅在
安装 system unit、daemon-reload 和 unit 控制的窄步骤调用 sudo；服务进程继续使用声明的非 root 用户。
指定非默认 `--unit-root` 时进入离线验证路径：外围 unit 必须存在于该隔离目录并通过
`systemd-analyze verify`，Core/Bridge unit 只原子写入该目录，不调用 sudo 或 reload 正式 systemd。
首次激活从系统 Python 发起时，UDS/gRPC probe 仍以候选 generation 的 Bridge Python 子进程运行；
系统 Python 缺少 `grpcio` 不得成为部署前置条件。

### 5.3 软件恢复

候选激活失败时停止候选，原子恢复旧 runtime.env，依次启动并真实验证旧 Bridge 与 Core，再写 failed
receipt。恢复也失败时停在 maintenance，不循环切换、不启动身份不确定的 Core，并输出精确人工恢复
命令。`recovery_failed` receipt 同时保存 candidate error、previous recovery error、maintenance stop
error（若有）和人工命令；不能因第二个异常覆盖第一次失败。候选 manifest、日志和旧恢复点全部保留。

软件恢复不等于数据回滚。包含 Workspace schema/data migration 的运行必须先走第 8 节恢复点与切换
合同；安装器不得用进程恢复掩盖已发生的数据或外部效果。

### 5.4 CLI

```text
akashic-release install [--commit SHA] [--yes]
akashic-release doctor
akashic-release rollback [--yes]
akashic-release pair-mobile
akashic-release migrate --snapshot-manifest PATH
```

`pair-mobile` 只访问当前 release 的 loopback WebChat 管理入口，在 SSH 终端用锁定的 `qrcode`
依赖直接绘制一次性二维码，等待已验签手机 claim，并要求 operator 输入相同的六位确认码后才批准。
默认 pairing offer 有效期为 8 分钟；延长操作窗口不改变 secret 哈希存储、一次性消费、设备签名或人工确认。

当前首版的 `migrate` 只校验预演 snapshot manifest 并输出 `plan_only` 阶段清单，明确返回
`automaticDataWrites=false`。它不停止旧端、不复制正式 state、不切换 ingress；这些动作仍需维护者在
目标机逐阶段批准。这样可以先用同一入口核对迁移输入，又不会让软件安装授权隐式扩大成数据迁移授权。

重复安装当前健康 generation 返回 `already_active` 并只执行 verify。active receipt、runtime.env、实际
Bridge/Core identity 任一不一致时，install/rollback 拒绝继续并要求先运行 doctor。

## 6. MCP 与 Skill 修复

active MCP 恢复预算耗尽只把对应 server/generation 标记为 degraded；对应工具调用明确失败，其他
插件、聊天和 Runtime 继续运行。candidate MCP 失败继续阻止 promote。只有未来有明确 `critical` 合同且
其 owner 无局部恢复动作时，才能把 MCP failure 升级为 runtime fatal；本次不新增这种声明。

skill-link ownership、pending transition journal、legacy adoption 和 persistence map 必须在同一 PR
交付。旧 workspace 在新 linker 首次启动前执行显式 adoption：只登记解析后仍位于批准 plugin roots
内的 symlink；用户文件、普通目录、越界 target、损坏链接和既有 ownership 文件全部 fail-loud且不删除。

`workspace/runtime/plugin-skill-links.json` 是 P2 可重建投影的 ownership/journal companion。它允许原子
更新 links/pending；只有在实际 link transition 提交后才能清除 pending。不能在 workspace 仍有既有
插件链接时把账本当缓存直接删除；恢复必须备份账本，或在完整验证旧链接和 canonical plugin roots 后
执行名称明确的 adoption。

## 7. 日志合同

Shell 日志不记录原始命令。命令可能包含 token、header、heredoc、查询和正文，text mode、journald 和
Loki 都执行相同默认隐私边界。结构化事件记录 description、command fingerprint、字节数、cwd、shell
kind、tty、execution/session correlation、wall time、finish reason 和 exit code；授权排查原文时按关联
identity 查询 SessionDB/tool history。

Telegram outbound 内容只记录 `content_fp`、字节数和既有 correlation，不使用语义错误的
`command_fp`，也不把消息正文写入全局日志。

## 8. 正式 Workspace 迁移

### 8.1 前置条件

- #353～#355、#352 和本安装 PR 按 stack 顺序通过相邻 diff 与累计 Review。
- 32 GiB 硬件完成稳定性测试。
- 历史泄漏凭据完成轮换。
- 2 TB 备份范围覆盖 Workspace、config、plugin-data、模型凭据、浏览器 profile、release manifest，
  并完成至少一次隔离恢复抽样。
- hua-home 外围 service/network 和私有仓库 release identity 通过只读核对。

### 8.2 阶段

1. **空状态安装**：隔离 workspace、端口、容器名和 ingress 验证 Core、Bridge、插件、MCP、OpenCLI、
   Feed，不接入正式 Channel、调度、域名或手机。
2. **维护与恢复点**：停止旧 ingress 和后台写入；SQLite 使用 online backup 与 integrity check；普通
   文件在 DB 窗口前后核对 metadata/SHA256，漂移时整轮重试。
3. **迁入候选 state**：恢复到 `/srv/data/services/akashic/state`；不复制 cache/venv；重新准备 plugin
   generation，执行 legacy skill-link adoption，核对 session/message/media/plugin-data 集合。
4. **隔离验收**：以正式数据、非正式入口启动；禁止主动任务和外发 Channel；验证 DB、Web/Mobile API、
   Shell/File、Plugin/MCP、OpenCLI、Feed、浏览器 profile 和真实模型 turn。
5. **正式切换**：再次证明旧端无写入，启用 Channel/调度/主动任务，切换域名与手机入口，证明只有新端
   拥有 ingress，写 cutover receipt。
6. **观察**：观察日志、资源、MCP recovery 和 delivery；保留旧端、迁移前快照与全部 generation。

### 8.3 迁移回退

正式入口开放前失败可停止候选并恢复旧端。入口开放后若能证明零新写入仍可切回。新端已经产生消息、
plugin-data 或外部效果时，禁止自动切回：先冻结两端、保留现场，再由维护者决定数据恢复或效果对账，
防止 split-brain 和事实丢失。schema 已迁移时，只有 manifest 明确声明旧版本兼容才允许旧软件继续使用
同一 state。

旧 Workspace、旧镜像、旧 generation 和迁移快照的删除不属于迁移操作，观察窗口结束后仍需名称明确的
独立授权。

## 9. PR 切分

```text
#353 HostBridge / Docker / systemd
  - 并发 admission/drain
  - 自包含 Compose 引用
  - V1 协议文档对账

#354 Plugin / MCP / control
  - MCP generation-local degraded
  - 移出依赖后层 adoption 的 skill-link ownership 改动

#355 外围合同 / skill-links / 迁移
  - ownership journal、legacy adoption、persistence map
  - compose.external-services.yaml

#352 Observability
  - content_fp 与非敏感 Shell correlation

#356 Core + Bridge installer
  - bootstrap、release manager、install/doctor/rollback/migrate
  - generation/activation state 与正式迁移 runbook
```

下层 PR 修复前为每个分支建立 backup ref，再按依赖顺序传播到上层。每张 PR 必须在自己的 base..head
独立成立；默认安装 `main` 最新 commit 后，不能接受只有最终栈顶才可启动的中间主线状态。

## 10. 验证与停止条件

### 10.1 确定性测试

- 两个不同 session 的 Bridge command 互相等待 marker，证明真并发而非 wall-time 猜测。
- takeover 关闭 admission、等待在途 operation、清理 execution 空集后才发布新 boot。
- active MCP 耗尽后对应工具失败，但无关 turn、其他 MCP 和 Core primary tasks 继续。
- legacy link 合法 adoption；用户文件、越界、损坏、既有 registry 全部保持原物并失败。
- latest-main 和指定 commit 解析、远端不可达、非完整 SHA、identity drift。
- 同代幂等、并发 installer、partial staging、manifest mismatch 和 unit 内容相等。
- checkout、image、venv、Bridge probe、Core health 各阶段故障注入与上一代恢复。
- 无 `grpcio` 的系统 Python 可以完成 bootstrap，Bridge probe 明确使用 generation-owned interpreter；
  previous 恢复验证也失败时写 `recovery_failed` receipt 并再次停止服务。
- token、Shell command、Telegram content 不进入日志。

runner fake 只验证命令编排和状态机，不计作 Docker/systemd 通过。最终 Gate 必须依次执行定向测试、
PR change-impact Gate、累计 stack 全量测试、本机隔离安装、hua-home 真实首次安装、同代重装、跨 commit
升级、故障注入恢复和一致性 Workspace 迁移预演。

### 10.2 停止条件

以下任一发生时停止，不扩大权限或伪造成功：

- release identity、远端 commit、manifest、image、Bridge 或 Core identity 无法闭合；
- 上一代恢复失败；
- 正式 Workspace 在准备阶段发生未授权写入；
- 备份无法隔离恢复或 SQLite/file snapshot 漂移无法收敛；
- 新旧入口同时拥有 ingress；
- skill-link adoption 无法区分用户路径与插件 projection；
- 日志、manifest、命令输出或 Git tracked source 暴露 secret。

## 11. 回滚点

- 代码：每个 stacked branch 的 pre-fix backup ref；installer 使用独立 worktree/branch。
- 软件：previous release manifest、runtime.env 备份、unit 备份、旧 image/checkout/venv。
- 数据：维护窗口的一致性 Workspace snapshot、SQLite integrity evidence、普通文件 inventory。
- 外部入口：切换前的域名、Channel、mobile 和 schedule owner 记录。

设计文档本身不授权实施正式迁移、停止线上 Runtime、修改外围私有仓库或删除旧状态。
