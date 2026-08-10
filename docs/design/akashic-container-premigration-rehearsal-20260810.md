# Akashic 容器正式迁移前预演报告

- 日期：2026-08-10
- 状态：预演与清理完成；正式 Workspace 尚未迁移
- 分支：`design/akashic-hua-home-container-migration`
- 最终预演提交：`31253d64ce1407d6b64dc77358ce670ab33e6f97`
- 目标主机：`hua-home`
- 上游设计：[Akashic 容器与 Linux 主机运行适配设计](akashic-container-cloud-runtime-adaptation.md)

## 1. 结论

目标架构可成立，而且不会把 Agent 限制成“只能操作容器”的版本：Core 的代码、依赖、插件运行时和
生命周期固定在容器内；模型可调用的 Shell、File、Process、Git、SSH、OpenCLI 统一经过同一 Unix
用户的 Host Bridge，在语义上等同于登录 hua-home 后操作。

```text
┌──────────────────────── hua-home ────────────────────────┐
│ system systemd units · User=huashen                      │
│ ├─ Host Bridge · huashen UID · mise · SSH/Git/OpenCLI   │
│ └─ Core Compose owner · 有界重启 · StartLimit           │
│                         │ gRPC UDS + boot lease          │
│ ┌───────────────────────▼──────────────────────────────┐ │
│ │ Core container                                      │ │
│ │ Supervisor → Guardian → Gateway                     │ │
│ │ Session / Memory / Plugin / MCP / managed services  │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                          │
│ independent services                                     │
│ ├─ persistent headed Chromium + OpenCLI extension        │
│ └─ RSSHub + Redis + browser workers                      │
└──────────────────────────────────────────────────────────┘
```

这不是安全沙箱。Bridge 使用 `huashen` 身份，Agent 能做该用户通过 SSH 能做的操作；容器边界用于固定
Core 环境和生命周期，不用于隔离用户数据。Core 不挂 Docker Socket、不使用 privileged，也不挂宿主
根目录。

正式模板是 system unit，才能真实 `Requires/After=docker.service`；本次没有把模板写入 `/etc`，而是用
同一 `huashen` 身份的 user transient unit 验证进程、cgroup 和重启语义。这是预演实现差异，不是正式
架构选择。

## 2. 本次预演边界

正式本机 Akashic 一直保持运行，未停止、未迁移、未切域名。预演只使用：

- 本机隔离副本：`/mnt/data/akashic-container-rehearsal-20260810`；
- hua-home 隔离根：`/srv/data/experiments/akashic-premigration-20260810`；
- 实验容器、网络、端口、systemd transient unit；
- WebUI-only 候选配置；Telegram、QQ、mobile、Feishu、proactive 和正式 schedule 均不启动；
- 从正式 Workspace 在线一致性复制的数据副本，绝不反向合并。

快照工具会：

1. 对 SQLite 使用 online backup 并逐库执行 `integrity_check`；
2. 在 DB 窗口前后核对普通文件集合、metadata 和 SHA256；发生并发漂移时整轮重试；
3. 验证 SessionDB 中指向 Workspace 内的媒体引用在副本中存在；
4. 保留 `schedules.source.json` 作为证据，将候选 `schedules.json` 替换为 `[]`；
5. 只复制 plugin manifest，不复制机器绑定 cache/venv；所有 marketplace 插件先 disabled；
6. 不复制旧 `~/.akashic-plugin/data`，正式状态 owner 是 `workspace/plugin-data`。

本轮实际副本约 2.6 GiB、10,300 个文件、49 个 SQLite DB；完整性检查全部通过。169 个现存
Workspace 媒体引用保留；2 个原数据中已经缺失的 meme 文件被明确记录，而不是由迁移制造。

## 3. 已验证能力

### 3.1 Core、Bridge 与真实模型

- 非 root、只读 rootfs、`cap_drop=ALL`、`no-new-privileges` 的 Core 真实启动并保持 healthy。
- WebUI 的 Chat、Dashboard、静态资源、HTTP health 和反向代理链可达；宿主只发布 loopback 端口。
- OpenCode Go `deepseek-v4-flash`、variant `high` 在 hua-home 真实完成 turn；没有回退模型。
- 主 Agent 通过 Bridge 读取/写入宿主文件、执行 Git、运行测试、创建独立 worktree 并提交本地 commit。
- programmatic turn、child turn、subagent 和 Drift 共用同一 backend 装配；Bridge 不创建或拥有 turn。
- Bridge 支持短/长命令、execution ID、增量 `write_stdin`、PTY、stop、进程组清理、文件和 raw bytes。
- Skill `requires.bins/env` 在 Host Bridge namespace 探测；容器内没有 OpenCLI 也不会误报缺失。

### 3.2 Runtime 身份与发布输入

正式发布入口使用用户批准的完整 40 位 commit，不能传 `main`、`HEAD` 或 tag。构建过程：

```text
approved commit
  → git archive exact tree
  → exact file inventory + archive SHA256
  → pinned Arch base digest + 2026/08/09 repository snapshot
  → hash-locked Python requirements + npm ci
  → Dashboard + Chat + plugin panels
  → runtime-info
  → local content-addressed image ID
  → deployment-owned external release manifest
```

Core 启动前同时核对镜像 `runtime-info`、外部 release manifest、部署 commit、只读宿主 checkout 的
HEAD/tree/clean 状态。部署前再由 Docker Engine 核对 exact `sha256:...` image ID；Compose 不接受
`:latest` 作为正式输入。

`.runtime-ready.json` 发布 `sourceCommit` 和 `hostCheckout`；`sourceTree`、source archive/manifest digest、
base image、依赖锁、pacman digest 和 image ID 由镜像内 `runtime-info` 与外部 release manifest 持有。
不要把 readiness 文件单独描述成完整供应链证明。

Host Bridge 的每一个 RPC 都带 expected release commit 和 toolchain digest，服务端逐请求拒绝错代；
不只依赖启动 doctor 或两秒 monitor。hua-home 已真实验证 mise profile：Python 3.14.6、Node 22.23.1、
npm 10.9.8、uv 0.12.3、OpenCLI 1.8.6、OpenCode 1.18.15。

Runtime 参考 checkout 必须是一代 shallow checkout，只含当前 commit。不要传整仓 Git bundle：历史中曾
跟踪一个带凭据的 config backup，完整 bundle 会把已删除秘密一起带到服务器。Agent 可以从 shallow
HEAD 创建修复 worktree，并把 `origin` 指向私有 GitHub；当前运行源码始终只读。

### 3.3 插件、MCP 与生命周期

- `status_commands` 完成 source/install → latest child → parent terminal 后 promote → stable 行为验证。
- `huayue-skills` 的 OpenCLI skill 在 Agent turn 中加载，并通过 Host Bridge 执行真实 OpenCLI。
- `feed-mcp` 重建后使用复制的 state；RSSHub URL 从候选副本的 localhost 改为容器 DNS，正式数据未写。
- Feed MCP 真实列出 34 个 source、查询 latest；OpenAI 与 Claude 路由抓取成功。
- active MCP 被 SIGKILL 后 0.25/1/3 秒有界恢复；同一 60 秒窗口连续耗尽后只把该
  server/generation 标记为 degraded，对应工具调用 fail-loud，Core 与其他 generation 继续运行。
- managed service 与 MCP candidate promotion 都要求当前 generation/process 仍存活并重做 readiness；
  candidate failure 不能晋升；只有 managed service 的既有 critical 合同继续进入 Core primary fatal
  waiter，active MCP recovery 耗尽不会升级为 runtime fatal。
- plugin-data 在 install/reinstall/uninstall 中保留；cache、manifest、skill projection 分别由明确 owner
  管理。skill 同名普通目录/用户 symlink 冲突必须在 promotion 前 fail-loud，不能 `rmtree`。

### 3.4 Supervisor 与容灾

- `agent_restart` 保持现有事务：唯一 caller、冻结准入、turn completed、回复实际 delivery 后，才在
  同一个容器内重启 Gateway；容器 ID 不变、boot ID 改变。
- Bridge 是 Core primary task。停止 Bridge 后 Core 在约 4 秒内非零退出并清除 readiness。
- Bridge 由 systemd `KillMode=control-group` 托管；对 unit 执行 SIGKILL 后，它创建的 600 秒宿主
  marker 进程也消失，没有跨 boot 残留。
- Compose 自身 `restart: no`；只有 systemd 是外层异常恢复 owner，避免 Docker 与 systemd 双重循环。
- `akashic-core.service` 使用 `Restart=on-failure`、3 次 StartLimit；正常人工 stop 不触发无限拉起。

### 3.5 最终发布代验收

- 最终远端镜像从 clean shallow checkout `31253d64...` 构建；remote image ID 为
  `sha256:b4d7162562b9ff483d97e036aa1a731ec6251e87afe15e9d85fc6ddab54c0553`。
- 实际维护窗口按 `Core/Bridge stop → RSS/Browser workers restart → OpenCLI browser recreate → Bridge start
  → live sidecar preflight → Core start → Docker health` 执行；没有中间 Core 代提前启动。
- 新 boot `7dc69eb2994d42aabbc4211df0875c03` 发布 exact commit/checkout；`/`、`/chat/`、
  `/dashboard/`、`/api/chat/health`、`/api/shell/state` 全部 200，SessionDB `integrity_check=ok`。
- 最终真实 Agent turn 为 `turn:5d711443-f666-43a6-8568-8acbf69a0781`，completed；registry 的
  `default/agent` 均绑定 OpenCode Go `deepseek-v4-flash` high，请求无 model override，运行日志五次记录
  `provider=opencode-go`。
- 该 turn 真实定位 release-bound `akashic-runtime`、验证同 commit Git checkout、plugin-doctor healthy、
  OpenCLI GitHub whoami 登录，并从 Feed latest 读到一条 GitHub Blog。
- 最终 clean HEAD 全量测试 `3103 passed, 2 skipped`；Terra xhigh 最终复审无 P0/P1。构建期间
  `npm audit` 仍报告 2 个 moderate、1 个 high，不能声称依赖漏洞为零。

## 4. 外围服务

本节记录的是预演事实，不表示外围实现由 Core 仓库拥有。预演后已完成 owner 对账：RSSHub、Redis、
Browserless、real-browser 与持久 OpenCLI Chromium 的 canonical source、image pins、Compose、systemd 和
release manifest 已迁到独立私有仓库
[`kachofugetsu09/akashic-home-services`](https://github.com/kachofugetsu09/akashic-home-services)。Core 仓库只
消费外部网络和 service unit 合同，不再构建、校验或重启这些容器。

### 4.1 OpenCLI 浏览器身份

主方案是独立长期有头 Chromium，而不是纯 headless、Playwright storageState 或临时浏览器 worker：

```text
Agent Shell → Host Bridge → host opencli daemon :19825
                                  ▲
                                  │ localhost extension connection
                     headed Chromium container
                     persistent /config + loopback noVNC
```

hua-home 已建立持久 profile 并真实登录 GitHub；`opencli doctor` 的 Daemon、Extension、Connectivity
全部通过，`github whoami` 返回正确账户。重启浏览器后 profile 和 extension 保留。加入
`--disable-gpu --disable-software-rasterizer` 后，空闲 CPU 从异常的多核占用降到约 0.4%。

noVNC 只监听 loopback，通过 SSH tunnel 人工登录。profile 能长期保存，但任何外部网站都可能因服务端
撤销、MFA、风控或自然过期而要求重新登录；合同是“自动检测，固定入口人工恢复”，不是承诺永不过期。

Browserless 和 real-browser 容器也完成健康验证，但它们只作为 RSS/短任务 worker，不替代 OpenCLI 的
个人持久身份。RSS 与 OpenCLI 已拆成两个 Compose 文件，互不要求对方的环境变量。

### 4.2 RSSHub / Feed MCP

RSSHub、Redis、Browserless、real-browser 使用独立 `akashic-services` 网络。Core 加入同一外部网络；
Core 内 Feed MCP 使用 `http://rsshub:1200`，宿主侧诊断使用 loopback `http://127.0.0.1:1200`。
不得让 Host Bridge Shell 依赖 Docker DNS 名称。

本次实验为避免占用正式默认端口，宿主发布端口实际是 `11200`；`1200` 是正式模板默认值。

Feed 有一个独立于容器化的上线前缺口：`feed_query latest` 在指定 source、limit=1 时可返回，但
`sources/search` 会在同一 stdio MCP 进程内强制全量 poll，真实两轮均超过 30 秒；并行请求还会让客户端
在服务端已经写出部分响应时超时。它没有造成 Core 假健康，错误会 fail-loud，但正式迁移前应把“返回
缓存”和“刷新所有订阅”拆开或重新设计并发/超时合同。

## 5. 开发、合并和更新流程

发布是手动触发，用户始终是最后合并者：

1. 开发机持续开发和测试；Agent 可从正在运行的 commit 创建 worktree、修复、commit、push 分支和开 PR。
2. 用户 review 并合并到私有仓库 main；没有自动部署。
3. 部署端 fetch 后解析用户批准的完整 commit；从该 commit 生成 image 和外部 release manifest。
4. 为 Host Bridge 建同 commit venv，运行 `mise install`、toolchain identity 和 Bridge doctor。
5. 用一代 shallow checkout 部署 exact image ID；先在候选 loopback 端口启动，核对 readiness、WebUI、
   plugin/MCP、OpenCLI 和真实 V4 Flash turn。
6. 外围仓库先用自己的 release manifest 验证并启动 `akashic-home-services.service`；OpenCLI browser 由
   同仓库独立维护。随后 Core 仓库的 `scripts/restart_host_runtime_release.py` 只重启 Host Bridge 与 Core，
   由 systemd dependency 保证外围 unit 已就绪，最后等待 Core Docker health。
7. 新代失败时停止候选并使用上一代 exact manifest/image；数据 migration 只允许 append-only、可回滚规则。

发布失败不自动猜测或静默回滚；它停在明确维护态，保留报错命令和上一代 env/manifest，由用户手动修复。
这是已确认的运维合同。正式脚本由 root 调 systemd，但所有服务进程仍以 `huashen` 运行。

Core 不自行 pull、build 或替换正在运行的镜像。Agent 若发现自身问题，读取 runtime identity 与同版本源码，
在独立 worktree 修复并提交 PR；合并和正式 deploy 仍由用户决定。

## 6. 正式迁移前仍需用户完成

- 内存到货并完成硬件稳定性测试。
- 轮换历史 config backup 中曾出现的 Telegram token 与 embedding API key；本分支已经删除该文件并让
  release preflight 拒绝 config/auth/secret 路径，但不会擅自改写 Git 历史或中断当前正式 Agent。
- 确认私有 GitHub remote 和 hua-home SSH/GitHub 凭据；正式 Agent 的 worktree 必须能 push 分支/开 PR。
- 验证全局备份真实覆盖正式 Workspace、plugin-data、config、plugin manifest、浏览器 profile 和 release
  manifest，并做一次抽样恢复。
- 在最终维护窗口停止旧正式写入、创建一致性恢复点、恢复到 hua-home、逐项验收后再切域名/手机入口。

## 7. 本次预演清理

所有对象都以本轮 run ID 精确命名。完成报告取证后已执行：

1. 停止并删除 `akashic-premigration-core-20260810`、两个实验 Chromium、RSSHub/Redis/browser workers；
2. 停止并 reset `akashic-premigration-bridge-20260810.service`；
3. 删除实验 Compose network、实验 images、临时 SSH tunnel；
4. 删除远端 `/srv/data/experiments/akashic-premigration-20260810` 和旧实验根
   `/srv/data/experiments/akashic-container-8eb23df6`；
5. 将本机 `/mnt/data/akashic-container-rehearsal-20260810` 移入同盘 Trash，并删除本轮 `/tmp`
   bundle/image/probe；
6. 单独清理含旧凭据 checkout 的 Trash 条目；
7. 保留 mise 和其固定工具安装，作为正式迁移前置；保留 OpenCode 凭据的远端变更备份，直到正式迁移决策；
8. 最后证明：无 run-ID 容器、unit、socket、进程、端口或实验路径；本机正式 Supervisor/Gateway 与
   boot ID 未变化，正式 SessionDB `integrity_check=ok`。

实际清理清单与恢复边界：

- 停止/reset 所有 `akashic-final*`、`akashic-premigration*` transient unit；
- 对 `akashic-final-core-20260810`、`akashic-services`、`akashic-opencli-browser` 分别执行精确 Compose down；
- 删除旧手工 OpenCLI 容器、实验 socket/token、三个 `/srv/data/experiments/akashic-*` 根；
- 将 `~/.config/akashic-container` 恢复到预演前备份，不删除为正式迁移准备的 mise/OpenCode 登录态；
- 删除本机一致性副本、release manifest、shallow checkout stage 和实验 Docker image；
- 关闭本次 loopback SSH tunnel，复核正式本机 Agent 的 PID/boot/DB。

最终清理证明：hua-home 的匹配 unit/container/network/experiment dir/process/listener/image 均为 0，
`~/.config/akashic-container` 已恢复为预演前的不存在状态。本机匹配实验 container/image/tunnel/tmp 为 0；
正式本机 Agent 仍是 boot `ab885e960d804290a1ea927a789bef88`、Gateway PID `3910723`、ready，正式
SessionDB `integrity_check=ok`。本机 2.6 GiB 一致性副本已移动到 `/mnt/data` 所在文件系统的 Trash，
原路径不存在、清空 Trash 前可恢复；远端三个精确实验根已删除。

清理审计还抓到一个早期手工 cgroup probe 进程，它没有经过 systemd unit，因而活到最终清理。已核对完整
cmdline 后终止。它不来自最终 Bridge 方案，但证明验收脚本本身也必须由 unit/cgroup owner 执行，不能用
裸 `nohup`/后台 Python 冒充生命周期验证。

为后续正式迁移只保留：mise/OpenCode 登录态、OpenCLI profile 的变更前备份、两份 runtime.env 变更前
备份和本机 Trash 中的隔离副本；没有保留正在运行的实验服务或可被误启动的实验配置。

清理不使用 `docker system prune`、宽泛 glob、递归删除 home 或无关 Docker network。远端实验状态删除后
不可直接恢复；本机 Trash 与 `~/.local/state/hua-home-change-backups` 中的变更前备份仍保留，直到正式迁移
完成。

## 8. 诚实边界

- candidate MCP 的 validation path 用于防止正常插件代码误写 production state，不是对恶意同 UID Python
  代码的安全沙箱；本项目明确不以牺牲 Agent 手脚换取该隔离。正式插件仍必须来自用户 review 的 commit。
- exact lock 和 snapshot 固定环境输入，但 Python 构建 wheel 的时间元数据可能让“冷构建逐字节相同 image
  ID”不成立；部署以本次实际构建产生的 content-addressed image ID 和外部 manifest 为准。
- 当前 release 尚未生成 SPDX/CycloneDX SBOM；正式 CI 发布前补齐，不影响本轮功能预演，但不能称供应链
  证明完整。
- 网站登录态不能保证永不过期；MFA/风控发生时由固定 noVNC 入口人工恢复。
