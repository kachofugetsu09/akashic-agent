# Akashic 容器与 Host Bridge 非迁移实验合同

- 状态：experiment completed / formal migration not started
- 日期：2026-08-10
- 上游设计：[Akashic 容器与 Linux 主机运行适配设计](akashic-container-cloud-runtime-adaptation.md)
- 目标主机：本机开发环境与 `hua-home`
- 模型：OpenCode Go `deepseek-v4-flash`，variant `high`

## 1. Role

- 负责范围：实现前置适配后，在不迁移正式Akashic状态的条件下验证LocalBackend、Python Host
  Bridge、容器Core、Supervisor、programmatic calling和宿主工具能力。
- 当前阶段：隔离实验已完成；正式数据迁移、正式服务部署与手机/域名切换均未开始。
- capability owner：Core拥有turn/plugin/MCP/Supervisor；Bridge拥有宿主Shell/File/Process；
  systemd拥有宿主服务与异常容器恢复。

## 2. 为什么采用“两地分层实验”

只在本机实验无法证明真实systemd、无GUI服务器、Docker、宿主用户、冷启动和跨namespace行为；
直接在hua-home边开发又会把协议错误、依赖错误和服务器状态混在一起。

采用两阶段：

```text
本机开发机
  → 单元/协议/Local回归
  → 本机容器+本机Bridge集成
  → 失败快速修复
             │
             ▼
hua-home隔离实验
  → systemd Bridge
  → 合成Workspace Core容器
  → V4 Flash High真实turn
  → restart/failure/cold-boot验证
```

结论：**先在本机完成确定性开发和故障注入，再SSH到hua-home做最终环境证明。** hua-home实验不是
正式迁移，不接入正式域名、手机、调度、主动任务或当前Workspace。

## 3. 受保护状态

实验不得写入、复制、删除或启动：

- `/home/huashen/.akashic/workspace` 的正式 sessions、memory、plugin-data、mobile identity与调度；
- 正式 `~/.akashic/config.toml`、`auth.json`、GitHub App PEM和浏览器Default profile；
- 当前运行的Akashic Supervisor、Gateway、端口、systemd unit和Cloudflare入口；
- hua-home现有NAS、备份、Mihomo、Cloudflare、Docker stacks与防火墙规则；
- 旧 `~/.akashic-plugin/data`，其删除在正式迁移验收后另行授权。

实验只允许使用带run ID的隔离对象：

```text
本机：mktemp或专用test workspace
hua-home：/srv/data/experiments/akashic/<run-id>/
容器名/网络/volume label：akashic-experiment-<run-id>
Socket：/run/akashic-experiment/<run-id>/
systemd unit：akashic-experiment-<run-id>-*.service
```

正式Workspace不得bind进候选容器；实验数据不得反向merge到正式Workspace。

## 4. OpenCode身份与模型

### 4.1 当前证据

- 本机OpenCode版本为 `1.18.11`。
- `opencode-go`凭据在 `~/.local/share/opencode/auth.json` 中是 `type=api`，不是机器绑定OAuth。
- `deepseek-v4-flash`支持 `low/high/max`。
- 2026-08-10本机以 `--model opencode-go/deepseek-v4-flash --variant high` 真实返回
  `V4FLASH_HIGH_OK`。
- hua-home已安装官方Arch包OpenCode `1.18.15`，并仅定向迁移 `opencode-go` 凭据；真实
  `deepseek-v4-flash/high` 请求已通过。

### 4.2 定向迁移方法

可以在不要求用户重新登录的情况下把OpenCode Go能力迁到hua-home，但不得复制整个OpenCode目录。

1. 用mise在hua-home安装锁定版本OpenCode。
2. 变更前备份远端已有 `auth.json`；不存在则记录 absent。
3. 本机只读取 `opencode-go` 这一条API credential，通过现有SSH加密通道直接传给远端临时0600文件；
   不在命令输出、日志、Git、环境快照或任务合同中显示key。
4. 在远端原子merge为0600 `auth.json`，不复制OpenAI、GitHub Copilot等OAuth，也不复制
   `opencode.db`、sessions、logs、snapshots、tool-output或plugins。
5. 先用 `opencode auth list` 验证provider存在，再执行最小V4 Flash High请求。
6. 失败时恢复远端备份或删除本次新建auth文件；不能把失败归因成模型不可用后继续实验。

如果供应商撤销API key，必须明确报告并停止；不得回退其他provider冒充V4 Flash High。

## 5. 实验矩阵

### E0 · 基线与隔离证明

- 记录本机和hua-home的hostname、architecture、Git base、Docker/OpenCode/mise版本、失败unit和挂载。
- 记录正式Workspace目录摘要、进程、监听器和容器清单，只用于证明实验前后未变化，不读取凭据正文。
- hua-home执行 `hua-home-server/scripts/health-check.sh`；不得对Seagate USB备份盘执行SMART。

通过：健康检查无新故障，所有实验路径与正式路径不重叠。

### E1 · LocalBackend回归

- 用现有Shell/File测试证明短命令、长命令、`write_stdin`、PTY、timeout、stop和文件操作行为不变。
- 主Agent、programmatic、subagent和Drift的工具装配都解析到同一Local backend contract。
- 缺依赖、无权限、错误cwd和进程清理失败必须保持原错误分类。

通过：现有统一执行测试与新增backend contract测试均通过，工具schema没有环境分叉。

### E2 · 本机Bridge协议

- 启动一次性 Python Bridge UDS，使用真实 gRPC client 和 V1 Protobuf `BytesValue` JSON envelope
  执行 exec、PTY、stdin、resize 和 raw file。
- 并行启动不同owner的命令，验证输出、execution ID和stop不串线。
- 断开boot lease，证明该boot进程组TERM→KILL并成为空集；其他boot/用户进程不受影响。
- 发送错误major、过期token、非法owner、超大frame和断流，全部fail-loud。

通过：没有僵尸、孤儿或跨owner输出，Bridge日志不含完整env、token或文件正文。

### E3 · 本机容器集成

- 用合成Workspace启动固定digest候选Core，正式profile只注册BridgeBackend。
- Bridge不可用时Core不得ready，且不得创建Local execution。
- Core启动后运行主turn、programmatic child、subagent和Drift探针；MCP/PluginManager仍在容器内。
- 核对 runtime-info 的commit/tree/base digest/lock/image identity与只读源码。

通过：容器内没有错误宿主副本写入，programmatic lineage和candidate snapshot不因Bridge改变。

### E4 · hua-home Host Bridge

- 先备份拟新增/修改的systemd、mise和release文件；使用带run ID的实验unit和Socket。
- 证明Bridge进程的UID/GID/补充组、HOME、umask、locale和mise toolchain与合同一致。
- 真实执行：文件读写、Git、SSH BatchMode、`gh auth status`、OpenCode、网络、进程检查和一个需要PTY
  的无副作用命令。
- 对比SSH与Bridge的工具版本；允许差异只包括SSH连接变量、prompt/history和桌面变量。

通过：用户通过SSH能执行的探针均可由Bridge执行，且没有依赖交互 `.zshrc` 的偶然PATH。

### E5 · hua-home完整候选Runtime

- 只使用实验Workspace、实验plugin home、实验config、实验端口和实验容器网络。
- 以OpenCode Go V4 Flash High运行真实turn，不用假provider或mock成功。
- Agent完成：读取宿主源码、创建临时Git worktree、修改测试文件、运行测试、提交本地commit；实验结束
  删除该实验worktree，不push、不创建真实PR。
- 运行一个canary插件：source test → host CLI install → attached programmatic child → turn后提交；验证
  PluginManager、MCP和managed service未经过Bridge。
- 生成文本与图片文件，经raw bytes进入Core附件链并在实验客户端读取。

通过：SessionDB保留真实tool trace和model binding，V4 Flash High完成工具链，实验写集只落run目录。

### E6 · Supervisor与失败注入

- 正常 `agent_restart`：caller为唯一active turn，terminal+delivery后只重启Gateway，新boot ready。
- 存在其他active turn时重启明确拒绝并恢复准入。
- 注入Gateway readiness失败，证明Supervisor失败退出，systemd使用同digest有界重启容器。
- 实验中停止Bridge，证明Core fail-loud退出；恢复Bridge后Core才能重新ready。
- 旧boot lease断开时清空Bridge jobs，验证workspace lock、旧容器和宿主job三个空集。
- 达到StartLimit后保持failed并可由 `akashic-host doctor` 独立诊断，不出现双restart loop。

通过：正常restart、容器升级重启、异常崩溃和部署换版本四种语义互不混淆。

### E7 · OpenCLI边界

无人配合阶段只验证OpenCLI binary、daemon/extension连接合同和持久服务重启，不迁移工作站Chromium
profile、不复制cookie。服务器专用profile的首次网站登录仍是正式迁移前的人工步骤，不是本轮阻塞。

## 6. 验收清单

实验完成必须提供一份机器可读manifest和一份人类报告，至少包含：

- Git commit/tree、image/CLI/Bridge摘要、protocol major和capabilities；
- 每个实验的run ID、开始/结束时间、命令类别、exit和artifact路径；
- Local与Bridge工具schema/结果差异；
- OpenCode binary版本、provider/model/variant和真实terminal，不含credential；
- Core boot/container identity、Supervisor commit、readiness和systemd restart证据；
- Bridge execution owner、lease断开、进程组空集和残留扫描；
- 实验前后正式Workspace摘要、正式进程/监听器/容器清单无变化；
- 清理结果与未运行项。

下面任一项失败都不能称为“能力等价”：

- Bridge失败后出现Local fallback；
- programmatic child没有使用Bridge或丢失candidate lineage；
- Shell完成但文件落在错误namespace；
- Core/plugin/MCP owner被搬到宿主；
- old boot宿主进程仍存活；
- V4 Flash High被其他模型替代；
- 正式Workspace发生写入；
- 实验资源清理不完整。

## 7. 自主范围

在用户批准开始实验后，可以自主完成：

- 本机代码、测试、一次性容器和临时Workspace实验；
- hua-home只读调查、带run ID的隔离目录/容器/Socket、备份后的实验systemd unit；
- 锁定OpenCode安装和仅 `opencode-go` API credential的定向迁移；
- 无副作用V4 Flash High、Git/SSH/OpenCLI健康探针；
- 实验资源清理与报告。

仍需单独确认：

- 迁移或停止正式Akashic Workspace；
- 手机重新配对、正式域名/Cloudflare入口切换；
- 人工登录服务器Chromium网站；
- Git push、PR、GitHub review/comment等外部写入；
- 删除旧工作站plugin-data；
- 新增sudo白名单或执行系统升级、关机、重启。

## 8. 停止与回滚

- 修改hua-home持久文件前按hua-home-server skill逐文件备份；实验使用独立名称，不覆盖正式unit。
- Bridge/Core协议身份、Workspace隔离或模型身份任一无法证明时停止，不扩大权限或复制更多状态。
- hua-home资源不足、现有服务异常或全局备份状态异常时停止远端实验。
- 清理只删除本轮manifest明确拥有的run ID对象；不使用宽泛glob，不删除共享image或网络。
- OpenCode认证迁移失败时恢复备份；若服务器原先无文件，只删除本轮创建且摘要匹配的文件。
- 实验结束后再次运行hua-home健康检查和正式状态摘要；不一致时保留现场并报告。

## 9. 2026-08-10 实验结果

### 9.1 固定身份与现场

- 已验证Runtime commit：`8eb23df61ae653aecac9c183736a0b1389ecfdc8`；tree：
  `163a7f5cb3d6dd3fa14ec22e1d3dfb09f4c37016`。
- 已验证镜像：`sha256:574b5cb2e2088ee842c1d59cbfde43ab1cbcee505d360956b7a5b1a1d56335dd`。
- hua-home实验根：`/srv/data/experiments/akashic-container-8eb23df6`；运行时引用为独立、clean、
  detached Git clone。不能只bind普通Git worktree子目录，因为其`.git`可能指向容器不可见的
  common gitdir。
- 机器可读证据清单：`/srv/data/experiments/akashic-container-8eb23df6/run-manifest.json`，sha256
  `7f567264e3520ffe48094f83ec4282b59c1b0706245699a2cbc160e3387cb27a`。
- 正式Akashic Workspace、正式插件数据、手机身份、浏览器profile、域名和端口均未迁入候选。
- 实验结束时Core容器为`exited(1)`、Compose restart count为0、readiness已清除；这是故障注入的
  预期终态，不是正式服务故障。

### 9.2 已通过

- 本机完整镜像包含Dashboard、Chat和plugin静态产物；只读rootfs、非root UID/GID、cap drop、
  no-new-privileges和Supervisor PID 1均真实启动。
- Core启动前严格核对image runtime-info commit/tree、部署commit、宿主checkout HEAD/tree和clean
  状态；readiness发布相同commit与checkout。
- V4 Flash High本机真实turn完成宿主write/read、Shell、OpenCode和Git核对；hua-home真实turn又完成
  7次工具调用，并嵌套执行`opencode-go/deepseek-v4-flash --variant high`，返回
  `HUA_HOME_NESTED_V4_HIGH_OK`。
- Host Bridge支持Shell、增量stdin、PTY、stop和File/raw image；本机6,491,882字节data URL跨过
  gRPC默认4MiB边界后成功返回。
- `agent_restart`在本机和hua-home都保持原事务语义；hua-home boot ID从
  `b748be48f07e4325a722c718cdc80d64`切换到`01c652cd38904d8eba27dfc4dc68cc3b`，
  容器本身没有被Compose替换或自动重启。
- Bridge进入Core受监督primary task。停止Bridge后本机Core exit 1且readiness清除。
- hua-home Bridge由systemd user transient unit托管且`KillMode=control-group`。Bridge创建300秒
  `sleep`后，确认进程与Bridge同属unit cgroup；对unit执行SIGKILL后，marker PID消失、Core exit 1、
  readiness清除、无自动重启。journal保留unit failed/SIGKILL历史；实验终态transient unit已被
  systemd GC，当前无unit或Bridge/long-job进程残留。
- hua-home实验SessionDB `integrity_check=ok`，保留2个completed programmatic turn；实验前后由本地
  `hua-home-server/scripts/health-check.sh`控制端脚本核对远端，均无failed unit，NAS、网络和备份
  定时器正常。该脚本不部署在hua-home本身。
- 当前worktree相关回归测试：176 passed；最终Terra xhigh只读交叉Review未发现阻止远端实验的
  代码P0。

### 9.3 暴露的问题与未完成项

- Fresh registry只有`[llm] registry="workspace"`时，首次迁移后缺少`llm.main`会fail-loud。本轮为
  继续容器实验，备份远端新DB后迁入了本地隔离实验的无明文密钥registry。正式部署前必须完善
  fresh onboarding/seed流程，不能把复制实验DB当发布步骤。
- hua-home尚未安装mise：当前sudo需要用户交互密码，无人值守阶段没有扩大权限。Bridge实验使用显式
  HOME/LANG/PATH和锁定Python venv完成；正式部署前仍需把mise/toolchain profile落成唯一环境owner。
- hua-home没有`/usr/bin/hostname`，SSH和Bridge都同样不可用；Agent通过`/proc/sys/kernel/hostname`
  与`/etc/hostname`确认主机名。这不构成namespace差异，但应在正式能力清单中记录。
- 镜像仍使用`archlinux:latest`、`pacman -Syu`和范围型Python依赖；本次通过传输同一已测镜像消除
  两机差异，但尚未达到可重复重建的生产发布合同。
- 本轮未完成canary插件安装、MCP managed-service失败回滚、真实Drift任务、subagent任务、SSH远端
  target、OpenCLI浏览器边车、手机/域名、冷启动持久unit与正式数据迁移；不得据此宣称全部迁移完成。
- 本轮没有把正式Workspace的前后目录摘要保存进run root。Compose mount证据证明Core没有bind正式
  Workspace，但不能仅靠当前现场证明Bridge历史上从未收到正式路径写请求；正式迁移Gate必须先补
  可复核的正式状态基线与迁移后对照。
