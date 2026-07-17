# 移动端与跨仓库语义 Gate 技术设计

- 状态：working design；已确认边界可执行，业务未知项不得由实现反向定义
- 日期：2026-07-18
- 目标读者：核心维护者、移动端维护者、插件维护者、评审者、Gate 实现者
- 关联条款：MOB-001、STA-001～STA-003、CTX-001、SES-001～SES-006、PLG-001、PLG-004、PLG-008～PLG-009、WSP-004、TST-001～TST-006
- 相关决策：[0002](../decisions/0002-context-reduction-is-a-nondestructive-projection.md)、[0003](../decisions/0003-core-capability-ownership-is-semantic.md)
- Gate 总体设计：[变更影响与跨仓库契约 Gate](../spark/2026-07-16-change-impact-contract-gate.md)

## 1. 目的与证据标签

这份设计把移动端、核心 runtime、协议快照和外部插件放进同一条可复现的验收链。它不新增产品需求，也不把某个候选 PR 的当前行为当成长期语义。

全文使用四种标签：

- **C（confirmed）**：用户已确认，或已经写入 `projectneed.md` 的稳定不变量。
- **F（fact）**：从指定代码、schema、测试或运行环境观察到的当前事实；只能说明“现在是什么”。
- **I（inference）**：为实现已确认边界提出的技术推断；可以进入候选实现和 Gate，不能据此扩大产品范围。
- **U（unknown）**：不同答案会改变产品行为、数据减少或恢复承诺，必须停止并由维护者决定。

评审结论必须保留标签。F 不能自动升级成 C；I 即使测试全绿，也不能反向修改 `projectneed.md`；U 不能用候选实现、平台惯例或“未来可能复用”补齐。

## 2. Owner 与数据增减合同

### 2.1 权威层次

**C：** Akashic 核心的 `sessions.db/messages` 是完整对话正文真源。正常收发只追加消息；只有用户主动撤销消息或删除会话时，名称明确的数据管理操作才可以减少正文，并携带目标、cascade、备份和审计证据。Prompt、runtime history view、客户端缓存和投影都不能改变这条保留规则。

**C：** Android Room 中从服务端 session、message、turn 和事件重放得到的行是本地投影，可以从服务端权威状态重建。重建只拥有服务端投影；它不拥有用户尚未提交完成的本地工作。

**C：** 本地 outbox、待发送消息、上传中的附件、失败后可重试状态和已有 draft 是客户端连续性状态。一次 `sync.reset_required`、cursor 回退、Room 投影清空或历史重新拉取不得把这些对象当作服务端投影删除。

```text
┌──────────────────────────────────┐
│ Core SessionDB                   │  完整会话真源；正常运行 append-only
└───────────────┬──────────────────┘
                │ event/history snapshot
                ▼
┌──────────────────────────────────┐
│ Android server projection        │  可清空并从同一服务端重建
└──────────────────────────────────┘

┌──────────────────────────────────┐
│ Android local continuity state   │  outbox / pending / failed / draft
└──────────────────────────────────┘  不属于 projection reset 的 write set
```

### 2.2 移动端持久对象

| 对象 | 类别 | 正常增加 | 允许更新 | 允许减少 | reset / restart 验收 |
|---|---|---|---|---|---|
| Room 中的服务端 session/message/turn 投影 | C：可重建投影 | 收到已校验事件或历史页后写入 | ack、delivery、turn 终态和 cursor 按协议推进 | 只由服务端权威删除或显式投影重建减少；不能扩大成核心历史删除 | reset 后由服务端重建；不得反向写 core SessionDB |
| outbox command 与本地待发送消息 | C：客户端连续性状态 | 用户提交动作时与本地展示状态原子创建 | `pending → in_flight → retry/acked/failed` | 只有服务端 ack、明确不可重试终态或用户取消协议允许减少 | projection reset、断线和进程重启后仍可恢复 |
| 附件 draft / upload transfer | C：客户端连续性状态 | 用户选择文件并完成受控复制后创建 | offset、上传状态和失败原因按传输协议更新 | 上传提交完成后的清理、用户明确移除或明确账户删除协议 | projection reset 不清除 pending/ready/failed draft；重启可继续或明确失败 |
| pending notification | C：持久交付快照 | 消息与 cursor 提交时在同一事务创建 | 通知投递尝试可以更新必要状态 | 只有系统通知已经成功交付、用户明确移除对应服务端，或另有已批准终止协议 | projection reset 后仍存在；进程被杀后仍能继续投递 |
| received attachment cache | C：可驱逐内容，描述符仍受消息引用保护 | 下载确认或本地已上传附件导入后创建 | LRU 时间、下载状态和确认 offset | 配额可驱逐内容；被消息引用的描述符与重新下载能力不能随意删除 | 重启先对账文件和 DB；不得把未确认字节发布成完整文件 |
| 文本 draft 是否需要新增独立持久表 | U | 未决定 | 未决定 | 未决定 | 不能从“附件 draft 要保留”推导出新文本草稿功能 |

`sync.reset_required` 的 write set 必须使用正向白名单描述，例如“删除并重建 server projection tables、重置该 device cursor”。实现若使用外键 cascade、`clearAllTables()` 或 destructive migration，必须证明它不会触达 local continuity state 和 pending notification。

### 2.3 持久通知的提交点

**C：** pending notification 不是可丢的内存事件，而是“这条已提交消息仍需交给 Android 通知系统”的持久快照。

推荐事务顺序：

```text
receive final/proactive message
  → transaction(message projection + cursor + pending notification)
  → commit
  → Android NotificationManager.notify(...)
  → success 后消费 pending notification
```

进程在事务提交后、`notify()` 前死亡时，重启必须再次看到 pending notification。进程在 `notify()` 后、消费前死亡时，稳定 notification ID 与 `onlyAlertOnce` 等平台机制负责幂等展示；不能为避免重复而在通知外部效果发生前删除快照。

服务端投影重建不是通知已交付证据，所以不能通过 message 外键 cascade 删除 pending notification。用户明确删除服务器、账户或会话时是否同时终止尚未投递通知，必须由对应 destructive command 明确写出 cascade，而不是依赖 Room 表结构偶然决定。

### 2.4 下载 offset 的确认点

**C：** `.part` 文件中已经 fsync 的字节只证明“客户端收到并保存了 binary frame”，不证明服务端的配对 `attachment.download.ok` 已经确认该 chunk 的 attachment ID、offset、next offset、大小、摘要和 complete 状态。

正确提交顺序：

```text
binary frame
  → 检查 attachment ID 与当前 confirmed offset
  → 截断到 confirmed offset
  → 写入 chunk + fsync(.part)
  → 暂不推进 DB confirmed offset

matching attachment.download.ok
  → 校验 command ID 和全部元数据
  → 非末块：推进 DB confirmed offset
  → 末块：校验 size + SHA-256
            → atomic move(.part → final)
            → DB state = cached
```

进程在 binary fsync 后、matching ok 前死亡时，启动恢复把 `.part` 截断到 DB confirmed offset，再从该 offset 请求。full-size `.part` 没有 matching ok 时也不能直接发布；DB offset 提前推进、文件长度自动成为 offset、或只靠最终 SHA-256 都是错误 mutant。

**F：** 移动端候选提交 `1c7ed8a` 已把 offset 推进移到 matching ok 之后，并增加未确认尾部截断测试。这个提交是实现证据，不是协议真源；stack 中每个 downstream head 仍需实际包含并重跑该场景。

## 3. 协议与外部仓库版本固定

### 3.1 客户端协议快照

**C：** 服务端中立协议由核心仓库拥有；移动端保存可离线构建和评审的精确快照，不能通过修改客户端快照反向定义核心语义。

移动端 `protocol/source.json` 至少绑定：

```json
{
  "source_repository": "https://github.com/<owner>/<core>",
  "source_commit": "<40-hex commit>",
  "source_path": "schema/mobile-realtime-v1.json",
  "snapshot_path": "protocol/mobile-realtime-v1.json",
  "sha256": "<snapshot sha256>"
}
```

Gate 必须从 `source_repository + source_commit + source_path` 读取原文，重算 SHA-256，并与客户端 snapshot 比较。分支名、PR URL、remote-tracking ref、本机 core checkout 和注释里的 commit 都不能代替 40 位 commit 与内容 hash。

协议新增能力时固定顺序为：

1. 在核心或中立协议仓库批准语义与 owner。
2. 提交协议 schema 和服务端实现，获得不可变 commit。
3. 移动端同步 snapshot，更新 `source.json`。
4. 分别运行 schema parity、核心实现、客户端 codec 和真实互操作场景。
5. 报告记录 core commit、mobile commit、schema hash 和场景结果。

### 3.2 插件与 MCP revision

**C：** 跨仓库 Gate 从 canonical GitHub `repository + 完整 ref` 查询远端 revision，并在本次 run 开始前冻结为 commit SHA。一个 GitHub 链接只解决“去哪里找”，移动 ref 本身不能提供可复现证据。

```text
repository + refs/heads/main
  → git ls-remote
  → resolved provider SHA
  → 下载/checkout 该 SHA
  → 安装到空 plugin home
  → 验收
  → report(repository, requested ref, resolved SHA)
```

本次 Gate 通过后 provider 又更新，不需要把旧报告伪装成失败：旧报告只证明旧的 `consumer SHA × provider SHA` 组合。下一次 PR、nightly、release 或手动重新验收重新解析 ref；resolved SHA 改变后，旧组合不能被复用为新版本证据。

**I：** Required check 的 cache key 应包含 consumer source digest、protocol digest、provider SHA、scenario catalog digest 和 Gate version。缺少任一项时重新运行，不使用本地插件 cache 猜测等价。

## 4. 语义干净的 Gate 环境

### 4.1 每次运行的隔离边界

G1/G2 每次创建独立目录：

```text
/tmp/akashic-change-gate-<run-id>/
├── workspace/       空 Akashic workspace
├── plugin-home/     空安装根
├── home/            不继承宿主用户配置
├── config.toml      本次生成
├── providers/       只含冻结 SHA 的 source
├── fixtures/        声明式输入
└── reports/         本次证据
```

- source 只读；只允许写本次 sandbox 和 tmpfs。
- 不挂载、不复制正式 workspace、`sessions.db`、正式 config、正式 plugin cache 或用户正文。
- 需要历史的场景通过正式写入入口创建，不复制线上数据库。
- provider 必须走真实安装与 runtime discovery，不从本机 editable checkout import。
- Gate 结束审计容器、网络、volume、进程和 sandbox 外 write set；cleanup 失败即失败。

### 4.2 真实 seam 与 mutant

普通单元测试只能证明局部函数符合当前断言。P0 Gate 还要跨越真正会漂移的 seam，并用已知错误证明 oracle 有杀伤力：

| 场景 | 必须穿过的真实 seam | 正确观察 | 必杀 mutant |
|---|---|---|---|
| context trim | SessionStore → runtime history view → PromptContext → provider retry → restart | prompt 可以缩小；SessionDB rows、ID、seq、正文和 embeddings 不变 | retry 后 DELETE/UPDATE 旧 messages，或重载只剩裁切窗口 |
| MCP finality/freshness | core MCP call → 真实插件进程 → provider 持久状态 → 正式读取接口 | success 返回时目标状态可见；刷新后 cursor/内容推进；重启仍成立 | 只排队后台任务便返回、停止刷新却持续返回合法旧 payload |
| mobile projection reset | 协议 reset event → Room transaction → history rebuild → reconnect | server projection 重建；outbox/pending/failed/draft/notification 保持 | `clearAllTables`、错误 FK cascade、destructive migration |
| attachment download | binary socket frame → `.part` fsync → matching ok → Room offset → restart | 只有 matching ok 推进 confirmed offset；未确认尾部重启被截断 | binary 到达即推进 DB offset、full `.part` 自动发布 |
| Observe turn identity | core lifecycle → EventBus → Observe SQLite → mobile RPC/display | 同一 assistant message 的稳定 ID 和 usage 可由移动端查询 | 在任一 seam 清空、重建或换掉 message ID |

mutant job 必须因对应 invariant 的状态差异失败。导入失败、fixture 未启动或超时不能算“成功杀死 mutant”。

### 4.3 Observe 真实链

**F：** 当前审查中的 Observe 候选使用以下链路保护移动端 message identity：

```text
SessionManager 分配并持久化 assistant message ID
  → AfterReasoning 构造 OutboundMessage
  → AfterTurn 构造 TurnCommitted
  → EventBus fanout
  → Observe _observe_turn_committed
  → TraceWriter 写 observe.db
  → kvcache.message_usage(session_id, message_id)
  → mobile panel 按同一 message ID 查询和展示
```

G2 不能只 import Observe 或调用一个伪造 reader。它必须用当前 core lifecycle 产生真实 committed turn，等待 TraceWriter 落库，再通过插件公开的 mobile RPC 读取。随后运行“清空稳定 message ID”的 mutant，确认 SQLite 或 mobile query 断言失败。

Observe 属于独立插件仓库，所以报告同时绑定 core consumer SHA、Observe remote ref、resolved SHA、安装产物 digest 和 mobile query 场景。插件未安装的公共贡献者可以运行 G1；required G2 由持有插件访问条件的环境返回明确的 `passed`、`failed` 或 `not_affected`。

## 5. Docker、Mobile Lab 与 Pixel 7 的证据层级

设备测试有价值，但不能替代可重复 Gate。验证从低层到高层累积：

| 层级 | 环境 | 证明什么 | 不能证明什么 |
|---|---|---|---|
| L0 | schema、JVM 单测、lint、构建 | codec、状态机、迁移代码和构建成立 | 真实进程、OS lifecycle、远端组合 |
| L1 | 公开 G1 Docker | 当前 core 在空 workspace 的确定性语义 | 私有插件、Android 平台行为 |
| L2 | 私有 G2 Docker | 冻结 provider SHA 的真实安装、进程、seam 和 mutant | 外部服务实时漂移、手机 OS 行为 |
| L3 | Mobile Lab | 独立 WSS、独立 workspace/plugin home 下的 core ↔ Android 互操作 | 正式线上状态；不得读取线上数据 |
| L4 | Pixel 7 + ADB | Room migration、进程杀死/恢复、后台连接、通知展示、附件文件系统和真实 Android 限制 | 普通贡献者 CI 可重复性 |

**F：** 当前 Mobile Lab 位于 `docker/mobile-lab/`，运行数据写入忽略版本控制的 `docker/debug/profiles/mobile-lab/`，不挂载正式 workspace，也不启动 Telegram、QQ 和 proactive。它适合 L3，不是正式环境。

**I：** 在控制器支持 run-specific profile 前，复用固定 `mobile-lab` profile 必须先停止旧容器，备份现有测试 profile，并在报告中记录备份、启动时间、core SHA、APK SHA 和 cleanup。任何路径解析到正式 workspace 时立即停止。

Pixel 验收固定遵守：

1. `adb devices` 核对唯一目标和设备序列，不对其他设备执行命令。
2. 读取已安装 package、version、signer 和 debuggable 状态；签名不兼容时不得卸载、清数据或覆盖用户现有 app。
3. 优先使用可保留数据的 `adb install -r` 或独立测试 application ID；没有安全安装路径就把设备项标记为 blocked。
4. 只连接 Mobile Lab 域名和测试配对，不连接正式 gateway。
5. 场景至少覆盖进程 kill/restart、Room migration、projection reset、本地未完成工作保留、notification delivery 和附件断点恢复。
6. 保存命令、设备/API 版本、APK digest、截图或 logcat、Mobile Lab report；测试后停止容器并审计残留。

Pixel 证据是当前维护者机器上的手工 L4 结果。CI 没有 Pixel 或 Android 虚拟设备时，PR 必须把 L0～L3 与 L4 分开报告，不能把“本机测过”伪装成每位协作者都能运行的 required check。

## 6. Stacked PR 的传播与评审

每张 PR 只拥有相邻 `base..head` 的新增语义；最终 head 拥有整个 stack 的累计行为。上游修复没有传播到 downstream，就不能说最终产品已修复。

```text
PR1 protocol/base
  ↓ include exact upstream fix commit
PR2 persistence
  ↓
PR3 upload
  ↓
PR4 download
  ↓
PR5 passive delivery
  ↓
PR6 interactions  ← 累计构建、迁移、Mobile Lab、Pixel 验收
```

修复和更新顺序：

1. 记录每层目标分支、base SHA、head SHA、protocol source commit/hash。
2. 在最早拥有问题的 PR 添加最小修复 commit，运行该层测试。
3. 把这个准确 commit 传播到每个 downstream branch；禁止只手工复制代码后声称已包含上游修复。
4. 每层重新审查相邻 diff，确保传播 commit 没有引入与该层 owner 无关的变化。
5. 在最终 head 运行 schema migration matrix、完整 Android 构建、Mobile Lab 和允许时的 Pixel 场景。
6. PR 报告列出 `base/head/protocol/core/provider` 组合和实际 Gate 报告；后续任何 head 变化使旧 source digest 失效。

跨仓库 core 或插件修复先在 owner 仓库得到独立 commit 和测试，再更新 consumer pin 或 G2 组合。移动端分支不能把未发布的本机 core/plugin checkout 当成通过证据。

## 7. 两次历史事故怎样进入 Gate

### 7.1 Context trim 事故

事故不是“DELETE SQL 写错”，而是执行者把 prompt 临时窗口误认成长期 session 保留范围。回归场景必须同时观察输入和持久状态：

1. 在空 workspace 通过正式 session API 写入多轮消息和 embeddings。
2. 保存完整 rows、ID、seq、正文、embedding 和最大 seq。
3. 让 provider 第一次抛出 ContextLengthError，确认后续 request 的 prompt history 缩小。
4. 比较 SessionDB 完整快照和 write set，确认旧 rows 无 UPDATE/DELETE。
5. 重启 core，再次读取全部历史；追加消息必须使用 `max(seq) + 1`。
6. 应用 DELETE mutant 后重跑；Gate 必须以 CTX-001/SES-005 状态差异失败。

只断言“重试成功”“token 数下降”或“单测期待删除”都不能保护原意。

### 7.2 MCP sync/async 事故

事故不是 Python 函数签名不同，而是主仓库把“调用成功”理解为终态可观察，插件却把它理解为“已安排后台刷新”。接口形状一致仍会返回合法旧数据。

确定性场景使用同一 Docker 私网中的真实协议数据源：

1. 发布 V1，启动真实 core 和冻结 SHA 的真实 provider。
2. 通过正式 MCP 调用看到 V1，并记录 provider cursor/SQLite。
3. 数据源推进到 V2。
4. 执行声明为普通成功的刷新或等待插件声明的自动刷新边界。
5. success 返回时，通过正式 MCP read、provider state 和 cursor 同时看到 V2。
6. 重启 provider/core 后仍看到 V2，且不重复提交。
7. 停止持续刷新或让调用只 enqueue 后立即 success；freshness/finality mutant 必须失败。

如果产品真正需要延迟完成，必须另行批准带 task ID、状态和 result 的异步协议；不能让同一个普通 success 在不同仓库拥有两种完成语义。

## 8. 业务未知与停止条件

以下问题当前不能由本设计决定：

1. **U：大附件历史加载。** 历史中大于等于 10 MiB 的附件应该 eager 下载、按需下载还是只显示描述符，属于带宽、缓存和体验策略。任何选项都不能削弱附件身份、摘要和显式下载错误。
2. **U：内部 durable inbox gap。** 由损坏或不一致数据库副本造成的本地 seq 缺口，产品是否承诺自动恢复，还是 fail-loud 后要求重新配对/重建，需要单独决定损坏恢复边界。
3. **U：旧消息编辑。** 核心 `update_message` 应保留原位 UPDATE，还是追加 correction/revision，仍由持久化状态地图跟踪；移动端不得自行定义。
4. **U：功能提案。** failed-message 手动重试、后台大文件续传、全新的文本 composer draft、搜索和 WebView/plugin UI 是独立 feature，不能作为当前修复的附带验收条件。

实现中发现以下任一情况立即停止，不用测试结果替代产品决定：

- 需要删除或覆盖 core SessionDB 正文才能让移动端同步成立。
- projection reset 无法在不删除 local continuity state 的情况下实现。
- 一个协议字段在 core、mobile 或 plugin 中存在两种合理但不同的完成/删除语义。
- 需要把 Android、Room、通知或界面概念加入中立核心协议。
- 设备验收要求卸载、清数据、使用正式 workspace 或连接正式 gateway。

## 9. 一次变更的可执行清单

```text
Read
  → INDEX / projectneed IDs / decision / this design / real code
Ownership
  → core | protocol | mobile | plugin；列出 authoritative state
Contract
  → semantic delta / protected state / allowed write set / unknowns
Pin
  → base/head + protocol commit/hash + provider repo/ref/resolved SHA
Isolate
  → clean worktree + empty workspace/plugin home + backup test profile
Verify
  → L0 → G1 → required G2 → Mobile Lab → allowed Pixel
Mutate
  → known wrong behavior must fail for the right invariant
Review
  → adjacent stacked diff + final cumulative head
Deliver
  → commits/PRs + report digests + unresolved U items only
```

交付报告至少包含：

- capability owner、consumer scope、runtime patch 理由和客户端替代方案；
- core/mobile/plugin 的 repository、base、head、requested ref、resolved SHA；
- protocol path、source commit、snapshot SHA-256；
- sandbox 路径策略、workspace/plugin-home 是否为空、正式路径不可达证据；
- 运行的 seam、mutant、L0～L4 结果和未运行理由；
- Room/SQLite/文件 write set，尤其是 reset、notification 和 attachment offset；
- stack 传播关系与最终累计 head；
- 仍为 U 的业务问题，不把它们写成已完成或默认方案。

## 10. 当前实施边界

**F：** 公开 Gate 已有 diff 选择、一次性 Docker sandbox 和报告入口；private companion 已能保存 GitHub provider revision 与真实 seam 场景。Mobile Lab 已与正式 workspace 隔离。

**I：** 下一步只把现有 Feed 与 Observe remote-revision 场景接入同一个 G2 Docker controller，并建立始终返回 `passed`、`failed` 或 `not_affected` 的外部状态。完成前不能把“本机脚本可运行”描述成完整 required Gate。

**U：** G2 外部状态由哪一个受保护 runner/仓库发布，以及 Pixel 手工证据是否作为 release promotion 的必需项，仍取决于仓库权限和发布策略；它们不影响本设计中的数据保护和版本固定合同。
