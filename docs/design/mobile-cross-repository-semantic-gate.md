# 移动端与跨仓库语义 Gate 技术设计

- 状态：working design；已确认边界可执行，业务未知项不得由实现反向定义
- 日期：2026-07-18
- 目标读者：核心维护者、移动端维护者、插件维护者、评审者、Gate 实现者
- 关联条款：GOV-005、MOB-001～MOB-004、STA-001～STA-003、CTX-001、SES-001～SES-006、PLG-001、PLG-004、PLG-008～PLG-009、WSP-004、TST-001～TST-008
- 相关决策：[0002](../decisions/0002-context-reduction-is-a-nondestructive-projection.md)、[0003](../decisions/0003-core-capability-ownership-is-semantic.md)、[0004](../decisions/0004-cross-repository-evidence-is-an-immutable-combination.md)
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
| pending notification | C：持久交付快照 | 消息与 cursor 提交时在同一事务创建 | 通知投递尝试可以更新必要状态 | 系统通知成功交付后可以按提交协议消费；服务端、账户或会话的破坏性删除是否终止待投递通知仍是 U，未确认前不得隐式 cascade | projection reset 后仍存在；进程被杀后仍能继续投递 |
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

服务端投影重建不是通知已交付证据，所以不能通过 message 外键 cascade 删除 pending notification。

**U：** 用户明确删除服务端、账户或会话时，是否同时终止尚未投递通知仍未决定。维护者确认前，destructive command 必须保持 pending notification，不得依赖 Room 表结构偶然 cascade；确认后再把终止条件、影响预览和恢复方式写入命令合同。

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

**F（历史 PR6 候选）：** `3f81275a52b0b87438f5d31041a71997edbac267` 已传播当时 PR5 的最终状态，并在栈顶重新运行附件、累计测试和设备 Gate。这个 commit 是历史实现证据，不是当前发布 head 或协议真源。

### 2.5 Room schema lineage 与汇合迁移

**C：** Room `user_version` 不是 schema identity。相邻 PR、已评审分支或早期公开候选可能使用同一个版本号，却拥有不同的列、外键和持久对象。迁移不能根据版本号猜来源，也不能用 destructive fallback 把分叉抹平。

```text
final PR5 v4: remote + media + notification + durable stop
                                   ┐
reviewed PR6 v4: media + notification + server sequence
                                   ├─ identify exact lineage ─→ canonical v5
public PR6 v3: media + server sequence
                                   ┘
```

每条已知 lineage 先核对表、列、索引和外键，再只增加目标版本缺少的结构。迁移后的保留集合取所有上游已批准状态的并集；例如加入 server sequence 不能丢失 remote pairing、pending notification 或 durable stop。只命中部分特征的未知 v4 必须 fail-loud，不能被当成“最接近”的已知版本。

**F（历史 PR6 迁移证据）：** `3f81275` 在 Pixel 7 上逐条重跑的 schema v5 迁移矩阵覆盖 final PR5 v4、reviewed PR6 v4、original public PR6 v3 和 canonical 1→5，并分别证明 partial v3 与 partial v4 fail-loud。当前独立移动仓库已演进到 schema v10；这组证据只负责旧分叉的汇合，后续 5→10 仍由当前仓库的逐版本 migration tests 负责。

### 2.6 主动消息的实时与历史身份

**C：** 被动消息按完整 Turn 提交，主动消息只在 dispatch 明确成功后追加到 SessionDB。主动实时事件不是第二条会话事实，而是同一条已发送消息的客户端投影。

```text
Core proactive delivery_id
          ├─ message.proactive payload
          └─ SessionDB message extra → history.page extra
                                │
                                ▼
                    Android 精确合并为一条消息
```

**I：** Core 在一次主动发送尝试开始时生成 `delivery_id`，通过只对内部出站调用开放的 metadata 通道交给支持该能力的 channel；dispatch 成功后把同一值随 assistant 消息追加到 SessionDB。该字段不改变 SessionDB message ID，也不提前创建会话消息。

**C：** Android 收到两种投影时优先按 `delivery_id` 合并。只有旧事件或旧历史没有该字段时，才允许使用带 `proactive=true`、相同文本、限定时间窗且唯一最近候选的兼容规则；候选不唯一时保留两条，不能猜测身份。

**C：** 历史页尚未到达时，Android 使用 `reply_to.delivery_id` 引用主动实时投影；Core 在同一 session 的主动 assistant 消息中唯一解析该身份，并把 canonical message ID 写进入站引用 metadata。历史页完成投影迁移后，Android 恢复使用 `reply_to.message_id`。`delivery_id`、`client_message_id` 与 `message_id` 是三个互斥协议身份，本地 `proactive:` 前缀不得越过客户端编码边界。

当前接受的边缘窗口是：Mobile 已接收后 Core 在追加历史前崩溃，该主动消息可能只保留在手机投影中。当前方案不为这个窗口新增 outbox、重试状态机或 SessionDB 表。

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

协议 source pin 与实际 runtime pin 是两个身份。旧客户端可以固定核心仓库中的归档 schema commit，而互操作 Gate 同时记录本次运行的 core commit/tree。两者不同不表示漂移；缺少任一身份才表示报告无法复现。

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
| Room lineage merge | 各已知旧 schema → migration → canonical schema export → DAO readback | 每条 lineage 的状态并集完整保留，未知形状 fail-loud | 只看 `user_version`、drop/recreate、遗漏某一分支列 |
| command catalog | request tracker → `command.list` → reconnect/reset/close → Compose filtering | 当前请求获胜；已取消迟到响应忽略；未知 ID 失败；目录在 source 变化后清空 | 旧响应覆盖新目录、断线后继续展示 stale catalog |
| cross-language bounds | Python provider → JSON schema → Kotlin codec → UI | Unicode code point、终态和错误分类在各端一致 | Python `len` 与 Kotlin UTF-16 `length` 分别定义协议 |

mutant job 必须因对应 invariant 的状态差异失败。导入失败、fixture 未启动或超时不能算“成功杀死 mutant”。

**F（当前公开 pilot）：** context 场景已经穿过真实 `DefaultReasoner.run_turn` retry，并用 SQLite trace、完整消息/embedding 快照、重启和 seq 续接观察持久状态。当前 fault injection 在测试 fixture 中直接执行历史 DELETE，再确认同一组快照与 write-set oracle 拒绝该状态；它不是 SQLite authorizer，也不是把错误 patch 注入真实 retry seam 的完整 mutant job。

**I：** 下一阶段用 SQLite authorizer 记录所有受保护写入尝试，并在一次性候选副本中把 DELETE/UPDATE mutant 注入真实 retry seam。只有健康路径通过、mutant 因 CTX-001/SES-005 状态差异失败，且导入错误、fixture 错误和超时均不能被计作 kill，才能把这部分从 pilot 提升为完整 P0 oracle。

### 4.3 Observe 真实链

**F：** Observe PR [#1](https://github.com/akashic-plugins/observe/pull/1) 的修复 commit 为 `b7f9d4ecee877d22b5452651d9abf699b2d30b7b`，canonical `main` 已解析为 merge commit `b434fa74b370fafcd0c64129fe1f641f73f0dbcf`，使用以下链路保护移动端 message identity：

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

**F（限定到候选 revision）：** [核心 PR #129](https://github.com/kachofugetsu09/akashic-agent/pull/129) 的仓库是 `https://github.com/kachofugetsu09/akashic-agent`，完整 ref 是 `refs/heads/feature/im-phone`；本次核对的 head 为 `83ca96ed70298d507a412fb3416914200acea2de`，tree 为 `954533025d6a18693bd0361db24289439ddfad5a`。该 revision 把 Unicode catalog 语义与 mobile command scope 两条同源分支合成一个 runtime；24 个移动场景映射到 26 个 provider test node，在无网络、源码只读、空 workspace/plugin home 的 Docker Gate 中通过。该 revision 还含 `docker/mobile-lab/`，其 README 与 Compose 把运行数据写入忽略版本控制的 `docker/debug/profiles/mobile-lab/`，不挂载正式 workspace，也不启动 Telegram、QQ 和 proactive。当前设计分支的 `origin/main@6a0616c82267c2045f89539ae3b1b204655f5d57` 不含该目录，所以在 PR #129 合入或准确 commit 被传播前，它只是跨分支候选事实，不能写成 main 已有能力。

**F（历史确定性设备证据）：** 移动端 `a707b9b0b6f3e3630d39a0786b57cd96c4b12c84` 的 debug 变体使用固定 `com.akashic.mobile.debug`。在 Pixel 7 / Android 16 上，排除需要一次性 `pairingOfferBase64/historySessionId` 的 2 个 live Gateway 场景后，64 个本地确定性 instrumentation tests 全部通过；那 2 项没有参数时按设计 fail-loud，不能计为通过或改成假 skip。这组历史结果还发现并修复了 AndroidTest 名称导致的 D8 失败，以及 `snapshotFlow` 在 measure/layout 期间同步 `scrollToItem` 导致的 Compose 重入崩溃。固定 `.debug` 只能隔离 release，不能证明它不会与设备上已有 debug package 冲突，因此这组结果不是当前 run-specific 安装 Gate 的证据。

**F（设备数据事故）：** 在 run-specific Gate 落地前，旧 raw connected task 曾以正式 application ID `com.akashic.mobile` 安装测试候选，覆盖 Pixel 7 上的正式 v0.8.0/code21。维护者后来从预先保存的 `base.apk` 恢复了 APK 与权限，但当时没有 app data 备份，且应用声明 `allowBackup=false`，所以被覆盖的数据无法恢复。这次事故有实际数据损失，不能因后续 binary 恢复或新 Gate 通过而写成“无影响”。

**F（隔离互操作组合证据）：** 当前移动端 PR6 head 为 `3f81275a52b0b87438f5d31041a71997edbac267`，tree 为 `e51f111064dcceef358557f856dcf758f4d08ef1`；这一次 Mobile Lab 设备运行固定核心 commit `f37a42826d9ad5e0988d8b26eba5dd7a20fb29b8` 与 tree `88365c13369b592290fd69918642b7166fc57c55`。run `rpr6live3f81275` 从生成后的 APK 读取 app/test identity 和 instrumentation target，得到 `com.akashic.mobile.review.rpr6live3f81275` 与 `com.akashic.mobile.review.rpr6live3f81275.test`，再以 `pm list packages -u` 得到 `collision_result=clear` 后安装。app APK SHA-256 为 `bc79e1314d61dd90356da919368f3190e496857e32e9eddba2279d3ff0dbe977`，test APK SHA-256 为 `b6629ef4eb23ef831d9430608bafdee664afaff7ddbd0a463831c9206f244c42`。

设备先运行 `pairSendAndReceiveFixedMedia`，显式 force-stop 后再运行 `processRestartResumesWithoutHistoryDuplicates`，两个 instrumentation phase 均通过，最终 `source_state_after_build=verified`、`test_result=passed`、`test_exit=0`、`cleanup_exit=0`、`gate_result=passed`。正式 package 在运行前后都只有 `com.akashic.mobile` v0.8.0/code21，`firstInstallTime/lastUpdateTime` 均为 `2026-07-18 05:16:55`，`ceDataInode=2589746`，本轮没有变化；run-specific app/test package、ADB reverse 与容器均已清理。维护者本机审计 bundle 名为 `mobile-device-gate-f37-3f81275-20260718`，其中 `live-final/` 是隔离 Gateway 根，`live-device-gate/` 保存设备报告；这个名称是本次事实定位，不是协作者必须使用的绝对路径。

**F（最终锁与设备 Gate oracle）：** PR6 的 runtime lock 固定核心 `83ca96ed70298d507a412fb3416914200acea2de` / tree `954533025d6a18693bd0361db24289439ddfad5a`。24 个移动语义场景映射到 26 个 core provider nodes，Docker Gate 通过；完整 Android 单测、AndroidTest 编译、lint 和 assemble 通过。fake-device Gate 覆盖构建期 dirty/HEAD/tree 漂移、app/test collision、app 安装失败、test APK 部分安装失败、0 test、assertion failure、process crash、成功、cleanup failure 和非法 runner argument；源码漂移在首次 ADB 前失败，并且只清理由本进程成功安装的 package。

Pixel 7 / Android 16 的 run `rpr6det3f81275` 在同一最终 head 上逐方法执行全部 64 个非网络 instrumentation，包含 9 条 Room 分叉迁移、session/history/projection reset、本地工作保留、附件、持久通知状态、密钥和 Compose 交互；64 个 phase 都是唯一 `OK (1 test)`，最终 `source_state_after_build=verified`、`cleanup_exit=0`、`gate_result=passed`。这补足当前候选的 L4 数据与交互证据，但不把 Pixel 结果冒充普通贡献者 CI，也不把持久通知状态测试写成真实通知栏展示或 OS 后台投递已经验证。

同一 head 上，run `rpr6zero3f81275` 请求不存在的 `LocalDeliveryStoreTest#methodThatDoesNotExist` 并得到 `OK (0 tests)`，正确以 `gate_result=failed_test`、`cleanup_exit=0` 失败；run `rpr6pass3f81275` 执行真实 `LocalDeliveryStoreTest#finalWithoutMessageIdFailsLoudly`，得到唯一测试开始/成功状态、`OK (1 test)`、`gate_result=passed` 和 `cleanup_exit=0`。两个 run 均有 `source_state_after_build=verified`，临时 package 均已清理。`rpr6live3f81275` 证明 `3f81275 × f37a428` 的实时互操作，Docker Gate 证明 `3f81275 × 83ca96e` 的协议与核心 provider 语义；二者不能合写成同一次 runtime 组合。

**F（当前独立仓库发布候选）：** [移动端 PR #30](https://github.com/kachofugetsu09/akashic-mobile/pull/30) 把 #7～#29 的拆分历史累计到 `1e95222a5c6aacd19dea25ebd0703dca6d3f6f37`，tree 为 `87df3459ee9b8bacfab0c004912986c36f261d96`。候选为 v0.8.5/code26；签名 APK SHA-256 为 `b5550e69ac80d26334be1cebfb1638073fc9bbff8a662d0a713e75a5d1c711bb`，APK Signature Scheme v2 通过，signer certificate SHA-256 仍为 `49bf31ed5c54c642d6f4fdd30a5310a8cb70e67666ad25d711b5f0e084e240bc`。该 release 层只恢复丢失的 Gate、修复确定性回归和收紧可重建缓存边界，不新增主题、布局、会话保留或数据库 schema。

**F（当前协议与 runtime 组合）：** 第二次重拆分曾漏掉旧 PR6 已有的 `protocol/`、`runtime-gate/` 和设备 Gate；当前 head 已恢复这些流程资产。协议快照固定核心 `5615b7df1cbc5092b2f28c9e321ebdf21c16c79a` 的 `schema/archive/mobile-realtime-v1-mobile-pr6.json`，SHA-256 为 `18f8f907c11b66df174699e8ff1d38adb598114e0caf6b25b6823b64cad1fcca`；实际 runtime 固定 `83ca96ed70298d507a412fb3416914200acea2de` / tree `954533025d6a18693bd0361db24289439ddfad5a`。profile `mobile-v0.8.5-release-v1` 把 24 个移动语义场景映射到 26 个 provider node；本机无网络、核心源码只读、tmpfs workspace/plugin home 的 Docker Gate 为 26/26 通过，同一 head 的 GitHub `runtime-contract` 也通过。

恢复场景映射时发现移动端把快捷命令说明上限实现为 Kotlin UTF-16 `length`，与核心 Unicode code point 合同不一致。`2481ad7` 改为按 code point 校验，并以 256 个 supplementary-plane emoji 通过、257 个拒绝以及 `/undo` 文本 round-trip 锁定两端语义；这是一条被 Gate 发现的兼容性修复，不是新命令功能。

**F（当前 Pixel 隔离证据）：** Pixel 7 run `rrel1e95222` 从干净 source commit/tree 构建 `com.akashic.mobile.review.rrel1e95222` 及其 test package，安装前 collision 为空。两个 incoming-share 持久化边界和 schema 9→10 durable stop 迁移各自得到唯一 `OK (1 test)`；最终 `source_state_after_build=verified`、`test_exit=0`、`cleanup_exit=0`、`gate_result=passed`，run-specific package 均已清理。该 run 不连接 Gateway，所以只属于 L4 本地持久化证据，不能替代上面的 Docker runtime 组合或历史 L3 实时互操作。

**F（本次流程偏差）：** 在恢复 run-specific Gate 之前，维护者虽已授权测试发布包，但执行者在无法通过 `run-as` 取得 app data 备份后，仍用 `adb install -r` 把 Pixel 上的 v0.8.0/code21 更新为当时的 v0.8.5/code26 候选。安装、签名和冷启动成功，之后只观察到配对页；由于没有安装前的私有数据库快照，数据影响只能记为 unknown，不能写成“数据未变”，这次安装也不能计入 TST-008 Gate。发现流程冲突后正式 package 不再被设备 Gate 触碰；后续 `rrel1e95222` 只证明以偏差后的 v0.8.5/package 时间为基线时，run-specific Gate 没有再次改变正式包。

**I：** 在控制器支持 run-specific profile 前，复用固定 `mobile-lab` profile 必须先停止旧容器，备份现有测试 profile，并在报告中记录备份、启动时间、core SHA、APK SHA 和 cleanup。任何路径解析到正式 workspace 时立即停止。

Pixel 验收固定遵守：

1. `adb devices` 核对唯一目标和设备序列，不对其他设备执行命令。
2. 为本次 run 生成唯一 app/test application ID；从实际 APK 读取两者和 instrumentation target，不根据 Gradle 配置或文件名猜测。
3. 从干净 source commit/tree 构建，同一 Android worktree 同时只运行一个 Gate；构建后、首次 ADB 前再次核对 clean、HEAD 和 tree。安装前保存正式 package 的 version、signer、install time 和可观察数据身份，并用 `pm list packages -u` 检查 app/test ID。任一 source drift 或 collision 都标记 blocked，不执行设备读取、install、clear 或 uninstall；安装禁止 replace，签名相同、版本较旧和 `adb install -r` 都不是覆盖许可。只有本进程成功安装的 package 才取得清理所有权。
4. 若任务确实需要触碰既有 package，先取得额外授权和经过恢复演练的数据级备份。只备份 `base.apk` 不能恢复 app data；无法备份时 blocked。
5. 只连接 Mobile Lab 域名和测试配对，不连接正式 gateway。需要验证进程恢复的多个 phase 之间显式 force-stop，不能把同一进程里的两个断言称为 restart。
6. 场景至少覆盖进程 kill/restart、Room migration、projection reset、本地未完成工作保留、notification delivery 和附件断点恢复；未覆盖项单独写未验证。
7. instrumentation oracle 核对每个 phase 的实际测试数量、指定方法、开始/成功状态、crash/aborted/assertion failure 标记；shell 退出码 0 或 `OK (0 tests)` 都不能通过。
8. 测试成功不提前写 Gate passed。cleanup 完成后写唯一 `gate_result`；清理失败非零退出并列出残留 package。
9. 保存命令、设备/API 版本、源码和 runtime identity、APK digest、package inventory、collision 结果、安装所有权、正式 package 前后快照、测试 phase、截图或 logcat、Mobile Lab core SHA/run ID/配对来源与 cleanup 结果。测试后移除 run-specific app/test package，停止容器，清理 ADB reverse，并审计残留。

Pixel 证据是当前维护者机器上的手工 L4 结果。CI 没有 Pixel 或 Android 虚拟设备时，PR 必须把 L0～L3 与 L4 分开报告，不能把“本机测过”伪装成每位协作者都能运行的 required check。

## 6. Stacked PR 的传播与评审

每张 PR 只拥有相邻 `base..head` 的新增语义；最终 head 拥有整个 stack 的累计行为。上游修复没有传播到 downstream，就不能说最终产品已修复。

并行评审时，每个 subagent 可以读取同一份固定工作手册和 commit，但每个 writer 必须拥有独立 worktree。主 agent 接受 finding 后再分配修复 owner；交接记录 worktree、branch、HEAD 和 dirty state。没有交接的 agent 不得在共享 worktree 中 commit 或 merge。本次评审中出现的来源不明 merge commit 证明“共享目录里大家都小心”不是可审计协议，必须用独占 writer 和 commit handoff 代替。

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
3. **U：服务端撤销 session 后的本地连续性。** 服务端撤销或删除 session、但客户端仍有 pending/failed/outbox/附件时，是保留服务端历史为只读并附着本地未完成状态，还是删除 server projection 只保留 unresolved local shell，仍未决定。两种答案会改变用户可见历史、session 列表和后续重试目标；客户端不得用当前 Room schema 或候选实现反向定义。
4. **U：破坏性删除与 pending notification。** 用户删除服务端、账户或会话时，尚未交给 Android 的 pending notification 是继续投递、显式取消，还是随目标一起终止，仍未决定。确认前 destructive command 不得隐式 cascade。
5. **U：旧消息编辑。** 核心 `update_message` 应保留原位 UPDATE，还是追加 correction/revision，仍由持久化状态地图跟踪；移动端不得自行定义。
6. **U：功能提案。** failed-message 手动重试、后台大文件续传、全新的文本 composer draft、搜索和 WebView/plugin UI 是独立 feature，不能作为当前修复的附带验收条件。

实现中发现以下任一情况立即停止，不用测试结果替代产品决定：

- 需要删除或覆盖 core SessionDB 正文才能让移动端同步成立。
- projection reset 无法在不删除 local continuity state 的情况下实现。
- 服务端撤销 session 时必须在“保留只读历史”和“只保留本地 unresolved shell”之间作选择。
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
- runtime commit/tree、scenario profile/catalog hash，以及协议 source 与 runtime pin 的关系；
- sandbox 路径策略、workspace/plugin-home 是否为空、正式路径不可达证据；
- 运行的 seam、mutant、L0～L4 结果和未运行理由；
- Room/SQLite/文件 write set，所有已知 schema lineage、最终 schema identity，尤其是 reset、notification 和 attachment offset；
- 每个写入 worktree 的唯一 owner、交接 HEAD 和 dirty state；
- stack 传播关系与最终累计 head；
- 仍为 U 的业务问题，不把它们写成已完成或默认方案。

## 10. 当前实施边界

**F：** 公开 Gate 已有 diff 选择、一次性 Docker sandbox 和报告入口。private companion `cac9582e41de45446374a85d06311f33dc4bad0e` 已为当前 catalog 的 20/20 provider 固定完整远端 branch ref，并把每个 `repository/requestedRef/resolvedCommit` 纳入计划摘要；它只接受 public plan 的 `planned`/`not_affected` 终态，并在 `unmappedChanges` 或 `touchedBaselineGaps` 非空时 fail-loud。只有 Feed/Observe 已有固定 SHA 安装和独立语义 scenario，它们仍是 G2 pilot，不等于统一 controller 或完整 required check。Mobile Lab 的隔离实现只存在于上文固定的 PR #129 revision，不是当前 main 事实。

**F（本次跨仓库审计）：** Gate 开跑前解析并冻结的插件身份如下。`change_source_pr_head` 只说明变更从哪里进入 canonical branch，不能代替 `requested_ref` 当时解析出的 `resolved_sha`。

| repository | requested_ref | resolved_sha | change_source_pr_head |
|---|---|---|---|
| `https://github.com/akashic-plugins/observe` | `refs/heads/main` | `b434fa74b370fafcd0c64129fe1f641f73f0dbcf` | `b7f9d4ecee877d22b5452651d9abf699b2d30b7b` |
| `https://github.com/akashic-plugins/status_commands` | `refs/heads/main` | `cee5bef98e6271c9eb069a6498b4ca072e85c878` | `5c1d4009bee04af271627819fd5731e1978b5dfe` |
| `https://github.com/akashic-plugins/proactive_feedback` | `refs/heads/main` | `d5227249f5ad195ab7693ae8c72690ee7db32e28` | `not_applicable`（直接进入 main 的 commit） |
| `https://github.com/akashic-plugins/feed-mcp` | `refs/heads/master` | `520ba10032089b1e056a9eecc5f2c1f459c75e5c` | `334276c4e972f1d80b0a353605d068abc5135b18` |

Observe 的移动端 turn identity seam、status_commands 的公开 session lookup、proactive_feedback 的 mobile UI v2 和 Feed freshness 都被核心移动协议变化触达，且修复位于各自 canonical plugin repository。移动端协议归档固定在核心 `5615b7df1cbc5092b2f28c9e321ebdf21c16c79a` 的 `schema/archive/mobile-realtime-v1-mobile-pr6.json`，SHA-256 为 `18f8f907c11b66df174699e8ff1d38adb598114e0caf6b25b6823b64cad1fcca`；实际插件 Gate runtime 另固定为上文 `83ca96e`。这组记录说明插件影响必须由远端 revision 和真实 seam 证明，不能只搜主仓库 import。

**F（当前公开插件发布锁）：** [核心 PR #135](https://github.com/kachofugetsu09/akashic-agent/pull/135) 的 head 为 `a196915f`，已把 Observe 更新到 `b434fa74`、status_commands 更新到 `cee5bef9`，并继续固定 proactive_feedback `d5227249`、fitbit-mcp `b95ec281` 与 emotion `0924217e`。干净 core head 上的插件合同 Gate 运行五个固定远端 checkout，共 31/31 个移动 UI 行为测试通过；GitHub `mobile-plugin-contract-gate` 通过。它锁定公开移动插件 ABI，仍不等于本节下方尚未完成的 20-provider private G2 终态。

**F（Gate 反向发现）：** 同一份 private plan 在 Feed `master@e1aa198` 上发现 lifespan poller 与首个 MCP 调用并发初始化 SQLite 时重复 `ADD COLUMN interest_scored_at`。这不是 freshness mutant 的预期失败，因此 Gate 拒绝通过。Feed PR [#2](https://github.com/akashic-plugins/feed-mcp/pull/2) 把 schema 初始化放进 `BEGIN IMMEDIATE` 并用旧 schema 并发测试覆盖；上表固定的 canonical revision 重跑后，background refresh、call finality、restart persistence 通过，两个 freshness mutant 都因目标不变量被杀死。

**F（固定组合结果）：** 这两份正式 G2 pilot 都消费同一份 public plan：base `5615b7d`、head `f37a428`、`sourceDigest=4b6b7d432c8ea7006038cb1f114ce46c22d4b0d79d9a0f6ba8d64ca59837d54f`、`publicPlanDigest=830860d642b56188a0b8e57093e7fd0080c2f9c9cec5ae56479d6d33488bd6bf`，以及 949 项、SHA-256 `5f82ccef0c3f3f1e89b8a4fc25a37e1548ca48fe6ff49d6f32842041b6e2cb90` 的显式 public inventory。它们不能改写成 merge commit `83ca96e` 上的同一次运行。

Feed 报告固定 `master@520ba100` 和 `feed_freshness` profile SHA-256 `86018e7225c36d47112a8fe64ad26c001eddd06c6cac70c438192100e3c9b4bb`；background refresh、call finality 与 restart persistence 通过，`async_accepted_before_visible` 和 `poller_stopped_stale_payload` 被杀死，报告 SHA-256 为 `c4f7b10f3afac0f1f7d450c98dd3a46c4e1b64ad1ae79c5d767687b56b66ae7f`。Observe 报告固定 `main@b434fa74` 和 `observe_turn_identity` profile SHA-256 `77a9e4740033c636978d4be303ece41ddd5390d22daf17a0e7ba7ef0fada672c`；五个 message identity observation 通过，`missing_assistant_message_id` 被杀死，报告 SHA-256 为 `febaf022f39b0fe63eb23377d8d61e13008780abb59b122095d5b14b4ec51431`。

status_commands `cee5bef` 的 9 个 Python/7 个 Node 测试和 proactive_feedback `d522724` 的 13 个 Python/5 个 Node 测试只是各自仓库的原生验证，不是上述 public plan 的组合报告。当前 plan 因 plugin group 的依赖闭包选中 20 个 provider，这证明它们需要得到终态，不证明它们全都受到同一种语义影响；除 Feed/Observe 外的 18 个 provider 仍缺少 `passed`、`failed` 或 `not_affected` 结果。完整 G2 因此仍按 `NOW.md` 保持未完成。

**I：** 下一步把现有 Feed 与 Observe remote-revision 场景接入同一个 G2 Docker controller，再为其余会被 private plan 选中的 provider 补独立语义 scenario 和可执行结果，并建立始终返回 `passed`、`failed` 或 `not_affected` 的外部状态。完整 ref 已固定，但 ref identity 不能代替语义 oracle；完成前不能把“本机脚本可运行”或两个 provider pilot 描述成完整 required Gate。

**U：** G2 外部状态由哪一个受保护 runner/仓库发布，以及 Pixel 手工证据是否作为 release promotion 的必需项，仍取决于仓库权限和发布策略；它们不影响本设计中的数据保护和版本固定合同。
