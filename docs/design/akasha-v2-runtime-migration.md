# Akasha V2 在线运行与确定性重放设计

- 状态：implemented；正式 workspace 尚未部署
- 日期：2026-07-27
- 决策：[0006](../decisions/0006-akasha-v2-is-the-canonical-explicit-memory-engine.md)
- 需求：MEM-009、SES-003、GOV-005、TST-002、TST-005

## 1. 目标与边界

本迁移把独立 `akasha-v2-engine` 接入 Akasic Agent，保留宿主的 MemoryPlugin、
自动上下文和 `recall_memory` 接口。完成后的可观察行为是：

- 每一轮合法 user/assistant turn 只在消息持久化成功后学习一次；
- 当前 query 可以看到历史 dense 精确命中和显式图的模式补全；
- 显式 recall 本身不学习、不改变待提交上下文；
- 在线逐轮增长与对同一份历史做全量 replay 得到相同逻辑状态；
- 重建、Docker 和报告都使用独立 workspace，不读写正式派生库。

这次不部署正式 workspace，不迁移旧 Akasha Graph，也不保留旧 fast/slow、reinforce
或可写检查器兼容层。桌面端和移动端重新提供面向 V2 schema 的只读 Inspector；旧配置
和旧 sidecar 的正式切换必须作为独立数据迁移执行。

## 2. 状态所有权

| 对象 | 性质 | 正常增加 | 允许原位更新 | 减少与恢复 |
|---|---|---|---|---|
| `sessions.db/messages` | 对话事实 | 宿主提交 user、assistant、tool 消息 | 本迁移不允许 | 只有用户数据管理操作可减少；SQLite backup 恢复 |
| `message_embeddings` | 固定重建输入 | turn commit 对真实落库文本计算并 upsert | 同 message/model 内容变化时由拥有者更新 | 完整重建不得补算；独立 embedding 迁移可在快照中补齐 |
| `akasha-v2-index.db` | 因果稀疏索引 | 每个 committed turn 增加 dense 指针、BM25 和时序统计 | 在线统计增量维护 | 可从固定 sessions 输入重建 |
| `akasha.db` | 派生显式记忆 | 每个 commit 增加/更新 hub、关系、激活与遗忘状态 | `MemoryCycle` 原子发布全状态 | 部署前备份；可由稀疏索引重放恢复 |
| pending retrieval ticket | 进程内临时状态 | 自动 context query 产生 | 同 session 新 context 可替换 | commit 消费；scheduler/skip 清除；进程退出可丢弃 |
| recall trace/report | 诊断证据 | query 或重建生成 | 不参与图状态 | 按独立诊断 retention 管理 |
| Inspector projection | 只读派生视图 | 每次请求从两个 V2 sidecar 重建 | 只缓存按文件签名失效的 dense 矩阵 | 不保存、无删除权限 |

`sessions.db` 是唯一历史事实源。Akasha 不保存一份可独立修改的消息正文，也不因检索、
遗忘或图清理删除原始消息。

## 3. 组件结构

```text
┌──────────────────────── Akasic Agent ────────────────────────┐
│                                                             │
│  Prompt context ── intent=context/effect=stateful ─┐         │
│                                                    ▼         │
│  recall_memory ── intent=answer/effect=read_only ─► query    │
│                                                    │         │
│  SessionStore ── INSERT user/assistant ──► TurnCommitted     │
│                                                    │         │
│                                                    ▼         │
│                       plugins/akasha/MemoryPlugin adapter     │
│                                                             │
│  Dashboard / Mobile ── read-only ──► Akasha Inspector       │
└────────────────────────────────┬────────────────────────────┘
                                 │ byte-identical mirror
                                 ▼
┌────────────────────── akasha-v2-engine ──────────────────────┐
│ SparseIndexBuilder → MemoryCycle.retrieve → RetrievalTicket  │
│                                      │                       │
│ TurnCommitted → MemoryCycle.commit ──┴→ graph/plasticity     │
│                                      │                       │
│                              deterministic persistence       │
└──────────────────────────────────────────────────────────────┘
```

宿主适配器只做五件事：读取宿主配置、调用 embedding provider、解析宿主事件、把
`MemoryRecord` 渲染成上下文、把 engine lifecycle 注册回宿主。稀疏特征、扩散、
连接预算、衰减、抑制和重放顺序都由 upstream 拥有。

## 4. 一轮在线闭环

### 4.1 自动上下文查询

1. 宿主在模型调用前构造 `MemoryQuery(intent="context", effect="stateful")`。
2. adapter 对 query 计算 dense，构造不属于历史的 cue turn。
3. `MemoryCycle.retrieve` 从 dense、BM25、burst/时序和已有关系形成 seed，再做局部
   residual diffusion，返回直接证据、模式补全和 retrieval ticket。
4. adapter 只为有稳定 `session_key` 的 context 查询保存 ticket。
5. 输出分成两条 lane：
   - 左脑：按 dense 分数选最多五个稳定 turn；
   - 右脑：按 completion 证据选候选，去掉已在左脑出现的 turn。
6. 选取完成后，每条 lane 按 turn 时间从近到远显示，不把算法分数误当时间顺序。

### 4.2 持久化与学习

1. 宿主先把 user 和 assistant 精确正文写入 `sessions.db`。
2. `TurnCommitted` 必须携带两个稳定 message ID；缺少 ID 直接失败。
3. adapter 对精确落库正文生成并保存 embedding。若 pending query 文本与落库 user
   正文一致，复用 query dense，只计算 assistant；否则两条正文重新 batch embed。
4. pending ticket 仍匹配时交给 `MemoryCycle.commit`；不存在或不匹配时在最新图状态上
   重新 retrieve 后提交。
5. commit 将当前 turn 加入稀疏索引和图，执行 Hebbian 关联、时序方向、连接预算、
   衰减与激活恢复，再原子发布 sidecar。

这个顺序保证“检索影响回答，已完成回答影响未来检索”，不会让未落库或失败的 turn
提前进入图。

### 4.3 显式 recall

`recall_memory` 构造 `intent="answer", effect="read_only"`。它可以读取同一张图并
返回记录，但 adapter 不保存 ticket，`MemoryCycle` 不 commit，已有 context pending
也保持对象身份不变。因此模型在一轮里是否调用 recall 只改变收到的上下文，不改变
图状态。

### 4.4 排除路径

- `session_key` 前缀为 `scheduler` 的 turn 不查询学习状态，也不进入 replay。
- user 或 assistant 的持久 metadata 含 `skip_post_memory` 时，同样排除。
- 中断占位 turn 不进入学习样本。新数据通过 assistant 的 `skip_post_memory` 表达；
  历史数据仅把 assistant 正文精确等于 `[interrupted]` 的 pair 视为中断。
- user 与 assistant 都没有文本的纯媒体 turn 不进入索引。
- 单边文本为空时，只要求非空一侧有 embedding；不得为纯空内容生成假向量。

## 5. 稀疏编码与模式补全

一个历史 turn 是稳定原子节点。它携带原始 message 指针、user/assistant dense、
增量 BM25 词项、时间与 burst 统计。检索先形成非负 seed，再进入同一 `MemoryCycle`：

```text
dense / BM25 / burst / temporal evidence
                    │
                    ▼
              sparse seed mass
                    │
                    ▼
        residual diffusion with restart
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   settled lower bound    remaining residual
         │                     │
         └──── residual ≤ tolerance ────► stop
```

每次 commit 用已激活模式形成稀疏 hub membership，不展开全连接 clique。短时间内的
正向关系强于反向关系；同一连接预算内，反复共同激活的关系获得份额，未共同激活关系
相对受抑制。遗忘按事件时间与重复激活共同决定，而不是只按数据库年龄单调删除。

在线与 replay 均按 `(timestamp, session_key bytes, user_seq, turn_id bytes)` 建立全局
因果顺序。所有 set/dict 到持久化输出前都有稳定排序；重放进程固定 BLAS 线程数，
不同 `PYTHONHASHSEED` 不得改变 canonical logical state。

## 6. 重建合同

`scripts/build_akasha_db.py` 只负责把宿主入口转发给 upstream CLI。完整重建分四段：

```text
只读打开 sessions.db
        │
        ▼
分类完成与中断 turn
        │
        ├─ 中断/显式跳过 ─► 保留原消息，不要求 embedding
        ▼
审计所有 eligible message embeddings
        │
        ├─ 缺失/损坏 ─► 写 JSON 缺口报告，fail-loud
        │                 不备份、不创建目标 DB
        ▼
临时目录构建 sparse index
        │
        ▼
同一个 MemoryCycle 全量 replay
        │
        ▼
备份旧 sidecar → staging 完整写入 → 原子替换 → 输出 run report
```

重建不调用 embedding provider。若历史确有缺口，必须由独立 embedding 迁移在源库副本
或经明确授权的正式库中补齐，之后重新执行严格重建。这样可以区分“固定输入重放”和
“改变固定输入”两种不同权限。

## 7. 在线与 replay 等价

比较对象不是 SQLite 文件字节，而是排除运行诊断时间、文件布局和查询 trace 后的
canonical logical state：

- turn 与 source message 身份；
- hub membership 与强度；
- 有向关系、连接预算和遗忘状态；
- context/burst 状态；
- 已提交 activation/plasticity 结果。

最小等价场景按下面顺序执行：

1. 空隔离 workspace 启动真实宿主 plugin。
2. context query 产生 pending。
3. 通过正式 session API 持久化 user/assistant。
4. 发出 `TurnCommitted`，确认状态版本增加。
5. 产生第二个 context pending。
6. 通过真实 `RecallMemoryTool` 调用 read-only recall，确认 sidecar logical hash 和
   pending 对象不变。
7. 提交第二轮。
8. 从同一 `sessions.db` 的干净索引全量 replay。
9. 比较 online 与 replay canonical logical hash。

全量 Gate 再对真实历史快照使用两个不同 `PYTHONHASHSEED` 重放，要求逻辑摘要一致。

## 8. Docker 与正式 workspace 隔离

Docker Gate 的 source、config、workspace、HOME、socket 和报告都位于唯一 `/tmp`
sandbox。正式 workspace 只在宿主侧通过 SQLite online backup 生成快照；容器不挂载
正式路径。Gate 前后记录以下正式文件证据：

- `sessions.db` SHA256、size、mtime；
- `memory/akasha.db` SHA256、size、mtime。

容器中的 scripted chat provider 只控制 Agent 回复与工具选择。Akasha embedding 使用
明确配置的真实 debug embedding endpoint；凭据只写权限为 `0600` 的临时配置，不进入
仓库、日志或报告。若 endpoint 或凭据不可用，Docker Gate 返回 blocked/failed，不能
用固定假向量声称在线路径通过。

Docker 场景必须观察：

- 第一次 user 输入进入 provider，assistant 落库，图状态版本增加；
- 第二次自动 context 确实进入 provider payload；
- provider 调用 `recall_memory`，工具返回且 sidecar 状态不变；
- 工具结果后的 assistant 正常提交并影响下一轮图；
- 容器停止后从其 `sessions.db` replay，逻辑状态与在线库一致；
- Compose 无残留容器，仓库与正式 workspace 摘要不变。

## 9. 输出合同

自动上下文沿用现有认知命名，不给每条记录标“语义”或“联想”：

```text
# Akasha memory now=07-06

## 左脑记忆：精确回忆
[07-06] user="..." assistant="最多 50 字..."

## 右脑联想：潜意识第一反应
[06-28] user="..." assistant="最多 50 字..."
```

两条 lane 先按各自证据选取，再按时间展示。去重使用稳定 turn ID，不用截断文本或
字符串相等；同一 turn 即使两侧分数不同也只出现一次。

### 9.1 Inspector 合同

Inspector 不是第二套检索算法，也不拥有任何记忆状态：

```text
memory_events ───────────────► 检索轮次列表
event_seeds ─────────────────► 直接线索
activation_runs/items ───────► 可选逐节点扩散路径
recall_runs/items ───────────► 显式模式补全
turn_dense + prior-only dot ─► 左脑 dense top 5
                               │
                               ▼
                     与运行时相同的去重、时间排序
                               │
                               ▼
                     实际 Prompt 记忆块预览
```

`activation_runs/items` 只在请求 capture 的重放目标上持久化，普通在线轮次可能没有
逐节点路径。此时 Inspector 仍能从 `recall_items` 展示最终补全，但必须把指标标成
“补全候选”，隐藏“扩散激活”明细，不能把没有 capture 误写成没有扩散。

桌面端通过插件 Dashboard 注册三个只读端点：overview、分页检索轮次和单轮详情。
移动端复用宿主通用 plugin UI 协议：当前 assistant 回复前显示本轮左右脑召回，
导航页显示最近检索并按需读取详情。两端都不暴露图快照、任意 SQL、reinforce 或写入
RPC；SQLite 连接使用 read-only URI 与 `query_only`。assistant 预览保持最多 50 字，
Dense 与显式补全按稳定 turn ID 去重后，分别按时间从近到远显示。

`dashboard_panel_inspector.ts` 属于 upstream 镜像；宿主构建生成的同名 `.js` 是被
Git 忽略的派生产物。镜像校验只排除这个明确命名的构建产物和 `UPSTREAM.json`，其他
意外文件仍会让完整文件集合校验失败。

## 10. 失败、回滚与部署前置

- upstream 镜像校验失败：停止构建，不从宿主镜像继续开发。
- embedding audit 失败：保留报告，不触碰现存 sidecar。
- replay 中断：目标 staging 不发布；现存 sidecar 的时间戳备份可恢复。
- online commit 失败：异常上抛，不能把空结果当成功；原始 session 消息仍保留。
- 正式部署前：备份旧 `config.local.toml` 和 `akasha.db`，在只读快照完成全量 Gate，
  再把规范 V2 config 与已验证 sidecar 作为同一个维护操作原子切换。
- 回滚时同时恢复旧插件代码、旧 config 和旧 sidecar；不能只切代码后继续打开另一代
  schema。

本分支只交付代码、隔离证据和迁移设计，不修改正式 workspace，也不把 PR 合入运行中
分支。

## 11. 验收命令

```bash
# upstream
pytest -q
python scripts/check_akasic_contract.py
python scripts/check_akasic_behavior.py

# host mirror and tests
.venv/bin/python scripts/check_akasha_v2_mirror.py \
  --upstream /mnt/data/coding/akasha-v2-engine
.venv/bin/python -m pytest -q
npm run typecheck
npm run lint
npm run build:dashboard
node --test tests/test_akasha_mobile_ui.mjs

# strict isolated replay
PYTHONHASHSEED=1 .venv/bin/python scripts/build_akasha_db.py \
  --sessions-db /tmp/<run>/sessions.db \
  --db-path /tmp/<run>/akasha-seed1.db \
  --embedding-model text-embedding-v4 \
  --embedding-dim 1024 \
  --require-complete-embeddings

PYTHONHASHSEED=987654321 .venv/bin/python scripts/build_akasha_db.py \
  --sessions-db /tmp/<run>/sessions.db \
  --db-path /tmp/<run>/akasha-seed2.db \
  --embedding-model text-embedding-v4 \
  --embedding-dim 1024 \
  --require-complete-embeddings

# public runtime gate
python docker/debug/gate.py run --base origin/main
```

## 12. 2026-07-27 隔离验收结果

正式 workspace 在 Gate 前后保持相同 SHA256、size 和 mtime。隔离 SQLite backup 中
共有 4800 个相邻 user/assistant pair，其中 6 个 assistant 是精确的
`[interrupted]` 占位。重放把它们分类为未完成 turn 后，得到 4794 个学习样本和
9556 条必须存在的 message embedding；全部 9556 条有效，没有补算向量，也未修改
正式 `sessions.db`。

全量重放得到 4634 个 hub 和 21254 条关系，单次约 157～216 秒。两种
`PYTHONHASHSEED` 使用相同 11 个冻结 query 产生相同 canonical logical state；
本次 SQLite 文件摘要也相同，但物理页摘要不作为语义等价判据。每个 query 的显式
召回数依次为：

| user seq | 3011 | 4740 | 5294 | 7877 | 8464 | 8566 | 9224 | 9624 | 9710 | 9892 | 10306 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recall turns | 6 | 18 | 8 | 15 | 9 | 20 | 5 | 15 | 15 | 27 | 11 |

所有 query 都低于既定的 40 条上下文舒适范围；详细私有正文只保存在调用者选择的
隔离报告中，不提交到公开仓库。与修复前逐项比较，11 个 query 的召回 turn 集合
没有增加或减少；变化只发生在 6 个中断 turn 及其派生 hub 和关系。

Docker Gate `AKV2-01` 至 `AKV2-06` 全部通过：第二轮 provider payload 确认包含自动
Akasha 上下文；`recall_memory` 调用前后 logical state 相同；第二轮持久化后状态改变；
停机 replay 与 online logical state 相同；Compose 无残留；正式 workspace 与仓库摘要
不变。
