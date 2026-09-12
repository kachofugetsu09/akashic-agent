# Akasha 宽 scope 分区设计：版本化拓扑 + 声明式重放

- 状态：proposal（对应 GitHub Issue #629；泛化 #370）
- 日期：2026-09-12
- 关联：[0006](../decisions/0006-akasha-v2-is-the-canonical-explicit-memory-engine.md)、[Akasha V2 在线与重放](../design/akasha-v2-runtime-migration.md)、[持久化状态地图](../design/persistence-state-map.md)、[未来方向与 Issue 拆分草案](../design/akashic-future-roadmap-issue-drafts.md)

## 1. 背景与目标

Akasha V2 当前只有一张 companion 图：`sessions.db` 中所有合法 turn 经稀疏索引进同一个 `MemoryCycle`。`session_key` 只负责 burst 连续性与因果序，不构成记忆分区。Issue 3 草案（#370）把分区定义为单字段 `project_id`，不足以表达调用方未来要引入的维度组合（project、devspace、worktree、普通 session，以及插件后装的未知维度）。

本设计把分区身份泛化为**宽 scope key**：一个有序字符串元组，分量数量可变。Akasha 对它完全不感知语义——不解析、不校验分量含义，只当 opaque 身份使用。

目标语义：

- 纯分区：每个 scope_key 对应一张完全独立的图 + 稀疏索引；`(a,b)` 与 `(c,b)` 共享分量也不互相命中。
- 同一 scope 下多个 session 共享一张图，burst 仍按 `session_key` 隔离。
- 维度可以后增、可以退役，由显式拓扑操作完成，绝不静默漂移。
- 无 key / 空 key → companion scope，即现有 `memory/akasha.db` 与 `memory/akasha-v2-index.db`，零迁移。

## 2. 设计原则（生产级分区基础设施的对应）

调研 Kafka、Pulsar、Couchbase、Vitess、FoundationDB 的分区机制后，取三条共识：

1. **身份与路由分离，中间插一张带版本的映射图。** Couchbase 的 vBucket map、Pulsar 的 bundle map、FDB 的 directory layer 都把"逻辑名 → 物理位置"做成可版本化的间接层。Kafka 的 `hash(key) % N` 把路由编码进函数，扩容即破坏同 key 有序性且官方不搬旧数据——这是反面教材：路由不得是隐式函数结果。
2. **改分区规则是显式 workflow。** Vitess `Reshard` 的 `copy → VDiff → SwitchTraffic → complete`、Ceph cluster map epoch 表明：拓扑变更必须有版本、有校验、有原子切换，不是后台漂移。
3. **Akasha 的独特优势。** 图是 `sessions.db` 的确定性派生物，"搬数据"只是按新路由重放投影——代价远低于中间件的物理迁移。本设计借鉴的是中间件的**控制面**（版本化映射、显式 workflow），数据面沿用"派生 sidecar + 确定性重放"的既有合同。

## 3. 术语与模型

```text
scope_facts      sessions.metadata 中的 namespaced dict：
                 {"project":"p1","devspace":"d2",...}
                 每个命名空间由一个 contributor 独占写入

contributor      注册的维度贡献者；声明稳定 order（rank + 名称 tie-break），
                 读取自己的 facts 命名空间，输出 0 或 1 个分量

resolver         版本化的 contributor 有序列表；纯函数
                 resolve(session_facts) → scope_key 元组
                 contributor 缺 facts 时分量省略，元组变长合法

scope_key        canonical 编码的不透明身份：
                 "v1:" + compact JSON array，元组有序，(a,b) ≠ (b,a)

topology ledger  append-only 的版本化映射图：
                 generation 单调递增；记录 resolver 组成与
                 scope_key → {dir, state, born_at_generation}
```

`scope_facts` 是事实，`scope_key` 是 resolver 的纯函数输出，**不持久化在 session 上**。sidecar 的 index metadata 记录 `built_under_generation`（等价 FDB directory 的 layer 校验），这是重放确定性的锚点。

## 4. 存储布局

```text
memory/
├── akasha.db + akasha-v2-index.db          # companion scope（现状，零迁移）
└── akasha-scopes/
    └── <sha256(scope_key)>/
        ├── manifest.json                   # 元组原文、算法版本、创建时间、topology generation
        ├── akasha.db
        └── akasha-v2-index.db
```

manifest 存原始元组供审计（hash 不可逆）；Core 侧另有 `scope-topology` ledger（见第 7 节持久化合同）。每 scope 一个 `OnlineMemoryRuntime`：独立 `MemoryCycle`、单写者 commit gate、独立因果序；`session_key` 在 scope 内部继续做 burst 隔离。

## 5. 顺序性：新增与卸载过程中的拓扑生命周期

### 5.1 contributor 两阶段生命周期

```text
declared ──(activation: generation+1 + reshard)──► active ──(显式退役)──► retired
```

- **declared**：新 contributor 已注册但**不进入 resolver**。此期间插件可为存量 session 回填 `scope_facts`，在线读写完全不变。
- **activation**：一次显式拓扑操作把 contributor 纳入 resolver，触发 reshard workflow——所有受影响 session **一次性**搬迁，而不是随 facts 到达逐个漂移。这消除了"回填期间拓扑持续抖动"的问题。
- **retired**：维度退役是独立的显式操作（又一次 generation + reshard），与插件卸载解耦。

### 5.2 卸载 ≠ 退役

插件卸载只移除 fact-writer 能力；contributor 注册是 topology ledger 里的**数据**，不随代码消失。卸载后：

- 存量 `scope_facts` 照常解析，scope key 不变，图照常服务；
- 新 session 不再获得该维度分量，维度进入"冻结"状态；
- 想让维度真正消失，必须走显式退役 + reshard。

例：resolver `[project, devspace, experiment]`，S1 facts `{project:akasic, devspace:d2, experiment:e5}` → `("project:akasic","devspace:d2","experiment:e5")`。卸载 experiment 插件后 S1 的 key 与图都不变；显式退役 experiment 后，S1 解析为 `("project:akasic","devspace:d2")`，与只带两个 facts 的 S2 **合并**进同一 scope——退役是合并不是删除，diff 预览必须人工确认。

### 5.3 在线路径的顺序保证

- topology ledger 单写者，注册/激活/退役以 CAS on generation 串行化；并发冲突败者重试。
- staged commit 携带 generation；publish 前发现 generation 已推进 → 拒绝并按新归属重走（等价客户端拿旧 vBucket map 写被拒）。
- reshard 的 diff + rebuild 在固定 sessions.db 读快照上计算；期间冻结受影响命名空间的 fact 写入（短维护窗口，写方拿到明确"拓扑变更中"错误而非静默排队）。
- 崩溃恢复序：ledger 先提交 generation；scope 打开时校验 `built_under_generation == ledger.generation`，落后则确定性 reconcile（从 source 重建）再服务。任一环节失败不产生"半个新拓扑"。
- 合并 scope 内因果序仍按 `(committed_at, session_key, user_seq, turn_id)`；不同 session 的 turn 按真实提交时间交错，确定性可重放。

## 6. Reshard workflow

```text
1. declare   拓扑 generation+1（contributor 激活/退役/序变更）
2. diff      membership diff：哪些 session 的 resolved key 变了 →
             新增/失去成员的 scope 清单；失败即中止，不动现存 sidecar
3. rebuild   受影响 scope 各自从 sessions.db 确定性重放；
             新 scope 的增量 build 按新归属自然纳入 session 全部历史
4. cutover   原子发布 sidecar（os.replace staging→正式）+ 提交 ledger generation
5. retain    失去全部成员的旧 scope 标 tombstoned，文件保留可读
```

workflow 的每一步写 ledger 回执；中断后按 ledger 与 sidecar 的 generation 对账恢复，不猜测现场。

## 7. 对话结构与写入接缝

- `sessions.metadata.scope_facts`：Session 创建时可带初始 facts；此后各 contributor 经**窄接缝**原位更新自己的命名空间（不得覆写他人），普通 turn 路径不持有该写入权限。
- 插件贡献维度的唯一方式：注册 contributor + 经窄接缝写自己命名空间的 facts。不开放任意 SQL、不开放直接改 session 行。
- 会话删除沿用现有级联；scope 图不随单 session 删除自动减少（与"派生重建"协议一致）。
- `MemoryQuery` / `TurnCommitted` 携带 resolved `scope_key`（缺省 companion）；adapter 侧按 `(session_key, generation)` 缓存解析结果。

## 8. 引擎边界与改动点

| 位置 | 改动 |
|---|---|
| `core/memory/engine.py`、`core/memory/events.py` | `MemoryQuery`/`TurnCommitted` 增加 `scope_key` 可选字段 |
| Core 新对象 | scope-topology ledger、resolver、contributor registry、session-metadata facts 窄接缝、`session_scope_map` 物化路由表 |
| `plugins/akasha/` adapter | ScopeRegistry：resolve→runtime 惰性打开，活到 shutdown；ticket 按 `(scope_key, session_key)` 保存；staged commit 带 generation fence |
| `infrastructure/sparse_index/builder.py`、`loader.py` | build 按 `session_scope_map` 过滤本 scope 成员；companion 只纳入无 facts 解析结果的 session |
| `projection.py`、`learning.py` | 构造 turn 前经路由表取 session 的 scope，路由到对应 index/graph |
| Inspector / Dashboard / Mobile | API 加 scope 参数，默认 companion；可枚举 ledger 中已知 scope |
| `scripts/build_akasha_db.py`、`application/rebuild.py` | 支持按 scope/按 generation 重建；输出 membership diff 报告 |
| upstream `akasha-v2-engine` | **零改动**（决策 0006 镜像约束） |

## 9. 性能与库优化

- **物化路由表 `session_scope_map(session_key, scope_key, generation)`**：resolver 结果按 generation 物化，增量 build 只做 `WHERE session_key IN (本 scope 成员)` + seq 高水位，不为每次 commit 全量解析所有 session metadata——路由是查表不是扫描。
- **embedding 零重算**：`message_embeddings` 留在 sessions.db 共享；reshard 与重建只重放索引与图，零 LLM/embedding 调用。
- 每 scope SQLite：WAL、`synchronous=NORMAL`、busy_timeout、单写者 lease；跨 scope 无锁竞争、可并行 commit；单 scope 损坏 fail-loud 不波及其他。
- 读路径：`(session_key, generation) → scope` 内存缓存，generation 推进即失效；runtime 惰性打开。
- scope 基数是用户可见成本：ledger 可枚举可审计；v1 不做 LRU/offload/GC（Weaviate 式 active/inactive/offloaded 列为未来逃生口）。

## 10. 持久化增改减合同

| 对象 | 正常增加 | 允许原位/逻辑变化 | 物理减少 |
|---|---|---|---|
| `scope_facts` | contributor 写自己命名空间 | 各 contributor 原位更新自己的 key，不覆写他人 | 只随显式数据管理操作 |
| topology ledger | 每次拓扑操作追加 generation 回执 | generation 记录不可变 | 不减少，是审计证据 |
| `session_scope_map` | 拓扑操作按快照物化 | 随 generation 整体重建替换 | 只随拓扑操作替换 |
| scope manifest | 首次打开原子创建 | immutable identity | 不自动减少 |
| graph/index sidecar | `MemoryCycle.commit` 演进 | 算法状态机内更新 | 只随显式重建或 interaction 撤销协调替换 |
| tombstoned scope | — | ledger 标 tombstoned | 独立命名操作，需备份与引用扫描 |

## 11. 非目标

- 不做跨 scope 联合检索；要并集由宿主 fan-out 多次查询，图不合并。
- 不做 scope LRU/offload/GC、自动 repair。
- 不动 upstream `MemoryCycle`、burst、权重、阈值。
- 退役 `memory2.db` 归档不参与 scoping。
- 不定义分量语义（project/devspace/worktree 的合法性由调用方与插件负责）。

## 12. 验收要点

- 不同形态 key（单分量/多分量/纯 session）并存，各 scope 文件与 runtime 互不影响；向 A 提交不改 B/companion 的 canonical 快照。
- 维度新增：declared 期 resolver 不变；activation 后一次性搬迁，membership diff 与实际重建逐项相等。
- 维度退役：合并语义经 diff 预览确认；旧 scope tombstoned 可读；in-flight commit 被 generation fence 正确拒绝。
- 卸载插件（不退役维度）：零拓扑变化，存量 scope 正常读写。
- 同 sessions.db 全量重放：每 scope canonical logical state 与在线一致；两个 `PYTHONHASHSEED` 一致。
- reshard 中途崩溃：ledger 未提交 → 旧 generation 照常服务；已提交 → 各 scope 打开时按 generation 对齐 reconcile，无半拓扑状态。
- upstream 镜像逐字节不变。
