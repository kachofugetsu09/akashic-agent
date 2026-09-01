# Compaction 与 Markdown 记忆普通插件化任务合同

- 状态：verified
- 日期：2026-08-31
- base：`51f50ad58be41106def8fb30662f1ceb6ecc563d`
- change_type：architecture + approved semantic delta
- capability_owner：`compaction` plugin、`markdown-memory` plugin、Session owner 的窄原子端口
- consumer_scope：provider 请求、Session ledger、Prompt、Wake、Subagent、QQ、Mobile inspection、Dashboard、Akasha
- runtime_patch：需要；补充来源无关 provider request seam、Session compaction port 与精确 workspace file grant
- client_only_alternative：无；权威 checkpoint 与 Markdown 文件都在 runtime workspace

## 1. 目标与完成标准

Core 不再构造或持有 compaction/Markdown 私有 runtime。两个能力由普通 v3 插件提供，内置只
代表发行方式。`PENDING.md` 与 optimizer 删除，committed included checkpoint 直接、幂等地
更新 `MEMORY.md` 和 `SELF.md`。依赖清单逐项对比后，除已批准语义差异外无能力退化。

完成需要：相关测试、差分 E2E、change Gate、三次独立 Terra xhigh 概念 Review 全部没有
must-fix，并提交 draft PR。

## 2. 状态合同

| 对象 | 正常增加/更新 | 逻辑失效 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| `sessions.db/messages` | 正常 Turn 只 INSERT | 仅显式用户撤销协议 | 仅 SES-003 管理动作 | Session owner；DB backup 与 audit |
| `session_compactions` | 插件经窄端口 INSERT generation 并推进 cursor | 显式 source 撤销使 generation/descendant 失效 | session 显式删除才 cascade | Session owner；ledger lineage、prepare、receipt |
| `session_compaction_prepares` | checkpoint saga 开始时 INSERT，提交/确定性恢复时清除 | 无 | 仅 owner 完成协议 | Session owner；source_ref + incarnation |
| `MEMORY.md` / `SELF.md` | Markdown 插件按 kind 原子替换 | 新版本 supersede 旧内容 | 不自动删除 | Markdown 插件；历史 backup + applied receipt |
| 旧 `PENDING.md` | 在线路径不再增加 | 一次迁移成功后归档 | 本任务不静默删除非空文件 | Markdown 插件迁移；迁移 backup + receipt |
| 旧 `consolidation_writes.db` | 新在线路径不再追加旧 pending receipt | 仅作为旧 saga/audit | 本任务不删除 | legacy recovery；原 DB + backup |
| plugin profile receipt | 每个 `source_ref + memory/self kind` INSERT immutable draft、before-image 与 applied receipt | 无 | 无已批准自动减少协议 | `markdown-profile-writes.db`；digest |

允许副作用只限 Git worktree 文件和一次性测试 workspace。不得写正式 Akashic workspace、发送
消息、部署、release 或删除现有用户数据。恢复点是
`/mnt/data/coding/akasic-agent-backups/pluginize-compaction-markdown-memory-20260831/origin-main.bundle`。

## 3. 三路扫描得到的依赖迁移清单

以下清单是实现后的逐项对比 oracle，不是按文件名完成打勾。

| 编号 | 现有依赖位置 | 当前能力 | 迁移目标与对比证据 |
|---|---|---|---|
| C01 | `agent/core/passive_turn.py` | 完整 payload gate、74%、overflow retry | 中性事件调用普通 compaction Service；payload/usage/error 差分一致 |
| C02 | `plugins/compaction/` 与 `agent/model_runtime/compaction_migration_v1.py` | logical unit、20k tail、六段摘要、fallback、持久化身份 | 实现归 compaction 插件；迁移身份由冻结的 v1 模块拥有，历史 Yoyo import path 仅保留转发；契约测试逐字段一致 |
| C03 | `session/compaction_runtime.py` | projection、prepare/receipt/ledger/cursor/recovery | Session 原子端口与插件策略拆开；Session write set 一致 |
| C04 | `session/store.py`、`session/manager.py` | ledger、fence、cursor、append-only provenance | 保持唯一 owner；messages 零 UPDATE/DELETE |
| C05 | `agent/looping/core.py`、`ports.py` | 构造/关闭私有 runtime | 删除 memory/compaction 专用 deps；按 frozen Root 取普通 Service |
| M01 | `agent/memory.py` | MEMORY/SELF/PENDING 文件与 receipt | 文件原子能力移入插件；PENDING API 删除；旧 receipt 保留只读恢复 |
| M02 | `core/memory/markdown.py` | extraction、draft、store、maintenance | direct MEMORY/SELF writer 移入插件，按 kind receipt 幂等 |
| M03 | `core/memory/optimizer.py` | 周期 PENDING 合并 | 删除，无替代循环 |
| M04 | `core/memory/runtime.py`、`bootstrap/memory.py`、`bootstrap/toolsets/memory.py` | 特权 runtime/facade/toolset | 删除私有 bootstrap；普通 loader/effect 拥有生命周期 |
| M05 | `bootstrap/tools.py`、`app.py`、`app_server.py` | 传递与关闭 memory runtime | 删除专用字段与 shutdown；generation Effect 关闭插件任务 |
| P01 | `agent/context.py`、`agent/core/prompt_block.py` | SELF(30)、MEMORY(35) 固定 Prompt block | Markdown 插件通过普通 ordered prompt section 保持内容和顺序 |
| P02 | `plugins/wake/plugin.py` | implicit MEMORY 与 `memory.recall.v1` | 普通 Turn 仍通过 ordered prompt event 看到 Markdown section；不新增 read Service |
| P03 | `plugins/subagent/plugin.py`、`prompts.py` | broad memory root、SELF 路径、spawn trace | 删除 memory root；只声明 `SELF.md` 与 trace 精确文件 |
| P04 | `infra/channels/qq_channel.py` | 从 SELF 解析 actor name | 保留既有 host 只读快照路径；不作为 Markdown 插件权限或专用 API |
| P05 | `infra/mobile_realtime/runtime_inspection.py` | 硬编码 MEMORY/SELF/PENDING | 保留已有 MEMORY/SELF 只读 inspection；删除已退役 PENDING 文档 |
| P06 | `bootstrap/dashboard_api.py` | 直接 MemoryStore 与 optimizer API | 删除 optimizer 操作和专用 Core 参数；Markdown 不新增 Dashboard 特权 |
| P07 | `plugins/akasha/plugin.py` | 声明整个 memory root，但不读 Markdown | 保留可配置 sidecar 所需 root；证明它不读 Markdown，recall 行为不变 |
| P08 | Scheduler/Wake/Subagent disabled sections | `memory`、`long_term_memory`、`self_model` 名称不一致 | 定义一个公开 section group/精确 sections；场景逐项证明可见性 |
| P09 | tests/eval/debug probes | 直接构造 runtime 或读取三份 Markdown | 改为正式插件安装链和一次性 workspace；保留等价 oracle |

## 4. 分阶段实现与 Review

1. 增加中性 request/Session 原子能力并迁移 compaction 普通插件；随后 Terra xhigh Review。
2. 增加 ordered prompt/file grant，迁移 Markdown 插件并删除 PENDING/optimizer；随后 Terra xhigh Review。
3. 按 P01～P09 迁移消费者并执行完整差分 E2E；随后 Terra xhigh 累计 Review。

每次 Review 都基于固定 head，报告 P0/P1/P2、真实失败路径和 must-fix 处置。架构修改发生在
review head 之后时重跑对应 Review。

## 5. E2E 差分矩阵

- 普通请求：messages、tools、model、output budget、provider call 次数逐字段相同。
- compaction：软水位前后一 token、hard edge、generation 0、增量 generation、tool batch、fallback、
  overflow attempt 2、included/excluded、取消与 shutdown。
- 持久化：SessionDB messages 的 id/seq/role/content/tool chain 完全一致；ledger/prepare/source digest
  一致；Markdown 只登记 direct-write semantic delta。
- 插件：candidate 不读正式 Session/Markdown，stable/hot reload 使用 frozen generation，禁用或
  shadow 不触发 Core fallback，listener/task 不重复。
- Markdown：同 source_ref 重放 no-op，异内容 fail-loud，MEMORY/SELF 独立失败，写前 backup，
  原子替换 crash point，旧非空 PENDING 一次迁移；旧 v3 receipt 不重放，新 v4 receipt 才进入
  direct profile 投影。
- 消费者：P01～P09 每项都有行为断言；Akasha recall 与 Markdown ordered prompt 是两个正交能力。

## 6. 验证结果

- 三路累计 Terra xhigh Review 在 rebase 到 `origin/main` 后均为 PASS，P0/P1 为 0。
- compaction、Markdown、consumer 与 #523 retry 交叉集合分别通过 152、141、369 项。
- selected basedpyright：0 errors；`git diff --check`：通过。
- memory-context 真实 Gate：PASS，报告 `20260831-160238-329c4fcf`，请求顺序为
  summary → business → Markdown，Session messages 保持只追加。
- 最终全量：3372 passed、6 skipped；selected basedpyright 为 0 errors，当前源码还须由最终
  Change Gate 和 PR CI 固定 source digest。
