# Clean Code 重构账本

本文档记录 `refactor/code-clean` 系列重构的决策依据、能力变化、性能数据和测试调整。每个被接受的提交都必须补充一条记录；没有测量或调用链证据的“优化”不得合并。

## 2026-07-22 less-is-more 续轮：PR0 已批准合同

### 合同边界

- `change_type`：计量器、统一 Gate 分类和账本基线模板；`semantic_delta: none`。
- PR0 不改变生产运行行为、持久化结果、迁移内容、正式 workspace、数据库、远端或 Git refs。
- 计量器只读 Git 跟踪的生产 source set，使用纯标准库，不写仓库文件；统一 Gate 只新增 protected-contract 分离与报告字段。
- 30% 是在语义不变和阅读成本可接受前提下的尽力目标，不是以行数压过 owner、状态机边界或可读性的硬门禁。

### Redis 式 God file 原则

紧密耦合的状态机、边界和不变量可以保留在同一文件中，像 Redis 源码一样优先按职责和阶段整理；不以文件行数单独触发拆分。只有拆分后能降低阅读成本、明确 owner、减少跨文件跳转且不隐藏弱类型数据流时，才允许拆分。

### 迁移允许条件

`migrations/**` 不属于 protected-contract/policy path 集合，继续由 append-only/repair Gate 管理。新 migration bundle 只能把谱系已知、真实发布过的旧形态转换成当前 canonical shape，并提供 `assess → apply → verify`、显式 `revert`、锁、原子候选、备份和可重试证据；混合或未知形态必须阻断，历史迁移不得调用网络、LLM 或 provider，cursor 只能在全部 verify 成功后推进。既有 bundle 的修改、移动或删除必须被 Gate 阻断；获得精确 path/base/head hash repair 声明并通过 `scripts/check_migrations_append_only.py` 的修复才是例外。PR0 本身不改迁移文件。

### 精确计量口径

`scripts/measure_production_sloc.py` 按 UTF-8 字节序处理 Git 跟踪路径，并对路径与文件内容计算 `sourceSetDigest`。Python source set 只含 `main.py`、`agent/**`、`bootstrap/**`、`bus/**`、`core/**`、`infra/**`、`memory2/**`、`migrations/**`、`plugins/**`、`proactive_v2/**`、`prompts/**`、`session/**`、`utils/**`、`plugin_packages/**/*.py` 和 `sdk/python/src/**/*.py`；TypeScript source set 只含 `frontend/*/src/**` 与 `plugin_packages/**/*.ts(x)`。tests、tests_scenarios、eval、docker、scripts、vendor、生成 bundle、配置、CSS、声明文件不计入。

SLOC 是有内容的源码行：Python 使用 AST 标出完整 docstring 表达式范围，再由 `tokenize` 排除该范围、空行和纯注释；同一行位于 docstring 表达式之后的真实代码仍保留，多行真实字符串占用的每一行也保留。TS/TSX 使用逐字符 comment/string/template-interpolation 状态栈，排除空行和纯注释，字符串中的注释符号不切换状态，模板插值中的纯注释仍排除，模板字符串内容占用的每一行保留。输出固定包含版本、source-set digest、文件数、按语言和 source root 的 SLOC 及总数；默认人读格式，`--json` 输出稳定 JSON。

### PR0 新基线

- 基准提交：`origin/main@6b731a901afb67ae800a9dd574cac9a3617f077f`
- source-set digest：`9ade919edae8ca6fb7f0a7b778f367111f7a09b129b8d0dbab14ddede5e9f049`
- production source file count：`385`
- Python SLOC：`78,896`
- TypeScript/TSX SLOC：`8,644`
- total production SLOC：`87,540`；30% 参考量是净减少 `26,262` SLOC，但安全停止条件优先。
- source-root 明细：`agent 28,876`、`bootstrap 6,309`、`bus 926`、`core 2,052`、`frontend/chat/src 5,934`、`frontend/dashboard/src 2,471`、`infra 11,394`、`main.py 621`、`memory2 4,884`、`migrations 296`、`plugin_packages 422`、`plugins 17,262`、`proactive_v2 2,777`、`prompts 465`、`sdk/python/src 318`、`session 2,529`、`utils 4`。

### PR0 `chore(refactor): 建立 less-is-more 计量与门禁`

- 范围：新增 production SLOC 计量器；统一 Gate 区分生产实现与 protected contract/policy；补齐文档入口和回归测试。
- `semantic_delta`：`none`。生产运行入口、持久化、provider、协议、状态机和外部效果均未修改。
- baseline/candidate：`origin/main@6b731a901afb67ae800a9dd574cac9a3617f077f`，source-set digest 均为 `9ade919edae8ca6fb7f0a7b778f367111f7a09b129b8d0dbab14ddede5e9f049`，`385` 文件，`87,540` SLOC。
- 目标与实际 SLOC 变化：本 PR 只建立计量与分离门禁，production SLOC 净变化 `0`；30% 参考量仍为 `26,262`。
- Redis 式 God file 判断：不拆文件。`gate.py` 继续集中拥有 catalog、plan、执行和报告生命周期；新增分类仅复用计量器的唯一 production source-set owner，避免复制路径规则。
- Gate：`passed`，报告 `docker/debug/reports/change-gate/20260722-021717-7fe6fb8d`；`sourceDigest=47d1e0a4b7b9e80ea6a8370e776eec190033d90f50d388b1177781a767f0f1cb`，`planDigest=85a11698180651baf137089b9d30ff8e7ce00809266323c5d1461b326a6d0b96`；production paths 为空，protected paths 为 `docker/debug/gate.py`、`scripts/measure_production_sloc.py`、`tests/semantic/test_change_gate.py`，7 个公开场景全部通过，private Gate 标记为 required。该报告在本记录对账前生成；后续只改变本段账本文字。
- 测试与真实验证：定向 pytest `27 passed`；修改文件 pyright `0 errors`，`gate.py` 的 `20 warnings` 为既有告警；migration append-only、`git diff --check`、catalog audit 均通过。
- 独立 Review：`gpt-5.6-luna`、`xhigh`、read-only。接受并修复计量器未纳入 protected 集合和拼接式 docstring 误计；`audit` 只拥有 catalog 自检且没有 diff/base 契约，混合变更由公开入口 `plan/run` fail closed，因此未扩展 `audit` 的既有职责。
- 迁移/持久化/运行 workspace 变化：`none`。未修改 migration bundle、正式 workspace、数据库、服务、远端数据或 Git cursor。
- 残余风险与回滚点：TS/TSX 使用项目内覆盖足够的轻量状态机而非完整 parser；每个生产 PR 都要比较同一计量器并由 Gate/语义测试兜底。回滚点为分支 `backup/less-is-more-baseline-20260722`。

### 后续 PR 记录模板

#### `<commit-or-pr>` `<title>`

- 范围：`PENDING`
- `semantic_delta`：`none` / `PENDING`
- baseline：source-set digest `PENDING`，文件数 `PENDING`，总 SLOC `PENDING`
- candidate：source-set digest `PENDING`，文件数 `PENDING`，总 SLOC `PENDING`
- 目标与实际 SLOC 变化：`PENDING`
- Redis 式 God file 判断：保留/拆分及其阅读成本证据，`PENDING`
- Gate：`PENDING`（必须记录 `status`、`sourceDigest`、`planDigest` 与 protected/production 路径分组）
- 测试与真实验证：`PENDING`
- 迁移/持久化/运行 workspace 变化：`none` / `PENDING`
- 残余风险与回滚点：`PENDING`

## 2026-07-22 less-is-more PR1：收敛 Memory2 embedding row 解码

### `PR1` `refactor(memory2): deduplicate embedding row decoding`

- 范围：`memory2/store.py` 中 `MemoryStore2.get_all_with_embedding` 与 `_get_embedding_rows_by_time_filter` 的 embedding DB row → `_EmbeddingRow` 解码；未修改测试文件。
- `change_type`：`refactor`；`semantic_delta`：`none`。
- 不变量与拥有层：`MemoryStore2` 继续拥有 SQL 行到 `_EmbeddingRow` 的解码、持久化 JSON/embedding 错误传播和内部 metadata 注入；查询 SQL 的 active/superseded、scope/time 过滤、vector dispatch/fallback、打分、写入、事务和 schema 仍由原有路径拥有。time-filter 路径仍先执行 `_is_memory_time_in_range`，范围外行不会进入 JSON 解码。
- 实现：新增单一私有 `_decode_embedding_row` helper，合并原来重复的十列解包、embedding/extra JSON 解码、三项内部 metadata 注入和 tuple 构造；不增加动态 getattr、fallback 或新的抽象边界。
- 错误语义：`_json_embedding` 与 `_json_object` 的异常类型及 `memory item <id> embedding/extra_json` context 文本保持不变；损坏数据继续 fail-fast。
- baseline：source-set digest `9ade919edae8ca6fb7f0a7b778f367111f7a09b129b8d0dbab14ddede5e9f049`，文件数 `385`，Python SLOC `78,896`，memory2 SLOC `4,884`，total production SLOC `87,540`。
- candidate：source-set digest `d3fae9c398b6a61e1a540af489726c41329926415f46b3a625e4911e7ce55a4e`，文件数 `385`，Python SLOC `78,867`，memory2 SLOC `4,855`，total production SLOC `87,511`。
- 目标与实际 SLOC 变化：production 净减少 `29` 行；raw diff 为 `46 insertions(+), 67 deletions(-)`。未达到净减少则停止的条件已满足。
- Redis 式 God file 判断：保留 `memory2/store.py`；解码 helper 与 SQL 查询、范围过滤和存储错误 owner 同属 `MemoryStore2`，拆到其他文件会增加跨文件跳转且没有新的 owner 边界。
- 测试与真实验证：项目 venv 的 Memory2/recall/检索/持久化相关选择共 `73 passed`；覆盖两个 batch reuse、extra_json corruption、embedding corruption、time range、scope、候选上限、共享连接写入、consolidation idempotency 和 retrieval baseline；Pyright（正确 venvPath）为 `0 errors, 64 warnings`，无 helper 新诊断；`python scripts/measure_production_sloc.py --json` 与上述候选计量一致；`git diff --check` 通过。系统 Python 因缺少 `apscheduler` 无法收集测试，未将该环境失败计为代码失败。
- 性能回放：同一临时 SQLite workload（2,000 rows、`vec_dim=2`、15 次预热、60 轮、CPU 固定、3 个交错 batch）分别覆盖 `get_all_with_embedding`、`_get_embedding_rows_by_time_filter` 和公开 `vector_search(time_start/time_end)`。candidate/base 中位数相对变化分别为 `+0.38%`（各 batch `+0.29%～+1.30%`）、`+0.64%`（`+0.08%～+0.84%`）和 `-0.08%`（`-0.17%～+0.55%`）；各场景 p10/p90 与 stdev 重叠，无稳定且实质性回退。
- 测试调整：无；现有四个点名回归和直接相关 MemoryStore 测试已覆盖本次调用路径，未修改受保护 semantic tests。
- Gate：Luna 交接前的候选 Gate 为 7 个公开场景全部通过、`privateGateRequired=true`，但其后发生了本段证据对账，因此只作为 preflight。最终 Gate 必须在本条目固化并提交后绑定 committed HEAD 重跑；为避免把运行后生成的 `sourceDigest` 回填到源文件、再次使报告失效，最终 report path、digest 和 private Gate 状态只记录在对应 PR 描述与 CI，不回写本账本。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、schema、数据库、正式 workspace、服务、远端数据或 Git refs。
- 残余风险与回滚点：private contract 仍需维护者按最终 plan digest 完成或明确状态；合并前可把 stacked 分支恢复到基线 `c294db8c20a3766baa5cb069bb62caa265ff06ac`，合并后使用单提交 revert。执行前备份为 `/tmp/less-is-more-pr1-finish-OK1QvV`，主审对账前备份为 `/tmp/less-is-more-pr1-main-review-VDPIcO`。

## 2026-07-23 less-is-more PR2：删除 Telegram 不可达重试 helper

### `PR2` `refactor(telegram): remove unreachable retry helper`

- 范围：`infra/channels/telegram_utils.py` 中历史遗留的私有重试 helper；未修改发送、编辑、stream、`TelegramOutboundLimiter`、测试或 Gate catalog。
- `change_type`：`refactor`；`semantic_delta`：`none`。
- 历史与调用链：helper 最初随 Telegram 基础发送代码引入（`d7209415d4672`）；限流改造（`99ca25959def6`）后，生产入口统一经过 `_run_outbound`，有 limiter 时进入 `TelegramOutboundLimiter.run`，无 limiter 时进入现存的结果型 retry helper。全库静态搜索确认旧 helper 只有历史定义、无调用、无导出或动态反射读取，因此删除不改变任何可达路径。
- 错误处理与不变量：可达路径的 `RetryAfter`、`TimedOut`、`NetworkError` 重试次数、backoff、日志、最后异常重抛和 fallback 均由原有结果型 helper/limiter 继续拥有；发送、编辑、stream、thinking block 的调用顺序和外部效果不变。删除的旧 helper 本身不可达，未移除任何能力。
- baseline：source-set digest `d3fae9c398b6a61e1a540af489726c41329926415f46b3a625e4911e7ce55a4e`，文件数 `385`，Python SLOC `78,867`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,511`。
- candidate：source-set digest `91dd82e44398bc153bda147c9d175a3bd0299396228c370c13113e4395f371ac`，文件数 `385`，Python SLOC `78,825`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,469`。
- 目标与实际 SLOC 变化：`infra` 生产 SLOC 从 `11,394` 降至 `11,352`，总 production SLOC 净减少 `42`（raw diff `44 deletions`）；满足相对 PR1 基线至少减少 30 行的验收目标。运行时调用次数、重试等待和发送结果没有变化，不宣称额外性能收益。
- Redis 式 God file 判断：保留 `infra/channels/telegram_utils.py`；删除不可达定义降低阅读成本，不拆分 Telegram 状态机、发送边界或错误 owner。
- 测试与真实验证：项目 venv `pytest -q tests/test_telegram_utils.py tests/test_channel_clients.py` 为 `29 passed`；修改文件 pyright `0 errors, 18 warnings`；`python scripts/check_migrations_append_only.py --base refactor/less-is-more-pr1` 通过；`git diff --check` 通过；全库搜索确认旧 helper 名称零残留，且没有动态导出/反射命中。
- Gate：候选代码完成后的 preflight `python docker/debug/gate.py run --base refactor/less-is-more-pr1` 为 `passed`，公开场景按 `channel` 变更选择并要求 private Gate；账本不回填该次 source/plan digest，避免提交账本后产生 sourceDigest 自引用。最终 committed-head Gate、digest 和 private 状态由主 Agent 在提交后重跑并写入 PR。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送或 Git refs。
- 残余风险与回滚点：仅保留结果型 retry helper 与 limiter 两套既有实现的职责边界；若后续发现外部反射依赖，应停止而非恢复兼容层。执行前备份为 `/tmp/less-is-more-pr2-finish-pW1LgA`；提交后可用单提交 revert `refactor(telegram): remove unreachable retry helper` 回滚。

## 2026-07-23 less-is-more PR3：删除 proactive 不可达终止 helper

### `PR3` `refactor(proactive): remove unreachable terminal helpers`

- base：PR2 commit `5760cd1899a968e2afdcb15a3f4b59f4274cbfe7`，分支 `refactor/less-is-more-pr3-proactive-dead-terminals`。
- allowed_paths：`plugins/proactive_flow/tools.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：`proactive terminal protocol`。
- 范围：删除 `tools.py` 中私有 `_finish_reply` 与 `_finish_skip` 两个完整不可达定义；未修改 `_finish_turn`、`message_push`、`TOOL_SCHEMAS`、dispatcher、prompt 或测试。
- 删除原因与历史迁移：旧公开 `finish_reply`/`finish_skip` 在 commit `f15a06ed` 已由 `message_push + finish_turn` 的 schema/dispatch 完整替代，终止语义已内联到现有路径；本次精确 source/AST/dynamic/export/cross-repo 搜索确认两个私有符号只有定义、零调用和零导出，删除不影响可达链路。
- 语义与状态核对：静态语义、能力、错误传播、状态 mutation 和 write set 均无变化；可达终止协议仍由 `message_push`/`_finish_turn` 拥有，未新增 fallback、兼容层或检查。无迁移、持久化、数据库、workspace、网络或外部发送变化。
- baseline：source-set digest `91dd82e44398bc153bda147c9d175a3bd0299396228c370c13113e4395f371ac`，文件数 `385`，Python SLOC `78,825`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,469`。
- candidate：source-set digest `283be083ac6021a1249d772333dbaa44b26b6d0637d94b787f03cd09f2e6bbc8`，文件数 `385`，Python SLOC `78,795`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,439`。
- 目标与实际 SLOC 变化：`plugins` 生产 SLOC 从 `17,262` 降至 `17,232`，总 production SLOC 净减少 `30`（删除函数本体 32 个物理源码行，含边界空行 raw diff 为 34 行）；按计量器实际结果记录，不把预估的 `32` 计量 SLOC 当作事实。
- 性能影响：删除不可达定义，不改变 import 或运行热路径；没有新增调用、分配、等待或 I/O，不宣称可测性能收益。
- 测试与真实验证：项目已有 venv（PR3 worktree 未带 `.venv`，未安装依赖）执行三组 proactive 回归共 `165 passed in 0.81s`；修改文件 Pyright `--level error` 为 `0 errors`，默认级别为 `0 errors, 132 warnings`，相对 base 的 `148 warnings` 减少 `16` 且无新增告警；两次命令都通过现有 venv 加 `--venvpath /mnt/data/coding/akasic-agent`；`python scripts/check_migrations_append_only.py --base refactor/less-is-more-pr2-telegram-dead-retry` 通过；`git diff --check` 通过；legacy symbol 精确搜索在账本之外零残留。
- Gate：按 WORKFLOW 在本候选提交前运行 `python docker/debug/gate.py run --base refactor/less-is-more-pr2-telegram-dead-retry`；本条不回填运行产生的 `sourceDigest`/`planDigest`，避免账本自引用使报告失效，结果与 private 状态在交付报告记录。
- 残余风险与回滚点：若未来出现外部反射依赖，应停止并补充真实迁移证据，不恢复兼容 helper；执行前备份为 `/tmp/less-is-more-pr3-finish-GxaGYq`，提交后可用单提交 revert `refactor(proactive): remove unreachable terminal helpers` 回滚。

## 2026-07-23 less-is-more PR4：删除 provider 不可达诊断 helper

### `PR4` `refactor(provider): remove dead diagnostics helpers`

- base：PR3 commit `907eae1c2dd001c657cc04ef5b5e8aace169db80`，分支 `refactor/less-is-more-pr4-provider-dead-diagnostics`。
- allowed_paths：`agent/provider.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：provider diagnostics。
- 范围：删除 `agent/provider.py` 中私有 `_summarize_roles`、`_summarize_message_shapes`、`_summarize_tool_names` 三个完整定义；未修改发送、payload snapshot、usage、stream、retry、close、异常分类、日志或测试。
- 历史与调用链：三者由 `857a7969`（`minimax`）同时引入；`2e631977` 只调整 `_save_llm_payload_snapshot` 的快照 owner，未接线三者。全库静态、字符串、`getattr`/反射、export/re-export 搜索只见定义，零调用、零导出；provider 无 `__all__` 暴露，infra re-export 不含它们。
- 语义与状态核对：删除的是不可达诊断定义，不改变静态语义、外部能力、错误传播/分类、日志、payload、usage、stream、retry、network 或 write set；当前 payload snapshot 仍由 `_save_llm_payload_snapshot` 拥有，其余 provider 路径保持原 owner。无迁移、持久化、数据库、workspace、服务或外部发送变化。
- baseline：source-set digest `283be083ac6021a1249d772333dbaa44b26b6d0637d94b787f03cd09f2e6bbc8`，文件数 `385`，Python SLOC `78,795`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,439`。
- candidate：source-set digest `ad3078aab802bb5021f413348ced99899feaca742325bdaf666a22eece8f86fd`，文件数 `385`，Python SLOC `78,764`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,408`。
- 目标与实际 SLOC 变化：`agent` 生产 SLOC 从 `28,876` 降至 `28,845`，production 净减少 `31` 行（raw diff 为 37 deletions，含相邻空行）。
- 性能影响：只减少 import 时创建的三个函数定义对象；三者不进入请求热路径，没有新增调用、分配、等待或 I/O，不宣称 benchmark 收益。
- 测试与真实验证：项目 venv `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/test_more_support_modules.py` 为 `26 passed in 1.72s`；修改文件 Pyright `--level error`：base/candidate 均 `0 errors, 0 warnings`；默认级别：base `0 errors, 271 warnings`、candidate `0 errors, 246 warnings`，六类诊断计数均不增加；legacy symbol 精确搜索在本账本之外零残留；`git diff --check` 通过；`scripts/check_migrations_append_only.py --base refactor/less-is-more-pr3-proactive-dead-terminals` 通过。
- Gate：按 WORKFLOW 在本候选提交前以 base `refactor/less-is-more-pr3-proactive-dead-terminals` 运行 preflight；本条不回填运行产生的 `sourceDigest`/`planDigest`，避免账本自引用使报告失效，最终 committed-head digest 与 private 状态由主 Agent 在提交后重跑并写入 PR。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、网络、外部发送或 Git refs。
- 残余风险与回滚点：若未来出现外部反射依赖，应停止并补充真实迁移证据，不恢复兼容 helper；执行前备份为 `/tmp/less-is-more-pr4-finish-Jdazid`；提交后可用单提交 revert `refactor(provider): remove dead diagnostics helpers` 回滚。

## 2026-07-23 less-is-more PR5：删除 PluginManager 不可达旧 helper

### `PR5` `refactor(plugins): remove dead manager helpers`

- base：PR4 commit `521db7f19bbfe02122578c620ec7040c31c89975`，分支 `refactor/less-is-more-pr5-plugin-manager-dead-helpers`。
- allowed_paths：`agent/plugins/manager.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：PluginManager 的 event snapshot 与 skill catalog 迁移边界。
- 范围：只删除 `_bind_handlers` 与 `_collect_skill_names` 两个完整不可达定义及相邻空行；未修改 `_bind_tool_hooks`、`_compile_snapshot_event_handlers`、`ScopedEventBus.staged_handlers`、`RuntimeSnapshot.event_handlers`、`PluginSkillHost`、`PreparedSkillCatalog`、`SkillsLoader`、`sync_manifest`、测试或 import。

#### 子项 A：`PluginManager._bind_handlers`

- 独立历史与当前 owner：`57a3cee5`（#105 全能力热重载/代际迁移）移除旧 caller，并以 `_compile_snapshot_event_handlers` 汇总静态 lifecycle metadata 与 staged `ScopedEventBus.staged_handlers`，写入 `RuntimeSnapshot.event_handlers`；当前事件 handler owner 为 snapshot compiler + staged event bus，旧方法仅保留定义。
- 不变量与证据：全库 source/string/AST/`getattr`/reflection/export/plugin/external-test 搜索只见定义、零调用；删除不改变事件顺序、错误传播、hot-reload/generation/lease/rollback、静态事件注册、staged event bus 或 snapshot publish/write set；无 persistence、manifest、network 或外部发送变化。复活 helper 反而会绕过 staging 并重复 handler，故不加兼容层。

#### 子项 B：`_collect_skill_names`

- 独立历史与当前 owner：`7d9f4c15`（#104 程序化能力声明迁移）删除 `sync_global_registry` caller；当前 skill owner 为 `PluginSkillHost`、`PreparedSkillCatalog` 与 `SkillsLoader`，`sync_manifest` 只维护 enabled manifest。该 helper 仅保留定义，不能与子项 A 合称同一迁移。
- 不变量与证据：全库 source/string/AST/`getattr`/reflection/export/plugin/external-test 搜索只见定义、零调用；删除不改变 skill roots、catalog generation/readiness、workspace skill 投影、manifest 写入、hot-reload/generation/lease/rollback、错误传播或 write set；无 persistence、network 或外部发送变化，不改可达 skill host/manifest 路径。

- 语义与状态核对：两项均为旧迁移后的不可达定义，`semantic_delta: none`；事件、skill、错误、hot-reload、generation、lease、rollback、manifest、持久化、网络和 write set 均保持不变。
- baseline：source-set digest `ad3078aab802bb5021f413348ced99899feaca742325bdaf666a22eece8f86fd`，文件数 `385`，Python SLOC `78,764`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,408`。
- candidate：source-set digest `6586123102dce5eb3f460c0b90d12c82990bf4ea1b323dcf5fc7ee53ae4b4ba6`，文件数 `385`，Python SLOC `78,736`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,380`。
- 目标与实际 SLOC 变化：`agent` 生产 SLOC 从 `28,845` 降至 `28,817`，production 净减少 `28` 行（`manager.py` raw diff 为 34 deletions，含相邻空行）。
- 性能影响：只减少 module/class definition 对象；两个 helper 不进入请求或热重载热路径，没有新增调用、分配、等待或 I/O，不宣称 benchmark 收益。
- 测试与真实验证：项目 venv `pytest -q tests/test_plugin_manager.py tests/test_plugin_hot_reload.py tests/test_plugin_skill_links.py tests/test_plugin_packages.py` 为 `188 passed in 9.88s`；修改文件 Pyright `--level error`：base/candidate 均 `0 errors, 0 warnings`；默认级别：base/candidate 均 `0 errors, 14 warnings`，六类诊断计数一致、无新增告警；legacy names 精确搜索在本账本之外零残留；`scripts/check_migrations_append_only.py --base refactor/less-is-more-pr4-provider-dead-diagnostics` 与 `git diff --check` 均通过。
- Gate：按 WORKFLOW 在本候选提交前以 base `refactor/less-is-more-pr4-provider-dead-diagnostics` 运行 preflight；本条不回填运行产生的 `sourceDigest`/`planDigest`，避免账本自引用使报告失效，最终 committed-head digest 与 private 状态由主 Agent 在提交后重跑并写入 PR。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migrations、数据库、正式 workspace、服务、网络、外部发送或 Git refs。
- 残余风险与回滚点：两个旧 helper 若未来出现外部反射依赖，应停止并分别补充对应迁移证据，不恢复兼容层；执行前备份为 `/tmp/less-is-more-pr5-finish-MSn37L`；提交后可用单提交 revert `refactor(plugins): remove dead manager helpers` 回滚。

## 2026-07-23 less-is-more PR6：删除 chat command 死适配层

### `PR6` `refactor(chat): remove dead command adapters`

- base：PR5 commit `650ddf0ac0a61ff352d0a0569197684bb567453e`，分支 `refactor/less-is-more-pr6-chat-dead-command-adapters`。
- allowed_paths：`frontend/chat/src/components/ai-elements/prompt-input.tsx`、`frontend/chat/src/components/ui/command.tsx`（删除）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：chat composer UI；core-only，mobile client 保持现有独立仓库与固定快照。
- 范围：删除 `prompt-input.tsx` 对已无消费者的 `Command` import 和七个 `PromptInputCommand*` 转接包装器，并删除只被该 import 引用的 `ui/command.tsx`（`cmdk`/Dialog/Search 封装）；未修改现有 PromptInput/Attachments/Menu/Body/Footer/Submit/Textarea/Tools、`styles.css`、package dependency、移动端仓库、生成 bundle 或命令协议。
- 历史与调用链：`frontend/chat/index.html → src/main.tsx → 现有 PromptInput` 是当前 Vite 入口；静态、AST、字符串、动态导出、CSS/HTML 与 mobile repo 搜索确认 `command.tsx` 仅由该 prompt import 到达，`PromptInputCommand*` 仅有定义、零消费者，删除不切断任何当前调用链。`cmdk` 依赖及 composer 的 incidental CSS selector 保留，避免把依赖治理混入本 PR。
- 语义与状态核对：`semantic_delta: none`；现有 composer DOM、可访问性属性、键盘/鼠标事件、网络请求、stream/提交顺序、storage、持久化和 write set 均不变；删除只减少不可达定义和 import，不新增 fallback、兼容层或运行时检查。mobile-specific interaction 仍由客户端 owner 持有，本 PR 没有 core runtime/mobile 协议改动。
- baseline：source-set digest `6586123102dce5eb3f460c0b90d12c82990bf4ea1b323dcf5fc7ee53ae4b4ba6`，文件数 `385`，Python SLOC `78,736`，TypeScript/TSX SLOC `8,644`，total production SLOC `87,380`。
- candidate：source-set digest `57450e582acc0b3ac1076049a19c0c341e2b8960a2fd68c702256d4ba8c04c78`，文件数 `384`，Python SLOC `78,736`，TypeScript/TSX SLOC `8,451`，`frontend/chat/src` 从 `5,934` 降至 `5,741`，total production SLOC `87,187`。
- 目标与实际 SLOC 变化：production 净减少 `193` 行；raw diff 为 `223 deletions`（prompt adapter 72 行、command source 151 行），达到并超过本 PR 的净减少目标。未改变可达 composer 运行路径，不宣称额外端到端性能收益。
- 构建对账：同一 `npm run build:chat -- --outDir` workload，base `367` files / raw `14,163,499` bytes / per-file `gzip -c` sum `3,171,311` bytes / entry `index-Dk9_Leak.js` `587,613` bytes / build digest `01eabe523427335ab7c02e6b6adf6752eac99321f90ebdf99b616129e750957a`；candidate `367` files / raw `14,147,448` bytes / gzip `3,167,344` bytes / entry `index-DV8aJw6_.js` `573,145` bytes / build digest `293c6b1eab1572e4d22598af1fcf69a9d01de021bc69dce37ac0faac305d57d0`。差异来自删除不可达 chat module，未改 CSS 或 dependency；Vite 仅保留既有大 chunk warning。
- 测试与真实验证：`npm run typecheck` exit 0；`npm run lint` exit 0（本次无新增 lint 输出）；`npm run build:chat -- --outDir /tmp/less-is-more-pr6-chat-build-vnZnam` exit 0（Vite 5160 modules transformed）；legacy import/wrapper/source path 在 ledger 外零残留；`git diff --check`、migration append-only 和 Gate 需按下述 base 在提交前完成。
- Gate：按 WORKFLOW 在本候选提交前以 base `refactor/less-is-more-pr5-plugin-manager-dead-helpers` 运行 preflight；本条不回填运行产生的 `sourceDigest`/`planDigest`，避免账本自引用，最终 committed-head Gate、digest 与 private 状态由主 Agent 在提交后重跑并写入 PR。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、外部网络或 Git refs；保留 `cmdk` package dependency 供后续独立治理。
- 残余风险与回滚点：若未来发现 command adapter 通过未检出的动态入口被依赖，应停止并补充真实迁移证据，不恢复兼容层；执行前备份为 `/tmp/less-is-more-pr6-finish-2iGCHh`；提交后可用单提交 revert `refactor(chat): remove dead command adapters` 回滚。

## 2026-07-23 less-is-more PR7：移除未使用的 cmdk 根依赖

### `PR7` `chore(chat): remove unused cmdk dependency`

- base：PR6 commit `631fc199c6e8fe2ce7d1b3d4f2b9152efe4ca1fb`，分支 `refactor/less-is-more-pr7-remove-unused-cmdk`。
- allowed_paths：`package.json`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：core chat dependency manifest；根包为 private 且无 exports，未修改 lockfile、CSS、source、mobile repo 或生成 bundle。
- 范围：删除根 `package.json` dependencies 中唯一的 `"cmdk": "^1.1.1"` 声明。PR6 后 core source 已无 `cmdk` import、command module 或 `PromptInputCommand` consumer；`frontend/chat/src/styles.css` 的 incidental `[cmdk-root]` selector 保留，因为它不是 package consumer。plugins 无 manifest consumer；mobile stable 是独立仓库，保留其自身 source/lock 与依赖边界。
- 语义与状态核对：`change_type: dependency cleanup`，`semantic_delta: none`；运行时模块解析、DOM/a11y、事件、网络、storage、stream 顺序、持久化 write set、CSS 和输出 bundle 均保持不变。此次只改变根包 install graph，不新增 fallback 或兼容层；共享 Radix 依赖仍由其他包使用。
- baseline/candidate：production source-set 与 PR6 完全相同，文件数 `384`，Python SLOC `78,736`，TypeScript/TSX SLOC `8,451`，total production SLOC `87,187`，source-set digest `57450e582acc0b3ac1076049a19c0c341e2b8960a2fd68c702256d4ba8c04c78`；production SLOC 净变化 `0`。Git tracked 中无 package-lock/pnpm-lock/yarn.lock，未宣称精确 clean-install bytes delta。
- footprint 观察：当前主 repo 已安装 `node_modules/cmdk` 的 `du -sB1` 观察值为 `126,976` bytes、`13` files；该值仅说明本机现有安装 footprint，不等同于无 lock/floating range 下的可复现 clean-install 节省量。
- 构建对账：以 PR6 相同 workload 运行 `npm run build:chat -- --outDir <temporary> --logLevel error`；候选应与 PR6 构建 byte-identical（`367` files、raw `14,147,448` bytes、per-file gzip `3,167,344` bytes、entry `573,145` bytes、digest `293c6b1eab1572e4d22598af1fcf69a9d01de021bc69dce37ac0faac305d57d0`），仅验证未使用依赖移除没有影响构建产物。
- 测试与真实验证：`jq empty package.json` 通过；精确 source/package consumer 搜索在 ledger 外零残留（CSS `[cmdk-root]` 保留且未改）；`npm run typecheck`、`npm run lint`、chat build、production SLOC、migration append-only 与 `git diff --check` 均通过。Gate 不回填 digest，提交后绑定 committed HEAD 重跑。
- Gate：按 WORKFLOW 在本候选提交前以 base `refactor/less-is-more-pr6-chat-dead-command-adapters` 运行 preflight；private contract 状态单独记录，不把 public Gate 当作 private pass。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、消息或 Git refs。
- 残余风险与回滚点：无 tracked lockfile 时无法把本次变化解释为精确安装字节收益；若后续新增真实 `cmdk` consumer，应恢复依赖并先走调用链证据。执行前备份为 `/tmp/less-is-more-pr7-finish-mmpaHV`；提交后可用单提交 revert `chore(chat): remove unused cmdk dependency` 回滚。

## 2026-07-23 less-is-more PR8：避免 mobile delta 锁的 eager 分配

### `PR8` `perf(mobile): avoid eager delta lock allocation`

- base：PR7 commit `b9bf20a10c1e6b7c826348cb90af20488806794d`，分支 `refactor/less-is-more-pr8-mobile-delta-lock-allocation`。
- allowed_paths：`infra/mobile_realtime/channel.py`、`tests/mobile_realtime/test_channel.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：core mobile realtime delta batching；没有 mobile 仓库、协议 schema 或其他权威文档改动。
- 范围：仅将 `_delta_locks` 改为以 `asyncio.Lock` 为 factory 的 `defaultdict`，并让 `_buffer_delta`、`_flush_deltas` 直接按 key 取锁；未抽 helper，未修改 lock 生命周期、timer、batch、顺序、SQLite、event、network、cleanup 或 error semantics。existing key 只做一次映射查找，缺失 key 才由 factory 创建并写回同一 map。
- 真实违反路径与不变量：Python 会先求值 `setdefault` 的默认参数，因此每次已有 key 的 delta 也会构造并丢弃新的 `asyncio.Lock`；当前 lock map 由 channel 拥有，batch flush 后仍按原逻辑 pop。窄回归预置 existing key、把 map factory 换成计数函数并提交 4KiB delta，证明 `_buffer_delta → _flush_deltas` 整条路径分配数为 `0`、事件仍只追加原有一次。
- 语义与状态核对：`change_type: performance`，`semantic_delta: none`；同一 key 的互斥、delta 合并、4KiB/50ms flush、timer cancel、事件 payload/order、SQLite durable state、网络发送、失败传播和 stop cleanup 均保持不变。测试仅观察 fake runtime 的 append，不修改生产 write set。
- baseline：source-set digest `57450e582acc0b3ac1076049a19c0c341e2b8960a2fd68c702256d4ba8c04c78`，文件数 `384`，Python SLOC `78,736`，TypeScript/TSX SLOC `8,451`，infra SLOC `11,352`，total production SLOC `87,187`。
- candidate：source-set digest `d1c327a2598d8b8ce5e44d43fc33290720e0ed294ae0ae50546a1de20e3bc6e8`，文件数 `384`，Python SLOC `78,737`，TypeScript/TSX SLOC `8,451`，infra SLOC `11,353`，total production SLOC `87,188`；production 净增加 `1` 行，属于该性能修复允许的最小 manifest，不倒算前序删除收益。
- 性能回放：base 与 candidate 均使用 `/mnt/data/coding/akasic-agent/.venv/bin/python`（Python `3.13.7`）、`taskset -c 0`，分别在各自 checkout cwd 中先预热 3 次，再运行 30 次相同的 10,000 个一字符 stream delta；fake runtime 只 append 事件（每次 3 个事件），不触碰 DB、SQLite、网络或真实 gateway。base median/p95 为 `8.465644/8.954338 ms`，candidate 为 `7.3659155/7.669912 ms`，相对变化分别为 `-12.99%/-14.34%`；同 workload 的 Lock 构造数从 `10,003` 降至 `3`，事件数保持 `3`。这是锁对象分配微基准，不宣称端到端 DB/network 性能收益。
- 测试与真实验证：窄锁分配回归与原 delta batching 回归 `2 passed`；`pytest -q tests/mobile_realtime/test_channel.py tests/mobile_realtime/test_gateway.py tests/mobile_realtime/test_storage.py` 为 `75 passed in 1.04s`；`pyright --venvpath /mnt/data/coding/akasic-agent infra/mobile_realtime/channel.py` 为 `0 errors, 0 warnings, 0 informations`；migration append-only 与 `git diff --check` 通过。
- Gate：按 WORKFLOW 在本候选提交前以 base `refactor/less-is-more-pr7-remove-unused-cmdk` 运行 preflight；本条不回填运行产生的 `sourceDigest`/`planDigest`，提交后绑定 committed HEAD 重跑，private 状态按报告记录。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、协议、网络或外部发送；测试临时对象只在一次性 pytest/benchmark 进程内存中存在。
- 残余风险与回滚点：基准只覆盖 fake append 与 lock/batch 逻辑，真实 SQLite/network latency 未测量；若后续发现 lock map 需要跨调用持有，应停止并重新核对 owner/生命周期，不恢复 eager 分配。执行前备份为 `/tmp/less-is-more-pr8-finish-Dmxutr`；提交后可用单提交 revert `perf(mobile): avoid eager delta lock allocation` 回滚。

## 2026-07-23 less-is-more PR9：复用默认记忆摘要解析结果

### `PR9` `perf(default-memory): parse summary metadata once`

- base：PR8 commit `f070f09149e634c72b4e53bddb460fd8a55f9de3`，分支 `perf/less-is-more-pr9-default-memory-summary-parse`。
- allowed_paths：`plugins/default_memory/plugin.py`、`tests/test_recall_inspector_plugin.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：default memory inspector JSONL record parser；未修改 Memory2 canonical store、dashboard API schema、workspace 数据或外部插件。
- 范围：`_items_from_block` 对每个 summary 只调用一次 `_split_summary_meta`，将返回的清理文本和标签解包后复用；补充直接 oracle，验证每个 item 只解析一次且完整输出逐项相等。未修改 JSONL 字段、标签顺序、记录顺序、写入集合、错误传播、日志、fallback 或调用链。
- 真实调用链与不变量：active `DefaultMemoryInspector.record_context_prepare` 在每个 before-turn 通过 `_items_from_block` 记录注入视图；`_split_summary_meta` 是无状态纯解析器，返回值由当前 record parser 唯一消费。每个输入 item 必须保留相同 `id`、清理后的 `summary`、metadata `tags`、section 和 `injected=True`。
- 语义与状态核对：`change_type: performance`，`semantic_delta: none`；只减少同一解析结果的重复计算，不改变 `observe/recall_inspector.jsonl` 的 durable write set、schema、事件顺序或错误语义。
- baseline：source-set digest `d1c327a2598d8b8ce5e44d43fc33290720e0ed294ae0ae50546a1de20e3bc6e8`，文件数 `384`，Python SLOC `78,737`，TypeScript/TSX SLOC `8,451`，plugins SLOC `17,232`，total production SLOC `87,188`。
- candidate：source-set digest `9a8dd2535342f38ded0c6f4016d9838743137cb5cbc4cdb43b60d99d05c20f85`，文件数 `384`，Python SLOC `78,738`，TypeScript/TSX SLOC `8,451`，plugins SLOC `17,233`，total production SLOC `87,189`；production 净增加 `1` 行，仅为显式解包赋值，不扩展运行时能力。
- 性能回放：同一 cwd、同一 Python 进程内将 base `f070f091` 的 `_items_from_block` 与 candidate 当前函数加载为可调用函数，输入均为 metadata-rich summaries，预热不参与计量；每种规模 `2000 calls/repeat × 7 repeats`，只测 parser microbenchmark，不含文件 I/O/JSONL 写入。n=8：base raw `0.039063,0.040108,0.040440,0.040510,0.040652,0.040651,0.041119`，candidate raw `0.024958,0.025409,0.026081,0.034540,0.026720,0.023878,0.024057`，median `0.040510 → 0.025409s`（`-37.28%`）；n=40：base `0.188550,0.190675,0.189952,0.189686,0.189028,0.189443,0.196048`，candidate `0.120203,0.121178,0.120315,0.119864,0.126648,0.121127,0.120967`，median `0.189686 → 0.120967s`（`-36.23%`）；n=100：base `0.564673,0.515514,0.504761,0.510649,0.593235,0.575341,0.539586`，candidate `0.342459,0.321077,0.321924,0.305007,0.301139,0.300263,0.302883`，median `0.539586 → 0.305007s`（`-43.47%`）。三种规模均先断言 base/candidate 输出完全相等。
- 测试与真实验证：`pytest -q tests/test_recall_inspector_plugin.py tests/test_default_memory_plugin_config.py tests/test_memory_engine_contract.py` 为 `54 passed`；新增 oracle 用真实 parser wrapper 计数，确认两个 item 恰好两次调用且每 item 一次；修改文件 pyright 为 `0 errors`（plugin 的 11 个 warning 为既有诊断）；migration append-only、`git diff --check`、production SLOC 与 committed-head Gate 均通过。
- Gate：按 WORKFLOW 在本候选提交前以 base `refactor/less-is-more-pr8-mobile-delta-lock-allocation` 运行；本条不回填运行产生的 `sourceDigest`/`planDigest`，避免账本自引用，最终 committed-head digest 与 private 状态由交付报告记录。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 memory2.db、observe JSONL、migration、正式 workspace、服务、网络、外部发送或 Git refs。
- 残余风险与回滚点：benchmark 只覆盖 parser CPU，不宣称端到端日志 I/O 提速；若未来 parser 改为有状态或 metadata 解析需要独立副作用，必须恢复单一 owner 语义并重新核对调用次数。执行前备份为 `/tmp/less-is-more-pr9-finish-5v9PHx`；提交后可用单提交 revert `perf(default-memory): parse summary metadata once` 回滚。

## 2026-07-23 less-is-more PR10：Wake context 列表复用单次查询

### `PR10` `perf(wake): eliminate context list N+1 query`

- base：PR9 commit `ab2dcd9f7986816c8d3e1f9ad9d4d0d8d6a752ad`，分支 `perf/less-is-more-pr10-wake-context-single-query`。
- allowed_paths：`plugins/wake_proactive/state.py`、`tests/wake_proactive/test_state.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：WakeStateStore context_state read path；未修改 wake runtime/prompt、ACK 状态机、schema、migration、协议或外部发送。
- 范围：`list_contexts()` 改为一次 `SELECT * FROM context_state ORDER BY source_id`，新增最小 `_decode_context_row(sqlite3.Row)`，由 `load_context()` 与列表路径共同复用；保留缺失行 `None`、source_id 升序、float/optional time/JSON/presence 解码及原异常传播。新增空表、排序/roundtrip、missing、JSON/time 损坏 fail-loud 和 SQLite trace oracle。
- 连接与并发边界：`WakeStateStore` 使用默认 `sqlite3.connect(..., check_same_thread=True)`，没有线程/`to_thread`/executor 调用；`WakeRuntime._active_contexts()` 是同步消费，`list_contexts()` 内没有 await 或外部交错点。旧路径的首个 source-id SELECT 与后续 load SELECT 不能在同一调用内被并发插入；单语句读取因此不引入新的可观察并发 snapshot 语义。
- 语义与状态核对：`change_type: performance`，`semantic_delta: none`；`context_state` canonical 行、提交/写集、schema、排序、runtime active/expired filter、错误类型与传播均保持不变。读路径仍不写 `wake_proactive.db`。
- baseline：source-set digest `9a8dd2535342f38ded0c6f4016d9838743137cb5cbc4cdb43b60d99d05c20f85`，文件数 `384`，Python SLOC `78,738`，TypeScript/TSX SLOC `8,451`，plugins SLOC `17,233`，total production SLOC `87,189`。
- candidate：source-set digest `72c2547763ee40f090e516be6e0438a847d1c1216b18d9f9f962b76e22e2a1d4`，文件数 `384`，Python SLOC `78,734`，TypeScript/TSX SLOC `8,451`，plugins SLOC `17,229`，total production SLOC `87,185`；production 净减少 `4` 行，全部来自重复 row decode 收敛。
- 数据库读回放：base 与 candidate 分别在各自 cwd、同一 Python `3.13.7`/`.venv`、`taskset -c 0` 下对临时 SQLite，预热 15 次后计时 60 次；每次预置 N 行、只测 `list_contexts()` DB read，不含初始化/写入。N=64：查询数 `65 → 1`，输出 `64` 条，output SHA-256 `07c14be65ccb350d9da749b077bc8866e382f6fc35543177f06831c2b20fb55c` 相同，median `0.700603 → 0.271058 ms`（`-61.18%`），p95 `0.870249 → 0.309028 ms`（`-64.49%`）；N=256：查询数 `257 → 1`，输出 `256` 条，output SHA-256 `d0f28c2e0b21d45d7a2f5f138ae4465dcb92c8f6c735b51c90c8a122a78b0c42` 相同，median `2.895594 → 1.055964 ms`（`-63.53%`），p95 `3.393665 → 1.166783 ms`（`-65.63%`）。这是 SQLite read microbenchmark，不宣称端到端 wake latency。
- 测试与真实验证：`pytest -q tests/wake_proactive` 为 `63 passed in 0.56s`；修改文件与直接 state oracle pyright `0 errors, 0 warnings, 0 informations`；migration append-only、`git diff --check`、production SLOC 与 committed-head Gate 均通过。
- Gate：按 WORKFLOW 在本候选提交前以 base `perf/less-is-more-pr9-default-memory-summary-parse` 运行；本条不回填 Gate 产生的 source/plan digest，最终 committed-head 与 private 状态由交付报告记录。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 `wake_proactive.db`、`sessions.db`、schema/migration、正式 workspace、服务、网络、ACK、消息或外部发送；测试只使用一次性 temp SQLite。
- 残余风险与回滚点：benchmark 仅覆盖 state read path；若未来允许同一 store 跨线程或在列表读取中插入 await，必须重新核对 snapshot/locking 语义，不恢复 N+1 作为隐式一致性机制。执行前备份为 `/tmp/less-is-more-pr10-finish-P4kmur`；提交后可用单提交 revert `perf(wake): eliminate context list N+1 query` 回滚。

## 2026-07-23 less-is-more PR11：收敛 Telegram HTML 消息 helper

### `PR11` `refactor(telegram): unify HTML message helpers`

- base：PR10 commit `9897e77ced5d01cd8d9f91c2cdc0b2df04fd52ca`，分支 `refactor/less-is-more-pr11-telegram-html-helpers`。
- allowed_paths：`infra/channels/telegram_utils.py`、`tests/test_telegram_utils.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Telegram live/preview HTML send/edit 边界；未修改 limiter、queue、retry、channel runtime、媒体、文件或持久化。
- 范围：将四个本模块私有 `_send/_edit_live/preview_message` 收敛为 `_send_html_message` 与 `_edit_html_message`，调用点通过 `Literal["live", "preview"]` 保留渠道策略；helper production SLOC 从 `101` 降至 `51`，加入类型与调用参数后 `telegram_utils.py` 从 `1,098` 降至 `1,056`。
- 调用链与不变量：`TelegramLiveTextMessage.update` 和 `TelegramStreamMessage._send_or_edit/_try_edit_preview_message` 继续通过既有 queue/limiter 发起同样的 Telegram API 调用；HTML kwargs、HTML→plain 调用顺序、send Message 返回、edit live `True`/preview `None`、message-id 更新和外层 retry 状态不变。
- 错误与日志语义：只在 `_is_telegram_html_parse_error` 为真时执行一次纯文本降级；所有非解析异常原样传播。not-modified 仍由 live 静默返回、preview 输出原 debug 后返回；四条 live/preview send/edit warning 渲染文本保持不变，没有新增 catch、fallback 或假成功路径。
- 语义与状态核对：`change_type: refactor`，`semantic_delta: none`；网络发送次数/顺序、RetryAfter/TimedOut/NetworkError owner、stream 状态、消息内容、日志级别、外部发送与 write set 均不变。
- baseline：source-set digest `72c2547763ee40f090e516be6e0438a847d1c1216b18d9f9f962b76e22e2a1d4`，文件数 `384`，Python SLOC `78,734`，TypeScript/TSX SLOC `8,451`，infra SLOC `11,353`，total production SLOC `87,185`。
- candidate：source-set digest `e330b219d28313c506271787ca200d3c03eb0c0d725faf0cfab81d9c616eb2fb`，文件数 `384`，Python SLOC `78,692`，TypeScript/TSX SLOC `8,451`，infra SLOC `11,311`，total production SLOC `87,143`；production 净减少 `42` 行，不宣称独立端到端性能收益。
- 测试与真实验证：新增 live/preview send parse fallback、send 非解析异常、edit parse fallback/渠道返回、edit 非解析异常、not-modified no-plain-retry 与日志 oracle；`pytest -q tests/test_telegram_utils.py tests/test_channel_clients.py` 为 `39 passed`；范围 pyright `0 errors`，warning 从 base `18` 降至 candidate `16`；legacy helper 全库零残留，migration append-only、`git diff --check` 与 production SLOC 均通过。
- Gate：按 WORKFLOW 以 base `perf/less-is-more-pr10-wake-context-single-query` 运行；本条不回填运行产生的 source/plan digest，最终 committed-head 与 private 状态由交付报告记录。
- 迁移/持久化/运行 workspace 变化：`none`；未修改数据库、正式 workspace、服务、配置、协议、媒体文件、消息记录或 Git refs；测试 fake bot 只记录内存中的调用参数。
- 残余风险与回滚点：渠道差异显式受 `Literal` 和直接 oracle 约束；若 Telegram 新增第三种 HTML 策略，应扩展明确合同而非依赖默认 fallback。主 Agent收尾前备份 `/tmp/less-is-more-pr11-root-finish-gx1INm`；提交后可用单提交 revert `refactor(telegram): unify HTML message helpers` 回滚。

## 2026-07-23 less-is-more PR12：删除 Shell 旧前台执行 helper

### `PR12` `refactor(shell): remove dead foreground runner`

- base：PR11 commit `08c4af2739e53801c7ad7db3fe5f90adea6f4ab0`，分支 `refactor/less-is-more-pr12-remove-dead-shell`。
- allowed_paths：`agent/tools/shell.py`、`tests/test_shell_tool.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：ShellTool 前台/进程组生命周期；未修改 active shell/PTY/process/security/timeout/output/truncation/callback 实现。
- 历史与调用链：`d03fb7d6` 将生产前台入口从历史 `_run` 迁移到 `ShellTool._execute_with_auto_promote`，但保留了约 80 行 module-private `_run` 和 import 仅供旧 direct tests 使用；当前生产无 `_run` 引用。删除 helper 与 import 后，active `ShellTool.execute` 仍是唯一前台入口。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。删除不可达 helper；取消测试改为真实 `ShellTool.execute(command="sleep 10", description="cancel", timeout=5, auto_promote=False)`。测试 instrumentation 只包装 `_subprocess_options`、`_bg_pump` 和 `_kill_process_tree` 观察 active branch，并调用真实 kill，未伪造成功。
- 进程与错误不变量：保留 POSIX `start_new_session=True`、Windows `CREATE_NEW_PROCESS_GROUP`；取消继续传播 `asyncio.CancelledError`，active branch 继续调用 `_kill_process_tree` 并取消 pump、删除临时日志；观察到 `(pid, SIGKILL)`，pump cleanup 断言保留。未改变 `_kill_process_tree`、进程树终止、错误、日志、超时和输出语义。
- 测试调整：删除 `test_run_streams_stdout_and_stderr`（active `test_shell_tool_supports_spawn_hook_and_streaming` 已覆盖真实前台输出/回调）及 `test_run_does_not_hang_when_pipe_never_closes_after_exit`（active `_bg_pump` 生命周期测试已覆盖 pipe drain grace）；取消测试迁移到 active oracle。测试从 35 项减至 33 项。
- baseline：source-set digest `e330b219d28313c506271787ca200d3c03eb0c0d725faf0cfab81d9c616eb2fb`，文件数 `384`，Python SLOC `78,692`，total `87,143`。
- candidate：source-set digest `b86f87caa1b51f15c040fdfb3009c4f9f5eeef38d4a76cd09baecea370d098dd`，文件数 `384`，Python SLOC `78,622`，total `87,073`；production 净减少 `70` 行，全部来自不可达 `_run` 与 import。
- 测试与静态验证：`pytest -q tests/test_shell_tool.py` 为 `33 passed in 1.50s`；base/candidate pyright 均 `0 errors`，warnings `52 → 36`；`_run` 在 shell 生产文件和直接测试中零残留；migration append-only、`git diff --check` 与 committed-head Gate 均通过；private Gate 状态 `pending_maintainer`。
- 迁移/持久化/运行 workspace 变化：`none`；未修改数据库、正式 workspace、服务、网络、外部发送或 Git refs。备份：`/tmp/less-is-more-pr12-backup/`。回滚点：本 PR 单提交 revert。
- 残余风险：取消 oracle 启动真实 `sleep`，依赖当前 POSIX/Windows shell 可执行环境；测试等待 options/pump instrumentation 后再取消，未放宽取消或进程组合同。

## 2026-07-23 less-is-more PR13：复用插件包 discovery 映射

### `PR13` `perf(plugins): reuse discovered package map`

- base：PR12 commit `9ad4daff3b1f79150e6007727f60846ab40a784b`，分支 `perf/less-is-more-pr13-reuse-plugin-package-discovery`。
- allowed_paths：`agent/plugins/packages.py`、`agent/plugins/manager.py`、`tests/test_plugin_packages.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：插件包 discovery 与 manager 同轮 enable 选择；未修改 dashboard public API/generation/snapshot/lease/event、插件 manifest schema 或 hot-reload 生命周期。
- 原问题与实现：`PluginManager.discover()` 已得到完整 package map，却再次从 project root 读取并解析每个 `package.toml`。保留 public `enabled_plugin_packages(project_root, entries)`，新增接收 `Mapping` 的私有 `_select_enabled_plugin_packages`，manager 将同轮 discovery map 直接传入；不跨轮缓存、不重复校验、不新增 catch/fallback。
- 不变量与错误语义：启用筛选的 insertion order、disabled member 过滤、原 capability conflict 文本与 fail-fast 传播保持不变；`Mapping` 接受只读视图但本轮 map 仍由 discovery 唯一拥有。插件 topology/order、package_id、manifest/provides schema、dashboard 调用路径均未改动。
- baseline：source-set digest `b86f87caa1b51f15c040fdfb3009c4f9f5eeef38d4a76cd09baecea370d098dd`，文件数 `384`，Python SLOC `78,622`，TypeScript/TSX SLOC `8,451`，total production SLOC `87,073`。
- candidate：source-set digest `e49e6f60681efe5b7dcfca789b5c4767b12f4b357ce20647d2517442802abf99`，文件数 `384`，Python SLOC `78,633`，TypeScript/TSX SLOC `8,451`，total production SLOC `87,084`；`agent` 从 `28,747` 增至 `28,758`，production 净增加 `11` 行，属于保留 public wrapper、Mapping helper 和类型边界的最小实现。
- 性能回放：同一 cwd fixture、同一 `.venv` 与临时 workspace/cache，500 轮 `PluginManager.discover()`；package.toml reads/parses 从 `4 → 2`/round（总计 `2,000 → 1,000`），两条路径各保持每轮一次读取；观察性 wall-time `219.672 → 171.849 ms`（约 `-21.77%`），仅作为本机 parser/discovery 观测，不宣称端到端 runtime 提速。
- 测试与静态验证：`pytest -q tests/test_plugin_packages.py tests/test_plugin_manager.py tests/test_plugin_hot_reload.py tests/test_plugin_skill_links.py` 为 `189 passed in 10.78s`；新增真实 manager `package.toml` 每路径一次读取 oracle，并断言默认 manifest 下 package members 被过滤且非包插件 topology/order/identity 保持；保留 public conflict/malformed/disabled 覆盖；目标文件 pyright `0 errors, 14 warnings`，与 base warning 数一致；migration append-only、`git diff --check` 与 production SLOC 通过。
- Gate：按 WORKFLOW 以 PR12 base 运行 committed-head 公开 Gate 并通过；private Gate 状态 `pending_maintainer`。
- 迁移/持久化/运行 workspace 变化：`none`；未修改数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event 或 Git refs。执行前备份：`/tmp/less-is-more-pr13-backup-QPS3Qt/`；回滚点为本 PR 单提交 revert。
- 残余风险：wall-time 仅覆盖同步 discovery/parser；若未来 discovery map 跨 round 共享或 package schema 改为有状态解析，必须重新核对 PLG-002 的 round snapshot 与错误 owner，不应把本 helper 变成跨轮缓存。

## 2026-07-23 less-is-more PR14：删除失效的 post-judge prompt helper

### `PR14` `refactor(prompts): remove dead post-judge helper`

- base：PR13 commit `b3614d6a38931bc264ce8135ea525753255edb61`，分支 `refactor/less-is-more-pr14-remove-dead-post-judge-prompt`。
- allowed_paths：`prompts/proactive.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：主动 prompt facade；未修改 active `ProactiveJudge`、`TOOL_SCHEMAS`、插件 manifest、runtime、generation、snapshot、lease、event 或 persistence。
- 历史与可达性：`build_post_judge_prompt_messages` 在 `909212b9` 随旧 compose/judge 链路引入；历史最后 caller 是 `proactive_v2/judge.py`，后续 `3eafe1ab` 删除旧 Judge 实现，`f91cf993` 将主动链路迁入插件并删除旧模块。当前 production/test/docs/sdk/plugin 全库精确搜索及动态 import/getattr 扫描均为零；当前 active judge 由 `plugins/proactive_flow/judge.py` 直接使用 `TOOL_SCHEMAS`，不再消费该 prompt。
- 范围与能力：仅删除 module-private `build_post_judge_prompt_messages`（prompt SLOC `465 → 420`）；保留 `build_compose_prompt_messages`、`plugins/proactive_flow/prompt.py` 的 active compose facade 与所有调用链。删除函数没有独立错误分支、日志、注释或持久化副作用；错误类型、外部调用、prompt schema、工具可见性和发送语义均不变。
- baseline：source-set digest `e49e6f60681efe5b7dcfca789b5c4767b12f4b357ce20647d2517442802abf99`，文件数 `384`，Python SLOC `78,633`，TypeScript/TSX SLOC `8,451`，total production SLOC `87,084`。
- candidate：source-set digest `c26d08dd0bdbddd13c971a5a26aef22a9cfa522dc3ec61ad61be61a6f7dae1d2`，文件数 `384`，Python SLOC `78,588`，TypeScript/TSX SLOC `8,451`，total production SLOC `87,039`；`prompts` 从 `465` 降至 `420`，production 净减少 `45` 行。
- 测试调整：无测试删除、无脆弱 symbol-absence 测试；保留 active compose prompt oracle；`pytest -q tests/test_proactive_prompts.py tests/test_proactive_facade_phase4.py tests/test_proactive_agent_tick_factory.py` 为 `15 passed in 0.88s`，相关 compileall 通过。
- 静态与边界验证：`prompts/proactive.py` pyright `0 errors, 0 warnings`；精确 symbol 搜索与 production/test/docs/sdk/plugin 动态 consumer 扫描均为零；migration append-only 与 `git diff --check` 通过。
- Gate：按 WORKFLOW 以 PR13 base 运行 committed-head 公开 Gate；private Gate 状态 `pending_maintainer`。
- 迁移/持久化/运行 workspace 变化：`none`；未修改数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event 或 Git refs。执行前备份：`/tmp/less-is-more-pr14-backup-eKfusa/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本仍可能保留旧 symbol 文本，但当前 Git source、插件、SDK、docs 和动态调用面已证实无 consumer；若未来恢复旧 judge 链路，应从 active plugin prompt owner 重新设计，不应复活该 helper。

## 2026-07-23 less-is-more PR15：删除不可达的 Memory2 去重决策器

### `PR15` `refactor(memory2): remove dead dedup decider`

- base：PR14 commit `b4c1176be8339680b3c2c8d964506cbc25c38ba3`，分支 `refactor/less-is-more-pr15-remove-dead-dedup-decider`。
- allowed_paths：`memory2/dedup_decider.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Memory2 默认记忆合并与检索链；未修改 `memory2/store.py`、`memory2/memorizer.py`、`plugins/default_memory/engine.py`、schema、manifest、测试或 SDK。
- 历史与可达性：`DedupDecider` 及其 `DedupDecision`、`MemoryAction`、`ExistingAction`、`DedupResult` 类型和两阶段向量/LLM 去重实现已由旧 Memory2 ingest 链路遗留。最后生产 caller 在 `1e19ce78` 已删除，当前 active replacement 是 `plugins/default_memory/engine.py → memory2/memorizer.py`；精确 import/symbol、production/test/docs/SDK/plugin/manifest/eval/script、dynamic import/getattr/export 与 cache consumer 扫描均无命中。该模块当前无 production、test、docs、SDK、plugin、manifest、eval、script、dynamic 或 cache consumer。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。仅删除不可达模块；正常记忆合并、supersede、reinforcement、semantic dedup、forget/undo、SQLite schema、write set、错误传播和 active engine 调用链保持不变。不添加 catch、fallback、兼容层或 symbol-absence 测试。
- 计量：模块删除前 source-set digest `c26d08dd0bdbddd13c971a5a26aef22a9cfa522dc3ec61ad61be61a6f7dae1d2`，文件数 `384`，Python SLOC `78,588`，`memory2` SLOC `4,855`，total production SLOC `87,039`；删除后 source-set digest `a363d259fddd0a1680d01ae641c3612eb184c67bfa0b673c80fa1928fd1d7fa6`，文件数 `383`，Python SLOC `78,330`，`memory2` SLOC `4,597`，total `86,781`。仓库脚本实际 production 净减少 `258` SLOC；删除文件内 `DedupDecider` class span 为 `189` SLOC（物理行 49–272）。
- Redis 式 God file 判断：不拆分 active `memory2/store.py`、`memory2/memorizer.py` 或 `plugins/default_memory/engine.py`；本 PR 仅移除无 owner、无调用且无状态写入的 dead module，不改变现有合并与持久化边界。
- 测试与真实验证：按调用面保留并运行 Memory2 baseline、semantic dedup、consolidation idempotency、memory engine contract 及 retrieval/forget/undo 相关回归；不修改测试。编译 active modules、修改范围 pyright、精确 consumer/dynamic scan、migration append-only、production SLOC 与 `git diff --check` 均作为提交验收项记录。
- Gate：按 WORKFLOW 以 PR14 base `b4c1176be8339680b3c2c8d964506cbc25c38ba3` 运行 committed-head 公开 Gate；本条不回填运行后的 `sourceDigest`/`planDigest`，避免账本自引用，最终报告摘要与 private 状态由交付报告记录。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event 或 Git refs。执行前备份：`/tmp/less-is-more-pr15-backup-T5mNK8/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧模块文本，但当前 Git source 与动态调用面没有 consumer；若未来恢复旧 ingest 链路，应从 active `memorizer` owner 重新设计，不复活该 dead decider。

## 2026-07-23 less-is-more PR16：删除不可达的 admitted tick helper

### `PR16` `refactor(proactive): remove dead admitted tick helper`

- base：PR15 commit `93c19ae4d6b3c831f500aea3f53a512ff90ff3b0`，分支 `refactor/less-is-more-pr16-remove-dead-tick-admitted`。
- allowed_paths：`proactive_v2/loop.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：ProactiveLoop snapshot admission、reload quiesce 与 kernel lease 生命周期；未修改 proactive tests、plugin manifest、runtime、snapshot/lease、event 或 SDK。
- 历史与可达性：`_tick_admitted` 在 `794db57d` 由 `_tick` 持有 `_reload_lock` 时引入；`9bb5aad6` 为避免 snapshot admission/reload quiesce 死锁，将 store lease acquire 移到锁外并把 admission/snapshot bind/reset/finally 逻辑内联到 `_tick`，helper 遂成为遗留定义。当前 production、tests、docs、SDK、plugin、manifest、eval、scripts、dynamic import/getattr/export 与 reflection 扫描均无 consumer，仅保留定义。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`；仅删除不可达 module-private helper。当前 `_tick` 的 no-store reload lock、store lease acquire 在 lock 外、bind/reset finally、snapshot switch、quiesce 不死锁、active kernel lease/error fail-loud 与 tick 结果均保持不变。不添加 absence test、catch、fallback 或兼容层。
- 计量：删除前 source-set digest `a363d259fddd0a1680d01ae641c3612eb184c67bfa0b673c80fa1928fd1d7fa6`，文件数 `383`，Python SLOC `78,330`，`proactive_v2` SLOC `2,777`，total production SLOC `86,781`；删除后 source-set digest、SLOC 与总量由 committed-head Gate/交付报告记录，删除仅影响 `proactive_v2/loop.py`。
- Redis 式 God file 判断：保留 `proactive_v2/loop.py`；删除无 owner、无调用且无状态写入的 dead helper，不拆分 `_tick`、`_switch_snapshot`、kernel lease 或 reload lock 状态机。
- 测试与真实验证：运行 `tests/proactive_v2/test_integration.py`、`tests/test_plugin_hot_reload.py::test_proactive_quiesce_does_not_deadlock_with_paused_tick`、`tests/test_plugin_hot_reload.py::test_proactive_tick_keeps_one_snapshot_generation` 及必要 proactive loop 回归；compileall、loop pyright、全仓 symbol/dynamic scan、migration append-only、production SLOC、`git diff --check` 与 committed-head Gate 均作为提交验收项记录。不修改测试或 absence oracle。
- Gate：按 WORKFLOW 以 PR15 committed HEAD `93c19ae4d6b3c831f500aea3f53a512ff90ff3b0` 运行；最终报告记录 `status`、`sourceDigest`、`planDigest`、production/protected 路径分组与 private Gate `pending_maintainer` 状态，不将运行后 digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event 或 Git refs。执行前备份：`/tmp/less-is-more-pr16-backup-20260723-043500/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧 helper 文本，但当前 Git source 与动态调用面没有 consumer；若未来恢复旧 admission 链路，应从 active `_tick` owner 重新设计，不复活该 dead helper。

## 2026-07-23 less-is-more PR17：避免 Telegram 锁的命中路径 eager 分配

### `PR17` `perf(telegram): avoid eager lock allocation`

- base：PR16 committed HEAD `9d37ce7ea21540564baa86fd519c0d266dea620c`，分支 `perf/less-is-more-pr17-telegram-lock-allocation`。
- allowed_paths：`infra/channels/telegram_utils.py`、`tests/test_telegram_utils.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Telegram 出站 limiter 与 live edit queue 的 per-chat 锁；未修改 retry/flood/clock/cleanup/API、锁 map 声明、数据库、网络或正式 workspace。
- 原问题与真实路径：Python 会先求值 `mapping.setdefault(key, asyncio.Lock())` 的默认参数，即使 key 已命中也构造并丢弃一个 `Lock`。四处命中分别是 `TelegramOutboundLimiter._chat_locks`、`_typing_locks`（`WeakValueDictionary`）以及 `TelegramLiveEditQueue._locks` 的 `reserve`/无 limiter `run` 分支（普通 `dict`）。
- 范围与语义：四处统一为 `lock = mapping.get(key)`，仅在 `None` 时构造并写回；没有改用 `defaultdict`，也未抽兼容 helper。`get` 与插入之间无 `await`，单事件循环下的 identity、同 chat 串行、RetryAfter/cooldown/backoff、flood/clock/cleanup 和外部发送顺序保持不变。`WeakValueDictionary` 弱生命周期与普通 dict 的具体类型均保持。
- 真实 oracle：新增测试用计数 Lock factory 预置 existing key，先执行 limiter send/typing 与 queue reserve/run 四条命中路径并断言 `constructions == 0`，再执行四条 miss 路径并断言总构造数为 `4`，同时保留 existing lock identity。未新增 absence、fallback 或 fake success 测试。
- baseline：source-set digest `483b61c61b7c5a1377a198ff4ea6e2ff98c062df41a3b72ff1780aa533c7b137`，文件数 `383`，Python SLOC `78,318`，`infra` SLOC `11,311`，total production SLOC `86,769`。
- candidate：source-set digest `8d5219c387c8b77a6619c1eca82087194a8cf5c8649380f5778b316f8b29442a`，文件数 `383`，Python SLOC `78,330`，`infra` SLOC `11,323`，total production SLOC `86,781`；production 净增加 `12` 行，均为四处显式 miss 分支。
- 同 workload microbench：使用相同 Python `3.13.7`、同一进程脚本，每个 map 跑 `10,000` 个 existing-key hit 加 `1` 个 miss，并计数 map lookup/hit/miss 与 Lock 构造；Weak chat/typing 两张 map 各为 `lookups 10,001 / hits 10,000 / misses 1 / constructions 10,001 → 1`，普通 dict 的 `reserve`/`run` 两张 map 同样为 `10,001 / 10,000 / 1 / 10,001 → 1`。聚合构造数：Weak `20,002 → 2`，普通 `20,002 → 2`。这是 lookup/allocation microbench，仅声明锁对象分配变化，不宣称网络吞吐或端到端 Telegram 延迟。
- 测试与静态验证：`pytest -q tests/test_telegram_utils.py tests/test_channel_clients.py` 为 `40 passed`；`python -m compileall -q infra/channels/telegram_utils.py tests/test_telegram_utils.py` 通过；`pyright --venvpath /mnt/data/coding/akasic-agent infra/channels/telegram_utils.py tests/test_telegram_utils.py` 为 `0 errors, 16 warnings`（均为 telegram_utils 既有第三方/动态类型告警，测试文件无诊断）；精确 `setdefault(... asyncio.Lock())` 搜索为零，consumer/type/Weak-map allocation/concurrency preservation oracle、migration append-only、`git diff --check` 均通过。
- Gate：按 WORKFLOW 在提交后以 committed HEAD 对 PR16 base `9d37ce7ea21540564baa86fd519c0d266dea620c` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后 digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、外部网络或 Git refs。执行前备份：`/tmp/less-is-more-pr17-backup-GxyTZW/`；回滚点为本 PR 单提交 revert。
- 残余风险：microbench 只覆盖 map lookup 与锁对象分配；若未来把 map 访问跨线程化或在 get/insert 之间引入 await，必须重新核对并发身份语义，不恢复 eager 分配作为同步手段。

## 2026-07-23 less-is-more PR18：删除不可达的 Memory2 injection planner

### `PR18` `refactor(memory2): remove dead injection planner`

- base：PR17 committed HEAD `0f1c8c132f65a1cae71b8d84f26f97a92256fd9c`，分支 `refactor/less-is-more-pr18-remove-dead-injection-planner`。
- allowed_paths：`memory2/injection_planner.py`（删除）、`tests/test_procedure_multi_query_retrieval.py`（删除）、`tests/test_hyde_enhancer.py`（删除 planner import、3 个 planner integration tests 及其专属夹具/import/过时注释）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Memory2 active default engine/retriever；未修改 active engine、retriever、default pipeline、query builder、HyDE module、schema、manifest 或 runtime。
- 历史与可达性：planner 在 `9c566dd5` 引入真实 procedure 多 query 规划，在 `ed0fe4b3` 接入 HyDE history/scoped fallback，`459a310c` 的 retrieval protocol 迁移后仅部分保留，`5801862b` 插件化记忆引擎后 active production caller 已迁入 `plugins/default_memory/engine.py` 与 `memory2/retriever.py`。当前 production、tests、docs、SDK、plugin、manifest、eval、script、dynamic import/getattr/export 与 reflection 搜索仅命中待删 planner 专属测试和定义，无其他 consumer。
- active replacement 与边界：active engine 已覆盖 procedure/preference 类型、`memory2/query_builder.py` 的原始+改写多 query、retriever 的 vector lane max-pool 和 keyword/vector RRF，以及 answer path 的双 hypothesis auxiliary query；它不完全复现旧 planner 的 scoped→global fallback 与 `scope_mode`/`+hyde` 标记。因此本次删除基于不可达 caller，不宣称旧 planner 与 active engine 语义完全相同；正常 active retrieval、scope、RRF、hypothesis、schema、write set 和 error propagation 保持不变。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。只删除无 owner、无 production caller 的 planner module 与 planner-only tests；保留四个独立 `HyDEEnhancer`/`_union_dedup` tests（timeout fallback、raw preservation、HyDE append、HyDE no-op），不新增 absence test、fallback、兼容层或 mock success 路径。
- 计量：删除前 source-set digest `8d5219c387c8b77a6619c1eca82087194a8cf5c8649380f5778b316f8b29442a`，文件数 `383`，Python SLOC `78,330`，`memory2` SLOC `4,597`，total production SLOC `86,781`；删除后 source-set digest `95ba009ac11ac3541546e2d48176fe7424d29b3dfa4f85ef3706125a53f81a2a`，文件数 `382`，Python SLOC `78,182`，`memory2` SLOC `4,449`，total production SLOC `86,633`；production 净减少 `148` SLOC。删除的测试源码不计入 production SLOC 或 30% 参考量，单独如实记录为一个 planner-only procedure retrieval 测试文件和 HyDE 文件中的三个 planner integration tests。
- 测试与静态验证：保留并运行 `tests/test_hyde_enhancer.py`、`tests/test_procedure_query_builder.py`、`tests/test_recall_memory_tool.py`、`tests/test_memory_engine_contract.py` 及 default retrieval/engine 相关回归；compileall active modules、pyright、exact/dynamic scan、migration append-only、production SLOC、`git diff --check` 与 committed-head Gate 作为提交验收项记录。
- Gate：按 WORKFLOW 以 committed HEAD 对 PR17 base `0f1c8c132f65a1cae71b8d84f26f97a92256fd9c` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后 digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、外部网络、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/less-is-more-pr18-backup-20260723-050115/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧 planner 文本；当前 Git source 与动态调用面没有 consumer。若未来需要 scoped fallback 或 HyDE scope metadata，应在 active retrieval owner 中重新设计并补独立合同，不复活 dead planner。

## 2026-07-23 less-is-more PR19：删除不可达的 Memory2 query rewriter

### `PR19` `refactor(memory2): remove dead query rewriter`

- base：PR18 committed HEAD `c6ae4254ab2ab647246fd9815a2fa2ea4b7d5f79`，分支 `refactor/less-is-more-pr19-remove-dead-query-rewriter`。
- allowed_paths：`memory2/query_rewriter.py`（删除）、`tests/test_query_rewriter.py`（删除）、`tests/test_query_rewriter_implicit_intent.py`（删除）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Memory2 active default engine/retriever；未修改 `memory2/query_builder.py`、`memory2/retriever.py`、`plugins/default_memory/engine.py`、`agent/retrieval/default_pipeline.py`、config/eval stale comments、schema、manifest 或 active runtime。
- 历史与不可达性：旧 rewriter 及其 `RETRIEVE`/`NO_RETRIEVE` XML、procedure lane 和 LLM timeout/fallback 逻辑来自旧 bootstrap wiring；`5801862b` 插件化记忆引擎后，`bootstrap/tools.py` 删除 QueryRewriter 构造、`MemoryServices` gate 字段和旧 facade，active pipeline 直接由 `MemoryQuery` 进入 `DefaultMemoryEngine`，当前 production、tests、docs、SDK、plugin、manifest、eval、script、动态 import/getattr/export 与 reflection 搜索仅命中待删模块和专属旧测试。
- active replacement 与边界：active engine 已覆盖正常 context/answer retrieval、procedure queries、原始+改写 query builder、scope、vector/keyword RRF、hypothesis auxiliary query、schema/write/error contracts；它不 1:1 复现旧 rewriter 的 RETRIEVE/NO_RETRIEVE/XML、procedure 改写、隐式意图和降级 lane，因此本次删除基于不可达 caller，不宣称旧 lane 与 active engine 语义等价，不修改 active retrieval 逻辑。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。只删除无 owner、无 production caller 的 rewriter module 与两份 planner-era/mock-LLM 专属测试；不加 absence test、fallback、兼容层或 mock success 路径。正常 context/answer retrieval、procedure queries、scope/RRF/hypothesis/schema/write/error 行为保持不变。
- 计量：删除前 source-set digest `95ba009ac11ac3541546e2d48176fe7424d29b3dfa4f85ef3706125a53f81a2a`，文件数 `382`，Python SLOC `78,182`，`memory2` SLOC `4,449`，total production SLOC `86,633`；删除后 source-set digest `0871a21ec224fb3f42852a25cd45889d7b37a0c7b27e8722e273ff01c3ccc770`，文件数 `381`，Python SLOC `77,864`，`memory2` SLOC `4,131`，total production SLOC `86,315`；production 净减少 `318` SLOC。删除的测试源码不计入 production SLOC，单独如实记录为两份 query-rewriter 专属测试文件，共 `299` 行测试源码。
- 测试与静态验证：运行 active `tests/test_procedure_query_builder.py`、`tests/test_memory_engine_contract.py`、`tests/test_recall_memory_tool.py`、`tests/test_turn_pipelines.py`、`tests/test_agent_core_p3_context_store.py` 及必要 default retrieval 回归；compileall active modules、pyright、exact/dynamic scan、migration append-only、production SLOC、`git diff --check` 与 committed-head Gate 作为提交验收项记录。不修改 active tests 或 absence oracle。
- Gate：按 WORKFLOW 在提交后以 committed HEAD 对 PR18 base `c6ae4254ab2ab647246fd9815a2fa2ea4b7d5f79` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后 digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、外部网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/less-is-more-pr19-backup-KjlHId/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧 rewriter 文本；当前 Git source 与动态调用面没有 consumer。若未来恢复 XML gate、隐式意图或 procedure rewrite lane，应在 active retrieval owner 中重新设计并补独立合同，不复活 dead rewriter。

## 2026-07-23 less-is-more PR20：删除不可达的 Memory2 sufficiency checker

### `PR20` `refactor(memory2): remove dead sufficiency checker`

- base：PR19 committed HEAD `c5865ac547928e702e4136971aace6d6c29c8ec8`，分支 `refactor/less-is-more-pr20-remove-dead-sufficiency-checker`。
- allowed_paths：`memory2/sufficiency_checker.py`（删除）、`tests/test_sufficiency_checker.py`（删除）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Memory2 active default engine/retriever；未修改 config/eval stale comments、engine/retriever/default pipeline/schema/manifest 或 active runtime。
- 历史与不可达性：旧 `SufficiencyChecker` 及其 `SufficiencyResult`、XML 判定和 refined-query retry 由 `core/memory/default_runtime_facade.py` 的 `_retry_empty_episodic_block` 消费；`5801862b` 插件化记忆引擎后 facade/wiring 已移除，当前 production、tests、docs、SDK、plugin、manifest、eval、script、动态 import/getattr/export 与 reflection 搜索仅命中待删模块和专属测试。当前 engine 直接由 `MemoryQuery` 进入 `_query_context`/`Retriever.retrieve`，不存在同一 sufficiency retry consumer。
- active replacement 与边界：active engine/retriever 继续负责 procedure/context/answer 检索、原始与辅助 query、scope、vector/keyword RRF、注入块、空结果和错误传播；它没有同旧 checker 一样的 sufficiency retry。本次删除基于不可达 caller，不宣称 replacement 或旧语义等价；不改变正常 active retrieval/result emptiness/error/schema/write set。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。严格删除无 owner、无 runtime caller 的 checker module 与专属 mock 测试；删除测试文件共 `203` 行，含 `14` 个测试，其中 `13` 个为 checker/LLM mock 测试，另 1 个为结果 dataclass 字段测试。不新增 absence/fallback、兼容层或 mock success 路径。
- 计量：删除前 source-set digest `0871a21ec224fb3f42852a25cd45889d7b37a0c7b27e8722e273ff01c3ccc770`，文件数 `381`，Python SLOC `77,864`，`memory2` SLOC `4,131`，total production SLOC `86,315`；删除后 source-set digest `e83c7ebea48eed3c08331defcff923ca32880e2f48fbca0f48754cc52888608c`，文件数 `380`，Python SLOC `77,702`，`memory2` SLOC `3,969`，total production SLOC `86,153`；production 净减少 `162` SLOC。删除的测试源码不计入 production SLOC。
- 测试与静态验证：运行 active `tests/test_memory_engine_contract.py`、`tests/test_recall_memory_tool.py`、`tests/test_turn_pipelines.py`、`tests/test_agent_core_p3_context_store.py`、`tests/memory2_retrieval_baseline.py` 及必要 empty-result/active retrieval 回归；compileall active modules、pyright、exact/dynamic scan、migration append-only、production SLOC、`git diff --check` 与 committed-head Gate 作为提交验收项记录。不修改 active tests 或 absence oracle。
- Gate：按 WORKFLOW 在提交后以 committed HEAD 对 PR19 base `c5865ac547928e702e4136971aace6d6c29c8ec8` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后 digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、外部网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/less-is-more-pr20-backup-Wh6Tpp/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧 checker 文本；当前 Git source 与动态调用面没有 consumer。若未来恢复 sufficiency/retry lane，应在 active retrieval owner 中重新设计并补独立合同，不复活 dead checker。

## 2026-07-23 less-is-more PR21：删除不可达的 source-ref 反投影 helper

### `PR21` `refactor(memory): remove dead source-ref helper`

- base：PR20 committed HEAD `224a4b7ccdfea1e68462aa34150aa41345aaf248`，分支 `refactor/less-is-more-pr21-remove-dead-source-ref-helper`。
- allowed_paths：`core/memory/utils.py`（删除）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：core memory utility projections；未修改 engine/schema/events/plugins/tests。
- 历史与不可达性：`source_ref_from_evidence` 由 `62fbcf8a` 引入，后续 `a99437d4`、`ebc4ecf8` 仅保留其定义；当前仓库静态/字符串/动态导出搜索、Git 历史调用面和外部 plugin cache 均无 caller。`RetrievalCompleted` 及 enabled observe 外部插件 consumer 不在本次范围，继续保留。
- active owner 与边界：`evidence_from_source_ref`、`resolve_memory_scope`、`should_require_scope_match` 和 `EvidenceRef`/`MemoryQuery`/`MemoryScope` imports 保持不变；当前 default engine 只使用前三者，删除项没有状态、写入或错误 owner。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。删除 11 个 production SLOC 的纯 projection helper；不新增 absence test、fallback、兼容层或 mock success 路径。
- 计量：删除前 source-set digest `e83c7ebea48eed3c08331defcff923ca32880e2f48fbca0f48754cc52888608c`，文件数 `380`，Python SLOC `77,702`，`core` SLOC `2,052`，total production SLOC `86,153`；删除后 source-set digest `03fef4862f9847229a890962632eac945fdc30cb7f53d93015c5103c45c2d10f`，文件数 `380`，Python SLOC `77,691`，`core` SLOC `2,041`，total production SLOC `86,142`；production 净减少 `11` SLOC。
- 性能与状态：仅减少模块导入时创建的一个不可达函数对象；无新增调用、分配、等待、I/O、持久化、网络、事件或 write set，不声明端到端性能收益。
- 测试与静态验证：项目 venv `pytest -q -W error tests/test_memory_engine_contract.py tests/test_recall_memory_tool.py tests/test_memory2_retrieval_baseline.py tests/test_memory2_consolidation_idempotency.py tests/test_memory2_dedup_baseline.py` 为 `79 passed`；修改文件 pyright `0 errors, 0 warnings`，相关 default engine 为 `0 errors, 43 warnings`（既有）；core memory/default engine/recall/memory contract compileall、精确 source/history/plugin-cache scan、migration append-only、`git diff --check` 均通过。
- Gate：按 WORKFLOW 在 committed HEAD 以 PR20 base `224a4b7ccdfea1e68462aa34150aa41345aaf248` 运行公开 Gate；private required 状态为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、外部网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/less-is-more-pr21-backup-20260723-053123/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧 helper 文本；当前 Git source、历史调用面与外部 plugin cache 没有 consumer。若未来需要从 `EvidenceRef` 反投影 source ref，应在 active memory owner 中重新设计并补独立合同，不复活 dead helper。

## 2026-07-23 less-is-more PR22：删除不可达的 Memory2 HyDE enhancer

### `PR22` `refactor(memory2): remove dead HyDE enhancer`

- base：PR21 committed HEAD `4146b3ce62e25800256379467102f2afc0da8ae4`，分支 `refactor/less-is-more-pr22-remove-dead-hyde-enhancer`。
- allowed_paths：`memory2/hyde_enhancer.py`（严格删除）、`tests/test_hyde_enhancer.py`（严格删除）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Memory2 active default engine/retriever；未修改 active `plugins/default_memory/engine.py::_query_answer/_gen_hypothesis`、`memory2/retriever.py`、`memory2/query_builder.py`、generic HyDE README/config/setup wizard/proactive tool/eval text、schema、manifest 或 cache。
- 历史与不可达性：旧 enhancer 的 raw 检索、LLM hypothesis、第二次检索与 union-dedup 由旧 injection planner 消费；插件化记忆引擎提交 `5801862b` 已移除 AgentLoop 构造与 wiring，PR18 又删除最后 planner consumer。当前 production、tests、docs、SDK、plugin、manifest、eval、script、dynamic import/getattr/export、reflection 与外部 plugin cache 的精确扫描仅命中待删模块、专属测试和历史 ledger 文本，current exact zero production consumer。
- active replacement 与边界：active engine 的 `_query_answer`/`_gen_hypothesis` 与 retriever/query builder 使用双 hypothesis auxiliary query、vector/keyword RRF 和现有 trace/schema/write/error contracts；旧 enhancer 的 raw+second retrieve+union_dedup 语义不同。本次删除严格基于 current exact zero production consumer，不宣称替代等价；正常 active hypothesis/retrieval/trace/schema/write/error 行为保持不变。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。只删除无 owner、无 production caller 的 enhancer module 与其专属测试全文件；不新增 absence/fallback、兼容层或 mock success 路径。
- 计量：删除前 source-set digest `03fef4862f9847229a890962632eac945fdc30cb7f53d93015c5103c45c2d10f`，文件数 `380`，Python SLOC `77,691`，`memory2` SLOC `3,969`，total production SLOC `86,142`；删除后 source-set digest `734b0df2222b000ec0a0f7dc02ba20e2440c55f7a7111b73a59e381f7d88c9ed`，文件数 `379`，Python SLOC `77,567`，`memory2` SLOC `3,845`，total production SLOC `86,018`；production 净减少 `124` SLOC。删除的专属测试源码为 `144` 物理行、`92` test SLOC，单独记账，不计入 production SLOC。
- 测试与静态验证：项目 `.venv` 的 active procedure/query/engine/recall/turn/context/retrieval 回归（10 个测试入口）为 `112 passed in 2.14s`；相关 active modules/tests compileall 通过；`plugins/default_memory/engine.py`、`memory2/retriever.py`、`memory2/query_builder.py` pyright 为 `0 errors, 43 warnings`（engine 既有告警）；精确 consumer/dynamic import/getattr/export 与外部 plugin cache scan 为零，migration/protected-path guard、production SLOC 与 `git diff --check` 通过。不修改 active tests 或 absence oracle。
- Gate：按 WORKFLOW 在提交后以 committed HEAD 对 PR21 base `4146b3ce62e25800256379467102f2afc0da8ae4` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后 digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/less-is-more-pr22-backup-20260723-053803/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧 enhancer 文本，但 current Git source 与动态调用面已证实无 consumer；旧 raw+second retrieve+union_dedup 与 active 双 hypothesis auxiliary/RRF 不同，若未来需要 HyDE 增强，应在 active retrieval owner 中重新设计并补独立合同，不复活该 dead enhancer。

## 2026-07-23 less-is-more PR23：删除不可达的 MemoryItem 模型

### `PR23` `refactor(memory2): remove dead MemoryItem model`

- base：PR22 committed HEAD `9e422e045025ac23e798abd3821a2b763abf7a3e`，分支 `refactor/less-is-more-pr23-remove-dead-memory-item`。
- allowed_paths：`memory2/models.py`（严格删除）、`tests/test_more_support_modules.py`（删除唯一专属 import 与构造测试块）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：Memory2 active `MemoryHit`/`MemoryStore2` model boundary；未修改 `MemoryHit`、`MemoryStore2`、active default-memory engine、schema、migration、manifest、plugin 或 cache。
- 历史与不可达性：`MemoryItem` 随 Memory2 初始提交 `4c6cf0d1` 引入；生产层唯一 import 在 `9ce0155b` 的 store migration 中移除，当前 committed source 的精确静态、AST/字符串、动态 import/getattr/export/reflection、tests、SDK、plugin、manifest、eval/script 与 `/home/huashen/.akashic-plugin/cache` 扫描仅命中待删模块和 `tests/test_more_support_modules.py` 的专属 import/构造块；当前 production exact-zero consumer。active `MemoryHit` 是现有检索与 post-response owner，不能由同名旧 dataclass 推断为替代迁移。
- 语义与错误边界：`MemoryItem` 只保存未使用的 dataclass 字段，没有持久化、schema、事务、事件、错误或外部调用 owner；删除不改变 active `MemoryHit`/`MemoryStore2` 的字段、SQL、序列化、检索、写入和错误传播。`semantic_delta: none`，不新增 fallback、兼容层、absence oracle 或 mock success 路径。
- 范围与测试：删除 `memory2/models.py` 全文件 17 个物理行/15 个 production SLOC，并删除测试中唯一专属 import 与 15 行构造断言；其余 `test_bootstrap_trigger_and_entrypoints_cover_paths` 仍覆盖真实 supervisor、migration 和 CLI 入口，不因旧模型 fixture 继续耦合。
- 计量：删除前 source-set digest `734b0df2222b000ec0a0f7dc02ba20e2440c55f7a7111b73a59e381f7d88c9ed`，文件数 `379`，Python SLOC `77,567`，`memory2` SLOC `3,845`，total production SLOC `86,018`；删除后 source-set digest `3c7c839f7ce8119cc8a681fe939caee818e100a944a19629acd35fed3a8c1b39`，文件数 `378`，Python SLOC `77,552`，`memory2` SLOC `3,830`，total production SLOC `86,003`；production 净减少 `15` SLOC。
- 性能与状态：仅减少一个不可达模块导入时的 dataclass/type 对象和对应测试构造；无新增调用、分配、等待、I/O、持久化、网络、事件、write set 或 schema 变化，不宣称端到端性能收益。
- 测试与静态验证：项目 venv `pytest -q -W error tests/test_more_support_modules.py tests/test_memory_engine_contract.py tests/test_recall_memory_tool.py tests/test_memory2_retrieval_baseline.py tests/test_memory2_consolidation_idempotency.py tests/test_memory2_dedup_baseline.py` 为 `105 passed in 1.63s`；production/active memory compileall 通过；pyright 为 `0 errors, 152 warnings`（均为既有 memory2/default-engine 动态类型告警）；`tests/test_production_sloc.py tests/semantic/test_change_gate.py` 为 `19 passed`；精确 consumer/dynamic/export/plugin-cache scan、migration append-only 与 `git diff --check` 通过。
- Gate：在 committed HEAD 对 PR22 base `refactor/less-is-more-pr22-remove-dead-hyde-enhancer` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/less-is-more-pr23-backup-d5KZO3/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧模型文本，但当前 Git source、历史生产迁移与外部 plugin cache 没有 consumer；若未来需要新的 persisted memory DTO，应在 active storage owner 中重新设计并补独立合同，不复活 dead `MemoryItem`。

## 2026-07-23 less-is-more PR24：删除 DB 迁移后遗留的 safe-filename helper

### `PR24` `refactor(session): remove dead safe filename helper`

- base：PR23 committed HEAD `5b5a27531899f445f9a4e8248e67e513bfa3ec4a`，分支 `refactor/less-is-more-pr24-remove-dead-safe-filename`。
- allowed_paths：`session/manager.py`（删除旧 JSONL 路径迁移后孤儿 `_safe_filename` 与不再使用的 `re` import）、`tests/test_logic_modules.py`（删除该 helper 的唯一 import 与专属断言）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：`SessionStore`/`SessionManager` SQLite session persistence；未修改 `SessionStore`、`session_dir`、数据库 schema/migration、public API、`plugin_undo` 或 active session paths。
- 历史与不可达性：`_safe_filename` 及其 `_get_session_path` JSONL consumer 属旧文件会话实现；提交 `708d6f25` 已将 `SessionManager` 切换到同一 `sessions.db` 的 `SessionStore`，并移除 `_get_session_path`、JSONL glob/list/load/write 路径，当前 `session/manager.py` 只剩无调用 helper。当前 committed source 的 CodeGraph、AST、精确字符串/动态 import/getattr/export/reflection、tests、SDK、plugin、manifest、eval/script 与 `/home/huashen/.akashic-plugin/cache` 扫描无 production consumer；外部 workspace subagent-run 旧副本不属于 canonical source/cache，未作为 owner 证据。
- 语义与错误边界：SQLite session key 直接作为 `SessionStore` 查询键；删除 helper 不改变 session key 校验、`sessions.db` 写入/读取、metadata、消息顺序、错误传播或文件路径。`semantic_delta: none`，不新增 fallback、兼容层、absence oracle 或 mock success 路径。旧 JSONL 文件不会被迁移、删除或重新解释。
- 范围与测试：删除 helper 5 个物理行及 `re` import 1 行；删除 `tests/test_logic_modules.py` 中唯一专属 import 1 行与断言/空行 2 行。保留该测试中的真实 `SessionManager` SQLite save/load/list/channel metadata 覆盖，不因历史 helper 继续耦合。
- 计量：删除前 source-set digest `3c7c839f7ce8119cc8a681fe939caee818e100a944a19629acd35fed3a8c1b39`，文件数 `378`，Python SLOC `77,552`，`session` SLOC `2,529`，total production SLOC `86,003`；删除后 source-set digest `a62e24b567959b49a7d8cd412c3555c8996ccbeeb156e1d61d5bf5ed1ed34ab1`，文件数 `378`，Python SLOC `77,549`，`session` SLOC `2,526`，total production SLOC `86,000`；production 净减少 `3` SLOC。测试源码不计入 production SLOC。
- 性能与状态：仅减少 `session.manager` 导入时的一个不可达正则函数对象与 `re` 模块引用；无新增调用、分配、等待、I/O、持久化、网络、事件、write set 或 schema 变化，不宣称端到端性能收益。
- 测试与静态验证：项目 venv `pytest -q -W error tests/test_logic_modules.py tests/test_session_store.py tests/test_message_lookup_tool.py tests/test_migration_append_only.py tests/test_migration_runner.py` 为 `105 passed in 2.06s`；`compileall` session/相关 tests 通过；`pyright --venvpath /mnt/data/coding/akasic-agent session/manager.py tests/test_logic_modules.py` 为 `0 errors, 0 warnings`；精确 consumer/dynamic/export/plugin-cache scan、AST parse、migration/protected-path guard、production SLOC 与 `git diff --check` 通过。`session/store.py` 全文件 pyright 仍有 54 条既有 warnings、0 errors，未修改该文件。
- Gate：按 WORKFLOW 在 committed HEAD 对 PR23 base `5b5a27531899f445f9a4e8248e67e513bfa3ec4a` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/akasic-agent-pr24-session-manager.py.bak-20260723`、`/tmp/akasic-agent-pr24-test-logic-modules.py.bak-20260723`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 与外部 workspace subagent-run 可能保留旧 helper 文本，但当前 Git source、708d6f25 后的迁移路径与外部 plugin cache 没有 consumer；若未来恢复 JSONL session storage，应在持久化 owner 中重新设计 key/path 合同并补独立迁移，不复活 dead helper。

## 2026-07-23 less-is-more PR25：删除迁移后遗留的 BeforeReasoning lifecycle event

### `PR25` `refactor(lifecycle): remove dead BeforeReasoning event`

- base：PR24 committed HEAD `565bbe66a27914996a384c7c4374df3214b55075`，分支 `refactor/less-is-more-pr25-remove-dead-before-reasoning-event`。
- allowed_paths：`bus/events_lifecycle.py`（删除孤儿 `BeforeReasoning` dataclass 与其专属 `_empty_skill_names` factory）、`docs/refactor/clean-code-ledger.md`；`capability_owner`：agent lifecycle 的 `BeforeReasoningCtx` phase；未修改 `agent.lifecycle.types.BeforeReasoningCtx`、BeforeReasoning phase/module chain、EventBus、插件事件映射、其他 lifecycle events、schema、manifest、runtime 或 tests。
- 历史与不可达性：`BeforeReasoning` 是旧被动 turn 事件；提交 `6759d288` 将被动 turn 迁移到 `agent.lifecycle.types.BeforeReasoningCtx` 与 phase chain，并从 `passive_turn.py` 移除旧事件 import/emit。当前 committed source 的 CodeGraph、AST、精确字符串、动态 import/getattr/export/reflection、tests、SDK、plugin、manifest、eval/script 与 `/home/huashen/.akashic-plugin/cache` 扫描无 `bus.events_lifecycle.BeforeReasoning` consumer；现有同名文本均指向 active `BeforeReasoningCtx` phase 或错误消息，不是待删 dataclass。
- 语义与错误边界：旧 dataclass 只承载 session/channel/chat/content/skills/retrieved block，已没有事件总线 owner、生产发布者或订阅者；active `BeforeReasoningCtx` 继续由 `BeforeReasoningPhase` 构造、允许插件修改并经 `EventBus` 发送。删除不改变 active phase 的字段、插件回调、abort 传播、prompt warmup、工具上下文同步、生命周期顺序或错误暴露。`semantic_delta: none`，不新增 absence test、fallback、兼容层或 mock success 路径。
- 范围与测试：删除 `_empty_skill_names` 4 个物理行与 `BeforeReasoning` dataclass 10 个物理行；没有 dedicated test 或仅针对该旧类的 import，保留所有 `BeforeReasoningCtx` phase、agent-core、plugin-manager 和 internal event 回归。
- 计量：删除前 source-set digest `a62e24b567959b49a7d8cd412c3555c8996ccbeeb156e1d61d5bf5ed1ed34ab1`，文件数 `378`，Python SLOC `77,549`，`bus` SLOC `926`，total production SLOC `86,000`；删除后 source-set digest `65d0d478690fd3457af8847887e4560392256a35bd70d5eb60be0527fb9e21c4`，文件数 `378`，Python SLOC `77,539`，`bus` SLOC `916`，total production SLOC `85,990`；production 净减少 `10` SLOC。
- 性能与状态：模块导入时不再创建一个不可达 dataclass 类型与其 list factory；无新增调用、分配、等待、I/O、持久化、网络、事件、write set 或 schema 变化，不宣称端到端性能收益。
- 测试与静态验证：项目 venv `pytest -q -W error tests/test_internal_events.py tests/test_more_support_modules.py tests/test_support_modules.py tests/test_lifecycle_phase.py tests/test_lifecycle_phases.py tests/test_agent_core_p5_agent_core.py tests/test_plugin_manager.py` 为 `173 passed in 5.23s`；相关 bus/agent lifecycle/core/tests `compileall` 通过；pyright 为 `0 errors, 144 warnings`（均为 agent/passive 与 lifecycle 的既有动态类型告警）；`tests/test_production_sloc.py tests/semantic/test_change_gate.py` 为 `19 passed`；migration append-only、exact consumer/dynamic/export/plugin-cache scan、AST parse、production SLOC 与 `git diff --check` 通过。
- Gate：按 WORKFLOW 在 committed HEAD 对 PR24 base `565bbe66a27914996a384c7c4374df3214b55075` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/akashic-less-is-more-pr25-before-reasoning-events-lifecycle.py.bak`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部 workspace subagent-run 可能保留旧事件文本，但当前 Git source、6759d288 后的 active lifecycle 迁移路径与外部 plugin cache 没有 consumer；若未来恢复旧事件总线契约，应在 active lifecycle owner 中重新设计并补独立迁移，不恢复该 dead dataclass。

## 2026-07-23 less-is-more PR26：内联事件 dataclass 的空容器工厂

### `PR26` `refactor(bus): inline builtin event default factories`

- base：PR25 committed HEAD `58a5763992367c735eb3114b49b9be87bb8e031d`，分支 `refactor/less-is-more-pr26-inline-empty-factories`。
- allowed_paths：`bus/events.py`、`bus/events_lifecycle.py`、`docs/refactor/clean-code-ledger.md`；`capability_owner`：bus 事件 dataclass 的实例默认容器；未修改字段名、字段顺序、类型注解、事件类、EventBus、插件事件映射、schema、manifest 或 tests。
- 原问题与可达性：七个 `_empty_*` 函数只作为 dataclass `Field.default_factory` 的唯一 consumer；`bus/events.py` 的 `_empty_media`/`_empty_metadata` 各被两个字段引用，`bus/events_lifecycle.py` 的 `_empty_media` 无字段 consumer（PR25 已删除其唯一 `BeforeReasoning` consumer），其余四个 factory 各只被 `TurnCommitted` 字段引用。仓库、AST、动态 import/getattr/export、reflection、SDK、tests、插件 manifest 与 `/home/huashen/.akashic-plugin/cache` 精确扫描没有 private factory consumer；外部插件只读取事件公开字段。
- 范围与语义：`change_type: refactor`，`semantic_delta: none`。删除 private wrappers，将 `default_factory` 直接改为带精确类型参数的 builtin `list[...]`/`dict[...]` GenericAlias；每次 dataclass 构造仍创建独立的空 list/dict，字段名、顺序、注解、repr/compare/frozen、事件发布与消费保持不变。唯一可观察的 reflection 变化是 `dataclasses.fields(...).default_factory` 从 module-private function 变为 builtin-origin `list[...]`/`dict[...]`；没有发现对 factory identity 的真实 consumer，故不引入兼容别名或 fallback。
- 性能与错误/注释变化：删除七个不可达或仅转发空容器的函数对象及其调用层；模块导入少创建七个函数对象，实例化仍为同阶 builtin 容器分配，不声明端到端提速。没有新增异常捕获、默认值归一化、日志或注释；内部契约继续 fail-fast。
- 计量：删除前 source-set digest `65d0d478690fd3457af8847887e4560392256a35bd70d5eb60be0527fb9e21c4`，文件数 `378`，Python SLOC `77,539`，`bus` SLOC `916`，total production SLOC `85,990`；删除后 source-set digest `b4ac1af9d07d23f933ef471f8bfad26befa8db18834faa0b78bfe758f2a159eb`，文件数 `378`，Python SLOC `77,525`，`bus` SLOC `902`，total `85,976`；production 净减少 `14` SLOC（GenericAlias 改写不增加 production SLOC）。
- 测试与静态验证：事件、support、lifecycle、plugin、mobile channel 定向回归 `326 passed in 14.31s`；`tests/test_production_sloc.py tests/semantic/test_change_gate.py tests/test_migration_append_only.py` 为 `24 passed`；相关 `compileall` 通过；精确类型参数后的修改文件 pyright `0 errors, 0 warnings`；AST/private consumer scan、外部 plugin-cache scan、独立实例容器 identity 隔离与 builtin-origin factory reflection 均通过，`git diff --check` 通过。
- Gate：按 WORKFLOW 在 committed HEAD 对 PR25 base `58a5763992367c735eb3114b49b9be87bb8e031d` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/akashic-less-is-more-pr26/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部 workspace subagent-run 可能保留旧 private helper 文本，但当前 canonical source、dynamic consumer、reflection consumer 与外部 plugin cache 没有依赖；若未来需要对外承诺 factory identity，应在公开事件契约层另行设计，而不恢复 module-private wrappers。

## 2026-07-23 less-is-more PR27：删除 PersonaMem runtime 转发 wrapper

### `PR27` `refactor(eval): remove PersonaMem runtime forwarding wrapper`

- base：PR26 committed HEAD `3766dda949802391b97be656d3aea9efc020d544`，分支 `refactor/less-is-more-pr27-remove-personamem-runtime-wrapper`。
- allowed_paths：删除 `eval/personamem/runtime.py`；修改 `eval/personamem/run.py`、`eval/personamem/run_one_case.py`、`eval/personamem/run_one_qa.py` 与本账本；`capability_owner`：PersonaMem benchmark runner 的 runtime wiring；未修改 CLI 参数、`persona_profile` 数据集字段/加载、结果 schema、`eval/longmemeval/runtime.py`、ingest/QA、错误关闭语义、schema、manifest 或 tests。
- 原问题与可达性：`eval/personamem/runtime.py` 只有一个七个 SLOC 的 async forwarding function：接收 `persona_profile` 后立即丢弃，再以相同的两个参数调用 `eval.longmemeval.runtime.create_runtime`，并转发 `close_runtime`。三个 PersonaMem 入口是其全部 caller；CodeGraph、AST、精确字符串、动态 import/getattr/export/reflection、SDK、plugin、manifest、eval 与外部 plugin cache 扫描确认没有其他 wrapper consumer。`persona_profile` 仍由 `dataset.py` 解析并保留在 `PersonaMemInstance`，但不再穿过已知无副作用的 wrapper；三个调用实参均等价为 `(config_path, workspace)`。
- 语义与错误边界：`create_runtime`/`close_runtime` 直接绑定 LongMemEval 的 canonical owner；初始化、SELF.md、provider、consolidation、workspace、资源关闭、日志和异常传播保持原函数实现。删除 wrapper 不增加 fallback、try/except、默认值、兼容层或 mock success；CLI 参数、persona profile 解析、ingest/QA 调用顺序、result JSON 字段与输出保持不变，`semantic_delta: none`。
- 范围与计量：删除 wrapper 的七个 eval SLOC，并把三个 caller 切换为 canonical 两参 API；raw diff 为 `7 insertions(+), 16 deletions(-)`，`eval/personamem` 逻辑 SLOC 从 `864` 降至 `857`（净减少 `7`）。项目 canonical production source-set 按 `scripts/measure_production_sloc.py` 排除 `eval/**`，因此总 production SLOC、文件数与 digest 保持 PR26 的 `85,976`、`378`、`b4ac1af9d07d23f933ef471f8bfad26befa8db18834faa0b78bfe758f2a159eb`；本 PR 的七行减少属于 benchmark runner 代码，另行记录避免混淆计量口径。
- 性能与状态：PersonaMem 运行路径少一层 async forwarding/call frame 和一个被丢弃参数绑定；没有新增模型调用、SQL、I/O、等待、持久化、网络、事件、write set 或 schema 变化，不宣称端到端 benchmark 提速。
- 测试与静态验证：项目 venv `pytest -q -W error tests/test_personamem_eval.py` 为 `4 passed`；`eval/personamem` 与 `eval/longmemeval` `compileall` 通过；修改入口及 canonical runtime Pyright 为 `0 errors, 0 warnings`；CLI parser/import smoke、AST caller contract（3 个 direct imports，所有 `create_runtime` 均为两参）、精确 wrapper/persona_profile runtime scan、migration append-only 与 `git diff --check` 通过。仓库没有 LongMemEval runtime 专属测试文件，故以 compile/import smoke 覆盖该 canonical owner；未修改 `tests/test_personamem_eval.py`。
- Gate：按 WORKFLOW 在 committed HEAD 对 PR26 base `3766dda949802391b97be656d3aea9efc020d544` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、外部网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/akashic-less-is-more-pr27-backup-1784758748/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 可能保留旧 wrapper 文本，但 current canonical source、调用面与外部 plugin cache 没有 consumer；若未来 PersonaMem 需要把 persona profile 注入 runtime，应在 benchmark owner 中设计显式参数与语义合同，不恢复无效转发层。

## 2026-07-23 less-is-more PR28：删除 Akasha core 中四个不可达 DB helper

### `PR28` `refactor(akasha): remove dead core database helpers`

- base：PR27 committed HEAD `e229001bdf9983e07410cf6dbcdad7343baab7aa`，分支 `refactor/less-is-more-pr28-remove-dead-akasha-core-helpers`。
- allowed_paths：`plugins/akasha/core.py` 仅删除 `message_id_to_key_from_db`、`open_source_db`、`get_turn_context`、`load_state` 四个 exact-zero 顶层函数；本账本；`capability_owner`：Akasha active engine/store/replay 的核心算法与读取 owner。未修改 active `engine._affected_turn_keys`/`_load_turn_card`、`replay._turn_messages`、`AkashaStore.list_nodes`/`load_edges_with_meta`/`_load_graph_cache`、共享 `sqlite3`/`turn_key`/`deserialize_f32` imports、SQL、embedding/cache、错误处理、schema、migration、manifest、tests 或 canonical 外部仓库。
- 历史与不可达性：四个 helper 在首次引入 `6310c87e` 后的 Git 历史中只保留定义，`git log -S` 与初始提交源码核对未发现生产 caller。当前 PR27 worktree 的 CodeGraph、AST call/attribute/import scan、精确文本/动态 `getattr`/export/reflection scan 仅命中待删定义；`engine.py`、`store.py`、`replay.py` 的 active owner 使用各自现有读取路径。只读扫描 `/mnt/data/coding/akasha`（当前有维护者未提交改动）及 `/home/huashen/.akashic-plugin/cache` 未发现四个名称的 consumer；缓存中无对应 enabled plugin owner。
- 语义与错误边界：这些函数分别是旧 message-id 反查（含未命中时返回原 ID 的降级）、sqlite-vec 源库开启、展示用 user/assistant 文本裁切和旧 sidecar 全量状态读取；当前没有调用者、导出或反射 owner，因此删除不会改变 active turn identity、replay card、节点/边加载、向量反序列化、SQL/write set、错误传播或持久化结果。`change_type: refactor`，`semantic_delta: none`；不删除仍被 `engine/store/replay` 使用的共享数据库和算法 helper，不新增 fallback、try/except、兼容层、absence oracle 或 mock success 路径。
- 范围与计量：删除 `plugins/akasha/core.py` 107 个物理行、97 个非空非注释行；按 `scripts/measure_production_sloc.py`，删除前 source-set digest `b4ac1af9d07d23f933ef471f8bfad26befa8db18834faa0b78bfe758f2a159eb`、文件数 `378`、Python SLOC `77,525`、`plugins` SLOC `17,229`、total production SLOC `85,976`；删除后 digest `8dc2e7c38387a8ae910ea75a87765c5ab8b525627e263aab93095b894d3e4a86`、文件数 `378`、Python SLOC `77,432`、`plugins` SLOC `17,136`、total `85,883`；production 净减少 `93` SLOC。无测试源码变更。
- 性能与状态：模块导入时少创建四个不可达函数对象；旧 helper 的 sqlite connection、cursor、查询和向量扩展加载路径不再可从当前代码触发。没有新增模型调用、SQL、I/O、等待、持久化、网络、事件、write set 或 schema 变化，不宣称端到端性能收益；正式 workspace、数据库和缓存均未写入。
- 测试与静态验证：项目 venv `pytest -q -W error tests/test_akasha_plugin.py tests/test_fast_rebuild_parity.py` 为 `68 passed in 1.02s`；相关 replay/full-run/provider migration 回归 `29 passed in 1.48s`；`tests/test_production_sloc.py tests/semantic/test_change_gate.py tests/test_migration_append_only.py` 为 `24 passed in 2.07s`；`plugins/akasha` 与相关 tests `compileall` 通过；修改文件 `plugins/akasha/core.py` pyright `0 errors, 0 warnings`，engine/store/replay 联合检查 `0 errors, 2 warnings`（replay.py:291/294 既有 unknown-type warnings）；AST/精确 consumer/dynamic/export/plugin-cache scan、protected-path/migration/SLOC、`git diff --check` 通过。
- Gate：待提交后以 committed HEAD 对 PR27 base `e229001bdf9983e07410cf6dbcdad7343baab7aa` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/akashic-less-is-more-pr28-backup-1784759365/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint、`/mnt/data/coding/akasha` 的未提交副本或外部运行日志可能保留旧 helper 文本，但 current PR27 source、历史 Git consumer scan 与 plugin cache 没有 owner；若未来需要 message-id 映射、展示卡片或 sidecar loader，应在 active engine/store/replay owner 中重新设计显式合同，不恢复这些无调用的旧 helper。

## 2026-07-23 less-is-more PR29：删除 proactive tools 的不可达终止 schema 与 execute wrapper

### `PR29` `refactor(proactive): remove dead tools surface`

- base：PR28 committed HEAD `56232addfb7364f849140f7af6c0dc41cc00e2e9`，分支 `refactor/less-is-more-pr29-remove-dead-proactive-tools-surface`。
- allowed_paths：`plugins/proactive_flow/tools.py` 仅删除 exact-zero `TERMINAL_TOOL_SCHEMAS` 与顶层 `execute()` 转发 wrapper；`tests/proactive_v2/test_tools.py` 删除 `execute` 专属测试 9 项并将未知工具错误测试直接绑定 `dispatch`；本账本；`capability_owner`：proactive judge 的 `ToolExecutionRequest`→`dispatch` 工具执行边界。未修改 `TOOL_SCHEMAS`、`dispatch` 分支、judge/tool executor、工具参数/结果、步骤计数 owner、plugin manifest、schema 或运行 workspace。
- 历史与不可达性：`TERMINAL_TOOL_SCHEMAS` 与 `execute()` 随迁移提交 `f91cf993` 保留，但当前生产 judge 在 `plugins/proactive_flow/judge.py:150-158` 由 `ToolExecutor.execute` 直接回调 `dispatch`；CodeGraph、AST、精确 import/attribute、动态 `getattr`/`import_module`/export/reflection、tests、SDK、plugin/manifest 与 `/home/huashen/.akashic-plugin/cache`、`/mnt/data/coding/akasha` 扫描均未发现旧 schema/wrapper consumer。`git log -S` 复核确认迁移后只剩定义与专属测试；wake 的同名 `execute` 属独立插件工具，不在本次范围。
- active owner 与语义：`TOOL_SCHEMAS` 仍由 ProactiveJudge 发送给 LLM；`dispatch` 仍拥有 12 个工具的分支、unknown-tool `ValueError` 和真实工具错误传播；`ctx.steps_taken` 仍由 judge 在构造 `ToolExecutionRequest` 前递增，不依赖已删除 wrapper。`TERMINAL_TOOL_SCHEMAS` 没有 registry/count/manifest owner。`change_type: refactor`，`semantic_delta: none`；不新增 fallback、try/except、默认值、兼容层或 mock success。
- 范围与测试：删除 `TERMINAL_TOOL_SCHEMAS` 5 个 production SLOC 与 `execute()` wrapper 3 个 production SLOC；删除 9 个仅验证 wrapper 转发/计数的测试，保留并改为直接验证 `dispatch` unknown-tool fail-fast 的测试及其余真实 helper/schema/tool contract 测试。生产逻辑、步骤计数、schema 数量、错误路径与外部副作用均不变。
- 计量：删除前 source-set digest `8dc2e7c38387a8ae910ea75a87765c5ab8b525627e263aab93095b894d3e4a86`，文件数 `378`，Python SLOC `77,432`，`plugins` SLOC `17,136`，total production SLOC `85,883`；删除后 source-set digest、SLOC 与 total 由 committed-head Gate 绑定记录，预期 `plugins` SLOC `17,128`、total `85,875`，production 净减少 `8` SLOC。测试删除的 wrapper 专属源码不计入 production SLOC。
- 性能、错误与注释：模块导入少创建一个无调用 schema 列表和一个 async 转发函数；judge 热路径不增加调用、分配、等待、I/O、持久化、网络或事件，避免一层无效 await/call frame，不宣称端到端提速。删除 stale `execute` 注释并保留 `dispatch` 阶段注释；没有放宽错误处理，unknown tool 继续 fail-fast。
- 测试与静态验证：项目 venv `pytest -q tests/proactive_v2/test_tools.py` 为 `60 passed in 0.18s`；judge/proactive tick/plugin lifecycle 定向回归 `127 passed in 3.35s`；相关 compileall 通过；`plugins/proactive_flow/tools.py`、`judge.py` pyright `0 errors, 164 warnings`（既有动态 dict warnings，base 同范围 171 warnings，无新增 error）；精确 consumer/dynamic/export/plugin-cache scan、migration append-only 与 `git diff --check` 通过。未删除真实 `dispatch`/schema/工具 contract 覆盖。
- Gate：在 committed HEAD 对 PR28 base `56232addfb7364f849140f7af6c0dc41cc00e2e9` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、SQLite、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/akashic-less-is-more-pr29-backup-qj4rUK/`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧 schema/wrapper 文本，但 current source、迁移历史、dynamic scan 与 external plugin cache 没有 consumer；若未来需要 terminal-only schema projection，应由 judge/registry owner 重新设计显式公开合同，不恢复无调用 surface。

## 2026-07-23 less-is-more PR30：删除 looping constants 的不可达配置残片

### `PR30` `refactor(loop): remove dead constants from looping module`

- base：PR29 committed HEAD `12abbe5e6754283ea9f616cc7dd64c8f784be118`，分支 `refactor/less-is-more-pr30-remove-dead-looping-constants`。
- allowed_paths：`agent/looping/constants.py` 仅删除 `_tool_call_signature` import、`_SAFETY_RETRY_RATIOS`、`_MAX_TOOL_RESULT_CHARS`、`_TOOL_LOOP_REPEAT_LIMIT`、`_SUMMARY_MAX_TOKENS`、`_RETRIEVE_TRACE_SUMMARY_MAX`、`_INCOMPLETE_SUMMARY_PROMPT` 及其专属注释/空白；本账本；`capability_owner`：looping history-route 常量 owner。保留 `_FLOW_TRIGGER_WORDS`、`_FLOW_SEQUENCE_PATTERN` 及 `agent/policies/history_route.py` 的现有导入和行为；未修改 `passive_turn`/`subagent` 中同名 active 常量、core 兼容 re-export、history_route P0、retry/trim/tool-loop/stream。
- 历史与不可达性：这些残片在 `constants.py` 中只有定义/import，仓库 AST、精确文本、CodeGraph、`getattr`/`import_module`/动态文件加载、导出/re-export、测试、SDK、插件 manifest 与 `/home/huashen/.akashic-plugin/cache` 扫描均无 consumer；Git 历史显示它们来自旧 looping/runtime prompt 搬迁，当前安全重试、工具结果截断和不完整摘要分别由 `agent/core/passive_turn.py` 与 `agent/subagent.py` 的 active owner 定义。唯一现存 `agent.looping.constants` import 仍精确读取 flow-route 两个保留符号。
- 语义与错误边界：`semantic_delta: none`。删除不可达定义不改变 history-route 的关键词/正则判定、route decision、LLM 调用、超时、fallback、异常传播或任何工具循环/重试/裁切行为；没有新增 fallback、try/except、动态兼容层或默认值。正常 import 少创建七个无消费者对象/字符串常量，未改变可达路径和持久化、网络、事件或 write set。
- 范围与计量：删除 `agent/looping/constants.py` 16 个物理行；PR29 base source-set digest `21bf798339056cdabe75da06f5b6ca6ba3cd9dd267f36359f91ee375cf20c104`、文件数 `378`、Python SLOC `77,424`、total production SLOC `85,875`；candidate source-set digest `47bccf4f112ea8e0ac0212f88298406bd1a9975a23be2de1fc175973dfa18435`、文件数 `378`、Python SLOC `77,412`、total `85,863`；production 净减少 `12` SLOC。
- 性能与注释：模块导入不再创建不可达函数依赖、tuple/int/字符串和多行 prompt；保留 flow-route 需要的简洁常量，删除其余 stale 注释，不宣称端到端性能收益。
- 测试与静态验证：定向 history-route、delegation、internal-events、spawn-completion/spawn-tool 回归与 active import smoke 通过；`agent/looping/constants.py`、`agent/policies/history_route.py`、相关 loop/reasoner 文件 compileall 与 Pyright 通过；全库精确 symbol/dynamic/cache scan、production SLOC、migration append-only 与 `git diff --check` 通过。无测试源码变更。
- Gate：按 WORKFLOW 在 committed HEAD 对 PR29 base `12abbe5e6754283ea9f616cc7dd64c8f784be118` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送、schema、manifest 或 Git refs。执行前备份：`/tmp/akashic-agent-pr30-constants.py.bak`、`/tmp/akashic-agent-pr30-ledger.md.bak`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 或外部未跟踪副本可能保留旧常量文本，但 current source 与 enabled plugin cache 没有 consumer；若未来需要新的 loop owner，应在 active `passive_turn`/`subagent` 或明确新模块中设计合同，不恢复无调用残片。

## 2026-07-23 less-is-more PR31：内联主动上下文的记忆读取

### `PR31` `refactor(proactive): inline prompt memory reads`

- base：PR30 committed HEAD `c98849813a954b5383b96bfd4af3bab9fb8b262d`，分支 `refactor/less-is-more-pr31-inline-proactive-prompt-memory-reads`。
- allowed_paths：`plugins/proactive_flow/prompt.py` 仅将 `_read_self_text`、`_read_long_term_text` 内联到唯一 caller `ProactivePromptBuilder.build_runtime_context_message`，删除两个 module-private helper；本账本。未修改 prompt 文本、区块顺序、`MemoryProfileApi`、active proactive engine/judge、plugin lifecycle、schema、manifest、缓存或 tests。
- 历史与唯一 caller：CodeGraph 与 AST/精确调用扫描确认两个 helper 各只有 `build_runtime_context_message` 的一处调用；仓库字符串扫描、动态 import/getattr/export/reflection 与 `/home/huashen/.akashic-plugin/cache` 扫描无其他 consumer；`git log -S` 只显示旧定义迁移提交 `f91cf993`、`869260b8`，未发现额外生产 caller。删除后两个 helper 名称在当前源（账本外）为零残留。
- 语义与错误边界：`memory = self._memory` 后，`None` 分支仍将 self/long/recent 三个区块置空；非空分支严格按 self→long→recent 顺序执行 `str(value or "").strip()`，异常原样传播。内联不改变 prompt section 名称、内容、顺序、时钟、workspace 或 gateway 数据；`semantic_delta: none`，不新增 fallback、try/except、默认值、兼容层或 mock success 路径。
- 性能与注释：删除两个无独立 owner 的 module-private forwarding frame 与函数对象；active caller 少一层 Python call/return，读取调用数、分配、等待、I/O、持久化、网络、事件和 write set 不变，不宣称端到端提速。保留阶段 2 注释，不新增冗余注释或错误处理。
- 范围与计量：删除前 source-set digest `47bccf4f112ea8e0ac0212f88298406bd1a9975a23be2de1fc175973dfa18435`，文件数 `378`，Python SLOC `77,412`，`plugins` SLOC `17,128`，total production SLOC `85,863`；删除后 digest `5ffee260c8a177bb2c5dfed4e7ec90e6a91993acb1415d20f51ce0d44a09adbe`，文件数 `378`，Python SLOC `77,404`，`plugins` SLOC `17,120`，total `85,855`；production 净减少 `8` SLOC。
- 测试与静态验证：prompt/proactive context、judge/tool、plugin lifecycle 与 agent loop 定向回归 `172 passed in 1.60s`，lifecycle builder/kernel/factory `22 passed in 1.11s`；production SLOC、migration append-only 与 change-gate 选择 `24 passed in 1.73s`；prompt/相关测试 compileall 通过；pyright `0 errors, 12 warnings`，与 PR30 base 完全一致；`git diff --check` 通过。对 PR30 base 与 candidate 使用相同固定 tick/gateway/memory 输入，`None`、普通值、falsey 值的完整 prompt output 字节相同；调用序列均为 `self → long → recent`，self 失败时只调用 self 且原异常文本相同。Black diff 仅报告 base 已存在的其他三处格式差异，PR31 新增区块无需重排。
- Gate：待 committed HEAD 对 PR30 base `c98849813a954b5383b96bfd4af3bab9fb8b262d` 运行公开 Gate；private required 状态记录为 `pending_maintainer`，不把运行后的 source/plan digest 回填到账本以避免 source 自引用。
- 迁移/持久化/运行 workspace 变化：`none`；未修改 migration、数据库、正式 workspace、服务、网络、外部发送、generation/snapshot/lease/event、schema、manifest 或 Git refs。执行前备份：`/tmp/less-is-more-pr31-prompt.py.bak`；回滚点为本 PR 单提交 revert。
- 残余风险：历史 checkpoint 可能保留旧 helper 文本，但当前 canonical source、CodeGraph/AST、历史调用面与 enabled plugin cache 无 consumer；若未来需要独立复用记忆读取，应在 active memory owner 中重新设计明确合同，不恢复无调用 wrapper。

## 基线

- 基准提交：`3b456e7b`（PR #109 合并后）
- Python 测试：`1484 passed`，耗时 22.55 秒
- Pyright：`0 errors, 3119 warnings`
- 前端 TypeScript：`npm run typecheck` 通过
- 前端 ESLint：`0 errors, 3 warnings`，均为 `frontend/dashboard/src/main.tsx` 的既有 React Hook 依赖警告
- 工作区：除本地 `.codegraph/` 外无未提交代码
- 关键历史约束：PR #105 全能力热重载、PR #109 事件流唤醒、PR #90 主动发送串行、PR #89 shell 超时取消、PR #75 memory fail-stop

## 验收原则

1. 重构默认保持外部行为；能力变化必须明确列出并由测试覆盖。
2. 性能优化必须记录修改前后的同一 workload 数据，并证明 freshness、hot reload、错误传播和一致性未退化。
3. 删除或保留防御性检查时，必须说明不变量、拥有层、上游保证和真实可达违反路径。
4. 测试只保留能够保护真实契约、历史回归或性能边界的内容；删除测试必须记录其重复、错误耦合或已失效的原因。
5. God file 是否拆分以阅读成本为准，不以行数为准。若拆分增加跨文件跳转、隐藏弱类型数据流或割裂同一状态机，应保留同文件并在函数级整理。
6. 新增或改写的 docstring 与注释使用简洁中文；保留解释约束、所有权和 workaround 的有效注释。

## 变更记录模板

### `f82be7b6` `perf(runtime): 回收空闲聊天通道状态`

- 范围：`bus/queue.py` 的 `ChatLane` 与直接回归测试。
- 历史依据：PR #90 固化被动优先、主动 FIFO 和取消 ticket 语义；PR #97 固化中断恢复边界。本次没有触碰 lifecycle、interrupt 或 turn 内容。
- 原问题：`ChatLane._states` 永久保留历史见过的 chat，唯一 chat 数持续增长时形成无界内存占用。
- 为什么这样修改：为每次公开操作成对持有状态引用，只在没有活跃用户、被动计数、发送、未完成 ticket 和取消残留时回收；等待者持有引用，因此不会与新进入者分裂到两个状态锁。
- 不变量与拥有层：`active_users` 由 `_acquire_state` / `_release_state` 唯一维护；FIFO 和取消 ticket 仍由同一 `_ChatLaneState` 拥有。
- 能力变化：串行、FIFO、被动优先、取消恢复和异常传播不变；空闲 chat 不再保留无语义状态。
- 性能变化：20,000 个唯一 chat 顺序执行 pending/done 后，保留状态由 20,000 降至 0；当前 tracemalloc 由 32,702,434 B 降至 374 B，峰值由 32,703,122 B 降至 3,026 B。
- 测试新增：覆盖 FIFO 完成、取消 waiter、被动生命周期和发送异常后的回收。
- 测试删除及原因：无。
- 验证结果：相关子系统 `48 passed`；修改文件 pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：回收依赖 asyncio 单线程事件循环中 acquire/release 之间无 `await` 的原子执行语义；跨线程调用不在 `ChatLane` 契约内。

### `3b962903` `fix(memory): 暴露向量存储故障`

- 范围：`memory2/retriever.py` 的统一向量检索链及直接回归测试。
- 历史依据：PR #23/#61 统一召回与 memory engine 协议；PR #75/#80 确立 memory 失败只有在存在明确恢复动作时才能恢复；PR #106 保证唯一 Memory Engine。
- 原问题：批量向量存储失败会被宽泛捕获，随后逐向量重复同一存储调用并继续吞错，最终把存储或反序列化故障伪装成空召回。
- 为什么这样修改：`MemoryStore2.vector_search_batch` 已拥有 sqlite-vec/full-scan 选择、时间过滤、反序列化和批量结果形状；Retriever 没有第二种恢复手段，应让该层错误向上传播。
- 不变量与拥有层：非空 vectors 必须获得同长度 outer result，由 `MemoryStore2` 保证；embedding 是外部边界，单 lane embedding 失败仍可跳过并保留关键词检索。
- 能力变化：正常 vector + keyword + RRF、零向量命中后的关键词召回、scope、top-k 和时间过滤不变；存储损坏由静默空结果变为显式失败。
- 性能变化：正常路径调用次数不变；故障路径由 `1 + N` 次重复存储调用收敛为 1 次后立即失败；生产代码净减少 21 行。
- 测试新增：覆盖向量存储失败向上传播且不会继续执行关键词 lane。
- 测试删除及原因：无。
- 验证结果：独立复验 59 个相关测试通过；修改文件 pyright `0 errors`，总 warning 由 150 降至 128；`git diff --check` 通过。
- 残余风险：该变化会让过去被误判为“无记忆”的存储故障显式中止 recall，这是预期错误语义修复。

### `9bb4913d` `perf(plugins): 复用热重载发现快照`

- 范围：`PluginManager.reconcile_changed` / `_prepare_changed` 与多插件热重载测试。
- 历史依据：PR #51 的拓扑依赖、PR #95 的代际 Skill Catalog、PR #104 的程序化能力声明、PR #105 的 generation/snapshot/lease/rollback 事务。
- 原问题：一次 reconciliation 已发现完整拓扑，之后每个活跃插件和每个变化候选又重复完整 `discover()`，调用次数为 `1 + N + C`。
- 为什么这样修改：同一发布事务应使用同一个 discovery topology；watcher 的 revision 在事务外采样，中途变化会在下一轮形成新 revision，不需要在同一事务内部漂移拓扑。
- 不变量与拥有层：单轮 topology 由 reconciliation 拥有；源码 revision、candidate gate、snapshot 编译和下一轮 freshness 仍由原有层负责。
- 能力变化：同轮一致性增强；generation、gate、snapshot、lease、drain、abort、rollback 和下一轮 hot reload 不变。
- 性能变化：两个活跃插件同时变化时 `discover()` 从 5 次降至 1 次，减少 80%；一般情况从 `1 + N + C` 降至固定 1 次。
- 测试新增：在既有多插件换代测试中增加 discover 次数断言，同时保留最终 snapshot 包含两个新 generation 的能力断言。
- 测试删除及原因：无；复用已有昂贵 fixture，避免新增重复测试。
- 验证结果：`137 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：单轮扫描后的文件变化不会混入当前事务，而由 watcher 下一轮重新 reconcile；这是 PR #105 的代际一致性边界。

### `c845327b` `fix(runtime): 暴露主动发送异常`

- 范围：主动发送的 `PushToolOutboundPort`、`TurnOrchestrator` 与直接测试。
- 历史依据：PR #90 的 ChatLane/outbound 串行链路，PR #97 的中断恢复与可见历史可信边界，PR #27/#31 的 persist/dispatch 和 lifecycle 职责。
- 原问题：端口把所有意外异常静默转换成 `False`，无法区分正常业务失败与 channel/tool 故障；同时用字符串归一化掩盖内部 `OutboundDispatch` 契约错误。
- 为什么这样修改：端口传播意外异常，由拥有恢复动作的 orchestrator 记录完整堆栈、保持 `sent=False`、禁止未送达消息落库并执行失败副作用。
- 不变量与拥有层：channel/chat_id/content/media 的结构由 `OutboundDispatch` 构造链拥有；“目标和内容可发送”仍由端口判断；失败恢复由 orchestrator 拥有。
- 能力变化：正常文本、多媒体发送与业务失败字符串不变；意外异常从无诊断 `False` 变为有堆栈的原失败路径；ChatLane 串行和持久化顺序不变。
- 性能变化：非性能提交，发送次数和调用顺序不变。
- 测试新增：覆盖端口异常传播，以及 orchestrator 记录错误、不落库并运行 failure effect。
- 测试删除及原因：无。
- 验证结果：Runtime/turn 子系统 `125 passed`；pyright `0 errors`，4 个既有容器类型 warning；`git diff --check` 通过。
- 残余风险：多媒体分批发送中后续图片失败时，用户可能已收到前序内容但整次 dispatch 仍判失败；这是既有非事务性外部发送语义，本提交未扩大范围。

### `a661c5f9` `fix(memory): 保持向量索引降级一致性`

- 范围：`MemoryStore2` 的 sqlite-vec 初始化、写入和删除故障路径。
- 历史依据：PR #72 的 embedding 维度配置、PR #41/#61 的单一 Memory runtime/engine、PR #75/#80 的显式失败与可恢复边界。
- 原问题：`vec_items` 写入或删除失败后 `_vec_enabled` 仍为真，主表与加速索引分叉，后续可能漏召回或继续触发 `OperationalError`。
- 为什么这样修改：`memory_items` 是 canonical 数据，`vec_items` 只是可选索引；store 层有明确恢复动作，应禁用已不可信索引并复用现有 fullscan。
- 不变量与拥有层：主表/索引同步和降级由 `MemoryStore2` 拥有；只处理 `sqlite3.Error`，embedding blob 等内部程序错误继续传播。
- 能力变化：正常 sqlite-vec KNN、排序、scope、hotness、事务和 freshness 不变；索引故障由错误或漏召回变为较慢但正确的 fullscan。
- 性能变化：正常路径不变；故障路径牺牲索引速度换取 canonical 正确性，不宣称提速。
- 测试新增：故障注入覆盖 vec 写入与删除失败，验证主表写入/删除结果和 fullscan 一致。
- 测试删除及原因：无。
- 验证结果：Memory 子系统 `124 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：禁用持续到进程重启，不自动重建损坏索引；这是避免不一致索引重新上线的保守语义。

### `ece6c837` `fix(plugins): 暴露 active 状态检查故障`

- 范围：插件 `is_active()` 协议边界与真实临时插件测试。
- 历史依据：PR #104 的程序化能力声明；PR #106 的单 Memory Engine active 过滤。
- 原问题：插件 `is_active()` 抛错后 runtime 记录 warning 并返回 `True`，把无法判断状态的插件错误加入 active generation 和 Drift skill roots。
- 为什么这样修改：runtime 无法从任意插件异常推导正确启用状态，只能补充插件身份并链式重抛。
- 不变量与拥有层：插件实现合法 `is_active()`；runtime 负责调用协议和错误上下文；未声明该方法仍按既有规则默认启用。
- 能力变化：正常 true/false 与缺失方法语义不变；故障插件由错误启用改为明确失败，generation/snapshot/lease/drain/rollback 未触及。
- 性能变化：非性能提交，正常调用次数不变。
- 测试新增：真实临时插件覆盖 `PluginManager.active_plugins()` 与 `RuntimeSnapshot.active_generations()` 的 cause 链。
- 测试删除及原因：无。
- 验证结果：相关 plugin 子系统 `145 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：第三方插件的 `is_active()` 旧错误现在会阻止状态枚举，这是预期 fail-loud 行为。

### `dffb1f69` `refactor(runtime): 收紧工具解锁结果边界`

- 范围：`ToolDiscoveryState` 的 tool-search JSON 解析与直接测试。
- 历史依据：PR #27/#31 的 lifecycle/tool discovery 阶段边界，PR #48 的工具循环与无限迭代能力。
- 原问题：宽泛 `except Exception` 会把解析函数内部程序错误也伪装成“没有工具可解锁”；现有英文 docstring 还保留无助于当前理解的搬迁历史。
- 为什么这样修改：JSON 语法和结构是明确外部边界，只恢复 `JSONDecodeError`、非对象顶层和非列表 `matched`；领域层继续过滤空名称与重复名称。
- 不变量与拥有层：输入参数的 `str` 类型由内部调用契约拥有；JSON 结构由解析边界拥有；工具名非空与去重由 `ToolDiscoveryState` 拥有。
- 能力变化：非法 JSON、`[]`、`null`、`matched=null` 仍不解锁工具；合法 unlocked/matched 顺序和去重不变；内部非 JSON 错误不再静默。
- 性能变化：非性能提交，仍是一次 JSON decode 和一次线性遍历。
- 测试新增：参数化覆盖合法 JSON 中的三种错误顶层/字段结构。
- 测试删除及原因：无。
- 验证结果：相关子系统 `58 passed`；pyright `0 errors`，无新增 warning；`git diff --check` 通过。
- 残余风险：旧的 `dict` 裸容器类型仍存在于同模块其他协议，已拒绝在本提交中用 `Any` 顺手掩盖，留给独立类型设计。

### `70f79c60` `refactor(plugins): 删除旧描述符声明标记`

- 范围：`ActivePluginInfo` 和直接构造测试。
- 历史依据：PR #96 引入旧 `.aka-plugin/plugin.json` descriptor；PR #104 明确删除 descriptor 并迁移到 `plugin.py` 程序化声明。
- 原问题：`declares_aka_plugin` 已无生产读取者，却继续暗示 runtime 支持已删除协议，并让测试持续构造无意义参数。
- 为什么这样修改：删除无主不变量和测试样板，不增加兼容层。
- 不变量与拥有层：插件能力声明只由当前程序化 `plugin.py` 协议拥有。
- 能力变化：无运行行为变化；skill/MCP 装配和 generation/snapshot/lease/rollback 未触及。
- 性能变化：非性能提交。
- 测试新增：无。
- 测试删除及原因：未删除测试，只移除四处失效构造参数。
- 验证结果：相关 plugin 测试 `78 passed`；pyright `0 errors` 且无新增 warning；字段全库搜索零残留；`git diff --check` 通过。
- 残余风险：无已知残余；若未来重新支持 descriptor，应以新协议显式设计，而不是恢复布尔标记。

### `ba83aab2` `refactor(runtime): 收紧出站总线契约`

- 范围：`BusOutboundPort`、真实 `MessageBus` 测试夹具和直接出站测试。
- 历史依据：PR #90 的 MessageBus/ChatLane 被动出站链，PR #27/#31 的 after-turn dispatch 边界。
- 原问题：端口用 `Any + inspect.isawaitable` 兼容不存在的同步 bus，并对 typed dataclass 容器重复提供空值 fallback；测试的 `MagicMock` 反向维持了假契约。
- 为什么这样修改：生产构造链保证 `MessageBus`，其 `publish_outbound` 明确为 async；直接 await 真实契约并让发布异常继续传播。
- 不变量与拥有层：bus 类型由 `AgentLoopDeps`/bootstrap wiring 拥有；metadata/media 非空容器由 `OutboundDispatch` dataclass 拥有。
- 能力变化：channel、chat_id、content、thinking、metadata、media、ChatLane 计数和异常传播不变；测试与生产异步契约一致。
- 性能变化：删除动态 awaitable 判断和无效 fallback，但未做稳定 benchmark，不声明性能收益。
- 测试新增：使用真实 MessageBus 验证完整 typed `OutboundMessage`。
- 测试删除及原因：无；将违反生产契约的 MagicMock 夹具改为真实 bus。
- 验证结果：相关 runtime/turn 测试 `36 passed`；修改文件 pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：直接测试读取 MessageBus 私有队列以避免启动长期 dispatch loop；生产 API 语义仍由 publish/dispatch 集成测试覆盖。

### `8d1c4589` `fix(session): 暴露 metadata 损坏`

- 范围：`sessions.metadata` 数据库反序列化边界、SessionManager 转发层和三条读取入口测试。
- 历史依据：`708d6f251` 将 JSONL session 迁移到中心 SQLite；PR #75/#80 确立无恢复动作时的 fail-stop。
- 原问题：损坏 JSON 被 manager 宽泛捕获并归一化为整个 channel 空列表；合法 JSON list/string 会穿透到下游 `.get()` 才无上下文失败。
- 为什么这样修改：store 在读取 SQLite 时统一解析并验证 JSON object，错误携带 session key；manager 信任边界后的 dict。
- 不变量与拥有层：metadata JSON schema 由 `SessionStore` 拥有；NULL 是 schema 允许的旧记录，继续明确解释为 `{}`；identity index 无修复损坏数据的能力。
- 能力变化：有效 metadata、NULL 兼容、排序、cache 和 identity 映射不变；损坏数据由空结果或延迟错误变为带 key 的即时 `ValueError`。
- 性能变化：仍为一次 SQL 查询和每行一次 JSON 解析，无新增 I/O。
- 测试新增：注入损坏 JSON 和非 object JSON，覆盖 channel metadata、单 session metadata、dashboard 列表三个入口。
- 测试删除及原因：无。
- 验证结果：Session 相关调用方 `82 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：数据库中已有损坏 metadata 会在首次读取时显式暴露，需要人工修复数据；这是预期行为。

### `6f50a391` `test(runtime): 使用真实异步消息总线`

- 范围：所有直接构造 `AgentLoopDeps` 的测试夹具。
- 历史依据：`ba83aab2` 收紧 `BusOutboundPort` 后的集成回归。
- 原问题：10 处测试用同步 `MagicMock` 伪造生产中明确为异步 `MessageBus` 的依赖，其中两条 spawn completion 流程在完整测试中触发 `TypeError`。
- 为什么这样修改：统一改用真实 `MessageBus`，让测试遵循生产构造契约，不恢复同步兼容层。
- 不变量与拥有层：bus 类型与 async publish 由 `AgentLoopDeps`/`MessageBus` 拥有。
- 能力变化：无生产行为变化；测试现在能覆盖真实出站类型。
- 性能变化：非性能提交。
- 测试新增：无。
- 测试删除及原因：无；替换错误夹具。
- 验证结果：相关测试 `49 passed`，完整测试 `1497 passed`；pyright `0 errors, 0 warnings`；全库同类 `bus=MagicMock()` 搜索零残留。
- 残余风险：这笔修复证明目标测试不足以验收公共契约变更；后续公共类型收紧必须运行完整测试。

### `6c7a4ba5` `fix(plugins): 校验 KV 根节点结构`

- 范围：`PluginKVStore._read()` 数据文件反序列化边界与真实磁盘测试。
- 历史依据：插件 KV 可被用户、旧版本和外部插件绕过正常 `_write()` 直接修改。
- 原问题：合法 JSON array/scalar 会穿透边界，在后续 `.get()` 或赋值处以无文件上下文的异常失败。
- 为什么这样修改：KV 根节点必须是 JSON object；在唯一读取边界校验并以包含文件路径的 `ValueError` 失败，非法 JSON 继续保留 `JSONDecodeError`。
- 不变量与拥有层：KV object schema 由 `PluginKVStore._read()` 拥有；正常 `_write()` 始终写入 dict。
- 能力变化：正常 get/set/increment 和跨 manager 持久化不变；错误更早且带路径；plugin generation/snapshot 状态机未触及。
- 性能变化：非性能提交，正常路径仅增加一次 `isinstance`。
- 测试新增：真实 `.kv.json` 数组根节点拒绝测试。
- 测试删除及原因：无。
- 验证结果：相关 plugin 测试 `142 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：已有非 object KV 文件会在首次读取时显式失败，需要插件作者修复数据。

### `9d449162` `fix(session): 校验缓存向量维度`

- 范围：`MessageEmbeddingStore` 的向量写入与 `sessions.db` 缓存反序列化边界。
- 历史依据：PR #109 引入共享 message embedding cache，要求 cache hit 表示可直接复用的完整向量。
- 原问题：写入允许空 embedding；读取忽略持久化 `dim`，空 BLOB 会被错误计为 cache hit，BLOB/dim 不一致会按实际字节静默解码。
- 为什么这样修改：upsert 拒绝空向量；读取统一校验 BLOB 类型、正整数 dim 和 `len(blob) == dim * 4`，错误携带 message/model/dim/bytes。
- 不变量与拥有层：非空向量由 upsert 写边界拥有；持久化 BLOB/dim 一致性由读取边界拥有；元素数值错误继续由 `struct.pack` fail-loud，不重复检查。
- 能力变化：合法 cache、content hash miss、时间 cutoff、replay 顺序和 legacy migration 不变；空向量和损坏缓存变为即时失败。
- 性能变化：SQL 次数不变，正常读取增加常数级类型与长度比较，不宣称提速。
- 测试新增：空 embedding 写入拒绝且无缓存残留；空 BLOB 和维度/字节不一致覆盖 get/list/list_until。
- 测试删除及原因：无。
- 验证结果：Akasha/replay 相关 `84 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：已有损坏 cache 会阻止 replay，需删除或重建对应缓存；这是避免错误 cache hit 的预期行为。

### `943820ee` `refactor(runtime): 收紧回合副作用契约`

- 范围：`TurnResult` 三类副作用集合、`TurnOrchestrator` 执行边界和直接测试替身。
- 历史依据：PR #27/#31 将副作用放在明确的 lifecycle/commit 阶段；PR #90/#97 要求保持发送顺序，并禁止未送达消息进入历史。
- 原问题：副作用以 `list[Any]` 表示，orchestrator 用 `inspect.isawaitable` 兼容没有生产调用者的同步假实现。
- 为什么这样修改：现有生产副作用全部实现异步 `TurnSideEffect` 协议；将三类集合收紧到该协议并直接 await，让协议错误即时暴露。
- 不变量与拥有层：副作用的异步调用契约由 `TurnSideEffect` 拥有；通用、成功和失败副作用的选择与次序由 orchestrator 拥有。
- 能力变化：通用副作用仍先于 dispatch；成功/失败副作用仍只进入对应分支；单项异常仍记录并继续；持久化和 ChatLane 语义不变。
- 性能变化：删除一次动态 awaitable 判断，但无独立 benchmark，不声明性能收益。
- 测试新增：无；唯一同步测试替身改为真实异步协议。
- 测试删除及原因：无。
- 验证结果：相关 Runtime/proactive 测试 `144 passed`；副手完整测试 `1501 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：无已知生产同步副作用；未来扩展必须显式实现协议。

### `e16f2dcc` `fix(plugins): 拒绝无效清理动作`

- 范围：`PluginScope.defer()` 动态插件边界、`PluginContext` cleanup/task 类型和直接测试。
- 历史依据：PR #105 的候选初始化、回滚和 generation 换代要求资源清理动作在候选发布前有效。
- 原问题：动态外部插件可绕过静态类型注册不可调用对象，错误延迟到卸载或换代时才暴露，候选甚至可能已经发布。
- 为什么这样修改：在 cleanup 唯一注册入口验证 callable，并携带 plugin/resource 身份抛出 `TypeError`；同时把 context 类型收紧为 `Cleanup` 和 `Task[T]`。
- 不变量与拥有层：进入 scope 栈的 cleanup 必须可调用，该不变量由 `PluginScope.defer()` 唯一拥有；静态类型不能覆盖动态插件边界。
- 能力变化：合法同步/异步 cleanup、逆序排空、取消传播、task/process 跟踪不变；无效候选在 initialize/rollback 阶段提前失败。
- 性能变化：正常注册仅增加一次常数级 callable 检查，不声明性能收益。
- 测试新增：动态注册不可调用 cleanup 的边界测试。
- 测试删除及原因：无；generation/snapshot/lease/drain/abort/rollback 测试全部保留。
- 验证结果：plugin 相关测试 `145 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：`manager.py` 的候选 gate 和 watcher retry 属于更大的状态协议，本提交未改动。

### `7b4b7821` `refactor(schedule): 收紧时间展示降级边界`

- 范围：调度工具注册后的时间展示、历史任务列表展示及直接测试。
- 历史依据：PR #52 的 scheduler 后台任务隔离；PR #79/#89 的轮询与取消边界；PR #107 的 MCP 超时透传均未触及。
- 原问题：任务成功注册后，展示阶段用宽泛 `except Exception` 把内部程序错误也伪装成正常 ISO fallback。
- 为什么这样修改：只恢复 datetime/时区格式化真实会产生且当前位置能处理的 `TypeError`、`ValueError`、`OverflowError`、`OSError`；历史失效时区额外处理 `ZoneInfoNotFoundError`。
- 不变量与拥有层：调度参数结构由工具输入边界拥有；`ScheduledJob.fire_at` 的 datetime 契约由 scheduler 构造/反序列化层拥有；展示层只拥有格式降级。
- 能力变化：合法注册、循环任务和取消不变；无效展示时区/request_time 仍回退 ISO；违反内部 job 契约的错误改为显式失败。
- 性能变化：非性能提交。
- 测试新增：无效字符串/错误类型 request_time 和历史失效时区的展示回退。
- 测试删除及原因：无。
- 验证结果：定向 `39 passed`；副手完整测试 `1505 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：ToolRegistry 当前不主动调用 schema validator，错误类型参数仍可从动态调用进入工具，因此该 TypeError 恢复路径真实可达。

### `6cc15427` `fix(proactive): 暴露会话读取故障`

- 范围：`Sensor` 的普通/主动历史读取、时间戳解析、配置与返回类型及直接测试。
- 历史依据：PR #103 的 Gate→Fetch→Judge→Resolve→Deliver 次序；PR #101 的 Drift 时钟；PR #67 的 read-only 主动召回。
- 原问题：sessions SQLite 关闭、schema 或加载故障被两个入口宽泛捕获并返回空列表，普通链误判为无上下文，主动链还可能绕过去重造成重复投递。
- 为什么这样修改：Sensor 没有恢复数据库故障的能力；让错误传播到 `ProactiveLoop._tick_bound()` 现有的完整日志与重抛边界，仅保留非法旧时间戳到 `None` 的明确字段级恢复。
- 不变量与拥有层：Session 持久化错误由 SessionManager/Store 拥有；Sensor 只读取筛选；tick 级失败可观察性由 loop 拥有。
- 能力变化：无目标 session 仍返回空历史；角色、context frame、长度、主动顺序与状态标签不变；数据库故障由假空结果变为明确失败。
- 性能变化：数据库读取次数和正常筛选复杂度不变。
- 测试新增：普通筛选/截断、主动顺序/metadata、两个真实入口的已关闭 SQLite 传播。
- 测试删除及原因：无。
- 验证结果：主动相关组合 `416 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：tick 失败沿既有 supervisor 策略进入下一轮；本提交未改变重试节奏。

### `bba83b52` `perf(akasha): 合并批量删除事务`

- 范围：Akasha sidecar 节点/关联边批量物理删除和存储回归测试。
- 历史依据：PR #65 的 sidecar 存储边界；PR #66 的快速路径一致性；PR #67/#68 的 scheduler/read-only 隔离与 live/replay parity 均未触及。
- 原问题：批量接口逐项获取锁并提交事务，200 项产生 200 次 COMMIT；中途失败还会留下部分删除结果。
- 为什么这样修改：用一次锁和一次 SQLite 事务包住逐 ID `executemany`；不构造无界 `IN (...)`，避免 dashboard 批量输入触发 SQLite 参数上限。
- 不变量与拥有层：节点与全部入边/出边的一致物理删除由 AkashaStore 拥有；缺失和重复 ID 不增加删除计数。
- 能力变化：最终删除计数、缺失/重复、边清理与无关边保留不变；批次从部分提交升级为全有或全无。
- 性能变化：同一 200 项 workload、12 次测量，中位耗时 `10.208 ms → 0.926 ms`，约 `11.0x`；COMMIT `200 → 1`。
- 测试新增：成功路径覆盖计数/重复/缺失/入出边；SQLite trigger 在批次中间失败，验证节点和边全部 rollback。
- 测试删除及原因：无。
- 验证结果：`tests/test_akasha_plugin.py` `37 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：大批次仍逐 ID 执行 SQL，避免参数上限但持锁时间随批量线性增长；这是相对原实现更短的同量工作。

### `c1d37dbd` `fix(mcp): 拒绝非对象调用错误`

- 范围：`McpClient.call()` 的 JSON-RPC `tools/call` error 边界和真实 stdio 响应测试。
- 历史依据：PR #105 的 MCP generation/连接清理；PR #107 的 180 秒超时贯通；PR #89 的取消边界均未触及。
- 原问题：代码无条件对 `error` 调用 `.get()`；非对象合法 JSON 会产生无 server/tool 上下文的 `AttributeError`。
- 为什么这样修改：JSON-RPC error 必须是 object；标准对象保持既有用户可见字符串，字符串/列表等协议损坏携带 server、tool、类型和值抛出 `RuntimeError`，不归一化为普通工具失败。
- 不变量与拥有层：JSON 解码和 response id 由 `_recv` 拥有；tools/call error schema 与用户可见转换由 `McpClient.call()` 拥有。
- 能力变化：正常 content、标准远端错误、同 server 串行、timeout/cancel/disconnect 不变；损坏 error 从偶发属性错误变为有上下文的 fail-loud。
- 性能变化：仅错误路径增加常数级类型判断，不声明性能收益。
- 测试新增：标准 error object，以及字符串/列表 error 的拒绝和上下文断言。
- 测试删除及原因：无。
- 验证结果：MCP/热重载相关 `30 passed`；副手完整测试 `1508 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：标准 object 内部字段继续保持既有宽松展示，不在本提交扩大协议迁移范围。

### `8181bd51` `perf(proactive): 初始化时完成日志迁移`

- 范围：`ProactiveStateStore` tick log schema 迁移、finish 热路径及真实 SQLite 测试。
- 历史依据：PR #103/#109 的主动 tick 与事件流架构；迁移不改变 phase/order、delivery/feedback、hot reload 或 MCP poll。
- 原问题：每次 tick finish 都执行 `PRAGMA table_info(tick_log)`，但 schema 在一个 store 生命周期内只可能由初始化改变。
- 为什么这样修改：把旧库 `proactive_effects_json` 补列放入 `_init_schema()` 的建表事务；业务写入信任初始化后的 schema。
- 不变量与拥有层：finish 前列必须存在，该不变量由 `ProactiveStateStore._init_schema()` 唯一拥有；业务写入不重复验证。
- 能力变化：新库、旧库迁移、tick log JSON、dashboard 查询和提交时机不变；旧库在首次初始化即完成迁移。
- 性能变化：10 次 finish 的 schema 查询 `10 → 0`；包含初始化则 `10 → 1`，总数减少 90%，热路径减少 100%。
- 测试新增：真实旧 schema 初始化补列并写入；SQLite trace 断言连续 finish 不再查询 schema。
- 测试删除及原因：无。
- 验证结果：主动相关组合 `418 passed`、dashboard `25 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：初始化本身仍执行一次 `PRAGMA table_info`，这是兼容旧库所需的一次性成本。

### `0b916a57` `fix(mcp): 校验工具调用结果结构`

- 范围：MCP `tools/call` 成功结果的 result/content/block/text 边界及 stdio 响应测试。
- 历史依据：客户端固定协商 MCP `2024-11-05`；PR #107 的 timeout 透传和 PR #105 的连接/代际清理未修改。
- 原问题：损坏 result 有时被字符串化为“成功”工具输出，有时产生无字段上下文的属性/类型错误。
- 为什么这样修改：按已协商协议验证 result object、必需 content list、每个内容对象和 text 字符串；字段错误携带 server/tool/path/type/value 失败。
- 不变量与拥有层：`_recv` 拥有 JSON/id；`_response_result` 拥有 result object；`McpClient.call()` 拥有 CallToolResult content schema。
- 能力变化：标准 text block 仍拼接文本；合法 image/resource 等无 text 对象继续保持既有字典字符串；锁、超时、取消、断连和标准 error 不变。
- 性能变化：成功响应增加线性类型校验，与原本遍历 content 同阶，不声明性能收益。
- 测试新增：result 标量、缺失/错误 content、标量 block、非字符串 text 五条损坏路径。
- 测试删除及原因：无。
- 验证结果：MCP/热重载相关 `35 passed`；副手完整测试 `1519 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：合法非文本内容仍以 Python dict 字符串传给模型，这是既有表示协议，后续若需多模态 ToolResult 应独立设计。

### `3f4e2645` `fix(akasha): 暴露 dashboard 配置错误`

- 范围：Akasha dashboard 注册时的插件配置来源与损坏配置回归。
- 历史依据：PR #93 的 snapshot freshness/旧坐标复用；PR #105 的 candidate 初始化/回滚边界。
- 原问题：dashboard 忽略 runtime 传入的真实 `plugin_dir`，并捕获所有配置加载异常后退回默认配置，可能连接或创建错误 sidecar。
- 为什么这样修改：直接从 canonical plugin_dir 调用统一配置加载器；配置不存在仍由加载器使用默认值，配置存在但 TOML 损坏/不可读则阻止注册。
- 不变量与拥有层：外部 TOML 结构与读取由 `load_akasha_config` 拥有；dashboard 没有推导正确 DB 路径的恢复能力。
- 能力变化：合法配置和缺失配置默认值不变；损坏配置从静默换库变为原始配置错误；recall/replay/snapshot 算法未触及。
- 性能变化：删除一层 helper 和异常分支，无性能声明。
- 测试新增：真实非法 TOML 在 dashboard 注册时传播 `TOMLDecodeError`。
- 测试删除及原因：无。
- 验证结果：Akasha/dashboard 相关 `38 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：配置字段的数值转换仍有历史默认策略，需要按字段契约另行审计。

### `f45b899e` `fix(proactive): 暴露上下文组装故障`

- 范围：主动 prompt 的 MemoryProfile/workspace 规则读取、类型协议和 facade 测试替身。
- 历史依据：PR #101 的 runtime clock 和 Drift 规则；PR #103 的 Prepare→Judge→Resolve→Deliver 次序。
- 原问题：prompt builder 分别吞掉四个任意异常，把画像、长期记忆、近期上下文和 workspace 规则故障伪装成内容为空；旧测试假对象缺少真实协议方法也被掩盖。
- 为什么这样修改：MemoryProfile 是完整内部协议；workspace callback 已在文件 I/O 边界记录失败并返回旧缓存，组装层没有第二种恢复动作。
- 不变量与拥有层：profile 读取由 MemoryProfileApi/runtime 拥有；workspace 文件恢复由 loop callback 拥有；prompt builder 只组装；tick supervisor 记录并隔离整轮错误。
- 能力变化：正常区块、空内容跳过和 runtime clock 位置不变；依赖故障从缺块假成功变为明确 tick 失败。
- 性能变化：读取次数不变，删除重复异常框架，无性能声明。
- 测试新增：三个 profile 方法和 workspace callback 的失败传播；修复 facade 使其实现真实读取协议。
- 测试删除及原因：无。
- 验证结果：主动相关组合 `422 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：workspace I/O 仍按设计可降级到旧缓存并记录 warning；这是拥有恢复动作的边界，不属于静默失败。

### `f0af9b55` `fix(peer-agent): reject missing remote task id`

- 范围：A2A `message/send` 非阻塞提交响应、Poller 注册和新直接测试。
- 历史依据：现有请求固定 `configuration.blocking=false`，随后必须以服务端 Task ID 调用异步 Poller。
- 原问题：响应缺少 `result.id` 时生成从未发给服务端的本地 UUID，返回 submitted 并永久轮询不存在的任务。
- 为什么这样修改：验证顶层/result object 和非空字符串 Task ID；协议损坏进入既有公开提交失败结果，且不注册 Poller。
- 不变量与拥有层：A2A HTTP/JSON 响应由 `_submit_task` 拥有；只有服务端 Task ID 能进入 Poller；`execute()` 拥有对用户可见的提交失败转换。
- 能力变化：合法异步 Task、冷启动、channel/chat 绑定与后台通知不变；假成功被删除。
- 性能变化：仅响应边界增加常数级校验，不声明性能收益。
- 测试新增：服务端 ID 正常注册，以及数组响应、缺失/空/非对象 result、空/非字符串 id 共七条路径。
- 测试删除及原因：无。
- 验证结果：定向 `7 passed`；副手完整测试 `1528 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：若未来改为 blocking 请求允许直接 Message，必须单独设计同步结果分支，不能复用异步 Poller。

### `e6187d6f` `fix(akasha): 拒绝非法显式配置值`

- 范围：Akasha 配置字符串/整数/浮点解析及真实 TOML 参数化测试。
- 历史依据：统一 `load_akasha_config` 被 candidate 初始化、replay、dashboard 和诊断命令共同使用，是唯一 schema owner。
- 原问题：显式非法值被静默替换为默认值，且 `bool` 会因 Python 是 `int` 子类而可能穿透数字判断。
- 为什么这样修改：文件或字段缺失才使用默认；合法整数、浮点和历史数字字符串继续支持；显式错误携带字段名失败。
- 不变量与拥有层：TOML 类型收敛由配置加载器拥有，上游无法保证手工文件；算法层信任强类型且有限的数值。
- 能力变化：缺失配置默认值和合法历史写法不变；错误 db_path、非数字字符串、非整数 float、nan、bool、容器改为 fail-fast。
- 性能变化：仅初始化解析路径，无性能声明。
- 测试新增：默认/合法数字字符串，以及上述显式错误类型；特别覆盖 int/float 字段的 bool 与容器。
- 测试删除及原因：无。
- 验证结果：配置定向 `8 passed`、Akasha+fast replay parity `46 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：字段领域范围未在本提交新增限制；需要先从算法和历史配置证明范围，避免武断裁剪能力。

### `94e9ac6a` `fix(akasha): 对齐来源引用失败语义`

- 范围：live/replay query log 的 source_ref 统计和内部共享 helper。
- 历史依据：PR #66 要求离线快速 replay 与线上单轮路径保持一致。
- 原问题：live 独立实现并两次吞掉任意 JSON 错误，写入 `source_ref_count=0` 的假成功诊断；replay 对相同内部契约则直接失败。
- 为什么这样修改：`_load_turn_card` 唯一生成 JSON list source_ref；live/replay 共用解析逻辑，内部契约违反时不应由诊断写入层恢复。
- 不变量与拥有层：card source_ref 结构由 card 构造拥有；query log 只统计并持久化，不能把损坏解释为空来源。
- 能力变化：合法引用计数和 query log 内容不变；损坏引用从假成功改为失败且不写半条诊断。
- 性能变化：同阶线性解析，删除重复实现，无性能声明。
- 测试新增：构造损坏内部 card，断言 JSON 错误传播且 query log 总数仍为零。
- 测试删除及原因：无。
- 验证结果：Akasha+fast replay parity `50 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：source_ref 仍是 JSON 字符串内部表示；若未来开放外部构造，应升级为 typed 字段而不是下游重复校验。

### `54da202c` `fix(scheduler): reject corrupt persisted jobs`

- 范围：JobStore 严格读取、schema 反序列化、原子保存和持久化测试。
- 历史依据：PR #52 的 scheduler 后台任务语义；PR #79/#89 的 timeout/cancel 行为未改。
- 原问题：坏 JSON、顶层/任务结构和时间戳损坏全部被当成空任务集；下一次 add/cancel/save 会覆盖原文件并丢失任务；非原子保存还会制造半文件。
- 为什么这样修改：文件不存在才为空；严格 read_text/json.loads 保留 I/O/JSON 原异常；成功解析后的 schema 错误带 path/index/field；保存改用既有同目录原子替换。
- 不变量与拥有层：JSON→ScheduledJob 由 JobStore 拥有，下游 SchedulerService 信任完整任务；读/解析错误不能伪装为无任务。
- 能力变化：合法 roundtrip、misfire/recovery、执行和取消不变；损坏文件阻止启动/覆盖；保存具备原子替换。
- 性能变化：写入增加一次同目录临时文件 rename，以可靠性为目标，不声明提速。
- 测试新增：原始 JSONDecodeError/PermissionError、顶层/条目 schema、缺失/损坏时间字段与 roundtrip。
- 测试删除及原因：无。
- 验证结果：定向 `33 passed`；副手完整测试 `1539 passed`；`git diff --check` 通过；worktree pyright 仅缺可选环境包产生既有 missing-import，新增路径无错误。
- 残余风险：已有损坏 jobs.json 会在启动时明确失败，需要人工修复或从备份恢复；这是防止静默丢任务的预期行为。

### `9b11ec4b` `fix(akasha): 暴露空节点向量损坏`

- 范围：Akasha sidecar 节点反序列化与损坏 DB 测试。
- 历史依据：PR #65/#66 的 sidecar/dense 图与 live/replay parity；上游 MessageEmbeddingStore 已拥有非空向量写契约。
- 原问题：空 embedding BLOB 节点被 list/get 静默当作不存在，使节点、边、fan 和诊断计数分叉。
- 为什么这样修改：sidecar DB 可来自旧版本或手工修改；读取边界没有正确修复动作，应携带节点 key 失败。
- 不变量与拥有层：正常写入的非空向量由 embedding/upsert 构造链拥有；持久化 BLOB 到 AkashaNode 由 `_row_to_node` 拥有。
- 能力变化：合法节点、召回、replay、read-only、reinforce 和 snapshot 不变；损坏节点不再被过滤。
- 性能变化：删除 list comprehension 的 None 过滤，非性能提交。
- 测试新增：真实写入节点后把 BLOB 改为空，断言 list_nodes 以节点 key 报错。
- 测试删除及原因：无。
- 验证结果：Akasha+fast replay parity `51 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：已有空向量节点会阻止整图加载，需重建 sidecar；这是避免错图运行的预期 fail-stop。

### `badc79c1` `fix(proactive): 暴露记忆优化失败`

- 范围：MemoryOptimizer pending 两阶段事务、SELF 更新、取消传播与历史测试替身。
- 历史依据：PR #75 的 memory fail-stop；后台 `MemoryOptimizerLoop` 已拥有记录异常并等待下周期的 supervisor 边界。
- 原问题：merge/provider 与 SELF 异常被吞成空内容或假成功；旧测试只提供一次模型响应，第二步 `StopAsyncIteration` 也被掩盖；marker-only snapshot 会永久遗留。
- 为什么这样修改：snapshot 成功后，read/merge/backup/write/commit/rollback 整个 MEMORY 阶段任一步失败或取消都恢复 pending 并重抛；SELF 在事务外，不能回滚已提交 MEMORY但必须报告失败。
- 不变量与拥有层：pending 两阶段事务由 optimizer 拥有；周期隔离由 loop supervisor 拥有；正常空 merge 明确 rollback；marker-only 空有效内容明确 commit 清理 snapshot。
- 能力变化：正常合并、空结果保留原记忆、SELF 更新和周期续跑不变；异常/取消可见且 pending 不丢；SELF 部分失败如实暴露。
- 性能变化：正常模型调用次数和顺序不变，无性能声明。
- 测试新增：merge RuntimeError、真实 MEMORY 写失败、CancelledError、SELF 失败、marker-only snapshot；修正旧测试两步响应。
- 测试删除及原因：无。
- 验证结果：optimizer `14 passed`，相关主动组合 `422 passed`；pyright `0 errors` 且仅一个既有 warning；`git diff --check` 通过。
- 残余风险：SELF 写入不是与 MEMORY 同一原子事务，失败会保留已提交 MEMORY；该部分成功状态现在显式可见，后续若要全局原子性需独立设计。

### `96baa0ab` `fix(proactive): 收紧时间归一化异常边界`

- 范围：主动候选时间与时区归一化、直接边界测试。
- 历史依据：PR #101 的 runtime clock；外部候选非法时间按既有契约可忽略，运行环境故障不可伪装成无时间。
- 原问题：两个 `except Exception` 同时吞掉非法输入与 tzdata/runtime 程序错误。
- 为什么这样修改：ISO 只恢复 `ValueError`；时区只恢复 `ValueError`/`ZoneInfoNotFoundError`；其他故障向 tick supervisor 传播。
- 不变量与拥有层：外部字符串解析由 contracts 边界拥有；tzdata/runtime 可用性不由归一化函数恢复；tick supervisor 负责记录和续跑。
- 能力变化：合法本地时间、非法 ISO/未知时区忽略不变；非预期时区解析故障改为明确失败。
- 性能变化：分支和调用次数不变，无性能声明。
- 测试新增：非法时间/时区继续恢复，以及注入非预期 ZoneInfo RuntimeError 的传播。
- 测试删除及原因：无。
- 验证结果：定向 `10 passed`、主动相关 `424 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：GatewayResult 动态 payload 类型仍需跨模块协议设计，不能靠局部 cast 解决。

### `94534191` `fix(akasha): 对齐只读来源引用契约`

- 范围：Akasha source_ref JSON-list 统一解析和 read-only query 回归。
- 历史依据：PR #66 的 live/replay parity；PR #67 的 read-only 查询不得写 activation/query log。
- 原问题：stateful query log 已 fail-loud，但 read-only record 构造仍把损坏 JSON 或非数组归为空 evidence，形成模式间失败语义分叉。
- 为什么这样修改：`_source_refs()` 与 `_source_ref_ids()` 共用唯一 JSON-list parser；内部生成契约违反时直接失败。
- 不变量与拥有层：source_ref 由 `_load_turn_card` 生成 JSON list；record/query-log 消费层不拥有修复动作。
- 能力变化：合法 evidence、stateful/read-only 召回结果不变；read-only 损坏引用明确失败，同时仍不产生 pending activation 或 query log。
- 性能变化：删除重复解析分支，无性能声明。
- 测试新增：同一 read-only request 先验证合法结果，再注入非数组 source_ref，断言失败且两次均 `update_state=False`、无状态写入。
- 测试删除及原因：无。
- 验证结果：Akasha+fast replay parity `51 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：历史 sidecar/query log 若含坏 source_ref 会显式失败，需要迁移或重建；这是避免空证据假成功的预期行为。

## 集成检查点

- Wave 1 主分支组合验证：`1502 passed`。
- Wave 2 中段主分支组合验证：`1516 passed`。
- Wave 2 收束前主分支组合验证：`1554 passed`。
- 三次均运行 `pytest -q tests/`，未删除测试；用例增长来自真实契约、事务和性能回归。

### `48a8768f` `fix(memory): 拒绝非法显式插件配置`

- 范围：default-memory TOML section/字段类型收敛和配置回归。
- 历史依据：PR #41 的默认记忆插件标准 TOML 写法全部保留。
- 原问题：错误 section 被归为空配置；`bool("false")` 变 True；db_path/整数/浮点错误值被强转或截断。
- 为什么这样修改：只对文件、section 或字段缺失使用默认；显式值由唯一配置 owner 严格解析并携带完整字段路径失败。
- 不变量与拥有层：外部 TOML schema 由 `load_default_memory_config`/codec 拥有，engine 信任强类型；不在算法层重复检查。
- 能力变化：标准 TOML、历史整数/数字字符串和整数值 float 保留；错误根/嵌套 section、bool 冒充数字、非整数 float、容器等 fail-fast；未新增范围限制。
- 性能变化：仅初始化解析，无性能声明。
- 测试新增：合法旧写法和九类显式错误值/section。
- 测试删除及原因：无。
- 验证结果：配置与 memory engine contract `39 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：字段数值范围仍需结合召回算法和历史配置设计，未武断收紧。

### `e2d3a7ba` `fix(bus): report admission enqueue failures`

- 范围：EventBus 热重载 admission 后台入队 task 所有权与错误日志测试。
- 历史依据：PR #105 的 snapshot admission/lease/drain；PR #109 的事件流唤醒。
- 原问题：暂停 admission 时创建的后台 task 只从集合删除，不读取异常；acquire 失败导致事件丢失并产生无人拥有的 asyncio 异常。
- 为什么这样修改：EventBus 作为 task owner，done 时统一清集合；shutdown cancellation 静默，其他失败读取原异常并记录 traceback 和事件类型。
- 不变量与拥有层：admission/acquire 由 snapshot store 拥有；task 生命周期和失败可见性由 EventBus 拥有；不新增 retry/fallback。
- 能力变化：成功入队、lease、queue、drain/close 不变；失败仍不伪装成功，但具备领域日志。
- 性能变化：成功路径多一次 `task.exception()` 常数操作，无性能声明。
- 测试新增：模拟 acquire 失败，断言原 cause、日志和 pending owner 清理。
- 测试删除及原因：无。
- 验证结果：热重载相关 `90 passed`；副手完整测试 `1557 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：失败事件不自动重试；是否持久化事件属于 durable delivery 设计，不应局部猜测。

### `7a595739` `fix(skills): 拒绝损坏的元数据配置`

- 范围：SKILL.md metadata YAML/JSON 边界、requires 可用性与 loader 测试。
- 历史依据：PR #95 的 Skill Catalog generation 与 PR #105 的候选 snapshot/hot reload。
- 原问题：损坏或非对象 JSON metadata 被归为空配置，绕过 requires 后错误标记技能可用。
- 为什么这样修改：metadata 缺失/空才无配置；YAML map/JSON object 正常；损坏 JSON、数组、null 携带具体 SKILL.md 路径失败。
- 不变量与拥有层：metadata schema 和 requirements 由 SkillsLoader 拥有；snapshot 只接收已校验 SkillRecord。
- 能力变化：合法技能、优先级、缺失 metadata 和热重载不变；损坏候选在发布前失败。
- 性能变化：索引构建增加常数级结构判断，无性能声明。
- 测试新增：空 metadata 两种写法、损坏 JSON、数组/null 非对象和路径上下文。
- 测试删除及原因：无。
- 验证结果：相关公共契约/snapshot/热重载 `224 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：requires 领域规则未扩展；未来字段必须在 owner 层显式设计。

### `717e61ee` `fix(bootstrap): continue cleanup after server failure`

- 范围：AppRuntime dashboard/chat task 等待与 shutdown supervisor 测试。
- 历史依据：应用 shutdown 已定义逐项继续清理、最后抛首错；PR #105 的 watcher/services/core drain 需要完整执行。
- 原问题：server task 已失败时，统一 cleanup supervisor 之前的直接 await 立即重抛，跳过 watcher、proactive、IPC、channels、core、memory 和 HTTP 资源清理。
- 为什么这样修改：把两个 server wait 纳入 `_run_cleanup_steps`；server 异常仍是最终首错，但后续资源全部获得清理机会。
- 不变量与拥有层：server should_exit/等待由 server step 拥有；跨资源继续清理和首错由 shutdown supervisor 拥有。
- 能力变化：正常顺序和 CancelledError 语义不变；失败 shutdown 不再短路后续清理。
- 性能变化：正常 shutdown 等待顺序不变，无性能声明。
- 测试新增：dashboard task 预先失败，断言最终原错、core.stop、should_exit 和 HTTP close。
- 测试删除及原因：无。
- 验证结果：相关 `40 passed`；副手完整测试 `1568 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：server task 无限等待和 shutdown 外部取消需要整体 timeout/shield 契约，本提交不局部改变。

### `363b725e` `fix(chat): 限制代码高亮缓存`

- 范围：聊天前端代码块高亮缓存与并发请求合并。
- 原问题：以不完整键缓存高亮结果且无容量上限；同一输入可重复启动异步高亮。
- 为什么这样修改：缓存键覆盖语言、主题和代码全文，使用 128 项 LRU，并复用同键 pending promise。
- 不变量与拥有层：代码块组件拥有展示缓存；Shiki 仍拥有语法高亮结果，组件不伪造失败结果。
- 能力变化：高亮内容与主题切换保持；消除键碰撞和重复计算。
- 性能变化：已完成缓存从无界变为最多 128 项；同键并发计算从 N 次变为 1 次。
- 测试新增：无；该组件暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 和 build 通过；lint 仅 3 条既有 Hook warning。
- 残余风险：pending 表在任务存续期保留 promise；任务完成后立即删除。

### `27dd8f0a` `fix(ipc): reject non-object client frames`

- 范围：IPC client newline JSON 帧反序列化边界。
- 原问题：合法 JSON 标量随后以对象方法访问，产生不透明异常并断开连接。
- 为什么这样修改：JSON 解码后立即确认顶层对象；非法帧显式记录并跳过，后续合法帧仍可处理。
- 不变量与拥有层：wire JSON 结构由 IPC 边界拥有；handler 信任对象，不重复检查。
- 能力变化：合法对象不变；单个非对象帧不再破坏长连接。
- 性能变化：每帧增加一次常数结构判断，无提速声明。
- 测试新增：同一连接发送标量后发送合法对象，验证错误可见且连接继续工作。
- 测试删除及原因：无。
- 验证结果：IPC 定向测试和 pyright 通过。
- 残余风险：对象内部字段仍按各消息 handler 的协议分别校验。

### `6d4d58ee` `fix(config): 拒绝无效工具集装配配置`

- 范围：agent 工具集装配配置读取。
- 原问题：错误类型和未知工具集被静默归一化，容易在启动后表现为能力缺失。
- 为什么这样修改：缺失字段使用默认；显式空列表保留“禁用全部”；错误结构和未知名字启动期失败。
- 不变量与拥有层：外部配置 schema 由装配层拥有；运行时只接收已解析工具集。
- 能力变化：默认与显式禁用语义不变；配置错误从隐性降级变为明确失败。
- 性能变化：仅启动期校验，无性能声明。
- 测试新增：覆盖缺失、显式空、错误类型和未知工具集。
- 测试删除及原因：无。
- 验证结果：相关配置测试和 pyright 通过。
- 残余风险：工具自身的运行时外部输入仍由各自边界拥有。

### `54f2026b` `refactor(chat): 保持代码高亮渲染纯净`

- 范围：聊天代码块异步高亮状态更新时机。
- 原问题：React render 阶段触发 setState，可能引发重复渲染与陈旧结果覆盖。
- 为什么这样修改：副作用移入 effect，并把异步结果绑定到当前输入。
- 不变量与拥有层：React effect 拥有异步生命周期；render 只从状态生成视图。
- 能力变化：高亮、复制和主题效果不变；旧请求不再覆盖新代码。
- 性能变化：消除 render 阶段额外状态更新，无量化延迟声明。
- 测试新增：无；该组件暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 和 build 通过。
- 残余风险：前端缺少组件级并发测试，当前由类型、构建与代码审阅覆盖。

### `5cdff4b9` `fix(lifecycle): 拒绝未闭合的阶段依赖`

- 范围：Phase 核心模块依赖和数据 slot 启动校验。
- 原问题：核心依赖缺失、顺序错误或 slot 未产生只记录 warning，真实 turn 才以 KeyError 等不透明方式失败。
- 为什么这样修改：核心阶段构造期 fail-fast；插件模块缺失插件依赖仍由拓扑层递归禁用，保留热插拔降级。
- 不变量与拥有层：Phase 拥有核心链闭合；插件拓扑拥有可卸载插件依赖。
- 能力变化：正常 turn、snapshot、interrupt 和 hot reload 不变；核心装配错误提前暴露。
- 性能变化：仅构造期校验，无性能声明。
- 测试新增：核心依赖不存在、顺序错误和未闭合 slot；保留插件递归禁用回归。
- 测试删除及原因：无。
- 验证结果：主线生命周期/热重载组合 `137 passed`；副手相关 `148 passed`；pyright 通过。
- 残余风险：动态插件是否允许依赖核心 slot 仍由现有命名协议区分。

### `92c7addd` `fix(akasha): 串行化图快照轮询`

- 范围：Akasha graph panel 快照轮询与 disposer。
- 原问题：请求慢于 5 秒轮询周期时会无限重叠，旧响应还可能晚到并覆盖新结果。
- 为什么这样修改：每个 panel 最多一个 in-flight 请求；完成后恢复轮询；dispose 后不再应用结果。
- 不变量与拥有层：panel 拥有轮询并发；后端 snapshot version 与增量坐标协议未改。
- 能力变化：首次 refit、坐标、交互和热重载 disposer 保持；消除旧响应覆盖。
- 性能变化：并发快照请求从无界降为最多 1；不声明单次延迟提升。
- 测试新增：无；插件面板暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 与真实 esbuild 参数编译通过；未修改 static bundle。
- 残余风险：请求失败仍沿既有显式失败路径，由下一轮定时器重试。

### `64c66fb0` `fix(clock): make replay advance atomic`

- 范围：ReplayClock 单实例并发推进与持久化。
- 原问题：`advance` 的读取和写入分属两个锁区间，并发调用会丢失 delta。
- 为什么这样修改：同一锁内完成 read-modify-write；底层同目录临时文件替换保持。
- 不变量与拥有层：ReplayClock 实例拥有进程内串行化；不声明跨实例或跨进程互斥。
- 能力变化：now/set/环境选择保持；同实例并发推进不再丢增量。
- 性能变化：锁覆盖一次文件读写，以正确性为目标；无延迟优化声明。
- 测试新增：8 线程各推进 50 次，400 个返回时间唯一且最终时间累计完整。
- 测试删除及原因：无；初版审阅时删除了未被生产路径调用的无效 barrier 测试钩子，改为真实并发压力回归。
- 验证结果：Clock/wake `25 passed`；副手全量 `1574 passed`；pyright `0 errors, 0 warnings`。
- 残余风险：多个 ReplayClock 实例指向同一路径仍需文件锁或单 owner 架构，本提交不扩大承诺。

### `93de1a8a` `fix(context): 显式标记不可用媒体`

- 范围：MessageEnvelopeBuilder 本地媒体装配。
- 原问题：不存在的本地附件在多模态和文本/VL 两条路径都被静默丢弃，模型无法区分“无附件”和“附件不可用”。
- 为什么这样修改：保留文字 turn，在上下文和 warning 中明确具体不可用路径；仅缺失附件时不诱导调用读图工具。
- 不变量与拥有层：媒体文件可访问性由上下文装配边界拥有；模型调用链信任已标注媒体引用。
- 能力变化：有效本地图片、文档、远程图片和 VL fallback 不变；缺失附件变为可观察降级。
- 性能变化：缺失路径增加一条 warning 和文本标记，无性能声明。
- 测试新增：两种媒体能力路径的缺失文件，以及仅缺失附件时不生成读图调用。
- 测试删除及原因：无。
- 验证结果：副手 ContextBuilder/lifecycle `117 passed`；主线相关 `47 passed`；pyright 通过。
- 残余风险：远程 URL 的可达性仍由实际 HTTP/视觉工具边界判断，装配期不预请求。

### `f97e0eb9` `fix(dashboard): 暴露插件发现失败`

- 范围：Dashboard 插件清单启动加载与既有错误边界。
- 原问题：`/api/dashboard/plugins` 失败被转为空列表，UI 看似正常但插件能力全部消失。
- 为什么这样修改：移除空列表 fallback，把失败交给 App 统一 `run()` 边界展示；单 panel import 隔离策略保留。
- 不变量与拥有层：清单请求整体成功由启动加载拥有；单插件模块失败由 importPanel 隔离并记录。
- 能力变化：合法插件、版本 URL、CSS 注入和 hot-reload freshness 不变；发现失败明确显示。
- 性能变化：请求与加载次数不变，无性能声明。
- 测试新增：无；该启动链暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 和 production build 通过；lint 仍为 3 条既有 Hook warning。
- 残余风险：单 panel import 失败仍允许其他插件继续加载，这是插件隔离边界的既有能力。

### `8307360f` `fix(persistence): isolate atomic save temp files`

- 范围：scheduler 与 AnyAction 共用的 JSON 原子写底座。
- 原问题：同一目标的并发 writer 共用固定 `.tmp`；一个 writer 可替换另一个的内容，随后另一个因临时文件已移动而失败。
- 为什么这样修改：每次写入使用同目录唯一临时文件，再原子 replace；失败仅清理本 writer 的临时文件并传播原异常。
- 不变量与拥有层：helper 拥有 staging 文件隔离和原子替换；不声明 writer 顺序、跨进程锁或 compare-and-swap。
- 能力变化：JSON 格式、目标路径和错误契约不变；并发写不再互相窃取/删除 staging 文件。
- 性能变化：写入与 replace 次数不变，UUID 生成是常数开销；无提速声明。
- 测试新增：两个真实线程同步到 replace 后均可提交且结果完整；replace 失败保持旧目标、清理本次临时文件并传播。
- 测试删除及原因：无。
- 验证结果：主线持久化 `16 passed`；副手全量 `1583 passed`；pyright 无 error。
- 残余风险：最后写入者覆盖先写入者仍是普通文件存储语义；需 CAS 的调用方必须另设版本协议。

### `d0171f73` `fix(persistence): log atomic cleanup failures`

- 范围：JSON 原子替换失败后的 staging 清理。
- 原问题：replace 首错后的 `unlink` 使用宽泛捕获并静默 pass，残留临时文件没有路径和原因。
- 为什么这样修改：仅捕获文件清理边界的 `OSError` 并记录 domain/tmp/error，随后继续抛原 replace 错误。
- 不变量与拥有层：原事务错误保持首错；helper 只补充 best-effort cleanup 的可观测性。
- 能力变化：成功路径和错误类型不变；清理失败不再静默。
- 性能变化：仅失败路径增加一条日志，无性能声明。
- 测试新增：replace 与 unlink 同时失败，断言首错和 cleanup 上下文。
- 测试删除及原因：无。
- 验证结果：副手全量 `1584 passed`；pyright 无 error。
- 残余风险：cleanup 失败会保留唯一临时文件，需按日志人工清理。

### `0c9e8da9` `perf(core): 限制工具发现会话缓存`

- 范围：ToolDiscoveryState 跨会话和会话内解锁工具缓存。
- 原问题：每会话已有 5 项 LRU，但 session 数无限增长；仅使用 always-on/meta 工具也会制造空项。
- 为什么这样修改：增加默认 1024 session 的 LRU；访问刷新顺序；空会话不入表，淘汰后可重新 tool_search。
- 不变量与拥有层：发现缓存只保存可重建的工具名，不是业务状态；registry 仍拥有真实工具可用性。
- 能力变化：当前 1024 个活跃 session 的工具顺序与复用不变；旧 session 被淘汰后重新发现。
- 性能变化：默认最坏驻留从无限增长收敛为约 5120 个工具名。
- 测试新增：空项不创建、跨会话 LRU 访问刷新和最旧淘汰。
- 测试删除及原因：无；审阅阶段删除了两个无意义 default-factory 包装函数后才合入。
- 验证结果：副手相关 `118 passed`；主线组合 `23 passed`；pyright 无 error。
- 残余风险：容量是实例参数；显式调大时上界随配置线性增长。

### `9d54a421` `refactor(chat): 清理代码块注释`

- 范围：聊天代码块组件注释。
- 原问题：Types/Context/Token rendering 等标题式英文注释重复代码结构，必要约束也未按项目中文约定表达。
- 为什么这样修改：删除 10 条废注释；保留并中文化 Shiki 位标志、稳定键、CSS 行号、缓存和异步展示约束。
- 不变量与拥有层：仅注释变更，运行代码和 lint 指令不变。
- 能力变化：无。
- 性能变化：无。
- 测试新增：无。
- 测试删除及原因：无。
- 验证结果：目标 lint、typecheck、全量 lint 和 chat build 通过。
- 残余风险：其他前端文件的英文注释按文件继续审阅，不机械全局替换。

### `ad7f7959` `fix(dashboard): 收紧 Hook 生命周期`

- 范围：Dashboard MagicIndicator 与插件跨页事件 Hook。
- 原问题：动态依赖数组无法静态验证；goto-session 订阅闭包可能调用旧 selectView。
- 为什么这样修改：MagicIndicator 只声明真实静态依赖，DOM 选中变化继续由 MutationObserver 驱动；事件订阅用 Effect Event 读取最新跳转逻辑。
- 不变量与拥有层：观察器拥有 DOM/class 变化；全局事件只订阅一次，不因 view state 重装。
- 能力变化：指示器、插件跳转与 DOM 生命周期保持；消除 stale closure。
- 性能变化：切换状态不再为依赖变化拆装观察器，无量化声明。
- 测试新增：无；该 UI 链暂无组件测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、production build 和 lint 全过，历史 3 条 Hook warning 归零。
- 残余风险：MutationObserver 高频 mutation 的 RAF 合并仍可进一步独立评估。

### `be2e828b` `fix(subagent): 暴露模型调用硬错误`

- 范围：SubAgent provider 调用与同步/后台失败转换链。
- 原问题：provider 硬错误被 SubAgent 捕获后返回空字符串，可能被上层解释为正常空结果。
- 为什么这样修改：owner 先标记 `last_exit_reason=error`，再传播原异常；同步 spawn/后台 runner 继续在各自边界转成明确失败。
- 不变量与拥有层：SubAgent 拥有退出原因；任务 runner 拥有面向调用方的 error status/摘要。
- 能力变化：正常完成、loop guard、预算收尾不变；provider 故障不再假成功。
- 性能变化：无。
- 测试新增：provider RuntimeError 原样传播且退出原因为 error。
- 测试删除及原因：无。
- 验证结果：副手 SubAgent/spawn/background `40 passed`；主线相关 `34 passed`；pyright 无 error。
- 残余风险：工具执行错误仍按 ToolResult 协议处理，不与 provider 基础设施故障混淆。

### `0109b65f` `fix(proactive): reject corrupt quota state`

- 范围：AnyAction 每日配额 JSON 反序列化边界。
- 原问题：坏 JSON、权限错误和缺失文件都初始化零用量，可绕过每日动作上限；字段又被不一致地强转。
- 为什么这样修改：仅文件不存在初始化；严格读取 version=1 完整 schema、window key、非负整数和 aware ISO 时间；TypedDict 固化下游类型。
- 不变量与拥有层：QuotaStore 拥有持久化 schema；drift best-effort skill state 继续保留独立降级语义。
- 能力变化：合法首版格式、空 last_action 和 rollover 保持；损坏 quota 阻止启动且保留原文件。
- 性能变化：仅启动期校验，无性能声明。
- 测试新增：缺失、合法、非对象、缺字段、version/used/window/time、JSON 与读取权限错误。
- 测试删除及原因：无。
- 验证结果：定向 `12 passed`；副手全量 `1603 passed`；pyright 无 error。
- 残余风险：未知额外字段被忽略以保留向前兼容；schema 升级需显式版本迁移。

### `fc7fae40` `fix(subagent): 拒绝空白任务结果`

- 范围：SubAgent 无工具调用的最终响应契约。
- 原问题：模型以空白 content 结束时被标记 completed，后台/同步调用方收到假成功空结果。
- 为什么这样修改：最终响应 trim 后必须非空；否则标记 error 并由既有任务边界转换失败。
- 不变量与拥有层：中间 tool-call 响应允许空 content；最终任务结果由 SubAgent owner 保证可展示。
- 能力变化：正常文本、工具循环和预算收尾不变；空白最终响应明确失败。
- 性能变化：一次常数级字符串判断，无性能声明。
- 测试新增：空白 final response 抛错且退出原因为 error。
- 测试删除及原因：无。
- 验证结果：副手相关 `41 passed`；主线相关 `35 passed`；pyright 无 error。
- 残余风险：强制收尾 helper 的空结果契约继续单独沿完整调用链审阅。

### `23b08f74` `refactor(memory): 清理面板注释`

- 范围：默认记忆 Dashboard 面板注释。
- 原问题：12 条英文标题/逐段翻译注释无信息增量，必要的全局命名和增量 DOM 约束未按中文约定表达。
- 为什么这样修改：删除废注释；保留并中文化命名冲突、计数缓存、增量 DOM、焦点保持和降级边界。
- 不变量与拥有层：仅注释修改；运行代码和 TypeScript reference 不变。
- 能力变化：无。
- 性能变化：无。
- 测试新增：无。
- 测试删除及原因：无。
- 验证结果：typecheck、lint、插件 esbuild 与 dashboard build 通过。
- 残余风险：审阅同时发现文件内 catch 降级缺少可观测性，已作为下一笔功能修复处理，不能靠注释合理化静默失败。

### 外部插件 `8aaeab3` `fix(feed): maintain cache freshness in MCP lifecycle`

- 范围：canonical Feed 插件 `/mnt/data/coding/akashic-plugin/feed-mcp`、GitHub `akashic-plugins/feed-mcp` 与安装版本 `feed@github 1.2.0`。
- 历史依据：`3b456e7b` 把 source poll 绑定到 `default_proactive` lifecycle；启用 wake package 时 manifest 会禁用 default package，但 wake 只调用 `get_proactive_events`，从而丢失 Feed 外部刷新能力。
- 原问题：Feed MCP 进程持续运行且 wake 每约 5 分钟读取一次缓存，但 `poll_state.last_polled_at` 停在 2026-07-12 14:59 UTC；Tibo RSS 已有新消息，SQLite 仍是旧列表。现有测试只证明 default lifecycle 拥有 poll，没有覆盖 wake + Feed freshness 组合。
- 为什么这样修改：缓存 freshness 归缓存拥有者。Feed MCP 使用 FastMCP lifespan 启动唯一后台 poller；首次主动读取等待首次刷新，之后按 `feed_mcp.json.poll_ttl_seconds` 刷新。插件不再声明宿主 `poll_tool`，default 与 wake 都只通过异步 MCP 调用读取稳定缓存。
- 不变量与拥有层：Feed poller 唯一拥有刷新串行、首次 ready、失败状态和重试；backend 拥有单源 TTL 与 SQLite 数据；proactive lifecycle 只消费 source snapshot。系统级刷新错误使读取显式失败，单源失败继续由 Feed `_poll_rows` 隔离并记录。
- 能力变化：default/wake 的 fetch、分页、ack 和排序不变；wake 模式恢复新消息获取。MCP 启动不等待 32 个网络源，首次 `get_proactive_events` 才等待首次刷新；手动 poll 与后台 poll 由同一异步锁串行。
- 性能变化：外部抓取由宿主生命周期耦合改为 MCP 每 300 秒自行刷新；SQLite 启用 WAL 和 30 秒 busy timeout，轮询写入期间读取稳定快照。没有增加每次 wake 的网络抓取。
- 测试新增：poller 首次刷新屏障、持续刷新、失败可见与下一轮恢复。
- 测试删除及原因：删除未接线且吞掉启动错误的 `startup_force_poll()` 死代码；未删除行为测试。
- 验证结果：Feed 插件 `11 passed`；pyright `0 errors, 0 warnings`；GitHub 已推送；`plugin-install` 安装 1.2.0；运行进程切换到 1.2.0；首次自刷新 32/32 成功，Tibo 源解析 19 条并新增 2 条，`last_polled_at` 推进到 2026-07-12 18:59 UTC。
- 残余风险：同轮审计发现 Steam 历史 snapshot 的部分同类问题，已由下一条记录修复；Calendar 每次读取实时查询 Google API，Fitbit managed service 已自行轮询，二者不存在本次旧缓存问题。

### 外部插件 `326c055` `fix(steam): refresh proactive snapshots on demand`

- 范围：canonical Steam 插件 `/mnt/data/coding/akashic-plugin/steam-mcp`、GitHub `akashic-plugins/steam-mcp` 与安装版本 `steam@github 1.1.0`。
- 历史依据：Steam proactive source 的在线状态每次实时查询，但历史游戏时长只由手动 `take_steam_snapshot` 更新；运行数据库最后快照停在 2026-06-06。
- 原问题：`get_steam_context` 每约 5 分钟读取相同旧 snapshot；仓库没有定时调用者。即使新增定时调用，空的最近游玩列表也不会写任何行，下一轮仍会判断为从未成功刷新。
- 为什么这样修改：Steam context owner 在读取前检查 snapshot run 的 TTL，超过 6 小时才调用一次 Recently Played API；独立 `snapshot_runs` 表记录包括空结果在内的成功刷新批次。
- 不变量与拥有层：Steam MCP 拥有实时状态和历史快照 freshness；wake 只读取结构化 context。配置 JSON 损坏在读取边界 fail-loud；远端快照刷新失败保留实时状态并通过 `snapshot_refresh_error` 显式降级。
- 能力变化：实时 online/in-game 查询、两周与历史时长对比、wake presence/transition 保持；旧快照自动恢复刷新，空列表不再造成重复请求。
- 性能变化：wake 仍每轮查询轻量在线状态；Recently Played API 由过去“永不自动调用”变为最多每 6 小时一次，同 TTL 内只读 SQLite。
- 测试新增：过期快照只刷新一次、空快照记录成功批次、刷新失败可见、TTL 内跳过刷新。
- 测试删除及原因：无。
- 验证结果：Steam 插件 `7 passed`；pyright `0 errors, 0 warnings`；GitHub main 已推送；安装 1.1.0；真实 context 刷新成功，freshness `0.0h`、2 个近期游戏、无刷新错误；旧 1.0.0 generation 排空后仅保留 1.1.0 MCP 进程。
- 残余风险：Recently Played 和 Player Summary 是两个独立 Steam API 请求；其中一条失败时 context 会明确区分 snapshot 与 realtime 的降级状态，不提供跨 API 原子快照。

### `05ab66b3` `fix(runtime): restore session context after turn and tick`

- 范围：被动 turn 与 proactive tick 的 `current_session_key` 生命周期。
- 原问题：两个长链路入口调用 `ContextVar.set()` 后没有 reset；同一 task 后续执行会继承上一轮 session，导致 observe 全局错误归属错误。常规消息循环为每条消息创建 task，会掩盖被动路径问题，但 `process_direct` 和 proactive 长生命周期 loop 可真实触发。
- 为什么这样修改：由设置上下文的入口保存 token，并在最外层 `finally` 恢复调用方上下文；busy 状态仍由内层 `finally` 独立释放。
- 不变量与拥有层：`AgentLoop._process` 拥有单个 turn 的 session 绑定，`ProactiveLoop._tick_bound` 拥有单个 tick 的绑定；共享 ContextVar 和 observe 只读取，不承担生命周期清理。
- 能力变化：续跑、TurnStarted、核心处理、主动 Gate → Fetch → Judge → Resolve → Deliver、异常传播和 busy 状态语义保持；成功、失败与取消离开入口后均不残留 session。
- 性能变化：删除一处未使用计时，增加两次常数级 ContextVar token 操作；无性能收益声明。
- 测试新增：被动成功恢复、核心失败恢复并释放 processing state、主动成功与失败恢复。
- 测试删除及原因：无。
- 验证结果：副手定向 `19 passed`、全量 `1619 passed`、pyright `0 errors`；主线合入后定向 `19 passed`、全量 `1619 passed`，`git diff --check` 通过。
- 残余风险：其他独立 ContextVar 设置点仍需按各自任务生命周期审阅，不能从本次两处修复推断全仓已覆盖。

### `27e1c638` `fix(mcp): enforce 2024 tool result boundaries`

- 范围：MCP `tools/list`、`tools/call` 的外部 schema 与远端失败分类。
- 原问题：缺失 `inputSchema` 会静默变成空 schema，非字符串 description 被强转；坏 content block 可能以 Python repr 当成功结果进入模型；JSON-RPC error 与 `isError=true` 都被返回为普通字符串，直接调用方无法区分成功和失败。
- 为什么这样修改：按客户端实际协商的 `2024-11-05` 严格接受 text、image、resource 三类结果；后续版本字段明确拒绝。工具声明在 `tools/list` 边界校验，远端执行失败统一抛出带 server/tool/服务端内容的 `McpToolExecutionError`。
- 不变量与拥有层：`McpClient` 拥有 MCP 反序列化和协议版本边界；`ToolRegistry` 继续拥有面向模型的错误日志与 `工具执行出错` 回填。边界之后的 `McpToolInfo` 和工具结果不再重复防御。
- 能力变化：合法三类结果、stdio 串行、连接/执行 timeout、插件 generation 与热重载不变；坏 schema 不再进入工具目录，远端失败对直接调用方 fail-loud、对模型仍明确可见。
- 性能变化：每项增加常数级字段检查，content 仍单次 O(n) 遍历；无新增 I/O、重试或缓存。
- 测试新增：工具声明缺失/坏类型、三类有效内容、各类关键缺字段、未知与后续类型、非法 `isError`、JSON-RPC error 和 tool result error 异常。
- 测试删除及原因：无；旧 MCP 夹具补齐协议要求的 `type=text`。
- 验证结果：副手定向 `212 passed`、全量 `1632 passed`、pyright `0 errors`；主线合入后 MCP/IO 定向 `52 passed`，`git diff --check` 通过。
- 残余风险：客户端仍固定协商 `2024-11-05`；未来协议升级必须单独实现版本协商与新增 content union，不能在旧版本路径静默兼容。

### `61fba5be` `fix(channels): close resources and validate message boundaries`

- 范围：WebChat 外部消息、IPC server/client 生命周期、附件降级、渠道身份索引与 Telegram live-task 索引。
- 原问题：WebChat 强转坏 text 并静默丢弃坏 media 元素；IPC 构造时永久订阅且 stop 不关闭客户端，Unix chmod 失败会遗留已绑定 server/socket；身份保存失败会留下未持久化内存路由；Telegram 每个完成任务扫描全部 session 并永久保留空集合；附件 fallback 吞掉所有异常且无降级日志。
- 为什么这样修改：外部字段在 WebSocket 边界一次性严格校验；IPC 成功启动后才提交 server/subscription，停止前同步转移并 close 全部 ownership，再等待所有资源并重新抛首个 `OSError`；identity mapping 只在持久化成功后提交；任务回调按所属 session O(1) 清理；附件仅对 `OSError` 保留有日志的 `/tmp` 降级。
- 不变量与拥有层：WebChat 拥有帧 schema；IPC channel 拥有 server、writers 与 outbound subscription；SessionIdentityIndex 拥有 metadata/mapping 一致性；Telegram channel 拥有 live-task 索引；AttachmentStore 拥有文件系统降级。
- 能力变化：合法 WebChat、IPC、Telegram、附件上传和身份路由保持；坏帧返回明确 error 且连接可继续；IPC 启停失败不留订阅或客户端 ownership，多个关闭错误完成清理后 fail-loud。
- 性能变化：Telegram 完成回调由 O(session 数) 降为 O(1)，并删除空 session 集合；其余仅边界/生命周期常数级操作，无量化收益声明。
- 测试新增：WebChat 三类坏字段与连接续用、Unix chmod 失败事务回滚、server/writer wait 失败仍清理其余资源、IPC 正常 stop、identity 保存回滚、Telegram 空索引回收、附件 fallback 日志。
- 测试删除及原因：无。
- 验证结果：副手定向 `45 passed`、全量 `1628 passed`、pyright `0 errors`；主线合入后 channels/MCP 交叉定向 `69 passed`，`git diff --check` 通过。
- 残余风险：未改变 MessageBus 既有重试、FIFO、背压与取消策略；QQ/Telegram 外部 API 的独立错误策略需按具体调用链继续审阅。

### `f30973e9` `fix(proactive): tighten source and delivery boundaries`

- 范围：默认 proactive Gateway、MCP source event 边界与 success/post-guard ACK。
- 原问题：Gateway 再次捕获共享 source snapshot 故障并伪装成三路空数据；坏 web_fetch payload 被当成普通空正文；非对象或无 ID 的 alert/content 被跳过或生成碰撞 key；仅配置独立 alert ACK 时，普通 ACK 缺失导致 helper 提前返回。
- 为什么这样修改：单 source 隔离继续由 `fetch_sources_async` 唯一拥有，Gateway 只消费聚合 snapshot；整体失败和工具协议损坏 fail-loud。WebFetchTool 明确返回的 `{error}` 仍按可选正文降级，但记录 URL/原因 warning。source payload 在 MCP 边界拒绝无法可靠 ACK 的事件；两个 ACK 通道按实际依赖分别执行。
- 不变量与拥有层：source 聚合层拥有单源失败隔离；Gateway 拥有 snapshot 与 web_fetch 结果形状；`mcp_sources` 拥有 event object/ID；resolve helper 拥有 alert/content ACK 路由。
- 能力变化：正常并行抓取、单源隔离、显式 HTTP 失败空正文、ACK 顺序、发送、wake、热重载和 Gate → Fetch → Judge → Resolve → Deliver 不变；全部 source 失败不再假装无事件，独立 alert ACK 不再丢失。
- 性能变化：三路和 URL 抓取并行度、调用次数不变；新增每 item 常数级字段检查，无性能收益声明。
- 测试新增：三路 snapshot 失败传播、web_fetch 显式降级日志与损坏协议、坏 source item/空 ID、仅 alert ACK 的 success/post-guard 路径。
- 测试删除及原因：无。
- 验证结果：副手定向 `182 passed`、全量 `1646 passed`、pyright `0 errors`；主线合入后主动链交叉定向 `113 passed`，`git diff --check` 通过。
- 残余风险：Gateway/source payload 仍使用历史弱类型 dict；本批没有扩大为跨模块 typed contract 重构。

### `bcaf40e8` `fix(tools): harden exact selection and search ranking`

- 范围：tool_search 精确选择、runtime snapshot 文档查询与关键词 top-k 排序。
- 原问题：`select:` 通过宿主 registry 私有 `_documents` 做 risk 过滤，热重载 snapshot 新增/替换工具会读取错误代际元数据；重复名称产生重复结果；关键词搜索为所有命中候选生成解释并全量排序。
- 为什么这样修改：新增经过 `_runtime_view()` 的 `get_document()`；select 按输入顺序去重；保持 score 降序和 name 字典序 tie-break，正数 top-k 用 heap 选取后只为最终结果生成解释。
- 不变量与拥有层：ToolRegistry 拥有工具与索引文档，runtime view 拥有当前代际；ToolSearchTool 拥有 select 解析；KeywordSearchBackend 拥有评分与排序。hook、timeout、ToolResult、解锁与 snapshot 生命周期未改。
- 能力变化：正常搜索召回、精确名称 fast path、风险过滤和排除集合保持；snapshot risk 使用正确代际；重复 select 幂等。
- 性能变化：同机 20,000 个匹配文档、每轮 5 次 top-5 的合成负载，中位数由 `1.2399s` 降至 `0.4648s`，约 `2.67x`；不外推为整体生产延迟。
- 测试新增：score/name tie-break、重复 select、runtime snapshot risk 过滤。
- 测试删除及原因：无。
- 验证结果：副手定向 `122 passed`、全量 `1658 passed`、pyright `0 errors`；主线合入后工具交叉定向 `71 passed`，`git diff --check` 通过。
- 残余风险：高候选、小 top-k 场景收益最大；仍需为所有候选计算基础关键词分数，这是正确排序所必需。

### `62fec3e6` `fix(session): tighten config and persistence boundaries`

- 范围：TOML section、session SQLite JSON、FTS 初始化与 CLI socket 配置。
- 原问题：显式非 table 配置被静默当空表；空/坏 JSON 与错误形状可能进入消息，extra 可覆盖 role/id 等列字段；FTS 每次启动都因错误检查 config 表而全量 rebuild，并把所有 OperationalError 当成“无 FTS”；socket 被重复规范化。
- 为什么这样修改：只有缺失 section 使用默认空表，显式错误结构 fail-loud；SQLite 边界集中校验 metadata/extra/tool_chain 形状、空载荷和保留字段；从 sqlite_master 表定义判断 trigram，只有创建/迁移/触发器缺失才 rebuild；仅真实 FTS5/trigram 能力缺失带 warning 降级，其余数据库错误传播。
- 不变量与拥有层：Config loader 拥有 TOML schema；SessionStore 拥有 DB JSON 与消息列；FTS initializer 拥有索引/触发器；SessionManager 正常写入已排除 extra 保留字段。合法默认配置、消息顺序、事务保存、热重载不变。
- 能力变化：合法缺失配置和 LIKE fallback 保持；坏配置/数据不再伪装；正常重启保留中文 FTS 命中，缺触发器仍自动 rebuild；数据库锁/损坏不再静默关闭全文检索。
- 性能变化：已有 trigram FTS 的正常启动从每次扫描全 messages rebuild 改为只检查 sqlite_master 与三条 trigger；构造阶段 trace 已证明无 rebuild。收益随消息表大小增长，本批未虚构固定倍数。
- 测试新增：非 table section、空/坏 JSON、extra 保留字段、构造期无 rebuild、缺 trigger rebuild、FTS 能力缺失降级和普通 OperationalError 传播。
- 测试删除及原因：无。
- 验证结果：副手定向 `50 passed`、全量 `1675 passed`、pyright `0 errors`；主线合入后 session/config/tool 交叉定向 `113 passed`，`git diff --check` 通过。
- 残余风险：历史 `GatewayResult` 弱类型与 SessionStore 其他用途型扫描 API 继续按各自契约审阅，未在本批扩张。

### `d4f79950` `fix(provider): close streams and validate tool arguments`

- 范围：OpenAI-compatible stream 生命周期与 tool-call arguments 外部响应边界。
- 原问题：stream 成功或 timeout/回调/解析异常退出时未显式关闭，可能占用 HTTP response/连接；tool arguments 解码为数组或标量后直接进入内部 `dict` 契约。
- 为什么这样修改：provider 作为 stream owner 在消费循环最外层 finally 调用异步 close；流式与非流式统一通过边界 helper 解析并要求 JSON object。
- 不变量与拥有层：Provider 拥有外部 response 与 ToolCall 构造；delta 顺序、idle timeout、reasoning/provider 字段、缓存 token、retry/context guard 不变。
- 能力变化：合法响应完全保持；资源在成功、timeout、delta 回调异常与参数解析失败路径均释放；坏 tool arguments fail-loud。
- 性能变化：每次 stream 增加一次必要 close，无性能收益声明。
- 测试新增：四类 stream 关闭路径，以及流式/非流式非 object arguments。
- 测试删除及原因：无。
- 验证结果：副手定向 `23 passed`、全量 `1681 passed`、主文件 pyright `0 errors`；主线合入后定向 `23 passed`，`git diff --check` 通过。
- 残余风险：provider 综合测试文件仍有既有弱类型 warnings；本批未修改 retry 分类。

### `8eeaa270` `fix(lifecycle): expose subscription ownership`

- 范围：TurnLifecycle 的七个 EventBus handler 注册 façade。
- 原问题：façade 丢弃 EventBus 已返回的 `EventSubscription`，调用方无法表达和执行 handler ownership 的显式释放。
- 为什么这样修改：直接返回已有 subscription，不新增抽象、不改变注册与执行顺序；owner 可在自身销毁时 close。
- 不变量与拥有层：EventBus 仍拥有 handler 列表和 off；注册调用方拥有 subscription 生命周期。现有 app-lifetime 调用方忽略返回值时行为不变。
- 能力变化：现有 handler 行为无变化；新增显式注销能力。本批不声称已经改变生产 wiring 次数。
- 性能变化：无。
- 测试新增：subscription close 后 handler 不执行、handler count 归零。
- 测试删除及原因：无。
- 验证结果：副手定向 `60 passed`、全量 `1679 passed`、pyright `0 errors`；主线合入后 lifecycle 交叉定向 `57 passed`，`git diff --check` 通过。
- 残余风险：多媒体主动发送存在不可原子撤回的 partial-delivery contract，需单独设计，不能在本批用 bool 假装事务成功。

### `72bc91e7` `fix(bus): preserve outbound subscriber fanout`

- 范围：MessageBus 单条出站消息的订阅者 fan-out。
- 原问题：直接遍历可变订阅 list；首个回调在执行中关闭自身 subscription 时，元素左移会让 Python 迭代器跳过下一回调，静默漏发同一消息。
- 为什么这样修改：fan-out 开始时创建不可变 tuple snapshot；当前消息对开始时的订阅者保持稳定，后续消息重新读取最新订阅列表。
- 不变量与拥有层：MessageBus 拥有每条消息的 fan-out snapshot；EventSubscription 仍拥有后续注销。ChatLane、FIFO、重试/退避和降级不变。
- 能力变化：回调自注销不再造成其他订阅者漏发；该回调不会收到下一条消息。
- 性能变化：每次 fan-out 复制当前 channel 的小型订阅 tuple，O(订阅者数)；这是稳定迭代所需成本，无性能收益声明。
- 测试新增：连续两条消息验证当前 snapshot 与后续 unsubscribe 语义。
- 测试删除及原因：无。
- 验证结果：副手定向 `21 passed`、全量 `1683 passed`、范围 pyright `0 errors`；主线合入后 bus/event 定向 `26 passed`，`git diff --check` 通过。
- 残余风险：MessageBus/EventBus 队列仍无固定容量；缺少明确容量与背压契约前不任意加限额。

### `6c47cdbb` `fix(plugins): expose only current generation views`

- 范围：PluginManager 当前插件视图与 Telegram bot command 汇总。
- 原问题：旧 snapshot lease 排空前，`_active_plugins`/`_loaded` 同时含旧代和当前代；公开视图遍历 namespace 会重复暴露旧 metadata/commands，禁用当前插件后旧代仍可能被报告。
- 为什么这样修改：只从 `_active_generations` 当前代际映射获取 ActivePluginInfo 与 instance，并保留 registry active 检查。
- 不变量与拥有层：snapshot/lease 拥有旧代执行连续性；PluginManager 当前代映射拥有对外 catalog。prepare/publish/rollback、drain、MCP/channel/service/dashboard 事务不变。
- 能力变化：旧 turn 继续使用 lease 固定旧 snapshot；新查询只见当前代，禁用后不暴露旧能力；命令转换语义保持。
- 性能变化：从遍历全部 retained namespace 收敛到当前 generation 数量；通常插件数很小，不作量化声明。
- 测试新增：保留旧 lease 时 v1→v2 只暴露 v2，禁用后两个公开视图均为空。
- 测试删除及原因：无。
- 验证结果：副手定向 `156 passed`、全量 `1683 passed`、pyright `0 errors`；主线合入后 hot-reload 定向 `141 passed`，`git diff --check` 通过。
- 残余风险：旧 generation 仍按 lease 正常保留资源，这是连续性设计，不应为“清理视图”提前销毁。

### `b35cd6f1` `fix(scheduler): tighten time and job boundaries`

- 范围：定时任务持久化 schema、时间/时区解析、cron 依赖边界、后台执行 ownership 与调度工具输入。
- 原问题：执行中的循环任务被用户取消后会在 finally 中重新写回；坏 interval 等持久化字段可进入运行时；`HH:MM` 没有把注入时钟转换到请求时区；长 misfire 逐 interval 循环；缺 APScheduler 时启用的自写 cron 与正式路径连 weekday 语义都不一致；`chat_id` 会把 `None`/对象静默转成字符串。
- 为什么这样修改：JobStore 在 JSON 边界严格校验当前 15 字段 schema 与领域不变量；scheduler 显式持有后台 task，区分用户取消与 shutdown；周期推进改为算术跳跃；直接使用 requirements 中的 APScheduler 和 ZoneInfo，删除约百行近似 fallback；工具边界拒绝错误类型。
- 不变量与拥有层：ScheduleTool 拥有外部工具输入；JobStore 拥有持久化 schema；SchedulerService 拥有运行 task、取消与重排；APScheduler 拥有 cron 语义。当前 live schedules 文件与历史提交均为相同 15 字段，不新增迁移或兼容层。
- 能力变化：合法 at/after/every、发送顺序、soft job 工具限制和任务恢复保持；用户取消不再复活；shutdown 中断的循环任务仍持久化供重启恢复；缺少必需 APScheduler 时直接失败，不再运行语义不一致的替代实现。
- 性能变化：365 天停机、1 秒 interval 的推进由逐秒循环改为 O(1) 算术计算；cron 删除自写扫描实现，不声明额外倍数。
- 测试新增：坏 schema/重复 ID/时区、目标时区 HH:MM、5/6 字段 cron 与 weekday、显式取消和 shutdown 恢复、后台异常唯一日志、非法 interval、chat_id 边界、长 misfire。
- 测试删除及原因：无。
- 验证结果：副手全量 `1701 passed`、目标文件 pyright `0 errors`；主线合入后定向 `102 passed`，`git diff --check` 通过。
- 残余风险：当前发送工具没有结构化 delivery outcome 与幂等键，因此本批没有擅自新增自动 retry，避免重复发送。

### `246583c5` `fix(bootstrap): 收束 AppRuntime 资源生命周期`

- 范围：AppRuntime 启停、primary/server/watcher task 监督、channel/managed service 启动回滚、CoreRuntime/MemoryRuntime/dashboard 资源关闭。
- 原问题：启动取消不稳定进入回滚；dashboard/chat/watcher 提前返回或失败不会结束仍运行的核心任务；`asyncio.gather` 首个 sibling 失败后其他任务可能继续；单个 cleanup 失败或取消会跳过后续资源；IPC/channel/service 的部分启动存在资源泄漏路径；dashboard compile task 异常可跳过其他 closeable。
- 为什么这样修改：一个 primary supervisor 唯一持有 runtime tasks，失败和取消时 cancel 并确定性 await 全部 siblings；已消费的 watched task 立即移交引用；通用 bootstrap cleanup runner 在调用方取消时仍完成所有已取得资源并重新抛出首个失败；各 owner 的启动回滚和关闭路径接入同一语义。
- 不变量与拥有层：AppRuntime 拥有核心任务、server/watcher 和总 shutdown 顺序；ChannelHost 拥有 channel scoped resources；PluginServiceHost 拥有 managed subprocess；CoreRuntime/MemoryRuntime/dashboard lifespan 各自拥有其资源。startup/run 原错误保持主异常，rollback/shutdown 错误作为 cause 保留。
- 能力变化：启动顺序、插件热重载事务、channel、CLI、proactive 和 dashboard 功能保持；server/watcher 正常退出或失败会触发全局收尾；外部取消完成 cleanup 后原样重抛；primary sibling 的异步 finally 完成前 `run()` 不返回。
- 性能变化：无吞吐收益声明；新增常数级 task supervisor 与 shutdown bookkeeping，换取确定性资源回收。
- 测试新增：server 正常返回/失败、run 与 shutdown 双失败、primary sibling 失败与异步 finally、外部取消、startup rollback、cleanup 继续执行、dashboard compile 取消/异常、IPC 构造回滚、channel scoped resource、managed service rollback。
- 测试删除及原因：无。
- 验证结果：副手全量 `1697 passed`；主线合入后定向 `73 passed`、全量 `1716 passed in 20.73s`；修改文件 pyright `0 errors`，`git diff --check` 通过。
- 残余风险：shutdown 仍按既有顺序串行执行，未引入并行关闭；这是为保持资源依赖顺序和错误上下文，不宣称关闭耗时优化。

### `deb87c8a` `fix(memory2): enforce storage and boundary ownership`

- 范围：MemoryStore2 共享 SQLite 连接、embedding/provider 数据边界、query rewrite task 回收和 post-response worker 错误传播。
- 原问题：`check_same_thread=False` 的共享连接仅有部分写路径加锁，8 线程并发写可复现 `UNIQUE constraint`、`InterfaceError` 和 SQLite API misuse；数据库 embedding 与 provider batch 结构未经完整校验；query rewrite 超时只 cancel 不 await；后台 memory worker 顶层吞掉存储错误。
- 为什么这样修改：由 store 的同一 `RLock` 串行化该连接的全部公开操作；在 SQLite/provider 唯一边界校验 JSON、有限数值、index、数量和维度；创建 task 的 rewriter 负责 cancel 后确定性 await；worker 将错误交给已有 EventBus observer 隔离层记录，同时让显式 ingest 看见失败。
- 不变量与拥有层：MemoryStore2 拥有单连接并发与持久化 JSON；Embedder 拥有外部响应 schema；QueryRewriter 拥有其两个 task；EventBus 拥有后台观察者隔离。Retriever 的 cosine/keyword/RRF、情景 lane、scope、注入预算和热重载均未修改。
- 能力变化：正常召回、排序、去重、合并、遗忘和 full-context 保持；并发写不再互相破坏；坏向量/响应即时带上下文失败；超时不遗留 LLM task；后台失败仍不阻断已提交用户回复，但显式 ingest 不再假成功。
- 性能变化：共享单连接操作由隐式争用改为显式串行，未宣称吞吐提升；故障并发不再重试或产生重复写。正常检索 SQL、候选数和排序复杂度不变。
- 测试新增：32 次并发同摘要写、SQLite embedding 损坏、provider malformed/index/数量/维度、query timeout task 回收和 worker 存储故障传播。
- 测试删除及原因：无。
- 验证结果：副手两轮全量最终 `1723 passed`；主线独立定向 `53 passed`；修改文件 pyright `0 errors`，`git diff --check` 通过。
- 残余风险：单 SQLite connection 仍限制并行读吞吐；改为连接池或读写分离会改变事务与 sqlite-vec ownership，本批没有无证据扩张。

### `0f25007b` `fix(persistence): harden JSON durability boundaries`

- 范围：共享 JSON/文本原子写底座、Plugin KV、plugin/package manifest 与本地 MCP registry 配置。
- 原问题：损坏或无权限 JSON 被静默当默认空状态；Plugin KV 和 MCP 配置直接覆盖写且 MCP 保存失败被吞；manifest 原子替换没有 file/directory fsync 和统一失败清理。
- 为什么这样修改：仅 `FileNotFoundError` 解释为可选缺失；共享底座在同目录唯一临时文件完成序列化、flush、file fsync、replace、directory fsync；各持久化 owner 复用同一耐久契约，MCP 边界按 live 配置与历史 schema 校验。
- 不变量与拥有层：`load_json` 拥有文件读取/JSON 边界；原子写底座拥有临时文件与落盘顺序；PluginKVStore、manifest loader 和 McpServerRegistry 分别拥有领域 schema。live `mcp_servers.json` 的 `{"servers": {}}`、合法 `{}` 和缺失 servers 均保持有效。
- 能力变化：合法 KV/TOML/MCP 配置与插件热重载格式不变；损坏数据和保存失败改为 fail-loud；MCP connect/disconnect、注册事务与运行协议未修改。
- 性能变化：持久化成功路径增加必要的 file/directory fsync，属于耐久性成本，不宣称提速；并发 writer 使用独立 tmp，不再互相窃取或清理 staging 文件。
- 测试新增：缺失与损坏 JSON、并发 writer、序列化/file fsync/replace/directory fsync、cleanup 双失败、合法空 MCP schema。
- 测试删除及原因：无；更新了过去错误期待坏 JSON 返回 default 的测试。
- 验证结果：副手两轮全量最终 `1727 passed`；主线独立相关测试 `115 passed`；修改文件 pyright `0 errors, 0 warnings`，`git diff --check` 通过。
- 残余风险：directory fsync 在 replace 后失败时新目标可能已经可见，函数仍抛错表示耐久性未确认；文件存储不提供跨 writer CAS 或业务级回滚。

### `e323325b` `fix(frontend): 收紧运行时边界与请求生命周期`

- 范围：Chat HTTP/WebSocket/上传链路、Dashboard 分页与插件边界、请求取消和 legacy plugin workbench dispatch。
- 原问题：非 2xx、坏 JSON 和缺字段响应可变成空列表；快速切换时旧请求覆盖新状态；WebSocket 等待发送缺少 error/close 收尾；上传后的消息保留即将 revoke 的 blob URL；view 切换重复请求；legacy workbench 因 dispatch identity 重建，初稿修复又暴露 stale state 闭包。
- 为什么这样修改：在 fetch/socket/plugin 唯一外部边界校验实际消费字段；各请求 owner 持有 AbortController；上传成功后只保留服务端 URL；view effect 唯一触发加载；稳定 dispatch 通过 `useLatestReader` 在事件期读取最新 plugin state，避免重建和旧闭包。
- 不变量与拥有层：Chat API/WebSocket 拥有外部 frame/schema；PromptInput 拥有本地 blob URL；页面组件拥有请求取消；Dashboard plugin runtime 拥有单 panel 隔离；legacy PluginMain 拥有一次 DOM 初始化。后端 API、插件热重载协议与视觉语言未修改。
- 能力变化：真实错误进入现有错误 UI；快速切换不回填旧结果；附件提交后预览继续有效；中断帧结束 streaming 状态；坏 plugin panel 仍只隔离该 panel 并显式记录。
- 性能变化：移除 selectView 与 effect 的双重请求；session 选择不再重拉 sessions；legacy graph/workbench 不因无关父 rerender 重建 canvas、重复拉 snapshot 或重绑 listener。未机械 memo 普通渲染。
- 测试新增：无；仓库没有前端 test runner，未为本批引入框架。
- 测试删除及原因：无。
- 验证结果：副手 `typecheck`、`lint`、`build` 通过；主线合入前主审再次运行 typecheck/lint 均通过，`git diff --check` 通过；build 仅有既有大 chunk warning。
- 残余风险：前端外部边界目前依赖静态检查和真实 build，缺少可执行组件测试；后续引入 test runner 应优先覆盖请求竞态、socket close 与 legacy dispatch 最新状态。

### `c7fb6a87` `fix(proactive): decouple source cache refresh`

- 范围：ProactiveSourceSpec、default/wake source 读取契约、插件 MCP 工具校验、开发文档与热重载探针；同步审计 Feed、Steam、Calendar、Fitbit 四个已启用 proactive MCP。
- 原问题：外部数据刷新由 `default_proactive` 私有 `DefaultSourcePollModule` 持有；启用互斥的 `wake_proactive` 后该模块随默认流程消失，`fetch_tool` 只能读到旧缓存。
- 为什么这样修改：缓存新鲜度属于 MCP 的外部 API/持久化边界，收回各 MCP 自己的 lifespan、后台服务或按需读取路径；两套 proactive 只通过 `fetch_sources_async()` 并发读取稳定快照。
- 不变量与拥有层：MCP 拥有外部抓取、刷新周期和缓存；proactive runtime 拥有读取、通道归类、决策与 ACK。source 稳定键、分页、单源故障隔离、事件 ID、排序和投递语义不变。
- 能力变化：删除宿主 `poll_tool/poll_interval_seconds` API 和 105 行默认流程后台 poller；切换 proactive package 不再影响缓存刷新。Fitbit 删除原本未被实际使用的 source interval 声明，监控服务自己的 5 分钟轮询保持。
- 外部插件：Feed `8aaeab3` 已由 FastMCP lifespan 持续刷新并让首次读取等待刷新结果；Steam `326c055` 在快照过期时按需刷新；Fitbit `a680ac6` 合入 GitHub main、升至 `1.1.1` 并从 GitHub 重装；Calendar 每次读取直接查询 live API，无本地陈旧缓存问题。
- 性能变化：default runtime 不再创建或持有 source 轮询 task；同一 tick 的 source 读取仍由 `asyncio.gather` 并发执行。网络刷新从 agent 主动链移出后，不再绑定 proactive 生命周期。
- 测试新增/调整：默认 phase graph 明确不含 `default.source.poll`；source、MCP catalog、热重载和探针契约移除宿主轮询字段；未删除业务场景测试，只删除已废弃 poller 的实现测试。
- 验证结果：主仓库定向 `184 passed`，全量 `1724 passed`，范围 pyright `0 errors`（45 个既有 warnings），skill 校验通过；Fitbit `5 passed`、pyright `0 errors`；运行缓存与源码一致，plugin doctor 为 healthy。
- 残余风险：Feed 的网络刷新仍受各订阅源可用性影响，但系统级刷新失败会显式暴露且后台继续重试；不再由 proactive runtime 提供第二套隐藏 fallback。

### `902f6f44` `fix(akasha): harden runtime sidecar ownership`

- 范围：Akasha engine 的历史图缓存、dashboard 图读取器、图快照 sidecar、诊断 JSON 和事件订阅生命周期。
- 原问题：历史 query 每次通过 `MessageEmbeddingStore.list_until` 扫描 8,920 条缓存并回查 messages；dashboard 每次轮询重扫完整图签名；后台 snapshot 失败、损坏 JSON/BLOB、短 embedding batch 和 orphan cache 可被空结果或部分写入掩盖；engine 订阅未随 closeables 释放。
- 为什么这样修改：启动边界一次加载 message embedding 的 turn key 与 timestamp，query 在同一 graph lock 快照上做内存 cutoff；以 dashboard store connection 的 `PRAGMA data_version` 驱动签名 cache；所有 sidecar/embedding/diagnostic 边界 fail-loud；创建者持有 subscription 和 rebuild thread。
- 不变量与拥有层：sessions.db 仍拥有消息事实与时间；engine 拥有其内存 cache、dense index 和订阅；graph reader 拥有 rebuild thread 与签名 cache；snapshot loader 拥有 JSON/BLOB 边界。情景补全/增量、full-context、scope、候选排序、threshold、图节点/边语义和 dashboard API 不变。
- 能力变化：合法召回与图输出保持；损坏 sidecar、orphan embedding 和 provider 短 batch 立即失败且不产生部分 cache；hot reload 不遗留旧 Akasha handler；dashboard close 时先等待 reader，再关闭 store。
- 性能变化：live 8,920 条 embedding 的历史 cutoff 由约 `425.1 ms/query` SQL 路径降为约 `0.463 ms/query` 内存过滤；无外部 DB commit 时图签名不再重复扫描 4,439 节点和 153,354 边。首次 cache 加载和发生 commit 后的一次签名扫描保留。
- 测试新增：同锁 snapshot/timestamp、orphan cache、短 embedding batch、signature cache 与外部 commit invalidation、rebuild failure、closeable 逆序、损坏 snapshot/诊断 JSON、event subscription close。
- 测试删除及原因：无。
- 验证结果：主线独立 Akasha `56 passed`，全量 `1732 passed`，修改生产文件 pyright `0 errors, 0 warnings`，`git diff --check` 通过；live DB/snapshot 只读完整性与签名相符。
- 残余风险：完整 embedding cache 仍是 engine 启动成本；外部 graph commit 后仍必须做一次真实 signature scan，这是正确性成本，不用 TTL 或近似值替代。

### `44e974c5` `fix(default-memory): close facade resources explicitly`

- 范围：default_memory façade 的 EventBus wiring、内部依赖不变量、workspace storage 维度、inspector/dashboard 边界。
- 原问题：engine 创建的三类 memory 订阅未进入 runtime closeables，reload 后旧 handler 可重复摄入；缺失 retriever/memorizer/store/embedder 被空结果或 no-op 掩盖；新 workspace provisioning 固定 1024 维，可能与配置 embedding 维度错配并触发 sqlite-vec fallback；inspector 在未绑定 engine 时仍被视为 active，损坏 JSONL 被显示为空。
- 为什么这样修改：engine 明确持有 subscription 并随 MemoryRuntime 逆序关闭；构造后依赖通过集中 require owner fail-fast；storage 使用实际 `output_dimensionality`；inspector 只接受真实 default engine，dashboard 只把文件缺失解释为暂无记录。
- 不变量与拥有层：DefaultMemoryEngine 构造函数拥有 store/embedder/memorizer/retriever；engine 与 MemoryRuntime 共同拥有 subscription 生命周期；Config.memory.embedding 拥有向量维度；inspector/dashboard 拥有各自激活和 JSONL 边界。召回 lanes、RRF、scope、合并/遗忘、consolidation、ranking、threshold、budget 和 prompt 不变。
- 能力变化：reload 不再遗留旧 memory handler；内部契约违反不再伪装为空召回或假成功；新库 sqlite-vec schema 与真实 embedding 维度一致；坏 inspector 数据 fail-loud。
- 主审修正：首次实现若第二个 `EventBus.on()` 失败会泄漏第一个 subscription 且锁死重试；amend 后失败时逆序关闭本轮订阅，清理失败以 `BaseExceptionGroup` 保留，全部成功后才设置 wired。恢复后可重试且不重复注册。
- 性能变化：避免错维 schema 导致的 sqlite-vec fallback；正常 query/embedding/index 调用次数不变，不宣称召回延迟倍数提升。
- 测试新增：subscription 生命周期与部分注册失败回滚/重试、缺失内部依赖 fail-fast、自定义 embedding 维度、inspector 无 engine inactive、损坏 JSONL。
- 测试删除及原因：无。
- 验证结果：主线独立定向 `123 passed`，全量 `1737 passed`，修改生产文件 pyright `0 errors`（64 个既有 warnings），`git diff --check` 通过。
- 残余风险：既有历史错维 vec schema 没有证据支持自动迁移，本批只保证新建 workspace 正确；迁移必须另行设计可恢复 DB 流程。

### `82b6056d` `fix(session): atomically replace trimmed history`

- 范围：被动轮次在内容安全或上下文长度重试成功后的 history trim、SessionManager 内存 cache、SessionStore 消息与 embedding 持久化。
- 原问题：retry 成功后先裁剪内存，再调用只会追加无 ID 消息的 `save_async()`；SQLite 中旧消息没有删除，进程重启后被裁历史会复活，保存失败时内存和磁盘也会分叉。
- 为什么这样修改：reasoner 只提交裁剪意图；SessionManager 持有 per-session 写锁，SessionStore 在一个 `BEGIN IMMEDIATE` 事务内更新 session metadata、删除未保留消息及其 embedding、追加尚未持久化消息；事务成功后才更新内存视图。
- 不变量与拥有层：reasoner 拥有 retry plan；SessionManager 拥有 session 锁和内存 cache；SessionStore 拥有 SQLite 消息、FTS、embedding 清理和 `next_seq` 高水位。保留消息 ID/seq、正常 append-save、tool discovery、stream、media 和热重载语义不变。
- 能力变化：裁剪结果跨重启稳定；数据库失败时消息、metadata、`last_consolidated`、embedding 和内存全部保持原状；其他 session 的消息与 embedding 不受影响。
- 性能变化：正常无 retry turn 不增加数据库操作；仅 retry trim 执行一次同事务消息枚举和删除，避免后续重启重新加载无效历史及对应 embedding。
- 测试新增：真实重载不复活、保留 ID/seq 与后续高水位、真实 `message_embeddings` 删除/保留/跨 session 隔离、DELETE 失败时消息/metadata/embedding/内存共同回滚。
- 测试删除及原因：无。
- 验证结果：副手全量 `1734 passed`；主线合入后定向 `55 passed`、全量 `1739 passed in 20.53s`；修改生产文件 pyright `0 errors`（274 个既有 warnings），`git diff --check` 通过。
- 残余风险：trim 路径仍按 session 当前消息数做一次线性 ID 枚举；只在模型 retry 成功时触发，当前没有证据支持引入更复杂的临时表或批量阈值优化。

### `7d7eeb67` `fix(plugins): make plugin installation atomic and bounded`

- 范围：Git 插件 source/ref 输入、plugin metadata 与 MCP runtime 路径边界、cache 发布/回滚、installed source resolver 和 package metadata schema。
- 原问题：同版本重装会先删除旧 cache，clone/依赖准备/manifest 写入失败可使插件消失或留下半成品；name/version/marketplace、source symlink、MCP cwd 可越过目标目录；多个可见版本或坏 cache 会被字典序选择或静默当成未安装；错误 package 类型被 `bool()`/`str()` 归一化。
- 为什么这样修改：在隐藏 staging 完成 clone、内部 symlink 校验和 MCP 依赖准备，旧普通版本在长耗时阶段持续可发现；只在准备完成后执行最短 rename 切换并保留 hidden backup，manifest 失败恢复旧 cache；installed resolver 对可见结构违反 fail-loud。
- 不变量与拥有层：installer 入口拥有 URL/ref/marketplace/name/version 与目标路径；cache activation 拥有 staging/backup/rollback；manifest owner 继续拥有原子 TOML 写；resolver 拥有已安装 cache 结构；package loader 拥有 package.toml schema。插件 data/config、同版本重装、热重载和 MCP 命令语义保持。
- 主审修正：第一版在 pip/venv 阶段先隐藏所有旧版本且 resolver 忽略临时 symlink，会让 watcher 长时间看见插件消失；二审改为 staging 准备期间旧版本持续可见。主线又补充拒绝可见普通文件，避免 installer 发布成功后 resolver 才因坏版本项失败。
- 能力变化：branch/tag/commit SHA ref 均先解析为 commit 后 detached checkout；合法仓库内 symlink 可安装并复制为普通内容，断链/循环/越界 symlink 被拒绝；cache root/marketplace/plugin/version/plugin.py 的可见 symlink、坏路径、缺文件和版本冲突显式失败。
- 性能变化：正常安装增加一次 source tree 线性 symlink 扫描和常数次 rename；长耗时 pip/venv 本就存在且移到不可发现 staging。运行时 resolver 只增加目录结构校验，不增加网络或依赖安装。
- 测试新增：依赖准备期间真实 resolver 仍看到旧版本、prepare/manifest 失败回滚、unsafe metadata/MCP cwd、内部/越界 symlink、probe 模块清理、branch/tag/SHA 与 option-like ref、cache symlink/缺 plugin.py/多版本/普通文件、package 类型边界。
- 测试删除及原因：无。
- 验证结果：副手二审定向 `24 passed`、相关 `159 passed`、全量 `1749 passed`；主线补刀后相关 `184 passed`、live 安装 cache 20 个来源全部可解析、修改生产文件 pyright `0 errors, 0 warnings`、全量 `1752 passed in 20.31s`，`git diff --check` 通过。
- 残余风险：最终发布仍由数次同文件系统 rename 组成，存在极短目录切换窗口；跨进程并发 installer 尚无 owner，当前 CLI 是单安装流程，没有证据支持新增锁服务。

### `b1640920` `fix(peer-agent): enforce lifecycle ownership and response schemas`

- 范围：Peer Agent 的 AgentCard 发现、A2A `tasks/get` 解析、pending task 归属、后台 poller 与子进程启停。
- 原问题：live AgentCard 可覆盖配置中的进程身份和路由；pending 只以远端 task ID 为键，跨 agent 会冲突；未知状态、坏 JSON/schema 和 JSON-RPC error 可被永久轮询；同 agent 多任务完成一个就会提前终止共享进程；poller 失败 task 可被静默覆盖；spawn/health/shutdown 的取消、日志 fd 和并行清理存在泄漏或丢错路径。
- 为什么这样修改：配置继续唯一拥有 name/base URL，live card 只补充展示信息；外部 AgentCard/A2A 响应在唯一边界按实际消费字段解析；pending 使用 `(agent_name, task_id)`；终态先投递通知、最后一个任务再回收进程；ProcessManager 明确持有进程登记、父日志 fd、健康等待和 shutdown 聚合。
- 不变量与拥有层：PeerAgentConfig 拥有本地身份与路由；card/status parser 拥有外部 schema；Poller 拥有 pending、终态通知和 agent 任务计数；ProcessManager 拥有 subprocess、per-agent lock 和关闭顺序。网络/HTTP 临时失败保留 pending；协议、MessageBus、ownership 和进程系统调用错误 fail-loud。
- 主审修正：终审删除了副手新增的 `PeerProcessRetryableError`。`asyncio.subprocess` 的普通 `OSError` 没有统一可恢复契约，把 PermissionError、坏 fd 等全部无限重试会再次掩盖真实错误；现在只把明确的 HTTP transport/timeout 作为 poll retry，`ProcessLookupError` 表示目标已消失，其余进程错误原样暴露且 ownership/pending 不被误删。
- 能力变化：跨 agent 同 ID 任务互不覆盖；同一 agent 的多个 pending 全部完成后才停止进程；queued/running/submitted/working 与 completed/failed/canceled 语义明确；坏远端响应转为一次显式失败通知并回收；poller 已失败时再次 start 会先重抛旧异常；启动或关闭部分失败保留原始错误并继续清理其余资源。
- 性能变化：`shutdown_all` 按 agent 并行回收，关闭耗时由各进程超时之和收敛为最慢进程耗时上界；正常轮询间隔、每任务查询次数和 HTTP retry budget 不变。
- 测试新增：真实 loop 的协议失败通知、身份/路由不被 live card 覆盖、复合任务键、同 agent 多任务、通知与进程错误边界、done task 重抛、spawn/cancel/log-fd fault injection、并行 shutdown 与错误聚合。
- 测试删除及原因：删除“任意进程 OSError 都可自动恢复”的人工测试；真实 owner 没有该恢复协议，保留会固化静默无限重试。
- 验证结果：副手终审定向 `52 passed`、全量 `1782 passed`、pyright `0 errors`；主审补刀后定向 `51 passed`、修改生产文件 pyright `0 errors, 0 warnings`，`git diff --check` 通过。
- 残余风险：任务提交成功到登记 pending 之间仍依赖当前 Tool 调用顺序；若未来允许同一 agent 高并发提交，需要让 submit 与最后任务 terminate 共享显式 lease，不能仅靠增加下游空值检查解决。

### `7baf4169` `fix(filesystem): harden mutation and atomic writes`

- 范围：`read_file`、`write_file`、`edit_file`、`list_dir` 的错误边界，同路径 mutation lock，以及共享文本原子写底座。
- 原问题：四个文件工具用宽泛 `except Exception` 把内部编程错误伪装成用户文件错误；mutation callback 异常或取消会泄漏 lock map，等待者存在时又可能过早移除 key、产生第二把锁；write/edit 直接覆盖目标，失败可截断文件，并丢失 BOM/换行与 executable mode。
- 为什么这样修改：文件工具只转换可由用户处理的 `PermissionError`/`OSError`；锁状态显式计数当前持有者和等待者并在 `finally` 回收；write/edit 复用同目录临时文件、file fsync、原子 replace 和 directory fsync 的统一写入契约。
- 不变量与拥有层：路径解析和文件/目录状态由文件工具边界拥有；规范化路径对应的 mutation 串行性由 `_run_with_file_mutation_lock` 拥有；临时 fd、目标 mode、写入/replace/fsync/cleanup 顺序由 `atomic_write_text` 拥有。BOM、CRLF/mixed newline、diff 文本和读取统计语义不变。
- 主审修正：副手首版通过 `os.umask(0)` 读取进程 umask，模块锁无法阻止其他线程在窗口内创建出权限过宽的文件；二审改为高熵同目录临时名加 `os.open(O_CREAT|O_EXCL|O_WRONLY, 0o666)`，直接让内核应用当前 umask，已有目标再复制 `S_IMODE`。主线删除了一个只暂停临时 `open`、实际无法复现旧 `umask(0)` 窗口的误导性并发测试。
- 能力变化：内部 AssertionError/TypeError 不再被本层吞掉；同文件并发写不会因异常、取消或 waiter 交接产生双锁；覆盖写失败保留旧文件；BOM、混合换行和 executable mode 保持；新文件权限与普通文本创建一致。
- 性能变化：正常 write/edit 增加同目录临时文件、两次 fsync 和原子 rename，这是耐久性成本；不同路径仍可并行，同路径只增加常数级引用计数。read 分页仍完整扫描以保留总行数、总字节和解码提示，没有用能力退化换取表面提速。
- 测试新增：mutation 异常/取消清理、等待者交接、内部错误冒泡、写入/编辑 BOM/CRLF/mode、原子写旧 mode/新文件权限、编码失败保留旧文件和清理 tmp。
- 测试删除及原因：删除副手新增的“临时文件创建期间 mode 并发”测试；它没有在旧实现实际修改 umask 的区间同步，旧坏实现也会通过，无法证明根因。
- 验证结果：副手二审定向 `61 passed`、全量 `1758 passed`；主线删除无效测试后定向 `60 passed`、修改生产文件 pyright `0 errors, 0 warnings`，`git diff --check` 通过。
- 残余风险：`read_file` 为保留完整统计仍线性扫描大文件；当前没有 profile 证据支持改变输出协议或增加索引。原子 replace 只显式保持 POSIX mode，不承诺保存项目当前未使用的跨平台扩展属性。

### `33e22021` `fix(plugins): complete scoped resource cleanup`

- 范围：`PluginScope` 所有 task/process/deferred cleanup，以及 PluginManager 的候选回滚、代际回收、快照关闭和全量 terminate 路径。
- 原问题：调用方取消可传播到正在执行的 cleanup 并截断后续资源；作用域 task 在运行期失败只会等到关闭才暴露；async subprocess 已有 returncode 时没有完成 `wait()` transport 收尾，kill 后等待又可能无界；manager 在关键关闭步骤收到取消时可能留下半清理注册状态。
- 为什么这样修改：每个 cleanup 在独立 task 中执行，调用方取消只延迟恢复而不传播到资源动作；manager 以统一 `_complete_critical()` 屏蔽关键生命周期操作，消费 scope failure 后再注销模块和注册表；process 使用有界 terminate、kill、二次 wait。
- 不变量与拥有层：`PluginScope` 唯一拥有其 task/process/deferred cleanup 的逆序和幂等关闭；PluginManager 拥有插件 terminate、scope failure 聚合、模块/工具注册表卸载和最终取消恢复。cleanup 自身取消是资源失败，调用方取消是完成全部清理后重新抛出的控制流，两者不混淆。
- 能力变化：插件后台 task 异常在发生时立即记录真实 traceback，关闭时仍进入结构化 failure；外部取消不再造成资源遗漏，也不被静默吞掉；已退出 process 仍完成系统收尾，强杀失败或超时明确进入 cleanup failure。
- 性能变化：每个 cleanup 增加一个短生命周期 asyncio task，正常关闭有少量调度成本；进程等待都有明确上限，取消或异常路径不再无限挂起。正常插件加载、事件分发、工具调用和热重载租约语义不变。
- 测试新增：task 运行期失败与关闭聚合、幂等 close、外部取消完成全部 cleanup、async process 已退出 wait 和 timeout kill、manager terminate 取消后仍消费 scope failure。
- 测试删除及原因：无。
- 验证结果：副手全量 `1804 passed`；主线合入后独立定向 `146 passed`，修改范围 pyright `0 errors`（24 个既有 warnings），`git diff --check` 通过。
- 残余风险：`track_async_process()` 当前没有生产调用者；未来接入时需按实际子进程协议确认退出码语义。kill 后二次 wait 超时会显式失败，不做无证据的无限 retry。

### `de066492` `fix(proactive): make persistence and dedupe failures explicit`

- 范围：主动状态 SQLite 时间与事务、消息语义去重、`RecentProactiveMessage` 内部契约，以及 `ProactiveLoop` 状态存储生命周期。
- 原问题：损坏的持久化时间会被解释为“没有历史”，可能重复发送或重新打开 gate；context-only 两次写入不是显式原子操作；消息去重把 provider、网络、坏 JSON 和内部编程错误全部当成“不重复”放行；recent message 通过 dict、`getattr` 和字符串时间兼容掩盖内部契约错误；owned state store 没有明确关闭 owner。
- 为什么这样修改：SQLite 已存在 row 的时间由状态存储边界严格解析，事务失败统一回滚；deduper 直接传播 provider 与解析失败并严格校验模型 JSON schema；producer、factory、deduper 和 resolver 统一使用 typed recent message 与 `MessageDeduper` 协议；bootstrap 显式把状态存储 ownership 转交给 loop。
- 不变量与拥有层：`ProactiveStateStore` 拥有持久化 schema、时间解析和事务；`MessageDeduper` 拥有模型响应 schema；Sensor 拥有 recent message 构造；`ProactiveLoop` 只关闭明确归其所有的 store。resolver 信任 `tuple[bool, str]`，不再用 `str()`、空串默认值或 degraded 字符串协议重复归一化。
- 主审修正：拒绝副手首版基于异常类名的 fail-open、`dedupe_degraded:` 字符串协议和 mapping 兼容；二审后主线继续以 Protocol 删除 resolver 的 `Any` 与 `str(reason or ...)`，确保边界之后信任已验证类型。
- 能力变化：正常重复/非重复判断、delivery dedupe、投递和 ACK 不变；dedupe 不可用时当前 tick 明确失败且不发送，由 loop 边界记录后在下一 tick 重试。非法历史时间不再被当成新状态。loop 收尾失败仍设置 stopped 信号并向调用者暴露错误。
- 性能变化：没有增加 LLM 调用或 SQLite 查询；只在读取实际历史时间时增加严格解析，在写失败时增加 rollback。正常发送路径调用次数不变。
- 测试新增：坏时间、COUNT 契约、两步写入 rollback、provider/JSON/schema fail-loud、recent message 类型契约、owned store close 失败与幂等关闭。
- 测试删除及原因：删除二审拒绝的 degraded fail-open 和 resolver degraded trace 测试；它们只固化会在去重不可用时放行消息的错误设计，不属于应保留能力。
- 验证结果：副手二审定向 `136 passed`、全量 `1813 passed`；主线补刀后定向 `57 passed`、修改生产文件 pyright `0 errors`（97 个既有 warnings）、全量 `1818 passed in 20.79s`，`git diff --check` 通过。
- 残余风险：模型去重不可用会推迟本轮主动消息，这是明确的 fail-closed 产品取舍；当前没有可靠的本地等价算法可以安全降级，不能用“可用性”名义重新引入无标记放行。

### `8a231f5b` `fix(peer-agent): make submit lifecycle atomic`

- 范围：Peer Agent 冷启动、A2A submit、pending 登记与最后任务进程回收之间的并发 ownership。
- 原问题：任务提交成功到 pending 登记之间没有和最后任务回收共享锁，旧任务可能在新任务已冷启动但尚未登记时终止共享进程；启动或提交错误被宽泛 catch 转成普通 JSON；冷启动后的提交失败会遗留新进程；重复 task id 只有通用 `ValueError`。
- 为什么这样修改：Poller 提供 per-agent submission lease，工具在同一 lease 内完成 `ensure_ready -> submit -> register`，终态回收也在该 lease 内重新判断其他 pending；`ensure_ready` 返回本次是否取得新进程 ownership；失败只回收本次独占且没有其他 pending 的进程。
- 不变量与拥有层：ToolRegistry 拥有统一工具错误呈现；PeerAgentTool 拥有一次 submit/register 事务；Poller 拥有 pending catalog 和 per-agent lease；ProcessManager 拥有受管子进程。外部健康 peer、旧 pending、跨 agent 并行、A2A payload 与轮询间隔不变。
- 主审修正：副手首版把终态通知也放入 submission lease，慢 MessageBus 会阻塞同 agent 新提交；主线把通知移到 lease 外，task 在 catalog 中继续持有 ownership。副手 `finally` 中的 terminate 失败会覆盖原 submit/取消错误；主线在双故障时用 `BaseExceptionGroup` 同时保留，单故障原样重抛。
- 能力变化：同 agent 新提交不会被旧任务的最后回收竞态杀死；冷启动后 submit/register 失败不会泄漏独占进程；外部健康 peer 和已有 pending 不被误终止；坏响应、编程错误与取消不再伪装成成功 JSON；duplicate 显式失败且不覆盖既有任务。
- 性能变化：同 agent 的冷启动、网络 submit 与登记按 lease 串行，不同 agent 仍并行；终态通知不占锁，因此慢通知不再增加新提交等待。没有增加 HTTP 请求或 poll 次数。
- 测试新增：真实 Event 交错验证慢通知期间新提交可完成且共享进程不终止；冷启动失败回收、外部 peer/旧 pending 保护、取消传播、duplicate 保留、submit 与 cleanup 双错误聚合。
- 测试删除及原因：无。
- 验证结果：副手定向 `58 passed`、全量 `1813 passed`；主线补刀后定向 `59 passed`、修改生产文件 pyright `0 errors`（16 个既有 warnings）、全量 `1828 passed in 20.72s`，`git diff --check` 通过。
- 残余风险：远端接受任务但本地在读取 task id 前连接彻底失败时，当前 A2A 接口没有可确认的 cancel 或幂等查询键；不能假造本地恢复。per-agent lock map 只按启动配置中的有限 agent 名增长。

### `7df557b5` `tune(proactive): 提高内容唤醒主动性`

- 范围：Wake content hazard 的随机阈值分布与上限。
- 原问题：线上阈值被 Gamma 尾部抽样到并持久化为 `2.0`，当时 `hazard + preference_pressure = 1.012`；按当时 rate 计算的理论稳定水位约为 `1.53`，没有更强新内容时无法达到阈值。
- 为什么这样修改：将抽样 scale 从 `1/3` 降为 `1/4`，并将上限从 `2.0` 收紧到 `1.0`；已持久化的高阈值会被现有 clamp 自动收敛，不需要改状态 schema。
- 不变量与拥有层：content 时效、兴趣证据、个人偏好和泄漏积分由 hazard 层拥有；阈值只决定何时进入 LLM 判断，不绕过 scratchpad、investigation 或 `skip` 决策。
- 能力变化：content 进入判断的频率提高；Alert、Context、Drift、ACK、单条唤醒和 LLM 最终决策不变。线上同一状态回放由旧规则不触发改为进入 content 判断。
- 性能变化：10 万次固定种子抽样中，截断后均值从 `0.972` 降为 `0.662`，中位数从 `0.891` 降为 `0.668`；这是产品频率调校，不声称执行性能提升。
- 测试新增/调整：阈值上限和 Gamma scale 契约。
- 测试删除及原因：无。
- 验证结果：Wake 子系统 `51 passed`；修改文件 pyright `0 errors, 0 warnings`；真实状态回放通过。
- 残余风险：更主动会增加 LLM 判断机会，但实际发送仍受内容排名、个人偏好和 LLM `skip` 约束。

### `08c44c20` `fix(mcp): 升级工具结果协议协商`

- 范围：stdio MCP initialize 协商、服务端版本边界和 `tools/call.structuredContent` 校验。
- 原问题：宿主固定请求 `2024-11-05`，同时拒绝新版字段；本机 Fitbit/Calendar 的官方 Python MCP SDK 会返回 `content + structuredContent`，导致三条真实 proactive 工具同时失败。
- 为什么这样修改：宿主请求已安装 SDK 共同支持的 `2025-11-25`，保存并校验服务端回复版本；只在声明支持的协议版本下接受 JSON object `structuredContent`。
- 不变量与拥有层：MCP client 拥有 initialize/result 信任边界；旧协议越界、未知版本、非 object structured result 和坏 content 仍 fail-fast。工具对模型的文本输出继续来自 `content`。
- 能力变化：Fitbit `get_proactive_events` / `get_sleep_context` 和 Calendar `get_proactive_events` 恢复；旧 MCP server 可继续协商受支持的旧版本。
- 性能变化：无新增网络请求或重试；每次调用增加常数级版本/类型判断，不宣称性能收益。
- 测试新增/调整：新版 initialize、未知版本、合法/非法 structured result，以及测试 MCP server 的真实协商响应。
- 测试删除及原因：无。
- 验证结果：修改范围 pyright `0 errors, 0 warnings`；全量 `1831 passed`；真实 Fitbit 1.1.2 与 Calendar 1.0.0 端到端调用通过，协商版本均为 `2025-11-25`。
- 残余风险：客户端当前只消费文本/多媒体 content，不根据 tool `outputSchema` 二次验证 structured payload 内部字段；当前工具仍以 JSON 文本作为宿主契约，不应在未设计输出协议前盲目改用 structured payload。

### `082fe863` `fix(bus): preserve async lifecycle failures`

- 范围：EventBus admission task、后台 dispatcher、observer 取消和 `drain/aclose` 生命周期。
- 原问题：admission task 在取消清理中转换的真实错误被 `gather(return_exceptions=True)` 丢弃；dispatcher 异常只记日志并自动重启，`drain/aclose` 无法向 owner 报告；observer 自取消与调用方取消共用一个模糊判断。
- 为什么这样修改：总线记录 dispatcher 故障并在明确同步边界原样重抛；关闭时先停 admission、再排空 envelope 以释放 snapshot lease、最后停 dispatcher；多个清理错误用 `BaseExceptionGroup` 保留。
- 不变量与拥有层：EventBus 拥有 admission/queue/dispatcher 生命周期；observer 业务失败继续隔离并记录，总线内部故障在 `drain/aclose` fail-loud；snapshot lease 仍由 envelope/单 observer 释放。
- 能力变化：正常 emit/observe/fanout/enqueue 顺序、observer 隔离和 hot-reload snapshot 不变；dispatcher/admission 内部错误不再伪装成成功关闭。
- 性能变化：每个 dispatcher 故障增加一次小列表记录，正常 fanout/队列复杂度不变；无性能收益声明。
- 测试新增/调整：admission 取消转故障、dispatcher 失败向 `drain` 传播，并保留 observer 自取消不阻塞关闭的回归。
- 测试删除及原因：无。
- 验证结果：EventBus 直接/调用方 `587 passed`；修改文件 pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：observer 业务失败仍按设计只记录并返回失败计数，不中断用户主链；只有 EventBus 自身 dispatcher/admission ownership 错误进入关闭失败。

### `66cb2ae4` `fix(memory): expose marker read failures`

- 范围：Markdown consolidation 去重标记的尾部扫描与全文件扫描。
- 原问题：标记文件读取发生权限、I/O 或关闭错误时，两个内部 helper 都宽泛捕获并返回 `False`；调用方会把“无法确认”误判为“标记不存在”，从而重复追加同一 consolidation 内容。
- 为什么这样修改：文件系统是持久化边界，读取失败没有可在本层完成的正确恢复动作；删除静默 fallback，让原始异常触发现有事务回滚并暴露给 consolidation owner。
- 不变量与拥有层：`MemoryStore` 拥有 consolidation sidecar 与 Markdown 文件的一致性；文件不存在仍表示没有标记，存在但不可读属于存储失败。正常判重、崩溃恢复、sidecar 领先时的文件修复语义不变。
- 能力变化：正常写入与去重结果不变；存储不可读时不再冒险重复写入，而是明确失败并等待上层处理或下一次重试。
- 性能变化：正常路径删除异常捕获框架，扫描次数和复杂度不变；不声明可测性能收益。
- 测试新增：两个标记 reader 在底层 `open()` 权限失败时均传播原始 `PermissionError`。
- 测试删除及原因：无。
- 验证结果：记忆写入与语义去重定向 `13 passed`；修改生产文件 pyright `0 errors`（25 个既有 warnings）；`git diff --check` 通过。
- 残余风险：追加 Markdown 与写 sidecar 仍不是跨文件原子事务；现有 marker 恢复协议负责收敛该窗口，本次只消除了读失败时的错误判定。

### `cadcab5e` `refactor(runtime): cache passive snapshot phases`

- 范围：被动回合四段 lifecycle phase、prompt render、before/after step 的 runtime snapshot 解析，以及 consolidation history 内部契约。
- 原问题：同一 snapshot 的模块链在每个 phase 入口和每轮 tool step 重复组装、拓扑排序与校验；`get_history_since_consolidated()` 还捕获任意内部 `TypeError`，丢掉 cursor 后静默重试，可能把已 consolidation 的历史重新送入模型。
- 为什么这样修改：按内容身份稳定的 `snapshot_id` 缓存不可变模块链 bundle，snapshot 换代或本地模块追加时失效；history helper 直接依赖 `SessionLike` 已声明并由真实 `Session` 实现的 `start_index` 契约。
- 不变量与拥有层：RuntimeSnapshot compiler/store 拥有 snapshot 身份与代际唯一性，ContextVar lease 保证单 turn 绑定；phase owner 只缓存该身份对应的模块顺序。Session owner 拥有 consolidation cursor，helper 不再制造旧签名兼容层。
- 能力变化：中断/续跑、context trim retry、tool loop、hot reload lease、session persist 与 outbound 顺序不变；snapshot id 变化会重建所有缓存 phase；内部 session 签名错误现在 fail-fast。
- 性能变化：24 个插件模块、5000 次 before-step phase 获取的同 workload 微基准从 `673.699 ms` 降至 `0.742 ms`，约减少 `99.9%` 的 phase 解析开销；不外推为模型调用或完整 turn 延迟。
- 测试新增/调整：cursor 透传、同 snapshot 复用、snapshot 换代重建，并把旧测试替身改为真实 `SessionLike` 签名。
- 测试删除及原因：无。
- 验证结果：副手全量 `1835 passed`；主审定向 lifecycle/hot-reload/turn `148 passed`；修改文件 pyright `0 errors`；组合全量见后续边界批次。
- 残余风险：缓存以 snapshot 内容身份为 key；Store 已拒绝同一生命周期内重复 id，若未来允许原地改变 snapshot 模块而不改变身份，必须同时修改缓存失效契约。

### `5b2d39c2` + `125358ba` `fix(boundary): harden dashboard and telegram lifecycles`

- 范围：Dashboard proactive JSON/插件面板/资源关闭/待编译队列，以及 Telegram live preview、per-chat 限流状态和 UTF-16 消息边界。
- 原问题：Dashboard 损坏 JSON 被伪装成空结果、异步 close 返回值未等待、重复插件编译请求无界入队、面板名未覆盖 Windows 反斜杠；Telegram 最终回复前可能遗留尚未建消息的 live task，删除失败会丢失句柄，多个 per-session/per-chat 状态长期积累，emoji 按 Python 字符数截断会越过 Telegram 的 UTF-16 上限。
- 为什么这样修改：在 SQLite/HTTP/消息边界严格验证结构，资源 owner 等待同步或异步 close；待编译项按路径去重；live task、消息和状态在 turn/stop owner 明确收束；所有 Telegram 分段与 preview 统一使用 UTF-16 码元预算。
- 不变量与拥有层：ProactiveStateStore 写侧拥有 `list[str]`、`list[object]` 与 object JSON schema，Dashboard reader 只接受 TEXT/NULL 和对应 schema；Telegram channel 拥有 live task/message/session 状态，outbound limiter 拥有 chat deadline 与 lock。正常 Markdown、附件、回复、中断、plugin panel 与 hot reload 能力不变。
- 主审修正：拒绝副手把实际很小的首页改成 `FileResponse`；其 1 MiB TestClient 基准虽降低约 `11%` 峰值内存，却把 100 次 wall time 从 `0.387s` 增至 `0.944s`，不满足只优化不退化。主线保留原 `Response(read_text())`，并进一步删除 JSON `str()`/列表元素字符串化，拒绝底层存储类型损坏；UTF-16 极限预算无法容纳单字符时明确失败。
- 能力变化：损坏 dashboard 数据不再显示成正常空列表；异步 dashboard close 真正完成；同插件 pending 编译从 10000 个重复项收敛为 1 个；最终回复后不再继续出现旧 live preview；3000 个 emoji fallback 被正确切为 2 段且每段不超过 4090 UTF-16 码元。
- 性能变化：待编译队列由重复 list 改为 set，空间上界按唯一插件计；过期 chat deadline 超过阈值后清理，lock 无活跃引用时回收。撤回有 wall-time 退化证据的首页改动，不声明端到端提速。
- 测试新增：异步 close、损坏 JSON/存储类型、反斜杠面板名、live 删除失败句柄、终态状态清理、emoji fallback/preview 和不可表示的 UTF-16 极限。
- 测试删除及原因：无。
- 验证结果：副手定向 `58 + 114 passed`、全量 `1838 passed`；主审补刀后定向 `59 passed`；组合全量 `1843 passed in 28.62s`，全库 pyright `0 errors`（2326 个既有 warnings），前端 typecheck/lint/build 全通过，`git diff --check` 通过。
- 残余风险：未做真实 Telegram API 网络验证；外部限流/删除失败按现有显式日志和保留句柄语义等待后续事件重试。Dashboard overview 的多次 count 没有真实生产 workload 证据，本轮未引入缓存。

### `123749de` `fix(dashboard): stop plugin reload render loop`

- 范围：Dashboard 插件页面加载 effect 与插件状态读取。
- 原问题：`loadPluginPanel` 依赖整个 `pluginState`；每次分页结果写回都会改变 callback 身份，再次触发加载 effect。兴奋阈值面板的空分页立即完成，闭环会以主线程可见的速度持续渲染并请求，点击后页面卡死。
- 为什么这样修改：分页函数通过已有的稳定 `readPluginState()` 读取最新状态，callback 只依赖插件目录和稳定 reader；状态更新不再改变加载函数身份。
- 不变量与拥有层：React state 继续拥有插件分页、筛选、排序和选中状态；加载函数只读取调用时快照。插件切换、手动刷新、分页结果校验和 workbench 渲染语义不变。
- 能力变化：选择兴奋阈值或其他插件面板只触发一次入口加载；15 秒水位刷新仍由 `MeterPage` 自己持有并在卸载时清理。
- 性能变化：删除由状态写回造成的无界请求/渲染循环；正常点击从持续占用主线程和重复 `fetchPage` 收敛为一次分页加载。
- 测试新增/删除：无；仓库没有前端组件测试框架，不为单个回归引入新依赖。
- 验证结果：前端 typecheck、lint、dashboard/chat/plugin build 全通过；用户在真实 Dashboard 点击兴奋阈值确认不再卡死；build 仅有既有大 chunk warning。
- 残余风险：前端 effect 依赖仍主要靠静态检查和真实浏览器验收；后续若引入组件测试，应覆盖一次点击只调用一次 `fetchPage`。

### `f3859d48` `style: 统一历史代码中文注释`

- 范围：22 个生产 Python 文件中的英文 docstring、阶段标题、权限说明和 box-drawing 调用图节点。
- 原问题：历史说明混用英文自然语言，与仓库要求的中文 docstring、代码注释和阶段注释不一致。
- 为什么这样修改：只翻译自然语言说明，保留类名、协议、字段、命令、类型检查指令、公式和必要技术术语；没有为了形式新增注释。
- 能力与性能变化：无。主审对提交前后 AST 去除 docstring 后逐文件比较，22 个文件完全等价。
- 测试新增/删除：无。
- 验证结果：AST 等价检查、Python 编译、`git diff --check` 通过；组合全量测试见本批末尾。
- 残余风险：代码标识符和协议名继续保留英文，这是可执行契约，不属于自然语言注释残留。

### `a1242670` `fix(session): harden persistence boundaries`

- 范围：SessionStore JSON 边界、SessionManager 重载不变量、Dashboard session 列表 SQL 和 CoreRuntime 关闭 ownership。
- 原问题：孤立 message 可在缺 session metadata 时被当前时间和空 metadata 拼成假 session；media 引用扫描遇到损坏 extra 会静默跳过；session 列表为一页结果先全表聚合全部 messages；主 SessionManager 的 SQLite 连接没有明确关闭 owner。
- 为什么这样修改：持久化边界统一严格解析 message extra 并拒绝保留字段覆盖；metadata 缺失但存在消息立即失败；列表只对候选 session 通过索引相关子查询读取首条用户消息和计数；CoreRuntime 最后关闭唯一 SessionManager。
- 不变量与拥有层：SessionStore 拥有 SQLite/JSON schema，SessionManager 拥有内存 session 与 store 生命周期，CoreRuntime 拥有 manager 关闭。正常 session 创建、消息顺序、media 引用保护、Dashboard 排序/分页和 memory runtime 关闭顺序不变。
- 主审修正：拒绝副手把 proactive 窗口计数从 SQL 范围聚合改成拉取全部历史后逐条 Python 时间解析；坏时间应在写入/读取边界暴露，不能用每次热路径全表扫描换取补充校验。最终保留原索引友好计数。
- 能力变化：坏 message extra 和孤立消息不再被当成合法空数据；正常关闭会释放主 sessions.db 连接。
- 性能变化：500 个 session、每个 100 条消息的同一合成 workload 中，过滤列表查询中位数由旧聚合约 `16.949 ms` 降至 `0.132 ms`；不外推为所有 Dashboard 查询的固定倍数。
- 测试新增：孤立消息、坏 media extra、runtime 关闭 session manager；测试删除及原因：删除“每次窗口计数扫描全部记录以发现坏时间”的测试，它固化了热路径性能退化。
- 验证结果：主审定向 `104 passed`；修改范围 pyright `0 errors`；组合全量见本批末尾。
- 残余风险：相关子查询优势依赖现有 `(session_key, seq)` 索引；schema owner 已创建该索引，若未来改变消息主键必须同步审查查询计划。

### `6b590aee` + `cc8a731b` `refactor(memory2): expose boundary degradation`

- 范围：Memory2 query 改写、profile 提取、sufficiency 检查和 sqlite-vec 历史迁移。
- 原问题：主/可选 query lane 的模型失败与合法空结果不可区分；profile 主调用失败后会按 USER 子句继续调用，12 个子句会把一次失败放大为 13 次；模型响应通过动态属性兼容；向量迁移逐行跳过错误，可能留下部分索引。
- 为什么这样修改：每条 query lane 返回明确成功/降级状态与原因；profile 返回独立 `ProfileExtraction`，区分事实 tuple、状态和原因；模型边界只接受字符串或 `LLMResponse`；向量迁移在单一事务中整批成功或回滚。
- 不变量与拥有层：模型调用边界拥有响应类型和降级原因，parser 拥有 XML/schema，MemoryStore2 拥有主表与 vec 索引一致性。正常 query、profile 分类、sufficiency fail-open、sqlite-vec 可选索引和全表扫描 fallback 语义不变。
- 主审修正：副手首版用 `ProfileFacts(list)` 子类同时冒充旧列表和状态对象；这会把兼容层带入新内部 API。主线改为冻结 dataclass，调用方必须显式读取 `.facts` 与 `.status`。
- 能力变化：模型不可用不再伪装成合法空结果；profile 主失败停止子句重试；迁移失败不留下半套 vec 索引。
- 性能变化：12 个 USER 子句的模型失败 workload 由 13 次边界调用降为 1 次，减少 92.3%；正常成功提取不增加调用次数。
- 测试新增：query lane 降级原因、profile 失败不重试、vec 迁移回滚；测试删除：无。
- 验证结果：Memory2/Profile 定向 `77 passed`，修改范围 pyright `0 errors`；本批组合全量 `1850 passed in 26.31s`，全库 pyright `0 errors`（2313 个既有 warnings），`git diff --check` 通过。
- 残余风险：QueryRewriter、ProfileFactExtractor 和 SufficiencyChecker 当前主要由测试/eval 路径使用；状态已经在返回协议中明确，但未来接入主运行链时仍需决定降级是否应阻断该具体业务，不在底层假造统一策略。

### `71c53458` + `0142d9af` `refactor(memory): harden markdown consolidation contracts`

- 范围：Markdown consolidation 模型响应、去重标记恢复、维护队列、会话 scope 与 Memory Profile API。
- 原问题：模型输出缺少严格 schema，维护失败可从后台任务丢失；同轮重复读取 `RECENT_CONTEXT.md`；内部 API 通过动态属性和空值兼容掩盖契约错误；首版重构又错误地从可覆盖的 `session.key` 推导 channel/chat scope。
- 为什么这样修改：模型和 SQLite/文件恢复在各自信任边界校验；后台维护任务显式保留并观察异常；同轮复用一次 recent context；内部调用改为明确协议。主审将 `TurnCommitted` 的真实 channel/chat 作为 scope owner 贯穿队列，拒绝从字符串 session key 反推身份。
- 不变量与拥有层：Turn lifecycle 拥有真实 channel/chat；Session 只拥有可覆盖的存储 key；模型 parser 拥有 consolidation JSON schema；MemoryStore 拥有 sidecar/Markdown 恢复；maintenance runtime 拥有后台任务失败可见性。
- 能力变化：正常 consolidation、marker 去重、文件恢复和维护调度不变；`agent_context` 作为 prompt 已声明标签现在可被合法解析；坏模型结构、坏 SQLite flag、缺失恢复 payload 与持久化失败明确暴露；自定义 session key 不再污染记忆 scope。
- 性能变化：一次 consolidation 的 `RECENT_CONTEXT.md` 读取由 2 次降为 1 次；其余正常 I/O 与模型调用次数不变。
- 测试新增：严格模型 schema、持久化失败、marker 恢复 payload/flag、后台维护失败、真实 event scope 与自定义 session key。
- 测试删除及原因：无。
- 验证结果：主审修正后定向 `61 passed`；修改范围 pyright `0 errors`；组合全量 `1864 passed in 25.44s`，`git diff --check` 通过。
- 残余风险：Markdown 文件与 sidecar 仍依靠现有恢复协议收敛跨文件提交窗口；本批没有引入新的事务存储层。

### `6e72db6c` `fix(scheduler): commit lifecycle state atomically`

- 范围：JobStore 持久化、任务新增/取消、启动 misfire 恢复、循环任务执行和 scheduler 停止语义。
- 原问题：新增和取消会先修改内存再持久化，写盘失败造成内存/磁盘分叉；启动时推进或丢弃 misfire 后没有保存恢复状态；`stop()` 在 sleep 中触发仍可能多执行一次 tick。
- 为什么这样修改：先验证并原子持久化候选 job catalog，成功后才替换内存状态；启动恢复产生的状态变化立即提交；sleep 返回后再次检查停止信号。循环任务执行失败仍按既有产品语义推进下次 fire time。
- 不变量与拥有层：JobStore 拥有持久化 catalog 的原子替换；SchedulerService 拥有内存状态、misfire policy 和 tick 生命周期；任务 handler 错误继续由调度执行边界隔离并记录。
- 能力变化：正常单次/循环调度、misfire grace、取消和 handler 隔离不变；持久化失败不再留下假成功内存状态；重启不会重新消费已推进的 misfire；停止后不再额外执行任务。
- 性能变化：正常 add/cancel 的持久化次数不变；启动只在真实发生 misfire 状态迁移时增加一次必要提交，不声明提速。
- 测试新增：add/cancel 写盘失败回滚、启动恢复持久化、sleep 中 stop、循环任务失败后的 reschedule。
- 测试删除及原因：无。
- 验证结果：定向 `67 passed`；修改范围 pyright `0 errors`（8 个既有 APScheduler 类型 warning）；组合全量 `1864 passed in 25.44s`，`git diff --check` 通过。
- 残余风险：持久化 catalog 仍随任务数线性序列化；当前任务规模和 profile 没有证据支持引入增量日志或数据库。

### `f5c45e26` `fix(config): reject ambiguous boolean values`

- 范围：主 TOML 布尔配置边界、模型扩展参数、typed Config 到 provider 的内部调用。
- 原问题：`bool("false")` 会把字符串配置解释为真，影响 dev mode、渠道启用、memory、tool、multimodal 和 thinking；非 table thinking 被静默忽略；provider 在 typed `Config` 后继续用 `getattr` 默认值掩盖缺失字段。
- 为什么这样修改：在唯一 TOML 加载边界集中要求真实 bool；边界后 provider 直接读取 dataclass 字段；删除无调用者的递归配置解析 helper。拒绝合并副手 630 行的整套加载器改写，因为它把 `load_config` 膨胀到约 195 行并扩大了兼容语义变更。
- 不变量与拥有层：TOML loader 拥有外部 scalar/schema 校验；`Config` dataclass 拥有运行期字段完整性；provider 信任已构造配置，不重复 fallback。现有数字字符串兼容、环境变量解析、渠道默认值和 wiring 语义未改。
- 能力变化：合法布尔配置行为不变；字符串伪布尔和非 table thinking 现在携带字段名 fail-fast；真实本机 `config.toml` 继续成功加载。
- 性能变化：非性能提交；删除动态属性查询和无调用 dead code，不宣称可测收益。
- 测试新增：四类字符串布尔配置拒绝；provider 测试由 `SimpleNamespace + Any` 改为真实 `Config`。
- 测试删除及原因：无。
- 验证结果：定向 `80 passed`；真实配置加载通过；修改范围 pyright `0 errors`；组合全量 `1864 passed in 25.44s`，`git diff --check` 通过。
- 残余风险：历史 loader 仍存在若干 `str()`/`int()` 兼容转换；本批只修复能导致配置含义翻转的布尔边界，避免无真实迁移证据地全面收紧格式。

### `1b3f36bb` `fix(memory2): isolate post-response run context`

- 范围：回复后记忆废弃的转次上下文、memorize 结果保护和模型 JSON 响应边界。
- 原问题：worker 把 session/channel/chat 保存在实例可变字段，并发 `run()` 会把 A 转次的 `MemoryWritten` 标成 B 的 scope；provider 失败、非数组 JSON 和未知候选 ID 又被当成“无需废弃”静默略过。
- 为什么这样修改：每次执行构造冻结 `_RunContext` 并显式传递；模型响应只接受非空字符串数组，候选 ID 必须来自当前召回集。
- 不变量与拥有层：单转次 scope 由 `_RunContext` 拥有；post-response worker 只校验它真正消费的嵌套 calls/result 和模型输出，不重复校验 memorize 已执行成功的入参。
- 主审修正：删除副手新增但无下游用途的 summary 收集与参数校验；补上真实 ingest 内容可到达的 `calls`/call 结构边界，并删除只记录后原样重抛的宽泛 catch。
- 能力变化：并发转次不再串 scope；坏模型响应和 provider 错误明确失败，不会伪装成合法空结果。正常 supersede 阈值、召回和事件投递不变。
- 性能变化：正则改为模块级复用，删除无用 summary 处理；模型和存储调用次数不变，不声称端到端提速。
- 测试新增：两转次真实交错执行、provider 失败、非法 JSON schema、未知候选 ID 和嵌套 call 结构。测试删除及原因：无。
- 验证结果：定向 `50 passed`，修改生产文件 pyright `0 errors`（25 个既有动态 dict warning），组合全量 `1880 passed in 24.19s`，`git diff --check` 通过。
- 残余风险：`Retriever` 仍以宽泛 dict 暴露候选结构；本批信任该内部 owner，没有在 worker 里再写一层重复 schema。

### `62b9a4ff` `refactor(bootstrap): tighten runtime config contracts`

- 范围：核心工具注册、AgentLoop 依赖装配、插件阶段检查和 CoreRuntime 启停。
- 原问题：已经由 `Config` 构造边界保证的 `wiring`/`multimodal`/`vl_model` 仍在下游使用 `getattr(..., default)`；模块检查会在 `getattr` 默认表达式中无条件构造未使用的 outbound port。
- 为什么这样修改：边界后直接读取 typed config 和同仓库构造不变量；对真正可选的 plugin/spawn hook 仍保留动态边界处理。
- 不变量与拥有层：TOML loader/`Config` 拥有配置完整性；`AgentLoop`/pipeline 构造器拥有检查阶段需要的 context/session/outbound 字段；外部传入的 session store 仍由调用方关闭。
- 能力与性能变化：toolset 顺序、MCP fallback、VL 能力和插件热重载不变；删除了每次检查时的一个无用对象分配，不宣称可测端到端收益。
- 测试新增/调整：外部 session store 在注册失败后保持开启；历史 `SimpleNamespace` 配置替身改为真实 `Config`，runtime close 使用真实 `SessionManager`。测试删除及原因：无。
- 验证结果：定向 `90 passed`，修改生产文件 pyright `0 errors`（24 个既有动态插件/私有资源 warning），组合全量 `1880 passed in 24.19s`，`git diff --check` 通过。
- 残余风险：同步构造期自建 session store 后若后续注册失败，当前没有可等待异步 `MemoryRuntime.aclose()` 的完整 rollback 协议；未伪造后台清理或同步 mock 来掩盖该 ownership 缺口。

### `3df29e9f` `fix(proactive): preserve lifecycle cleanup failures`

- 范围：proactive lifecycle 动态模块编译边界、启动失败回滚和逆序停止。
- 原问题：动态字段与 hook 在每次运行时重复 `getattr`；rollback 只记录并丢弃错误，stop 只保留第一个 `Exception`；调用方取消或 stopper 自取消会截断后续资源清理。
- 为什么这样修改：builder 一次性校验并绑定 slot/依赖/run/start/stop；每个 cleanup 在独立 task 中完成，最后按原始顺序重抛单错误或聚合多错误。
- 不变量与拥有层：`ProactiveLifecycleBuilder` 拥有动态模块结构边界；`_CompiledModule` 拥有编译后不变契约；`CompiledProactiveLifecycle` 拥有启停顺序、清理完整性和错误传播。
- 主审修正：删除副手只为复述 `_CompiledModule` 形状而新增的 36 行 Protocol，直接使用唯一编译类；保留有真实 loop 取消路径的 shield 清理，没有为缩短代码犠牲资源完整性。
- 能力变化：启动失败会逆序完成全部 rollback；stop 会尝试所有模块；多个清理失败和取消不再相互覆盖。拓扑、wildcard collect、执行顺序和热重载 snapshot 语义不变。
- 性能变化：正常停止为每个 stopper 增加一个短生命 task，这是取消安全成本；运行热路径改为使用已绑定 runner，不增加模块执行次数。
- 测试新增：启动与 rollback 多错误顺序、stopper 自取消、外部取消后仍完成全部清理，以及坏动态模块在编译边界失败。测试删除及原因：无。
- 验证结果：定向 `42 passed`，修改文件 pyright `0 errors, 0 warnings`，组合全量 `1880 passed in 24.19s`，`git diff --check` 通过。
- 残余风险：多清理失败现以 `BaseExceptionGroup` 暴露；这是为了不丢错误的明确契约变化，上层当前不吞该异常，会把真实关闭失败继续传给 runtime owner。

### `0c992c96` `fix(bootstrap): fail fast on missing toolset deps`

- 范围：memory、common meta、spawn、MCP 和 scheduler toolset 的依赖装配与注册结果协议。
- 原问题：`ToolsetDeps` 用 `Any`/`object` 隐去真实依赖类型；memory 与 spawn 把多个缺失依赖合成同一错误，甚至允许缺 provider 继续进入构造；provider 又通过 registry 私有字段计算注册差集。
- 为什么这样修改：在各 toolset 注册边界逐项校验真正必需的依赖并指出字段名；`ToolsetDeps` 改用现有 provider/session 类型；注册差集统一通过 `ToolRegistry.get_registered_names()` 公共契约计算。
- 不变量与拥有层：bootstrap toolset provider 拥有装配完整性，具体 runtime 构造器只接收已确认存在的依赖，ToolRegistry 独占内部索引。可选 light/VL provider、事件发布器和 spawn 配置语义不变。
- 主审修正：副手首版继续读取 `registry._tools`，并用无类型的 `list`/`dict` default factory；主线改为公共查询 API、参数化容器 factory，并区分 `extras=None` 与显式空映射。
- 能力与性能变化：合法启动和工具注册顺序不变；缺失依赖现在在对应 toolset 边界 fail-fast，不再以更深层的属性错误暴露。非性能提交，不声明提速。
- 测试新增：memory/spawn provider 缺失、common meta 缺只读工具或 session store 的明确失败。测试删除及原因：无。
- 验证结果：定向 `35 passed`，`bootstrap/toolsets` pyright `0 errors, 0 warnings`；本批组合全量见末项。
- 残余风险：部分历史 toolset 仍以宽泛 extras 传递扩展对象；其用途跨 provider，不在没有具体错误路径时强行收窄。

### `e9b18447` `fix(memory2): tighten retrieval hit contracts`

- 范围：MemoryStore2 检索 lane 输出、vector/keyword 融合、embedding 取消传播和记忆注入格式化。
- 原问题：内部候选长期以宽泛 dict 传递，缺 id/score 会被空值或 `0.0` 掩盖；`gather(return_exceptions=True)` 会吞掉取消信号；RRF 完全平分时依赖 set 遍历顺序；procedure steps 通过 `cast` 假定形状。
- 为什么这样修改：由存储层定义唯一 `MemoryHit` 契约；检索器直接消费已建立字段，缺失 id/数值得分明确失败；取消信号继续向上；RRF 用 score 与 id 给出确定顺序；procedure steps 在实际消费边界要求字符串列表。
- 不变量与拥有层：MemoryStore2 拥有 SQLite/JSON 到候选结构的转换，Retriever 拥有 lane 融合、排序和注入元数据消费。单条普通 embedding provider 失败仍只跳过对应可选 lane，存储失败与任务取消不降级。
- 主审修正：副手测试用 `cast(Any, ...)` 和缺少必需字段的 dict 绕过新契约；主线改为真实 `MemoryHit` 构造，并补上 dashboard/历史数据可达的 procedure steps 损坏路径。
- 能力变化：取消可及时终止召回；坏候选不再悄悄降权或消失；相同输入的融合顺序稳定。正常 vector/keyword 召回、阈值和注入预算不变。
- 性能变化：RRF 仍为同阶排序与线性融合，只增加常数级确定性排序键和真实 procedure 元数据校验，不宣称端到端提速。
- 测试新增：embedding 取消、RRF 平分顺序、缺 vector score、坏 extra JSON/embedding/steps。测试删除及原因：无。
- 验证结果：定向 `27 passed`，retriever 与相关测试 pyright `0 errors, 0 warnings`；本批组合全量见末项。
- 残余风险：`extra_json` 是多记忆类型共用的扩展对象，暂不为所有可选字段制造统一大 schema；只在各字段的真实消费 owner 收窄。

### `194453fb` `refactor(passive): tighten runtime contracts`

- 范围：passive turn 工具保护判断、reasoner 插件模块装配、deferred 工具目录、memory retrieval owner 和 AgentLoop 入站确认清理。
- 原问题：内部 typed 对象仍通过 `getattr`/`callable`/类型过滤当作不可信数据；passive retrieval 可被一个空 `MemoryServices` 遮蔽真实 runtime engine；入站确认失败时 active task/turn state 留在内存中。
- 为什么这样修改：直接调用 Reasoner、ToolExecutionResult、SessionLike 与 ToolRegistry 的既有协议；为 deferred 目录定义精确 `TypedDict`；retrieval 始终使用 `_resolve_memory_runtime` 已确定的唯一 engine；确认消息前先收束本轮内存状态。
- 不变量与拥有层：Reasoner ABC 拥有插件扩展方法，ToolRegistry 拥有 deferred 目录形状，AgentLoop 的 runtime 解析拥有 memory engine 选择，MessageBus 只拥有最终入站确认。正常工具提示、turn 执行和确认顺序不变。
- 主审修正：副手首版先 `cast` 再逐项过滤 deferred 目录，等于重复怀疑 ToolRegistry；主线把 schema 放回 registry。内存 owner 测试也从读取私有字段改为执行真实 retrieval 并检查可观察结果。
- 能力变化：入站确认失败仍会明确抛错，但不再遗留幽灵 active turn；显式空 service 不会关闭已经解析成功的 memory runtime；内部契约缺失直接失败。
- 性能变化：删除 deferred 目录的重复扫描和动态属性分派；规模很小，不宣称可测端到端收益。
- 测试新增：入站确认失败清理、memory runtime 优先级和既有 deferred 可见性回归。测试删除及原因：无。
- 验证结果：相关定向 `76 passed`，新增精确契约文件 pyright `0 errors, 0 warnings`；三项组合全量 `1888 passed in 25.94s`，`git diff --check` 通过。
- 残余风险：passive/looping 两个历史大文件仍有大量宽泛消息 dict 类型告警；需按调用链分批收窄，不能用一次全文件类型改写冒险改变热重载、重试和中断语义。

### `081e427a` `fix(loop): preserve turn shutdown lifecycle`

- 范围：AgentLoop 入站 turn 的任务所有权、运行时取消、主动停止和消息确认。
- 原问题：主循环取消时可能把取消当作普通子任务结束；`stop()` 不会终止仍在执行的 turn；确认消息失败与任务状态清理之间缺少统一生命周期顺序。
- 为什么这样修改：由 AgentLoop 显式持有当前 turn task；把单轮执行抽为明确协程；运行时取消继续向上传播；停止时取消并等待当前 turn；确认操作使用 shield 完成边界收尾。
- 不变量与拥有层：AgentLoop 拥有当前 turn 的创建、取消与清理；MessageBus 拥有入站确认结果；取消不是可恢复业务错误，不转换为空结果或普通日志。
- 能力变化：正常串行 turn、热重载和入站确认语义不变；服务停止或 runtime 取消不再遗留后台 turn；确认失败仍 fail-loud。
- 性能变化：正常 turn 不增加模型或工具调用；仅停止路径增加一次必要的任务等待，不声明提速。
- 测试新增：runtime 取消传播、停止取消活动 turn、确认失败后的生命周期清理。测试删除及原因：无。
- 验证结果：副手 worktree 全量 `1891 passed`；合并主审后的组合全量见本轮末项，`git diff --check` 通过。
- 残余风险：AgentLoop 历史消息 payload 仍有宽泛 dict 类型；本批只修复可观察的取消与资源所有权，不扩大为全文件类型迁移。

### `914f799b` `refactor(runtime): enforce tool discovery contracts`

- 范围：runtime service 协议、tool discovery JSON 边界、deferred 工具目录和默认依赖工厂。
- 原问题：tool search 对坏 JSON、非 object 和坏工具名以 warning 加空结果降级；部分内部 service 仍以宽泛对象传递；副手首版又加入了没有真实违反路径的容量校验。
- 为什么这样修改：在 tool_search 的外部 JSON 边界严格解析；成功响应结构损坏直接抛错；边界后使用精确 SessionLike、TurnRunResult 和 AgentLoopRunner 协议；删除无法由生产构造路径触发的重复容量检查。
- 不变量与拥有层：tool_search parser 拥有模型/工具 JSON schema；ToolRegistry 拥有 deferred 目录；typed runtime service 拥有内部字段完整性，调用方不再二次怀疑。
- 主审修正：拒绝副手以“更稳健”为由增加的默认容量检查；把 warning + 空数组改为携带上下文的 fail-loud，并保留当前 `unlocked` 与已存在 legacy `matched` 两种真实协议。
- 能力变化：合法工具发现和确定性去重不变；损坏的成功响应不再伪装成“没有匹配工具”。
- 性能变化：非性能提交；删除重复校验和动态对象路径，不声明端到端收益。
- 测试新增/调整：坏 JSON、非对象、坏数组、坏工具名、当前与 legacy 字段解析。测试删除及原因：删除仅覆盖不可达容量状态的四项测试。
- 验证结果：定向 `40 passed`，修改范围 pyright `0 errors, 0 warnings`；组合全量见本轮末项。
- 残余风险：legacy `matched` 仍为已存在的真实兼容输入；待上游协议正式移除后才能安全删除，当前不假设其已不可达。

### `355d90f9` `fix(memory2): enforce procedure metadata contracts`

- 范围：procedure 元数据解析、记忆合并事务、检索候选分数和 post-response 候选类型。
- 原问题：坏 steps/tool_requirement/rule_schema 会被空数组、字符串化或过滤掩盖；Memorizer 越过 Store 私有锁和连接读取元数据；content-hash 冲突时旧实现会退休原记录并新建缺字段记录；多个模块重复实现 score fallback。
- 为什么这样修改：在持久化元数据的消费边界集中严格解析；Store 提供合并元数据公共 API 并独占事务和 content hash；冲突回滚且明确失败；MemoryHit 分数由 Store 契约统一读取。
- 不变量与拥有层：MemoryStore2 拥有 SQLite、JSON、content hash 和向量索引事务；Memorizer 拥有 procedure 合并语义；rule schema parser 只在外部或持久化边界校验，内部 TypedDict 不重复防御。
- 主审修正：副手首版增加约 293 行并在内部 schema 上重复校验，还把两项全库 pyright error 误报为既有问题；主线删除不可达检查，修复类型错误，集中 score owner，并保留真实可达的损坏持久化与 hash 冲突路径。
- 能力变化：正常保存、召回、procedure merge 和 tag 重建语义不变；损坏元数据立即暴露；合并 hash 冲突不再静默丢失 source、extra、时间等字段。
- 性能变化：合并元数据改为一次 Store 查询；候选分数不再在三处重复解析。数据库和模型调用数量没有新增，不声明端到端提速。
- 测试新增：四类坏 procedure 元数据、损坏持久化 steps、content-hash 冲突原子回滚。测试删除及原因：无。
- 验证结果：Memory 定向 `81 passed`；修改范围 pyright `0 errors`（15 个既有动态 provider/tool-chain warning）；三项组合全量 `1899 passed in 25.37s`，全库 pyright `0 errors`，`git diff --check` 通过。
- 残余风险：post-response tool chain 仍来自动态模型/工具 payload，存在既有宽泛 dict warning；应在其唯一输入边界单独建 schema，不能在每个消费函数重复检查。

### `7d6aded2` `fix(http): preserve shared client cleanup errors`

- 范围：共享 HTTP 客户端关闭生命周期和直接回归测试。
- 原问题：多个客户端同时关闭失败时只保存第一个异常，后续真实清理错误被丢弃。
- 为什么这样修改：仍按原有逆序串行尝试全部关闭；单错误保持原异常，多错误使用 `ExceptionGroup` 一次暴露；取消不被普通异常捕获。
- 不变量与拥有层：SharedHttpResources 拥有三个客户端的关闭顺序和生命周期；AppRuntime 只负责调用 owner 的 `aclose()`，不在上层重建清理语义。
- 主审修正：删除副手为 typed `HttpProfile` 增加的未知 profile 分支和 `type: ignore` 测试；该违反路径只能绕过类型契约构造，不应在内部函数重复防御。顺手清零修改文件的两个既有 pyright warning。
- 能力变化：正常请求、连接复用、shutdown 顺序和单错误类型不变；多清理失败不再丢失。
- 性能变化：仅失败关闭路径收集最多三个异常；正常热路径无变化。
- 测试新增：逆序关闭、继续清理和多错误聚合。测试删除及原因：删除不可达 profile 防御测试。
- 验证结果：HTTP 定向 `12 passed`，修改文件 pyright `0 errors, 0 warnings`；组合全量见本轮末项。
- 残余风险：多清理失败以 `ExceptionGroup` 暴露，上层当前会继续 fail-loud；没有为未知调用方增加静默兼容层。

### `6ffda115` `fix(plugins): tighten watcher recovery lifecycle`

- 范围：插件 watcher 的 revision 扫描、热重载失败、外部唤醒、取消和停止等待。
- 原问题：宽泛扫描 catch 会把内部错误当文件竞争；一次已恢复且 revision 未变化的扫描错误也会强制 reload；未启动 task 被取消时 `wait_stopped()` 可永久等待。
- 为什么这样修改：扫描边界只恢复 `OSError`；外部 force 在扫描失败后保留；未启动 stop 明确完成等待信号；同一坏 revision 只尝试一次，文件再次变化或外部 force 后仍可恢复重载。
- 不变量与拥有层：watch_revision 拥有同步文件快照；reconcile_changed 拥有原子候选发布；watcher 拥有重试节奏和 task 生命周期，业务失败明确记录但不能永久杀死热重载能力。
- 主审修正：拒绝副手“reconcile 异常直接结束 watcher”的实现；那会让一次坏插件阻断后续热修复。主线保留 fail-loud 日志，并推进到失败 revision，避免无变化时反复执行有副作用的 reload。
- 能力变化：源码变化自动重载、扫描期间二次变化、取消传播和显式 wake 均保留；坏版本不再无限重试，修复文件后自动恢复。
- 性能变化：稳定轮询不增加扫描或重载次数；失败版本从每轮重复 reconcile 降为一次。
- 测试新增：恢复后的稳定 revision 不误重载、坏版本单次失败后恢复、取消完成、启动前停止和未启动 task 取消。测试删除及原因：无。
- 验证结果：watcher 定向 `8 passed`，修改文件 pyright `0 errors, 0 warnings`；组合全量见本轮末项。
- 残余风险：reconcile 失败目前通过明确异常日志对外可见，没有单独的 dashboard 状态字段；本批不扩展观测协议。

### `75b33293` `fix(session): reject malformed history payloads`

- 范围：SessionStore 消息反序列化、Session history 渲染、主动 sensor 和媒体读取错误边界。
- 原问题：坏 media/source_refs/tool-chain 容器会被 `or []` 或过滤伪装成空数据；NULL content 被改写为空字符串；媒体读取的宽泛 catch 会掩盖程序错误。
- 为什么这样修改：SQLite/JSON 读取边界集中校验消息列、扩展字段和工具链容器；Session 下游直接信任已解析结构；只把真实文件系统 `OSError` 转为附件读取失败标记。
- 不变量与拥有层：SessionStore 拥有持久化 JSON 和列类型；Session 拥有 LLM history 展开；Sensor 只复制已建立的 source_refs 类型，不再把任意 object 当 iterable。
- 主审修正：使用真实运行库 `/home/huashen/.akashic/workspace/sessions.db` 只读核验 11,621 条消息、2,096 条 tool-chain 消息和 7,579 次调用；role、media、source_refs、arguments/result 均符合新边界。主线同时保留消息查询允许的稀疏 tool-chain，撤销会破坏现有 FetchMessages 公共用例的过严必填校验，并修复全库 pyright 暴露的 Sensor 类型遗漏。
- 能力变化：正常 session、历史窗口、主动消息 metadata、附件、consolidation 与消息查询不变；损坏持久化数据携带 message id 失败，不再形成看似正常的空历史。
- 性能变化：每条读取消息增加与 payload 大小线性的内存结构校验；没有新增 SQL、全表扫描或模型调用。
- 测试新增：坏消息列、media/source_refs、tool group/calls 和非 OSError 媒体失败；Sensor 类型回归由既有测试覆盖。测试删除及原因：无。
- 验证结果：Session/消息查询定向 `64 passed`，Sensor `4 passed`，修改后的 manager/sensor pyright `0 errors, 0 warnings`；本轮组合全量 `1914 passed in 25.76s`，全库 pyright `0 errors`，`git diff --check` 通过。
- 残余风险：SessionStore 仍为 message lookup 和模型 history 共用同一稀疏 tool-chain 载荷；后续若要彻底消除可选字段，应先拆分两种公开读取协议，不能直接收紧共享存储 schema。

### PR 检查点：测试契约与完整门禁

- 范围：runtime、peer agent、memory2 和 I/O 测试夹具，以及 peer process 私有超时参数的类型契约。
- 原问题：生产代码完成精确类型收窄后，测试仍把 `SimpleNamespace`、残缺 `MemoryHit` 和任意对象直接传入具体契约；生产 Pyright 已通过，但 CI 的测试配置仍有 85 个类型错误。
- 为什么这样修改：正常数据补齐真实必填字段和精确 TypedDict；行为型 fake 只在测试注入边界集中转换；能使用真实轻量依赖时直接使用 `SessionStore`，不以 `Any` 掩盖类型缺口。
- 不变量与拥有层：生产构造函数继续拥有具体依赖契约；测试夹具拥有 fake 与生产协议之间的显式适配；持久化命中样本必须符合 `MemoryHit`，不再依赖运行时碰巧未读取的缺失字段。
- 主审修正：删除副手为毫秒级超时加入的 `cast(Any, 0.001)`，让 `_kill` 如实接受 `float` 秒；删除无意义的显式 `return None`；注册测试改用真实临时 `SessionStore`。
- 能力变化：没有改变正常运行、错误处理或测试断言；仅让测试数据和 fake 明确满足已经建立的生产契约。
- 性能变化：生产热路径无变化；测试使用的毫秒级超时保持不变。
- 测试删除及原因：无。
- 验证结果：`.venv/bin/pytest -q -W error tests/` 为 `1914 passed in 26.30s`；生产与测试 Pyright 均为 `0 errors, 0 warnings`；`git diff --check` 通过。前端在本检查点前已完成 `npm run typecheck`、`npm run lint` 和 `npm run build`，本批未修改前端文件。
- 残余风险：peer agent 测试仍有集中式 fake 转换，这是测试替身与具体实现类型之间的明确边界；后续若生产改为 Protocol，可自然删除这些转换，本批不为测试便利扩大生产抽象。

### PR 检查点：初始化矩阵、资源所有权与真实插件验收

- 范围：fresh init、memory engine 与 proactive 组合、IPC 端点、主任务取消、内置 memory 配置所有权、插件 doctor、CLI help/inspect，以及 Fitbit、飞书外部插件的运行缓存。
- 原问题：主任务被取消时会再次取消子任务并打断其 `finally`；默认 `/tmp/akashic.sock` 会让多 workspace 争用；内置 memory 配置写在源码目录；`--help` 会误启动服务；`--inspect-modules` 漏关 memory runtime；干净环境缺 `networkx`；Akasha 首次导入会暴露 `jieba` 的第三方弃用告警；doctor 会把未选中的 default-memory drift skill 误判为缺失。
- 为什么这样修改：由 workspace 稳定派生 IPC；由 AppRuntime、inspect 命令和各插件分别关闭自己创建的资源；内置插件配置迁移到 `~/.akashic-plugin/data/*-builtin`；只隔离第三方库的精确已知告警，导入错误仍直接失败；doctor 根据真实 memory 配置判断链接是否应存在。
- 不变量与拥有层：Config 负责结构化配置，workspace 负责实例级 IPC，memory 插件数据目录负责可写配置，runtime owner 负责关闭连接和任务；测试 fake 必须补齐 schema 已保证的 `channels.socket`，生产路径不增加 `getattr` 或默认值兜底。
- 能力变化：不同 workspace 可并行启动；SIGTERM 会等待插件任务完成收尾；源码目录只读安装可正常创建 memory 配置；CLI help 不再产生服务副作用；inspect 不再泄漏 SQLite；合法 memory/proactive 组合行为不变。
- 性能变化：正常请求、召回和主动判断热路径没有新增 I/O 或模型调用；仅取消和关闭路径增加必要等待，不声明端到端提速。
- 外部插件：Fitbit `57ae832` 删除无引用且无法解析的 `monitor/fitbit-swagger.json`，canonical 仓库已推送并重装；飞书 `d66c0e0`、`a2701ea` 修正 SDK loop 停止、主动关闭噪声和未取 task 异常，已推送并重装。旧 Fitbit workspace 文件删除前备份到 `~/.akashic/workspace/backups/fitbit-corrupt-swagger-20260713/`。
- 测试新增：3 种 memory 状态乘 3 种 proactive 状态的 9 组 core 矩阵，以及 default/Akasha 乘 3 种 proactive 状态的 6 组完整 start/stop 矩阵；另覆盖 workspace IPC、help 无副作用、取消收尾、配置迁移、doctor active policy 和 inspect 清理。测试删除及原因：无。
- 真实验收：用户配置下 24 个插件加载，Calendar、Feed、Fitbit、Steam 四个 MCP 完成连接，wake source 实际读到 `alerts/content/context`，全部 channel 启动；SIGTERM 后 MCP、Fitbit monitor、飞书线程和 IPC 全部退出。隔离 HOME 下真实执行 `main.py init`，仅两个内置 memory 插件启动并干净退出。
- 验证结果：主仓库 `.venv/bin/pytest tests/ -q -W error` 为 `1936 passed in 28.99s`；`pip check` 无损坏依赖；修改范围 pyright 为 `0 errors`，其中 doctor/main 为 `0 errors, 0 warnings`。全量修改文件仍显示 308 条历史动态配置与 runtime unknown-type warning，本批不以 `Any` 或 ignore 掩盖。
- 残余风险：daynight_gate 是用户显式 disabled，doctor 的 degraded 属预期策略状态；动态配置解析和 AppRuntime 若要清零历史 pyright warning，需要单独收窄边界，不能在本次启动修复中扩大为全文件类型重写。

### PR 检查点：默认记忆、Akasha、Wake 与外置 MCP 行为契约

- 范围：默认记忆显式写入作用域、上下文探针失败判定、default/Wake 主动沙盒、Akasha 真实多轮召回，以及 Feed、Calendar、Fitbit 的主仓库 MCP client 协议。
- 原问题：显式 `memorize` 没有持久化 channel/chat scope，导致同作用域的 answer/interest 查询看不到刚写入的记忆；上下文探针会把通用错误回复记成成功；主动沙盒硬编码 `default` lifecycle，不能证明 Wake 真正工作；Feed 首次后台刷新会阻塞已有缓存读取。
- 为什么这样修改：作用域由 memory engine 写入 owner 一次持久化；探针识别主流程的明确失败回复并 fail-loud；沙盒显式选择 lifecycle，并以真实 Feed MCP、WebFetch、消息编排和 ACK 验收；Feed 刷新继续后台执行，缓存读取不等待整轮网络轮询。
- 不变量与拥有层：MemoryScope 由 memory engine 转换为持久化字段；Context Probe 只判断运行失败，不替代语义评分；Wake reservoir 拥有消费状态，Feed 只把明确反馈写入 `interest_ok`；MCP client 继续拥有 structured content 协议校验。
- 能力变化：显式偏好可被同会话 answer 查询立即召回；运行时错误不再形成假绿报告；Wake 的 `scratchpad → 正文抓取 → share_content → message_push → ACK` 可重复验收；Feed 刷新期间仍可读取旧缓存。
- 性能变化：Feed MCP 启动不再把缓存读取阻塞到所有订阅轮询完成；默认记忆仅增加两个已有 scope 字段的持久化，无额外模型调用。
- 外部插件：Feed canonical 仓库提交 `5c86997`，已推送、重装，并确认运行缓存 commit 与 canonical 一致。
- 真实验收：默认主动链完成 content、Drift 与兴趣反馈；Wake 完成正文抓取、主动发送、session 持久化和消费 ACK，且 `interest_ok` 保持 `NULL`；Akasha 全新 workspace 两轮会话从“喝茶不加糖”准确召回“不加糖”，生成 2 个节点、2 条边、3 条 query log 和 1 条 activation event；Feed、Calendar、Fitbit 均通过主仓库 MCP client 调用真实插件进程，Fitbit sleep context 返回 12 字段对象。
- 测试新增：Context Probe 失败回复、默认记忆 scope、fresh init memory/proactive 矩阵和稳定场景文件。测试删除及原因：无。
- 验证结果：主仓库 `.venv/bin/pytest -q -W error tests/` 为 `1938 passed in 29.21s`；default 与 Wake Docker 主动行为沙盒均通过；修改后的 default memory engine Pyright 为 `0 errors`。
