# Issue 1179 · 边界②③审查入口

本次承接原会话 `01a125c7-3c5e-78b1-88ea-7321bf19e8e5` 的三个停点。
边界①已在 main；这里交付边界②非账本 Core 收口与边界③ Ledger/整体验收。
全部 PR 为 Draft，尚未合并或部署，正式 workspace 未写入。

基线 `ca3d0d20baa420dd6b847dedfff53308b1d17bb6`。唯一 writer 的隔离 worktree 为
`/mnt/data/coding/akashic-boundary23-1179`；原 checkout 与其未提交内容未修改。
恢复分支 `backup/1179-boundary23-start-20261011`；各层 commit 是源码恢复点。

## 先看重大决定

用户授权自主决定，并要求重大选择独立 PR。以下均有显式标题与单独 ADR，待审查：

| 决定 | PR | 需要重点判断的行为 |
|---|---|---|
| [0103](../decisions/0103-processes-follow-provider-lifetime.md) | [#1245](https://github.com/kachofugetsu09/akashic-agent/pull/1245) | HostExecution 换代/卸载关闭所属后台进程；取消一次等待仍保留本代进程。代码回退不能复活进程 |
| [0104](../decisions/0104-public-codecs-without-ui-runtime.md) | [#1253](https://github.com/kachofugetsu09/akashic-agent/pull/1253) | 公共合同可有纯编码；UI 停用时 CLI/Gateway 仍读基本事实，业务展示明确缺席 |
| [0105](../decisions/0105-ledger-owns-business-storage.md) | [#1257](https://github.com/kachofugetsu09/akashic-agent/pull/1257) | Ledger 拥有唯一业务库与关闭责任；历史迁移只搬 owner，保留 ID/SQL/数据 |
| [0106](../decisions/0106-derived-vectors-use-separate-storage.md) | [#1259](https://github.com/kachofugetsu09/akashic-agent/pull/1259) | 向量复制到派生库，旧表保留；回退不会把新向量自动同步到旧表 |
| [0107](../decisions/0107-restore-missing-vectors-before-reading-memory.md) | [#1260](https://github.com/kachofugetsu09/akashic-agent/pull/1260) | 派生库缺失先按原空间补算；模型不可用/空间不符明确失败，原图不重写 |
| [0108](../decisions/0108-bundles-are-composition-inputs.md) | [#1261](https://github.com/kachofugetsu09/akashic-agent/pull/1261) | base/mode/workspace patch 整行覆盖，无自动依赖选择 |
| [0109](../decisions/0109-workspace-owns-plugin-choices.md) | [#1264](https://github.com/kachofugetsu09/akashic-agent/pull/1264)、[#1265](https://github.com/kachofugetsu09/akashic-agent/pull/1265) | workspace 拥有启停；既有配置仍由配置事务独占；Yoyo 保存原文与计划后退休旧全局开关 |
| [0110](../decisions/0110-ledger-checks-use-existing-message-semantics.md) | [#1267](https://github.com/kachofugetsu09/akashic-agent/pull/1267) | 不变量插件拒绝重复 abandon/旧序号；未知发送失败并注明可能已送达，不自动重发 |
| [0111](../decisions/0111-remove-unconsumed-business-ports.md) | [#1278](https://github.com/kachofugetsu09/akashic-agent/pull/1278) | 删除无维护消费者端口及旧通知；Drift 原数据与结算保持 |
| [0112](../decisions/0112-akasha-reads-validated-consumption.md) | [#1279](https://github.com/kachofugetsu09/akashic-agent/pull/1279) | 损坏的消费进度明确报错，不再被面板解释为空记录 |

## 按 base 关系审查主栈

每层的 base 是前一层。#1257 是唯一超过约 500 新增行的层：前置已拆开公开合同，
这层原子移动 2,448 行存储/迁移实现并删除 2,460 行旧位置，避免中间状态有双 owner
或重新引入兼容壳。其它层按语义单独交付；不能把净删除当成新增行额度。

| PR | 前置 | 固定 head | 内容 |
|---|---|---|---|
| [#1245](https://github.com/kachofugetsu09/akashic-agent/pull/1245) | main | `80c409a2` | [重大决定 ADR-0103] HostExecution 换代和卸载时关闭所属进程集合 |
| [#1246](https://github.com/kachofugetsu09/akashic-agent/pull/1246) | #1245 | `93657488` | [codex] Tools 拥有完整公共合同，删除 Core 工具合同壳 |
| [#1247](https://github.com/kachofugetsu09/akashic-agent/pull/1247) | #1246 | `bf8f0220` | [codex] 文件操作与 Shell 环境通过 HostExecution 服务取得 |
| [#1248](https://github.com/kachofugetsu09/akashic-agent/pull/1248) | #1247 | `9ef58eae` | [codex][决策] Bridge 与物理执行后端归 HostExecution，Core 停止认领 Bridge |
| [#1249](https://github.com/kachofugetsu09/akashic-agent/pull/1249) | #1248 | `95ffd18b` | [codex] 配置 HTTP 路由归 UI，删除 Core 初始化转发壳 |
| [#1250](https://github.com/kachofugetsu09/akashic-agent/pull/1250) | #1249 | `669f1d9e` | [codex] Ledger 前置：可信调用身份和依赖来源由内核提供 |
| [#1251](https://github.com/kachofugetsu09/akashic-agent/pull/1251) | #1250 | `03a94067` | [codex] 通用凭据合同归 Core 凭据服务，解除 Channel 反向依赖 |
| [#1252](https://github.com/kachofugetsu09/akashic-agent/pull/1252) | #1251 | `48eaf38b` | [codex] 文件操作返回物理结果，删除 Core 工具结果壳 |
| [#1253](https://github.com/kachofugetsu09/akashic-agent/pull/1253) | #1252 | `d3a29ccd` | [重大决定 ADR-0104] UI 拥有公共消息编码，停用后 CLI 仍读取基本事实 |
| [#1254](https://github.com/kachofugetsu09/akashic-agent/pull/1254) | #1253 | `e317ea20` | [codex] Ledger 前置：消息服务只公开协议，存储实现独立 |
| [#1255](https://github.com/kachofugetsu09/akashic-agent/pull/1255) | #1254 | `67a7a473` | refactor(ledger): 分离消息值与存储协议 |
| [#1256](https://github.com/kachofugetsu09/akashic-agent/pull/1256) | #1255 | `443b3eb2` | refactor(ledger): 分离附件与 binding 存储端口 |
| [#1257](https://github.com/kachofugetsu09/akashic-agent/pull/1257) | #1256 | `7527c684` | refactor(ledger): [重大决定] 将业务存储及入站交接移交普通插件 |
| [#1258](https://github.com/kachofugetsu09/akashic-agent/pull/1258) | #1257 | `f4c78443` | refactor(ledger): 删除旧入站队列状态机和空绑定壳 |
| [#1259](https://github.com/kachofugetsu09/akashic-agent/pull/1259) | #1258 | `f485bfea` | refactor(ledger): [重大决定] 派生向量独立存储并保留旧表 |
| [#1260](https://github.com/kachofugetsu09/akashic-agent/pull/1260) | #1259 | `d0fc2a9e` | fix(akasha): [重大决定] 丢失派生库后按原空间补算再读图 |
| [#1261](https://github.com/kachofugetsu09/akashic-agent/pull/1261) | #1260 | `ba25e2cb` | feat(plugins): [重大决定] 声明三个 bundle 及整行替换语义 |
| [#1262](https://github.com/kachofugetsu09/akashic-agent/pull/1262) | #1261 | `65cf8c23` | refactor: 发行制品使用 TOML bundle，删除默认 JSON profile |
| [#1263](https://github.com/kachofugetsu09/akashic-agent/pull/1263) | #1262 | `69d322a8` | refactor: 实际提交并启动 base、headless、minimal 组合 |
| [#1264](https://github.com/kachofugetsu09/akashic-agent/pull/1264) | #1263 | `3cf21e1e` | [重大决定] workspace patch 独占启停，制品与配置各保留一个 owner |
| [#1265](https://github.com/kachofugetsu09/akashic-agent/pull/1265) | #1264 | `853f9d12` | [1179 · 重大决定 ADR-0109] 原子切换 workspace 插件选择并退役全局启停 |
| [#1266](https://github.com/kachofugetsu09/akashic-agent/pull/1266) | #1265 | `9ec6c530` | [1179] 用旧 Core 和真实写入故障验收 bundle 选择迁移 |
| [#1267](https://github.com/kachofugetsu09/akashic-agent/pull/1267) | #1266 | `c193b8bd` | [1179 · 重大决定 ADR-0110] Ledger 不变量插件沿用现有 Turn 和未知效果合同 |
| [#1268](https://github.com/kachofugetsu09/akashic-agent/pull/1268) | #1267 | `c0f44a83` | [1179] 验收 Ledger generation 换代与事务提交并发 |
| [#1269](https://github.com/kachofugetsu09/akashic-agent/pull/1269) | #1268 | `209a6e21` | [1179] 图片表示与模型错误诊断归还插件 |
| [#1270](https://github.com/kachofugetsu09/akashic-agent/pull/1270) | #1269 | `c2b39eaa` | [1179] 退役主动流程的离线交接工具退出 Core |
| [#1271](https://github.com/kachofugetsu09/akashic-agent/pull/1271) | #1270 | `6a767708` | [1179] R7–R11 固定 Core 服务、词汇、数据库与 Web 边界 |
| [#1272](https://github.com/kachofugetsu09/akashic-agent/pull/1272) | #1271 | `7fe6a24d` | [1179] EventMail 公开外部来源合同，保持异步 v2 语义 |
| [#1273](https://github.com/kachofugetsu09/akashic-agent/pull/1273) | #1272 | `f430a56c` | [1179] Programmatic 公开调用参数，删除消费方复制模型的前提 |
| [#1274](https://github.com/kachofugetsu09/akashic-agent/pull/1274) | #1273 | `57a47305` | [1179] Akasha 公开只读召回出处，删除无人使用的单条服务 |
| [#1275](https://github.com/kachofugetsu09/akashic-agent/pull/1275) | #1274 | `04b1e368` | [1179] Models 公开现有调用历史读取合同 |
| [#1276](https://github.com/kachofugetsu09/akashic-agent/pull/1276) | #1275 | `c79d9301` | [1179] Markdown Memory 公开现有写入历史合同 |
| [#1277](https://github.com/kachofugetsu09/akashic-agent/pull/1277) | #1276 | `8b8e0945` | refactor(content): 公开文本引用和区间合同 |
| [#1278](https://github.com/kachofugetsu09/akashic-agent/pull/1278) | #1277 | `a8ca7134` | [重大决定][P6] 删除无维护消费者的业务端口 |
| [#1279](https://github.com/kachofugetsu09/akashic-agent/pull/1279) | #1278 | `1a5327e4` | [重大决定] Akasha 面板对损坏的消费进度明确失败 |
| [#1280](https://github.com/kachofugetsu09/akashic-agent/pull/1280) | #1279 | `af8ea4c4` | test: 逐个拔除 49 个内置插件并验证冷启动 |
| [#1281](https://github.com/kachofugetsu09/akashic-agent/pull/1281) | #1280 | `7b448cd8` | test: 用独立 BLOB provider 验证 Ledger 附件端口 |
| [#1282](https://github.com/kachofugetsu09/akashic-agent/pull/1282) | #1281 | `388e58c7` | 验证独立 UI 查询 provider 的安装与替换 |
| [#1283](https://github.com/kachofugetsu09/akashic-agent/pull/1283) | #1282 | `fa2af10f` | 复核发布选择切换后的 Timer 替换场景 |
| [#1284](https://github.com/kachofugetsu09/akashic-agent/pull/1284) | #1283 | `504c9e57` | 验证独立只读 Gateway 与同一 SDK 的查询合同 |
| [#1285](https://github.com/kachofugetsu09/akashic-agent/pull/1285) | #1284 | `92343ca1` | 验证替换模型驱动时 Models 与 Reply 不重新 apply |
| [#1286](https://github.com/kachofugetsu09/akashic-agent/pull/1286) | #1285 | `df1c7fd4` | 用真实压缩结果验证独立摘要归档 provider |
| [#1287](https://github.com/kachofugetsu09/akashic-agent/pull/1287) | #1286 | `f423ff05` | 验证旧 Core 的附件身份与待处理入站记录原样接管 |
| [#1288](https://github.com/kachofugetsu09/akashic-agent/pull/1288) | #1287 | `e5ca4757` | 删除 MCP 宿主无人消费的字符串端点映射 |
| [#1289](https://github.com/kachofugetsu09/akashic-agent/pull/1289) | #1288 | `8e9c4e90` | 收口目录读取类型和 Telegram 凭据表示 |
| [#1290](https://github.com/kachofugetsu09/akashic-agent/pull/1290) | #1289 | `6662fdbd` | 证明非幂等发送崩溃后不会假成功或自动重发 |
| [#1291](https://github.com/kachofugetsu09/akashic-agent/pull/1291) | #1290 | `d9b30c12` | 对账动态和外部能力消费者，补全静态调用目录 |
| [#1292](https://github.com/kachofugetsu09/akashic-agent/pull/1292) | #1291 | `29cd6115` | 补齐 Timer 借用消费者零重新 apply 的替换证据 |

## 隔离验收

所有运行场景使用临时 workspace、plugin home 与配置。以下是实际运行结果，
不是单元 mock 或只检查 import 的推断；完整端口限制见 [第二 provider](plugin-second-providers.md)。
#1245～#1292 的 48 个固定 head 均已通过 `check-and-test` 与 `plugin-boundary` CI。

| 范围 | 入口与结果 |
|---|---|
| Core E1～E4 | `plugin_boundary.py check --base ca3d0d20`：R1/R2/R3/R7/R8/R9/R11 均零，R10 精确九个 key；真实 CLI 负样本使各门失败 |
| 类型/迁移 | Core、全部 `plugins/`、既有 tests 类型检查零错误；Yoyo 迁移门通过 |
| E5 三组合 | `d9b30c12` 制品三模式全部通过；minimal=0、headless=42、base=49；headless 完成真实 CLI 回复并干净退出 |
| E6 禁用矩阵 | `builtin_disable_matrix_scenario.py` 遍历全部 49 个 row；只有依赖闭包缺服务，无失败 Fiber；2,157 次无关插件观察的 apply 增量均为 0，全部 49 项禁用选择冷启动后仍成立 |
| E7 代表替换 | Ledger 附件、UI 查询、Timer、Gateway 查询协议、Models 驱动贡献、Compaction 摘要归档均有独立实现；借用消费者不重跑，硬依赖按原规则重激活 |
| Core 规模 E8 | `measure_production_sloc.py --json`：agent 14,937 + bootstrap 1,234 + core 833 + infra 134 + main 633 + utils 206 + migrations 342 = **18,319** |
| 旧 Core 数据 | `binding_upgrade_scenario.py` 用 bdcc1660 生成绑定、消息、附件、身份、pending handoff；新 Ledger 可读，十张源表完整行与 uploads 文件摘要不变 |
| Ledger 并发/崩溃 | `ledger_runtime_scenario.py` 真实入站提交后 SIGKILL、四次重启与阻塞事务换代；原消息不重复、不丢失，旧事务排空后新代可读 |
| 派生库与学习 | `ledger_derived_migration_scenario.py`、`akasha_derived_recovery_scenario.py`：只复制、不删源表；丢库补算后相同非空召回，源库与图字节不变 |
| 选择迁移 | `bundle_choice_migration_scenario.py`：旧代码生成状态；写 patch 后失败再重试，共享 cache/config 多 workspace 与新 workspace 隔离 |
| 未知外部效果 | `delivery_unknown_crash_scenario.py`：外部文件已写后 SIGKILL；恢复 failed/可能已送达，发送次数 1，原消息不变 |
| 压缩 | `compaction_workflow.py` 五个真实 HTTP 场景通过，含完整 Turn、开放 Turn、短尾、软/硬失败，原消息不减少 |
| 进程/MCP | `host_process_plugin_scenario.py` 与 `mcp_plugin_scenario.py` 验证实际进程、取消、换代、重启、卸载和清理；MCP 旧字符串端点通道已删除 |

`binding_upgrade_scenario.py` 故意同时包含旧模块导入与旧构造参数，必须在两个版本
的独立进程运行；不声称它通过当前模块树的单版本 Pyright。产品类型门没有豁免。

新制品源为 `d9b30c1264d2a51cf802ecb1d7eadf44d9334f66`，其后只有 Timer 验收与文档。
本机原始日志位于 `/tmp/1179-*`；制品在 `/mnt/data/coding/1179-reviewed-distribution`。
三模式证据位于 `/mnt/data/coding/1179-build-tmp/akashic-bundle-modes-7lhaqcyj`；
禁用矩阵完整报告位于 `/mnt/data/coding/1179-build-tmp/akashic-disable-matrix-zx0bgx6j/report.json`。
这些本地路径不是远端 CI artifact。重现应使用上表脚本和固定 commit 重新构建。

## 当前 Fleet 外部源码

维护范围只认 Fleet `cc1c812` 的 `.gitmodules`/gitlinks。14 个仓库均从各自当前主分支
建立独立 worktree 与恢复分支，未修改安装 cache、旧 checkout 或 Fleet gitlink。
它们删除重复合同/兼容导入，改用实际提供方公开合同；CI 固定 Core `1a5327e4`。
本栈后续没有改变这些外部合同的行为；不把来源 CI 通过当成线上激活证据。

| 插件 | Draft PR | 固定 head |
|---|---|---|
| calendar | [#17](https://github.com/akashic-plugins/calendar-mcp/pull/17) | `9b5821f3` |
| citation | [#9](https://github.com/akashic-plugins/citation/pull/9) | `b85a295a` |
| feed | [#24](https://github.com/akashic-plugins/feed-mcp/pull/24) | `5a17718b` |
| fitbit | [#25](https://github.com/akashic-plugins/fitbit-mcp/pull/25) | `96d48b2f` |
| github-watch | [#16](https://github.com/kachofugetsu09/github-watch/pull/16) | `5b2a5eb3` |
| huayue-skills | [#11](https://github.com/akashic-plugins/huayue-skills/pull/11) | `94a9e1f6` |
| meme | [#13](https://github.com/akashic-plugins/meme/pull/13) | `ab990975` |
| observe | [#20](https://github.com/akashic-plugins/observe/pull/20) | `e75b01d7` |
| proactive_feedback | [#23](https://github.com/akashic-plugins/proactive_feedback/pull/23) | `26df1ae5` |
| setup_helper | [#9](https://github.com/akashic-plugins/setup_helper/pull/9) | `eeca6aa8` |
| shell_restore | [#10](https://github.com/akashic-plugins/shell_restore/pull/10) | `e1bb8e11` |
| shell_safety | [#9](https://github.com/akashic-plugins/shell_safety/pull/9) | `56dbe9ca` |
| status_commands | [#13](https://github.com/akashic-plugins/status_commands/pull/13) | `c913e474` |
| steam | [#17](https://github.com/akashic-plugins/steam-mcp/pull/17) | `9e7ab07b` |

14 个外部 PR 的实际 CI 已全部通过，包含各仓库既有回归与类型检查；本机还执行了
Feed/Steam MCP、GitHub Watch/Status/Observe/Proactive 的物理 I/O 排空场景。
所有外部源码 PR 应在 Core 公共合同层合并后按依赖顺序合并，再显式更新 Fleet gitlink、
正式安装与验收。本次没有执行这些发布动作。

## 文档、限制与恢复

当前架构由 [组合方向](agent-plugin-composition-direction.md)、[状态地图](persistence-state-map.md)、
[能力消费者对账](capability-consumer-audit.md) 与上述 ADR 拥有。#1179 早期 E9 指向的
根 `CLAUDE.md` 在当前仓库已被 `.gitignore` 明确归为本机文档；不强行纳入版本，
不覆盖用户本机指导。版本化终态说明在组合方向文件，`INDEX` 与 `NOW` 已改指向本入口。

第二 provider 是针对明确端口的替换证据，不是可直接替代全部产品功能的备选发行版：
附件例子只支持本地文件；Gateway 只读；模型驱动只做文本；摘要只读首代 v2；
UI 示例没有默认配额和超时策略。Timer 硬依赖消费者会重激活，借用消费者不重跑。
不把这些差别隐藏在兼容分支或假的成功回执里。

前端区域 owner slot 化属于独立 P5；真实 Android、Telegram 账号、生产模型与
hua-home 部署不在本次隔离验收中。正式迁移、删旧表、删除旧记忆库和重建生产状态
都没有执行，不能仅凭这份记录关闭这些后续工作。

恢复源码使用每层父 commit；恢复运行状态使用迁移保存的原文、完整数据库与原版本。
派生库复制保留旧表，未发明自动向旧版本同步新向量的协议；代码回退不能撤销已发送
消息或恢复已被关闭的后台进程。所有 Draft PR 保持可审查，不自动合并。
