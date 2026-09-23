# NOW

这份文件只保存 Akashic Agent 当前仍未完成的工作。事项完成后删除，不保留“已完成”记录。

## P1 · Akasha 学习图一次性重放

`plugins/akasha` 已收口为单一学习实现（重建 = 空图 + 无切换上界重放同一个
`MessageConsumer`），旧的稀疏索引、`legacy_prefix`、frozen history、离线 CLI 与 repair
通道退役。待完成：

- 用插件自有 migration bundle 登记的一次性重放完成线上切换，并按
  `docker/debug/akasha_replay_compare.py` 核对历史前缀（本机副本实测：legacy 复现
  5549/5581、相对顺序无逆序、交集身份与时间瞬间零差异）。
- 重放会丢掉旧图里 26 条早期 `remember` 事件。维护者已确认它们不是 Akasha 现行反馈
  通道的产物（属于很早期的 tool result），接受该偏差，不另立 carry-over 迁移。
- `memory/akasha-v2-index.db`（约 511MB）退役后的物理删除单独授权。

## P0 · 插件正交化实施与最终验收

按 [0065](decisions/0065-plugin-boundary-checks-do-not-grant-core-ownership.md) 与
[阶段验收](design/plugin-boundary-foundation.md#7-验收标准) 完成已授权的 stacked PR 实施。
当前优先完成全部本地插件的行为回归、默认组合与无 checkout 的分发产物验收。
维护者已明确将第三方外部插件的进一步迁移与验收后置；已提交改动保留，
不将第三方迁移完成作为本地插件阶段的退出条件。
当前维护者要求只交付 stacked Draft PR；仅做静态检查和独立只读评审，
不运行测试、Gate 或 CI。真实运行验收须另行获得授权，不因实现完成自动开始。
验收必须覆盖 Core-only CLI/AppRuntime、独立子集和异名 provider、generation/归档生命周期、
实际 Message 与持久送达闭环；不能以 import 数量清零代替这些证据。

## P0 · Akashic Channel 与 Web/Mobile Adapter 实现

[Akashic Channel 与 Web/Mobile Adapter 规格](design/akashic-channel-client-adapters.md) 已确认
一个 `akashic` Channel、两个薄 adapter 和一次 breaking rekey。渠道归属按
[0067](decisions/0067-clients-are-ordinary-plugin.md) 修订为普通插件。实现已获授权，当前核对
Session/Message 全身份迁移、配置、Akasha 和 Android 强制全量同步；不得直接迁正式 workspace。

## P1 · 移动端主题 token 边界

[`移动端投影审计 D2`](design/mobile-projection-audit.md) 已确认原生壳 Compose 色板与 Core WebUI CSS token 是两个渲染层的表示，不是重复 owner。仍需决定色值一致性由构建期产物还是显式 token 边界保证。

- 移动端用户 checkout 存在未提交 Theme diff（Theme.kt 等 5 个文件）；D2 决策（原生壳与 WebUI token 边界）完成前不得合入。

## P0 · 单图插件系统与局部换代（Issue 750）

- T-fb9a7f 已闭合 Selected-Load A 的真实 selection identity oracle，并完成 Core T02 局部行为证据：selection 1/1、Core 其余四组 25/25。详见设计文档 T-fb9a7f 段与 `/tmp/i750-core.YE9wsb/`；不外推为 Issue 750 完成，完整回归/Gate/CI/正式运行/final enable 仍为 WIP。

- T-b0ab9e（T06-Models R2）上批已知事实：`reader.__self__` 替换、同路径断言、Models 生产/测试对账及文档已由主审与独立只读 review 静态接受，累计静态材料未运行行为测试；T-fa9271/T-83b918 原失败稿不追认为整批通过。生产 Models/Core/clients/Bindings/Manager 未改，持久化 delta 为 0。ModelsStore/CAS、credentials、descriptor/schema、Message、model_calls、continuation、embedding identity 与可选 chat API 注入保留；不能把静态结论写成行为验收或最终 enable。

- T-d44273 的 R5 记录已由 T-a9246e 承接：R4 production 修复已由主审与独立概念 Gate 接受并冻结；R4 的两类 red 要准确区分为 C1 child CE 被 TaskGroup 忽略导致 source-local warning 缺失（未击穿 watcher），C2 sync `on_close` error 进入 TaskGroup 导致健康 peer 无法完成。R5 原 artifact `/tmp/i750-reply-r5.b4WgrJ/` 仍保留为历史事实；当前 A cleanup/oracle、Source registration wake 与真实执行证据以 T-a9246e 段为准，不写成 Issue 750、Gate、部署、正式运行或 final enable 完成。
- Source wake 当前状态：R2 ordering、R4 tests、T-8677b9 production 与 T-a0440f 测试收尾均已接受；production hash `24aabacea42855c20f8bb0fed47f17b34f708cb32fa5672dad9f719b6fa37b18` 已过主审及独立 `gpt-5.6-terra/xhigh` 概念 Gate，T-a0440f targeted 1/1、Reply 23/23 green。Source head observation 不持久化，重启/registration replacement 仍可重读历史。T-82189d 已完成 Subagent→Conversation→Reply owner 接线与定点验证，但报告尚未通过：Reply 内 `ToolProgramFactory` 调用仍缺 Tools owner scope，局部 drain 也被 `delivery_policy` readiness failure 阻断；详见 [Issue 750 Source wake](design/issue-750-plugin-publication-simplification.md)。其它 provider/RPC/consumer、snapshot/fence、offline/candidate/freeze、T06/T07、累计 Gate、运行验收与 final enable 仍未完成。
- Models 不再有三张运行 registration 表或 `sealed`：config 可读性、vision binding 保留在 Models 本地边界，enabled connection 的 `open+close` 延后到真实首用或显式 probe；`StoredSnapshot.revision`、CAS、credentials、descriptor/schema、Message、model_calls、continuation 与 embedding identity 不变。当前只做 AST、hash、diff 检查，未执行测试、Gate、CI、应用 import、插件安装、业务进程、正式 workspace 或部署。
- 公开卸载实现（T05-B）与 T05-C 生产接线静态通过：同一 Manager owner 覆盖 disable、已选但无 Fiber、FAILED/active+draining、selection CAS、accepted handoff、真实 hard-consumer 排空和现有 cache/manifest finalizer；finalizer 通过 `complete_critical(to_thread(...))` 持有真实删除线程，状态目标从成功 accepted 结果推导，不新增第二套卸载 receipt/schema，不删除 plugin-data、Message、附件、历史 binding、归档或 Delivery 记录。T-129aa6 的 T05-C-R2/T05-D 测试与文档静态通过；T-9990aa 整体未通过，T-fb6d31 生产修复静态通过，T-7ff900 A1～A4/B 测试源已由主审与独立只读 review 静态通过但未运行；测试、Gate、CI、运行验收和迁移演练仍未执行。
- T-fbf63b 已删除无生产消费者的 EventBus RuntimeSnapshot lease/store 保护层、旧 admission 等待任务与单字段 envelope，并删除 Manager/ValidationHost 的绑定调用；generic queue/dispatcher、handler Task、错误/取消隔离、drain/join 和关闭责任保留。生产删除与 Inspector 坏节点 admission oracle，以及当前累计 `tests/test_plugin_external_loader.py` 已由主审与独立只读静态接受；T-7912fd 原 R1 缺口不作为已通过稿，行为测试未运行。owner 链为：`Core/ValidationHost → EventBus queue/dispatcher → generic handler Task → drain/join`；`Plugin Context → Root EventRegistry → Fiber Effect`（独立，不桥接）。不删除 RuntimeSnapshot 其他消费者，不改持久事实；当前只做 AST、内存 compile、diff/空白、符号和哈希核验。
- T-71cb50 修订 EventBus R1 两项测试源：self-cancel 使用未置位真实 `asyncio.Event` 交付取消，queue 关闭后核对 empty 并调用公开 `drain()`；同时删除无生产消费者的 `RuntimeMessageDisplay` wrapper 与唯一旧 snapshot generation 测试。生产展示继续只有 `app_server.py:message_display` 和 Manager `core.message_display.v1` 两个 live-Root consumer。上述 EventBus R1 尾修、RuntimeMessageDisplay 删除、T-fbf63b 生产删除与 Inspector oracle 已由主审与独立只读静态接受，行为测试未运行。T-4be90d 整体 review 未通过；T-131de5 原稿不追认为通过稿；后续修正另行记录，未运行测试。N1/N2、UI、其它 provider/RPC、scope 外 consumer、current_snapshot/fence、offline/trusted watcher、candidate/freeze、T06/T07、运行验收和 final enable 仍未完成。

## P0 · 独立语义验收

- 将 CTX-001 当前的 trace、完整状态快照和 fixture `DELETE` pilot 升级为 SQLite authorizer 与一次性候选真实 retry seam mutant；导入失败、fixture 失败或超时不得计为 mutant kill。
- 建立受保护路径 policy：`semantic_delta: none` 的普通实现改动不能同时修改 P0 oracle、mutant 或 coverage baseline 来获得全绿。
- 建立轻量 `change-intent` 校验，检查实际 diff、允许路径、受保护状态和副作用是否超出声明。

## P1 · 工作流扩展

- 按 [`容器与 Host Bridge 非迁移实验合同`](design/akashic-container-host-bridge-experiment-contract.md) 完成 mise/锁文件前置、本机 Local/Bridge/容器分层验证和 hua-home 隔离候选运行时；在 capability matrix、Supervisor 故障注入、OpenCode V4 Flash High 与正式状态零写入证据齐全前，不启动正式 workspace 迁移。
- 把 `projectneed.md` 中其他 P0 不变量逐步迁入可执行契约，优先处理 MEM-001、MEM-002、OUT-001、PLG-001、PLG-004、WSP-001 和 BAK-001。
- 为高风险 refactor 增加 base/candidate 差分回放，核对持久 write set、事件、外部调用和错误分类。
- 由维护者继续确认 [`design/persistence-state-map.md`](design/persistence-state-map.md) 的 INT-009、INT-010、INT-012～INT-014，以及旧消息编辑和 turns retention；INT-001～INT-008、INT-011 已提升为 projectneed 条款。
- 把已确认的持久化状态地图转成机器可读备份 manifest，补齐目录快照、global companion state 和隔离恢复演练；确认 snapshot 能启动只读 runtime，并读取会话、记忆、调度、插件数据和主动流程连续性。

## P2 · PTY resize 既有规格差距

SH-003 写有 resize，但当前 ShellProcessManager 和 Host Bridge 尚无 resize 入口。
Protocol V2 保留已有 PTY 输入、输出和 stop，不把协议升级视为 resize 验收通过；后续独立确认实现范围。
