# Issue 750：单图插件系统与局部换代任务拆分

## 2026-09-23 · 开发基线与 A1/B1 集成

T-3ed982 修复 Gate 模型影响映射后，独立 `gpt-5.6-terra/xhigh` 的 T-a1ea29
接受 `3d72a597` 为开发基线，不代表最终发布。下方“本地集成继续执行”和“WIP 基线”
是此前阶段的历史记录；当时的 Tools owner、DeliveryPolicy readiness 和旧测试入口
阻塞已被 T-e69982 的集成修复及最终验证覆盖，不应继续作为当前缺陷。

三条 lane 从该基线隔离开发，主审串行集成。T-a03bcb 独立接受以下相邻改动：

- A1 `e280d323`：只删除 Manager 五个无消费者的私有候选函数，共209行；
  公共拒绝入口、当前 live Root、选择与真实清理责任不变。基线与 A1 的同四项
  原生对照各通过；A1 干净提交累计 Gate 27/27 通过且无资源残留。
- B1 `9752a356`：真实 distribution A/B 回归证明普通 `ensure_profile` 保持 A，
  显式公开在线安装才选择并运行 B，重启仍读 B 归档，peer、旧 receipt/归档与用户数据
  保持。主审原生三模块33项通过；不以 worker 的 selector 诊断运行作为验收。

主审在 integration 上先快进 A1，再以 `e4ed9e73` 导入 B1；两个 lane 的绿色结果
不替代联合树。这批联合验证、最终 HEAD/tree、源码与计划摘要、逐项 Gate 和恢复材料
记录在 `/mnt/data/issue750-ab1-integration-20260923.3z1ln0/`。
该目录中的命令结果才表示本次联合验收，不把待运行项写成通过。
共享历史基线 tag 保持不动；未 push、修改 PR、部署或触碰正式 workspace。

后续已审查源码继续串行纳入 integration：

- A 的共享固定输入准备最终提交 `d1f5a99f` 经 T-8555a8 接受。Manager 与未来离线
  采用入口复用同一 archive/config/identity/compile/environment 准备算法，返回完整
  `archive_ref`，不另暴露裸 code 引用；离线准备不执行插件或构造 Root/Fiber/Scope。
  原生98项及两份类型检查通过；集成对应提交为 `cf445133`、`a9ccf34b`。
- C1 真实组合测试曾发现 RawInbound 的 provider identity 在 handoff 中丢失。
  C2 `97cb8e14` 保留原断言并修复生产，经 T-369cac 接受，集成为 `1ac8d5ff`。
  新 handoff 在既有 metadata_json 内保存版本化身份对，含显式 `(None, None)`；
  恢复前剥离内部字段。缺字段的旧行只保持原 sender/chat_id 解释，不回填或声称找回
  原始身份。重投仅在身份及其余字段精确等价时幂等；坏新格式 fail-loud，不改 schema、
  既有 row 或物理删除权限。C1/C2 关键38项、Channel34项、Store/Mobile等65项及
  两份类型检查均原生通过；组间有重复控制，不合称137个不同测试。

两批的分支绿证据不替代新联合树。此次原件备份、精确来源、联合命令与最终结果统一在
`/mnt/data/issue750-ac-integration-20260923.JAkW3G/`；其中绑定最终源码的命令结果
才是联合验收结论。未运行的 Gate、远端 CI 或正式状态不得由分支结果代替。

剩余工作是 T06 其余可达性/持久清理边界、T07 显式离线发布和最终联合验收。
普通换镜像重启仍按既有 selection 读取旧归档，不能声称自动采用新版 Akasha。
远端 CI、真实发布/恢复和 final enable 仍未验收。

## 2026-09-23 · R1 语义与验收映射

本轮从集成分支 `33bd6d41` 和固定 `main@ae444d47` 继续。执行者只有一个
`gpt-6-sol/high` integration writer；未更新 PR、正式 workspace 或外部插件环境。
恢复归档、隔离测试的 argv/JUnit/source hash 和 Gate 报告在
`/mnt/data/issue750-green-r1-20260923.68KJfw/`，最终本地验收状态由
`/home/huashen/.huagenteam/team/tasks/T-e69982/reply.md` 与其中指向的结果文件确定。
T-71d1a1 对 Gate 模型影响映射提出 request-changes；随后 T-3ed982 修正，
T-a1ea29 已独立接受修复后的开发基线，准确边界见上节。
先前本地测试结果保留，远端 CI、正式部署与 Issue 750 最终验收未完成。

R1 继续保持 0072 的单一 live Root：selection CAS 是持久选择，运行中的 Fiber 是
另一个事实。首次空选择先提交固定归档再加载；B 已被选中但加载失败时保留 B 与错误，
不能自动复活 A；显式 retry 从 B 的原归档建立新 generation。卸载与更新先返回
accepted，旧 owner 的调用和资源仍由 Manager 排空，失败保留 owner 供重试。
这些路径由当前 `test_plugin_hot_reload`、`test_plugin_stable_runtime`、
`test_plugin_uninstall_root_drain`、`test_plugin_update_source` 和
`test_plugin_retired_activity_recovery` 验证；旧 candidate 晋升、discard 与全图
snapshot publication 用例不再充当单图的成功 oracle。删除文件到现行 owner 的逐项映射
见测试清理 ledger。

停止期发现两处可达竞态：Subagent watcher 等待子任务结算后，Fiber 可能已撤销新
OwnerCall；此时只放弃本次新接纳，保留持久指针由下次启动读取，其余异常继续抛出。
Web adapter 关闭只关闭该 binding 的新接纳与 follower，真实 Web transport 的
`_stopping` 仍由 transport stop 拥有；因此已捕获的旧回调可排空，新 binding 可接入。
Subagent 恢复 12 项与 Web 入口 45 项已在隔离环境复验。

结构映射只更新精确 `agent.plugin_composition.host` 模块许可、四个实际 role key、
由 typed RPC 生成的控制 schema，以及 Gate 的 Models 删除路径与退役 nodeid。
`plugin_uninstall_drain_finality` 现执行 live MCP 换代、Socket accepted/排空、
cleanup retry、隔离 host grant、Shell owner 与 plugin-data 保留。Gate baseline
除计算所得 `catalogDigest` 外没有改 `base`、`coveredP0` 或 `acceptedGaps`；
27 个公开场景、原 84 项矩阵和全量 Python 的最终结果必须分别读本轮运行证据。

## 2026-09-23 · 本地集成继续执行

用户恢复了唯一集成 worker 的执行授权：在隔离集成 worktree 上正常合入固定
`main@ae444d47281b5b25bb467a76dd25eab02c95ec08`，修复 T-82189d 暴露的 Tools、
DeliveryPolicy 和 live Root consumer，以及主干 Reply 存活性回归。实施指定
`gpt-6-sol/high`；后续 Luna 实现槽指定 `gpt-6-luna/xhigh`。本轮不分发三路任务，
独立概念 review 仍按 WORKFLOW 保留。

隔离验收中 Subagent 12/12、Reply follow 23/23、主干 Reply liveness 31/31、
消息命令 10/10、扩展 consumer 8/8、Web 101/101 和 SDK 2/2 已通过。全量 Python、
测试类型检查与结构/Gate 尚未通过：旧 candidate/snapshot 测试仍需逐项迁移；
`plugin_boundary.toml`、`schema/app-server-v2.json` 与 Gate inventory/baseline
需要超出本轮手工修改范围的决定。准确运行产物、每个冲突的取舍和恢复目录保存在
`/mnt/data/issue750-green-integration-20260923.3dXxg9/`。这些局部结果不构成
共同绿色基线、远端 CI、正式状态验收或 final enable。

## 2026-09-23 · WIP 基线与并行交接

本次只冻结累计源码、准备隔离工作区并交付 Draft PR，随后暂停；没有修复
T-82189d 的新失败，也没有分发后续任务。下述记录是当前交接状态，不把后文各批
历史通过项提升为系统通过。Draft PR 用于累计审查，不关闭 Issue 750、不允许合并或部署。

### 基线与已知失败

- 累计源码检查点：`e34161c728d54429ce36ef06cada75f7a89c3819`，基于
  `69a67e7f3d464d1bd749d738d2d31402229fde60`。它保存 173 个累计变更路径，
  包含 T-82189d 未验收实现；旧 worktree 的 `.done` 保留但不提交。
- 各新工作区从上述检查点之后的同一个交接文档提交创建；准确 HEAD、tree、路径和
  分支由本机 `/mnt/data/issue750-parallel-baseline-20260923.RXUmH9/environments.json`
  记录。原 `feature/issue750-local-plugin-graph` 保留为交接基线，不再作为并行 writer 目录。
- T-82189d：Subagent 1 passed / 11 failed；Reply follower 23/23；Message Commands
  10/10；八项定点 consumer 测试 5 passed / 3 failed。Tools owner scope、
  `delivery_policy` 启动 owner 和 Core 旧 snapshot oracle 尚未闭合。局部 drain 没有证明
  更新成功；被中止、无 JUnit 的探索运行不计入验收。
- 本次仅执行累计 Python 源的 AST/内存 compile（150 个通过），没有重跑行为测试或 Gate。
  累计 staged diff 检查仍有 `agent/plugin_composition/host.py` 文件末尾空行告警；
  为保持源码检查点原样，没有顺手修改该生产文件。

### 主线同步不是自动通过

预检目标为 `origin/main@ae444d47281b5b25bb467a76dd25eab02c95ec08`，比原基线多七个
提交。`git merge-tree --write-tree --name-only` 报告以下 26 个冲突路径；它只生成
预览对象，没有修改任何 checkout/index，也没有留下未完成的 merge：

```text
agent/plugin_composition/admission.py
agent/plugin_composition/context.py
agent/plugin_composition/runtime_catalog.py
agent/plugins/manager.py
agent/plugins/reload_journal.py
bus/queue.py
docker/debug/wake_v3_provider_e2e.py
plugins/channels/provider.py
plugins/models/state.py
plugins/reply/follow.py
plugins/stable_view/plugin.py
tests/test_channel_input.py
tests/test_commands_provider.py
tests/test_mcp_binding_scope.py
tests/test_message_bus_admission.py
tests/test_message_model_selection.py
tests/test_mobile_ui_provider.py
tests/test_plugin_hot_reload.py
tests/test_plugin_latest_models.py
tests/test_plugin_runtime_control.py
tests/test_plugin_uninstall_root_drain.py
tests/test_plugin_update_source.py
tests/test_plugin_updates_api.py
tests/test_reply_follow.py
tests/test_stable_view_plugin.py
tests/test_subagent_messages.py
```

下一轮先由唯一集成 owner 处理主线语义差异，再冻结新的共同起点并下发任务。
不得让多个 worker 各自解决这 26 个冲突；不得用整文件 ours/theirs 覆盖主线修复。
当前新工作区可隔离检查，但不代表已经与最新 main 集成或可发布。

### 工作区与写入规则

全部目录位于 `/mnt/data/coding/`；本轮只创建环境，不启动 agent 或分派代码任务。

| 用途 | worktree 目录 | 分支或模式 |
|---|---|---|
| 集成与 Draft PR | `akasic-agent-issue750-integration-20260923` | `codex/issue750-integration-20260923` |
| 实现槽 A | `akasic-agent-issue750-worker-a-20260923` | `codex/issue750-worker-a-20260923` |
| 实现槽 B | `akasic-agent-issue750-worker-b-20260923` | `codex/issue750-worker-b-20260923` |
| 实现槽 C | `akasic-agent-issue750-worker-c-20260923` | `codex/issue750-worker-c-20260923` |
| 独立只读评审 | `akasic-agent-issue750-review-20260923` | detached HEAD；合同禁止写入 |

```text
冻结的共同起点
├─ 实现 A ─┐
├─ 实现 B ─┼─ 各自提交与证据 → 唯一集成 owner → 累计验证 → Draft PR
├─ 实现 C ─┘                                      │
└─ 独立 reviewer ← 精确待审 SHA ────────────────────┘
```

1. 一个 worktree 同时只有一个 writer；任务必须固定起始 SHA、允许路径和验收范围。
   模型与任务分配留待下一轮，不因有空闲槽就同时修改同一接口。
2. Core/Manager/公共接口有唯一责任人；共享 API 变化先集成，再更新依赖任务起点。
   `NOW`、本设计、决策与产品规则由集成 owner 唯一写入，worker 只在回执提出文档变更。
3. worker 不直接推 Draft PR head。集成 owner 按依赖顺序接收精确提交、审查 diff 并运行
   累计测试；Git 无冲突不等于行为兼容。不得跨 worktree 复制未提交文件当成合并。
4. 仅在依赖文件 hash 相同时复用现有虚拟环境作只读解释器，不安装或升级共享依赖。
   每次运行使用独立 TMPDIR、plugin home、workspace、basetemp 和实际分配的端口；
   清除继承的 `AKASHIC_*` 和测试 Git 配置。依赖变更必须独立建环境，不能污染其他槽。
5. `/mnt/data/issue750-parallel-baseline-20260923.RXUmH9/run-python.py` 提供本机隔离
   Python 入口；它不自动运行测试。测试仍须显式选 nodeid、超时与证据目录，禁止指向正式
   workspace/cache。只读 review 通过合同和前后 hash 核验约束，不声称 detached HEAD
   本身提供文件系统只读权限。
6. 任何 must-fix、冲突或未通过测试都留在 Draft 状态；完整 Gate、CI、运行验收和
   final enable 另行完成。创建 PR 不授权 merge、auto-merge、release 或 deploy。

### 恢复与停止点

本次恢复目录为 `/mnt/data/issue750-parallel-baseline-20260923.RXUmH9`：保存原 HEAD
bundle、binary patch、index、174 项原始状态（含 `.done` 和删除记录）、副本 hash、
主线合并预览，以及 T-82189d 最终 console/JUnit 等证据。新增环境与 Git 提交不改正式
workspace 或数据库；恢复应在新目录按 manifest 重建，不覆盖用户 checkout。
旧 worktree 中还观察到一个此前启动的 `test_task_scopes.py` pytest 进程，未确认清理
归属，因此未擅自终止；新环境不复用其 TMPDIR 或运行 workspace。没有开启新业务进程。

本轮交付后停下，等待下一轮明确拆分与 hgt 派遣。

- T-750784 → T-d80010 → T-673b11（Reply-Source owner R1→R2→R3）：R1 整批不接受，仅保留
  `TaskServiceClosed`/`formal=False` 的窄事实与 typed 接纳方向；R1 没有新增 follower 回归，
  前置 red 也没有原始 command/console/JUnit/exit 产物。R2 收敛 `follow.py` 的真实取消结算：
  `Task.on_done` + 局部 Event 等待 physical finally，再用公开 `Task.join()` 读取终态，不创建
  join waiter、不依赖 task factory。R3 又收口同拍 marker/caller 取消计数、child
  `CancelledError` 的 cleanup cause 与测试 finally；当前行为证据、原始 red/green artifact、
  AST/compile、diff 与 frozen152/152 只记录在下方 T03/Reply 段，不能写成 Issue 750、Gate、
  部署或 final enable 已完成。T-673b11 的 R3 整批随后被主审与独立
  gpt-5.6-terra/xhigh 概念闸门暂拒；T-cfe587 R4 的 production 修复已由主审与独立概念 Gate
  接受并冻结；T-d44273 R5 只补测试 cleanup/oracle 与可核验执行记录，不改变 production。
- 本轮保留 Message/schema、intent/receipt、selection/journal/history/archive、descriptor、
  `root_ref`、metadata、credential、正式 workspace 与持久迁移 delta=0；隔离测试的 SQLite、
  archive、receipt 和 plugin/home/workspace 写入不是正式迁移。T-a9246e 已实现本地
  Source registration wake，并以真实隔离 W1/W2/W3/W4/W5 覆盖 late registration、LOADING
  registration、同名换代、来源独有结算失败和撤销；Subagent→`REPLY_PROGRAM.report`→Reply
  writer、其它 provider/RPC/consumer、legacy snapshot/fence、offline/candidate/freeze、
  T06/T07、累计 Gate/CI、运行验收和 final enable 仍为 WIP。

- 状态：第三十九版，2026-09-22。T-fb9a7f 的 Selected-Load A/Core T02 局部行为证据已完成（selection 1/1、Core 其余四组 25/25）；T-e87866 已完成 Commands provider 与 Message Commands 的真实局部验证：provider 13/13、message 9/9，reply_entry 0/1，合计 22 passed、1 failed，失败定位为 Reply follower 进入 Conversation source 时缺少 Conversation owner call，详见 T03 Commands 段与 `/tmp/i750-cmd.l6NEiF/`。T-e87866 当批不改 production/测试源码或正式 durable data；失败入口及 Source/Reply/Conversation owner 迁移、其它 provider/RPC/consumer、`current_snapshot`/fence、offline/candidate/freeze、T06/T07、完整回归/Gate/CI、运行验收和 final enable 仍为 WIP，不宣称 Issue 750 完成。
- T-83b918（T06-Models R1）上述代码与文档修订、T-b0ab9e（T06-Models R2）的 `reader.__self__` 替换与同路径断言已由主审与独立只读 review 静态接受，累计 Models 静态材料未运行行为测试；T-fa9271/T-83b918 原失败稿不追认为整批通过。删除无生产消费者的 Core `RuntimeModelControl` 与 `model_catalog` companion、ModelsStore/CAS、credentials、descriptor/schema、Message、model_calls、continuation、embedding identity 和可选 `create_chat_app(model_control=...)` 合同均保持不变。生产 Models/Core/clients/Bindings/Manager 未改，持久化 delta 为 0；不能据此写成行为验收或 Issue 750 完成。
- 当前接手点：Web/Dashboard 使用 `DASHBOARD_ROUTES` 的真实 host route tuple、贡献 Context 的 `RUNTIME_STARTING`/ACTIVE 初始化、registration Effect 前置登记与失败 close retry；HTTP/WS 通过 live Root/UI provider/原 contributor scope 和 `root.generation_id + catalog.identity + module generation` fence。Mobile 使用单一 live Root、真实 `UI_SLOTS` 与贡献 Context 自持 async catalog/asset/query scope；三个入口都在目标 scope 执行 `available`，合法 retained permit 可在 UNLOADING 完成，child 取消仍排空物理线程，Root cleanup 可重试。Core 只提供 activation-local `RuntimeScope.wait_admission_closed()`。持久化 delta 为 0。

### T-3c5a49 · Channel 局部生命周期 review 修复（静态待审）

本轮保留 T-c76e09 的宿主只读事实；独立 review 发现其行为源仍有缺口，本轮修复后由
`agent/plugin_composition/host.py` 的 frozen/slotted
`HostInfo(boot_id, validation)` 独占；Manager 继续从 `_host_boot_id` 提供正式 boot，candidate
复用同一 boot；验证测试使用普通隔离 Root/Fiber 与隔离 identity。Channel provider 的声明、binding、adapter 与停止
Effect 由贡献 Fiber 持有；每次 activation 使用 `ctx.generation_id` 运行身份和新的
`binding_token`，ready 后在该 Context 的 `runtime_scope` 中局部开放。

```text
HostInfo(boot_id, validation)
        │ 只读注入
        ▼
贡献 Fiber/Context ── RUNTIME_STARTING → closed ready → ACTIVE/health → open
        │                         │
        │ Effect                  └─ capture → child Task → adapter/input/control/ack
        ▼
adapter stop → listener settle → durable defer/保留失败 owner
```

`ChannelBindingLease` 只保留 exact binding 的 transport in-flight claim 与 close 责任；
`CHANNEL_INPUT`、控制回执、outbound delivery、presentation callback 和 recovery 经过贡献
Context 的真实 scope。普通 input 的 prepare → commit → complete、durable handoff、identity、
attachment、真实 receipt 与失败保留语义未改。bootstrap 与隔离 validation Root 的恢复入口只取
当前 live Root 或隔离 Root 的 `CHANNELS` provider，不重建 Root、不扫描 cache。

本轮测试源使用真实 live Root、局部 binding 和隔离 CompositionRoot/Fiber；crash 夹具在
`provider_started` 的 edge `fsync` 后立即 SIGKILL，确认 provider-calls 尚不存在。已持有的
request/attachment/durable owner 在 UNLOADING 期间继续完成必要清理；新的 request/recovery
必须拒绝或保留 pending。所有测试、Gate、CI、应用 import、插件安装、业务进程和正式
workspace 仍未运行。review 仍未通过，T05 公开安装/accepted→active、其它 provider/RPC/UI、
scope 外 admission/manager、候选路径最终删除、freeze、T06/T07 与最终 enable 仍为后续工作。

### T-b980b1 · Sources 入口与 Channel 只读修正（静态待审）

本轮把 `Sources` 的实际宿主 Context 保存到唯一 registry，并把每次入口固定为一条明确的
嵌套 scope 链：

```text
raw ingress → Channel contributor Context
           → Sources Context.runtime_scope()
           → selected Source Context.runtime_scope()
           → conversation.accept → MessageWriter.bind → Message
```

已持有的 selected callback permit 继续保护旧 contributor；其卸载期间新入口拒绝，不改选默认
来源，也不在 Core 增加跨 owner 借用。测试源补了真实 Sources callback 排空、旧/新 Channel
activation、attachment read lease、durable settlement 和隔离 validation 的静态证据；T-9f83dd
的生产 Channel provider 与 crash runner 仍冻结。本轮只做 AST/内存 compile、diff/hash 和
`git diff --check`，未运行测试、Gate、CI、应用 import、插件安装、业务进程、正式 workspace
或部署；T04/T05、其它 provider/RPC/UI、scope 外 manager、候选路径删除、freeze、T06/T07 与
最终 enable 仍为 WIP。

- 已确认：一张运行图；安装直接请求生效；按依赖局部换代；不保留候选晋升、自动回退或完整源码 HMR；普通安装不得重建全局图。
- 第三版的“整 Gateway 重启、Supervisor 承接发布”方案不再采用。本版取代该提案，不把旧版评审结论算作本版通过。
- [0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md) 与 PLG 条款已由 T01 对账改写为单图语义，T01 修订已通过 Codex 静态 review；本文件不声称尚未迁移的公开安装链和运行消费者已符合新合同。
- 证据基线：Akashic `69a67e7f3d464d1bd749d738d2d31402229fde60`；DSH `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8`。
- 阅读顺序：第 1～4 节解释目标和边界；第 5 节是任务拆分；第 6 节是最终验收；第 7 节提供 DSH 源码对照。

### Models M2 实现边界（本批）

Models 现在以一个私有 registration record 作为 driver 的唯一运行事实：record 同时保存实际
`Context`、definition 和一次 `uuid4().hex` registration identity；按 `driver_id` 的索引只保存
当前可接纳 record。登记 Effect 撤销时只在索引仍指向原 record 时移除可用入口，然后沿原
record 清理 auth attempt；同名新登记不会接管旧 attempt。Models 自身构造一次 instance UUID，
execution、embedding 和实际 descriptor/binding 路径以实际选中的 `(driver_id, registration UUID)` 集合
生成稳定 namespace；settings/discovery 只使用选中的 owner scope，不凭空生成 runtime namespace。
`plugin_snapshot_id` 仍是 wire 字段但不再来自 RuntimeCatalog 或全局 revision。

Models 的 public async 入口取得自己的 `Context.runtime_scope()`，再固定本次实际 driver scope；
一次调用的 connections、registrations、settings、role 和 capabilities 在首个 driver await 前固定。
enabled connection 的 `open+close` 旧 seal 预检不再在整 Root 启动时扫描，改在真实首用或显式
settings probe 边界执行；config 可读性和 vision binding 仍由 Models 的本地 store/选择边界校验。
同步 catalog、describe、selection 与 contributors 读取当前唯一 registration，不引入 sync lease。

Auth attempt 固定原 registration。starting attempt 在 start await 前登记并以 `state=None` 表示
provider 尚未返回；finish/cancel/probe 继续使用原 record，registration Effect cleanup 负责卸载后
的 cancel retry，失败保留 attempt 和 Effect。Models settings 不再暴露候选设置 source，也不再有
第二个 settings store；`plugin_update` 直接使用自己的 validation boundary。

本批保留 ModelsStore revision/CAS、credential、descriptor/schema、Message、model_calls、continuation
及 embedding identity/缓存键。DSH `c389f96b` 这里只作 Cordis Fiber/Effect/Loader 局部生命周期对照，
不作为模型绑定或 auth 清理保证。M2 生产源码与测试源待 review；本批只做 AST/compile、diff/hash
和 `git diff --check` 静态检查，不声称测试或运行验收通过。

### T-59017c · Shell 单 owner 与 Tools drain

本批的 cleanup 入口在 Shell 注册时绑定原始 `Context`、`ShellOwners` 和 Shell 的
`PluginTasks` admission；`run_reply` 不再把 caller Context 传入。清理 Task 归 Shell
owner，原 Reply Task 只转交已有 `ExternalRootPermit`，Task 完成后释放一次；普通
cancel/pause 由 `shield` 等待真实清理，只有明确 abandon 才让 caller waiter 离开；进程
终止失败保留 owner 并写入 Shell incident。清理阶段对历史
binding 只执行 `BINDINGS.describe`，不重新 `open` Tools binding。

```text
Reply Task
    │ cleanup(reader, source, from_seq, task, drain)
    ▼
Shell TOOL_CLEANUP ──固定 Shell Context/Owners/TaskAdmission──┐
    ├─ Tools.drain_calls ──注入的 Tools admission──> 旧 Effect Task.join()
    └─ BINDINGS.describe ──> ShellOwners.release_tool ──> PluginProcesses
```

Models nested embedding 在 chat execution 内固定 revision 6，外部新 binding 读取
revision 8；只比较 `plugin_snapshot_id`、`identity` 和 `dimensions` 等稳定字段。上述
生产静态基线与测试源只做 AST/compile、diff/hash 和 `git diff --check` 静态检查；
T-84dd9a 已补 Models full descriptor equality、Tools unload/closed-admission oracle，以及
真实 local Root 的 Shell C1–C4 测试源。主审指出 Tools/Shell 源中的卸载屏障顺序、公共入口
oracle、真实消息 ID 和 C4 Task 结算仍有静态问题；T-5d8705 已按这些问题修正，测试仍未
执行，仍待主审复核，不能写成行为验收通过。

### T04-A-R3/B2 当前静态证据（待主审）

- 失败路径为：`_reconcile_changed` → `_update_live_generation` → selection CAS → 旧 generation/Fiber owner 排空 → `_start_local_generation` → `_dispose_generation` 结算；显式 `retry_runtime_recovery` 只读已选 `archive_ref`，先清理 active/draining owner，再创建新的 generation/Scope，不复活已关闭实例、不回写 A。
- 局部就绪检查目标 generation、边移除前捕获且仍登记的实际下游 Fiber，并按当前 Root 的 provider 身份和声明依赖补查仍在图中的消费者；即使消费者失败后清空 `dependency_store`，也不丢失当前硬边。健康项按真实 `entry.owner` 检查，不按 plugin id 或 receipt 字符串猜测；绑定闭包按每个真实 Context/Fiber 的冻结 provider 依赖展开，Core provider 不成为归档组件。
- Fiber `_load` 在 apply/本 Fiber 生命周期/required health 完成后检查当前 Task 的取消状态；Manager 在清理后、mount 后和新 retry load 前复核 operation 提交权，因此吞掉 `CancelledError` 的插件不能迟到发布为 ACTIVE。对应测试源码覆盖 compile 失败、B 启动失败后的同归档显式 retry、A/B 清理失败、无关 optional child 与必需消费者、真实 Manager BINDINGS、同代多 child 的不同依赖、foreign/stale Context、同代 archive component 去重，以及真实 local update/revoke/物理结算/显式 retry。
- `RuntimeCatalog` 现在只从正式 live Root 的当前 Fiber、health、incident 与 Manager `_active_generations` 投影；Manager capability 要求同一 Root、当前 Task 的实际 OwnerCall 或精确生命周期借用，以及 Fiber 明确声明 `RUNTIME_CATALOG`。保留 `snapshot_id` 线字段，但它只是 `root.generation_id` 与现有 composition revision 的显示身份，不是 lease、freeze 或第二生命周期；stable_view 通过 `COMMANDS + RUNTIME_CATALOG` 的自身 scope 读取，MCP 不可用时保留插件树并由 runtime_inspection adapter 转回既有错误合同。
- B2 测试源码改为真实 Manager/probe Context 与声明依赖，覆盖 live replacement 的新 Fiber/activation/health、旧 child 消失、无关 generation、读 catalog 不 load/freeze，以及无 permit/未声明/foreign/旧代与同 Fiber stale scope；stable_view 增加真实 Manager + Commands + live catalog，仍保留 formatter 的窄单元夹具。IncidentView 保存真实 Fiber ID，receipt 的 `incident_counts` 仍是按 owner 名的兼容聚合；本轮还修正 `Context.require_runtime_owner` 的过宽声明门闸，并把 Manager catalog 声明检查留在能力边界；未迁移的 Models、其它 provider、RPC、scope 外 admission/manager 消费者、UI/current_snapshot/fence、公开入口和候选路径仍是 WIP。
- Models M1-R1 的生产 scope/Effect/discover 修复已通过主审和独立静态 review；既有 Models M2 定点测试源仍未执行、待主审复核。T-59017c 仅修正 nested embedding 的 6/8 revision oracle，不把它写成运行验收通过。`StoredSnapshot.revision` 仍是 SQLite 设置/CAS 真相；不把 RuntimeCatalog 的显示 `snapshot_id` 当成 `plugin_snapshot_id`、model revision、binding/embedding identity 或持久化指针，不改 schema、descriptor、凭据、Message 或 durable data。
- 本批仅做 AST/compile、diff/hash、`git diff --check` 静态检查；最终 scope diff、文件 SHA、恢复点和保护 SHA 写入工单回执。T05 公开入口、T06/T07 清理以及运行验收仍未完成，不据此声称系统可启动、可部署或已通过 Gate。

## 1. 从插件的职责出发

### 1.1 我们要留下什么

插件应该能独立安装、提供服务、依赖其他服务，并在退出时清理自己登记的资源。
插件系统不需要知道 Fitbit、记忆、模型各自怎样处理业务。

| 责任 | 唯一 owner | 不顺带承担什么 |
|---|---|---|
| 下载、构建和固定代码及环境 | 安装链 | 不运行一套完整候选应用来决定业务是否正确 |
| 挂载插件、解析服务依赖、管理 Fiber/Effect | 组合内核 | 不枚举工具、MCP、UI 等业务能力类别 |
| 将一次安装应用到当前图 | 宿主中的窄加载入口 | 不持有 stable/latest 两套运行图，不做业务数据回滚 |
| 工具、UI、MCP、进程等能力 | 普通 provider 插件 | 不再以“整图封存”作为动态登记的前提 |
| 配置含义、业务正确性、数据及迁移 | 相应插件 | 不交给 Core 通过版本号、阶段名或统一 recover API 猜测 |
| 真实进程、连接和资源退出 | 原 Scope/provider/宿主 | 不用内存指针恢复冒充资源或外部效果已经恢复 |

保留 `apply(ctx)`、`ctx.config`、`inject`、`provide`、`effect`。
不要求业务插件新增 prepare/commit/rollback 三阶段接口。
历史 Message、binding 和资源回执仍是事实，不因换代码而删除或重跑。

“主要改插件系统”不等于“一行插件代码也不改”。
Tools、UI、模型及资源 provider 中与全局 snapshot 绑定的部分要迁移；
普通业务插件不应为了这次重构重新实现自己的升级流程。

### 1.2 今天为什么复杂

当前机制各自有来由，问题在于把一次局部更新变成了整套应用的替换：

| 上游需求 | 今天的机制 | 下游成本与问题 |
|---|---|---|
| 一次工作使用稳定能力 | Turn/job 等持有整个 runtime snapshot 的 lease | Fitbit 更新也会等与它无关的长工作 |
| 更新不能混用运行实例 | Root 装配后冻结；新版本使用全新 Root | 未变化插件也重新 import、apply、start |
| 新代码先验证 | latest 候选、隔离调用、结果、撤销与晋升 | 同一版本先建候选，再重建正式实例，还要处理两者清理 |
| 发布要有完整结果 | Manager、SnapshotStore、selection、journal 配合推进 | 选择、初始化、接纳和失败恢复交织，状态分支增多 |
| UI 等目录稳定 | 部分 provider 在整图发布时封存注册 | 后续插入一个插件也无法只更新它的登记 |
| 出错仍能恢复服务 | 新正式实例失败后重建旧组合 | 恢复代码并不能撤销新代码已经写入的数据 |

具体证据：

- [Fiber._reconcile / CompositionRoot.freeze](../../agent/plugin_composition/context.py)：628、863 行起，冻结后不再进行正常局部重装配。
- [RuntimeSnapshotCompiler / RuntimeSnapshotStore](../../agent/plugins/snapshot.py)：93、1078 行起，编译冻结 Root，并限制跨 snapshot 共享实际实例。
- [PluginManager._switch_ready / _replace_formal_root](../../agent/plugins/manager.py)：1578、1671 行起，关闭候选并替换完整正式 Root。
- [fresh-root 测试](../../tests/test_plugin_fresh_root.py)：124 行起，明确要求候选和正式组合中的每个插件都是新实例。
- [Ui.register / seal](../../plugins/ui/plugin.py)：52、112 行起，一次性封存后不再允许登记。
- 旧发布等待测试已退役；[现行更新调用排空](../../tests/test_plugin_hot_reload.py)与[卸载期限及清理 owner](../../tests/test_plugin_uninstall_root_drain.py)保护有限排空，旧行为及替代边界见[§6.4 验收映射](#64-t01-验收映射本-worktree-副本补充)。

[Issue #750](https://github.com/kachofugetsu09/akashic-agent/issues/750) 报告了发布长时间等待、
整组装载和 Akasha 重放拖慢可用性等现象。这些生产现象来自 issue，并非本次重新测量。
源码中 Akasha 启动等待 rebuild 和首轮 consume，见 [Akasha](../../plugins/akasha/plugin.py)：488～539 行。
更新 Fitbit 不应触发它的启动链；不必先改造 Akasha 的业务，才能解决这个问题。

根因不是“所有 owner 都太多”，而是**把局部代码替换提升成了全局运行事务**。
拓扑、业务数据、实际资源各有 owner 是必要分工；多套运行图与多阶段晋升不是本次目标所需。

## 2. 目标：一张运行图，只替换受影响部分

### 2.1 “不重建全局图”的硬合同

普通安装、更新、禁用和卸载必须满足：

1. 当前 CompositionRoot 对象和 boot 身份不变，不调用完整 Root 构建或 Gateway 重启作为兜底。
2. 不为验证新插件另外装配一张完整候选图。
3. 未受影响插件的模块对象、Fiber、服务对象、任务和外部进程保持原样；不重新执行它们的 apply/start/stop。
4. 只处理变更插件、它拥有的子 Fiber，以及因实际依赖变化而必须启停的消费者。
5. 不等待全局 lease 清零，不暂停所有来源，不重新广播所有插件的启动回调。
6. 可以扫描依赖登记、保存完整的“已选择输入”记录、重新生成只读目录。它们不是重建运行图；第一版不为这些扫描另建复杂的增量缓存。

这是待实现并验收的保证，不是当前代码已经做到的事实。
冷启动本来就需要首次建图；修改 Core/解释器也属于独立部署，不混进普通插件安装。

### 2.2 Fitbit 更新时究竟动谁

~~~text
                       ┌ Tools / MCP / Timers 等公共 provider ┐
                       └──────────────┬──────────────────────┘
                                      │ 被 Fitbit 使用
                                      ▼
                       ┌ Fitbit 旧实例 → Fitbit 新实例 ┐
                       └──────────────┬─────────────────┘
                                      │ 提供某个服务
                                      ▼
                       ┌ 依赖该服务的消费者及子作用域 ┐
                       └───────────────────────────────┘

┌ 模型、Akasha、Shell 等无关分支 ┐
│ 原对象、原任务、原资源继续运行 │
└───────────────────────────────┘
~~~

图中的“无关”以实际依赖为准，不按插件名字硬编码。

- Fitbit 使用 Tools，不代表更新 Fitbit 要重启 Tools。只撤销、重建 Fitbit 在 Tools 中的登记。
- 消费者把 Fitbit 服务作为硬依赖时，先停止消费者，再停止 Fitbit；新服务可用后再恢复消费者。
- 可选功能放在嵌套 Fiber 中，只启停该子作用域。不能为缩小影响范围，偷偷把真实硬依赖改成可选依赖。
- 新增插件也可能让原来等待依赖的 Fiber 变为可用，必须处理这类激活。
- 更新一个所有插件都依赖的基础服务，影响范围可能很大。仍不新建 Root，但不能承诺它只影响一个节点。
- 同一个事件的监听者不是彼此的硬依赖，不能因为它们订阅同名事件就全部重启。

DSH 对照：D01、D02。它扫描相关服务依赖并按 provider 身份变化启停 Fiber，不要求另建完整图。

### 2.3 不再把整个 Turn 固定到旧图

保留的是“正在执行的实际调用不能失去资源”，不是“整轮对话占有所有插件”。

Tools、RPC、MCP、Dashboard 等在实际调用边界取得目标 owner 的工作许可。
停止该 owner 前关闭它的新调用接纳，等待已开始的调用及清理；无关 owner 不受影响。
闲置的工具目录、长期 listener 或仅持有历史 binding，不占全图许可。

一个消费者的本次 activation 仍固定自己的实际依赖。
不把每次方法调用都偷偷转到全局最新对象，也不靠动态代理掩盖旧引用失效。
已有模型执行可以继续固定其所选模型和必要 provider；它不应因此占住 Fitbit。
旧 ToolRef 失效时明确返回“未执行/引用失效”，后续通过正常能力 view 取得新引用，不重放未知效果。

这个变化需要调整来源、调用边界和公共 provider，不能直接删除 lease 计数。
没有受管理的生命周期、偷偷保存别的插件对象或在 import 中启动无主线程的代码，
不能仅靠一张拓扑图保证安全热更新；T03 要盘点真实消费者并修正这类越界。

## 3. 安装、检查、生效和失败

### 3.1 三种检查不混为一谈

| 检查 | 在哪里做 | 能证明什么 |
|---|---|---|
| 构建、语法编译、包内容和环境准备 | 安装准备阶段 | 制品可构建，所需文件存在，已检查的代码语法合法 |
| 入口、配置、硬依赖、冲突、初始化、必需资源就绪 | 局部加载阶段 | 该插件及必需能力在当前环境实际可用 |
| 业务行为是否符合用户要求 | 普通测试与 Agent 的正常调用 | 已测场景符合预期，不承诺所有未来输入都正确 |

Python 编译通过不能证明 import、依赖连接或所有业务路径成功。
能提前检查的错误在停旧插件之前拒绝；动态声明与正式资源只在真实加载时判定。
不为这些检查运行一张完整候选图，也不把正式 import 说成无副作用预检。

因此，“Fitbit 能用，但把最近七天误算成最近三十天”允许通过安装与激活。
这属于业务 bug，由 Agent 修改、测试并再次安装；不需要底座维护验证 Agent 或晋升协议。

### 3.2 一条安装路径，不分内部和外部特权

~~~text
Agent Shell / 普通安装工具 / 外部 CLI
                    │
                    ▼
┌ 准备固定新制品 → 编译与可提前完成的检查 ┐
└───────────────────┬────────────────────┘
                    ▼
┌ 宿主持锁，原子保存新的已选择插件输入 ┐
│ 返回“已接收应用请求”，不是“已经可用” │
└───────────────────┬──────────────────┘
                    ▼
┌ 关闭受影响分支的新工作 → 等待实际在途调用 ┐
│ → 逆依赖关闭旧节点 → 加载新节点          │
└───────────────────┬─────────────────────┘
                    ▼
┌ 新节点就绪，局部开放；或明确报告该分支失败 ┐
└─────────────────────────────────────────┘
~~~

推荐继续复用现有 [PluginSelection](../../agent/plugins/selection.py) 的原子写入与并发基线检查：
它保存“用户已选择安装哪些固定插件输入”，不再表示“整图已验证成功”。
旧文件名不要求为了改名立即迁移；是否升级格式由 T01/T04 核对读取者后决定。
完整输入记录只是代码引用列表，不是第二张运行图。

先保存选择，再执行会触碰正式资源的加载，失败不自动写回旧版本。
选择与运行状态明确分开：安装目标可以是 B，但当前处于停止 A、启动 B 或 B 失败。
状态查询直接投影 Loader/Fiber 事实，不另造一个能决定版本的状态库。

内部 Shell 发起安装不能等待自己所在调用被释放后才返回。
宿主接管应用任务，安装入口在自己的调用 scope 退出前不等待完成；外部入口使用同一协议。
发起者随后查询结果即可，不等待整个父 Turn 结束，也不由模型回复中的“批准”决定生效。
控制入口不能依赖正在被替换插件继续运行，且不把管理权限开放给任意网络调用。

应用操作第一版串行，现有 workspace/plugin-home 排他边界保留。
不新增 Supervisor 发布协议、多 workspace 并发安装或分布式更新协调器。
离线安装在取得维护锁后更新同一选择，但只能报告“已安装、未启动”，不能报告 active。

### 3.3 失败和取消的简单规则

| 失败位置 | 结果 |
|---|---|
| 下载、构建、编译或可提前完成的检查失败 | 不改变选择和当前图；返回具体错误 |
| 提交选择发生冲突或明确失败 | 不开始局部换代；返回错误，不覆盖其他安装 |
| 选择写入结果不确定 | 停止本次继续执行并报告不确定，不回写旧值；核对原有存储证据 |
| 已选中 B，旧分支排空超时 | B 尚不可用；该分支关闭新接纳，实际在途工作和清理仍有 owner；无关服务继续 |
| 旧资源释放失败 | 不启动会冲突的新资源；保留句柄、依赖和失败原因，不扩成全局停服 |
| B import/apply/必要启动或资源就绪失败 | 清理 B 已取得的资源；B 标为失败，硬依赖消费者不可用；不自动启动 A |
| B 启动成功，后来发现业务结果错误 | 按业务错误处理，修改代码并重新安装；系统不擅自退回 A |
| 进程在选择提交后退出 | 下次正常启动读取 B；不续跑候选晋升，不猜测 A 安全，不新增启动令牌制度 |

应用任务有有限等待时间。超时只终结等待与报告，不证明任务、进程或外部效果已消失。
取消请求在选择提交前可以停止安装；提交后不能当作“撤销成功”。
旧资源尚未清理完时，后续变更可明确拒绝，不为此引入并发发布与跨操作接管机制。

不提供专门 rollback 命令。用户以后显式安装旧版本也是一次普通安装，
仍由该版本解释现有数据，不享有数据恢复保证。
保留普通服务重连、资源清理重试和现行宿主监督；本设计不顺带改变 RUN-004/RUN-015 的宿主策略。

### 3.4 “不回退”为什么没有要求复制整个 HMR

采用 DSH 的 Loader/Fiber 思路，不逐行移植其 TypeScript 实现：

- 借鉴插件条目的局部替换、依赖变化传播、Effect 归属与真实初始化等待。
- 不采用 Loader 中“新实例失败后重新启动旧实例”的分支。
- 不采用 HMR 的源码监视、ESM/CJS 缓存恢复及框架变化后的全量退出兜底。
- 明确安装已经给出变更插件及固定版本，无需再从任意源码文件变化推断重载目标。
- DSH 的清理错误处理不能覆盖本项目 PLG-006：失败必须保留实际资源 owner，不只打日志就继续装新资源。

仅参考完整 HMR 的“识别变更插件后局部处理”这一边界。
第一版不支持“保存任意源码即生效”；将来增加开发入口时，也必须调用同一个局部加载入口。
DSH 本身区分 startup/live profile，不用它来证明任意运行模式都支持无感更新。

## 4. 改动边界与明确取舍

| 保留 | 改造 | 退出 |
|---|---|---|
| 服务键、硬依赖、嵌套 Fiber、Effect | 冻结整图改为局部依赖变化 | stable/latest 双运行图 |
| 固定制品、独立模块命名空间、环境身份 | Manager 收敛为加载及安装应用入口 | 候选运行、验证晋升、撤销晋升 |
| 实际调用与资源借用 | 全图 lease 改为实际 owner 的调用保护 | 全局排空与新建正式替身 Root |
| 普通 provider、配置和数据 owner | 注册表随 owner 生命周期增减 | 自动重建旧组合 |
| 原子安装选择与历史来源 | 状态投影、UI 目录及 source 接纳局部更新 | 为上述流程服务的 publication/journal 分支 |
| Supervisor/Guardian/Controller 的真实资源责任 | 只调整与插件局部生命周期耦合的边界 | 第三版新增 Supervisor 发布控制面的计划 |

接受局部短时不可用，不承诺无缝切流、保留所有连接或让所有旧请求成功。
同进程插件若阻塞事件循环、崩溃进程或越过托管接口，拓扑不能提供进程级故障隔离。
不为此新增逐插件沙箱；也不能把正常异步初始化异常升级为整进程退出。

Python 的插件私有模块可以按固定版本重新加载。
会替换已加载的进程级共享库、解释器或不可重入原生组件的变更，必须明确报告不支持在线应用，
不能通过清空整个 sys.modules、修改共享环境或自动重启偷偷完成。
安装前能识别的这类变更应在改变选择前拒绝；识别范围由 T04 用实际环境加载实现验证。

## 5. 七个实施任务

这是责任和依赖拆分，不按旧 PR 编号安排，也不预估未经验证的删行收益。
每步都先迁真实消费者，再删旧入口；“删掉了用户曾经使用的功能”不算无语义变化的冗余删除。

~~~text
T01 合同与验收基线
          │
          ▼
T02 局部 Fiber 生命周期
          │
          ▼
T03 实际调用与公共 provider
          │
          ▼
T04 单插件加载与安装应用
          │
          ▼
T05 切换公开入口与状态查询
          │
          ▼
T06 删除旧发布系统
          │
          ▼
T07 显式迁移与累计验收
~~~

T02～T04 可先在隔离夹具中实现并验证；不能在旧 snapshot writer 仍控制生产图时，
只去掉 freeze 就开放局部修改。T05 是一次权威路径切换，最终不保留两套可选择的运行模型。
仓库实现、运行验收、外部插件迁移与正式部署分别取得所需授权；任务拆分不等于已授权部署。

### T01 · 对账合同，先写“不重建”的验收条件

**输入与下游：** 维护者确认的新方向 → 明确取代哪些旧承诺，给 T02～T07 同一套判断标准。

**范围：** projectneed 第 10 节、相关 RUN 条款、0071 的后续决策、NOW 中的旧晋升任务，
以及现有插件组合和安装边界测试。

**怎么做：**

1. 新决策取代 0071 中“完整重建、冻结服务绑定、候选晋升”的部分，保留 0070 数据归属和 PLG-006 清理责任。
2. 修订 PLG-001～005、007～009、012～014、016、018 中的候选/整图/晋升语义；PLG-010 保留卸载不删数据，改为局部执行。
3. RUN-007/009 去掉对整图 snapshot 的依赖；保留模型执行自己的固定配置及实际 provider 保护。RUN-016 删除候选与自动恢复旧 Workload 的更新承诺，保留 Controller 权限和真实退出责任。
4. 对账 PLG-011/015 的 generation 身份、诊断和查询：它们不因去掉 snapshot 就消失。RUN-010 动态设置、RUN-015 Core/Bridge 部署不归入本次插件安装。
5. 定义“变更插件 + 一个硬消费者 + 一个可选子 Fiber + 一个无关长任务”的最小验收拓扑，记录 Root、模块、Fiber、服务、任务及资源身份。

**删除性质与 DSH：** 这是合同修订，不是纯冗余删除。目标行为参考 D01～D04；数据与宿主边界来自本项目，不伪称 DSH 提供。

**退出条件：** 每条改变的现行承诺有新表述；新验收能识别旧实现的全量重建，旧功能测试不得仅改断言掩盖未迁移消费者。

### T02 · 让现有组合内核支持局部启停

**输入与下游：** 插件挂载/服务变更 → 内核只启停受影响 Fiber，供 Loader 使用。

**主要文件：** [context.py](../../agent/plugin_composition/context.py)、[effect.py](../../agent/plugin_composition/effect.py)、
[events.py](../../agent/plugin_composition/events.py)；现有组合 lifecycle 测试。

**怎么做：**

1. 复用 Fiber._reconcile、provider epoch 和 _reconcile_dependents；移除“发布后必须冻结，依赖变化只允许 dispose”的限制。
2. 服务变化沿实际硬依赖传播；保留每次 activation 的 dependency_store，不引入第二套全量拓扑模型或全局最新服务代理。
3. 把每个受管理 owner 的接纳/实际调用保护接入现有作用域。局部停止先拒绝新调用，再等待本分支实际使用者；不读取全局 lease 归零作为条件。
4. 局部 apply、必要启动与 required health 完成后，才将新 owner 的能力视为可用并恢复消费者。LOADING 中的半成品登记不能被普通调用使用。
5. 生命周期启动/停止只发给本次变化的 owner；启动传播按依赖关系，停止按反向依赖。保留普通事件原有顺序和错误语义，不把业务事件改成依赖调度器。
6. 清理失败保持原 Fiber、句柄和必要依赖；不可用状态与错误可以查询，不把一条支路的普通失败变成 Root.dispose。

**保留/删除：** 保留抗取消、逆序清理、失败 owner 和重入检查；替换 whole-Root freeze 限制。
不删除只是名称含 snapshot 的依赖绑定局部值。

**DSH 对照：** D01 的 _refresh/_setEpoch/_reload、D02 的 provide/notify。局部接纳与失败保留沿用 Akashic 既有责任，DSH 不直接证明该部分完成。

**退出条件：** 依赖失去/恢复、替换、新增 provider、缺依赖、冲突、环、子 Fiber 和清理失败有确定结果；
无关 Fiber 的 apply/start/stop 次数均不增加，Root 身份不变。

#### T02-N1-R2 · 单 registration 撤销与 caller 自等待合同（T-a2817b 整批未通过；T-a510da 生产与 T-518ba4 R2 测试源静态接受；测试未运行）

主审否决 T-1040ce 的“先删除旧 registration、再允许新 provide 交错”和“notify 显式 retry”设计。
T-a2817b 的 N1 实现方向可保留，但整批 production/test review 未通过：`_service_provider` 的
UNLOADING 错误合同与 Effect cleanup 子 Task 的 lifecycle-borrow 自等待仍漏检，且两处新增测试把
cleanup 后置 Event 当成排空前屏障。T-a510da 只修正这两个生产点和 T1-T6 的真实测试时序/失败收尾；其
P1/P2 生产修复与 T-518ba4 的 R2 测试源已由主审与独立只读 review 静态接受，行为测试未运行。N1
本身不拥有 ACTIVE late provide/reprovide 通知；该 N2 边界已由 T-2f3d82 实现并静态接受。不新增第二张
运行图、retired registry、通知队列、跨 Task 借用或恢复 API。
`_Provider.revision`、`Effect`、Fiber activation、`dependency_store`、现有 transition/reconcile 和
`_active_provider` 仍是唯一事实来源；`receipt()`/`topology_view()` 继续从这些事实投影。

**事实 owner 与两条轴：**

1. `root._providers[key]` 只有一个“当前登记”事实，负责冲突检查、当前值和新调用是否可接纳；
   当前登记的 `_Provider` 对象及其 `revision` 是一次注册的精确身份。提供者 Fiber 的 registration
   `Effect` 持有撤销责任；Effect 关闭成功前，资源责任不从该 Fiber 消失。
2. 消费者 activation 的 `dependency_store[key]` 持有它已经接受的那一个 `_Provider` 身份；其
   `OwnerCall`/runtime scope 持有物理在途调用直至释放。因而“当前可以接纳新调用”和“旧调用/旧注册仍
   需要物理排空”是两个正交事实：新登记不改写旧消费者的绑定，旧排空也不能删除新登记。
3. `Context.get`、`_active_provider`、`_service_provider`、`bindings.py:103` 的 frozen binding、
   `binding_contributors`/`plugin_service_owners` 与 `manager.py:969` 的消费者投影必须各自继续使用
   当前登记或 activation 已冻结的登记；它们不能把 `key` 当成跨代句柄。`_service_provider` 的原始
   当前表读取只表示登记投影，不等于 ACTIVE 接纳：非 `revoking` 的 LOADING/UNLOADING owner 仍
   返回登记 Context，实际 `runtime_scope`/`_begin_call` 再给出 `OWNER_UNAVAILABLE`；只有
   `revoking` 才在 raw lookup 处要求精确 target permit 或 lifecycle borrow。

**N1 正常撤销路径：**

1. `_register_provider` 保持同步并返回刚创建的精确 `_Provider`；`Context.provide` 的 setup 只做同步
   登记并返回 cleanup。cleanup 捕获该对象，不只捕获 `(key, owner)`；LOADING 只登记，ACTIVE 成功边界
   由 N2 对精确 registration 做一次通知。
2. 原 caller 的 close guard 先同步扫描实际 `_reconcile_dependents` direct wait set，再扩展该 Fiber 的
   children/descendants 与其所供硬消费者。guard 识别当前 Task 的实际 `OwnerCall`、`_transition_owner`
   和 exact `(Context, Task)` lifecycle borrow；没有自等待后，才把同一 record 标为 `revoking`，使
   fresh `_active_provider`/dependency snapshot/跨 owner 查找不可用；record 仍留在唯一 `_providers` 表。
3. 现有 gather/排空完成且没有登记 Fiber 的 `dependency_store` 仍持 exact record 后，才按对象身份删除
   表项并让 Effect 成功移除 owner。LOADING 仍沿既有 owner ACTIVE 入口；ACTIVE late notify 由 N2 在
   registration Effect 成功建立后负责。

**单 registration 责任与失败保留：**

1. 旧 record 在整个排空和失败期间保持当前表项；`revoking=True` 只撤销新的 fresh 查找，不是 owner
   Fiber 的 ACTIVE 标记，也不是第二份状态表。任何相同 key 的新 provide（同 owner 或别 owner）都继续
   `DUPLICATE_SERVICE`，不允许旧失败和新登记并行。
2. 消费者 activation 的 `dependency_store[key]` 仍保留已接受的 exact `_Provider`，所以合法在途调用和
   cleanup 可读 frozen 旧值；新的普通查找不可用。只有所有真实持有者完成后才物理删除该同一对象，
   `revision` 不因 retry 改写；Effect.closed 后的重复 `aclose` 幂等。
3. consumer reconcile、cleanup 或取消失败时，`revoking` record、同一 Effect、owner、Fiber.error 和
   原始异常继续保留；retry 只重试原 registration close，不提供不存在的 notify retry，也不回滚旧 Root、
   replay 业务效果或自动恢复旧组合。清理失败不改变其它 key、peer Fiber 或 Root 身份。

**自等待与同步拒绝：**

1. 每次非 closed service registration `Effect.aclose` 都在创建或 join `_close_task` 前、实际 caller
   Task 上执行 Core-private 同步 guard；已有安全 close task 正在排空时，后来的 joiner 也必须先过 guard。
   普通 Effect 不设置该 guard，既有 setup 重入和 cleanup 自等待语义不变。
2. guard 只即时扫描真实等待集合：direct seed 与当前 `_reconcile_dependents` 一致，再递归纳入 owned
   children/descendants、其 registered provider 的声明硬消费者和 frozen dependency_store 责任；不建
   反向缓存、不把无关 provider 子树或全 Root 算入。当前 Task 持受影响 `OwnerCall` 时复用
   `REENTRANT_CALL_WAIT`，持 `_transition_owner` 或 exact lifecycle borrow 时复用
   `REENTRANT_LIFECYCLE_WAIT`；只按绑定里的 Context 对象和 Task 身份匹配，不把继承的 ContextVar 当授权。
3. 命中任一自等待必须**变更前拒绝**：不置 `revoking`、不改表、不改 composition revision、不删表项，
   原 Effect/record/消费者绑定完整保留。caller 释放 permit 后显式重试原 `registration.aclose`；N1
   不声称 Fiber.dispose、Root.dispose 或用户自行 create_task 形成的所有跨 owner 循环都已闭合。

**T-a510da R1 定点修正与测试收尾（生产与测试源静态接受，测试未运行）：**

1. P1 删除 `_service_provider` 对非 `revoking` owner 的 ACTIVE 预检；LOADING/UNLOADING 的实际新
   admission 继续由返回的 Context、`runtime_scope` 和 `_begin_call` 负责，保持 `OWNER_UNAVAILABLE`。
2. P2 在同一临时 wait-set 中检查 exact lifecycle borrow；Effect cleanup 子 Task 借用受影响 Fiber
   的 Context 时变更前抛 `REENTRANT_LIFECYCLE_WAIT`，不依赖 ContextVar 继承，也不增加 owner/task 表。
3. T1 使用真实 `RuntimeScope.wait_admission_closed()` 作为排空前屏障，并在释放后验证 frozen read、
   exact record 删除和无关 peer 的新 scope/identity 不变；T2 为本轮并发/失败测试补 gate、permit、task、Root
   的 finally 结算；T3 核对 first/second caller 与真实 cleanup Task 身份并重复取消；T4 覆盖已有 close
   task 的真实 joiner guard；T5 用 entered RuntimeScope 的原生 child 验证 ContextVar 不传递授权；T6
   分别覆盖 consumer.dispose 与独立 consumer Effect.aclose 的 exact lifecycle-borrow 自等待，以及同 owner
   另一 key 的可用性。以上仅写测试源，未运行。

**T-518ba4 R2 四项测试源定点修订（主审与独立只读静态接受，测试未运行）：**

1. A 修正 composition guard 测试的确定性顺序：先确认 first/second caller 与唯一 cleanup Task，向 first
   分两次投递取消并在释放 gate 前确认两者仍等待同一 cleanup、owner 与 Effect 仍在；释放后再分别确认
   first 的 `CancelledError`、second 的成功结算、cleanup 只执行一次且 owner 移除。
2. B 修正 revoking raw lookup fixture：由当前 Task 从 consumer Fiber 取得真实 `OwnerCall`，进入
   `RuntimeScope(call)` 后等待 admission closed；不把 `Context.runtime_scope()` 的裸 `yield None` 当作 scope。
3. C 让新 Task 先调用 `root._service_provider(SVC)` 核对精确旧 Context/value，再对该返回 Context 进入
   `runtime_scope` 并断言 `OWNER_UNAVAILABLE`；不把父 Task 已解析的 Context 当作 P1 lookup oracle。
4. D 在同一原 provider Context/Fiber/activation 下验证失败后仍 `DUPLICATE_SERVICE`、临时失败 Effect
   不遗留；原 Effect retry 成功后以同一 owner reprovide 新值、新 revision、新 Effect，并确认已 closed
   旧 Effect 的重复 close 不删除或撤销新 registration。

T-518ba4 的 R2 只修订上述四项测试源，已由主审与独立只读 review 静态接受；T-2f3d82 的 N2 另改
`context.py`、本地生命周期测试源与三处状态文档。所有行为测试、Gate、CI、应用 import 和运行验收仍未
执行。其它 provider/RPC、scope 外消费者与最终 enable 仍为 WIP。

#### T02-N2 · ACTIVE late provide/reprovide 通知（T-2f3d82 生产与 T-cbecef 测试/文档静态接受；行为测试未运行）

N2 补上同一 Root 内“provider 已 ACTIVE、consumer 仍 PENDING”的最小通知闭环，不引入第二个 registry、
通知队列、全局 gate、ContextVar 授权或自动恢复。它只在 registration Effect 已经成功建立 owner hold 后运行：

```text
ACTIVE Context.provide
    ├─ frozen/duplicate + exact self-wait guard（同步、变更前）
    ├─ _register_provider → add_effect 成功（registration/Effect 保留）
    └─ physical notify task → exact live registration → affected dependents reconcile
```

```text
registration Effect owner
└─ 本 key 的真实消费者协调
   ├─ 坏 C → C 与实际必需下游 D 不可用，原错误/资源 owner 保留
   ├─ 健康 P、sibling S、无关 peer U → 继续 scope/业务读取
   └─ 其它 key 登记不自动重试坏支路；显式原 owner cleanup 才 retry
```

登记成功、consumer ready 与失败 owner 保留是三个独立事实；通知错误或 caller cancellation 不是原子回滚。

1. ACTIVE 新 key 和同一 Context/Fiber/activation 的成功 reprovide 都只通知当前 exact `_Provider`；
   LOADING 的多次 `provide` 只登记，仍由正常 `_owner_became_active` 一次协调全部 key。通知不会调用
   `_owner_became_active`，也不会把 registration success 伪装成 consumer readiness。
2. ACTIVE preflight 先复用 frozen/duplicate 检查，再按 key+owner 的真实 wait set 检查当前 Task 的
   `OwnerCall`、transition owner 和 exact `(Context, Task)` lifecycle borrow；没有 fake `_Provider`、
   第二张 owner/task 表或 inherited ContextVar 授权。同步拒绝不改 provider table、revision、Effect
   或 composition revision。
3. setup 仍同步登记并返回 exact-record cleanup；`add_effect` 成功后才建物理通知 Task。task creation
   失败关闭未开始的 coroutine；`_await_critical` 在 caller cancellation 时仍等待物理通知 settle。
   通知失败不撤销已建立的 registration/Effect。
4. 通知再次确认 exact current registration、`revoking=False`、owner ACTIVE 后，只调用
   `_reconcile_dependents((registration.key,), exclude=owner)`。consumer apply/start/required health
   失败留在真实 consumer Fiber；provider、健康 sibling 与 unrelated branch 继续工作，不自动 retry、
   rollback、root rebuild 或恢复旧 instance。
5. reconcile/cleanup 错误向 providing caller 传播，同时保留 provider registration/Effect、失败
   consumer Fiber、dependency binding 和原始错误；same-key provide 仍 duplicate，显式原 consumer
   dispose/原 registration cleanup 才能重试。取消只取消 caller 的等待，不取消物理通知。

T-2f3d82 的六项测试源原先覆盖 ACTIVE exact key late wake/reprovide 与 identity、不在 RUNTIME_STARTED 前
唤醒的 LOADING 多次登记、duplicate/frozen priority、P→bad consumer→D 的局部失败隔离、cleanup failure
的保留与显式 retry，以及通知期间 caller cancellation 的 physical settle；但 Fiber/Handle API 使用和局部
失败 oracle 不完整，整批未通过。T-cbecef 修复了真实 raw Fiber/Handle 边界、失败错误层级、原 registration
与 Effect 责任、无关分支 scope 和 Task/Root finally 收尾，不改生产等待算法；修订后的测试源与文档已由
主审及独立只读 review 静态接受，仅行为测试未运行。

**失败、取消与显式 retry 的共同时序：**

1. 正常顺序是“原 caller guard → 同 record 标 `revoking`/撤 fresh 查找 → 现有 consumer 协调与物理
   排空 → 检查 exact record 残留 → 对象身份删除表项 → Effect 成功移除 owner”。拒绝发生在第一步，
   cleanup 失败发生在后续步骤；两者都不伪造成功。
2. consumer apply/start/health、reconcile、依赖 drain 或 Effect cleanup 失败时，Fiber.error、同一
   Effect/record、失败 owner 与原始异常保留；`gather` 返回成功但 `dependency_store` 仍持 exact record
   时，按 `DEPENDENT_CLEANUP_PENDING` 失败，先显式完成该 consumer 的真实 dispose，再重试原 close。
3. caller cancellation 沿既有 Effect shield/物理 settle：取消等待不等于清理成功，责任仍由同一 Effect/owner
   持有，重复 close 仍幂等。N1 不把 cancellation 变成后台 stop_task，也不提供新的公开恢复入口。

**下一代码单元的边界与最小 oracle：**

T-a510da 实际范围是 `context.py` 的 P1/P2 两个生产定点与两个指定测试源的 T1-T6 修正；`effect.py`
及 `bindings.py`、Manager 只读核对，不在 R1 修改。`receipt()`/`topology_view()` 继续投影现有表项与
责任。不得改 Message、schema、cursor、receipt wire、descriptor、`root_ref`、metadata、credential、
插件数据、其它 provider/RPC、Loader 或 ACTIVE late-notify。

最小确定性 oracle 为：

- 关闭 ACTIVE owner 的单个 registration 时，consumer 的已接纳调用继续读 frozen 旧值；fresh 查找不可用；
  revoking record/Effect 在 drain 期间仍在唯一表中，释放 permit 后才删除；无关 peer identity/lifecycle
  不变；
- consumer cleanup 真实失败、`_reconcile` 因 `_dispose_requested` 跳过或 caller cancellation 时，原
  record/Effect/Fiber.error/责任保留；same-key provide 继续 duplicate；显式 dispose/close retry 后才
  可删除；已 closed close 幂等；
- 原 caller 持 direct consumer、下游 consumer 或 owned child 的真实 permit 时，close 变更前分别得到
  `REENTRANT_CALL_WAIT`；持真实 transition owner 时得到 `REENTRANT_LIFECYCLE_WAIT`；无关 permit 不误拒；
- frozen `Context.get`、target-own permit/lifecycle 的 `_service_provider` 读取保留，ordinary 新 Task/raw
  child 不能借 ContextVar 取得 revoking 服务；`_active_provider`、provided services 和
  `plugin_service_owners` 不把 revoking 当可用服务；不产生第二份 registration 状态。

该合同记录 T-a2817b 整批 production/test review 未通过；T-a510da 的 P1/P2 生产修复与 T-518ba4 的
四项 R2 测试源已由主审与独立 review 静态接受，测试未执行。T-2f3d82 的 N2 production hook 与
T-cbecef 的六项测试源、文档已由主审及独立 review 静态接受；不能将任一静态结论写成行为验收。

**T02/T03a/T03b-Tools/T03b-Content/T03b-Materials/T03b-Bindings/T03b-Senders 实施状态（2026-09-20，A-R1/B1 生产与测试源及此前 Content fixture 修正已通过主审及独立只读复核；T-ae2487 Content 三项测试修正已通过主审当前 diff/hash/AST 静态复核；Materials 与 Bindings.open 生产已通过主审及独立只读静态复核；本轮 Materials/Bindings.open 测试修正与 T03b-Senders 生产接线及测试源待只读 review；测试未执行）：**

- 已落地：`FiberHandle.acquire_call(expected_activation)` 同步原子校验调用方预期 token 与当前可接纳 activation——不匹配报 `STALE_ACTIVATION`、非 ACTIVE 报 `OWNER_UNAVAILABLE`、Root Fiber 报 `ROOT_CALL_ADMISSION`、无当前 Task 报 `OWNER_CALL_CONTEXT`；`OwnerCall` 仅代表一次已接纳未释放的许可（只能由 acquire_call 返回，已 release 的许可 `__aenter__` 即拒绝、body 不执行）；`_unload` 顺序：自等待检查（在任何状态撤销之前）→撤 token/UNLOADING→依赖方退出→排空本 owner 在途调用（`_await_critical`）→子/Effect 逆序清理；自等待拒绝覆盖两条真实路径——`dispose()` 调用方任务持本 activation 调用时报 `REENTRANT_CALL_WAIT`，`reconcile()` 在 transition 已被持锁且本任务持有本 owner 在途调用时同错（无锁/同 epoch 的 reconcile 不误拒）。
- 既有机制沿用：provider epoch + `_reconcile_dependents` 只唤醒实际硬依赖者；`dependency_store` 冻结本次 activation 绑定；`apply(ctx)` 完成后才置 ACTIVE；FAILED owner 不自动重启旧实例；Effect 逆序抗取消清理、失败保留 owner/句柄并可显式重试。
- **批次 A 已完成——启动/健康屏障与局部 scope 接入（R1 已通过；B/C 仍未完成）：**

  当前实现的事实约束：`ACTIVE` 晚于 apply、本 owner STARTING/STARTED 与 required health；同一 Fiber 可有多次 activation，旧 Context 以对象身份拒绝；`Context.get` 在 LOADING 允许本 owner 已登记的 provide，跨 owner 仍要求 ACTIVE；`EventRegistry._active_listeners` 继续过滤非 ACTIVE，LOADING/UNLOADING 生命周期走按 Fiber 的专用内核 dispatch，不放开普通过滤、不按 plugin_id 广播；`ctx.runtime_scope`（context.py:155-177）已改为本 Fiber/activation 的唯一 `RuntimeScope`，旧 snapshot lease 直构与运行期 fence 仍归批次 B。

  **i. 真实/目标调用链**
  - 迁移前的冷启动历史链：Manager 校验 → `_mount_module`×N → `receipt().ready` → compile+`root.freeze` → closed snapshot → Manager 全图 `RUNTIME_STARTING`/`RUNTIME_STARTED`（manager.py:1399-1462；插件如 akasha plugin.py:527-539 在 start 内进 `ctx.runtime_scope`）。这是仍有真实 snapshot 消费者时保留的 legacy start/stop/cleanup 责任，不是当前 App/stdio 宿主的常驻监督合同。
  - 本批正式宿主链：`core.start/load_all` → 同一 live Root → 每个 Fiber `STARTING/STARTED/health`；宿主不再从 Manager 订阅常驻 snapshot loop。
  - 局部换代（目标）：Loader/安装入口 → `root._mount`/`fiber.dispose` → `_load`：apply 完成 → 本 Fiber 专用 `RUNTIME_STARTING` → `RUNTIME_STARTED`（真实启动代码在这两个按 Fiber 派发的 listener 内，不走全局 listener）→ 本 owner required health 检查 → 置 ACTIVE + `_owner_became_active` 唤醒消费者、放行后台工作。停止反向：`_unload` 撤 token/UNLOADING → 依赖方退出 → 排空在途调用 → 本 Fiber `RUNTIME_STOPPING` → Effect 逆序清理。普通插件仍只写 apply/`ctx.on`/inject，业务插件不改。

  **ii. 唯一执行位置与事实 owner**：每次 activation 的 start/stop 由内核 `_load`/`_unload` 唯一执行；事实 owner 是 activation token（非 Fiber 对象、非 plugin_id），同一 Fiber 重载即新 activation、新 start。Manager/Loader 最终只"请求装载/等待结果"，不广播生命周期；不保留常驻双模式或第二份 started 表。

  **iii. owner startup 的资源访问**：选窄的 owner-self 读取——`Context.get` 在本 Fiber LOADING 时允许读**本 Fiber 自己**已登记的 provide（`owner is self._fiber`），其他 owner 仍要求 ACTIVE；不开全局 fallback、不允许跨 LOADING owner 互读。`ctx.runtime_scope` 的本地化是前置依赖（见 v 的 T03a）：局部 activation 需要不经过整图 snapshot lease 的 scope 入口，接口为"按 Fiber/activation 的 scope 获取"，阶段上先于 start 执行点。

  **iv. 迁移期 seam（已被 T03a-5 合同取代）**：原"局部装载标记 seam"废弃——T03a-5 定为同一不可分割单元内直接切换：内核 `_load`/`_unload` 对每次 activation 统一 dispatch 生命周期（冷启动与局部换代同链），不新增第二套 lifecycle owner。T-57630a 只删除 Manager 的 resident `run_runtime_services` 与 App/stdio 的正式 bootstrap 调用；`start_runtime`、`_start_runtime_snapshot`、`_start_closed_runtime_snapshot`、`_stop_runtime_snapshot*`、snapshot store 及真实消费者需要的清理责任保留。freeze 屏障随 fence/lease 消费者迁移一并解除，不先 no-op。

  **v. 校正后的实施依赖**（最小 Unicode 依赖图）：

  ```
  T02a 拓扑/调用保护（已落地：epoch 传播、acquire_call、排空、自等待拒绝）
    │
    └─→ 接入批次 A→B1→B2→C（T03a-5-R1 合同：A 内核件 → B1 Task 消费者 → B2 其余消费者/fence → C 候选路径移除+启用验收）
              │
              ├─→ T03b provider/工具/RPC 调用面接 acquire_call
              ├─→ T04  Loader 局部安装应用
              └─→ T05  公开入口/状态切换 + 旧 Manager 全图路径退役 → T06 删除
  ```

各独立可审交付：T02a=内核调用保护+本测试文件；接入实现=批次 A（内核件）→B1（Task 消费者）→B2（其余消费者/fence）→C（候选路径移除与启用验收），未闭合批次标 WIP；其余照旧。T02 退出条件不减：ACTIVE 必须晚于 apply+startup+required health。
- 暂留屏障：旧 snapshot lease 仍服务未迁移消费者，不能拥有 live generation；Tools 的 target open、参数准备与授权回调，以及 Content 的 ACTIVE 定义快照、贡献者 scope、动态 prepare 和视图逆序释放已按实际 owner scope 接线；Materials 生产 bind 已按 provider/贡献者 scope 接线，Bindings.open 已改为目标 provider 的 local scope；本轮 Manager 已接入 generation provenance 与局部 operation owner；Models M1/M2 本地 owner、namespace、auth 和 settings source 收敛已完成本批静态改动，仍待复核。Shell owner-entry 已按单 owner/TaskAdmission 接入；RPC/UI、scope 外 admission/manager 消费者、其它 provider、current_snapshot/fence 和公开入口仍待后续迁移；生产调用面尚未整体接入 `acquire_call`。
- 测试：相关新增/修正测试均按真实签名、Event 屏障和实际 `Receipt`/`Message` 构造补齐（无 sleep/忙循环），未执行；不得视为运行验收。

### T03 · 迁移实际调用和公共能力登记

**输入与下游：** T02 的局部生命周期 → Tools/RPC/UI 等消费者能在同一 Root 上观察能力增减。

**主要文件与职责：**

| 范围 | 要改什么 | 必须保留什么 |
|---|---|---|
| Context.runtime_scope、RuntimeScope；bindings.py | 从整图租约改为实际 owner/依赖的调用保护 | 实际引用、历史 metadata、原错误，不复活历史 Root |
| plugins/tools、plugins/reply、plugins/reply_program | 调用时保护目标；后续步骤从当前获授 view 取引用 | 工具授权、schema 与 ToolResult 只结算一次；旧引用不能静默指向新实现 |
| plugins/context/materials.py、plugins/models | 按实际贡献者和所选模型持有资源 | 单次准备及 ModelExecution 的一致性，不扩大为全图占用 |
| plugins/ui、Dashboard、Mobile 查询 | 目录随登记增减更新，取消“整个 Root 只能 seal 一次” | 模块合同检查、资源 owner、公开资产身份与客户端 revision |
| plugins/mcp、managed_processes、workloads | 同 Root 内使用 owner/activation 身份，登记随贡献方退出 | 真实握手、借用、停止回执与失败资源句柄 |
| bootstrap/app_server.py、渠道与来源插件 | RPC/输入接纳及生命周期按实际服务 owner 处理 | 消息持久接纳、取消、来源 Task 与发送回执的现有责任 |

**怎么做：**

1. 枚举全局 lease/runtime_scope 的入口及服务对象长期持有者，按“实际调用、硬依赖、只读目录”分类迁移，不能只按 import 搜索删文件。
2. 保留各 provider 自己的注册表；登记和撤销归贡献 Fiber 的 Effect。可缓存派生目录，但不建中央能力总表。
3. provider 仅对 active owner 暴露可调用能力；收回登记后，旧引用给出明确未执行反馈。
4. UI 可以重新生成轻量目录，但不能重建所有插件或重新启动所有 Dashboard 后端。移除的模块按现有客户端协议失效。
5. 盘点已声明 inject、嵌套 inject，以及 get/require 后保留对象的真实路径。长期依赖必须体现在已有依赖/作用域关系中，不增加任意 Python 对象的动态追踪系统。
6. 普通业务插件 API 尽量不变；确需改外部插件时只改其源码仓库，经正式安装验证，不直接改 cache。

**删除/替代：** 整轮全图占用、整图目录 seal 被具体调用保护和动态登记替代；不是无替代的删除。

**DSH 对照：** D05 的登记与 Effect 同归属，D06 的目录直接投影；保留 Akashic 比示例更严格的引用和清理合同。

**退出条件：** 无关长 Turn 不阻塞 Fitbit 更新；已开始的 Fitbit 调用不会被拆掉底层资源；
后续工具/查询获得新能力或明确不可用；普通消息、模型绑定和资源回执不变。

#### T-131de5 · MCP 真实 live Root fixture R1（原稿不追认为通过稿，后续修正另行记录）

T-4be90d 的整体测试源 review 未通过；T-131de5 原 draft 不追认为通过稿；后续修正另行记录，测试仍未运行。普通 MCP
调用复用同一真实 Manager/live Root/probe 与 SERVICE open closure；Manager 分支验证全量关闭，
local 分支只处置真实 probe contribution Fiber。hard consumer 由 `root.mount(..., inject=(SERVICE,))`
建立并以 cleanup Event 形成排空前屏障；无关 peer 提供自己的 service、Effect 与生命周期记录，
验证同一 Root、Context、activation、Effect 和计数不变。局部 owner 链为：

```text
plugin artifact + ExecutionAccess → contribution Context → MCP Session Effect
    → host/client/process → disconnect confirmation → Effect release
```

顺序 oracle 是真实握手进入 `McpClient.connect` 后再开始 drain：probe 进入 `UNLOADING`、旧
Session/OwnerCall 仍被持有时，peer 仍可 scope/require，保存的旧 open closure 对新调用给出
`OWNER_UNAVAILABLE`；放行握手、使用 gate 与 consumer cleanup 后才等待原 owner 结算。失败/取消
路径在同一 finally 回收 Task、gate、Manager 和 MessageLog，不以整 Root dispose 或
`return_exceptions=True` 掩盖错误。候选环境仍使用独立 Root、固定 `CodeOwner` 与
`ExecutionAccess(candidate=True)`，构造前即隔离 host secret，不调用 Manager candidate publication。
MCP 生产 owner、wire、持久化、descriptor、`root_ref`、metadata、credential 和正式 durable data
不变，本轮 production delta 为 0；其它 provider/RPC、scope 外 `current_snapshot`、offline/trusted
watcher、candidate/freeze、T06/T07、Gate 与运行验收仍是 WIP。

#### T-e70bc3 · T03 Commands provider R2（静态接受；T-e87866 行为验证待 review）

T-471123 的三个 Commands production 文件已由主审与独立只读 review 静态接受并冻结；T-e70bc3 只修订 `tests/test_commands_provider.py` 与本设计、NOW、0072 对账，source 与状态对账先前已静态接受，均不改 production。T-69d63e R1 已修订的 Message workspace 初始化与 Commands 基础生命周期项保持不变；R1 整体仍因取消分支缺少有界等待、排空原 Task 未验证 nested 两 owner scope/冻结 provider 读取、recover 错误后续健康调用未验证而不追认通过。本 R2 补上取消等待 timeout、原 handler 的 nested scope 与 contributor provider identity、admission barrier 后 capture 结算、`recover_error` 原始错误传播和独立 health command。

T-e87866 首次行为验证按三个独立组运行：provider 组 13/13 passed（6 个精确函数展开为 13 cases），message 组 9/9 passed（6 个精确函数展开为 9 cases），reply_entry 组 0/1，合计 22 passed、1 failed、无 skip/error/timeout，artifact 根目录为 `/tmp/i750-cmd.l6NEiF/`。provider 覆盖 Core 不提供 Commands、foreign/current/stale Context、immutable view、binding/registration Effect、双 owner drain 和 handler/result/cancel/recover；message 覆盖真实 identity、receipt 恢复、不重放、unknown、取消/new input、abandon 与输入顺序。

唯一失败是 `tests/test_message_commands.py::test_default_reply_short_circuits_command_before_model_or_tool`：测试已完成 fixture load 和真实 `CHANNEL_INPUT` 接纳，但 reply follower 在 `plugins/reply/follow.py:52-53` 的 `ctx.runtime_scope()` 内调用 Conversation source；`plugins/conversation/plugin.py:101-115` 用 Conversation `ctx` 执行 `MESSAGE_WRITERS.bind`、`TASKS.open`，最终在 `agent/plugin_composition/messages.py:77` → `agent/plugin_composition/context.py:180-196` 抛出 `CompositionError("OWNER_CALL_CONTEXT", "授权需要当前 Context 的 OwnerCall 或生命周期借用")`。因此 10 秒等待 Output 是后果，不是根因；清理时 `plugins/reply/follow.py:85` 的 TaskGroup 又把同一错误沿 `manager.terminate_all()` 暴露。该失败属于真实 production reply/source 入口与 teardown，不是 fixture/setup 或冻结 oracle 误报。

最小下一批边界：只在 source owner 创建 Task、由同一真实 Task 持有 Conversation 的 MessageWriters/TASKS scope，并在该 Task 内按 permit 进入 Reply program scope；`task.join()` 保持在创建 scope 外。先用本失败的 default-reply entry 回归验证，不扩大为新 registry、总 lease、schema 或协议迁移。Source registration wake、Subagent→REPLY_PROGRAM.report→Reply writer 的第三 owner、其它 provider/RPC/consumer、`current_snapshot`/fence、offline/candidate/freeze、partial startup/task factory、T06/T07、全量回归/Gate/CI/正式运行和 final enable 仍为 WIP。

#### T-c6ec4e · T03 Reply/Source owner wiring（R0/R1 整体不接受；窄事实保留）

前序 T-c6ec4e/T-750784 的整体概念与测试合同不接受；只保留真实 Task service close 的 typed
信号、`formal=False` 普通拒绝、Source admission 边界捕获和双 owner scope 的窄事实。R1 的
`tests/test_reply_follow.py` 字节未改，只有原三项旧测试；其前置 red 仅写成文字，缺少原始
command/console/JUnit/exit，因此不作为已证明的 red。R2 只在现有真实 Manager/Root/Source
fixture 上补取消结算回归，不把旧 green 外推为所有 factory、monitor 或 owner 边界已覆盖。

R2 当时的生产主链目标是：raw monitor 只等待 `reply_scope.wait_admission_closed()` 与
`source_scope.wait_admission_closed()`；拿到 exact Source Task 后同步注册 `Task.on_done`，
以局部 Event 等待 physical finally，随后调用公开 `Task.join()` 取得正常、child-cancel 或
真实错误。R3/R4 后续复核发现，caller cancellation、child `CancelledError`、同步
`Task.cancel()` cleanup error 和 monitor error 的原对象保留必须按实际 owner 结算路径分别证明，
不能用这段当时目标替代失败路径 oracle。Source Task
仍由 SourceSession/Tasks 独占，`follow` 只做一次 cancel/join 协调；两个 `runtime_scope`
入口只把 `OWNER_UNAVAILABLE`/`STALE_ACTIVATION` 转换为局部不可用，open/start/body 的其他
错误继续由原 owner 观察。

```text
Reply/Source admission
  ├─ raw monitor（空 Context，只等 admission）
  └─ Source Task capture（与 monitor capture 不同）
       ↓
admission close → 当前 drive 一次取消 → exact Source Task at-most-once cancel
       ↓
Task.on_done 物理 finally → public Task.join 终态 → Reply/Source permit 归还
```

R2 当时记录使用临时 plugin/home/workspace/SQLite/archive fixture，最终 artifact 为
`/tmp/i750-reply-r2.XTKLDt/`；前置 controlled red 的原始 command/console/JUnit/exit 在
`/tmp/i750-reply-r2.wZ6E5X/`，退出码为 1。主审本轮确认该旧 command 不能独立核验环境隔离，
controlled mutant 也不是 R1 原样 pre-fix red；这些只作历史材料，不作为本轮 red 或独立隔离证明。

- I `tests/test_reply_follow.py` 全模块：7/7，退出码 0；包含 4 个新增真实取消/物理结算回归。
- II `tests/test_message_commands.py::test_default_reply_short_circuits_command_before_model_or_tool`：1/1，退出码 0。
- III 两个 R1 task-scope 收尾 nodeid：2/2，退出码 0。

AST/内存 compile、`git diff --check` 与 frozen151/151 另有回执证据；本轮不宣称独立 review
通过，也不是 Issue750 system Gate、CI、部署、正式运行或 final enable。`Message`/schema、
intent/receipt、descriptor/root_ref/metadata/credential、selection/journal/history/archive 的
正式持久化 delta 为 0；测试临时 DB/archive 写入不属于正式数据迁移。Source registration
wake、Subagent→`REPLY_PROGRAM.report`→Reply writer、其它 scope-out provider/RPC、legacy
snapshot/fence、offline/candidate/freeze、后续 None/closed gate 与完整 owner/race 覆盖、T06/T07、
累计 Gate/CI 与最终启用继续是 WIP。

MCP 的 T-a30391 累计测试源保持静态已接受、未运行；T-131de5 原 draft 不追认为通过稿，后续修正另行记录。

#### T-d80010 → T-673b11 · T03 Reply/Source owner R2/R3（R3 暂拒；事实保留）

R1 已接受的窄事实保留：真实 `_closed` 的 `admit`、group、idle 和 exclusive 闸门抛
`TaskServiceClosed`，`PluginTasks(formal=False)` 仍抛原普通 `RuntimeError("当前不能接纳正式
Task")`；Reply 只在 `source.open()`/`session.start()` 的正式接纳边界处理 typed 关闭，不按
错误文本、不窥探私有 `_closed`，source/body 的其它 `RuntimeError`、`CompositionError` 和
业务错误沿原 owner 观察。

R2 保留的唯一收尾路径如下；R3 在同一链上补足取消归因与 child 终态：

```text
Reply/Source scope capture
  ├─ raw monitor（空 Context；不继承授权）
  └─ Source Task capture（独立于 monitor）
       ↓
admission close → drive cancel → exact Source Task at-most-once cancel
       ↓
Task.on_done physical finally → public Task.join 终态 → permits release
```

`None` 的 restart gate 等待仍发生在 captures/monitors 归还之后；existing Task 不重跑程序。
R2 新增的真实回归覆盖 monitor 停止期间 caller 取消、Source Task 取消与 cleanup 错误并发、
admission monitor 内部取消叠加 caller，以及持续拒绝 task factory 的 public join 路径。未新增
的 None/closed gate 即时释放、monitor 两次创建失败、旧 Task join→closed 再 admit race、其它
provider/RPC/consumer 等边界继续是下一批；不能把 R3 当时的 7/7 绿测扩成全合同覆盖。
R3 当时没有自称独立 review 通过；其后续审查结论为暂拒，当前 R4/R5 状态见下段。

R3 的两个真实根因与处理是：

1. marker 与 caller 在同一个 exact drive await 同拍到达时，消费一次内部 marker 后若取消计数仍
   非零，只记录“外部取消事实待传播”，不盲目继续 `uncancel()`，也不伪造平台没有独立交付的
   caller message/object；monitor、scope 与 Source Task 物理结算完成后才传播该事实。
2. `on_done` 已证明 exact Source Task 完成后，public `Task.join()` 的 `CancelledError` 按本次
   waiter await 的新增取消归因，不能拿历史 `cancelling()` 计数吞掉 child 终态。follow 主动取消的
   纯 child CE 仍可按既有局部语义忽略；child CE 带真实非取消 cleanup cause 时保留同一对象及
   `__cause__`，不克隆、不改写、不把它替换成文本。

R3 同时把四个 R2 follower 例的 gate release、exact watcher/Source Task 结算和 fixture 的 body/
cleanup 错误合并收口；factory rejection 例不再由拒绝 factory 关闭 coroutine，最终 cleanup 才
负责未消费对象。真实隔离 harness 清除继承的 `AKASHIC_*`，每组新建 plugin/home/workspace/
basetemp，保留完整 argv、console、JUnit、exit，并使用 TERM 180 秒后 10 秒 KILL 的单进程边界。
R2 production 上 A 同拍序列实际为 green，不能伪造为前置 red；B 的真实 child CE/cause oracle
以 exit 1 红，artifact 为 `/tmp/i750-reply-r3.9rZYlv/B/`。R3 修复后 A/B 分别在
`/tmp/i750-reply-r3.ZYp3Lz/{A,B}/` green；最终完整 I artifact 为 `/tmp/i750-reply-r3.5eIkkK/I/`。

- I `tests/test_reply_follow.py` 全模块：9/9，退出码 0；包含 R2 四例与 A/B 两个根因回归。
- II `tests/test_message_commands.py::test_default_reply_short_circuits_command_before_model_or_tool`：1/1，退出码 0。
- III 两个指定 task-scope nodeid：2/2，退出码 0；最终 II/III artifact 根目录为 `/tmp/i750-reply-r3.5eIkkK/{II,III}/`。

R2 的 10 次 green（I 7/7、II 1/1、III 2/2）与 `/tmp/i750-reply-r2.wZ6E5X/` 的 joiner/factory
controlled mutant 仍是历史证据：后者不是 R1 原样 pre-fix red，旧 command 也不能独立核验本轮要求的
隔离环境/timeout。R3 的 B red 才是上一轮 production-before 修复的实际根因证据；A 在当前 R2
不红则如实保留。上述结果不宣称独立概念 Gate、Issue750 system Gate、CI、部署、正式运行或 final
enable；正式 Message/schema、intent/receipt、descriptor/root_ref/metadata/credential、
selection/journal/history/archive 与 durable data delta 仍为 0。

#### T-cfe587 · T03 Reply/Source settlement owner R4（production 已接受；R5 测试收尾待 review）

T-673b11 的 R3 整批被主审与独立 `gpt-5.6-terra/xhigh` 概念闸门暂拒：原 A 测试的
source 程序没有吞掉自己的纯 child `CancelledError`，因此旧 green 不能证明 P1；同时，
孤立 child CE+cleanup cause 与同步 `Task.cancel()`/`on_close` 失败被错误地当作 follower
cleanup error。R4 的 production 修复随后已由主审与独立概念 Gate 接受并冻结；R5 不重新
评审 production，只修测试 cleanup/oracle 和执行记录，不改 `Task`、`TaskGroup`、
`SourceSession`、Manager 或其他 owner。

R4 的唯一 owner 规则是：

1. 每次 iteration 的 `source_settlement_errors` 只保存当前 exact Source Task 的同步
   cancel/on_close 失败和 physical `Task.join()` 终态错误；它是局部事实列表，不是 registry，
   Source Task 仍由 `SourceSession/Tasks` 拥有。
2. monitor、captured scope 和 exact Source Task 都完成物理结算后才作最终判断；caller 后到也
   只能追加原始 caller 事实，不能改写 Source Task 对象。
3. 只有 Source settlement error、没有 caller cancel、monitor/scope/pending hard error 时，
   用原异常 `exc_info` 逐项 warning，停止当前 source drive；R1 的“不重试”仅覆盖该 drive
   完成结算后由 unrelated registration 变化触发的唤醒，不覆盖同 Session peer 的既有
   changed-session 重扫；健康 peer 继续运行。
4. 若存在 caller/follower hard error，则把 Source errors 原对象加入最终 error tree；纯 owner
   cancel 的 child CE 仍忽略，普通程序异常的既有 source-local warning 保持不变。

```text
Source Task owner
  ├─ physical cancel/join → source_settlement_errors（当前 iteration）
  └─ pure owner CE → 正常结算；真实 cleanup/sync error → 原对象保留
                 ↓
follower/monitor/scope 完成 → source-only: warning + stop drive
                         └─ 伴随 caller/hard error: 合并 error tree 后传播
```

R4 真实红绿证据使用独立 `Root.mount` 的 fault Source 和健康 conversation peer；没有新
registry、Task factory 或测试专用 Fiber。production-before 的两项 C 回归以 exit 1 红，
原始 console 为 `/tmp/i750-reply-r4-red.LuEiTO/console-production-before-bound-final-red.txt`：
C1 是 child CE 被 TaskGroup 忽略、source-local warning 缺失，并非 watcher 被击穿；C2 是
sync `on_close` error 进入 TaskGroup，健康 peer 无法完成。R4 修复后 console 观察到
targeted 2、I 11、II 1 绿，但没有保存完整 command/JUnit/逐 attempt 隔离与 timeout，不能把
这组观察升级为独立可复核验收，也不猜旧测试运行 cwd。

T-d44273 的 R5 只在四条允许路径内补 A/C 测试收尾和状态记录；production `follow.py` SHA
仍为 `ee6ce8abba170675f90f020509179d7e61fbb7761652f1a2c48ac82319a27e9c`。新 harness artifact
为 `/tmp/i750-reply-r5.b4WgrJ/`，cwd 为
`/mnt/data/coding/akasic-agent-issue750-implementation-20260920`，使用
`/mnt/data/coding/akasic-agent/.venv/bin/python`、pytest 9.0.3、清除继承的 `AKASHIC_*`、
`PYTHONDONTWRITEBYTECODE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、
`-p pytest_asyncio.plugin -p no:cacheprovider`，每组 timeout 180 秒、TERM grace 10 秒：

```text
I: /mnt/data/coding/akasic-agent/.venv/bin/python -B -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/test_reply_follow.py --basetemp /tmp/i750-reply-r5.b4WgrJ/I/basetemp --junitxml=/tmp/i750-reply-r5.b4WgrJ/I/results/junit.xml
II: /mnt/data/coding/akasic-agent/.venv/bin/python -B -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/test_message_commands.py::test_default_reply_short_circuits_command_before_model_or_tool --basetemp /tmp/i750-reply-r5.b4WgrJ/II/basetemp --junitxml=/tmp/i750-reply-r5.b4WgrJ/II/results/junit.xml
```

I JUnit 为 11 tests、0 failures、0 errors、0 skipped；II 为 1、0、0、0，均 exit 0 且未超时。
本轮源 SHA 为 `tests/test_reply_follow.py=deacfc706fa3b86d63ae522bb790567bf7f0b1c42763eb258f2f542eb0358a3b`、
`tests/test_message_commands.py=42e11683a7a1ac0999eb8d05a291cb4a569e929675ba8492942b6b5bce8132ac`；
恢复点为 `/mnt/data/issue750-reply-owner-r5-preflight.Co8EHR`，本轮编辑备份为
`/mnt/data/issue750-reply-owner-r5-edit-backup.vLlb9x`。R5 结果仍待主审与独立只读 review，
本回执不自行宣布 R5 通过；不外推为 Issue750 system Gate、CI、部署、正式运行或 final enable。
正式 Message/schema/intent/receipt/selection/journal/history/archive 与 durable data delta 仍为 0。

#### T-a9246e / T-b9bd66 · T03 Source registration wake 与 R5 A（R1 修订，历史记录）

本段保留 R1 的历史证据与边界；R1 不是当前 Source wake 接手点，当前状态见下方 R2/R3。

T-a9246e 的 exact registration、activation-ready `ctx.spawn`、visible-only 首次发布和 R5 A
方向保留，但其 production/test/docs review 暂不接受：P1a dedicated LOADING route 会回退
default、P1b 只依赖异步 Source snapshot、P2 TaskGroup factory failure 会遗留 coroutine；其
W2/W3/W4 oracle 及 W1 与最终测试版本的 red binding 也不足。旧 17 green 与 R5 A 是历史事实，
不追认旧 oracle 已完整验收。

本 R1 只改以下六个允许路径中的三个生产/测试文件，不改 `Task`、`TaskGroup`、`SourceSession`、
Manager、Core、Message/schema 或正式 durable data：

```text
SOURCES.accept ──┬─ all registrations decide route
                 └─ visible only controls entries()/Reply discovery

catalog.follow ─┐
sources.changes ─┴→ wake/event only → current _items identity → TaskGroup drive
                                      ├─ replacement: public old Task.join() → new Source
                                      └─ revoke: exact Source scope stops; no stale open/start
```

P1a 让 dedicated channel 在 LOADING 时仍命中其 exact registration，由 owner admission 保持
`CompositionError(code="OWNER_UNAVAILABLE")`，不回退 default；activation-ready 后原输入只由
dedicated 接纳一次。P1b 让 watcher 只负责唤醒，main action 与每轮 Source scope 进入前都从
权威 `_items` 核对 exact identity。P2 统一 safe-create：`TaskGroup.create_task` 交接失败时
关闭尚未消费的 coroutine，且不会留下 drive 的 active bookkeeping。三项都不增加 registry、
revision、failure 表或第二套 scheduler。

W2 现在以真实 writer 的 fault Input 先形成 catalog head，再由独立 LOADING contributor 持 gate；
放行后只有一次实际 Task 和唯一 complete fault Output。W3 以真实 public `Task.join()` 返回门闩
证明旧 `on_done` 到达而 production join 未返回时 new program 不会启动，并在 replacement 前、
窗口与之后保留独立 peer。W4 直接取消已接纳的 fault Source Task 并保留 cleanup cause，等待
production join 和精确 warning；fault registration/owner 仍 ACTIVE，再经真实 unrelated
`changes()` barrier 验证无重试，healthy session 仍只产一条 Output。W5 原 on_done/join 与
不 restart 语义保持。

最终源码绑定与隔离证据：

- P1a/P1b/P2 production-before red 分别为 `/tmp/i750-source-wake-r1-final-red-p1a.2Vk9lN/P1a/`、
  `/tmp/i750-source-wake-r1-final-red-p1bp2.BH9v47/P1b/`、`/tmp/i750-source-wake-r1-final-red-p1bp2.BH9v47/P2/`；
  均为行为/资源 oracle 的 test failure，非 setup/import/harness error。
- W1 用最终测试版本绑定精确旧 production：
  `/tmp/i750-source-wake-r1-final-red-w1.g0epFu/W1/`，旧
  `plugins/sources/plugin.py=8a90801371bc060023425420013f2bb17e67c4a621139aeda427b0db2a82d3aa`、
  `plugins/reply/follow.py=ee6ce8abba170675f90f020509179d7e61fbb7761652f1a2c48ac82319a27e9c`；
  JUnit 1 test/1 failure、exit 1、无 timeout，失败是历史注册后未唤醒的 `TimeoutError`。随后
  已恢复当前 R1 production hash。
- 最终 targeted `/tmp/i750-source-wake-r1-targeted-final.0HLrHb/targeted/` 为 9 tests、0 failure/error/skip、
  exit 0；Reply full `/tmp/i750-source-wake-r1-full-final.k1EzkS/full/` 为 20/20；冻结 default command
  `/tmp/i750-source-wake-r1-command-final.UCcZDG/default-command/` 为 1/1。三组均无 timeout，使用固定
  `/mnt/data/coding/akasic-agent/.venv/bin/python`、pytest 9.0.3、隔离 pluginhome/workspace/tmp、
  `PYTHONDONTWRITEBYTECODE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 与 frozen pytest plugins。
- 当前生产 hash 为 `plugins/sources/plugin.py=b9a574264bc36b27bc065939d8bab6d9497141d53d0ff622f262d41cadf75875`、
  `plugins/reply/follow.py=98e45aa4433420470339c753b06ebdcbf96085ad85699fed0fcf6463720bd169`；最终测试
  hash 为 `tests/test_reply_follow.py=908c9b0d06b68a843202f95940ae2a72ff5873ff3a32b538fb04b643c530eca5`，
  frozen `tests/test_message_commands.py=42e11683a7a1ac0999eb8d05a291cb4a569e929675ba8492942b6b5bce8132ac`。
  每个最终 artifact 的 `source-manifest-preexec.sha256` 另列四文件，其中 harness 自带 `sha256.json`
  保持原三文件格式。
- `/mnt/data/issue750-source-wake-r1-preflight.5FwAmM` 是 recovery/frozen source；本轮测试备份为
  `/tmp/i750-source-wake-r1-test-backup.or3Od7/`，文档编辑前备份为
  `/tmp/i750-source-wake-r1-docs-current.Slj8zs/`，临时 production 恢复副本为
  `/tmp/i750-source-wake-r1-production-current.ZVtI9A/`。frozen152/152、AST/内存 compile、
  `git diff --check` 通过，正式 durable delta 为 0。

仍未闭合的历史边界必须保持窄写法：若 source registration 仍存在且发生 source-only settlement
error，之后同 Session peer Message 仍可能按既有 changed-session 规则重扫并重试旧 Input；本轮
“无重试”只证明 unrelated registration 变化，不是所有 peer 变化的全局保证。failure/retired
registry 是 PROHIBITED ALTERNATIVE，不是实现 TODO；同 Session gap、其它 provider/RPC/consumer
race、snapshot/fence、offline/candidate/freeze、T06/T07、累计 Gate/CI、正式运行与 final enable
仍为后续 WIP，待主审与独立只读 review。

#### T-cef8cd · T03 Source wake R2（production 已接受；R3 测试收尾待审）

T-b9bd66 的 P1a、P1b、原始 P2 保留为接受事实，但 R1 整批不接受：新增
`asyncio.eager_task_factory` 路径暴露了 P2 的 active owner ordering，且 W2/W3/W4/W5
仍有 physical cleanup/oracle 缺口。该结论来自 `gpt-5.6-terra/xhigh` 审查，不把
reviewer 意见写成完成声明。R1 起始 hash 为
`plugins/reply/follow.py=98e45aa4433420470339c753b06ebdcbf96085ad85699fed0fcf6463720bd169`、
`tests/test_reply_follow.py=908c9b0d06b68a843202f95940ae2a72ff5873ff3a32b538fb04b643c530eca5`。

R2 唯一 production 变更是 `plugins/reply/follow.py:schedule`：先登记 exact
`active[key]`，再交接 drive；交接失败时仅在 `active[key] is wake` 时删除并
原样抛错。未修改 `plugins/sources/plugin.py`、Core、Task、SourceSession、Manager，
没有新增 registry、failure 表、fallback scheduler、Message/schema/receipt 或 durable owner。
该 production ordering 已由主审与独立 `gpt-5.6-terra/xhigh` 概念 Gate 接受并冻结；R2 的
eager red/green、targeted 8/8、Reply full 22/22、default command 1/1 及 static/frozen153
检查已完成。R2 整批仍不写成 source/test 全部接受：本轮 T1 queued 时序与 W2/P1a/W3/eager
测试 cleanup 缺口由 R3 收口。
最终 hash 为
`plugins/reply/follow.py=66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247`；
冻结 Sources 仍为
`plugins/sources/plugin.py=b9a574264bc36b27bc065939d8bab6d9497141d53d0ff622f262d41cadf75875`。

测试源补齐合同 T1-T5：T1 以 test-owned wrapper 持有真实 queued drive，撤销 Effect 后再放行
并证明旧 open 未调用；W2 保存 exact Source Task、production public `Task.join`/
`on_done`，等待真实 Fiber calls-idle 后核对 raw permits 为零；W3 使用独立 mounted
replacement contributor，记录同一 healthy peer 的 raw Fiber、Context、public activation token、
owned Effect、lifecycle/cleanup counters，并覆盖 replacement 前、窗口中、完成后三个真实
peer Message 窗口；W4 捕获 exact child `CancelledError`，核对 warning 的 `exc_info`
与 cleanup cause，并用真实 main `entries()` 的 `call_soon` barrier 后检查 exact
Source identity；W5/P1a 的 mount、collector、dispose Task 在 gate 释放后消费原 Task 一次。
最终测试 hash 为
`tests/test_reply_follow.py=86cb2a0e2b7efdf896f54a448f287532d1ec72ca3479eb90de9d306c7e82d56b`。

最终同一测试源版本的 eager red/green 与分层证据：

- `/tmp/issue750-source-wake-r2-evidence-20260923/eager-before-final/`：起始 Follow
  hash 下 1 test/1 failure、exit 1、无 timeout；失败是同名 replacement `new_started`
  超时，不是 setup/import/fixture failure。
- `/tmp/issue750-source-wake-r2-evidence-20260923/eager-after-final/`：修复后 1/1
  green，exit 0、无 failure/error/skip/timeout。
- `/tmp/issue750-source-wake-r2-evidence-20260923/targeted-r2c/`：8/8 green；
  `/tmp/issue750-source-wake-r2-evidence-20260923/full-reply-r2/`：Reply 全模块 22/22
  green。

新 harness 的 `harness_python` 是 `/usr/bin/python` 3.14.7；实际 child 始终是
`/mnt/data/coding/akasic-agent/.venv/bin/python` 3.13.7，pytest 9.0.3、
pytest-asyncio 1.3.0。每组记录实际 argv/cwd、四文件 pre-exec SHA、清除的 `AKASHIC_*`
名称、独立 pluginhome/workspace/basetemp/TMPDIR、console/JUnit/exit/timeout；旧 artifact
保持不变。R2 static/frozen153、最终 hash/AST/空白审计已完成；R3 只处理测试 oracle 与
finally 物理结算，结果待本轮主审与独立只读 review。

本轮不声称 Issue 750、Gate、CI、部署、正式运行或 final enable 完成；仍登记 Source 的
same-session peer retry gap、其它 provider/RPC/consumer race、snapshot/fence、
offline/candidate/freeze、T06/T07 与累计系统验收继续是 WIP。正式 Message/schema/receipt/
selection/journal/history/archive 与 durable data delta 仍为 0。

#### T-6ee259 · T03 Source wake R3（历史结果；三项异常路径缺陷由 R4 收口）

R3 只修改 `tests/test_reply_follow.py` 与本文件、`docs/NOW.md`、决策 0072：queued oracle
把 peer 输入移到旧 drive 物理结算且确认未 open 之后；W2/P1a/W3 使用 `asyncio.wait` 等待而不以
timeout 取消 owner Task，并只消费 exact result 一次；W2 在 `before_start` 失败时先放行 gate
并物理收口 mount/collector；W3 与 eager 对挂载任务和每个真实 Effect 独立清理，保留 body 与
cleanup 错误。production `follow.py`、Sources/Core/Task/SourceSession/Manager 不变。
新隔离 artifact 为 `/tmp/issue750-source-wake-r3-evidence-20260923.aRHRGi/`：targeted-r3
为 5/5，Reply full-r3 为 22/22，均 exit 0、无 failure/error/skip/timeout。两组均使用
`/mnt/data/coding/akasic-agent/.venv/bin/python` 3.13.7、pytest 9.0.3、pytest-asyncio 1.3.0，
并保留实际 argv/cwd、隔离 roots、console/JUnit、四文件 pre-exec SHA；测试 SHA 为
`tests/test_reply_follow.py=3fb9b2a568164e2415bb8ee7e5e3d9febe0a104da5dd41c3c6c64ec6d91a51a3`，
冻结 `tests/test_message_commands.py=42e11683a7a1ac0999eb8d05a291cb4a569e929675ba8492942b6b5bce8132ac`，
production `follow.py=66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247`，
Sources `plugin.py=b9a574264bc36b27bc065939d8bab6d9497141d53d0ff622f262d41cadf75875`。
AST/内存 compile、`git diff --check`、四文件 scope、frozen154/154 与正式 durable data delta=0
均通过；R3 的 queued/W3/eager 独立 Effect cleanup 已接受，但 whole R3 仍因 W2
`collector_joined` lexical binding、三处 bounded wait 超时后无界等待，以及 W2 collector
对带 cause 的 `CancelledError` 误忽略而未接受。该历史结果不写成 Issue 750、Gate、CI、部署、
正式运行或 final enable 接受。

#### T-26b5a5 · T03 Source wake R4（三项测试异常路径收尾，已接受）

R4 只修改 `tests/test_reply_follow.py` 与本段、`docs/NOW.md`、决策 0072；production
`follow.py`、Sources/Core/Task/SourceSession/Manager 继续冻结。W2 补齐 `before_start` 的
`collector_joined` nonlocal；W2/P1a/W3 在 bounded `asyncio.wait` 超时后立即以明确断言失败，
finally 先释放 gate、再在 timeout 外用 `asyncio.wait` 物理收口并消费 exact Task result 一次，
删除 R3 中与 joined 同值的 retrieved flags；W2 collector 只忽略本 cleanup 主动取消且
`CancelledError.__cause__ is None` 的纯取消，其它取消与 cause 保留。

新 artifact 为 `/tmp/issue750-source-wake-r4-evidence-20260923.x4nLCk/`：targeted-r4
为 3/3，Reply full-r4 为 22/22，均 exit 0、无 failure/error/skip/timeout。两组均使用
`/mnt/data/coding/akasic-agent/.venv/bin/python` 3.13.7、pytest 9.0.3、pytest-asyncio 1.3.0，
并保留实际 argv/cwd、隔离 roots、console/JUnit、四文件 pre-exec SHA；测试 SHA 为
`tests/test_reply_follow.py=7c060ed60332ca06a618bdfc9c0dc8ef873d6c42425e92da207b4f82fb55614d`，
production frozen SHA 仍为 follow `66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247`、
Sources `b9a574264bc36b27bc065939d8bab6d9497141d53d0ff622f262d41cadf75875`。AST/compile、
W2 symtable binding、无界 timeout fallthrough source check、`git diff --check`、四文件 scope、
frozen154/154 与 formal durable data delta=0 均通过；R4 已由主审与独立只读 review 接受，
不外推为 Issue 750、Gate、CI、部署、正式运行或 final enable 完成。

#### T-8677b9 · T03 Source wake R5（production 已接受；测试批次待收口）

R5 只修改 `plugins/reply/follow.py`、本测试与 Source wake 三份状态文档。owner 仍是 Reply
follower 的进程内通知观察：合并 Message-head 与真实 Source identity-history 两条调度循环，
每个被检查的 `(session_id, source_name)` 读取既有 `reader.head(source=...)`；真实 identity
变化仍无条件扫描历史，并在 `schedule` 前记录该 head，普通 Session 变化则仅在 source head
不同于本 follower 已观察值时调度。active revoke/handoff、SourceSession 判定和失败 Control
不变；没有增加 failure/retired 状态、cursor、SourceSession/Core 协议或取消框架。

production-before red 使用起始 `follow.py` hash
`66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247` 与最终回归源码 hash
`79c62421c578f39e07427f5b8a7b53ef178d6d7bdd6f237d6e2a26785f7ce122`：真实 fault Source
晚注册并执行历史 Input，child `CancelledError` 保留原 cleanup cause 与 warning；conversation
peer 的新 Input 被 follower 实际扫描且完整 Output 可见，但旧实现把 fault Source 启动两次，
故 1 项回归按预期在启动次数断言失败。修复后 targeted 1/1、Reply full 23/23、冻结 default
入口 1/1、SourceSession retry/pause controls 2/2 均通过，无 failure/error/skip/timeout。

artifact `/tmp/issue750-source-head-evidence-20260923.mj1Qbk/` 保留 red、green 各组的命令、
cwd、环境隔离路径、console/JUnit、exit/timeout、六份 pre-exec source snapshot 与 hash；
运行器和 child 均为固定 Python 3.13.7，pytest 9.0.3、pytest-asyncio 1.3.0。green production
hash 为 `24aabacea42855c20f8bb0fed47f17b34f708cb32fa5672dad9f719b6fa37b18`，回归源码为
`79c62421c578f39e07427f5b8a7b53ef178d6d7bdd6f237d6e2a26785f7ce122`；冻结 Sources、
SourceSession、default 与控制测试 hash 见各组 `source/sha256.json`。本地 observed head 随
follower 结束即丢弃，不是成功/失败或消费事实；进程重启、真实 registration replacement
仍依 SourceSession 既有日志规则重读，因此不声称跨重启/换代不重放。R5 production hash
`24aabacea42855c20f8bb0fed47f17b34f708cb32fa5672dad9f719b6fa37b18` 已通过主审及独立
`gpt-5.6-terra/xhigh` 概念 Gate；原 production red/green 证据有效，但整批曾因下述测试 A/B
缺口不接受。正式 durable data delta=0，不代表 Issue 750、累计 Gate、CI、部署、正式运行或
final enable 完成。其它 provider/RPC/consumer、snapshot/fence、offline/candidate/freeze、
T06/T07 与最终验收仍开放。

##### T-a0440f · T03 Source wake R1（仅测试收尾，已接受）

本轮只改本测试函数与本段、`docs/NOW.md`、决策 0072；production 保持上述已接受 hash。修复 A
将 `cleanup_release` 放进 `async with running(...)` 内层 `finally`，确保 assertion/setup 异常时
先放行 Source cleanup，再由 fixture 物理收尾。修复 B 用仅在本函数生效的 `Task.join` delegate
观察两个 exact Source Task：首个 follower join 捕获原 child `CancelledError` 与 cleanup cause，
并与 warning 的异常对象核对；第二个 Task 的 `on_done` 与 follower join 均完成，成功结果与
complete Output 一并核对。移除了会二次读取取消结果的测试方 join；首个失败 Task 在实际
on_done/follower join 后也核验 Source Fiber 在途调用已归还。没有改共享 fixture、其他测试或
production。

最终测试 SHA `561fe762c686617e248999ffba554fa11b49eb60f0b07b755eb4eadf361e2692` 绑定原始
`follow.py` `66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247` 的隔离 red：
1 项行为失败，fault Source 在同 Session peer 扫描后启动数为 2，非 setup/timeout。修复版
targeted 1/1、Reply full 23/23 green，均无 error/skip/timeout。artifact
`/tmp/issue750-source-head-r1-evidence-20260923.bhy0c4kf/` 保存两组真实 argv/cwd、环境隔离、
四份源码快照及 SHA、console/JUnit/exit/timeout；Python 3.13.7、pytest 9.0.3、pytest-asyncio
1.3.0。T-a0440f 测试收尾已接受并冻结；既有 R5 production Gate PASS 与本批测试 review 分开记录。
formal durable delta=0。

#### T-82189d · T03 Subagent parent report owner wiring（行为部分通过，整体阻塞）

新增窄服务 `conversation.complete.v1`：Conversation owner 在自身 `runtime_scope()` 内打开
SourceSession 并等待 `complete(program)`；Subagent 只硬依赖该 callable，`_announce` 等待它完成；
Reply 的 `report` 也在 Reply owner scope 内执行。旧 `conversation.v1` 不保留 alias，专用旧 consumer
仍为 PENDING/`INACTIVE_SERVICE`；没有新增 Message、binding、receipt 或 schema。

```text
Subagent _announce
    └─ conversation.complete.v1
         └─ Conversation owner scope → SourceSession.complete(program)
              └─ Reply report owner scope → reply.execute.v1
```

固定解释器下，Subagent 全组 12 项为 1 passed、11 failed（artifact
`/tmp/i750-subagent-I.k0xvgM/`）：旧 key consumer 的 PENDING 合同通过；真实报告链在
`reply_program.run_reply → ToolProgramFactory.create_menu/bind_reply → MESSAGE_WRITERS.bind` 以
`OWNER_CALL_CONTEXT` 失败，因为 Reply scope 不能授权 Tools owner Context。该 owner 位于本任务允许
修改范围之外，未改 Tools/Core 来绕过它。drain 测试已观察到 Conversation 停止新接纳、旧 callable
拒绝新调用、已接纳 Task 返回并物理完成；随后 Manager 更新因 `delivery_policy` Fiber FAILED 而
未能通过依赖 readiness，因此局部更新成功条件仍未证明。

其余精确验证：Reply follow 23/23、message commands 10/10；跨 Source/Core/Delivery/Message/Models
的 8 项复验为 5 passed、3 failed，artifact `/tmp/i750-subagent-IV-final.BeYCY5/`。两项失败是
`delivery_policy` 启动缺 OwnerCall/lifecycle borrow，一项 Core 测试仍要求 `current_snapshot`；该
snapshot 分支与相关 production 修改不在本任务范围。Models 直连选择/原子 metadata 用例现已通过。
所有测试仅用隔离临时目录，正式 durable data delta=0；未运行 Gate/CI、部署或正式运行。当前实现
仍须解决或另行授权处理 Tools owner 边界，并在 delivery readiness 可用环境重做 drain 验收，之后再
由主审与独立只读 review 判断是否接受。


```text
贡献Context → registration Effect → _registrations/_names
consumer → 当次freeze view → Commands scope → exact handler scope
         → handler/recover/result → release
```

Command view 保存真实 provider Context 与精确贡献 Context；一次已接纳调用按
`Commands provider Context.runtime_scope()` 嵌套 selected contributor scope，handler/recover 和
结果校验完成后逆序释放。登记 Effect 移除后，新的 freeze view 不再包含该名字；已有 immutable view
仍保留原定义，只由实际 Context/owner admission 决定可用性，不把旧 view 改写成 unknown。失活 owner
明确返回 `OWNER_UNAVAILABLE`，真正没有匹配的当次 view 才保留 unknown/default 语义。

R2 测试源继续使用 live Root、真实 provider/contributor scope、Effect/lifecycle/permit 与 bounded Event gate，动态 fixture 写盘前先 AST parse/内存 compile；Message 的既有 receipt/不重跑断言只证明已加载归档源码不受 checkout 源文件改写影响，不证明 live binding 换代后的精确 provenance/handler 语义。
T-e87866 只运行合同指定的 13 个精确 function nodeid，实际展开 23 cases：provider 13/13、message 9/9，reply_entry 0/1，合计 22 passed、1 failed；artifact 为 `/tmp/i750-cmd.l6NEiF/{provider,message,reply_entry}/{console.log,junit.xml}`。失败为真实 reply/source owner 边界，根因与调用链见上文，不将 timeout 或 teardown ExceptionGroup 写成独立第二个 bug。
T-e87866 当批不改 production、测试源码或 formal durable data：Message、intent、receipt、binding、history、archive、descriptor、root_ref、metadata、credential、wire、schema 与正式数据增改减均为 0；测试允许写隔离临时 plugin/home/workspace、SQLite/archive/selection/Message/intent/receipt fixture。除上述 23 cases 外未运行其它测试、整模块、全量回归、Gate、CI、正式 workspace、插件安装、业务进程、部署或正式运行。其它 provider/RPC/consumer、Source registration wake、Subagent、partial startup/task factory、`current_snapshot`/fence、offline/candidate/freeze、T06/T07 与最终 enable 仍为 WIP，当前状态待 review。



#### T03a-0 消费者盘点与最小 scope 合同（2026-09-20 盘点；原 enter-才-acquire 方案已撤销，完整 scope 迁移仍未实现）

**真实消费者分类**（按调用路径，非 import 清单）：

| 类 | 真实路径 | 现在保护什么 / 多保护什么 | 最窄行为 | 迁移与删除点 |
|---|---|---|---|---|
| 1 普通短调用/实际目标 owner | `Context.runtime_scope`（context.py:155-177）已按本 Fiber activation 取得 `OwnerCall`/唯一 `RuntimeScope`；调用方：content:320、wake/runtime:228-304、materials:195、reply/follow:52、computer:359-365、markdown_memory:799、scheduler:42/48、subagent:272/293、delivery_policy:34-99、delivery/plugin:116、mcp:135、plugin_update:109/160、senders:119、stable_view:131、bootstrap/tools.py:179-190 recover_input、bindings.open:150-165 | 批次 A 已不再由 facade 持整图 lease；manager/bindings 等直构与 fence 仍可能持有旧 lease | 许可绑定到发起 Context 所属 Fiber 的 activation（OwnerCall）；上游 provider 资源由"依赖方先退出"顺序传递保护 | 批次 B 迁移旧 `get_current_runtime_lease` 读取与直构点后删除旧路径 |
| 2 生命周期自身启动/停止 | akasha plugin.py:521/532（start/follow）、channels/provider.py:940（start 读 snapshot_id）、tasks.py:152-160（Task._run 进出 scope） | 生命周期回调内绑整图 lease | 内部借用（transition owner 表述已废弃；T03a-5-R1 合同：binding 携带 (Context, 实际执行 Task)，由 _load/_unload 与 Effect `_close_task` 经窄 binder 显式建立/finally reset，不靠 ContextVar 继承、不泄给后台 Task） | 接入批次 A 随内核 start/stop 接线 |
| 3 capture 给子 Task/长执行 | shell.py:370-400（固定 Shell Context/TaskAdmission、child capture、create_task 失败 close、shield 取消仍等）；models/state.py:504/565/803/814；ui/plugin.py:133/139；react/plugin.py:324 `partial(react, capture_scope=ctx.capture_runtime_scope)`（bound method 跨 activation）；tasks.py:59-70（Task 复制 lease + 重置 task-bound vars） | 未迁移消费者仍可能持整图 lease；Shell 已固定单 owner | **T-59017c Shell entry**：Shell cleanup Task 只从注册时固定的 Shell admission 接纳，raw child 沿用同一 Shell scope；其它消费者仍按各自迁移批次处理，不在本批扩大范围 | T03a 后续步骤改 RuntimeScope 内部；Shell 的 caller Context 不再作为 cleanup 参数 |
| 4 Binding 持久 metadata/真实引用（事实已纠正） | bindings.py:66-… bind（验 lease root、service，归档 components/root_ref/service/metadata）；channels/provider.py:879 `_bindings` 是内存 dict 键 `(snapshot_id, channel_name)`，:942-949 已有 uuid `binding_token` 与 `activation_token`；bus/queue.py durable 用 handoff_id/channel/session/message 身份 | 内存键与运行期 fence 按 snapshot_id | binding 归档身份不变；运行期 fence 与内存键随消费者迁移（T03b 按真实消费者改）；持久描述若无已证实变化不迁，不发明"全图冻结 ID" | T03b/T04 按真实消费者迁运行 binding/fence；本轮不写数据 |
| 5 UI/状态只读身份（非单纯展示） | ui/plugin.py 的 live catalog/state；**frontend/dashboard/src/webHost.ts 与 agent/plugins/dashboard_host.py 用 snapshotId/catalogId/module generation 拒绝旧页面请求**；plugins/ui/web.py `build_web_ui_catalog` identity 含 registration UUID、plugin_id、generation_id、模块/样式/contract 摘要；Mobile 使用 Root generation_id 与 registration UUID 派生 plugin_revision | snapshotId/catalogId/plugin_revision 兼作请求 fence | Web snapshotId 用 Root generation_id，catalogId 与 Mobile revision 都来自真实登记事实；不为只读字段持全图 lease | Web/Dashboard 与 Mobile 目标接线，均待 review |

#### T03a-UI · 单一登记图与公开身份契约（Dashboard R3/A1-A3 与 Mobile R2 静态通过；测试未运行）

这一节是当前 Web/Dashboard 实现与待审验收合同。当前真实 owner 和消费者如下；
Core 只提供组合、Fiber/Effect 和实际 owner scope，不另建第二 registry、全局 `ACTIVE` 或 UI 专用
lease。

| owner / 事实 | 当前真实登记、资源和消费者 | 只读投影 | 不由它拥有的事实 |
|---|---|---|---|
| `plugins/ui/plugin.py:Ui` | `Ui.register` 创建一个共享 `registration_uuid` 的 `Registration`；登记 Effect 先可见，再由同一 contributor 的 `RUNTIME_STARTING`/ACTIVE helper 建立 `DashboardResources`；Effect 持有 close/retry | `_entries` 是唯一登记表；`build_web_ui_catalog` 与 bindings 是无副作用当前 ACTIVE 投影 | Loader/PluginGeneration 持有模块、importer 与代码；Workload 持有进程；安装选择、归档来源、业务 artifact 不由 UI 持有 |
| `plugins/ui/mobile.py:MobileUiSlots` | `_registrations` 以 `MobileUiBinding` 为唯一记录；`register_mobile` 的 Effect 撤销自己的 binding；binding 持有实际贡献 Context 与 registration UUID | `bindings()`/`contributors()` 只投影当前 Context；旧 owner 入口拒绝 `STALE_ACTIVATION` | 设备连接、query 结果、Message、artifact 与 plugin-data 不由 slots 持有 |
| Dashboard | `DashboardResources` 持有 app/module/返回 closeables；bootstrap 在 `core.start/load` 前把同一 app.routes 交给 Manager；live host 使用 Root/UI provider，binding 携带原 contributor Context | app、route 与 Web 请求的 Root generation/catalog/module/generation fence | DashboardHost 不是资源 owner；Workload 进程也不归 host；资源关闭后不得裸调用 app |
| Mobile query | `agent/plugins/mobile_ui.py:PluginMobileUiProvider` 持有 ThreadPoolExecutor、`_admitted_queries`、`_draining_queries`；`mobile_realtime/plugin_ui.py` 只持设备/插件 gate 与 request Task | catalog/asset/query 的当前 registration projection 与物理线程 drain | diagnostic `ContextVar` 不是 permit；ticket、设备撤销、plugin selection 与 archive 由各自 owner 解释 |
| 客户端 wire | Web `frontend/dashboard/src/webHost.ts` 读取 state 并发送 header/WS identity；Mobile 使用 `plugin_revision`、`catalog_revision`、request owner 与 cancel | 页面/请求接受或拒绝其声明的当前 fence | 不把 per-request UUID、`id(ctx)` 或 RuntimeCatalog 展示 id 当成 incarnation |

**登记、ACTIVE 与关闭顺序：**

```text
Ui/MobileUiSlots.register + registration Effect
    → 登记在实际 Root 中可见
    → 请求在实际目标 owner 上取得 admission
    → 关闭新 admission
    → 排空已接纳的 HTTP/query 与物理线程
    → Effect 关闭 DashboardResources/移动端资源
```

`plugins/ui/apply` 提供 `UI` 与 `UI_SLOTS`，registration Effect 是每个 contributor 的登记与撤销 owner；
Mobile 不再 seal，也不维护 snapshot projection。`DashboardResources.build` 在入口
就把 `_started` 置为 true；它只在真实 contributor 的 `RUNTIME_STARTING`/ACTIVE scope 内执行，route tuple
来自 host 的 `DASHBOARD_ROUTES`，不读目录、不创建第二 host route、不容忍失败。build/import/route 冲突失败
不能重放 build；已经产生的 closeables 仍由原 Registration Effect 持有并可重试 close，原注册成功关闭后
才允许新注册再 build。ACTIVE 目录读取只返回当前投影，不重放模块副作用。启动/关闭 hook 仍明确不支持；
Loader 持有永久的 module/importer/code，DashboardResources 持有本次 closeables。注册类型/owner 与 Web
合同由 provider 在边界校验；`build_web_ui_catalog` 只做纯 contract/digest/总字节投影，
`_require_routes_available` 在 contributor 初始化前检查 route 冲突。可选 contract 语义保留。

**身份：内容、来源与运行时 incarnation 分开。** `resolve_web_module` 的 JS/CSS/contract byte
digest 只回答制品内容；当前 `build_web_ui_catalog` 的 identity 已包含 registration UUID、plugin id、generation id
和这些摘要，并非“只 hash bytes”。registration UUID 已是现有登记记录的单一事实：同一 Context
重新登记会生成新 UUID，catalog projection、live initializer、host 和请求 fence 只读取实际登记记录，
不另造 activation 表或旁路状态：

- Web 在 `Ui.register` 完成 provider/owner/path/contract 校验、创建 `Registration` 前生成一次，
  写入该记录；`Ui.register` 是唯一 writer，Effect cleanup 只删除自己这条记录。catalog projection、
  live initializer、host 和请求 fence 都只读该 UUID、generation、activation 与制品摘要。
- Mobile 在 `MobileUiSlots.register_mobile` 完成 asset 和 callback 校验后生成一次，写入现有
  `MobileUiBinding`；该注册函数是唯一 writer，`_register`/cleanup 不另造 identity。provider、
  catalog、asset、query 和 cancel 只读实际 binding。
- Web 的 artifact catalog bytes 仍是现有排序模块、原始 bytes、hash 与 contract 的 wire payload；
  artifact digest、payload identity 和 registration-aware runtime fence 是三个不同事实。正式 live
  Root 由 `agent/plugins/manager.py:802` 以 `plugins-live:` 加随机 token 创建；现有 wire 的
  `snapshotId` 使用 `root.generation_id` 作为 Root incarnation。现有 `catalogId` wire slot 继续
  是 opaque aggregate fence，固定 JSON 元组按 `plugin_id` 排序：
  `(plugin_id, registration_uuid, generation_id, module_sha256, stylesheet_sha256 or "", contract_sha256)`，
  对 UTF-8 JSON 做 SHA-256；不新增 schema 或字段，不使用 Root 全局 revision，也不拉回 SnapshotStore。
- Mobile 的 `plugin_revision` 对固定 JSON 元组
  `("mobile-ui", root.generation_id, plugin_id, registration_uuid)` 的 UTF-8 编码做 SHA-256，
  不依赖不存在于 `PluginRuntime` 的 `source_revision`，也不另造 generation cache；asset byte sha
  仍只回答内容。`catalog_revision` 仍是实际目录 payload 的 hash。现有 ticket/header 字段保持
  opaque，不改签名或 schema。

同一 `Context` 在相同 activation 内重新登记必须得到新 registration UUID；同一 Fiber 新 activation
也必须得到新 runtime fence。无关 UI 插件的登记变化只改变 UI aggregate，不重启无关 backend；无关
非 UI 插件不得改变 UI fence。Root incarnation 由现有 `root.generation_id` 表示，不能用 generation
以外的全局 revision、byte hash、archive provenance 或 per-request UUID 冒充它。

**Web owner 与 socket 关闭前置条件：** `Ui.bootstrap/state` 必须先进入 UI `Context.runtime_scope`
再执行业务读取。`DashboardBinding`/resolver 必须携带现有 Registration 的原始 contributor
Context；resolver 调 `ctx.require_runtime_owner(UI, ctx.require(UI))`，ASGI app 是可执行插件代码，
不能在 scope 外裸调用。Core 只提供已经实现的 `RuntimeScope.wait_admission_closed()`：它只等待
该 scope 所持 OwnerCall 捕获的 activation-local Event，不调用业务回调、不取得第二份资源所有权，
也不改变正常 HTTP/query 排空。

```text
Fiber 撤 token / UNLOADING → admission Event.set → 现有 consumer 与 OwnerCall drain
                                      └→ Dashboard future monitor wait
                                           → cancel app → host join → app scope release
```

实现边界是：Fiber 在撤 token、置 `UNLOADING` 后同步 set Event，再执行现有
`_owner_became_inactive → consumer/call drain → children → STOPPING → Effect` 顺序；Event 引用随
OwnerCall 及其 `_retain` 固定在原 activation，重试卸载不 clear，下次 activation 新建 Event。Event
不是新接纳条件，monitor 读到它也不能得到 `_current_runtime_scope`、capture 或执行许可；scope
close 只结算 permit，已 close 的 scope 仍可观察原 Event。

Dashboard 已接线：从现有 Registration 取得 exact Context，UI lookup 只做 catalog/binding/fence
选择与 target capture；stale/forbidden 的 status/code 决定在 scope 内形成，实际拒绝 wire 在 scope
退出后发送。已准入 request 在 target scope 内 capture 给 app child；独立 monitor 等同一 scope 的
撤接纳 Event，返回后 cancel app，host 负责 cancel/join app 与 monitor。create_task 失败或 app
首指令前取消时，创建方 close 未 enter 的 captured scope；已 enter 的 scope 只能由 app Task 的
finally close。正常 app 先结束则 cancel/join monitor，不等待未来 unload；反复取消不能提前释放
物理工作，不存新 socket/owner 表。Web/Dashboard consumer 已接线，R3 生产接线与 A1-A3 测试源已静态通过；
行为测试未运行。Mobile consumer 同批已迁移到 async host 与真实 Root/Context scope；R2 已由主审与独立只读复核静态通过，测试未运行。

**Mobile async 自持 scope 与结算：** `PluginMobileUiProvider` 只接受当前
`CompositionRoot`。`catalog`、`asset`、`query` 先从该 Root 的 `UI_SLOTS` 取得真实 provider Context；
catalog 对每个当前 ACTIVE registration 进入其贡献 Context，asset/query 也在目标 scope 内调用同步
`available`。false 会从 catalog 省略并让 asset/query 明确不可用，不读资产或提交 handler；callback
异常保持原样传播。asset/query 固定目标 binding、核对 registration-derived revision，再进入目标
Context；外层 client request scope 只是 client boundary，不是 UI 授权。没有 Root snapshot、lease、
fork 或同步/异步双模式 fallback。

```text
Root owns executor
    └─ UI registration lookup
         └─ exact contributor permit
              └─ capture / child Task
                   └─ physical thread
                        └─ release → registration Effect
```

`plugins/ui/mobile.py` 的 `_registrations` 是唯一登记真相；每次注册只生成一个
`uuid4().hex`，binding 保存原贡献 Context，Effect 关闭只删除自己的记录。同一 Context 关闭旧记录后
重新登记得到新的 UUID；同 Fiber 新 activation 也得到新 revision。`available` 仍是同步 callback，
但只在实际目标 Context scope 内运行。

query 的第一步是冻结实际 registration、generation 与 permit，再创建 coroutine/Task；已由同一 Task
保留目标 permit 的旧请求在 UNLOADING 中完成旧 handler，新 Task 的 stale/non-ACTIVE 请求明确失败。
每个请求必须按以下表一次结算：

| 阶段 | owner / 状态 | 失败或取消时的唯一结算 |
|---|---|---|
| slot reserve | provider 的 `_admission_lock`、`_admitted_queries` 与 scheduler request map | 未 reserve 不释放；已 reserve 只能由 query done/取消路径释放一次 |
| target admission/capture | 实际 MobileUiBinding 的 Fiber/activation 与 request scope | capture 失败退还 slot；不能留下 partial OwnerCall |
| coroutine/create_task | `PluginMobileUiProvider.query` 的 child Task | 同步 create 失败立即退还 capture 与 slot；首条指令前取消也走同一 done 结算 |
| executor submit | provider 已登记的 physical query 与 bounded ThreadPoolExecutor | submit 失败保留可观察错误并释放逻辑 admission；不得伪造成功 |
| timeout/caller cancel | scheduler/provider 只撤销等待方并进入 `_draining_queries` | shield/取消不能提前释放目标 permit；必须等待物理 worker settlement |
| physical thread finish | `run_in_executor` wrapper 完成、取回异常/结果 | 线程真正返回后才从 draining/admitted 移除 |
| final release | query Task done callback、OwnerCall、slot | release 与 map 删除幂等一次；provider `aclose` 关闭新 admission，累计取消并等待旧 work 与 executor shutdown；取消先发生而 shutdown 再失败时同时保留两类错误，失败保留同一 owner/retry |

**本批 Web/Dashboard 与 Mobile 实现、删除合同（Dashboard R3 生产与 T98 A1-A3、Mobile R2 静态通过；测试未运行）：**
本批盘点并迁移 Web/Dashboard 与 Mobile 的真实调用面：

- 登记与协议：`plugins/ui/plugin.py` 的模块级 `apply` 与 `Ui.register/catalog/bootstrap/state`，`plugins/ui/mobile.py` 的 `MobileUiSlots` construction、
  `register_mobile/_register`，以及 `agent/plugin_composition/ui.py`、`ui_slots.py` 的 protocol、
  record 和 binding 字段；
- Dashboard：`plugins/ui/dashboard.py` 的 resolver/build/closeables、`plugins/ui/web.py` 的
  `resolve_web_module/build_web_ui_catalog/_validate_web_contracts`、`agent/plugins/dashboard_host.py`
  的 live registry lookup、middleware、`_web_request_matches`、HTTP/WS app task，及
  `bootstrap/dashboard_api.py` 的真实 construction；
- Mobile：`MobileUiProvider`、handler/thread/query admission、Mobile wire fence、realtime
  channel/provider 的 catalog/asset/query/cancel 已迁移到 async host 与真实 Root/Context owner；
  gateway、ticket、设备 gate 和 wire shape 不变；本批只记录源码与测试源静态状态；
- wire 与测试：Web `checkCurrent`、headers、WebSocket identity，Mobile `plugin_revision`/
  `catalog_revision` 与 ticket 首先按 opaque 字段检查，只有源级契约证明需要时才改 passthrough；
  现有生命周期/客户端测试必须观察真实 Task、listener、线程、owner 和 wire 结果。

本批已删除 Web/Dashboard 直接调用点上的 whole-Root UI seal listener、`_catalog` 第二投影、
旧 snapshot lease 的 construction/read、以 `snapshot.accepting_leases` 停止轮询及只为旧 preparer
服务的分支；Mobile 同批删除 `FrozenMobileUiRegistry`、`_frozen`、`seal`、host 的 snapshot/lease/
`current_snapshot`/`source_revision` 和同步 catalog/asset 双模式。不能删除整个 SnapshotStore、
历史读取或其它仍有真实消费者的 candidate/freeze 结构，除非其所有直接消费者已经在同一批完成对账。

**确定性 oracle（仅设计）：** 三个入口分别执行 `available`、false/异常回执、同一 Context
re-register、同一 Fiber 新 activation、无关 UI 与
非 UI identity、旧 HTTP/async query drain、物理 timeout thread、精确
`RuntimeScope.wait_admission_closed()` 驱动的 socket cancel/join、新请求拒绝、owner permit 下
`available` 同步 true、合法 UNLOADING retain、query 的 reserve/capture、
create_task 与首指令前取消三种早失败、app/resource close failure 保留 owner 并 retry，以及原有
wire/auth/ticket/bytes contract；Root shutdown cancellation/retry 与 stale registry projection
也必须可观察。DSH 固定 commit `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8` 只可
作为 Provider/Fiber shape 对照，不能证明 Akashic OwnerCall 或 UI fence 等价。本批已实现 Core
等待原语并接入 Web/Dashboard consumer；T-7aeefe 的 Mobile R2 源码与测试源已定点修订，补齐
取消后二次等待、真实 timeout、取消与 shutdown 失败组合及 finally 清理保留，仍待独立 review，测试未运行。

#### T03a-Models · M2 当前合同（T-c46d11 review 未通过；T-59017c 定点返修待复核）

这一节取代此前把 Models 拆成多张运行表、把 M1 旧缺口继续写成当前事实的描述。现在唯一的
registration record 同时持有实际 driver `Context`、definition 和 registration UUID；`driver_id` 索引
只指向当前 record。registration Effect 负责登记和撤销，撤销时先删除仍指向原 record 的索引，再清理
原 record 的 auth attempt；同名新登记不得接管旧 attempt。Models 自己持有一个 instance UUID，只有
实际产生 binding/descriptor 的 execution、embedding 和 descriptor 路径按选中的
`(driver_id, registration UUID)` 集合生成 namespace；settings/discovery 不凭空生成 runtime namespace。

**当前唯一连接 owner：** `_DriverScope` 先按 selected connection 去重，进入每个实际 driver
Context 的 `runtime_scope()`，再为每个 selected connection 预登记一个 driver-owned Effect。Effect
setup 只返回 cleanup，不等到 `open()` 返回后才登记；因此首次 `open` await 期间以及同一 scope 的后续
connection 仍由固定 driver owner 保护。holder 只保存成功返回的 `DriverConnection`；open 失败没有外层
半资源，driver definition 自己负责其内部失败结算。

```text
Models Fiber -> ModelsState/index
      │
      └─ selected driver Fiber
           ├─ registration Effect -> ModelsState driver registration
           └─ per-operation connection Effect -> DriverConnection.close
                                      (失败保留在原 driver Fiber)

Models auth cleanup ─────────────────────────────── independent of connection scope
```

execution、embedding、model check、new/updated connection probe 和 first-model check 都先冻结
selected connection，再进入 `_driver_scope`；`definition.probe` 也在 selected driver scope 内执行。
`_sync_models` 与 `_discover_new_connection` 的真实 `definition.discover` 同样只在选中 driver
Context 的 `runtime_scope()` 内执行，enrich/CAS 留在 scope 外；异常和取消都必须释放该 owner。
连接关闭按 Effect 逆序尝试全部资源，然后释放 driver scopes；清理直接调用 `DriverConnection.close`，
复用现有 Effect lifecycle binder、shield、重复取消等待和失败 Effect 保留语义，不调用 Core 的
`DriverConnection.aclose()` 子 Task 包装。一个 close 失败不会跳过其他连接，也不会移除失败 owner。
config 可读性和 vision binding 仍由 Models 本地 store/选择边界校验；enabled connection 的
`open+close` 只在真实首用或显式 settings probe 边界执行，不再作为整 Root seal 预检。

`_sync_models` 与 `_discover_new_connection` 的真实 `definition.discover` 只在选中 driver Context
的 `runtime_scope()` 内执行，enrich/CAS 留在 scope 外；异常和取消都必须释放该 owner。

`ModelsStore.read_snapshot` 与 `StoredSnapshot.revision` 是普通 SQLite 设置/CAS 真相，不是运行期
owner；保留它们。候选 settings source 已删除，`plugin_update` 只在自己的 validation boundary
解释更新输入，不再复用 Models 的第二个 settings store。

**既有持久身份与 M2 runtime namespace：** `state.py` 的 `_binding_id` 把
`plugin_snapshot_id` 放进 binding hash；`store.py:336-342` 把 descriptor 写入
`model_calls.binding_json`；`projection.py:46-63/219/315/337` 比较 model facts、continuation
和当前 binding；`compaction/message_summary.py:186` 用 binding 相等性避免同一模型重复。因此
`plugin_snapshot_id` 不是显示字段，不能改成 RuntimeCatalog 的 `snapshot_id`，也不能静默清空旧
continuation。保持现有 wire shape 和历史 JSON，新的不兼容 continuation 必须 fail-loud。

M2 只引入一个简单 namespace，不引入 Core activation token 序列化、owner UUID map 或第二套生命
周期表。`ModelsState` 持有一个 `uuid4().hex` 的 Models instance UUID；每次成功的 driver registration
分配一个 registration UUID。namespace 是 Models instance UUID 加上按稳定顺序排列的 selected
`(driver_id, registration UUID)` 集合的 hash。相同注册集合在同一 Models instance 内稳定；同一 definition
关闭后重新注册会得到新 UUID；无关 plugin update 不改变它；同一 Context 可以登记多个 driver ID；不
生成或持久化 ownerID。

完成 M2 仍保留 `agent/plugin_composition/models.py` 的 `EmbeddingSpaceDescriptor.identity`、schema、
descriptor、Message/model_calls 旧 JSON、CredentialHandle 和 cache；不做 history migration，不以
namespace 变化静默清理 continuation。embedding 持久绑定仍由 model/space identity 和 dimensions
解释。

**本批可审代码单元：** T-c46d11 的真实 facade、complete/embed、hard-dependency reload、idle dispose
和 expiry/finish Event 链总体静态成立，但整批 review 确认了两个失败断言与四个关键 oracle 空洞。
此前定点测试在同一 `CompositionRoot + PluginRuntime + driver Fiber/Effect` fixture 中覆盖异常组内层
`RuntimeError`、旧 auth Effect 的首次失败/显式重试计数、同 Context 重登记、历史 `model_calls`
完整记录与 credential payload；T-59017c 另外修正 execution 内嵌套 embedding 的设置冻结 oracle。
源码已通过 AST/内存 compile 的初步检查，但尚未执行，定点返修仍待主审复核，不能写成运行验收或行为通过。
除这些测试与本状态文档外，不改 Models descriptor、Store schema、Message、凭据或 durable data；DSH
固定 `c389f96b` 只作本地 Fiber/Effect/Loader 对照，不能抹掉本项目的 model-call、continuation
和严格 cleanup 责任。

另核：`turn_scope.py` 只有 TurnExecutionScope ContextVar（Prompt/权限/副作用），无 snapshot 读取，不列入迁移。

**T03a-1/-R1 已落地并经 Codex+独立静态复核通过（测试未运行）与撤销的方案**：

- 已落地：非 Root Fiber 每次 `_load` 创建新 `Context` 并在 apply 前成为 `fiber.context`（context.py:1011-1013）；`Context._require_current`（`fiber.context is self` 对象身份，不比较 token/ACTIVE，避免误拒排空期已保护工作）统一守住全部能力入口——直接检查 `fiber`/`runtime_scope`/`capture_runtime_scope`/`get`/`mount`/`effect`/`provide`/`health`/`on`/`spawn`，事件分发 `emit`/`serial`/`parallel`/`transform`/`observe`（堵住旧 Context 直达 Root 事件注册表的泄漏），以及 `report_incident`/`require_runtime_owner`；`require`/`inject` 经 get/mount 委托覆盖；`runtime`/`config`/`data_root`/`workspace_*`/`diagnostics` 等不可变 Root/Fiber 元数据不扩大限制。排空期 fiber.context 不变，已接纳调用与 Effect 清理仍读自己 activation 的 dependency_store；Root Context 身份稳定。事件注册表、分发顺序、listener 注册与错误语义、`parallel` 的 per-listener Task 创建均不变——检查只加在各分发入口进入注册表之前。
- **撤销** T03a-0 原提议的"enter 时才 acquire_call"整体迁移方案：它无法在 capture→child enter 之间保护资源，且会把已接纳工作的嵌套使用误当新调用拒绝。
- `require_runtime_owner` 的现状与后续约束：现在在读取 service 状态前先拒 stale Context（`STALE_ACTIVATION`），再要求当前 OwnerCall/生命周期借用、同 Root dependency 与 service identity；Root fallback 对普通 service owner 仍是合法合同。只有有额外信任边界的能力（当前为 Manager 的 `read_runtime_catalog`）在能力入口明确检查声明依赖并返回 `UNDECLARED_SERVICE`，不把该门闸泛化到所有 service。`context_owner` 的 legacy ACTIVE 限制仍是独立遗留路径，本步未迁移；生命周期借用与后台 Task 资格继承仍需完整迁移设计。
- 实现批准表述纠正：不是"只重写 context 四函数+tasks 就不动其它消费者"——bindings.py:158、Manager（manager.py:415/462/518/1456）等仍直接 `RuntimeScope(lease)`，其它调用者直读当前 snapshot；后续公共 scope 切换必须有完整兼容/切换清单，不能半改调用链，不引入 legacy/local 双模式。

**生命周期歧义收敛**：`_load` 顺序确定为 apply → 本 owner `RUNTIME_STARTING` → `RUNTIME_STARTED`（两者均属本 owner 初始化，不重复调用 start）→ required health 检查 → ACTIVE/唤醒；不得在最后那个真实 start 执行前先报 health 通过。初始化失败先清理已获资源：清理成功→FAILED，失败→UNLOADING 保留资源/依赖（同 PLG-006）。activation token 是身份，Fiber/内核是执行 owner；同 activation start/stop 不双发，Manager 不持另一本 started 表。冷启动最终与局部换代共享同一 owner 生命周期；旧冷启动链仅为迁移前事实。"Loader 局部装载标记"降级为未审提议：优先做未接线内部能力、T05 一次切消费者；若临时标记不可避免，须列字段/唯一写者/消费者/移除 diff 范围。

**三个例子调用链**（批次 A 内核已接线，公共消费者迁移仍归 B）：
1. 普通工具调用（tools → reply）：调用点 `async with ctx.runtime_scope()` 对本 Fiber activation 取 `OwnerCall`/`RuntimeScope` 保护；require 走本 activation dependency_store；owner 换代后旧 Context 在捕获处 `STALE_ACTIVATION`，不进入新 activation。
2. Shell 单 owner cleanup（shell.py:370）：注册时固定原始 Shell Context、ShellOwners 与 Shell `TaskAdmission`；清理 Task 由 Shell admission 接纳，raw child 用 `capture_runtime_scope()` 保护同一 Shell scope；create_task 失败显式 close，普通 cancel/pause 仍等真实清理，只有明确 abandon 释放 caller waiter。
3. Akasha startup（plugin.py:521/532）：内核 `_load` 持 transition 发 RUNTIME_STARTING/STARTED；回调内 `ctx.runtime_scope()` 经生命周期 binding（Context+实际执行 Task）授权（T03a-5-R1 合同）；watcher spawn 的后台任务经 `_activation_ready` 闸后自行走正常许可路径，不冒用生命周期授权。

#### T03a-2/T03a-3 局部 RuntimeScope：合同已撤回，原子保留原语已落地

**T03a-2 原合同经 Codex + 独立复核未通过，其 scope/facade/tasks 接线提议已撤回**（原表述见 T-3b9c4e 任务记录——该稿从未 commit）。以下事实纠正替代原方案，不留会被误实施的描述：

1. capture 无条件 `acquire_call`（要求 ACTIVE）会在排空中拒绝已保护父调用的 Shell cleanup（shell.py:370-400），本批不采用。保留已受保护资源：Shell cleanup Task 复用注册时固定的 Shell `TaskAdmission`，不在 caller scope 重开 `TASKS`，不重新接纳历史 Tools binding；普通 cancel/pause 等待该 Task，明确 abandon 才只释放 caller waiter。
2. "Context facade/tasks 接新构造、同时保留 `RuntimeScope(lease)`"不是未接线能力——facade/tasks 本身就是生产入口，构成长期双模式，撤销。
3. T03b-Tools 已将 tools/plugin.py 的 target open、参数准备和授权回调接到各自真实 Context scope；T-9f83dd 修复 Channel D3 的终态结算并补齐指定局部 scope/validation 测试源，但这不等于 bindings、scope 外 manager 消费者或其他 provider 的 service 入口接纳已完成，不得把局部接线写成全链路闭合；本轮也不加全图可达性缓存或扫描。
4. `Effect.aclose` 实际在 effect.py:74-77 `create_task(self._close())`——cleanup 运行在独立 task，不是 `fiber._transition_owner`；因此批次 A 固定零参数 binder，在该 `_close_task` 内按实际 Task 建立并 reset 窄生命周期借用，不放宽 Task 身份门。
5. 取消事实按代码写：`Tasks.Task.cancel`（tasks.py:128-136）未运行时只置取消标记、不 cancel 底层 Task，`_run` 进 `finally` 释放 scope；Shell `run` 创建失败显式 close、parent 取消则 shield 等 child 完成。不存在通用"外层 close 未 enter child"兜底，也不把直接取消私有底层 Task 推广成公共合同。

**T03a-3 已落地并经 Codex+独立静态复核通过（测试未运行）**：私有 `OwnerCall._retain()`（context.py）——从仍被当前 Task 持有、未 release 的许可派生同 Fiber、同 activation、独立 `OwnerCall` 的新许可，登记进同一个 `_in_flight_calls`；源已 release→`OWNER_CALL_RELEASED`，当前 Task 不是源 entry 的 owner（含无 Task）→`OWNER_CALL_CONTEXT`，全部检查先于新增；不判断 ACTIVE、不比较 Fiber 当前 token——UNLOADING 期间源许可仍受保护，故仍可 retain；源/派生许可各自独立 release，最后一份释放才 `_calls_idle`。`_retain` 是 Core-only 原语，不提供跨 Task 转交、不承诺公共 API；批次 A 的 `RuntimeScope` capture/child 交接已消费它，tasks.py、bindings/manager 直构点仍归批次 B。

**T03a-4/-R1 已收敛为批次 A 的唯一 `RuntimeScope`（测试未运行）**：此前私有 `_CallScope` 只用于静态验证，现已删除其类名，不留双构造。`RuntimeScope` 构造消费当前 Task 独占、未释放的一份 OwnerCall（来自 `acquire_call` 或 `_retain`）；移交后调用方不再释放/再次移交同一 call——这是 Core 调用方遵守的私有合同，普通违约由既有 released/in-flight 检查 fail-loud，不加反向指针/票据/所有权表；`capture()` 同步 `_retain`，返回前保护即存在、无 capture→enter 空窗；`__aenter__` 同步完成"许可在途未释放 → `_adopt_call` 把记账归属改当前 Task → 绑 ContextVar"，不判 ACTIVE、不比最新 token；`close()` 幂等——未 enter 对象可显式关闭释放自己的许可，已 enter 只能由 entered_task reset+释放，错误 Task 在任何状态变更前被拒。`_current_runtime_scope()` 按 `entered_task is current_task` 过滤，ContextVar 隐式继承不算授权。原生 Task 首指令前取消由创建方显式关闭未 enter scope；本类不建 Task、不加 watchdog。`_adopt_call` 仅为 scope enter 服务，非通用 Task 转交 API。跨 owner 借用、生命周期/Effect 清理归属、UI fence 身份仍是后续迁移缺口。

**后续接入点（批次 B/C，范围须按真实消费者重新定）**：provenance/UI 来源、`RuntimeScope(lease)` 直构点（bindings.py:158、manager.py:415/462/518/1456/2690/2750）与剩余旧 snapshot 读取者（channels/provider.py:940 等）的完整切换清单仍归 T03b/T05；已删除的 admission lease 不再是当前合同；批次 A 已完成 capture→child Task 交接、生命周期 binding 与 Effect 清理归属。

**批次 A 测试源（只写不运行）**：精确旧 activation 拒绝；无关分支不被占用；排空中已接纳许可仍用旧依赖；retain 的释放顺序/排空期/非持有者拒绝；`RuntimeScope` capture 无空窗/未 enter close/运行前取消与 ContextVar 继承不授权；LOADING 借用与 native child 不冒用；startup/health 完成后才 ACTIVE；Effect 独立 close 与业务主动 `aclose` 共用窄 binder；STOPPING 失败保留资源且 retry 不重发；同插件不同 Fiber 的 owner 精确 dispatch。测试未执行。

#### T03a-5 局部 scope 与生命周期接入合同（R1 已通过；批次 A 已实现，测试未运行）

**（R1 已通过；T03a-5 首稿未通过。本节替换首稿全部规则与 §9-T02 早期"启动/健康屏障接入方案"、"三个例子调用链"的示意表述——同一决定的早期草稿不再各自成立。）**

**主合同（R1）**

1. **许可表改为可返回真实许可**：`Fiber._in_flight_calls` 由 `dict[call_token, Task]` 改为 `dict[OwnerCall, Task]`，删除 `_call_token`；`_begin_call`/`_retain_call`/`_adopt_call`/`_end_call`/`_reject_self_call_wait` 共用这一张表。`Fiber._call_owned_by_current_task()` 扫该表返回当前 Task 实际持有的 `OwnerCall`（或 None）——这是嵌套 scope 的事实来源，不凭 token 重建许可、无第二索引。
2. **`runtime_scope()` 规则（顺序固定）**：先 `reject_executor_context_access`+`_require_current` 拒 stale Context → 当前 Task 持匹配本 Context 的生命周期 binding → 本 Task 已持有本 Fiber 的 `OwnerCall`（经 `_call_owned_by_current_task`）→ `call._retain()` 派生 `RuntimeScope` → 当前 activation ACTIVE → `acquire_call` 正常新接纳 → 非 ACTIVE 且无已有保护 → 拒绝。**不是"无许可一律拒"**：akasha `follow`（plugin.py:519-523）后台每轮 scope 本来就是第一份新许可，ACTIVE 新接纳是正常路径。
3. **生命周期借用绑定实际执行 Task，不靠 ContextVar 继承**：binding 携带二元事实 `(Context, Task)`，仅当 `task is asyncio.current_task()` 且 `context is self` 才借用。由执行方显式建立并 `finally` reset：`_load`/`_unload` 在 transition Task 内各回调段建立；`Fiber.add_effect` 在创建 `Effect` 时固定当次 Context 的零参数同步 binder，`Effect.aclose()` 保持无参数，而 `Effect._close` 在自己的 `_close_task` 内调用该 binder 建立 `(context, current_task)` 借用并 reset；不声明"自动继承即授权"，不建全局授权表/任务图/公共授权框架。Context 是每 activation 身份：`_unload` 已把 token 置 None，binding 比对的是 Context 对象身份而非当前 token，故旧 cleanup 可由它自己的 Context 授权、不会误授给新 activation。原生后台 Task（subagent/plugin.py:153 在 RUNTIME_STARTED 内 `asyncio.create_task`、channels/provider.py `_spawn_owned` :1070-1085 不经 `ctx.spawn`）即使继承了 ContextVar 值也因 `task is not current_task` 不获权；`capture_runtime_scope` 不复制 lifecycle 借用，detached 工作只从已接纳许可 retain（shell.py:353-375 形态）。
4. **activation-local 生命周期事实**：每次 activation 的事实挂在该 activation 自己的记录上（`_activation_ready` Event、`stopping_completed` 标志，随 `_load` 重建），不设 Manager 侧第二本 started Root 表；旧 activation 在 UNLOADING/清理失败后仍由同一 Fiber 的 activation 记录与保留的 `dependency_store` 引用。`stopping_completed` 只在本 owner `RUNTIME_STOPPING` 串行 dispatch 成功且非 Bail 后置位；STOPPING 本身失败则在 Effect 清理/依赖清空前传播，保留 owner/资源/固定依赖，显式 retry 只重试失败的停止阶段——是既有严格 cleanup retry（核对 manager.py:508-531、effect.py:80-105），不是业务回滚；不用 dispatch 前的 `stopping_started` 提前置位导致失败被永久跳过，也不聚合错误后继续释放依赖资源；先成功 listener 在重试中可能重复，沿用 serial 现有合同，不新增逐 listener 事务。
5. **spawn 闸覆盖真实取消路径**：`ctx.spawn` 的 wrapper Task 立即创建返回（cancel/await 约定不变），运行用户 coroutine 前 `await` 本 activation 的 `_activation_ready`（捕获的当次 Event，旧 coroutine 不会等到新 activation ready）。wrapper 首指令前被原生 `cancel()` 时 `finally` 不执行——故结算事实最少且一致：Effect cleanup `cancel`+`join` wrapper 后，若 wrapper 未运行用户 coroutine，由 cleanup 侧负责 close 该 coroutine；wrapper 正常路径的 `finally` 用同一"是否已运行"事实防双关/漏关，不加 watchdog，后台 watcher 不持永久 OwnerCall。消费者事实分三类：启动 watcher（akasha:539、scheduler:127、wake:131 等，stop 中 cancel/await）；运行期工作（plugin_update/latest.py:159 `finish` 等，不属启动闸语义）；原生 Task（subagent:153、channels `_spawn_owned`）不经 `ctx.spawn`、不受闸保护，其入口接纳需求单列边界，不重写所有业务插件。
6. **顺序**：`apply → RUNTIME_STARTING → RUNTIME_STARTED → 本 owner required health → ACTIVE(ready.set + _owner_became_active)`；LOADING 仅 owner-self provide 可读（`Context.get`/`require` 允许 `provider.owner is self._fiber`），跨 owner 仍 ACTIVE-only；启动失败清理成功→FAILED、失败→保留 owner/句柄/固定依赖且不可用、不自动回旧。

**生命周期事件接线**：`EventRegistry` 新增按 owner 过滤的串行 dispatch（`listener.owner is fiber`，绕过 `_active_listeners` 的 ACTIVE 过滤；顺序/Bail/错误语义与 serial 相同）；`_load` 在 apply 后对本 Fiber dispatch STARTING→STARTED，`_unload` 在 Effect 逆序清理前 dispatch STOPPING（依 `stopping_completed` 幂等）；同一 activation 只发一次。跨 owner（A→B→C）按实际调用边界取得各 owner scope：本批 T03b-Tools 已保护 Tools、target、prepare、authorize 四个真实 owner；T03b-Content 的 bind 只快照 ACTIVE 定义，按首次出现顺序去重取得贡献者 scope，动态 prepare 在全部 scope 取得后执行，视图关闭后按贡献者逆序释放、Content scope 最后释放。剩余 service 入口与其他 provider 仍归后续批次；不发明 service 代理/可达性缓存。

**实际迁移表（scope/fence/snapshot 全部真实读取者）**

| 组 | 真实消费点 | 处置 | owner/替代合同 | 前置与验收 |
|---|---|---|---|---|
| 调用保护型 scope | `runtime_scope`/`capture_runtime_scope` 全部调用点（content:320、wake、materials、reply、computer、markdown_memory、scheduler、subagent、delivery_policy、delivery、mcp、plugin_update、senders、stable_view:129-132、bootstrap/tools.py:178-193 recover_input、akasha、tools:470/497/547 等）；tasks.py:62/152-160 Task 复制；manager.py:415/462/518/1456/2690/2750 `RuntimeScope(lease)` 直构 | 保留迁移 | facade 当前返回唯一 `RuntimeScope`；Task 直构已由 B1 改为 capture/enter，Manager 等直构归 B2 删除或迁移 | 批次 A 验收=scope 不再由 facade 引用 lease、跨 activation 旧 Context 拒绝；剩余直构迁移归 B2 |
| 运行期 fence 与内存键 | channels/provider.py `_bindings`/`ChannelBindingLease`（:90/:161-262/:882/:942-1031/:1155）；bindings.open（bindings.py:150-165）；executor.py:116-118 `get_current_runtime_snapshot`；validation.py:97-107 候选路径 recover_input | 保留迁移 | fence 键改 binding owner 的 (fiber, activation) 身份；`executor`/`recover_input` 的 Root 解析改走当前公开入口（T05） | fence 迁移与 facade 切换同批；验收=旧 activation 的 binding/envelope 被拒；持久归档身份不变 |
| UI 请求 fence（非展示） | ui/plugin.py:135/141 当前读取不存在的 `scope.snapshot_id`；webHost.ts:93-107/225-241；dashboard_host.py:327-342；mobile_ui.py:172/217/274（query lease/capture/`get_current_runtime_snapshot`） | 保留迁移 | Web snapshotId 采用 `root.generation_id`，catalogId 采用 ACTIVE registration 固定 JSON 元组 hash；Mobile plugin_revision 采用 `("mobile-ui", root.generation_id, plugin_id, registration_uuid)` 固定 JSON 元组 hash；同制品新 activation 必须区分，不恢复 SnapshotStore 或全局 revision | 归 T03b/T05；启用验收前完成，不以旧 lease 假闭合 |
| 持久事实不变 | bindings 归档 metadata/root_ref/service；durable handoff_id/channel/session/message 身份 | 纯持久，不改 | `ChannelBindingLease` 当前是运行期对象；表中“持久字段”不是已证实的存储 schema，不在批次 A 发明迁移 | 无已证实变化不迁，不发明"全图冻结 ID" |

**实施批次与最终启用（R6）**

候选构建路径必须先行移除或替换：Manager 仍对候选 Root `mount`/`_mount_module`（manager.py:~2991 SnapshotSealing、~3286）且 source admission 拒绝 candidate starting（:1387/:1493）——`Fiber._load` 一旦接入 start 会提前启动/拒绝旧候选，故候选路径移除是启用前置，不以 legacy/local 标记保留两套生产模式。允许分批实现，未闭合批次如实标 WIP，不为中间 diff 造兼容层：

1. 批次 A-R1（内核修订）：保留批次 A 的 `dict[OwnerCall, Task]`、唯一 `RuntimeScope`、生命周期 binding、`_activation_ready`+spawn 闸、owner dispatch、`stopping_completed`、owner-self LOADING 读；补齐 STARTING 责任与 stale `spawn` coroutine 关闭，并修正内核测试 oracle。→ P1/P2 与 Root oracle 已静态通过，未运行测试。
2. 批次 B1（Task 消费者）：`tasks.py` 的 `Task.__init__` 只从当前真实 entered `RuntimeScope` 同步 capture，Task 结束释放局部许可；构造失败同步关闭 `_run` coroutine 和未 enter scope。→ 生产与测试源已静态通过，未运行测试。
3. 批次 B2/T03b 后续（剩余消费者与 fence）：bindings、scope 外 manager 消费者、Manager/UI/current_snapshot/candidate/freeze、RPC 与其他 provider 继续迁移，删除直构与旧 lease 路径；本轮 T03b-Tools 已完成 tools provider 的实际 owner scope 接线，T-9f83dd 仅修复 Channel D3 并补测试源，均待独立 review。
4. 批次 C（最终启用验收）：候选构建路径移除/替换、UI fence 替代落地、freeze 屏障解除、全部测试与 Gate；此前不声称闭合。Loader（T04）、公开入口（T05）仍按原 T02–T07 目标。

**确定性测试 oracle（只设计）**：STARTING→STARTED→health→ACTIVE 顺序与依赖序启动/反向停止；spawn 闸前 unload 关闭未启动 coroutine（含 wrapper 首指令前取消由 cleanup 侧 close）；生命周期 scope 仅在实际 callback Task 可用、`Effect.aclose` 的 `_close_task` 经窄 binder 获权、原生后台 Task 继承 ContextVar 也不获权；嵌套 scope 经 `dict[OwnerCall,Task]` 查得真实许可 retain；STOPPING 失败保留资源且 retry 不重发已成功 STOPPING；候选路径移除前旧候选不被 `_load` start 提前启动。

#### T03b-Channel · 局部 binding 与来源接纳合同（当前实现合同）

本节记录当前已经接入的 owner、生命周期和验收合同，不恢复旧 SourceAdmission 或全 Root
snapshot fence。生产持久身份、Message/schema、descriptor、`root_ref`、metadata、credential
和 durable handoff 不变；本轮只修复局部运行资源的行为边界。

**唯一事实与 owner 表**

| 事实/边界 | 唯一 owner | 合同与保留项 |
|---|---|---|
| 选择与变更范围 | `PluginSelection` + Manager | 只决定实际变更 Fiber/activation；不让 Channel provider 拥有全 Root selection、snapshot 或候选图。 |
| activation 与 binding 身份 | 贡献插件的 Fiber/`Context` + `PluginChannels` 的 `_ChannelBindingState` | 每次 activation 生成新的 `binding_token`；`plugin_context` 是真实 Context 身份。`snapshot_id`/`generation_id` 只保留在 wire、attempt、provenance 与 receipt 字段，不再是 graph selector、lease 或全局 fence。卸载先清 `activation_token`，UNLOADING 时不能拿当前 token 匹配旧 binding。 |
| 来源接纳、adapter 与在途 | 每个 `_ChannelBindingState` 与其 adapter | adapter 启动时保持 closed；绑定级 admission 先关，再 drain `in_flight`/`drain_event`，再执行 adapter stop。`ChannelBindingLease.aclose` 仍释放精确 binding 和实际传输责任；不能用删除 snapshot 字段换掉清理语义。 |
| 声明、registry、listener、adapter Effect | 贡献 Context 的 `Effect` | 注册物随 contribution Context 撤销；删除 Channel 对全局 `SNAPSHOT_SEALING/seal` 的依赖，不建立第二张 registry/owner 表。 |
| raw callback、request、commit | raw callback 的当前 Task + 精确 `plugin_context` | callback 当前 Task 先 `async with contribution_ctx.runtime_scope()`，再在已 entered scope 内 capture 给 commit 子 Task；创建失败或子 Task 首指令前取消时由创建方关闭未 enter scope，entered scope 由 enter 的 Task 关闭。不得把 entered `RuntimeScope` 塞进 `ChannelBindingLease`，也不得通过私有 `_call` 绕授权。startup 的 `open_scope` 只允许 `_start_binding` 的真实 `start_task` 做生命周期借用，raw listener 不继承该例外。`InboundEnvelope` 保留 exact binding、`handoff_id`、channel/session/message、immutable `Message`、descriptor、`root_ref`、metadata、credential、schema。 |
| durable inbound 与恢复 | `MessageBus` + `InputCustody`/`InboundHandoffStore`/SessionAdmissions | 普通 `prepare` 在 await 前登记、`CHANNEL_INPUT` commit 后关闭；durable 只在 exact handoff 完成后删除，失败保留 receipt/owner 并按现有 retry/defer 处理。Channel 不删除 durable 数据，不 replay 授权。 |
| outbound | Delivery/Senders 业务发送 + Channel provider 控制回执 | 不再经过已删除的 MessageBus 出站 queue；Delivery/Senders 负责真实业务发送，Channel provider 仍用 `ChannelBindingLease.deliver` 直发控制回执，capture 旧贡献 owner、结算 receipt 并保留失败证据，不把已接纳回执静默改为拒绝。 |
| identity、attachment 与 schema | 各自 `ChannelIdentities`、artifact/attachment store、Message/descriptor owner | 保留 `handoff_id/channel/session/message`、immutable envelope、`root_ref`、metadata、credential、schema 与 identity failure receipt；不把这些持久/业务事实移入通用 Core owner 表。 |
| host recovery 与局部 admission | 隔离 validation Root/Fiber；live Root 由 Manager 编排，Channel provider 持有 binding | `recover_input` 只解析当前 live/isolated Root 的 `CHANNELS`；局部 binding 自己关闭 admission、排空并 stop。HostInfo 的 `boot_id`/`validation` 是宿主事实，不用 root revision 冒充 boot identity。 |

`bus/event_bus.py` 曾有独立的 snapshot observer/queue lease 路径，但不直接消费 `CHANNEL_INPUT`、`PluginChannels`、`ChannelBindingLease` 或 adapter ownership；T-fbf63b 已删除这层耦合，generic queue/dispatcher 仍不承担 Plugin Context 的 owner 责任，不能用 Channel 合同替代 Root EventRegistry。

**正常输入、局部卸载与失败结算**

```text
贡献 Fiber apply
  │ Effect 注册 declaration/binding（closed）
  ▼
RUNTIME_STARTING
  │ 新 binding_token；factory/attach；adapter.start（仍 closed）
  ▼
ACTIVE 前的 ready gate
  │ ctx.spawn 等 Fiber ACTIVE + required health；成功后才 open 该 binding
  ▼
raw adapter input
  │ 校验 exact binding/token/admission → 当前 Task runtime_scope → capture 子 Task scope
  ▼
MessageBus.prepare ──> CHANNEL_INPUT commit ──> ordinary close
                              └──────────────> durable complete / retain / defer

局部更新/卸载
  │ 关闭该 binding admission（新输入明确拒绝）
  ▼
旧 envelope、lease、transport in_flight drain
  ▼
adapter.stop（其已接纳任务由 adapter 持有；成功后 listener settle）
  ▼
Effect/declaration close；新 activation 生成新 binding_token

任一 start/open/commit/stop 失败
  │ 保留 exact owner、failure receipt、incident 或 durable pending
  └─> 由同一 owner 显式 retry/defer；不暂停整个 Root、不借新 activation、
      不 replay 授权
```

ready gate 的取消在 adapter ready 前，必须由既有 `ctx.spawn` wrapper cleanup 关闭 waiter，binding Effect stop 关闭 adapter/listener；当次 ACTIVE 后，真正 open 仍须在贡献 Context 的 `runtime_scope` 内完成，并在卸载竞态中明确拒绝。open 失败则保持 closed 并报告 Channel domain failure，不能把后台 Task 失败冒充 readiness 成功或失败。普通 bus prepare/complete 只覆盖短的 input commit，不等待整 Turn reply；durable cleanup/retry 和 outbound provider receipt 是独立责任。

**推荐最小方案与 Core seam**

1. Channel provider 不拥有全局 selection、snapshot lease 或全 Root admission；它只维护每个贡献 Context 的 binding、adapter、listener 和在途计数。
2. 使用现有 Fiber/Context/OwnerCall/`RuntimeScope` 与 provider 已有 `_ChannelBindingState.in_flight`；每次 activation 一个新 token，Context identity 与 admitted protection 分离，不在 UNLOADING 比当前 activation token。`ctx.spawn` ready gate 足以表达 ACTIVE/health 后开放，不新增通用 `RUNTIME_READY` 或 graph lease。
3. 复用 `Effect.aclose` 作为 declaration/registration 的逆序边界，直接由现有 close Task、shield、生命周期 binder、失败保留和显式 retry 负责 `_stop_binding`；删除下一批中的重复 `state.stop_task`。adapter.stop 仍先正常收束 listener，成功后再取消剩余 listener；严格失败时保留资源，不删除 owner。
4. 当前方案使用既有 `Context.runtime_scope`、`capture_runtime_scope`、`OwnerCall._retain` 和 `ctx.spawn`：raw callback 先进入贡献 scope，再 capture 子 Task；未 enter scope 的创建方负责首指令前取消结算，entered scope 由进入它的 Task 关闭。已持有 request、attachment lease 和 adapter stop/finally 不再申请新的 activation scope；只使用原 owner 的在途保护和注入的 InputCustody。ready gate 不能绕过 scope，原生 listener Task 不能继承 start/cleanup 生命周期权限。

**本轮实现与剩余边界**

- `plugins/channels/plugin.py:apply`：移除 `SOURCE_ADMISSION` 注入与 `SNAPSHOT_SEALING` listener，登记 per-activation ready gate；`plugins/channels/provider.py:PluginChannels.__init__/register/_start_binding/_open_request_scope/_acquire_control_binding/_admit_inbound/_handle_control/_deliver/_stop_binding`：改用 local binding admission/context/token/in-flight，保留 exact envelope、durable、控制回执与 adapter stop 语义，并直接 await `_stop_binding`。
- `agent/plugin_composition/channels.py` 的 `ChannelBindingLease`、`InboundEnvelope`、`OutboundEnvelope`、`ChannelFactoryContext`：收敛为 exact binding contract，保留所有 wire/attempt/identity 字段和 `aclose` 责任；旧 `admission.py` Channel watcher/lease 路径已删除，当前 validation 只使用 HostInfo 与隔离 Root/Fiber。
- `bootstrap/tools.py`、隔离 validation Root/Fiber、live Root/Manager 与 `bus/event_bus.py` 各自保留自己的 owner；Channel recovery 只经当前 Root 的 `CHANNELS`，不复制 Root 或跨 host 读取。`bus/queue.py` 的旧出站 queue 已删除，EventBus 只保留 generic observer queue/dispatcher。
- 本轮只修复 Channel D3 的 durable defer/settle 取消结算、crash oracle、request/attachment/validation 测试源；D1/D2/D4/D5 生产结论保持此前静态通过且未修改。不删除 durable IDs、metadata、identity receipt、adapter stop failure owner，也不新增兼容层。
- 源级 oracle：同一 Root 替换贡献 Fiber 时新 binding token 与旧 Context 隔离；closed→ready→ACTIVE/health→open；UNLOADING 中已接纳 request/attachment/durable cleanup 可完成而新 work 拒绝；stop failure 保留 owner 并可 retry；crash edge 只有 `provider_started`，provider-calls 不存在；validation 只写隔离资源。以上均只做 AST/compile、diff/hash 静态检查，未运行测试。

**DSH 对照（c389f96b，只读）**：`vendor/cordis/src/fiber.ts` Fiber（:184）构造时创建 ctx（:236）、effect 经 `runner.collect` 收集 disposer（:230/:359-364）、dispose/restart 等 `inertia` 排空 PENDING effect（:294-313）、`:649 _reload` 复用同一 ctx 按 epoch 固定 store、`vendor/loader/src/config/entry.ts:291 _start` 只挂载并 `await fiber.await`、`reflect.ts:277 provide`/`314 notify` 按实际 inject 通知——可对照局部拓扑、effect 清理归属与初始化时序。**有意差异**：Akashic 每 activation 换 Context、严格 OwnerCall/在途排空/retain、`runtime_scope` 与 task-bound 生命周期借用均为本项目自有责任，DSH 复用 ctx 且无这些机制，不声称同款；不复制其 rollback/cache/loader.exit。

### T04 · 收敛为单插件 Loader，安装直接请求应用

**输入与下游：** 固定制品与安装选择 → T02/T03 的局部挂载，不再建立候选 Root。

**主要文件：** [manager.py](../../agent/plugins/manager.py)、[install.py](../../agent/plugins/install.py)、
[importer.py](../../agent/plugins/importer.py)、[selection.py](../../agent/plugins/selection.py)、
generation、archive 与 Python environment 的现有实现。

**当前源级耦合与唯一责任：** `PluginGeneration` 同时记录 `module_path`、`scope`、
`generation_id`、配置/数据身份、权威 `archive_ref` 和真实 `fiber` 关系。
Manager 的 `_active_generations` 是 live Root 唯一的 loaded-generation owner map；
`RuntimeSnapshot.generations` 仅是尚未迁移的旧消费者事实，不能再拥有或投影 live generation。
`_load_one` 只固定输入和写归档，静态编译覆盖归档内插件自有 Python 源码；选择提交后才由
`_archived_generations`/live load 导入固定归档并解析入口，随后真实调用 `apply`；
`_mount_generation_composition` 保存 `_mount_module` 返回的 Fiber。
`_dispose_generation` 先关闭 Fiber，再关闭 generation 的 factory、Scope 和 module；
旧 snapshot drain 若试图回收 live owner 会明确失败，而不是替换或删除它。

推荐单案是让现有 `PluginGeneration` 成为 loaded input/code owner，下一实现批次只为它
保留一个 `fiber: Fiber | None` 关系；不复制 archive/version、第二张 active 图或独立图。
Manager 仍是串行编排者，执行 generation 的永久 close，但 Root 只拥有拓扑和 Fiber，
不再拥有 module/importer/PluginScope 的永久清理。Fiber 的 activation unload/reload
由内核在同一 Fiber 上得到新的 activation Context/apply 关系；一次新的 generation load
不会复用旧 plugin Context。它的 Effect 可随 activation 清理，却不能负责删除 module。
永久移除的顺序是 affected Fiber 与 hard consumer 排空并 dispose 成功，再由同一个
generation owner 关闭 `PluginScope`、注销 importer/`module_path`；任一清理失败都保留
generation、module、scope 和明确的 retry owner。import/apply/start/health 失败也由这份
generation 负责失败清理；失败清理失败时不丢句柄、不回写旧选择。依赖暂时失去只走
Fiber `_unload`，随后同一 Fiber 可再次 activation，不触发 module close。

**本轮 Manager 主链实施：** 这是一次有真实调用者的 Manager 主链替换，入口是
现有 `reconcile_changed`/`_load_one`，不是先增加 Loader wrapper；完整切换前允许 WIP，
但不保留 local/legacy 两条生产分支。

| 当前对象/调用 | 保留的权威事实 | 同批删除的旧路径 | 目标唯一 owner/消费者 |
|---|---|---|---|
| `manager.py:_load_one`、`_archived_generations` 的 Root `_defer_internal_cleanup` | `PluginGeneration.module_path`、`scope`、`archive_ref`、`generation_id` | module/scope 由临时或整图 Root 兜底 | generation code owner；`Manager._dispose_generation` 是唯一永久 close 消费者 |
| `manager.py:_mount_generation_composition` 的 `_ = await root._mount_module(...)` | `_mount_module` 返回的真实 Fiber | 丢弃 Fiber、用 `runtime_snapshot` 代替局部 activation 身份 | `generation.fiber`；Loader 直接等待/停用该 Fiber，Root 只持拓扑 |
| `_active_generations` 与 `RuntimeSnapshot.generations` | live Root 的唯一 loaded-generation owner；旧 snapshot 仅服务未迁移消费者 | snapshot 投影、第二张 version table、plugin_id/ACTIVE 猜旧 Context | Manager 以 live Root 的真实 Context 精确取得 generation 与 `archive_ref` |
| `_provide_composition_services` 的完整 `mount_order` 构造 | Root-stable `ExecutionAccess` 与 `CredentialClients` provider 身份 | 每次局部变更重建全表、按 plugin_id 复用新配置、`requested` 决定凭据 provider 是否存在 | Root/Manager 宿主 provider 内唯一 `(plugin_id, generation_id)` owner/factory 表；Loader 是唯一 add/remove 写者，永久 close 后删除目标项，Root shutdown 才关闭剩余项 |
| `Bindings.bind` | 当前 OwnerCall、真实 provider/contributor Context、依赖闭包和 generation `archive_ref` | snapshot lease、plugin_id/global ACTIVE 或 descriptor 猜旧 Context；不改 descriptor schema | Manager generation lookup 是唯一 archive provenance；foreign Root、raw ContextVar、旧 activation 均 fail-loud |
| `RuntimeCatalog`、`stable_view`、runtime inspection adapter | live Root 当前 Fiber/health/incident、`_active_generations` 与 MCP 可用性；wire `snapshot_id` 保留为显示身份；Incident 以真实 Fiber ID 过滤 | `RuntimeSnapshot`/lease、冻结 topology、stable builder 与 MCP 缺失时的整段遮蔽；按当前名字归属旧代 incident | Manager capability 接收显式 caller Context，只接受同 Root、当前 OwnerCall/生命周期借用与声明依赖；普通 consumer 在自身 scope 调 `reader(ctx)` |

执行/凭据授权不使用运行时全图扫描：冷启动的正式 Root 无条件提供这两个稳定 facade，
其表由当时 generations 初始化，局部 Loader 只增删目标代际；`ExecutionAccess.bind`
按 runtime 的 `(plugin_id, generation_id, plugin_dir)` 直接取并核对，`CredentialClients`
按相同二元身份取 factory。旧 Context 因代际键不匹配，不能得到新配置；无关 factory/client
不重建、不关闭。`ControllerAccess` 继续借用 ExecutionAccess 和正式 Workload Controller，
不扩展业务恢复。除这两个已确定宿主能力外，live Root 一次提供可安全稳定的宿主 facade；
真实资源缺口在所选代码加载时明确失败，不用导入阶段预判代替真实加载。不打开 candidate 开关、不建双模式。后续迁移
`current_snapshot`、Manager 直构 lease、UI/admission 及其它 provider 前，不能宣称整条
公开安装链已上线。

**固定输入到局部应用的主链：**

~~~text
请求 B
  │
  ├─ prepare + compile：固定来源/归档/环境/配置/manifest，禁止 import/apply/start
  ▼
PluginSelection.commit(B, expected=A)  ──失败/结果不确定 → 不 dispose A，停止并报告
  │
  ▼
永久 dispose 老目标 A（仅 affected Fiber + hard consumer）
  ├─失败 → 选择仍为 B；旧 owner/句柄保留，显式 retry，不自动回写 A
  └─有限排空超时 → 只结束本次等待/报告；`_await_critical` 继续持有真实责任，不能称清理完成
  ▼
同一 Root：导入固定 B、解析入口 → apply → start → required health
  ├─import/apply/start/health 失败 → B 失败并关闭已取得资源，不自动启动 A
  │                         cleanup 失败 → B generation 保留 module/importer/scope 待 retry
  ▼
恢复 B 的 hard consumer
  └─恢复失败 → B 已选但 affected branch 不可用，错误进入结果；无关分支继续
~~~

这里没有整图操作、通用事务、恢复框架或自动 rollback；选择提交是持久输入的 commit point，
不是 active 证明。新输入启动失败不复活旧实例，下一次明确请求才使用当前已选 B 重试。
禁用/卸载复用同一局部永久 dispose；批量请求逐项处理，不承诺批量原子成功。

**下一工单的准确范围：**

- 允许生产文件：`agent/plugins/generation.py`（仅 Fiber 关系）、
  `agent/plugins/manager.py`（局部 Loader 主链、generation close、稳定宿主表的唯一写点）、
  `agent/host_bridge/plugin_execution.py`（代际键 owner 表与直接 bind）、
  `agent/plugin_composition/credentials.py`（代际键 factory 表与直接 create）、
  `agent/plugin_composition/bindings.py`（exact lease/Context/activation 校验）。
  `selection.py` 的现有 `commit`、archive、FreshPluginImporter 和公共 `Context` 直接复用，
  不新增持久字段或通用 Context API。
- 允许测试文件只复用真实消费者的既有边界：
  `tests/test_plugin_hot_reload.py`（`reconcile_changed` 局部主链）、
  `tests/test_plugin_composition_lifecycle.py`（同 Root/Fiber/Effect oracle）、
  `tests/test_plugin_channel_credentials.py` 与 `tests/test_plugin_bindings.py`（代际授权）。
  本批已补充/修订这些文件中的静态真实路径 oracle；按任务合同未执行测试。
- 必须同批删除：局部路径对 `_compile_generation_snapshot`、`_build_validation_host`、
  `_replace_formal_root` 的调用；Root 对 generation module/scope 的 deferred cleanup；
  `mount_order` 每次局部变更重建 Execution/Credential 表；按 plugin_id 取得新 factory
  的旧路径；以及没有调用者的 Loader 私有转发层。`runtime_snapshot` 只有在旧消费者迁移
  批次仍有真实消费者时暂留，并在该批次明确删除点，不得被当作局部 loaded input owner；
  旧 candidate 入口本批明确拒绝。
- 退出条件：`reconcile_changed` 真正走同一 Root 的 prepare→selection→dispose→mount→hard
  consumer；同一固定输入幂等；编译失败选择/图不动；新启动失败不回旧；清理失败可见且
  owner 可 retry；无关实例/生命周期计数/在途工作不变；Execution/Credential 旧 Context
  不能获得新代际授权。否则该批只算前置，不能称 Loader 已切换。

**最少真实 oracle（只设计）：** 同一 Root 上记录 unrelated plugin 的 import/apply/start/stop/
Effect 与在途请求身份；局部变更后这些计数与身份不变。断言 module 仅在永久卸载或失败清理
成功后释放，依赖 activation 重启不释放 module；旧 Context 的 `generation_id` 不能 bind
新执行/凭据；compile 失败不改 selection/graph；B start/health 失败不回 A；cleanup 失败
仍能找到真实 generation/module/scope owner。排空采用事件屏障与受控调度；超时只证明本次
等待截止，不证明 `_await_critical` 背后的清理已经结束。

**怎么做：** 保留安装链的来源固定、构建、代码归档、环境准备与错误传播；只准备发生变化的
制品，复用 FreshPluginImporter 私有命名空间；相同固定输入且当前 active 时无操作，相同输入
失败时只能由明确请求重试；重复请求只查询原结果。安装、禁用、卸载与批量项都进入上述同一
局部入口，结果同时报告 selection、实际 owner 状态和错误。

**DSH 对照：** 固定基线 `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8` 中，
`vendor/loader/src/config/entry.ts:130-139` 的 `_dispose` 只释放指定 Fiber，
`:142-246` 的 `update` 区分 no-op、局部 patch、dispose 与替换，`:269-301` 的 `_await`/
`_start` 等待 Fiber 并传播错误；`vendor/cordis/src/fiber.ts:222-236` 创建一次 Context，
`:611-695` 以 epoch 复用同一 Fiber 做 reload/unload，`:704-723` 提供 await/restart，
`vendor/cordis/src/registry.ts:316-335` 返回该 Fiber。Akashic 借鉴“保留 Fiber 关系、等待
实际启动/清理”，不复制 DSH `entry.ts:232-240` 的自动回登旧实例、HMR cache 或 rollback。
固定基线实际 CLI 是 `apps/cli/src/plugin.ts:59-90` 的 `reconcilePlugins` 和
`:120-135` 的 `runPlugin`；没有猜测的 `packages/cli/src` 路径。严格 OwnerCall、持久
PluginSelection 与局部排空仍是 Akashic 自有约束。

**退出条件：** 新增、替换、禁用、卸载均保持 Root 和无关实例身份；编译失败不动旧图；
新实例失败只影响真实依赖分支；真实加载错误不会被包装成安装成功。完成本节不等于
current_snapshot、公开入口、运行验收或部署已闭合。

### T05 · 切换公开入口，移除 latest/promotion 协议

**输入与下游：** 内外安装请求 → 唯一安装应用入口 → Agent/CLI/Web/Mobile 看到真实结果。

**主要文件：** [PluginUpdates](../../agent/plugin_composition/plugin_updates.py)、[plugins/plugin_update](../../plugins/plugin_update/)、
[main.py](../../main.py)、[bootstrap/app_server.py](../../bootstrap/app_server.py)、安装 watcher/trusted batch、
相关 ControlService schema 和实际前端消费者。

**怎么做：**

1. plugin_install 与 CLI plugin-install 都走 T04。安装请求返回后由宿主持有应用任务，避免内部 Shell 等待自身被卸载。
2. 保留窄的安装、卸载、状态查询能力；移除 plugin_latest run/revert、open_validation、publish/discard 及其专属授权。普通 programmatic 调用能力不删除。
3. 删除 main.py 的 plugin-promote/plugin-discard 入口及对应 control callback；旧入口明确报告不再支持，不静默映射成新的安装语义。
4. watcher 不再另行晋升或直接 _replace_formal_root。固定选择有变化时由同一加载入口应用；修改 checkout 不隐式改变已安装制品。
5. 离线 trusted batch 使用同一选择写入逻辑，不再靠清空 selection 让下次启动猜测。在线仍通过现有受认证控制面，不增加第二个 writer。
6. UpdateStatus 和插件目录改为安装选择与当前 Fiber 状态的投影。实际前端消费者按搜索结果逐一迁移，不预设所有 frontend 文件都有候选逻辑。
7. 与父 Turn 晋升授权专用的环境标记在消费者清空后删除；Shell 进程归属、权限、退出和通用请求身份不删除。

#### T05 当前最小公开安装与卸载合同（T-9990aa 整体未通过；T-fb6d31 生产修复静态通过、测试源有 B1～B4 错误；T-66262a 生产窄简化静态通过；测试与运行验收仍未完成）

T-799b45 已完成生产、status/卸载尾项与 T05-C 生产接线的静态修正；T-3b0e07 已完成 T05-D 页面 provider 迁移的生产静态修正。T-129aa6 的 T05-C-R2/T05-D 测试源、fixture、negative-control、LOADING、scope 与 cleanup 顺序证据已静态通过；T-9990aa 的 T05-E 整批未通过，T-fb6d31 的生产修复静态通过但测试源有 B1～B4 错误，T-66262a 的生产窄简化静态通过，不能据此宣称 T05 或 Issue 750 已启用。当前边界如下：

- `PluginUpdates.install`、`bootstrap/app_server.py:install` 和 CLI 现在都走同一 `PluginManager.install`；CLI 缺少 `--update-id` 时生成非空 UUID。
- `ManagerOperation` 持有一次性 `accepted` Future。调用者只等待 selection CAS，取消或断开不撤销宿主继续排空和挂载的任务；caller cancellation 只放弃等待，宿主仍持有安装/清理 owner。宿主 deadline 则结束 accepted 等待并撤销迟到的提交许可；已进入的物理 installer、清理与 busy owner 仍须结算，不能把它们写成已经取消或成功。
- `agent/plugins/manager.py:_load_one` 已能把固定归档、静态 compile、配置 revision 和 `archive_ref` 放入一个 generation；`_update_live_generation` 才执行 selection CAS、旧目标排空和同一 live Root 挂载。公开 install 不能用 `reconcile_changed` 顺便安装其它 checkout 变化。
- `install_git_plugin(stage_candidate=False)` 只结算固定制品；Manager 随后从精确制品构造 generation，写入 `input_ref`，再做 selection CAS。`candidate_pointer` 仍只表示安装恢复点。

实现沿用已有 `UpdateStatus`，不新增 InstallReceipt、第二状态表或版本选择器：

```text
source/ref
   ↓ installer(stage_candidate=False)
archive + input_ref
   ↓ 唯一 selection CAS
accepted Future ──→ caller scope 结束
   ↓ 同一 live Root
旧 Fiber 排空 → 目标挂载 / readiness
   ↓
UpdateStatus(active) + notify
```

```text
UpdateStatus
├─ update_id: 本次请求唯一身份
├─ plugin_id: 目标插件身份
├─ input_ref: 本次固定 PluginArchive descriptor；旧行可为 null
├─ selection: selected | not_selected | unknown
├─ runtime: generation_id、archive_ref、fiber_state(active/loading/unloading/failed/disposed/missing)
├─ state: accepted | active | failed | unknown
└─ error: 实际错误文本
```

`accepted` 只表示本次 `input_ref` 已通过唯一 `PluginSelection` 的 CAS；它不表示 active。
查询必须同时读取当前 selection、`Manager._active_generations[plugin_id]` 和实际 generation/Fiber：只有 selection
包含同一 `input_ref`、generation 的 `archive_ref` 相同且 Fiber 为 `ACTIVE` 才返回 `active`。找不到当前运行实例不能用空目录、`journal.phase=committed` 或 HTTP 200 代替 active。
错误不建立七阶段持久协议；保留实际 owner 错误文本，不能把未知结果写成回退成功。

当前持久实现给现有 `plugin_updates` 增加可空 `input_ref`：

1. `input_ref` 由 PluginUpdates/journal owner 在固定 archive descriptor 后、selection CAS 前只填一次；不复用或改写 `candidate_pointer` 的含义。
2. 旧行保持 `NULL`，不从当前 plugin/source revision、cache 目录或 `committed` phase 猜测，也不自动重放；查询旧行只能返回 `input_ref=unknown`，不能宣称本次 active。
3. 不删除旧字段、旧行或历史事件。迁移源先建立命名 SQLite 恢复点并核对旧/新 schema identity；旧行保持 NULL，`PluginUpdates.read` 与公开状态查询是实际读取者。

失败和恢复结果固定如下：

| 情况 | selection | 运行事实 | 回执与恢复 |
|---|---|---|---|
| compile/import 前静态失败 | 不变 | 旧 generation 不动 | `failed`，保留实际错误文本，不返回 accepted |
| selection CAS 冲突或写入不确定 | 不把旧值改回 | 旧 owner 不动 | `failed` 或 `unknown`；停止并报告，不能 dispose 旧 owner |
| 已选 B 但 B start/apply/readiness 失败 | 保留 B | B 失败/保留真实 owner，A 不自动启动 | `failed`；显式 retry 按同一固定 selection 处理 |
| 旧 owner cleanup 失败 | 保留 B | 旧 draining owner 保留，B 尚未冒充 active | `failed`；显式 retry 原 owner |
| caller 取消 | caller 只失去等待；宿主随后仍可完成 CAS，不回写旧值 | 宿主任务继续物理结算 | 查询返回 accepted/active/failed；取消不当作 rollback |
| 任意同 `update_id` 重复 | 不变 | 不重拉、不重跑 | 所有重复安装都 fail-loud；只能读取原回执，不比较输入来放行重跑 |
| 进程重启 | 启动读取当前 selection | 不续跑未完成 operation；按 selection 重建并重新核对 Fiber | 有固定 `input_ref` 才能判定；旧行/缺 ref 返回 `unknown`，不猜历史晋升 |

安装宿主使用已有 `_start_operation(background=True)`/`ManagerOperation` 持有应用任务：selection 成功、旧 owner 开始排空前提供一次性的 `accepted` 同步点，调用方随即释放自己的 scope；局部排空、挂载和 readiness 在宿主任务中继续。线程安装取消仍须等线程和目录 owner 物理结算后，才能归还安装资源。`accepted` 永远不是 `active`。

本批真实责任范围已覆盖上述 Manager、journal、control、CLI、plugin_update 消费者和迁移源；`latest.py`、`validation.py`、旧 install/open_validation/publish/discard 公开入口已删除或停用。历史 OWNER_STATE 请求由窄 decoder 丢弃已退役 validation 字段，不执行旧验证；active/failed 通知结果 ID 与 follow 去重、以及历史请求 decoder 已接通。可信 UI/目录投影与其它历史边界仍只读取同一 selection/Fiber 事实，不在本轮扩展 writer 或恢复协议。

T-e721a4 的 `CoreRuntime.inspect_modules` production 已由主审与独立只读 review 静态接受；T-fbf63b 的 Inspector 坏节点 admission oracle 与当前累计 `tests/test_plugin_external_loader.py` 也已由主审与独立只读 review 静态接受，T-7912fd 原 R1 缺口不作为已通过稿。production 唯一链路仍是：冷查询取得 publication lock，发现没有 Root 时调用既有 `PluginManager.load_all`，再重读 `live_root`；load 成功却没有 Root 时明确抛出 Core 不变量错误；已有 Root 则直接调用 `root.topology_view()` 输出 identity、revision、Fiber 和 listener。热查询不再重复 load、不重建或编译 Root、不 acquire snapshot lease、不 freeze，也不把 `receipt.ready` 当查询门禁；无关坏分支仍由既有 Fiber/receipt 负责状态和错误。load 的真实错误与取消继续原样传播。当前累计 Inspector 测试源显式初始化 selection，使用真实 live Root 挂载/撤销 Fiber、缺依赖 PENDING、peer `runtime_scope` 和实际变更后的旧路径负控；Gate、CI、应用 import 仍未执行。

WIP 范围明确为：UI/目录查询、其它 provider/RPC、scope 外 manager 消费者、`current_snapshot`/fence、离线 watcher/trusted batch、候选路径删除与 freeze、T06/T07 及最终 enable；它们改为读取同一 selection/当前 Fiber 投影，但本轮不扩展、不删除历史消息或 OWNER_STATE 数据，不增加第二个 selection writer。普通 programmatic 调用继续保留；DSH 只作三处固定只读对照：`apps/cli/src/plugin.ts:59-90,120-135` 的安装后依赖层投影、`vendor/loader/src/config/entry.ts:291-306` 的 import/apply 后等待 Fiber、`packages/host/plugin-inventory/src/index.ts:46-88` 的每次直接读取 Loader/Fiber 当前事实。DSH 没有本项目的 OwnerCall、PluginSelection 或持久回执，不把这些对照扩写成等价实现。

**DSH 对照：** D06 插件目录不另存生命周期真相；D07 安装入口提供确定输入。
“无候选晋升”是本项目已确认的功能移除，不要求 DSH 有同名旧功能才可删除。

**退出条件：** 内部 Shell 自安装和外部 CLI 得到同一种选择/状态语义；
旧 latest/promote 路径不可继续控制版本；已接收与已激活可明确区分；
查询不到运行实例时不把空目录当成健康成功。本批仍未运行测试、Gate、CI、应用 import、插件安装、正式 workspace 或部署，不能写成 Issue 750 完成。

#### T05-B · 公开卸载实现（T-799b45 生产/status/卸载尾项静态通过；测试未执行）

公开 `uninstall` 现在只由 `PluginManager` 持有：入口取得同一 `ManagerOperation`，先验证安装项，
在 operation owner 已取得后写入 manifest disabled，再按完整 selection 做目标组件的 CAS 移除；
CAS 成功后一次性返回 `{plugin_id, state: "accepted", selection_ref}`。同一宿主任务随后排空目标
generation、目标 Fiber 的真实硬依赖消费者以及已有 draining owner，所有实际资源关闭后才调用现有
`finalize_uninstall_plugin` 删除 cache 和 manifest entry。该物理删除通过当前 Manager operation 的
`complete_critical(asyncio.to_thread(...))` 结算，调用方取消或 deadline 不会截断仍在写目录的线程，也不会提前返回
`removed`；实际 manifest/cache 事实继续可查询。`bootstrap/app_server.py:uninstall` 只保留到
Manager 的薄委托，不同步等待宿主任务，也不另立 receipt、schema、Root/lease 或重启路径。

```text
Manager.uninstall
  → operation owner / validate installed / manifest enabled=false
  → selection CAS remove target components
  → accepted handoff
  → target Fiber + hard consumers + draining owners close
  → finalize_uninstall_plugin(cache + manifest)
```

卸载状态查询复用现有 `plugin_status`：manifest 的 installed/enabled、selection target ref（包括无
active Fiber）、当前 generation 的平面字段、draining generation、当前 Manager operation/task、accepted
结果和 cache 实际存在性分别投影。operation 诊断只描述当前 Manager Task，不是跨重启或跨后续操作的永久卸载历史；operation 的目标插件只从成功的 `UpdateStatus` 或卸载 dict accepted
结果推导；pending/failed accepted 不从 operation 请求字段猜目标，不增加状态 cache 或 schema。accepted
不表示 removed；目录不存在也不单独证明卸载完成。安装继续使用 `UpdateStatus`，卸载使用显式 dict 结果，
不新增通用状态协议。进程重启只读取持久 manifest 和 selection，不自动续跑未完成 operation。

已选但无 Fiber、Fiber FAILED、active 加 draining 的目标都走同一目标 owner；禁用 reconcile 的
清理重试不以“active map 为空”永久 busy。CAS 冲突或提交失败不 dispose、不返回 accepted；manifest
已 disabled 的部分事实可以保留，不能声称回滚。finalizer 失败按实际错误暴露，cache 可能部分变化，
后续显式同 owner retry 重新读取 manifest；不创建 recovery queue。卸载只删除代码 cache、manifest
entry 和能力投影，保留 plugin-data、Message、附件、历史 binding、归档、root_ref、metadata、
Delivery/journal 历史与凭据；恢复依赖 source copy，不能把它称作用户数据备份。

本批只完成静态实现与 review 材料：没有运行测试、Gate、CI、应用 import、插件安装、业务进程、正式
workspace 或部署。DSH 对照固定为 `/mnt/data/source-code/deepseek-harness` @
`c389f96bf3a9b6807cb71ed6bdad5849be0df6d8`：`vendor/loader/src/config/entry.ts:127-139`、
`:179-193` 与 `vendor/cordis/src/fiber.ts:680-699`、`:704` 只用于说明局部 Fiber/Effect
清理与等待；它不提供本项目的 rollback、cache recovery、accepted、PluginSelection 或
plugin-data 等价保证，也不引入 DSH 的 weak cleanup 或 re-login 语义。

#### T05-C · 控制输入与动态 RPC 的 live-provider 局部 scope（T-799b45 生产、T-129aa6 测试静态通过）

本批只迁移 `bootstrap/app_server.py` 的两个旧整图租约入口。每次调用读取同一
`PluginManager.live_root`；缺失时抛出 `RuntimeClosedError`，不构造 Root、不启动插件、不退回
`snapshot_store`。输入接纳从 live Root 的 `CHANNEL_INPUT` 精确取得 provider Context 与 callable，
在该 Context 的 `runtime_scope` 内调用同一个 callable；动态 RPC 仍先用 `rpc_method_key(name)`
核对边界，`service_value` 没有 ACTIVE provider 时保留 Method not found，否则在无 await 窗口取得
exact provider Context 与 `RpcMethod`，并在同一 scope 内让 Router 完成 schema validate 与 invoke。

```text
Control
  │
  ▼
live Root exact provider
  │
  ▼
owner RuntimeScope ── schema + handler / input
  │
  ▼
release
```

旧 `RuntimeMessageDisplay` snapshot 展示 wrapper 已退役；其它 snapshot 消费者、业务 RPC 内部的可选 provider
保护、其它 provider/RPC、`current_snapshot`/fence、offline trusted watcher、candidate/freeze、
T06/T07 和最终 enable 仍是 WIP。测试源使用真实 `CoreRuntime`、live CompositionRoot、provider
Fiber/Context/Effect、`build_control_service` 和 `ConnectionRouter`，覆盖目标 owner 排空、旧请求完成、
新请求不可用、peer scope 不变、旧 schema 拒绝和新 provider schema 生效；仅写源，不执行。

新切口的 DSH 只读借鉴限定为 `/mnt/data/source-code/deepseek-harness` @
`c389f96bf3a9b6807cb71ed6bdad5849be0df6d8` 的 `vendor/cordis/src/reflect.ts:277-304`
`provide`（exact value 与 Fiber 绑定）和 `:314` `notify`（只影响依赖方）。Akashic 的
OwnerCall/RPC scope 不是 DSH 现成等价物，不复制旧实例、cache recovery 或整图恢复。

#### T05-D · 页面展示的本页 provider scope（T-3b0e07 生产、T-129aa6 测试静态通过）

`project_message_rows` 已从 `agent/plugins/snapshot.py` 移到
`agent/plugin_composition/message_view.py`。helper 只根据当前 `MessagePage` 实际出现的
`ContentPart.kind` 与 `ToolCall` 选择 `message.display:<kind>` 和
`tools.display-name.v1`；通过 live `CompositionRoot` 的 `service_value` 与 exact
`_service_provider` 取得 provider Context，在每个 Context 上只进入一次 `runtime_scope`，
同步调用既有 `message_rows`，然后由 `AsyncExitStack` 释放 scope。Control 不产生 part；未出现的
provider 不进入 scope，缺失或非 callable provider 保持 unavailable 语义。

```text
MessagePage → 本页 kind/ToolCall → live Root exact provider Context
            → local scopes → 同步 message_rows → release → rows
```

`bootstrap/app_server.py` 的 `message_display` 每次读取同一个 `manager.live_root`；Manager 的
`core.message_display.v1` closure 把同一个 Root 传给 helper，不再把 snapshot store 当作展示
provider。当前生产真实 consumer 只有这两个入口；本批删除 `bootstrap/message_display.py` 的
`RuntimeMessageDisplay` wrapper 及仅覆盖它的旧 snapshot generation 测试。reply/UI 等其它历史读取仍保留其原 owner。测试源覆盖真实
CompositionRoot/provider Fiber/Context/Effect、同 Context 去重、无关 provider 不入 scope、同一
Root 的局部替换、LOADING/缺失 unavailable、renderer 异常释放，以及 build-control-service 的
真实 Message/read 链；保留 `test_live_message_projection_uses_only_page_providers_and_releases_scopes`
与 `test_live_message_projection_releases_scope_on_renderer_error` 两个高价值 oracle。T-3b0e07 的测试源另有 asyncio import、ToolCall writer check_call、同步
compile negative-control、peer runtime scope、RPC LOADING/missing method、handler→Effect cleanup
顺序等缺口；T-129aa6 只补这些测试与文档，动态 fixture 源仍在写入前做 AST parse 与内存 compile，
全部仅供静态 review，未执行。

#### T05-E · control reply.status 长订阅与重复 follower 删除（T-9990aa 整体未通过；T-fb6d31 生产修复静态通过；T-7ff900 测试源静态通过但未运行）

真实消费者只有 `bootstrap/app_server.py` 的 Control follow；`bootstrap/reply_status.py` 的
`RuntimeReplyStatus` 原先和 `agent/plugins/snapshot.py:follow_reply_status` 各自读取同一
`reply.status.v2`。现行实现只保留 live Root 订阅器；`plugins/akashic_clients/channel.py` 的自有
`_follow_reply_status` 是另一条未迁移路径，本节不声称它已闭合。

```text
外部 control follow
        │
        ▼
Root.context.inject(REPLY_STATUS)
        │  optional subscriber Fiber
        ▼
Fiber-owned Effect pump ──> 私有有界/合并 channel ──> 外部 frame
        │                              ▲
        │ provider 失活时 cancel + join │ Root Effect 只负责 teardown wake
        ▼                              │
ReplyState 及其 close 由 reply provider 拥有 ─────────┘
```

subscriber 只冻结自己的 `ReplyRead` 依赖并启动 pump；pump 不持有外部 socket，也不长期持有
provider `OwnerCall`。`snapshot_id` 仅由本次 subscriber dependency store 中冻结的 live Root
`generation_id` 与精确 provider registration revision 组成；无关 Fiber 不改变它，重新登记 provider
才改变它。私有 channel 合并临时快照，但首个 reader acquisition/iteration/`aclose` 错误是不可覆盖的
终态；取消保持取消语义，初始化和后续 activation 的 spawn/apply 失败都经同一终态唤醒真实 follow
消费者。provider 失活和正常 reader 结束会尝试产生 unavailable 边界，A 的预览不会被 B 重放；这不
承诺 Root teardown 前一定先交付该帧。Root teardown 先关闭 subscriber children，再关闭 Root-owned
teardown Effect；PENDING subscriber 的等待由同一 channel 终止事件唤醒。subscriber generator/pump
归 subscriber，`ReplyState` 及其 close/真实回复工作归 reply provider，不增加 Core registry、watcher、
第二运行图或 durable data。

本批删除 `snapshot.py` 的旧 `_ReplyStatusReader` 与 `follow_reply_status`，以及
`bootstrap/reply_status.py` 的 Store/lease/generation 分支；保留 snapshot.py 仍有消费者的
`RuntimeSnapshotCompiler/Store` 及其它历史读取。T-fb6d31 测试源迁移为同一 Root 的普通 provider
Fiber、真实 `CoreRuntime`/`PluginManager`/`ControlService.follow`、`ReplyState`、真实 Task/Effect
边界，并保留 Web/Mobile preview、正式 Message、cursor、receipt、auth/protocol 与取消断言；
T-fb6d31 的生产 R1～R4 修复已静态通过；T-7ff900 A1～A4/B 测试源已由主审与独立只读 review
静态通过但未运行。现行 Core 的 ACTIVE Context late provide 已由 T-2f3d82 接入 exact registration 通知、
critical physical settle 与失败 owner 保留；T-a2817b 的 T02-N1 整批 production/test review 未通过，
T-a510da 的 P1/P2 与 T-518ba4 的 R2 测试源已静态接受；T-2f3d82 的六项 N2 测试源因真实 Fiber/Handle
API 与局部失败 oracle 缺口未通过，T-cbecef 正在返修。当前剩余证据是返修后的测试源静态 review、行为测试，
以及 scope 外 provider/RPC/消费者的独立接线；不能写成产品全链路或运行验收已闭合。当前仍未运行测试、
Gate、CI、应用 import 或运行验收。

T05-E 与客户端 adapter 的 T05-F 分开核算；T-7ff900 的 C 只读交接确认客户端下一步必须把
Web/Mobile 的 message 与 reply 长 follower 一起处理：

```text
client follow Task
  → open_request_scope/open_message_scope 包住 message/reply reader
  → binding in-flight + contribution Context OwnerCall
  → provider UNLOADING 等待 OwnerCall
  → registration Effect 尚未执行
  → close_admission/follower cancel 尚未发生
```

当前 T05-E 仍未覆盖客户端 adapter 的 scope 收窄；不把它写成 T05-F 或 Issue 750 的完成证据。

#### T05-F · Web/Mobile 客户端长订阅 scope 收窄（T-24e3c0 不通过；T-cbbf5f R2 已静态通过；T-8e7365 源码待 review）

T-24e3c0 的测试源存在 fixture 解包、`CompositionRoot.mount`/`Context.mount` 返回值、
`ChannelFactoryContext.open_scope` 与真实 `_GenerationAkashicAdapter` owner 链的确定性错误；
独立生产 review 还确认了 Web 迟到 follower 登记和 Mobile `ready`/`send_lock`/`start_follow`
关闭窗口。T-1f0e1c 修订了允许的测试源与文档；T-cbbf5f R2 补齐生产闸门和真实回归源，并已由主审与
独立只读 review 静态通过；T-c0c299 的 A1～A3 测试源静态通过，A4 与 UI 合同未通过，
T-8e7365 修订 A4 和 T03a-UI 设计合同；T-57f277 与 T-37d602 整体 review 未通过；T-7b0520 R2 与 T-33865f R3 已完成 Dashboard host、wire 和真实测试源定点修订，Dashboard 生产与 T98 A1-A3 静态通过；T-98b426 完成 Mobile seam 的 live Root/Context、async consumer 与测试源迁移，T-7aeefe 完成 Mobile R2 的取消/shutdown 失败组合和真实 oracle 定点修订，Mobile R2 已由主审与独立只读复核静态通过，测试未运行。

客户端 owner 链必须按真实状态顺序理解：

```text
短 scope 取得 readonly reader → scope 退出
        │
        ├── 长 follower Task 持有 listener
        │       └── 每页 display 自己取得短 scope
        │
        ▼
provider 失活 → consumer 停止 → close_admission/cancel + join
        → consumer Effect/adapter cleanup 完成 → provider 资源 close
```

硬依赖的作用是让 provider 在 `UNLOADING`、撤销新接纳后继续等待真实 consumer cleanup，
不是保证 provider 先于 consumer cleanup 才失活。`MessageLog` listener、binding/in-flight claim、
reply iterator 的 `aclose` 和 adapter stop 都必须由实际 owner 结算；新的 request 在 owner 不可用
时明确拒绝。订阅 child 的现有顺序和关闭结算为：

```text
创建/登记 child → ready → wire OK → start_follow
      ╲ close_admission/stop at any window
       → 同一 child cancel/ready settlement/map cleanup
```

已经开始发送的 OK 不可撤回；关闭后不放行 `start_follow`。真实
Web 测试源通过 `CompositionRoot`、`PluginRuntime`、`Channels.register`、`Effect`、
`ChannelFactoryContext.open_scope` 和 `_GenerationAkashicAdapter` 检查实际 adapter close、
迟到登记和旧 follower join；Mobile 测试源用真实 WebSocket/gateway 覆盖首指令取消、ready 后
锁门和 OK 后 start gate。display callback 的短 permit、reply iterator 的 `aclose`/错误传播及
provider `UNLOADING` 窗口也有源级 oracle。以上清理失败保留 owner、首指令取消、send 竞态和
display 结算均尚未行为测试；旧 fake scope counter 只保留为 transport 回归。

本批持久化 delta 为 0；Message、cursor、receipt、auth、schema、descriptor、`root_ref`、
metadata、credential 不变。Core 活动 Context 的动态 reprovide 通知缺口、其它 provider/RPC、
UI/fence/current_snapshot、offline watcher、candidate/freeze、T06/T07、运行验收和最终 enable
继续是 WIP；不引入第二套 graph、lease 或 bootstrap helper。

**T05 UI wiring gate（Web/Dashboard R3、T98 A1-A3 与 Mobile R2 静态通过；测试未运行）：** T03a-UI 的 Core 前置条件现在唯一是
`RuntimeScope.wait_admission_closed()`；它是绑定原 activation 的只读 Event wait，不是 callback、
Effect listener、全局 revision polling 或 `snapshot.accepting_leases` 轮询。T-7b0520 R2 在 T-37d602
基础上收敛 Web/Dashboard 的 registration/apply、Dashboard app/module/Workload、HTTP/WS wire
fence 与 owner 结算；T-33865f R3 又把 stale/forbidden 拒绝的 wire await 移出 UI lookup scope，补齐
背压下 UI owner 可 dispose 的真实 HTTP/WS oracle，并修正同 Fiber 硬依赖重激活与错误结算测试源；
UI 自持 scope、`HOST_INFO.validation` 可独立记为静态结论，wire close 不严格
等待 Fiber Effect 完成，而是在 app Task 物理 finally/permit 释放后发送，Fiber 随自身 drain 执行
Effect。Mobile handler/thread/query admission、Mobile wire fence 已迁移到单一 live Root/Context owner，旧 Mobile seal
及 snapshot/source_revision 双轨已删除。生产代码
已删除 Web/Dashboard 调用点的 whole-Root UI seal、Dashboard 旧 snapshot lease construction/read
和旧停止轮询。Web/Dashboard 验收必须分别证明 local replacement/同 Fiber reactivation、旧 HTTP
drain、socket monitor cancel/join、无关插件 identity、同 artifact fence、exception/close retry；
Mobile R2 静态复核已通过但行为测试未运行；不能用 catalog 可见、HTTP 200、pending/done 行或 worker 文案替代行为验收证据。

### T06 · 删除旧发布系统，而不是换名保留

#### T-b0ab9e · Models R2 跨 archive 测试修订（累计静态接受，未运行）

T-fa9271 只保留无真实生产消费者的 `agent/plugins/model_control.py` 与
`agent/plugins/model_catalog.py` 删除结果；T-83b918 的 Core 消息 owner scope、Stats
Dashboard/Mobile 真实 provider 错误链与 in-flight/cleanup 证据、Wake debug Path/catalog
证据及文档对账已由主审与独立只读 review 静态接受；T-b0ab9e 的
`reader.__self__` 替换和同路径断言也已静态接受。T-fa9271/T-83b918 原失败稿不追认为整批通过；真实消费者改为：

```text
公开 RPC / 真实 Dashboard / 窄 reader
  └─ 同一 live Root → exact provider Context
       → Models / selected driver scope → 原 SQLite / 真实 model_calls
```

测试和 debug 不再构造 Core wrapper、第二 registry、snapshot lease 或 fake FastAPI
router；`build_control_service(core).resolve_method()` 取得真实 `RpcMethod`，参数由
operation schema 校验后 invoke。旧 wrapper 专属的 503/lease-count oracle 删除；客户端
可观察的 missing 404、Mobile `model_stats_unavailable`、隐私、持久 model_calls 和
programming-error 不吞仍保留。Core 消息测试使用实际 Tools/Materials/Content/Catalog
provider 的短 scope，Stats 测试使用真实 Web identity、ModelsStore reader 与 Models/UI
provider 结算证据；Wake debug 使用真实 `MODEL_CATALOG` 与独立 `CHAT_MODELS` scope。Models
Dashboard 仍由 Models contributor 自己注入 `MODEL_CALL_STATS`；可选
`create_chat_app(model_control=...)` 生产 API 不变。

这只是测试/debug/文档返修和两个死文件删除，不改变 Core/Models 生产 owner、数据库 schema、
descriptor、credentials、Message、receipt、history 或正式 workspace；行为测试、Gate、
CI、应用 import、插件安装和部署均未运行。

**输入与下游：** T05 已切换真实消费者 → 移除失去责任的实现，保留必要历史读取与资源责任。

| 删除目标 | 删除前必须证明 | 替代或保留 |
|---|---|---|
| manager 中候选准备、验证宿主、_switch_ready、_replace_formal_root、_build_and_publish_root 及失败重建旧 Root | 公开入口和内部 boot/terminate 路径不再依赖它们；拆出仍需要的首次加载与最终关闭 | T04 局部加载；保留一次冷启动和真实退出 |
| snapshot.py 的多运行图 store、pending/provisional、全图 lease、跨图发布事务 | T03 已迁完调用、binding、查询与清理消费者 | 单 Root、owner 的实际调用保护；只读结果对象可保留 |
| reload_journal.py 中候选晋升、旧图重建和双图恢复推进 | 无在途旧更新；旧外部资源与历史账有明确处理和只读入口 | 不新增同等复杂的新 journal；真实资源账仍由其 owner 保留 |
| plugin_update/latest.py、validation.py 中候选专属流程 | 没有动态工具、提示、CLI 或测试入口残留 | 普通 Agent 测试与 programmatic 调用不受影响 |
| `bootstrap/message_display.py` 的 `RuntimeMessageDisplay` 与旧 generation 测试 fixture | 生产只剩 `app_server.py:message_display` 与 Manager 的 `core.message_display.v1`，二者都直接使用 live Root 的 `project_message_rows` | 删除 wrapper 与仅覆盖旧 snapshot 切代的测试；保留本页 provider scope、missing/LOADING、局部替换和 renderer failure oracle |
| 发布专用回调、候选开关、重复投影、失效 helper | 静态/动态消费者、配置入口、安装产物和兼容义务均已核对 | 无独立行为的冗余直接删除，无需额外 DSH 替代 |

不能把以上文件按文件名整删：archive、generation identity、历史 binding、诊断、终止、真实失败资源清理可能仍有消费者。
进程/Workload 的候选清理也不能因“以后不建候选”就抛弃旧残留；T07 先核对和处理。

#### T-fbf63b · EventBus 残留保护层与坏节点 admission oracle

本批已按生产消费者和 owner 证据删除 EventBus 的 RuntimeSnapshot lease/store 绑定、获取/继承/释放路径、旧 admission 等待任务与单字段 queued envelope；同步删除 Manager 保存并绑定 Bus 的无读者字段，以及 ValidationHost 的绑定调用。EventBus 的 generic queue、dispatcher、handler Task、observer 错误/取消隔离、`task_done`/`drain`/`join`、dispatcher 错误聚合和 `aclose` 责任保留。T-fbf63b 的生产删除与 Inspector 坏节点 admission oracle，以及当前累计 `tests/test_plugin_external_loader.py` 已由主审与独立只读静态接受；T-7912fd 原 R1 缺口不作为已通过稿，行为测试未运行。

```text
Core/ValidationHost → EventBus queue/dispatcher → generic handler Task → drain/join
Plugin Context → Root EventRegistry → Fiber Effect（独立，不桥接）
```

本批不删除 `snapshot.py` 或其其他消费者，不改 EventRegistry、Channel binding、Message/schema、历史 journal/selection/archive、candidate 清理、offline/trusted watcher 或 scope 外 provider/RPC；它们仍为 T06/T07 WIP。持久化 delta 为 0。仅做 AST、内存 compile、diff/空白、全仓符号搜索和哈希核验，不运行测试、Gate、CI、应用 import、插件安装、业务进程、正式 workspace 或部署。

T-71cb50 的 EventBus R1 两项测试源尾修与上述退役 wrapper/旧 fixture 删除已由主审与独立只读静态接受；行为测试未运行。

#### T-57630a · 退役常驻 runtime loop 与正式宿主入口（production 已冻结；T-d8e47c production/test review 未通过；T-4f17d6 Cold-Start 累计静态接受，未运行）

本批只收窄真实宿主边界，不扩大到全部 candidate/snapshot 删除。Manager 删除唯一常驻
`run_runtime_services` loop 及其开关字段；保留仍有真实消费者的 snapshot start/stop/store、candidate/history、cleanup、operation/CAS、accepted、physical owner 与 `terminate_all` 责任。App 不再在正式启动中调用 `start_runtime` 或登记旧 loop；只有实际 `host_bridge_monitor` 等 runtime task 才创建 primary supervisor，Dashboard/PluginWatcher 仍是独立真实监督对象，意外空集合明确失败。stdio 直接等待 `StdioAppServer.run()`，EOF/错误后沿原 `ControlService → Bus → Core.stop → Manager.terminate_all → Fiber` 清理链退出。

```text
core.start/load_all → 同一live Root → 每个Fiber STARTING/STARTED/health
AppRuntime.run Task（精确 owner）→ 实际Dashboard/PluginWatcher/[HostBridge] → host shutdown
外层 stop/cancel/restart
 └─ main.serve exact Tasks
      ├─ runtime → AppRuntime physical cleanup → Core/Root resources
      └─ stop/restart/settings → 全部结算 → 原取消/实际错误链
stdio → StdioAppServer/ConnectionRouter → 物理 readline 线程 → EOF/error → 原Core.stop
```

host readiness/宿主存在不等于所有插件健康：Fiber 的 STARTING/STARTED/required health/ACTIVE
仍由该 activation owner 解释，host 只监督真实宿主入口。T-57630a production 三文件与旧 runner 删除已由主审及
独立 review 静态接受并冻结；原整批不接受，因为 `tests/test_runtime_smoke.py` 相对备份只删了旧 runner stub，
没有交付 D1-D4 的 host 测试源码，不能写成只差运行。T-b0292d R1 新增测试源不接受：A1 的
`dashboard_started` 只表示进入 `serve` 包装，且仅检查外层任务未完成，App 没有 primary 时仍可能已经进入
shutdown 等待窗口；B 把 `main.serve` 外层 Task 当成取消 owner，实际 `runtime.run()` 是其内部另一个 Task，
finally 不能证明精确的 `AppRuntime.run` 已回收；C 释放输入后直接取消或等待 runner，没有等待
`asyncio.to_thread(readline)` 的物理线程 `finished`，线程可能脱离 finally。T-74fbda R2 的 A1/A2/A3、真实
fixture 与直接 `AppRuntime.run` 的 B error/cancel 已由主审与独立 review 静态接受，未运行；T-aba7dc R3 已补齐
C 的 finally 尾项并由主审与独立 review 静态接受，未运行：释放物理输入后，把 finished 等待放进 try，把
runner retrieve/join 放进独立 finally；若尚未开始 read 则跳过 Event 等待。T-d8e47c 的 production/test review
未通过，缺口精确为：A1 同拍 caller cancel 与 child 完成时错误判断取消来源；A2 `shield` 已完成 fast path 后再次
`result()` 可能消耗取消异常及其 cause；A3 runtime/stop/restart/settings 后发生错误可能被 first-error 规则静默；
B1 cleanup Fiber 的 apply 永不返回；B2 第二次 caller cancel 未证明 runtime 只收到一次取消请求。T-f9462b 的 A1/A2
production 与 B1/B2 测试源、T-a3c9a2 的 production A3 与真实 ready/FD 边界已由主审及独立 review 静态接受，未运行；
但 R2 整批仍未通过，测试源缺口是 same-turn callback 调度后的 `serving.done()` 断言、原 runtime
`CancelledError` 身份，以及两条主路径的一次性 retrieve/finally。T-8e1435 的三项目标与 T-fd28c1 的一行 CLI R4 修正
已由主审与独立 review 静态接受，未运行；此前 restart transport_failure 的 cause identity 已恢复为
`assert caught.value.__cause__ is runtime_cancel_error`，不把旧 R2 恢复点当作本轮编辑前证据。T-13021b production
删除唯一 live cold-start `receipt.ready` 全局中止已静态接受；其初稿测试的 pre-bad 基线、dispose 窗口和
current-catalog 证据缺口不接受。T-0d18d9 R1 补齐主要 cold-start 结构，但仍因外层
`RUNTIME_CATALOG` 缺直接 import、以及未在 bad FAILED/downstream PENDING 存续窗口执行真实 peer scope 而整批不接受。
既有 T-4f17d6 T06-Host Cold-Start 累计 production/test 结论已由主审与独立 review 静态接受并冻结，未运行；它不是本轮
Selected-Load 接手点。该累计材料保留真实 `RUNTIME_CATALOG`、peer scope/permit、bad/downstream 状态、host 存续、
dispose 后 peer 身份/Effect/lifecycle/cleanup、SIGTERM/cleanup gate/runtime done callback、取消与 restart 的错误链，
以及真实 `StdioAppServer.run`/`ConnectionRouter`/`ControlService`/`Bus`/`Core`/`Manager`、EOF/OSError、HTTP/WorkspaceLock
结算；partial startup/task factory failure 仍是前一合同明确未扩展的边界。

#### T-0b0e05 · Selected-Load B R2 first-null 与真实 host owner 收尾（A R4 静态接受；B R4 已接受；A 行为验证见 T-fb9a7f）

T-1d7e36 的 production/test review 未通过，不是“只需运行”：fatal pre-Fiber owner、shared retention 清理集合、
catalog 当前 archive/state 字段和真实 source oracle 均有缺口。T-5c7b99 的 production `manager.py`/runtime catalog
P1-P4 已静态接受并冻结；T-11ad0b R2 仍不接受，尾项是 A3 peer mount 时机与真实 inner operation/cancellation
settlement、A2 gate finally、B3 runner error/cancel retrieval 和 C 的直接 `sys`/cleanup recovery。T-b6731d R3 的
stable 测试/文档及上述 A1/A2/A3 主要链已由主审与独立只读 review 静态接受，整批仅因本 R4 的 A/B 两项仍未闭合。
T-1174f2 的 Selected-Load A R4 已由主审与独立只读 review 静态接受，行为未运行：A3 释放 cleanup gate 后在
timeout 外等待真实 `load_task` 并用局部 finally 记录结果消费；B 用 `asyncio.wait` 分离精确 runner 的物理完成与
`.result()` 原始终态，保留真实 `CancelledError.__cause__`。不把 source compile 结论写成行为验收。
已有 `_active_generations`/`_building_roots` 负责渐进 owner 跟踪，selected archive 的 import、`from_module` 与身份核对
只在 live 入口窄 wrapper 中写入当前 generation 的原始 `load_error/state=failed`；legacy path 保持共享导入逻辑但不写
live 失败语义。清理顺序仍是 Scope/owner 先于 module removal，取消和 `SystemExit` 原样传播；没有 fake Fiber、incident
或 fallback Root。失败 generation 保留 exact current/selection，cleanup failure 才保留 current+draining owner 并由
explicit retry 推进，成功修复则再换代。

```text
selected archive
  └─ pre-Fiber live wrapper → failed generation/load_error
       ├─ owner cleanup → module removal
       └─ cleanup failure → current+draining retained → explicit retry
```

runtime catalog 同步投影 `archive_ref`、`state`、`load_error`、Fiber/ready 和对象身份 `cleanup_pending`；无 Fiber
不猜测 Fiber、incident 或 ready。A R4 的 cleanup gate 物理 join、精确 runner error/cancel retrieval 和现有 cold
restart 对账已静态接受。T-fb9a7f 随后完成唯一 selection 重读尾修，并按五组精确 nodeid 运行 Core T02：
selection 1/1、baseline 4/4、admission 3/3、registration 12/12（9 个 nodeid，含参数化 3+2）、
late_provide 6/6，共 26/26；artifact 为 `/tmp/i750-core.YE9wsb/`。测试仅使用隔离临时
plugin/home/workspace 与真实应用 import、局部生命周期/SQLite/archive/selection 夹具，未执行 migration、正式安装/部署、网络、
外部 MCP/server、业务服务、Gate 或 CI；正式 durable data delta=0。

T-0b0e05 的 B R2 修正不改已静态接受的 R1 source/artifacts/bootstrap 合同；R2 production、install/hot-reload 测试源与三份状态文档按静态事实接受并冻结。窄 typed content scan 保留 strict resolver 合同；Manager 是唯一
`_source_failures` owner，`plugin_status.source_failures` 与 selected generation/Fiber 失败分开。first-null 只对完整 prepare/compile
成功输入执行一次现有 `PluginSelection` CAS，全失败 `commit(())`；取消/CAS/共享错误原样传播，已有 selection 仍只加载 exact archive。
watcher/SIGHUP 不自动安装 repaired-unselected source，也不因 source scan 错误或暂失 deactivate；新增选择成员须显式 install，移除成员须显式
disable/uninstall，已选健康 source 更新仍走受控 prepare/replacement/CAS。source error 只在同进程同一 source 完整 prepare+compile replacement
成功后清除，metadata/revision scan 和重启 archive load 不能擦除；source diagnostic 不跨重启持久化，archive-only restart 没有真实 source
provenance 时只报告 source_unavailable/来源未知，不猜路径。无新增 durable owner/schema/writer，正常 archive 准备可能留下不可变产物，正式
durable data delta=0。正常不可变 archive 准备可能留下产物，首次 null 的 selection CAS 是未来运行会写入的持久选择；本开发未触碰正式
durable data，实际正式数据 delta=0，不等于产品运行无写入变化。

T-a4905e 的 R3 source 修复已静态接受；scan 8 项与两项 first-null smoke 真实通过，loader 历史在第 7 项停止的失败不是冻结测试 owner，而是 `_check_existing_schema` 的真实 `mode=ro` 连接未关闭。T-dfc862 R4 只在 `ReloadJournal` 构造期修复该连接的 `try/finally close`，并新增 valid-schema 与 missing-required-index 两个真实连接回归：旧实现两个参数均在“连接仍开放”断言处 red，修复后两个参数均 green；同一修复下完整 loader 10 个 nodeid 全部通过。测试只写隔离临时 workspace/selection，未执行 migration；正式 durable data delta=0。B R4 已由主审及独立 review 接受，不能写成 Issue750 或最终 enable 已完成。

T-855f8f 承接 Selected-Load A 的真实隔离行为验证，按 core/view/host 三组运行精确 nodeid：fixture、view/host 和 core 的其它 4 cases 已接受；原 7 cases 中 `test_live_cold_cancel_cleans_imported_generation_owner` 因冻结测试 oracle 以 `plugin_root.name` 误认归档末级 `tree` 为插件 ID 而失败，不能追认原整批全通过。该失败及 `/tmp/i750-av.SdkiWD/` 证据保留为历史事实，不把本轮已接受的 host fixture 重列为 pending。

T-648c5e 是该剩余项的定点行为验证：只修改 `test_live_cold_cancel_cleans_imported_generation_owner` 函数内协调，从 first 的真实 selection components/descriptor 建立有序 `(plugin_id, archive_ref, archive.open(code).resolve())`，observer 用 `(module_name, resolved code_dir)` 在当前 `_active_generations` 唯一匹配 exact generation，保存 scope/archive/error，并直接委托真实 `_import_plugin`。identity 与 public-install 精确测试各 1 passed；但该版本在 `load_all()` 原始 `CancelledError` 抛出后没有再次读取 selection，因此其“selection 未回滚”文字只是未闭合 claim，不作为已证明事实。其余 scope/module/importer、Root/active/draining/building cleanup 与 public-install 的 accepted→selected failed B→显式 fresh retry 事实保留；两组 artifact 为 `/tmp/i750-a-id.2CDBQG/identity/{console.txt,junit.xml}` 与 `/tmp/i750-a-id.2CDBQG/public-install/{console.txt,junit.xml}`。

T-fb9a7f 补上唯一允许的 `assert manager._selection.read() == selected`，位置严格在 `load_all()` 取消之后、`terminate_all()` 之前；selection identity 测试 1/1 通过。其余五组 Core T02 局部行为也均通过：baseline 4/4、admission 3/3、registration 12/12（9 个 nodeid，含参数化 3+2）、late_provide 6/6，共 26/26；每组均有独立 console/JUnit artifact，根目录为 `/tmp/i750-core.YE9wsb/`。本轮实际执行隔离临时 plugin/home/workspace、真实应用 import 与局部生命周期/SQLite/archive/selection 夹具，未执行 migration、正式安装/部署、网络、外部 MCP/server、业务服务、Gate 或 CI；正式 durable data delta=0。该结果待独立 review，不构成 Issue750 或 final enable 完成。

本 R2 沿用且本批核对的 owner 图固定为：

```text
结构校验（pointer/cache/path/symlink/existence）
        │
        ▼
所选 source 内容分类（identity / compile；真实 SyntaxError/UnicodeError）
        │
        ▼
Manager 受控诊断与完整 prepare
        │
        ├── first-null 单 CAS ──► live Root
        └── selected replacement ─► live Root

watch_revision 线程 ──纯读 scan/metadata──► Manager reconcile owner
```

T-830fd2 原稿缺口包括 installed pointer 双目标内容误解析、installed artifact 消失被降级、猜测 source root、诊断丢失底层错误，
以及 hot-reload/runtime-smoke 缺少真实 CAS/取消/共享错误和 AppRuntime/host first-null 链；T-d338ca 的 R1 已在 13 个路径内补齐这些
主要源与边界，T-0b0e05 R2 继续修正 P/T/D 尾项，未运行行为测试。

T-b0ab9e 的 Models 累计静态结论作为已知事实保留，不追认原失败稿。本批未改 Message、schema、cursor、
receipt、descriptor、root_ref、metadata、credential、archive/history/selection/journal 或正式 durable data。
T-13021b 历史上移除 live initial 的全局 `receipt.ready` 中止；本轮 T-dfc862 只改 journal schema-read close、对应真实回归与状态文档。其它 scope 外
provider/RPC/consumer、UI 外
`current_snapshot`/fence、offline/trusted watcher、candidate/freeze、Commands live binding/provenance、
T06/T07、行为验收与 final enable 仍为 WIP。R3/R4 的 targeted 测试已执行相关应用 import 与隔离临时 SQLite/selection 写入；没有运行其它测试、Gate、CI、正式 workspace、插件安装、业务进程或部署。

**DSH 对照：** 替代行为已经由 T02～T05 的 D01～D07 解释。
最后一行这类经证明无副作用、无独立功能的冗余删除，不要求引用 DSH。

**退出条件：** 正常安装调用链不再经过旧发布模型；不存在两个版本选择 writer；
删除不是靠关闭错误检查、丢弃清理责任或删除历史数据获得。

### T07 · 显式升级与最终累计验收

**输入与下游：** 单图实现及已选输入 → 可从当前安装状态启动，并交付可复核的真实结果。

**范围：** 现有显式升级脚本、安装/集成夹具、文档和按授权安排的跨仓库验收；不自动操作正式 workspace。

**怎么做：**

1. 先在副本盘点旧 selection、制品、环境、安装清单、未决 journal 与资源回执；真实未决效果不能被写成“迁移成功”。
2. 从原正式选择生成新加载输入，不扫描 latest 猜测目标。初始空选择和损坏/缺失状态分开处理，不在普通启动中自动修复未知格式。
3. 若格式兼容，复用原子选择文件，不为了新术语重写全部历史；若必须转换，显式执行、先备份、逐项核对前后输入。
4. 更新文档与现行合同；旧决策保留历史并标明被取代，不整段删除证据。
5. 运行第 6 节累计验收。先用可控插件夹具及兼容当前 API 的本地插件，再用正式安装链验证外部插件。
6. Fitbit 是场景说明，不把旧 API 的外部 worktree 当成已通过的生产实例。外部源码适配、安装、真实资源验证单列结果。
7. 按当前 WORKFLOW 和维护者授权执行相关测试、独立只读评审、Gate；源码通过、运行通过、CI 与正式部署分开报告。

**DSH 对照：** 加载结果参考 D04/D06；持久化转换、消息保留与部署恢复属于本项目合同，不移植 DSH 缓存回退。

**退出条件：** 第 6 节全部有实际证据；只读历史和真实清理 owner 仍可用；
不存在为了通过验收而隐式全局重启的分支。正式部署需单独批准。

## 6. 最终验收：怎样证明真的没有重建全局图

以下是实施后的验收要求，本次文档工作没有执行这些测试。

### 6.1 最重要的三个判据

1. **身份不变。** 更新前后 Root、boot 和无关模块/Fiber/服务对象相同；无关任务和实际进程没有替换。
2. **生命周期不动。** 无关插件 import/apply/start/stop/cleanup 的计数不增加。即便复用了 Root 对象但偷偷重装全部插件，也必须判失败。
3. **请求继续。** 让 Fitbit 在途调用受控暂停，同时让无关插件完成一个真实请求；随后释放 Fitbit 调用，局部换代完成。不以 HTTP 200 或列表非空代替。

在安装测试中将完整 Root 构建入口设为“被调用即失败”。
替换旧 [fresh-root 测试](../../tests/test_plugin_fresh_root.py) 的全量新实例目标；
重写 [整图卸载测试](../../tests/test_plugin_uninstall_root_drain.py) 的等待边界，
但保留真实资源退出、抗取消与错误传播断言。

### 6.2 其余必要场景

- 新增插件不动旧插件；新 provider 出现只激活实际等待它的节点；相同输入 active 时不重复初始化。
- 硬消费者按依赖关闭/恢复，可选子 Fiber 只影响自身；公共 Tools/UI provider 不因贡献变化重启。
- 编译失败保持选择和运行图；动态配置、缺依赖、冲突、环、apply/必要启动失败不误报 active。
- 新插件 LOADING 时半成品工具/路由不可调用；失败后没有残余登记或无主资源。
- 旧 ToolRef 明确失效，当前获授 view 更新；没有借名字静默替换实现、提升权限或重放业务效果。
- 内部 Shell 安装不自等；更新调用者所依赖服务时也有明确结果，不无限等待整个 Turn。
- 有限排空、清理失败、取消和迟到完成保留责任；普通安装失败不把无关分支停掉。
- CLI/RPC、Tools、MCP、UI、后台来源和模型实际消费者均覆盖，不只验证内核夹具。
- B 激活失败后不自动启动 A；正常重启使用已选 B，不依赖旧 candidate/journal 推进。
- 安装选择提交竞争有明确冲突；同一请求不重复应用；批量部分失败如实报告。
- 不支持在线更新的共享环境变更明确拒绝，没有“失败后重启整机试试”的隐式兜底。
- 业务功能错误可在正常调用中暴露并再次安装修正，不需要系统晋升协议。
- 安装/更新/卸载不减少 Message、附件、plugin-data、历史 binding、归档与回执。

计时分别记录准备、局部排空、关闭、加载、必要资源就绪。
不预设加速倍数；“不重建”的身份与行为证据先于性能数字。
所有并发场景优先用事件屏障和受控调度，不依赖 sleep 猜测先后。

### 6.3 持久化与恢复边界

| 对象 / owner | 正常增加与允许更新 | 逻辑失效与物理减少 | 恢复证据 |
|---|---|---|---|
| 代码制品、环境、输入记录 / 安装链 | 新安装增加固定材料，不原位覆盖已使用代码 | 旧材料可不再被当前选择引用；本轮不自动 GC | 输入摘要、原路径要求、归档及环境备份 |
| 已选择输入 / 安装应用入口 | 持锁、检查前驱后原子替换；旧记录保留 | 旧选择退出当前作用，不抹掉历史 | 原 selection、前驱、可读的完整制品 |
| 原 journal、验证消息、binding / 原 owner | 本轮停止旧推进路径；读取和真实资源结算按原 owner | 不因代码删除而删数据库、伪造终态或清理候选数据目录 | 原生备份、未决清单、真实资源回执 |
| 临时任务、连接、进程 / Scope 和 provider | 创建时登记归属；状态由真实 owner 更新 | 只释放自己已确认的临时资源；失败保留句柄和依赖 | 关闭结果与外部宿主/Controller 的身份、回执 |
| plugin-data、消息、附件 / 业务 owner | 仍按原协议增加、迁移及更新；消息正文正常只追加 | 普通安装/卸载无物理删除权 | 按[状态地图](persistence-state-map.md)备份并核对完整性 |

实现代码可以恢复到旧提交，不代表新插件写过的数据能被旧代码读取。
不得把 Git 恢复点、旧 selection 或旧模块缓存当成数据回滚证明。

### 6.4 T01 验收映射（本 worktree 副本补充）

以下是验收设计，尚未运行；本工单禁止执行测试、Gate 与 CI。
它给 T02～T07 同一套判断标准，不改变上文第 6 节既有要求。

**最小验收拓扑：** ownership 边与服务依赖边分开，累计最小集覆盖以下角色；
专门负例按目的选择相关角色，不必机械包含全部角色。

~~~text
┌ host（不硬依赖 changed 的稳定宿主）──────────┐
│  ├─ optional child Fiber ──inject──▶ changed 的服务
│  └─ 其他无关子功能                          │
└──────────────────────────────────────────────┘

hard consumer ──inject──▶ changed 的服务
changed ──owns──▶ 自有子 Fiber（随 changed 启停）
unrelated：与 changed 无实际依赖的长任务 / 在途真实请求
~~~

- `changed`：被安装、更新、禁用或卸载的插件；它拥有的自有子 Fiber 随其退出。
- `hard consumer`：把 changed 的服务作为硬依赖，先停后恢复。
- `host` 与其 `optional child`：host 本身不硬依赖 changed；child inject changed 的服务。
  changed 变化只让该 child 停启，host 与其无关子功能的身份和生命周期计数不变——
  这才能发现“把可选消费者的整个宿主插件重启”的实现错误。
- `unrelated`：与 changed 无实际依赖的长任务/长 Turn；其请求持续、身份与计数不变。

| 现有测试 / 调用路径 | 保留的保护 | 被批准取代的整图断言 | 新行为 oracle | 落点 |
|---|---|---|---|---|
| `tests/test_plugin_fresh_root.py`（候选与正式组合每插件全新实例） | 新节点确实是新实例，不复活旧对象 | “每次更新候选+正式组合全部重建” | 更新前后 Root、boot、无关模块/Fiber/服务/任务/进程身份相同；无关 import/apply/start/stop/cleanup 计数不增加 | T02、T06 |
| `tests/test_plugin_uninstall_root_drain.py`（整 Root 卸载排空） | 真实资源退出、抗取消、错误传播 | 整图 lease 归零才允许换代 | 只等待受影响分支的实际在途调用与清理；unrelated 长任务不阻塞 changed 更新 | T02、T03 |
| `tests/test_plugin_publication_admission.py`（`test_publication_wait_outlives_commit_deadline` 断言超过 deadline 仍 publishing 且无错误） | 停止可等待、失败含义不丢 | “超过提交期限仍继续 publishing”这一旧行为本身 | 排空超时只终结等待与报告；保留实际 owner、关闭状态与失败原因，不扩成全局停服。有限排空是新 oracle，不由该旧测试保护 | T02、T04 |
| `manager.py` `_switch_ready` / `_replace_formal_root` / `_build_and_publish_root` | 一次冷启动建 Root、真实退出 | 候选→正式整 Root 替换、失败重建旧组合 | 完整 Root 构建入口在安装测试中设为“被调用即失败”；新增/替换/禁用/卸载只挂载/撤下相应节点 | T04、T05、T06 |
| `snapshot.py` RuntimeSnapshotCompiler/Store、跨 snapshot lease | 只读结果对象可保留 | 多运行图 store、全图 lease、跨图发布事务 | 单 Root；调用保护按实际 owner/activation；B 激活失败不自动启动 A | T02、T03、T06 |
| `reload_journal.py` 候选晋升/旧图重建推进 | 真实资源账由原 owner 保留、只读历史可查；不重新拉起、不续跑未提交验证（0056 既有约束） | 双图恢复推进、以旧图重建作为恢复路径 | 改变的是持久选择的提交时点与恢复目标：进程在选择提交后退出时，下次正常启动读取 B，不续跑未完成的应用，不猜 A 安全 | T06、T07 |
| `plugins/plugin_update` latest run/revert、open_validation、publish/discard | 普通 programmatic 调用能力 | 候选试用/晋升/撤销专属入口与授权 | 内外安装同一 accepted/active 语义；旧入口明确报告不再支持，不静默映射 | T05 |
| `plugins/ui` register/seal、Dashboard、Mobile 查询 | 模块合同检查、资源 owner、客户端 revision | 整图 seal 一次后不可再登记 | 目录随贡献 Fiber 登记/撤销增减；不重建所有插件或重启所有 Dashboard 后端 | T03 |
| Tools/RPC/MCP 调用边界（`runtime_scope`、bindings） | 工具授权、schema、ToolResult 只结算一次、历史 binding 事实 | 整 Turn 全图 lease | 停止 owner 前关闭新调用接纳并等待实际在途调用；旧 ToolRef 明确失效，获授 view 取新引用，不重放效果 | T03 |
| `selection.py` PluginSelection 原子写入 | 持锁、前驱基线检查、原子替换、旧记录保留 | “stable=整图已验证成功”的语义 | 选择是输入引用列表；accepted≠active；写入结果不确定时停止并报告，不回写旧值 | T04 |
| 编译/构建与可提前检查 | 制品固定、原子选择、错误传播 | 用完整候选图当预检 | 编译失败不改变选择和当前图；不运行全量构建（负控：全量构建被调用即失败） | T04 |
| 启动失败路径（新实例 apply/资源就绪失败） | 清理已取得的 B 资源、保留失败事实 | 自动恢复旧组合 A | B 标为失败、硬依赖消费者不可用并如实报告；不自动启动 A；普通重启使用已选 B | T02、T04、T07 |
| 清理失败路径（PLG-006、Fiber dispose） | 逆序、抗取消、保留失败 owner/句柄/依赖 | 整 Root dispose 兜底 | 单支路清理失败保留原 Fiber、句柄与必要依赖，可查询；不升级为 Root.dispose | T02 |
| 内部 Shell 自安装（`plugin_updates`、install 调用栈） | 来源 Task、权限、真实退出责任 | 持有 lease 的调用等待自身排空 | 发起调用先返回 accepted，宿主持有应用任务，避免自等；内外 CLI 同一语义 | T04、T05 |

**oracle 使用规则：**
- “无关”按实际依赖判定，不按插件名硬编码；累计最小集必须覆盖上述全部角色，专门负例按目的选择相关角色。
- 身份证据优先于性能数字；计时分别记录准备、局部排空、关闭、加载、必要资源就绪。
- 并发场景用事件屏障与受控调度，不依赖 sleep；HTTP 200 或目录非空不证明 active。
- 修改既有测试时不得只改断言掩盖未迁移消费者；被取代的整图断言逐条标注取代依据。

## 7. DSH 实现索引与借鉴边界

所有 DSH 引用均为本机固定源码基线 `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8`。
根目录为 `/mnt/data/source-code/deepseek-harness`；行号用于此次核查，长期以符号为准。

| 编号 | 具体实现 | 本方案采用什么 / 不采用什么 |
|---|---|---|
| D01 | [Fiber._refresh / _setEpoch / _reload / _unload](/mnt/data/source-code/deepseek-harness/vendor/cordis/src/fiber.ts:611)；[Fiber.await](/mnt/data/source-code/deepseek-harness/vendor/cordis/src/fiber.ts:704) | 按依赖身份变化启停，等待初始化；不照搬只记日志后继续的清理错误策略 |
| D02 | [ReflectService.provide](/mnt/data/source-code/deepseek-harness/vendor/cordis/src/reflect.ts:277)、[notify](/mnt/data/source-code/deepseek-harness/vendor/cordis/src/reflect.ts:314) | 服务登记归 Fiber Effect，变化仅刷新相关消费者；允许扫描登记，不要求第二套反向图缓存 |
| D03 | [Entry.update](/mnt/data/source-code/deepseek-harness/vendor/loader/src/config/entry.ts:142) | 无变化不动、局部禁用/替换；214 行起预导入不等于无副作用；232～241 行自动启动旧实例不采用 |
| D04 | [Entry._await / _start](/mnt/data/source-code/deepseek-harness/vendor/loader/src/config/entry.ts:269) | 等待 Fiber 初始化并传播错误，不把包准备完成当作运行就绪 |
| D05 | [ScopedLayers.effect](/mnt/data/source-code/deepseek-harness/packages/core/scope/src/store.ts:226)、[Tools.register](/mnt/data/source-code/deepseek-harness/packages/core/tools/src/index.ts:1027) | 登记与撤销归贡献 Context；不引入多代 Overlay，DSH 的 agent scope 层也不是版本 snapshot |
| D06 | [PluginInventoryGateway.list](/mnt/data/source-code/deepseek-harness/packages/host/plugin-inventory/src/index.ts:53-88) | 直接读取 Loader Entry/Fiber，不另存一套生命周期状态 |
| D07 | [runPlugin](/mnt/data/source-code/deepseek-harness/apps/cli/src/plugin.ts:120)、[reconcilePlugins](/mnt/data/source-code/deepseek-harness/apps/cli/src/plugin.ts:59) | 包安装成功后更新加载输入；这段 CLI 自身不证明在线激活成功 |
| D08 | [HMR.partialReload](/mnt/data/source-code/deepseek-harness/vendor/hmr/src/index.ts:400)、[缓存与插件恢复](/mnt/data/source-code/deepseek-harness/vendor/hmr/src/index.ts:482) | 理解模块依赖与运行依赖的区别；不复制缓存恢复、源码 watcher 和完整 HMR |
| D09 | [HMR 框架变化退出](/mnt/data/source-code/deepseek-harness/vendor/hmr/src/index.ts:259)、[PROFILE_TEMPLATES](/mnt/data/source-code/deepseek-harness/packages/boot/app-boot/src/profile.ts:110) | 明确 DSH 存在全量退出及 startup/live 策略；不把这些当成普通安装全量重建的许可 |

这不是“DSH 原样搬过来就天然满足全部合同”。
DSH 给出局部组合的具体实现；Akashic 的持久消息、严格清理与安装身份仍需按各自边界实现和验收。

## 8. 当前交付与恢复点

静态与局部行为状态：T04 Manager 局部 Loader、generation Fiber、精确 host facade、OwnerCall/Context binding provenance、有限 operation owner，以及 B2 live RuntimeCatalog/stable_view/runtime_inspection adapter 已完成当前批次静态实现；T-fb9a7f 的 Selected-Load A/Core T02 局部行为测试已通过 26/26（selection 1 + Core 25）。T-e87866 的 Commands/Message/Reply 精确验证为 provider 13/13、message 9/9、reply_entry 0/1，合计 22 passed、1 failed；失败发生在真实 Reply→Source→Conversation 入口，`Conversation` 的 `MessageWriters/TASKS` owner call 缺失，证据与下一批最小边界见 T03 Commands 段，artifact 为 `/tmp/i750-cmd.l6NEiF/`。本批只使用隔离临时 fixture，不改 production、测试源码或正式 durable data；其它 provider/RPC/consumer、Source wake、Subagent、`current_snapshot`/fence、offline/candidate/freeze、T06/T07、完整回归/Gate/CI、运行验收和 final enable 仍为 WIP。
当前仍是 WIP：其它 provider/RPC、scope 外 manager 消费者、UI 之外的 `current_snapshot`/fence、offline/candidate/freeze、T06/T07、完整运行验收和最终 enable；本轮结果待独立 review，不将其外推为 Issue 750 完成。

前两轮 Devin 评审任务 T-edb4be、T-1e8840 针对旧方案。
其中对真实代码、owner、数据与资源边界的核查仍有参考价值；
对整进程发布协议的认可或修改意见不能算作本版局部换代方案已评审。

T01 文档修改前的恢复点：`/mnt/data/coding/issue750-t01-recovery-mXOWmV/`；
T01-R1 修订前版本另备于 `/mnt/data/coding/issue750-t01r1-recovery-lmFh4V/`。
第三版设计与编辑前索引的恢复点（历史）：
`/mnt/data/coding/issue-750-design-v4-recovery-20260920.t1JYe8/`。
实施前重核主线及在途 0071 工作，不能直接将本基线的行号和待删列表套到新主线上。
