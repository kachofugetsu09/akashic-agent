# 0072 · 单张运行图与局部插件换代

T-d80010 → T-673b11 是 Reply/Source owner R2/R3 的最小取消结算收口。R1 整体概念/测试合同
不接受；只保留 `TaskServiceClosed`、`formal=False` 普通拒绝、typed 接纳窄边界和 owner
capture 事实。R2 用 `Task.on_done` + 局部 Event 等待 physical finally，再用公开 `Task.join()`
读取 Source Task 终态，不创建 join waiter、不依赖 task factory；raw monitor 与 Source Task
capture 仍是两条不同事实，不增 registry 或授权 scope。R3 在同一链上记录 marker/caller 同拍
取消的可交付外部事实，并保留带真实 cleanup cause 的 child CE 原对象；不盲目 uncancel、不伪造
未被平台独立交付的 caller message/object，也不把纯 owner cancellation 误报成 cleanup error。

R2 的 I/II/III 10 次 green 与 `/tmp/i750-reply-r2.wZ6E5X/` controlled mutant 是历史事实；旧
command 不能独立核验本轮要求的环境隔离，controlled mutant 也不能当作 R1 原样 pre-fix red。R3 在 production-before 上实际复现的 B red 位于
`/tmp/i750-reply-r3.9rZYlv/B/`，A 同拍序列实际 green 并如实保留。R3 修复后 I 9/9、II 1/1、
III 2/2；最终 artifact 为 `/tmp/i750-reply-r3.5eIkkK/{I,II,III}/`，targeted artifact 见设计段和回执。AST/内存 compile、`git diff --check` 与
frozen152/152 通过，仍待主审与独立只读 review；不改变 Message/schema/receipt/selection/
journal/history/archive、正式 workspace 或 durable data，也不代表 T03 全部消费者、T05 公开
入口、Gate、部署或 final enable 已完成。

T-673b11 的 R3 整批随后被主审与独立 `gpt-5.6-terra/xhigh` 概念闸门暂拒；这是历史审查
结论，不把 R3 的 green 记录当作当前接受。T-cfe587 的 R4 production 修复随后已由主审与
独立概念 Gate 接受并冻结；T-d44273 R5 只修测试 cleanup/oracle 与执行记录。当前 iteration 单独持有
`source_settlement_errors`，只接收 exact Source Task 的同步 cancel/on_close 失败与物理
`Task.join()` 终态错误；SourceSession/Tasks 仍是 Task owner，不新增 registry、Task factory
或跨任务 owner 表。monitor、captured scope、Source Task physical settlement 全部完成后，
source-only 错误才以原异常 `exc_info` warning 并停止当前 drive；R1 的“不重试”仅针对该 drive
完成结算后由 unrelated registration 变化触发的唤醒，不覆盖同 Session peer 的既有
changed-session 重扫；
caller/follower hard error 存在时，原 Source errors 进入最终 error tree。纯 owner cancellation
仍忽略，普通程序异常仍走既有 source-local warning。

```text
Source Task → physical cancel/join → source_settlement_errors
                         ↓
             local final decision after all cleanup
              ├─ source-only: warning + stop drive
              └─ caller/hard error: preserve and propagate error tree
```

R4 的真实行为证据：production-before 两项 C 回归 exit 1，原始 console 为
`/tmp/i750-reply-r4-red.LuEiTO/console-production-before-bound-final-red.txt`；C1 是 child CE
被 TaskGroup 忽略、source-local warning 缺失，并非 watcher 被击穿，C2 是 sync `on_close`
error 进入 TaskGroup、健康 peer 无法完成。修复后 console 观察到 targeted 2、I 11、II 1 绿，
但缺少完整 command/JUnit/逐 attempt 隔离与 timeout，不能升级为独立可复核验收。T-d44273
R5 新 harness artifact 为 `/tmp/i750-reply-r5.b4WgrJ/`：I JUnit 11/11、II JUnit 1/1，
均 exit 0 且 0 failure/error/skip、未超时；cwd、完整 argv、源 hash 与恢复点记录在设计文档
T-cfe587 段。R5 仍待主审与独立只读 review，不能外推为 Issue750 system Gate、CI、部署、
正式运行或 final enable。

```text
Reply/Source admission
  ├─ raw monitor（空 Context，只等 admission）
  └─ Source Task capture（独立 owner 事实）
       ↓
current drive cancel → exact Source Task at-most-once cancel
       ↓
Task.on_done physical finally → public Task.join → permits release
```

T-a9246e 的 Source registration wake 方向保留，但该批 production/test/docs review 暂不接受；
R1 是历史交接点，当前 Source wake 状态见下方 R2/R3。它仍不增加 Core registry、Task factory、Message callback 或
持久 cursor：`Sources` 的唯一 registration record 在 Effect 建立时先处于 LOADING，所有同名和
channel 冲突都对现存 registrations 检查；只有该 record 所属 Context 的 `ctx.spawn` 通过本
activation-ready 闸后，`entries()` 才看到 Source。只读 `changes()` 只通知 follower 唤醒，
不调用 Reply/business callback，也不承担 Message 持久化。

```text
registration Effect
      │ exact record + activation-ready spawn
      ▼
visible Source snapshot ──┐
catalog heads snapshot ───┴→ follower local wake/event
                              └→ TaskGroup drive
                                   ├─ replacement: settle old → start current
                                   └─ revoke/error: stop affected drive only
```

P1a 的 route 由全部 registration 决定，visible 只控制 `entries()`/Reply discovery；LOADING
dedicated 由真实 owner admission 返回 `OWNER_UNAVAILABLE`，不回退 default。P1b 的 watcher 只
唤醒，main action 与每轮 Source scope 前都从当前 `_items` 核对 exact identity；P2 的 safe-create
在 TaskGroup factory 失败时关闭未交接 coroutine，并清理 drive bookkeeping。`follow` 仍按
`(session_id, source_name)` 串行排空 replacement/revoke，W2/W3/W4/W5 的真实隔离 oracle 见设计
文档 R1 段；测试不改变正式 Message/schema/receipt/history/archive 或 durable data。

本 R1 的最终 evidence 是 targeted 9/9、Reply full 20/20、default command 1/1，分别位于
`/tmp/i750-source-wake-r1-targeted-final.0HLrHb/`、`/tmp/i750-source-wake-r1-full-final.k1EzkS/`、
`/tmp/i750-source-wake-r1-command-final.UCcZDG/`；最终测试版本重绑定的旧 production W1 red 位于
`/tmp/i750-source-wake-r1-final-red-w1.g0epFu/`，其失败是历史 late-registration wake timeout，
不是 setup/import error。P1a/P1b/P2 的 before-red artifact、四文件 source manifest、恢复点和
当前 hash 见设计文档；frozen152/152、AST/compile、diff-check 通过，正式 durable delta 为 0。

边界必须保持准确：若 Source registration 仍存在且发生 source-only settlement error，同 Session
peer 的后续 Message 仍可能触发现有 changed-session 重扫并重试旧 Input；本 R1 的“不重试”只
覆盖 unrelated registration 变化，不是所有 peer 变化。failure/retired registry 是 PROHIBITED
ALTERNATIVE，不是实现 TODO；该 same-session gap、其它 provider/RPC/consumer race、snapshot/fence、
offline/candidate/freeze、T06/T07、累计 Gate/CI、部署、正式运行与 final enable 仍为后续 WIP，待
主审与独立只读 review。

T-cef8cd 记录 R2 收口：T-b9bd66 的 P1a/P1b/原始 P2 接受事实保留，但 R1 整批因 eager
factory P2 与 W2/W3/W4/W5 oracle/cleanup 缺口暂拒；审查模型为 `gpt-5.6-terra/xhigh`。
R2 的唯一 production 修复在 `plugins/reply/follow.py:schedule`，先把 exact wake 写入
`active[key]` 再 create drive，create 失败只删除仍指向该 wake 的键；Sources production
`plugins/sources/plugin.py` 及 Core/Task/SourceSession/Manager 不变。最终 production hash
为 `66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247`，测试 hash 为
`86cb2a0e2b7efdf896f54a448f287532d1ec72ca3479eb90de9d306c7e82d56b`，R1 起始 hash 见设计文档。

R2 测试源实际覆盖：queued drive 撤销前首指令、eager closed-admission 同名换代、W2
production public `join`/`on_done` 与 raw Fiber permits、W3 独立 replacement
contributor/三窗口 peer identity/Effect/lifecycle/cleanup、W4 exact child CE/cause 与 follower
main `entries()` action barrier、W5/P1a 原 Task 一次性 retrieve。它仍保留 R1 的真实
live Root、Message Input/Output、Effect 与 owner scope 边界，不添加 registry、cursor、
failure/retired 表或 same-session retry 机制；same-session peer retry gap 仍是历史 WIP。

R2 production ordering 已由主审与独立 `gpt-5.6-terra/xhigh` 概念 Gate 接受并冻结。R2 evidence：
`/tmp/issue750-source-wake-r2-evidence-20260923/eager-before-final/` 在起始
Follow hash 下为 1 test/1 behavior failure、exit 1、无 timeout；同一最终测试源的
`eager-after-final/` 为 1/1 green；`targeted-r2c/` 为 8/8，
`full-reply-r2/` 为 22/22。新 harness 明确区分 `/usr/bin/python` 3.14.7 的
harness 与 `/mnt/data/coding/akasic-agent/.venv/bin/python` 3.13.7 的 child，child
使用 pytest 9.0.3/pytest-asyncio 1.3.0；四文件 pre-exec source snapshot、argv/cwd、
isolated roots、console/JUnit/exit/timeout 均保留；static/frozen153、default command 1/1
与上述 exact R2 checks 已完成。R2 source/test 整批仍不接受，原因是 T1 queued oracle 与
W2/P1a/W3/eager cleanup 缺口；上述是局部行为证据，不是 Issue 750、Gate、
CI、部署、正式运行或 final enable 验收；formal durable data delta=0。

T-6ee259 的 R3 只收口测试生命周期与状态文档：不引入新行为机制、failure/retired registry
或 same-session retry。R3 targeted-r3 5/5、full-reply-r3 22/22 的 artifact 与 SHA 保持有效；
queued/W3/eager 独立 Effect cleanup 已接受，但 whole R3 因 W2 lexical binding、bounded wait
超时无界 fallthrough、W2 collector cause 误忽略三项异常路径缺陷暂不接受。

T-26b5a5 的 R4 只补上述三项：W2 `collector_joined` nonlocal；W2/P1a/W3 bounded wait 超时
立即失败，finally 释放 gate 后用无 timeout 的 `asyncio.wait` 物理收口并仅消费 exact result
一次，删除 retrieved 冗余标志；W2 只忽略本 cleanup 主动且无 cause 的纯 `CancelledError`。
新 artifact `/tmp/issue750-source-wake-r4-evidence-20260923.x4nLCk/` 的 targeted-r4 为 3/3、
full-reply-r4 为 22/22，均 exit 0 且无 timeout；AST/compile、W2 symtable、source fallthrough
检查、`git diff --check` 与 frozen154/154 已核验。测试 SHA 为
`7c060ed60332ca06a618bdfc9c0dc8ef873d6c42425e92da207b4f82fb55614d`，production frozen hashes
仍为 follow `66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247` 与 Sources
`b9a574264bc36b27bc065939d8bab6d9497141d53d0ff622f262d41cadf75875`；R4 已由主审与独立只读
review 接受，不外推为 Issue 750、Gate、CI、部署、正式运行或 final enable 完成。

T-8677b9 的 R5 仅在 Reply follower 内加入 `observed_source_heads[(session_id, source_name)]`。
Message 变化与真实 Source identity-history 合并为单次扫描；late registration / replacement
对历史 head 无条件建基线，且在 `schedule` 前写入；后续 peer 只会唤醒 source head 实际前进的
条目。真实旧实现 red 是 peer 已被 follower 扫描、健康 conversation Output 可见后，fault
Source 启动数由 1 变 2；最终源码下 targeted 1/1、Reply 23/23、冻结 default 1/1、两个
SourceSession controls 2/2 green。artifact 与六文件 pre-exec hashes：
`/tmp/issue750-source-head-evidence-20260923.mj1Qbk/`。observer 是易失通知基线，不是 terminal
receipt；重启或 registration replacement 仍可能按现有日志合同重读历史。无新持久化写入，
R5 production 精确 hash `24aabacea42855c20f8bb0fed47f17b34f708cb32fa5672dad9f719b6fa37b18`
已通过主审与独立 `gpt-5.6-terra/xhigh` 概念 Gate；R5 整批当时因本测试 A/B 结算证据缺口
未接受。

T-a0440f 仅修该回归函数：异常路径在 `running` fixture 退出前释放 gate；局部 `Task.join`
delegate 保留真实结果/异常，核对首个 child CE/cause/warning 对象与第二个 Task 成功
`on_done`/follower join，移除 test-side 二次 join。原生产 hash `66edde08c3b6b87b5ae87e3588543f3ef0522b94c54bf947769c94368afa0247` red 1/1 行为失败，修复后 targeted 1/1、Reply 23/23 green；新证据及四源快照在
`/tmp/issue750-source-head-r1-evidence-20260923.bhy0c4kf/`，最终测试 SHA 为
`561fe762c686617e248999ffba554fa11b49eb60f0b07b755eb4eadf361e2692`。生产未改；T-a0440f 测试收尾
已接受并冻结。其它 provider、Subagent、snapshot/fence、offline/candidate/freeze、
T06/T07、累计 Gate/运行验收/final enable 仍未闭合。

T-82189d 将 Subagent parent report 的调用从 `conversation.v1` factory 改为唯一
`conversation.complete.v1` callable：Conversation owner 包住 SourceSession `complete(program)`，
Subagent `_announce` 等待该调用，Reply `report` 建立 Reply 自身 scope；旧 key 不作兼容 alias，
旧 consumer 留在 PENDING。Subagent 12 项为 1 passed/11 failed；Reply follow 23/23 与 message
commands 10/10 通过。报告链仍在 `ToolProgramFactory.bind_reply` 使用 Tools owner Context 时因
缺失 OwnerCall 失败；本任务不得修改 Tools/Core。局部 drain 已观察到旧 Task 物理完成，但
`delivery_policy` FAILED 阻断 Manager readiness；Core builtin 测试仍依赖 scope 外的
`current_snapshot` 分支。8 项复验为 5 passed/3 failed，证据在
`/tmp/i750-subagent-I.k0xvgM/`、`/tmp/i750-subagent-II.USNRxD/`、
`/tmp/i750-subagent-III.cXfAGb/`、`/tmp/i750-subagent-IV-final.BeYCY5/`。未改 formal durable data，
不构成 Issue 750 完成或行为验收通过；需解决/授权 Tools owner 边界并重做可运行的 drain readiness
验收，再交主审与独立只读 review。

T-b0ab9e（T06-Models R2）上批已知事实：T-fa9271 删除 Core 的 `RuntimeModelControl`
与无消费者的 `model_catalog` companion、T-83b918 的测试/debug scope、错误传播、in-flight、
Path/catalog 证据及文档修订均保留；上述 R1 与 T-b0ab9e 的 `reader.__self__` 替换已由主审
与独立只读 review 静态接受。T-fa9271/T-83b918 原失败稿不追认为整批通过。未改
Models/Core/clients/Bindings/Manager production，未运行行为测试：

```text
公开 RPC / 真实 Dashboard / 窄 reader
  └─ 同一 live Root → exact provider Context
       → Models / selected driver scope → 原 SQLite / 真实 model_calls
```

ModelsStore/CAS、credentials、descriptor/schema、Message、model_calls、continuation、
embedding identity 与 optional chat injection 保留。wrapper 专属的 snapshot lease、
lease-count 和 503 oracle 不再作为当前模型合同；其它 provider/RPC、scope 外
`current_snapshot`/fence、offline/candidate/freeze、T06/T07、行为验收与 final enable 仍是
WIP。

T-131de5 承接 T-e721a4、T-cbecef、T-2f3d82、T-518ba4、T-a2817b、T-7aeefe 与 T-7912fd 的当前文档对账：Dashboard R3 生产接线与 T98 A1-A3 测试源已静态通过；Mobile T03a-UI R2 已由主审与独立只读复核静态通过，测试未运行。T98 的 Mobile 整批 review 不通过；R2 只闭合 `aclose` 先取消、后 executor shutdown 失败时的双错误保留，以及真实二次取消、timeout、组合异常和 finally 清理 retry 测试源。T-a2817b 的 Core T02-N1 整批 production/test review 未通过；T-a510da 的 P1/P2 生产修复与 T-518ba4 的四项 R2 测试源已由主审与独立只读 review 静态接受，测试未运行；T-2f3d82 的 N2 production hook 与 T-cbecef 的六项测试源、文档已由主审及独立只读 review 静态接受，测试未运行；T-e721a4 的 production inspect 已由主审与独立只读 review 静态接受；T-fbf63b 的生产删除与 Inspector 坏节点 admission oracle，以及当前累计 `tests/test_plugin_external_loader.py` 已由主审与独立只读 review 静态接受，T-7912fd 原 R1 缺口不作为已通过稿；T-131de5 原 draft 不追认为通过稿，后续修正另行记录，未运行行为测试。Dashboard 的 UI 自持 scope、`HOST_INFO.validation`、Core/routes 启动静态结论仍分别记录，不把源码通过写成行为验收。持久化 delta 为 0，不改 Message、cursor、receipt、auth、schema、descriptor、`root_ref`、metadata、credential 或正式 durable data。

- 状态：Dashboard R3 生产与 T98 A1-A3 测试源静态通过；Mobile T03a-UI R2 已由主审与独立只读复核静态通过，测试未运行。Mobile 的 owner 链为：Root 持有 executor → UI registration lookup → exact contributor permit → target-scope `available` → capture/child → physical thread → release → registration Effect；catalog/asset/query 自持真实 Root/Context scope。catalog 仍只投影 ACTIVE，asset/query 的 false 不读资产或提交 handler，callback 异常不吞；同 Task retained permit 在 UNLOADING 完成，新 Task 拒绝；`aclose` 取消后仍等待 worker/shutdown，若 shutdown 随后失败则同时保留取消与原始失败，Root cleanup 可重试。T-a2817b 的 Core T02-N1 整批 production/test review 未通过；T-a510da 的 P1/P2 生产修复与 T-518ba4 R2 四项测试源已由主审与独立只读 review 静态接受，测试未运行；T-2f3d82 的 N2 production hook 与 T-cbecef 的六项测试源已由主审及独立只读 review 静态接受，测试未运行；T-e721a4 的 production inspect 已由主审与独立只读 review 静态接受；T-fbf63b 的 Inspector 坏节点 admission oracle 与当前累计 `tests/test_plugin_external_loader.py` 已由主审与独立只读 review 静态接受，T-7912fd 原 R1 缺口不作为已通过稿；T-131de5 原 draft 不追认为通过稿，后续修正另行记录，未运行行为测试：它不改变上述 Mobile owner 链，也不改变 Message、cursor、receipt、auth、schema、descriptor、`root_ref`、metadata、credential 或正式 durable data。旧 seal、snapshot lease、`current_snapshot`、`source_revision` 和同步/异步双模式已删除；wire 与持久化合同不变。

```text
UI lookup scope
    └─ target-scope available → capture/child enter
          └─ owner UNLOADING: retained Task completes; new Task rejects
                ├─ physical settlement → permit release → wire close
                └─ Fiber drain → Effect cleanup → Root retry on failure
```
- 日期：2026-09-21
- 关联条款：PLG-001～PLG-018、RUN-007、RUN-009、RUN-016、CTRL-003、STA-001、CAP-002、ERR-001
- 取代：[0071](0071-plugin-composition-and-whole-runtime-updates.md) 中“完整重建 Root、发布后冻结服务绑定、stable/latest 候选晋升、失败自动回旧组合”的部分；[0026](0026-plugin-rollout-is-owned-by-the-parent-turn.md) 的父 Turn 终点晋升授权、[0036](0036-plugin-composition-keeps-promotion-owner.md) 的晋升 owner、[0056](0056-plugin-update-crashes-return-to-stable.md) 的崩溃回旧 stable、[0038](0038-operator-trust-can-publish-offline-plugin-batches.md) 的离线批次按旧提交协议发布的部分
- 保留：[0070](0070-plugins-own-persisted-data.md) 的插件数据归属、PLG-006 的清理责任、普通 programmatic 调用能力、0071 中“底座只解释组合、插件解释业务”的分工原则
- 完整设计与任务拆分：[Issue 750：单图插件系统与局部换代任务拆分](../design/issue-750-plugin-publication-simplification.md)

T-fbf63b 对 T06 的静态对账已删除 EventBus 的 RuntimeSnapshot lease/store 绑定、继承/释放和旧 admission 等待层，并删除 Manager/ValidationHost 的无读者绑定调用；generic queue、dispatcher、handler Task、observer 错误/取消隔离、`drain`/`join` 与 `aclose` 仍由 EventBus 负责。T-fbf63b 的生产删除与 Inspector 坏节点 admission oracle，以及当前累计 `tests/test_plugin_external_loader.py` 已由主审与独立只读静态接受；T-7912fd 原 R1 缺口不作为已通过稿，行为测试未运行。

```text
Core/ValidationHost → EventBus queue/dispatcher → generic handler Task → drain/join
Plugin Context → Root EventRegistry → Fiber Effect（独立，不桥接）
```

这不是全量删除 RuntimeSnapshot：`snapshot.py` 的其他消费者、历史 journal/selection/archive、candidate 清理、offline/trusted watcher、scope 外 provider/RPC 仍是 T06/T07 WIP；Message/schema、descriptor、`root_ref`、metadata、credential 与正式 durable data 不变，持久化 delta 为 0。

T-71cb50 对账 T05-D/T06：删除无生产消费者的 `bootstrap/message_display.py:RuntimeMessageDisplay` 与唯一旧 snapshot generation 测试；现行展示只保留 `app_server.py:message_display` 和 Manager `core.message_display.v1` 两个 live-Root consumer，沿用 `MessagePage → liveRoot exact provider Context → 本页 local scopes → message_rows → release`。EventBus R1 两项测试源尾修、RuntimeMessageDisplay 删除与 T-fbf63b 的生产删除/Inspector oracle 已由主审与独立只读静态接受；行为测试未运行。

T-57630a（T06-Host）production 三文件与旧 runner 删除已由主审及独立 review 静态接受并冻结；原整批不接受，
原因是 `tests/test_runtime_smoke.py` 相对备份只删了旧 runner stub，没有交付 D1-D4 的 host 测试源码，不能写成
“只差运行”。T-b0292d R1 只补测试源与本段对账，主审不接受：A1 仅以外层 Task 未完成判断，可能落入没有
primary 时的 shutdown 等待窗口；B 的 `main.serve` 外层 Task 不持有内部 `runtime.run()`，不能证明精确 App
owner 已回收；C 未在取消/等待 runner 前等待 `to_thread(readline)` 的物理线程结算。T-74fbda R2 的 A1/A2/A3、
真实 fixture 与直接 `AppRuntime.run` 的 B error/cancel 已由主审与独立 review 静态接受，未运行；R2 的 C 仍有
finally 尾项：物理 `stdin.finished.wait` 超时或断言失败会跳过尚未 retrieve 的 exact runner。T-aba7dc R3 已修订
`test_real_stdio_entry_waits_for_physical_input_and_cleans_resources` 的 finally，并由主审与独立 review 静态接受，
未运行：释放物理输入后，把 finished 等待放进 try，把 runner retrieve/join 放进独立 finally；尚未开始 read 时跳过
Event 等待。T-d8e47c 的 production/test review 未通过，缺口精确为：A1 同拍 caller cancel 与 child 完成时错误判断
取消来源；A2 `shield` 已完成 fast path 后再次 `result()` 可能消耗取消异常及其 cause；A3 runtime/stop/restart/settings
后发生错误可能被 first-error 规则静默；B1 cleanup Fiber 的 apply 永不返回；B2 第二次 caller cancel 未证明 runtime
只收到一次取消请求。T-f9462b 的 A1/A2 production 与 B1/B2 测试源、T-a3c9a2 的 production A3 与真实 ready/FD 边界
已由主审及独立 review 静态接受，未运行；但 R2 整批仍未通过：same-turn callback 调度后的 `serving.done()` 断言、
原 runtime `CancelledError` 身份和两条主路径的一次性 retrieve/finally 尚未闭合。T-8e1435 的三项目标与 T-fd28c1
的一行 CLI R4 修正已由主审与独立 review 静态接受，未运行；此前 restart transport_failure 的 cause identity 已恢复为
`assert caught.value.__cause__ is runtime_cancel_error`，不把旧 R2 恢复点当作本轮编辑前证据。本轮 T-13021b production
删除唯一 live cold-start `receipt.ready` 全局中止已静态接受；其初稿测试的 pre-bad 基线、dispose 窗口和
current-catalog 证据缺口不接受。T-0d18d9 R1 补齐主要 cold-start 结构，但仍因外层
`RUNTIME_CATALOG` 缺直接 import、以及未在 bad FAILED/downstream PENDING 存续窗口执行真实 peer scope 而整批不接受。
既有 T-4f17d6 T06-Host Cold-Start 累计 production/test 结论已由主审与独立 review 静态接受并冻结，未运行；它不是本轮
Selected-Load 接手点。该累计材料保留真实 `RUNTIME_CATALOG`、peer scope/permit、bad/downstream 状态、host 存续、
dispose 后 peer 身份/Effect/lifecycle/cleanup、SIGTERM/cleanup gate/runtime done callback、取消与 restart 的错误链，
以及真实 `StdioAppServer.run`、`ConnectionRouter`、ControlService/Bus/Core/Manager、EOF/OSError、HTTP/lock cleanup；
partial startup/task factory failure 仍是前一合同明确未扩展的边界。

T-1d7e36 对 T06 的 A 曾写入，但 production/test review 未通过，不能视为“只差运行”：fatal pre-Fiber owner
没有覆盖 partial archive/importer，shared retention 可能缩小清理集合，catalog 缺少当前 archive/state，且真实
source/oracle 测试不足。T-5c7b99 的 production `manager.py`/runtime catalog P1-P4 已由主审与独立生产 review
静态接受并冻结；其 R1 测试源不接受。T-11ad0b R2 仍不接受，尾项是 A3 peer mount 时机与真实 inner
operation/cancellation settlement、A2 gate finally、B3 runner error/cancel retrieval 和 C 的直接 `sys`/cleanup
recovery。T-b6731d 是 R3 测试/文档返修，已由主审与独立只读 review 静态接受主要链，未运行；整批仍只因下面两项收尾未过。
T-1174f2 是 R4 测试/文档返修；Selected-Load A R4 已由主审与独立只读 review 静态接受，行为在当时未运行。A3 释放 cleanup gate 后在 timeout 外等待真实 `load_task` 物理收尾并在局部 finally 记录结果消费；B 用 `asyncio.wait` 分离精确 runner 的物理完成与 `.result()` 原始终态，纯取消仅在本 finally 主动请求且没有真实 cleanup cause 时忽略。它不改变 production，也不把测试源写成行为验收。


T-0b0e05 承接 T-d338ca 的 B R1：R1 的 P1-P4 已由主审与独立只读 review 静态接受；本 R2 只修固定 7 路径内的 P/T/D 尾项，仍待主审与独立只读 review，行为测试未运行：(1) `source_resolver` 保留 strict consumer，新增
窄 typed content scan；`_load_one` 只把源码 Syntax/Unicode compile 内容错误分类，IO、权限、path/cache/pointer/identity/archive
错误仍 fail-loud；(2) Manager 单一持有 `_source_failures`，`plugin_status` 顶层投影，metadata/revision scan 和重启 archive load 不清除，
同一 source 完整 prepare+compile replacement 才清除；(3) first-null 沿既有 `_load_live_initial` 恰好一次 CAS，健康 subset 或 `()`，
提交后 A 的 import/Fiber failure 不改 selection；(4) watcher/SIGHUP 只 reconcile exact selection，修复未选 source 不自动装，source 暂失
不 deactivate；新增选择成员须显式 install，移除成员须显式 disable/uninstall，已选健康 source 更新仍走受控 prepare/replacement/CAS。无 source-error durable owner/schema/writer；正常 archive 准备可能留下不可变产物，正式 durable data delta 仍为 0。archive-only restart 没有真实 source provenance 时只报告 source_unavailable/来源未知，不猜路径；R1 的 P1-P4 已静态接受，但本 R2 的 P/T/D 尾项导致整批仍不接受，待 review、未运行行为测试。核心清理路径为：

```text
selected archive
  └─ live pre-Fiber wrapper → load_error/state=failed
       ├─ Scope/owner cleanup → module removal
       └─ retained current+draining on cleanup failure → explicit retry
```

T-a4905e 为 first-null B R3：R2 production 与 install/hot-reload 测试源按既有静态接受事实冻结；本轮只修 real-app smoke 的 FiberHandle/raw Fiber oracle、timeout 与精确 runner 结果消费顺序，以及带 shutdown cause 的 `CancelledError` 传播。scan `8 passed`、两项 smoke `2 passed`；loader 历史在第 7 项停止的失败后来定位为生产 `_check_existing_schema` 未关闭 `mode=ro` SQLite 连接，不归因于冻结测试。

T-dfc862 为 first-null B R4：唯一生产改动是 `ReloadJournal._check_existing_schema` 对自建只读连接执行显式 `try/finally close`，保持 schema/index 只读检查、错误和 cause 不变。新增回归在真实临时 journal 上覆盖 valid-schema 与 missing-required-index；旧实现两个参数均以“捕获连接仍可执行” red，修复后两个参数均 green；随后完整 loader 10 个 nodeid `10 passed`。测试实际执行应用 import 与隔离 SQLite/selection 写入，但不执行 migration；正式 durable data delta=0。R4 只影响上述五个允许路径，AST/内存 compile、`git diff --check` 与 147 项冻结 hash 通过；B R4 已由主审及独立 review 接受，不能据此宣称 Issue 750、Gate 或 final enable 完成。

T-855f8f 是 Selected-Load A 的真实隔离行为验证：fixture、view/host 与 core 其它 4 cases 已接受；原 7 cases 中 `test_live_cold_cancel_cleans_imported_generation_owner` 因冻结测试 oracle 把归档 `code_dir` 末级目录 `tree` 当作 plugin ID 而失败，不能追认原整批全通过，证据保留在 `/tmp/i750-av.SdkiWD/`。

T-648c5e 是该剩余项的 exact archive identity oracle 定点修复与两项行为验证：identity 与 public-install 各 1 passed，且 public-install 的 accepted→selected failed B→显式 fresh retry 事实保留；但 identity 版本在 `load_all()` 原始 `CancelledError` 抛出后没有再次读取 selection，因此“selection 未回滚”只是未闭合 claim，不作为已证明事实。artifact 为 `/tmp/i750-a-id.2CDBQG/`。

T-fb9a7f 补上唯一允许的 `assert manager._selection.read() == selected`，位置严格在 `load_all()` 取消之后、`terminate_all()` 之前；selection identity 1/1 通过。合同指定的 Core T02 五组也全部通过：baseline 4/4、admission 3/3、registration 12/12（9 个 nodeid，含参数化 3+2）、late_provide 6/6，共 26/26；artifact 根目录为 `/tmp/i750-core.YE9wsb/`。测试使用隔离临时 plugin/home/workspace、真实应用 import 与局部生命周期/SQLite/archive/selection 夹具，未执行 migration、正式安装/部署、网络、外部 MCP/server、业务服务、Gate 或 CI；正式 durable data delta=0。本结果待独立 review，不构成 Issue750、Gate 或 final enable 完成。


App 只有在真实 HostBridge 等 runtime task 存在时创建 primary；无 primary 时继续监督真实 Dashboard/PluginWatcher，
空监督集合明确失败。保留 snapshot start/stop/store、candidate/history/cleanup 与 operation/CAS/accepted/physical
owner/`terminate_all` 责任，因为它们仍有真实消费者；不把旧 snapshot 生命周期全量删除写成本批结果。Host
readiness/宿主存在与所有插件健康分开：

```text
core.start/load_all → 同一live Root → 每个Fiber STARTING/STARTED/health
AppRuntime.run Task（精确 owner）→ 实际Dashboard/PluginWatcher/[HostBridge] → host shutdown
外层 stop/cancel/restart
 └─ main.serve exact Tasks
      ├─ runtime → AppRuntime physical cleanup → Core/Root resources
      └─ stop/restart/settings → 全部结算 → 原取消/实际错误链
stdio → StdioAppServer/ConnectionRouter → 物理 readline 线程 → EOF/error → 原Core.stop
```

Fiber 的 STARTING/STARTED/required health/ACTIVE 仍由该 activation owner 解释，host 只监督真实宿主入口。新增
task/gate/scope 的 finally、join 和物理清理责任均留在测试源；本轮 T-4f17d6 延续 T-fd28c1 的边界，不以“finally 保留”替代上述场景，也不以
fake Core/Root、sleep、`return_exceptions` 或立即 `create_task` 后的断言替代真实 host boundary 证据。删除冲突的旧
candidate runner 和 restart runner 依赖。T-b0ab9e Models 累计静态接受为已知事实，不追认原失败稿。
T-13021b 历史上移除 live initial 的全局 `receipt.ready` 中止；本轮 Selected-Load R4 只改测试与状态文档。其它 scope 外
provider/RPC/consumer、UI 外 `current_snapshot`/fence、
offline/trusted watcher、candidate/freeze、Commands live binding/provenance、T06/T07、行为验收和 final enable
仍为 WIP。未改 Message、schema、cursor、receipt、descriptor、root_ref、metadata、credential、archive/history/
selection/journal 或正式 durable data；只做 AST/内存 compile、diff/空白、符号/hash 核验，未运行测试、Gate、CI、
应用 import、插件安装、业务进程、正式 workspace 或部署。

T-4be90d 整体 review 未通过；T-131de5 继续收口 MCP 的旧 fixture 依赖，不扩大生产迁移范围：普通调用沿
`plugin artifact + ExecutionAccess → contribution Context → MCP Session Effect → host/client/process → disconnect confirmation → Effect release`
进入真实 live Root；binding 由真实 Manager `BINDINGS` 与 SERVICE provider Context 建立；候选环境测试使用独立 Root、固定 `CodeOwner` 和 `ExecutionAccess(candidate=True)`，不调用 Manager candidate publication。T-131de5 原稿不追认为通过稿；后续修正另行记录，未运行测试、Gate、CI 或行为验收。MCP 生产 owner、wire、持久化、descriptor、`root_ref`、metadata、credential 和正式 durable data 不变，production delta 为 0；其它 provider/RPC、scope 外 `current_snapshot`、offline/trusted watcher、candidate/freeze、T06/T07 与 final enable 仍为 WIP。

T-471123 的三个 Commands production 文件已由主审与独立只读 review 静态接受并冻结；T-e70bc3 R2 测试源与状态对账先前已静态接受，未运行测试，不改 production。T-69d63e R1 的 Message workspace 初始化与基础生命周期修订已静态接受，但 R1 整体仍有取消分支无有界等待、排空原 Task nested 两 owner scope/冻结 provider 读取、recover 错误后健康调用等缺口，不追认整批通过。

T-e87866 首次行为验证按三个独立组运行：provider 13/13 passed（6 个精确函数展开）、message 9/9 passed（6 个精确函数展开）、reply_entry 0/1，合计 22 passed、1 failed，无 skip/error/timeout；artifact 为 `/tmp/i750-cmd.l6NEiF/`。唯一失败真实进入 Reply→Source→Conversation：`plugins/reply/follow.py:52-53` 以 Reply scope 调用 source，`plugins/conversation/plugin.py:101-115` 用 Conversation `ctx` 创建 `MESSAGE_WRITERS`/`TASKS`，在 `agent/plugin_composition/context.py:180-196` 抛 `OWNER_CALL_CONTEXT`；Output timeout 是后果，teardown 的 TaskGroup/`terminate_all()` 只重现同一根因。该边界待 review，不把 provider/message 绿扩大为 Reply/Subagent 全链路通过。

T-e87866 当批不改 production、测试源码或 formal durable data；仅使用隔离临时 plugin/home/workspace 与 Message/intent/receipt/SQLite/archive/selection fixture。下一批最小写入范围应只处理 source owner 创建 Task、同一 Task 的 Conversation owner scope 与 Reply program permit/`task.join()` 边界；Source registration wake、Subagent 第三 owner、其它 provider/RPC/consumer、`current_snapshot`/fence、offline/candidate/freeze、T06/T07、全量回归/Gate/CI、正式运行和 final enable 仍为 WIP。

T-c6ec4e 已完成这条最小 Reply/Source wiring，但待独立只读 review。`plugins/reply/follow.py` 不再把 Reply scope 当作 Source owner：它预捕获两个真实 Context 的 permit，Source scope 内执行 `open/start`，实际 Source Task 自动捕获 Source call，wrapper 在该 Task 内进入 Reply capture；Conversation command closure 再在自己的 `ctx.runtime_scope()` 中运行 `run_commands`。两个 raw monitor 只等待 admission 关闭，不授权、不 capture、不登记第二张 owner 表；取消后先停 monitor、精确取消一次 active Task、物理 join，再释放 permit。`None`、existing Task、wrapper 未进入、首指令取消和 monitor 建立失败均有本地结算路径；错误只在两个 scope-entry 边界局部吞 `OWNER_UNAVAILABLE`/`STALE_ACTIVATION`，Source open/start/body 的 CompositionError 仍可见。

```text
Reply drive accepts two Contexts → Source Task auto Source call → wrapper Reply call
    → Conversation command own scope → actual Commands provider/contributor
    → original receipt/Message
two admission events → drive cancel → exact Task physical join
    → permits returned → Core STOPPING/Effect
```

T-c6ec4e 三组隔离行为证据为：`tests/test_message_commands.py::test_default_reply_short_circuits_command_before_model_or_tool` 1/1；`tests/test_reply_follow.py` 三个 live Root/Manager nodeid 3/3；`tests/test_message_commands.py` 全模块 10/10。测试动态 Python fixture 写盘前 AST parse/in-memory compile；正式 Message、schema、intent/receipt、descriptor/root_ref/metadata/credential、selection/journal/history/archive 均无 delta，临时 SQLite/archive 不构成正式持久化。此为概念 Gate 证据，不是独立 review、Issue750 system Gate、CI、部署、正式运行或 final enable；Source registration wake、Subagent report、其它 provider/RPC/consumer、legacy snapshot/fence、offline/candidate/freeze、T06/T07 与累计 Gate 仍为 WIP。


```text
贡献Context → registration Effect → _registrations/_names
consumer → 当次freeze view → Commands scope → exact handler scope
         → handler/recover/result → release
```

R2 补上取消等待 timeout、原 handler 在排空窗口的 nested 两 owner scope 与 contributor provider
identity 读取、admission barrier 后的未 enter capture 结算、`recover_error` 原始错误传播和错误后
独立 health command。登记 Effect 移除后，新的 freeze view 不再包含名字；已有 immutable view 仍保留
原定义，只由真实 Context/owner admission 失效，不改写成 unknown/default。健康上游、兄弟和无关 peer
继续可用；坏节点只影响自己与实际硬下游。

Message 的既有 receipt/不重跑测试只证明已加载归档源码不受 checkout 源文件改写影响，不证明 live binding 换代后的精确 provenance/handler 语义；本批的 provider/message 绿证据也不覆盖 Reply→Source→Conversation owner 迁移。
T-e87866 仅运行上述 23 cases；测试实际写入隔离临时 Message/intent/receipt/SQLite/archive/selection fixture，正式 durable data 增改减为 0，不改 Message、history、archive、descriptor、root_ref、metadata、credential、wire 或 schema。未运行整模块、全量回归、Gate、CI、正式 workspace、插件安装、业务进程、部署或正式运行；其它 provider/RPC、scope 外 `current_snapshot`/fence、offline/trusted watcher、candidate/freeze、Source wake、Subagent、T06/T07、行为验收与 final enable 仍为 WIP。


## 背景

[Issue #750](https://github.com/kachofugetsu09/akashic-agent/issues/750) 报告发布长时间等待、整组装载与无关插件启动链拖慢可用性。核查确认根因不是 owner 分工过多，而是把一次局部代码替换提升成了全局运行事务：Turn/job 持有整图 snapshot lease、更新重建完整 Root、候选与正式两套运行实例、失败时自动回旧组合。本次简化针对这个目的，不是按名字删除“snapshot/generation”词汇，也不恢复另一套生命周期真源或通用业务恢复系统。

## 决定

1. 运行时只有一张 live CompositionRoot。安装直接请求应用；普通插件新增、更新、禁用、卸载不重建完整 Root、不以 Gateway 重启兜底、不为验证另外装配完整候选图。
2. 局部变更只触及变更节点、它拥有的子 Fiber 和实际硬依赖消费者。共享上游 provider 不随贡献插件重启；未受影响模块、Fiber、服务、任务与进程身份及生命周期不动。
3. 构建、编译与可提前完成的校验失败不改变已选输入和当前图；实际 import、配置、依赖、apply 和必要资源完成后才视为 active。业务逻辑错误由正常测试与修复重装处理。
4. 保留 `apply(ctx)`、`ctx.config`、`inject`、`provide`、`effect`；主要改系统与公共 provider，不要求普通业务插件实现 prepare/commit/rollback。
5. 去掉 stable/latest 双运行图、候选试用/晋升/撤销和失败自动回旧版本；不复制完整 HMR 的源码 watcher 与缓存恢复。局部生命周期参考 DSH 的 Loader/Fiber/Effect，对照见设计文档第 7 节。
6. 安装输入只有一个持久选择 owner，沿用 PluginSelection 的原子写入与并发基线检查；选择与运行状态明确分离。先原子选择 B，再局部应用；B 失败只使实际依赖分支不可用，无关分支继续，不自动启动 A。
7. 不再以整 Turn 全图 lease 阻塞局部替换；保护的是实际调用、实际资源和本次 activation 的实际依赖。旧引用不能静默转成新实现或重放效果。
8. 内部 Shell 与外部 CLI 使用同一安装语义；宿主持有应用任务，安装发起调用先返回 accepted，accepted 不等于 active。普通业务测试不再是系统晋升协议。
   caller cancellation 只结束 caller 的等待；宿主仍持有物理 installer、清理和挂载 owner。宿主 deadline 还会撤销迟到的提交许可，但不会把已进入的物理 installer 或 cleanup 写成取消或成功；其 busy 状态必须继续可观察并可恢复。
9. 接受局部短时停顿与同进程隔离边界；保留清理失败 owner、历史事实、权限边界与真实退出责任，不把合理 owner 压成一个巨型 Manager。
10. 不修改 RUN-010 动态配置与 RUN-015 Core/Bridge 独立部署恢复策略；不引入新版 Supervisor 发布协议和启动令牌制度。

公开卸载已接入同一 Manager owner：disable → selection CAS remove → accepted → 目标 Fiber/硬消费者及 draining owner 排空 → 现有 cache/manifest finalizer；finalizer 通过 `complete_critical(asyncio.to_thread(...))` 持有真实物理删除线程，deadline/caller cancellation 不提前宣称 removed。状态投影同时读取 manifest、selection、当前 generation 平面字段、draining Fiber、operation task/accepted 和 cache 实际存在性；目标插件只从成功的 `UpdateStatus` 或卸载 dict accepted 结果推导。已选但无 Fiber、FAILED、active+draining、CAS 冲突、cleanup/finalizer 失败都保留真实错误和可重试 owner，不新增第二套 receipt/schema，不以目录消失或 HTTP 200 宣称完成。普通卸载保留 plugin-data、Message、附件、历史 binding、归档、root_ref、metadata、Delivery/journal 历史与凭据；T05-B/T05-C 已有源码静态材料，测试和迁移演练仍未执行。

本批 T04-A-R3/B2 的静态失败路径证据是：局部 reconcile 先提交 selection，再由旧 active/draining generation owner 负责物理排空；B 的 import/apply/start/readiness 失败进入同一 dispose 结算，失败 owner 留在 draining，显式 retry 按 selection 的固定 archive 重建新 Scope。取消在 operation 被撤销后仍等待实际 Fiber 结算，吞取消的 apply 不能迟到发布；就绪检查使用 Root 中仍登记的实际 Fiber、当前 provider 身份、声明依赖与健康 owner；Core provider 只授权依赖，不加入插件归档组件。live RuntimeCatalog 只读当前 Root，不复用 snapshot lease 或冻结拓扑；同名 Fiber 的 Incident 以不可复用 Fiber ID 过滤，reader 接收显式 caller Context，不猜 ambient scope。Models M1 production static review 已通过；T-d69c83 的 Models M2 production/既有测试修正已通过主审与独立只读静态复核；T-c46d11 整批 review 未通过，T-59017c 的 Models oracle、Shell single-owner entry 与 Tools drain 已写入但未执行、待主审复核。T-60b287 的两条 Shell 测试链只完成静态源修正，T-eab8b1 的旧 Bus 出站删除与 durable 测试迁移只完成静态源修改，T-c76e09 已完成 Channel provider 的 HostInfo/局部 Fiber 合同静态迁移；所有这些仍待主审复核，运行验收仍未执行。Models M2 保留 ModelsStore/CAS、credentials、descriptor/schema、Message、model_calls、continuation 与 embedding identity，并把 enabled connection 的 `open+close` 从整 Root seal 预检延后到真实首用或显式 probe。

本批 Shell/Tools 调用链固定为：

```text
Reply Task → TOOL_CLEANUP(reader, source, from_seq, task, drain)
           → Shell owner TaskAdmission
           → Tools drain 的旧 Task.join + BINDINGS.describe
           → ShellOwners.release_tool → PluginProcesses
```

清理失败保留真实 process owner 并记录 Shell incident；普通 cancel/pause 仍等待实际
cleanup 完成，只有明确 abandon 才放弃 caller 的等待，已接纳的 Shell cleanup Task 继续排空。Tools 不在 Shell caller scope 重开 `TASKS`，也不重开
历史 Tools binding；schema、Message、descriptor、`root_ref`、metadata 和 durable data
不变。DSH `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8` 仍只作 Fiber/Effect 局部生命周期
对照，不是本批生产证据。

T-84dd9a 已补 Models full descriptor equality、Tools unload/closed-admission oracle，以及
真实 local Root 的 Shell C1–C4 测试源；T-5d8705 已修正 Tools/Shell 的卸载等待顺序、公共入口
拒绝、消息 ID 和 C2/C4 Task 结算；T-60b287 再把 C2/C4 的公共 cleanup、真实 Task 取消与同源
新 owner 链补齐。以上仍待主审复核，测试未执行，只能算静态源审查材料，不能写成行为验收通过。

T-eab8b1 删除了 `bus/queue.py` 中无生产消费者的旧 Channel 出站 queue、dispatcher、receipt
状态与拒绝壳，撤掉 `bootstrap/tools.py` 的旧 dispatcher binding 和 `bootstrap/app.py` 的空
dispatcher task；`_closed`、入站队列、ChatLane 的入站计数、durable handoff、SessionAdmissions
和关闭失败责任保留。旧出站测试已删除，durable delivery 取消与 SIGKILL fixture 改由现有
`PluginDurableDeliveries` sender callback 证明；T-c76e09 已删除
`tests/test_runtime_smoke.py:463` 的旧 `dispatch_outbound` monkeypatch，并将 crash fixture
改为 provider edge `fsync` 后 SIGKILL、确认 provider-calls 尚不存在。该证据仍只完成源码静态迁移，未运行。

T-9f83dd 修复后的局部 Channel 合同如下：HostInfo 只读提供 boot/validation；贡献 Fiber 的
Context 负责本次 activation 的 scope；注册 Effect 是 adapter 的唯一关闭 owner；binding
lease 只负责 exact transport claim。ready 必须先 closed，ACTIVE/required health 后才在
贡献 Context 内 open；stop 先关新接纳、排空真实调用并等待 adapter stop，成功后再结算 listener
与 durable reservation，失败则保留原 owner 供显式重试；已持有 request、attachment lease 和
durable stop/finally 不申请新 activation scope。Channel 不再持有 RuntimeSnapshotLease，也不由
Manager 广播全图 open/recovery。Bootstrap 和隔离 validation Root 只通过 live Root/隔离 Root 的
公开 `CHANNELS.recover_inbound` 恢复 pending handoff。crash oracle 在 `provider_started` edge
`fsync` 后 SIGKILL，并断言 provider-calls 尚不存在；这些仍是静态源证据，未运行测试。

T-b980b1 的入口收口保持这一边界：`raw ingress → Channel contributor Context → Sources
Context.runtime_scope() → selected Source Context.runtime_scope() → conversation.accept`。
Sources 不重选默认来源，不建立第二 registry，也不替 Core 借用 contributor 的权限；旧 callback
在其 permit 释放前继续完成，新入口在 contributor UNLOADING 时明确拒绝。该批只完成 AST/内存
compile、diff/hash 与 `git diff --check`，未执行测试、Gate、CI、应用 import、插件安装、业务
进程、正式 workspace 或部署。

## 理由

0071 的整图换代允许暂停接纳并完整重建，适合低频大变更，但让 Fitbit 级更新支付全图冻结、双实例与恢复成本。改为按实际依赖局部启停后，保护对象从“整张图”收窄到“实际调用与实际依赖”，与 0070 的数据归属、PLG-006 的清理责任兼容：谁拥有资源仍由谁关闭，谁拥有数据仍由谁解释。不引入 prepare/commit/rollback 三阶段接口，是因为固定制品、原子选择和局部 Fiber 生命周期已能覆盖已确认需求；为 1% 假设场景新增通用恢复协议不符合本次已批准取舍。

## 影响

- projectneed 第 10 节按本决定改写，逐条语义变化体现在各条款新表述中；RUN-007、RUN-009、RUN-016、CTRL-003 中与整图 snapshot/候选晋升耦合的表述同步修订。
- 0071、0026、0036、0056、0038 保留历史正文，只标明被取代范围。
- 旧 Devin v1/v2 评审针对整图发布方案，对真实代码、owner 与资源边界的核查仍有参考价值，但不构成本版方案的评审通过证据。
- 实施按设计文档 T02～T07 执行；本记录本身不授权 commit、部署或解除测试/Gate 限制。

## 验收

最终验收标准在设计文档第 6 节及 §6.4 验收映射，覆盖最小拓扑（变更节点 + 硬消费者 + 稳定宿主的可选子 Fiber + 无关长任务）、Root/boot/模块/Fiber/服务/任务/资源身份不变、无关生命周期计数不增加、编译失败不动图、启动失败不回退、清理失败保留 owner、内部 Shell 不自等。当前状态：验收设计已完成，尚未运行任何测试。
