# 插件 v3 Mobile UI/query capability 任务合同

- 状态：implemented / independent review passed
- 日期：2026-08-16
- 实现基线：`19f2cca2`
- 实现提交：`2c6e4f71`、`b173f551`
- 关联条款：PLG-001～PLG-004、PLG-008、PLG-011、PLG-014、MOB-006、TST-001～TST-008
- 上游：[v3 production readiness checklist](plugin-v3-production-readiness-checklist.md)、[v3 package contributions](plugin-v3-package-contributions-task-contract.md)
- 参考但不直接合并：旧 capability lane `6214af1c`、`c828b31d`

## 1. 目标与边界

本 PR 让 v3 Fiber 通过 `core.ui_slots` 登记 Mobile UI 静态资产、动态可用性与只读 query，
同时保留现有移动协议、有界 worker、timeout 和 snapshot lease。插件继续拥有业务 projection；
Core 只拥有路径/大小校验、stable publication、query admission 与 generation 生命周期。

```text
v3 Fiber ── Effect register ──► PluginUiSlots in candidate Root
                                      │ freeze
                                      ▼
candidate RuntimeSnapshot.mobile_ui_registry
                                      │ promotion / formal rebuild
                                      ▼
stable registry ── lease ──► catalog / asset / bounded query
```

`semantic_delta: compatible`。本任务不迁移具体插件，不改变 Android/WebUI 协议，不写 Session、
workspace 或 plugin-data，不把 Dashboard backend 重做成动态 slot，也不把旧 `mobile_ui()` 方法复制为
新的 namespace 固定声明。

## 2. Owner 与公开合同

1. `UI_SLOTS = ServiceKey("core.ui_slots")` 由每个 CompositionRoot 提供。插件调用
   `PluginUiSlots.register_mobile(ctx, MobileUiDefinition(...), query=..., available=...)`；登记是调用
   Fiber 的 Effect，duplicate、freeze 后登记、非法 handler 或非法资产在 candidate compile 前 fail-loud。
2. 模块和 stylesheet 必须是调用 Fiber `ctx.runtime.plugin_dir` 内的 `.js`/`.css` 文件；Core 固化内容、
   SHA-256、字节数、navigation 与已知 slots，总大小继续限制为 240 KiB。query/available 必须是同步 callable。
3. `MobileUiRegistry` 是 snapshot-owned immutable value，descriptor 覆盖 owner、asset hashes/bytes、
   navigation 与 slots，并进入 snapshot identity。handler 只存在 exact snapshot binding，不进入内容 hash；
   generation source revision 继续绑定其代码身份。
   每个 live handler 还绑定 Fiber activation token；Fiber dispose/restart 后旧 binding 立即不可解析，cleanup
   不能重新打开已经 freeze 的 registry。
4. candidate registry 只存在 candidate snapshot，不得写回 stable `PluginGeneration.contributions`。这是对旧
   capability lane 的明确修订：candidate clone handler 不能替换 stable generation 的 query callback。
   formal rebuild 必须重新登记正式 Root handler，payload replacement 复制正式 registry。
5. `PluginMobileUiProvider` 的 catalog/asset 只读 current stable snapshot；query 取得 exact stable lease 后，
   从同一 snapshot registry 解析 handler。timeout 或 caller cancellation 后 lease 持有到同步 worker 真正退出。
   worker/queue/结果大小和 `MobileUiRpcInvalidRequest` 的现有错误映射保持不变。
   query 结果由 Core 做递归 JSON 校验：mapping key 必须是字符串，拒绝 cycle、非有限浮点和非 JSON 值；
   校验后才允许进入 Mobile transport。
6. 迁移期 v2 Mobile UI 先在 generation contribution 中冻结 asset/query/available 三元组；provider 不再把任意
   generation instance cast 成 `Plugin`。v2 与 v3 同名 contribution 在 snapshot compile 阶段 fail-loud；
   最后一个 v2 consumer 迁走后删除 legacy 三元组与 Manager 固定收集。

## 3. 状态与安全边界

- 静态 asset 只读取 immutable plugin artifact；本任务没有持久写入。
- query 是受支持的只读 ABI，不是 Python 安全沙箱。同 UID 插件仍可能通过闭包绕过约定；真正恶意隔离需要
  独立进程/权限域，本任务不以 wrapper 伪装实现。
- candidate discard、load failure、Root dispose 与 terminate 必须移除 registration、handler 引用与 Effect；
  stable lease 未归零时旧 handler 继续可用，归零后才释放旧 Root/module。
- v2 query 的业务数据 owner 与 schema 不变；本 PR 不迁移、复制或删除任何 plugin-data。

## 4. 验证与停止条件

- unit：path/symlink/slot/navigation/size、sync handler、duplicate、freeze、Effect cleanup、descriptor identity；
- real Manager mixed v2/v3 stable：相同 catalog/asset/query 协议，v3 provider 不依赖 `Plugin` cast；
- candidate：latest catalog 不公开，candidate handler 不进入 stable generation；discard 零引用，promotion 后正式
  handler 才可见，descriptor drift 阻止 publication；
- provider：stale revision、dynamic unavailable、worker offload、queue bound、timeout/cancel 后 lease drain、
  invalid request、执行异常、非 JSON/超 192 KiB 结果；
- mobile UI、loader、hot reload、snapshot 定向回归，Basedpyright error-level、compileall、`git diff --check`。

任何 candidate asset/handler 提前公开、query 脱离 exact lease、超时提前释放 generation、非法 asset 写入
artifact/workspace、或 v2 provider 行为漂移都停止交付。

实现 head 的独立复核未发现 P0/P1。集成分支已运行 Mobile UI、Manager、lifecycle、loader、kernel 与
hot-reload 累计回归 `372 passed`；相关 Basedpyright 为 `0 errors`，compileall 与 `git diff --check` 通过。
最终 exact plugin lock 由 fleet source/API compatibility 与 Mobile Gate 对账；正式 workspace
替换证据仍由拥有部署输入的发布流程负责，本合同不把定向回归写成生产替换证据。

## 5. 回滚

代码恢复点为 `19f2cca2`。本任务没有正式运行数据变化；回滚只撤销该能力源码、测试和合同。
