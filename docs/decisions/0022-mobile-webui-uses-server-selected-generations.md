# 0022 · 移动 WebUI 使用服务端选择的不可变 generation

- 状态：accepted
- 日期：2026-08-03
- 关联条款：WEBUI-001～WEBUI-006、MOB-001～MOB-004、GOV-005、TST-006～TST-008
- amends：[0018](0018-chat-webui-has-one-source-and-two-adapters.md)
- amended by：[0029](0029-main-gateway-reconciles-mobile-webui-stable.md)
- 设计：[服务端发布的移动 WebUI OTA](../design/server-published-mobile-webui.md)

## 背景

决策 0018 把 `frontend/chat` 固定为唯一源码真源，并要求 Android 把由干净提交生成的 ZIP 放进 APK。这个基线保证离线构建和可复现性，但任何颜色、布局、抽屉 island 或既有桥交互变化都要重新发布 APK，无法满足服务端明确发布界面后让已配对设备收敛的产品目标。

直接让 WebView 打开远程页面会把网络可用性、服务端瞬时状态和页面执行混成同一个启动条件，也会把凭据、任意导航和不完整资源带进 Web 容器。客户端维护一条包含检查、下载、等待页面状态和激活的总状态机，则会让每个网络与生命周期 edge case 增加新的全局跳转。

## 决定

1. APK 内固定 ZIP 继续作为 embedded baseline，而不是移动 WebUI 的唯一交付来源。Android 无网络、未配对、目标不兼容、资源损坏或候选激活失败时始终能使用 baseline。
2. Core 发布者拥有 WebUI generation、manifest、按内容摘要寻址的资源、Stable/Preview 指针和发布 journal。候选资源完整写入并校验后，单一发布事务才改变当前 `ReleaseView`；Runtime 只读已提交状态，不在请求路径构建 WebUI，也不因启动、重启或 watcher 自动发布。
3. Stable 从指定提交和完整构建上下文确定性生成。Preview 可以固化未提交输入及其 provenance；它只有在提交后重建出相同 generation 时才能提升 Stable。清除 Preview、提升和回滚都是显式指针操作，不修改 generation 内容。
4. 已配对客户端通过中立协议读取当前 `ReleaseView`，再经同一服务端身份和短期认证读取 manifest 与静态资源。客户端按 `server identity + generation + manifest digest` 识别目标，不按发布序号、时间或 semver 判断新旧；通知只提示重新解析，不直接授予目标。
5. Android 与未来的 iOS 只保留四个事实：`desired`、`ready(desired)`、`serving` 和 `fallback`；行为收敛为 `Resolve`、`Ensure` 和 `Present`。下载与激活由不同 owner 协调，`Present` 只在 UI session 边界使用本地完整验证的资源。
6. 每个服务端拥有独立的客户端 CAS、generation、marker 和 rejected target。embedded baseline 不参与 OTA GC；清理或重置只能减少可证明未引用的派生 UI 资源，不能触碰业务状态、配对身份或其他服务端缓存。
7. WebUI 只表达产品界面并调用版本化 capability。新增原生能力或不兼容 bridge/snapshot 时仍发布移动二进制；现有 GitHub APK updater 与安装权限保持独立，本决定不提前改造成 Google Play 发行模式。
8. `Resolve/Ensure` 只返回 `Ready`、`RetryAfter`、`WaitFor(trigger)` 或 `RejectTarget`。同 Target 的 `WaitFor(space)` 和永久 reject 都不因前台、重连或重复 hint 自动下载；必须等 Target/兼容指纹变化或名称明确的用户动作。candidate 由 process-scope attempt lease 持有，健康提交前的 Activity 重建不得将它提升为 serving。

## 理由

这个选择保留 0018 的单一源码真源、离线 baseline 和原生状态 owner，同时把“服务端当前选择什么”“本地是否完整拥有”“下一 UI session 展示什么”分给三个明确动作。不可变 generation 和内容摘要复用减少下载量，却不在 serving 目录原位打补丁；显式指针让 Preview、提升和回滚可审计，也避免客户端推测服务端发布历史。

具体采用的成熟不变量及官方来源记录在关联设计的“成熟实践与采用理由”中。外部实践只说明为什么选择 embedded baseline、兼容分组、内容摘要和 session-boundary activation，不覆盖本仓库的数据 owner 与安全合同。

## 状态与减少协议

| 对象 | 正常增加 | 允许原位变化或逻辑失效 | 允许物理减少 | 恢复证据 |
|---|---|---|---|---|
| Core generation 与资源 | 显式 build/import 提交完整不可变对象 | 不原位更新；指针变化只令旧对象不再可达 | 显式 GC 只能删除当前 Stable/Preview 和发布事务均不可达的对象 | 当前 `ReleaseView`、manifest 摘要、资源摘要与发布 journal |
| Stable/Preview 与 journal | 显式 publish/clear/promote/rollback 原子提交；journal 追加 | 指针可以原子替换，Preview 可以显式清空；journal 不改写 | 指针行不以删除资源代替更新；journal retention 另立合同 | 旧或新完整 `ReleaseView`，不能出现半提交选择 |
| 客户端 WebUI 缓存 | `Ensure` 写临时对象，完整校验后提交 verified generation | desired/serving/fallback/attempt marker 按 owner 事务变化 | 安全 GC 或用户明确重置单个服务端 UI 缓存；必须跳过 pinned 对象 | embedded baseline、最近健康 fallback、verified marker 与逐文件摘要 |
| 消息、草稿、outbox、附件、配对与插件状态 | 继续按各自 owner 合同变化 | 本决定不授权任何更新或逻辑失效 | 本决定不授权任何删除 | 各自数据库、文件和 owner 的既有恢复证据 |

## 影响

- Core 增加平台中立发布仓、显式发布命令、已认证发布发现和静态资源数据面；这是 `runtime_patch: required`，因为当前选择、跨设备收敛和回滚属于服务端权威语义。只在客户端实现会复制每台设备的发布真相，不能形成同一服务端的 Stable/Preview。
- Android 增加按服务端隔离的派生缓存、兼容选择、激活与回退；Room、outbox、附件、通知、Keystore 和系统能力继续由现有原生 owner 管理。
- 0018 第 4 项的固定 ZIP 变为 embedded baseline 与离线构建证据；“每次 WebUI 更新都同步移动仓库 ZIP”不再是已配对设备获得 UI-only 更新的唯一方式。
- 协议 source、客户端 snapshot、实际 Core runtime、场景目录、Android commit、APK digest 和真机 run 继续按 0004 固定为不可变验收组合。

## 验收

- Core 显式发布 Preview 或 Stable 后，已配对 Android 在不安装新 APK 的情况下下载并在下一 UI session 使用对应 generation；未发布、离线或失败时继续使用本地 serving 或 baseline。
- 构建、发布、下载、摘要、进程、WebView renderer 和健康握手的失败注入均不产生混合 generation，不让未健康页面取得写动作，也不破坏回退链。
- 同一地址上的不同服务端身份、不同设备、多个服务端缓存和迟到回调不能串改 desired、serving 或资源。
- 更新、回滚、GC 和重置前后的消息、草稿、outbox、附件、阅读位置、配对和插件状态逐项保持其 owner 合同。
- 两仓库 targeted tests、固定协议组合、隔离 Docker Gate 和 run-specific 真机 Gate 均绑定当前源码与真实执行数量；旧报告、`OK (0 tests)` 或只看返回值不能通过。
