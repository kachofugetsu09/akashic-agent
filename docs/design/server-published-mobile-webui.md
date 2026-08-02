# 服务端发布的移动 WebUI OTA 设计

- 状态：proposed；产品边界、三步协调模型和 edge case 语义已经确认，持久化落点、线协议字段、资源上限和实施合同仍待确认
- 日期：2026-08-03
- 关联决策：[0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md)
- 关联条款：WEBUI-001～WEBUI-003、MOB-001、STA-001～STA-003、CAP-001～CAP-002、ERR-001、TST-006～TST-008
- 涉及仓库：`akasic-agent`、`akashic-mobile`

## 1. 文档边界

本文固化已经确认的产品边界、发布模型、Android 三步协调模型、回滚语义和 edge case。它不是实现批准，也不把尚未修改的现行合同描述成已经支持 OTA。

本文使用三种标记：

- **F（fact）**：当前代码或现行合同已经能证明的事实。
- **C（confirmed）**：维护者在本轮设计中已经确认的目标语义。
- **U（unknown）**：仍会影响持久状态、协议兼容或实施范围的待确认项。

在新决策被接受、相关需求条款完成勘误、两仓库实现和 Gate 通过前，当前 APK 内固定 ZIP 仍是唯一已实现路径。

## 2. 用户意图与成功边界

**C：** 产品界面由同一份 WebUI 表达。顶部栏、抽屉、抽屉中的 island、会话列表、消息、输入区、插件面板和产品设置都属于 WebUI。调整样式、布局或只使用已有桥能力的交互时，服务端发布新 WebUI 即可，不要求重新发布 APK。

**C：** Kotlin 和未来的 Swift 只拥有平台可信能力、设备状态和 Web 容器，不重复实现产品界面。新增原生系统能力、桥协议或不兼容快照字段时，仍需要发布 Android/iOS 二进制。

**C：** 移动端不直接打开远程网页。它从已配对服务端发现发布版本，下载并验证不可变资源，写入每个服务端独立的本地缓存，再从本地可信 origin 加载。

**C：** Android 先实现。协议、manifest 和状态语义保持平台中立，未来供 Swift/WKWebView 使用；本轮不创建 iOS 项目。

**C：** Google Play 适配不属于本轮范围。当前 GitHub APK 检查、下载、安装入口和权限保持原状；未来真的准备提交 Google Play 时再单独设计发行边界，本设计不提前删除或隐藏现有能力。

用户能观察到的最终边界是：

| 变化 | 是否只发布 WebUI | 是否发布移动二进制 |
|---|---:|---:|
| 调整颜色、间距、动效、抽屉布局 | 是 | 否 |
| 在抽屉增加只使用现有数据和动作的 island | 是 | 否 |
| 调整消息、输入区、插件面板的组合方式 | 是 | 否 |
| 使用 manifest 已兼容的桥字段或 capability | 是 | 否 |
| 新增相机、通知、文件选择、系统分享等原生能力 | 否 | 是 |
| 改变 bridge/snapshot 协议且旧客户端不能兼容 | 否 | 是 |
| 修改 Kotlin/Swift 生命周期、数据库、网络或安全逻辑 | 否 | 是 |

## 3. 当前事实与合同冲突

- **F：** `frontend/chat` 是桌面和移动 WebUI 的唯一源码真源；`mobile-native.tsx` 是移动入口。
- **F：** `scripts/package-mobile-web.sh` 当前只接受整个 Git tree 干净的源码，生成完整 ZIP、SHA-256 和 schema v1 manifest。
- **F：** Android 当前把固定 ZIP 作为 Gradle 输入，构建时校验并解包进 APK。
- **F：** Android 通过 `WebViewAssetLoader` 从 `https://appassets.androidplatform.net` 加载 APK 资产并阻止外部资源请求。Native→Web 使用指定 origin 的 `postWebMessage`；Web→Native 当前仍通过 `addJavascriptInterface`，在服务端下发 JS 前需要收窄。
- **F：** Room、outbox、附件传输、通知、Keystore、配对、系统 Activity result 和生命周期由 Android 原生层拥有。
- **F：** Core 已有配对后的 WSS 与同源认证 HTTPS 传输模式，可以作为发布发现与资源下载的现有信任基础。
- **F：** [0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md) 和移动仓库的现行合同要求固定 ZIP、干净 commit、构建期解包且无网络 fallback。

因此，服务端发布 OTA 不是对现有实现的自然解释，而是对“移动产物怎样交付”这一部分合同的有意改变。它不改变 WEBUI-001～WEBUI-003 的单一源码真源和状态 owner；后续接受设计时必须用新决策明确勘误 0018，并同步修改移动仓库对应合同。

### 3.1 成熟实践与采用理由

本设计不引入 Expo、Ionic 或 OCI 依赖，只采用它们已经验证的发布不变量：远程选择与本地运行分离、资源不可变、下载与激活分离、原生兼容性先于 WebUI 选择、失败时始终存在内嵌回退。

| 成熟实践 | 官方行为 | 本设计采用的抽象 |
|---|---|---|
| [Expo Updates](https://docs.expo.dev/versions/latest/sdk/updates/) | 默认启动时检查并下载，下一次启动才应用；binary 中保留 embedded update | `Ensure` 可以后台运行，`Present` 只在 UI session 边界运行；APK baseline 永久可用 |
| [Expo runtime versions](https://docs.expo.dev/eas-update/runtime-versions/) | 远程更新必须匹配原生 runtime compatibility | `Target` 在下载前检查 native、bridge、snapshot 和平台兼容范围 |
| [Expo error recovery](https://docs.expo.dev/eas-update/error-recovery/) | 首次内容出现前失败可以回退；已经运行后的回退不能假装撤销持久副作用 | candidate 健康前不开放有副作用 bridge；回退只切换 UI，不回滚原生业务事实 |
| [Ionic Appflow background strategy](https://ionic.io/docs/appflow/deploy/setup/capacitor-sdk) | 推荐后台下载、下次启动应用；需要前台重载时只阻止少量 critical workflow | 页面 modal、手势和选区不进入原生状态机，只保留一个 `canReplaceUi` 原生判定 |
| [Ionic differential updates](https://ionic.io/docs/appflow/deploy/differentials) | manifest 为每个文件保存 hash，只下载变化文件 | `Ensure(Target)` 按 content hash 复用 blobs，不在 serving 目录原位打补丁 |
| [OCI Distribution Spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md) | blob 由 digest 标识，manifest 引用 blobs，可变 tag 选择不可变内容 | generation 是不可变内容；Stable/Preview 只组成当前 `ReleaseView` |
| [Android WebViewAssetLoader](https://developer.android.com/develop/ui/views/layout/webapps/load-local-content) | 应用内资源从 HTTPS 语义的本地 origin 加载 | 下载和校验由原生完成，WebView 只看本地 generation |
| [Android WebView termination handling](https://developer.android.com/develop/ui/views/layout/webapps/handle-termination) | renderer 退出后不得复用旧 WebView；重复崩溃不能无限重载同一页面 | 单次重建 serving，重复失败回 fallback并 `RejectTarget` |
| [Android WorkManager unique work](https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/manage-work) | unique work 避免同一目标重复排队，并定义替换策略 | 每个 server 只有一个 `Ensure` owner；新 `Target` 取代旧工作 |

这些实践共同指向一个更小的心智模型：客户端不维护“正在检查、等待激活、重新下载”等平行状态机，而是持续回答三个问题：服务端当前想要什么、本地是否已经完整拥有它、下一次 UI session 应展示什么。

## 4. 能力和状态所有权

```text
┌────────────────────── akasic-agent / Core ──────────────────────┐
│ WebUI 源码与构建          发布者               Runtime           │
│ 产品界面、主题、组件  →  manifest + blobs  →  只读发现与下载接口 │
└───────────────┬───────────────────────┬─────────────────────────┘
                │ WSS 发布描述           │ 同身份 HTTPS 资源
                ▼                       ▼
┌──────────────────── Android / future iOS ───────────────────────┐
│ 配对与认证 → desired/ready → serving/fallback → 本地 origin       │
│ Room/outbox/附件/通知/系统能力/生命周期/激活协调器由原生层拥有    │
└──────────────────────────────┬───────────────────────────────────┘
                               │ versioned snapshot/patch/actions
                               ▼
                     ┌──────────────────────┐
                     │ WebView 产品界面     │
                     │ 无凭据、无任意网络、 │
                     │ 无文件路径与发布权限 │
                     └──────────────────────┘
```

| 对象或能力 | 权威 owner | WebUI 权限 |
|---|---|---|
| 主题、排版、导航、抽屉、island、消息和输入区视觉 | `frontend/chat` | 创建和渲染 |
| WebUI generation、manifest、Stable/Preview 指针 | Core 发布者 | 无 |
| 会话和消息真源 | Core SessionDB | 只读取原生层给出的投影 |
| 移动会话投影、阅读位置、草稿、outbox | 移动原生存储 | 展示并发出受约束动作 |
| 配对身份、TLS、token、下载 ticket | 移动原生层与 Core | 不可读取 |
| 附件上传下载、系统文件选择、相机、通知、分享 | 移动原生层 | 只调用 capability 白名单 |
| 插件运行状态和真实外部效果 | Core/插件 runtime | 只查询投影并提交显式动作 |
| WebUI 下载、校验、激活、回滚、GC | 移动原生层 | 只能参与 reload handshake |
| APK/未来 IPA 的构建、签名、安装和商店发布 | 平台发行链 | 无 |

WebUI 只能接收：

1. 版本化完整 snapshot 与有序 patch。
2. 业务动作 bridge，例如发送、重试、切换会话和插件查询。
3. 原生 capability 白名单及其明确成功、失败或取消结果。
4. reload 前的通用 durable ack 和 reload 后的 ready/healthy 握手。

WebUI 不得获得设备凭据、任意文件路径、任意 SQL、任意网络、发布指针写入、资源缓存写入、APK 安装或绕过原生校验的能力。

## 5. 目标发布结构

### 5.1 每个服务端独立发布

**C：** 已配对服务端是 WebUI 的信任根和版本 owner。每个 `server_id` 拥有独立 Stable、Preview、generation 集合、客户端缓存、激活状态和 rejected TargetKey 集合。服务端 A 的资源、指针或失败不得影响服务端 B。

```text
WebUI source inputs
       │
       ▼
┌───────────────┐   验证成功   ┌──────────────────────────────┐
│ 离线构建候选  ├─────────────→│ 不可变 generation            │
└───────────────┘              │ manifest + CAS blobs/archive │
                               └──────────────┬───────────────┘
                                              │ 原子提交
                         ┌────────────────────┴──────────────────┐
                         ▼                                       ▼
                  Stable pointer                         Preview pointer
                         │                                       │
                         └───────── current ReleaseView ─────────┘
                                              │
                           WSS descriptor + HTTPS content
                                              │
                                              ▼
                               per-server native local cache
```

构建、校验和资源写入发生在候选区。只有全部成功后，单一发布者才原子提交 Stable/Preview 指针组成的 `ReleaseView`。Runtime 只读取已提交发布，不在请求路径运行 Node/Vite，也不拥有发布写权限。

`ReleaseView` 是“服务端现在选择什么”的完整快照，至少包含 server identity、Stable target、可选 Preview target 和每个 target 的 manifest digest。规范化选择内容计算 `selection_digest`；审计 journal 可以记录每次发布事件，但客户端不按 journal 序号、发布时间或 semver 推断新旧。

这一点有意采用 OCI mutable tag 的语义：客户端每次只解析当前指针。显式回滚可以重新得到一个历史上出现过的 `selection_digest`，这是相同 desired state，不需要伪造一个“更大版本号”。每个 server 的 `Resolve` 单飞串行，迟到响应用本地 resolve token 丢弃，因此服务端备份恢复也不需要客户端理解 release lineage。

### 5.2 三层来源

| 来源 | 用途 | 生命周期 |
|---|---|---|
| Embedded baseline | APK 永久安全底座、首次启动和最终回退 | 永不被 OTA 覆盖或 GC |
| Stable | 服务端默认生产 WebUI | 由显式发布命令原子推进或回滚 |
| Preview | 服务端级试验 WebUI | 由显式发布、清除或提升操作改变 |

**C：** Preview 对该服务端配对的全部设备生效，不是单设备选择。Preview 效果不好时，发布者可以清除 Preview 或把指针回退到旧 generation；客户端下一次成功 Resolve 后收敛到当前 ReleaseView。

**C：** 保存源码、构建成功、文件 watcher 或重启 Core 都不得自动发布。Stable 和 Preview 都必须由名称明确的发布命令提交。Watcher 只能发送“有新发布可检查”的提示。

### 5.3 Preview 与 Stable 可复现性

- Preview 允许使用有未提交 WebUI 输入的源码，但构建前必须固化 base commit、patch/tree digest、完整 build context、产物 digest、`reproducible=false` 和 dirty provenance。
- Stable 只从指定 commit 的隔离 source snapshot、固定 lockfile/config/toolchain 构建，并且只接受 `reproducible=true` 的 generation。
- 当前开发 checkout 的后端、测试、文档甚至 WebUI dirty 都不影响从指定 commit 发布 Stable；它们不会进入隔离 source snapshot。
- dirty WebUI 输入不能通过忽略标记或伪造 commit 进入 Stable。
- 同一 generation ID 对应不同内容时 fail-loud；完全相同内容重复发布可以成为 no-op，但不能伪造新的资源内容。
- `reproducible=true` 的 Preview 可以原样提升 Stable。dirty Preview 必须先提交其真实输入，并从隔离 snapshot 重建出完全相同 generation；若产物不同，先发布新的 Preview 重新验证，不能把未复现的 dirty artifact 提升 Stable。

### 5.4 Manifest 语义

generation manifest 至少表达以下语义；最终字段名和编码在协议设计中确认：

- `manifest_schema`
- `generation_id`
- source repository、commit/base、WebUI 输入 digest 和 dirty provenance
- bundle/render digest 与构建工具身份
- bridge protocol 与 snapshot protocol 范围
- minimum native build、platform 和可选 WebView 约束
- entrypoint
- 每个文件的相对路径、content hash 和 size
- archive hash、总解包 size 和文件数量
- `reproducible`
- builder/publisher identity 和可复现构建信息

channel、Stable/Preview、发布时间、发布事件和当前选择不进入 generation manifest；它们属于 `ReleaseView` 或发布 journal。这样同一 Preview generation 才能原样提升 Stable。

`generation_id` 由经过规范化的 manifest 内容和资源内容决定，不由可变 channel、服务器路径或客户端时间决定。`TargetKey = server_identity + generation_id + manifest_digest`；generation 自身不可变。

## 6. 增量同步与缓存

**C：** 增量的含义是 `Ensure(Target)` 按 content hash 复用 blobs，不是在 serving 目录原位覆盖文件，也不是把补丁执行权限交给 WebView。

1. 客户端先获取并验证 manifest。
2. 客户端按 `server_id + content_hash` 检查已有 blobs。
3. 冷缓存或缺失比例较大时下载完整 archive；少量缺失时只下载缺失 blobs。
4. 校验每个 blob、完整文件树和 bundle digest。
5. 从不可变 blobs 物化 verified generation，使 `ready(Target)=true`。
6. serving generation 在 Present 健康提交前保持不变。

客户端不得跨 `server_id` 共享 CAS，即使 hash 相同。这样可以防止服务端身份撤销、保留策略和故障归属被全局去重混淆。

所有可用网络默认允许后台下载，不按 Wi-Fi/蜂窝做产品级禁用；下载必须受单文件、文件数、总压缩量、总解包量、压缩比、时间、并发和磁盘余量的硬限制。具体数值仍是 U。

## 7. Android 组件边界

目标 Android 原生层包含以下职责，名称表达职责而不是预先锁定类结构：

| 组件 | 职责 |
|---|---|
| `ReleaseResolver` | 串行读取当前 `ReleaseView`，得到唯一 compatible `desired Target`；hint 只令它重新解析 |
| `ArtifactEnsurer` | 每 server 单 owner，把 desired target 变成完整、已验证的本地 artifact；统一重试、等待和拒绝语义 |
| `UiSessionController` | 在 UI session 边界选择 serving、执行短暂 activation marker、健康提交和回退 |
| `LocalWebUiStore` | 保存 per-server blobs、verified generation、serving/fallback marker，并提供本地 asset handler 与安全 GC |

这些组件不得修改 SessionDB、Room 消息、outbox、草稿、附件、配对密钥或插件真源。WebUI 更新失败只改变派生资源、发布选择和诊断状态。

## 8. 客户端三步协调模型

客户端不实现一条把 `CHECKING`、`DOWNLOADING`、`WAITING_SAFE_POINT`、`ACTIVATING` 串起来的全局状态机。检查网络和替换 WebView 是两个独立 owner；把它们硬连成状态机会让每个新 edge case 都增加跳转。

客户端只持久化或推导四个事实：

| 事实 | 含义 |
|---|---|
| `desired` | 最近一次成功、已认证 `Resolve` 得到的 compatible Target；没有则为 baseline |
| `ready(desired)` | `LocalWebUiStore` 能否证明 desired 的 manifest、blobs 和物化目录完整且匹配 |
| `serving` | 当前 UI session 实际加载的 generation 或 embedded baseline |
| `fallback` | 最近一次健康 serving；不可用时为 embedded baseline |

所有行为只有三个幂等动作：

```text
authenticated trigger
        │
        ▼
┌────────────────┐       desired Target
│ Resolve(server)├─────────────────────────┐
└────────────────┘                         ▼
                                    ┌────────────────┐
                                    │ Ensure(Target) │  网络、hash、CAS、磁盘
                                    └───────┬────────┘
                                            │ ready
                                            ▼
UI session boundary ───────────────→┌────────────────┐
                                    │ Present(Target)│  只读本地资源，不等网络
                                    └───────┬────────┘
                                            ▼
                                          serving
```

`Resolve` 回答“服务端当前想要什么”；`Ensure` 回答“本地是否完整拥有它”；`Present` 回答“这次 UI session 展示什么”。下载失败不改变 serving，激活等待不占用下载 worker，WebView 页面状态也不影响后台下载。

维护者判断行为时只使用下面五条规则：

| 问题 | 唯一规则 |
|---|---|
| 什么时候检查 | 配对成功、重连、进入前台、hint、手动检查时 `Resolve`；同 server 请求合并 |
| 什么时候下载 | 当前 desired 不是 baseline，且 `ready(desired)=false` 时 `Ensure` |
| 什么时候等待 | 只有 `RetryAfter` 的计时、`WaitFor` 的明确事件，或 ready 后等待 UI session 边界 |
| 什么时候重新下载 | 当前 desired 改变，或 ready/partial 的完整性证据失效；只补缺失或损坏 blob |
| 什么时候替换页面 | `ready(desired)=true`、desired 与 serving 不同、到达 UI session 边界且 `canReplaceUi=true` |

### 8.1 Resolve：什么时候检查

以下事件令每 server 的 resolver 重新读取当前 `ReleaseView`：

- 配对认证完成。
- 已认证连接重新建立。
- 应用进入前台。
- 收到 release hint。
- 用户手动检查。

同一 server 同时只有一个 resolve 请求。请求期间到达的多个 hint 合并为一个 dirty bit；当前请求结束后最多再查一次。hint 不携带可直接下载或激活的 target。客户端用本地 resolve token 忽略旧请求结果，不比较远端 publication sequence、时间或 semver。

选择规则固定为：compatible Preview、compatible Stable、当前 serving、embedded baseline。Preview 和 Stable 都只是 `ReleaseView` 中的指针；服务端回滚后，下一次成功 Resolve 直接得到旧 generation 作为新 desired。

### 8.2 Ensure：什么时候下载、等待和重新下载

`Ensure(Target)` 是每 server 唯一、latest-wins 的后台工作。只有同时满足以下条件才访问网络：

1. target 来自当前已认证 `ReleaseView`。
2. target 与 native、bridge、snapshot 和平台兼容。
3. 本地还不能证明该 target 已 ready。

它先取得 descriptor 锚定的 manifest digest，再只下载缺失 blobs。进程重启、应用进入前台、Runtime 重启或重复 hint 都不会令已验证 target 重下。

Resolve/Ensure 对外只产生四种协调结果，不新增全局状态：

| 结果 | 典型条件 | 下一步 |
|---|---|---|
| `Ready` | target 已完整验证，或本次补齐后验证成功 | 等待 `Present`；不再访问网络 |
| `RetryAfter` | 离线、超时、连接中断、临时 5xx | 保持 serving；有界退避，reconnect 或手动检查可提前触发 |
| `WaitFor(trigger)` | 未认证、空间不足、native 不兼容、系统限制后台工作 | 不轮询；只等明确的 auth、space、binary upgrade、OS 或新 ReleaseView 事件 |
| `RejectTarget` | manifest/path 非法、当前发布 404、重复 hash 冲突、超硬限制 | 同一 TargetKey 不再自动尝试；只接受新 target 或用户显式重试 |

重新下载只发生在以下情况：

- 当前 desired 改变，新的 `Ensure` 替换旧任务；已验证、同 server 的 blobs 继续复用。
- partial 有相同 strong ETag/content digest 和 Range 证据时续传；否则只重下该 blob。
- 本地已存 blob 自检损坏时删除该 blob 并补一次；连续从服务端取得错误内容则 `RejectTarget`。
- ticket 401 时重新认证获取一次绑定 ticket；仍失败转 `WaitFor(auth)`，不匿名重试。
- 用户执行“重置此服务端 UI 缓存”后显式手动重试，或服务端提交了不同 TargetKey。

新 desired 出现时，旧下载可以安全取消，也可以完成当前不可分割写入；无论哪种方式，只有当前 desired 可以进入 `Present`。临时文件、旧 target 和已验证共享 blobs 的清理由 store 引用关系决定，不由 worker 猜测。

### 8.3 Present：什么时候替换 WebView

`Present` 永远不等待网络。它只在本地已经 `ready(desired)` 时考虑切换，并且只在 UI session 边界运行：

- 冷启动创建第一个 WebView 前。
- 应用从不可见回到前台、准备重新开放 UI 动作前。
- 切换到另一个 server 的 UI session。
- 用户点击“立即应用已下载更新”。

Activity 因旋转或系统重建但仍属于同一 UI session 时，继续加载原 serving，不借机切换版本。应用持续停留前台时不强制打断；界面只显示“已下载，将在下次进入应用时使用”，用户可以显式立即应用。

原生层只暴露一个 `canReplaceUi` 判定，不理解页面里的 modal、选区、抽屉、手势或各类 island：

1. 当前没有等待系统回调的文件选择、保存、权限或设置页请求。
2. incoming share 已复制到原生持久区。
3. 原生层可以生成一份完整、身份明确的 snapshot。
4. 没有另一次 activation、server switch、reset 或 revoke 持有 owner。

streaming、stop、outbox、上传和下载本身不阻止替换，只要它们由原生 owner 持久化并能进入 snapshot。重要草稿和 island 输入必须使用通用 durable draft/capability；只存在于 DOM 的临时展开、选择和未提交表单不升级成原生状态。

### 8.4 短 activation transaction

满足边界后，唯一 `UiSessionController` 执行以下短事务：

1. 重新读取当前 desired、ready 证据、server identity 和 `canReplaceUi`。
2. 关闭旧页面的普通 action admission；fence 后请求明确返回 `ui_reloading`。
3. 写入 `attempting TargetKey`、fallback、activation nonce 和当前 native/WebView compatibility fingerprint。
4. 销毁旧 WebView，从 generation-specific 本地 URL 创建 candidate WebView。
5. candidate 先以只读 bridge 完成 exact origin、nonce、generation 和协议握手。
6. 原生发送完整 snapshot；页面完成 root render 和首次 visual acknowledgment。
7. 原子提交 serving、清除 attempting，再开放有副作用的业务动作。

candidate 自己的 `reportHealthy` 只是证据之一，不能单独提交 serving。关键资源错误、协议身份不符、snapshot apply 失败、首次 visual acknowledgment 超时或 renderer termination 都使事务失败。

### 8.5 恢复、回滚和 rejected target

- activation 失败：立即加载 fallback；fallback 不可用时加载 embedded baseline。
- 启动发现 `attempting`：说明上次 activation 未提交。直接使用 fallback，并把精确 TargetKey 标为不再自动激活；诊断记为 `activation_incomplete`，不虚构为 renderer crash。用户可显式重试，新 TargetKey 也可重新尝试。
- serving 单次 renderer termination：销毁旧 WebView并有界重建同一 serving；窗口内重复发生则回到 fallback/baseline，并拒绝该环境下的 TargetKey。
- 普通服务端回滚：Resolve 得到旧 generation，Ensure 复用本地 blobs，下一次 UI session 边界 Present。
- emergency revoke：这是唯一可以绕过普通 session 边界的路径。先关闭 bridge admission，再回到已验证 fallback 或 baseline；原生持久状态不回滚。
- “样式不好看”不是健康失败。发布者显式清除 Preview 或把 Stable/Preview 指针改回旧 generation；客户端只协调 desired state。

## 9. 清理与重置

**C：** embedded baseline 永久保留。客户端只 pin 当前 serving、fallback、ready desired 和 attempting 所引用的 generation 及其 blobs。Stable/Preview 是服务端选择，不自动让客户端永久保留两个 channel 的全部历史。

正常稳定状态通常只有 serving 与 fallback 两代；desired 正在下载或激活时短暂增加一代。最终实现同时设置每 server generation 数和总字节硬上限，具体数值在真实 bundle 测量后确认。达到上限时先删未引用对象；若 pinned 集合本身超过预算则 `WaitFor(space)`，不得破坏回退链。

客户端提供两个名称明确的动作：

| 动作 | 允许改变 | 不得改变 |
|---|---|---|
| 清理未使用 UI 资源 | 删除各服务端未被 serving/fallback/ready desired/attempting 引用的派生 blobs、archive 和 orphan staging | pinned generation、baseline、业务数据和凭据 |
| 重置此服务端 UI 缓存 | 取消并等待该 server 的 Ensure/Present owner 退出，回到 baseline，删除其派生 UI 缓存，并将当前 TargetKey 置为需手动重试 | 配对身份、Room、outbox、草稿、附件、SessionDB、插件数据和其他服务端缓存 |

不提供任意本地版本选择器。正常版本选择由服务端当前 `ReleaseView` 决定；客户端只保留恢复所需 generation。重置后不会在下一个周期立即偷偷重下同一 TargetKey；用户手动重试或服务端 target 改变后才重新 Ensure。低磁盘时先运行安全 GC，空间仍不足则保留当前 UI、停止下载并明确报告，不删除 pinned 资源制造“成功”。

持久对象的最低语义如下；具体目录和 schema 是 U：

| 对象 | 正常增加 | 允许更新或逻辑失效 | 物理减少条件与 owner | 恢复证据 |
|---|---|---|---|---|
| 服务端 immutable generation | 成功构建并验证后新增 | 被新指针 supersede，不改内容 | 无指针/租约引用且满足保留协议时由发布 GC 删除 | manifest、blob hash、发布日志 |
| 服务端 CAS blob/archive | 以 content hash 新增 | 只改变引用可达性 | 无 generation/候选引用时由发布 GC 删除 | hash、size、引用扫描 |
| Stable/Preview pointer 与 `ReleaseView` | 发布事务创建 | 单 writer 原子替换当前选择；Preview 可显式清除 | 不直接删除所指内容；旧 generation 按 GC 协议处理 | 当前 ReleaseView、selection digest、发布 journal |
| 客户端 generation/blob cache | Ensure 验证后新增 | serving/fallback/desired/rejected 改变引用状态 | 仅上述清理、重置或安全 orphan recovery | hash、manifest、per-server reference set |
| `attempting` marker | Present 关闭旧 bridge 后写入 | 成功清除；未提交恢复为 rejected-for-auto TargetKey | 只能由完成或启动恢复事务清除 | fallback、nonce、TargetKey、启动恢复报告 |
| embedded baseline | 随 APK 安装 | APK 升级替换整个应用版本 | OTA 和 UI GC 永远无权删除 | APK 签名和内嵌资源 hash |

## 10. 信任和网络边界

**C：** 不新增独立 WebUI 签名密钥。信任链来自现有配对服务端身份：

```text
paired server identity
        │ authenticated WSS
        ▼
release descriptor ── short-lived bound ticket ──→ same server HTTPS
        │                                         manifest / blobs
        └──────────────── hashes + limits ────────────────────┘
```

- descriptor 必须来自已认证 WSS，并绑定 `server_id`、设备、`selection_digest`、Stable/Preview TargetKey 和 manifest digest。
- 下载 ticket 短时有效、一次或有界使用，绑定资源和配对 epoch；401 只能重新获取 ticket，不能降级匿名下载。
- HTTPS 不跟随任意 host redirect；最终 TLS/服务端身份必须与配对身份一致。
- 设备 revoke 后停止新发现和下载，关闭该服务端 bridge admission，并按明确产品动作决定是否保留已验证缓存；不得继续把旧 ticket 当授权。
- WebView 继续只加载原生提供的本地可信 origin，generation 进入 URL path；entry HTML 使用 `no-store`，content-hash asset 可以 immutable。CSP 不开放任意网络，外部链接交给系统浏览器。
- WebView navigation、subresource 和 service worker 不得绕过 asset handler 访问发布服务器或第三方。
- Web→Native bridge 使用 exact-origin、main-frame 的单一版本化消息 envelope；Android 优先使用带 `allowedOriginRules` 的 [WebViewCompat.addWebMessageListener](https://developer.android.com/reference/androidx/webkit/WebViewCompat#addWebMessageListener)。candidate 健康前不开放有副作用 capability。Android 官方把加载不可信内容的 JavaScript interface 列为高风险边界，见 [WebView native bridge security](https://developer.android.com/privacy-and-security/risks/insecure-webview-native-bridges)。
- archive 必须拒绝绝对路径、`..`、重复规范化路径、symlink、特殊文件、zip bomb、超限文件数和超限解包体积。
- manifest、文件 hash、archive hash 或 content length 不一致时 fail-loud，不使用“尽量能打开”的部分版本。

## 11. Edge case 合同矩阵

每个 case 必须归入发布事务、`Resolve`、`Ensure`、`Present` 或 Store 中的一个 owner。新增 case 只能选择已有结果 `Ready / RetryAfter / WaitFor / RejectTarget` 或 Present 的 commit/fallback，不得为单个异常再造一条全局状态。

### 11.1 构建与发布

| ID | 场景 | 必须结果 |
|---|---|---|
| PUB-001 | 当前 Git worktree 有无关后端、测试或文档 dirty | Stable 从指定 commit 的隔离 source snapshot 构建；当前 checkout dirty 不参与判断 |
| PUB-002 | WebUI 构建输入含未提交变化 | 只允许 Preview；固化 base commit、patch/tree digest、产物和 `reproducible=false` |
| PUB-003 | 构建、manifest、磁盘或校验失败 | 不创建可见 generation，不改变 ReleaseView |
| PUB-004 | 资源写入后、ReleaseView 提交前崩溃 | 当前选择不变；候选成为可证明未引用的 orphan |
| PUB-005 | ReleaseView 提交过程中崩溃 | 恢复后只能读到完整旧选择或完整新选择 |
| PUB-006 | 并发 Stable/Preview 发布 | 单 writer 或 compare-and-swap；后到者基于当前 ReleaseView 重验 |
| PUB-007 | 同一 generation ID 对应不同内容 | fail-loud，不覆盖任何旧内容 |
| PUB-008 | 相同内容重复 Preview | 允许 no-op；复用同一 generation/blobs |
| PUB-009 | Preview 提升 Stable | 只允许 `reproducible=true` 或已从提交输入重建出相同 generation 的 Preview；指针改变与 Preview 清除一次事务提交 |
| PUB-010 | watcher、保存源码或 Runtime 重启 | 不自动发布；最多在已提交后发 hint |
| PUB-011 | Stable 回滚到旧 generation | 改变当前指针；若完整选择与历史相同，允许得到相同 selection digest |
| PUB-012 | Stable 回滚时 Preview 仍存在 | 明确提示 Preview 仍优先，不暗中清除 |
| PUB-013 | 发布者在事务提交前发出 hint | hint 必须由 commit 后 outbox/回调产生；过早 hint 不得暴露候选 |
| PUB-014 | 发布 GC 与指针提交并发 | ReleaseView 可达对象在同一引用/租约协议下 pin；不得提交指向缺失 blob 的选择 |
| PUB-015 | commit 相同但 lockfile、构建配置、toolchain 或环境不同 | WebUI build context fingerprint 不同；Stable 可复现证据必须覆盖全部真实输入 |
| PUB-016 | 发布仓从备份恢复到旧选择 | 恢复后的当前 ReleaseView 是 desired truth；审计 journal 记录恢复，但客户端不比较历史序号拒绝它 |

### 11.2 发现、认证和身份

| ID | 场景 | 必须结果 |
|---|---|---|
| AUTH-001 | 未配对 | 只用 baseline，不请求远程 descriptor |
| AUTH-002 | WSS 未认证或会话失效 | 不接受 ReleaseView，继续 serving/baseline；`WaitFor(auth)` |
| AUTH-003 | 收到 release hint | 经认证重新查询，不直接信任 hint 内容 |
| AUTH-004 | 旧 Resolve 在本地目标已改变后返回 | resolve token 不匹配即丢弃；同一 server 不并发提交两个结果 |
| AUTH-005 | ticket 过期 | 重新鉴权获取 ticket；不匿名重试 |
| AUTH-006 | ticket 设备、资源或 pairing epoch 不匹配 | 拒绝并报告认证失败 |
| AUTH-007 | 设备被 revoke | 立即关闭 bridge admission并回 baseline；停止 Resolve/Ensure，保留业务数据和已验证缓存直到明确清理 |
| AUTH-008 | HTTP redirect 到其他 host | 拒绝，不携带 ticket 跳转 |
| AUTH-009 | TLS identity 改变 | 停止更新并要求重新配对，不静默接受 |
| AUTH-010 | server identity 变化但地址相同 | 视为新服务端；旧缓存不能继承 |
| AUTH-011 | 地址改变但固定 server identity 相同 | 只有经过配对 profile 的显式重绑定/验证才沿用缓存；redirect 不能完成重绑定 |
| AUTH-012 | 相同 selection digest 或 TargetKey 返回不同规范化内容 | fail-loud，拒绝整个 ReleaseView/Target |

### 11.3 网络与传输

| ID | 场景 | 必须结果 |
|---|---|---|
| NET-001 | Wi-Fi、蜂窝或受限但可用网络 | 均可自动下载，仍受统一硬限制 |
| NET-002 | 下载中断 | 只对有明确 range/identity 的资源有界续传，否则从该 blob 重下 |
| NET-003 | 长时间离线 | 保持 serving；最后一次已认证 desired 仍可使用本地 ready artifact，重连后重新 Resolve |
| NET-004 | 401 | 更新一次绑定 ticket；仍失败转 `WaitFor(auth)` |
| NET-005 | 当前 Target 的 manifest/blob 404 | `RejectTarget` 并报告发布仓损坏，不解释为“没有更新” |
| NET-006 | 412、strong ETag 或内容身份变化 | 丢弃该 blob partial并重新 Resolve，不拼接不同内容 |
| NET-007 | Content-Length、实际字节或 hash 超限/不符 | 删除临时对象并 fail-loud |
| NET-008 | archive 路径逃逸、symlink、特殊文件或压缩炸弹 | 在物化前拒绝整个候选 |
| NET-009 | Ensure A 期间 Resolve 得到 Target B | B 替换唯一 worker；A 不可 Present，已验证 blobs 可复用 |
| NET-010 | 临时错误连续发生 | `RetryAfter` 有界退避并可观察，不阻塞 serving |
| NET-011 | 应用或设备重启 | 先读取 verified marker/hash；完整 target 不因重启重新下载 |
| NET-012 | 本地单个 blob 自检损坏 | 只删除并补齐该 blob；从服务端连续得到错误内容后 `RejectTarget` |
| NET-013 | 设备离线后服务端已清除 Preview | 设备无法猜测远端变化，继续使用最后已认证 desired；重连 Resolve 后收敛，不按本地时间伪造清除 |
| NET-014 | 客户端或服务端 wall clock 回拨/跳变 | 时间只用于展示、TTL 或退避，不用于判断 ReleaseView 新旧 |

### 11.4 存储、GC 和人工清理

| ID | 场景 | 必须结果 |
|---|---|---|
| STO-001 | 下载前空间不足 | 先安全 GC；仍不足则停止候选下载并报告 |
| STO-002 | GC 遇到 serving/fallback/ready desired/attempting 引用 | 跳过，不以低空间为由删除 |
| STO-003 | 多进程或多个更新任务竞争缓存 | per-server store lock 保证单 writer |
| STO-004 | 发现 `.partial`、orphan staging 或无提交 marker | 只清理能证明未引用的对象 |
| STO-005 | serving 资源损坏 | 回到 verified fallback/baseline，并对当前 desired 重新 Ensure |
| STO-006 | 清理未使用 UI 资源 | 只删未引用派生资源，不改变版本指针和业务状态 |
| STO-007 | 重置此服务端 UI 缓存 | 取消并等待 owner，回 baseline，只删其派生 UI 缓存；同 TargetKey 等手动重试 |
| STO-008 | 两个 server 有相同 content hash | 仍保存为身份隔离的缓存，不跨 server 引用 |
| STO-009 | GC 或重置中途进程死亡 | 通过引用扫描和事务 marker 恢复，不留下 serving 指向空目录 |
| STO-010 | embedded baseline | 永不参与 OTA GC |
| STO-011 | pinned 集合自身超过 generation/byte cap | 保留回退链并 `WaitFor(space)`；不谎报清理成功 |
| STO-012 | OS 清除 cache 目录 | serving/fallback 放 app-private durable files；丢失的 partial/未引用 cache 可重建 |
| STO-013 | app data 被恢复到另一设备但 Keystore/配对身份不可用 | 不执行旧远程 UI；回 baseline并等待重新配对，业务数据按其独立恢复合同处理 |
| STO-014 | reset/clear 与 Ensure/Present 并发 | 同一 store/coordinator owner 先取消并等完成，再删除；不得边读边删 |

### 11.5 生命周期与进程恢复

| ID | 场景 | 必须结果 |
|---|---|---|
| LIFE-001 | 应用在后台 | Ensure 可以继续；不创建或替换 WebView |
| LIFE-002 | 冷启动存在 ready compatible desired | Present 只读本地资源，不等待网络；失败立即 fallback |
| LIFE-003 | Activity 重建或旋转 | 同一 UI session 继续 serving；application-scope owner 不借机切 desired |
| LIFE-004 | Ensure 时进程死亡 | 下次从可信 marker恢复或重验，不把 partial 当 ready |
| LIFE-005 | 启动发现 `attempting` | 视为 activation 未提交，使用 fallback，并禁止同 TargetKey 自动重试；不虚构 crash 原因 |
| LIFE-006 | APK 升级 | 重新检查全部 cached generation 与新 native/bridge 兼容性 |
| LIFE-007 | APK 降级 | 不加载要求更高 native build 的缓存；baseline 必须可用 |
| LIFE-008 | WebView provider/version 变化 | compatibility fingerprint 与 rejected key 重新评估，不能永久继承旧判断 |
| LIFE-009 | 应用长时间停留前台 | 不强制中断；显示 ready 状态，下一 UI session 或用户立即应用 |
| LIFE-010 | 从文件选择、权限或设置页返回前台 | 先完成原生 continuation；该次 resume 暂不 Present |
| LIFE-011 | WebView provider 缺失、禁用或无法创建 | 显示原生恢复页并提示系统修复；不能假称 baseline 已成功渲染 |
| LIFE-012 | APK 升级时旧 `attempting` 尚在 | 新 binary 先用自己的 baseline，废止旧 attempt，再按新兼容性重新 Resolve/Ensure |
| LIFE-013 | serving 健康提交后应用被 force-stop 或 OS kill | serving marker 已提交，下一次可继续使用；不当作 candidate 失败 |

### 11.6 用户行为与页面状态

| ID | 场景 | 必须结果 |
|---|---|---|
| UX-001 | 正在输入或 IME composition | 自动 Present 不在前台交互中途发生；草稿继续通过通用原生 owner 持久化 |
| UX-002 | durable draft/阅读锚点写入尚未得到原生 ack | `canReplaceUi=false`；不为每种页面表单增加独立 blocker |
| UX-003 | stream、stop 或 resync | stream/stop 不阻止；只有原生暂时无法生成完整 snapshot 时 `canReplaceUi=false` |
| UX-004 | 文件选择、保存、权限或设置 Activity result | 等待原生 continuation 完成 |
| UX-005 | incoming share 仍依赖临时 URI | 等待复制进原生持久区 |
| UX-006 | 页面拥有 pending plugin query | 销毁时显式 cancel并返回；迟到结果由 nonce 拒绝，不等待查询完成 |
| UX-007 | modal、文本选择、拖动或手势进行中 | 原生不感知；自动 Present 只在 UI session 边界发生 |
| UX-008 | drawer、surface、滚动锚点 | 需要恢复的内容进入通用 snapshot/durable state；否则允许回默认状态，不保持 DOM |
| UX-009 | island 中有未提交表单 | 重要输入使用通用 durable draft/capability；纯 DOM 临时表单不升级为 native owner |
| UX-010 | 已持久 outbox、上传或下载仍进行 | 不阻塞；新页面从原生 snapshot 恢复 |
| UX-011 | 用户动作恰好跨 fence | fence 前正常提交；fence 后明确 `ui_reloading`，不得重复 |
| UX-012 | 用户切换服务端 | 创建新 UI session；分别按新 server 的 desired/ready/serving 选择 |
| UX-013 | 用户点击“立即应用” | 执行一次通用 durable ack/提醒；`canReplaceUi=false` 时明确等待原因，不绕过原生 owner |
| UX-014 | 更新 ready 后用户一直不离开前台 | 不强制 reload；显示一次非阻塞状态，避免反复提示 |

### 11.7 WebView、bridge 和健康检查

| ID | 场景 | 必须结果 |
|---|---|---|
| WEB-001 | 页面声明错误 generation/bridge/snapshot | 健康检查失败并回滚 |
| WEB-002 | 旧 WebView 的迟到 callback | nonce/generation 不匹配，忽略并记录 |
| WEB-003 | WebUI 调用客户端不支持 capability | 返回明确 unsupported；不得假成功 |
| WEB-004 | 页面尝试外部网络或任意 navigation | 原生阻止；外链按白名单交系统浏览器 |
| WEB-005 | entrypoint 或关键资源加载失败 | 回滚，不渲染半版本 |
| WEB-006 | 初始化 JS、bridge、root render 或 snapshot apply 失败 | 回滚并 `RejectTarget` |
| WEB-007 | candidate 在健康窗口 renderer crash | 回滚并 `RejectTarget` |
| WEB-008 | serving 页面单次运行期 crash | 有界重建同版本 |
| WEB-009 | serving 页面窗口内重复 crash | 回滚 fallback/baseline 并 `RejectTarget` |
| WEB-010 | UI 只是主观不好看但技术健康 | 客户端不自作回滚；由服务端发布者清除/回滚 Preview 或 Stable |
| WEB-011 | 通用 durable ack 超时或返回不完整 | 取消本次 Present，serving 继续；不枚举页面内部组件 |
| WEB-012 | reportHealthy 来自错误 nonce 或 snapshot | 拒绝，不清除 attempting |
| WEB-013 | iframe 或错误 origin 调用 native bridge | exact-origin、main-frame listener 拒绝；不使用向所有 frame 暴露的大接口 |
| WEB-014 | Service Worker 或 browser cache 返回旧 generation 资源 | Service Worker 不得联网；generation-specific URL、HTML no-store、hash asset immutable |
| WEB-015 | 路径大小写、Unicode normalization 冲突或 MIME 与扩展不符 | manifest 规范化阶段拒绝整个 target |
| WEB-016 | action 在 reload 前后重复、capability 已废弃 | 版本化 envelope + request ID；幂等与终态由原生 owner 决定 |
| WEB-017 | 页面自报 healthy 但尚未产生可见首帧 | 原生 visual acknowledgment 未到，不提交 serving、不开放写动作 |

### 11.8 连续发布、多设备和多服务端

| ID | 场景 | 必须结果 |
|---|---|---|
| CON-001 | 下载 A 时发布 B | latest desired wins；A 不激活 |
| CON-002 | A ready、尚未 Present 时 Resolve 得到 B | A 不再是 desired；Ensure B，下一边界不能激活 A |
| CON-003 | 合格 Preview 提升 Stable 并清除 Preview | 一次发布事务；设备最终指向同一已验证 generation |
| CON-004 | Stable 回滚但 Preview 未清除 | Preview 设备继续 Preview，并向操作者明确提示 |
| CON-005 | 某一设备 `RejectTarget` | 只影响该设备与 compatibility fingerprint，不自动全局回滚服务端 |
| CON-006 | 部分设备长期离线 | 上线后只 Resolve 当前 ReleaseView，不逐个重放发布事件 |
| CON-007 | emergency target 本地缺失 | 直接 baseline；后台再取合法 fallback target |
| CON-008 | server A 失败、server B 正常 | 缓存、指针、rejected TargetKey 和通知完全隔离 |
| CON-009 | 从备份恢复发布仓 | 已认证 Resolve 直接接受恢复后的当前 ReleaseView；不比较客户端历史序号或时间 |
| CON-010 | 短时间收到大量 hint | 合并为 dirty bit；单飞 Resolve 结束后最多补查一次 |
| CON-011 | 旧 Resolve/Ensure callback 晚于 server switch | owner token 不匹配即忽略；不能改变新 UI session |
| CON-012 | 同一应用同时维护多个 server UI 缓存 | 每 server 独立 Resolve/Ensure/Store；全局容量只触发安全 GC，不串改 desired |

### 11.9 兼容性与二进制发行

| ID | 场景 | 必须结果 |
|---|---|---|
| BIN-001 | Android 纯 UI/CSS/已有桥交互 | 服务端 WebUI 发布，不发 APK |
| BIN-002 | 新原生 capability 或不兼容 bridge | 设置 minimum native build 并发布二进制 |
| BIN-003 | 无兼容 generation | 保持 serving/baseline，并明确提示升级应用 |
| BIN-004 | 当前 GitHub APK 发行 | updater、下载、安装入口和权限保持原状；与 WebUI OTA 使用独立 owner |
| BIN-005 | 未来准备提交 Google Play | 另开设计和政策审查；本轮不提前删除 GitHub updater。Google 对 WebView JavaScript 与 APK/原生代码自更新的边界见 [Device and Network Abuse](https://support.google.com/googleplay/android-developer/answer/16559646?hl=en) |
| BIN-006 | 未来 iOS | 复用中立 manifest/Resolve/Ensure/Present；哪些 WebUI 变化可 OTA 需按 [App Review Guidelines 2.5.2](https://developer.apple.com/app-store/review/guidelines/) 单独确认 |
| BIN-007 | 服务端下发 dex/JAR/.so 或原生可执行内容 | 永久拒绝；WebUI OTA 只接受声明的静态 Web 资源类型 |
| BIN-008 | APK 升级带来新 baseline，但服务端仍选择旧 generation | 按 compatibility 和当前 ReleaseView 选择，不以 build 时间判断；无兼容 target 时使用新 baseline |

## 12. 用户可见行为

- 正常 Resolve、Ensure 和等待下一 UI session 默认安静进行，不弹阻塞对话框。
- 激活成功可以使用轻量非阻塞提示；频繁 Preview 更新不应持续打扰。
- 回滚、资源损坏、磁盘不足、需要升级原生应用、配对身份变化和 emergency revoke 必须明确可见。
- 手动检查只展示从四个事实推导出的结果：已是当前版本、正在补齐、已下载等待下一次进入、`WaitFor` 的明确原因或 `RejectTarget` 的明确错误。
- “立即应用”只在 ready 时出现；`canReplaceUi=false` 时显示原生 continuation/snapshot 等粗粒度原因，不展示内部状态名。
- “清理未使用 UI 资源”和“重置此服务端 UI 缓存”必须明确不会删除聊天、草稿、附件或配对；执行结果报告释放空间和当前来源。

## 13. 分阶段实施方向

以下只是设计顺序，不是当前实施授权：

1. 接受本设计后，新增决策记录并勘误 0018 与移动仓库固定 ZIP/no-network 合同。
2. 确认服务端发布仓的持久化根、备份/恢复、增改减和发布事务。
3. 固化平台中立 generation manifest、ReleaseView、WSS descriptor、HTTPS ticket/content 和 bridge envelope。
4. 实现 Core 隔离构建、不可变 generation、CAS、Stable/Preview 发布和只读 Runtime 接口。
5. Android 分别实现 `ReleaseResolver`、`ArtifactEnsurer`、`UiSessionController` 与 `LocalWebUiStore`，不再实现平行的下载/等待/激活总状态机。
6. 运行 Core/Android targeted tests、change-impact Gate、跨仓库固定组合和真机故障注入。
7. 当前 GitHub APK updater 保持原状；未来真的准备 Google Play 或 iOS 时分别建立新合同，不作为首版 OTA 前置工作。

## 14. 验收边界

设计落地至少需要证明：

- 纯 UI 改动从 Core 发布后，已配对 Android 无需 APK 更新即可在下一 UI session 或用户显式立即应用时使用。
- WebView 始终加载经过认证发现、完整校验并从本地可信 origin 提供的不可变资源。
- 任意构建、发布、下载、校验、激活、renderer crash 或进程死亡故障都保留 serving/fallback/baseline 中至少一个可启动版本。
- 所有传输失败都能唯一归入 `RetryAfter`、`WaitFor(trigger)` 或 `RejectTarget`；不会因为新 edge case 增加全局更新状态。
- 连续发布最终只 Present 最近一次成功 Resolve 的 desired Target；不会在 serving 目录混合两个 generation。
- 草稿、阅读位置、outbox、附件、消息、配对和插件真实状态在更新、回滚、GC 和重置前后符合各自 owner 合同。
- Preview dirty provenance 可审计，Stable 能从声明的 WebUI 输入重建相同产物。
- 所有安全上限、archive 攻击、ticket 失效、身份变化和恶意 navigation 都有独立失败测试。
- 多服务端、多设备、冷启动、后台、Activity result、IME、streaming、连续崩溃、备份恢复、旧 callback 和低磁盘场景有明确 oracle，且分别绑定 Resolve/Ensure/Present/Store owner。
- Android 真机证据与 Core CI、跨仓库组合证据分层报告，不用模拟成功替代设备行为。

## 15. 待继续商讨

以下内容尚未批准，不能从本文推断实现：

1. 服务端发布仓位于 Akashic `<workspace>`、独立 deployment state root，还是由配置显式选择；Stable/Preview、generation、CAS 和 publication journal 是否进入正式备份。
2. 各持久对象的具体目录、schema、锁、事务、retention 数值、备份恢复和跨版本迁移。
3. generation manifest、ReleaseView、selection digest 与 WSS/HTTPS 的精确字段名、命令、事件、错误码、endpoint 和 ticket 生命周期。
4. 文件数、大小、压缩比、并发、超时、重试、健康窗口、崩溃窗口和 rejected TargetKey 重试窗口的具体常量。
5. bridge snapshot/patch/capability 的精确兼容矩阵、single-envelope 迁移和通用 durable ack；不再为页面各类临时状态设计专用 flush 字段。
6. 两仓库 PR 切分、任务合同、Gate 选择、真机矩阵和正式发布操作手册。
7. Google Play 与 iOS 的未来发行资格和具体改动；当前 GitHub APK updater 不变，该项不阻塞首版 OTA。
