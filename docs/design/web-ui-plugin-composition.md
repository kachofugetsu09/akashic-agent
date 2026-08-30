# 2236 WebUI 插件组合设计

- 状态：accepted / implementing
- 日期：2026-08-30
- 关联需求：WEBUI-001～WEBUI-007、PLG-001～PLG-004、PLG-006、PLG-008、PLG-010～PLG-011、PLG-014～PLG-016、ONB-001、MOB-001
- 关联决策：[0051](../decisions/0051-web-ui-composes-ordinary-plugin-modules.md)、[0037](../decisions/0037-plugin-runtime-is-pure-v3.md)、[0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md)、[0022](../decisions/0022-mobile-webui-uses-server-selected-generations.md)、[0043](../decisions/0043-paper-brand-tokens-replace-material-visual-semantics.md)、[0050](../decisions/0050-model-revision-lives-in-ordinary-plugin.md)；已被取代的 [0008](../decisions/0008-plugin-runtime-publishes-only-committed-snapshots.md) 只保留 committed snapshot 不变量的历史说明
- 上游设计：[模型普通插件与 Provider 组合规格](model-plugin-ordinary-capability-spec.md)、[v3 包级 contribution](plugin-v3-package-contributions-task-contract.md)、[v3 DashboardContext](plugin-v3-dashboard-context-task-contract.md)、[v3 Mobile UI/query capability](plugin-v3-mobile-ui-query-task-contract.md)

## 1. 结论

可行。2236 只需要补两项彼此正交的通用能力：

1. **Web module 发布**：普通插件从自己的 artifact 发布浏览器 JS、CSS 和静态资源；Core 把资源校验后绑定到 exact plugin snapshot。
2. **递归 mount 组合**：页面拥有自己的子挂载点；其他普通插件等待该挂载点出现，再登记一个可撤销的 UI contribution。

对插件作者只呈现一套 `ctx.ui` API。两项内部能力不能强行揉成一个对象：module publication 处理服务端 artifact、摘要和 snapshot；mount composition 处理浏览器中的父子关系、顺序和卸载。它们沿用现有 `Service → inject → Effect → snapshot/lease`，不建立第二套插件 manager、generation、权限或热更新模型。

```text
Core Web Host
└── web.root                            唯一原始 mount
    └── shell-ui                        普通插件，拥有品牌顶栏和页面导航
        └── shell.pages
            ├── conversation-ui         普通插件，拥有会话侧栏与对话页
            ├── workbench-ui            普通插件，拥有工作台页
            │   └── workbench.panels    Dashboard 面板迁入
            ├── runtime-ui              普通插件，拥有知识与运行页 adapter
            └── models                  普通插件，拥有模型页
                └── models.connection-types
                    ├── openai-compatible
                    ├── codex
                    └── opencode-go
```

“知识与运行”由 `runtime-ui` 顶层页面 contribution 保留。删除该插件只删除页面 adapter，不删除 MCP、Skill、job、runtime inspection、Akasha 或移动端的底层能力。

## 2. 用六岁小孩能懂的话解释

Core 只是一块有电的空地，留一个总插座：`web.root`。`shell-ui` 插进来后才出现屋顶、门牌和顶栏，并再留一个 `shell.pages` 插座。

- “对话”插件插进 Shell，就出现对话房间。
- “工作台”插件插进来，就出现工作台房间。
- “知识与运行”插件插进来，就出现运行信息房间。
- “模型”插件插进来，就出现模型房间。

模型房间自己又留了一个小插座：`models.connection-types`。Codex、OpenCode Go 和 OpenAI-compatible 各自把自己的连接按钮和表单插进去。拔掉 Codex 插件，只少 Codex 那一块；房子、模型房间和已经保存的 Connection 都不会消失。

房子不需要知道“Codex”是什么。Core 只知道三件事：哪一包资源属于哪个插件、哪个插座是谁声明的、拔插件时要把它插入的东西一起撤掉。

## 3. 用户意图与成功标准

用户希望 L 形 2236 WebUI 本身由平等、非特权、可外置安装的普通插件拼成：

- 顶部保留“对话”“工作台”“知识与运行”“模型”，不再由 `frontend/dashboard` 写死。
- 页面可以继续声明自己的嵌套 UI 扩展点；Provider UI 是第一条纵向组合证明。
- `models`、`codex`、`opencode-go`、`openai-compatible` 即使移出本仓库，作为普通插件正式 install 后仍能提供同样页面和行为。
- 新增第四种 Provider 只新增插件，不修改 Core、Shell 或 `models` 的 Provider 分支。
- 将来 Dashboard、Onboarding 和其他顶栏页面可以使用同一个原子，但首版不提前发布没有真实消费者的 slot、layout schema 或 UI DSL。

这是本轮 UI 插件化的北极星。Onboarding 只是将来的一个普通消费者，不是本轮能力、实现或验收的前置条件。

“普通插件”验收的是正式安装链和运行行为，不是源码目录看起来像插件，也不是 manifest 有一个 `builtin` 标志。

## 4. 当前事实、已确认选择与未知

### 4.1 实现后的真实调用链

`frontend/dashboard/src/main.tsx` 只启动通用 Web Host。`shell-ui` 注册顶栏、history 和
`shell.pages.v1`；`conversation-ui`、`workbench-ui` 与 `models` 分别注册页面。删除任一普通插件，
对应页面就不再进入 catalog，Core 没有页面名称分支。

`models` 声明 `models.connection-types.v1` 并拥有模型状态、默认模型、Embedding 模型和 Connection
布局；OpenAI-compatible、Codex 与 OpenCode Go 插件各自注册认证和连接 UI。Provider 的出现不再由
`PROVIDER_TEMPLATES` 或 Host 分支决定。

嵌套样式沿同一父子关系生效：Host 只提供 paper token 和隔离作用域；`models` 在自己的
`settings-dialog` 下提供 contract 中列出的表单 class，Provider 只使用这组公开 class；Workbench
同样只在 `workbench.panels.v2` 子树提供表格、详情和工作台视觉词。子插件样式不能越过
自己的 module root 反向修改父级或兄弟，父插件样式可以向自己的子 mount 继承。

本轮先把插件化前的页面作为逐像素金标准，因此从旧页面搬入普通插件的 CSS 可以在 module 私有
作用域内继续消费 0043 已允许的 Material/`--ak-color-*` 兼容别名；这不把旧别名提升成 Host 或新插件
公共语义。等 1:1 迁移验收完成后，视觉系统迁移应在独立变更中保持截图基线或明确接受视觉变化，
不能把架构迁移与品牌重做混成一次不可归因的变化。

插件 UI 与 Dashboard 数据 API 现在组成一条链：

```text
Web module
  → package contribution
  → exact snapshot WebUiCatalog
  → content-digest JS/CSS
  → Host 激活 module 并组合 Mount

Dashboard module
  → exact snapshot DashboardBinding
  → ctx.http 携带同一 snapshot/module/generation identity
  → Core HTTP route

Mobile UI
  → UI_SLOTS.register_mobile(...)
  → candidate Root freeze
  → RuntimeSnapshot.mobile_ui_registry
  → catalog / content-digest asset / bounded read-only query
  → Android WebView runtime
```

Web Host 复用既有 candidate isolation、原子发布、Effect 清理和 exact snapshot query 不变量，
并增加父 Mount 撤销时对子登记的递归清理。Mobile 仍由自己的 `UI_SLOTS` 和 generation owner 管理，
没有被改名或并入 2236。

旧 Dashboard source-directory discovery、请求期编译、import map、`AkashicDashboard` global 和
浏览器 panel adapter 已删除。`dashboard_module` 只保留插件自己的数据 API owner，不再拥有浏览器 UI。

### 4.2 已确认选择

- Core 只拥有通用插件组合、资源边界、发布和诊断；不拥有顶栏、产品页面或 Provider 名称。
- `models` 是模型状态和模型页的领域 owner；Provider 插件拥有自己的认证、transport、图标、说明和连接 UI。
- WebUI 继续使用纸张品牌 token；新插件 UI 不把旧 Material 别名提升为新公共语义。
- 同 UID 普通插件是受支持 API 的隔离，不是恶意代码安全沙箱。
- 本实现已获授权；数据迁移和删除运行数据仍不在授权范围。

### 4.3 已处理的现有决策冲突

[0051](../decisions/0051-web-ui-composes-ordinary-plugin-modules.md) 已勘误 [0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md)：共享对话实现仍以 `frontend/chat` 为唯一真源，桌面顶层页面注册和 adapter 改由普通 `conversation-ui` 插件拥有；Android baseline/OTA、Room 和 Bridge owner 不变。

[0022](../decisions/0022-mobile-webui-uses-server-selected-generations.md) 已经定义 Mobile 产品 WebUI 的不可变 generation、Stable/Preview 和客户端 CAS。本设计不得复制这些 owner。2236 的 Web module catalog 只是 exact plugin snapshot 的派生投影，没有独立 Stable 指针、journal、retired manager 或持久 generation。

### 4.4 保留边界

- Conversation 继续复用 `frontend/chat` 的产品实现，但顶层 entry、readiness 与 adapter 由
  `conversation-ui` 普通插件拥有；Host 不提供 iframe 或 Chat 专用 API。
- Workbench 面板只通过 `workbench.panels.v2` 登记。v2 固定结构化 entry，并让 Host 为计数、分页与详情读取提供 `AbortSignal`；旧 Dashboard browser ABI 已在最后一个仓库内
  consumer 迁完后删除；插件自己的 `dashboard_module` HTTP route 保留。
- Web module 暂不增加任意 UI DSL、跨插件 DOM 查询、全局 event bus 或第二套 generation。
- Web module 的资源预算等有第二个真实容量问题再设计，不照抄 Mobile 240 KiB。

## 5. 最少概念

本设计只增加两个公共概念：

| 概念 | 唯一含义 | 不拥有 |
|---|---|---|
| `WebModule` | 一个插件 artifact 内经校验、按摘要寻址的浏览器入口和资源集合 | 页面顺序、领域数据、独立 generation、网络权限 |
| `Mount` | 一个 UI owner 声明的、有父节点、有 entry schema 和 cardinality 的组合位置 | 静态资源、业务数据库、全局路由 manager、权限继承 |

`Registration` 只是当前 Fiber 拥有的 `Effect`，不是第三个 domain manager。`WebUiCatalog` 只是 `RuntimeSnapshot` 的不可变派生值，不拥有第二套生命周期。浏览器中的 `MountTree` 是当前 catalog 执行后的可观察拓扑，不是权威持久状态。

`BrowserCatalogSession` 是 `Mount` 的浏览器端生命周期实现，不是第三个插件概念。它只拥有当前 tab 已激活 module 的 token、disposer 和 mount ledger；不拥有服务端 snapshot、业务状态或持久 generation。

首版 mount cardinality 只有：

- `single`：最多一个 entry。
- `list`：多个 entry，按稳定 `order` 和 `id` 排序。

首版不提供 `keyed`、`chain`、session scope、任意 JSON schema、跨插件 DOM 查询或全局 event bus。出现第二个无法由 `single/list` 直接表达的真实消费者后，再单独设计。

## 6. Core 最小原子

### 6.1 Web module 发布

普通插件以无副作用的包级 contribution 声明入口，例如 `web_module = "web_module.js"`。首版每个 module 只能发布一个 ESM 和一个可选 CSS 文件。ESM 只能导入 Host 已经公开的 `react`、`react/jsx-runtime`、`react-dom/client` 和 `@akashic/web-ui-v1`；不能引用 lazy chunk、远程包、外部字体、图片或其他运行时静态文件。`@akashic/web-ui-v1` 只含主题与 Material 控件等全局原子，不含工作台图表、分页、领域布局或产品页面。小图标由 bundle 或 data URL 自带。这个限制用更少的生命周期换来可证明的一致性，出现真实的大包消费者后再设计分块。

Web Host 在导入任何插件前发布唯一 React/ReactDOM 实例。Shell、Workbench 和普通子插件都把这三个包视为外部 Host SDK；父插件的激活顺序、CSS 或 mount 存在与否不负责偷偷初始化 renderer。

Core 在插件 `apply()` 前：

1. 解析路径并拒绝越出插件 artifact、symlink escape、错误 MIME 和超限资源。
2. 校验入口只导入 Host SDK，并冻结 JS、CSS、字节数与 SHA-256。
3. 将 descriptor 放入 candidate snapshot 的 `WebUiCatalog` 派生投影。
4. 资源或声明校验失败时拒绝 candidate；通过后随同一个 RuntimeSnapshot 原子发布。
5. 一次 `WebUiBootstrap` 响应携带 catalog 与全部 JS/CSS bytes。服务端只在发送该响应期间短租它对应的 snapshot，响应完成或连接取消即释放；浏览器完整接收并核对全部摘要后才启动任何 module。

浏览器 catalog 只接受当前 exact snapshot 中 active Fiber 的 module。普通产品请求不执行或猜测页面树；浏览器 Host 在执行任何 module 前先核对整个 catalog 的字节数与摘要。

### 6.2 浏览器 module ABI

所有外置 module 只导出一个版本化入口：

```ts
export function activate(ctx: WebHostContextV1): () => void
```

`WebHostContextV1` 的类型合同独立发布供构建时检查；运行时对象只包含 `ctx.ui` 和默认路由到同 owner Dashboard route 的 `ctx.http`。snapshot、catalog、module 和 generation identity 由 Host 私下附加到请求，不暴露成插件领域状态。entry 的 `render(host, view)` 只获得自己的 DOM host 与自己声明的 child mount view。这里约束普通插件的公共 ABI 和误路由，不是同 realm 恶意代码的安全隔离；安装 Web module 等同于信任其浏览器代码。

Host 创建 `BrowserCatalogSession` 后激活全部 module。每次 `activate()` 都在 per-module transaction 中运行：新 registration/inject subscription 先挂到暂存 token，只有返回有效幂等 disposer 才整体提交；抛错或返回无效值会按 token 逆序撤销全部 entry/child/subscription，再记录 module error。该 contribution 显示通用错误态，其他 module 继续工作，不留下幽灵 entry。

`activate()` 必须同步完成登记。全部 module 激活后，Host 关闭该 catalog 的 registration admission；未出现的被注入 mount 使对应 module 失败。新拓扑只能来自新 catalog，运行时可见性变化留在 component state，不通过 timer/promise 改 registry。session replacement、reload 或 page close 时，Host 逆序调用 module disposer，最后释放 mount ledger。

server Fiber dispose 只会让旧 catalog 变成 stale，不能隔空执行浏览器 disposer。递归清理由浏览器 session 自己执行；server snapshot 仍按自己的 lease 规则排空。这两个生命周期只有 catalog identity 对齐，不互相伪装。

### 6.3 递归 mount registry

Core Web Host 在浏览器启动时预声明唯一根 mount `web.root.v1`。`shell-ui` 注入它并声明 `shell.pages.v1`；每个已验证 module 获得窄 `ctx.ui`：

```ts
ctx.ui.inject(MOUNT_KEY, (mount) =>
  mount.register({ id, order, render, children })
)
```

- `inject` 使用既有依赖语义：mount 未声明时等待；owner 消失时停止并撤销当前 entry。
- `register` 返回当前 module activation 拥有的 Effect/disposer。
- `children` 由当前 entry owner 声明。父 entry 被撤销时，Core 按逆序递归撤销整个子树。
- parent component 只取得一个按 `children` 静态收窄的 `renderMount(key, ownerProps)`；Host 不渲染未声明的 child。声明、渲染范围和递归 cleanup 使用同一张 ledger。
- duplicate mount、duplicate entry、版本不匹配、`single` 冲突、循环父子关系和 freeze 后登记都 fail-loud。
- entry `props` 由 mount owner 的版本化合同解释。Core 通用 registry 只解释 `id`、`order`、render handle、parent 和 lifecycle。

`web.root.v1` 的合同只允许一个根 renderer，由 Core Web Host 拥有。`shell.pages.v1` 的合同由 `shell-ui` 拥有，包含导航 label、icon、route key 和页面 renderer。导航与页面是同一个 entry 的两个投影，不能拆成 `NAV_ITEMS` 与 `PAGES` 两个 registry，否则二者会漂移。

### 6.4 Core Web Host

Host 只拥有：

- 空 HTML 启动面、`web.root.v1` 和 paper/ink/rule/typography/status token root；
- 全局 loading、空组合、单页加载失败和 stale catalog 恢复界面；
- module loader、catalog identity、诊断和刷新提示。

Host 不拥有：

- Akashic 品牌顶栏、导航、history、`conversation`、`workbench`、`models` 或任何 Provider ID；
- 会话侧栏、工作台侧栏或全局 `left.sidebar`；
- 模型 readiness、默认模型、embedding 或 Provider auth；
- 页面业务 API、数据库、credential、Dashboard query 或 Mobile bridge。

L 形区域不是一个全局侧栏原子。顶部横条属于 `shell-ui`；下面的左侧区域属于活动 page 插件。对话页可以放会话列表，工作台可以放模块列表，模型页可以不放左栏。这样改变一个页面布局不会迫使其他页面或 Core 改接口。

### 6.5 样式跟随组合树

样式不增加第二套 registry。Host 只发布 paper、ink、rule、typography 和 status token；每个普通插件的
CSS 随自己的 module 一起安装，并只拥有自己创建的 DOM。父 entry 同时拥有 child host，因此可以在
自己的根节点下提供低优先级的排版与控件基线，子插件自然继承 token，并用自己的 class 补充领域布局。

```text
Host token root
└── shell-ui                 顶栏、导航、页面容器
    ├── workbench-ui         工作台侧栏、面板排版与控件基线
    │   ├── akasha           检索列表与详情布局
    │   └── other panel      自己的领域布局
    └── models              连接行、系统模型、Provider 对话框基线
        ├── codex            登录流程特有布局
        └── opencode-go      登录流程特有布局
```

父插件只能选择自己的根节点和它创建的 child host，不能选择子插件私有 class；供子插件覆盖的基线使用
低 specificity。子插件不能选择父插件私有 DOM，也不复制父级视觉规则。依赖关系决定 DOM 嵌套，DOM
嵌套负责继承；Core 不解释 CSS 内容、插件名称或页面类型。删除父插件会连同 mount 和视觉上下文一起
删除整棵子树，这与现有 Effect 生命周期一致。

直接调用 `child.render(entryId, host)` 时，Host 自动把子 module 的 stylesheet scope 安装到 `host`。
Workbench 这类宽接口不会直接调用 `child.render`，而是由子 entry 按语义槽位提供同一种
`mount(host, dispatch) → void | disposer`。子插件可以在 mount 内创建 React root，也可以直接操作 DOM；
父插件不区分实现。父插件在调用 mount 前使用同一个通用 `child.style(entryId, host)`，并在插槽卸载时调用返回的 disposer。
这只是把同一条 entry ownership 投影到父插件拥有的 child host，不增加样式 registry，也不让 Host 认识
Workbench、Provider 或任何 CSS class。

Host 在安装 stylesheet 前拒绝无法由 `@scope` 隔离的全局命名 at-rule，包括 `@keyframes`。插件若需要
关键帧动画，使用浏览器 Web Animations API；字体和其他全局名字由 Host token 与资源层提供。一个 child host 同时
只能有一个 stylesheet owner，重复绑定会 fail-loud，组合必须通过 DOM 嵌套表达父子继承。

Workbench 的命令式 `renderMain`、`renderDetail`、`renderNavBody`、`renderFilters` 与
`renderTopbarAction` 统一返回 `void | disposer`。返回 disposer 的插件拥有自己建立的请求、timer、
listener 和临时窗口；Workbench 在重绘、切换或卸载对应 DOM 时先清理 renderer，再释放该 child host
的 stylesheet scope。React 插件把 `root.unmount()` 作为同一个 disposer 返回，不建立第二套生命周期。

## 7. 页面和 Provider 插件怎样组合

### 7.1 顶层页面

```text
┌──────────────────────────────────────────────────────┐
│ Akashic │ 对话 │ 工作台 │ 知识与运行 │ 模型    主题 │  shell-ui
├─────────┬────────────────────────────────────────────┤
│         │                                            │
│ page    │       active page plugin                   │  页面自己决定
│ sidebar │                                            │  是否需要左栏
│         │                                            │
└─────────┴────────────────────────────────────────────┘
```

| 插件 | 注入 | 注册 | 自己拥有 |
|---|---|---|---|
| `shell-ui` | `web.root.v1` | 唯一 Shell；声明 `shell.pages.v1` | 品牌顶栏、页面导航、route/history |
| `conversation-ui` | `shell.pages.v1` | `conversation` page | 会话侧栏、消息、composer、desktop adapter |
| `workbench-ui` | `shell.pages.v1` | `workbench` page；声明 `workbench.panels.v2` | Session/Plugin 工作台布局、最新读取与 panel adapter |
| `runtime-ui` | `shell.pages.v1` | `runtime` page | 知识与运行的 desktop adapter |
| `models` | `shell.pages.v1` | `models` page；声明 `models.connection-types.v1` | catalog、Connection、Binding、默认 chat/embedding 的 UI |

page 合同不包含 readiness、onboarding 或 redirect。首版迁移期间保留现有 `/api/shell/state → models` 跳转 adapter；它必须被标为模型特判删除点，并在硬编码 Shell 退场时一并删除，不等待 Onboarding。没有默认聊天模型时，对话插件显示自己的不可用状态，用户仍可手动进入模型页。将来 Onboarding 另做普通消费者，不能为了它先把“通用恢复目标”塞进所有页面合同。

### 7.2 模型页面中的 Provider UI

```text
models page
├── 已连接                    MODEL_CATALOG
├── 系统模型与 embedding      MODEL_CATALOG + MODEL_SETTINGS
└── 添加连接                  models.connection-types.v1
    ├── OpenAI-compatible     endpoint / API Key / manual model
    ├── Codex                 device auth / account / discovery
    └── OpenCode Go           local auth or API Key / discovery
```

`models` 只通过公开 child props 给出窄动作：打开/关闭流程、提交 provider-neutral auth command、刷新 catalog、显示 receipt。Provider child 的受支持 ABI 不包含 `ModelsState`、SQLite、credential store、generic command 或其他 Connection；同 realm 信任边界仍按 8.1 的说明处理。

`models.connection-types.v1` 的 props 类型和运行时 schema 由 `models` 的公开 contract artifact 拥有。该 contract 作为可独立安装、带 lock/version/schema digest 的前端构建依赖发布；Provider 在构建时依赖它，产出的 bundle 只运行时导入 Host SDK，并在 module descriptor 声明接受的 contract ID/digest。candidate validator 把它与当前 `models` module 发布的 descriptor 核对。Provider 不能 import `models` 的页面、store 或 Python/TypeScript 实现，也不能复制 schema。Core 只比较通用 ID/digest envelope，不解释模型字段。

Provider 的 UI 与 backend driver 在同一个普通插件 artifact 中发布并共享插件身份，但它们通过公开的两个正交接口组合：backend 注入 `MODEL_DRIVERS`，browser module 注入 `models.connection-types.v1`。UI 挂到模型页不代表获得 backend Service；backend driver 注册也不自动显示 UI。

Driver 与 UI 是两个独立 contribution，不增加 `ui=required` 或 `ui=none` 状态。首批三个 Provider 的发行 artifact 同时携带两者；ordinary-plugin Gate 分别证明 driver 注册和浏览器 entry。缺少 UI 时 driver 仍可供已有 Connection 使用，只是不显示“添加连接”入口；缺少 driver 时 UI 的动作会明确失败。这样不把发行完整性变成 Core 或模型执行 ABI，浏览器行为仍由唯一 Host 和 E2E 验证。

卸载 Provider 时：

- 对应“添加连接”入口和在途 auth UI 随 Effect 消失。
- 已保存的 Connection、Model、Binding 和 revision 仍由 `models` 保留。
- catalog 把相关 Connection 显示为 `driver unavailable`，不猜测 transport，不改写或删除数据。
- 正在使用旧 exact snapshot 的 server operation 按现有 lease 排空；新操作明确失败。

## 8. 数据、动作与信任边界

Web module 是视图代码，不是新的业务数据面。生产 session 首版复用现有设置、Dashboard、Chat 和插件 API；正常调用通过 Host 注入、绑定当前 catalog/module identity 的窄 client。它是唯一受支持的 ABI，不是对同 realm 主动绕过的安全隔离。只有第二个真实消费者证明现有边界不够时，才抽取新的通用 query/action API。

边界 owner 如下：

| 边界 | owner | 集中校验 |
|---|---|---|
| plugin artifact → Web module | Core loader | 路径、MIME、摘要、大小、import、active generation |
| catalog → browser | Core Web Host API | snapshot identity、descriptor schema、cache headers |
| browser module → mount | mount registry | active module token、parent、version、cardinality、duplicate |
| browser → settings/action HTTP | Host client + 现有 control host | catalog/module token、current snapshot、身份、同源/CSRF、request schema、snapshot lease |
| settings command → model state | `models` | revision CAS、领域规则、probe、credential commit |
| Provider wire | Provider driver | 外部协议、auth、model discovery、错误映射 |

Host client 自动携带 `snapshot_id + catalog_id + module_id + generation_id`，并把请求路由到同一插件通过既有 `dashboard_module` 注册的 route。服务端不把旧请求落到 current handler。父子 UI 正常通过 mount props/callback 组合，不借 HTTP 调用兄弟插件。identity header 是 exact generation 的路由事实，不是不可伪造的 capability；这个边界防止 stale 和普通实现误路由，不抵抗同一 JS realm 中主动绕过 Host API 的代码。不可信 UI 需要另立 iframe/worker/process 设计。

## 9. Catalog、更新与并发

一次浏览器启动先取得一个 `catalog_id`。它由 current exact RuntimeSnapshot 的插件身份、module descriptor 和摘要派生，不是模型 revision，也不是 Mobile WebUI generation。

```text
candidate plugin Root
  → validate and freeze WebUiCatalog
  → publish same RuntimeSnapshot
  → browser loads exact catalog
  → module apply builds MountTree
```

- catalog 中的全部 module 必须来自同一个 committed snapshot，不能混用新旧资源。
- active page 不在用户填写表单时热替换。发现新 catalog 后，当前 `BrowserCatalogSession` 标记 stale 并显示轻量“界面已更新，重新打开”提示。
- session 已把所有 JS/CSS 装入内存，因此 stale 后仍可切换并显示任何已发布页面；所有新 Host HTTP 请求只返回 `stale_catalog`，不能提交旧表单或落到 current handler。
- HTML 与 `WebUiBootstrap` 使用 `no-store`；浏览器可以把已验证 bytes 按摘要放入普通 cache。首版没有独立 asset 请求、lazy chunk 和延迟静态资源，因此不需要 WebUI 专用 retention manager、浏览器长 snapshot lease 或 GC grace period。
- 每个 Host client 请求自动携带 `snapshot_id`、`catalog_id`、`module_id` 和 `generation_id`。既有 Dashboard middleware 不等待、不 fallback：它租用该 exact snapshot，核对 active Web module，再只匹配同 owner 的插件路由。任一不符立即返回 `stale_catalog` 或 `forbidden_contract`。
- 成功 lease 绑定整个 HTTP handler、插件 entrypoint 和 action commit，直到 response 或持久事务结束才释放。检查后不能重新读取 current Root，现有 settings/Chat/Dashboard handler 必须通过该绑定进入；不能只在 router 前检查一次 catalog。

Core 不为浏览器长时间持有 server snapshot lease。每次请求只租调用瞬间仍可接收请求的 exact snapshot，不等待下一代，也不尝试重新租已经 retired 的 snapshot。已加载的 DOM 不是 server operation 已回滚或仍可提交的证据。

### 9.1 Candidate 边界

candidate 只验证它真正拥有的静态 artifact 事实：路径不越界、资源可读且有界、入口符合同步 ABI、只导入 Host SDK、摘要与 exact snapshot 投影一致。它不启动 Chromium、Node、临时 HTTP 服务或第二套 DOM；这些依赖会把普通插件安装变成新的部署生命周期，也无法证明真实交互正确。

JS 语法、首次 `activate`、mount 冲突、首屏 render 和 disposer 由生产使用的唯一 `BrowserCatalogSession` fail-loud 并按 module 隔离。仓库内及正式发布的插件另外通过真实浏览器 E2E 验证主要交互。这里不把“CI 测过”伪装成任意第三方代码的运行时安全证明。

### 9.2 生产资源边界

声明合同要求 module 只使用 `WebUiBootstrap` 内的 JS/CSS；静态图标只允许 `data:`，插件数据请求只通过 Host client 进入同 owner 的 Dashboard 路由。CSP 同时阻止外部 origin、worker 与常见外链资源，但同 realm 插件仍可主动创建同源或 Blob 资源，因此这里不是恶意代码 sandbox，也不把 CSP 声称为完整资源证明。

生产 violation 在请求发出前拒绝，并把对应 module 放入可诊断 error entry；其他页面保持可用。这里仍不是针对恶意同 realm 插件的完整 sandbox，而是让“一次 bootstrap 包含完整可执行资源”成为可验证合同。

## 10. 失败、取消与清理

| 场景 | 行为 |
|---|---|
| 一个 module 下载/校验失败 | 该 page/contribution 进入错误态；其他 page 保持可用；candidate 阶段则拒绝整体发布 |
| page render 抛错 | page 级 error boundary；Host 导航仍可用 |
| child render 抛错 | child entry error boundary；父页面和兄弟 entry 不白屏 |
| parent plugin 卸载 | server 将旧 catalog 标 stale；browser session replacement/close 时先停 admission，再逆序撤销 child、parent entry 和 module disposer |
| child plugin 卸载 | server 将旧 catalog 标 stale；新 session 不含 child，旧 session 关闭时只撤销该 entry，不重建 parent |
| auth flow 中卸载 Provider | 取消 UI attempt；backend attempt 按 models 合同 cancel/expire，不提交 credential |
| catalog 更新时有脏表单 | 不热换；提示用户保存/放弃后刷新 |
| browser 请求旧 catalog | current snapshot identity 核对失败并返回 `stale_catalog`，保留用户可理解的刷新路径，不 fallback 到 current handler |
| duplicate/cycle/version mismatch | candidate fail-loud；stable 不变 |
| cleanup 抛错 | 汇总所有失败并标记 snapshot/plugin health；不静默吞掉 |

## 11. 持久状态与恢复

本设计没有新的用户权威持久状态。

| 对象 | 正常增加 | 原位更新或逻辑失效 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| Web module descriptor/catalog | candidate compile 派生 | 不原位更新；随 snapshot 替换而不可达 | 随 plugin snapshot/artifact 安全 GC | Core；plugin artifact、snapshot identity、摘要 |
| BrowserCatalogSession/MountTree | catalog activation 在内存建立 | 标记 stale；registration/disposer 改内存 ledger | 页面关闭、刷新或 session replacement 后释放 | Core Web Host；catalog、activation token、cleanup ledger |
| 浏览器 asset cache | 按摘要下载 | immutable，不原位更新 | 浏览器 cache policy 或安全 GC | Core/browser；内容摘要 |
| 模型 Connection/Model/Binding/Revision | 继续按模型规格 | 继续按模型规格 | 本设计不授权自动减少 | `models`；SQLite、operation backup、revision |
| Session/Message、plugin-data、Mobile state | 各自现行 owner | 本设计不改变 | 本设计不授权删除 | 各自数据库、文件和现有恢复合同 |

回滚 UI 组合实现只恢复旧 Shell 和旧 Provider 模板入口；不回滚、迁移或删除模型、会话、plugin-data 或 Mobile WebUI generation。

## 12. 迁移顺序

### 阶段 0：固定决策和消费者地图

- 以 0051 接受本设计并勘误 0018 的桌面页面入口 owner。
- 扫描 `frontend/**/src`、Dashboard module、Mobile UI、Onboarding、runtime inspection、所有内置与外部插件 cache/source。
- 固定 `WebModule`、`Mount`、catalog identity、错误码和 ordinary-plugin Gate。

### 阶段 1：只实现通用 Host 和组合原子

- Core 增加静态 `web_module` contribution、artifact 校验和一次性 snapshot-bound `WebUiBootstrap` endpoint。
- 新 Host 只含空根 mount、loader、token 和错误/更新界面；`shell-ui` 普通插件提供品牌、history 与页面 mount。
- 用一个外置 fixture 插件证明 root page、nested child、卸载、candidate reject 和冷启动。
- 不迁移任何产品页面前先删除 fixture 之外没有消费者的 API 字段。

### 阶段 2：迁移工作台

`workbench-ui` 注册顶层 page 和 `workbench.panels.v2`。Shell 与 Workbench 的可编辑源码位于各自
`plugins/*/web/`，聚合构建器按入口发现插件，只消费插件源码和 `@akashic/web-ui-v1` 的公开 Tailwind preset，
不再把 `frontend/dashboard/src` 或 Dashboard 私有配置当作隐藏输入。面板迁为只依赖 Host SDK 的 Web module 后，旧 Shell dashboard 分支、浏览器 panel adapter、源码扫描和请求期编译一起删除；插件自己的 Dashboard
HTTP/data ABI 继续由 `dashboard_module` 拥有。

### 阶段 3：迁移模型页和 Provider 子 UI

- `models` 注册 page、声明 `models.connection-types.v1`，拥有 catalog/default/embedding/Connection 布局。
- 三个 Provider 插件分别注册 child UI；删除 `PROVIDER_TEMPLATES` 和 Provider 图标/表单分支。
- 用卸载 Codex、安装第四 Provider、保留 unavailable Connection 做纵向 Gate。
- Onboarding 以后可注入同一 Provider mount 或复用 Provider-neutral action，不复制 Provider 表。

### 阶段 4：迁移对话页

完成 0018 勘误和 shared source 分界后，由 `conversation-ui` 注册 page 并直接挂载共享 Chat 实现。
SessionDB 只追加、Web/Mobile adapter、stream 局部更新、Android baseline/OTA 和 bridge owner 不变；
旧 Shell chat iframe 分支已经删除。

### 阶段 5：删除硬编码与兼容层

- 删除 `ShellView` union、固定导航按钮、runtime page 和三套 iframe 特判。
- 删除 `/api/shell/state → models` 兼容跳转，不等待 Onboarding；对话页自己显示模型未就绪和前往模型页的普通动作。
- 删除 `PROVIDER_TEMPLATES`、Provider 特定图标选择和 dialog 类型分支。
- 删除最后一个消费者已经迁完的 Dashboard adapter、legacy globals 和兼容 CSS。
- 对每个删除点核对源码、安装 cache、动态入口、运行 catalog 和外部普通插件兼容义务。

## 13. 正交性与概念完整性检查

### 13.1 变化轴

| 变化 | 应修改 | 不应修改 |
|---|---|---|
| 新增顶层页面 | 新 page 插件 | Core Host、其他页面、模型状态 |
| 改一个页面左栏 | 该 page 插件 | root mount、其他页面 |
| 新增 Provider | 新 Provider 插件 | Core、Shell、`models` Provider 分支 |
| 改 Codex 登录 | `codex` UI/backend | OpenAI-compatible、模型 catalog schema |
| 改默认模型 | `models` state/UI | ReAct、Host、Provider transport |
| 改 Shell history | `shell-ui` | Core Host、page domain、Provider |
| 插件升级 | exact plugin snapshot/catalog | Mobile Stable 指针、model revision |
| 改 Android 原生布局/bridge | Mobile adapter/native | 2236 mount tree、模型状态 |

任一实施 diff 违反表中“不应修改”列时，必须说明真实边界；没有不可避免边界就继续收敛。

### 13.2 直接性

普通任务应只有一条短路径：

```text
新增页面：publish WebModule → inject shell.pages
新增 Provider：register MODEL_DRIVERS → register models.connection-types
卸载插件：dispose Effect
发布更新：candidate validate → publish RuntimeSnapshot
```

若新增页面需要改导航数组、路由 union、iframe switch 和 readiness switch，设计失败。若新增 Provider 需要改 Core、models 页面和 Provider 模板表，设计失败。

### 13.3 明确拒绝的方案

- 不做通用 JSON UI DSL 或第二套 renderer adapter。React/ReactDOM 是版本化 Host SDK 的一部分，Core 只负责同一实例；插件把 React root 包在普通 mount/disposer 内，父插件不解释组件语义。
- 不拆独立导航 registry 和页面 registry。
- 不做全局 `left.sidebar`。
- 不把 Provider metadata 或能力表搬进 Core。
- 不让 Provider import `models` 或兄弟插件实现。
- 不把 iframe 当长期默认插件隔离；它只可作为迁移 adapter 或未来不可信 UI 的独立安全设计。
- 不复制 Mobile WebUI generation、Stable/Preview、CAS 或 plugin snapshot manager。
- 不把 mount 嵌套误解为权限继承。
- 不提前照搬参考实现的 `keyed`、`chain`、scope 和全局 slot vocabulary。
- 不强迫 Web 和 Mobile 使用同一页面树；共享的是身份、能力、状态语义和可复用组件。

## 14. 验收合同

### 14.1 Core 与普通插件

- Core production path 不包含 Shell、`conversation`、`workbench`、`models`、`codex`、`opencode`、`openai` 等产品/来源分支。
- page 的导航和 renderer 是同一 entry；不存在必须同步的第二张导航表。
- 将每个目标插件源码移出仓库并清空旧 cache 后，从外部 artifact 正式 install；禁止额外 `PYTHONPATH`、repo-relative import、旧 Dashboard globals 和 Core `/chat`/`/settings` iframe 掩盖实现。
- 收集 Python `__file__`、JS module URL/digest 与 catalog provenance；验证 candidate → promotion → cold boot → upgrade → revert → uninstall → reinstall。
- 联合场景同时移走 `models`、`openai-compatible`、`codex`、`opencode-go` 四个源码目录，只从正式 artifact 完成 UI action、auth、discovery、chat 和 embedding。
- 在仓库源码不存在时，按锁定的 `models.connection-types.v1` 或 `workbench.panels.v2` contract package 独立构建 Web bundle；运行时 JS 只导入 Host SDK，构建输入只含插件自身和公开类型合同，运行时核对 schema digest。
- `shell-ui`、`conversation-ui` 与 `workbench-ui` 分别通过同一外置 Gate；最终 Gate 在 iframe、`PROVIDER_TEMPLATES`、legacy globals 和 repo 内目标源码都不存在时重跑，迁移 adapter 通过不能冒充普通插件证明。
- `builtin` 插件和外部插件经过相同 loader、candidate、snapshot、asset、mount 和 cleanup 路径。

### 14.2 模型纵向组合

- 移除 `codex` 只移除 Codex 新建入口；已有 Codex Connection 保留并显示 `driver unavailable`。
- 新增第四 Provider 只增加一个普通插件；Core、Host 和 `models` 无源码 diff。
- `models` 页面仍能设置默认 chat、role 和 embedding；Provider child 无法直接读写其他 Connection 或 credential。
- 真实 Codex/OpenCode/OpenAI-compatible 登录/连接、模型发现、chat 和 embedding 继续由各自 driver 完成；UI 不复制 transport。
- 首批 Provider 的同一普通 artifact 同时通过 driver 与 UI entry 验收；删掉 `web_module` 的 fixture 证明 driver 可以独立 headless，删掉 driver 的 fixture 证明 UI 不取得隐含 backend 权限。

### 14.3 发布与生命周期

- candidate 资源、路径、同步入口或摘要校验失败时 stable catalog 和页面保持不变；浏览器 ABI 失败只隔离对应 contribution。
- catalog 中全部资源来自同一个 exact snapshot；网络记录和页面诊断不出现 mixed snapshot。
- browser catalog session replacement/close 时 parent dispose 递归撤销 child，child dispose 不重建 parent；cleanup 顺序和失败可观察。server snapshot retirement 不伪装成已执行浏览器 disposer。
- active form 不被热更新替换；stale request 明确提示刷新，不偷偷落到 current snapshot。
- 更新后不刷新、首次进入先前未打开页面时，页面代码仍已完整加载并显示；Host HTTP 明确返回 stale，不出现 chunk 404。
- catalog bootstrap 与 plugin publish/uninstall 并发时，单个响应只包含同一 snapshot 的完整 bytes；取消会释放短 lease，成功不会出现 catalog 已到但首个 module 404。
- action admission 与 publish/uninstall 并发时，要么取得请求声明的 exact lease 并让完整 handler/commit 排空，要么立即 stale；不能等待后落到新 Root。
- 浏览器网络记录证明生产 session 除一次 bootstrap、catalog state 和同 owner Dashboard 请求外没有资源请求。
- 一个 page/child 崩溃不让 Host、导航、父页面或兄弟 contribution 白屏。

### 14.4 用户体验与可访问性

- 2236 顶栏只显示已注册页面，默认顺序稳定；对话、工作台、知识与运行和模型均由各自普通插件注册。
- 插件化前的同数据截图、DOM 几何和主要交互是迁移金标准；除已确认删除的旧 Akasha 模型配置块外，迁移不得改变可见体验。
- 浏览器 back/forward、deep link、刷新、无模型时的对话不可用提示和无页面空态行为确定。
- 键盘可以进入顶栏、切换页面、返回触发按钮；焦点在 page/child 卸载后回到可预测位置。
- 320 px、常用桌面宽度、200% zoom、浅色/深色、reduced motion 和屏幕阅读名称保持可用。
- 页面布局继续使用 paper/ink/rule/typography/status；插件不能依赖 Host 私有 DOM 或 CSS selector。

### 14.5 受保护状态

- UI 组合、候选验证、安装、卸载和回滚不 UPDATE/DELETE `sessions.db/messages`，不改变 Room、outbox、附件、配对或 Mobile serving generation。
- Provider 卸载不删除 Connection、Model、Binding、credential 或模型 revision。
- candidate 只写既有 validation root 和派生缓存；discard 后不留正式 plugin-data 或 credential。

## 15. 未来展望

当三个真实页面和 Provider 子 UI 都通过 ordinary-plugin Gate 后，同一个 mount 原子可以自然承载：

- Onboarding 注册顶层或临时恢复页面；Akasha、Proactive 和 Provider 只在安装时贡献自己的步骤。
- Dashboard、Project、插件管理器、诊断和其他顶栏页面成为普通插件。
- 页面 owner 在需要时声明 toolbar、inspector 或 settings 子 mount；没有消费者前不加入 Core 词汇。
- Runtime inspection 展示 catalog、module digest、mount parent、entry owner、Effect 和失败 provenance。
- candidate preview 对比新旧 MountTree，让维护者在发布前看到新增、删除和冲突。
- 插件可以在自己的 DOM host 内使用 Host React、打包后的 Web Components 或普通 DOM；Core 不增加 renderer adapter，也不把 React component 当跨平台 wire schema。

最终系统仍只有一套组合哲学：插件发布能力，消费者注入能力，注册由 Effect 拥有，一次操作观察 exact snapshot。UI、模型、ReAct、Channel、Scheduler 和 Akasha可以各自变化，却使用同一种生命周期语言。这同时满足正交性和概念完整性。

## 16. 参考实现的采用与舍弃

本设计参考 `/mnt/data/source-code/deepseek-harness`：

- `docs/subsystems/client-modules.md` 的 client module graph 和 revision 思路；
- `packages/client/ui-slots/src/index.ts` 的递归 slot、parent-owned children 和 dispose subtree；
- `packages/client/ui-settings/src/client/contract/slots.ts` 的领域 owner 声明嵌套 UI contract。

采用的是“module 发布与 UI mount 分轴”“父 owner 声明 children”“递归 dispose”。没有照搬它的全部 cardinality、scope 或 slot 名称，因为 Akashic 首批三个页面只需要 `single/list` 和一个 root scope。参考仓库证明形状可行，Akashic 的 plugin snapshot、Mobile generation、持久状态和安全 owner 仍由本仓库现行合同决定。
