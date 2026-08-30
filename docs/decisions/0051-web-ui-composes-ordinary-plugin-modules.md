# 0051 · WebUI 由普通插件递归组合

- 状态：accepted / implementing
- 日期：2026-08-30
- amends：[0018](0018-chat-webui-has-one-source-and-two-adapters.md) 的桌面页面入口 owner
- 关联条款：WEBUI-001～WEBUI-007、PLG-001～PLG-004、PLG-006、PLG-008、PLG-010～PLG-016、ONB-001、MOB-001

## 背景

2236 当前把顶栏、对话、工作台、知识与运行、模型页写死在 Dashboard React 树中；模型页又把
Codex、OpenCode Go 和 OpenAI-compatible 的入口写在固定 Provider 表里。Dashboard 面板虽能从插件
目录加载，浏览器侧却再次扫描源码目录和文件名，没有使用 committed plugin snapshot 作为唯一
发现事实。于是插件后端、页面、导航和面板拥有不同生命周期，删除或新增一种能力要同步修改多处。

现有 v3 runtime 已经拥有普通插件、candidate、exact snapshot、Service、inject、Effect 和 cleanup。
WebUI 不需要第二个插件系统，只需要把浏览器资源发布和父子挂载投影进同一 snapshot。

## 决定

Core Web Host 只发布一个空的 `web.root.v1` 挂载点，并拥有资源校验、exact-snapshot bootstrap、
module 激活、递归 cleanup、通用错误面和 paper token。Core 不拥有品牌顶栏、产品页面、Dashboard
面板或 Provider 名称。

`shell-ui` 是普通插件：它注入 `web.root.v1`，注册唯一 Shell，并声明 `shell.pages.v1`。它拥有
Akashic 顶栏、页面导航、当前 route 和浏览器 history。`conversation-ui`、`workbench-ui`、`runtime-ui`
与 `models` 分别注入 `shell.pages.v1`；导航和页面 renderer 是同一个 entry 的两个投影。

`workbench-ui` 声明 `workbench.panels.v2`。Akasha、Wake 以及外置 Dashboard 面板插件通过同一个
Web module ABI 注入该 mount，不再由 Dashboard 扫目录或依赖 `AkashicDashboard` global。
`models` 声明 `models.connection-types.v1`；Codex、OpenCode Go 和 OpenAI-compatible 各自在自己的
普通插件 artifact 中同时发布 driver 与连接 UI。新增 Provider 或面板只增加插件，不修改 Core、
Shell 或父页面的名称分支。

```text
Core Web Host
└── web.root.v1
    └── shell-ui
        └── shell.pages.v1
            ├── conversation-ui
            ├── workbench-ui
            │   └── workbench.panels.v2
            │       ├── akasha
            │       ├── wake
            │       └── installed panel plugins
            ├── runtime-ui
            └── models
                └── models.connection-types.v1
                    ├── openai-compatible
                    ├── codex
                    └── opencode-go
```

Web module 是插件包级静态 contribution，资源字节和摘要随 candidate snapshot 冻结。Mount 是浏览器
内存中的父子组合点，registration 由 module activation 的 disposer 拥有。两者不是新的持久
generation、权限继承或领域状态。服务端 API 仍由原领域 owner 提供；Host 给 UI 的公共 ABI 只暴露
按当前 module 路由的窄 client，以及把已登记子 entry 的隔离 stylesheet 应用到 child host 的通用
`style(entryId, host)`。同 realm 浏览器代码属于安装时信任，不把这个 client 伪装成安全 sandbox。

`@scope` 只隔离 selector，不隔离 document-global 名字。插件 stylesheet 不得声明 `@font-face`、
`@property`、`@counter-style`、`@layer` 或 `@keyframes`。关键帧动画使用不发布 document-global
名字的 Web Animations API。artifact 边界
集中拒绝违反项，避免一个普通插件覆盖兄弟插件的动画或卸载后改变兄弟视觉。

0018 的共享消息、移动 WebView 和单一对话实现约束继续有效；桌面顶层页面的注册与 adapter 改由
`conversation-ui` 普通插件拥有。Android baseline、Room、outbox、Bridge 和 Mobile WebUI generation
不随本决定迁移。

## 理由

只有 Core 必须先存在，因此它只保留无法插件化的启动原点。顶栏也是产品能力，让它成为普通插件后，
删除或替换 Shell 不要求 Core 认识任何页面。父页面声明自己的子 mount，令布局、面板和 Provider
沿各自变化轴独立演进；所有层仍服从同一套“发布能力、注入能力、Effect cleanup、exact snapshot”
语言。

## 影响

- `frontend/dashboard` 的固定 `ShellView`、导航按钮、runtime 页和 Provider 模板逐步删除。
- 旧 `dashboard_module` 后端 route 在消费者迁完前保留；旧浏览器 panel discovery、运行时编译和 globals
  在最后一个面板迁完后删除。
- 内置与外置插件必须走同一 loader、candidate、catalog、mount 和 cleanup 路径。
- “知识与运行”由 `runtime-ui` 普通插件保留；移除该插件只移除对应页面，不删除底层能力或持久数据。
- 迁移以插件化前页面为视觉和交互金标准；只有旧模型页的 Akasha 独立配置块按产品决定移除。
- Onboarding 以后可以作为普通页面或临时流程注入，不是本次实现前置条件。

## 验收

- [x] Core production path 不包含 conversation、workbench、models、Provider 或面板插件名称。
- [x] 移除 `shell-ui` 只留下通用空组合面；安装另一个 Shell 无需修改 Core。
- [x] 对话、工作台、知识与运行和模型只因各自 Web module 注册而出现。
- [x] Workbench 面板只来自 `workbench.panels.v2`；每次异步读取携带 Host 拥有的 `AbortSignal`，且不存在源码目录扫描或 `AkashicDashboard` global。
- [x] Codex、OpenCode Go 和 OpenAI-compatible 登录入口由各自 module 动态注册；移除插件后入口消失。
- [x] 外置第四 Provider 和外置面板从正式 install 生效，不修改 Core、Shell、models 或 workbench。
- [x] 单次 bootstrap 的 catalog、JS 和 CSS 全部来自一个 exact committed snapshot；失败 candidate 不改变 stable UI。
- [x] Candidate 只验证静态 artifact；浏览器 ABI 由唯一 Host 按 module 隔离，不引入第二套 headless 生命周期。
- [x] 父插件 CSS 沿 child host 向下生效；子 module CSS 只作用于自己的 entry host，不能修改父级或兄弟。
- [x] 父 entry 撤销会递归撤销子 mount；一个 module 失败不使 Shell 或兄弟 entry 白屏。
- [x] UI 迁移、卸载和回滚不改写 Session、模型 Connection、credential、plugin-data 或 Mobile generation。
