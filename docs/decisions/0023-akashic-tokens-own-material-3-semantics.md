# 0023 · Akashic Token 拥有 Material 3 设计语义

- 状态：accepted
- 日期：2026-08-05
- 关联条款：WEBUI-001～WEBUI-007

## 背景

主题统一后，各页面虽然共享浅色、深色和暖纸主题，仍主要使用 `bgSurface`、`actionSoft` 等局部角色和手写控件。Chat、Mobile WebUI、6321、Dashboard 与插件面板会因此分别解释层级、选择和运行状态；只增加圆角或卡片不能形成 Material 3 的颜色语义。

`@material/web` 2.5.0 提供可访问的 Material 3 Web Components，但官方已经声明维护模式。维护状态不会阻止本项目采用成熟组件，也意味着 Akashic 不能把主题所有权交给组件库内部默认值。

## 决定

1. `frontend/theme/src/theme-catalog.json` 是唯一颜色真源。每个主题完整声明 Material 通用角色和 Akashic 领域角色。
2. Theme Runtime 从同一份 Catalog 输出 `--md-sys-color-*`、`--ak-sys-color-*`，并在迁移期输出既有 `--ak-color-*` 兼容别名；旧页面不能反向定义新角色。
3. 按组件逐项引入 `@material/web`，不使用全量 barrel。按钮、筛选 chip、进度和适合 Shadow DOM 的独立控件优先复用成熟组件；需要 Radix 组合、虚拟列表或原生文本行为的现有控件可以保留，但必须消费同一套系统 token。
4. primary 只用于主要动作和关键选择；success、warning、error、trace、info 使用各自前景与容器配对，并同时保留文字或图标提示。界面层级优先由 surface container 与留白表达，不把卡片、胶囊、边框和阴影无差别铺满页面。
5. 这次视觉系统不取得 SessionDB、Room、outbox、配对、插件运行状态、Bridge/API 或 WebUI 发布指针的所有权。

## 目标结构

```text
┌──────────────────── Akashic Theme Catalog ────────────────────┐
│ Material roles                         Akashic domain roles    │
│ primary · tertiary · surface · error   success · warning      │
│                                        trace · info            │
└──────────────────────────┬─────────────────────────────────────┘
                           │ runtime validation + CSS emission
             ┌─────────────┼──────────────────┐
             ▼             ▼                  ▼
      --md-sys-*      --ak-sys-*       --ak-color-* aliases
             │             │                  │ migration only
             └─────────────┼──────────────────┘
                           ▼
       ┌─────────────── UI component language ───────────────┐
       │ 6321 │ Web Chat │ Mobile WebUI │ Dashboard │ Plugins│
       └──────────────────────────────────────────────────────┘
```

## 理由

产品拥有角色映射，才能在组件库停止演进或个别页面保留现有交互 primitive 时继续保持一致。Material 系统角色为通用控件提供成熟语言，Akashic 领域角色则避免把“成功”“等待”“工具轨迹”等不同含义都涂成品牌主色。

## 影响与回滚

- Theme Catalog schema 升级为 v2；构建产物继续附带完整目录，旧 CSS 变量在迁移期保持可用。
- Material Web 只进入前端构建，不写正式 workspace，也不改变服务端或移动 Bridge 协议。
- 回滚到变更前源码即可恢复旧目录和控件；没有数据库、客户端缓存或发布指针迁移。

## 验收

- 每个主题完整声明并校验 Material 与 Akashic 领域角色。
- 6321、Chat/Mobile WebUI、Dashboard 与插件公开控件从同一目录取得语义颜色。
- light、dark、warm-paper 均可构建；Mobile 状态、共享 Chat、typecheck 和 lint 通过。
- 视觉验收覆盖桌面与移动宽度、主要/次要/危险动作、选择、成功、警告、错误和工具轨迹；无法运行浏览器时必须明确保留为未验证项。
