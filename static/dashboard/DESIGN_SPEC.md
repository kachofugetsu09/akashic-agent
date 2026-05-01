# Akashic Dashboard — 设计规范

本文档是仪表盘所有视觉与交互决策的唯一权威来源。
请严格遵循。不要自行发明新值，不要使用内联样式，不要使用 Tailwind 或 Bootstrap 类。

设计基调定位在暖奶油色画布、珊瑚色强调、衬线 display 标题、人文无衬线正文——刻意温暖且具有编辑刊物气质。

---

## 1. 字体

从 Google Fonts 加载——已在 `index.html` 中引入，请勿删除：

```html
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

| 用途 | 字体栈 |
|------|--------|
| Display / 标题 | `var(--display)` → `"Cormorant Garamond", "EB Garamond", "Times New Roman", serif` |
| UI / 正文 | `var(--sans)` → `"DM Sans", "Inter", -apple-system, BlinkMacSystemFont, sans-serif` |
| 代码 / ID / 时间戳 / 数字 | `var(--mono)` → `"JetBrains Mono", "ui-monospace", "SF Mono", monospace` |

**规则：**
- 所有 `font-family` 值必须引用 `var(--sans)`、`var(--mono)` 或 `var(--display)`。严禁在其他地方硬编码字体名称。
- Display 使用 weight 400 + 负字间距。衬线字体永远不加粗。
- 正文段落 weight 400，标签 weight 500。只使用人文无衬线体——不用几何无衬线。

**背景说明：** 衬线 display 是品牌的声音——Cormorant Garamond weight 400 + 负字间距。DM Sans 是人文无衬线体，屏幕渲染清晰，作为正文搭配。

---

## 2. 色彩令牌

所有颜色定义在 `:root` 中。组件 CSS 中永远不要使用原始十六进制值。

```css
:root {
  /* ── 暖奶油色画布 ── */
  --bg:           #faf9f5;   /* canvas — 页面背景，暖奶油色 */
  --bg-soft:      #f5f0e8;   /* surface-soft — 悬停、区域分割 */
  --paper:        #fffaf5;   /* 卡片/面板表面 */
  --paper-strong: #fffdf8;   /* 抬升表面（弹窗） */

  /* ── 边框 — 发丝线色调 ── */
  --line:         #e6dfd8;   /* hairline — 默认边框 */
  --line-strong:  #cdb89d;   /* 悬停/聚焦边框 */
  --line-soft:    rgba(205, 184, 157, 0.15);  /* 细微分割线 */

  /* ── 文字 — 暖墨色 ── */
  --text:         #141413;   /* ink — 接近黑色 */
  --text-soft:    #6c6a64;   /* muted — 次要文字 */

  /* ── 强调色 — 暖珊瑚色 ── */
  --accent:       #cc785c;   /* 主珊瑚色 */
  --accent-soft:  #f0d9cc;   /* 珊瑚色浅色背景 */
  --accent-hover: #a9583e;   /* 珊瑚色按压/悬停变深 */

  /* ── 语义色 ── */
  --green:        #5db872;
  --green-soft:   #e1f0e7;
  --yellow-soft:  #f5edce;
  --red-soft:     #f2d6ce;
  --blue-soft:    #e4ebf5;

  /* ── 层级 — 色块优先，阴影最少 ── */
  --shadow: 0 1px 3px rgba(20, 20, 19, 0.08);

  /* ── 圆角 — 层级化比例 ── */
  --radius:    8px;     /* md — 按钮/输入框/导航项 */
  --radius-lg: 12px;    /* lg — 内容卡片 */
  --radius-xl: 16px;    /* xl — 大型容器 */
}
```

### 语义映射

| 用途 | 令牌 |
|------|------|
| 页面背景 | `var(--bg)` |
| 面板/侧边栏 | `var(--paper)` |
| 输入框/下拉框背景 | `var(--paper)` |
| 弹窗背景 | `var(--paper-strong)` |
| 默认边框 | `1px solid var(--line)` |
| 聚焦/悬停边框 | `var(--line-strong)` |
| 主要文字 | `var(--text)` |
| 次要/占位文字 | `var(--text-soft)` |
| 激活/选中强调 | `var(--accent)` |
| 强调色浅色背景 | `var(--accent-soft)` |

### 角色徽章颜色

| 角色 | bg | text |
|------|----|------|
| `user` | `var(--accent-soft)` | `var(--accent)` |
| `assistant` | `var(--green-soft)` | `var(--green)` |
| `system` | `var(--yellow-soft)` | `#8b6b09` |
| `tool` | `var(--blue-soft)` | `#276489` |

### 渠道徽章颜色（渠道从 `key.split(':')[0]` 推导）

| 渠道 | bg | text |
|------|----|------|
| `telegram` | `var(--blue-soft)` | `#276489` |
| `cli` | `#ece6db` | `var(--text-soft)` |
| `qq` | `#efe0f7` | `#74488d` |
| `scheduler` | `var(--yellow-soft)` | `#8b6b09` |
| 未知 | `#ece6db` | `var(--text-soft)` |

---

## 3. 字体比例

| 类/上下文 | 字号 | 字重 | 字体族 |
|-----------|------|------|--------|
| 默认正文 | 13px | 400 | sans |
| 区域标签（SESSIONS、FIELDS…） | 11px | 600 | sans, 大写, letter-spacing 0.06em |
| 徽章/标签 | 10.5px | 500 | mono |
| 表头 | 11px | 600 | sans, 大写, letter-spacing 0.06em |
| 表格单元格 — session key | 11px | 400 | mono |
| 表格单元格 — 序号/时间戳 | 11.5px | 400 | mono |
| 表格单元格 — 内容预览 | 13px | 400 | sans |
| 详情面板标题 | 22px | 400 | **display**（衬线, -0.3px 字间距） |
| 详情标签 | 11px | 600 | sans, 大写, letter-spacing 0.06em |
| 详情字段值 | 12.5px | 400 | sans |
| 详情字段值（ID/键/时间戳） | 12px | 400 | mono |
| 弹窗标题 | 22px | 400 | **display**（衬线, -0.3px 字间距） |
| JSON 树 | 12px | 400 | mono |
| 按钮 | 12.5px | 500 | sans |

Display 衬线（`var(--display)`）仅用于突出标题：详情面板标题、弹窗标题和品牌标记。其余文字一律使用 `var(--sans)` 或 `var(--mono)`。

---

## 4. 间距

基础单元：**4px**。

### 仪表盘间距值

| 用途 | px |
|------|----|
| 图标间距、徽章内边距 | 4px |
| 徽章内部间距、小内边距 | 6px |
| 项之间默认间距 | 8px |
| 标准水平内边距 | 10px |
| 面板内边距 | 12px |
| 窗格内边距、表单元格内边距 | 14px |
| 区块外边距、工作区内边距 | 16px |
| 弹窗上下内边距 | 20px |
| 弹窗内边距 | 24px |

布局中永远不要使用 7px、9px、11px 等奇数。仅在微调字体/行高时可以例外。

### 间距参考

| 令牌 | px |
|------|----|
| `xxs` | 4 |
| `xs` | 8 |
| `sm` | 12 |
| `md` | 16 |
| `lg` | 24 |
| `xl` | 32 |
| `xxl` | 48 |
| `section` | 96 |

---

## 5. 布局

### 外壳

```
┌──────────────────────────────────────────────────┐  height: 48px  topbar
├──────────────────────────────────────────────────┤
│  sessions-pane (256px) │ messages-pane (flex:1) │ detail-pane (400px) │
│                        │                        │                      │
│                        │                        │                      │
└──────────────────────────────────────────────────┘
```

```css
.shell         { height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
.topbar        { height: 48px; flex-shrink: 0; }
.workspace     { flex: 1; display: flex; overflow: hidden; min-height: 0; }
.sessions-pane { width: 256px; flex-shrink: 0; }
.messages-pane { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.detail-pane   { width: 400px; flex-shrink: 0; }
```

**规则：** 工作区使用 `flex`，不要用 CSS grid。详情面板可以在未选中时隐藏（`display:none`），grid 模式会使这一操作更难。

### 顶栏

- 高度：`48px`
- 背景色：`var(--paper)`
- 边框：`border-bottom: 1px solid var(--line)`
- 内边距：`0 16px`
- 品牌在左侧（固定宽度 ~212px），筛选器在中间（`flex: 1`），可选操作按钮在右侧。

### 会话面板（左侧边栏）

- 背景色：`var(--paper)`
- `border-right: 1px solid var(--line)`
- 头部：`padding: 12px 12px 8px`，`border-bottom: 1px solid var(--line)`
- 列表可滚动：`overflow-y: auto`，`padding: 4px 0 12px`
- 会话项：`margin: 1px 6px`，`border-radius: var(--radius)`，`border: 1px solid transparent`

### 消息面板（中央表格）

表格列模板：`34px 80px 60px 1fr 90px 72px 60px`
— 复选框 | session-key | 序号 | 内容 | 时间戳 | 角色 | 操作

- 表头：`min-height: 36px`，`background: var(--bg-soft)`，sticky 顶部
- 表行：`min-height: 44px`，`border-bottom: 1px solid var(--line-soft)`，cursor pointer
- 表底：`border-top: 1px solid var(--line)`，`background: var(--bg-soft)`

### 详情面板（右侧）

- `border-left: 1px solid var(--line)`
- 头部：`padding: 12px 14px`，`border-bottom: 1px solid var(--line)`，flex 行 + 关闭按钮
- 内容体：可滚动，`padding: 16px`

---

## 6. 组件

### 6.1 按钮

仅三种变体。没有其他。

```css
/* Primary — 用于弹窗中的主要 CTA，或创建操作 */
.btn-primary {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  padding: 6px 13px;
  font-size: 12.5px;
  font-weight: 500;
  cursor: pointer;
}
.btn-primary:hover { background: var(--accent-hover); }

/* Ghost — 次要操作、取消、翻页 */
.btn-ghost {
  background: transparent;
  color: var(--text-soft);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 6px 13px;
  font-size: 12.5px;
  font-weight: 500;
  cursor: pointer;
}
.btn-ghost:hover { background: var(--bg-soft); border-color: var(--line-strong); color: var(--text); }

/* Ghost danger — 破坏性次要操作（删除） */
.btn-ghost.danger {
  color: #b03a3a;
  border-color: var(--line);
}
.btn-ghost.danger:hover { border-color: #b03a3a; background: var(--red-soft); }
```

**规则：** 列表或表格中永远不要使用带背景色的危险按钮——只使用 ghost-danger。`btn-primary` 仅用于确认和创建操作。

仅图标按钮（编辑 ✎、删除 ✕ 在表格行中）：

```css
.icon-btn {
  background: none;
  border: none;
  padding: 2px 4px;
  font-size: 13px;
  color: var(--text-soft);
  border-radius: var(--radius);
  cursor: pointer;
}
.icon-btn:hover { color: var(--text); background: var(--line-soft); }
.icon-btn.danger:hover { color: #b03a3a; }
```

### 6.2 输入框

```css
input[type="text"],
input[type="number"],
textarea {
  border: 1px solid var(--line);
  background: var(--paper);
  color: var(--text);
  border-radius: var(--radius);
  padding: 6px 12px;
  font-family: var(--sans);
  font-size: 12.5px;
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
}
input:focus, textarea:focus {
  border-color: var(--line-strong);
  box-shadow: 0 0 0 3px rgba(204, 120, 92, 0.1);
}
```

带图标的搜索输入框——用 `.search` div 包裹，图标绝对定位 `left: 10px`，输入框 `padding-left: 28px`。

### 6.3 下拉框

面向用户的筛选器可以使用 React 受控 `<select>`，但必须沿用 `styles.css` 中的统一 select 样式，不要引入新的下拉组件或内联样式。

基础样式：

```css
select {
  border: 1px solid var(--line);
  background: var(--paper);
  border-radius: var(--radius);
  padding: 6px 28px 6px 12px;
  font-size: 12.5px;
  /* SVG 箭头通过 background-image 实现，right 10px center */
}
.custom-select.open .cs-trigger {
  border-color: var(--line-strong);
  box-shadow: 0 0 0 3px rgba(204, 120, 92, 0.1);
}
```

下拉选项：

```css
.cs-dropdown {
  position: absolute; top: calc(100% + 4px); left: 0;
  background: var(--paper-strong);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  z-index: 200;
  animation: cs-drop-in 0.1s ease;
}
.cs-option { padding: 8px 12px; font-size: 12.5px; cursor: pointer; }
.cs-option:hover { background: var(--bg-soft); }
.cs-option.active { color: var(--accent); background: var(--accent-soft); font-weight: 500; }
```

### 6.4 徽章 / 标签

```css
.badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 10.5px;
  font-weight: 500;
  font-family: var(--mono);
  white-space: nowrap;
}
```

使用内联 `style` 或修饰符类来设置 §2 表格中的 bg/color 组合。

### 6.5 活跃会话标签（顶栏筛选指示器）

```css
.active-session-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  background: var(--accent-soft);
  border: 1px solid rgba(204, 120, 92, 0.2);
  border-radius: var(--radius);
  font-size: 12px;
  max-width: 280px;
}
.active-session-chip code { font-family: var(--mono); color: var(--accent); font-weight: 500; }
.active-session-chip button { background: none; border: none; color: var(--text-soft); cursor: pointer; }
.active-session-chip button:hover { color: var(--accent); }
```

### 6.6 会话列表项

结构（两行）：
```
row1: [渠道徽章] [:uid-part]  [计数标签]
row2: [相对时间]              [✎ ✕ — 仅在悬停时显示]
```

激活态：`background: var(--accent-soft); border-color: rgba(204, 120, 92, 0.2);`
悬停态：`background: var(--bg-soft);`
默认态：`background: transparent; border: 1px solid transparent;`

操作按钮默认 `opacity: 0`，`.session-item:hover` 时 `opacity: 1`。不要永久显示——会显得杂乱。

```css
.session-actions { opacity: 0; transition: opacity 0.12s; }
.session-item:hover .session-actions { opacity: 1; }
```

### 6.7 表格行状态

| 状态 | CSS |
|------|-----|
| 默认 | 透明背景 |
| 悬停 | `background: var(--bg-soft)` |
| 激活（详情打开） | `background: var(--accent-soft); box-shadow: inset 2px 0 0 var(--accent)` |
| 选中（复选框） | `background: var(--yellow-soft)` |
| 激活+悬停 | 保持激活背景——不要在悬停时回退 |

### 6.8 批量操作栏

仅当 `selectedCount > 0` 时出现在表格上方。使用强调色浅色背景：

```css
.batch-bar {
  padding: 8px 16px;
  background: var(--accent-soft);
  border-bottom: 1px solid rgba(204, 120, 92, 0.2);
  display: flex; align-items: center; gap: 12px;
}
.batch-bar span { font-size: 12.5px; color: var(--accent); font-weight: 500; }
```

### 6.9 详情面板区块

详情面板中的每个部分使用此模式：

```html
<div class="detail-block">
  <div class="detail-label">区块标题</div>
  <!-- 内容 -->
</div>
```

```css
.detail-block { margin-bottom: 16px; }
.detail-label {
  font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-soft); margin-bottom: 6px;
}
```

内容框（用于消息内容字段——支持 markdown）：

```css
.detail-content {
  padding: 12px 16px;
  background: var(--bg-soft);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  font-size: 13px; line-height: 1.8;
}
```

字段键值网格（用于元数据字段）：

```css
.detail-row {
  display: flex; gap: 12px;
  padding: 8px 0; border-bottom: 1px solid var(--line);
}
.detail-row-label { width: 120px; flex-shrink: 0; font-size: 12px; color: var(--text-soft); }
.detail-row-val   { flex: 1; font-size: 12.5px; word-break: break-all; }
```

### 6.10 JSON 树查看器

**永远不要使用 `<pre>` 显示 JSON 数据。** 使用 `app.js` 中的 `makeJsonViewer(data)` DOM 函数，它返回可折叠的树节点。它：
- 自动解析嵌套的 JSON 字符串（双编码的 `result` 字段等）
- 默认折叠深度 ≥ 3 的对象/数组（深度 0–2 展开）
- 颜色：字符串绿色，数字强调色，布尔蓝色，null 灰色

在 `innerHTML` 模板中使用时，使用 `jvPlaceholder(data)` 辅助函数，它生成 `<div data-jv="...">` 标记，然后在设置 `innerHTML` 后调用 `attachJsonViewers(container)`。

```css
.json-tree {
  padding: 12px 16px;
  background: var(--bg-soft);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  font-family: var(--mono); font-size: 12px; line-height: 1.75;
  overflow-x: auto; word-break: break-all;
}
.jt-children { padding-left: 16px; border-left: 1px solid var(--line-soft); margin-left: 4px; }
.jt-str  { color: #2f7d62; }
.jt-num  { color: var(--accent); }
.jt-bool { color: #276489; }
.jt-null { color: var(--text-soft); font-style: italic; }
```

### 6.11 弹窗

```css
.modal-backdrop {
  position: fixed; inset: 0;
  background: rgba(20, 20, 19, 0.25);
  backdrop-filter: blur(4px);
  z-index: 400;
}
.modal {
  position: fixed; left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  width: min(500px, calc(100vw - 24px));
  max-height: calc(100vh - 40px);
  overflow-y: auto;
  background: var(--paper-strong);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  box-shadow: 0 4px 24px rgba(20, 20, 19, 0.12);
  padding: 24px;
  z-index: 500;
}
```

弹窗内容顺序：标题（衬线）→ 副标题（灰色，13px）→ 表单网格 → 操作按钮（右对齐）。

### 6.12 空状态

```css
.empty-state {
  padding: 48px 16px;
  text-align: center;
  color: var(--text-soft);
  font-size: 13px;
}
```

---

## 7. 交互规则

| 触发 | 效果 |
|------|------|
| `input:focus` / `select:focus` | `border-color: var(--line-strong)` + `box-shadow: 0 0 0 3px rgba(204,120,92,0.1)` |
| 按钮悬停 | 见各变体规则。Primary 按压变为 `--accent-hover` |
| 行悬停 | `background: var(--bg-soft)`——可点击行永远不要使用 `cursor: default` |
| 激活行 | 强调色左边框 `box-shadow: inset 2px 0 0 var(--accent)` |
| 自定义下拉框展开 | 触发器获得聚焦光环 |
| 会话项悬停 | 通过 `opacity` 过渡显示编辑/删除图标 |

过渡时长：背景/颜色 `0.1s–0.15s ease`。微交互永远不要超过 `0.2s`。

---

## 8. 滚动条样式

```css
::-webkit-scrollbar        { width: 6px; height: 6px; }
::-webkit-scrollbar-track  { background: transparent; }
::-webkit-scrollbar-thumb  { background: rgba(205, 184, 157, 0.4); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(205, 184, 157, 0.6); }
```

Firefox：`scrollbar-width: thin; scrollbar-color: rgba(205, 184, 157, 0.4) transparent;`

---

## 9. 反模式——不要这样做

| 错误做法 | 正确做法 |
|---------|---------|
| 下拉框引入新组件或内联样式 | 使用现有 select 样式 |
| JSON 数据使用未样式化 `<pre>` 块 | 使用 `.json-tree` 样式 |
| 表格预览中显示原始 markdown 语法 | 先 `stripMarkdown()` |
| 工作区使用 `display: grid` | 使用 `display: flex` |
| 组件 CSS 中硬编码十六进制颜色 | 使用 `var(--token)` |
| 表中始终显示三列 | Session-key 列在会话激活时可隐藏 |
| `font-family: monospace` | `font-family: var(--mono)` |
| 错误消息使用 `alert()` | 弹窗/内联错误提示 |
| 顶栏下拉框 `width: 100%` | `width: auto`（由内容决定尺寸） |
| 会话列表中始终显示编辑/删除按钮 | 仅 `:hover` 时通过 `opacity` 显示 |
| `background: blue` 或 OS 原生下拉选项 | 自定义暖色调下拉菜单 |
| 胶囊/徽章 `border-radius: 4px` | `border-radius: 999px` |
| `font-weight: bold` | `font-weight: 600` 或 `500`——不要使用 `bold` 关键字 |
| `z-index: 9999` | 弹窗 `500`，下拉框 `200`，顶栏 `3` |
| 画布使用冷灰色或纯白色 | 暖奶油色 `#faf9f5` |
| 衬线 display 加粗 | Display 衬线保持 weight 400 |
| 强调色使用冷蓝色或饱和青色 | 珊瑚色 `#cc785c` 是品牌电压 |
| 页面背景使用渐变 | 纯色奶油画布（色块优先） |

---

## 10. 文件结构

```
static/dashboard/
  index.html        — 页面骨架 HTML，不含逻辑，不含内联样式
  styles.css        — 全部样式，:root 中定义变量
  app.js            — 全部 JS：状态、API 调用、渲染函数、工具函数
  DESIGN_SPEC.md    — 本文件
```

**规则：** 永远不要添加第四个 JS 文件。所有代码保留在 `app.js` 中。超过 ~1200 行时才考虑拆分。

---

## 11. 页面背景

```css
body {
  background: var(--bg);  /* 纯色奶油画布 — #faf9f5 */
}
```

仪表盘遵循**色块优先**的层级哲学：深度来自表面对比（画布 vs 卡片 vs 浅色区域），而不是装饰性背景或阴影。奶油色画布是默认基底——没有径向渐变，没有背景噪点。
