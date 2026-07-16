# 工具详情结构化复制

## 任务与边界

Agent IM 的工具详情已经能展示脱敏参数、结果和单次耗时，但日用时只能长按选择局部文本。首版只补齐“把现有安全投影稳定带走”的动作，不新增服务端协议、存储、完整原始参数拉取或工具重放。

```text
工具节点
└─ 展开详情（既有 surface）
   ├─ 参数                         [复制]
   │  command  printf toolcopyresult
   └─ 结果                         [复制]
      toolcopyresult
```

## 设计

### Better UI

| Before | After |
|---|---|
| 依赖长按选择，参数与结果没有稳定动作 | 标题行附着 48dp 文字动作，操作对象与按钮在同一视觉平面 |
| 复制后没有反馈 | 原位切换 check +“已复制”，1.6 秒后恢复，不弹 toast 或成功卡片 |
| 折叠动画只隐藏视觉内容 | 折叠面同时 `aria-hidden + inert`，不可见按钮不进入焦点树 |

### Better Colors

没有新增颜色。动作、焦点和按下 state layer 复用 `--m-primary`；参数、结果和标题继续使用既有 surface / on-surface-variant，紫色仍只承担 Agent 过程状态。

### Better Typography

| Before | After |
|---|---|
| 参数/结果只有 11px 分区标题 | 标题保持低强调层级，旁边增加 12px 中等字重动作；不与工具名争夺主次 |
| 值使用等宽字体 | 继续只让参数值和结果使用等宽字体，动作与标题保持系统正文字体 |

### Kill AI Slop

- 不新增外层卡片、弹窗、彩色图标方块、渐变或阴影。
- 完整胶囊只用于有明确组件类别的 Material text button，不把标签、状态和正文都做成胶囊。
- 扫描前后均为 38 个文件、9 组、57 个命中；差异仅是新增 CSS 推动既有命中行号。

## 实现所有权

- `message-view.tsx`：决定安全投影的复制文本、分区动作与流式成功状态。
- `mobile-native.tsx`：只把现有 `AkashicNative.copyText` 注入 shared message view。
- `mobile-native.css`：拥有 48dp 触控、primary state layer、焦点和 reduced-motion。
- Android、Gateway、Room 和插件协议无改动。

## 验证

### 自动门禁

- `npm run typecheck`
- `npm run lint`
- `npm run test:mobile-web-state`：15/15
- `npm run build:mobile-web`
- `git diff --check`
- `clients/android/scripts/build-release.sh`：release unit、Lint、R8、assemble、v2 signature

### Pixel 7 / Mobile Lab

- workspace：`docker/debug/profiles/mobile-lab/workspace`，与正式 workspace 隔离。
- 真机输入：`use shell tool to run printf toolcopyresult and report`。
- `mobile_command_receipts` 新增 `message.send completed`，session 为 `mobile:74cf8e16-a7ab-4077-b58e-b057480b91ac`。
- 时间线真实出现 thinking → `shell 完成 · 9ms` → thinking → 最终 `toolcopyresult`；截图 `/tmp/pixel7-send-enter2.png`。
- 完成后折叠为“已思考 5s”，最终回答与 token 尾注位置正常；截图 `/tmp/pixel7-tool-copy-final-collapsed.png`。
- review 修复后重新构建并无损安装 release APK；应用 PID 日志无 FATAL、RenderProcessGone、event gap 或协议错误。

### 尚未伪装为通过的项目

release WebView 不开放调试；ADB TAB 会把焦点带出 WebView，曾误启动系统天气应用。没有把这条自动化副作用记成 Akashic 崩溃，也没有声称已经自动完成“点击复制后粘贴比对”。既有消息复制已验证同一原生 bridge；下一轮先建立可复用、不会越出应用的 release WebView 交互驱动，再把参数与结果剪贴板往返纳入固定真机门禁。

## 独立 Review

- 无 blocker。
- 修复折叠按钮仍可聚焦：disclosure 增加 `inert`。
- 修复流式结果更新后旧剪贴板仍显示成功：成功状态绑定 `{ section, text }`。
- 触控高度从 44dp 提升到 Material 3 的 48dp 基线。
- shared WebChat 与 Android bridge 边界、折叠态序列化成本、颜色和形状语义通过审阅。
