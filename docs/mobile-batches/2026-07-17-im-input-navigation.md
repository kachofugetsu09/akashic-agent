# 移动端日常 IM 输入与导航批次

## 范围

本批只补齐高频会话操作，不修改 Agent 核心协议：

- 点击用户或 Akashic 消息中的引用，跳回当前会话里的原消息并短暂高亮。
- 原消息不在当前投影时，在引用原位显示 1.8 秒“原消息不在当前记录中”，不跳会话、不弹新卡片。
- 输入框从 1 行随内容长到 6 行；按 16 px 字号、1.5 行高和 20 px 纵向 padding 计算，超过 164 px 后只滚动输入框内部。
- Gboard 普通回车只换行，IME 组合期间不发送；桌面键盘仅 `Ctrl/Meta + Enter` 发送。
- 消息选择模式在现有复制、引用旁增加 Android 系统文本分享，共用同一份会话文本格式。

## 交互与状态所有权

```text
当前消息中的引用
       │ 点按
       ▼
只在当前 snapshot.messages 解析目标
       ├── 命中 ──> 复用 jumpToMessage ──> 居中、高亮并播报身份与时间
       └── 缺失 ──> 引用原位显示轻反馈 ──> 1.8 秒后恢复预览

输入 / 草稿恢复 / 宽度变化
       │
       ▼
测量 textarea.scrollHeight ──> 44…164 px ──> 超过上限内部滚动
       │
       └── composer ResizeObserver ──> 会话底部留出真实输入区高度

长按消息进入选择模式 ──> 现有文本 formatter
       ├── UTF-8 ≤ 64 KiB ──> Android ACTION_SEND 文本
       └── UTF-8 > 64 KiB ──> FileProvider 临时 .txt
                                │
                                └── chooser 成功启动后才退出选择模式
```

引用目标由当前消息投影拥有，跨会话和已清理历史不会被猜测解析。输入草稿仍由既有 Android owner 持久化；Web 只根据真实内容高度呈现。系统分享是终端动作，不增加服务端消息类型；原生以 request/result 明确确认 chooser 是否成功启动，pending 期间冻结整个选择操作组，失败时恢复操作并保留选择供重试。

## Material 3 设计决策

- 引用预览继续是消息正文里的结构层，不增加圆角卡片或阴影；主色只表示“可导航引用”和“目标缺失”状态。
- 分享沿用顶部选择模式的 44 px 圆形触控目标，与复制、引用处在同一操作层级。
- 输入框沿用现有 composer surface；高度变化表达内容容量，不用装饰性动画。
- 字号、行高和项目现有字体栈不变。输入保持 16 px，避免 Android WebView 输入缩放；引用作者与摘要继续使用既有紧凑层级。
- `kill-ai-slop` 扫描没有发现本批新增命中。仓库已有扫描项位于 showcase、通用 AI elements、代码字体和真实圆形按钮，本批不扩大范围处理。

## 自动验证

| 能力 | 验证 | 结果 |
| --- | --- | --- |
| 引用只解析当前投影，覆盖用户/助手/缺失目标 | `mobile-message-state.test.mjs` | 通过 |
| 回车、快捷发送、IME 组合态 | `mobile-message-state.test.mjs` | 通过 |
| 1–6 行高度与超限滚动 | `mobile-message-state.test.mjs` | 通过 |
| Android 文本 chooser payload、64 KiB UTF-8 边界 | `MobileTextShareTest` | 通过 |
| TypeScript / ESLint / mobile bundle | 定向前端门禁 | 通过 |

新 worktree 需要本机 `clients/android/local.properties` 和根 `package-lock.json` 才能进入 Android 编译；两者都是未跟踪环境文件，不属于本批提交。

## Pixel 7 验收清单

1. 分别点按引用用户消息和 Akashic 消息的引用条；原消息进入视口并短暂高亮，TalkBack 明确播报“已跳到你/Akashic HH:MM 的消息”。
2. 在历史不完整的引用上点按；只在引用原位看到“原消息不在当前记录中”，会话位置不跳动。
3. 用 Gboard 连续输入 1、3、6、7 行；前 6 行 composer 向上生长且最后一条消息不被遮挡，第 7 行开始输入框内部滚动。
4. 普通回车产生换行；中文联想上屏不发送；外接键盘 `Ctrl/Meta + Enter` 只发送一次。
5. 长按后选择一条及多条消息，点击分享；系统 chooser 出现后选择模式关闭，分享文本与复制文本格式一致。制造 chooser 无法启动的场景时，选择仍保留并显示“分享未打开，请重试”。另以超过 64 KiB 的文本确认 chooser 收到 `.txt` 文件而非发生 Binder 崩溃；过期分享文件在下一次分享或页面启动时清理。
6. 切换会话并返回，文字与引用草稿恢复后输入框高度立即匹配内容；横竖屏切换后不截断文本。

真机验收需在汇总分支构建 APK 后执行，本 worktree 不并行运行完整 Release 构建。
