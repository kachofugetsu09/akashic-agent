# 2026-07-17 会话级输入与引用草稿

## 目标

手机上的输入不是 WebView 临时状态，而是当前电脑、当前会话拥有的本地工作：切换会话、应用进后台或进程重启后，未发送文字和引用目标都应回到原位置；发送失败保留，服务端确认接收后才清除。

```text
会话 A 的文字 / 引用 ─┐
会话 B 的文字 / 引用 ─┼─► Room（server + session）─► 当前 composer
WebView 隐藏 / 进程重启 ┘                         └─ 发送成功才清除
```

## 设计与所有权

- Room schema v7 新增 `composer_drafts`，一条会话最多一份文字与引用目标；外键跟随电脑和会话删除，引用目标删除时置空，不把消失的消息伪装成仍可引用。
- Android `LocalDeliveryStore` 是唯一写入 owner：在 WebView 边界限制文字和 ID，校验电脑、会话与引用归属，并与阅读锚点、canonical identity 迁移共用同一串行写序。
- React 只持有当前渲染副本。输入以 250 ms debounce 写回；切会话、页面隐藏和卸载前立即 flush。原生 snapshot 是恢复和发送确认后的最终 owner。
- 切会话时 Android 只有在 `serverId + sessionId` 与本地 composer 投影一致后才发 snapshot，避免 `combine` 把 A 的最新草稿临时配给 B。附件草稿同时获得相同隔离。
- 消息图、会话目录、附件与 composer 在同一个 `sessionState.flatMapLatest` 投影内完成首帧后再整体发布；不会出现 B 草稿先到、B 消息图仍是 A 而被过滤为空的中间帧。
- 发送请求捕获当时的会话、文字和引用。拒绝时不清；接受时仅在当前草稿仍等于已发送副本时清除，因此等待确认期间继续输入或切换会话不会丢字。
- optimistic / streaming 消息经历一段或两段 canonical ID 迁移时，引用草稿与阅读锚点一起移动；迟到的 WebView 写入也先解析 alias。
- 已从电脑删除但仍有 composer 草稿的会话属于“有本地工作”，不能被本机直接移除。

## Material 3 与 ExtraGram 取舍

本组复用 ExtraGram 已验证的“草稿属于对话、回复上下文与文字一起恢复”交互语义，没有复制 Telegram 的页面结构。现有输入栏和回复 state layer 已经表达任务，因此没有新增卡片、徽章、颜色或提示：恢复发生在原位置，视觉上只出现用户本来就在编辑的内容。

五项 UI 检查结果：`better-ui` 确认切换与确认时无跳变；`better-colors` 不新增无语义颜色；`better-typography` 复用现有输入与引用层级；`material-3` 保持 composer/回复 state layer；`kill-ai-slop` 确认没有新增卡片、胶囊、渐变、阴影或装饰文案。

## 自动验证

- 移动 Web 状态：`25 passed`，覆盖会话捕获、缺失引用清理、发送接受/拒绝与确认期间继续输入。
- `npm run typecheck`、`npm run lint -- --max-warnings=0`、`npm run build:mobile-web`：通过。
- Android JVM：`:app:testDebugUnitTest` 通过。
- Pixel 7 Room instrumentation：迁移、草稿 DAO 与完整 `LocalDeliveryStoreTest` 共 `47 passed`。首轮真实发现旧 ID 写入未跟随两段 canonical alias，修复后整组复跑通过。
- release 门禁：`testReleaseUnitTest lintRelease assembleRelease`、R8 与 APK v2 签名通过；最终验收 APK SHA-256 为 `a91d63b8718c57656a1356ca56ebb69f642103974869f1ad88950398b6266fcf`。

## Pixel 7 / 隔离 Mobile Lab

设备 `28151FDH200478` 无损覆盖同签名 release；没有卸载或清数据，正式 workspace 未访问。由于正式包不可调试，Android 安全边界拒绝 `run-as` 导出私有数据库，因此安装前无法生成应用数据备份；覆盖安装保留原配对与 Room 数据。

1. 首轮 A 写入后切 B，真机暴露 B 短暂收到 A 草稿。根因是 Android `combine` 在 `sessionState` 已切 B、Room 子流仍保留 A 最新值时发出混合 snapshot；现在按 `(serverId, sessionId)` 抑制不一致投影。
2. 修复后 A 与 B 显示各自不同草稿；回到 A 的截图为 `/tmp/pixel7-draft-return-a.png`，B 在 A 发送后仍保留自己的截图为 `/tmp/pixel7-draft-b-preserved-after-a-send.png`。
3. A 的文字与“回复 Akashic · OK”经 force-stop / 冷启动完整恢复，截图为 `/tmp/pixel7-draft-reply-restart.png`。
4. A 真实发送 `reply only DRAFT_OK`，隔离 Agent 回复 `DRAFT_OK`；服务端接受后文字与引用同时清空，B 草稿不受影响。截图为 `/tmp/pixel7-draft-after-send2.png`。
5. 独立 Review 要求补测“发送后立即切会话”：A 发出 `reply only SWITCH_OK` 后在 accepted 前切到 B，B 保持空 composer；返回 A 后已发送草稿没有复活，真实回复为 `SWITCH_OK`。截图为 `/tmp/pixel7-draft-accepted-while-b.png`、`/tmp/pixel7-draft-accepted-switch-return-a.png`。
6. WebView 入口使用 `?appVersion=20` 并设置 `LOAD_NO_CACHE`；覆盖安装后 logcat 实际加载带版本入口，旧协议 bundle 不会从固定 URL 缓存复用。
7. 最终 logcat 无 FATAL、RenderProcessGone、Room migration、event sequence gap 或协议校验错误。

## 独立 Review

- High：发送成功前切到 B 时，原逻辑只 flush B 而不清 A。现在 accepted 始终清理 sent-session 的持久化副本；只有仍在 A 且已继续编辑时才保留新内容。状态测试与上述 Pixel 7 真实回合共同覆盖。
- High：独立的 Room 子流可能组成“B composer + A message graph”中间帧，进而把有效引用误判为消失。现在同一 session projection 原子切换，并在真机 A/B 往返中复测。
- Medium：strict snapshot v5 配合固定 appassets URL 可能复用旧 bundle。现在入口按 app version cache-bust，WebView 禁止读取缓存；单元测试覆盖版本 URL 与导航白名单。
- 修复后复核无剩余 Blocker、High 或 Medium；Room schema、DAO、canonical 两段迁移与跨会话 fail-loud 边界未发现额外问题。

## 不扩大的边界

- 草稿只在 Android 本机持久化，不新增 Agent 核心、Gateway 消息、云端草稿同步或多设备冲突协议。
- 没有为草稿增加第二套附件模型；附件继续使用既有 `attachment_transfers`，只统一其切会话投影隔离。
- 没有通过默认值或静默 fallback 掩盖跨电脑、跨会话引用；这些内部契约仍 fail-fast、fail-loud。
