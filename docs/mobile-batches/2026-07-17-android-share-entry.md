# Android 系统分享入口

## 目标

从浏览器、相册和文件管理器分享文字或文件到 Akashic 时，内容进入当前 mobile 会话的既有 composer，由用户确认后再发送。移动端不自动触发 Agent，不新增消息协议，也不复制附件上传实现。

```text
Android ACTION_SEND / ACTION_SEND_MULTIPLE
              │ 外部信任边界
              ▼
     解析文字与 content URI
              │
              ├─ 一次绑定当前 session owner
              │
              ├─ 文字 ──► 等待目标 Web composer owner
              │              │ 持久化提交成功后消费
              │              ▼
              │         目标会话 Room 草稿提交成功
              │
              └─ 文件 ──► 同一目标的 AttachmentDraftStore
                             │ 私有复制、SHA-256、大小限制
                             ▼
                        上传进度与失败恢复
```

## ExtraGram 与 Akashic 取舍

参考 ExtraGram `LaunchActivity` 的成熟边界：忽略从任务历史恢复的旧分享、把 URL 与 subject 合成可读文本、区分 `SEND` / `SEND_MULTIPLE`、在发送前保留用户确认。Akashic 首版不移植联系人、贴纸、媒体编辑和多收件人路由；它只把内容放入当前 Agent 会话的输入任务面，继续复用现有会话抽屉切换和发送确认。

## 状态所有权

- `MainActivity` 只拥有 Android Intent 边界和一次性入口；`file://` 被拒绝，畸形 Parcelable 显式提示失败。文字与外部 URI 先进入原生持久接收层，文件在授权期内复制到私有 staging，提交后才清除 Intent action。
- `MainViewModel` 按 Intent ID 排队，并在首次可处理时一次绑定当前 session。文字先提交、附件随后导入，两部分不会在快速切会话时拆开；连续分享也不会把已经消费的旧文字重新追加。
- WebView 可以早于首个 Room snapshot ready，但不能提前确认文字已消费。JavaScript 只在目标会话 composer owner 就绪后合并并保留现有引用，原生 Room 真正提交成功才消费队列；写入失败保留原分享。
- 文件继续由 `AttachmentDraftStore` 复制到私有目录；provider 没有 `DISPLAY_NAME` 时才按 MIME type 生成 `shared-<ULID>.<ext>`，不从 URI 猜路径。
- 文件只有在私有复制和 Room 提交成功后才出队；容量、权限或 IO 失败使用 Material 3 Snackbar 提供“重试 / 放弃”，不会伪装成功。草稿空间不足也不截断共享文字，而是保留并要求用户精简后重试。
- 进程重建从 `IncomingShareStore` 恢复 FIFO、目标 session 与私有 staging；文字和文件分别成功后才清理对应部分。重试会先切回已绑定会话；目标已不存在时保留错误并要求放弃，不会静默遮蔽后续分享。
- Intent 接收 ID 在外部边界按规范 UUID 校验；同一次 launch 在 durable enqueue 前后重放时，由“队尾、未 claim、未 prepared、10 秒内”的 SHA-256 receipt 去重。超过短窗口的同内容主动分享仍会形成新任务。
- 文字在写 Room 前冻结最终合并值与草稿基线；恢复时 compare、canonical reply 解析和 write 在 `LocalDeliveryStore` 同一 mutex + Room transaction 内完成。较新的用户编辑不会被旧 prepared 值覆盖。
- 文件在接收时生成稳定 attachment ULID；Room 已完整提交但 receipt 尚未消费时，恢复会识别同一批 ID，而不是复制第二组附件。
- 发送、上传、重试、metered network 和会话失效规则仍由既有 owner 决定；本批没有第二套队列。

## 五技能设计复核

- Better UI：分享完成后直接回到对话和现有 composer，不弹新页面、不增加确认卡；只有真实导入失败才用短暂 Snackbar 提供恢复动作，用户仍需点击发送。
- Better Colors：没有新增颜色。上传进度、连接、错误和 Agent 过程继续使用既有语义映射。
- Better Typography：共享文字保持 16 px composer 正文；文件名继续使用现有附件行层级。
- Material 3：复用既有输入 surface、附件行、进度条和系统 Toast；形状与触控目标不变。
- Kill AI Slop：本批没有新增卡片、胶囊、渐变、阴影、发光点或装饰 icon tile。

## 自动验收

1. `normalizeSharedText` 覆盖 URL + subject、纯文字、空内容和 65,536 字符边界。
2. Intent parser 合并并去重 ClipData / `EXTRA_STREAM`，拒绝 `file://`，忽略从任务历史恢复的旧 Intent。
3. 共享文字追加而不重写已有草稿，保持引用 ID，并受既有 65,536 字符持久化上限约束。
4. Android debug unit 与 androidTest Kotlin 完整编译；release unit、Lint、R8、assemble 和 v2 签名通过。
5. Web mobile state、TypeScript、ESLint 和生产 bundle 通过。

实际门禁结果：

- `node --test frontend/chat/src/mobile-message-state.test.mjs`：`35/35`。
- `npm run typecheck`、`npm run lint`：通过。
- `:app:testDebugUnitTest :app:compileDebugAndroidTestKotlin -x buildMobileWeb`：通过，共执行 40 个 Gradle task。
- release unit、Lint、R8、assemble 与 v2 签名：通过；最终验收 APK 为 `0.8.0 (21)`，SHA-256 `11f288291621eb94ecd5decdc84172e45e230317c6dcb3ee14b15e7a0317c0da`。
- Pixel 7 debug instrumentation：分享接收、附件草稿和本地投影共 `54/54`；覆盖伪造 ID 拒绝、Intent 重放去重、Store 重建、稳定附件 ID、prepared CAS 防覆盖与 missing reply 等价解析。

## Pixel 7 验收

1. 从 Chrome 分享“标题 + URL”：Akashic 回到当前会话，文字出现在 composer，未自动发送；强停并重开后文字仍在。最终持久接收层验收标记为 `DURABLE_SHARE_FINAL_1640`，重启证据 `/tmp/pixel7-durable-share-final.png`。
2. 在已有文字和引用草稿时再分享文字：原草稿在前、共享文字在后，引用条不消失。
3. 从 Android 系统分享面板分享 `akashic-share-file-final.txt`：Akashic 成为真实系统分享目标，文件先进入私有 staging，随后进入原有附件队列并显示真实文件名与“上传完成”，可以从原位移除；最终截图为 `/tmp/pixel7-durable-share-file-final.png`。
4. 连续触发两个文字分享 Intent：队列测试与真机草稿确认两段文字各追加一次，不重复旧内容。一次多文件的解析、去重和顺序由 Android 边界测试覆盖，不伪装成已执行真机项。
5. 从最近任务恢复 Akashic 不再次导入旧分享；应用日志无 FATAL、RenderProcessGone、event sequence gap 或 WebView 协议错误。

最终签名 APK 覆盖安装后，Pixel 7 冷启动接收 `PIXEL7_SYSTEM_SHARE_FINAL_1732`，原草稿、原引用和附件均保留，共享文字只追加一次且未自动发送；截图为 `/tmp/pixel7-system-share-final-1732.png`。

5. 用失效 `content://` 真机触发读取失败：只出现分享 owner 的 Snackbar，文案不泄漏 URI/异常细节，提供“重试 / 放弃”，没有再污染连接错误；截图为 `/tmp/pixel7-share-final-concise-failure.png`。

首次文件验收没有在系统列表看到 Akashic。对比设备已安装 base APK 与当时产物后确认：设备实际 APK 哈希为 `28c2d334…3880`，Manifest 没有 `SEND` filter；不是分享解析或 MIME 兼容问题。无损覆盖安装新产物后，`dumpsys package` 同时注册 `SEND` / `SEND_MULTIPLE`，随后真实系统分享闭环通过。

## 独立 Review 收口

六轮独立 review 从真实崩溃窗口继续追到 owner 收口：先发现跨 session、过早 ACK/出队和静默截断，再发现内存队列、盲写 prepared 草稿、附件重复导入、Intent 路径穿越与重放误判。最终用 durable receipt、一次 claim、原子草稿 CAS、稳定附件 ID 和短窗口 replay dedupe 逐项关闭；最终复核无 High/Medium，仅保留“系统超过 10 秒才重放崩溃前 Intent 时可能 at-least-once”的 Low 权衡。

真机只连接隔离 Mobile Lab；正式 workspace 和正式插件目录不写入测试消息。
