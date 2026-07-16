# Android WebView 迁移与插件 UI 协议

## 迁移边界

Android 仍负责系统信任边界与长连接；WebView 负责需要快速迭代、并与 WebChat 保持一致的会话界面。

```text
┌──────────────────── Android 原生层 ────────────────────┐
│ 扫码/相机  Keystore  WebSocket  Room  文件/分享  通知 │
│                       │ 版本化 JSON                     │
│                       ▼                                │
│ ┌────────────────── WebView 会话层 ──────────────────┐ │
│ │ 抽屉  消息  Markdown/LaTeX  思考时间线  输入/附件 │ │
│ │                    │ 插件 slots                    │ │
│ │                    ▼                              │ │
│ │ 左右脑召回  工具前内容  回答后内容  抽屉面板      │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

不迁入 WebView：配对、凭据加密、网络重连、离线持久化、上传下载、文件 URI、前台服务和通知。这些能力需要 Android 生命周期、系统权限或本地安全边界。

迁入 WebView：会话抽屉、消息顺序、Markdown/KaTeX、思考与工具时间线、跟随滚动、命令面板、输入区、附件展示和插件自管理 UI。它们复用 `frontend/chat` 已有组件和依赖，不再维护第二套 Compose 渲染器。

## 插件 UI 接入

插件声明两个静态资产和一个受控 RPC：

```python
class ExamplePlugin(Plugin):
    @classmethod
    def mobile_ui_module(cls) -> str | None:
        return "mobile_ui.js"

    @classmethod
    def mobile_ui_stylesheet(cls) -> str | None:
        return "mobile_ui.css"

    async def mobile_ui_call(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        ...
```

`mobile_ui.js` 是默认导出定义对象的 ES module。插件拥有容器内部 DOM、交互和清理逻辑；核心只提供稳定插槽与 `context.request()`。

```js
export default {
  slots: {
    "turn.before_reasoning": {
      mount(host, context) {
        host.textContent = "加载中";
        context.request("panel.current").then((result) => {
          host.textContent = result.label;
        });
        return () => host.replaceChildren();
      },
    },
  },
};
```

| 插槽 | 语义 |
| --- | --- |
| `turn.before_reasoning` | 本轮思考时间线之前 |
| `turn.before_tool` | 每个工具调用之前，`context.block` 提供当前工具块 |
| `turn.after_answer` | 最终回答之后 |
| `drawer.panel` | 会话抽屉中的插件区域 |

资产按插件真实 ID 和 SHA-256 版本化。Android 拉齐整批资产并校验摘要后才发布；WebView 在全部模块加载成功后原子替换旧定义和样式。模块加载有 5 秒截止时间，失败版本会隔离且保留旧 UI。RPC 服务端截止时间为 20 秒，回包在 Android 保留到 WebView 分批 ACK，重连不会静默丢失 Promise。

## Akasha 左右脑设计

Akasha 在 `turn.before_reasoning` 插入一个统一召回组，而不是两张独立卡片：

```text
┌─ ● 左脑 · 精确回忆                         3 条 ─┐
└───────────────────────────────────────────────┘
┌─ ◆ 右脑 · 联想记忆                         2 条 ─┐
└───────────────────────────────────────────────┘
                 ▼ 展开后共享列表平面
```

- 蓝色圆点只表示精确召回；亮紫色菱形只表示联想召回。
- 色彩使用 OKLCH，并复用移动端 surface/on-surface 语义变量。
- 默认折叠，不占用思考主路径；展开行满足 44px 触控尺寸。
- 依靠同色 state layer、分组和留白表达层级，不使用阴影或卡片套卡片。
- 正文沿用 16px 基准和约 1.45 行高，分数使用紧凑标签字号。

## 提交与验证记录

| Commit | 能力 | 验证 |
| --- | --- | --- |
| `d0ac0c38` | Mobile Web 构建基础 | Web bundle、Android assets 合并 |
| `ed7560c5` | 会话 UI 迁入 WebView | typecheck、lint、Android unit/build |
| `2f419b85` | 服务端插件 UI 协议 | 139 项服务端测试、pyright |
| `4343db99` | Android/Web 插件运行时与 Akasha UI | 43 项定向测试、typecheck、lint、Android unit/build、子代理语义复审 |
| `b4331fb8` | 移除退休 Compose 对话 UI | Android unit、androidTest 编译、生产引用审阅 |

最终发布前执行：

```bash
npm run typecheck
npm run lint
uv run pytest -q tests/test_plugin_mobile_ui.py tests/mobile_realtime/
ANDROID_HOME=/home/huashen/Android/Sdk clients/android/gradlew -p clients/android \
  :app:testDebugUnitTest :app:compileDebugAndroidTestKotlin :app:assembleRelease
```

历史原生实现可由标签 `archive/mobile-native-0.6.2-20260715` 恢复。

## 0.7.0 发布结果

- 完整服务端测试：`2238 passed`
- 前端：typecheck、lint、dashboard/chat/plugin production build 通过
- Android：release unit、lint、R8、assemble 通过
- APK：v2 签名校验通过，证书 SHA-256 为 `49bf31ed5c54c642d6f4fdd30a5310a8cb70e67666ad25d711b5f0e084e240bc`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.0>
- ADB 安装：发布时没有已连接设备，未在本轮远程安装

### 0.7.1 白屏修复

`0.7.0` 错误启用了 `WebSettings.blockNetworkLoads`，连 `WebViewAssetLoader` 使用的受控 HTTPS appassets origin 也被阻止，导致 React 首帧前白屏。`0.7.1` 改由 `MobileWebClient` 精确放行 `appassets.androidplatform.net`，其他 origin 返回 403；主文档的网络错误和 HTTP 错误会显示原生错误页与重新加载入口。

私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.1>

### 0.7.2 旧 WebView 兼容修复

静态复现确认旧 WebView 缺少 `Array.prototype.at` 时，Markdown 依赖会在 React 首帧期间抛错并留下空根节点。移动入口现在通过 `core-js` 在应用模块执行前补齐 `Array.prototype.at`、`Object.hasOwn` 和 `structuredClone`；Material 颜色令牌同时增加 sRGB fallback，支持 OKLCH 的 WebView 继续使用原色。React 渲染异常会显示可重新载入的错误状态，不再直接白屏。

- 兼容门禁：产物加载前删除三项原生 API，Kotlin 形状的 snapshot、Markdown 和公式文本仍完整渲染
- 验证：TypeScript、ESLint、release JVM、Lint、R8、assemble 与 APK v2 签名均通过
- APK：`0.7.2`（versionCode 11），SHA-256 `f73c8c2251ae38f9a6a3579d513af2cef4af50f4127b6511acd5fd9317afaaf3`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.2>
- ADB 安装：发布时没有已连接设备，等待真机覆盖安装后确认实际 WebView 行为

### 0.7.3 配对与首帧稳定性修复

二维码入口只接收 JSON object，再由严格协议解码器统一校验字段、签名、有效期和证书；非对象内容会返回稳定错误。相机提高分析分辨率并启用旋转与 QR 专用提示，同时增加可滚动、适配键盘且限制为 32 KB 的手动粘贴入口。

Mobile Web 首帧改用 React ready 握手和 10 秒页面代际 deadline，入口 JS/CSS 失败或脚本未挂载时显示原生重新加载页。Native snapshot 在协议边界严格校验，只有通过校验并提交 React 后才停止请求；无效快照会明确显示原因并继续等待有效状态。

- 复审：两轮 Android/Web 只读 gate 后无发布 blocker
- 验证：TypeScript、ESLint、debug/release JVM、Kotlin 编译、Lint、R8、assemble 与 APK v2 签名均通过
- APK：`0.7.3`（versionCode 12），8,255,218 bytes，SHA-256 `3f26c1c87c40ad3c76c309274d04c1c3938104034a4ff15a7daa5973de8c35df`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.3>
- 真机验收：等待覆盖安装后验证 Pixel 7/Pixel 9 的扫码、手动粘贴和会话首帧

### 0.7.4 Akasha 历史召回恢复

Mobile Web 宿主现在为每条普通 assistant 历史消息挂载 turn 插槽，插件可以按消息身份恢复对应轮次的数据；主动推送消息不会挂载 turn 插件。Akasha mobile UI 会用 `session_id + message_id` 定位该轮真正注入模型的 `context` 查询日志，完整展示 Dense / Ripple，不再截成各 6 条，也不会用最近一轮覆盖旧轮。

召回项按 `happened_at` 从新到旧排列。新查询日志会直接保存该字段；旧日志缺失时，从服务端 `sessions.db` 按原消息 ID 只读补齐。因此清理手机缓存、重新安装或换机后，只要服务端仍保留 mobile 消息和 Akasha 查询日志，历史轮次同步完成后会重新显示对应召回；没有 Akasha 日志的轮次保持为空。

- 验证：Akasha pytest 60 项、Pyright、TypeScript、ESLint、debug JVM、assemble 与生产 Mobile Web 构建通过
- 生产只读验收：assistant seq `22/24/26` 分别绑定 Akasha seq `21/23/25`，返回 `4+15 / 9+9 / 10+8` 条，历史时间全部可解析
- APK：`0.7.4`（versionCode 13），8,255,238 bytes，SHA-256 `c153402f92b1550772053e2ab930c92d8c2d756bcb471a5bade34ecb210a5697`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.4>
- 真机验收：Pixel 7 覆盖安装成功，系统报告 versionCode 13 / versionName 0.7.4，MainActivity 正常前台运行

### 0.7.5 Akasha 插件会话绑定修复

Mobile Web 插件 RPC 现在沿渲染槽位显式传递 `session_id` 与活动 `turn_id`，不再由原生异步任务读取可能已经切换的当前会话。历史消息与 Akasha 查询日志因此保持同一会话身份，左右脑面板加载不会再制造跨会话请求。

插件明确拒绝参数时，服务端返回持久化的 `plugin_invalid_request` RPC 错误；该错误只影响对应插件面板，不再穿透 ASGI 并关闭整条 WebSocket。

- APK：`0.7.5`（versionCode 14），8,255,242 bytes，SHA-256 `fcfdf17741d5e64c435c2df69343fdfb511fbb71690c7d48901e01ea148e8a4f`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.5>
- 真机验收：Pixel 9 Pro XL 覆盖安装成功并保留配对数据，系统报告 versionCode 14 / versionName 0.7.5；重新连接后完成全量历史同步，连接状态正常，旧消息不再显示跨 session 错误

### 0.7.6 Mobile WebView 交互修复

快捷命令面板改为相对 Android WebView 可视视口定位，键盘和底部安全区变化时不再被 composer 父容器裁切。思考与工具时间线从首节点中心开始，并在末节点中心结束；长思考文本不再让轴线伸到最后一个节点下方。

- 验证：TypeScript、ESLint、release JVM、Lint、R8、assemble 与 APK v2 签名均通过
- APK：`0.7.6`（versionCode 15），8,255,330 bytes，SHA-256 `1f287beae92e0dfb30d24866d4e4776dad1c912c93211ae88d3b6a4d40694ad9`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.6>
- 真机验收：Pixel 7 与 Pixel 9 Pro XL 均覆盖安装成功并保留应用数据，系统报告 versionCode 15 / versionName 0.7.6；MainActivity 正常前台运行，启动阶段没有 AndroidRuntime 或 Mobile Web 错误

### 0.7.7 双向引用与实时链路自愈

用户消息与 Agent 最终回答都支持左滑引用，消息时间、日期分隔和独立复制动作随历史同步恢复。引用目标由服务端按同会话 canonical 消息重新解析，并进入 Agent 的真实模型上下文。

坏 `message.send` 使用 `4410` 隔离当前 outbox 命令，后续消息在重连后继续发送；一般协议错误与版本不兼容保留待发内容。服务端 resume 发现 durable event 缺号时发送 `sync.reset_required`，复用现有全量重建协议恢复，不再让客户端陷入永久重连。

- 验证：Python `2269 passed`；TypeScript、ESLint、release JVM、Lint、R8、assemble 与 APK v2 签名均通过
- APK：`0.7.7`（versionCode 16），8,260,170 bytes，SHA-256 `4dbcba376dd0c63ca49c5d9d389c168af5f06632efd43b44c254d4d75b6c0cde`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.7>
- 真机验收：Pixel 7 覆盖安装成功并保留应用数据，系统报告 versionCode 16 / versionName 0.7.7，MainActivity 正常前台运行且 logcat 无 FATAL；正式 Pixel 9 的缺号 cursor 已通过 reset 推进，durable inbox 清零并停止重连风暴

### 0.7.8 会话搜索、未读锚点与工具详情

会话顶栏新增本地搜索，支持从用户正文、Agent 回答和附件名跳转前后结果；流式输出期间保持当前阅读位置。离开底部后按逻辑 Agent turn 记录未读，首次点击回到底部动作会先定位首条未读，再回到真正底部。普通重连保留未读和滚动语义，只有明确的破坏性投影重建才建立新的已读基线。

思考时间线内的工具节点可原位展开最终参数、结果摘要或失败内容。实时耗时由服务端 monotonic clock 计算，不使用手机收帧间隔；历史没有可靠单工具时间戳时省略耗时。参数投影会隐藏常见敏感键、Authorization/Bearer、环境变量赋值和 argv 凭据，并按结构与 UTF-8 字节收敛到单调用 8 KiB，避免参数随完整会话热快照反复放大。

- 独立 Review：经过 UTF-8 帧预算、凭据字符串/argv、最终参数、服务端耗时、非成功状态、旧桌面历史、WebView 快照性能和无障碍语义复核，最终 no findings
- 验证：mobile channel/gateway `40 passed`、Pyright 零错误、Mobile Web 状态 `11 passed`、TypeScript、ESLint、Android 定向 JVM 与 23 条不依赖外部 pairing 参数的真机 instrumentation 通过
- 构建：release unit、Lint、R8、assemble 与 APK v2 签名通过；签名证书 SHA-256 `49bf31ed5c54c642d6f4fdd30a5310a8cb70e67666ad25d711b5f0e084e240bc`
- APK：`0.7.8`（versionCode 17），8,281,638 bytes，SHA-256 `2d8b3f6b64eaa955a4034885153fb0070e0a5d8dab90949f2b3f4ad3bf05c45e`
- 私有发布：<https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.8>
- 真机验收：Pixel 7 无损覆盖安装并保留配对和历史；系统报告 versionCode 17 / versionName 0.7.8，手机 `base.apk` 与构建产物哈希一致。MainActivity 正常前台运行，工具详情可展开，logcat 无 FATAL、WebView render crash、event gap、4406 或协议错误
