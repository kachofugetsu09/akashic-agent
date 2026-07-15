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
