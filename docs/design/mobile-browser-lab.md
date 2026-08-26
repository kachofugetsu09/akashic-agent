# Mobile Browser Lab

## 1. 问题和用户意图

Akashic Mobile 使用 Android 原生能力壳承载 React WebUI。普通颜色、排版、间距、消息组件和流式渲染变化不应要求开发者每次连接手机或启动模拟器。维护者需要在浏览器中直接运行真实 Mobile WebUI，手工调试样式，并复用同一入口完成确定性浏览器验收。

本设计实现 `WEBUI-001`～`WEBUI-003`，不改变 Android、SessionDB、Room、outbox、配对、密钥或 WebUI generation 的 owner。

视觉回归以 [Akashic 纸张品牌系统](akashic-paper-brand-system.md) 为准。`focus=1` 只渲染生产 Mobile viewport，不保留 Lab 控制条，方便远程维护者直接验收像素结果。

## 2. 已确认事实

- `frontend/chat/src/mobile-native.tsx` 是 Android WebView 与 Browser Lab 共用的生产 React 应用；`mobile-entry.tsx` 只安装 Android transport 并调用共享挂载器。
- `mobile-bridge.ts` 把 WebUI 动作编码为带 generation 身份的固定方法 envelope。
- native 通过 `window.AkashicMobile` 投递完整 snapshot、stream patch、state patch 和结果事件。
- `scripts/webui-performance/run-browser-scenarios.mjs` 已能在 Chromium 中构建真实移动 WebUI，并验证 300 条历史和 600 段流式 patch。
- Android 原生壳仍拥有 Compose 配对、设置、恢复提示、WebView 生命周期以及系统能力。浏览器不能证明这些行为。

## 3. 借鉴的成熟做法

Capacitor 采用同一套 Web 应用加平台能力实现：Android/iOS 调用原生插件，浏览器加载 Web 实现；缺少实现时返回明确的 `UNIMPLEMENTED`，不伪造成功。官方 TestApp 同时支持浏览器启动和原生平台运行。Mobile Browser Lab 只采用这套边界方式，不引入 Capacitor runtime，也不把现有 Kotlin 宿主迁移成其他框架。

Playwright 提供设备 viewport、触摸、color scheme 等浏览器仿真以及稳定环境内的截图比较。Lab 让手工检查和 Playwright 使用同一构建结果；截图基线必须在固定 Chromium、字体和运行环境中生成，不能跨机器把像素差异直接判为产品退化。

## 4. 目标结构

```text
┌──────────────────── Browser Lab ─────────────────────┐
│ 场景、设备、主题控制                                 │
│                                                      │
│  fixture ── snapshot / patch ──┐                    │
│                                ▼                    │
│                    ┌──────────────────────┐          │
│                    │ mobile-native.tsx    │          │
│                    │ 生产 React + CSS     │          │
│                    └──────────┬───────────┘          │
│                               │ 同一 bridge envelope │
│                               ▼                      │
│                    Browser capability adapter        │
└──────────────────────────────────────────────────────┘

┌────────────────── Android 生产环境 ──────────────────┐
│ Native snapshot / patch → 同一 Mobile WebUI          │
│ 同一 bridge envelope → Kotlin / Room / 系统能力      │
└──────────────────────────────────────────────────────┘
```

`mobile-lab-frame.html` 只在测试构建中安装浏览器 transport，然后通过生产 `mobile-native-mount.tsx` 挂载同一个 `MobileNativeApp`。它不反向 import Android 入口，也不复制消息 React 树或 CSS。外层 `mobile-lab.html` 只提供场景控制、设备画布和 Bridge 记录，不进入正式 Mobile WebUI generation。

## 5. 能力和失败边界

浏览器 adapter 可以确定性实现：

- snapshot、stream patch、terminal 和发送结果投递；
- 内存中的发送、停止、主题切换、草稿和阅读动作；
- Bridge envelope 可见记录。

相机、系统文件选择、通知、Android 设置、真实分享、密钥、Room、网络重连和 WebView 生命周期没有浏览器等价实现。触发这些方法时，Lab 显示“需要 Android 原生环境”，不返回成功结果。

外网分享只发布构建后的 fixture Lab。它没有正式 workspace、凭据、SessionDB、任意网络和文件能力。Cloudflare Tunnel 只转发 loopback listener，不把仓库开发服务器或正式 Akashic runtime 暴露到公网。

## 6. 固定入口

```bash
npm run serve:mobile-web-lab
npm run test:mobile-web-lab
```

服务固定监听 `127.0.0.1`。端口默认是 `4174`，可用 `AKASHIC_MOBILE_WEB_LAB_PORT` 指定。需要临时远程查看时，把 Cloudflare Tunnel origin 指向 `http://127.0.0.1:4174`。

## 7. 验收

1. Lab 构建不改变 `build:mobile-web` 的正式产物入口。
2. 浏览器接受完整 snapshot 并显示生产聊天页面。
3. 可见流式回答由多段 `receiveStreamPatch` 生长，并由 terminal patch 结束。
4. 输入框发送经过真实 `sendMessage` envelope，Browser adapter 投递发送结果和后续流式回答。
5. 原生专属能力显示明确不可用状态。
6. 浏览器没有未处理异常；TypeScript、lint、现有 mobile state 和 Bridge 测试通过。

## 8. 回滚

删除 Lab 专用 HTML、TypeScript、CSS、Vite 配置和 `scripts/mobile-web-lab`，恢复 `package.json`、`docs/INDEX.md` 和 `mobile-bridge.ts` 的对应改动即可。正式 Mobile WebUI、Android 仓库和运行 workspace 没有迁移数据或持久状态需要恢复。
