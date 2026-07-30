# 0007 · 移动插件控制面与查询数据面显式分离

- 状态：accepted
- 日期：2026-07-28
- 关联条款：MOB-001、MOB-003、MOB-006、PLG-003、PLG-011、TST-006～TST-008
- superseded by：[0009](0009-akasha-mobile-recall-preserves-semantic-lanes.md)（仅取代第 5 项的每 lane 五项与 16KiB 上限）

## 背景

移动端通过一条 WebSocket 同时接收会话事件、插件目录、插件资源和插件查询正文。
Akasha 当前回复下方的 recall 卡片会返回 Inspector 的完整 lane 行，其中包含界面不渲染
的长正文。Pixel 7 真实测量中，服务端查询只需要约 18ms，而 28～104KiB 的内联结果
从 WebSocket 发送到 WebView JavaScript 需要约 2.3～5.9s；同一结果命中本地缓存后只需
约 21ms。瓶颈位于大 JSON 共享实时链路、原生解析和同步 JavaScript 注入，不在 Akasha
检索算法。

插件热插拔仍需要版本、generation 和取消语义。直接让每个插件自建 HTTP endpoint
会绕过 Core 的设备认证、revision lease 和调度；把所有插件立即迁出 WebSocket 又会
扩大兼容范围。

## 决定

1. WebSocket 继续拥有认证、resume、实时事件、插件 catalog/asset、查询授权和取消。
   既有 `plugin.ui.query` 保持内联 reply，不改变未迁移插件。
2. 插件显式选择 HTTPS 时发送 `plugin.ui.query.prepare`。Core 在已认证连接上校验查询，
   返回绑定设备、连接代际、规范化请求摘要、服务端身份和 30s 有效期的 ECDSA ticket。
3. Mobile 从当前活动的已校验 WebSocket endpoint 派生同源 HTTPS 地址，用 POST 提交
   ticket 绑定的原查询。Core 验签、重新读取设备撤销状态，再复用既有 plugin query
   scheduler、generation lease、owner 和取消路径执行。
4. ticket 是短期、无状态、只读的 bearer capability；不写 workspace、数据库、
   durable inbox、command receipt 或 session。HTTP response 明确 `no-store`，客户端
   只把 immutable turn 结果写入已有的可重建本地缓存。
5. Akasha `recall.current` 首个使用该数据面，返回
   `akasha.recall-card.v1`：每条 lane 最多五项，只包含 100 字用户预览、50 字助手预览、
   时间和可选分数，最坏 Unicode fixture 的完整卡片小于 16KiB。桌面 Inspector 与
   `inspector.detail` 保持完整投影。
6. Native 到 WebView 的插件 catalog/result 改用异步 `WebMessage`，不再把结果拼成
   `evaluateJavascript` 源码。Mobile 继续在 TypeScript 边界解析和校验消息。

```text
┌───────────────┐  plugin.ui.query.prepare  ┌────────────────────┐
│ Mobile Web UI │ ─────────────────────────▶ │ Authenticated WS   │
└───────┬───────┘ ◀──── signed grant ─────── │ control plane      │
        │                                     └────────────────────┘
        │ HTTPS POST + request-bound ticket
        ▼
┌────────────────────┐  generation lease  ┌──────────────────────┐
│ Core HTTP data     │ ──────────────────▶ │ Akasha card-v1 DTO  │
│ plane + scheduler  │ ◀────────────────── │ read-only provider   │
└─────────┬──────────┘                     └──────────────────────┘
          │ compact JSON + gzip
          ▼
┌────────────────────┐  async WebMessage  ┌──────────────────────┐
│ Android cache      │ ─────────────────▶ │ WebView renderer     │
└────────────────────┘                    └──────────────────────┘
```

## 理由

控制面需要长连接的顺序、状态和取消；查询正文需要独立请求、HTTP 压缩、大小边界和
不阻塞实时事件的传输。短期签名 ticket 让 HTTP 复用已有设备身份而不新增 cookie、
长期 token 或服务端 ticket 表。请求摘要绑定阻止 ticket 被换参复用，实时撤销检查让
已经签发的 ticket 不能绕过设备撤销。

显式 opt-in 保留旧插件兼容性，也让迁移单位保持在插件语义投影，而不是一次重写全部
插件。Akasha 自己知道哪些字段对 recall 卡片有意义；Core 和 Mobile 都不应猜测或裁切
记忆语义。

## 影响

- Core runtime 新增一个同监听器 HTTPS route、ticket issuer 和协议命令，但不新增
  权威持久状态。
- Akasha 插件 revision 会因 module 与 card DTO 改变；新 module 会检查
  `transport=https` capability，旧 Mobile 明确提示更新，不把 HTTPS 请求静默退回内联。
- Mobile 继续使用现有结果缓存目录和驱逐协议，不需要 Room migration。
- Cloudflare 或 LAN endpoint 同时承载 WSS 与 HTTPS；ticket 不允许客户端指定 host，
  redirect 被禁用。
- 其他插件保持内联 WebSocket 查询，后续是否迁移由各插件需求单独决定。

## 验收

- 协议 schema、Python parser 和 Kotlin codec 都接受 prepare/ready，旧 query 测试不变。
- ticket 篡改、请求换参、过期和设备撤销都在插件执行前失败。
- HTTP 查询仍受 provider revision、generation lease、设备并发、plugin 并发、owner
  cancel 和断线 cancel 约束。
- Akasha 最坏 Unicode card fixture 小于 16KiB，结果中不出现 `user_text`、
  `assistant_text` 或未渲染 Inspector 字段。
- Mobile HTTPS 使用活动 endpoint 的既有 LAN pin 或 tunnel system trust，拒绝 absolute
  path、redirect 和超过 192KiB 的响应。
- Native/WebView 通过异步 message 交付 catalog/result；TypeScript、Kotlin 单测、Core
  定向测试、跨仓库 schema parity 和隔离 Pixel 7 首次打开/缓存复开计时全部通过。
