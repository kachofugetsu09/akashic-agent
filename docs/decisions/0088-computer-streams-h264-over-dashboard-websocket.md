# 0088 Computer 经 Dashboard WebSocket 传输 H.264

- 状态：proposed（实现随 stacked Draft PR 提交，待维护者评审）
- 依据：维护者要求调查公网浏览器接管卡顿，以现状为基线尝试并提交 stacked PR。
- 关联：PLG-017、RUN-016、WEBUI-008、0084

## 选择与理由

人工接管默认使用 Selkies 2.0.0 的 H.264 WebSocket 客户端。画面由同一 Computer
容器中的软件编码服务捕获，目标为 1280×800、30 fps、4 Mbps。输入与文字剪贴板
沿同一 generation 的 Dashboard 通道到达同一 Xvnc 桌面，复用主 Chromium profile。
控制服务继续拥有启停和空闲回收，面板、编码连接、轮询和重连不产生持续使用占用。

```text
┌────────────────────────────────────┐
│ 公网浏览器：Computer 面板与工具栏   │
│ 独立 iframe：Selkies 视频及输入     │
└─────────────────┬──────────────────┘
                  │ 原 HTTPS / Tunnel，generation-bound WS
┌─────────────────▼──────────────────┐
│ Dashboard：固定客户端资源及流代理  │
└─────────────────┬──────────────────┘
                  │ 普通 Workload 的私有 stream 端口
┌─────────────────▼──────────────────┐
│ Selkies 软件编码 / X11 输入         │
│ 同一 Xvnc、Chromium、持久 profile   │
└────────────────────────────────────┘
```

高变化桌面的 RFB 数据量会挤占有限下行。固定镜像的隔离实验中，模拟 100 ms RTT、
8 Mbps 下行时，原 noVNC 设置的滚动约 7 fps、画面年龄 P95 569 ms；降低 JPEG
质量约 14 fps、P95 252 ms；Selkies 约 30 fps、P95 123 ms。数据的条件与局限见
[设计和验收](../design/computer-h264-display.md)。这些数字不代表正式公网升级后的结果。

优先复用现有 HTTPS 入口。WebRTC 可减轻 TCP 丢包后的排队，但真实公网还需要
验证 ICE、TURN、转发地址和防火墙；当前 Tunnel 路径没有提供媒体 UDP 通道。
本轮不增加第二套公开入口。隔离 hua-home 的 60 fps 试验能提高运动画面的帧率，
但滚动约 41 fps、带宽和 CPU 开销更高，首版采用已测得较稳定的 30 fps。

## 接入和边界

Selkies 的全局变量、样式和输入监听位于独立 iframe；它是命名空间隔离，不是新的
权限边界。面板继续拥有键盘映射、剪贴板草稿、全屏、活动提醒、休眠和重连状态。
客户端通过现有 `ctx.http.request` 读取，WebSocket URL 由同一 activation 生成。
旧 catalog / generation 的资源不能被新客户端静默接管。

固定上游 commit `3ec56fb1538cf077c27156f5ab75b6595a83c461`、源码下载校验值和 Python
依赖。源码补丁只交入完整 WebSocket 地址、稳定 localStorage 地址和剪贴板发送回执。
构建对准确入口作匹配，升级不匹配时失败；修改源码和 MPL 许可证随镜像保留，并由
固定 Dashboard 资源路径提供。输入 adapter 依赖该版本的 `_sendKeyEvent` 和 context
启停接口，升级必须重新核对 held-key、失焦和剪贴板行为。
另一窗口接管的通知沿固定版本的 `KILL` 消息识别；升级也必须核对该消息与重连行为。

服务不取得宿主 GPU、设备、host network、额外 capability 或公网端口。声音、摄像头、
麦克风、手柄、命令、文件传输、分享和打印关闭。流代理只开放显示 WebSocket 和固定
客户端/源码资源，不代理 Selkies 的任意 HTTP 控制 API。剪贴板发送完成前不注入 Ctrl+V，
失败、超时和断线明确反馈。另一窗口取得控制后暂停自动重连，只有人工明确接管才
重新连接，避免窗口互相抢占。

## 回退与重议

本轮保留私有 RFB bridge。回退恢复上个插件 artifact 与固定镜像，不迁移或删除 profile。
profile、HOME、config 和既有截图的 owner、写入及删除条件仍由 0084 和状态地图定义。

若正式公网升级后仍受 RTT / 丢包排队限制，或软件编码明显影响 Agent 操作，再分别
测量 TURN 路径或硬件编码。浏览器兼容性和丢包场景未验证前，不据此承诺同等体验。
