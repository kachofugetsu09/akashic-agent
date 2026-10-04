# Computer H.264 显示：结构和验收

- 状态：stacked Draft PR 的实现与隔离验收；尚未正式部署。
- 选择理由：[0086](../decisions/0086-computer-streams-h264-over-dashboard-websocket.md)。
- 生命周期：[0084](../decisions/0084-computer-keeps-identity-without-an-idle-desktop.md)。

## 当前问题与基线

用户在电脑浏览器经公网域名接管 Computer，滚动和操作感觉不连续。基线为 Computer
镜像 `6a2cb0b57e912d48dae55b1b4c1d62c43d52108692bb330b038c1ba00bb6a365`，TigerVNC
1.12.0、1280×800、noVNC quality 7 / compression 2、无 GPU。实验保持同一桌面镜像，
用每帧绘制的时间条码记录客户端可见帧和画面年龄，以真实鼠标输入测反馈。

2026-10-04 隔离实验的代表结果：

| 条件 | 滚动可见帧率 | 画面年龄 P95 | 输入反馈 |
|---|---:|---:|---:|
| 原 RFB，模拟 100 ms RTT / 8 Mbps | 7.4 fps | 569 ms | 点击 P50 152 ms |
| RFB quality 4，其他条件相同 | 14.5 fps | 252 ms | 点击 P50 152 ms |
| Selkies 30 fps / 4 Mbps，同一模拟网络 | 约 30 fps | 123 ms | 点击约 152 ms |
| hua-home 隔离 Selkies，相同模拟网络 | 约 31 fps | 111 ms | 点击约 160 ms |
| hua-home 隔离 Selkies 60 fps / 6 Mbps | 约 41 fps | 124 ms | 点击约 158 ms |

hua-home 30 fps 运动画面约 29 fps、P95 192 ms；60 fps 运动画面约 54 fps、P95 127 ms。
CPU 分别约为一核和 1.3 核，整个实验容器的内存约 1.27 GiB。这包含桌面和 Chromium，
不是独立编码器开销。以上是固定高变化页面，实际文字可读性仍须人工检查。

只读探测原公网入口的 ping/pong 中位数约 493 ms，SSH 经内网到 origin 约 4.5 ms。
这是共享 Web 入口探测，不能归因于某个运营商或 Cloudflare 跳点，也不能直接替代
Computer 的输入到画面测量。升级编码能够减少带宽与排队，不能消除已有公网 RTT。
本轮没有公网 A/B、丢包模拟、TURN 或 Safari / Firefox / Android 实机结果。

实现后的完整面板 → Dashboard relay → 隔离流服务链路，模拟 100 ms RTT / 8 Mbps
时，滚动约 30.8 fps、画面年龄 P95 139 ms、2.76 Mbps；运动约 31.0 fps、P95
140 ms、3.16 Mbps。十次真实点击反馈 P50 176 ms、P95 191 ms。这个结果包含面板
接入开销，不能用前述裸客户端数据替代。

## 实现与 owner

`docker/computer/start.sh` 随桌面启动 Selkies；编码器先释放 XShm 和输入连接，然后
关闭 X server。`gateway.mjs` 的桌面 readiness 包含流服务 health。Workload 使用命名
私有端口 `stream`，没有 Core 专属 Computer 分支。

`plugins/computer/dashboard.py` 通过原 Dashboard 租约代理 WebSocket，禁用上游
WebSocket 压缩，沿已有取消与关闭路径结束双方连接。只开放固定 JS、修改源码和
许可证路径；保持现有 Web identity、同源和 generation 校验。

`plugins/computer/web/display.ts` 拥有当前 iframe、transport、Blob URL、发送回执和
超时请求。它将 native 鼠标输入交给 Selkies，把可信键盘事件交给既有面板映射。
先释放输入，再移除 iframe、关闭 transport、取消请求和释放 Blob。面板继续拥有
连接/休眠显示、后台 hold、唯一重连计时器和剪贴板草稿；发送失败不宣称成功。

文字剪贴板通过固定版本的发送回执排序。回执说明客户端已完成发送；并不是任意
外部应用完成粘贴的证明。人工操作和 Agent 操作沿各自既有 owner 到同一 display，
两者不另建主 profile writer。

## 状态与恢复

本轮只增加可重建的镜像依赖和运行时视频资源，没有正式 workspace 数据写入。
profile、HOME、config 正常随主浏览器原位更新；休眠让进程、JS 和 DOM 失效，保留
身份目录。截图仍由原 MCP owner 增加和保留，传输优化不授予新的删除权。
详见[状态地图](persistence-state-map.md)。回退使用上个 artifact / 固定镜像和同一数据目录。

## 验收边界

每层 PR 控制新增约 500 行以内：先镜像和生命周期，再 generation-bound 面板接管，
最后合同与验收记录。按依赖顺序评审和采用。

已完成的隔离验证：

- 完整 Docker 构建、固定依赖的 `pip check`、相关类型/语法和插件边界检查。
- 新客户端经实际 Dashboard relay 显示同一主桌面；真实键盘、右/中键、拖动、滚轮。
- 新文字及 Unicode 双向剪贴板、失焦时释放 held keys、失败与断线反馈，以及另一窗口接管后的暂停与人工取回。
- 320px、200% 字体、剪贴板草稿/焦点和全屏；无横向溢出，dispose 无残留 iframe。
- 打开面板后停止实际操作，桌面按时休眠、编码器结束；显式唤醒沿同一 profile。
- 代表性高变化页面在模拟 100 ms RTT / 8 Mbps 下保持约 30 fps，无持续帧队列增长。

组件夹具使用真实插件 Dashboard 路由和隔离 Workload；它不代替完整 Root 换代与
旧 generation 排空验收。正式采用还需固定 release/artifact、真实 Controller、完整
generation 切换，以及用户实际公网浏览器的接管验收。

## Agent 操作位置反馈

桌面输入完成后，Native backend 读取 X11 的实际位置和屏幕尺寸。主浏览器的
CDP 鼠标/触摸输入完成后，将主页面 CSS 坐标、缩放与窗口装饰换算到桌面位置；
隐藏标签清除标记，匿名 headless 浏览器不发布主桌面反馈。设备仿真和多屏未验收。
位置读取失败单独报告并清除标记，不重试输入、不改写已完成操作的回执。
主浏览器位置读取最多等待 250 ms，避免页面脚本阻塞让辅助反馈长期占住输入调用。

```text
┌───────────────────────┐
│ Native / 主浏览器输入 │
└──────────┬────────────┘
           │ 已执行的位置反馈
┌──────────▼────────────┐   ┌──────────────────────────┐
│ Gateway：内存最新位置│ → │ Dashboard：只读 WebSocket│
└───────────────────────┘   └──────────────────────────┘
```

Gateway 只在内存保留最新序号、位置与五秒有效期；新位置覆盖旧位置，休眠和停止
清空，重启不恢复。通道最多 32 个消费者，慢消费者断开，订阅不唤醒、不 touch，
不写 activity、profile 或 workspace。Dashboard 使用当前 generation 的租约转发；
浏览器向该只读通道发送消息会被拒绝。这个反馈不证明外部应用已接受点击。

面板参考 Codex 光标的箭头、描边、蓝色光晕、长移动弧线和点击缩放反馈，使用
独立 SVG 与既有品牌 token。动画只在位置更新后短暂运行，不让实际输入等待动画。
热点对齐 iframe 中实际视频区域，缩放、留白和面板全屏沿同一位置换算。
人工指针/键盘操作、失焦、到期、失败或断线立即隐藏；减少动态效果时直接定位。
标记不接收指针事件、不获取焦点，断线通过状态文字说明，面板 dispose 释放订阅、
观察器、计时器和动画。不增加独立重连 owner；重新连接面板时恢复订阅。

隔离浏览器验收覆盖 Native 与主浏览器点击、普通窗口装饰和浏览器缩放的截图校准、
320px/200% 字体、实际视频留白、全屏、减少动画、人工接管、五秒到期、匿名浏览器
隔离和 dispose。实际公网接管仍需用户验收。

上游资料：[固定版本源码](https://github.com/selkies-project/selkies/tree/2.0.0)、
[Core API](https://github.com/selkies-project/selkies/blob/2.0.0/addons/selkies-web-core/README.md)、
[TURN](https://webrtc.org/getting-started/turn-server)。
