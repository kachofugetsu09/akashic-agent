# Citation + Meme 纯 v3 WebUI E2E 任务合同

## 1. 目标

在一次性 Docker workspace 中按 installed artifact 布局装入 exact-commit 的纯 v3
Citation 与 Meme，启动完整 supervised Gateway，并只通过公开 WebUI HTTP/WebSocket
完成一轮被动回复。Gate 必须同时证明模型 prompt、最终消息、持久化 metadata、媒体读取、
Dashboard、能力检查与进程清理一致。

本任务不修改插件行为，不读取正式 workspace，也不删除 Core v2 兼容分支。Core v2 的最终
移除继续由独立迁移清单管理。

## 2. Owner 与运行边界

- Core 拥有 installed artifact 解析、stable snapshot、WebUI channel、SessionDB commit、
  Dashboard 路由和 supervised 进程生命周期；
- Citation 拥有 `citation.protocol` 与 `cited_memory_ids`；
- Meme required Fiber 依赖该 Service，拥有 prompt、回复媒体、Skill 和 Dashboard；
- `model-gate` 是 Compose 私网内真实 OpenAI-compatible HTTP provider，只固定确定性回复；
- Gate host 拥有临时 checkout、临时 workspace、Compose project、报告和最终 cleanup。

```text
exact plugin commits
        │
        ▼
installed stable artifacts ──► supervised Gateway ──► WebUI-only channel
                                      │                       │
                                      ▼                       ▼
                                model HTTP request       WebSocket turn
                                      │                       │
                        Citation prompt → Meme prompt         │
                                      │                       │
                                      └──────────┬────────────┘
                                                 ▼
                                  SessionDB + media + Dashboard
                                                 │
                                                 ▼
                                      graceful stop + zero residue
```

## 3. 行为合同

1. plugin home 只发布锁内 Citation 与 Meme 的 stable/latest 指针；artifact 内容来自 fresh
   detached checkout，manifest 只启用 `citation@webui` 与 `meme@webui`。
2. 隔离 `/app/plugins` 只保留 Web Shell 导入所需的 `default_memory` bootstrap package，
   配置关闭 Memory，因此 active runtime capability 列表只能由两个目标 v3 插件贡献；
   这不改变 Core 源码。
3. 配置只启用 `[channels.chat]`；Telegram、QQ、mobile realtime 与 proactive 均关闭。
4. WebUI `/api/shell/state` 与 `/api/chat/health` ready 后，客户端通过 `/ws` 创建 session，
   发送一条用户消息，并等待同 session 的 `message.final`。
5. model-gate 返回 `答复正文\n§cited:[mem_1]§ <meme:shy>`。模型请求中的 Citation
   protocol 必须早于 Meme prompt；最终 WebUI 正文必须为 `答复正文`，只带 fixture 图片。
6. WebUI messages API 必须读到同 session 的 user/assistant 两条 append-only 消息；assistant
   包含 `cited_memory_ids=[mem_1]` 与相同 media，数据库 `integrity_check=ok`。
7. WebUI media API 必须返回 fixture 的 exact bytes；Dashboard categories 必须返回 `shy`；
   runtime capabilities 必须列出两个 installed plugin 和 `meme-manage` Skill。
8. 被动链路前后 `workspace/memes` 摘要相同。Gate 只允许临时 SessionDB、runtime、
   plugin-data、Skill link 和日志等正常运行写入。
9. 两个 installed artifact 的逐文件摘要、artifact inventory 与 stable/latest 指针对在
   运行完成及 Gateway 优雅停止后都必须与安装时相同；Dashboard 编译、PluginWatcher 和
   dispose callback 都不得修改或重发布锁内 generation。
10. Gateway 必须经 SIGTERM 优雅停止；停止后的 `workspace/memes` 仍须与运行前一致。
    Compose down 后本项目不得残留 container、network 或 volume，临时 sandbox 必须删除。

## 4. 失败、证据与恢复

- 浮动 revision、artifact/manifest 漂移、非 WebUI channel、prompt 顺序、最终输出、
  SessionDB、媒体、Dashboard、能力或 cleanup 任一不符都 fail-loud；
- 报告固定 Core head/tree、Gate version、scenario hash、lock hash、provider commit/tree、
  config hash、installed pointers/artifact inventory、模型请求状态/顺序索引/payload SHA、
  WebUI frame、持久消息、运行后与停止后的 workspace 摘要和 cleanup；报告不记录请求
  headers、凭证或完整 prompt；
- Gateway 日志和运行报告只写被 Git 忽略的 Gate 报告目录；不记录凭证；
- 失败时仍执行 Compose down 与 sandbox 删除。源码恢复点为
  `backup/plugin-v3-webui-e2e-before-20260816`。

## 5. 验收

```bash
python docker/debug/plugin_passive_webui_v3_e2e.py --require-clean-core
python -m basedpyright --level error docker/debug/plugin_passive_webui_v3_e2e.py
git diff --check
```

真实报告必须为 `status=passed` 且 `cleanup.residuals=[]`。没有启动真实 Gateway、没有经过
公开 WebUI 或 cleanup 失败时，不能称为 E2E 通过。
