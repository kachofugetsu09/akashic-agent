# 0135 · 模型连接的额度由驱动可选提供

- 状态：accepted / implemented for review
- 日期：2026-10-28
- 关联：设置 → 模型、用量；参考 magpie（yetone/magpie）的额度卡片

## 问题与决定

订阅登录的连接（ChatGPT/Codex、OpenCode Go）都有服务商限定的用量窗口，此前设置页看不到，
只能等到 429 才知道用完了。各服务商读取方式完全不同，Models 不应知道任何一家的接口。

`ModelDriverDefinition` 增加可选的 `read_quota(descriptor, credential) -> QuotaReading`：
驱动用自己保存的凭据向服务商读取（返回 None 表示这个连接没有额度，同一驱动接多家服务时用），返回套餐、账号、若干窗口（已用比例、重置时间、时长）、
余额文本。没有公开接口的驱动不提供，设置页就不显示额度。窗口名由合同里的
`quota_window_label` 按时长统一生成（5 小时、每周、每月）。

Models 按连接缓存一分钟；用户点刷新也至少间隔 10 秒，并发请求合并成一次读取。
读取失败时保留上一次成功读数并附上原因，界面照常显示旧值和读取时间。
`GET /api/dashboard/models/quotas`（同名 RPC `models/quotas`）只读，不改目录 revision。

本次实现两家：

| 驱动 | 来源 |
|---|---|
| codex | `GET {backend-api}/wham/usage`，与 Codex CLI `/status` 同源；401 时轮换令牌重试一次 |
| opencode-go | `GET {host}/zen/go/v1/usage`，rolling / weekly / monthly 三个窗口；403 表示 key 不属于订阅者 |
| openai-compatible | 按服务主机查 `plan_quota.PLAN_SOURCES`：智谱 `open.bigmodel.cn`、Z.ai `api.z.ai` 读 `/api/monitor/usage/quota/limit`（裸 key 作 Authorization），5 小时 / 每周 / MCP 每月；其他主机返回 None，不出现在额度列表 |

API Key 余额（DeepSeek、Kimi、OpenRouter 等）以同一接口由对应驱动或插件后续接入。

## 取舍

- 不落盘：额度是服务商的实时事实，重启后第一次打开设置页重新读取即可；走势图另行设计。
- 不从模型响应头被动更新：Codex 的 `x-codex-*` 头只在 Codex 驱动内部可见，等有界面需要实时值时再接。
