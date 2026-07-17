# Emotion 主动状态移动面板

## 目标与所有权

这组能力回答一个 Akashic 特有的手机端任务：Agent 当前会采用什么语气，真实反馈又如何
改变下一次主动发送的把握。数据解释、RPC、移动模块和样式全部由 `emotion` 插件拥有；
Agent 核心继续只负责通用插件目录、资产分发和 `plugin.ui.call`。

桌面 Dashboard 保留完整的 VAD、事件与 effect 审计。手机不复制每次 proactive tick 生成
的 effect 表，只显示真正造成 valence 或 dominance 增量的反馈。

```text
┌─ 插件目录：主动状态
│
├─ 当前语气 ───────┬─ 主动发送门槛
│  唯一主指标       └─ 有效影响数量
│
├─ 查看状态指标
│  └─ 愉悦度 / 活跃度 / 主动把握（按需展开）
│
└─ 最近影响（平面列表）
   ├─ 反馈类型 + 用户原话预览
   └─ 主动把握增减
```

## Task-first Material 3

- 蓝色表达当前中性状态，青绿色表达更愿意继续，深琥珀表达降低主动把握；颜色只承担状态语义。
- 当前语气、主动门槛和有效影响组成同一个指标组；最近影响保持平面列表，没有把 26 条反馈做成卡片墙。
- 原始 VAD 默认折叠，不让内部模型指标抢过“现在会怎么表现”这一主任务；折叠态同步 `inert` 与 `aria-hidden`。
- 当前行为由 `db.py` 的 `describe_behavior()` 统一解释，主动 effect 和移动 overview 复用同一个领域 owner，UI 不复制阈值规则。
- 字体沿用 Roboto / Noto Sans SC，状态数字使用 tabular figures；无渐变、阴影、玻璃拟态、发光点或装饰性胶囊。
- fallback 颜色对浅色 surface 的对比度为 `5.31:1` 至 `6.21:1`，满足正文可读性要求。

## 实施与验证

插件源码提交 `48f18d0` 已通过
[PR #2](https://github.com/akashic-plugins/emotion/pull/2) 合入 `main`，merge commit 为
`0db706f`。Docker Mobile Lab 最终从远端 `main` 重新安装，缓存 HEAD 与 merge commit 一致；
正式插件缓存和正式 workspace 均未写入。

- Python 插件测试：`3 passed`。测试真实构造 10 条周期 effect 噪声、4 条非零反馈和 1 条零增量反馈，移动列表只返回 4 条有效影响。
- 移动模块测试：`4 passed`，覆盖任务文案、状态映射、插件所有权和指标 disclosure 的 inert 行为。
- Pyright：`0 errors`；存在 3 个仓库既有/低优先级 warning，没有用 fallback 掩盖。
- 独立 review 两轮通过；当前状态不再使用可能过期的 last effect，噪声过滤和零增量过滤均有回归测试。
- `plugin-doctor emotion@mobile-lab` 的安装、策略、MCP 均为 ok；实验环境未挂载 `feedback-preference-context` drift skill，因此整体报告 degraded，这与移动看板无关。

Pixel 7 连接独立 Mobile Lab 后验证：

1. 插件目录从 2 个原位更新为 3 个，并出现插件自有“主动状态”入口。
2. 只读复制正式 emotion DB 后，概览显示“自然 / 保持门槛 / 26 个有效影响”。
3. 正式快照包含 `3630` 条周期 effect 和 `28` 条反馈事件；移动端只显示其中 `26` 条非零影响，没有复制 tick 噪声。
4. 最近影响显示真实用户原话预览，并用青绿或深琥珀分别表达主动把握增加和减少。
5. 展开状态指标后显示愉悦度、活跃度和主动把握；再次收起后三个原始指标从 Android 无障碍树消失。
6. 从远端 `main` 回装插件后重新进入面板，真实状态和 26 条影响保持一致。
7. Android 和服务端日志无 FATAL、RenderProcessGone、event sequence gap、插件 RPC 或 WebView 异常。

截图证据：

- `/tmp/pixel7-plugin-directory-emotion.png`
- `/tmp/pixel7-emotion-populated.png`
- `/tmp/pixel7-emotion-metrics-expanded.png`

## 隔离与恢复

- 正式 `/home/huashen/.akashic/workspace` 只通过 SQLite `.backup` 读取快照，没有写操作。
- Mobile Lab 插件目录备份：`/mnt/data/coding/backups/mobile-lab-plugin-home-before-emotion-20260717-075649.tar.gz`。
- Mobile Lab 数据备份：`/mnt/data/coding/backups/mobile-lab-emotion-data-before-20260717-075701/`。
- 实施日志备份：`/mnt/data/coding/backups/mobile-agent-im-implementation-log-before-emotion-20260717-080146.md`。
- 所有可变验证只发生在隔离 workspace；正式会话和手机日用会话不受影响。
