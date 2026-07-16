# 主动反馈移动面板

## 目标与所有权

这组能力回答一个 Akashic 特有的手机端任务：主动消息发出后，用户是否真的继续了这个
话题，以及系统为什么把它判断为继续、明确引用或没有继续。

数据、RPC、移动模块和样式全部由 `proactive_feedback` 插件拥有。Agent 核心继续只负责
通用的插件目录、资产分发和 `plugin.ui.call`，没有新增插件名、数据库路径或业务分支。
移动端复用插件既有 `ProactiveFeedbackDashboardReader`，桌面 Dashboard 保留完整审计表，
手机不复制桌面表格。

```text
┌─ 插件目录：主动反馈
│
├─ 继续率 ─────────┬─ 明确引用
│  唯一主指标       └─ 高可信信号
│
└─ 最近回应（平面列表）
   ├─ 全部 / 继续 / 引用
   └─ 点开一条
      主动发出
          │
      用户回应
          │
      助手继续
```

## Task-first Material 3

- 青绿色只表达“话题被继续”，紫色只表达明确引用，蓝色只表达高可信信号；没有用颜色装饰普通列表。
- 三个指标是同一语义组，最近回应保持平面列表；只有用户主动展开时才出现关联时间线，没有事件卡片墙。
- 筛选使用一个 Material 3 segmented control，不把每个状态做成独立胶囊或 badge。
- 关系详情使用插件已有预览，不新增第二套持久化；折叠态同步 `inert` 与 `aria-hidden`，隐藏内容不会留在键盘或 Android 无障碍树。
- “未能判断”使用深琥珀文字而不是错误红；fallback 文字对浅色 surface 的对比度为 `7.20:1`。
- 字体沿用 Roboto / Noto Sans SC，数字使用 tabular figures；无渐变、阴影、玻璃拟态、发光状态点或装饰图标。

## 实施与验证

插件源码提交 `153be14` 已通过
[PR #2](https://github.com/akashic-plugins/proactive_feedback/pull/2) 合入 `main`，merge commit
为 `3fa085e`。Docker Mobile Lab 最终从远端 `main` 安装，缓存 HEAD 与 merge commit 一致；
正式插件缓存和正式 workspace 均未写入。

- Python 插件测试：`12 passed`。
- 移动模块测试：`5 passed`，覆盖类型语义、展开 ARIA/inert、筛选失败后恢复成功。
- Pyright：`0 errors, 0 warnings`。
- `plugin-doctor proactive_feedback@mobile-lab`：healthy。
- watcher 热加载 generation 6，再以无障碍修复更新为 generation 7；没有重启 Agent、Android Activity 或 WebSocket。

Pixel 7 连接独立 Mobile Lab 后验证：

1. 插件目录从 1 个运行中插件原位更新为 2 个，并出现插件自有“主动反馈”入口。
2. 空数据库显示真实空态，不伪造继续率或事件。
3. 只读复制正式反馈 DB 到隔离 workspace 后，概览显示 `707` 条、`30%` 继续率、`53` 次明确引用和 `158` 个高可信信号。
4. “全部 / 继续 / 引用”筛选通过真实 `plugin.ui.call` 返回 `707 / 160 / 53` 条。
5. 展开真实事件后显示“主动发出 → 用户回应 → 助手继续”，再次点击收起。
6. 第一轮真机发现视觉折叠内容仍进入 Android 无障碍树；补 `inert + aria-hidden` 后，折叠态不再出现三个关系节点，展开态只暴露当前一条，收起后再次消失。
7. Android 和服务端日志无 FATAL、RenderProcessGone、协议间隙、插件 RPC 或 WebView 异常。

截图证据：

- `/tmp/pixel7-proactive-plugin-list.png`
- `/tmp/pixel7-proactive-empty-panel2.png`
- `/tmp/pixel7-proactive-populated.png`
- `/tmp/pixel7-proactive-filter-follow.png`
- `/tmp/pixel7-proactive-expanded-final.png`
- `/tmp/pixel7-proactive-a11y-collapsed.png`

## 隔离与恢复

- 正式 `/home/huashen/.akashic/workspace` 只通过 SQLite 只读快照读取，没有写操作。
- Mobile Lab 插件目录备份：`/mnt/data/coding/backups/mobile-lab-plugin-home-before-proactive-feedback-20260717-0737.tar.gz`。
- Mobile Lab 数据备份：`/mnt/data/coding/backups/mobile-lab-proactive-data-before-20260717-0740/`。
- 隔离 workspace 只导入最近 30 条反馈对应的 90 条消息预览；正式会话和手机日用会话不受影响。

