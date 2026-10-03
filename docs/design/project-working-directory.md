# Project 默认目录与 Session 当前目录

- 状态：accepted / implementation in progress
- 语义与理由：[0084](../decisions/0084-project-default-and-session-working-directory.md)
- 关联条款：SES-011、SH-004、CTX-009

## Owner 和创建边界

Projects 的 owner_records 保存默认目录，绑定只在同一事务里从 null 写入 path。rename/archive 合并当前记录，不覆盖已绑定字段。

SessionAdmission 注册普通 owner 初始化函数。MessageLog 只有插入新 Session 时才在同一个 SQL 事务内执行各 owner 初始化；失败回滚 Session 和 owner 记录，重试已有 Session 不再次初始化。旧 Session 缺少目录记录表示旧的未指定状态，不运行全库迁移。Web scoped 输入、CLI conversation、programmatic、scheduler、wake 和 subagent 仍走各自既有接纳路径，不由 scope 校验函数写 cwd。

后续工具插件拥有 Session 当前目录；Projects 只注册默认取值查询。Core 不解释 Project、路径或 AGENTS。文件 backend 提供路径解析、探测、目录页和有界原文读取，业务规则发现由工具插件组合。

```text
┌─ SessionAdmission 创建 ─┐    ┌─ Projects 默认查询 ─┐
└───────────┬───────────┘    └──────────┬──────────┘
            └────────────┬────────────┘
                         ▼
┌─ standard_tools：Session 目录唯一 owner ─┐
│ 请求材料       工具 prepare       只读 UI │
└────────────────────────────────────────┘
```

## 持久状态和失败

| 对象 | 增加 | 原位更新 | 逻辑失效 | 物理减少与恢复 |
|---|---|---|---|---|
| Project 默认目录 | 显式绑定一次 | null → path | 探测失败保留路径 | 不自动减少；workspace 备份恢复 |
| Session cwd | 创建时一次快照 | 成功切换递增 revision | 缺失/离线保留路径 | 不自动减少；owner 数据随 workspace 保留 |
| AGENTS 材料 | 每次新请求读取 | 替换临时材料 | 删除/不可读撤下旧规则 | 只释放请求投影，不改历史 |
| 外部 worktree | 现有 Shell 明确创建 | 普通 Git 操作 | 实际 Git/文件错误 | 不自动删除、prune、reset 或复制 dirty 内容 |
| 消息与记忆 | 既有追加/学习 | 本功能不获得改写权 | 不因目录失效而失效 | 本功能不获得删除权 |

取消未提交操作不落盘；提交后按 durable receipt 判定。路径失效与未指定不同，恢复时仍探测原路径；Shell/PTY 保持启动目录。准备过的参数不因后来切换而重新解析。规则读取失败允许聊天并明确要求暂停依赖规则的仓库修改。

## 实施与验收边界

第一层接通通用创建事务钩子和长期合同；尚未接通目录功能。后续依次接通 backend、目录 owner、Project API、文件/Shell prepare、独占批次、AGENTS 材料和界面。各层以独立 stacked PR 交付，栈顶核对累计行为。

真实隔离组件/API 验证创建回滚和重试、并发绑定、消息内容/身份/顺序、Session 独立、worktree 切换、prepared 恢复、PTY 与失效路径。Host Bridge 必须实测宿主执行，不能以容器内 exists 判断替代。UI 另核对键盘、取消、错误、长路径、窄屏、字体和草稿保持。正式状态与部署不在本实现任务范围。
