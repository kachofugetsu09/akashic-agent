# 0085 · Project 固定默认目录，Session 独立当前目录

- 状态：accepted
- 日期：2026-10-03
- 关联条款：SES-010、SES-011、SH-004、CTX-009、WSP-004
- 部分替代：[未来方向草案](../design/akashic-future-roadmap-issue-drafts.md) 第 6 节；补充 [0073](0073-session-scope-routes-akasha-graphs.md)

## 背景与决定

Project 组织聊天和记忆，代码目录只提供执行起点。用户可以永不关联目录；首次关联后默认目录固定，同值重试幂等，改绑和清空明确失败。已有聊天不因后来关联目录发生变化。

Session 在真实创建事务中一次快照默认目录，包括 null 和已经失效的路径。当前目录由普通工具插件独占，切换只改变这个 Session 的执行位置与版本，不改变 scope、Project、其他 Session、消息或记忆。旧 Session 缺少目录记录时保持未指定，不追溯读取 Project 的新默认值。

```text
┌─ Projects：默认目录写一次 ─┐
└────────────┬─────────────┘
             │ 新 Session 创建事务
             ▼
┌─ 工具插件：Session cwd + revision ─┐
│ Prompt 读取         prepare 固定目标 │
└───────────────────────────────────┘
```

Agent 使用现有 Shell 创建 worktree，再显式调用目录切换工具。切换独占工具批次，成功后下一次模型请求刷新 cwd 和 AGENTS。文件相对目标与 Shell cwd 在 prepare 固定；恢复沿原 receipt 执行，不重新读取当前目录。

目录选择、校验、AGENTS 和文件执行使用同一 backend。不存在或暂不可访问的目录保留路径，绝不创建空目录、回到运行 workspace 或原 Project。规则读取失败仍可聊天，但明确暂停依赖规则的仓库修改。

## 取舍与重议条件

不采用 coding/chat 模式、自动 worktree 开关、项目指令编辑器和 Session 目录表单：它们引入与组织身份无关的第二套行为模型。不可变 Session 根会妨碍 Agent 在同 Turn 使用新 worktree，因此只固定 Project 默认值。

不把目录放进 scope：位置变化不是记忆分区变化。AGENTS 只属于请求材料，不进入永久用户消息或全局 persona；规则提醒不重放，带实时材料的请求不接续可能保留旧规则的 opaque provider 会话。每层 override 优先，规则从最近 Git 根到 cwd 依次读取，累计上限 32 KiB；深层规则由 Agent 在进入相关目录前读取。

代价是移走 Project 目录后不能原位改绑；可新建 Project，或显式切换原 Session。若以后需要迁移 Project，必须另定消息、记忆和身份合同，不能借目录设置实现迁移。

## 恢复与验收

新增与切换只增加或原位更新 owner 记录，无自动减少协议。源码用 Git 恢复；正式 owner 状态需要 workspace 一致备份，回退源码不删除新记录。

验收覆盖原子绑定、新旧 Session、并发隔离、同 Turn 刷新、混合批次拒绝、prepared 恢复、失效路径和真实 Host Bridge；UI 覆盖取消、焦点、草稿、窄宽与 200% 字体。调用链及阶段状态见 [技术设计](../design/project-working-directory.md)。
