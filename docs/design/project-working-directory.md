# Project 默认目录与 Session 当前目录

- 状态：accepted / implemented; isolated acceptance complete
- 语义与理由：[0084](../decisions/0084-project-default-and-session-working-directory.md)
- 关联条款：SES-011、SH-004、CTX-009

## Owner 和创建边界

Projects 的 owner_records 保存默认目录，绑定只在同一事务里从 null 写入 path。rename/archive 合并当前记录，不覆盖已绑定字段。

SessionAdmission 注册普通 owner 初始化函数。MessageLog 只有插入新 Session 时才在同一个 SQL 事务内执行各 owner 初始化；失败回滚 Session 和 owner 记录，重试已有 Session 不再次初始化。旧 Session 缺少目录记录表示旧的未指定状态，不运行全库迁移。Web scoped 输入、CLI conversation、programmatic、scheduler、wake 和 subagent 仍走各自既有接纳路径，不由 scope 校验函数写 cwd。

standard_tools 拥有 Session 当前目录；Projects 只注册默认取值查询。Core 不解释 Project、路径或 AGENTS。文件 backend 提供路径解析、探测、目录页和有界原文读取，业务规则发现由工具插件组合。

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

创建事务、物理 backend、目录 owner、Project API、文件/Shell prepare、独占批次、AGENTS 材料和界面均已接通。各层独立 stacked PR，正式部署与主审验收仍是独立动作。

真实隔离组件/API 验证创建回滚和重试、并发绑定、消息内容/身份/顺序、Session 独立、worktree 切换、prepared 恢复、PTY 与失效路径。Host Bridge 必须实测宿主执行，不能以容器内 exists 判断替代。UI 另核对键盘、取消、错误、长路径、窄屏、字体和草稿保持。正式状态与部署不在本实现任务范围。


## API 与工具合同

`standard_tools.working_directory.v1` 是普通插件能力。Projects 可独立加载；没有目录能力时仍能创建未绑定项目，绑定入口明确报错。已绑定项目不能在缺少目录 owner 的组合里新建 Session，以免丢失默认目录。运行 workspace 仍保存聊天、记忆、artifact 和回执，不被代码目录替换。

Project UI 查询 `project.bind_directory` 接受 Project ID 和执行主机的绝对路径。backend 规范化并校验已有可访问目录后，Projects 在一个事务内执行 null → path；竞争提交只有一个值获胜。同值重试返回原记录，不再次改写。`project.directory` 返回路径和实时状态，`directory.browse` 只返回一页直接子目录。

`directory.current` 以真实 Session ID 返回 path、revision、实时状态与 AGENTS 来源，不返回规则正文。`set_working_directory` 不接受任意 Session ID，只从实际 CallSource 取所属 Session；prepare 固定绝对目标与 expected revision。目录 owner 使用 CAS，并把 cwd 更新和效果 receipt 同事务保存。同路径切换不增加 revision，失败保留原状态。

切换工具通过通用 `exclusive_batch` 元数据声明独占。ReAct 在解码整批后拒绝所有混合调用；Tools 在 prepare 和任何物理效果之前复核持久 Output，直接提交非法批次也不能产生部分效果。普通工具注册不增加该元数据。

文件 prepare 固定绝对 path 与既有 allowed_dir 限制；默认 read/list 根随已设置的 Session cwd，显式配置的限制保留。Shell prepare 固定 cwd；显式命令 cwd 或插件 working_dir 保持优先。write_stdin 和 task_stop 只找原进程 owner，运行中的 PTY 不因切换而迁移。

## backend 与规则读取

路径解析、目录页、可用性和规则原文共用 File backend。local 在当前执行主机读取；Host Bridge 通过已有 FileTool RPC 增加的 PathInfo 操作在宿主读取，不用容器内 exists 判断宿主。RPC 的结构与大小校验集中在物理边界，业务插件不复制 host client。

规则从 cwd 向上寻找最近的 .git 文件或目录，再从该根逐层读到 cwd。没有 Git 标记时只读 cwd。每层只采用 AGENTS.override.md 或 AGENTS.md 之一，累计不超过 32 KiB，不扫描整个子树。材料要求 Agent 在修改更深子目录前读取那一层适用规则。删除、权限错误、超限和主机离线分别返回当前规则状态；错误不保留部分规则供执行。

规则提醒声明 `replay=false`：当前请求有正文，但 model.facts 不保存 reminder。存在实时提醒时 Context 以完整消息投影重新构造请求，不接续可能保留旧规则的 opaque provider continuation。未设置目录时不贡献规则提醒，保留既有请求行为。代价是放弃带目录材料请求的 provider 会话续接优化；普通历史、工具 call ID、真实回执和模型调用审计仍保留。已冻结的失败/恢复请求依然按既有 ReAct prepare 合同恢复，不因磁盘后来变化重新付费。

## UI 与验证

Project 行只有一个目录入口：未绑定时浏览执行主机、展示完整候选路径和不可改绑说明，再明确提交；已绑定时只读查看状态。取消、Escape 和失败不创建目录。Session 只读栏显示当前路径及 AGENTS 来源，工具回执、窗口重新聚焦和可见页面轮询刷新状态，不重挂 composer。

隔离真实 API 验证并发绑定、同值重试、旧/未设置 Session 不回填及独立切换。真实 PluginManager、Tools 和 Models 账本配合受控本地 model driver，验证同一 Turn 混合批次整批拒绝、随后单独切换、下一模型请求的新规则及旧规则撤下。实际 Git worktree、所有相对文件工具、默认 Shell cwd、运行中 PTY 和缺失路径均已核对。

恢复验证在真实 Tools 写入 prepared 回执后退出进程；第二进程切换 cwd；第三进程使用原归档 binding 恢复，仍写入原绝对目标。Host Bridge 的独立 gRPC 进程验证宿主 cwd、浏览、规则读取、有界原文与离线状态。Chromium 使用生产 chat bundle 和真实插件 HTTP API 验证取消焦点、草稿、固定后的只读入口、Session 状态、320px/200% 字体和长路径。其他页面与完整既有检查由 CI/后续主审核对，不能把本地组件验收当作正式部署。
