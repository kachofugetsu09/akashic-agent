# 移动端运行时检查设计

## 1. 目标

移动端提供一个只读入口，查看当前 workspace 的六份固定 Markdown、当前
`SchedulerService` 任务，以及当前已发布插件快照中的插件、Skills 与 MCP。
该入口不提供编辑、启停、执行或任意路径读取能力。

## 2. Owner 与调用链

```text
┌──────────────────────┐
│ Mobile WebSocket     │  校验 command 与 payload
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ RuntimeInspection    │  固定文档 allowlist、稳定 JSON 投影
│ Service              │
└──────┬────────┬──────┘
       │        │
       ▼        ▼
┌────────────┐ ┌──────────────────────┐
│ Scheduler  │ │ RuntimeSnapshotStore │
│ Service    │ │ lease                │
└────────────┘ └──────────────────────┘
```

- 文档只允许 `RuntimeInspectionService` 中声明的六个 ID，不能传文件路径。
- 定时任务只读 `SchedulerService.list_jobs()`，不直接反序列化
  `schedules.json`。
- 插件、Skills 与 MCP 在一次 `RuntimeSnapshotStore` 租约内投影，确保同一
  回复不混用两个 generation。
- MCP 只暴露 server、工具名、描述与输入 schema；不暴露 command、env、
  cwd、认证材料或配置。

## 3. 协议命令

| 命令 | 请求字段 | 成功结果 |
|---|---|---|
| `runtime.document.list` | 无 | 六份固定文档摘要与可用状态 |
| `runtime.document.get` | `document_id` | 文档元数据与 Markdown |
| `scheduler.job.list` | 无 | 当前任务摘要 |
| `scheduler.job.get` | `job_id` | 当前任务详情与 Markdown |
| `runtime.capability.list` | 无 | snapshot、插件、Skills、MCP 列表 |
| `runtime.mcp.get` | `owner_id`, `server_name` | MCP 工具详情与 Markdown |

所有命令复用移动协议已有 command receipt 与最大帧校验。文档正文限制为
192 KiB，避免成功读取后产生无法投递的回复。

## 4. VEDA 大写迁移

新真源为 `memory/VEDA.md`，默认模板为 `prompts/VEDA.md`。迁移状态如下：

| 迁移前状态 | 行为 |
|---|---|
| 只有合法小写文件 | 备份后原子重命名 |
| 只有合法大写文件 | 已满足 |
| 两份合法且字节相同 | 备份后删除小写副本 |
| 两份内容不同 | 阻止启动并报告冲突 |
| 任一文件为空、损坏或不可读 | 阻止启动 |
| 已有 workspace 两份都不存在 | 阻止启动 |

回滚只在目标摘要仍与迁移时相同时执行，不覆盖迁移后的用户修改。

## 5. 验收

1. 任意路径不能绕过六个文档 ID allowlist。
2. 定时任务列表反映 service 内存中的当前状态。
3. 热重载期间每个能力回复来自单一 snapshot lease。
4. MCP 回复不包含启动命令、环境变量和本地路径。
5. 大小写迁移覆盖重命名、去重、冲突、损坏和安全回滚。
6. checked-in 移动协议 schema 与服务端模型生成结果一致。
