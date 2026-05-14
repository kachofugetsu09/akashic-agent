# Drift 系统指南

## 先理解它是什么

Drift 是一个**你写模型可以做什么、模型照着执行**的后台任务系统。

- **什么时候跑**：proactive 拉了一圈啥也没有（无 alert、无 content、无 context fallback）
- **做什么**：你写在 `drift/skills/<skill-name>/SKILL.md` 里的事
- **怎么做**：SKILL.md 是一份分步操作指南——先读哪个文件、跑什么脚本、怎么判断、什么时候发消息——模型一步步按着走
- **跟 proactive 的本质区别**：proactive 的行为是代码里写死的 system prompt，drift 的行为是你写的 SKILL.md

**一个 drift skill 就是一个 agent run**：它拿到一套工具（read_file / write_file / shell / fetch_messages / message_push...），拿着你写的 SKILL.md 当 system prompt，一步一步执行，最后调 `finish_drift` 收尾。

## 什么是 Drift（展开）

当 proactive gateway 拉完数据发现三路都空时，agent 不空转，而是进入 Drift 模式利用空闲时间做后台工作。简单说：**没新闻可推的时候就干点后台活儿**。

```
tick
  └── DataGateway.run() → 无 alert / 无 content / 无 context
       └── DriftTurnPipeline.run()
            ├── scan_skills()      读取 drift/skills/*/SKILL.md
            ├── filter_skills()   跳过 requires_mcp 未满足的 skill
            ├── build_context()   注入记忆、近期上下文、skill 列表
            └── tool_loop(max_steps)
                 ├── read_file / write_file / edit_file
                 ├── recall_memory / web_fetch / web_search
                 ├── fetch_messages / search_messages / shell
                 ├── message_push      最多一次
                 ├── mount_server      可挂载 MCP server
                 └── finish_drift      必须声明 message_result
```

## Drift 的核心约束

1. **每次重新选择**：不默认继续上次的 skill，每轮重新比较所有 skill
2. **message_push 限制**：最多推送一次用户消息；推送后只允许 write_file / edit_file / finish_drift 收尾
3. **必须 finish_drift**：执行结束前必须调用，填写 `message_result`
4. **message_result 与实际一致**：
   - `"sent"` — 本轮成功调用了 `message_push`
   - `"silent"` — 本轮没有推送消息
5. **到达 max_steps 不强收**：如果模型没主动调 finish_drift，本轮保持未完成
6. **最小间隔**：`drift.min_interval_hours` 控制连续两次 drift 的最小间隔

---

## Drift Skill 格式

每个 skill 是一个目录，放在 `~/.akashic/workspace/drift/skills/<skill-name>/` 下，核心文件是 `SKILL.md`。

### 哪些文件你写、哪些 agent 写

| 文件 | 维护方式 | 说明 |
|------|---------|------|
| `drift/skills/<name>/SKILL.md` | **你写** | drift 任务定义，agent 每轮当 system prompt 读。也可以让主 agent 用内置 skill `create-drift-skill` 帮你生成 |
| `drift/skills/<name>/state.json` | **agent 写** | skill 执行时自动更新状态，不用管 |
| `drift/skills/<name>/*.md` | **agent 写** | 工作文件（audited.md、queue.md、backlog.md 等），skill 执行时自动读写 |
| `drift/skills/<name>/scripts/*.py` | **你写** | 固定脚本，skill 通过 `shell` 工具调用 |
| `drift/drift.json` | **agent 写** | DriftTurnPipeline 自动写运行记录（recent_runs），不用管 |
| `drift/drift_note.md` | **agent 写** | 跨轮次的自由笔记，agent 可读可写 |

内置了一个 skill 放仓库里（`agent/skills/`），用来创建新的 drift skill。

### SKILL.md 结构

```yaml
---
name: <skill-name>
description: <一句话描述>
---

## 目标

## 工作文件
（列出这个 skill 会读写的工作文件路径）

## 工作流程
1. ...
2. ...

## 要求
- 约束和规则
```

---

## 真实案例

### 案例一：audit-dirty-memories（记忆审计）

**目标**：随机抽检一条带 `source_ref` 的长期记忆，回溯原始消息，判断记忆摘要是否准确。

**工作流程**：
1. 脚本抽样（`sample_memory_for_audit.py`）→ 随机选一条未审计的记忆
2. `fetch_messages` 读取原始消息上下文
3. 对比摘要与原文做"高置信可疑判断"
4. 干净 → 静默记录（`message_result="silent"`）
5. 可疑 → 发消息告诉用户哪条记忆为什么可疑（`message_result="sent"`）

**实际运行记录**（`drift.json`）：
```json
{
  "skill": "audit-dirty-memories",
  "run_at": "2026-05-08T14:10:48Z",
  "one_line": "审计记忆 7cf7657414cb：摘要声称 Falcons 阵容查询，source_ref 却是测试消息，内容完全不匹配，判定可疑已报告",
  "message_result": "sent"
}
```

**工作文件**：
- `audited.md`：已审计的 memory_id 列表（防止重复）
- `state.json`：运行状态
- `scripts/sample_memory_for_audit.py`：固定抽样脚本

### 案例二：explore-curiosity（好奇心探索）

**目标**：补足用户画像中的生活化信息空白，一次只问一个轻量、自然的问题。

**工作流程**：
1. 读 `queue.md`，取第一行问题
2. `message_push` 发送
3. 删除该行，写回剩余问题

如果 queue 为空，基于长期记忆生成 5 个问题放入 queue。

**实际运行记录**：
```json
{
  "skill": "explore-curiosity",
  "run_at": "2026-05-08T20:02:54Z",
  "one_line": "推送睡前小说话题，更新队列",
  "message_result": "sent"
}
```

**queue.md 示例**：
```
有没有什么你最近特别想买的东西，或者一直在忍住没买的？
```

**规则**：
- 问题必须轻量、自然、像朋友随口一问
- 优先问：音乐偏好、开源项目、运动习惯、食物口味、日常消遣
- 禁止问太大、太虚、太像采访的问题
- 避开长期记忆里已经明确有答案的信息

### 案例三：review-drift-gaps（Drift 自我反思）

**目标**：定期回顾 Drift 全局行动历史，找出一直没推进的方向，维护有界 backlog（最多 10 项）。

**工作流程**：
1. 读 `drift.json` 获取最近运行历史
2. 扫描各 skill 的 `state.json` 获取当前状态
3. 对比上轮快照，判断停滞方向（长期无进展 / 阶段停滞 / 待办未推进）
4. 更新双计数器（停滞轮数 / 恢复轮数）
5. 按优先级排序后写回 `backlog.md`

**核心逻辑**：
- 显式跳过自身（review-drift-gaps）
- 停滞轮数越多的方向排越前
- 连续恢复 3 轮的方向自动从 backlog 移除
- 不调用 message_push，纯后台记录

---

## 写自己的 Drift Skill

### 最小示例

```markdown
---
name: my-skill
description: 每天备份一次 conversation 精华到 notion
---

## 目标
定期把用户最近对话中的值得回顾的内容同步到 notion 数据库。

## 工作流程
1. `fetch_messages` 获取最近 24 小时的对话
2. 提取值得回顾的内容（用户明确提到的计划、决策、偏好变化）
3. 如果有可同步的内容 → write_file 写入 notion API 格式 → shell 调用 notion API
4. 没有新内容 → 静默结束

## 工作文件
- `skills/my-skill/last_sync.md`：上次同步时间

## 要求
- 不调用 message_push（此 skill 纯后台）
- finish_drift(message_result="silent")
```

### 关键工具

| 工具 | 用途 |
|------|------|
| `read_file` / `write_file` / `edit_file` | 读写 drift 工作文件 |
| `recall_memory` | 检索长期记忆 |
| `fetch_messages` | 读取被动对话历史 |
| `web_fetch` / `web_search` | 获取外部信息 |
| `shell` | 运行脚本 |
| `message_push` | 推一条消息给用户（最多一次） |
| `finish_drift` | 保存状态并结束本轮 |

### 注意事项
- `finish_drift` 的 `message_result` 必须和本轮实际动作一致
- `finish_drift.next` 写这个 skill 下次该做什么，不写全局调度愿景
- 把状态写在 `state.json` 里，把数据写在专门的工作文件里
- 只读工具（read_file / fetch_messages 等）放在前面，写操作放在后面
- 如果 skill 需要 MCP server，可以用 `mount_server` 挂载
