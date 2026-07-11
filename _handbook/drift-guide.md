# Drift 系统指南

## 先理解它是什么

Drift 是一个**你写模型可以做什么、模型照着执行**的后台任务系统。

- **什么时候跑**：proactive 拉了一圈啥也没有（无 alert、无 content、无 context fallback）
- **做什么**：你写在 `drift/skills/<skill-name>/SKILL.md` 里的事
- **怎么做**：SKILL.md 是一份分步操作指南——先读哪个文件、跑什么脚本、怎么判断、什么时候发消息——模型一步步按着走
- **跟 proactive 的本质区别**：proactive 的行为是代码里写死的 system prompt，drift 的行为是你写的 SKILL.md

**一个 drift skill 就是一个 agent run**：它拿到一套工具（read_file / write_file / shell / fetch_messages / message_push...），带着 runtime 注入的 Drift Briefing，拿着你写的 SKILL.md 当 system prompt，一步一步执行，最后调 `finish_drift` 收尾。

## 什么是 Drift（展开）

当 proactive gateway 拉完数据发现三路都空时，agent 不空转，而是进入 Drift 模式利用空闲时间做后台工作。简单说：**没新闻可推的时候就干点后台活儿**。

```text
┌─ proactive gateway 无 alert / content / context
│  └─ DriftTurnPipeline
│     ├─ 扫描并比较 drift skills
│     ├─ 注入记忆、近期上下文与连续性前情
│     ├─ select_skill 或 idle_drift
│     └─ 执行一个原子动作
│        ├─ 可选 message_push（最多一次）
│        └─ finish_drift
│           ├─ completed / paused
│           ├─ skill continuum
│           ├─ self_update
│           └─ self_observation journal
└─ done
```

## Drift 的核心约束

1. **每次重新选择**：不默认继续上次的 skill，每轮重新比较所有 skill
2. **message_push 是 fire-and-forget**：最多推送一次；成功后本轮动作已经完成，只能调用 `finish_drift`。未来真有用户回答时，它会作为新会话上下文和记忆进入，但 Drift 不保存“等待回答”，也不能推断“用户没回”
3. **必须 finish_drift**：执行结束前必须调用，填写 `status`、`briefing` 和 `self_update`
4. **message_result 由 runtime 记录**：调用过 `message_push` 就是 sent，否则是 silent，不由 skill 自报
5. **status 表示系统自己的进度**：
   - `"completed"` — 本轮小闭环已完成，不强行生成下一步
   - `"paused"` — 本轮没做完，必须在 `scratchpad_update` 写清下次从哪里继续
6. **到达 max_steps 会收尾**：如果模型没主动调 finish_drift，runtime 会进入 wrap-up phase，只允许调用 `finish_drift` 保存接续点
7. **最小间隔**：`drift.min_interval_hours` 控制连续两次 drift 的最小间隔

## Drift 的自我连续性

Drift 会被反复触发。它既要保留当前意图，也要能从多轮行为中形成可修正的暂定认识。

```text
┌─ self_state
│  ├─ 上轮意图与选择原因
│  └─ 宽松的 next_tendency，不是下一轮指令
├─ recent_drift_runs
│  └─ 最近真实做过的活动
└─ self_observation journal
   ├─ question       首次提出暂定观察
   ├─ reinforce      后续重复证据加强
   └─ revise         反例或主动变化修正
```

`finish_drift.self_update` 必须包含：

- `reflection`：本轮与近期行为是什么关系。
- `pattern`：`ordinary`、`repeat`、`change` 或 `contradiction`。
- `next_tendency`：下次可能想做什么的宽松倾向，不能写等待用户回答。
- `observation`：当 pattern 不是 ordinary 时必填，保存 statement、basis 和 effect。

这些观察只属于 Drift 自身的空闲行为，不写入用户长期记忆，也不是稳定人格结论。

---

## Drift Skill 格式

每个 skill 是一个目录，放在 `~/.akashic/workspace/drift/skills/<skill-name>/` 下，核心文件是 `SKILL.md`。

### 哪些文件你写、哪些 agent 写

| 文件 | 维护方式 | 说明 |
|------|---------|------|
| `drift/skills/<name>/SKILL.md` | **你写** | drift 任务定义，agent 每轮当 system prompt 读。也可以让主 agent 用内置 skill `create-drift-skill` 帮你生成 |
| `drift/drift.db` | **runtime 写** | 保存 run、skill continuum、cursor、journal、self_state 和 self_observation |
| `drift/skills/<name>/*.md` | **按 skill 需要读写** | 工作文件（audited.md、读书笔记、临时材料等），不是系统级连续性的唯一来源 |
| `drift/skills/<name>/scripts/*.py` | **你写** | 固定脚本，skill 通过 `shell` 工具调用 |

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
4. 干净 → 静默记录
5. 可疑 → 发消息告诉用户哪条记忆为什么可疑

**实际运行记录**（`drift.db`）：
```json
{
  "skill": "audit-dirty-memories",
  "run_at": "2026-05-08T14:10:48Z",
  "status": "completed",
  "briefing": "审计记忆 7cf7657414cb：摘要声称 Falcons 阵容查询，source_ref 却是测试消息，内容完全不匹配，判定可疑已报告",
  "message_result": "sent"
}
```

**工作文件**：
- `audited.md`：已审计的 memory_id 列表（防止重复）
- `scripts/sample_memory_for_audit.py`：固定抽样脚本
- 连续性前情：由 `drift.db` 的 `skill_continuum` 保存；如果一轮抽到候选但没判断完，用 `status="paused"` 写清 memory_id、source_ref 和当前阶段

### 案例二：explore-curiosity（好奇心探索）

**目标**：补足用户画像中的生活化信息空白，一次只问一个轻量、自然的问题。

**工作流程**：
1. 阅读 Drift Briefing 里的本 skill 前情，避免短期重复
2. 基于长期记忆、最近上下文和当前状态，现场判断一个轻量自然的问题
3. 如果适合聊天，`message_push` 发送
4. 如果不适合打扰，`finish_drift(status="completed")` 静默闭环

**实际运行记录**：
```json
{
  "skill": "explore-curiosity",
  "run_at": "2026-05-08T20:02:54Z",
  "briefing": "推送睡前小说话题",
  "status": "completed",
  "message_result": "sent"
}
```

**规则**：
- 问题必须轻量、自然、像朋友随口一问
- 优先问：音乐偏好、开源项目、运动习惯、食物口味、日常消遣
- 禁止问太大、太虚、太像采访的问题
- 避开长期记忆里已经明确有答案的信息
- `message_push` 成功后本轮立即闭环，不保存“等待回答”。用户以后真的回答时，由会话和记忆链路自然关联

### 案例三：review-drift-gaps（Drift 自我反思）

**目标**：定期回顾 Drift 全局行动历史，找出长期 paused 或反复失败的方向。

**工作流程**：
1. 读取 `drift.db` 里的 recent runs 和 skill_continuum
2. 找出长期 paused 或最近频繁失败的 skill
3. 生成轻量健康摘要
4. `finish_drift(status="completed", self_update={...})`

**核心逻辑**：
- 显式跳过自身（review-drift-gaps）
- 不再把各 skill 的 `state.json` 当权威来源
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
- `skills/my-skill/last_sync.md`：可选工作文件。跨轮连续性优先使用 Drift Briefing 和 `finish_drift` 的 scratchpad_update。

## 要求
- 不调用 message_push（此 skill 纯后台）
- 完成同步后调用 `finish_drift(status="completed", briefing="同步了 X 条内容", self_update={...})`
- 如果没做完，调用 `finish_drift(status="paused", briefing="同步中断", scratchpad_update="下次从 ... 继续", self_update={...})`
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
- `finish_drift.status` 必须是 `completed` 或 `paused`
- `completed` 不需要编造执行断点；`paused` 必须写 `scratchpad_update`
- 连续性前情由 runtime 写入 `drift.db`，下一轮通过 Drift Briefing 注入
- 工作文件可以继续使用，但不要把它们写成模型必须手工维护的复杂状态机
- 只读工具（read_file / fetch_messages 等）放在前面，写操作放在后面
- 如果 skill 需要 MCP server，可以用 `mount_server` 挂载
