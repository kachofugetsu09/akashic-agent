# 0145 · 消息出处只判定一次，自动来源共用中心循环

- 状态：accepted / implemented for review
- 日期：2026-11-06
- 关联条款：RUN-001～RUN-003、PLG-003、PLG-006、MEM-009、MEM-011、SES-005、TST-001
- supersedes：无；延续 [0039](0039-react-core-atoms-keep-sources-unprivileged.md) 中“来源保持非特权”的方向

## 背景

Ledger 消息已有 `author` 与 `source` 两列，但“这句话是不是人说的”由各插件自己判断：
Akasha 召回、Akasha 检查面板、Markdown 记忆、Content 的 `is_user_input`、前端九处 `author === "user"`、
fleet 中 status_commands 与 proactive_feedback 各写一份，迁移历史的 `history.provenance` 规则也复制了多份。
更严重的是程序调用（programmatic）以 `author="user"` 写入外部程序的提示词：GitHub Watch 的机器提示
在记忆、摘要和界面里都被当成了用户原话。

学习 lane 的默认值 `("conversation", "programmatic", "legacy-unattributed")` 也在 Akasha 与 Markdown 记忆里各写一份；
Wake 的“近期对话”又写了第三种列表。

Wake 还保留了自己的外层分阶段驱动：每阶段固定工具、阶段 Input、只许内部程序调用的决策工具。
其中模型执行已复用中心 `reply.execute`，但阶段编排、工具授予与决策回收仍是 Wake 私有特权。

## 决定

### 1. 出处在写入器绑定时固定

`author` 就是出处，不另存一列。作者分两类：

| 作者 | 主体（Actor） | 含义 |
|---|---|---|
| `user` | human | 人在通道里写的 |
| `program` | program | 外部程序经公开来源代表自己发言 |
| `assistant` | model | 模型输出 |
| `tool` | tool | 工具结果 |
| `app` | host | 宿主控制 |
| 其余 | automation | 写入插件自选的稳定名字（如 `wake`、`scheduler`），表示该插件自动产生的消息 |

作者已由 `MESSAGE_WRITERS.bind` 固定，单条消息不能另选作者，所以不另存出处列，也不在读取侧重新推断。
不强制作者等于插件名：外置插件有自己的作者名，子任务等运行时也会由其他插件的调用代写。
程序调用改写 `author="program"`。

迁移历史（`legacy-unattributed` / `legacy-attribution-unknown`）在读取时按保全的 `history.provenance.role` 派生主体；
存储行一律不改写，Akasha 的来源摘要与已学习样本保持原身份。

### 2. 读取方只调用一处判定

`plugins/ledger/contract.py` 末尾的出处一节是唯一解释（跨插件只能导入 contract，见 R3）：

- `origin(message) -> Origin(actor, producer)`；
- `is_human_input`：只认人的 Input，用于“用户原话”证据（Markdown 记忆的用户事实）、界面气泡、编辑与重试；
- `is_principal_input`：人或外部程序的 Input，用于召回触发与回合起点，保持程序调用原有的召回行为；
- `DIALOGUE_SOURCES`：对话 lane 的唯一默认值，学习与对话上下文共用。

插件不再比较作者字符串，也不再各自解析 `history.provenance` 的角色。
界面行携带服务端派生的 `actor`，前端只读这个字段。

### 3. 自动来源共用中心循环

Wake 不再拥有私有阶段驱动。筛选、调查、分享写成内置 skill，由同一个 `reply.execute` 以普通工具搜索解锁工具；
决策工具按会话出处与候选快照校验，不再按“调用者是不是 Wake 内部程序”放行。
每次 Wake 仍新建一个 internal / excluded 会话，上下文保持很浅；归一化的是循环与出处，不是会话。

## 代价与边界

- 程序调用的新输入不再是用户原话：Markdown 记忆不能以其建立用户事实，摘要与界面把它显示为程序输入。
  召回与学习资格不变（仍由会话 `learning` 属性和对话 lane 决定）。
- 旧 programmatic 行仍是 `author="user"`，按存储原样解释为人；不做历史改写。
- 只改新消息的模型渲染，避免已缓存前缀一次性失效。

## 验收

隔离实例真实发消息：对话输入仍为气泡并可编辑、重试；程序调用输入显示为程序事件行；
Wake 一轮跑完筛选到分享，会话仍是新建的内部会话；Akasha 召回在对话与程序调用中照常触发。
