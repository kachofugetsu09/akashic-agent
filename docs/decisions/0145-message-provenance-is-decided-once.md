# 0145 · 消息出处只判定一次，自动来源共用中心循环

- 状态：accepted / implemented for review
- 日期：2026-11-06
- 关联条款：RUN-001～RUN-003、PLG-003、PLG-006、MEM-009、MEM-011、SES-005、TST-001
- supersedes：无；延续 [0039](0039-react-core-atoms-keep-sources-unprivileged.md) 中“来源保持非特权”的方向

## 背景

Ledger 消息已有 `author` 与 `source` 两列，但“这句话是不是用户说的”由各插件自己判断：
Akasha 召回、Akasha 检查面板、Markdown 记忆、Content 的 `is_user_input`、前端九处 `author === "user"`、
fleet 中 status_commands 与 proactive_feedback 各写一份，迁移历史的 `history.provenance` 规则也复制了多份。
迁移历史里用户的输入因为作者不是 `user`，在界面上不能编辑、重试，也不算回合起点。

学习 lane 的默认值 `("conversation", "programmatic", "legacy-unattributed")` 也在 Akasha 与 Markdown 记忆里各写一份；
Wake 的“近期对话”又写了第三种列表。

Wake 还保留了自己的外层分阶段驱动：每阶段固定工具、阶段 Input、只许内部程序调用的决策工具。
其中模型执行已复用中心 `reply.execute`，但阶段编排、工具授予与决策回收仍是 Wake 私有特权。

## 决定

### 1. 出处在写入器绑定时固定

`author` 就是出处，不另存一列。作者分两类：

| 作者 | 主体（Actor） | 含义 |
|---|---|---|
| `user` | user | 经任一渠道从外部进入的输入：Web、Telegram、QQ、飞书、程序调用 |
| `assistant` | model | 模型输出 |
| `tool` | tool | 工具结果 |
| `app` | host | 宿主控制 |
| 其余 | automation | 写入插件自选的稳定名字（如 `wake`、`scheduler`），表示该插件自动产生的消息 |

作者已由 `MESSAGE_WRITERS.bind` 固定，单条消息不能另选作者，所以不另存出处列，也不在读取侧重新推断。
不强制作者等于插件名：外置插件有自己的作者名，子任务等运行时也会由其他插件的调用代写。

程序调用也是一种渠道，不再细分“人还是程序”：渠道背后是谁，由渠道在接纳会话时声明的会话属性约束
（programmatic 默认 internal / excluded，不进入学习，除非调用方显式 `persist_memory`）。
真正需要区分的是“外部经渠道进入”与“插件自己产生”，前者是 `user`，后者是插件名。

迁移历史（`legacy-unattributed` / `legacy-attribution-unknown`）在读取时按保全的 `history.provenance.role` 派生主体；
存储行一律不改写，Akasha 的来源摘要与已学习样本保持原身份。

### 2. 读取方只调用一处判定

`plugins/ledger/contract.py` 末尾的出处一节是唯一解释（跨插件只能导入 contract，见 R3）：

- `origin(message) -> Origin(actor, producer)`；
- `is_user_input`：用户的 Input，用于召回触发、“用户原话”证据（Markdown 记忆的用户事实）、回合起点、界面气泡、编辑与重试；
- `DIALOGUE_SOURCES`：对话 lane 的唯一默认值，学习与对话上下文共用。

插件不再比较作者字符串，也不再各自解析 `history.provenance` 的角色。
界面行携带服务端派生的 `actor`，前端只读这个字段。

### 3. 自动来源共用中心循环

Wake 不再拥有私有阶段驱动。筛选、调查、分享写成内置 skill，由同一个 `reply.execute` 以普通工具搜索解锁工具；
决策工具按会话出处与候选快照校验，不再按“调用者是不是 Wake 内部程序”放行。
每次 Wake 仍新建一个 internal / excluded 会话，上下文保持很浅；归一化的是循环与出处，不是会话。

## 代价与边界

- 现有消息的解释不变：conversation 与 programmatic 的输入本来就是 `author="user"`；只有迁移历史里的用户输入恢复为用户输入。
- 存储行不做历史改写；只改读取侧的判定位置。
- 升级前已提交、尚未送达的 Content Wake 决定位于旧 `investigate` 阶段；恢复时从该阶段读取，不改写旧会话。

### 已知限制

- 界面行的 `actor` 是必填字段：前端（`static/chat`）与 `plugins/ui` 同属一个发行版、一起发布，
  不存在“新前端读旧服务端”的组合，因此不做缺字段兼容。

## 验收

隔离实例真实发消息：对话与程序调用输入仍为用户输入并可编辑、重试；
Wake 一轮跑完筛选到分享，会话仍是新建的内部会话；Akasha 召回在对话与程序调用中照常触发。
