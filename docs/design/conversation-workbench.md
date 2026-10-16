# 对话工作台：会话上下文、意图路由与正交标签

决策见 [0121](../decisions/0121-conversation-workbench-tabs.md)，需求来源 #1419。本文给出合同、事实 owner 与切片顺序；
每个切片落地后在这里补最终合同与验证，未落地的内容不代表已完成。规格评审（brief-25）的取舍记录在第 7 节。

## 1. 分层

```text
┌──────────────────────────── 聊天 iframe（/chat） ─────────────────────────────┐
│ 当前会话 ─┐   附件卡片 / 插槽按钮 ─┐ (点击处理函数内)      输入框草稿 ◀─┐       │
└───────────┼────────────────────────┼────────────────────────────────────┼───────┘
            │ akashic.chat-session   │ akashic.workbench-intent           │ akashic.composer-quote
┌───────────▼────────────────────────▼────────────────────────────────────┴───────┐
│ conversation-ui：展开/宽度/当前标签/窄屏 + 转交会话 ID + 按 accepts 路由 + 核对会话后转发引用 │
└───────┬──────────────────┬──────────────────┬──────────────────┬────────────────┘
        │ conversation.tools.v2：每个标签只看到自己的 view
   ┌────▼────┐       ┌─────▼──────┐     ┌─────▼──────┐     ┌─────▼─────┐
   │ 电脑     │       │ 终端        │     │ 改动        │     │ 预览       │
   │computer │       │standard_    │     │workspace_   │     │preview_ui │
   └────┬────┘       │tools 自有   │     │changes      │     └─────┬─────┘
        │            └─────┬──────┘     └─────┬──────┘           │ ARTIFACT_READ
        ▼                  ▼ 自有记录          ▼ host.processes.v1  ▼
   computer workload   jobs.notices.v1    WORKING_DIRECTORY     ledger
                       (停止后台命令)       TURN_PROJECTION 等
```

规则：
- 宿主不认识任何标签、意图或事实的含义；增减能力只增减插件。
- 标签之间不通信；跨插件只经 owner 的 `contract.py` 窄服务（PLG-016）。
- 事实只有一个 owner：命令记录与终端标签都归 `standard_tools`（shell 工具的 owner），轮次改动归 `workspace_changes`，附件归 ledger。
  命令记录只有一个消费者，按 0111 不发布跨插件合同，出现第二个消费者时再由 owner 发布。
- 不做全局工具事件总线；消费者不解析别人的工具结果文本。

## 2. 宿主合同 `conversation.tools.v2`

合同文件 `packages/akashic-conversation-tools-v2/contract.json`，摘要取文件 sha256，与其它挂载点一致。
v1 只有电脑一个消费者：S2a 把电脑迁到 v2 并同时删除 v1，不保留双 ABI。

```ts
interface WorkbenchTab {
  id: string; label: string; order?: number;
  icon?: string;                       // 16px 线性 SVG，启动器与窄屏使用
  accepts?: readonly string[];         // 能处理的意图名，例如 "workspace.changes.open.v1"
  render(host: HTMLElement, view: WorkbenchTabView): Disposer;   // 首次被选中时才调用
}
interface WorkbenchTabView {
  readonly active: boolean; onActiveChange(listener: (active: boolean) => void): Disposer;
  readonly session: string | null; onSessionChange(listener: (session: string | null) => void): Disposer;
  requestAttention(noticeId: string): void;
  onIntent(listener: (intent: WorkbenchIntent) => IntentStatus | Promise<IntentStatus>): Disposer;
  quote(request: QuoteRequest): QuoteStatus;
}
interface WorkbenchIntent { id: string; name: string; session: string; payload: unknown }  // payload JSON ≤ 16 KiB
type IntentStatus = "handled" | "invalid" | "failed";
interface QuoteRequest { session: string; text: string; source: string }  // text ≤ 8 KiB，source ≤ 200 字
type QuoteStatus = "inserted" | "stale" | "rejected";
```

- **会话**：聊天 iframe 在会话切换时发 `akashic.chat-session {sessionId|null}`；宿主只转交，不缓存业务数据。
  两端都核对 `event.origin` 与 `event.source`，沿用现有 `akashic.rail-action-activate` 的做法。
- **懒挂载**：标签首次被选中才 `render`；之后切走只发 `onActiveChange(false)`，标签自行暂停后台工作。
- **按会话记忆**：当前标签和展开状态按会话 ID 记在 `sessionStorage`；宽度仍全局一份，双击分隔条恢复默认。
- **启动器与铺满**：顶栏右侧依次为"+"（列出全部标签与图标）、铺满/还原、关闭。铺满时对话列收起，`Esc` 或再次点击还原。
  可见按钮 28px，窄屏触控热区 44px（0051）。

### 2.1 意图

1. 聊天端只在点击或键盘激活的处理函数里同步发出 `akashic.workbench-intent {id, name, session, payload}`；
   不用 `navigator.userActivation` 作为来源证明。Agent 侧事件的代码路径不调用这个入口，这是“只有用户能打开工具区”的保证方式。
2. 宿主校验外层：名字格式、JSON 字节上限、`session` 等于当前会话。不通过时回 `invalid`。
3. 按标签顺序选第一个 `accepts` 含该名字的标签（稳定的第一接受者，不广播），展开工具区、激活标签、投递意图。
4. 宿主异步回 `akashic.workbench-intent-result {id, status}`，`status` 为 `handled | unhandled | invalid | failed`。
   `unhandled` 与 `failed` 时聊天端回到原有行为（例如附件预览弹窗）。
5. 宿主把当前可接受的意图名集合以 `akashic.workbench-intents {names}` 告诉聊天端，只用来决定是否显示“在侧栏打开”，不代表一定成功。
6. 意图名带版本（`.v1`）。payload 的结构由该意图的领域 owner 在自己的 README 与 `contract.py` 里定义，例如
   `workspace.changes.open.v1` 由 workspace_changes 定义 `{turn_id, path?}`。接收标签校验领域引用和读取权限。
   聊天插槽上下文增加 `workbench: { accepts(name): boolean; open(name, payload): Promise<status> }`，由聊天端转发。

### 2.2 引用回输入框

- `quote` 只能在标签自己的点击或键盘处理函数里调用，并带上发起时记下的 `session`。
- 宿主在三种情况下拒绝：`session` 不是当前会话（返回 `stale`），标签已被撤销，或长度超限（返回 `rejected`）。
  通过后转发 `akashic.composer-quote {session, text, source}`。
- 聊天端再次核对当前会话，然后调用现有 `insertDraft(text, {append:true})`，文本前加来源行，例如 `> 改动 · src/a.ts:12-20`。
  不新增 Input 内容种类，也不自动发送。
- 同一 realm 里的插件是安装时受信任的代码（0051）。这些校验保护的是交互正确性（异步旧结果不会写进别的会话），不构成恶意插件沙箱。

### 2.3 Agent 活动

Agent 活动只经 `requestAttention` 亮徽标；宿主不因活动展开工具区（0092）。

## 3. 终端（standard_tools 自有）

首版是只读命令记录，按轮分组显示：命令、cwd、状态、退出码、耗时、输出。

- **记录**：shell 工具在 prepare 固定 cwd（SH-004）后写入 `running`，每次工具调用读到输出、或执行结束时更新。
- **输出口径**：首版的输出就是 Agent 通过工具已经读到的部分。后台命令在 Agent 下次读取前不会出现新输出，界面注明“显示 Agent 已读取的输出”。
  实时滚动需要 host_execution 提供不推进读取位置的有界输出观察合同，另立决策（S4b）。
  不另起轮询 `write_stdin`，否则会抢走 Agent 的输出。
- **停止**：后台命令走现有 `JOB_NOTICES.stop_job(session_id, job_id)`，与聊天里的任务列表是同一个入口。
  前台命令不提供停止，由对话的中断处理。
- **存储**：standard_tools 的 OWNER_STATE，每会话固定 200 个槽位 `command:{session}:{slot}`，循环覆盖。
  OwnerStore 没有 delete，所以固定槽位就是物理上限。`running` 记录被覆盖前先标 `unknown`，避免完成时复活旧槽。
  命令每条 ≤ 4 KiB，输出尾部 ≤ 16 KiB，超出部分截断并标注。
- **故障**：进程重启后无法确认结局的记录标 `unknown`，不猜成功。
- **隐私**：记录内容与对话里的工具结果相同，没有新增数据源，保存在同一个 sessions.db，备份范围与对话相同。插件停用时不采集。
- **路由与标签**：dashboard 路由 `GET /api/dashboard/standard-tools/commands?session=&before=` 与停止接口 `POST .../commands/stop`。
  标签 accepts `shell.command.open.v1 {call_id}`。

## 4. 每轮改动（workspace_changes）

首版只做 git 仓库，统计口径是**观测窗口内的工作目录差异**，不等同于“Agent 本轮做的改动”。
非 git 目录显示“不是 git 仓库，暂不统计改动”。

### 4.1 窗口与时序

- **轮次身份**：沿用 Computer 的做法，`turn_id` 取该轮所属 open Turn 第一条 Input 的 message_id，是本功能从现有投影派生的标识。
- **开始**：`SOURCE_CHANGED_V3` 的同步监听只把 `(session, source)` 放进队列，由插件自己的后台任务读出该 Turn 并拍起始快照。
  按 `(session, source, turn_id)` 只拍一次：同一轮追加输入不重置起点，resume 不新建。
- **结束**：后台任务用 `MESSAGE_CATALOG.follow` 加 `TURN_PROJECTION` 看到该轮 `complete/quiet/abandoned` 后，拍结束快照并计算 diff。
- **完整性标记**：每条记录写明 `started_at`、`baseline_done_at`、`ended_at`、`end_done_at`，并带两个标记：
  - `baseline_late`：起始快照完成晚于该轮第一条 ToolCall 的提交；
  - `end_late`：拍结束快照前，下一轮已经开始，或还有该会话的后台命令在运行（abandon 后旧工具也可能还在跑）。
  任一为真时，界面写“可能包含窗口外的改动”或“可能遗漏改动”，不当作精确归因。
- **重启**：重启后找不到起始快照的轮次标为 `unavailable`，不靠通知补算历史。

### 4.2 git 快照

- **范围**：经 `host.processes.v1` 以本插件身份执行固定 argv；路径作为独立参数传入，环境用显式非交互配置，`hard_timeout_s=20`。
  - 起始时用 `WORKING_DIRECTORY.snapshot(session)` 取路径，`git rev-parse --show-toplevel --absolute-git-dir` 解析仓库（linked worktree 同样适用）。
    本轮的仓库在起始时固定，结束时仍对同一仓库拍快照。
  - 目录没设置、不存在、不是仓库、git 损坏或没有权限，分别给出原因，不当作“没有改动”。
- **步骤**：`GIT_INDEX_FILE=<git-dir>/akashic-changes-<随机>` → `read-tree HEAD`（unborn 时用空树）→ `add -A`（整个工作树）→ `write-tree` → 删除临时 index。
  取消或超时时，等进程组退出后再清理临时 index。
- **会动和不会动的**：
  - 不改用户的 index、HEAD、refs。
  - `git add` 仍会把对象写入对象库。
  - 仓库若声明了 filter 属性（含 LFS），首版不拍快照，标为“不支持：仓库使用 filter”，避免触发外部程序。
  - 子模块内部的改动不覆盖，存在 `.gitmodules` 时注明。
  - 只有未跟踪且被忽略的文件不统计；已跟踪文件照常统计。`.gitignore` 不是隐私保证。
  - 用户同时写文件时，快照不是原子的。
- **diff**：轮次结束时立刻用 `git diff --no-ext-diff --no-textconv -z --numstat` 加 `--patch` 计算并保存，不依赖树对象长期存在。
  原因是裸树没有引用，gc 后不可读；插件不对用户仓库执行 gc。
- **上限**：单轮最多列 2000 个文件；patch ≤ 256 KiB，超出只保留文件列表与行数，并标 `truncated`。
  同一仓库同一时刻只跑一个快照。

### 4.3 存储与范围

- **存储**：OWNER_STATE 每会话固定 50 个槽位 `turn:{session}:{slot}`，循环覆盖。
  每槽 `{turn_id, ending_message_id, repo, mode, window, files[], patch, truncated, baseline_late, end_late, error}`，
  每会话不超过约 13 MiB。插件停用时不采集；卸载保留历史（0070）。
  跨会话的总量随会话数增长：物理清理需要 ledger 提供窄清理合同，另行处理，列为已知限制。
- **范围**：
  - **本轮**：读存储。
  - **未提交**：按需拍一次快照，与 HEAD 比较，包含未跟踪的新文件；unborn 时与空树比较。
  - “本会话净 diff”首版不做：它需要长期保活树对象。
- **回复下方摘要**：`turn.after_answer` 只出现在 complete 的最终回答上，按 `messageId == ending_message_id` 查询，
  显示“改动了 N 个文件 +a −r”，计算中显示“统计中”。quiet/abandoned 的轮次只在标签的轮次列表里出现。

## 5. 预览（preview_ui）

用 `ARTIFACT_READ` 读取附件，在侧栏查看附件、文本、表格和 visuals，accepts `artifact.open.v1 {message_id, artifact_id}`。
聊天附件卡片先发意图，收到 `unhandled` 或 `failed` 时仍打开原弹窗。

## 6. 切片与验收

| 切片 | 内容 | 依赖 |
|---|---|---|
| S0 | 本设计、0121、WEBUI-008 增补 | — |
| S1 | 电脑标签顶栏：一行头部、状态胶囊、截图、复制/外部打开、窄宽度 ⋮（#1421） | — |
| S2a | v2 合同文件、会话同步、电脑迁到 v2 并删除 v1 | S0 |
| S2b | 懒挂载、按会话记忆、启动器、铺满、底部空白修复 | S2a |
| S3a | 意图路由与异步回执、聊天插槽 `workbench` 上下文 | S2a |
| S3b | 引用回输入框 | S2a |
| S4 | standard_tools 命令记录与终端标签（Agent 已读输出、停止后台命令） | S2a |
| S7a | workspace_changes 窗口时序与固定槽位存储 | S0 |
| S7b | git 快照与 diff 执行、上限与不支持判定 | S7a |
| S8a | 改动标签（本轮 / 未提交，文件列表 + diff，引用到对话） | S3、S7b |
| S8b | 回复下方改动摘要 | S7b |
| S9 | 预览标签与附件卡片的意图回退 | S3a |

每个切片一个 PR，新增约 500 行以内。验收用真实组件，不新增单元测试（TST-001）：
- 前端：真实构建 + 浏览器，1440 / 900 / 360、浅色 / 深色、减少动效。
  另验 A/B 会话快速切换后旧异步结果被拒、标签插件停用/启用、Agent 活动不展开工具区、不写草稿、无接受者时回退。
- 后端：真实 AppRuntime 与 host bridge 场景脚本。验同轮追加输入、暂停恢复、abandon、后台命令跨轮、重启后 `unavailable`，
  以及用户 index、refs、既有消息与附件保持原样。

## 7. 评审取舍与延后项

| 评审意见（brief-25） | 处理 |
|---|---|
| 实时输出需要非消费读取 | 首版只显示 Agent 已读的输出并注明；实时滚动另立决策（S4b） |
| 文件工具没有真实前后内容回执 | 首版不统计非 git 目录；需要 host_execution 在文件锁内返回 typed 回执，另立决策 |
| 异步快照的起止边界 | 改称观测窗口差异，补 `end_late`、按轮只拍一次、重启 `unavailable` |
| 树对象会被 gc | 结束时立即算 diff；取消“本会话净 diff” |
| 持久数据没有删除接口 | 固定槽位循环覆盖；跨会话清理列为已知限制 |
| 引用缺会话与手势绑定 | `quote` 带 session，宿主与聊天端两次核对，只在点击处理函数里调用 |
| git 范围与副作用 | 写明整个工作树、对象写入、filter/LFS 不支持、子模块不覆盖、超时与上限 |
| 意图 payload 与回执 | 意图名带版本，payload 由领域 owner 定义，异步回执带 id |
| 摘要入口只在 complete | 按 ending_message_id 查询；其余轮次只在标签里 |
| 切片过大、需求去向 | 重新拆分；延后项见下 |

延后项（不随 S0～S9 视为完成）：
- 终端实时输出、交互终端；
- 非 git 目录的改动统计；
- 本会话净 diff；
- “活动”标签；
- 文件树与记忆标签；
- 电脑的原始尺寸、让电脑休息、正常状态下重启、匿名页后退/刷新：需要改后端语义，0115 规定重启只能从 failed 发起。
