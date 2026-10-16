# 对话工作台：会话上下文、意图路由与正交标签

决策见 [0121](../decisions/0121-conversation-workbench-tabs.md)，需求来源 #1419。本文给出合同、事实 owner 与切片顺序；
每个切片落地后在这里补最终合同与验证，未落地的内容不代表已完成。

## 1. 分层

```text
┌──────────────────────────── 聊天 iframe（/chat） ────────────────────────────┐
│ 当前会话 ─┐   附件卡片 / 插槽按钮 ─┐ (用户手势)        输入框草稿 ◀─┐            │
└───────────┼────────────────────────┼─────────────────────────────────┼────────────┘
            │ akashic.chat-session   │ akashic.workbench-intent        │ akashic.composer-insert
┌───────────▼────────────────────────▼─────────────────────────────────┴────────────┐
│ conversation-ui：展开/宽度/当前标签/窄屏 + 转交会话 ID + 按 accepts 路由意图 + 转发引用 │
└───────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┘
        │ conversation.tools.v2（每个标签只看到 view）                                   
   ┌────▼────┐       ┌─────▼─────┐      ┌─────▼──────┐     ┌─────▼─────┐
   │ 电脑     │       │ 终端       │      │ 改动        │     │ 预览       │   各自独立插件
   │computer │       │terminal_ui │      │workspace_   │     │preview_ui │   各自 dashboard 路由
   └────┬────┘       └─────┬─────┘      │changes      │     └─────┬─────┘
        │                  │            └──┬───────┬──┘           │
        │      standard_tools.command_log.v1│       │file_edits.v1 │ artifact 读取
        │                  ▼               ▼       ▼              ▼
        │            standard_tools（shell / 文件工具的事实 owner）   ledger
        ▼                               host_execution（进程、文件）、working_directory
   computer workload
```

规则：
- 宿主不认识任何标签、意图或事实的含义；增减能力只增减插件。
- 标签之间不通信；跨插件只经 owner 的 `contract.py` 窄服务（PLG-016）。
- 事实只有一个 owner：命令与文件编辑归 `standard_tools`，轮次改动归 `workspace_changes`，附件归 ledger。
- 不做全局工具事件总线；消费者不解析别人的工具结果文本。

## 2. 宿主合同 `conversation.tools.v2`

合同文件 `packages/akashic-conversation-tools-v2/contract.json`，摘要取文件 sha256，与其它挂载点一致。

```ts
interface WorkbenchTab {
  id: string; label: string; order?: number;
  icon?: string;                       // 16px 线性 SVG，启动器与窄屏标签使用
  accepts?: readonly string[];         // 能处理的意图名，例如 "workspace.changes.open"
  render(host: HTMLElement, view: WorkbenchTabView): Disposer;   // 首次被选中时才调用
}
interface WorkbenchTabView {
  readonly active: boolean; onActiveChange(listener: (active: boolean) => void): Disposer;
  readonly session: string | null; onSessionChange(listener: (session: string | null) => void): Disposer;
  requestAttention(noticeId: string): void;
  onIntent(listener: (intent: WorkbenchIntent) => void): Disposer;  // 宿主先激活标签再投递
  quote(request: { text: string; source: string }): void;           // 插入输入框草稿
}
interface WorkbenchIntent { name: string; session: string; payload: unknown }  // payload JSON ≤ 16 KiB
```

- **会话**：聊天 iframe 在会话切换时发 `akashic.chat-session {sessionId|null}`；宿主只转交，不缓存业务数据。
- **懒挂载**：标签首次被选中才 `render`；之后切走只发 `onActiveChange(false)`，标签自行暂停后台工作。
  v1 标签保持现状（挂载即渲染），过渡期两种版本可以并存。
- **按会话记忆**：当前标签和展开状态按会话 ID 记在 `sessionStorage`；宽度仍全局一份，双击分隔条恢复默认。
- **启动器与铺满**：顶栏右侧依次为"+"（列出全部标签、图标与快捷键）、铺满/还原、关闭。
  铺满时对话列收起，`Esc` 或再次点击还原。
- **意图路由**：
  1. 聊天端只在用户手势内发 `akashic.workbench-intent`（检查 `navigator.userActivation.isActive`）。
  2. 宿主按标签顺序找第一个 `accepts` 含该名字的标签，展开工具区、激活标签、投递意图；没有接受者时回复
     `akashic.workbench-intent-result {handled:false}`，聊天端回到原有行为（例如预览弹窗）。
  3. 宿主把当前可接受的意图名集合以 `akashic.workbench-intents {names}` 告诉聊天端，聊天端据此决定是否显示"在侧栏打开"。
- **引用回输入框**：`quote` 转发为 `akashic.composer-insert {text}`，聊天端调用现有 `insertDraft(text, {append:true})`，
  文本前加来源行（例如 `> 改动 · src/a.ts:12-20`）。不新增 Input 内容种类，不自动发送。
- **Agent 活动**：只经 `requestAttention` 亮徽标；宿主不因活动展开工具区（0092）。
- **聊天插槽**：Plugin UI 插槽上下文增加 `workbench: { accepts(name): boolean; open(name, payload): boolean }`，
  由聊天端转发上面的意图消息。

## 3. 事实合同

### 3.1 `standard_tools.command_log.v1`

```python
class CommandRecord(BaseModel):
    call_id: str            # 工具调用所在 Output 消息 ID + part 序号
    session_id: str; turn_id: str        # turn_id = 该轮第一条 Input 的 message_id
    command: str            # ≤ 4 KiB，超出截断并标注
    cwd: str; tty: bool; background: bool
    state: Literal["running", "exited", "timeout", "terminated", "unknown"]
    exit_code: int | None; started_at: datetime; ended_at: datetime | None
    output_tail: str        # 最近 16 KiB；output_truncated 标记是否截断
    output_truncated: bool

class CommandLog(Protocol):
    def page(self, session_id: str, *, before: str | None, limit: int) -> CommandPage: ...
    def follow(self, session_id: str) -> AsyncIterator[CommandRecord]: ...   # 先回放进行中的命令，再推送变化
COMMAND_LOG = ServiceKey[CommandLog]("standard_tools.command_log.v1")
```

shell 工具在 prepare 固定 cwd（SH-004）后写入记录，执行结束更新。每个会话保留最近 500 条，存于 `standard_tools`
自己的 OWNER_STATE；超限淘汰最旧并在页首标注。`unknown` 用于进程重启后无法确认结局的记录，不猜成功。

### 3.2 `standard_tools.file_edits.v1`

```python
class FileEditRecord(BaseModel):
    call_id: str; session_id: str; turn_id: str
    path: str               # 绝对路径
    kind: Literal["create", "modify", "overwrite"]
    patch: str | None       # unified diff，≤ 64 KiB；超出为 None 并给出行数统计
    added: int; removed: int; binary: bool

FILE_EDITS = ServiceKey[FileEdits]("standard_tools.file_edits.v1")  # page(session_id, turn_id) / follow(session_id)
```

`edit_file` 用调用参数里的新旧文本与实际匹配位置生成 diff；`write_file` 在写入前读出旧内容。
只记录工具成功执行后的结果。

### 3.3 `workspace_changes`（轮次改动 owner）

- **轮次身份**：沿用 Computer 的做法，`turn_id` 为所属 open Turn 第一条 Input 的 message_id。
- **起止**：`SOURCE_CHANGED_V3` 看到 Input 提交时异步拍起始快照；`MESSAGE_CATALOG.follow` +
  `TURN_PROJECTION` 看到该轮 `complete/quiet/abandoned` 时拍结束快照并计算 diff。
- **git 快照**：经 `host.processes.v1` 以本插件 owner 执行，工作目录取 `WORKING_DIRECTORY.snapshot(session)`：
  `GIT_INDEX_FILE=<git-dir>/akashic-changes-<随机>` → `read-tree HEAD`（无 HEAD 时空树）→ `add -A` → `write-tree` → 删除临时 index。
  不改用户 index、分支和 refs，`.gitignore` 忽略的文件不统计。diff 在轮次结束时立即算出并保存，之后对象被 gc 也不影响。
- **非 git 目录**：改用 `FILE_EDITS` 汇总该轮记录，模式记为 `file_tools`，界面注明"shell 命令造成的改动未统计"。
- **可能不完整**：起始快照完成时间晚于该轮第一条工具调用的提交时间时，标记 `baseline_late`。
- **存储**：OWNER_STATE 每轮一条 `{turn_id, ending_message_id, cwd, mode, files[{path,status,added,removed,binary}],
  patch(≤ 512 KiB), truncated, baseline_late, error}`；每会话最近 200 轮。
- **范围**：本轮（存储）、本会话（首轮起始树到末轮结束树，按需计算，非 git 时合并记录）、未提交（`git diff HEAD`，按需）。
- **失败**：git 不可用、目录不存在、超时都写进 `error`，不生成空改动冒充"没有改动"。

## 4. 标签插件

| 插件 | 标签 | accepts | 数据 | 说明 |
|---|---|---|---|---|
| computer | 电脑 | — | 现有 | 顶栏改版：两行布局、状态胶囊、⋮ 菜单、容器查询收起 |
| terminal_ui | 终端 | `shell.command.open` | `COMMAND_LOG` | 只读命令记录，按轮分组，运行中实时滚动；交互终端另立决策 |
| workspace_changes | 改动 | `workspace.changes.open` | 自有 | 文件列表 + diff（并排/统一）、范围切换、选中行"引用到对话"；`turn.after_answer` 显示"改动了 N 个文件 +a −r" |
| preview_ui | 预览 | `artifact.open` | ledger 附件读取 | 附件、文本、表格、visuals 在侧栏查看；聊天附件卡片优先发意图，无接受者时仍用弹窗 |

所有标签工具栏遵守同一规格：28px 方形无底色按钮、悬停提示带快捷键、不可用时写明原因、按容器宽度把次要按钮收进 ⋮。
规格写在本文与各插件样式里，不建共享组件包（web 模块不能互相 import）。

## 5. 切片与验收

| 切片 | 内容 | 依赖 |
|---|---|---|
| S0 | 本设计、0121、WEBUI-008 增补 | — |
| S1 | 电脑标签顶栏改版 | — |
| S2 | v2 合同文件、会话上下文、懒挂载、按会话记忆、启动器、铺满、底部空白修复 | S0 |
| S3 | 意图路由、引用回输入框、插槽 `workbench` 上下文 | S2 |
| S4 | `standard_tools.command_log.v1` | S0 |
| S5 | 终端标签 | S2、S4 |
| S6 | `standard_tools.file_edits.v1` | S0 |
| S7 | `workspace_changes` 快照、diff 与存储 | S6 |
| S8 | 改动标签与回复下方摘要 | S3、S7 |
| S9 | 预览标签 | S3 |

每个切片一个 PR，新增约 500 行以内。前端切片用真实构建 + 浏览器在 1440 / 900 / 360、浅色 / 深色、减少动效下截图对比；
后端切片用真实 AppRuntime 与 host bridge 场景脚本取证（TST-001，不新增单元测试）。

## 6. 未决问题

- 起始快照与第一次工具调用之间没有同步屏障；若 `baseline_late` 比例偏高，再请 reply owner 提供可等待的轮次开始信号。
- 大仓库 `git add -A` 的耗时与临时 index 体积需要实测，必要时按目录大小跳过并标注。
- 交互终端需要 host_execution 提供输出流订阅，另立决策；本轮只做只读命令记录。
