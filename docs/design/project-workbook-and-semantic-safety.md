# 项目工作手册与语义安全技术设计

- 状态：proposed
- 日期：2026-07-16
- 目标读者：维护者、coding agent、评审者、CI 实现者
- 关联条款：WBK-001～WBK-006、COM-001～COM-004、PRM-001～PRM-008、CTX-001～CTX-006、SES-003、TST-001～TST-005
- 相关决策：[0001](../decisions/0001-project-workbook-is-shared-reality.md)、[0002](../decisions/0002-context-reduction-is-a-nondestructive-projection.md)
- Prompt 参考：[OpenAI · Prompting guidance for GPT-5.6](https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6)

## 1. 这份设计解决什么问题

2026-07-13 合并的 [PR #111](https://github.com/kachofugetsu09/akashic-agent/pull/111) 在上下文重试路径中加入了持久历史删除。重构的原始目标是让模型请求在窗口超限后缩小输入，并让重试结果跨进程保持一致。实现把“缩小本次模型窗口”解释成“让数据库永久只保留窗口内消息”，同时删除对应 embeddings。普通测试也被改成期待旧历史消失。

2026-07-14 的 [PR #124](https://github.com/kachofugetsu09/akashic-agent/pull/124) 撤销了删除路径，并从备份和缓存恢复可恢复数据。修复后的实现保留数据库完整历史，只缩短当前进程中的 `session.messages`。事故已经止血，产生事故的组织和架构条件仍然存在：

1. `history` 和 `trim` 同时描述三种不同对象。
2. 上下文代码从 SessionManager 获得了超出职责的持久化写入权限。
3. 实现者可以在同一改动中重写普通测试的期望。
4. PR #111 包含 178 个提交和 225 个文件，单条高风险语义变化被大量正确改动淹没。
5. 项目意图散落在代码、测试、旧会话和维护者记忆里，agent 只能自己补齐空白。
6. 多 worktree 和多执行者扩大了基线差异，却没有强制共享现实的交接协议。

本设计交付两套互相配合的机制：

- 项目工作手册让每个新会话从同一份当前现实开始。
- 可执行语义安全层让 agent 即使误解，也拿不到破坏数据的接口，并会被独立验收拦住。

## 2. 已确认的事故链

### 2.1 PR #111 做了什么

事故提交 [`82b6056d`](https://github.com/kachofugetsu09/akashic-agent/commit/82b6056d3bc0559c5b7f8aefcbcd02878efce852) 新增了以下路径：

```text
DefaultReasoner.run_turn
  │
  ├─ 模型请求因 ContextLengthError / safety 进入 retry plan
  │
  ├─ 较小窗口重试成功
  │
  └─ SessionManager.trim_history_async(retained ids)
       │
       └─ SessionStore.persist_session(..., retained_message_ids)
            │
            ├─ DELETE FROM messages WHERE id NOT IN retained ids
            └─ DELETE corresponding message_embeddings
```

重构账本把“进程重启后旧历史再次出现”记为原问题，把数据库替换为裁切结果记为正确修复。新增测试断言：

- 裁切后数据库消息从 4 条变成 2 条。
- 被裁消息的 embeddings 同时消失。
- 关闭并重载后只能看到 2 条历史。
- DELETE 失败时数据库和内存共同回滚。

这些测试完整证明了代码实现了“原子删除”。它们没有回答“用户是否授权删除”。被测实现和验收标准使用了同一份错误语义。

### 2.2 PR #124 如何修复

修复提交 [`a08e27d8`](https://github.com/kachofugetsu09/akashic-agent/commit/a08e27d8dd3e34b6c8b9e61d8bd6e52cc48b521b) 删除 `retained_message_ids` 参数和 `_delete_messages_not_retained_locked()`，把 `trim_history_async` 改为：

1. 在 session lock 内计算保留的运行时消息。
2. 只提交 session metadata 和尚未持久化的新增消息。
3. 提交成功后替换进程内 `session.messages`。
4. 保留 `sessions.db/messages` 和 `message_embeddings` 全部历史。

对应测试改为断言数据库仍有 4 条消息、全部 embeddings 保留、重载后完整历史恢复、下一条 seq 从原最大值继续。

### 2.3 当前实现仍有哪些含糊点

当前主线已经没有上下文裁切触发的 DELETE。调用流程仍保留一个容易再次误解的结构：

```text
DefaultReasoner.run_turn
  │
  ├─ source_history = get_history_since_consolidated(...)
  ├─ _build_attempt_plans(total_history)
  │    ├─ full history + full dynamic sections
  │    ├─ full history + drop skills
  │    ├─ full history + drop memes
  │    ├─ full history + drop long-term memory
  │    ├─ full history + drop retrieved memory
  │    ├─ 50% history + all dynamic sections dropped
  │    └─ 0% history + all dynamic sections dropped
  │
  ├─ provider raises ContextLengthError
  ├─ later attempt succeeds
  └─ attempt > 0 时调用 trim_history_async(window)
```

即使只删除了 `skills_catalog`、历史窗口没有缩小，成功后仍会调用 `trim_history_async`，日志仍写“修剪 session 历史”。这条路径目前不会删除数据库，但它把“本次 prompt 退化计划”和“长期修改当前 session runtime view”绑在一起。下一次重构很容易再次问：既然 runtime view 已裁切，为什么重载后又恢复？

目标设计要彻底分离两件事：

- 一次请求的退化计划只产生 `PromptContext`。
- 进程内历史缓存是否回收由独立 cache owner 决定。
- 完整会话历史的保留与删除由显式数据管理流程决定。

## 3. 根因分析

### 3.1 语义同名

事故前的设计缺少三种明确名词：

| 对象 | 生命周期 | 是否权威 | 是否允许因 token 预算缩小 |
|---|---|---:|---:|
| persistent conversation history | 跨进程、跨版本 | 是 | 否 |
| runtime history view | 当前进程或当前 turn | 否 | 可以 |
| prompt history | 单次模型请求 | 否 | 可以 |

一个 `history` 名称覆盖三行，`trim_history_async` 没有指出改哪一行。agent 根据局部目标把“重载后不复活”理解成正确持久化，语义上自洽，产品意图上错误。

### 3.2 写入权限大于职责

Context/Reasoner 只需要读取历史、构造 prompt 和更新少量运行元数据。它得到的是完整 SessionManager，SessionManager 又拥有完整 SessionStore。调用者可以调用一个看似普通的 manager 方法触发 SQL DELETE。

注释只能告诉实现者“不要做”。窄接口可以让实现者“做不到”。

### 3.3 Oracle 与实现共同漂移

测试名、fixture 和断言跟着实现一起变化。CI 只能证明新代码与新断言一致，不能证明新断言仍符合用户意图。全量测试全绿反而增强了错误改动的可信度。

### 3.4 大 PR 降低审计信号

PR #111 同时处理 runtime、memory、plugin、scheduler、filesystem、前端和大量错误边界。每个局部改动都有测试和账本说明。评审者很难从数百个正确的 fail-loud 改动中识别出一条未经批准的数据保留变化。

### 3.5 没有共享现实的开工门槛

“上下文裁切不删历史”当时不是一个稳定、可引用、受保护的项目条款。agent 不需要在开工前声明受保护状态，也没有材料要求它区分 prompt budget 和 data retention。

## 4. 目标与非目标

### 4.1 目标

1. 新维护者在十五分钟内找到需求、当前工作、相关决策和执行模板。
2. 高风险任务在写代码前暴露关键假设、权限和状态分类。
3. Context/Prompt 代码在类型和依赖图上拿不到 destructive port。
4. `semantic_delta: none` 的 refactor 无法自行修改受保护 oracle 获得全绿。
5. CTX-001 的已知 DELETE mutant 每次都会失败。
6. worktree 和多 agent 使用同一基线、稳定 ID 和共享证据交接。
7. Prompt 保持结果导向、不反复陈述、可停止，并能用真实任务回归。

### 4.2 非目标

- 本轮不一次性为全部语义条款实现 CI。
- 本轮不把所有项目历史装入每次 agent 上下文。
- 本轮不建立复杂的文档生成平台或新数据库。
- 本轮不依赖另一个 LLM 对实现做主观批准。
- 本轮不改变 Akashic Agent 的持久会话保留和显式删除语义。

### 4.3 从协作原理到工程机制

| 协作原理 | PR #111 中的失效形态 | 本设计的工程机制 |
|---|---|---|
| 核对先于假设 | “裁切历史”被自行解释成替换数据库历史 | 高风险歧义门禁、任务合同、状态分类 |
| 工作手册提供共享现实 | 项目意图散落在旧会话、账本和普通测试 | INDEX、WORKFLOW、projectneed、NOW、decisions、writing rules |
| 执行时隐藏，问责时展开 | context 模块拿到完整 manager，评审却难看见单条危险语义 | 窄依赖、完整 diff、write set、决策依据 |
| 减少沟通不能消灭必要沟通 | 大改动一路执行，数据保留变化没有独立确认点 | semantic delta、一个高风险语义一个改动、阶段门禁 |
| 执行可以树状，信息必须网状 | worktree 各自漂移，集成时才发现前提不同 | 稳定条款 ID、共享 Git 状态、验收前 workbook diff |

## 5. 项目工作手册架构

```text
                         ┌──────────────────────┐
                         │ docs/projectneed.md  │
                         │ 长期需求 / 不变量 ID │
                         └──────────┬───────────┘
                                    │ 约束
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ docs/WORKFLOW.md     │  │ docs/decisions/      │  │ docs/writing-rules   │
│ 开工与交付纪律        │  │ 决策理由与勘误        │  │ 文档所有权与写法       │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     ▼
                           ┌──────────────────────┐
                           │ docs/NOW.md          │
                           │ 当前未完成 / 接手点   │
                           └──────────┬───────────┘
                                      │ 指向
                                      ▼
                           ┌──────────────────────┐
                           │ code / tests / logs  │
                           │ 当前实现证据          │
                           └──────────────────────┘
```

### 5.1 启动读取算法

任何非简单任务按下面的顺序进入项目：

1. 读取 `INDEX.md`，再读取 `projectneed.md` 第 1～6 节和路由表命中的领域；修改任务同时读取 `WORKFLOW.md`，跨层、高风险任务读取全文。
2. 读取 `NOW.md`，确认任务是否已有 owner、阻塞或既定接手点。
3. 用模块名、条款 ID、错误名和关键名词搜索 `decisions/README.md`。
4. 只打开命中的决策和设计，不批量加载全部历史。
5. 读取真实代码、配置、日志、数据库和测试。
6. 如果代码、需求和决策冲突，停止实现，输出冲突表。
7. 写任务合同，满足开工门禁后再修改文件。

### 5.2 冲突表格式

```markdown
| 对象 | projectneed 要求 | 当前实现 | 决策记录 | 影响 |
|---|---|---|---|---|
| CTX-001 | persistent history 不变 | 当前是否满足 | ADR-0002 | 删除风险 / 无风险 |
```

agent 不能用“代码现在这样，所以需求应该这样”解决冲突。代码是实现证据，需求是意图 owner。

### 5.3 信息隐藏的使用边界

执行阶段只给任务所需材料和权限。问责阶段允许展开全部相关证据：

```text
执行者输入 = 当前目标 + 相关条款 + 相关代码 + 必需工具
评审者输入 = 执行者输入 + 完整 diff + oracle + 日志 + 状态变化 + 决策依据
```

隐藏无关信息可以降低窗口压力。隐藏决定成败的要求、写入证据或失败日志会破坏问责。

## 6. 可复制的 Agent 任务合同

OpenAI 的 GPT-5.6 prompt guidance 建议把结果、重要约束、可用证据和完成标准写清，让模型自行选择高效路径；审批边界、工具路由和停止条件需要单独说明。仓库使用以下合同：

```markdown
## Role

你负责 [任务边界]。当前阶段是 [research/design/implementation/review]。

## Goal

[一句话写用户最终能看到的结果。]

## Success criteria

- [可以独立判断 true/false 的结果 1]
- [结果 2]
- [必须完成的验证]

## Evidence

- 已知事实：[路径、日志、PR、数据库、条款]
- 必须先读取：[最小材料]
- 未确认：[会改变方案的未知项]

## Semantic change

- change_type: fix|feature|refactor|migration|docs
- semantic_delta: none|compatible|breaking
- protected_state: [...]
- allowed_effects: [...]
- forbidden_effects: [...]
- invariants: [CTX-001, ...]

## Autonomy

- 可自主：[只读调查、范围内本地编辑、非破坏性验证]
- 需确认：[外部写入、删除、迁移、付费动作、扩大范围]

## Tools

- [工具]：何时使用、关键返回值、失败如何解释
- 独立读取可以并行；依赖前一步结果的操作保持串行

## Output

- [交付文件、报告字段、格式]

## Stop rules

- 满足全部 success criteria 后停止
- 缺少 [关键事实] 时只询问最小缺口
- [重试上限] 后仍失败则报告阻塞，不猜测成功
```

### 6.1 合同编写规则

- `Goal` 写结果，不写几十步操作清单。
- `Success criteria` 必须能由测试、状态快照、diff 或人工明确判断。
- `Constraints` 只保留真实不变量。判断型选择写成条件规则。
- 相同规则只出现一次。项目级规则引用条款 ID。
- `Tools` 只列任务相关工具，避免让无关接口进入选择空间。
- `Stop rules` 阻止无止境搜索，也阻止证据不足时过早交付。

### 6.2 高风险歧义门禁

下面四个问题只要有一个答不出来，就不能进入 implementation：

1. 被“裁切、清理、同步、替换”的具体类型是什么？
2. 哪些对象跨进程仍然存在，谁拥有它们？
3. 失败后要保持哪份状态，恢复动作由谁执行？
4. 哪个独立 oracle 会发现错误理解？

任务合同示例：

```yaml
change_type: refactor
semantic_delta: none
goal: 降低超长 prompt 的输入规模，同时保留完整会话历史
protected_state:
  - sessions.db/messages
  - sessions.db/message_embeddings
  - message id and seq high-water mark
allowed_effects:
  - rebuild PromptContext
  - reduce prompt history window
  - update retry trace
forbidden_effects:
  - delete or update persistent messages
  - delete message embeddings
  - change retention policy
invariants: [CTX-001, CTX-002, CTX-003, SES-002, SES-003]
stop_rules:
  - all targeted tests and CTX-001 semantic gate pass
  - if current code requires a database delete, stop and report a requirement conflict
```

## 7. 上下文架构

### 7.1 目标数据流

```text
┌─────────────────────────────┐
│ SessionHistoryReader        │
│ 只读完整持久历史             │
└──────────────┬──────────────┘
               │ tuple[StoredMessage, ...]
               ▼
┌─────────────────────────────┐
│ ConversationSnapshot        │
│ 不可变，只读                 │
└──────────────┬──────────────┘
               │ select semantic turns
               ▼
┌─────────────────────────────┐
│ RuntimeHistoryView          │
│ 进程内可重建，不写数据库      │
└──────────────┬──────────────┘
               │ render + trim plan
               ▼
┌─────────────────────────────┐
│ PromptContext               │
│ 单次请求，可裁切             │
└──────────────┬──────────────┘
               │ ModelRequest
               ▼
┌─────────────────────────────┐
│ Provider                    │
└─────────────────────────────┘

┌─────────────────────────────┐
│ SessionDestructivePort      │
│ 显式用户/管理员删除          │
└─────────────────────────────┘
       ▲ 与上方数据流无依赖边
```

### 7.2 类型草案

下面的类型用于标出语义边界，不要求一次新增五套运行时抽象。当前仓库已有可复用对象：

| 目标概念 | 当前落点 | 本轮处理 |
|---|---|---|
| 持久历史 owner | `session/store.py::SessionStore` | 保持唯一 SQLite owner |
| 运行时 session | `session/manager.py::Session` | 只有 history window 确实缩小时才裁切 `messages`，且不调用 store |
| prompt 输入与结果 | `agent/lifecycle/types.py::PromptRenderInput`、`PromptRenderResult` | 继续使用，不新增同义 wrapper |
| 退化计划（历史实现，已退役） | `agent/prompting/budget.py::ContextTrimPlan` | 旧 prompt retry 设计记录；当前由 session Context Gate 与完整 logical unit compaction 取代 |
| retry orchestrator | `agent/core/passive_turn.py::DefaultReasoner.run_turn` | 移除 `SessionManager` 依赖和 `trim_history_async` 调用 |
| 显式删除入口 | `agent/control/service.py::delete_thread` → `SessionManager.delete_session` | 保留，并与 prompt 路径保持无依赖边 |

下面草案仅用于现有类型无法表达只读边界的情形。名称可以按仓库风格调整，三层语义不能合并：

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StoredMessage:
    id: str
    seq: int
    role: str
    content: object


@dataclass(frozen=True)
class ConversationSnapshot:
    session_key: str
    messages: tuple[StoredMessage, ...]
    max_seq: int


@dataclass(frozen=True)
class PromptContext:
    messages: tuple[dict[str, object], ...]
    disabled_sections: frozenset[str]
    source_message_ids: tuple[str, ...]


class SessionHistoryReader(Protocol):
    def load_snapshot(self, session_key: str) -> ConversationSnapshot: ...


class SessionDestructivePort(Protocol):
    def delete_session(self, session_key: str, *, intent_id: str) -> None: ...
```

如果后续引入 `PromptContextBuilder`，它只接收 `ConversationSnapshot` 和动态区块，输出 `PromptContext`。它不接收 manager、store、SQLite connection 或 filesystem。当前实现先让 `DefaultReasoner` 使用既有 `PromptRenderInput` 完成相同隔离，不为改名新增并行抽象。

### 7.3 退化计划

每个 prompt 区块声明耐久等级和裁切顺序：

| 等级 | 示例 | 预算不足时的动作 |
|---|---|---|
| decorative | meme、非必要展示 | 最先移除 |
| rebuildable | skills catalog、retrieved memory、long-term memory 投影 | 可再次查询，先于历史移除 |
| conversational | 较早完整语义回合 | 动态区块移除后再缩窗 |
| current-turn | 当前用户消息、当前工具因果链 | 所有前项移除后仍需保留；仍超限时明确失败 |
| canonical | sessions.db 完整历史 | 不参与 prompt 裁切 |

计划由纯函数生成：

```python
def build_prompt_attempts(
    snapshot: ConversationSnapshot,
    dynamic_sections: "DynamicSections",
    budget: int,
) -> tuple[PromptContext, ...]:
    """按耐久等级生成有界的模型请求候选。"""
```

候选只描述本次请求。某次 retry 成功后不自动修改 `ConversationSnapshot` 或持久历史。

### 7.4 上下文压缩 handoff

长任务完成主要阶段后可以压缩；实际文件使用 [`docs/templates/context-handoff.yaml`](../templates/context-handoff.yaml)，核心字段如下：

```yaml
goal: 当前用户可见目标
success_criteria:
  - 尚未完成前不能删除的条件
verified_facts:
  - fact: 已核对事实
    evidence: path/symbol/pr/log
assumptions:
  - assumption: 尚未核对的假设
    risk: low|high
decisions:
  - decision_id: ADR-0002
changed_files:
  - path
remaining_work:
  - 只保留未完成事项
validation:
  passed: []
  pending: []
  failed: []
```

压缩内容是任务继续执行的状态，不是持久项目需求。恢复时按路径和条款再次取证；不能从摘要措辞推导新的删除、权限或保留规则。

## 8. 权限拆分方案

### 8.1 依赖规则

| 调用层 | 可依赖 | 禁止依赖 |
|---|---|---|
| `agent/prompting/**` | snapshot types、token estimator | SessionStore、SQL、destructive port |
| `DefaultReasoner` context retry（历史实现，已退役） | `PromptRenderInput`、旧 `ContextTrimPlan`、retry trace | SessionManager、SessionStore、delete session/messages/embeddings |
| `session` persistence owner | SQLite store、transaction | provider 和 prompt policy |
| control/dashboard delete | destructive port、audit、backup policy | prompt trim helpers |

### 8.2 静态门禁

第一版不用引入大型 policy 平台。一个 Python AST 脚本即可检查：

- `agent/prompting/**` 不导入 `session.store`。
- `DefaultReasoner` 不保存 `SessionManager`，context retry 模块不引用 `delete_session`、`DELETE`、`remove_messages` 和 embedding 删除接口。
- `SessionDestructivePort` 只在批准的管理入口注入。
- 新出现的 `trim_history`、`replace_history`、裸 `history` public API 需要显式 allowlist 和评审。

静态规则只挡常见路径，不能替代运行时 write-set。

### 8.3 SQLite 运行时门禁（完整目标）

完整 oracle 应对被测连接安装 authorizer：

```python
PROTECTED_TABLES = {"messages", "message_embeddings"}


def authorizer(action, arg1, arg2, database, trigger):
    if action in {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_UPDATE,
    }:
        if arg1 in PROTECTED_TABLES:
            violations.append((action, arg1, arg2))
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK
```

完整测试还应使用独立 observer connection 核对完整规范化快照。authorizer 发现违规尝试，observer 发现实际状态变化；两者覆盖不同失败面。当前 pilot 只有 trace 与 observer 快照，实际差距见第 13 节。

## 9. 独立语义验收

### 9.1 CTX-001 黑盒测试

完整目标测试从真实 SessionManager 或更高入口触发，不以纯函数为唯一被测入口：

1. 在隔离数据库创建含 user/assistant/tool/media/embedding 的大会话。
2. 保存 `messages`、`message_embeddings` 和最大 seq 的规范化快照；session metadata 单独记录为审计证据。
3. 安装 SQLite authorizer 与 trace。
4. 触发 context overflow，并让后续较小 prompt 成功。
5. 断言本次 provider payload 满足预算和语义回合边界。
6. 断言受保护表没有 INSERT/DELETE/UPDATE 尝试。
7. 用 observer connection 核对完整内容一致。
8. 关闭 manager，再次加载，断言完整历史可见。
9. 追加一条消息，断言 seq 等于原最大值加一。

Phase 1 不把全部 session metadata 设为硬不变量。当前 `trim_history_async` 会更新 `updated_at` 和 `last_consolidated`，所以 metadata 只进入 before/after 报告。Phase 2 的 runtime-only mutator 只能在 history window 变小时修改内存中的 `messages` 和 `last_consolidated`，不能写数据库或刷新 `updated_at`。这个分阶段规则确保第一版 oracle 能锁住数据不丢失，同时诚实暴露当前剩余副作用。

测试名建议：

```text
tests/semantic/session/test_ctx_001_context_projection.py
```

### 9.2 语义 mutant

维护最小错误 patch：

```diff
+ store.delete_messages_not_retained(session_key, retained_ids)
```

CI 的 mutant job：

1. 在临时 worktree 应用 patch。
2. 只运行 CTX-001 semantic test。
3. 期望测试失败，且失败原因包含 protected DELETE 或 snapshot mismatch。
4. 如果测试仍然成功，mutant job 必须失败，并报告 oracle 无效。

### 9.3 Oracle 所有权

```text
普通实现改动可修改：
  agent/**、session/**、tests/unit 或现有普通 tests

需要独立 owner：
  docs/projectneed.md
  docs/decisions/**
  contracts/**
  policies/**
  tests/semantic/**
  tests/semantic_mutants/**
  migrations/**
```

即使个人仓库没有多人审批，也使用两阶段提交：先提交 contract/decision，明确暂停点；用户确认后再提交实现。CI 检查 `semantic_delta: none` 是否同时修改受保护路径。

## 10. Worktree 和多 Agent 协议

### 10.1 创建

```bash
git fetch origin main
git worktree add -b feature/<task> ../akasic-agent-worktrees/<task> origin/main
```

记录：

```yaml
target_branch: main
base_commit: <sha>
worktree: <path>
owned_invariants: [CTX-001]
owned_paths:
  - agent/prompting/**
  - tests/semantic/session/**
```

### 10.2 分工

- 一个执行者只拥有边界清楚的文件和不变量。
- 两个执行者不能并行修改同一 semantic oracle 或同一 ADR。
- 调查任务可以并行；任何写入前由主 owner 汇总事实和冲突。
- 子任务返回证据路径、假设、结论和未解决问题，不能只返回“已完成”。
- 接口决定和受影响任务通知是跨 owner 修改接口的前置条件。

### 10.3 验收前同步

1. 获取目标分支最新提交。
2. 核对从 base 到最新目标分支的 INDEX、WORKFLOW、projectneed、NOW 和相关 ADR。
3. 解决需求或 owner 冲突后再 rebase/merge。
4. 再次运行任务合同中的 semantic tests。
5. 检查实际 diff 是否仍在 owned paths 和允许副作用内。

worktree 提供文件隔离，不提供语义同步。跳过第二步会让每条分支基于不同现实继续漂移。

## 11. 从开工到合并的标准流程

本节记录流程形成时的设计理由和阶段退出条件。[`WORKFLOW.md`](../WORKFLOW.md) 是当前可执行入口；步骤或命令发生变化时更新执行入口，本节只在设计理由或阶段边界变化时同步。

```text
┌──────────────┐
│ 1. Bootstrap │  读工作手册、代码和证据
└──────┬───────┘
       ▼
┌──────────────┐
│ 2. Contract  │  目标、成功标准、语义变化、权限、停止条件
└──────┬───────┘
       ▼
┌──────────────┐
│ 3. Confirm   │  高风险歧义向用户核对
└──────┬───────┘
       ▼
┌──────────────┐
│ 4. Implement │  窄接口、最小 diff、阶段更新
└──────┬───────┘
       ▼
┌──────────────┐
│ 5. Verify    │  targeted → semantic → type/build → mutant
└──────┬───────┘
       ▼
┌──────────────┐
│ 6. Reconcile │  同步目标分支、检查 workbook diff
└──────┬───────┘
       ▼
┌──────────────┐
│ 7. Deliver   │  diff、证据、阻塞；NOW 完成即剔除
└──────────────┘
```

### 11.1 阶段退出条件

| 阶段 | 退出前必须成立 |
|---|---|
| Bootstrap | 任务相关需求、当前实现、历史决策和未知项已列出 |
| Contract | 成功标准可判断，语义 delta、权限和停止条件明确 |
| Confirm | 高风险未知项已确认，或任务停止等待用户 |
| Implement | 实际改动不超范围，没有新增未声明副作用 |
| Verify | 目标测试成功；未运行项和原因明确 |
| Reconcile | 目标分支新变化未使任务合同失效 |
| Deliver | 文档、代码、决策和 NOW 互相一致 |

## 12. Prompt 维护策略

### 12.1 保持 stable prefix

长期不变量和协作纪律放在 `projectneed.md` 与 `WORKFLOW.md`。动态任务、NOW、日志和临时证据放在后部按需读取。这样可以减少大 prompt 前缀频繁变化，也让缓存行为更稳定。

### 12.2 删除反复陈述，不删除完成标准

Prompt 精简优先删除：

- 多处反复出现的“先确认”“不要删除”“运行测试”。
- 不能改变行为的示例。
- 模型已经稳定完成、且无回归证据的流程性唠叨。
- 与任务无关的工具说明。

必须保留：

- 用户可见目标。
- 成功标准和停止条件。
- 安全、业务、证据和权限边界。
- 依赖上下文的工具路由。
- 输出和验证要求。

### 12.3 Prompt 回归集

至少维护以下真实任务：

| Case | 输入特征 | 必须观察的行为 |
|---|---|---|
| P-01 | 普通局部修复 | 自主修改并跑目标测试，不反复请示 |
| P-02 | “裁切历史”含糊 | 先区分 prompt view 与 persistent history |
| P-03 | 用户明确要求删除 | 建备份、确认范围、走 destructive port |
| P-04 | 只要求诊断 | 只读调查，不擅自实现 |
| P-05 | 工具返回空结果 | 做有界 fallback，不断言事实不存在 |
| P-06 | 长任务跨里程碑 | 生成完整 handoff，保留目标和未完成事项 |
| P-07 | 两条指令冲突 | 开工前暴露冲突，不静默任选 |
| P-08 | 多 worktree 基线漂移 | 验收前检查 workbook diff |

每次只修改一组 prompt 规则，使用相同模型、reasoning effort、工具集和 case 再次运行。记录：

- 任务是否成功。
- 是否发生不必要提问。
- 是否越权写入。
- 是否漏验证或过早停止。
- token、工具调用、轮次、延迟和成本。

只有原成功标准全部满足，资源下降才记为改进。

## 13. CI 实施计划

### 当前 pilot 与完整 oracle 的差距

**F（当前 pilot）：** 现有落点是：

```text
tests/semantic/test_context_history_contract.py
tests_scenarios/contracts/oracles.py
tests_scenarios/contracts/scenarios.toml
```

`test_full_context_projection_preserves_append_only_history` 使用真实 `SessionManager` 和 `DefaultReasoner.run_turn`，只替换有界模型结果。它证明单一 `full_context` 计划从 session projection 读取全部历史，记录 SQLite trace，比较完整 messages/embeddings 快照，重启后再追加消息验证 seq 续接。`test_history_oracle_rejects_historical_delete_mutant` 直接在 fixture 数据库执行历史 DELETE，证明同一组快照和 write-set oracle 会拒绝已知坏状态。

这个 pilot 已经保护“full-context projection 和 compact 不能减少持久历史”的主要事故路径，但它还不是完整 Phase 1：

- 没有 SQLite authorizer，因此不能稳定记录所有被回滚、由 CTE 表达或经 trigger 触发的受保护写入尝试。
- fault injection 直接修改 fixture，不是在一次性候选源码中把 DELETE/UPDATE 注入真实 compaction seam。
- 还没有独立 protected-path policy 阻止普通实现改动同时降低 oracle、mutant 或 coverage baseline。
- G2 仍是 Feed/Observe 单场景 runner；统一 Docker controller 和 aggregate external status 尚未完成。

**I（完整 Phase 1 目标）：** 在保留当前健康场景的基础上新增 SQLite authorizer/write guard，并在一次性候选副本中应用真实 seam mutant。mutant job 只有在健康路径先通过、错误候选因 CTX-001/SES-005 状态差异失败时才算 kill；导入失败、fixture 未启动和超时一律是 Gate failure。建议新增落点：

```text
tests/semantic/helpers/sqlite_write_guard.py
tests/semantic_mutants/ctx_001_delete_messages.patch
```

Phase 1 完整后锁住“绝不丢持久历史”。Phase 2 再收紧 runtime 副作用：临时 payload projection 不改 session，compaction 只追加 ledger/checkpoint 并推进会话游标。当前缺口继续保留在 `NOW.md`，不能用 pilot baseline 或普通 pytest 全绿冒充已经完成。

### Phase 0 · 文档和模板

交付：

- `docs/INDEX.md`
- `docs/WORKFLOW.md`
- `docs/projectneed.md`
- `docs/NOW.md`
- `docs/decisions/`
- `docs/writing-rules.md`
- `docs/templates/`

验收：所有链接存在；projectneed 只有一份规范正文；NOW 只含未完成事项；核心文件不再被 `.gitignore` 排除。

### Phase 1 · CTX-001 独立门禁（pilot 已有，完整门禁未完成）

交付：

- 保留当前真实 retry、完整快照、trace、重启和 seq 续接 pilot。
- 增加 SQLite authorizer/write-set helper。
- 增加一次性候选源码上的 CTX-001 seam mutant patch。
- 增加防止实现与受保护 oracle 同改的 policy，并让 CI 区分 mutant kill 与环境失败。

验收：当前实现满足健康门禁；真实 seam DELETE/UPDATE mutant 因指定状态差异被杀死；普通单元测试改写不能绕过 Gate。完成前只能称 pilot。

### Phase 2 · 切断 Context 的持久化写入边

交付：

- `agent/core/passive_turn.py::DefaultReasoner` 删除 `session_manager` 构造参数和 `_session_manager` 字段。
- `run_turn` retry 成功记录 `selected_plan`、`disabled_sections` 和 window，不再调用 `trim_history_async`。
- history window 小于原窗口时调用名称明确的 runtime-only mutator；它只修改内存中的 `session.messages` 和 `last_consolidated`，不刷新 `updated_at`，也不调用 store。
- `agent/looping/core.py::_assemble_passive_runtime` 不再把 `self.session_manager` 注入 reasoner。
- `tests/test_session_compaction_runtime.py` 核对 runtime-only projection 不改写权威 Session，并固定 retry/commit 边界。旧 safety-retry 细分测试已在 2026-09-02 测试预算清理中移除。
- Phase 1 两个 case 开启对应的 runtime view 断言。

验收：`rg '_session_manager|trim_history_async' agent/core/passive_turn.py` 无匹配；prompt retry 的 protected store write set 为空；runtime view 的变化与 selected window 完全一致；retry trace 能解释本次发送了哪个窗口。

### Phase 3 · 清理含糊 API 并建立静态边界

交付：

- 全仓和 canonical plugin source 查询 `trim_history_async` 调用者。没有生产调用者就删除方法；runtime 裁切改用 `replace_runtime_history_view` 或等价的纯内存接口，该接口不能写 store。
- 历史 `ContextTrimPlan` 只描述过 prompt section，现已随旧 retry 路径退役；当前实现不得恢复同义兼容壳。
- 继续复用 `PromptRenderInput`、`PromptRenderResult`，不新增同义 context wrapper。
- 只有持久历史需要脱离 `Session` 加载时才引入 `SessionHistoryReader` 与 immutable snapshot。
- 显式用户删除逐步收口到 `SessionDestructivePort`，入口仍是 `agent/control/service.py::delete_thread`。
- 增加 import/AST policy，禁止 `DefaultReasoner`、`agent/prompting/**` 依赖 store 或 destructive port。

验收：prompt/context dependency graph 无 destructive edge；无修饰的 public `trim_history` 不再存在；control/dashboard 的显式删除测试仍然成功。

### Phase 4 · 变更治理

交付：

- `change-intent` schema/checker
- 受保护路径检查
- PR 模板中的 semantic delta、capability delta、oracle 和 rollback 字段
- base/candidate 差分回放框架

验收：超范围 diff、refactor 同改 oracle、未声明 destructive capability 会阻止合并。

### Phase 5 · 扩展其他 P0 不变量

按 `NOW.md` 顺序扩展 MEM-001、MEM-002、OUT-001、PLG-001、PLG-004、WSP-001 和 BAK-001。每次只新增一个独立 oracle 和一个 mutant，避免再次形成超大治理 PR。

## 14. 失败处理和回滚

### 文档迁移失败

核心文件改动前保留清晰备份。链接或结构验证失败时恢复单个文件，不改原工作区的用户未提交内容。

### Semantic test 引入后失败

先判断当前实现违反条款，还是 oracle 读取了错误 owner。不能先改断言。输出受保护状态的 before/after 和 write trace，由用户决定修实现或提出规格变化。

### 权限拆分影响显式删除

保留旧 destructive path 到 control/dashboard 的 adapter，先迁移 read-only caller。显式删除的现有行为用独立测试锁定，不能为了拿走 context 权限而破坏用户删除功能。

### 目标分支发生语义变化

暂停 rebase 后的实现工作，更新冲突表。用户确认新语义后更新任务合同；不能把 rebase 冲突的机械解决当成需求确认。

## 15. 完成定义

陌生维护者的文档阶段验收演练如下：

1. 从根目录找到固定读取顺序。
2. 用 CTX-001 定位“上下文裁切不得删除持久历史”。
3. 从决策索引找到事故原因和目标架构。
4. 从 `NOW.md` 找到尚未实现的 semantic gate 和权限拆分。
5. 复制任务合同，写出一个合格的 context refactor 变更声明。
6. 按 Phase 1 的文件、步骤和断言实现 CTX-001 黑盒测试。
7. 知道哪个文件能改、哪个 oracle 不能在普通 refactor 中顺手修改。
8. 知道何时自主继续，何时必须询问用户，何时停止并报告缺失证据。

Phase 1 完整完成时还要满足：

- 当前实现满足 CTX-001 semantic test。
- DELETE mutant 稳定失败。
- `semantic_delta: none` 同改受保护 oracle 会被 CI 拒绝。
- 测试报告能显示受保护状态、实际 write set 和失败原因。

当前 trace + fixture DELETE pilot 不等同于上述完整完成条件。只有 authorizer、真实 seam mutant、受保护路径 policy 和 CI 失败分类都落地后，才能把 Phase 1 标记为完成。届时“别人读完能动”才成为可演练的结果，不依赖作者在旁边补充口头背景。
