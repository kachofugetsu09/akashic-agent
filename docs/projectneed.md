# Akashic Agent 项目需求与语义不变量

这份文件是 Akashic Agent 的长期需求规范。它回答“系统必须保持什么”，供新会话、维护者、coding agent、评审者和 CI 使用。

实现细节、临时进度和历史讨论不放在这里：

- 当前未完成事项见 [`NOW.md`](NOW.md)。
- 新会话阅读入口见 [`INDEX.md`](INDEX.md)。
- 决策理由与勘误见 [`decisions/`](decisions/README.md)。
- 文档维护规则见 [`writing-rules.md`](writing-rules.md)。
- 上下文事故复盘与后续技术方案见 [`design/project-workbook-and-semantic-safety.md`](design/project-workbook-and-semantic-safety.md)。
- 当前持久化对象、owner 与待确认意图见 [`design/persistence-state-map.md`](design/persistence-state-map.md)。

## 1. 规范用语

- **必须**：合并门槛。违反即为错误，除非先批准需求变更。
- **不得**：已知危险路径。普通实现不得绕开。
- **应该**：默认工程方案。偏离时要记录理由和替代验证。
- **可以**：在不破坏其他条款的前提下自由选择。
- **权威状态**：用户和系统下一次运行仍应看到的事实。
- **运行时视图**：从权威状态派生、只为本次执行服务的临时表示。
- **语义变化**：用户可见行为、持久化结果、外部副作用、错误分类或数据保留规则发生变化。

条款 ID 是稳定引用地址。修改措辞时保留原 ID；含义改变时必须写决策记录，说明兼容性和迁移方式。

### 阅读路由

所有非简单任务读取第 1～6 节。再按任务范围展开：

| 任务范围 | 继续读取 |
|---|---|
| Prompt、上下文、会话、历史 | 第 7 节 |
| MEMORY、SELF、PENDING、Memory2 | 第 8 节 |
| AgentLoop、MessageBus、出站 | 第 9 节 |
| 插件加载、热重载、generation | 第 10 节 |
| Workspace、文件、Shell、迁移 | 第 11 节 |
| 调度、主动流程、备份、控制面 | 第 12 节 |
| 高风险 refactor、CI、验收 | 第 13～14 节 |

一个改动跨越多个 owner、会修改持久数据或会产生外部不可逆效果时读取全文。这样保留共同前提，也避免把无关领域全部塞入执行窗口。

## 2. 项目目标

### OBJ-001 连续、可恢复的个人 Agent

Akashic Agent 必须在多轮会话、进程重启、插件换代和工作区切换后保留用户授权保存的事实。临时预算、展示窗口和缓存策略不得改写数据保留范围。

### OBJ-002 可观察的自主执行

系统可以主动执行任务，但每个写入、发送、删除、进程和外部调用都必须有明确 owner、提交时机和失败语义。合法跳过、降级和故障必须可区分。

### OBJ-003 可演进而不丢语义

重构可以改变内部结构和性能，不得借“清理、裁切、压缩、统一、原子化”等名义改变未经批准的外部语义。高风险不变量必须由独立于实现的验收器保护。

### OBJ-004 每次协作从同一份现实开始

新会话不依赖维护者脑内背景、旧聊天记忆或 agent 猜测。项目工作手册必须用最少文本提供当前需求、决策理由、未完成事项和协作纪律。

## 3. 项目工作手册

### WBK-001 共享现实

`INDEX.md`、`projectneed.md`、`NOW.md`、`decisions/`、`writing-rules.md` 和根目录 `AGENTS.md` 共同构成项目工作手册。它们必须进入版本控制，不能只存在于某个工作区、会话或个人记忆中。

### WBK-002 文档各司其职

| 文档 | 只回答什么 | 不得混入什么 |
|---|---|---|
| `INDEX.md` | 新会话先读什么、怎样按任务继续展开 | 产品语义、临时进度、历史全文 |
| `projectneed.md` | 长期需求和语义不变量 | 临时进度、会话转录、易过期测试数字 |
| `NOW.md` | 当前尚未完成什么 | 已完成记录、长期设计说明 |
| `decisions/` | 为什么作出某项决定，何时被取代 | 待办清单、无结论讨论 |
| `AGENTS.md` | coding agent 如何开工、核对和交付 | 具体模块的临时实现方案 |
| `writing-rules.md` | 文档写到哪里、怎样保持一致 | 产品需求本身 |
| `design/` | 一个问题的技术结构、迁移和验收 | 项目全部长期需求的副本 |

### WBK-003 按需展开历史

新会话默认读取当前工作手册。历史记录、旧会话和记忆库只在当前材料不足时按主题查询；查询结果必须用当前代码、当前配置或当前权威数据复核。不得用自动注入的陈旧记忆覆盖当前事实。

### WBK-004 完成即剔除

一项工作完成后必须从 `NOW.md` 删除。完成记录由 Git、PR 和决策记录承担。`NOW.md` 只保留现状、阻塞和未完成事项，让新会话准确找到接手点。

### WBK-005 转述不能代替引用

需要复用需求时优先引用条款 ID 或相对链接。不得把关键约束反复改写成多个版本；转述带来的含义变化必须先回到原条款核对。

### WBK-006 新会话从索引进入

每个新会话进入仓库后的第一个主动读取动作必须是 `docs/INDEX.md`。索引必须给出稳定的公共入口、按任务分类的继续阅读顺序、权威冲突规则和真实证据入口。索引只做路由，不复制需求正文；执行者按需展开，不能用“节省上下文”为理由跳过公共合同，也不能把全部历史无差别注入。

## 4. 协作与变更治理

### COM-001 核对先于假设

需求中的空白如果会改变持久状态、外部副作用、权限、数据保留或兼容性，agent 必须先写出自己的理解，明确会改变与不会改变的对象，并等待用户确认。低风险局部假设可以继续，但必须显式标注，不能伪装成已核对事实。

### COM-002 沟通成本按风险分配

核对不要求每一步都请示。可逆、局部、语义明确的实现由 agent 自主完成；不可逆、跨层或多种解释会产生不同后果的决策必须提前核对。高风险核对应该成为默认路径，事后补救不能替代开工确认。

### COM-003 执行时收窄，问责时展开

执行者只获得完成任务所需的代码、接口和权限，减少无关上下文。评审者必须能读取完整 diff、相关需求、决策、测试、日志和状态变化。信息隐藏用于降低执行负担，不能用于遮蔽责任和证据。

### COM-004 树状执行，网状信息

主 agent 可以把任务分成树状子任务；所有参与者仍必须从同一版本的工作手册和代码基线出发。跨任务的事实、接口变化和阻塞要写回共享状态或明确发送，不能只沿层级口头转述。

### GOV-001 开工前声明语义变化

非简单改动必须声明 `change_type` 和 `semantic_delta`，列出允许变化、受保护状态、允许副作用、关联不变量和验证方式。`semantic_delta: none` 表示所有外部行为和持久结果保持不变。

### GOV-002 规格变化与实现变化分开批准

普通 refactor 不得同时改实现和受保护语义来制造全绿。需求或不变量的变更要先批准规格和决策，再修改实现。实现者可以补普通单元测试，不得独自降低语义 oracle。

### GOV-003 Diff 是评审单位

评审以基线与候选之间的 diff 为单位，重点检查新增权限、写集合、删除路径、错误分类和外部副作用。大规模格式变化、无关重构和批量改名不得遮蔽语义差异。

### GOV-004 一个高风险语义一个可审阅改动

数据、权限、会话、记忆、插件发布和外部发送等高风险改动应该按不变量拆分。不得把大量相互独立的重构塞入一个无法逐条核对的 PR。

### GOV-005 Worktree 不得制造私有现实

独立 worktree 必须记录目标分支和基线提交。开工前读取该基线中的工作手册，验收前同步目标分支并检查工作手册差异。同一份权威文档、语义契约、分支或 worktree 同一时刻只允许一个 writer；其他 agent 可以并行只读评审，但不得在同一 worktree 写文件、提交或切换分支。

Writer 交接前必须把允许范围内的修改提交成可引用 commit，或恢复到明确的 clean HEAD，再记录 worktree、分支、HEAD、dirty state 和下一位 owner。共享文件系统不等于共享写权限；没有完成交接的后台 agent 不得继续提交，接手者也不得把来源不明的 merge 或文件变化当成自己的结果。

### MOB-001 核心按权威语义演进，不按客户端便利性扩张

移动端提出的需求默认由移动端仓库或客户端适配层拥有。修改 Akashic 核心运行时必须同时证明：该能力属于既有或已批准的 Akashic 语义；权威状态或跨客户端一致性确实由核心或中立协议拥有；接口不包含 Android、iOS 或单一产品界面细节；只在客户端实现会复制、猜测或破坏权威语义。

“未来可能复用”“所有移动端可能都需要”“放在核心更方便”不能单独成为 runtime patch 的理由。平台普遍能力仍由平台层拥有，例如 Android 前台服务、通知、Room、缓存、图标和手势；Akashic 移动端专属交互仍由移动端产品拥有，例如命令面板和富文本展示。只有 session、turn、ack、resume、附件传输确认、取消终态等需要服务端权威状态或跨客户端一致语义的能力，才进入核心或中立协议边界。

跨仓库客户端任务必须在开工和评审时记录 `capability_owner`、`consumer_scope`、`runtime_patch`、`runtime_patch_reason`、`authoritative_state_owner` 和 `client_only_alternative`。存在核心改动却无法填写这些字段时停止并等待维护者确认，不得用候选实现反向证明核心本来就应拥有该能力。

### MOB-002 投影重建只减少可重建服务端投影

移动端从服务端 session、message、turn、事件和历史页得到的本地行属于可重建投影；`sync.reset_required`、cursor 回退或历史重拉只能清理正向白名单中的服务端投影和对应 cursor。它们不得删除 outbox、pending/failed 本地消息、附件 draft 与 transfer、待投递通知、持久 stop 或其他尚未完成的本地工作。

服务端明确删除 session 不属于投影重建。它与本地未完成工作如何共同展示、阻止或减少仍需独立产品决定；确认前不得借 reset、外键 cascade、`clearAllTables()` 或 destructive migration 偶然删除本地连续性对象。

### MOB-003 协议语义不能由语言原生类型偷偷改写

跨语言协议的长度、顺序、终态、取消和迟到响应由协议定义，不由 Python、Kotlin、JavaScript 或数据库的默认 primitive 定义。协议说 Unicode code point 时，各端都按 code point 验证；协议说请求已取消时，已知取消请求的迟到响应可以忽略，未知 response ID 仍须 fail-loud。临时命令目录等连接级投影在 reconnect、reset、source 变化或 terminal close 后失效，不能伪装成持久权威状态。

### MOB-004 数据库迁移识别真实 schema lineage

数据库 `user_version` 只表示版本号，不能单独证明表、列、索引和外键形状。若多个已发布或已评审分支曾使用同一版本号但 schema 不同，迁移必须识别每一种已知 lineage，逐一验证保留集合并汇合到唯一目标 schema；未知或部分匹配的形状 fail-loud，不得猜测、清库或用 destructive fallback 获得启动成功。

迁移验收至少覆盖每个已知来源 schema 到最终版本的真实建库与数据保留，并提交当前目标 schema identity。Stacked PR 的最终 head 必须同时保留所有上游持久状态，不能只证明其中一条相邻迁移路径。

## 5. Agent 任务合同

本节参考 [OpenAI · Prompting guidance for GPT-5.6](https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6)，并按本项目的数据与权限边界收窄。外部指南提供设计依据，不会自动覆盖本文件条款；指南更新需要评审后再修改 PRM 条款。

### PRM-001 Prompt 从结果和完成标准开始

复杂任务的 prompt 必须先写用户可见结果、成功条件和停止条件，再补约束、证据与工具。只要模型可以自行选择安全路径，就不逐步规定实现过程。安全、权限、数据和业务不变量继续使用明确的“必须/不得”。

### PRM-002 指令只保留一个权威版本

同一条审批、语言、验证或副作用规则只写一次。任务 prompt 使用条款 ID 引用项目规则，不复制整段。发现两条指令对同一情形给出不同动作时，开工前解决冲突。

### PRM-003 自主范围和批准边界明确

prompt 必须区分只读调查、设计、实现、评审和外部协调。读取项目材料、修改已授权范围内的本地代码和运行非破坏性验证可以自主完成；外部写入、破坏性操作、付费动作和显著扩展范围需要确认。

### PRM-004 工具路由说明前置条件

工具说明必须交代用途、适用时机、关键返回值和失败含义。正确性依赖检索、发现或校验时，这些步骤是写入前置条件。独立读取可以并行；结果会改变下一步决策时保持串行；并行结果在写入前统一汇总。

### PRM-005 缺少证据时使用最小补救

工具返回空、部分结果或异常狭窄结果，需要尝试一到两个有意义的替代读取。关键事实仍然缺失，就指出缺口并缩小结论或提出最小问题。没有证据不能自动推导成事实不存在。

### PRM-006 每轮检查是否已经满足目标

每个主要工具结果都需要触发一次目标检查：核心请求能否用现有证据完整回答。已经满足成功标准就停止搜索；缺少必要事实就只补最小缺口。减少工具轮次不能压过正确性、必需验证和用户要求的证据。

### PRM-007 长任务只在阶段变化时更新

首次工具调用前给出一到两句开工说明。后续只在主要阶段开始、关键发现改变方案或出现真实阻塞时更新。更新内容包含一个具体结果和下一步，不逐条播报常规工具调用。

### PRM-008 Prompt 变化按真实样例回归

优化 agent 指令时先保留当前模型和 reasoning 基线，用代表性任务建立结果。每次只删除或修改一组指令、示例或工具，再跑同一组样例。只有结果继续满足原验收时，token、延迟和成本下降才算改进。

## 6. 状态分类与权限边界

所有设计先确定状态类别，再决定谁能修改：

```text
┌────────────────────┐
│ A. 权威持久事实     │  sessions/messages、MEMORY/SELF/PENDING、jobs、plugin-data
└─────────┬──────────┘
          │ 只读快照
          ▼
┌────────────────────┐
│ B. 受保护派生索引   │  embeddings、FTS、vec index、可重建 sidecar
└─────────┬──────────┘
          │ 派生
          ▼
┌────────────────────┐
│ C. 临时运行时视图   │  PromptContext、窗口、cache、candidate、snapshot binding
└────────────────────┘

┌────────────────────┐
│ D. 外部不可逆效果   │  消息发送、远程 API、子进程、服务切换、文件发布
└────────────────────┘
```

### STA-001 权威状态只有一个 owner

每类权威状态必须有唯一拥有层。其他模块使用窄接口读取或请求变更，不能各自维护一份可独立漂移的“真相”。

### STA-002 临时视图不得反向定义保留策略

C 类对象可以裁切、再建和回收，但不能因为自身容量、显示或性能要求删除 A 类事实。B 类索引可由显式维护流程再次生成；运行时降级不得顺带改写 A 类事实。

### STA-003 每类持久状态都要声明增、改、减

持久状态的设计和评审必须分别说明：正常运行怎样增加数据、哪些字段允许原位更新、什么动作造成逻辑失效、什么动作造成物理删除，以及每种变化由哪个 owner 执行。逻辑 supersede、消费完成和状态终结不等于物理删除。没有明确减少协议的对象，普通运行、重构、容量优化和缓存清理都不得自行减少。

### CAP-001 权限和接口按任务最小化

只读计算接收只读快照；元数据更新只获得白名单字段 writer；删除由独立 destructive port 拥有。不得向上下文、展示、检索或验证模块传入带任意写入和删除接口的 repository。

### CAP-002 外部效果必须有提交协议

D 类效果只能由拥有 prepared、committed、failed 和必要补偿语义的层执行。只恢复内存指针不能算外部世界已回滚。

### ERR-001 失败必须保留含义

不存在、空结果、合法跳过、明确降级、输入错误、数据损坏和内部故障必须可区分。只有拥有正确恢复动作的边界才能捕获异常并降级；其余错误 fail-fast、fail-loud。

## 7. 上下文和会话

### CTX-001 上下文裁切是非破坏性投影

本次 `PromptContext` 可以因模型窗口超出预算而改变。当前进程的 runtime history view 只有在选中的 history window 确实缩小时才能缩短。只移除 skills、memory 等动态区块不得改变 runtime history view。`sessions.db/messages`、`message_embeddings` 和完整历史内容必须保持不变。上下文裁切不承担归档、保留或删除职责。

验收至少核对裁切前后的完整持久快照和数据库 write set；关闭并再次加载后仍能看到全部历史；追加消息从原最大序号继续。

### CTX-002 先移除可再生内容

预算不足时按耐久等级处理：先减少装饰性内容，再减少可再次查询的 skills、meme、长期记忆和检索结果；只有这些内容已经移除仍超限，才缩小发送给模型的历史窗口。当前用户指令和最近完整语义回合优先保留。

### CTX-003 窗口以完整语义回合为边界

Prompt 历史不得从孤立 assistant 或 tool result 开始。assistant 工具调用和对应结果成对保留；合法 user 边界或明确的 proactive assistant 边界拥有窗口起点。长工具结果只允许在临时模型视图中截短。

### CTX-004 派生上下文不得伪装成用户原话

skills、长期记忆、检索结果和 recent context 必须带来源和信任级别，作为 system context 或独立数据块进入请求。当前 user message 始终独立；工具授权不能由提示词内容决定。

### CTX-005 新设计不得使用无修饰的 history

新增接口、变量和设计文档必须区分 `persistent history`、`runtime history view` 和 `prompt history`。只写 `history`、`trim history` 或 `replace history` 且无法判断对象类别，设计不能通过评审。

### CTX-006 在里程碑压缩，保持任务函数不变

长任务只在完成调查、确定设计、完成实现或完成验证等主要里程碑后压缩上下文。压缩结果至少保留目标、成功标准、已核对事实、关键假设、决定、未完成事项、文件/条款引用和验证状态；格式见 [`templates/context-handoff.yaml`](templates/context-handoff.yaml)。压缩内容是当前任务的 opaque handoff，不得把摘要措辞反向当成新的项目需求。

### SES-001 回合持久化全有或全无

同一批 session metadata、消息和序列分配必须在一个事务中提交。任一步失败时数据库不出现半批消息，内存对象也不得获得并不存在的稳定 ID。

### SES-002 消息序列单调且不复用

同一 session 的 seq 在数据库事务内唯一递增。裁切运行时视图、进程重载和并发追加不得降低高水位或复用旧序号。

### SES-003 破坏性删除只接受用户显式意图

删除 session、messages 或随之级联的派生索引，必须来自用户主动发起的撤销或删除操作，并经过名称明确的管理命令。命令必须携带用户动作来源、精确目标、cascade 语义、备份和审计证据。裁切、压缩、检索、展示、重放、保留期猜测和普通 refactor 不得调用这些接口。

### SES-004 损坏数据在存储边界失败

存储层遇到持久化 JSON、列类型、tool chain、metadata、embedding BLOB 或维度损坏，必须带 session/message 上下文抛错。不得返回空列表、空对象或 cache miss。

### SES-005 对话正文在正常运行中只追加

`sessions.db/messages` 保存完整对话正文。正常收发消息只能 INSERT 新行，并在所属 session 内分配单调、不复用的 `seq`；不得 UPDATE 或 DELETE 既有正文。只有用户主动撤销消息或删除会话时，独立数据管理操作才可以按 SES-003 减少数据。`sessions` 元数据和 `turns` 状态可以按各自状态机原位更新；FTS、embedding 等派生索引可以随显式撤销/删除同步变化或通过独立维护流程重建，但不能反向决定原始消息的保留。旧消息编辑是否允许原位 UPDATE 不由本条猜测，必须另行确认后再形成条款。

### SES-006 附件随消息引用保留

消息仍引用的附件属于会话数据，必须保持可读。附件清理只能从完整引用关系出发，先识别真正孤儿，再经过 dry-run、备份和名称明确的删除操作；在引用计数、cascade 和恢复协议落地前不得自动 GC。文件年龄、当前 prompt 是否使用、索引是否命中和代码重构都不能成为删除依据。

## 8. 记忆系统

### MEM-001 档案重写同时验证结构和事实保全

替换 MEMORY 或 SELF 要求模型输出 Markdown 结构合法、必需 section 完整，且受保护事实没有无理由消失。结构合法不代表语义完整；删除 pinned fact 必须有显式 tombstone、来源和理由。

### MEM-002 PENDING 合并遵守两阶段事务

优化开始时冻结旧 snapshot，处理中到达的新事实写入新 PENDING。只有 MEMORY 提交成功后才能删除 snapshot；异常、取消和重启恢复必须把旧 snapshot 与新追加按顺序合并，事实不能丢失。

### MEM-003 破坏性重写前留下不可覆盖恢复点

MEMORY、SELF、RECENT_CONTEXT 和 PENDING 使用同目录临时文件、fsync 与原子 replace。覆盖前保留已校验的唯一历史备份；备份失败时不得继续覆盖。

### MEM-004 事实摄入按 source_ref 幂等

同一 source_ref/kind 最多追加一次。文件和索引任一侧领先时，恢复流程必须确定性收敛，不能出现两份相同事实或漏记；无法判定的分叉显式失败。

### MEM-005 canonical 事实与派生索引分离

`memory_items` 拥有事实，向量和 FTS 只负责加速。索引写入、删除或初始化失败后立即停用该索引，使用 canonical full scan 或显式失败，不能继续查询已不可信索引。

### MEM-006 只有可恢复 lane 才能降级

关键词 lane 仍能给出合法结果且外部 embedding lane 失败，系统可以带降级证据继续。MemoryStore 读取、反序列化和形状错误必须传播；取消不得被转换为空召回。

### MEM-007 每次转次拥有冻结上下文

session、channel、chat、source_ref 和预算在每次 post-response run 创建的不可变上下文中传递。并发转次不得共享实例可变字段；本轮新增记忆不能在本轮被立即 supersede。

### MEM-008 长期记忆状态不可互相替代

`MEMORY.md`、`SELF.md`、尚未提交的 `PENDING.md` 和 `memory2.db` 都属于必须持久保存的记忆状态。前三者分别承担人类可读档案、自我档案和事务队列，`memory2.db` 保存结构化记忆、强化、替换和人工管理结果；只保留其中一份不能证明可以无损恢复其余内容。`RECENT_CONTEXT.md` 是可重建投影，不拥有这些长期事实。

### MEM-009 Akasha 使用固定输入确定性重建

`akasha.db` 和 graph snapshot 是派生 sidecar。完整重建只读取 `sessions.db/messages`、对应的 `message_embeddings`、固定算法和固定配置，不引入 LLM 重新解释历史，也不重新生成已经存在的 embedding。同一组输入必须得到可复现的图；缺少或模型不匹配的 embedding 必须使完整重建失败并报告缺口，不能静默跳过消息后仍声称成功。

## 9. 运行时、并发和出站

### RUN-001 同一聊天中被动回复优先

同一 channel/chat 下，主动、计划和工具发送必须等待正在执行的被动 turn 与被动 outbound 完成；多个非被动发送严格 FIFO。不同聊天可以并行。

### RUN-002 取消和异常不能卡死通道

等待 ticket 被取消时必须跳过；发送失败必须复位 sending 并通知等待者；空闲 chat state 最终回收。原始错误继续向 owner 暴露。

### RUN-003 活动回合的 owner 唯一

AgentLoop 唯一拥有活动 turn task 的取消和 cleanup。无论成功、失败或取消，都恢复临时 session context。terminal event、inbound complete 和 delivery ack 各自由一个层提交，保证恰好一次。

### OUT-001 未送达内容不得进入可见历史

dispatch 明确成功是写入用户可见历史、presence、dedupe 和 success 状态的前置条件。部分送达必须有独立状态，不能冒充成功或完全失败。

### OUT-002 回合副作用的顺序和分支确定

通用、成功和失败副作用必须属于明确 phase。单个独立副作用失败时继续尝试同阶段其他项并保留所有错误；失败分支不得运行成功副作用。

## 10. 插件 generation 与 snapshot

### PLG-001 候选插件不得污染正式状态

候选在 commit 前只能使用 generation 私有 staging、只读 session/memory 和 staged event bus。初始化失败后，正式 KV、session、memory、事件和外部服务必须与开始前一致。

### PLG-002 单次 reconciliation 使用一个发现快照

同一轮候选准备、禁用和发布使用同一个不可变 topology revision。扫描后的文件变化只进入下一轮。

### PLG-003 在途请求绑定同一 runtime snapshot

一个 turn、job、event 或 proactive tick 从 admission 到结束只能看到同一 snapshot。新代只服务新请求；旧代在全部 lease 归零后清理。无主后台任务不得意外继承绑定。

### PLG-004 发布对外观察必须原子

candidate 在所有 invariant 通过前不接受公开请求。commit 临界区一次切换 current 与 admission；失败继续使用 previous。恢复指针但无法撤销已发生外部效果不算完整回滚。

### PLG-005 独占 endpoint 先停 admission 再排空

端口、channel 和 managed service 换代前暂停新请求，等待旧 lease 归零，再切换 endpoint。失败时先恢复旧 endpoint 和 admission，随后清理候选；持有当前 lease 的调用栈不得发起会等待自身的切换。

### PLG-006 清理逆序、抗取消并保留全部失败

插件 task、process、subscription 和 catalog cleanup 按注册逆序执行。调用方取消不能截断清理；每项都要尝试，完成全部清理后聚合错误。

### PLG-007 Watcher 单轮失败不终止生命周期

一次 scan 或 reconcile 失败只影响当前 revision，旧插件继续服务。相同失败 revision 不得无限重试；后续变化或显式 wake 可以恢复。stop 必须可等待。

### PLG-008 动态协议和冲突 fail-loud

active 检查错误、generation key 错配、名称冲突、依赖缺失和拓扑环必须在发布前拒绝。不得以“后注册覆盖前者”或默认 active 掩盖错误。

### PLG-009 Skill 和 MCP 通过插件安装发布

Skill、Drift skill 和 MCP server 都由插件包声明并通过插件安装系统进入 Akashic。插件的 `skill_roots`、`drift_skill_roots` 和 `mcp_servers` 是能力来源；安装阶段准备代码与 MCP runtime，generation readiness 全部通过后再原子发布 catalog。workspace 中的 skill 软链接只是当前插件 generation 的可重建投影，不是 canonical source。独立 `mcp/servers/*.toml` 和 workspace 内手工 skill 目录不属于目标安装模型；现有兼容路径必须迁移到插件，不能继续扩展成第二套能力所有权。

### PLG-010 卸载插件默认保留 plugin-data

插件代码、安装清单和 workspace 内 `plugin-data` 使用不同生命周期。普通卸载只移除插件 cache、manifest entry 和能力投影，必须保留 `<workspace>/plugin-data/<plugin>-<marketplace>/`。永久删除插件数据需要名称不同的用户操作、影响预览、独立备份和再次确认，不能作为卸载的隐式 cascade。

## 11. Workspace、文件和进程

### WSP-001 Workspace 可写状态显式归属

会话、记忆、附件、plugin-data、socket、运行日志和运行密钥都从显式 workspace 派生。全局插件缓存和 credential store 必须列入明确 global state 清单；运行时不得隐式回退 HOME。

### WSP-002 数据路径不能通过片段或符号链接逃逸

plugin、marketplace、snapshot 等名称必须是安全单片段；resolved path 位于 workspace；已存在父组件不得是 symlink。高风险写入需要 OS 级 no-follow 或隔离边界。

### WSP-003 数据迁移离线、持锁并原子发布

迁移先获得 workspace 单实例锁。SQLite 使用在线 backup 与 integrity check；全部内容写到唯一 staging，再一次性发布。目标已存在时拒绝合并，源数据保留到独立清理步骤。

### WSP-004 Workspace 是 Akashic 运行数据根

`<workspace>` 表示由 `--workspace`、`AKASHIC_WORKSPACE` 或主配置选中的 Akashic 运行实例主要工作区。它承载会话、长期记忆、附件、调度、主动流程、plugin-data、能力投影、诊断和运行控制状态，不是源码仓库、Git checkout 或 Git worktree。插件代码、Skill/MCP 的 canonical source、全局插件清单和凭据可以位于 workspace 之外，必须作为明确 companion state 管理。Git worktree 只承载代码、测试和项目工作手册；任何代码 worktree 都不得把自己的目录当成正式运行数据根。

### FS-001 文件写入限于 allowed root

路径按 canonical target 校验；同一目标的 mutation 串行，不同目标可并行；写入使用原子替换。失败和取消必须释放锁并保留旧完整文件。

### FS-002 Edit 精确且无歧义

old text 不存在时失败；多次匹配而未声明 replace-all 时拒绝猜测。编辑保留 BOM、换行和 mode，并在锁内重读最新内容。

### SH-001 Shell 生命周期有界

前台超时或取消终止整个进程树；后台任务有硬上限、状态查询和显式 stop。路径字符串检查只防误操作，不得冒充安全沙箱；运行不可信命令使用容器、namespace 和最小权限。

## 12. 调度、主动流程、备份和控制面

### SCH-001 损坏调度文件不得解释为空任务集

只有文件不存在可以返回空。I/O、JSON、根类型、字段、相同 ID 出现两次、时间和枚举错误都必须带路径与索引失败；损坏状态不得进入 `save([])`。

### SCH-002 调度状态按候选提交

add、cancel 和 reschedule 先构造 candidate，持久化成功后才替换内存。stop 后不再产生新 tick；关闭回收 in-flight，停止期间周期任务不重排。

### PRO-001 主动流程的空、跳过和失败可区分

sensor、session、context 和 store 故障不得伪装成“无事件”。decision、dedupe、presence 和成功状态只在真实送达后提交；合法 skip 带 reason，内部错误可观察。

### PRO-002 主动、Wake 和 Drift 状态必须连续恢复

`proactive.db`、`wake_proactive.db` 和 `drift/drift.db` 中影响 delivery dedupe、cooldown、pending ack、reservoir consumption、hazard timer、cursor、journal 和下一轮选择的内容属于运行连续性。启用对应功能时，备份和恢复必须保留这些状态；日志表与连续性表可以制定不同 retention，但不得把整库按诊断日志清空。

### PRO-003 Wake 用主动历史保持连续性而不预设重复惩罚

Wake 判断内容时必须把最近被动对话与已经送达的主动消息作为两个明确区分的运行时区块；主动消息保留实际发送时间，不能伪装成用户陈述或本轮候选。模型把主动历史作为理解近期连续性的背景，并保持对用户及其关注事项的开放好奇；话题聊过、结论相同、事件反复发生或发送次数较多，都不能单独推导出用户疲劳、不感兴趣或禁止再次分享。当前事件是否值得主动告诉用户仍由模型结合正文证据、长期偏好、真实用户反馈、最近上下文和时机自主判断，不增加按主题、URL、相似度或次数硬编码的 share/skip 规则。

### BAK-001 备份必须能验证和恢复

普通文件完整复制，SQLite 使用 backup API 与 integrity check；临时 snapshot、manifest 和 hash 全部完成后原子发布，新快照成功后才 prune。必须定期恢复到隔离 workspace 并运行应用级只读 smoke。

### CTRL-001 控制协议严格握手和 typed params

连接先完成版本、token 和 initialize/initialized 握手。JSON-RPC envelope、method 和 params 在唯一边界严格校验；未知字段和宽松类型转换不得触发高权限动作。

### CTRL-002 控制 owner、thread、turn 和终态一致

一个 workspace 同时只有一个 runtime owner。本地控制 socket/token 不能跨 workspace；turn terminal 每次恰好一次，送达前断连标记失败。thread/delete 是显式破坏操作。

## 13. 独立验收要求

### TST-001 语义 oracle 独立于实现

P0 不变量必须由受保护的 semantic test、policy 或黑盒观察器验证。普通实现 agent 不得在同一 refactor 中同时修改 oracle 的预期结果。

### TST-002 核对完整状态和 write set

持久化语义不能只核对返回值或行数。验收应规范化完整内容，记录 INSERT、UPDATE、DELETE、文件写入、事件和外部调用；即使违规事务最终回滚，也要看见写入尝试。

### TST-003 用已知错误验证验收器

每个 P0 oracle 应有至少一个语义 mutant 或等价故障注入。例如 CTX-001 主动加入 `DELETE FROM messages` 后，门禁必须稳定失败。如果已知错误仍能通过，测试本身没有完成验收职责。

### TST-004 Refactor 做差分回放

`semantic_delta: none` 的高风险重构应在 base 和 candidate 上回放同一组脱敏输入。Prompt 文本可以按声明变化；持久 write set、事件、外部调用、错误分类和用户可见结果不得出现未声明差异。

### TST-005 可恢复性要实际演练

备份、rollback 和 previous snapshot 只有经过隔离恢复、重载和关键路径 smoke 后才算有效。文件存在或指针恢复不能单独证明可恢复。

### TST-006 变更影响由版本化 Gate 决定

代码改动必须由版本控制中的 capability、state 和 scenario 索引解释，再从 Git diff 选择语义场景。未知可执行改动先运行全量公开场景，最终仍要 fail-loud，不能由实现者临时猜测或缩减测试。每个场景使用一次性测试 workspace、plugin home、config 和 HOME，不读取正式运行状态。

公开 Gate 只输出能力组、场景和 plan/source/catalog digest，不要求贡献者安装私有插件，也不得暴露 provider 身份。private runtime 用同一 plan digest 把能力组映射到真实 provider；`privateGateRequired=true` 时，公开 Gate 通过只表示 G1 完成，跨仓库总体验收在 G2 明确返回 `passed`、`failed` 或 `not_affected` 前仍属待验证。

跨仓库 Gate 必须从私有 catalog 的 GitHub `repository + 完整 ref` 查询远端当前 revision，并在开跑前冻结成 commit SHA。安装、验收和报告都绑定该 SHA；本地 checkout、remote-tracking ref、开发者机器上的插件 cache 或手工传入路径不能满足 required check。这样主仓库候选始终与 Gate 启动时远端真实插件版本组成一个明确、可复现的验收组合。

### TST-007 跨仓库证据绑定不可变组合

跨仓库报告必须同时绑定 consumer commit、协议 source repository/commit/path/hash、运行时 commit/tree、provider repository/requested ref/resolved commit、scenario catalog/profile/hash 和 Gate 版本。协议历史源与当前运行时可以来自不同 commit，但两者都必须单独固定；分支名、PR URL、浮动 GitHub 链接、本机 checkout 和已安装 cache 不能代替不可变身份。

任一输入变化都会形成新的验收组合，旧报告仍可作为历史证据，但不能复用为新组合通过。客户端离线快照的源文件必须存在于固定 commit；核心需要保留已发布协议的归档来源，不能让后续 schema 演进使旧客户端的 source pin 失效。

### TST-008 CI 与真实设备证据分层报告

确定性单测、构建、Docker Gate、隔离互操作和真实设备分别证明不同边界。CI 没有 Pixel 或 Android 虚拟设备时，不能把维护者本机 ADB 结果伪装成所有贡献者可运行的 required check；设备结果必须记录设备/API、应用 ID、APK 与源码身份、测试 profile 和实际场景。

设备 Gate 只接受干净 source commit/tree，同一 Android worktree 同时只运行一个 Gate。候选构建完成后、首次 ADB 调用前必须再次核对 worktree clean、HEAD 和 tree 与起始值相同；任一漂移必须以 `failed_setup` 且零设备调用结束，不能把构建期间产生的未提交 APK 归因到旧 commit。在任何安装、清数据或卸载之前，必须从本次生成的 app/test APK 读取实际 application ID 与 instrumentation target，并为本次 run 生成唯一的 run-specific application ID。随后用 `pm list packages -u` 核对设备上已安装和保留数据的 package；app 或 test package 任一 collision 都必须 fail-loud 并标记 blocked，不能用签名相同、版本较旧、`adb install -r` 或“只是 debug 包”推断可以覆盖。安装不得 replace；只有本进程确认安装成功的 package 才取得清理所有权，部分安装失败不得卸载未拥有的 package。

`adb shell am instrument` 的进程退出码不能单独充当 oracle；Gate 必须核对声明的测试数量、指定方法、开始/成功状态和失败标记，0 test、crash、aborted 或 assertion failure 都不能记为通过。测试阶段通过后仍不能提前声明 Gate 通过；清理完成后才能写唯一终态。清理失败必须非零退出、标记 `gate_result=failed_cleanup` 并列出残留 package。

正式应用及设备上既有 package 属于受保护状态。Gate 必须记录测试前后的 package、版本、安装身份和可观察数据身份，且不得覆盖、卸载、清空或连接正式应用状态。`base.apk` 只能恢复 binary，不能代替 app data 备份；若任务确实需要触碰既有 package，必须另获授权并先取得经过恢复演练的数据级备份，否则 blocked。测试结束还要证明 run-specific app/test package、ADB reverse、容器和测试 workspace 已清理。CI 继续承担固定逻辑的可重复 Gate，维护者设备只补充 OS lifecycle、Room migration、通知、文件系统和真实 Compose 交互证据。涉及实时 Gateway 的设备证据还必须绑定 Mobile Lab core SHA、run ID 和非正式配对来源；客户端 package 隔离不能证明服务端 workspace 已隔离。

## 14. 需求变更流程

1. 指出受影响条款、当前语义和拟议语义。
2. 说明为什么现有语义不再成立，以及对持久数据和外部行为的影响。
3. 新建决策记录；breaking 变化写迁移、备份、回滚和兼容窗口。
4. 先批准规格变化，再提交实现。
5. 更新或新增独立 oracle，并用语义 mutant 验证。
6. 实现完成后从 `NOW.md` 删除对应事项。

证据不足的步骤沿用现有条款，不能用实现代码反向推导“需求原本就是这样”。
