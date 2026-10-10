# 兼容截止：hua-home 数据盘点

- 状态：只读事实盘点 / 后续方案 proposed；不授权正式数据转换或删除。
- 日期：2026-10-11（Asia/Shanghai）。
- 源码：边界①栈顶 [#1241](https://github.com/kachofugetsu09/akashic-agent/pull/1241)，盘点基线 `b7bf1e7d`。
- 数据：`hua-home:/srv/data/services/akashic/state/workspace`。
- 约束：[状态地图](persistence-state-map.md)、[ADR-0066](../decisions/0066-yoyo-current-baseline.md)、[部署手册](operator-deployment.md)。

## 结论与范围

可以让后续运行只接受一种当前结构，但必须先证明旧事实仍能读取、身份与恢复含义没有改变。
当前正式数据确实使用若干旧表示，直接删除读取分支会让下一次启动或历史读取失败。
历史审计事实、恢复协议和配置拒绝入口不能按 `legacy` 名称统一删除。

本次只读取激活记录、当前插件选择、SQLite 结构与汇总、文件大小及回执；数据库使用
`mode=ro` 和 `query_only`，没有实例化会初始化或升级数据库的 runtime store。
没有执行迁移、改写消息、删除数据、部署或启动服务。正文、配置值和凭据不进入报告。
每个数据库独立读取，本报告不是跨库一致备份，也不是恢复演练。

激活记录的 Core 是 `bdcc1660b60e721f70004f997eccac32735f0ea1`，不是 PR 栈顶。
检查时 Core、Host Bridge 和 home-services 的 systemd unit 已停止，Core 与 workloads
容器已退出；Room 仍在运行。`active.json` 的 `active` 是此前激活结果，不能当作当前
运行证明。本报告说明保留数据，不证明新栈在正式实例上的行为已经验收。

## 实际数据与源码判断

| 对象 / owner | hua-home 实际形状 | 清理判断 |
|---|---|---|
| Session / MessageLog | 737 个会话；`sessions` 仍含旧列，已有 attributes、软删除和 title；metadata 为 NULL 的 424 行，旧 last_consolidated 非零 2 行、last_user_at 非空 213 行、last_proactive_at 非空 3 行 | 需要显式结构转换。旧列含真实值，NULL/default 含义与旧列保留需确认，不能在重建表时顺手丢掉 |
| Message / MessageLog | 33,012 条消息；当前消息表含 metadata；ToolResult 有 3 条 `unknown` | 旧 outcome reader 有真实消费者；保留外部效果不确定的事实，不直接归为确定失败。消息正常路径只追加，不因截止格式授权重写正文 |
| 入站接纳 / Channel 与 custody owner | `inbound_handoffs` 0 行；旧 compaction prepare 0 行 | 这些表的现有数据无需转换；不代表所有去重消费者或持久恢复机制可删 |
| 插件选择 / PluginSelection 与安装 owner | 65 个当前输入全部为 v5：52 builtin、13 installed；外层 stable 已是 v2 | v5 reader 当前有真实消费者。按安装来源生成 v6 输入并重新提交选择，不能只改 version |
| 绑定 / Bindings 与服务合同 owner | 3,698 条：v1 2,792 条、v2 906 条；`core.commands` 2 条 | 原 descriptor 是哈希身份的一部分，不能直接 UPDATE 改名。#1241 已将历史名称声明移到 Commands 合同，Core 通用解析 |
| 摘要 / compaction | 当前 `plugin:compaction@release` 有 v0 3 条、v2 25 条、8 个 head；完整当前父链仍引用全部 3 条 v0 | 旧 reader 仍必需。迁入记录保留旧行出处，不能补造 v2 所需的模型输入、调用或省略范围 |
| 未选摘要状态 / 原 owner | `plugin:compaction` 另有 v0 3 条、v2 1 条、2 个 head；旧表 `session_compactions` 5 行 | 与当前 owner 分开；保留数据不等于当前入口，不能据此自动删除或转换 |
| EventMail / eventmail | 当前选择指向 `eventmail@release`，数据库 user_version=3 | 对这一目标无需 v1/v2 转换；删除升级器仍需声明旧 workspace 的截止与升级入口 |
| Markdown profile / markdown_memory | `PENDING.md` 0 字节，snapshot 与 retired 文件不存在；旧 pending 迁移 receipt 0 条 | 本目标没有待转换 PENDING 内容；空文件不能证明迁移已执行，不伪造回执 |
| Models / models | 9 个连接：openai-compatible 6、gemini 1、opencode-go 2；旧 driver ID 0 个，但 `catalog_provider_id` 9 个非空 | 旧 driver ID 本目标无需再转换；独立 provider hint 仍有消费者，需核对 config 合并冲突后转换 |
| 插件配置 / 各插件配置 owner | 当前 65 个所选 data_dir 内没有 `config.local.toml`；其他目录有 9 个同名旧文件 | 先按 selection 限定当前目标；历史目录不自动清理。`check_config_format` 是显式拒绝旧格式，不是读取 fallback |
| Wake 规则 / wake | `wake@release/legacy-rules` 规则正文 7,474 字节，receipt 的摘要与正文相符 | 当前 runtime 实际读取的目标私有规则归档，名字含 legacy 不代表死代码；删除会改变 Prompt |
| Yoyo / migration owner | 63 条成功迁移记录；当前 17 个脚本中 14 个已有同 ID 回执；Gateway 两步与合同重启一步未记录 | 不能将新栈三个待执行脚本当作完成的历史。回执还需与 artifact/hash、目标状态核对，不能仅按 ID 批量退役 |
| 主动岛 / 离线 handoff 与目标插件 | 旧 proactive 与 wake 库保留；当前 114 个 source item 与 1 份规则全部匹配 lineage 源摘要；114 个 Feed 回执的身份和源摘要匹配，action 全是 `cutover_superseded`；退休回执的 14 项当前 block 精确匹配，全部备份文件摘要通过 | 已有显式切换替代记录，并非等待复制 114 个旧 item。本次未发现新的未交接源事实或未退休 block；完整 adapter/目标 digest 验收仍需核对。不重新交接，不删除旧库 |

两个源码数字也需修正：`session/log.py::_session_schemas` 是一个当前结构加两条旧
lineage 各五种形状，共 11 个规范化结构，不是三乘五。`core.commands` 写死映射在
#1241 已删除，但磁盘上的历史名字仍真实存在。

`bus/queue.py` 从持久 handoff 重建入站消息的正常恢复路径必须保留。只删除已明确
截止的旧 metadata 投影或 dedupe 表示，不删除当前崩溃恢复协议。

## 推荐推进顺序

```text
┌──────────────────────────────┐
│ 边界①合并、固定升级来源      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ owner 盘点 → 隔离副本转换验证 │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 维护窗口：备份 → 显式转换    │
│ 失败停在可恢复步骤，不记成功  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 核对事实与恢复 → 写截止标记  │
│ 当前运行只读当前表示          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 删除对应旧 reader → Ledger   │
└──────────────────────────────┘
```

推荐一次明确的截止发布，不把逐个数据转换插进当前长 PR 栈。本轮提交盘点与过时文档修正。
后续只转换真实存在的旧表示；离线升级器由数据 owner 提供，Core 只处理通用版本检查。
业务归档可以采用当前、明确的审计表示，保留原事实；不要求它伪装成一次新原生生成。

转换前逐项固定输入和 artifact、实际写入对象、目标结构、恢复点及完整性比较：

- Session：固定当前结构，核对全部原行、NULL/default 含义、旧列、索引、触发器和外键。
- 选择：所有当前输入升级后才提交新选择；保留原 selection 和安装来源，不重建虚假命令声明。
- 绑定：先决定不可变旧 descriptor 的读取义务。若采用新绑定，需要明确所有引用和历史身份的
  保留协议；不能为了改名重写历史 Message。
- 摘要与工具结果：保留真实来源和外部效果不确定性；需要改变表示时先确认可观察语义。
- Models：将旧 hint 转到 driver config 前核对值冲突、身份与调用账目，原凭据不出现在报告。
- 插件独立状态：由实际选择的插件 owner 转换，未选目录与备份不在默认写入范围。

每步先建立相应 SQLite 原生备份或一致目录恢复点；转换失败保留错误和实际已提交结果。
验收比较受保护记录的完整内容、身份、顺序及引用，不以行数相同代替事实相同。
当前没有授权减少旧表、历史目录、receipt、Message 或备份。

## 截止机制的设计边界

最低版本标记只是拒绝旧 workspace 的入口，不能代替各 owner 的转换与检查。只有所有必需
步骤验证成功才能写标记；缺失、损坏或低版本都明确失败，并指向固定的桥接版本与命令。
新 workspace 由当前 owner 初始化；导入旧备份或安装旧插件仍须在各自边界检查版本。
桥接版本、截止版本与命令尚未确定，本轮不新增 startup gate，也不扩大正式数据写权限。

不推荐用 `legacy` / “旧”关键词扫描作为唯一静态门：它会误伤 Wake 规则归档和真实审计
记录，也找不到没有这些词的兼容分支。若新增兼容规则，建议在持久格式入口显式登记
旧形状、owner、桥接入口、截止版本及删除条件，静态检查验证登记完整和过期项。
R7/R10 仍按 Issue 1179 的实际验收定义实现，不能用关键词命中数代替。
