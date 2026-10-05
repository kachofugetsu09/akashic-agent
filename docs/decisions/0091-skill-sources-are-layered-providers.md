# 0091 · Skill 来源是分层 Provider，本地目录不必打包成插件

- 状态：accepted（来源分层）；资源归档与 mtime 缓存由 [0092](0092-plugin-runtime-uses-installed-files.md) 取代
- 日期：2026-10-06
- 关联条款：PLG-009、PLG-014、PLG-016、CTX-004、WKS-001
- 变更范围：显式调整 [PLG-009](../projectneed.md) 对 Skill 来源的表述，并重议 [持久化状态地图](../design/persistence-state-map.md) 的 G-004A

## 背景与决定

现状只有一条 Skill 来源：插件制品通过 `INSTALLED_ASSETS.register` 声明的固定资产目录。
`SkillCatalogParser.parse` 硬编码 `source="plugin"`，workspace `skills/` 下的真实目录不再
被加载（`PluginSkillLinker` 删除后，软链接投影与手工目录一起失效）。

维护者要求对齐 dsh 的正交 Provider 结构：Skill 不必打包成插件，放到本地目录即可被发现和
加载。本次据此把来源拆成三个并列 Provider，按就近优先合并：

```text
优先级（高 → 低）        目录                         source
─────────────────────────────────────────────────────────
1  workspace            <workspace>/skills            workspace
2  user                 ~/.akashic/skills             user
3  plugin               InstalledAsset.root_dir       plugin
```

- 高层同名 Skill 静默覆盖低层；**同一来源内重名仍然报错**，插件层保留原有的
  `插件 Skill 名称重复` 失败语义。
- `SkillRecord.source` 从 `Literal["plugin"]` 扩展为
  `Literal["plugin", "workspace", "user"]`，`load_skill` 参数、返回协议、
  `SKILL_INSPECTION` 只读投影字段和 frontmatter/`requires` 门禁都不变。
- 三种来源共用一条归档链：`save_skill` 仍把目录写进不可变的 `PluginArchive`，
  相对路径基准仍是归档后的内容寻址目录。
- workspace 目录里指向插件归档的软链接不再被视为独立 Skill，避免同一份资产被登记两次。

## 取舍与重议条件

**不采用**“所有 Skill 必须打包成插件”：它把一次文案改动变成一次插件发布，也让 ToB 场景下
的本地定制必须走完整制品流程；dsh 的正交做法已经证明本地层可以只承担“发现+读取”。

**不采用**在 `PluginSkillLinker` 之外再建一套 workspace 软链接投影：投影会带来第二份
可独立漂移的状态，与“一个事实一个 owner”冲突。文件系统即来源，读取时直接解析。

**不采用**让本地目录绕过 `PluginArchive`：归档是 `base_directory` 与快照一致性的唯一依据，
绕过会让恢复读不到原样材料。

代价是本地目录成为一条新的能力入口，需要 workspace 写权限即可生效；容器化部署中
`~/.akashic/skills` 落在镜像 HOME 内，宿主注入需要显式挂载。因此本次明确：

- 工作区目录按 `local_signature`（根目录与子目录 mtime）在 generation 内重新解析，
  新增/删除/修改的 Skill 在下一次读取时可见，不需要重装插件。
- 本地来源不声明 `requires` 之外的权限，不改变工具可见性，也不是长期记忆或用户事实证据。

什么前提下值得重议：如果本地目录被证明成为绕过插件评审的后门，就收回到“本地目录只做
只读覆盖，不参与默认投递”；如果 dsh 之外的实现证明分层优先级会制造歧义，再讨论是否
改为显式 `enabled` 声明。

## 恢复与验收

新增来源只增加读取路径，不迁移、不改写任何既有数据；插件资产与 `PluginArchive` 的
内容寻址都不变。回退源码即可回到“只认插件资产”，本地目录只是不再被发现，不会被删除。

验收使用真实目录与真实解析：三来源各放一个同名 Skill 校验覆盖顺序，去掉 workspace
校验退回 user，全空校验退回 plugin，同一来源重名校验仍报错，workspace 内的插件投影
软链接不重复登记，签名对新目录可见。之后在隔离实例中做一次端到端验收，确认
`load_skill` 能读到本地来源的正文。

## 影响

`docs/projectneed.md` 的 PLG-009 与 `design/persistence-state-map.md` 的 G-004A 记录的
是“手工目录待迁移并收窄”的旧方向；本次由维护者授权改为“本地目录是并列来源”，
两处表述需要在合并前同步更新。
