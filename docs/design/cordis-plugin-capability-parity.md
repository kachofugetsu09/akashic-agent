# Cordis 插件迁移能力等价验收设计

- 状态：accepted / Core host implemented / Citation-Meme pilot ready for review
- 核对日期：2026-08-14
- 文档基线：`akashic-agent@d1b1295b8490ffe899f27476bf97ae7b261ef76e`
- 当前运行基线：`akashic-agent@07068e2bfb2dac0173298ed0c60a7f5c466ad745`
- 对照实现：`deepseek-harness@47f943859bef60e4160492346772ded9b24f765a`
- 理论来源：`cordiverse/paper@948a07b369c62adb3b12e102458be5c18dfb69b9`
- 关联条款：OBJ-003、GOV-001～GOV-005、PLG-001～PLG-013、TST-001～TST-007、STA-001～STA-003、CAP-001～CAP-002、ERR-001
- 关联设计：[插件递归自验证运行时设计](recursive-plugin-self-validation.md)、[持久化状态地图](persistence-state-map.md)、[移动端与跨仓库语义 Gate](mobile-cross-repository-semantic-gate.md)

## 1. 结论与范围

Akashic 可以采用 Cordis 的 service、typed event、effect、fiber 和配置组合机制，减少固定插件基类与生命周期方法。迁移不能把“插件能够加载”当成能力等价。每个当前插件必须在旧 Akashic 与 Cordis 候选上运行同一组输入，并比较目录、Prompt、工具、事件、持久状态、外部调用、清理结果和用户可见输出。

本设计保留现有插件发布语义：候选隔离、父 Turn 授权、stable/latest 内部双快照、generation lease、领域 oracle、自动提交或丢弃。Cordis 负责挂载、依赖等待和可逆注册，不取得插件代码晋升、持久数据恢复或外部效果回滚的所有权。

本设计同时作为迁移前的能力基线与验收方法。维护者已批准组合内核、实验插件和 Citation/Meme 首组迁移，边界见[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)与[任务合同](plugin-composition-kernel-task-contract.md)。当前交付仍不批准以下动作：

- 不修改当前正式 runtime、workspace、plugin-data 或插件 manifest。
- 不把当前安装缓存当成可编辑源码。
- 不把 Citation/Meme 试点自动扩大到其他现有插件。
- 不批准修复当前 Wake ACK 故障或改变缺失依赖的错误语义。
- 不批准向真实渠道、GitHub、浏览器账号或外部 API 发送测试效果。

### 1.1 实施收敛：选择组合语义，不复制整套工具链

第一阶段转译 Cordis 的 Context、Service、Inject、Fiber 与 Effect，并让 Root/Fiber 直接拥有 scope 生命周期；它吸收 DeepSeek Harness 对重入卸载、owner 先登记、UNLOADING 禁止新 Effect、child 先归属后发布、epoch 防陈旧激活和 observer 隔离的加固。它不移植独立 service isolation scope、Loader、Include、HMR、Timer、Logger、Schemastery 或 CosmoKit：Akashic 已有 artifact、安装、generation、热重载和晋升 owner；Timer 与 Logger 可以像 Job 一样成为普通 Service；配置继续使用 Python 类型和现有 Pydantic 边界。

实现后的 publication seam 在候选 lease 排空后由 Core 封存拓扑、写集和外部效果回执；晋升前再次复核。Service 实例本身仍可承载运行时可变状态，不能为了消除 TOCTOU 而把 Cordis 的动态服务错误冻结成序列化值。任何被 `ExternalEffectGate` 拒绝的尝试即使被插件捕获，也会令候选不再 ready。

因此本文后续出现的“Cordis 候选”表示采用上述组合语义的 Python 候选拓扑，不表示逐包、逐 API 或 TypeScript ABI 完整兼容。能力等价仍以可观察行为为准。

## 2. 本设计与现有自验证设计的关系

[插件递归自验证运行时设计](recursive-plugin-self-validation.md)回答“同一套 Akashic runtime 怎样让父 Turn 验证候选插件并决定是否发布”。本设计回答另一个问题：“把整个插件宿主替换为 Cordis 后，怎样证明当前全部能力没有发生未批准变化”。

```text
┌──────────────────────────┐
│ 现有插件发布合同          │
│ candidate / stable       │
│ parent Turn / child Turn │
│ generation lease         │
└─────────────┬────────────┘
              │ 保留
              ▼
┌──────────────────────────┐
│ Cordis 挂载与运行机制     │
│ service / event / effect │
│ fiber / disposer / HMR   │
└─────────────┬────────────┘
              │ 产生旧/新两份证据
              ▼
┌──────────────────────────┐
│ 能力等价 Gate             │
│ catalog / turn / state   │
│ effect / lifecycle / UI  │
└──────────────────────────┘
```

Cordis 的动态 package 工具不是发布协议。它只在当前进程内定义和运行临时 package，运行后的工具或 Prompt 贡献从后续模型请求开始可见；重启后消失，不持久安装，也不会自动晋升。`effect` 能回收注册，但不会撤销文件、数据库、消息或远程调用。

## 3. 当前真实运行基线

### 3.1 运行身份

2026-08-14 的只读检查得到：

| 对象 | 当前身份 |
|---|---|
| document source | commit `d1b1295b8490ffe899f27476bf97ae7b261ef76e`，tree `866433e22a392556a9734d2a612cd97a0b1a4f3f` |
| live Akashic source | commit `07068e2bfb2dac0173298ed0c60a7f5c466ad745`，tree `fa54b8c664c0ab7a544bdca60c36e54f0769bfeb` |
| DeepSeek Harness source | commit `47f943859bef60e4160492346772ded9b24f765a`，tree `f904efab9ef435201d6ba4da88a34d6366568272` |
| Cordis paper source | commit `948a07b369c62adb3b12e102458be5c18dfb69b9`，tree `9843926bd597bf184536fe9b2961bcc77f245bb6` |
| manifest | SHA-256 `5a1a5a42472ea3e3272936883a3aab4d044cfbc69e6a91b7b8c8a8d3fdb29d90` |
| stable snapshot | `930ac2d927380dee` |
| latest snapshot | `930ac2d927380dee` |
| candidate | 无 |
| skill projection | 10 个已登记链接，pending 为空 |

本次检查的 live process、主配置、workspace 和 plugin home 是一组本机运行实例。验收工具必须从运行进程读取这些选择，不能从当前 shell 的工作目录、默认 HOME 或另一个 Git checkout 猜测。

`plugin-status` 当前只返回 stable/latest snapshot 和 candidate 状态，没有逐插件返回 active generation、artifact digest、config revision、data owner 和 lease。迁移开始前需要补充只读能力基线导出；仅有一个 snapshot ID 不能证明其中每个插件的来源。

### 3.2 启用、加载和实际贡献不是同一集合

当前 manifest 有 13 个启用 plugin 条目和一个启用 package。`wake-proactive` package 展开为三个运行时插件，因此启动日志报告 16 个 loaded plugin：

```text
citation
emotion
huayue-skills
meme
observe
plugin_undo
proactive_feedback
setup_helper
shell_restore
shell_safety
status_commands
akasha
default_memory
wake_drift_flow
wake_proactive
wake_proactive_flow
```

当前 memory engine 是 Akasha。`default_memory` 虽然已加载，但 `is_active()` 为 false，不应贡献 before-turn 模块、Skill、Dashboard 或写入。因此当前基线是 16 个 loaded implementation、15 个有效贡献者。

以下对象已安装但禁用：

```text
computer-use-linux@github
daynight_gate@github
default-proactive package
```

禁用对象的当前等价含义是没有 tool、Prompt、Skill、MCP、UI、job、listener、进程或写入。它们另需在一次性环境做 cold-enable 兼容测试，但不进入当前完整运行时的输出比较。

历史 `plugin-data` 目录不证明插件仍然安装或启用。当前目录、当前 manifest、active generation 和实际 runtime contribution 必须分别列出。

### 3.3 当前外部 artifact 身份

表中 artifact digest 使用排序 tar 计算，统一 mtime、owner 和 group，排除 `.git`、`__pycache__` 与 `*.pyc`，保留其余 tracked 和 untracked 文件、mode 与符号链接。它描述本机实际安装内容，不代替 canonical source commit。

| 插件 | 状态 | 版本 | source commit | artifact SHA-256 |
|---|---|---:|---|---|
| citation | active | 1.0.0 | `bf56c30df144f36e640174c5c8fe0723fc20fa02` | `2f43cf73324a069ee1f1db080eb971e8591dd46bf6cf68ab88c1caa7b2b3cfb2` |
| emotion | active | 1.1.0 | `ff47f6e9d83a090babb220a2fb82299d94322538` | `02930f6bcceaa78461d9bdbb0c72af5b193db7386c108969e7b6b3c96b5c5f72` |
| huayue-skills | active | 1.0.2 | `c3a8baf297da0ba83bfd91ca7f6a643e85b244de` | `1cced531b485b3dc3e903afdcef96dd261174ce8404b3423deb466277ef2bfc3` |
| meme | active | 1.0.0 | `db97d3390404070b1bb6664947e85f368d551350` | `289f4395c1a783d1c166a6204b7c87030981900deadc8d321b63d9b58e96253d` |
| observe | active | 1.2.0 | `4d85b9dc64ef0d8d96c5a635586ca17dd94b59cd` | `688c8637f903eff313af9e35b1e1e695224a0d73ddb3fe27f9ff71c8e790383a` |
| plugin_undo | active | 1.0.0 | `2b2d90e4f3dbf211a127ed3fdde8194f812c642e` | `70f43570922f3dd7f7218cc38e9167a12e23786cd872195821f936bbfc8e31ca` |
| proactive_feedback | active | 1.1.0 | `b8a0e0eca4d16614bb1bd663616ee1c9ba1297ed` | `e4003df42e36fa7a222bf5d25e8b8fdc651059cca60a69e6ac1adeb14488ea31` |
| setup_helper | active | 1.0.0 | `c3dcd813473f60f77cb384b75461a92611a1bb28` | `d6861b304d9451869358e0230e8d4c2efde5535e084cfcd786bef68b360ad157` |
| shell_restore | active | 1.0.2 | `091fdca06df763354ba8c2693a16dffc477604d9` | `d3b942b67f0771f5b5ce8fb52efdde4274dc212168dc5a0f37a656cc9c72cb54` |
| shell_safety | active | 1.0.0 | `84ddac1573cc34bb49df1243b3beda579dcea5ef` | `13a0bef7b094cfbe72363a045bd6b14c3bfe8e53126199cbe7b677bb688f6b51` |
| status_commands | active | 1.1.0 | `cf61f99284be2b09636b6633f7832fa61e20926b` | `69aec2a122045cd733176bd60868736dc22fd4153d40da3cae2ee5e05463d421` |
| computer-use-linux | disabled | 1.1.1 | `31325bb0552455fd4cc5d29efaf1477a7b10bc0` | `341813f1ff0db0246a406edcde1d0cb471c2f272fa26f2d7adffe16229aa7e10` |
| daynight_gate | disabled | 1.0.0 | `2b95c6aed10d7ebd671a50ad215a61934f10e363` | `f8ea93ae543e59f1ae1855d406a6298ed5e643c3e7f36ef37596f4f158f88dac` |

`meme` 安装目录存在未跟踪的 Dashboard 文件；`shell_restore` 从 `.pointers.json` 指向的 immutable artifact 运行。多个插件的本地开发 checkout 与 installed cache commit 不一致。因此旧侧基线必须取实际 stable artifact，迁移作者再从 artifact identity 追溯 canonical source；不能直接使用手边最新源码重建“旧实现”。

### 3.4 当前 Skill 投影

当前 runtime registry 拥有 10 个插件投影：

| owner | 类型 | 名称 |
|---|---|---|
| huayue-skills | Skill | `anthropic-diagram`、`codex-usage`、`gh-cli`、`image-generation-nano`、`opencli`、`paper-explainer`、`playwright-browser`、`yt-dlp-downloader` |
| meme | Skill | `meme-manage` |
| emotion | Drift skill | `feedback-preference-context` |

Skill 等价既包括目录和 frontmatter，也包括正文、脚本、资源、命令意图、可见条件和清理行为。只比较名称会漏掉绝大部分能力。

## 4. 已确认事实、冲突和未知边界

### 4.1 已确认事实

- Cordis 的插件通过 service key 获取能力，通过 typed event 通信，通过 effect/fiber 回收注册和后台工作。
- 配置行顺序不承担 service 依赖语义；硬依赖由 inject 表达，可选能力由使用点查询。需要确定执行先后的处理使用拥有明确合同的 serial event 或领域 service；当前迁移不引入通用 waterfall。
- Akashic 的 phase slot/requires/produces、event priority、tool hook 顺序和 package 展开共同决定当前行为，不能只翻译插件基类方法名。
- 当前 tool hook 依 plugin ID 排序后顺序执行，后一个 hook 看到前一个 hook 改写后的参数。`shell_restore` 与 `shell_safety` 必须作为组合场景验证。
- 当前 citation/meme 的关键顺序由显式 slot 依赖和 after-reasoning event placement 共同形成，必须作为一个组合迁移。
- Cordis disposer 只能撤销它拥有的注册和资源；插件自行执行的数据库、文件、网络和渠道操作需要领域提交、隔离或补偿协议。

### 4.2 当前文档与实现冲突

1. `persistence-state-map.md` 的 `memes/manifest.json` 条目写着未找到生产 reader/writer，但当前已安装 `meme` 插件会读取该 manifest 并按 mtime 重载。迁移不能依据旧条目把它判成废弃状态。
2. PLG-008 要求依赖缺失和拓扑环在发布前 fail-loud。当前 `agent.lifecycle.phase` 对部分缺失依赖记录 warning 并禁用模块。迁移前要确认这是待修缺口还是有意兼容语义，不能让 Cordis 的等待行为无期限掩盖配置错误。
3. 当前 Wake runtime 持续存在 `feed@github:subscriptions` ACK source 缺失。它是已观察故障，不是自动要求候选复现的正确能力。

### 4.3 尚未确认

- 是否先修复 Wake ACK source，再冻结迁移基线；或把它登记为一个有明确 owner 的批准差异。
- 缺失 phase dependency 的最终语义是发布失败，还是允许某类显式 optional dependency。
- 真实模型 A/B 的场景数量、通过阈值和允许波动范围。
- Cordis 化是否只替换插件宿主，还是同时重画被动 ReAct、proactive、scheduler 与 UI 边界。本设计默认只替换插件宿主和所需能力 seam。
- 迁移期间是否允许同时改变插件的用户可见行为。本设计默认 `semantic_delta: none`，每个例外单独批准。

## 5. 能力等价的定义

能力等价不是内部类、文件或 SQLite bytes 完全相同。它表示在固定输入、固定插件组合和固定外部环境下，旧实现与候选具有相同的可观察结果：

```text
┌──────────────── 能力基线 ────────────────┐
│ runtime / config / manifest / artifact  │
│ workspace snapshot / clock / random     │
│ model / external adapter / scenario     │
└────────────────┬────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────┐     ┌──────────────┐
│ Akashic old  │     │ Cordis new   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │ 规范化差分回执      │
       └─────────┬──────────┘
                 ▼
       独立 oracle / mutant / verdict
```

### 5.1 精确比较

以下对象原则上精确相同：

- enabled、inactive、disabled 集合与 package 展开结果。
- Tool 名称、参数 schema、risk、可见 scope 和实际执行 pipeline。
- Skill、Drift skill、MCP 声明、正文和资源摘要。
- Prompt section 内容、位置、排序和可见条件。
- phase、event、job、source、channel 和 listener 的依赖与执行顺序。
- Session event、错误分类、取消终态、queue drop 和 delivery envelope。
- 用户可见文本中的协议清理结果、media、引用和 metadata。
- HMR/unload 后剩余的 service、tool、listener、timer、task、process 和 UI registration。

### 5.2 逻辑比较

SQLite page、WAL、rowid 分配和物理文件布局可以不同，但 schema identity、排序后的完整逻辑 rows、允许的状态转换和 write set 必须一致。文件使用内容、mode、目标路径和原子发布结果比较，不要求临时文件名相同。

### 5.3 允许归一化的对象

时间戳、UUID、PID、端口和随机选择只有在场景开始前登记到 normalization allowlist 才能忽略或映射。Prompt 顺序、Tool schema、事件顺序、错误类型、INSERT/UPDATE/DELETE、文件目标和外部调用不得由通用 normalizer 删除。

### 5.4 批准差异

已知故障修复或产品语义变化通过单独的 `declared_delta` 进入 verdict。每条差异要引用受影响条款、旧结果、新结果、数据与外部影响、迁移和新 oracle。没有批准的差异仍按失败处理。

## 6. 差分回执

每个 scenario 在旧、新两侧各产生一组同结构结果：

```text
parity-run/<run-id>/
├── identity.json
├── catalog.json
├── turn.jsonl
├── state.json
├── effects.jsonl
├── lifecycle.json
├── ui/
└── verdict.json
```

### 6.1 `identity.json`

至少固定：

- core repository、commit 和 tree。
- Cordis/DeepSeek Harness repository、commit 和 tree。
- plugin source repository、requested ref、resolved commit、artifact digest 和 manifest entry。
- runtime snapshot、plugin generation、config revision、data owner 和 lease identity。
- scenario catalog/profile/hash、模型 generation、clock/random fixture 和 adapter set。
- workspace、plugin home、config 和 HOME 的隔离 run identity；不记录 secret 值。

### 6.2 `catalog.json`

记录完整实际贡献，不从 manifest 推断：

- service 定义、provider 和 consumer。
- tools、schema、risk、render intent、scope。
- Skill、Drift skill、MCP、脚本和资源摘要。
- Prompt sections、context providers 和排序。
- events、dispatch mode、listeners 和注册顺序。
- lifecycle、phase、slot、requires、produces、jobs、sources、channels 和 managed services。
- desktop/mobile UI slots、assets、RPC schema、navigation 和 version。

### 6.3 `turn.jsonl`

记录 admission identity、输入 Message、冻结 snapshot、实际模型 request、Tool call/result、phase/event 顺序、Session event、最终输出和错误。任何模型可见输入必须能从这份运行证据或持久 session event 重建。

### 6.4 `state.json`

记录每个持久对象的 before/after logical snapshot 与尝试过的 write set：

- SQLite schema、完整规范化 rows、INSERT/UPDATE/DELETE 和事务结果。
- 文件创建、内容替换、rename、move、chmod 和 delete 尝试。
- 即使事务回滚，也保留违规写入尝试。
- untouched 对象记录 byte digest，证明没有被顺带改变。

### 6.5 `effects.jsonl`

记录 LLM、embedding、MCP、HTTP、GitHub、browser、shell、channel、child process、socket 和 service switch 的调用 envelope、幂等键、终态和补偿结果。默认 adapter 只记录并返回确定性 fixture，不触发真实外部效果。

### 6.6 `lifecycle.json`

记录 prepare、activate、admission、retire、lease drain、terminate、dispose、HMR 和 crash recovery。结束时重新枚举所有注册、task、process、port、socket 和临时目录；没有 owner 的残留即失败。

### 6.7 `ui/`

记录 plugin roster、RPC DTO、desktop/mobile 浏览器 snapshot、交互结果和 render failure。UI package load 成功不等于组件成功渲染；render error 必须进入 verdict。

## 7. 持久状态与权限

本表只定义验收怎样观察状态，不授予新的写入或删除权限。

| 对象 | 正常增加或更新 | 逻辑失效 | 物理减少条件 | 验收 owner 与恢复证据 |
|---|---|---|---|---|
| `sessions.db/messages`、turn 与 citation metadata | 正常 Turn 按既有事务追加；turn 按状态机更新 | 既有 compaction/interaction 规则 | 只允许用户显式撤销或删除 | SessionStore；完整 rows、write set、interaction snapshot |
| Akasha sidecar 与 embedding | completed Turn 后按固定输入增加或重建 | source generation fence | 只由明确 rebuild/interaction 撤销流程替换 | Akasha owner；source digest、logical state、recall result |
| `memory2.db`、Markdown memory | 既有引擎按事件增加、强化或原子重写 | supersede、consume、terminal | 只按既有 forget/optimizer 协议 | memory owner；before/after、备份、idempotency receipt |
| `plugin-data/` | active plugin 按自己的 schema 增加或更新 | 插件自行定义 | 普通卸载不得删除；永久删除需独立用户操作 | plugin repository；目录快照、schema、restore smoke |
| observe/emotion/proactive-feedback DB | Turn 或领域事件增加，状态机更新 | retention/terminal 仅按既有合同 | 没有批准协议时不得自动减少 | 各插件 owner；完整 rows、queue/event receipt |
| proactive/Wake/Drift DB | tick、run、reservoir、ack、cursor 和 journal 增加或更新 | consume、ack、terminal | 只按领域事务与已批准 retention | runtime owner；dedupe、cursor、pending ack、restart smoke |
| schedules、quota、plugin manifest/pointers | 既有管理动作增加或更新当前值 | disabled、superseded pointer | 只按 cancel/uninstall/GC 合同 | 对应 owner；journal、manifest、pointer、recovery |
| meme、Skill 和 shell 管理文件 | 用户或插件工具创建、改写或移动目标 | manifest/category 状态 | 只按用户显式工具意图 | 文件 owner；复制目录、content hash、restore path |
| 外部消息和远程 API | 只在领域提交点发生 | failed/unknown 是终态，不等于回滚 | 已提交效果通常不能物理撤销 | channel/provider；recording sink、delivery receipt、补偿说明 |
| candidate 与差分产物 | 每次隔离 run 新建 | verdict terminal 后不再参与 admission | 仅由 run owner 在证据发布或保留期确认后清理 | parity runner；manifest、cleanup report、previous restore |

旧、新 runtime 只能写各自的一次性 workspace、plugin home、config 和 HOME。正式 workspace、正式 plugin-data、正式渠道和当前进程作为只读基线，不作为 replay 目录。

## 8. 当前插件的领域 oracle

### 8.1 必须共同迁移的组合

| 组合 | 必须保持的顺序与结果 | 主要 oracle |
|---|---|---|
| citation + meme | `citation.prompt → meme.prompt`；citation after-reasoning 先处理 cited IDs，保留 meme tag；meme 消费 tag 后 cleanup 不泄漏协议 | Prompt snapshot、cited IDs、最终文本、media path/count、code block 排除 |
| shell_restore + shell_safety | restore 先改写受支持的 `rm`，safety 再判断实际 args；拒绝不能执行子进程 | original/executed argv、allow/deny、文件 move、restore tree、无额外 process |
| proactive_feedback + emotion | committed Turn 产生唯一 feedback；emotion 消费同一 event；DriftFinished 触发 pending/context merge | dedupe key、score、VAD/effect rows、pending/context 文件、cursor |
| wake_proactive + wake_proactive_flow + wake_drift_flow | prompt emotion → start → ingest → content → drift → schedule | reservoir、quarantine、hazard、drift、ack、next wakeup、send envelope |

### 8.2 单插件与能力组

| 插件或能力组 | 当前能力 | 必须观察的结果 |
|---|---|---|
| Akasha | memory engine、recall/context、after-reasoning feedback、Inspector、rebuild | TurnCommitted 后的 sidecar logical state、recall lanes、citation IDs、source fence、failure recovery |
| default_memory inactive | 已安装但不属于 active engine | active graph 中没有 handler、Skill、UI、trace 或写入；cold-enable 环境另测完整能力 |
| observe | Turn/Proactive/Retrieval/Memory 事件、global error、writer/retention、desktop/mobile UI | event identity、queue/drop count、DB rows、watermark、retention rollback、hook disposal |
| plugin_undo | `/undo`、SessionDB 与 memory rollback | dry-run、目标选择、session delete、memory cleanup、部分失败的显式错误；只对复制 DB 运行 |
| setup_helper | `/chatid`、`/myid` short-circuit | 不启动模型、不写持久状态、返回当前渠道身份和配置指引 |
| status_commands | memory/kvcache command 与 mobile projection | 只读查询、不创建缺失 session、command 与 mobile DTO 语义一致、DB bytes 不变 |
| huayue-skills | 8 个外部 Skill | frontmatter/body/resource digest、触发条件、command intent、sink receipt、未授权外部效果为零 |
| meme-manage | meme catalog 管理 Skill | manifest、图片目录、原子写入、失败恢复和 Dashboard 更新 |
| emotion Drift skill | feedback preference context | pending/context 增改、LLM fixture、cursor/journal、失败不丢 pending |
| computer-use-linux disabled | Skill + MCP 的冷能力 | 当前装配零贡献；cold-enable 时 MCP readiness、process/UI cleanup |
| daynight_gate disabled | proactive gate | 当前装配零贡献；cold-enable 时固定 clock/random 下的 gate result |
| default-proactive disabled | 另一套 proactive package | 当前装配零贡献；若未来替换 Wake，必须作为批准语义变化处理 |

### 8.3 huayue-skills 外部效果

这些 Skill 不能在自动 parity 中直接操作真实系统：

- `gh-cli` 可能创建 Issue、PR、push 或 merge。
- `opencli`、`playwright-browser` 可能启动持久浏览器、使用 cookie 或操作登录网站。
- `image-generation-nano` 调用外部图像 API 并写图片。
- `yt-dlp-downloader` 下载网络内容并写文件，可能安装依赖。
- `codex-usage` 可能安装或启动本地服务。
- `anthropic-diagram` 写图表与图片。
- `paper-explainer` 读取网络论文。

确定性 Gate 记录 Skill 实际发出的命令、请求和文件意图。真实领域 smoke 必须在前述 Gate 全绿后单独授权，并使用测试账号、测试仓库或受控 endpoint。

## 9. Cordis 映射原则

迁移按能力 owner 映射，不按旧基类方法逐项复制：

| Akashic 语义 | Cordis/DeepSeek Harness 目标 |
|---|---|
| Tool registry 与 schema | `ctx.tools` service + scoped registration |
| Prompt module | system-prompt section/provider；排序由领域 owner 明确拥有 |
| before/after interception | typed event；serial/parallel/emit mode 写入公开 contract |
| Skill/Drift skill | `ctx.skills` provider/catalog/invalidation + disposer |
| MCP/managed service | Service Definition / Provider / Consumer + readiness |
| job/background task | `ctx.jobs` 或 plugin fiber-owned task |
| generation/snapshot | Akashic rollout owner 保留；Cordis fiber 作为一代的资源 scope |
| dashboard/mobile UI | versioned client slot + RPC DTO + render receipt |
| plugin-data | 领域 repository service；不交给通用 effect 回滚 |
| proactive/scheduler | 独立 Message/Turn producer，复用 Agent seam，不复制被动链路的固定 Prompt/Tool 组合 |

DeepSeek Harness 的 `tools/pre-execute` 当前只表达 allow/deny/ask，不拥有输入改写。`shell_restore` 应放入 shell service provider/consumer wrapper，记录原始 args 与实际执行 args；不能为了复用 guard API 丢失改写能力。

依赖不能使用任意数字 priority 作为主要公共协议：

- service existence 使用 inject/coeffect。
- 需要有序的领域流程使用 serial 或拥有显式 step graph 的 service；只有 `Bail` 能提前终止 serial。
- 可以同时执行且没有顺序依赖的观察者使用 parallel。
- 纯同步广播使用 emit；异步 listener 注册到 emit 时 fail-loud。
- listener 只使用 generation 内稳定注册顺序；不引入 priority 或 listener dependency DAG。
- 同步并发通过受限 ExecutorService 执行纯任务，不在线程中暴露 Context/Fiber。
- 相互排斥的 provider 使用唯一 service key、显式 config selection 和 load-time conflict check。Akasha 与 default memory 由同一个 memory service 选择，不让两个 provider 同时偷偷生效。

## 10. 模型行为验证

模型不能作为自己的验收器。验证分成确定性与真实模型两层。

### 10.1 确定性层

两侧使用相同的 scripted model、clock、random、session fixture、external response 和 Tool result。旧、新运行时都执行真实插件和真实下游代码，只替换 LLM、网络、时间等昂贵或不确定边界。

该层精确比较：

- 完整 Prompt 与 Tool view。
- model request、Tool call/result 和循环次数。
- phase/event/session log。
- 最终文本、media、metadata 和错误。
- state write set 与 external effect intent。

### 10.2 真实模型层

两侧固定同一 provider/model generation、system context 和场景输入，成对运行。独立 oracle 重新读取文件、数据库、渠道 sink 和 Tool trace，不用回答中的关键词判断成功。

真实模型文本允许表达不同，但必须满足同一领域 rubric、禁止副作用和任务结果。场景数量、重复次数和通过阈值由维护者另行确认；在确认前，单次成功只能作为 smoke，不能形成统计等价结论。

## 11. Gate 层级

| Gate | 通过条件 |
|---|---|
| G0 · Baseline | live process、config、manifest、artifact、active generation、data owner 和 scenario identity 全部固定；任何来源不明即停止 |
| G1 · Catalog | active/inactive/disabled、tool、Skill、MCP、Prompt、event、job、UI、RPC 和 topology 精确一致 |
| G2 · Plugin | 每个插件的正常、空输入、错误、取消、reload 和 dispose 场景通过 |
| G3 · Composition | citation/meme、shell、feedback/emotion、Wake 等组合顺序和结果通过 |
| G4 · Runtime | 真实 Loader 启动完整装配，比较完整 request、session log、state 和用户输出 |
| G5 · Recovery | 每个持久提交、service switch、delivery 与 disposal 崩溃点都能恢复 previous 或显式 fail-loud |
| G6 · Model/UI | 真实模型领域 smoke、desktop/mobile render 与 RPC DTO 通过 |
| G7 · Shadow | 复制生产状态、禁止真实发送的长时运行没有写集、资源、ACK、queue 或行为漂移 |

所有 required Gate 通过后，才允许沿现有 parent Turn rollout 协议发布候选。任一 required Gate 失败时，stable 保持旧实现；不得修改 normalizer、oracle 或 fixture 来隐藏差异。

## 12. Mutant 与故障注入

每个 P0 oracle 至少杀死一个已知错误：

| 领域 | 最小 mutant |
|---|---|
| citation/meme | 交换 Prompt 顺序，或在 citation cleanup 中提前删掉 meme tag |
| shell | 跳过 safety，或让 safety 检查原始而不是改写后的 args |
| feedback/emotion | 同一 Turn 发出两次 feedback event |
| observe | 把另一个 message identity 写入当前 turn row |
| plugin_undo | 多删除一个 interaction，或 memory cleanup 失败后报告成功 |
| memory | 对 `sessions.db/messages` 注入 UPDATE/DELETE |
| Wake | 重复 delivery、提前清 pending ACK 或丢失 next wakeup |
| lifecycle | HMR 后残留 listener、task、tool、port 或 process |
| UI | client package load 成功但组件 render 抛错 |

Mutant 因依赖缺失、fixture 失败或测试超时而未运行，不计为 kill。失败必须指向对应 invariant 和可观察差异。

## 13. 外部副作用与回滚

### 13.1 默认隔离

- 每次 run 使用复制的 workspace、plugin home、config 和 HOME。
- Channel 使用 recording sink，不连接正式 QQ、Telegram、Mobile 或 Web delivery owner。
- LLM、embedding、MCP、HTTP、GitHub、browser 和 package manager 使用 recording/deny adapter。
- shell 运行在复制目录与受控 process owner 中。
- `plugin_undo` 只操作复制的 SessionDB 与 memory state。
- candidate service 使用独立 loopback endpoint 和 plugin-data 副本。

### 13.2 真实效果验证

写型插件只有满足以下一种条件才执行真实领域验收：

1. 插件提供与正式路径共用领域校验的事务或 dry-run；或
2. 使用隔离 workspace、测试账号和受控 endpoint；或
3. 用户明确授权具体效果，并声明幂等键、before/after oracle、费用与不可撤销边界。

stable/latest pointer 只恢复代码和运行时选择，不撤销已经发生的文件、数据库、消息和远程 API。恢复测试必须重新启动 previous、重读关键状态并检查外部 receipt；不能以 pointer 已恢复结束。

## 14. 分阶段迁移

### Phase 0：冻结能力基线

- 提供只读 runtime inventory，导出逐插件 generation、artifact、config、scope、lease 和 contribution。
- 生成当前 artifact、Skill、MCP、UI、schema 和 state owner 清单。
- 解决或明确 Wake ACK、缺失 phase dependency 与 meme 状态地图冲突。
- G0 未通过前不编写插件迁移代码。

### Phase 1：建立 Cordis 宿主与差分 runner

- 建立最薄的 Message → Turn → react → Message 主链。
- 提供 session、model、prompt、tool、file、shell、job、UI 等必要 service seam。
- 实现 catalog/state/effect/lifecycle receipt 与独立 comparer。
- 用一个故意错误的最小插件证明 Gate 能失败。

### Phase 2：迁移 citation + meme

该组合副作用较小，却能覆盖 Prompt 依赖、after-reasoning、metadata、最终文本、media、Skill 和 UI，是第一组完整试点。G0～G4 和对应 mutant 全部通过后，才继续扩大范围。

#### Phase 2 候选交付身份

2026-08-14 的首组候选固定为以下不可变组合。分支和 PR 只用于导航，验收身份以 commit 为准：

| 层 | repository / commit | Review 入口 |
|---|---|---|
| Core 组合栈顶 | `kachofugetsu09/akashic-agent@6d38dc2f99d2bdd41159935975ae4eb5109300c5` | PR #395～#403 |
| v3 静态合同 | `akashic-plugins/plugin-contracts@4dd69dd621e029e51e99aa428443fa3a4ec1f6cf` | PR #1 |
| Citation 双入口迁移 | `akashic-plugins/citation@7527251b88c7530b20685f38b5dbab6107fc1f5b` | PR #2 |
| Citation v2 去壳 | `akashic-plugins/citation@8ce75703fa9a426a0cf6b9dcf3fde0d744d31244` | PR #3，base 为 PR #2 |
| Meme 双入口迁移 | `akashic-plugins/meme@3ca7e1415d50c05a7f475595b26032f3db9faae2` | PR #2 |
| Meme v2 去壳 | `akashic-plugins/meme@00f899f70b25ea24e278b386332469f5f0351acf` | PR #3，base 为 PR #2 |

候选交付已经观察到以下结果：

- 双入口迁移组合用真实 `PluginManager` 比较旧/v3 Prompt 顺序、最终 reply、cited IDs、media 与 meme tag，9 个测试通过。
- 去壳组合从空 plugin home 加载 Citation 与 Meme，验证 `citation.protocol` 依赖、listener 顺序、Skill catalog、Dashboard binding 和最终回答，8 个测试通过。
- v3 静态合同分别验收迁移态双入口和去壳后的纯模块入口；Core 各层的 targeted、静态检查与 change-impact Gate 均绑定各自提交。
- 全部测试使用一次性 workspace 和 plugin home；没有写入正式 workspace、正式 plugin-data、manifest、渠道或远程 API。

这组证据足以把实现作为 Draft PR 栈交给维护者逐层评审，也证明删除 v2 壳层有可重放的迁移基线。它尚未形成第 6 节规定的完整 `identity/catalog/turn/state/effects/lifecycle` 回执，也没有实际运行故意交换顺序的 mutant；因此不能把当前候选标记为完整 G0～G4 能力等价，更不能据此切换正式 runtime。完整回执和 mutant 是 Phase 2 发布前剩余验收，不得通过放宽 oracle 省略。

### Phase 3：迁移只读命令、Skill catalog 与 shell 组合

- setup_helper、status_commands。
- huayue-skills 的静态 catalog 与 recording intent。
- shell_safety，再实现 shell provider wrapper 承载 shell_restore。

### Phase 4：迁移状态型事件插件

- observe。
- proactive_feedback + emotion + Drift skill。
- 每组先完成 schema、queue、write set、event identity、restart 和 dispose parity。

### Phase 5：迁移 memory provider

- 先保持 Akasha active、default_memory inactive 的当前选择。
- 分别完成 Akasha active suite 与 default memory cold-enable suite。
- 不在迁移中顺带改变 memory schema、recall 算法或数据保留。

### Phase 6：迁移 Wake 与主动执行

- 把 Wake 三个 runtime plugin 作为一个能力组。
- 固定 source/MCP/LLM/tool fixture，比较 reservoir、hazard、drift、ACK、schedule 和 outbound。
- 空输入、合法 skip、source failure、model failure、delivery unknown 和 restart 分开验收。

### Phase 7：完整 UI、shadow 与发布

- 对齐 desktop/mobile plugin roster、RPC、navigation、assets 和 render。
- 在复制生产状态、禁止真实发送的环境完成长时 shadow。
- 按现有 parent Turn candidate rollout 做最后一次 previous recovery，再批准正式切换。

## 15. 停止条件

出现任一条件都不能声明能力等价：

- active generation 不能绑定到精确 artifact、config、data owner 和 lease。
- manifest、startup log、skill links 与 runtime contribution 不一致。
- installed artifact 与 canonical source 的关系不明确。
- phase/event/hook 顺序、tool schema、Skill/MCP、UI/RPC 有未批准差异。
- state write set、错误分类、queue drop、ACK 或 delivery envelope 不一致。
- citation/meme、memory、feedback/emotion 或 Wake 的领域 oracle 不一致。
- 真实外部效果无法隔离、授权或验证。
- rollback 只恢复 pointer，没有启动 previous 并验证状态与 endpoint。
- 只通过单元测试，没有真实装配、session log、持久状态和用户结果证据。
- normalizer 忽略了未经批准的差异，或 mutant 没有被 Gate 杀死。

## 16. 发布建议

本设计已经 accepted，当前交付采用 Draft stacked PR。Review 先逐层检查 Core PR #395～#403，再检查 v3 合同、Citation/Meme 双入口迁移和各自去壳 PR；最后按上节的不可变组合重跑累计行为。迁移 PR 与去壳 PR 分开，保证 reviewer 可以先确认等价证据，再决定是否删除旧入口。

本轮不部署、不修改正式插件清单，也不把 Draft PR 的通过声明成生产发布。维护者确认相邻 diff、累计组合、完整 Phase 2 回执与 mutant 后，才按现有 parent Turn rollout 协议批准正式候选；Phase 3 及之后继续使用独立小 PR，不把另一组插件夹进本栈。
