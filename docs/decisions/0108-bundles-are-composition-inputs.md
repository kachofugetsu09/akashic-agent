# 0108 · Bundle 是显式组合输入

- 状态：proposed / modes implemented, user choice migration pending
- 日期：2026-10-11
- 依据：#1179 P4；用户授权自主决定并单独提交重大选择。

组合输入按发行版 `base.toml`、指定 mode、用户 patch 的顺序读取。每个 row 有稳定 ID，
声明实际 plugin 身份、config 和 disabled；后层同 ID **替换整行**，不深合并 config。
每行必须写 plugin；省略 config 表示空对象，省略 disabled 表示 false。因此 patch
不能只写一个 disabled 并隐式继承隐藏字段。两个 row 指向同一插件时报错。

┌──────────────┐   ┌───────────┐   ┌────────────┐
│ 发行版 base  │──►│ mode      │──►│ 用户 patch │
└──────────────┘   └───────────┘   └──────┬─────┘
                                        ▼
                              ┌────────────────────┐
                              │ 完整选择提交       │
                              │ plugin-stable.json │
                              └────────────────────┘

Bundle 不发现 provider、不补依赖、不决定服务优先级，也不保存运行状态。只有完整选择
提交后才运行；重启仍读取已提交选择。默认 base 明列当前 48 个插件；headless 禁用
UI、客户端页面及 Channel 连接，保留 Gateway/CLI 和后端能力；minimal 禁用全部 row。
用户 patch 可以在 mode 之后明确重新启用插件，所以“零插件”指没有额外启用 patch 的 minimal。

发行制品 schema 3 携带三个 TOML bundle 及各自摘要；namespace 由制品 marketplace
字段确定。旧 JSON profile 和解析器已删除，安装命令使用 `--bundle`、`--ensure-bundle`。
`AKASHIC_PLUGIN_BUNDLE` 选择 mode，缺省为 base；安装路径必须与该值一致。部署和
runtime 使用同一进程配置，Docker 与本地启动器把它传到两侧。修改 mode 必须走正式
发行安装/发布入口，提交新选择之后启动；不能只换环境变量而直接启动旧选择。
空组合同样提交非 null Root，宿主 journal 独立于插件数量初始化。全部制品的公共 API
仍可登记，但只有选中的实现挂载；停用 UI 不会使 Gateway 的协议编码失去类型定义。

本层开放三个 mode；用户 patch 接入、旧 Config/manifest 启停迁移继续下一层。
在完成迁移前，已有 manifest 的禁用选择仍保留，mode 可以进一步禁用，不能覆盖它。
历史首次安装 receipt 的 profile 标签继续只读，仍是旧来源证据。回退需要同一历史
发行版的 Core、制品与安装器，不能混用旧制品和新安装器。

`scripts/bundle_modes_scenario.py` 使用真实发行制品和 AppRuntime，核对完整选择、
启停与退出后的端点撤回；headless 通过当前 `main.py exec` 命令完成完整回复。
旧 issue 中的 `main.py cli` 已被原生命令入口替代，不恢复别名。模型请求发给隔离
HTTP 服务，经过真实 OpenAI-compatible driver、Models、Reply、Ledger 和 Gateway。
它证明本地协议和组合链路，不能替代远程模型服务或浏览器验收。
