# 0108 · Bundle 是显式组合输入

- 状态：proposed / base artifact implemented, mode activation pending
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
目前发行安装只接纳 base，headless/minimal 的实际选择在启停来源迁移后开放，不能把
静态声明当作启动验收。历史首次安装 receipt 的 profile 标签继续只读，仍是旧来源证据。
回退需要同一历史发行版的 Core、制品与安装器，不能混用旧制品和新安装器。
接入层还须迁移旧启停选择、删除旧字段，并验证三个 mode 的实际启动/关闭及 CLI 回复。
本次场景没有写正式 config 或 manifest。
