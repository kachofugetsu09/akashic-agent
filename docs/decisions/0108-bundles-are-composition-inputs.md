# 0108 · Bundle 是显式组合输入

- 状态：proposed / input format implemented, activation pending
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

本层只加入输入格式和三个声明文件，尚未替代发行安装器及旧启停来源，不能宣称三个
mode 已从正式启动链验收。接入层须迁移旧启停选择、删除旧字段和 JSON profile，并验证
实际启动/关闭及 CLI 回复。旧 config 文件与 manifest 没有被本层写入。
