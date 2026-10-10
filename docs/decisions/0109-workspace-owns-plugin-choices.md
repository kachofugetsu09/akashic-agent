# 0109 · Workspace 拥有插件选择

- 状态：proposed / patch API implemented, activation pending
- 日期：2026-10-11
- 依据：#1179 P4、ADR-0108；用户授权自主决定并把重大选择单列 PR。

全局 `manifest.toml` 的 enabled 与 Core `disabled_builtin` 会让同一插件有两个
启停 owner，也让共享安装目录的 workspace 相互改变选择。它们迁移到各 workspace 的
`bundle.patch.toml`；实际运行仍只由已提交的 `plugin-stable.json` 决定。

```text
┌───────────────────────────┐
│ artifact pointer / 发行制品 │  有哪些代码可用
└───────────────────────────┘
┌────────┐   ┌────────┐   ┌──────────────────────┐
│ base   │ → │ mode   │ → │ workspace bundle patch│  选择哪些代码
└────────┘   └────────┘   └───────────┬──────────┘
                                    ▼
                        ┌────────────────────────┐
                        │ plugin-stable.json     │  已提交的完整运行输入
                        └────────────────────────┘
```

安装、启用、禁用和安装失败回退写当前 workspace 的 patch。卸载删除 cache 后保留
显式 disabled row，防止下次 base 组合重新启用它。库存直接由制品与 artifact pointer
证明，不另建一份列表。共享 cache 的物理安装/卸载仍受现有发布锁约束；本决定没有把
共享代码改成 workspace 私有副本。

启停操作保留 row 的 config；回退新增选择时恢复 row 缺席。row ID 支持完整插件身份，
使外置插件不需要另造编码名。明确更换 row 的 provider 不触发自动依赖求解。

**配置选择：** 合并时仍整行替换 config，但它是新插件数据目录的初始配置。已有
`config.input.json` 由原来的插件配置事务和 revision 独占，不被 bundle 在重启时覆盖。
这保留了用户已保存配置及失败恢复的合同；若要让 bundle 持续拥有全部配置，应另做
配置 owner 迁移，不能保留两份同时可写的权威配置。

**发行选择：** 新发行版直接提交已验证 sources，不再先复制到全局 cache、再用首次
receipt 排除这些副本。旧 receipt 继续只读证明历史 cache 归属，不能作为当前选择。
新 receipt 应明确记录新入口和空的历史 cache 集合。

切换必须原子交付：Yoyo 先保存旧 config、旧全局清单、旧 patch 与完整目标的恢复计划，
复制启停选择，再移除 Core 旧字段；停机重试只接受原值或计划目标值。旧全局文件不再
由运行时读写，保留给尚未迁移的共享 workspace 及旧版本恢复。已有用户 patch 高于
迁入开关；既有配置、消息、附件和 plugin-data 保持原值。未结算安装须先由旧 owner 结算。

本 PR 提供 patch API；下一层接通所有写入、一次性迁移和恢复验收后删除旧读取。
验收须覆盖旧版本生成数据、故障后重试、两个 workspace 共享 cache、卸载后不复活，
以及 base/headless/minimal、实际 CLI 回复和已有部署迁移场景。恢复时须使用旧 Core，
并按 `runtime/before-bundle-choices.json` 恢复原 config/patch；旧全局清单原文留在计划中。
