# 0050 · 模型 revision 由普通插件拥有

- 状态：accepted
- 日期：2026-08-29
- supersedes：0027 中 Core 拥有模型 generation manager 的部分；0039 中 Core 直接冻结模型 generation 的部分
- 关联条款：RUN-005～RUN-012、ONB-001、PLG-003、PLG-014、PLG-016、WSP-001

## 背景

0027 已经确立 workspace 模型注册库、运行时切换和执行期冻结语义，但当前实现由 Core 的
`ModelRegistry` 同时拥有持久 revision、provider 构造和第二套 generation lease。这样新增 Provider
仍要修改 Core，内置插件也只能依赖私有 provider 对象，不能作为普通插件外置安装。

插件 v3 已经拥有 candidate/stable 发布、exact runtime snapshot lease、Service、inject、Effect 和
cleanup。模型再维护一套 generation manager 会复制同一生命周期，并让 plugin snapshot 与模型
generation 形成不必要的双重 fence。

## 决定

Core 只拥有通用插件组合与 exact runtime snapshot lease，不拥有模型、Provider、认证、角色绑定或
能力字段。Turn owner 可以请求公开、稳定的 consumer role，但不解析 role 到 model 的映射。一个普通 `models` 插件成为模型领域唯一 owner：它保存 Connection、Model、Binding 和
单调 Revision，并在一次 exact plugin snapshot lease 内复制不可变 `ModelExecution`。

`models` 插件通过公开 Service 分面提供聊天执行、embedding、只读目录、设置事务和 driver 注册。
普通 Provider 插件只注册 driver；OpenAI-compatible、Codex 和 OpenCode Go 不获得 Core 分支或内置
权限。Turn、Akasha、设置 API 和其他消费者只注入自己需要的分面。

模型 revision 继续用于 workspace SQLite 的 CAS、备份和恢复，不拥有 lease、retired generation、
manager 或独立发布生命周期。一次执行期间仍冻结完整 provider、model、credential 与能力；下一次
执行读取最新 committed plugin snapshot 和最新模型 revision。

这里的 credential 冻结指 connection ID、auth identity 和窄 `CredentialHandle` 不变，不复制固定
token payload。同一 identity 的 refresh 不增加 revision；当前 execution 的下一次 outbound request
通过原 handle 读取刷新后的 payload。endpoint、auth identity 或 API Key 选择变化仍增加 revision。

第一阶段保留现有 `model-registry.sqlite3` 路径和 schema，不移动正式数据。Provider config 与 credential
格式只允许先扩展 reader、再显式迁移；当前 artifact 必须读取所有曾经 committed 的格式。Provider
卸载只使对应 Connection unavailable，保留其配置、credential 和 plugin-data，并明确拒绝新执行。

## 理由

这一选择只保留一套发布与存活机制。增加 Provider 只增加插件；修改默认模型只增加模型 revision；
运行中的 Turn 只持有一个 exact plugin snapshot lease。Core 不再承担来源名称、协议映射或模型能力
对照表，内置与外部插件可以使用同一公开能力边界。

## 影响

- `agent.model_runtime`、`agent.provider` 和 `bootstrap.providers` 的模型领域实现迁入普通插件 artifact。
- ReAct 仍由 Core 拥有 Turn admission、取消和 terminal，但通过 `CHAT_MODELS` 取得执行绑定。
- Akasha 通过 `EMBEDDINGS` 获得绑定模型，不再读取 endpoint、key 或 wire format。
- 设置/control 的 catalog/settings API 通过 `MODEL_CATALOG` 和 `MODEL_SETTINGS` 工作，不按 Provider ID 分支；现有 Onboarding 前端固定入口和动态 Web contribution 属于后续 UI 规格。
- Git 回滚只恢复代码和 artifact；不能冒充新的模型 revision、消息或外部调用已经回滚。

## 验收

- [ ] Core 公共路径不包含 Provider 名称、endpoint、认证流程、能力表或模型专用 generation manager。
- [ ] `models` 与三个首批 Provider 都能移出仓库，通过正式 install、冷启动、调用、卸载和重装 Gate。
- [ ] 一个 Turn 只租 exact plugin snapshot，并在其中冻结一个 `ModelExecution`；切换 revision 不影响在途执行。
- [ ] passive、job、compaction、vision、memory、Akasha、设置与模型目录消费者全部迁到公开 Service。
- [ ] 现有模型库和 Session selection 不搬迁、不静默改写；消息正文继续只追加。
- [ ] 缺失 driver、认证失败、限流、超时、协议错误和卸载均保留原始失败含义，不 silent fallback。
