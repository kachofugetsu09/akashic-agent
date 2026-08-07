# 0028 · 模型凭据随 workspace connection 保存

- 状态：accepted
- 日期：2026-08-07
- 勘误：[0027](0027-runtime-models-use-generation-leases.md) 中“凭据继续由全局 CredentialStore 保存”的部分
- 关联条款：RUN-009～RUN-012、ONB-001、WSP-001、BAK-001

## 背景

0027 已把 Provider connection、model 和 role 迁入 workspace 的
`model-registry.sqlite3`，但 credential id 仍指向全局 `~/.akashic/auth.json`。
这样一个可运行模型要同时恢复 workspace 数据库和 HOME 下的第二个 owner，首次设置事务也要跨
SQLite 与 JSON 两个提交边界。用户确认当前是单人本地 Companion，接受用文件权限保护明文
secret，希望先采用最小结构，不引入 keyring、加密层或 credential generation。

## 决定

1. Provider API Key、Codex access/refresh token 与账号路由字段直接作为 JSON payload 保存到对应
   `model_connections` 行；`auth_kind` 标识 payload 类型。Base URL、Provider、模型与凭据由同一
   workspace 数据库拥有。
2. `model-registry.sqlite3`、可能出现的 WAL/SHM 和设置/迁移备份使用 `0600`。设置状态、日志、
   API 响应和 Observe 不返回或复制 secret。
3. API Key connection 与模型候选在一个 SQLite 事务提交。Codex 登录可以先建立没有模型的
   connection，登录成功后原位写 token；选择模型后复用同一 connection。
4. Codex token refresh 原位更新 credential payload，不增加模型 revision。Provider、Base URL、
   模型、角色或显式 API Key 设置变化仍通过模型设置事务增加 revision，下一 execution generation
   生效。
5. 未发布的 `20260807_01_model_registry_database` 直接完成一次迁移：从 inline key 或旧 JSON
   credential 复制被模型引用的凭据，验证数据库和新配置后才允许 Yoyo 落账。旧 JSON 不自动删除，
   因为它可能仍被其他 workspace 或非模型配置引用；迁移后本 workspace 的模型运行不再读取它。
6. 不新增加密、系统 keyring、credential 独立表、credential generation 或自动删除。删除来源与
   secret 仍需以后名称明确的独立操作。

```text
┌──────────────────── workspace ────────────────────┐
│ model-registry.sqlite3                            │
│ connection(provider · base_url · auth_payload)   │
│        └── model definitions ── role bindings    │
└───────────────────────┬───────────────────────────┘
                        ▼ execution start
               immutable model generation
```

## 理由

connection 本来就是 API Key 或登录订阅的产品身份。把 secret 留在同一行，使 onboarding 保存、
workspace 恢复和运行时读取都只有一个权威 owner；现有 `CredentialStore` 调用形状可以保留为窄
访问接口，不需要再造 secret 服务。`0600` 符合当前单人本地部署边界，也明确保留未来需要更强
保护时的升级空间。

## 影响与回滚

- workspace 备份包含模型 secret，必须按 secret 处理，不能上传到普通诊断或提交到 Git。
- 旧 `auth.json` 保留为迁移前恢复证据和非模型兼容状态，不再是已迁移模型的运行时 fallback。
- 迁移或设置失败时用 SQLite backup API 恢复数据库 preimage，并恢复 `config.toml`；失败不得写
  Yoyo 成功回执或部分发布模型 revision。
- 源码回滚到 0027 实现时，可从迁移 operation backup 恢复旧配置与 JSON credential；不能只回退
  SQLite schema 后继续启动。

## 验收

- 全新 onboarding 后只备份 workspace 模型数据库即可恢复已配置 Provider connection 和模型 secret。
- API Key、Codex refresh token 不出现在设置 state、日志、Observe 或 `config.toml`。
- 数据库、WAL/SHM 和备份均为 `0600`；权限过宽时 credential 读取 fail-loud。
- 迁移首次执行、失败、修复后重试和重复启动保持幂等，失败前后的 config、数据库与旧 JSON 字节可核对。
- Codex token refresh 与 API Key 模型请求都从 workspace 数据库读取，且现有 execution lease 语义不变。
