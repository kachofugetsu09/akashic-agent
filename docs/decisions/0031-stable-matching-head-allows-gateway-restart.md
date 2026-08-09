# 0031 · Stable 与本地 HEAD 一致时允许 Gateway 重启

- 状态：accepted / implemented
- 日期：2026-08-09
- 修订：[0029 · main Gateway 对账移动 WebUI Stable](0029-main-gateway-reconciles-mobile-webui-stable.md)
- 关联条款：WEBUI-004～WEBUI-006、GOV-005、TST-006～TST-008
- 设计：[服务端发布的移动 WebUI OTA](../design/server-published-mobile-webui.md)

## 背景

0029 用 `HEAD == origin/main` 限制 Gateway 的自动 Stable 发布权限，但实现先检查该条件，再读取已有发布事实。远端 `main` 前进后，一个本地源码和当前 Stable 都没有变化、两者 `source_commit` 仍完全一致的 clean `main` 会在重启时失败。远端是否出现更新因此反向改变了已发布版本的启动资格。

## 决定

Gateway 在 clean `main` 启动时先区分已有版本重启与新版本自动发布：

```text
当前 Stable.source_commit == HEAD
              │ 是
              └──────────────► no-op，正常启动
              │ 否
              ▼
        HEAD == origin/main
          │ 是          │ 否
          ▼             ▼
    按 0029 自动发布    fail-loud
```

`origin/main` 只拥有尚未发布 HEAD 的自动发布授权，不拥有已一致 Stable 的持续启动授权。匹配路径只读取并完整校验当前 `ReleaseView` 和 Stable manifest，不增加状态、不构建、不写 publication journal，也不改变 Stable 或 Preview。Stable 为空、manifest 损坏或 Stable `source_commit` 与 HEAD 不一致时不能进入该路径。

feature branch、detached HEAD 和 dirty tree 仍不触发自动发布。发布 journal 继续记录一个 source commit 是否曾成功成为 Stable，并继续防止重启覆盖显式 rollback；本修订不改变这项幂等语义。

## 理由

当前 Stable 指针和 immutable manifest 已经保存 `source_commit`，足以证明运行源码与现有 Stable 的版本一致，不需要新增数据库字段、缓存或恢复状态。把远端同步检查限制在自动发布分支，可以保留可信发布来源，同时避免远端推进使一个未变化的已发布版本突然无法重启。

## 影响

- 本地 clean `main` 可以继续运行与其 HEAD 一致的已有 Stable，即使远端已有更新。
- Stable 与 HEAD 不一致时，自动发布仍只接受与 `origin/main` 完全一致的 HEAD。
- 该判断不会 pull、构建、推进指针或追加 journal。
- Preview 的选择、优先级和生命周期不变。

## 验收

- 本地 `main` 落后 `origin/main`、当前 Stable `source_commit == HEAD` 时启动成功。
- 上述路径不构建、不追加 publication journal，Stable/Preview 和 `selection_digest` 逐项不变。
- 本地 `main` 落后 `origin/main` 且当前 Stable 为空或 `source_commit != HEAD` 时仍 fail-loud。
- 与 `origin/main` 一致的新 HEAD 继续按 0029 自动发布；已成功发布的 source commit 继续 no-op。
