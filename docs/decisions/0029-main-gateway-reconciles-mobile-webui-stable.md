# 0029 · main Gateway 对账移动 WebUI Stable

- 状态：accepted
- 日期：2026-08-08
- 修订：[0031 · Stable 与本地 HEAD 一致时允许 Gateway 重启](0031-stable-matching-head-allows-gateway-restart.md)
- 关联条款：WEBUI-004～WEBUI-006、GOV-005、TST-006～TST-008
- amends：[0022](0022-mobile-webui-uses-server-selected-generations.md)
- 设计：[服务端发布的移动 WebUI OTA](../design/server-published-mobile-webui.md)

## 背景

0022 要求所有 Stable 都由显式命令发布。这个边界保护了开发分支和未提交输入，却让普通用户拉取已经合并的服务端 `main` 后仍可能长期使用旧 WebUI。服务端代码与默认移动界面因此形成两个需要人工同步的生产版本，启动成功也不能证明当前 `main` 已成为 Stable。

## 决定

Gateway 启动时只对一个受限来源执行自动 Stable 对账：当前分支必须是 `main`，当前 HEAD 必须与本地 `origin/main` 完全一致，且发布 journal 中没有该 source commit 的成功 Stable 记录。满足条件后，Gateway 复用 0022 的隔离、可复现 Stable 发布命令；发布失败直接中止启动，不能以旧 Stable 伪装新服务已经完整就绪。

```text
git pull main
      │
      ▼
Gateway 启动 ── 非 main / 未同步 ──► 不自动发布
      │
      ▼
source commit 已有 Stable 记录？ ── 是 ──► no-op
      │ 否
      ▼
可复现构建并原子提交 Stable ── 失败 ──► Gateway 启动失败
      │ 成功
      ▼
继续启动，Preview 指针保持原值
```

feature branch、detached HEAD、dirty tree、源码保存、普通构建和 watcher 不取得发布权限。开发者仍用显式 Preview 验证未合并界面。自动 Stable 不清除 Preview；客户端继续按既有优先级解析当前完整 `ReleaseView`。

## 理由

`main == origin/main` 把自动动作限制在已经进入远端生产历史的提交。以 source commit 的成功 Stable journal 作为幂等事实，可以在重启和显式回滚后避免重复构建，同时不把“当前指针不是该提交”误判为从未发布。复用唯一发布者保留不可变 generation、可复现构建和原子指针语义，没有增加第二条发布实现。

## 影响

- 拉取并启动远端 `main` 成为用户获得同提交默认移动 WebUI 的标准路径。
- Stable 构建失败现在属于 Gateway readiness 失败，旧进程或旧 Stable 保持原状。
- 显式回滚仍然有效；重启不会把已经发布过的当前提交再次强推为 Stable。
- Preview 的发布、清除、提升和优先级不变。

## 验收

- feature branch 和未与 `origin/main` 同步的 main 不改变发布仓；后者 fail-loud。
- 首次启动新 main 只产生一次成功 Stable 发布，同提交后续启动不再构建或追加发布。
- Stable 发布失败时 Gateway 不进入 ready，旧 ReleaseView 保持完整可读。
- 自动对账前后的 Preview target 完全相同。
- 真实合并、拉取和重启后，发布 journal 的 source commit、Stable generation 与运行进程 HEAD 一致。
