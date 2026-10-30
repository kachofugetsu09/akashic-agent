# 0139 · 部署失败回到原版本，外置插件坏掉不拖垮 Core

- 状态：accepted / implemented for review
- 日期：2026-10-30
- 关联条款：RUN-004、MIG-001、PLG-013
- 修订：[0074](0074-deployment-policy-belongs-to-operator.md) §5「不自动恢复旧 runtime/env」、[0091](0091-shell-self-deployment.md)「不能盲目回滚旧镜像」；在下面的条件内改为自动回到原版本。

## 背景

2026-10-30 部署 50f88f60 时，生产停机约 25 分钟：

1. 当天上午，shell_safety 3.0.1 导入失败。它作为外置插件，仍然留在已选组合里，状态是 failed。
2. 新版启动、健康检查都通过了，但就绪检查要求所有已选插件都 ACTIVE。发布器因此判定失败，停掉了新实例，并保持停机。
3. 原版本也带同一个就绪检查，手工回滚同样起不来。

Agent 通过 [0091](0091-shell-self-deployment.md) 自部署时，这种失败没人能救：服务停了，Agent 也就没了，只能等部署者上机。

## 决定

1. **外置插件不阻止启动。**
   - 就绪检查里，已选的外置插件（没有发行版来源的输入）起不来时，记入回执的 `runtimeCheck.degraded_external`，Core 照常启动。
   - Core 与内置插件仍须全部 ACTIVE。这样即使外置插件坏了，Agent 仍然活着，可以用 `plugin_install` 重装或换版本。
   - 代价：shell_safety 这类守卫插件坏掉时，启动后它的限制不生效，与撤下该插件一致（见 [#1766](https://github.com/kachofugetsu09/akashic-agent/pull/1766) 对限制 fail-open 的说明）。
   - `install` 结果带 `degradedExternal`，`akashic-release status` 能直接看到；部署 Skill 要求先重装守卫插件，重装好之前不做删除类操作。
2. **无迁移的失败回到原版本。**
   - 停机前在内存保存 `plugin-stable.json` 原文。在线预检报告没有待执行迁移时，回执标记 `rollback: eligible`。
   - 停机后的任何失败（停止确认、发布、启动、健康、就绪检查）都会依次：
     1. 把选择文件原子写回原文，前提是当前选择仍是原 Root（无需改）或本次发布的 Root。选择里的输入记录不能凭 ref 重建，`PluginSelection.commit` 只接受当前或本进程准备的输入，所以写回原文而不是重新提交；
     2. 恢复原单元、CLI 与 runtime.env；
     3. 启动原版本并核对健康。候选启动即退出会让单元进入 `start-limit-hit`，部署用户不能 `reset-failed`，所以在 2 分钟限流窗口内每 10 秒重试启动。
   - 健康等待发现 Core 单元已 failed 就立即判失败，不再按就绪超时（生产 1800 秒）空等。
   - 发布已提交新选择、但在返回前崩溃时，回执里还没有新 Root。停机期间只有本次发布会写选择，所以失败阶段为 publishing 时接受任意新 Root。
   - 失败发生在停机确认之前时，回退会把原实例停下再拉起一次，回执不区分这两种情况。
   - `akashic-release resume` 没有回退点，失败后照旧保留现场。自部署 worker 走 `install`，不受影响。
   - 成功后，回执记为 `rolled_back`，命令仍以失败退出，并说明已回到哪个版本。
3. **以下情况保持原行为，保留停机现场：**
   - 有待执行迁移：数据可能已经前移，原代码不一定能读；
   - 选择已被其他写入者改过；
   - 回退本身失败（回执记录 `rollbackError`）。
4. **离线安装不再删除被替换的旧产物。** 这与运维手册「旧归档与环境不自动回收」一致。当前选择可能仍引用旧产物，删除后发布器读选择会 `FileNotFoundError`。

## 取舍

- **不做通用降级。** 回退只覆盖「本次没有任何迁移」的情况，靠预检的 `migration_ids` 判断，不推断 schema 兼容性。
- **原版本必须仍可启动。** 原镜像和原 checkout 都保留在发行目录里，只要不被清理就行。
- **回退会停机一段时间。** 生产实测，候选启动即崩溃时约停 2.5 分钟，其中约 1.5 分钟是在等启动限流窗口过去。
