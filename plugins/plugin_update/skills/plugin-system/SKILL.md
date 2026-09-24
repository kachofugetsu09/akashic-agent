---
name: plugin-system
description: 当用户要求安装、更新、卸载 Akashic 插件，改变 agent 能力，或询问插件何时生效、如何检查结果时使用。说明正式入口、局部换代、状态核对与数据保留。
when_to_use: 安装插件、更新插件、卸载插件、停用能力、插件自更新、Skill 或 MCP 更新、确认插件是否生效或已移除。
metadata: {"akashic": {"always": false}}
---

# 安装、更新和卸载 Akashic 插件

先确认用户要改变哪个插件、目标源码版本及允许的副作用。只读询问只查询，不顺手安装、卸载或重启。Skill 只提供说明，不增加权限；只使用本次实际可见的工具。

## 当前模型

- 插件代码和 Skill/MCP 资源来自固定安装制品，由普通插件注册。内置插件与外部插件走相同的安装、选择和生命周期规则，没有内置特权入口。
- `PluginSelection` 保存完整的已选输入；唯一 live Root 中，各插件 Fiber 持有自己的运行资源。**已安装、已选中、ACTIVE、业务验证通过是不同事实。**
- 更新或卸载只排空目标及其实际硬依赖消费者，不重建整个 Root；无关插件、手机连接和任务不应因局部变更被重启。若目标本身承载当前渠道或其依赖，明确告知相关功能可能暂时不可用。
- 旧调用保留原绑定；新版本只服务新取得的调用。不要把旧工具引用当成已经切换，也不要把一轮对话开始当作更新提交时机。

## 什么时候更新

| 情况 | 应做什么 |
| --- | --- |
| 用户明确要求安装新能力，或更新指定插件 | 核对来源和版本后，通过正式安装入口提交一次请求。 |
| 已授权修改了插件代码、依赖、SKILL.md 或 MCP 资源 | 在源码仓库验证、commit；远程来源先 push，再重新安装该固定 commit。只改源码不等于线上已更新。 |
| 上游 Git 有新提交，或普通对话开始、Core 重启 | 不自动拉取、安装或升级。重启恢复已提交的选择；上游更新本身不构成授权。 |
| 改了配置 | 按该插件自己的配置接口和生效协议处理；不能承诺改文件后立即生效。安装输入冻结的配置需要显式重新准备/发布。 |
| 需要数据库迁移或同时更换 Core 与多个插件 | 交给明确授权的停机发行流程，先目标 Core/插件 Yoyo 迁移，再提交完整选择并启动核验；普通在线安装不替插件执行迁移。 |
| 安装超时、结果不明或资源关闭失败 | 查询原请求与实际状态，不反复安装、不盲目重启或自动降级旧代码。 |

镜像中带有新版内置插件，不表示正在运行的选择已经采用它。发行升级只按原选择及来源身份采用允许替换的成员；未选、禁用或外部覆盖的成员不能偷偷启用或覆盖。

## 先找到当前运行时

优先使用实际可见的 `plugin_install` 工具。状态与卸载使用当前运行时 CLI；不存在名为 `plugin_uninstall` 的专用模型工具时，不要编造工具调用。

Host Bridge 的 Shell 会提供固定版本启动器 `AKASHIC_RUNTIME_CLI`，并把 `akashic-runtime` 加到 PATH。确认它及目标配置/workspace，再只读查询：

```sh
"${AKASHIC_RUNTIME_CLI:?需要当前 runtime 启动器}" plugin-status \
  --config "${AKASHIC_CONFIG:?需要目标配置}" \
  --workspace "${AKASHIC_WORKSPACE:?需要目标 workspace}"
```

本地运行若没有此启动器，先定位当前服务的源码入口、Python 环境、配置和 workspace，再用对应的 `python /绝对路径/main.py` 执行同一命令。不要猜开发 checkout、默认 HOME 或系统 Python；不要为管理操作再启动一个 Core。示例中的目标 ID/SHA 必须替换为查询得到的真实值。

## 安装与更新

1. 确认源码根包含 V3 `plugin.py`，身份为 `name@marketplace`。从源码仓库修改；不要编辑安装 cache、归档或 pointer。指定可获取的完整 Git commit；版本号相同也可能是不同源码。
2. 使用可见 `plugin_install` 的真实 schema：`source`、`marketplace`、`ref`，需要时再传 `sparse`。更新已有插件必须沿用其正确身份，避免换 marketplace 意外多装一份。工具自动关联发起消息和结果通知，不传旧的 validation 参数。
3. 保存返回的 `update_id`，按下面的状态分层检查。不要等待当前工具调用自身被排空；收到接单回执后允许本次调用结束，由宿主继续执行。

只有工具不可用且当前任务允许 Shell 管理时，才使用同一个在线 runtime owner 的 CLI：

```sh
"$AKASHIC_RUNTIME_CLI" plugin-install \
  --source '已核对的 Git URL 或绝对源码路径' \
  --marketplace '已核对的来源名' --ref '完整目标 commit' \
  --update-id '本次新请求的唯一ID' \
  --config "$AKASHIC_CONFIG" --workspace "$AKASHIC_WORKSPACE"

"$AKASHIC_RUNTIME_CLI" plugin-status '返回的update_id' \
  --config "$AKASHIC_CONFIG" --workspace "$AKASHIC_WORKSPACE"
```

```text
固定源码/环境/配置 → 完整 selection 提交 → accepted
                                           │ 宿主持有应用任务
                                           ▼
                            排空受影响分支 → 新 Fiber ACTIVE
                                           └→ 失败保留真实状态和错误
```

- `accepted`：选择已接纳，不等于可用。等待原渠道 active/failed 通知，或适度查询原 `update_id`。
- `active`：核对该请求的 `input_ref`、`selection_state`、当前 `archive_ref` 与 `fiber_state`；确认是目标输入，不是其他更新留下的 ACTIVE。
- `failed` 或选择已被后续更新替代：如实报告错误及当前选择。自动恢复旧代码不保证数据兼容，不能伪称回滚。
- 最后做与需求相关的能力检查：Skill 要能出现在目录且可由 `load_skill` 读出新正文；MCP 要能列出所需工具；业务功能另验真实结果。ACTIVE 不替代这些检查。

新 Skill/工具在新取得的 prompt/tool binding 中可见，旧绑定仍读原归档。先确认更新 ACTIVE，再在后续新绑定中检查，不承诺固定“下一轮必定成功”。当前架构不再使用 `plugin_latest run/revert`、候选晋升或父 Turn 结束发布。

## 卸载

1. 用 `plugin-status` 确认**完整插件 ID**和影响范围。卸载普通功能不等于删用户数据；不要把“停用一下”擅自扩大为删除私有数据库或凭据。
2. 提交一次卸载请求：

```sh
"$AKASHIC_RUNTIME_CLI" plugin-uninstall '目标name@marketplace' \
  --config "$AKASHIC_CONFIG" --workspace "$AKASHIC_WORKSPACE" --json
```

3. 收到 `accepted` 后结束本次管理调用，再用 `plugin-status` 观察。宿主先从选择中移除目标，再排空该 Fiber、硬依赖消费者及旧的 draining owner，最后移除安装代码和 manifest 条目。不要在同一调用里反复卸载或忙等自身排空。
4. 确认最终状态：目标不在 selection，`installed=false`、`cache_exists=false`、无 active generation、`draining_generations=[]`、无 cleanup pending。若状态列表已无目标，也要核对安装记录/cache 缺席、无相关进行中或失败 operation，以及原 plugin-data 仍在。仅 `enabled=false`、命令 exit 0 或 `accepted` 均不是完成。

保留 `<workspace>/plugin-data/<plugin>-<marketplace>/`、消息、附件、凭据及历史归档/回执；正常卸载不删除这些数据。需要删除数据必须另有明确授权、恢复点和该数据 owner 的入口。

如果 operation 报错、截止时间到期、连接丢失或 cache 仍在，报告实际残留和原错误。查询不是重试；只有原因已处理且明确授权后才重新提交。进程重启不自动续跑未完成的卸载，不用手动删目录或修改 manifest/selection 掩盖失败。

## 交付时说清楚

报告目标插件 ID、源码 commit、请求 ID、选择与 Fiber 的实际状态、能力检查结果、保留的数据和未完成事项。不把安装成功说成业务成功，不把选择恢复说成外部效果回滚。不要创建 workspace Skill 软链接、旧候选目录或第二份安装账本来补能力。
