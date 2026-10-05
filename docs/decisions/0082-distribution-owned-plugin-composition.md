# 0082 · 内置代码随部署，外置输入保留

- 状态：accepted
- 日期：2026-09-30；2026-10-01 按维护者澄清修订
- 关联条款：ONB-002、PLG-003、PLG-007、PLG-013、PLG-016、PLG-019、MIG-001、STA-001～STA-003
- supersedes：0074 第 1、3 项的内置组合与逐 ID 迁移审批策略；0080 中源码退役不会改变下一次部署组合的限制

> 运行输入的归档方式由 [0092](0092-plugin-runtime-uses-installed-files.md) 取代；内置来源、外置选择、数据身份与迁移顺序继续有效。

## 决定与理由

部署的 distribution 是内置代码来源。部署者选择新版本时，仍在分发中的内置插件采用该版本，已不在分发中的内置插件退出加载；默认 profile 删除一项不等于代码退役。外置插件保留原来选择的代码、配置和 Python 环境，除非部署清单明确更新它。

```text
┌───────────────────────┐   ┌────────────────────────┐
│ distribution 固定代码 │   │ 外置插件原 selection   │
└───────────┬───────────┘   └────────────┬───────────┘
            └───────────┬───────────────┘
              原配置 / 原启用选择 / Core 与内置 Yoyo
                        │
                一次完整 PluginSelection CAS
                        │
               同一普通插件图与生命周期
```

内置/外置只说明代码来源，不授予插件额外 API、能力或生命周期。原 `name@marketplace` 和 plugin-data 路径不变。已有用户配置和启用选择保留；新默认项建立一次普通 manifest 选择，旧 receipt 中已被用户卸载的项不重新加入。内置卸载表达为停用并保留 manifest=false；外置同 ID 覆盖被卸载时，照常移除外置 cache，但保留已经写入的 false，防止底层内置默认复活。其它外置卸载不变。

不增加部署 journal、批次状态机或第二张运行选择。首次安装仍复用原 installer 和 receipt；以后直接准备 image source 的普通不可变 descriptor，不改写首次 cache。Python requirements 仍由 PythonEnvironments 拥有；镜像构建用目标 Python 下载 wheels，停止期准备固定环境，运行期只读取准备好的引用。

## 维护者确认的意图与范围

- 内置本体是 preset：选择新版包含新增默认功能和内置退役，已有显式停用保留。默认加载、代码版本和数据迁移各有 owner。
- Yoyo 先于新版激活，范围仅是 Core 和当前发行版的内置插件。内置 migration 来源读取全部随包源码，不受默认 profile 或启用状态裁剪。外置插件自己负责兼容与迁移，其安装、部署和启动不解析 `migration.catalog.toml`。
- 采用 DSH 的“当前内置组合加用户选择”思路，不移植额外的来源优先级、配置层级或自动 provider 求解器。
- 替换依据已有 ServiceKey 或独占 claim，用户明确停用 A、启用 B。inject 表示消费依赖，不是独占证明。同名或安装先后不能证明行为互斥。重复独占提供者继续报错；本次不新增自动替换、卸载后自动恢复默认或失败回退。
- 既有同名 cache 优先规则属于来源兼容，不升级为上述替换协议。若外置与当前内置占用同 ID，二者共用 data root，发布/启动在任何 Yoyo 前拒绝该 owner 冲突，即使外置已经停用也不能绕过；替代实现须使用独立名称和插件 ID。新的外置安装在写入 cache/data 前拒绝占用内置 ID，部署清单也拒绝这种目标。本次不猜测或迁移已有外置数据。历史曾混用同 ID 后又卸载的目录，不能只凭当前 cache/selection 证明归属；这类旧安装须先做明确的数据归属核验，不宣称已覆盖全部历史路径。

Yoyo 是顺序与成功账本，不是任意脚本的数据安全证明。每个 step 仍须声明写入、事务、恢复和重试合同。失败后不启动新版，不把恢复旧选择等同于恢复数据。本次不增加自动数据清理或强制全量备份。

## 已有安装与来源证明

首次 `distribution-install.json` 永久保留为历史证据。只有 receipt 的完整 plugin ID、artifact 路径、Git revision、distribution commit、源码 provenance 和实际代码树都匹配时，旧 cache 才属于该 distribution。该证明独立于当前 selection 和 profile，退役后的旧 cache 因而不会在第二次启动复活。被替换或修改的同 ID cache 不按名字猜成内置；selection/cache 漂移明确拒绝。

新的 v4 component descriptor 增加可选 `distribution_source` 来源属性，值为构建 commit，读取时核对代码归档中的 `.akashic-source.json`。这是不可变来源声明，不是另一张可修改版本表。没有此属性的旧 descriptor 不改写；首次转换必须使用 receipt 的精确证据，或下述维护者显式批准的历史转换。普通 bare builtin 或其他显式 source 不因 source_type=builtin 被接管。配置更新沿用 descriptor 复制，保留该属性和 Python 环境引用。

### 多次历史升级的显式转换

2026-10-01 维护者批准 hua-home 已逐项核验的 52 个旧安装归入发行版：代码由正式发布回执或历史镜像精确证明，当前 cache 与 Root 一致，66 个已选数据目录无冲突。首次 receipt 不覆盖这类后续升级，不能改写它来补齐历史。

部署者先核验正式来源证据和数据归属，再在部署清单的 `distribution_adoption` 中列出精确输入；它不是按名称自动推断来源的开关。清单绑定当前 Root、目标 distribution commit，以及各项原 component、artifact pointer、Git revision、代码与静态 manifest 摘要、data_dir、原源码 commit 和审计证据摘要。发布器重新核对本机输入；正式来源证据及数据归属的人工审核由部署者负责，摘要不能代替审核。

转换凭证追加到现有 `PluginArchive`，由本次完整 Root CAS 同时引用。带凭证的 Root 使用 v2；普通 Root v1 继续可读。后续 Root 提交保留同一凭证引用，避免启动遍历整条历史链；引用不复制代码选择或产生新的可变来源表。未被当前 Root 引用的孤立凭证不生效，首次 receipt、原 cache、旧 Root 和业务数据都不改写、不减少。

```text
已批准清单 ──核对当前 Root/cache/data──► 不可变归属凭证
                                           │
新版普通组件 ──────────────────────────────┼──► 一次完整 Root CAS
                                           │
                       后续配置/启停 Root 保留同一凭证引用
```

普通启动按该凭证核对保留的旧 cache，然后将其排除出 installed 扫描；已加载的转换项必须是有效发行版 descriptor。停用或退役不释放历史数据身份；同 ID 外置安装继续拒绝，复用数据须另有显式恢复/身份转换流程。cache pointer、代码或数据目录漂移明确失败。没有完整发布回执时先核对实际 Root：未提交可重试原清单；已提交则不重放转换，按实际 Root 准备普通清单或使用已有发布回执 resume。恢复旧 Core 须同时恢复其支持的旧 Root，不能让旧读者解析 v2。

保留既有 installed 同名优先规则（包括禁用的外置来源），不从未被选择的 cache 猜测应运行的新外置版本。已有选中内置与另一同名外置 cache 冲突时，预检报告歧义，而不是让两者都从组合消失。

## 状态、失败与恢复

- 唯一运行选择仍是 `plugin-stable.json`。准备或迁移失败时不提交新 Root；归档、环境和新默认 choice 可能已准备，不能因此声称它们已运行。
- 新 Root CAS 是完整组合提交。后续 import/apply/readiness 失败保留真实结果，不声称业务数据回滚。旧完整 Root 与全部代码归档保留；恢复必须使用兼容 Core 并核对实际选择，不能仅改启动标记或删除 cache。
- `manifest.toml` 仍拥有启用选择；已有值不被 profile 覆盖。当前选择引用的 accepted/selected/failed 配置请求必须先由原 owner 恢复；文件投影成功且实际实例 ready 后才结算为 active。停止期内置配置从迁移后的持久输入归档，发布重试仍保留迁移结果；外置继续保留原选中配置。
- Session、Message、凭据、附件和 plugin-data 不被复制、删除或重建。首次 receipt、旧代码 cache、历史 Root/descriptor、环境引用不自动减少。
- 选择发行版即接受必要的 Core 与内置 Yoyo，执行完成后才提交并启动新组合。删除部署清单的 `migrations` ID 列表；旧清单需移除此字段。可选备份、停止期双锁和实际运行验收继续有效；不兼容的外置 runtime 明确失败，不自动重装。

## 验收

复用概念基线与静态检查，不新增单元测试。隔离的真实 Git bundle、installer、Manager 和 selection 场景核对升级、退役、profile 可选项、停用/卸载、同名外置覆盖、脏旧源码、失败重试、内置自动迁移、外置迁移排除、配置与环境引用、第二次启动和受保护文件字节。容器构建、生产发布与运行验收分别报告；本 PR 不部署生产。

可重跑的手工场景（不加入 `tests/`）：

```bash
.venv/bin/python scripts/deployment_composition_scenario.py
.venv/bin/python scripts/deployment_composition_scenario.py --with-wheels
.venv/bin/python scripts/deployment_migration_scenario.py
.venv/bin/python scripts/deployment_adoption_scenario.py
```

默认只使用本地临时 Git 源和空 requirements；`--with-wheels` 额外下载一个公开测试依赖并在固定插件解释器中执行。脚本创建独立目录、记录当前 commit/tree/dirty 状态并留下 `result.json`、bundle、数据库和 receipt；从不连接已有安装或 provider。`--output` 只接受尚不存在的目录。Session/Message 同时核对完整 SQL dump 与数据库字节，配置、plugin-data、旧 cache 和 receipt 核对完整文件内容。
