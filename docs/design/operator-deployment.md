# 部署操作手册

状态：实现已完成，本地验证通过。下文验证记录来自部署前；正式部署以目标机 release/active 回执为准。
依据：[0082](../decisions/0082-distribution-owned-plugin-composition.md)；备份与失败恢复沿用 [0074](../decisions/0074-deployment-policy-belongs-to-operator.md)。

## 职责与主流程

部署者决定目标 Core commit、插件映射、外置数据兼容性，以及是否备份；选择发行版即接受其 Core/内置迁移。
发布工具固定输入、先完成 Core/内置 Yoyo、串行发布并核对实际运行身份。它不能从代码差异推断业务数据可降级。

```text
┌─────────────────────────────────────────────┐
│ 部署者：commit + 可选清单 + 可选 --backup     │
└─────────────────────┬───────────────────────┘
                      ▼
        构建固定产物 → 在线预检与缓存准备
                      │ 失败：现有服务继续运行
                      ▼
        停 Core/Bridge → 取得写入锁
                      ▼
        可选备份 → Core/内置迁移 → 显式外置安装
                      ▼
        必要时一次完整 Root CAS → 启动 → 实际身份/选中 Fiber 验收
                      ▼
                  active.json
```

内置组合及必要迁移随发行版更新；备份和外置目标仍显式选择：

- 无 `--backup`：不创建部署备份。既有备份和历史记录保留。
- 无 `--plan`：更新 Core/Bridge 与 distribution 拥有的内置代码，保留外置插件的原选择；预检列出待迁移，停止期自动执行。
- 清单 `targets: []`：不额外更新外置插件；内置组合按 [0082](../decisions/0082-distribution-owned-plugin-composition.md) 跟随新 distribution，保留用户选择。
- 清单不再包含 `migrations`；旧版计划须删除此字段。Yoyo 从目标发行版发现 Core 和全部内置 step，已成功 ID 不重跑；外置迁移格式不参与解析。同 ID 外置输入会与内置共用数据目录，因此新安装和部署拒绝这类接管；已有冲突先核对数据归属，改用独立插件身份。历史曾混用后又卸载的目录也需人工核验，不能从当前 cache 缺失推断数据安全。
- 必要内置迁移不隐式开启全状态备份，step 自身的恢复合同仍有效。

`--no-activate` 仅准备 source、image、Bridge venv 和 manifest，不改运行单元、CLI 或正式 state。
已有 selected Root 的新版启动也必须先完成 Core/内置迁移；失败就停止启动。外置插件自己负责其数据兼容与迁移。

### 缓存粒度与耗时

内置插件按完整内容（包括生成的 UI 文件与执行位）比较；发行 commit 的来源标记不属于代码变化。
内容、迁移后配置、Python 环境和 Core binding 合同都相同，就保留原 descriptor、代码归档和历史来源。
只改配置时继续引用原代码；依赖或合同变化仍生成新输入。旧 Root、归档和环境不自动回收。

空 requirements 或仅使用离线命名 wheel 的环境，按 requirements 字节、wheel 树、runtime root 和解释器身份复用。
在线、本地构建和 editable requirements 仍绑定完整源码，避免把构建输入误当作无关代码。
Bridge 的 release 路径指向按解释器路径/文件摘要与锁文件内容准备的环境，venv 本身不移动；源码仍从精确 release checkout 加载并验收。
旧 commit 目录继续有效。首次采用新的环境 key 可能需要填充一次缓存，后续命中直接复用。
终端产品入口 `./start` 的运行依赖同样按 Python、requirements 与 SDK 完整树复用，生成的 Core/Web 分发仍固定到各自 commit。
构建阶段只共享 pip/npm 下载缓存，不把未校验的下载、可变构建目录或其他版本的 Web 产物当作当前发布输入。
从源码构建需要 Docker Buildx（Arch 包名 `docker-buildx`）；发布器在生成 Web 分发前检查它，
并显式使用 BuildKit 构建，将完成的镜像加载到本机 Docker。旧 builder 不支持下载 cache mount。Core 与插件 wheel 构建默认使用官方 PyPI；
直接调用分发 builder 时可用既有 `--pypi-index-url` 覆盖 Core 下载源。锁文件与哈希校验不随下载源变化。

在线预检只允许增加可重建的代码归档和 Python 环境；容器将正式 state 设为只读，仅两个 cache 根可写。
此时不发布 descriptor 选择、不写配置或业务数据库，不执行迁移。停止期仍重新持锁、核对清单并读取迁移后的配置。
缓存失败发生在停机前；留下的完整不可变对象可复用，未完成对象不伪装成功。

`release.timing` JSON 日志记录单调时钟的实际秒数和成功/失败，包含 Web/wheel 构建、镜像/Bridge、分发校验、
每个插件的 hash/copy/fsync、依赖安装、输入复用及 stop/publish/start/health/runtime 阶段。
`reused` 表示该次真实命中，不能从总耗时推断。阶段存在嵌套，不把所有秒数相加；比较同名外层阶段，再按插件分项定位。
发布容器的明细写入 publication 回执的 `timings`；Host 阶段记录在发布器 stderr，启动准备见 Core 日志。

## 日常更新

日常操作只有三步：固定目标版本并核对实际数据影响 → 用发布器更新 → 验证受影响的业务结果。发布器已承担构建、预检、停止期发布、身份与选中 Fiber 核对，不重复手工执行同一组检查。历史安装归属转换和失败恢复只在遇到对应情况时展开。

备份按 `projectneed.md` BAK-001 判断，理由见 [0083](../decisions/0083-short-workflow-preserves-design-intent.md)：

| 本次实际影响 | 处理 |
|---|---|
| 源码、文档、纯前端变化，没有数据改写或待迁移 | 不额外备份，保留 Git/发行版恢复身份 |
| 仅重建派生物，完整来源与重建方式已核实 | 通常不备份派生物；保留来源不授权自动删除它 |
| 改写无法可靠重建的数据、配置或数据协议 | 先备份受影响状态及其必要关联，核对一致性与恢复方法 |
| 目标版本还包含其他改动或待迁移 | 以实际部署范围判断，不能只看最后一个 PR |

`--backup` 仍备份整个 state，不是按插件备份开关；必要的局部备份由部署者在对应 owner 停写后准备，全状态备份只在确有需要时选择。已有迁移的恢复要求继续有效。无需为每次部署新增审批表；只在授权或数据边界不明时先核对。

bootstrap 会固定目标 main SHA，使用目标版本的发布器：

```bash
curl -fsSL https://raw.githubusercontent.com/kachofugetsu09/akashic-agent/main/scripts/install-akashic.sh \
  | sh -s -- --yes
```

这会更新 Core/Bridge 和内置代码，保留外置版本，自动执行必要 Core/内置迁移，不自动创建全状态备份。需要固定版本或备份时：

```bash
sh scripts/install-akashic.sh --commit <40位SHA> --backup --yes
```

本地已有目标版本的干净 checkout 时，也可运行：

```bash
python3 scripts/akashic_release/cli.py install \
  --source-checkout /path/to/target-checkout --commit <40位SHA> --yes
```

必须使用目标版本脚本。已安装的 `akashic-release` 从当前 `runtime.env` 加载当前版本发布器；
从旧版首次切到此实现时，使用上面的 bootstrap 或目标 checkout，旧 CLI 不认识新选项。

## 更新指定外置插件

先读取实际基线：

```bash
cat /srv/data/services/akashic/state/workspace/runtime/plugin-stable.json
```

将 `root_ref` 原样填入部署清单。以下例子同时更新一个 distribution 内插件和一个外部插件：

```json
{
  "schema_version": 1,
  "expected_root_ref": "<当前64位Root>",
  "targets": [
    {"plugin_id": "assets@release", "bundled": true},
    {
      "plugin_id": "example@local",
      "bundle_relative_path": "example.bundle",
      "bundle_sha256": "<bundle文件的SHA256>",
      "target_commit": "<外部插件40位commit>"
    }
  ]
}
```

例中的插件 ID 必须换成当前已选择且启用的真实 ID。`bundled: true` 从**目标镜像的 distribution**
按插件名取固定 bundle；外部 bundle 必须含指定 commit。没有列出的外置插件连同顺序、配置输入保持原选择；内置代码自动采用目标分发。内置迁移后的配置自动进入新 descriptor，无需额外 bundled target。
内置分发缺少的代码退出加载，数据与历史材料保留；新默认项得到一次初始选择。外置插件的添加、删除和重命名仍由插件控制面负责。

准备外部 bundle 后核对哈希：

```bash
git -C /path/to/plugin-repo bundle create /path/to/inputs/example.bundle HEAD
sha256sum /path/to/inputs/example.bundle
```

存在非空 Python requirements 的目标还要在该 target 中声明离线 wheel 目录：

```json
"offline_wheels": {
  "relative_path": "example-wheels",
  "tree_sha256": "<wheel_tree_sha256输出>"
}
```

wheel 必须适配目标镜像的 Python/平台，并包含完整传递依赖。使用
`agent.plugins.python_environment.wheel_tree_sha256(Path(...))` 计算目录摘要；发布器在离线预检中
核对依赖闭包，不在停止期临时联网补依赖。

运行：

```bash
sh scripts/install-akashic.sh --commit <Core的40位SHA> \
  --plan /path/to/deploy.json --inputs /path/to/inputs --yes
```

需要备份时追加 `--backup`。输入文件由部署者管理；发布器保存清单副本，每次 publish 将 bundle/wheels 复制到临时目录并复核摘要。
仅更新 bundled 插件时可省略 `--inputs`。Core 与全部内置插件的必要迁移在停止期自动执行；停用内置插件的保留数据也在其迁移范围内。外置插件的迁移由自身操作流程负责，发布器不执行其 bundle。

Core 先升级自己拥有的账本结构；在内置业务 step 与组合发布前检查配置 owner。未结算时先用原 runtime 恢复原事务，文件投影与实际实例 ready 后才能继续。内置迁移成功后读取持久配置进入新归档，随后发布失败再重试也保留已迁移配置。迁移失败不提交新 Root，但已完成 step 的数据写入和成功账本不会因此撤销。

原发行版的正常启动入口允许这类恢复：仅当代码、来源、完整启用组合和运行时身份均与当前选择相同，且没有待迁移项时，保留已有 Root 直接启动原 owner。此时不会从尚未恢复的配置文件重新归档。改版、增删启用项或待迁移仍会阻止发布，须先完成原实例恢复。

## 备份范围

### 首次回执无法覆盖的历史安装

历史多次升级不能靠改写 `distribution-install.json` 伪装成首次安装。先逐项核验正式发布回执或历史镜像与实际代码树，核对当前 Root/cache、真实数据目录及历史混用风险，并取得维护者对数据归属的批准。

显式清单可增加一次性的 `distribution_adoption`，包含目标 `distribution_source_commit` 和 `entries`。每项需提供 `plugin_id`、原 `component_ref`、相对 `artifact_pointer`（`.artifacts/<id>`）、`source_revision`、`code_sha256`、`manifest_digest`、`data_dir`、原 `source_commit`、实际 `source_path` 和审核材料的 `evidence_sha256`。源码目录名不必等于插件名。摘要固定已审核材料，不替代来源/数据审核。当前 Root 仍由顶层 `expected_root_ref` 绑定。

在线预检和停止期均重新核对本机输入。通过后，转换凭证与新版组件由一个 Root CAS 发布；原 cache、首次回执、外置选择和数据保留。第二次启动无需再次提供转换清单。带凭证的 Root 为 v2，旧 Core 恢复必须同时采用受支持的旧 Root；参考 [0082](../decisions/0082-distribution-owned-plugin-composition.md)。

### 可选全状态备份

升级的 `--backup` 在停止期、写入锁内、任何迁移和插件安装前备份整个 `state/`，以及
`runtime.env`。单元与 CLI 只有内容变化且选择备份时才保存旧文件。备份位于发行根的 `backups/`，
`state` 使用 SQLite backup API 和完整性检查，manifest 最后发布。它不包含远程服务、外部源码或
服务器的全部 secret；外部 owner 的停写、备份和恢复仍由部署者安排。

首次初始化尚无已有 Root，此时如需保留导入前 state，由部署者在安装前自行备份。
已发布 migration step 内部的事务恢复文件属于该 step 的既有合同，不通过 `--backup` 改写历史脚本。
今后的 step 不应自行强加全状态备份策略；它必须说明写入范围、失败重试和必要的局部恢复机制。

## 失败后怎么处理

预检失败：现有服务不停止。修正清单、缺失输入或兼容问题后重跑 `install`。

停止期失败：服务保持停止，错误给出 `activation/deploy-*.json` 路径。先看其 `phase`、`detail` 和
`publication`。工具不自动切回旧 image，也不声称业务数据或外部效果回滚。

若有完整 `publication`，而错误发生在单元安装、启动或 readiness：修复环境后只重试启动验收：

```bash
akashic-release resume --attempt /srv/data/services/akashic/activation/deploy-<id>.json
```

若当前安装的 CLI 仍是旧版，用目标 checkout 的 `scripts/akashic_release/cli.py resume`。
`resume` 核对 Root、完整组件、image 和后续 active 变化，不重复迁移或插件安装。即使 active 回执已经写入，显式 resume 也会重启并重新验收。

若没有完整发布结果：先检查当前 Root、安装指针、迁移成功账本与插件自身的恢复要求，再基于**实际当前
Root**重写清单、运行 `install`。已经完整安装的清单目标可以继续归档发布；第三种 cache 状态、未结算的
reload/install owner 仍明确失败。迁移没有成功回执时是否可重试由 step 合同决定，不能伪造成功 ID。
Root 可能提交但回执丢失时，工具不猜测回滚，也不提供“一键强制放行”。

旧 `failed-*`、`attempt-external-*`、settlement 记录原样保留作证据，不再作为永久部署门禁。
新的成功只由当前持久 Root、实际运行身份和 `active.json` 证明；不靠改历史状态制造“已恢复”。
旧 `rollback`、`settle-restored`、`upgrade-bundled`、`adopt-bundled` 和 `--external-plan` 入口退役。
要降级，部署者先确认旧代码可读取当前数据，或自行恢复其选择的恢复点，再按普通 `install` 指定旧 commit
和明确插件映射；旧版本脚本若不支持此合同，需要部署者单独准备降级操作，不能盲目运行旧安装器。

## 验收与证据

发布器保留镜像/源码/Bridge 身份核对、Bridge RPC、Core health，并有界等待实际 control socket，
确认全部选中 Fiber ACTIVE 且完整 Root/顺序匹配，然后才写 `active.json`。

```bash
akashic-release doctor
cat /srv/data/services/akashic/activation/active.json
```

`doctor` 本身是基础身份和健康检查。发布验收中的 Fiber 核对也不能代替业务功能：涉及 UI、Mobile、
外部消息或插件数据时，部署者仍应验证对应入口、持久回读和实际消费结果。
本手册不把本地 scenario 或服务在线称作正式环境业务验收。

本轮本地验证：既有概念基线 41 项通过；主代码/测试类型、插件边界、迁移不可变性、控制/Bridge 协议生成物
和前端类型检查通过。临时真实 Git、正式插件 installer、Root、Yoyo 与 SQLite 验证了无备份更新、
未列出插件保持、安装完成后继续发布、过期 Root 拒绝、显式备份与迁移执行一次；本机只读 Docker
容器复跑了发布场景与 Root probe。容器使用已有依赖镜像挂载当前代码，只证明该隔离路径，
不是新 release image、完整 systemd 升级或正式业务验收。未新增或改写单元测试。
