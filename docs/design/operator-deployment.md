# 部署操作手册

状态：实现已完成，本地验证通过。下文验证记录来自部署前；正式部署以目标机 release/active 回执为准。
依据：[0074](../decisions/0074-deployment-policy-belongs-to-operator.md)。

## 职责与主流程

部署者决定目标 Core commit、插件映射、数据兼容性、批准哪些迁移，以及是否备份。
发布工具固定输入、阻止未批准的迁移、串行发布并核对实际运行身份。它不能从代码差异推断业务数据可降级。

```text
┌─────────────────────────────────────────────┐
│ 部署者：commit + 可选清单 + 可选 --backup     │
└─────────────────────┬───────────────────────┘
                      ▼
        构建固定产物 → 在线只读预检
                      │ 失败：现有服务继续运行
                      ▼
        停 Core/Bridge → 取得写入锁
                      ▼
        可选备份 → 已批准迁移 → 显式插件安装
                      ▼
        必要时一次完整 Root CAS → 启动 → 实际身份/选中 Fiber 验收
                      ▼
                  active.json
```

备份、迁移和插件更新是三个独立选择：

- 无 `--backup`：不创建部署备份。既有备份和历史记录保留。
- 无 `--plan`：只换 Core/Bridge，保留所有插件输入；发现待迁移则在预检中失败。
- 清单 `targets: []`：不更新插件。初装 profile 只用于首次安装，不是升级范围。
- 清单 `migrations: []`：不允许执行待迁移。批准列表必须覆盖全部实际 pending ID；已成功 ID 不重跑，未知 ID 拒绝。
- `--backup` 不批准迁移；批准迁移也不隐式开启全状态备份。

`--no-activate` 仅准备 source、image、Bridge venv 和 manifest，不改运行单元、CLI 或正式 state。
已有 selected Root 的普通启动只检查迁移，不再偷偷执行新 step；首次显式初始化仍允许建库流程。

## 日常更新

以下命令适用于**本实现合并后**。bootstrap 会固定目标 main SHA，使用目标版本的发布器：

```bash
curl -fsSL https://raw.githubusercontent.com/kachofugetsu09/akashic-agent/main/scripts/install-akashic.sh \
  | sh -s -- --yes
```

这是 Core/Bridge 更新，不做部署备份、不自动更新插件。需要固定版本或备份时：

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

## 更新指定插件或执行迁移

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
  ],
  "migrations": []
}
```

例中的插件 ID 必须换成当前已选择且启用的真实 ID。`bundled: true` 从**目标镜像的 distribution**
按插件名取固定 bundle；外部 bundle 必须含指定 commit。没有列出的插件连同顺序、配置输入保持原选择。
这个发布入口不负责添加、删除或重命名插件；这些操作仍由现有插件控制面负责。

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
仅更新 bundled 插件时可省略 `--inputs`。未批准迁移时，错误会列出缺少的 ID；部署者阅读对应
Core/插件 step 的写入范围和重试合同后，再将 ID 加入 `migrations`。框架不会自动替你批准。
迁移若修改插件配置，该插件也应明确列入 targets，使安装后的新配置重新归档到本次 Root。

## 备份范围

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
