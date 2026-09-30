# 0082 · 内置代码随部署，外置输入保留

- 状态：accepted
- 日期：2026-09-30
- 关联条款：ONB-002、PLG-003、PLG-007、PLG-013、PLG-016、MIG-001、STA-001～STA-003
- supersedes：0074 第 1 项中内置代码不随部署更新的策略；0080 中源码退役不会改变下一次部署组合的限制

## 决定与理由

部署的 distribution 是内置代码来源。部署者选择新版本时，仍在分发中的内置插件采用该版本，已不在分发中的内置插件退出加载；默认 profile 删除一项不等于代码退役。外置插件保留原来选择的代码、配置和 Python 环境，除非部署清单明确更新它。

```text
┌───────────────────────┐   ┌────────────────────────┐
│ distribution 固定代码 │   │ 外置插件原 selection   │
└───────────┬───────────┘   └────────────┬───────────┘
            └───────────┬───────────────┘
              原配置 / 原启用选择 / 显式迁移预检
                        │
                一次完整 PluginSelection CAS
                        │
               同一普通插件图与生命周期
```

内置/外置只说明代码来源，不授予插件额外 API、能力或生命周期。原 `name@marketplace` 和 plugin-data 路径不变。已有用户配置和启用选择保留；新默认项建立一次普通 manifest 选择，旧 receipt 中已被用户卸载的项不重新加入。内置卸载表达为停用并保留 manifest=false；外置同 ID 覆盖被卸载时，照常移除外置 cache，但保留已经写入的 false，防止底层内置默认复活。其它外置卸载不变。

不增加部署 journal、批次状态机或第二张运行选择。首次安装仍复用原 installer 和 receipt；以后直接准备 image source 的普通不可变 descriptor，不改写首次 cache。Python requirements 仍由 PythonEnvironments 拥有；镜像构建用目标 Python 下载 wheels，停止期准备固定环境，运行期只读取准备好的引用。

## 已有安装与来源证明

首次 `distribution-install.json` 永久保留为历史证据。只有 receipt 的完整 plugin ID、artifact 路径、Git revision、distribution commit、源码 provenance 和实际代码树都匹配时，旧 cache 才属于该 distribution。该证明独立于当前 selection 和 profile，退役后的旧 cache 因而不会在第二次启动复活。被替换或修改的同 ID cache 不按名字猜成内置；selection/cache 漂移明确拒绝。

新的 v4 component descriptor 增加可选 `distribution_source` 来源属性，值为构建 commit，读取时核对代码归档中的 `.akashic-source.json`。这是不可变来源声明，不是另一张可修改版本表。没有此属性的旧 descriptor 不改写；首次转换必须使用 receipt 的精确证据。普通 bare builtin 或其他显式 source 不因 source_type=builtin 被接管。配置更新沿用 descriptor 复制，保留该属性和 Python 环境引用。

保留既有 installed 同名优先规则（包括禁用的外置来源），不从未被选择的 cache 猜测应运行的新外置版本。已有选中内置与另一同名外置 cache 冲突时，预检报告歧义，而不是让两者都从组合消失。

## 状态、失败与恢复

- 唯一运行选择仍是 `plugin-stable.json`。准备失败或迁移未获批准时不提交新 Root；归档、环境和新默认 choice 可能已准备，不能因此声称它们已运行。
- 新 Root CAS 是完整组合提交。后续 import/apply/readiness 失败保留真实结果，不声称业务数据回滚。旧完整 Root 与全部代码归档保留；恢复必须使用兼容 Core 并核对实际选择，不能仅改启动标记或删除 cache。
- `manifest.toml` 仍拥有启用选择；已有值不被 profile 覆盖。原配置的选中投影优先于过期文件；未结算配置事务必须先由原 owner 恢复。显式 target 可采用被批准迁移产生的新配置。
- Session、Message、凭据、附件和 plugin-data 不被复制、删除或重建。首次 receipt、旧代码 cache、历史 Root/descriptor、环境引用不自动减少。
- 0074 的显式 migration ID、可选备份、停止期双锁和实际运行验收继续有效。新版本启动不批准任何 Core 或业务数据迁移；不兼容的外置 runtime 明确失败，不自动重装。

## 验收

复用概念基线与静态检查，不新增单元测试。隔离的真实 Git bundle、installer、Manager 和 selection 场景核对升级、退役、profile 可选项、停用/卸载、同名外置覆盖、脏旧源码、失败重试、迁移拒绝、配置与环境引用、第二次启动和受保护文件字节。容器构建、生产发布与运行验收分别报告；本 PR 不部署生产。

可重跑的手工场景（不加入 `tests/`）：

```bash
.venv/bin/python scripts/deployment_composition_scenario.py
.venv/bin/python scripts/deployment_composition_scenario.py --with-wheels
```

默认只使用本地临时 Git 源和空 requirements；`--with-wheels` 额外下载一个公开测试依赖并在固定插件解释器中执行。脚本创建独立目录、记录当前 commit/tree/dirty 状态并留下 `result.json`、bundle、数据库和 receipt；从不连接已有安装或 provider。`--output` 只接受尚不存在的目录。Session/Message 同时核对完整 SQL dump 与数据库字节，配置、plugin-data、旧 cache 和 receipt 核对完整文件内容。
