# 主动推送插件包

插件包是安装、启用和展示的产品单元；插件模块是运行时组合单元；信息源通过全局贡献目录共享。

```text
┌─ default-proactive
│  ├─ default_proactive ─── Runtime 与 Lifecycle
│  ├─ proactive_flow ────── 内容判断流程
│  ├─ drift_flow ────────── Drift 流程
│  └─ Dashboard ─────────── proactive.db
│
└─ wake-proactive
   ├─ wake_proactive ────── Runtime、采集与调度
   ├─ wake_proactive_flow ─ 内容蓄水池与兴趣唤醒
   ├─ wake_drift_flow ───── 低价值信息时的 Drift
   └─ Dashboard ─────────── wake_proactive.db
```

两个包都提供独占的 `proactive.runtime` capability，同时启用时拒绝启动。包清单只表达组合和能力，公共运行时仍只理解 Lifecycle、Module、Source 和 Capability。

```text
┌─ Package Resolver
│  ├─ 读取用户 [packages]
│  ├─ 展开 members
│  ├─ 检查成员唯一归属
│  └─ 检查 provides 独占冲突
│
└─ Global Source Catalog
   ├─ Feed
   ├─ Calendar
   ├─ Fitbit
   └─ Steam
```

Feed 等信息源不属于任一主动推送包。当前激活的主动推送包从同一个 `ProactiveSourceSpec` catalog 获取内容，包系统不复制、代理或改写来源数据。

旧清单中的成员插件开关会在同步时折叠成包开关，随后删除重复成员项。Dashboard 页面、后端路由和数据库读取器归对应包所有；包未启用时，不注册页面与 API。
