# Core 静态边界

状态：#1179 的 R7～R11 已实现；运行可替换性仍需真实组合验收。

`python scripts/plugin_boundary.py check --base origin/main` 固定以下边界：

| 规则 | 失败条件 |
|---|---|
| R7 | Core 声明的 key 缺少可解析的 Core provider，或 Core 提供业务 key |
| R8 | Core 导入 `plugins.*.contract`（R1 还禁止其他插件 import） |
| R9 | 宿主、组合内核、bootstrap、main 的标识符或运行时字符串出现领域词 |
| R10 | Core 声明集合不等于固定九个 key，或角色不是 core |
| R11 | 未登记的 Core 函数直接打开 SQLite，或 web_shell 之外导入 ASGI 框架 |

R9 排除注释和 docstring，标识符中下划线也视作词界；账本只能减少。检查不是安全
沙箱：它解析字面声明、import 别名和可解析的 provide 路径，不执行动态 Python。
能力目录公开未解析的 provider，不能据此宣称零消费者或可独立替换。

┌─────────────────────┐    ┌─────────────────────┐
│ Core 静态边界检查   │    │ 真实组合与故障验收  │
│ 禁止业务重新进入   │    │ 验证生命周期和数据 │
└─────────────────────┘    └─────────────────────┘

SQLite 白名单只包含插件 reload journal、迁移回执及 release 完整备份函数，以及
已经发布的这类历史迁移。它不允许在这些函数中增加业务表 owner；这种语义仍需要
代码评审。Ledger 独占业务库。离线旧数据交接工具位于 `scripts/proactive_island/`，
不随运行时 Core 装载。

`core/net/http.py` 保留进程共享连接池、HTTP 重试预算和传输原语，不导入 Web
服务框架，不发布业务 ServiceKey。模型错误解释已归 Models。`bootstrap/web_shell.py`
是唯一 Core ASGI owner，只根据声明的端点代理并在 runtime 不可用时响应外壳页面。

验证使用实际 CLI：当前源码通过；临时添加包含业务 key、插件合同 import、领域名
和别名 SQLite connect 的 Core 文件时，R7～R11 均失败；临时文件未执行、验证后撤回。
