# Veda workspace 人格设计

## 背景

Akashic 的身份和人格此前由 Python 常量直接拼入 Main、Proactive 和 Drift prompt。`MEMORY.md` 与 `SELF.md` 已经是 workspace 中可持续维护的 Markdown，但人格没有独立真源，修改需要发布代码，也无法让用户明确调整后从下一轮统一作用于三条链路。

人格文本还混有工具验证、输出格式和 emoji 等行为约束。把两者整体交给可编辑文件，会让人格修改同时绕过系统工作规则；继续在多个 prompt builder 复制人格，又会产生所有权漂移。

## 设计

1. `<workspace>/memory/VEDA.md` 是运行时人格真源，Main、Proactive 和 Drift 每次组装 prompt 时严格读取同一文件。
2. Veda 只描述 Akashic 的人格和关系定位。工具、安全、事实核验、检索、格式与持久化约束保留在代码拥有的行为规范中。
3. 只有 Main Agent 在用户明确要求修改人格或 Veda 时可以写入。后台 optimizer、Proactive、Wake 和 Drift 只读。
4. 本轮 prompt 在模型执行前冻结；本轮对 Veda 的写入只从下一次 prompt 组装开始生效。
5. 合法 Veda 必须存在、为非空 UTF-8。缺失或损坏时失败，不使用代码 fallback，也不自动修复。
6. 仓库保存版本化默认 Markdown。workspace 初始化只在缺失时创建；已有合法内容不覆盖。退役的 Git migration 脚本仅作为历史源码保留，不再自动执行。
7. `python main.py veda-reset` 是 runtime 加载前的显式恢复入口。它先保存原始字节和 hash，再原子发布当前版本默认内容。

## 状态边界

- Veda 是必须纳入 workspace 备份和恢复核对的权威 Markdown。
- 初始化与代码升级不得覆盖用户定制人格；恢复默认必须使用名称明确的命令。
- Veda 损坏会阻止 Agent runtime 或下一次 prompt 组装，需要维护者运行恢复命令。
- 移动端继续作为 Core 消费者，不增加人格协议、数据库或客户端状态。

## 验收

- 三条 Agent 链路观察到同一 Veda 内容，修改后下一次组装生效。
- `MEMORY.md`、`SELF.md`、session 历史和主动流程状态保持不变。
- 缺失、空文件和非法 UTF-8 都 fail-loud，错误指向 `veda-reset`。
- migration 只创建缺失文件，revert 不删除用户后来修改的 Veda。
- `veda-reset` 对现有原始字节形成可校验恢复点，并且重复执行默认内容时不新增备份。
