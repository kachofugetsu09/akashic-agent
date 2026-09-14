# Prompt

本包提供人格、行为规则和输入时间材料，依赖 `context.materials.v3`。
消费者通过 Context 的授权配置选择材料 provider；Core 不预置这个选择。

正式安装后，首次配置必须先运行通用 setup 向导。它发现并执行已安装 stable 制品根目录的
`configure.py`，只在 workspace 缺失时创建 `memory/VEDA.md`；已有合法内容保持原始字节：

```sh
python main.py setup --config /path/to/config.toml --workspace /path/to/workspace
```

容器发行入口也使用同一命令：先让 distribution entrypoint 完成 profile 安装，再以 `setup` 作为
容器命令运行一次向导，然后启动 `supervise`。纯安装不会猜测或写入 VEDA，未完成 setup 时首个
Prompt 读取会明确报告缺失。

显式恢复使用安装产物中的命令：

```sh
python /path/to/installed/prompt/persona.py --workspace /path/to/workspace
```

命令将默认模板写入 `memory/VEDA.md`。已有内容先保存原始字节备份，输出 JSON
包含目标、备份路径与两个 SHA-256；内容已经相同时不重写。运行前核对 workspace
和输出中的恢复点。模板随本包发布，可以由外部包替换，Core 无须修改。

正常 Prompt 读取不会创建、重置或修复缺失、空白或非 UTF-8 的人格文件。
更新插件不会覆盖已有 VEDA；恢复后的内容从下次材料组装生效。
