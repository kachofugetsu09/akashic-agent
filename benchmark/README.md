# Benchmark

该目录保存隔离评估 harness。其可执行文件由 change-impact catalog 的
`evaluation` capability group 负责。

`computer_runtime`（装配 `computer` 插件，提示词来源还点名 `markdown_memory`）与 `computer_driver` 的 `--agent` 模式（读取 `computer` 的技能文件）依赖已迁到 extra 仓库的插件，只能在 extra 仓库用 `python scripts/extra_ci.py compose --core <Core 检出> --synth <目录>` 合成出的目录里运行：合成目录是 Core 源码加上 extra 的 `plugins/`、`frontend/plugins/`。`computer_driver` 的非 Agent 模式不受影响。
