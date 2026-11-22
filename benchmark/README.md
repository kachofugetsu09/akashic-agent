# Benchmark

该目录保存隔离评估 harness。其可执行文件由 change-impact catalog 的
`evaluation` capability group 负责。

`computer_runtime`、`computer_driver` 依赖已迁到 extra 仓库的 `computer`（以及 `markdown_memory`）。它们只能在 extra 仓库用 `python scripts/extra_ci.py compose --core <Core 检出> --synth <目录>` 合成出的目录里运行：合成目录是 Core 源码加上 extra 的 `plugins/`、`frontend/plugins/`。
