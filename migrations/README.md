# Workspace migrations

`yoyo/` 是唯一自动执行的迁移目录。每个顶层 Python 文件使用 Yoyo migration ID
和 `__depends__` 声明依赖；已经进入主分支的文件只追加、不修改。

其余子目录是 Git cursor 系统留下的历史脚本，只供调查旧实现，不进入 Yoyo
catalog，也不再承诺自动执行或兼容当前 runtime。
