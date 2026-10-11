---
name: codex-delegate
description: 把长代码库任务委托给本机 Codex CLI 后台执行，完成时自动通知当前会话，并可续聊同一个 Codex 会话。当用户说用 codex skill、codexskill、codex delegate、委托 codex、后台 codex、让 codex 去做、继续那个 codex 会话时使用。
metadata: {"akashic": {"always": false, "requires": {"bins": ["codex"]}}}
---

# Codex Delegate

## 目标

用后台 `shell` 直接启动 `codex exec`。启动后本轮可以继续和用户对话或结束；Codex 结束时，
运行时会把完成通知送回当前会话（回合进行中注入下一步，空闲时唤醒一轮）。不要再包一层 subagent，
也不要用 `write_stdin` 长等待或轮询。

Codex 以 bypass 模式运行：没有审批和沙箱，能以当前用户权限执行任何命令、修改任何文件。
只委托用户明确要求交给 Codex 的任务。

## 流程

```
新任务
├─ shell(command -v codex && codex --version && echo $HOME)
├─ write_file(<HOME>/.akashic/codex-runs/<run>/prompt-1.txt)
├─ shell(run_in_background=true, cwd=<repo>)      → 返回 execution_id，本轮可继续
│   codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
│     --json -o <dir>/result-1.md - < <dir>/prompt-1.txt > <dir>/events-1.jsonl
└─ 收到「后台任务 shell-N 已结束」通知
    ├─ read_file(<dir>/result-1.md)               → 向用户汇报
    └─ shell(head -1 <dir>/events-1.jsonl)        → 记下 thread_id，写入 <dir>/session.txt

续聊同一会话
├─ read_file(<dir>/session.txt)
├─ write_file(<dir>/prompt-2.txt)
└─ shell(run_in_background=true, cwd=<repo>)
    codex exec resume <thread_id> --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
      --json -o <dir>/result-2.md - < <dir>/prompt-2.txt > <dir>/events-2.jsonl
```

## 使用规则

1. 先检查 `codex` 是否可用并取得家目录：`shell(command="command -v codex && codex --version && echo $HOME")`。失败就告诉用户本机没有 Codex CLI，不要继续。
2. 每个委托任务用一个运行目录 `<HOME>/.akashic/codex-runs/<run>/`（`<HOME>` 是上一步输出的绝对路径；`read_file`、`write_file` 不展开 `~`），`<run>` 取简短英文任务名加日期，例如 `refactor-auth-20261011`。prompt、结果、事件流和 session id 都放在这里，不写 `/tmp`。
3. 任务说明写进 `prompt-<n>.txt`，再用 `- < prompt-<n>.txt` 从 stdin 读入；不要把任务说明拼进命令行，避免引号和换行破坏 prompt。
4. 启动命令必须同时满足：
   - `run_in_background=true`，`cwd` 设为用户给出的 repo 路径；
   - 带 `--dangerously-bypass-approvals-and-sandbox` 和 `--skip-git-repo-check`；
   - 带 `--json` 并把 stdout 重定向到 `events-<n>.jsonl`；
   - 带 `-o <dir>/result-<n>.md`，结果以这个文件为准；
   - 不加 `--ephemeral`，否则会话不会保存，无法续聊；
   - 不加 `&`、`nohup`、`disown`、`| tee`，后台化交给 `run_in_background`。
5. 启动后告诉用户已派发、运行目录在哪，然后结束本轮或继续处理用户的其他话题。不要调用 `write_stdin` 等待，不要轮询结果文件。
6. 收到完成通知后：
   - 读 `result-<n>.md` 向用户汇报；status 不是 completed 或结果文件为空时，用 `shell(tail -20 <dir>/events-<n>.jsonl)` 查看失败原因；
   - 首次运行完成后，`events-1.jsonl` 第一行是 `{"type":"thread.started","thread_id":"..."}`，把 thread_id 写入 `session.txt`，并在回复里告诉用户。
7. 续聊用 `codex exec resume <thread_id> ... -`。resume 不支持 `--cd` 和 `-s`，工作目录用 shell 的 `cwd` 指定，与首次保持同一个 repo。
8. 同一个 Codex 会话同时只能有一个运行：上一轮还没收到完成通知时不要再 resume；用户要改方向就先 `task_stop(execution_id=N)`，再发新的 resume。
9. 自动续聊必须传明确的 thread_id，不用 `--last`，避免串到别的 Codex 会话。
10. 用户已经给出 repo 路径时，不要先自己探索这个仓库；prompt 里只写用户目标、输出要求和 repo 路径，让 Codex 自行发现入口和相关文件。不要塞进自己猜的文件列表，除非文件来自用户原文。
11. 默认硬超时是 4 小时（shell 的 `timeout`）；只有用户要求更短时才显式传入。

## 推荐调用形态

```text
shell(command="command -v codex && codex --version && echo $HOME", description="检查 codex")

write_file(
  path="/home/me/.akashic/codex-runs/refactor-auth-20261011/prompt-1.txt",
  content="仓库 /path/to/repo。目标：<用户目标>。请自行在整个仓库中定位相关实现。完成后用简短中文总结改了什么、怎么验证。"
)

shell(
  command="D=/home/me/.akashic/codex-runs/refactor-auth-20261011; codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --json -o $D/result-1.md - < $D/prompt-1.txt > $D/events-1.jsonl",
  description="后台 codex",
  cwd="/path/to/repo",
  run_in_background=true
)
```

完成通知到达后：

```text
read_file(path="/home/me/.akashic/codex-runs/refactor-auth-20261011/result-1.md")
shell(command="head -1 /home/me/.akashic/codex-runs/refactor-auth-20261011/events-1.jsonl", description="取 codex 会话 id")
write_file(path="/home/me/.akashic/codex-runs/refactor-auth-20261011/session.txt", content="<thread_id>")
```

续聊：

```text
shell(
  command="D=/home/me/.akashic/codex-runs/refactor-auth-20261011; codex exec resume <thread_id> --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --json -o $D/result-2.md - < $D/prompt-2.txt > $D/events-2.jsonl",
  description="续聊 codex",
  cwd="/path/to/repo",
  run_in_background=true
)
```

## 注意

- 收到通知前不要把 Codex 的结果当成已知事实向用户汇报。
- `events-*.jsonl` 里可能有 Codex 的配置告警（`"type":"error"` 条目），不代表任务失败；以退出状态和结果文件为准。
- 没有启用 jobs 插件时，shell 会提示后台完成不会通知；这时告诉用户需要手动让你去读结果文件。
