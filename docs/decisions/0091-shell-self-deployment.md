# 0091 · Shell 提交自部署，宿主等待正常回合结束

- 状态：accepted
- 日期：2026-10-06
- 关联：RUN-004、RUN-013、SH-003、STA-002、MIG
- 范围：Docker 生产路径；直接 Python 的 Supervisor 入口保留。

## 决定与理由

用户要求不用新模型工具，让 Agent 通过 Shell 完成检查、合并和提交，然后正常回复并结束。
采用 `akashic-release submit --commit SHA` 和独立用户 systemd service；复用既有发布器。
Shell 自动携带原 Session、CallRef 与 boot，Agent 不查内部编号，也不等待自己重启。

systemd 只管理进程。Core 根据既有 Message、TurnProjection、FinalOutputDelivery 与
工作许可核对停止准备，不能用 session 数量、PID 消失或 sleep 代替正常结束证据。
Skill 规定提交后只回复，不保证模型绝不追加工具调用；追加调用会继续占活动许可。
回合失败、送达失败或排空超时不会触发发布，原准入恢复。

Docker 直接运行 Gateway，AppRuntime 接管 Web Shell 和 readiness。Supervisor 与
Guardian 的进程管理不再叠加 Docker/systemd；容器使用 init。应用启动失败期间 Web
可能不可用，宿主 CLI/journald 提供诊断，不新增常驻部署 daemon。

## owner 与失败

工具和消息仍由原 owner 追加。宿主请求按原调用去重，保存目标与任务状态，不减少旧记录。
发布器在同一 release.lock 下准备、预检、等待、停止、迁移、发布和验收。
正常关闭回执不等于部署成功；Core/Controller 回执、容器退出与 Bridge service
结果必须同时成立。清理或迁移失败保留维护现场，不能盲目回滚旧镜像。

停止范围仅限 Akashic Core、Host Bridge 与 Controller 持有的精确 Workload 租约。
不得停止宿主 user manager、桌面、Docker daemon、外部服务或无关容器。
机器级 E2E 使用独立内核虚拟机；验收同时观察无关服务持续存活。

备份由该次授权决定，默认关闭。共享 SQLite/plugin-data 维持单写入者，接受短暂维护窗口。
电源故障或外部强杀不能保证回合完成；缺失证据明确失败，不生成假交接。

操作流程由 [部署 Skill](../../plugins/standard_tools/skills/deploy-akashic/SKILL.md) 拥有；随默认 Shell 插件加载。
