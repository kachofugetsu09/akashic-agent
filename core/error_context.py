from __future__ import annotations

from contextvars import ContextVar

# 当前正在处理的会话 key。在每条消息/每个 proactive tick 的处理 task 起点设置，
# observe 的全局错误采集器在 logging 钩子里读取它，给错误打上 session 归属。
# 放在 core 层是为了让主循环与 observe 插件共享同一个 ContextVar 对象。
current_session_key: ContextVar[str | None] = ContextVar(
    "akashic_current_session_key", default=None
)

# 当前 turn 的客户端消息标识（mobile message.send 的 client_message_id），
# 用于跨端 turn 时间链在 provider/context 里程碑处保持同一关联身份。
current_client_message_id: ContextVar[str] = ContextVar(
    "akashic_current_client_message_id", default=""
)

# 一次逻辑 provider 调用的中性 telemetry 身份（唯一 neutral owner 定义在这里）。
# 高层（passive_turn 的 _call_provider / _call_compaction_summary）在逻辑调用
# 起点 set、finally 精确 reset；底层（ChatCompletions transport/http/raw、
# nonstream 总 span）只读，保证同一逻辑调用的所有里程碑按 provider_call_id join。
# provider_attempt=1/2 对应业务 provider retry；provider_operation 区分
# business 与 compaction_summary。默认值不可被误当成真实身份。
current_provider_call_id: ContextVar[str] = ContextVar(
    "akashic_current_provider_call_id", default=""
)
current_provider_attempt: ContextVar[int] = ContextVar(
    "akashic_current_provider_attempt", default=0
)
current_provider_operation: ContextVar[str] = ContextVar(
    "akashic_current_provider_operation", default=""
)
