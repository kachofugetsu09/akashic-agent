from __future__ import annotations

from contextvars import ContextVar

# 当前正在处理的会话 key。在每条消息/每个 proactive tick 的处理 task 起点设置，
# observe 的全局错误采集器在 logging 钩子里读取它，给错误打上 session 归属。
# 放在 core 层是为了让主循环与 observe 插件共享同一个 ContextVar 对象。
current_session_key: ContextVar[str | None] = ContextVar(
    "akashic_current_session_key", default=None
)
