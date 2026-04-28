from __future__ import annotations

from typing import TYPE_CHECKING, cast

from agent.core.runtime_support import AgentLoopRunner, TurnRunResult
from agent.looping.ports import SessionServices
from bus.events import InboundMessage, OutboundMessage, SpawnCompletionItem

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.passive_turn import PassiveTurnPipeline
    from agent.tools.registry import ToolRegistry

async def process_spawn_completion_event(
    *,
    item: SpawnCompletionItem,
    key: str,
    session_svc: SessionServices,
    context: "ContextBuilder",
    pipeline: "PassiveTurnPipeline",
    tools: "ToolRegistry",
    memory_window: int,
    run_agent_loop_fn: AgentLoopRunner,
    dispatch_outbound: bool = True,
) -> OutboundMessage:
    # 1. 先读取 session 和内部事件，准备要给主模型的回传消息。
    session = session_svc.session_manager.get_or_create(key)
    event = item.event
    label = event.label or "后台任务"
    task = event.task.strip()
    status = (event.status or "incomplete").strip()
    result = event.result.strip()
    exit_reason = event.exit_reason.strip()
    retry_count = event.retry_count

    _EXIT_LABELS: dict[str, str] = {
        "completed": "正常完成",
        "max_iterations": "迭代预算耗尽（任务可能不完整）",
        "tool_loop": "工具调用循环截断（任务可能不完整）",
        "error": "执行出错",
        "forced_summary": "强制汇总（任务可能不完整）",
        "cancelled": "已取消",
    }
    exit_label = _EXIT_LABELS.get(exit_reason, exit_reason or "未知")

    if retry_count >= 1:
        guidance = (
            "⚠️ 已重试一次，不再重试。\n"
            "请直接将已获得的结果汇报给用户，说明已完成的部分和未完成的部分。"
        )
    else:
        guidance = (
            "**处理指引（按顺序判断，选其一执行）**\n"
            "1. 结果完整回答了原始任务 → 直接向用户汇报，不提及内部机制\n"
            "2. 退出原因是【迭代预算耗尽】或【工具调用循环截断】，且核心信息明显不足 → "
            "调用 spawn 重试；task 中说明上次卡在哪、这次从哪继续；"
            "run_in_background=true；同时简短告知用户正在补充\n"
            "3. 结果为空或明显出错 → 直接告知用户失败，询问是否需要重试\n"
            "重试只允许一次。"
        )

    current_message = (
        f"[后台任务回传]\n"
        f"任务标签: {label}\n"
        f"原始任务: {task or '（未提供）'}\n"
        f"退出原因: {exit_label}\n"
        f"执行结果:\n{result or '（无结果）'}\n\n"
        f"{guidance}\n\n"
        "禁止在回复中提及 subagent、spawn、job_id、内部事件等内部概念。\n"
        "必要时可读取结果里提到的文件来补充说明。"
    )

    # 2. 再调用主模型生成用户可见回复。
    tools.set_context(channel=item.channel, chat_id=item.chat_id)
    from agent.core.types import ContextRequest
    initial_messages = context.render(
        ContextRequest(
            history=session.get_history(max_messages=memory_window),
            current_message=current_message,
            channel=item.channel,
            chat_id=item.chat_id,
            message_timestamp=item.timestamp,
        )
    ).messages
    final_content, tools_used, tool_chain, _, _thinking = await run_agent_loop_fn(
        initial_messages,
        request_time=item.timestamp,
        preloaded_tools=None,
    )
    if final_content is None:
        if status == "completed":
            final_content = "后台任务已完成。"
        elif status == "incomplete":
            final_content = "后台任务未全部完成，部分工作尚未收尾。"
        elif status == "cancelled":
            final_content = "后台任务已取消。"
        else:
            final_content = "后台任务执行出错。"

    # 3. 走 AfterReasoning + dispatch 流程，经过插件链。
    marker = f"[后台任务完成] {label} ({status})"
    if exit_reason:
        marker += f" [{exit_reason}]"
    pseudo_msg = InboundMessage(
        channel=item.channel,
        sender="spawn",
        chat_id=item.chat_id,
        content=marker,
        timestamp=item.timestamp,
        media=[],
        metadata={"skip_post_memory": True},
    )
    parsed_tool_chain = cast(list[dict[str, object]], tool_chain)
    return await pipeline.post_reasoning(
        msg=pseudo_msg,
        session_key=key,
        turn_result=TurnRunResult(
            reply=final_content,
            tools_used=tools_used,
            tool_chain=parsed_tool_chain,
        ),
        dispatch_outbound=dispatch_outbound,
    )
