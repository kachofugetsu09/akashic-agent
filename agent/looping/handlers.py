from __future__ import annotations

from typing import TYPE_CHECKING

from bus.events import InboundMessage, OutboundMessage, SpawnCompletionItem

if TYPE_CHECKING:
    from agent.core.passive_turn import PassiveTurnPipeline

async def process_spawn_completion_event(
    *,
    item: SpawnCompletionItem,
    key: str,
    pipeline: "PassiveTurnPipeline",
    dispatch_outbound: bool = True,
) -> OutboundMessage:
    # 1. 先读取内部事件，准备要给主模型的回传消息。
    event = item.event
    label = event.label or "后台任务"
    task = event.task.strip()
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

    # 2. 复用被动 session-aware 主链；它负责 prompt、compaction gate、持久化和 dispatch。
    pseudo_msg = InboundMessage(
        channel=item.channel,
        sender="spawn",
        chat_id=item.chat_id,
        content=current_message,
        timestamp=item.timestamp,
        media=[],
        metadata={
            "skip_post_memory": True,
            "omit_user_turn": True,
            "skip_memory_retrieval": True,
        },
    )
    return await pipeline.run(
        pseudo_msg,
        key,
        dispatch_outbound=dispatch_outbound,
    )
