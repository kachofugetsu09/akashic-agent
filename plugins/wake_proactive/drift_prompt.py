from __future__ import annotations

from plugins.wake_proactive.drift_drive import DriftDriveResult


def build_drift_messages(
    *,
    memory_text: str,
    proactive_context: str,
    recent_session: str,
    drive: DriftDriveResult,
) -> list[dict[str, str]]:
    system = (
        "现在是一次低信息价值窗口中的轻量主动交流判断。"
        "你只能调用一次 share_drift 或 idle_drift。"
        "只有当固定记忆、主动规则和截至当前时间的近期对话共同支持一个自然话题时才发送；"
        "不要编造新事实，不要假装用户刚表达过未出现的内容，不要汇总资讯，也不要强行提问。"
        "消息应像熟悉用户的人自然想起一件相关的事，简短、具体、允许不发送。"
        f"\n当前 leisure 指标：idle_hours={drive.idle_hours:.2f}, "
        f"rate={drive.rate:.4f}, repetition={drive.repetition_suppression:.3f}。"
    )
    user = (
        f"【固定 MEMORY.md】\n{memory_text}\n\n"
        f"【固定 PROACTIVE_CONTEXT.md】\n{proactive_context}\n\n"
        f"【截至当前时间的最近对话】\n{recent_session}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
