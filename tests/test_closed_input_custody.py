"""已关闭输入 owner 不能用空状态伪装恢复或结算成功。"""
import pytest

from bus.queue import MessageBus


@pytest.mark.asyncio
async def test_closed_bus_rejects_late_custody_operations():
    bus = MessageBus()
    await bus.aclose()
    with pytest.raises(RuntimeError, match="message bus 已关闭"):
        await bus.defer_durable_inbound("original-handoff")
    with pytest.raises(RuntimeError, match="message bus 已关闭"):
        await bus.recover_durable_inbounds()
    with pytest.raises(RuntimeError, match="message bus 已关闭"):
        await bus.settle_rejected_inbound(
            channel="probe", session_key="original-session", provider_message_id="original-message",
        )
