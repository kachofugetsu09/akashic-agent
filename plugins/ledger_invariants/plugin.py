"""提交前检查消息顺序与 Turn 前缀，不保存第二份账本。"""
from agent.plugin_composition import Context
from plugins.ledger.contract import APPEND_CHECKS, Control, Message, MessageConflict, MessageReader

api_version = 3
name = "ledger_invariants"
version = "1.0.0"
desc = "Ledger 追加顺序与 Turn 关闭前缀检查"
inject = (APPEND_CHECKS,)


def check_append(message: Message, reader: MessageReader) -> None:
    """以同一事务内的既有前缀为依据；暂停和恢复不构成 Turn 关闭。"""
    head = reader.head()
    if message.seq <= head:
        raise MessageConflict("追加 seq 必须严格递增")
    body = message.body
    if not isinstance(body, Control):
        return
    if body.action == "abandon":
        finished = reader.latest_finished_output_seq(message.source, after_seq=-1, through_seq=head)
        closed = reader.scan_controls(
            lambda rows: max((control.through_seq for _, control in rows
                              if control.action == "abandon"), default=-1),
            source=message.source, after_seq=-1 if finished is None else finished, through_seq=head,
        )
        if body.through_seq <= max(-1 if finished is None else finished, closed):
            raise MessageConflict("abandon 不能重新关闭已经结束的前缀")


async def apply(ctx: Context) -> None:
    await ctx.require(APPEND_CHECKS).register(ctx, check_append)
