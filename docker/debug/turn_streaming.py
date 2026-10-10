"""用完整投影作 oracle，核对分页、闭段恢复和正文引用寿命。"""
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import weakref

from plugins.ledger.contract import CallRef, ContentPart, Control, Input, Message, Output, ToolCall, ToolResult
from plugins.turn_projection.plugin import TurnProjection
from plugins.ledger.log import MessageLog
from plugins.ledger.contract import ContentReferences


def check() -> None:
    """跨页 abandon 和迟到结果不串段；扫描只保留有界正文。"""
    projection = TurnProjection()
    now = datetime.now(UTC)
    bodies = [
        Input(()), Output((ToolCall('tool', {}),), 'continue'), Input(()),
        Control('abandon', 1), ToolResult(CallRef('m1', 0), 'success', ()),
        Output((), 'complete'), Input(()), Output((), 'quiet'), Input(()),
    ]
    messages = tuple(Message(f'm{i}', 's', i, now, 'author', 'source', body)
                     for i, body in enumerate(bodies))
    turns = projection.project(messages, 'source')
    assert projection.project(iter(messages), 'source', include_closed=False) == (turns[-1],)
    assert [turn.status for turn in turns] == ['abandoned', 'complete', 'quiet', 'open']
    assert turns[1].message_ids == ('m2', 'm5') and not turns[1].observations
    for closed in turns[:-1]:
        tail = (m for m in messages if m.seq > closed.through_seq)
        assert projection.project(tail, 'source', after_seq=closed.through_seq) == tuple(
            t for t in turns if t.through_seq > closed.through_seq)

    refs: list[weakref.ReferenceType[Message]] = []
    def large_stream():
        for index in range(1024):
            message = Message(f'b{index}', 's', index, now, 'author', 'source',
                              Input((ContentPart('text', str(index) + 'x' * 65536),)))
            refs.append(weakref.ref(message))
            yield message
            # 投影可以保存全部 ID，但不能保存此前正文。
            assert sum(ref() is not None for ref in refs) <= 2
    pending = projection.project(large_stream(), 'source')
    assert len(pending[0].message_ids) == 1024
    assert not any(ref() is not None for ref in refs)

    with TemporaryDirectory() as directory:
        log = MessageLog(Path(directory) / 'sessions.db')
        try:
            writer = log.writer('s', author='user', source='source', body_types=(Input,),
                                content={'text': lambda _: ContentReferences()})
            for index in range(130):
                writer.append(f'i{index}', Input((ContentPart('text', str(index)),)))
            reader = log.reader('s')
            assert reader.scan(lambda rows: projection.project(rows, 'source')) == projection.project(reader.snapshot(), 'source')
            escaped = reader.scan(lambda rows: rows)
            assert list(escaped) == []
            assert len(reader.snapshot(after_seq=63, through_seq=100)) == 37
            print('PASS: tail replay, late results, body release, real SQLite pages')
        finally:
            log.close()


if __name__ == '__main__':
    check()
