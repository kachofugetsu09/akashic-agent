"""真实消息库验证大前缀不被增量 reader 留存，小前缀仍可复用。"""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import weakref
import sqlite3

from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input


async def check(path: Path) -> None:
    log = MessageLog(path)
    try:
        def writer(session: str):
            return log.writer(session, author='user', source='source', body_types=(Input,),
                              content={'text': lambda _: ContentReferences()})
        small = writer('small')
        small.append('s0', Input(()))
        reader = log.reader('small').incremental()
        first = reader.snapshot()
        small.append('s1', Input(()))
        assert reader.snapshot()[0] is first[0]
        assert reader.snapshot(through_seq=0) == first

        large = writer('large')
        large.append('large0', Input((ContentPart('text', 'x' * (5 * 1024 * 1024)),)))
        reader = log.reader('large').incremental()
        for read in (lambda: reader.snapshot(),):
            messages = read()
            reference = weakref.ref(messages[0])
            del messages
            assert reference() is None, '大正文被 reader 留存'
        messages = await reader.snapshot_async(through_seq=0)
        reference = weakref.ref(messages[0])
        del messages
        # 让已完成的异步 worker 释放自身结果引用。
        await asyncio.sleep(0)
        assert reference() is None, '异步 reader 留存大正文'
        large.append('large1', Input(()))
        assert [m.message_id for m in reader.snapshot(after_seq=0)] == ['large1']
        assert len(reader.snapshot(through_seq=0)) == 1
        print('PASS: small reuse, large release, async release, fixed prefix and tail')
    finally:
        log.close()


if __name__ == '__main__':
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'sessions.db'
        asyncio.run(check(path))
        # 已发布且仍允许打开的无 metadata 列 schema 仍可读取。
        with sqlite3.connect(path) as connection:
            connection.execute('ALTER TABLE messages DROP COLUMN metadata')
        log = MessageLog(path)
        try:
            reader = log.reader('small').incremental()
            assert len(reader.snapshot()) == 2
            assert len(asyncio.run(reader.snapshot_async(through_seq=1))) == 2
            print('PASS: existing schema without metadata stays readable')
        finally:
            log.close()
