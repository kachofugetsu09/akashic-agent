"""用真实 SQLite、第二写入者和事务验证固定消息读面；只写临时目录。"""
import asyncio
import gc
import hashlib
import json
import sqlite3
import threading
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory

from session.log import MessageLog, OwnerTransaction
from session.message import ContentPart, ContentReferences, Input
from session.message_codec import encode_body


def body(text: str) -> Input:
    return Input((ContentPart('text', text),))


def writer(log: MessageLog):
    return log.writer('session', author='user', source='source', body_types=(Input,),
                      content={'text': lambda _: ContentReferences()})


def digest(path: Path) -> str:
    with sqlite3.connect(path) as db:
        rows = db.execute('SELECT * FROM messages ORDER BY seq').fetchall()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


async def exercise(path: Path) -> None:
    """逐次提交、竞争写入、回滚和取消后，始终与真实 SQL 完整消息对照。"""
    log = MessageLog(path)
    other = MessageLog(path)
    read = log.reader('session')
    append = writer(log)
    outside = writer(other)
    try:
        # 1. 多个读者共享已提交前缀，历史快照不随连续提交增长。
        append.append('first', body('first'))
        first = read.committed_snapshot()
        before = digest(path)
        assert tuple(first) == read.snapshot()
        assert digest(path) == before
        outside.append('middle', body('middle'))
        second = log.reader('session').committed_snapshot()
        assert second.extends(first) and len(first) == 1
        append.append('last', body('last'))
        third = await read.committed_snapshot_async()
        assert third.extends(second) and len(second) == 2
        assert tuple(third) == read.snapshot()
        short = read.committed_snapshot(through_seq=first.through_seq)
        assert tuple(short) == tuple(first) and not short.extends(third)

        # 2. 外部中段改写/删除换前缀；已经签发的视图仍保持原内容。
        before_middle = third[1]
        with sqlite3.connect(path) as db:
            db.execute('UPDATE messages SET body=? WHERE id=?', (encode_body(body('changed')), 'middle'))
        changed = read.committed_snapshot()
        assert not changed.extends(third)
        assert third[1] is before_middle and third[1].body == body('middle')
        assert changed[1].body == body('changed')
        assert tuple(changed) == read.snapshot()
        with sqlite3.connect(path) as db:
            db.execute('DELETE FROM messages WHERE id=?', ('middle',))
        deleted = read.committed_snapshot()
        assert not deleted.extends(changed) and tuple(deleted) == read.snapshot()

        # 3. pinned RO 与未提交写入只能读取自己的 SQL 视图，不能污染共享读面。
        with read.read_snapshot():
            pinned = read.committed_snapshot()
            outside.append('external', body('external'))
            assert tuple(read.committed_snapshot()) == tuple(pinned)
        current = read.committed_snapshot()
        assert len(current) == len(pinned) + 1
        assert not current.extends(pinned)
        owner = log.owner('snapshot-check')
        before = digest(path)
        def rollback(transaction: OwnerTransaction) -> None:
            transaction.append(append, 'rollback', body('rollback'))
            local = read.committed_snapshot()
            assert local[-1].message_id == 'rollback'
            assert not local.extends(current)
            raise ValueError('requested rollback')
        try:
            owner.transact(rollback)
        except ValueError as error:
            assert str(error) == 'requested rollback'
        else:
            raise AssertionError('transaction did not roll back')
        assert digest(path) == before
        assert tuple(read.committed_snapshot()) == read.snapshot()

        # 4. 确定性阻塞提交，取消等待者后排空写入；不丢已成立的事实。
        entered, release = threading.Event(), threading.Event()
        def pending(transaction: OwnerTransaction) -> None:
            transaction.append(append, 'cancelled-waiter', body('committed'))
            entered.set()
            if not release.wait(10):
                raise TimeoutError('commit gate was not released')
        task = asyncio.create_task(owner.transact_async(pending))
        await asyncio.to_thread(entered.wait)
        task.cancel()
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('caller cancellation was lost')
        final = read.committed_snapshot()
        assert final[-1].message_id == 'cancelled-waiter'
        assert tuple(final) == read.snapshot()
        before_restart = digest(path)
    finally:
        other.close()
        log.close()
    reopened = MessageLog(path)
    try:
        assert tuple(reopened.reader('session').committed_snapshot()) == tuple(final)
        assert digest(path) == before_restart
        # 5. 大消息随最后一个消费者释放，不由日志永久保留。
        large = reopened.writer('large', author='user', source='source', body_types=(Input,),
                                content={'text': lambda _: ContentReferences()})
        large.append('large', body('x' * 262144))
        snapshot = reopened.reader('large').committed_snapshot()
        ref = weakref.ref(snapshot[0])
        del snapshot
        gc.collect()
        assert ref() is None
    finally:
        reopened.close()
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    print('PASS: append, external writes, fixed prefix, pinned RO, rollback, cancellation, restart, release, integrity')


if __name__ == '__main__':
    with TemporaryDirectory(prefix='akashic-message-snapshot-') as directory:
        asyncio.run(exercise(Path(directory) / 'sessions.db'))
