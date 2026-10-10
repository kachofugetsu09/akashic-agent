"""旧制品生成向量，新 Yoyo 复制并核对源库完整性。"""
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv) != 2:
    raise SystemExit("usage: ledger_derived_migration_scenario.py <old-core-checkout>")
BEFORE = Path(sys.argv[1]).resolve()
from agent.migrations.runner import MigrationRunner
with tempfile.TemporaryDirectory(prefix='akashic-derived-migration-') as directory:
    root=Path(directory)
    sources=root/'plugins'
    shutil.copytree(ROOT/'plugins/ledger', sources/'ledger',ignore=shutil.ignore_patterns('__pycache__'))
    workspace=root/'workspace'
    runner=MigrationRunner(repo_root=ROOT, config_path=root/'config.toml',workspace=workspace,plugin_dirs=[sources])
    # 旧制品先初始化账本，再写入实际旧格式消息与向量。
    seed='''from pathlib import Path
from agent.migrations.runner import MigrationRunner
from plugins.ledger.log import MessageLog, SessionAttributes
from plugins.ledger.embedding_store import MessageEmbeddings, MessageEmbeddingStore
from plugins.ledger.contract import Input,ContentPart,ContentReferences
import sys, shutil
workspace=Path(sys.argv[1]); config=Path(sys.argv[2]); repo=Path.cwd()
old_sources=workspace.parent/'old-plugins'
shutil.copytree(repo/'plugins/ledger',old_sources/'ledger')
MigrationRunner(repo_root=repo,config_path=config,workspace=workspace,plugin_dirs=[old_sources]).run()
log=MessageLog(workspace/'sessions.db'); log.ensure_session('s',SessionAttributes())
writer=log.writer('s',author='user',source='scenario',body_types=(Input,),content={'text':lambda part:ContentReferences()})
message=writer.append('saved',Input((ContentPart('text','saved vector'),)))
records=MessageEmbeddings(log).bind(lambda message: 'saved vector')
records.save(message,model='fixed',embedding=[1.0,0.0])
old=MessageEmbeddingStore(workspace/'sessions.db')
old._db.execute("INSERT INTO message_embedding_migrations VALUES ('seed','2026-10-11',1)")
old._db.commit(); old.close(); log.close()
'''
    env=dict(os.environ, AKASHIC_PLUGIN_HOME=str(root/'home'),AKASHIC_PLUGIN_DISTRIBUTION='')
    subprocess.run([sys.executable,'-c',seed,str(workspace),str(root/'config.toml')],cwd=BEFORE,env=env,check=True)
    path=workspace/'sessions.db'
    before=path.read_bytes()
    with sqlite3.connect(path) as db:
        rows={table:db.execute(f'SELECT * FROM {table}').fetchall() for table in ('messages','message_embeddings','message_embedding_migrations')}
    result=runner.run()
    assert result.migrations == ('20261011_01_derived_storage',),result
    assert runner.run().state=='current'
    assert path.read_bytes()==before
    with sqlite3.connect(workspace/'sessions-derived.db') as db:
        for table in ('message_embeddings','message_embedding_migrations'):
            assert db.execute(f'SELECT * FROM {table}').fetchall()==rows[table]
    from plugins.ledger.log import MessageLog
    from plugins.ledger.embedding_store import MessageEmbeddings
    log=MessageLog(path)
    store=MessageEmbeddings(log,workspace/'sessions-derived.db')
    message=log.reader('s').get('saved')
    assert store.bind(lambda message:'saved vector').read(message,model='fixed',dimension=2)==(1.0,0.0)
    store.close(); log.close()
    assert path.read_bytes()==before
    (workspace/'sessions-derived.db').unlink()
    log=MessageLog(path); store=MessageEmbeddings(log,workspace/'sessions-derived.db')
    assert store.bind(lambda message:'saved vector').read(message,model='fixed',dimension=2) is None
    store.bind(lambda message:'saved vector').save(message,model='fixed',embedding=[1.,0.])
    store.close(); log.close()
    assert path.read_bytes()==before
    print('old-real-data; all-copied-rows; repeat-current; derived-loss-cache-miss; source-bytes-unchanged')
