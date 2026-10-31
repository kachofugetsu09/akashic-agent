"""新建真实附件 Runtime；不接模型、不写已有 workspace。运行前须取得性能锁。"""
import argparse
import asyncio
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'sdk/python/src')]
SESSION = 'akashic:attachment-benchmark'
PREVIEW = 512 * 1024


def digest(workspace):
    """只读核对正文、附件引用、导入账和完整制品字节。"""
    with sqlite3.connect(f'file:{workspace / "sessions.db"}?mode=ro', uri=True) as db:
        tables = {name: db.execute(f'SELECT * FROM {name} ORDER BY {key}').fetchall()
                  for name, key in [('messages', 'session_key,seq'), ('attachments', 'artifact_id'),
                                    ('attachment_imports', 'artifact_id'), ('message_attachments', 'message_id,ordinal')]}
    files = {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((workspace / 'uploads/artifacts').glob('*.bin'))}
    return {'tables': hashlib.sha256(json.dumps(tables, ensure_ascii=False).encode()).hexdigest(), 'files': files}


def samples():
    """用生成时已知的内容建立独立 oracle；不导入或执行前端算法。"""
    lines = [f'line {i:04d}: immutable attachment 中文正文' for i in range(600)]
    yield 'medium.txt', '\n'.join(lines) + '\n', lines, False, None, 'performance'
    records = [['id', 'title', 'note']] + [[str(i), f'item {i}', 'quoted, field'] for i in range(300)]
    out = io.StringIO(newline=''); csv.writer(out, lineterminator='\n').writerows(records)
    yield 'medium.csv', out.getvalue(), out.getvalue().splitlines(), False, {'columns': records[0], 'rows': 300}, 'performance'
    yield 'empty.txt', '', [], False, None, 'correctness'
    yield 'trailing.txt', 'one\ntwo\n', ['one', 'two'], False, None, 'correctness'
    yield 'no-trailing.txt', 'one\ntwo', ['one', 'two'], False, None, 'correctness'
    for name, delimiter in [('quoted.csv', ','), ('quoted.tsv', '\t')]:
        out = io.StringIO(newline='')
        csv.writer(out, delimiter=delimiter, lineterminator='\n').writerows([
            ['编号', '说明'], ['1', f'包含{delimiter}分隔符'], ['2', '双"引号\n换行']])
        yield name, out.getvalue(), out.getvalue().splitlines(), False, {'columns': ['编号', '说明'], 'rows': 2}, 'correctness'
    for count in (4999, 5000, 5001):
        rows = [f'row {i}' for i in range(count)]
        yield f'lines-{count}.txt', '\n'.join(rows), rows[:5000], count > 5000, None, 'correctness'
    for boundary in (PREVIEW, 2 * 1024 * 1024, 8 * 1024 * 1024):
        for delta in (-1, 0, 1):
            size = boundary + delta
            yield f'bytes-{size}.txt', 'x' * size, ['x' * min(size, PREVIEW)], size > PREVIEW, None, 'correctness'
    # 512KB 边界在四字节字符内部；第二行被截断，预览只保留完整首行。
    text = 'first\n' + 'a' * (PREVIEW - 6 - 1) + '😀' + '\nlast'
    yield 'utf8-boundary.txt', text, ['first'], True, None, 'correctness'


async def run(state, port, auth_enabled=False):
    """沿现有启动 helper 的隔离流程运行真实插件、附件 API 和 WebSocket。"""
    # 1. 只创建新 state；静态资产和所有运行状态都归本次实例。
    state.mkdir(parents=True, exist_ok=False)
    os.chmod(state, 0o700)
    plugins = state / 'plugins'
    shutil.copytree(ROOT / 'plugins', plugins, ignore=shutil.ignore_patterns('__pycache__', 'static'))
    for plugin, name, entry in [('ui', 'dashboard', 'index.html'), ('akashic_clients', 'chat', 'chat-app.js')]:
        assets = ROOT / 'static' / name
        if not (assets / entry).is_file():
            raise RuntimeError('先构建附件 worktree；fixture 不负责构建')
        destination = plugins / plugin / 'static'
        destination.mkdir()
        shutil.copytree(assets, destination / name)
        for asset in (destination / name).rglob('*'):
            if asset.is_file():
                os.utime(asset, (1767225600, 1767225600))
    for key in tuple(os.environ):
        if key.startswith('AKASHIC_'):
            del os.environ[key]
    os.environ.update(HOME=str(state / 'home'), XDG_CONFIG_HOME=str(state / 'home/.config'),
                      AKASHIC_PLUGIN_HOME=str(state / 'plugin-home'), AKASHIC_CORE_ROOT=str(ROOT),
                      AKASHIC_EXTRA_PLUGIN_DIRS=str(plugins))
    from bootstrap.init_workspace import init_workspace
    from bootstrap.app import AppRuntime
    from bootstrap.web_shell import create_web_shell_app
    from agent.config_models import Config
    from agent.plugins.bundles import set_plugin_choice
    from plugins.ledger.contract import AttachmentKind, ContentPart, Input, SessionAttributes
    from plugins.ledger.log import MessageLog
    from plugins.ledger.artifact_store import ArtifactStore
    from plugins.ledger.attachments import ChannelAttachmentArtifactStore
    from plugins.content.plugin import check_text, check_artifact
    import uvicorn
    workspace, config = state / 'workspace', state / 'config.toml'
    init_workspace(config_path=config, workspace=workspace)
    set_plugin_choice(workspace, 'computer', enabled=False)
    set_plugin_choice(workspace, 'web-auth', enabled=auth_enabled)
    if auth_enabled:
        # 用真实 owner 为本次新 workspace 生成初始凭据；不让启动日志打印密码。
        from plugins.web_auth.state import OwnerAuth
        OwnerAuth(workspace / 'web-auth').ensure()
    marker = dict(kind='akashic-attachment-oneoff', state=str(state), workspace=str(workspace),
                  chat=str(plugins / 'akashic_clients/static/chat'), pid=os.getpid(), port=port,
                  root=str(ROOT), authEnabled=auth_enabled)
    (state / '.akashic-oneoff-fixture.json').write_text(json.dumps(marker))
    # 2. 正式 ArtifactStore import 与 MessageLog append；不手写 SQL、不伪造响应。
    log = MessageLog(workspace / 'sessions.db')
    metadata = ArtifactStore(workspace / 'sessions.db')
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
    (state / 'originals').mkdir()
    manifest = []
    try:
        log.ensure_session(SESSION, SessionAttributes(learning='excluded'))
        writer = log.writer(SESSION, author='user', source='benchmark', body_types=(Input,),
                            content={'text': check_text, 'artifact_ref': check_artifact})
        for index, (name, source, preview, truncated, table, group) in enumerate(samples()):
            data = source.encode('utf-8')
            ref = await artifacts.import_bytes(data, kind=AttachmentKind.FILE, filename=name,
                                               media_type='text/plain')
            mid = f'attachment-{index:03d}'
            writer.append(mid, Input((ContentPart('text', f'附件验收 {name}'), ContentPart('artifact_ref', ref.artifact_id))))
            (state / 'originals' / name).write_bytes(data)
            # splitlines 仅用于这些无 CR 的原文；与前端缓存或构建产物无关。
            manifest.append(dict(name=name, id=ref.artifact_id, messageId=mid, group=group,
                                 bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), preview=preview,
                                 truncated=truncated, lines=len(source.splitlines()) if len(data) <= 2 * 1024 * 1024 else None,
                                 table=table, copyable=len(data) <= 8 * 1024 * 1024))
    finally:
        metadata.close(); log.close()
    (state / 'attachment-oracle.json').write_text(json.dumps(dict(session=SESSION, samples=manifest), ensure_ascii=False))
    before = digest(workspace)
    (state / 'attachment-integrity-before.json').write_text(json.dumps(before))
    # 3. 正常 Runtime 启停；关闭时证明浏览器只读过程没有改写持久事实。
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    loop = asyncio.get_running_loop()
    try:
        await app.start()
        app.plugin_watcher.stop()
        await app.plugin_watcher_task
        app.plugin_watcher = None
        print(json.dumps({'origin': f'http://127.0.0.1:{port}', 'samples': len(manifest)}), flush=True)
        server = uvicorn.Server(uvicorn.Config(create_web_shell_app(workspace), host='127.0.0.1', port=port, log_level='warning'))
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, setattr, server, 'should_exit', True)
        await server.serve()
    finally:
        try:
            await app.shutdown()
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
            after = digest(workspace)
            (state / 'attachment-integrity-after.json').write_text(json.dumps(after))
            if before != after:
                raise RuntimeError('附件或消息事实改变，验收失败')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--port', type=int, default=24719)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--auth', action='store_true', help='仅为新隔离实例启用真实 web-auth')
    args = parser.parse_args()
    state = args.state.resolve()
    if args.verify:
        marker = json.loads((state / '.akashic-oneoff-fixture.json').read_text())
        assert marker['kind'] == 'akashic-attachment-oneoff' and marker['state'] == str(state)
        expected = json.loads((state / 'attachment-integrity-before.json').read_text())
        assert digest(state / 'workspace') == expected, '附件或正文完整性不一致'
        print('附件与正文完整性一致')
    else:
        asyncio.run(run(state, args.port, args.auth))
