#!/usr/bin/env python3
"""用一次性 Compose project 验证镜像首次启动、网页和数据卷重启。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.request import ProxyHandler, build_opener
import uuid


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = """
import hashlib, json, os
from pathlib import Path

assert os.getuid() != 0, '运行进程必须使用普通用户'
for directory in ('/opt/venv', '/opt/core', '/opt/distribution'):
    root = Path(directory)
    for path in (root, *root.rglob('*')):
        assert not os.access(path, os.W_OK), path
distribution = Path('/opt/distribution')
report = json.loads((distribution / 'distribution.json').read_text())
for plugin in report['plugins']:
    source = distribution / 'sources' / plugin['name']
    json.loads((source / '.akashic-source.json').read_text())
    assert not os.access(source, os.W_OK), source
state = Path('/data')
paths = ('startup.json', 'config.toml', 'plugin-home/manifest.toml',
         'workspace/runtime/distribution-install.json',
         'workspace/runtime/plugin-stable.json')
print(json.dumps({'uid': os.getuid(), 'source_commit': report['source_commit'],
                  'files': {name: hashlib.sha256((state / name).read_bytes()).hexdigest()
                            for name in paths}}))
"""


def check_web(origin: str) -> list[str]:
    """通过映射端口读取真实网页和插件模块，不发送模型请求。"""
    opener = build_opener(ProxyHandler({}))
    with opener.open(origin + '/', timeout=10) as response:
        assert b'<html' in response.read().lower(), '网页没有返回 HTML'
    with opener.open(origin + '/api/shell/state', timeout=10) as response:
        assert json.load(response)['chatReady'], '聊天网关尚未就绪'
    with opener.open(origin + '/api/chat/web-ui/bootstrap', timeout=10) as response:
        modules = json.load(response)['modules']
    assert modules, '默认插件没有提供 Web 模块'
    for module in modules:
        assert module['module'], module['pluginId']
    plugins = sorted(module['pluginId'] for module in modules)
    names = {plugin.split('@', 1)[0] for plugin in plugins}
    assert {'onboarding', 'conversation-ui', 'shell-ui'} <= names, plugins
    return plugins


def main() -> None:
    """只管理本次创建的 project，保留日志后删除其一次性数据卷。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='已经构建的本地镜像')
    parser.add_argument('--revision', required=True, help='镜像应包含的完整源码 commit')
    parser.add_argument('--compose', type=Path, default=ROOT / 'compose.yaml',
                        help='仓库构建配置或发行附件')
    parser.add_argument('--output', type=Path, help='尚不存在的验收输出目录')
    args = parser.parse_args()
    if args.output is None:
        output = Path(tempfile.mkdtemp(prefix='akashic-compose-smoke-'))
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    project = 'akashic-smoke-' + uuid.uuid4().hex
    override = output / 'compose.json'
    override.write_text(json.dumps({'services': {'akashic': {'image': args.image}}}) + '\n')
    environment = dict(os.environ, AKASHIC_PORT='0', AKASHIC_REVISION=args.revision)
    command = ['docker', 'compose', '-p', project, '-f', str(args.compose.resolve()),
               '-f', str(override)]

    def compose(*arguments: str, capture: bool = False) -> str:
        """每次命令都绑定同一个隔离 project 与镜像。"""
        result = subprocess.run(command + list(arguments), env=environment, cwd=ROOT,
                                check=True, text=True, capture_output=capture)
        return result.stdout.strip() if capture else ''

    print(f'验收 project：{project}；日志：{output}', flush=True)
    startup_logs: set[str] = set()
    try:
        # 1. 从空卷启动，通过公开的 Compose healthcheck 等待就绪。
        compose('up', '-d', '--no-build', '--pull', 'never', '--wait', '--wait-timeout', '600')
        address = compose('port', 'akashic', '2236', capture=True)
        origin = 'http://' + address
        modules = check_web(origin)
        before = json.loads(compose('exec', '-T', 'akashic', 'python', '-c', SNAPSHOT, capture=True))
        assert before['source_commit'] == args.revision, before['source_commit']
        print('首次启动通过：普通用户、只读制品、真实网页和插件模块', flush=True)
        cold_logs = compose('logs', '--no-color', capture=True)
        (output / 'cold-compose.log').write_text(cold_logs + '\n')
        startup_logs.update(re.findall(r'/data/(startup-[0-9a-f]+\.log)', cold_logs))

        # 2. 重新创建容器而保留同一数据卷，核对持久安装与原选择。
        compose('down')
        compose('up', '-d', '--no-build', '--pull', 'never', '--wait', '--wait-timeout', '600')
        origin = 'http://' + compose('port', 'akashic', '2236', capture=True)
        assert check_web(origin) == modules, '重启后的 Web 插件集合改变'
        after = json.loads(compose('exec', '-T', 'akashic', 'python', '-c', SNAPSHOT, capture=True))
        assert after == before, '重启改写了安装记录、配置或插件选择'
        report = {'project': project, 'image': args.image, **after, 'web_modules': modules,
                  'cold_start': 'passed', 'volume_restart': 'passed'}
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('重建容器通过：配置、安装记录和插件选择保留', flush=True)
    finally:
        # 3. 日志与失败详情留在宿主；只删除本次 UUID project 的临时卷。
        try:
            logs = compose('logs', '--no-color', capture=True)
            (output / 'compose.log').write_text(logs + '\n')
            startup_logs.update(re.findall(r'/data/(startup-[0-9a-f]+\.log)', logs))
            container = compose('ps', '-a', '-q', capture=True)
            if container:
                # docker cp 也能读取已经退出的容器；不把配置或凭据写入验收产物。
                for name in sorted(startup_logs):
                    subprocess.run(['docker', 'cp', container + ':/data/' + name, str(output / name)],
                                   check=True)
        finally:
            compose('down', '--volumes', '--remove-orphans')


if __name__ == '__main__':
    main()
