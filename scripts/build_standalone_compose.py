#!/usr/bin/env python3
"""从仓库 Compose 生成只使用固定发行镜像的独立启动附件。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """复用同一份运行配置，仅替换镜像的取得方式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='带 sha256 digest 的发行镜像')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r'ghcr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}', args.image) is None:
        parser.error('--image 必须是固定 digest 的 GHCR 镜像')
    # 1. 不展开端口变量，让下载附件的人仍能通过 AKASHIC_PORT 选择端口。
    result = subprocess.run(
        ['docker', 'compose', '-f', str(ROOT / 'compose.yaml'), 'config',
         '--no-interpolate', '--no-normalize', '--format', 'json'],
        check=True, capture_output=True, text=True,
    )
    document = json.loads(result.stdout)
    service = document['services']['akashic']
    del service['build']
    service['image'] = args.image
    # 2. JSON 也是合法 YAML；附件没有源码目录、构建参数或额外环境文件依赖。
    with args.output.open('x') as output:
        output.write(json.dumps(document, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
