"""本机命令：能读写 workspace 的进程就是主人，不需要密码，也不需要运行中的 Gateway。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .state import TICKET_S, USERNAME, OwnerAuth


def _auth(workspace: Path) -> OwnerAuth:
    auth = OwnerAuth(workspace / "web-auth")
    if (initial := auth.ensure()) is not None:
        print(f"已创建账号 {USERNAME}，初始密码 {initial}")
    return auth


async def login_main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    """打印一条一次性登录链接；首次进入后仍需设置自己的密码。"""
    parser = argparse.ArgumentParser(prog="web-login", description=login_main.__doc__)
    parser.add_argument("--url", default=f"http://localhost:{os.environ.get('AKASHIC_WEB_PORT', '2236')}",
                        help="浏览器访问 Akashic 的地址")
    options = parser.parse_args(arguments)
    ticket = _auth(workspace).issue_ticket()
    print(f"一次性登录链接（{TICKET_S // 60} 分钟内有效，只能使用一次）：")
    print(f"{options.url.rstrip('/')}/_auth/login#t={ticket}")
    return 0


async def passwd_main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    """重置为临时密码并让所有设备退出；登录后必须设置新密码。"""
    argparse.ArgumentParser(prog="web-passwd", description=passwd_main.__doc__).parse_args(arguments)
    password = _auth(workspace).reset_password()
    print(f"已重置，所有设备已退出。用户名 {USERNAME}，临时密码 {password}")
    print("登录后需要设置新密码。")
    return 0
