#!/usr/bin/env python3
"""
手动触发 skill action 的测试脚本。

用法：
  python trigger_skill_action.py                     # 随机抽取一个 skill action
  python trigger_skill_action.py my-action-id        # 触发指定 action
  python trigger_skill_action.py --socket /path.sock # 指定 socket 路径

前提：agent 已在后台运行（python main.py）
"""

import asyncio
import json
import sys

DEFAULT_SOCKET = "/tmp/akasic.sock"


async def trigger(socket_path: str, action_id: str | None) -> None:
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
    except FileNotFoundError, ConnectionRefusedError:
        print(f"[错误] 无法连接到 agent socket: {socket_path}")
        print("请先启动 agent：python main.py")
        sys.exit(1)

    payload = {"type": "command", "command": "trigger_skill_action"}
    if action_id:
        payload["action_id"] = action_id

    print(f"[发送] {json.dumps(payload, ensure_ascii=False)}")
    writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
    await writer.drain()

    # 等待 command_result（skill action 可能运行较久，最多等 10 分钟）
    print("[等待] skill action 执行中，最多等待 10 分钟……")
    try:
        line = await asyncio.wait_for(reader.readline(), timeout=600)
    except asyncio.TimeoutError:
        print("[超时] 10 分钟内未收到响应")
        writer.close()
        sys.exit(1)

    if not line:
        print("[错误] 连接已断开，未收到响应")
        sys.exit(1)

    data = json.loads(line)
    ok = data.get("ok", False)
    message = data.get("message", "")
    status = "成功" if ok else "失败"
    print(f"[结果] {status} — {message}")
    writer.close()
    sys.exit(0 if ok else 1)


def main() -> None:
    args = sys.argv[1:]
    socket_path = DEFAULT_SOCKET
    action_id: str | None = None

    i = 0
    while i < len(args):
        if args[i] == "--socket" and i + 1 < len(args):
            socket_path = args[i + 1]
            i += 2
        else:
            action_id = args[i]
            i += 1

    print(f"[配置] socket={socket_path}  action_id={action_id or '(随机)'}")
    asyncio.run(trigger(socket_path, action_id))


if __name__ == "__main__":
    main()
