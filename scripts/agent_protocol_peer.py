"""Deterministic external CLI peer, not a real Codex/Claude acceptance claim."""
import json
import os
import sys
import time

mode = os.environ.get("PEER_MODE", "codex")
case = os.environ.get("PEER_CASE", "success")
def send(frame):
    data = (json.dumps(frame, ensure_ascii=False) + "\n").encode()
    for i in range(0, len(data), 7):
        os.write(1, data[i:i + 7])

def receive():
    return json.loads(sys.stdin.buffer.readline())

if mode == "codex":
    init = receive()
    assert init["method"] == "initialize"
    send({"id": init["id"], "result": {}})
    assert receive()["method"] == "initialized"
    thread = receive()
    assert thread["params"]["sandbox"] == "read-only"
    assert thread["params"]["approvalPolicy"] == "never"
    send({"id": thread["id"], "result": {"thread": {"id": "thread"}}})
    servers = receive()
    assert servers["method"] == "mcpServerStatus/list"
    send({"id": servers["id"], "result": {"data": [{"name": "leak"}] if case == "mcp" else []}})
    if case == "mcp":
        sys.exit(0)
    turn = receive()
    send({"id": turn["id"], "result": {"turn": {"id": "turn"}}})
    if case == "hang":
        time.sleep(120)
    if case == "eof":
        sys.exit(0)
    if case == "malformed":
        os.write(1, b"not-json\n")
        sys.exit(0)
    if case == "approval":
        send({"id": 999, "method": "item/commandExecution/requestApproval", "params": {}})
        assert receive()["result"]["decision"] == "decline"
    if case == "unsupported":
        send({"id": 998, "method": "item/unknown/request", "params": {}})
        assert receive()["error"]["code"] == -32601
        send({"method": "item/completed", "params": {"threadId": "thread", "turnId": "turn",
              "item": {"type": "agentMessage", "phase": "future_phase", "text": "progress"}}})
    for identity, text in [("other", "MUST-NOT-LEAK"), ("thread", "已完成：中文碎包")]:
        send({"method": "item/completed", "params": {"threadId": identity, "turnId": "turn",
              "item": {"type": "agentMessage", "phase": "final_answer", "text": text}}})
    send({"method": "turn/completed", "params": {"threadId": "thread", "turn": {
        "id": "turn", "status": "failed" if case == "failed" else "completed"}}})
else:
    assert receive()["type"] == "user"
    send({"type": "system", "subtype": "init", "session_id": "session"})
    if case == "hang":
        time.sleep(120)
    if case == "eof":
        sys.exit(0)
    if case == "malformed":
        os.write(1, b"[]\n")
        sys.exit(0)
    send({"type": "assistant", "session_id": "wrong" if case == "cross-session" else "session",
          "message": {"content": [{"type": "tool_use", "name": "Read"}]}})
    send({"type": "result", "session_id": "session", "subtype": "error" if case == "failed" else "success",
          "result": "已完成：中文碎包"})
sys.stdout.flush()
