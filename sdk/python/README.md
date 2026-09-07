# Akashic Python SDK

SDK 使用 Message v2 协议，连接已运行的 Gateway。发送返回原始消息的持久化 ACK；模型回复与工具结果随后追加到同一 Session。

```python
from uuid import uuid4
from akashic_sdk import Akashic

with Akashic.connect("/path/to/workspace/akashic.sock") as client:
    session_id = client.session_create()["session_id"]
    message_id = uuid4().hex  # 在发送前保存；网络失败重投时复用同一个 ID。
    ack = client.message_send(session_id, "整理最近的上下文", message_id=message_id)
    print(ack)
    print(client.message_read(session_id))
```

`session_create()` 只分配身份，首条输入提交后会话才出现在列表。`session_list()` 返回分页列表；`message_read()` 通过 `after_seq`、`through_seq` 和 `limit` 读取完整消息页。重投相同 ID 和正文得到同一 ACK；更改原文会明确报错。

异步订阅使用 `session_follow()`；同步客户端具有相同方法：

```python
from akashic_sdk import AsyncAkashic

async with await AsyncAkashic.connect(endpoint) as client:
    async with await client.session_follow(session_id, after_seq=saved_seq) as feed:
        async for event in feed.events():
            if event["type"] == "messages.appended":
                await process_messages(event["items"])
                saved_seq = event["next_after_seq"]  # 处理完成后持久保存。
            elif event["type"] == "reply.status":
                show_current_reply(event)
```

`messages.appended` 携带完整消息页；`reply.status` 只描述当前活动与草稿，不能作为持久完成回执。断线后从已处理的 `seq` 继续订阅即可补齐消息。每个 Session 同时保留一个订阅；同连接重新 follow 会替换旧订阅，旧订阅的关闭不会影响新订阅。

关闭订阅或连接只停止读取。需要停止回复时，发送一条具有新 `message_id` 的 `/stop` 消息；重试原输入时使用 `message_send(session_id, message_id=new_id, retry_of=original_input_id)`，不能同时更改正文或模型选择。

每个订阅只允许调用一次 `events()`；多消费者会争抢同一队列，因此第二次调用立即报错。需要独立消费时建立另一个客户端订阅，并分别保存游标。

每个订阅有独立有界队列。慢读者超过 `queue_size` 时收到 `SlowConsumerError`，其他 Session 和请求继续工作；调用者从已保存游标重读。SDK 不自动重发请求。服务端错误、协议损坏和连接关闭分别抛出 `RemoteError`、`ProtocolError`、`ConnectionClosedError`。

插件管理等控制操作可使用 `request(method, params)`。操作使用明确的 `update_id`，不依赖父 Turn；连接断开后通过 `plugin/update` 查询已有操作。

loopback TCP 需传入 `workspace_token`：

```python
client = await AsyncAkashic.connect("127.0.0.1:2236", workspace_token=token)
```

SDK 不启动 workspace owner。父进程托管可运行 `python main.py app-server --stdio` 并使用 JSON-RPC NDJSON；当前 Python 客户端使用 Unix socket 或 loopback TCP。服务端只接受 v2，旧 Thread/Turn 方法没有兼容入口。请求参数见 [协议 schema](../../schema/app-server-v2.json)。

程序调用使用同一连接上的来源方法；Session ID 由调用方先生成并保存：

```python
session_id = "programmatic:" + uuid4().hex
input_id = uuid4().hex
client.request("programmatic/session/admit", {"session_id": session_id})
client.request("programmatic/message/send", {
    "session_id": session_id, "message_id": input_id, "text": "检查当前项目",
})
result = client.request("programmatic/message/result", {
    "session_id": session_id, "input_id": input_id,
})
```

程序 Session 默认内部展示、排除学习。首次 `admit` 可传 `persist_memory=True`；以后不能更改。同 ID 的相同准入和相同输入可幂等重试。`result.status` 为 `open` 时继续从 `through_seq` 订阅消息后查询；`complete`/`quiet` 返回确切 `ending_message_id` 和 `ending_seq`，`pause`/`failure` 是可恢复的非成功结果，`abandoned` 表示该输入已放弃。停止使用 `programmatic/message/pause`，参数为 Session ID 和新的 Message ID；恢复使用 `programmatic/message/resume`，另传原 `input_id`。

```bash
python main.py exec --new --final-only "检查当前项目"
python main.py exec --session programmatic:ID --message-id INPUT_ID "后续输入"
python main.py exec --session programmatic:ID --resume INPUT_ID
```

`exec` 在发送前输出可恢复身份；`--json` 输出 JSON 行，`--detach` 在接纳后退出，`--persist-memory` 仅用于 `--new`。成功退出码为 0，暂停为 130，失败或放弃为 1。SIGINT 明确请求暂停，普通断线只停止读取。旧 `--thread`、`--runtime` 和父 Turn 发布凭据不属于新接口。
