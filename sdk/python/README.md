# Akashic Python SDK

```python
from akashic_sdk import Akashic

with Akashic.connect("/path/to/workspace/akashic.sock") as client:
    thread = client.thread_start()
    result = thread.run("整理最近的上下文")
    print(result["finalResponse"])
```

异步 API 使用 `await AsyncAkashic.connect(endpoint)`，`Thread.turn()` 返回可消费
`stream()`、`interrupt()` 和 `result()` 的 turn handle。连接断开不会取消服务端 turn；重新连接
后可通过 `thread_resume()` 和协议 `turn/read` 恢复状态。

```python
async with await AsyncAkashic.connect(endpoint) as client:
    thread = await client.thread_resume(thread_id)
    handle = await thread.turn("继续分析")
    async for event in handle.events():
        if event["method"] == "item/assistantMessage/delta":
            print(event["params"]["delta"], end="")
    result = await handle.result()
```

长任务可调用 `await handle.interrupt()`；同步 API 对应 `handle.interrupt()`、
`handle.events()` 和 `handle.result()`。远端业务错误抛出 `RemoteError`，协议损坏、慢消费者和
连接关闭分别抛出 `ProtocolError`、`SlowConsumerError`、`ConnectionClosedError`。SDK 只连接
已运行的 gateway，不会隐式启动第二个 workspace owner。父进程托管模式请启动
`python main.py app-server --stdio` 并直接使用 JSON-RPC NDJSON 流；当前 Python facade 连接
Unix socket 或 loopback TCP。

`RemoteError.retryable` 直接反映服务端错误数据中的重试语义。当前 thread 已有 active turn
时，新的 `turn()` 会得到 `retryable=True`；SDK 不自动重试，调用方应等待、停止当前 turn 或稍后重发。

loopback TCP 连接需显式传入 workspace token：

```python
client = await AsyncAkashic.connect("127.0.0.1:2236", workspace_token=token)
operation = await (await client.thread_resume(thread_id)).consolidate()
```

consolidation 的最终结果通过全局 `operation/completed` notification 返回。
