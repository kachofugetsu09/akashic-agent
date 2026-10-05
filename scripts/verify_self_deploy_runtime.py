"""用真实 HTTP 模型、Shell、控制协议和临时 workspace 核对停止边界。"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from aiohttp import web
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]
from agent.config_models import Config
from agent.config import resolve_app_server_endpoint
from agent.plugin_composition.config_input import save_config
from bootstrap.app import build_app_runtime
from bootstrap.runtime_readiness import RuntimeReadiness
from akashic_sdk import AsyncAkashic


def prepare_workspace() -> tuple[Path, Path, Path]:
    """建立独立 HOME、workspace、插件来源和正式初始化基线。"""
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-self-deploy-e2e-"))
    print("EVIDENCE", sandbox, flush=True)
    sources = sandbox / "plugins"
    work = sandbox / "state/workspace"
    config = sandbox / "state/config.toml"
    names = "channels commands sources content context tools conversation react turn_projection reply_program reply tool_search assets standard_tools models openai_compatible delivery delivery_policy programmatic message_push".split()
    for name in names:
        shutil.copytree(
            ROOT / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    os.environ.update(
        HOME=str(sandbox / "home"),
        AKASHIC_PLUGIN_HOME=str(sandbox / "state/plugin-home"),
        AKASHIC_EXTRA_PLUGIN_DIRS=str(sources),
        AKASHIC_WORKSPACE=str(work),
        AKASHIC_BOOT_ID="local-e2e",
        PYTHONPATH=str(ROOT / "sdk/python/src") + ":" + str(ROOT),
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "init",
            "--config",
            str(config),
            "--workspace",
            str(work),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    save_config(
        work / "plugin-data/context-builtin",
        {"prompt_sources": {"skills": "standard_tools"}},
    )
    return sandbox, work, config


async def start_model_server(
    sandbox: Path, *, command: str | None = None, host: str = "127.0.0.1", port: int = 0
) -> tuple[web.AppRunner, int]:
    """控制外部 HTTP 模型响应；内部组件使用真实实现。"""
    requests = []
    shell_command = command or f"{sys.executable} {sandbox}/arm.py"

    async def models(request):
        return web.json_response({"data": [{"id": "fixture", "object": "model"}]})

    async def chat(request):
        body = await request.json()
        requests.append(body)
        (sandbox / "requests.json").write_text(json.dumps(requests))
        messages = body["messages"]
        tools = body.get("tools", [])
        if not tools:
            message: dict[str, Any] = {"role": "assistant", "content": "OK"}
            finish = "stop"
        elif any(m["role"] == "tool" for m in messages):
            message = {"role": "assistant", "content": "已提交，正常结束本轮。"}
            finish = "stop"
        else:
            shell = next(
                t["function"]["name"]
                for t in tools
                if t["function"]["name"].endswith("shell")
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "fixture-shell",
                        "type": "function",
                        "function": {
                            "name": shell,
                            "arguments": json.dumps(
                                {
                                    "command": shell_command,
                                    "description": "Submit local deployment",
                                    "login": False,
                                    "yield_time_ms": 10000,
                                }
                            ),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        if not body.get("stream"):
            return web.json_response(
                {
                    "id": "fixture",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )
        delta = dict(message)
        delta.pop("role", None)
        for i, c in enumerate(delta.get("tool_calls", [])):
            c["index"] = i
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            (
                "data: "
                + json.dumps(
                    {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                )
                + "\n\ndata: [DONE]\n\n"
            ).encode()
        )
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/models", models)
    app.router.add_post("/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


async def configure_model(
    client: AsyncAkashic, port: int, *, host: str = "127.0.0.1"
) -> None:
    """通过正式 RPC 创建并选择本地 HTTP 模型。"""
    result = cast(
        dict[str, Any],
        await client.request(
            "models/command",
            {
                "type": "create_connection_with_model",
                "connection": {
                    "expected_revision": 0,
                    "connection_id": "fixture",
                    "name": "Fixture",
                    "driver_id": "openai-compatible",
                    "endpoint": f"http://{host}:{port}",
                    "auth_identity": "fixture",
                    "credential": {"api_key": "fixture"},
                },
                "model": {
                    "expected_revision": 0,
                    "model_id": "fixture",
                    "connection_id": "fixture",
                    "kind": "chat",
                    "model": "fixture",
                    "capabilities": {
                        "context_window": 32000,
                        "max_output_tokens": 4096,
                        "supports_tool_calls": True,
                    },
                    "capability_sources": {},
                },
            },
        ),
    )
    print("MODEL", result, flush=True)
    assert result["status"] == 200, result
    revision = result["body"]["revision"]
    result = cast(
        dict[str, Any],
        await client.request(
            "models/command",
            {
                "type": "set_default",
                "expected_revision": revision,
                "role": "default",
                "model_id": "fixture",
            },
        ),
    )
    assert result["status"] == 200, result


async def main() -> None:
    """在真实最终回复送达后核对停止、取消、旧 boot 拒绝与关闭。"""
    sandbox, work, config = prepare_workspace()
    runner, port = await start_model_server(sandbox)
    loaded = Config.load(config, workspace=work)
    runtime = build_app_runtime(
        loaded, work, readiness=RuntimeReadiness(work, "local-e2e")
    )
    endpoint = resolve_app_server_endpoint(loaded.app_server.listen, work)
    (sandbox / "arm.py").write_text(
        f"""import asyncio,json,os,sys\nsys.path[:0]={repr([str(ROOT),str(ROOT/'sdk/python/src')])}\nfrom akashic_sdk import AsyncAkashic\nfrom pathlib import Path\nasync def main():\n p={{**json.loads(os.environ['AKASHIC_CALL_CONTEXT']), 'request_id':'local-request','timeout_s':60}}\n Path({str(sandbox/'stop.json')!r}).write_text(json.dumps(p))\n async with await AsyncAkashic.connect({endpoint!r}) as c:\n  print(json.dumps(await c.request('runtime/prepare-stop', {{**p,'arm_only':True}})))\nasyncio.run(main())\n"""
    )
    try:
        # 2. 通过真实模型配置和消息接口驱动 Shell。
        await runtime.start()
        print("ROOT", runtime.core.plugin_manager.live_root.generation_id, flush=True)
        async with await AsyncAkashic.connect(endpoint) as client:
            await configure_model(client, port)
            session = "programmatic:self-deploy-e2e"
            await client.request(
                "programmatic/session/admit",
                {"session_id": session, "persist_memory": False},
            )
            subscription = await client.session_follow(session)
            await client.request(
                "programmatic/message/send",
                {
                    "session_id": session,
                    "message_id": "local-input",
                    "text": "Submit local update and finish.",
                },
            )
            async with asyncio.timeout(40):
                async for event in subscription.events():
                    page = await client.message_read(session, limit=100)
                    (sandbox / "messages.json").write_text(
                        json.dumps(page, ensure_ascii=False)
                    )

                    if any(
                        m["body"].get("kind") == "output"
                        and m["body"].get("finish") == "complete"
                        for m in page["items"]
                    ):
                        break
            results = [m for m in page["items"] if m["body"]["kind"] == "tool_result"]
            assert (
                len(results) == 1 and results[0]["body"]["outcome"] == "success"
            ), results
            p = json.loads((sandbox / "stop.json").read_text())
            # 3. 原 frame 已 flush；送达 claim 必须能跨普通 route 的回收。
            ack = cast(dict[str, Any], await client.request("runtime/prepare-stop", p))
            print("DRAINED", ack, flush=True)
            assert ack["state"] == "drained"
            await client.request("runtime/cancel-stop", p)
            assert runtime.core.restart_gate.accepting
            try:
                await client.request("runtime/prepare-stop", {**p, "boot_id": "wrong"})
            except RuntimeError as error:
                print("STALE_REJECTED", str(error), flush=True)
            else:
                raise AssertionError("wrong boot accepted")
            await subscription.close()
    finally:
        await runtime.shutdown()
        await runner.cleanup()
    requests = (sandbox / "requests.json").read_text()
    assert "deploy-akashic" in requests, "部署 Skill 未进入真实模型上下文"
    assert "agent_restart" not in requests, "未托管 runtime 暴露了自重启工具"
    receipt = json.loads((work / "runtime/closed/local-e2e.json").read_text())
    assert receipt["state"] == "closed"
    print("PASS", receipt, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
