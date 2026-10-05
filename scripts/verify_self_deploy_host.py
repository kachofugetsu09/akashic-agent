"""只在已初始化的独立虚拟机里验证真实 Shell 自部署与停止范围。"""

import argparse
import asyncio
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]

from akashic_sdk import AsyncAkashic
from agent.config import resolve_app_server_endpoint
from agent.config_models import Config
from scripts.akashic_release.doctor import read_environment
from scripts.verify_self_deploy_runtime import start_model_server, configure_model


def read_service(unit: str, *, user: bool = False) -> str:
    command = [
        "systemctl",
        *(["--user"] if user else []),
        "show",
        unit,
        "-p",
        "MainPID",
        "-p",
        "NRestarts",
        "-p",
        "ActiveState",
    ]
    result = subprocess.check_output(command, text=True)
    assert "ActiveState=active" in result, result
    return result


def read_sentinels() -> dict[str, object]:
    """检查无关系统服务、用户服务、容器和机器 boot 身份。"""
    container = json.loads(
        subprocess.check_output(["docker", "inspect", "unrelated-sentinel"], text=True)
    )[0]
    assert container["State"]["Running"], container["State"]
    return {
        "boot": Path("/proc/sys/kernel/random/boot_id").read_text(),
        "external": read_service("akashic-home-services.service"),
        "user": read_service("unrelated-user.service", user=True),
        "container": (
            container["Id"],
            container["State"]["StartedAt"],
            container["RestartCount"],
        ),
    }


async def wait_reply(client: AsyncAkashic, session: str, message_id: str) -> dict:
    """由真实输入触发回合，等待普通最终回复进入消息日志。"""
    subscription = await client.session_follow(session)
    try:
        await client.request(
            "programmatic/message/send",
            {
                "session_id": session,
                "message_id": message_id,
                "text": "Submit the authorized update, then finish this turn.",
            },
        )
        async with asyncio.timeout(60):
            async for _ in subscription.events():
                page = await client.message_read(session, limit=100)
                index = next(
                    i
                    for i, item in enumerate(page["items"])
                    if item["id"] == message_id
                )
                if any(
                    item["body"].get("finish") == "complete"
                    for item in page["items"][index + 1 :]
                ):
                    return page
        raise RuntimeError("未得到普通最终回复")
    finally:
        await subscription.close()


async def verify(args: argparse.Namespace) -> None:
    """发起真实宿主部署，比较最终消息、运行身份和无关服务。"""
    environment = read_environment(args.runtime_env)
    if environment.get("AKASHIC_ENVIRONMENT") != "isolated-vm-e2e":
        raise ValueError("此脚本只能修改显式标记 isolated-vm-e2e 的一次性部署")
    work = Path(environment["AKASHIC_WORKSPACE"])
    before_ready = json.loads((work / ".runtime-ready.json").read_text())
    before_sentinels = read_sentinels()
    heartbeat = Path.home() / "sentinel-heartbeat.log"
    before_lines = heartbeat.read_text().splitlines()
    args.evidence.mkdir(parents=True, exist_ok=False)
    command = f"{shlex.quote(str(Path.home() / '.local/bin/akashic-release'))} submit --commit {shlex.quote(args.commit)} --timeout 1800"
    runner, port = await start_model_server(
        args.evidence, command=command, host="0.0.0.0"
    )
    config = Config.load(environment["AKASHIC_CONFIG"], workspace=work)
    endpoint = resolve_app_server_endpoint(config.app_server.listen, work)
    session = "programmatic:host-self-deploy-e2e"
    try:
        # 1. 真正的 Docker Core → Host Bridge Shell → 独立用户 systemd worker。
        async with await AsyncAkashic.connect(endpoint) as client:
            await configure_model(client, port, host=args.fixture_host)
            await client.request(
                "programmatic/session/admit",
                {"session_id": session, "persist_memory": False},
            )
            first = await wait_reply(client, session, "host-update-input")
            (args.evidence / "before-messages.json").write_text(
                json.dumps(first, ensure_ascii=False)
            )
        results = [
            item for item in first["items"] if item["body"]["kind"] == "tool_result"
        ]
        assert len(results) == 1 and results[0]["body"]["outcome"] == "success", results
        shell_result = json.loads(results[0]["body"]["parts"][0]["value"])
        assert shell_result["process_status"] == "succeeded", shell_result
        accepted = json.loads(shell_result["output"].strip())
        assert accepted["status"] == "accepted", accepted
        request_path = args.root / "run/self-deploy" / (accepted["requestId"] + ".json")
        print("ACCEPTED", accepted, flush=True)

        # 2. 只由外部观察者等待；Agent 没有追加 Shell 或 load_tools。
        deadline = time.monotonic() + 1800
        last_status = None
        while True:
            record = json.loads(request_path.read_text())
            if record["status"] != last_status:
                print("JOB", record["status"], flush=True)
                last_status = record["status"]
            assert read_sentinels() == before_sentinels, "更新影响了无关运行对象"
            if record["status"] in {"failed", "active"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("宿主更新未在期限内完成")
            await asyncio.sleep(0.5)
        (args.evidence / "job.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2)
        )
        assert record["status"] == "active", record
        after_ready = json.loads((work / ".runtime-ready.json").read_text())
        assert after_ready["bootId"] != before_ready["bootId"], after_ready
        assert after_ready["sourceCommit"] == args.commit, after_ready
        requests = (args.evidence / "requests.json").read_text()
        assert "agent_restart" not in requests, "Docker runtime 暴露了旧重启工具"

        # 3. 新 boot 读取原消息，再执行下一回合；原正文、身份和顺序不减少。
        async with await AsyncAkashic.connect(endpoint) as client:
            reloaded = await client.message_read(session, limit=100)
            assert reloaded["items"] == first["items"], "停止/迁移改写了原消息"
            second = await wait_reply(client, session, "host-after-update-input")
            assert (
                second["items"][: len(first["items"])] == first["items"]
            ), "新回合改写了原消息"
            (args.evidence / "after-messages.json").write_text(
                json.dumps(second, ensure_ascii=False)
            )
        assert (
            "deploy-akashic" in (args.evidence / "requests.json").read_text()
        ), "默认 Docker 组合缺少部署 Skill"
        after_lines = heartbeat.read_text().splitlines()
        assert after_lines[: len(before_lines)] == before_lines
        ticks = [float(value) for value in after_lines[len(before_lines) - 1 :]]
        assert (
            len(ticks) > 1 and max(b - a for a, b in zip(ticks, ticks[1:])) < 5
        ), "无关服务心跳中断"
        assert read_sentinels() == before_sentinels
        print(
            "PASS",
            {
                "oldBoot": before_ready["bootId"],
                "newBoot": after_ready["bootId"],
                "commit": args.commit,
                "originalMessages": len(first["items"]),
                "unrelatedServicesUnchanged": True,
            },
            flush=True,
        )
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--fixture-host", required=True, help="Docker Core 可访问的虚拟机 IP"
    )
    parser.add_argument("--evidence", type=Path, required=True)
    asyncio.run(verify(parser.parse_args()))


if __name__ == "__main__":
    main()
