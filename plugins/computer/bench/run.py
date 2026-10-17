"""通过正式 Gateway 驱动完整 Akashic，运行固定版本 MiniWoB。"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, cast
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.computer_runtime.runtime import runtime
from browser import prepare
from akashic_sdk import AsyncAkashic

UPSTREAM = "a7658beb68e1d3572329e06734c5822bd284ddc8"
PROMPT = """Complete this browser interaction task using the computer skill and computer tool.
Use the existing task tab at {url}. Task: {instruction}
Inspect visible UI and perform ordinary user input with the documented Browser API.
Do not reload, navigate away, create another task instance, or restart the episode.
Do not read source files, scripts, hidden task state, reward globals, core, or genProblem.
Do not use shell, HTTP requests, or direct JavaScript DOM mutation to solve the task.
Read-only inspection of visible DOM and screenshots is allowed. Finish when the UI task is done.
"""

# 这些页面不接受通用 episode 时间预算；单独保留诊断，不混入可比目标集。
UNSUITABLE = {"hover-shape": "upstream_reward_not_implemented",
              "moving-items": "internal_9.9_second_timer",
              "chase-circle": "internal_animation_timer",
              "stock-market": "price_series_expires_after_10_seconds",
              "simon-says": "sequence_ends_before_cold_observation"}


def measurements(directory, status, score):
    """从真实回执计算收尾与错误；违规标记不代替人工完整审计。"""
    rows = json.loads((directory / "messages.json").read_text())
    calls = [part for row in rows for part in row["body"].get("parts", []) if part["kind"] == "tool_call"]
    violations = sorted({part["name"] for part in calls if part["name"] in {"shell", "web_search", "web_fetch"}})
    if score["starts"] != 1:
        violations.append("episode_restart")
    errors = sum(row["body"]["kind"] == "tool_result" and row["body"]["outcome"] != "success" for row in rows)
    positive = score["done"] and score["raw"] > 0
    return {"tool_calls": len(calls), "tool_errors": errors, "violations": violations,
            "completed": positive and status == "complete" and not violations,
            "smooth": positive and status == "complete" and not violations and errors == 0,
            "finish_after_done_ms": score["elapsed_ms"] - score["done_ms"] if score["done"] else None}


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git(path: Path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


async def interact(identity, prompt, directory, timeout, page):
    """从公开程序入口提交，保留持久消息；截止时由正式 pause 终止 Turn。"""
    session, message = "programmatic:" + uuid4().hex, uuid4().hex
    save(directory / "session.json", {"session_id": session, "input_id": message})
    async with await AsyncAkashic.connect(identity["rpc"], max_message_bytes=32 * 1024 * 1024) as client:
        await client.request("programmatic/session/admit", {"session_id": session, "persist_memory": False})
        async with await client.session_follow(session) as feed:
            await client.request("programmatic/message/send", {"session_id": session, "message_id": message, "text": prompt})
            try:
                async with asyncio.timeout(timeout):
                    with (directory / "events.jsonl").open("w") as log:
                        async for event in feed.events():
                            log.write(json.dumps(event, ensure_ascii=False) + "\n")
                            log.flush()
                            if event["type"] == "messages.appended":
                                result = cast(dict[str, Any], await client.request("programmatic/message/result", {"session_id": session, "input_id": message}))
                                if result["status"] != "open":
                                    save(directory / "reply.json", result)
                                    return result["status"]
                    raise ConnectionError("Session subscription ended before result")
            except TimeoutError:
                try:
                    save(directory / "deadline-score.json", await page.score())
                    await page.screenshot(directory / "end.png")
                finally:
                    await client.request("programmatic/message/pause", {"session_id": session, "message_id": uuid4().hex})
                return "deadline"
            finally:
                rows, seq = [], -1
                while True:
                    page = await client.message_read(session, after_seq=seq, limit=200)
                    if not page["items"]:
                        break
                    rows.extend(page["items"])
                    seq = rows[-1]["seq"]
                save(directory / "messages.json", rows)
                calls = {part["value"]["call_record_id"] for row in rows for part in row["body"].get("parts", [])
                         if part["kind"] == "model.facts"}
                stats = [await client.request("models/call_stats", {"call_id": call}) for call in sorted(calls)]
                save(directory / "model-calls.json", stats)


async def case(args, task, seed):
    """每次都启动新宿主与浏览器；环境失败单独保留，不冒充任务失败。"""
    directory = args.output / f"{task}-s{seed}"
    directory.mkdir(mode=0o700)
    result = {"task": task, "seed": seed, "status": "infra_error", "success": False}
    if task in UNSUITABLE:
        result.update(status="unsupported", reason=UNSUITABLE[task])
        save(directory / "result.json", result)
        return result
    start = time.monotonic()
    try:
        async with runtime(directory, max_steps=args.max_steps, endpoint=args.endpoint,
                           model=args.model, key_file=args.key_file) as identity:
            server, page, url, instruction = await prepare(identity, args.upstream / "miniwob/html", task, seed, args.timeout)
            try:
                prompt = PROMPT.format(url=url, instruction=instruction)
                (directory / "prompt.txt").write_text(prompt)
                await page.screenshot(directory / "start.png")
                if args.mode == "agent":
                    status = await interact(identity, prompt, directory, args.timeout, page)
                else:
                    status = "noop"
                if status == "deadline":
                    score = json.loads((directory / "deadline-score.json").read_text())
                else:
                    score = await page.score()
                    await page.screenshot(directory / "end.png")
                result.update(status=status, score=score, success=score["done"] and score["raw"] > 0,
                              audit="pending" if args.mode == "agent" else "not_applicable")
                if args.mode == "agent":
                    result.update(measurements(directory, status, score))
            finally:
                await asyncio.to_thread(server.shutdown)
                server.server_close()
    except Exception as error:
        # 用例边界记录完整失败并继续扫任务；不把异常转换成成功。
        result.update(status="infra_error", success=False, error=f"{type(error).__name__}: {error}")
        (directory / "error.txt").write_text(traceback.format_exc())
    result["seconds"] = round(time.monotonic() - start, 3)
    save(directory / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://api.commandcode.ai/provider/v1")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--mode", choices=["agent", "noop"], default="agent")
    args = parser.parse_args()
    args.upstream, args.output, args.key_file = args.upstream.resolve(), args.output.resolve(), args.key_file.resolve()
    if git(args.upstream, "rev-parse", "HEAD") != UPSTREAM or git(args.upstream, "status", "--porcelain"):
        parser.error("需要干净、固定提交的 MiniWoB checkout")
    available = sorted(path.stem for path in (args.upstream / "miniwob/html/miniwob").glob("*.html"))
    tasks = args.tasks or available
    if len(available) != 130 or set(tasks) - set(available) or args.timeout <= 0 or args.max_steps <= 0:
        parser.error("任务列表或预算无效")
    args.output.mkdir(parents=True, mode=0o700)
    sources = [*Path(__file__).parent.glob("*.py"), * (ROOT / "benchmark/computer_runtime").glob("*.py")]
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "key_file"}
    manifest.update(source=git(ROOT, "rev-parse", "HEAD"), upstream=UPSTREAM, tasks=tasks,
        scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    save(args.output / "manifest.json", manifest)
    source_patch = git(ROOT, "diff", "HEAD")
    (args.output / "source.patch").write_text(source_patch)
    scripts = args.output / "scripts"
    scripts.mkdir()
    for path in sources:
        (scripts / path.name).write_bytes(path.read_bytes())
    results = []
    for task in tasks:
        for seed in args.seeds:
            result = await case(args, task, seed)
            if git(ROOT, "rev-parse", "HEAD") != manifest["source"] or git(ROOT, "diff", "HEAD") != source_patch:
                raise RuntimeError("运行中源码发生变化；本批结果无效，请保留证据后使用新目录重跑")
            results.append(result)
            save(args.output / "results.json", results)
    if any(result["status"] == "infra_error" for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
