"""用真实 ReplyState、Task 与 WebSocket 同源的 asdict 帧验证生成中的解码速度事实：
首段起表、每个非空增量累加、重试清空并重新起表。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

from agent.plugin_composition.tasks import Task
from plugins.reply.status import ReplyState

SESSION = "akashic:reply-preview-fixture"


def check(checks: list[dict[str, object]], ok: bool, label: str, detail: object = "") -> None:
    checks.append({"label": label, "ok": bool(ok), "detail": detail})
    if not ok:
        raise SystemExit(f"失败：{label} {detail}")


async def hold(_task: Task) -> object:
    await asyncio.Event().wait()
    return None


def preview_frame(state: ReplyState) -> dict[str, object]:
    """与 ReplyRead.follow 推给 WebSocket 的是同一个 asdict 投影。"""
    activity = asdict(state.read.snapshot(SESSION)[0])
    preview = activity["preview"]
    if not isinstance(preview, dict):
        raise SystemExit("快照缺少预览")
    return preview


async def run() -> dict[str, object]:
    state = ReplyState()
    task = Task(hold, asyncio.Event())
    checks: list[dict[str, object]] = []
    try:
        with state.open(task, SESSION, "akashic") as preview:
            with preview("draft-1") as delta:
                await delta({"call_record_id": "call-a"})
                before = preview_frame(state)
                check(checks, before["output_deltas"] == 0, "未出字时没有增量", before)
                check(checks, before["decode_ms"] is None, "未出字时没有解码时长", before)

                await delta({"content_delta": "你"})
                first = preview_frame(state)
                check(checks, first["output_deltas"] == 1, "首段正文计一个增量", first)
                check(checks, isinstance(first["decode_ms"], float) and first["decode_ms"] < 200,
                      "首段到达即起表", first["decode_ms"])

                await asyncio.sleep(0.05)
                await delta({"thinking_delta": "想"})
                await asyncio.sleep(0.6)
                await delta({"content_delta": "好"})
                steady = preview_frame(state)
                check(checks, steady["output_deltas"] == 3, "正文与思考增量一起累加", steady)
                decode_ms = steady["decode_ms"]
                check(checks, isinstance(decode_ms, float) and 500 <= decode_ms < 2000,
                      "解码时长随到达时刻单调增长", decode_ms)
                # 与界面同一算法：增量数 / 解码秒数。
                rate = steady["output_deltas"] / (float(decode_ms) / 1000)
                check(checks, 3 <= rate <= 6, "生成中的估算速度落在实际到达区间", round(rate, 2))

                await delta({"retry_status": "退避 1s"})
                cleared = preview_frame(state)
                check(checks, cleared["output_deltas"] == 0 and cleared["decode_ms"] is None,
                      "重试清空增量与解码时长", cleared)
                check(checks, cleared["text"] == "" and cleared["thinking"] == "",
                      "重试不保留上一次尝试的草稿", cleared["text"] + cleared["thinking"])

                await asyncio.sleep(0.05)
                await delta({"content_delta": "再"})
                restarted = preview_frame(state)
                check(checks, restarted["output_deltas"] == 1, "重试后重新累计", restarted)
                check(checks, isinstance(restarted["decode_ms"], float) and restarted["decode_ms"] < 200,
                      "重试后重新起表", restarted["decode_ms"])
                check(checks, "truncated" not in restarted or restarted["truncated"] is not True,
                      "增量事实不改变草稿截断标记", restarted.get("truncated"))
    finally:
        task.cancel()
        try:
            await task.join()
        except asyncio.CancelledError:
            # 收尾取消就是本场景的结束方式；不把它当成检查失败。
            pass
        state.close()
    return {"session": SESSION, "checks": checks}


def main() -> None:
    report = asyncio.run(run())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"通过 {len(report['checks'])} 项")


if __name__ == "__main__":
    main()
