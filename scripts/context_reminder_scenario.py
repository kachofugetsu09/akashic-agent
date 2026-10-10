"""通过真实 Message、ReAct 和模型调用账核验材料位置、实时撤下与重启。"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.models.contract import (
    LLMResponse,
    ModelContinuation,
    ModelRequest,
    ToolCall,
)
from plugins.context.api import Materials, Reminder, material_data
from plugins.tools.execution import Result
from session.message import ContentPart, Input, Output
from tests.support.message_react import runtime


def plain(value):
    """将实际请求的不可变 JSON 转成可比较的普通数据。"""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def stored_rows(path: Path):
    """独立连接读取完整 Message 行，核对正常执行没有改写历史。"""
    with sqlite3.connect(path) as db:
        return db.execute("SELECT * FROM messages ORDER BY seq").fetchall()


async def scenario(path: Path, *, live_only: bool = False) -> dict:
    """受控 provider 产生工具调用，真实执行链读取和更新一次性材料文件。"""
    path.mkdir()
    material_path = path / "materials.json"
    material_path.write_text(json.dumps({"replay": "REMINDER_A", "live": "LIVE_X"}))
    requests: list[ModelRequest] = []
    checks: list[str] = []
    last_step = 4 if live_only else 7

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    async def complete(request):
        requests.append(request)
        step = len(requests)
        return LLMResponse(
            "done" if step >= last_step else None,
            [] if step >= last_step else [ToolCall(f"tool-{step}", "example", {"step": step})],
            thinking=f"provider reasoning {step}",
            continuation=ModelContinuation("model", {"step": step}),
        )

    async def invoke(_key, arguments):
        state = json.loads(material_path.read_text())
        step = arguments["step"]
        if step == 2:
            state["live"] = None if live_only else "LIVE_Y"
        elif step == 3 and not live_only:
            state["replay"] = "REMINDER_B"
        elif step == 5:
            state["live"] = None
        material_path.write_text(json.dumps(state))
        return Result("success", (ContentPart("text", f"material update {step} committed"),))

    async def prepare(_snapshot):
        state = json.loads(material_path.read_text())
        reminders = [] if live_only else [Reminder("memory", state["replay"], 1)]
        if state["live"] is not None:
            reminders.append(Reminder("environment", state["live"], 2, replay=False))
        return material_data(Materials("system", tuple(reminders)))

    # 1. 真实 Input、工具提交、调用账与 Output 贯穿同一次执行。
    async with runtime(path, complete, invoke, max_steps=10, material_source=prepare) as (source, log, _store, run):
        await source.accept("first-input", Input((ContentPart("text", "执行材料切换"),)))
        task = await source.start(run)
        assert task is not None
        await task.join()
        check(len(requests) == last_step, "模型完成全部计划步骤")
        rows = [plain(request.messages) for request in requests]
        text = [json.dumps(row, ensure_ascii=False) for row in rows]
        check(rows[1][:len(rows[0])] == rows[0], "不变材料保留完整上一请求前缀")
        check(all(request.continuation is None for request in requests[:2]), "保留原有实时材料请求的续接策略")
        if live_only:
            check("LIVE_X" not in text[2], "没有可回放材料时也能撤下实时材料")
            check(rows[3][:len(rows[2])] == rows[2], "撤下实时材料后前缀保持稳定")
        else:
            check("LIVE_X" not in text[2] and text[2].count("LIVE_Y") == 1, "实时正文使用新值且只出现一次")
            check(text[3].count("REMINDER_A") == text[3].count("REMINDER_B") == 1, "变化材料保留各自首次事实")
            check("REMINDER_B" in str(rows[3][-2]) and "LIVE_Y" in str(rows[3][-1]), "新材料追加到变化后的请求边界")
            check(rows[4][:len(rows[3])] == rows[3], "变化后的材料固定位置")
            check("LIVE_" not in text[5], "撤下材料不从历史恢复正文")
            check(rows[6][:len(rows[5])] == rows[5], "撤下后新的请求继续扩展前缀")
            check(requests[6].continuation == ModelContinuation("model", {"step": 6}), "完整可回放请求仍可接续供应商状态")
        facts = [cast(Mapping[str, object], part.value) for message in log.reader("s").snapshot()
                 if isinstance(message.body, Output) for part in message.body.parts
                 if isinstance(part, ContentPart) and part.kind == "model.facts"]
        check("LIVE_" not in str(facts), "实时材料不进入持久 replay")
        check(all(value["continuation"] is not None for value in facts[:2]), "保留供应商返回的原有协议续接事实")
        check(all(value["thinking"] for value in facts), "模型原有思考正文完整保留")
        before = stored_rows(path / "sessions.db")

    # 2. 关闭所有组件后，使用同一数据库重新装配；新 Input 拥有新身份。
    async with runtime(path, complete, invoke, max_steps=2, material_source=prepare) as (source, _log, _store, run):
        check(stored_rows(path / "sessions.db") == before, "重新装配保留完整原 Message 行")
        await source.accept("second-input", Input((ContentPart("text", "重新启动后继续"),)))
        task = await source.start(run)
        assert task is not None
        await task.join()
        after = stored_rows(path / "sessions.db")
        check(after[:len(before)] == before, "后续执行只追加消息")
        last_rows = plain(requests[-1].messages)
        check(last_rows[:len(rows[-1])] == rows[-1], "重启后重建相同历史前缀")
        if not live_only:
            check(json.dumps(last_rows).count("REMINDER_B") == 2, "相同正文在不同 Input 下保留独立身份")
    return {"checks": checks, "requests": len(requests), "live_only": live_only}


def main():
    """输出独立验收记录；只使用一次性目录，不调用外部 provider。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    with tempfile.TemporaryDirectory(prefix="akashic-context-reminder-") as directory:
        root = Path(directory)
        result = {"replay_and_live": asyncio.run(scenario(root / "replay")),
                  "live_only": asyncio.run(scenario(root / "live", live_only=True)),
                  "limits": ["Controlled provider; no external model or billing evidence",
                             "Real Message, ReAct, model ledger, tool execution and component restart",
                             "Runtime plugin uninstall and compaction require separate installed-runtime checks"]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: len(value["checks"]) for key, value in result.items() if isinstance(value, dict)}))


if __name__ == "__main__":
    main()
