"""把 AKASHIC_TIMING=1 + AKASHIC_LOG_FORMAT=json 的启动日志切成逐轮阶段表。

用法：python phases.py <startup.log> [--session <会话 id 后缀>]
每轮从 web.input.received 开始，到第一个 model.driver.begin 结束；同一阶段只取首次出现，
其他会话的阶段不计入；无 session_id 的模型阶段只在本轮 model.begin 之后计入。
"""

import argparse
import json
from collections.abc import Iterator


def _events(path: str) -> Iterator[tuple[int, str, str, str]]:
    with open(path, encoding="utf-8", errors="replace") as log:
        for line in log:
            if '"runtime.timing"' not in line:
                continue
            try:
                row = json.loads(line[line.index("{"):])
            except ValueError:
                continue
            yield (int(row["measurement_value"]), str(row.get("phase", "")),
                   str(row.get("session_id") or ""), str(row.get("source") or ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--session", default="")
    options = parser.parse_args()
    turns: list[list[tuple[int, str, str, str]]] = []
    for event in sorted(_events(options.log)):
        if event[1] == "web.input.received":
            turns.append([event])
        elif turns:
            owner = next((session for _, _, session, _ in turns[-1] if session), "")
            if not event[2] or not owner or event[2] == owner:
                turns[-1].append(event)
    for turn in turns:
        sessions = {session for _, _, session, _ in turn if session}
        if options.session and not any(session.endswith(options.session) for session in sessions):
            continue
        start, seen, previous = turn[0][0], set[str](), turn[0][0]
        print("----", next(iter(sessions), "-"))
        for at, phase, session, source in turn:
            # 无 session 的模型阶段只在本轮 model.begin 之后才属于本轮，之前的是后台调用。
            if phase in seen or (not session and phase.startswith("model.") and "model.begin" not in seen):
                continue
            seen.add(phase)
            print(f"{(at - start) / 1e6:9.0f}ms  +{(at - previous) / 1e6:7.0f}ms  {phase:28s} {source}")
            previous = at
            if phase == "model.driver.begin":
                break


if __name__ == "__main__":
    main()
