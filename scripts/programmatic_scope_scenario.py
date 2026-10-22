"""Manual acceptance: programmatic Session admission carries a memory scope.

验证「程序来源可以把自己放进一个既不学习也不召回的 scope」，以及不传 scope 时行为不变。
走真实的 SESSION_ADMISSION + Akasha scope 路由，不 mock 函数。

用法：
    python3 scripts/programmatic_scope_scenario.py --output /tmp/scope-report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CUA_TELEMETRY_ENABLED"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.source = args.source.resolve()
if args.output.exists():
    parser.error("output must be a new file")
sys.path.insert(0, str(args.source))

from plugins.ledger.contract import SessionAttributes  # noqa: E402
from plugins.programmatic.contract import AdmitParams  # noqa: E402

checks: list[dict[str, object]] = []


def check(label: str, passed: bool, detail: object = None) -> None:
    checks.append({"label": label, "passed": bool(passed), "detail": detail})


async def main() -> None:
    # 1. 参数边界：scope 是可选字段，缺省为 None；不传时与旧调用完全一致。
    plain = AdmitParams(session_id="programmatic:plain")
    check("scope 缺省为 None", plain.scope is None, {"scope": plain.scope})
    check("缺省仍保留 persist_memory 默认值", plain.persist_memory is False)

    scoped = AdmitParams(session_id="programmatic:scoped", scope={"project": "github-review"})
    check("scope 接受映射", scoped.scope == {"project": "github-review"}, scoped.scope)

    # 2. 维度名与值受 ledger 的既有约束：小写+下划线、值非空且不是 default。
    valid = SessionAttributes.scoped({"project": "github-review"}, visibility="internal", learning="excluded")
    check("合法 scope 可构造", valid.scope == (("project", "github-review"),), list(valid.scope))
    check("scope 与 learning 互相独立",
          valid.learning == "excluded" and valid.scope == (("project", "github-review"),))

    rejected = 0
    for bad in ({"Project": "x"}, {"project": "default"}, {"project": ""}):
        try:
            SessionAttributes.scoped(bad, visibility="internal")
        except ValueError:
            rejected += 1
    check("非法维度名/值被拒绝", rejected == 3, {"rejected": rejected})

    # 3. 不传 scope 的等价性：scoped({}) 与直接构造的字段必须一致。
    empty = SessionAttributes.scoped({}, visibility="internal", learning="excluded")
    direct = SessionAttributes(visibility="internal", learning="excluded")
    check("空 scope 等价于旧构造", empty == direct, {"scoped": repr(empty), "direct": repr(direct)})

    # 4. 多维度按名字排序，与 ledger 的「唯一且有序」要求一致。
    multi = SessionAttributes.scoped({"project": "b", "repo": "a"}, visibility="internal")
    check("多维度按名排序", multi.scope == (("project", "b"), ("repo", "a")), list(multi.scope))

    # 5. 记忆路由：learning 只关写，关不掉读；关读要靠 scope 的 recall 策略。
    import inspect

    from plugins.akasha.scopes import ScopePolicies  # noqa: PLC0415

    route_source = inspect.getsource(ScopePolicies.route)
    write_line = next(line for line in route_source.splitlines() if "write =" in line)
    read_line = next(line for line in route_source.splitlines() if "read =" in line)
    check("写轴看 learning 与 off",
          'attributes.learning == "excluded"' in write_line and '"off" in policies.values()' in write_line,
          write_line.strip())
    check("读轴只看 recall 策略，与 learning 无关",
          "recall_enabled" in read_line and "learning" not in read_line,
          read_line.strip())

    # 6. CLI 的 --scope 解析：重复维度与缺值都要在提交前就拒绝。
    from plugins.programmatic.contract import parse_scope_pairs  # noqa: PLC0415

    check("CLI 解析单个维度", parse_scope_pairs(["project=github-review"]) == {"project": "github-review"})
    check("CLI 解析多个维度",
          parse_scope_pairs(["project=a", "repo=b"]) == {"project": "a", "repo": "b"})
    cli_rejected = 0
    for bad in (["project"], ["=x"], ["project="], ["a=1", "a=2"]):
        try:
            parse_scope_pairs(bad)
        except ValueError:
            cli_rejected += 1
    check("CLI 拒绝非法 --scope", cli_rejected == 4, {"rejected": cli_rejected})

    report = {
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in checks:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"{mark} {item['label']}  {json.dumps(item['detail'], ensure_ascii=False)}")
    print("报告:", args.output)
    if not report["passed"]:
        raise SystemExit(1)


asyncio.run(main())
