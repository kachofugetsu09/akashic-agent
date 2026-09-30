#!/usr/bin/env python3
"""读取原始结果与人工 Browser 审核记录，不修改原始证据。"""

import argparse
import hashlib
import json
from pathlib import Path


def score(output, review_path=None):
    """模拟 provider、未审核或无效观察均不能进入模型成绩。"""
    manifest = json.loads((output / "manifest.json").read_text())
    rows = json.loads((output / "results.json").read_text())
    if manifest["agent"] is None:
        raise ValueError("Reference solver results are not Agent scores")
    # 1. 人工审核绑定实际轨迹摘要，不能沿用另一轮同名题目的结论。
    review = json.loads(review_path.read_text()) if review_path else {"cases": {}}
    entries = review["cases"]
    labels = {f"{row['task']}-{row['variant']}" for row in rows}
    if not isinstance(entries, dict) or set(entries) - labels:
        raise ValueError("Review contains unknown cases")
    if entries and (
        not isinstance(review["reviewer"], str) or not review["reviewer"].strip()
    ):
        raise ValueError("Review requires a named reviewer")
    for label, entry in entries.items():
        if (
            entry["validity"] not in {"valid", "invalid"}
            or not isinstance(entry["reason"], str)
            or not entry["reason"].strip()
        ):
            raise ValueError(f"Review requires validity and reason: {label}")
        actual = hashlib.sha256(
            (output / label / "agent.json").read_bytes()
        ).hexdigest()
        if entry["agent_sha256"] != actual:
            raise ValueError(f"Review refers to another trajectory: {label}")
    # 2. 先确认运行与清理，再决定观察有效性；原始 reward 始终单列保留。
    cases = []
    for row in rows:
        label = f"{row['task']}-{row['variant']}"
        validity = "valid" if not manifest["agent"]["browser"] else "pending"
        if label in entries:
            validity = entries[label]["validity"]
        eligible = (
            manifest["completed"]
            and manifest["cleanup"] == "passed"
            and manifest["agent"]["provider_kind"] == "real"
            and row["status"] == "evaluated"
            and row["cleanup"] == "passed"
            and validity == "valid"
        )
        cases.append(
            {
                "case": label,
                "validity": validity,
                "eligible": eligible,
                "reward": row.get("reward"),
                "solved": bool(eligible and row["reward"] == [1.0]),
            }
        )
    count = sum(case["eligible"] for case in cases)
    solved = sum(case["solved"] for case in cases)
    return {
        "provider_kind": manifest["agent"]["provider_kind"],
        "cases": cases,
        "eligible": count,
        "solved": solved,
        "success_rate": solved / count if count else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.output, args.review), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
