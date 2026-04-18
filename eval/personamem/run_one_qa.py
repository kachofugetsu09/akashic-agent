from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .dataset import load_dataset
from .metrics import extract_option_label
from .qa_runner import format_tool_trace, run_qa_instance
from .runtime import close_runtime, create_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QA only for one PersonaMem instance.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


async def _run(args: argparse.Namespace) -> None:
    instances = load_dataset(args.questions, args.contexts)
    inst = next((item for item in instances if item.question_id == args.question_id), None)
    if inst is None:
        raise SystemExit(f"question_id not found: {args.question_id}")

    rt = await create_runtime(args.config, args.workspace, inst.persona_profile)
    try:
        result = await run_qa_instance(rt, inst, timeout_s=args.timeout)
        result["predicted_label"] = extract_option_label(
            result["predicted_answer"], result["all_options"]
        )
        result["is_correct"] = result["predicted_label"] == result["gold_answer"]
    finally:
        await close_runtime(rt)

    print(f"question_id: {result['question_id']}")
    print(f"predicted : {result['predicted_answer']}")
    print(f"pick      : {result['predicted_label']}")
    print(f"gold      : {result['gold_answer']}")
    print(f"correct   : {result['is_correct']}")
    print(f"error     : {result['error']}")
    print("--- TRACE ---")
    print(format_tool_trace(result.get("tool_chain") or []))
    print("--- JSON ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
