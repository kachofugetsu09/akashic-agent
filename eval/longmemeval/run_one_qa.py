from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.config import load_config

from .dataset import load_dataset
from .metrics import judge_answer, token_f1
from .qa_runner import format_tool_trace, run_qa_instance
from .runtime import close_runtime, create_runtime


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run QA only for one LongMemEval instance.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--question-id", required=True)
    p.add_argument("--timeout", type=float, default=180.0)
    return p


async def _run(args: argparse.Namespace) -> None:
    instances = load_dataset(args.data)
    inst = next((x for x in instances if x.question_id == args.question_id), None)
    if inst is None:
        raise SystemExit(f"question_id not found: {args.question_id}")

    rt = await create_runtime(args.config, args.workspace)
    try:
        result = await run_qa_instance(rt, inst, timeout_s=args.timeout)
        cfg = load_config(args.config)
        result["judge_correct"] = await judge_answer(
            rt.core.provider,
            cfg.model,
            question=result["question"],
            gold=result["gold_answer"],
            predicted=result["predicted_answer"],
        )
    finally:
        await close_runtime(rt)

    print(f"question_id: {result['question_id']}")
    print(f"predicted : {result['predicted_answer']}")
    print(f"gold      : {result['gold_answer']}")
    print(f"f1        : {token_f1(result['predicted_answer'], result['gold_answer']):.4f}")
    print(f"judge     : {result['judge_correct']}")
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
