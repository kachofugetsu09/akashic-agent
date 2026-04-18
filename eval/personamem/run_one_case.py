from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from eval.longmemeval.ingest import ingest_instance

from .dataset import load_dataset
from .metrics import extract_option_label
from .qa_runner import format_tool_trace, run_qa_instance
from .runtime import close_runtime, create_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one PersonaMem case end-to-end: ingest, then QA."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--qa-only", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> None:
    instances = load_dataset(args.questions, args.contexts)
    inst = next((item for item in instances if item.question_id == args.question_id), None)
    if inst is None:
        raise SystemExit(f"question_id not found: {args.question_id}")

    total_sessions = len(inst.haystack_sessions)
    total_turns = sum(len(session) for session in inst.haystack_sessions)
    session_turns = [len(session) for session in inst.haystack_sessions]
    t0 = time.monotonic()

    rt = await create_runtime(args.config, args.workspace, inst.persona_profile)
    try:
        if not args.qa_only:
            print(
                f"[ingest] start  sessions={total_sessions}  turns={total_turns}",
                flush=True,
            )

            def _on_progress(done: int, total: int) -> None:
                ingested_turns = sum(session_turns[:done])
                elapsed = time.monotonic() - t0
                print(
                    f"[ingest] {done}/{total} sessions  turns={ingested_turns}/{total_turns}  elapsed={elapsed:.1f}s",
                    flush=True,
                )

            await ingest_instance(rt, inst, force=True, on_progress=_on_progress)
            elapsed = time.monotonic() - t0
            print(f"[ingest] done  elapsed={elapsed:.1f}s", flush=True)
        else:
            print("[ingest] skipped (--qa-only)", flush=True)

        print("[qa] start", flush=True)
        result = await run_qa_instance(rt, inst, timeout_s=args.timeout)
        elapsed = time.monotonic() - t0
        print(f"[qa] done  elapsed={elapsed:.1f}s", flush=True)
        result["predicted_label"] = extract_option_label(
            result["predicted_answer"], result["all_options"]
        )
        result["is_correct"] = result["predicted_label"] == result["gold_answer"]

        trace = format_tool_trace(result.get("tool_chain") or [])
        self_md_path = args.workspace / "memory" / "SELF.md"
        self_md_content = (
            self_md_path.read_text(encoding="utf-8")
            if self_md_path.exists()
            else "(missing)"
        )
        cfg = rt.core.config
        agent_cfg_text = (
            f"agent_model    = {cfg.agent_model or cfg.model}\n"
            f"agent_base_url = {cfg.agent_base_url or cfg.base_url}\n"
            f"main_model     = {cfg.model}\n"
            f"main_base_url  = {cfg.base_url}\n"
            f"light_model    = {cfg.light_model or '(none)'}\n"
        )
        trace_path = args.workspace / "trace.log"
        trace_path.write_text(
            f"=== Agent Config ===\n{agent_cfg_text}\n"
            f"=== SELF.md (injected as prompt block) ===\n{self_md_content}\n"
            f"=== Final Answer ===\n{result.get('predicted_answer') or '(empty)'}\n\n"
            f"=== ReAct trace ===\n{trace}\n\n"
            f"=== Raw Tool Chain JSON ===\n"
            f"{json.dumps(result.get('tool_chain') or [], ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
    finally:
        await close_runtime(rt)

    print(f"question_id: {result['question_id']}")
    print(f"predicted : {result['predicted_answer']}")
    print(f"pick      : {result['predicted_label']}")
    print(f"gold      : {result['gold_answer']}")
    print(f"correct   : {result['is_correct']}")
    print(f"error     : {result['error']}")
    print(f"trace     : {trace_path}")
    print("--- TRACE ---")
    print(trace)
    print("--- JSON ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
