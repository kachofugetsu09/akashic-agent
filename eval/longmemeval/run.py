"""CLI entry point for the LongMemEval benchmark.

Usage:
  # Full run:
  python -m eval.longmemeval.run \\
      --config eval/longmemeval/config.toml \\
      --data eval/longmemeval/data/longmemeval_akashic.json \\
      --workspace /tmp/lme_bench

  # 2 concurrent workers:
  python -m eval.longmemeval.run ... --workers 2

  # Resume (skip already-ingested):
  python -m eval.longmemeval.run ... --resume

  # QA only / ingest only:
  python -m eval.longmemeval.run ... --qa-only
  python -m eval.longmemeval.run ... --ingest-only

  # Smoke-test with first N instances:
  python -m eval.longmemeval.run ... --limit 5

Results are written to --output (default: eval/longmemeval/results/<timestamp>.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger("eval.longmemeval")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run LongMemEval benchmark against the akashic agent runtime."
    )
    p.add_argument("--config", required=True, type=Path, help="Path to config.toml")
    p.add_argument("--data", required=True, type=Path,
                   help="Path to longmemeval_akashic.json")
    p.add_argument("--workspace", type=Path, default=Path("/tmp/lme_bench"),
                   help="Workspace directory (created on first run)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON (default: eval/longmemeval/results/<ts>.json)")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process the first N instances (0 = all)")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent workers (default: 1)")
    p.add_argument("--resume", action="store_true",
                   help="Skip ingest for already-ingested instances")
    p.add_argument("--resume-auto", action="store_true",
                   help="Auto-resume: reuse per-instance results, else QA-only on existing memory, else ingest+QA")
    p.add_argument("--qa-only", action="store_true",
                   help="Skip ingest entirely")
    p.add_argument("--ingest-only", action="store_true",
                   help="Run ingest + consolidation only, skip QA")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="Per-question agent timeout in seconds (default: 180)")
    p.add_argument("--type", dest="question_type", default=None,
                   help="Filter to a specific question_type (e.g. single-session-preference)")
    return p


# ── rich display helpers ──────────────────────────────────────────────────────

def _make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
        transient=False,
    )


def _judge_str(jc) -> str:
    if jc is None:
        return "—"
    return "✅" if jc else "❌"


def _f1_str(f1: float) -> str:
    icon = "✅" if f1 >= 0.8 else ("⚠" if f1 >= 0.3 else "✗")
    return f"{icon} {f1:.2f}"


def _instance_result_path(workspace: Path) -> Path:
    return workspace / "result.json"


def _load_instance_result(workspace: Path) -> dict | None:
    path = _instance_result_path(workspace)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("failed to load cached result: %s", path)
        return None


def _save_instance_result(workspace: Path, result: dict) -> None:
    _instance_result_path(workspace).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _has_ingested_session(workspace: Path, question_id: str) -> bool:
    state_path = workspace / "ingest_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            return bool(state.get("completed") is True)
        except Exception:
            logger.warning("failed to inspect ingest state: %s", state_path)
    return False


def _workspace_has_partial_data(workspace: Path, question_id: str) -> bool:
    db_path = workspace / "sessions.db"
    if not db_path.exists():
        return False
    session_key = f"lme:{question_id}"
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "select count(*) from messages where session_key = ?",
                (session_key,),
            ).fetchone()
            return bool(row and int(row[0]) > 0)
        finally:
            conn.close()
    except Exception:
        logger.warning("failed to inspect sessions db: %s", db_path)
        return False


def _reset_instance_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)


# ── per-instance worker ───────────────────────────────────────────────────────

async def _process_instance(
    inst,
    *,
    args,
    judge_model: str,
    sem: asyncio.Semaphore,
    progress,
    overall_task,
    worker_task,
    console,
    results: list,
    counter: list,
    t_start: float,
) -> None:
    from agent.config import load_config
    from .ingest import ingest_instance
    from .metrics import judge_answer, token_f1
    from .qa_runner import format_tool_trace, run_qa_instance
    from .runtime import close_runtime, create_runtime

    async with sem:
        inst_workspace = args.workspace / inst.question_id
        inst_workspace.mkdir(parents=True, exist_ok=True)

        short_id = inst.question_id[:8]
        qt = inst.question_type

        if args.resume_auto:
            cached = _load_instance_result(inst_workspace)
            if cached is not None:
                results.append(cached)
                counter.append(1)
                progress.update(worker_task, description=f"[cyan]{short_id}[/]  cached",
                                completed=1, total=1)
                progress.update(overall_task, advance=1)
                judged = [r for r in results if r.get("judge_correct") is not None]
                if judged:
                    acc = sum(1 for r in judged if r["judge_correct"]) / len(judged)
                    f1avg = sum(token_f1(r["predicted_answer"], r["gold_answer"]) for r in results) / len(results)
                    progress.update(
                        overall_task,
                        description=f"[bold]Overall[/]  judge={acc:.0%}  F1={f1avg:.2f}",
                    )
                progress.update(worker_task, description="[dim]idle[/]", completed=0, total=1)
                return

        should_ingest = not args.qa_only
        if args.resume_auto:
            should_ingest = not _has_ingested_session(inst_workspace, inst.question_id)
            if should_ingest and _workspace_has_partial_data(inst_workspace, inst.question_id):
                _reset_instance_workspace(inst_workspace)

        rt = await create_runtime(args.config, inst_workspace)
        try:
            # ── Ingest ────────────────────────────────────────────────────────
            if should_ingest:
                n_sessions = len(inst.haystack_sessions)

                def _on_progress(done: int, total: int) -> None:
                    progress.update(
                        worker_task,
                        description=f"[cyan]{short_id}[/]  ingest {done}/{total}",
                        completed=done,
                        total=total,
                    )

                progress.update(worker_task, description=f"[cyan]{short_id}[/]  ingest 0/{n_sessions}",
                                completed=0, total=n_sessions)
                await ingest_instance(rt, inst, force=not args.resume, on_progress=_on_progress)
            elif args.resume_auto:
                progress.update(worker_task, description=f"[cyan]{short_id}[/]  [yellow]qa-only[/]",
                                completed=0, total=1)

            if args.ingest_only:
                progress.update(overall_task, advance=1)
                counter.append(1)
                return

            # ── QA ───────────────────────────────────────────────────────────
            progress.update(worker_task,
                            description=f"[cyan]{short_id}[/]  [yellow]agent[/]",
                            completed=0, total=1)
            result = await run_qa_instance(rt, inst, timeout_s=args.timeout)
            results.append(result)

            # ── Judge ─────────────────────────────────────────────────────────
            if not result["error"]:
                provider = rt.core.provider
                result["judge_correct"] = await judge_answer(
                    provider, judge_model,
                    question=result["question"],
                    gold=result["gold_answer"],
                    predicted=result["predicted_answer"],
                )
            else:
                result["judge_correct"] = None
            if args.resume_auto:
                _save_instance_result(inst_workspace, result)

        finally:
            await close_runtime(rt)

        # ── Print completed result ────────────────────────────────────────────
        if not args.ingest_only:
            f1 = token_f1(result["predicted_answer"], result["gold_answer"])
            jc = result.get("judge_correct")
            counter.append(1)
            n_done = len(counter)
            n_total = args._n_total

            # ETA
            elapsed_total = time.monotonic() - t_start
            avg = elapsed_total / n_done
            eta_s = avg * (n_total - n_done)
            eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.1f}m"

            # Write trace to file (prepend SELF.md so we can verify prompt injection)
            trace = format_tool_trace(result.get("tool_chain") or [])
            self_md_path = inst_workspace / "memory" / "SELF.md"
            self_md_content = self_md_path.read_text(encoding="utf-8") if self_md_path.exists() else "(missing)"
            cfg = rt.core.config
            agent_cfg_text = (
                f"agent_model    = {cfg.agent_model or cfg.model}\n"
                f"agent_base_url = {cfg.agent_base_url or cfg.base_url}\n"
                f"main_model     = {cfg.model}\n"
                f"main_base_url  = {cfg.base_url}\n"
                f"light_model    = {cfg.light_model or '(none)'}\n"
            )
            trace_path = inst_workspace / "trace.log"
            trace_path.write_text(
                f"=== Agent Config ===\n{agent_cfg_text}\n"
                f"=== SELF.md (injected as prompt block) ===\n{self_md_content}\n"
                f"=== ReAct trace ===\n{trace}",
                encoding="utf-8",
            )

            # Panel content
            jstr = _judge_str(jc)
            pred_text = (result["predicted_answer"] or "(empty)")[:120]
            gold_text = (result["gold_answer"] or "")[:120]
            question_text = (result["question"] or "")[:120]

            body = Text()
            body.append("  Q     ", style="dim")
            body.append(question_text + "\n")
            body.append("  pred  ", style="dim")
            pred_style = "bold green" if jc else ("bold red" if jc is False else "bold")
            body.append(pred_text + "\n", style=pred_style)
            body.append("  gold  ", style="dim")
            body.append(gold_text, style="green")
            if result["error"]:
                body.append(f"\n  err   {result['error']}", style="red")

            title = (
                f"[dim][{n_done:03d}/{n_total}][/]  [bold cyan]{short_id}[/]  "
                f"[dim]{qt}[/]"
            )
            subtitle = (
                f"judge={jstr}  {_f1_str(f1)}  "
                f"[dim]{result['elapsed_s']:.0f}s  ETA {eta_str}[/]"
            )
            console.print(Panel(body, title=title, subtitle=subtitle, padding=(0, 1)))
            console.print(f"  [dim]trace  {trace_path}[/]")

            progress.update(overall_task, advance=1)
            judged = [r for r in results if r.get("judge_correct") is not None]
            if judged:
                acc = sum(1 for r in judged if r["judge_correct"]) / len(judged)
                f1avg = sum(token_f1(r["predicted_answer"], r["gold_answer"]) for r in results) / len(results)
                progress.update(
                    overall_task,
                    description=f"[bold]Overall[/]  judge={acc:.0%}  F1={f1avg:.2f}",
                )

        progress.update(worker_task, description="[dim]idle[/]", completed=0, total=1)


# ── main ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    import sys

    from agent.config import load_config
    from .dataset import SUPPORTED_QUESTION_TYPES, load_dataset
    from .metrics import score_results
    from .runtime import close_runtime, create_runtime

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    data_path = args.data
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)

    instances = load_dataset(data_path)
    instances = [i for i in instances if i.question_type in SUPPORTED_QUESTION_TYPES]
    if args.question_type:
        if args.question_type not in SUPPORTED_QUESTION_TYPES:
            choices = ", ".join(SUPPORTED_QUESTION_TYPES)
            print(f"ERROR: unsupported --type {args.question_type!r}; choices: {choices}")
            sys.exit(1)
        instances = [i for i in instances if i.question_type == args.question_type]
    if args.limit > 0:
        instances = instances[: args.limit]
    args._n_total = len(instances)

    base_workspace = args.workspace
    base_workspace.mkdir(parents=True, exist_ok=True)

    bench_config = load_config(args.config)
    judge_model = bench_config.model

    console = Console()
    console.print(Rule(f"[bold]LongMemEval[/]  {len(instances)} instances  workers={args.workers}"))

    results: list[dict] = []
    counter: list = []
    t_start = time.monotonic()

    progress = _make_progress(console)
    with progress:
        overall_task = progress.add_task("[bold]Overall[/]", total=len(instances))
        worker_tasks = [
            progress.add_task(f"[dim]Worker {i+1}  idle[/]", total=1, completed=0)
            for i in range(args.workers)
        ]

        sem = asyncio.Semaphore(args.workers)

        async def _run_with_worker(inst, wt):
            return await _process_instance(
                inst,
                args=args,
                judge_model=judge_model,
                sem=sem,
                progress=progress,
                overall_task=overall_task,
                worker_task=wt,
                console=console,
                results=results,
                counter=counter,
                t_start=t_start,
            )

        # Round-robin worker task assignment
        coros = [
            _run_with_worker(inst, worker_tasks[i % args.workers])
            for i, inst in enumerate(instances)
        ]
        await asyncio.gather(*coros)

    if args.ingest_only:
        console.print("[green]Ingest-only complete.[/]")
        return

    # ── Final scores ─────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    scores = score_results(results)
    ov = scores["overall"]

    judged = [r for r in results if r.get("judge_correct") is not None]
    judge_acc = sum(1 for r in judged if r["judge_correct"]) / len(judged) if judged else 0.0

    table = Table(title=f"Results  —  elapsed {elapsed/3600:.1f}h", show_header=True,
                  header_style="bold", min_width=70)
    table.add_column("Question Type", style="cyan", min_width=32)
    table.add_column("judge", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("n", justify="right")
    table.add_column("errors", justify="right")

    table.add_row(
        "[bold]Overall[/]",
        f"[bold]{judge_acc:.1%}[/]",
        f"[bold]{ov['f1']:.4f}[/]",
        f"[bold]{ov['em']:.4f}[/]",
        str(ov["n"]),
        str(ov["errors"]),
        end_section=True,
    )
    for qt, s in sorted(scores["by_type"].items()):
        qt_judged = [r for r in results if r.get("question_type") == qt
                     and r.get("judge_correct") is not None]
        qt_acc = sum(1 for r in qt_judged if r["judge_correct"]) / len(qt_judged) if qt_judged else 0.0
        table.add_row(qt, f"{qt_acc:.1%}", f"{s['f1']:.4f}", f"{s['em']:.4f}",
                      str(s["n"]), str(s.get("errors", 0)))

    console.print(table)

    # ── Save ─────────────────────────────────────────────────────────────────
    output = args.output
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        output = results_dir / f"{ts}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "data": str(args.data),
        "workspace": str(base_workspace),
        "workers": args.workers,
        "scores": scores,
        "judge_acc": judge_acc,
        "results": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n  Saved → [bold]{output}[/]")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
