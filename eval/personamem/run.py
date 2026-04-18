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

logger = logging.getLogger("eval.personamem")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PersonaMem benchmark against the akashic agent runtime."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to config.toml")
    parser.add_argument("--questions", required=True, type=Path, help="Path to questions_*.csv")
    parser.add_argument(
        "--contexts",
        required=True,
        type=Path,
        help="Path to shared_contexts_*.jsonl",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/personamem_bench"),
        help="Workspace directory (created on first run)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-auto", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--type", dest="question_type", default=None)
    return parser


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


def _has_ingested_session(workspace: Path) -> bool:
    state_path = workspace / "ingest_state.json"
    if not state_path.exists():
        return False
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
    session_key = f"pm:{question_id}"
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


async def _process_instance(
    inst,
    *,
    args,
    sem: asyncio.Semaphore,
    progress,
    overall_task,
    worker_task,
    console,
    results: list,
    counter: list,
    t_start: float,
) -> None:
    from eval.longmemeval.ingest import ingest_instance

    from .metrics import extract_option_label
    from .qa_runner import format_tool_trace, run_qa_instance
    from .runtime import close_runtime, create_runtime

    async with sem:
        inst_workspace = args.workspace / inst.question_id
        inst_workspace.mkdir(parents=True, exist_ok=True)
        short_id = inst.question_id[:8]

        if args.resume_auto:
            cached = _load_instance_result(inst_workspace)
            if cached is not None:
                results.append(cached)
                counter.append(1)
                progress.update(worker_task, description=f"[cyan]{short_id}[/]  cached", completed=1, total=1)
                progress.update(overall_task, advance=1)
                progress.update(worker_task, description="[dim]idle[/]", completed=0, total=1)
                return

        should_ingest = not args.qa_only
        if args.resume_auto:
            should_ingest = not _has_ingested_session(inst_workspace)
            if should_ingest and _workspace_has_partial_data(inst_workspace, inst.question_id):
                _reset_instance_workspace(inst_workspace)

        rt = await create_runtime(args.config, inst_workspace, inst.persona_profile)
        try:
            if should_ingest:
                n_sessions = len(inst.haystack_sessions)

                def _on_progress(done: int, total: int) -> None:
                    progress.update(
                        worker_task,
                        description=f"[cyan]{short_id}[/]  ingest {done}/{total}",
                        completed=done,
                        total=total,
                    )

                progress.update(worker_task, description=f"[cyan]{short_id}[/]  ingest 0/{n_sessions}", completed=0, total=n_sessions)
                await ingest_instance(rt, inst, force=not args.resume, on_progress=_on_progress)
            elif args.resume_auto:
                progress.update(worker_task, description=f"[cyan]{short_id}[/]  [yellow]qa-only[/]", completed=0, total=1)

            if args.ingest_only:
                progress.update(overall_task, advance=1)
                counter.append(1)
                return

            progress.update(worker_task, description=f"[cyan]{short_id}[/]  [yellow]agent[/]", completed=0, total=1)
            result = await run_qa_instance(rt, inst, timeout_s=args.timeout)
            predicted_label = extract_option_label(result["predicted_answer"], result["all_options"])
            result["predicted_label"] = predicted_label
            result["is_correct"] = bool(
                not result["error"] and predicted_label == result["gold_answer"]
            )
            results.append(result)
            if args.resume_auto:
                _save_instance_result(inst_workspace, result)
        finally:
            await close_runtime(rt)

        if not args.ingest_only:
            counter.append(1)
            n_done = len(counter)
            n_total = args._n_total
            elapsed_total = time.monotonic() - t_start
            avg = elapsed_total / n_done
            eta_s = avg * (n_total - n_done)
            eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.1f}m"

            trace = format_tool_trace(result.get("tool_chain") or [])
            trace_path = inst_workspace / "trace.log"
            self_md_path = inst_workspace / "memory" / "SELF.md"
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
            trace_path.write_text(
                f"=== Agent Config ===\n{agent_cfg_text}\n"
                f"=== SELF.md (injected as prompt block) ===\n{self_md_content}\n"
                f"=== Final Answer ===\n{result.get('predicted_answer') or '(empty)'}\n\n"
                f"=== ReAct trace ===\n{trace}\n\n"
                f"=== Raw Tool Chain JSON ===\n"
                f"{json.dumps(result.get('tool_chain') or [], ensure_ascii=False, indent=2)}\n",
                encoding="utf-8",
            )

            body = Text()
            body.append("  Q     ", style="dim")
            body.append((result["question"] or "")[:120] + "\n")
            body.append("  pred  ", style="dim")
            pred_style = "bold green" if result["is_correct"] else "bold red"
            body.append((result["predicted_answer"] or "(empty)")[:120] + "\n", style=pred_style)
            body.append("  pick  ", style="dim")
            body.append(f"{result.get('predicted_label') or '(parse-failed)'}\n", style=pred_style)
            body.append("  gold  ", style="dim")
            body.append(f"{result['gold_answer']}  {result['gold_option'][:90]}", style="green")
            if result["error"]:
                body.append(f"\n  err   {result['error']}", style="red")

            title = f"[dim][{n_done:03d}/{n_total}][/]  [bold cyan]{short_id}[/]  [dim]{inst.question_type}[/]"
            subtitle = (
                f"acc={'✅' if result['is_correct'] else '❌'}  "
                f"[dim]{result['elapsed_s']:.0f}s  ETA {eta_str}[/]"
            )
            console.print(Panel(body, title=title, subtitle=subtitle, padding=(0, 1)))
            console.print(f"  [dim]trace  {trace_path}[/]")
            progress.update(overall_task, advance=1)

        progress.update(worker_task, description="[dim]idle[/]", completed=0, total=1)


async def _run(args: argparse.Namespace) -> None:
    import sys

    from .dataset import SUPPORTED_QUESTION_TYPES, load_dataset
    from .metrics import score_results

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    if not args.questions.exists():
        print(f"ERROR: questions file not found: {args.questions}")
        sys.exit(1)
    if not args.contexts.exists():
        print(f"ERROR: contexts file not found: {args.contexts}")
        sys.exit(1)

    instances = load_dataset(args.questions, args.contexts)
    instances = [item for item in instances if item.question_type in SUPPORTED_QUESTION_TYPES]
    if args.question_type:
        if args.question_type not in SUPPORTED_QUESTION_TYPES:
            choices = ", ".join(SUPPORTED_QUESTION_TYPES)
            print(f"ERROR: unsupported --type {args.question_type!r}; choices: {choices}")
            sys.exit(1)
        instances = [item for item in instances if item.question_type == args.question_type]
    if args.limit > 0:
        instances = instances[: args.limit]
    args._n_total = len(instances)

    args.workspace.mkdir(parents=True, exist_ok=True)
    console = Console()
    console.print(Rule(f"[bold]PersonaMem[/]  {len(instances)} instances  workers={args.workers}"))

    results: list[dict] = []
    counter: list[int] = []
    t_start = time.monotonic()

    progress = _make_progress(console)
    with progress:
        overall_task = progress.add_task("[bold]Overall[/]", total=len(instances))
        worker_tasks = [
            progress.add_task(f"[dim]Worker {index + 1}  idle[/]", total=1, completed=0)
            for index in range(args.workers)
        ]
        sem = asyncio.Semaphore(args.workers)

        async def _run_with_worker(inst, worker_task):
            return await _process_instance(
                inst,
                args=args,
                sem=sem,
                progress=progress,
                overall_task=overall_task,
                worker_task=worker_task,
                console=console,
                results=results,
                counter=counter,
                t_start=t_start,
            )

        coros = [
            _run_with_worker(inst, worker_tasks[index % args.workers])
            for index, inst in enumerate(instances)
        ]
        await asyncio.gather(*coros)

    if args.ingest_only:
        console.print("[green]Ingest-only complete.[/]")
        return

    elapsed = time.monotonic() - t_start
    scores = score_results(results)
    overall = scores["overall"]

    table = Table(title=f"Results  —  elapsed {elapsed/3600:.1f}h", show_header=True, header_style="bold", min_width=70)
    table.add_column("Question Type", style="cyan", min_width=40)
    table.add_column("acc", justify="right")
    table.add_column("parsed", justify="right")
    table.add_column("n", justify="right")
    table.add_column("errors", justify="right")
    table.add_row(
        "[bold]Overall[/]",
        f"[bold]{overall['accuracy']:.1%}[/]",
        f"[bold]{overall['parsed_rate']:.1%}[/]",
        str(overall["n"]),
        str(overall["errors"]),
        end_section=True,
    )
    for question_type, score in sorted(scores["by_type"].items()):
        table.add_row(
            question_type,
            f"{score['accuracy']:.1%}",
            f"{score['parsed_rate']:.1%}",
            str(score["n"]),
            str(score["errors"]),
        )
    console.print(table)

    output = args.output
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        output = results_dir / f"{ts}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "questions": str(args.questions),
        "contexts": str(args.contexts),
        "workspace": str(args.workspace),
        "workers": args.workers,
        "scores": scores,
        "results": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n  Saved → [bold]{output}[/]")


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
