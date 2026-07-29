"""Orchestrate one deterministic causal memory rebuild."""

from __future__ import annotations

import hashlib
import json
import resource
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cycle import MemoryCycle
from ..domain.features import BurstAwareFeaturePool
from ..domain.graph import DynamicMemoryGraph
from ..domain.model import (
    Capture,
    ContextState,
    MemoryConfig,
    PlasticityResult,
    SeedEvidence,
)
from ..infrastructure.loader import load_turns
from ..infrastructure.persistence import (
    canonical_json,
    logical_state_sha256,
    sha256_file,
    write_memory_database,
)

DEFAULT_TARGET_SEQUENCES = (
    7877,
    10306,
    8566,
    9224,
    8464,
    9892,
    4740,
    9624,
    9710,
    5294,
    3011,
)
DEFAULT_TARGET_SESSION = "telegram:7674283004"


@dataclass(frozen=True)
class RebuildSummary:
    """Report deterministic output identity and non-deterministic runtime cost."""

    turns: int
    sessions: int
    hubs: int
    relations: int
    targets: int
    elapsed_seconds: float
    peak_rss_kib: int
    database_sha256: str
    logical_state_sha256: str
    progress: tuple[dict[str, float | int], ...]


def rebuild_memory(
    index_path: Path,
    output_path: Path,
    *,
    run_report_path: Path | None = None,
    config: MemoryConfig = MemoryConfig(),
    target_sequences: tuple[int, ...] = DEFAULT_TARGET_SEQUENCES,
    target_session: str = DEFAULT_TARGET_SESSION,
    max_turns: int | None = None,
) -> RebuildSummary:
    """Replay all causal turns, persist graph state, and return its identity."""

    # 1. Validate immutable inputs and construct the single causal state machine.
    config.validate()
    started = time.perf_counter()
    turns = load_turns(index_path, max_turns=max_turns)
    cycle = MemoryCycle(
        config,
        turn_capacity=len(turns),
        feature_pool=BurstAwareFeaturePool(turns),
    )
    targets = _target_nodes(turns, target_sequences, target_session, max_turns)

    # 2. Run the single read-before-write event state machine.
    events, evidence, captures, context, progress = _replay(
        turns,
        cycle,
        targets,
        started,
    )

    # 3. Write deterministic state separately from runtime measurements.
    metadata = deterministic_metadata(index_path)
    database_hash = write_memory_database(
        output_path,
        turns=turns,
        graph=cycle.graph,
        events=events,
        evidence=evidence,
        captures=captures,
        context=context,
        burst_members=cycle.burst_members,
        config=config,
        metadata=metadata,
        recalls=cycle.recalls,
    )
    summary = RebuildSummary(
        turns=len(turns),
        sessions=len({turn.session_key for turn in turns}),
        hubs=len(cycle.graph.hubs),
        relations=len(cycle.graph.source),
        targets=len(captures),
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        database_sha256=database_hash,
        logical_state_sha256=logical_state_sha256(output_path),
        progress=tuple(progress),
    )
    if run_report_path is not None:
        _write_run_report(run_report_path, summary, config, metadata)
    return summary


def _replay(
    turns: list,
    cycle: MemoryCycle,
    targets: set[int],
    started: float,
) -> tuple[
    list[PlasticityResult],
    list[SeedEvidence],
    list[Capture],
    ContextState,
    list[dict[str, float | int]],
]:
    captures: list[Capture] = []
    progress: list[dict[str, float | int]] = []
    for event, turn in enumerate(turns):
        ticket = cycle.retrieve(
            turn,
            capture_paths=event in targets,
            include_completion=True,
        )
        committed = cycle.commit(
            turn,
            ticket,
        )
        if event in targets:
            captures.append(
                Capture(event, ticket.evidence, ticket.diffusion)
            )
        if committed.diffusion.pushes >= 100_000:
            print(
                canonical_json(
                    {
                        "hot_event": event,
                        "pushes": committed.diffusion.pushes,
                        "seed_support": len(committed.evidence.seed),
                    }
                ),
                flush=True,
            )
        _record_progress(progress, event, turns, cycle.graph, started)
    if cycle.context is None:
        raise RuntimeError("memory rebuild produced no context")
    return (
        list(cycle.events),
        list(cycle.evidence),
        captures,
        cycle.context,
        progress,
    )


def _target_nodes(
    turns: list,
    sequences: tuple[int, ...],
    session: str,
    max_turns: int | None,
) -> set[int]:
    requested = set(sequences)
    targets = {
        turn.node_id
        for turn in turns
        if turn.session_key == session and turn.user_seq in requested
    }
    found = {turn.user_seq for turn in turns if turn.node_id in targets}
    missing = sorted(requested - found)
    if missing and max_turns is None:
        raise ValueError(f"target sequences not found: {missing}")
    return targets


def _record_progress(
    progress: list[dict[str, float | int]],
    event: int,
    turns: list,
    graph: DynamicMemoryGraph,
    started: float,
) -> None:
    completed = event + 1
    if completed % 500 != 0 and completed != len(turns):
        return
    item = {
        "turns": completed,
        "hubs": len(graph.hubs),
        "relations": len(graph.source),
        "seconds": round(time.perf_counter() - started, 3),
    }
    progress.append(item)
    print(canonical_json(item), flush=True)


def deterministic_metadata(index_path: Path) -> dict[str, str]:
    return {
        "code_sha256": _package_hash(),
        "git_commit": _git_commit(),
        "numpy_version": np.__version__,
        "source_index_sha256": sha256_file(index_path),
        **_index_identity(index_path),
    }


def _package_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*.py"),
        key=lambda item: str(item.relative_to(root)),
    ):
        digest.update(
            str(path.relative_to(root)).encode("utf-8")
        )
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _index_identity(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        values = dict(
            connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        )
    finally:
        connection.close()
    return {
        f"sparse_index_{key}": values[key]
        for key in (
            "embedding_model",
            "index_version",
            "jieba_dictionary_sha256",
            "jieba_version",
            "lexical_normalizer_version",
            "turns_missing_embeddings",
        )
        if key in values
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _write_run_report(
    path: Path,
    summary: RebuildSummary,
    config: MemoryConfig,
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "config": asdict(config),
        "deterministic_metadata": metadata,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
