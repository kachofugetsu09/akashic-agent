"""Command-line entrypoint for deterministic Akasha rebuilds."""

from __future__ import annotations

import os

for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .application.rebuild import rebuild_memory
from .config import load_akasha_config
from .domain.model import MemoryConfig
from .infrastructure.sparse_index import (
    BuildConfig,
    EmbeddingAudit,
    audit_source_embeddings,
    build_sparse_index,
)


def main() -> None:
    """Validate CLI input, build the causal index, and rebuild memory."""

    # 1. Parse the public sessions-db workflow and the internal index workflow.
    arguments = _parser().parse_args()
    config = _memory_config(arguments)

    # 2. Build from the canonical sessions database or use an explicit index.
    if arguments.sessions_db is not None:
        summary = _rebuild_from_sessions(arguments, config)
    else:
        _backup_existing(arguments.db_path)
        summary = rebuild_memory(
            arguments.index,
            arguments.db_path,
            run_report_path=arguments.run_report,
            config=config,
            target_sequences=tuple(arguments.seq),
            max_turns=arguments.max_turns,
        )

    # 3. Emit one machine-readable completion record.
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild deterministic Akasha explicit memory.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sessions-db", type=Path)
    source.add_argument("--index", type=Path)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-report", type=Path)
    parser.add_argument(
        "--seq",
        nargs="*",
        type=int,
        default=[],
    )
    parser.add_argument("--target-session", default="")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--embedding-report", type=Path)
    parser.add_argument(
        "--require-complete-embeddings",
        action="store_true",
    )
    parser.add_argument("--restart", type=float, default=0.25)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--activation-power", type=float, default=2.0)
    parser.add_argument("--recurrent-budget", type=float, default=1.0)
    parser.add_argument("--reverse-temporal-ratio", type=float, default=0.25)
    parser.add_argument("--disable-forgetting", action="store_true")
    return parser


def _memory_config(arguments: argparse.Namespace) -> MemoryConfig:
    if arguments.config is not None:
        if not arguments.config.is_file():
            raise FileNotFoundError(arguments.config)
        return load_akasha_config(arguments.config).memory_config()
    return MemoryConfig(
        restart=arguments.restart,
        tolerance=arguments.tolerance,
        learning_rate=arguments.learning_rate,
        activation_power=arguments.activation_power,
        recurrent_budget=arguments.recurrent_budget,
        reverse_temporal_ratio=arguments.reverse_temporal_ratio,
        forgetting_enabled=not arguments.disable_forgetting,
    )


def _rebuild_from_sessions(
    arguments: argparse.Namespace,
    config: MemoryConfig,
):
    """Build a private temporary sparse index before atomic memory output."""

    # 1. Audit the immutable source before touching the current sidecar.
    build_config = BuildConfig(
        embedding_model=arguments.embedding_model,
        embedding_dimension=arguments.embedding_dim,
    )
    audit = audit_source_embeddings(
        arguments.sessions_db,
        build_config,
    )
    report_path = arguments.embedding_report
    if report_path is not None or not audit.complete:
        _write_embedding_audit(
            report_path
            or arguments.db_path.with_name(
                f"{arguments.db_path.name}.embedding-audit.json"
            ),
            audit,
            build_config,
        )
    if arguments.require_complete_embeddings and not audit.complete:
        raise ValueError(
            "sessions source has "
            f"{len(audit.issues)} invalid or missing embeddings"
        )

    # 2. Derive the complete index from the audited sessions source.
    with tempfile.TemporaryDirectory(prefix="akasha-rebuild-") as directory:
        index_path = Path(directory) / "sparse-index.db"
        build_sparse_index(
            arguments.sessions_db,
            index_path,
            build_config,
        )

        # 3. Replay the exact same MemoryCycle used by online growth.
        target_sequences = tuple(arguments.seq)
        target_session = arguments.target_session
        if target_sequences and not target_session:
            raise ValueError("--target-session is required when --seq is used")
        _backup_existing(arguments.db_path)
        return rebuild_memory(
            index_path,
            arguments.db_path,
            run_report_path=arguments.run_report,
            config=config,
            target_sequences=target_sequences,
            target_session=target_session,
            max_turns=arguments.max_turns,
        )


def _write_embedding_audit(
    path: Path,
    audit: EmbeddingAudit,
    config: BuildConfig,
) -> None:
    """Atomically publish a machine-readable source preflight report."""

    # 1. Serialize deterministic evidence without runtime timestamps.
    payload = {
        "embedding_model": config.embedding_model,
        "expected_dimension": config.embedding_dimension,
        "audit": asdict(audit),
    }
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # 2. Replace only the requested report after a complete local write.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _backup_existing(path: Path) -> Path | None:
    """Create one recoverable rebuild backup before atomic replacement."""

    if not path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    backup = path.with_name(f"{path.name}.bak-{timestamp}")
    shutil.copy2(path, backup)
    return backup


if __name__ == "__main__":
    main()
