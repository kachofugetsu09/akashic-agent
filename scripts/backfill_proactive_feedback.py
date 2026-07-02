from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.net.http import (  # noqa: E402
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from plugins.proactive_feedback.db import FeedbackEvent, insert_feedback, open_db
from plugins.proactive_feedback.scorer import (
    EmbedBatch,
    iter_user_assistant_turns,
    parse_quote_parts,
    proactive_since_previous_user,
    recent_proactive_messages,
    score_followup,
)
from memory2.embedder import Embedder


@dataclass(frozen=True)
class BackfillStats:
    scanned: int = 0
    with_candidates: int = 0
    embedded: int = 0
    inserted: int = 0
    skipped: int = 0
    failed: int = 0


async def _no_embed(texts: list[str]) -> list[list[float]]:
    _ = texts
    raise RuntimeError("quoted feedback must not call embedding")


async def run_backfill(
    *,
    workspace: Path,
    project_root: Path,
    clear: bool,
    limit: int | None,
    dry_run: bool,
    include_pua: bool,
) -> BackfillStats:
    sessions_db = workspace / "sessions.db"
    feedback_db = workspace / "proactive_feedback" / "proactive_feedback.db"
    if not sessions_db.exists():
        raise FileNotFoundError(sessions_db)

    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    if clear and not dry_run:
        _reset_feedback_db(feedback_db)
    source = sqlite3.connect(sessions_db)
    source.row_factory = sqlite3.Row
    sink = open_db(feedback_db)
    try:
        embedder = None
        stats = BackfillStats()
        turns = iter_user_assistant_turns(source)
        if limit is not None:
            turns = turns[:limit]

        scanned = with_candidates = embedded = inserted = skipped = failed = 0
        for session_key, user, assistant in turns:
            scanned += 1
            quote = parse_quote_parts(user.content)
            if quote.quoted_text:
                candidates = recent_proactive_messages(
                    source,
                    session_key=session_key,
                    before_seq=user.seq,
                    limit=64,
                )
                embed_batch = _no_embed
            elif not include_pua:
                skipped += 1
                continue
            else:
                candidates = proactive_since_previous_user(
                    source,
                    session_key=session_key,
                    before_seq=user.seq,
                )
                if candidates:
                    if embedder is None:
                        embedder = _build_embedder(project_root)
                    embed_batch: EmbedBatch = embedder.embed_batch
                    embedded += 1
                else:
                    embed_batch = _no_embed
            if not candidates:
                skipped += 1
                continue
            with_candidates += 1
            try:
                scored = await score_followup(
                    embed_batch=embed_batch,
                    user=user,
                    assistant=assistant,
                    candidates=candidates,
                    allow_pua=not bool(quote.quoted_text),
                )
            except Exception:
                failed += 1
                continue
            if scored is None:
                skipped += 1
                continue
            written = True
            if not dry_run:
                event_id = insert_feedback(
                    sink,
                    FeedbackEvent(
                        session_key=session_key,
                        user_message_id=user.id,
                        assistant_message_id=assistant.id,
                        proactive_message_id=scored.proactive.id,
                        feedback_type=scored.feedback_type,
                        confidence=scored.confidence,
                        pa_score=scored.pa_score,
                        pua_score=scored.pua_score,
                        lag_seconds=scored.lag_seconds,
                        candidate_count=scored.candidate_count,
                        matched_by=scored.matched_by,
                        reason=scored.reason,
                    ),
                )
                written = event_id is not None
            if written:
                inserted += 1
            else:
                skipped += 1
        stats = BackfillStats(
            scanned=scanned,
            with_candidates=with_candidates,
            embedded=embedded,
            inserted=inserted,
            skipped=skipped,
            failed=failed,
        )
    finally:
        source.close()
        sink.close()
        clear_default_shared_http_resources(resources)
        await resources.aclose()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill proactive feedback events from sessions.db")
    _ = parser.add_argument("--workspace", type=Path, default=Path.home() / ".akashic" / "workspace")
    _ = parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    _ = parser.add_argument("--clear", action="store_true", help="clear existing feedback events before writing")
    _ = parser.add_argument("--dry-run", action="store_true", help="score without writing")
    _ = parser.add_argument("--include-pua", action="store_true", help="also score the first user reply after each proactive block")
    _ = parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stats = asyncio.run(
        run_backfill(
            workspace=args.workspace,
            project_root=args.project_root,
            clear=args.clear,
            limit=args.limit,
            dry_run=args.dry_run,
            include_pua=args.include_pua,
        )
    )
    print(
        "scanned={scanned} with_candidates={with_candidates} inserted={inserted} "
        "embedded={embedded} skipped={skipped} failed={failed}".format(
            scanned=stats.scanned,
            with_candidates=stats.with_candidates,
            embedded=stats.embedded,
            inserted=stats.inserted,
            skipped=stats.skipped,
            failed=stats.failed,
        )
    )


def _reset_feedback_db(db_path: Path) -> None:
    for path in (
        db_path,
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm"),
    ):
        path.unlink(missing_ok=True)


def _build_embedder(root: Path) -> Embedder:
    data = tomllib.loads((root / "config.toml").read_text())
    embedding = data["memory"]["embedding"]
    api_key = str(embedding["api_key"])
    if api_key.startswith("$"):
        api_key = os.environ[api_key[1:]]
    return Embedder(
        base_url=str(embedding["base_url"]),
        api_key=api_key,
        model=str(embedding.get("model", "text-embedding-v3")),
        output_dimensionality=embedding.get("output_dimensionality"),
    )


if __name__ == "__main__":
    main()
