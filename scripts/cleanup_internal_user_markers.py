from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
    )
    _ = parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    db_path = args.workspace / "sessions.db"
    if not db_path.exists():
        raise SystemExit(f"sessions.db 不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT id, session_key, seq, content
        FROM messages
        WHERE role = 'user' AND content LIKE '[后台任务完成]%'
        ORDER BY session_key, seq
        """
    ).fetchall()
    print(f"命中 {len(rows)} 条内部完成 marker。")
    for message_id, session_key, seq, content in rows:
        print(f"{message_id}\t{session_key}\t{seq}\t{content}")

    if not args.apply or not rows:
        conn.close()
        return

    backup_path = db_path.with_suffix(".db.before-internal-marker-cleanup")
    shutil.copy2(db_path, backup_path)
    _ = conn.execute(
        """
        DELETE FROM messages
        WHERE role = 'user' AND content LIKE '[后台任务完成]%'
        """
    )
    conn.commit()
    conn.close()
    print(f"已清理，备份: {backup_path}")


if __name__ == "__main__":
    main()
