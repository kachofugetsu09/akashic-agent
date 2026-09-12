from contextlib import closing
import sqlite3


def snapshot(path):
    """Return a stable SQLite dump for read-only persistence assertions."""
    with closing(sqlite3.connect(path)) as connection:
        return "\n".join(connection.iterdump())
