"""SQLite helpers for Unclaw local persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def open_connection(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection configured for dictionary-like row access."""

    path = Path(database_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the base SQLite schema used by the runtime."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_active INTEGER NOT NULL CHECK (is_active IN (0, 1))
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
        ON sessions (updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_created_at
        ON messages (session_id, created_at);

        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            event_type TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_events_session_created_at
        ON events (session_id, created_at DESC);
        """
    )
    connection.commit()
