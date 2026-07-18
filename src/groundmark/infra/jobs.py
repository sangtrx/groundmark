from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundmark.models import ParseOptions


@dataclass(slots=True)
class Job:
    id: str
    status: str
    input_path: str
    filename: str
    media_type: str | None
    options: ParseOptions
    result_path: str | None
    error: str | None
    attempts: int
    created_at: float
    updated_at: float

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "filename": self.filename,
            "media_type": self.media_type,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


class JobStore:
    def __init__(self, database_path: Path, lease_seconds: int = 900) -> None:
        self.database_path = database_path
        self.lease_seconds = lease_seconds
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    media_type TEXT,
                    options_json TEXT NOT NULL,
                    result_path TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)"
            )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            status=row["status"],
            input_path=row["input_path"],
            filename=row["filename"],
            media_type=row["media_type"],
            options=ParseOptions.model_validate_json(row["options_json"]),
            result_path=row["result_path"],
            error=row["error"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def enqueue(
        self,
        input_path: Path,
        filename: str,
        media_type: str | None,
        options: ParseOptions,
        job_id: str | None = None,
    ) -> Job:
        job_id = job_id or "job_" + uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(id,status,input_path,filename,media_type,options_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    "queued",
                    str(input_path),
                    filename,
                    media_type,
                    options.model_dump_json(),
                    now,
                    now,
                ),
            )
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Job | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def claim(self) -> Job | None:
        now = time.time()
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status='queued' OR (status='processing' AND lease_until < ?)
                    ORDER BY created_at ASC LIMIT 1
                    """,
                    (now,),
                ).fetchone()
                if not row:
                    connection.execute("COMMIT")
                    return None
                connection.execute(
                    """
                    UPDATE jobs SET status='processing', attempts=attempts+1,
                    lease_until=?, updated_at=?, error=NULL WHERE id=?
                    """,
                    (now + self.lease_seconds, now, row["id"]),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(row["id"])

    def complete(self, job_id: str, result_path: Path) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status='succeeded', result_path=?, lease_until=NULL,
                updated_at=?, error=NULL WHERE id=?
                """,
                (str(result_path), now, job_id),
            )

    def delete(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def fail(self, job_id: str, error: str, retry: bool = False) -> None:
        now = time.time()
        status = "queued" if retry else "failed"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status=?, error=?, lease_until=NULL, updated_at=? WHERE id=?
                """,
                (status, error[:4000], now, job_id),
            )
