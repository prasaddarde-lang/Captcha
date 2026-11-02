"""SQLite persistence for execution history, verification requests and model registry."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class DatabaseManager:
    """Thin, thread-safe wrapper around a SQLite file.

    A new connection is opened per operation so the manager can be shared across
    threads (FastAPI workers, operator console, RPA subprocesses).
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        self.initialize()

    # ------------------------------------------------------------------ infra
    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                if self.db_path != ":memory:":
                    conn.close()

    def initialize(self) -> None:
        sql = SCHEMA_FILE.read_text(encoding="utf-8")
        with self.connection() as conn:
            conn.executescript(sql)
            # additive migration for databases created before the image model existed
            cols = {r[1] for r in conn.execute("PRAGMA table_info(executions)")}
            for col in ("detection_source", "image_model_version", "image_category"):
                if col not in cols:
                    conn.execute(f"ALTER TABLE executions ADD COLUMN {col} TEXT")

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------- executions
    def record_execution(self, **fields: Any) -> int:
        signals = fields.pop("signals", None) or []
        cols = [
            "task_id", "platform", "browser", "action", "mode", "source_host", "status",
            "captcha_detected", "captcha_type", "confidence", "resolution_method",
            "model_version", "duration_ms", "error_code", "message",
            "detection_source", "image_model_version", "image_category",
        ]
        values = [fields.get(c) for c in cols]
        values[cols.index("captcha_detected")] = 1 if fields.get("captcha_detected") else 0
        with self.connection() as conn:
            cur = conn.execute(
                f"INSERT INTO executions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", values
            )
            exec_id = int(cur.lastrowid)
            conn.executemany(
                "INSERT INTO detection_signals (execution_id, captcha_type, signal, weight) VALUES (?,?,?,?)",
                [(exec_id, s["captcha_type"], s["signal"], float(s["weight"])) for s in signals],
            )
        return exec_id

    def get_execution(self, exec_id: int) -> dict | None:
        rows = self.query("SELECT * FROM executions WHERE id = ?", (exec_id,))
        return rows[0] if rows else None

    def list_executions(self, limit: int = 100) -> list[dict]:
        return self.query("SELECT * FROM executions ORDER BY id DESC LIMIT ?", (limit,))

    # ---------------------------------------------------- verification queue
    def create_verification(self, request_id: str, task_id: str, captcha_type: str | None,
                            source_host: str | None, screenshot_path: str | None,
                            ttl_seconds: int) -> dict:
        expires = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO verification_requests (request_id, task_id, captcha_type, source_host, "
                "screenshot_path, status, created_at, expires_at) VALUES (?,?,?,?,?,'PENDING',?,?)",
                (request_id, task_id, captcha_type, source_host, screenshot_path, utc_now(), expires),
            )
        return self.get_verification(request_id)  # type: ignore[return-value]

    def get_verification(self, request_id: str) -> dict | None:
        self.expire_verifications()
        rows = self.query("SELECT * FROM verification_requests WHERE request_id = ?", (request_id,))
        return rows[0] if rows else None

    def list_verifications(self, status: str | None = None) -> list[dict]:
        self.expire_verifications()
        if status:
            return self.query(
                "SELECT * FROM verification_requests WHERE status = ? ORDER BY created_at", (status,)
            )
        return self.query("SELECT * FROM verification_requests ORDER BY created_at DESC")

    def update_verification(self, request_id: str, status: str, operator: str | None = None,
                            notes: str | None = None) -> bool:
        with self.connection() as conn:
            cur = conn.execute(
                "UPDATE verification_requests SET status = ?, operator = COALESCE(?, operator), "
                "notes = COALESCE(?, notes), completed_at = ? WHERE request_id = ? AND status = 'PENDING'",
                (status, operator, notes, utc_now(), request_id),
            )
            return cur.rowcount > 0

    def expire_verifications(self) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                "UPDATE verification_requests SET status = 'EXPIRED', completed_at = ? "
                "WHERE status = 'PENDING' AND expires_at IS NOT NULL AND expires_at < ?",
                (utc_now(), utc_now()),
            )
            return cur.rowcount

    # --------------------------------------------------------------- feedback
    def record_feedback(self, actual_type: str, execution_id: int | None = None,
                        predicted_type: str | None = None, reviewer: str | None = None) -> int:
        if execution_id is not None and predicted_type is None:
            ex = self.get_execution(execution_id)
            predicted_type = ex["captcha_type"] if ex else None
        with self.connection() as conn:
            cur = conn.execute(
                "INSERT INTO detection_feedback (execution_id, predicted_type, actual_type, reviewer) VALUES (?,?,?,?)",
                (execution_id, predicted_type, actual_type, reviewer),
            )
            return int(cur.lastrowid)

    # --------------------------------------------------------- model registry
    def register_model(self, version: str, file_path: str, accuracy: float | None,
                       macro_f1: float | None, metrics_json: str | None,
                       created_by: str | None, notes: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO model_versions (version, file_path, status, accuracy, macro_f1, metrics_json, "
                "created_by, notes) VALUES (?,?,'CANDIDATE',?,?,?,?,?)",
                (version, file_path, accuracy, macro_f1, metrics_json, created_by, notes),
            )

    def get_model(self, version: str) -> dict | None:
        rows = self.query("SELECT * FROM model_versions WHERE version = ?", (version,))
        return rows[0] if rows else None

    def list_models(self) -> list[dict]:
        return self.query("SELECT * FROM model_versions ORDER BY created_at DESC")

    def set_model_status(self, version: str, status: str, approver: str | None = None) -> None:
        with self.connection() as conn:
            if status == "APPROVED":
                conn.execute(
                    "UPDATE model_versions SET status = ?, approved_by = ?, approved_at = ? WHERE version = ?",
                    (status, approver, utc_now(), version),
                )
            elif status == "DEPLOYED":
                conn.execute("UPDATE model_versions SET status = 'RETIRED' WHERE status = 'DEPLOYED'")
                conn.execute(
                    "UPDATE model_versions SET status = 'DEPLOYED', deployed_at = ? WHERE version = ?",
                    (utc_now(), version),
                )
            else:
                conn.execute("UPDATE model_versions SET status = ? WHERE version = ?", (status, version))

    def deployed_model(self) -> dict | None:
        rows = self.query("SELECT * FROM model_versions WHERE status = 'DEPLOYED' ORDER BY deployed_at DESC LIMIT 1")
        return rows[0] if rows else None

    # ----------------------------------------------------------- api attempts
    def record_api_attempt(self, execution_id: int | None, site: str | None, endpoint_host: str | None,
                           success: bool, http_status: int | None, error: str | None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO api_attempts (execution_id, site, endpoint_host, success, http_status, error) "
                "VALUES (?,?,?,?,?,?)",
                (execution_id, site, endpoint_host, 1 if success else 0, http_status, error),
            )
