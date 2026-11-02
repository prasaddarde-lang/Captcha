-- CAPTCHA Management Framework - local event history (SQLite)
-- Only metadata is stored: no challenge images, audio, answers or tokens.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS executions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT NOT NULL,
    platform          TEXT,
    browser           TEXT,
    action            TEXT NOT NULL,
    mode              TEXT NOT NULL,
    source_host       TEXT,
    status            TEXT NOT NULL,
    captcha_detected  INTEGER NOT NULL DEFAULT 0,
    captcha_type      TEXT,
    confidence        REAL,
    resolution_method TEXT,
    model_version     TEXT,
    duration_ms       INTEGER,
    error_code        TEXT,
    message           TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_exec_task   ON executions(task_id);
CREATE INDEX IF NOT EXISTS idx_exec_type   ON executions(captcha_type);
CREATE INDEX IF NOT EXISTS idx_exec_status ON executions(status);

-- Detection signals that fired (selector / attribute names only, never content)
CREATE TABLE IF NOT EXISTS detection_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id  INTEGER NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    captcha_type  TEXT NOT NULL,
    signal        TEXT NOT NULL,
    weight        REAL NOT NULL
);

-- Human-in-the-loop verification requests
CREATE TABLE IF NOT EXISTS verification_requests (
    request_id      TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    captcha_type    TEXT,
    source_host     TEXT,
    screenshot_path TEXT,
    status          TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING|COMPLETED|REJECTED|EXPIRED|CANCELLED
    operator        TEXT,
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    expires_at      TEXT,
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_vr_status ON verification_requests(status);

-- Operator / reviewer feedback on detection (ground-truth label for the *type* only)
CREATE TABLE IF NOT EXISTS detection_feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id   INTEGER REFERENCES executions(id) ON DELETE SET NULL,
    predicted_type TEXT,
    actual_type    TEXT NOT NULL,
    reviewer       TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Classifier model registry (rule-weight models) with approval workflow
CREATE TABLE IF NOT EXISTS model_versions (
    version       TEXT PRIMARY KEY,
    file_path     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'CANDIDATE',  -- CANDIDATE|APPROVED|DEPLOYED|REJECTED|RETIRED
    accuracy      REAL,
    macro_f1      REAL,
    metrics_json  TEXT,
    created_by    TEXT,
    approved_by   TEXT,
    approved_at   TEXT,
    deployed_at   TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Official-API / provider verification attempts (no tokens stored)
CREATE TABLE IF NOT EXISTS api_attempts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id  INTEGER REFERENCES executions(id) ON DELETE CASCADE,
    site          TEXT,
    endpoint_host TEXT,
    success       INTEGER NOT NULL,
    http_status   INTEGER,
    error         TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
