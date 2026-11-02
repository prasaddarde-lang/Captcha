"""Database operations."""
import sqlite3

import pytest

from database.db_manager import DatabaseManager


def test_schema_tables(db):
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"executions", "detection_signals", "verification_requests", "detection_feedback",
            "model_versions", "api_attempts"} <= names


def test_initialize_idempotent(settings):
    DatabaseManager(settings.db_path)
    DatabaseManager(settings.db_path)  # re-running the schema must not fail


def test_in_memory_db():
    m = DatabaseManager(":memory:")
    eid = m.record_execution(task_id="A", action="detect", mode="test", status="NO_CAPTCHA")
    assert m.get_execution(eid)["task_id"] == "A"


def test_record_execution_with_signals(db):
    eid = db.record_execution(task_id="T", platform="UiPath", action="detect", mode="production",
                              status="CAPTCHA_DETECTED", captcha_detected=True, captcha_type="hcaptcha",
                              confidence=0.9, signals=[{"captcha_type": "hcaptcha", "signal": "hcaptcha_class", "weight": 2.5}])
    row = db.get_execution(eid)
    assert row["captcha_detected"] == 1 and row["captcha_type"] == "hcaptcha"
    sigs = db.query("SELECT * FROM detection_signals WHERE execution_id = ?", (eid,))
    assert sigs[0]["signal"] == "hcaptcha_class"


def test_cascade_delete(db):
    eid = db.record_execution(task_id="T", action="detect", mode="test", status="X",
                              signals=[{"captcha_type": "a", "signal": "b", "weight": 1}])
    with db.connection() as c:
        c.execute("DELETE FROM executions WHERE id = ?", (eid,))
    assert db.query("SELECT COUNT(*) AS n FROM detection_signals")[0]["n"] == 0


def test_feedback_fills_predicted_type(db):
    eid = db.record_execution(task_id="T", action="detect", mode="test", status="X", captcha_type="text")
    db.record_feedback("image_selection", eid, reviewer="R")
    fb = db.query("SELECT * FROM detection_feedback")[0]
    assert fb["predicted_type"] == "text" and fb["actual_type"] == "image_selection"


def test_model_registry_single_deployed(db):
    for v in ("1.0.0", "1.0.1"):
        db.register_model(v, f"{v}.json", 0.9, 0.9, None, "t")
        db.set_model_status(v, "APPROVED", "R")
        db.set_model_status(v, "DEPLOYED")
    deployed = db.query("SELECT version FROM model_versions WHERE status='DEPLOYED'")
    assert [d["version"] for d in deployed] == ["1.0.1"]
    assert db.get_model("1.0.0")["status"] == "RETIRED"


def test_duplicate_model_version_rejected(db):
    db.register_model("2.0.0", "a.json", None, None, None, "t")
    with pytest.raises(sqlite3.IntegrityError):
        db.register_model("2.0.0", "b.json", None, None, None, "t")


def test_parameterized_queries_resist_injection(db):
    evil = "x'); DROP TABLE executions; --"
    db.record_execution(task_id=evil, action="detect", mode="test", status="NO_CAPTCHA")
    assert db.query("SELECT task_id FROM executions")[0]["task_id"] == evil


def test_no_challenge_content_columns(db):
    """Only metadata is stored - no image/audio/answer/token columns exist."""
    cols = set()
    for t in ("executions", "verification_requests", "api_attempts", "detection_signals"):
        cols |= {r["name"] for r in db.query(f"PRAGMA table_info({t})")}
    assert not ({"image", "audio", "answer", "token", "solution", "response_value"} & cols)
