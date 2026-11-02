"""Detection history analytics: volumes, accuracy, verification outcomes and recurring failures."""
from __future__ import annotations

from collections import Counter, defaultdict

from database.db_manager import DatabaseManager

FAILURE_STATUSES = ("TIMEOUT", "ERROR", "NETWORK_ERROR", "EXPIRED", "REJECTED", "TEST_FAILED")


class CaptchaAnalytics:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def _group(self, column: str) -> dict[str, int]:
        rows = self.db.query(f"SELECT COALESCE({column}, 'n/a') AS k, COUNT(*) AS n FROM executions GROUP BY k ORDER BY n DESC")
        return {r["k"]: r["n"] for r in rows}

    def summary(self) -> dict:
        tot = self.db.query(
            "SELECT COUNT(*) AS total, SUM(captcha_detected) AS detected, AVG(duration_ms) AS avg_ms, "
            "AVG(CASE WHEN captcha_detected=1 THEN confidence END) AS avg_conf FROM executions")[0]
        total = tot["total"] or 0
        return {
            "total_executions": total,
            "captcha_detected": tot["detected"] or 0,
            "detection_rate": round((tot["detected"] or 0) / total, 4) if total else 0.0,
            "avg_duration_ms": round(tot["avg_ms"] or 0, 1),
            "avg_confidence": round(tot["avg_conf"] or 0, 4),
            "by_type": self._group("captcha_type"),
            "by_status": self._group("status"),
            "by_method": self._group("resolution_method"),
            "by_platform": self._group("platform"),
            "by_model_version": self._group("model_version"),
        }

    def detection_accuracy(self) -> dict:
        """Accuracy of detection/classification against operator-confirmed labels."""
        rows = self.db.query("SELECT COALESCE(predicted_type,'none') AS p, actual_type AS a FROM detection_feedback")
        if not rows:
            return {"labelled_samples": 0, "accuracy": None, "confusion": {}, "per_type": {}}
        confusion: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            confusion[r["a"]][r["p"]] += 1
        correct = sum(confusion[a][a] for a in confusion)
        per_type = {a: round(c[a] / sum(c.values()), 4) for a, c in confusion.items()}
        return {
            "labelled_samples": len(rows),
            "accuracy": round(correct / len(rows), 4),
            "per_type": per_type,
            "confusion": {a: dict(c) for a, c in confusion.items()},
        }

    def verification_stats(self) -> dict:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM verification_requests GROUP BY status")
        by_status = {r["status"]: r["n"] for r in rows}
        total = sum(by_status.values())
        t = self.db.query(
            "SELECT AVG((julianday(completed_at) - julianday(created_at)) * 86400.0) AS s "
            "FROM verification_requests WHERE status = 'COMPLETED'")[0]["s"]
        attempts = self.db.query("SELECT success, COUNT(*) AS n FROM api_attempts GROUP BY success")
        api = {("success" if r["success"] else "failure"): r["n"] for r in attempts}
        return {
            "human_requests": total,
            "human_by_status": by_status,
            "human_completion_rate": round(by_status.get("COMPLETED", 0) / total, 4) if total else None,
            "avg_seconds_to_complete": round(t, 1) if t is not None else None,
            "official_api_attempts": api,
        }

    def recurring_failures(self, min_count: int = 3) -> dict:
        mis = self.db.query(
            "SELECT COALESCE(predicted_type,'none') AS predicted, actual_type AS actual, COUNT(*) AS n "
            "FROM detection_feedback WHERE COALESCE(predicted_type,'none') <> actual_type "
            "GROUP BY predicted, actual HAVING n >= ? ORDER BY n DESC", (min_count,))
        hosts = self.db.query(
            f"SELECT COALESCE(source_host,'inline') AS host, status, COUNT(*) AS n FROM executions "
            f"WHERE status IN ({','.join('?' * len(FAILURE_STATUSES))}) GROUP BY host, status "
            f"HAVING n >= ? ORDER BY n DESC", (*FAILURE_STATUSES, min_count))
        sig = self.db.query(
            "SELECT s.captcha_type, s.signal, COUNT(*) AS n FROM detection_feedback f "
            "JOIN detection_signals s ON s.execution_id = f.execution_id "
            "WHERE COALESCE(f.predicted_type,'none') <> f.actual_type "
            "GROUP BY s.captcha_type, s.signal HAVING n >= ? ORDER BY n DESC", (min_count,))
        recs = []
        for m in mis:
            if m["predicted"] == "none":
                recs.append(f"'{m['actual']}' challenges missed {m['n']}x: review signatures for this type")
            else:
                recs.append(f"'{m['actual']}' classified as '{m['predicted']}' {m['n']}x: consider retraining weights")
        for h in hosts:
            recs.append(f"host {h['host']} hit {h['status']} {h['n']}x: check timeout / operator staffing / site API")
        return {"misclassification_patterns": mis, "failing_hosts": hosts,
                "signals_in_misclassifications": sig, "recommendations": recs}

    def daily_trend(self, days: int = 14) -> list[dict]:
        return self.db.query(
            "SELECT substr(created_at,1,10) AS day, COUNT(*) AS total, SUM(captcha_detected) AS detected, "
            "SUM(CASE WHEN status IN ('VERIFIED','TEST_VERIFIED','OFFICIAL_API_ROUTE','NO_CAPTCHA') THEN 1 ELSE 0 END) AS ok "
            "FROM executions WHERE created_at >= date('now', ?) GROUP BY day ORDER BY day", (f"-{int(days)} days",))
