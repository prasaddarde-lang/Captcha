"""Human-in-the-loop verification.

Flow:
  1. The engine creates a verification request (stored in SQLite) and notifies operators.
  2. An authorized operator completes the CAPTCHA *in the robot's browser window*
     (or the headed Playwright window) and marks the request complete via the
     CLI (``main.py operator complete``), the REST API, or the /operator web page.
  3. The RPA workflow either blocks (``wait_for_operator: true``) or polls with
     ``action: check_status`` and resumes when the status is VERIFIED.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import httpx

from config import Settings
from core.security import get_secret, sanitize_text
from database.db_manager import DatabaseManager

log = logging.getLogger("captcha_fw.human")

FINAL_STATES = {"COMPLETED", "REJECTED", "EXPIRED", "CANCELLED"}


class Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self.end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0


class HumanVerificationManager:
    def __init__(self, db: DatabaseManager, settings: Settings,
                 http_client: httpx.Client | None = None,
                 notifier: Callable[[dict], None] | None = None):
        self.db = db
        self.settings = settings
        self.http = http_client
        self.notifier = notifier

    # ---------------------------------------------------------------- create
    def create_request(self, task_id: str, captcha_type: str | None, source_host: str | None,
                       screenshot_path: str | None, ttl_seconds: int) -> dict:
        request_id = f"HV_{uuid.uuid4().hex[:16]}"
        req = self.db.create_verification(request_id, task_id, captcha_type, source_host,
                                          screenshot_path, ttl_seconds)
        self.notify(req)
        log.info("verification request %s created for task %s (%s)", request_id, task_id, captcha_type)
        return req

    def notify(self, req: dict) -> None:
        """Tell operators a verification is waiting. Payload carries no challenge content."""
        payload = {
            "event": "captcha_verification_required",
            "request_id": req["request_id"],
            "task_id": req["task_id"],
            "captcha_type": req["captcha_type"],
            "source_host": req["source_host"],
            "expires_at": req["expires_at"],
        }
        if self.notifier:
            try:
                self.notifier(payload)
            except Exception as exc:
                log.warning("custom notifier failed: %s", exc)
        webhook = get_secret("CAPTCHA_FW_NOTIFY_WEBHOOK")
        if webhook:
            if self.settings.require_https and not webhook.lower().startswith("https://"):
                log.warning("notification webhook ignored: HTTPS is required")
                return
            try:
                client = self.http or httpx.Client(timeout=self.settings.http_timeout)
                client.post(webhook, json=payload)
            except httpx.HTTPError as exc:
                log.warning("notification webhook failed: %s", exc)

    # ------------------------------------------------------------- operators
    def complete(self, request_id: str, operator: str, notes: str | None = None) -> bool:
        ok = self.db.update_verification(request_id, "COMPLETED", sanitize_text(operator, 80),
                                         sanitize_text(notes, 500) if notes else None)
        log.info("verification %s completed by %s: %s", request_id, operator, ok)
        return ok

    def reject(self, request_id: str, operator: str, notes: str | None = None) -> bool:
        ok = self.db.update_verification(request_id, "REJECTED", sanitize_text(operator, 80),
                                         sanitize_text(notes, 500) if notes else None)
        log.info("verification %s rejected by %s: %s", request_id, operator, ok)
        return ok

    def cancel(self, request_id: str) -> bool:
        return self.db.update_verification(request_id, "CANCELLED", "rpa", "cancelled by workflow")

    def get(self, request_id: str) -> dict | None:
        return self.db.get_verification(request_id)

    def pending(self) -> list[dict]:
        return self.db.list_verifications("PENDING")

    # ------------------------------------------------------------------ wait
    def wait(self, request_id: str, deadline: Deadline,
             clearance_check: Callable[[], bool] | None = None) -> str:
        """Block until the request reaches a final state or the deadline passes.

        ``clearance_check`` (optional) inspects the live browser page: when it reports
        that the challenge is no longer present / the widget response is filled, the
        request is completed automatically ("clearance observed").
        Returns the final DB status or ``TIMEOUT``.
        """
        interval = max(0.2, self.settings.operator_poll_interval)
        while True:
            req = self.db.get_verification(request_id)
            if req is None:
                return "CANCELLED"
            if req["status"] in FINAL_STATES:
                return req["status"]
            if clearance_check is not None:
                try:
                    if clearance_check():
                        self.complete(request_id, "auto", "clearance observed in browser after operator action")
                        return "COMPLETED"
                except Exception as exc:
                    log.warning("clearance check failed: %s", exc)
            if deadline.expired():
                return "TIMEOUT"
            time.sleep(min(interval, max(deadline.remaining(), 0.05)))

    # ------------------------------------------------------------ retention
    def purge_old_screenshots(self) -> int:
        """Delete challenge screenshots older than the retention window."""
        cutoff = datetime.now() - timedelta(hours=self.settings.screenshot_retention_hours)
        removed = 0
        for p in Path(self.settings.screenshot_dir).glob("*.png"):
            if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        return removed


def format_pending_table(rows: list[dict]) -> str:
    if not rows:
        return "No pending verification requests."
    lines = [f"{'REQUEST_ID':<22}{'TASK_ID':<20}{'TYPE':<16}{'HOST':<28}EXPIRES"]
    for r in rows:
        lines.append(f"{r['request_id']:<22}{r['task_id']:<20}{str(r['captcha_type']):<16}"
                     f"{str(r['source_host'])[:26]:<28}{r['expires_at']}")
    return "\n".join(lines)


def to_json(rows: list[dict]) -> str:
    return json.dumps(rows, indent=2)
