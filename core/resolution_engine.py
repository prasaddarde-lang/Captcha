"""Authorized resolution engine.

Resolvers are tried in order; the first that returns an outcome wins:

  1. ``TestModeResolver``      - mock challenges only (mode=test on allow-listed hosts / synthetic fixtures)
  2. ``OfficialApiResolver``   - the site publishes an official API: route the robot to it instead of the UI
  3. ``HumanInTheLoopResolver``- an authorized operator completes the challenge

There is deliberately no resolver that answers CAPTCHAs automatically.
Each resolver is independent and can be replaced or extended via ``ResolutionEngine(resolvers=[...])``.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from config import Settings
from core.detector import DetectionResult
from core.human_verification import Deadline, HumanVerificationManager
from core.response_handler import Status, TaskRequest
from core.security import get_secret, host_of
from database.db_manager import DatabaseManager

log = logging.getLogger("captcha_fw.resolution")

PROVIDER_SITEVERIFY = {
    "recaptcha": "https://www.google.com/recaptcha/api/siteverify",
    "hcaptcha": "https://api.hcaptcha.com/siteverify",
    "turnstile": "https://challenges.cloudflare.com/turnstile/v0/siteverify",
}


@dataclass
class ResolutionContext:
    request: TaskRequest
    detection: DetectionResult
    deadline: Deadline
    session: Any = None                 # core.browser.BrowserSession when a live page is available
    detector: Any = None                # used for clearance checks on the live page
    site_entry: dict = field(default_factory=dict)
    execution_id: int | None = None

    @property
    def host(self) -> str | None:
        return host_of(self.request.url or self.detection.snapshot.url)


@dataclass
class ResolutionOutcome:
    status: Status
    method: str
    message: str
    request_id: str | None = None
    next_action: str | None = None
    official_api: dict | None = None
    screenshot_path: str | None = None


class Resolver(Protocol):
    name: str

    def can_handle(self, ctx: ResolutionContext) -> bool: ...
    def resolve(self, ctx: ResolutionContext) -> ResolutionOutcome | None: ...


def is_test_target(ctx: ResolutionContext, settings: Settings) -> bool:
    """Test mode may only touch allow-listed hosts or synthetic fixtures marked data-mock-captcha."""
    host = ctx.host
    if host and host != "file":
        return host in settings.test_mode_hosts or host.endswith(".test") or host.endswith(".localhost")
    return ctx.detection.features.mock_marker is not None


# --------------------------------------------------------------------------- test mode
class TestModeResolver:
    name = "test_mode"

    def __init__(self, settings: Settings):
        self.settings = settings

    def _config(self) -> dict:
        path = self.settings.test_data_dir / "test_responses.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("test_responses.json is invalid; using defaults")
        return {"default": "pass", "by_type": {}, "delay_seconds": 0}

    def can_handle(self, ctx: ResolutionContext) -> bool:
        return ctx.request.mode == "test"

    def resolve(self, ctx: ResolutionContext) -> ResolutionOutcome:
        if not is_test_target(ctx, self.settings):
            return ResolutionOutcome(
                Status.POLICY_BLOCKED, self.name,
                "Test mode is only allowed on allow-listed test hosts or synthetic mock fixtures. "
                "Security controls on production websites are never disabled.")
        cfg = self._config()
        ctype = ctx.detection.classification.captcha_type or "unknown"
        response = ctx.request.test_response or cfg.get("by_type", {}).get(ctype) or cfg.get("default", "pass")
        delay = float(cfg.get("delay_seconds", 0))
        if delay:
            time.sleep(min(delay, ctx.deadline.remaining()))
        if response == "timeout":
            time.sleep(ctx.deadline.remaining())
            return ResolutionOutcome(Status.TIMEOUT, self.name, "Simulated verification timeout (test mode)")
        if response == "fail":
            return ResolutionOutcome(Status.TEST_FAILED, self.name, f"Simulated verification failure for '{ctype}'")
        return ResolutionOutcome(Status.TEST_VERIFIED, self.name, f"Simulated verification success for '{ctype}'")


# ----------------------------------------------------------------------- official API
class OfficialApiResolver:
    """When the target site offers a documented API, the right move is to use the API
    instead of automating the CAPTCHA-protected UI. Configured in sites.json."""

    name = "official_api"

    def __init__(self, settings: Settings, db: DatabaseManager, http_client: httpx.Client | None = None):
        self.settings = settings
        self.db = db
        self.http = http_client

    def can_handle(self, ctx: ResolutionContext) -> bool:
        api = ctx.site_entry.get("official_api")
        return bool(api and api.get("enabled", True) and api.get("base_url"))

    def resolve(self, ctx: ResolutionContext) -> ResolutionOutcome | None:
        api = ctx.site_entry["official_api"]
        base = api["base_url"].rstrip("/")
        if self.settings.require_https and not base.lower().startswith("https://"):
            log.warning("official API for %s skipped: HTTPS required", ctx.host)
            return None
        token = get_secret(api.get("auth_env"))
        if api.get("auth_env") and not token:
            log.warning("official API credential env var %s is not set; falling back", api.get("auth_env"))
            return None
        headers = {"Accept": "application/json"}
        if token:
            headers[api.get("auth_header", "Authorization")] = f"{api.get('auth_scheme', 'Bearer')} {token}".strip()
        url = base + api.get("health_path", "")
        client = self.http or httpx.Client(timeout=self.settings.http_timeout)
        last_error: str | None = None
        status_code: int | None = None
        for attempt in range(self.settings.http_retries + 1):
            if ctx.deadline.expired():
                break
            try:
                r = client.get(url, headers=headers, timeout=min(self.settings.http_timeout, max(ctx.deadline.remaining(), 0.5)))
                status_code = r.status_code
                if r.status_code < 400:
                    self.db.record_api_attempt(ctx.execution_id, ctx.request.site or ctx.host, host_of(base), True, r.status_code, None)
                    return ResolutionOutcome(
                        Status.OFFICIAL_API_ROUTE, self.name,
                        "Site provides an official API; continue the workflow through the API instead of the UI.",
                        next_action="Call the official API endpoints listed in official_api",
                        official_api={"base_url": base, "docs_url": api.get("docs_url"),
                                      "auth_env": api.get("auth_env"), "site": ctx.request.site or ctx.host})
                if r.status_code in (401, 403):
                    last_error = f"authorization failed ({r.status_code})"
                    break  # retrying will not help
                if r.status_code == 429:
                    last_error = "rate limited (429)"
                    break  # respect rate limits: do not hammer
                last_error = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                last_error = f"network error: {type(exc).__name__}"
            time.sleep(min(0.5 * (2 ** attempt), ctx.deadline.remaining()))
        self.db.record_api_attempt(ctx.execution_id, ctx.request.site or ctx.host, host_of(base), False, status_code, last_error)
        log.warning("official API unavailable for %s: %s", ctx.host, last_error)
        ctx.site_entry.setdefault("_notes", []).append(f"official API unavailable: {last_error}")
        return None


def verify_provider_token(provider: str, token: str, site_entry: dict, settings: Settings,
                          remote_ip: str | None = None, http_client: httpx.Client | None = None) -> dict:
    """Server-side verification of a widget token **for a site you own**, using the
    provider's documented siteverify endpoint. The token must come from a human who
    completed the widget; the secret is read from the env var named in sites.json."""
    if not site_entry.get("owned"):
        raise PermissionError("Provider verification is only available for sites marked owned=true in sites.json")
    if provider not in PROVIDER_SITEVERIFY:
        raise ValueError(f"Unsupported provider '{provider}'")
    secret = get_secret(site_entry.get("provider_secret_env"))
    if not secret:
        raise RuntimeError(f"Environment variable {site_entry.get('provider_secret_env')} is not set")
    endpoint = site_entry.get("siteverify_url") or PROVIDER_SITEVERIFY[provider]
    data = {"secret": secret, "response": token}
    if remote_ip:
        data["remoteip"] = remote_ip
    client = http_client or httpx.Client(timeout=settings.http_timeout)
    r = client.post(endpoint, data=data)
    r.raise_for_status()
    body = r.json()
    return {"success": bool(body.get("success")), "error_codes": body.get("error-codes", []),
            "hostname": body.get("hostname"), "score": body.get("score"), "action": body.get("action")}


# ------------------------------------------------------------------- human in the loop
class HumanInTheLoopResolver:
    name = "human_in_the_loop"

    def __init__(self, settings: Settings, hv: HumanVerificationManager):
        self.settings = settings
        self.hv = hv

    def can_handle(self, ctx: ResolutionContext) -> bool:
        return ctx.request.mode == "production"

    def resolve(self, ctx: ResolutionContext) -> ResolutionOutcome:
        req = ctx.request
        shot = ctx.detection.snapshot.screenshot_path
        if req.screenshot and not shot and ctx.session is not None and self.settings.screenshots_enabled:
            shot = ctx.session.screenshot(self.settings.screenshot_dir, req.task_id)
        # `timeout` is the time the operator has to complete the verification. In blocking mode
        # the waiter's deadline must win (it cancels the request), so the DB expiry gets a grace period.
        ttl = req.timeout + 30 if req.wait_for_operator else req.timeout
        vr = self.hv.create_request(req.task_id, ctx.detection.classification.captcha_type,
                                    ctx.host, shot, ttl_seconds=ttl)
        rid = vr["request_id"]
        if not req.wait_for_operator:
            return ResolutionOutcome(
                Status.HUMAN_VERIFICATION_REQUIRED, self.name, "Operator verification required",
                request_id=rid, screenshot_path=shot,
                next_action="Show the browser to an operator, then poll with action=check_status and this request_id")

        if ctx.session is not None:
            ctx.session.bring_to_front()
        clearance = None
        if ctx.session is not None and ctx.detector is not None:
            def clearance() -> bool:
                det = ctx.detector.detect(ctx.session.snapshot())
                return (not det.detected) or det.is_solved
        final = self.hv.wait(rid, ctx.deadline, clearance)
        mapping = {
            "COMPLETED": (Status.VERIFIED, "Operator verification completed; workflow may resume"),
            "REJECTED": (Status.REJECTED, "Operator rejected the verification request"),
            "CANCELLED": (Status.REJECTED, "Verification request was cancelled"),
            "EXPIRED": (Status.EXPIRED, "Verification request expired"),
            "TIMEOUT": (Status.TIMEOUT, "Timed out waiting for operator verification"),
        }
        st, msg = mapping[final]
        if final == "TIMEOUT":
            self.hv.cancel(rid)
        return ResolutionOutcome(st, self.name, msg, request_id=rid, screenshot_path=shot)


# ----------------------------------------------------------------------------- engine
class ResolutionEngine:
    def __init__(self, settings: Settings, db: DatabaseManager,
                 hv: HumanVerificationManager | None = None,
                 http_client: httpx.Client | None = None,
                 resolvers: list[Resolver] | None = None):
        self.settings = settings
        self.db = db
        self.hv = hv or HumanVerificationManager(db, settings, http_client)
        self.resolvers: list[Resolver] = resolvers or [
            TestModeResolver(settings),
            OfficialApiResolver(settings, db, http_client),
            HumanInTheLoopResolver(settings, self.hv),
        ]

    def site_entry(self, request: TaskRequest, host: str | None) -> dict:
        sites = self.settings.load_sites()
        key = request.site or host
        return dict(sites.get(key, {})) if key else {}

    def planned_method(self, ctx: ResolutionContext) -> str:
        for r in self.resolvers:
            if r.can_handle(ctx):
                return r.name
        return "none"

    def resolve(self, ctx: ResolutionContext) -> ResolutionOutcome:
        for r in self.resolvers:
            if not r.can_handle(ctx):
                continue
            outcome = r.resolve(ctx)
            if outcome is not None:
                notes = ctx.site_entry.get("_notes")
                if notes and outcome.method != "official_api":
                    outcome.message += f" ({'; '.join(notes)})"
                return outcome
        return ResolutionOutcome(Status.ERROR, "none", "No authorized resolution method is available for this request")

    def check_status(self, request_id: str) -> ResolutionOutcome:
        vr = self.hv.get(request_id)
        if vr is None:
            return ResolutionOutcome(Status.INVALID_INPUT, "human_in_the_loop", f"Unknown request_id {request_id}")
        mapping = {
            "PENDING": (Status.PENDING, "Waiting for operator"),
            "COMPLETED": (Status.VERIFIED, f"Verified by {vr.get('operator') or 'operator'}; workflow may resume"),
            "REJECTED": (Status.REJECTED, "Operator rejected the verification request"),
            "CANCELLED": (Status.REJECTED, "Verification request was cancelled"),
            "EXPIRED": (Status.EXPIRED, "Verification request expired"),
        }
        st, msg = mapping.get(vr["status"], (Status.ERROR, "Unknown state"))
        return ResolutionOutcome(st, "human_in_the_loop", msg, request_id=request_id,
                                 screenshot_path=vr.get("screenshot_path"))
