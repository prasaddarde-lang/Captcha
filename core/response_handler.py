"""Request/response contracts (Pydantic), status codes and JSON formatting for RPA callers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from core.security import TASK_ID_RE, sanitize_text, validate_url

MAX_HTML_CHARS = 5_000_000


class Status(str, Enum):
    NO_CAPTCHA = "NO_CAPTCHA"
    CAPTCHA_DETECTED = "CAPTCHA_DETECTED"
    HUMAN_VERIFICATION_REQUIRED = "HUMAN_VERIFICATION_REQUIRED"
    VERIFIED = "VERIFIED"
    OFFICIAL_API_ROUTE = "OFFICIAL_API_ROUTE"
    TEST_VERIFIED = "TEST_VERIFIED"
    TEST_FAILED = "TEST_FAILED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    TIMEOUT = "TIMEOUT"
    INVALID_INPUT = "INVALID_INPUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    ERROR = "ERROR"


# Process exit codes for BAT / command-line callers.
EXIT_CODES: dict[Status, int] = {
    Status.NO_CAPTCHA: 0,
    Status.VERIFIED: 0,
    Status.TEST_VERIFIED: 0,
    Status.OFFICIAL_API_ROUTE: 0,
    Status.CAPTCHA_DETECTED: 10,
    Status.HUMAN_VERIFICATION_REQUIRED: 11,
    Status.PENDING: 12,
    Status.TEST_FAILED: 20,
    Status.REJECTED: 21,
    Status.EXPIRED: 22,
    Status.TIMEOUT: 30,
    Status.INVALID_INPUT: 40,
    Status.POLICY_BLOCKED: 41,
    Status.NETWORK_ERROR: 50,
    Status.ERROR: 99,
}

Action = Literal["detect", "detect_and_handle", "check_status", "cancel"]
Mode = Literal["production", "test"]


class TaskRequest(BaseModel):
    """Input contract shared by CLI, BAT, REST API and all RPA adapters."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str
    platform: str = "generic"
    browser: str = "chrome"
    action: Action = "detect_and_handle"
    timeout: int = Field(default=120, ge=1, le=3600)
    mode: Mode = "production"

    # Page source - provide ONE of these (checked in that order)
    html: str | None = Field(default=None, max_length=MAX_HTML_CHARS)
    html_file: str | None = None
    cdp_endpoint: str | None = None      # attach to the RPA's own browser, e.g. http://127.0.0.1:9222
    url: str | None = None               # page URL (used to pick tab in CDP mode, or to open the page)
    image_file: str | None = None        # cropped screenshot of the suspected challenge element (image model)

    site: str | None = None              # key in sites.json; defaults to the URL host
    screenshot: bool = False             # capture a screenshot for the operator (where authorized)
    headless: bool = True
    wait_for_operator: bool = False      # block until an operator completes verification
    request_id: str | None = None        # for check_status / cancel
    test_response: Literal["pass", "fail", "timeout"] | None = None  # test mode only

    @field_validator("task_id")
    @classmethod
    def _task_id(cls, v: str) -> str:
        if not TASK_ID_RE.match(v):
            raise ValueError("task_id must be 1-64 chars of letters, digits, '_', '-', '.'")
        return v

    @field_validator("platform", "browser")
    @classmethod
    def _short_text(cls, v: str) -> str:
        return sanitize_text(v, 40)

    @field_validator("url", "cdp_endpoint")
    @classmethod
    def _url(cls, v: str | None) -> str | None:
        return validate_url(v) if v else v

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, v: str | None) -> str | None:
        if v and not TASK_ID_RE.match(v):
            raise ValueError("invalid request_id")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "TaskRequest":
        if self.action in ("check_status", "cancel") and not self.request_id:
            raise ValueError(f"action '{self.action}' requires request_id")
        if self.action in ("detect", "detect_and_handle") and not (
            self.html or self.html_file or self.cdp_endpoint or self.url or self.image_file
        ):
            raise ValueError("provide one of: html, html_file, cdp_endpoint, url, image_file")
        if self.test_response and self.mode != "test":
            raise ValueError("test_response is only allowed in test mode")
        return self


class Signal(BaseModel):
    captcha_type: str
    signal: str
    weight: float


class TaskResponse(BaseModel):
    task_id: str
    status: Status
    captcha_detected: bool = False
    captcha_type: str | None = None
    captcha_subtype: str | None = None
    confidence: float | None = None
    resolution_method: str | None = None
    message: str = ""
    request_id: str | None = None
    screenshot_path: str | None = None
    next_action: str | None = None
    official_api: dict[str, Any] | None = None
    signals: list[Signal] = Field(default_factory=list)
    model_version: str | None = None
    detection_source: str | None = None        # metadata | image_model | metadata+image_model
    image_model_version: str | None = None
    image_category: str | None = None          # image-model category (e.g. grid, math) when it ran
    image_confidence: float | None = None
    duration_ms: int | None = None
    exit_code: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @model_validator(mode="after")
    def _exit(self) -> "TaskResponse":
        self.exit_code = EXIT_CODES.get(self.status, 99)
        return self

    def to_json(self, include_signals: bool = True, indent: int | None = 2) -> str:
        exclude = None if include_signals else {"signals"}
        return self.model_dump_json(indent=indent, exclude=exclude, exclude_none=False)


def error_response(task_id: str | None, status: Status, message: str) -> TaskResponse:
    safe_id = task_id if task_id and TASK_ID_RE.match(task_id) else "UNKNOWN"
    return TaskResponse(task_id=safe_id, status=status, message=sanitize_text(message, 1000))


def parse_request(raw: str | dict) -> TaskRequest:
    """Parse JSON text or dict into a TaskRequest. Raises ValueError with a readable message."""
    if isinstance(raw, str):
        raw = raw.strip().lstrip("﻿")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Input is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})") from exc
    else:
        data = raw
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object")
    try:
        return TaskRequest.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        raise ValueError(f"Invalid input: {problems}") from exc
