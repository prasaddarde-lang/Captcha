"""Shared fixtures: isolated settings/DB per test, mock HTTP transport, synthetic fixtures."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "test_data" / "fixtures"


@pytest.fixture
def fixture_html() -> Callable[[str], str]:
    return lambda name: (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from config import get_settings, reset_settings

    env = {
        "CAPTCHA_FW_DB_PATH": str(tmp_path / "db" / "events.db"),
        "CAPTCHA_FW_LOG_DIR": str(tmp_path / "logs"),
        "CAPTCHA_FW_SCREENSHOT_DIR": str(tmp_path / "screenshots"),
        "CAPTCHA_FW_TEST_DATA_DIR": str(tmp_path / "test_data"),
        "CAPTCHA_FW_MODELS_DIR": str(tmp_path / "models"),
        "CAPTCHA_FW_SITES_FILE": str(tmp_path / "sites.json"),
        "CAPTCHA_FW_OPERATOR_POLL_INTERVAL": "0.2",
        "CAPTCHA_FW_HTTP_RETRIES": "1",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in ("CAPTCHA_FW_API_KEY", "CAPTCHA_FW_NOTIFY_WEBHOOK"):
        monkeypatch.delenv(k, raising=False)
    reset_settings()
    s = get_settings()
    yield s
    reset_settings()


@pytest.fixture
def write_sites(settings):
    def _write(sites: dict) -> None:
        settings.sites_file.write_text(json.dumps({"sites": sites}), encoding="utf-8")
    return _write


class MockHTTP:
    """Route table for httpx.MockTransport: {(method, url_prefix): handler(request) -> Response}."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {}
        self.calls: list[httpx.Request] = []

    def add(self, method: str, prefix: str, handler) -> None:
        self.routes[(method.upper(), prefix)] = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        for (m, prefix), h in self.routes.items():
            if request.method == m and str(request.url).startswith(prefix):
                return h(request)
        raise httpx.ConnectError("no mock route", request=request)


@pytest.fixture
def mock_http():
    return MockHTTP()


@pytest.fixture
def http_client(mock_http):
    with httpx.Client(transport=httpx.MockTransport(mock_http)) as c:
        yield c


@pytest.fixture
def db(settings):
    from database.db_manager import DatabaseManager
    return DatabaseManager(settings.db_path)


@pytest.fixture
def framework(settings, db, http_client):
    from core.framework import CaptchaFramework
    return CaptchaFramework(settings=settings, db=db, http_client=http_client)


@pytest.fixture
def make_request():
    from core.response_handler import TaskRequest

    def _make(**kw) -> TaskRequest:
        base = {"task_id": "T_1", "platform": "pytest", "action": "detect_and_handle", "timeout": 10}
        base.update(kw)
        return TaskRequest(**base)
    return _make
