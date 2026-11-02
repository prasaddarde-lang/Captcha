"""Timeouts, network failures and browser-session error handling (with fake sessions)."""
import time

import httpx

from core.browser import BrowserUnavailable
from core.detector import PageSnapshot
from core.framework import CaptchaFramework
from core.response_handler import Status


class FakeSession:
    """Stands in for BrowserSession: returns a canned snapshot, optional delays/errors."""

    html = "<div class='g-recaptcha'></div><script src='https://www.google.com/recaptcha/api.js'></script>"
    snapshot_delay = 0.0
    enter_error: Exception | None = None
    solve_after: int | None = None  # simulate an operator solving the widget after N snapshots

    def __init__(self, **kw):
        self.kw = kw
        self.calls = 0
        self.closed = False

    def __enter__(self):
        if self.enter_error:
            raise self.enter_error
        return self

    def __exit__(self, *a):
        self.closed = True

    def snapshot(self, screenshot_dir=None, task_id="t"):
        time.sleep(self.snapshot_delay)
        self.calls += 1
        filled = bool(self.solve_after and self.calls > self.solve_after)
        shot = None
        if screenshot_dir:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            shot = str(screenshot_dir / f"{task_id}.png")
            open(shot, "wb").write(b"\x89PNG")
        return PageSnapshot(html=self.html, url=self.kw.get("url"), response_filled={"recaptcha": filled},
                            screenshot_path=shot, source="cdp")

    def screenshot(self, directory, task_id):
        return None

    def bring_to_front(self):
        pass


def _fw(settings, db, cls):
    return CaptchaFramework(settings=settings, db=db, session_factory=cls)


def test_hard_timeout_when_page_hangs(settings, db, make_request):
    class Slow(FakeSession):
        snapshot_delay = 30

    t0 = time.monotonic()
    r = _fw(settings, db, Slow).run(make_request(cdp_endpoint="http://127.0.0.1:9222", timeout=1))
    assert r.status == Status.TIMEOUT
    assert time.monotonic() - t0 < 10
    assert db.list_executions(1)[0]["status"] == "TIMEOUT"


def test_live_session_detection_with_screenshot(settings, db, make_request):
    r = _fw(settings, db, FakeSession).run(make_request(cdp_endpoint="http://127.0.0.1:9222",
                                                        url="https://portal.example.org/login", screenshot=True))
    assert r.status == Status.HUMAN_VERIFICATION_REQUIRED
    assert r.screenshot_path and r.screenshot_path.endswith(".png")
    assert db.get_verification(r.request_id)["screenshot_path"] == r.screenshot_path


def test_screenshots_disabled_by_policy(settings, db, make_request):
    settings.screenshots_enabled = False
    r = _fw(settings, db, FakeSession).run(make_request(cdp_endpoint="http://127.0.0.1:9222", screenshot=True))
    assert r.screenshot_path is None


def test_live_session_operator_solves_in_browser(settings, db, make_request):
    class Solving(FakeSession):
        solve_after = 2

    r = _fw(settings, db, Solving).run(make_request(cdp_endpoint="http://127.0.0.1:9222", wait_for_operator=True, timeout=10))
    assert r.status == Status.VERIFIED
    assert db.get_verification(r.request_id)["notes"].startswith("clearance observed")


def test_network_error_from_browser(settings, db, make_request):
    class Down(FakeSession):
        enter_error = ConnectionError("ECONNREFUSED 127.0.0.1:9222")

    r = _fw(settings, db, Down).run(make_request(cdp_endpoint="http://127.0.0.1:9222"))
    assert r.status == Status.NETWORK_ERROR and r.exit_code == 50


def test_page_load_network_error(settings, db, make_request):
    class Err(FakeSession):
        enter_error = RuntimeError("page.goto: net::ERR_NAME_NOT_RESOLVED at https://nowhere.invalid")

    r = _fw(settings, db, Err).run(make_request(url="https://nowhere.invalid"))
    assert r.status == Status.NETWORK_ERROR


def test_browser_unavailable(settings, db, make_request):
    class NoPw(FakeSession):
        enter_error = BrowserUnavailable("Playwright is not installed. Run install.bat")

    r = _fw(settings, db, NoPw).run(make_request(url="https://x.example"))
    assert r.status == Status.ERROR and "install.bat" in r.message


def test_unexpected_exception_becomes_error(settings, db, make_request):
    class Boom(FakeSession):
        enter_error = KeyError("weird")

    r = _fw(settings, db, Boom).run(make_request(url="https://x.example"))
    assert r.status == Status.ERROR and r.exit_code == 99


def test_html_file_outside_allowed_dirs(framework, make_request, tmp_path):
    r = framework.run(make_request(html_file="C:/Windows/win.ini" if __import__("os").name == "nt" else "/etc/passwd"))
    assert r.status == Status.INVALID_INPUT


def test_html_file_missing(framework, make_request):
    r = framework.run(make_request(html_file="test_data/fixtures/does_not_exist.html"))
    assert r.status == Status.INVALID_INPUT


def test_history_failure_never_breaks_workflow(framework, make_request, fixture_html, monkeypatch):
    monkeypatch.setattr(framework.db, "record_execution", lambda **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    r = framework.run(make_request(html=fixture_html("no_captcha.html")))
    assert r.status == Status.NO_CAPTCHA


def test_http_timeout_on_official_api(settings, db, make_request, fixture_html, write_sites, monkeypatch):
    write_sites({"portal.example.org": {"official_api": {"base_url": "https://api.example.org", "auth_env": "TK"}}})
    monkeypatch.setenv("TK", "t")

    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    fw = CaptchaFramework(settings=settings, db=db, http_client=httpx.Client(transport=httpx.MockTransport(slow)))
    r = fw.run(make_request(html=fixture_html("recaptcha_mock.html"), url="https://portal.example.org/"))
    assert r.status == Status.HUMAN_VERIFICATION_REQUIRED and "ReadTimeout" in r.message
