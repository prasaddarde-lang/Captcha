"""Live browser against a local mock website, plus installation / BAT launcher checks."""
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading

import pytest

from core.browser import BrowserSession, BrowserUnavailable
from core.response_handler import Status
from learning.dataset_manager import SyntheticDatasetManager
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def mock_site(tmp_path_factory):
    from config import get_settings
    site = SyntheticDatasetManager(get_settings()).write_mock_site(tmp_path_factory.mktemp("site"))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _session_factory(**kw):
    # Use Edge (preinstalled on Windows) if no Playwright browser was downloaded; never hit the internet.
    kw["browser"] = "msedge"
    return BrowserSession(block_external=True, **kw)


@pytest.fixture(scope="module")
def browser_ok():
    try:
        with BrowserSession(browser="msedge", headless=True, timeout_s=30):
            return True
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no browser available for Playwright: {exc}")


@pytest.mark.parametrize("page,expected", [("recaptcha", "recaptcha"), ("hcaptcha", "hcaptcha"),
                                           ("text", "text"), ("none", None)])
def test_live_browser_detection(browser_ok, mock_site, settings, db, make_request, page, expected):
    from core.framework import CaptchaFramework
    fw = CaptchaFramework(settings=settings, db=db, session_factory=_session_factory)
    r = fw.run(make_request(url=f"{mock_site}/{page}.html", mode="test", timeout=60, screenshot=True))
    assert r.captcha_type == expected
    if expected:
        assert r.status == Status.TEST_VERIFIED  # localhost is an allowed test host
        assert r.screenshot_path and os.path.exists(r.screenshot_path)
    else:
        assert r.status == Status.NO_CAPTCHA


def test_browser_unreachable_page(browser_ok, settings, db, make_request):
    from core.framework import CaptchaFramework
    fw = CaptchaFramework(settings=settings, db=db, session_factory=_session_factory)
    r = fw.run(make_request(url="http://127.0.0.1:9/never", timeout=30))
    assert r.status in (Status.NETWORK_ERROR, Status.ERROR)


def test_cdp_endpoint_unreachable(settings, db, make_request):
    from core.framework import CaptchaFramework
    fw = CaptchaFramework(settings=settings, db=db)
    r = fw.run(make_request(cdp_endpoint="http://127.0.0.1:9", timeout=20))
    assert r.status in (Status.NETWORK_ERROR, Status.ERROR, Status.TIMEOUT)
    assert r.exit_code != 0


# --------------------------------------------------------------- installation
def test_requirements_cover_imports():
    reqs = (ROOT / "requirements.txt").read_text().lower()
    for pkg in ("pydantic", "fastapi", "uvicorn", "httpx", "playwright", "opencv-python-headless", "numpy", "pytest"):
        assert pkg in reqs


def test_install_bat_steps():
    text = (ROOT / "install.bat").read_text(encoding="utf-8")
    for needle in ("sys.version_info >= (3,12)", "-m venv", "pip install -r requirements.txt",
                   "playwright install chromium", "SETUP RESULTS", "pytest"):
        assert needle in text


@pytest.mark.skipif(os.name != "nt", reason="Windows BAT launcher")
@pytest.mark.parametrize("args,code,status", [
    (["samples\\sample_input_test.json"], 0, "TEST_VERIFIED"),
    (["samples\\sample_input_detect_only.json"], 10, "CAPTCHA_DETECTED"),
    (["samples\\missing.json"], 40, "INVALID_INPUT"),
])
def test_run_bat(settings, tmp_path, args, code, status):
    env = {**os.environ, "CAPTCHA_FW_PYTHON": sys.executable}
    out = tmp_path / "result.json"
    cmd = ["cmd.exe", "/c", str(ROOT / "run.bat"), *args, str(out)]
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120, encoding="utf-8")
    assert p.returncode == code, p.stdout + p.stderr
    assert json.loads(p.stdout)["status"] == status
    if code != 40:
        assert json.loads(out.read_text(encoding="utf-8"))["status"] == status


@pytest.mark.skipif(os.name != "nt", reason="Windows BAT launcher")
def test_run_bat_stdin_and_passthrough(settings):
    env = {**os.environ, "CAPTCHA_FW_PYTHON": sys.executable}
    payload = json.dumps({"task_id": "BAT_STDIN", "mode": "test", "html_file": "test_data/fixtures/audio_mock.html"})
    p = subprocess.run(["cmd.exe", "/c", str(ROOT / "run.bat")], input=payload, cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=120, encoding="utf-8")
    assert p.returncode == 0 and json.loads(p.stdout)["captcha_type"] == "audio"
    p2 = subprocess.run(["cmd.exe", "/c", str(ROOT / "run.bat"), "operator", "list"], cwd=str(ROOT), env=env,
                        capture_output=True, text=True, timeout=120, encoding="utf-8")
    assert p2.returncode == 0
