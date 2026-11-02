"""Human-in-the-loop verification."""
import threading
import time

import httpx

from core.human_verification import Deadline, HumanVerificationManager
from core.response_handler import Status


def test_create_and_complete(db, settings):
    hv = HumanVerificationManager(db, settings)
    req = hv.create_request("T1", "recaptcha", "portal.example.org", None, 60)
    assert req["status"] == "PENDING" and req["request_id"].startswith("HV_")
    assert hv.complete(req["request_id"], "Jane")
    got = hv.get(req["request_id"])
    assert got["status"] == "COMPLETED" and got["operator"] == "Jane"
    # a final request cannot be changed again
    assert not hv.reject(req["request_id"], "Bob")


def test_reject(db, settings):
    hv = HumanVerificationManager(db, settings)
    rid = hv.create_request("T1", "hcaptcha", None, None, 60)["request_id"]
    assert hv.reject(rid, "Jane", "wrong page")
    assert hv.get(rid)["status"] == "REJECTED"


def test_expiry(db, settings):
    hv = HumanVerificationManager(db, settings)
    rid = hv.create_request("T1", "text", None, None, 1)["request_id"]
    time.sleep(1.3)
    assert hv.get(rid)["status"] == "EXPIRED"
    assert hv.pending() == []


def test_wait_returns_when_operator_completes(db, settings):
    hv = HumanVerificationManager(db, settings)
    rid = hv.create_request("T1", "recaptcha", None, None, 30)["request_id"]
    threading.Timer(0.5, lambda: hv.complete(rid, "Jane")).start()
    t0 = time.monotonic()
    assert hv.wait(rid, Deadline(10)) == "COMPLETED"
    assert time.monotonic() - t0 < 5


def test_wait_timeout(db, settings):
    hv = HumanVerificationManager(db, settings)
    rid = hv.create_request("T1", "recaptcha", None, None, 30)["request_id"]
    assert hv.wait(rid, Deadline(0.6)) == "TIMEOUT"


def test_wait_clearance_check_autocompletes(db, settings):
    hv = HumanVerificationManager(db, settings)
    rid = hv.create_request("T1", "recaptcha", None, None, 30)["request_id"]
    calls = {"n": 0}

    def clearance():
        calls["n"] += 1
        return calls["n"] >= 2

    assert hv.wait(rid, Deadline(10), clearance) == "COMPLETED"
    assert hv.get(rid)["operator"] == "auto"


def test_notifier_and_webhook(db, settings, monkeypatch):
    seen = []
    posted = []

    def handler(request):
        posted.append(request)
        return httpx.Response(200)

    monkeypatch.setenv("CAPTCHA_FW_NOTIFY_WEBHOOK", "https://hooks.example.test/notify")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    hv = HumanVerificationManager(db, settings, client, notifier=seen.append)
    hv.create_request("T1", "turnstile", "x.example", None, 30)
    assert seen and seen[0]["event"] == "captcha_verification_required"
    assert len(posted) == 1 and b"turnstile" in posted[0].content


def test_webhook_requires_https(db, settings, monkeypatch):
    posted = []
    monkeypatch.setenv("CAPTCHA_FW_NOTIFY_WEBHOOK", "http://insecure.example/notify")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: posted.append(r) or httpx.Response(200)))
    HumanVerificationManager(db, settings, client).create_request("T1", "text", None, None, 30)
    assert posted == []


def test_webhook_failure_does_not_break(db, settings, monkeypatch):
    def boom(request):
        raise httpx.ConnectError("down", request=request)

    monkeypatch.setenv("CAPTCHA_FW_NOTIFY_WEBHOOK", "https://hooks.example.test/notify")
    hv = HumanVerificationManager(db, settings, httpx.Client(transport=httpx.MockTransport(boom)))
    assert hv.create_request("T1", "text", None, None, 30)["status"] == "PENDING"


def test_end_to_end_async_flow(framework, make_request, fixture_html):
    """RPA gets HUMAN_VERIFICATION_REQUIRED, operator completes, RPA polls -> VERIFIED."""
    r1 = framework.run(make_request(html=fixture_html("recaptcha_mock.html"), url="https://portal.example.org/login"))
    assert r1.status == Status.HUMAN_VERIFICATION_REQUIRED and r1.request_id
    r2 = framework.run(make_request(action="check_status", request_id=r1.request_id))
    assert r2.status == Status.PENDING
    framework.hv.complete(r1.request_id, "Operator One")
    r3 = framework.run(make_request(action="check_status", request_id=r1.request_id))
    assert r3.status == Status.VERIFIED and r3.exit_code == 0


def test_end_to_end_blocking_flow(framework, make_request, fixture_html):
    def complete_soon():
        time.sleep(0.6)
        for r in framework.hv.pending():
            framework.hv.complete(r["request_id"], "Operator Two")

    threading.Thread(target=complete_soon, daemon=True).start()
    resp = framework.run(make_request(html=fixture_html("hcaptcha_mock.html"), wait_for_operator=True, timeout=10))
    assert resp.status == Status.VERIFIED


def test_blocking_flow_timeout_cancels_request(framework, make_request, fixture_html):
    resp = framework.run(make_request(html=fixture_html("hcaptcha_mock.html"), wait_for_operator=True, timeout=1))
    assert resp.status == Status.TIMEOUT
    assert framework.hv.get(resp.request_id)["status"] == "CANCELLED"


def test_cancel_action(framework, make_request, fixture_html):
    r1 = framework.run(make_request(html=fixture_html("recaptcha_mock.html")))
    r2 = framework.run(make_request(action="cancel", request_id=r1.request_id))
    assert r2.status == Status.REJECTED


def test_unknown_request_id(framework, make_request):
    assert framework.run(make_request(action="check_status", request_id="HV_nope")).status == Status.INVALID_INPUT


def test_purge_old_screenshots(db, settings):
    import os
    hv = HumanVerificationManager(db, settings)
    old = settings.screenshot_dir / "old.png"
    new = settings.screenshot_dir / "new.png"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    past = time.time() - 3 * 24 * 3600
    os.utime(old, (past, past))
    assert hv.purge_old_screenshots() == 1
    assert not old.exists() and new.exists()
