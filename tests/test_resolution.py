"""Resolution engine: test mode policy, official API routing, provider verification, network failures."""
import httpx
import pytest

from core.resolution_engine import verify_provider_token
from core.response_handler import Status

PORTAL = "https://portal.example.org/login"
API_SITE = {
    "portal.example.org": {"official_api": {"base_url": "https://api.example.org/v1", "health_path": "/status",
                                            "auth_env": "PORTAL_TOKEN", "docs_url": "https://dev.example.org"}}
}


# ------------------------------------------------------------------ test mode
@pytest.mark.parametrize("answer,status", [("pass", Status.TEST_VERIFIED), ("fail", Status.TEST_FAILED)])
def test_test_mode_responses(framework, make_request, fixture_html, answer, status):
    r = framework.run(make_request(mode="test", html=fixture_html("recaptcha_mock.html"), test_response=answer))
    assert r.status == status and r.resolution_method == "test_mode"


def test_test_mode_timeout_response(framework, make_request, fixture_html):
    r = framework.run(make_request(mode="test", timeout=1, html=fixture_html("audio_mock.html"), test_response="timeout"))
    assert r.status == Status.TIMEOUT


def test_test_mode_blocked_on_production_host(framework, make_request, fixture_html):
    r = framework.run(make_request(mode="test", html=fixture_html("recaptcha_mock.html"), url="https://www.real-bank.com/login"))
    assert r.status == Status.POLICY_BLOCKED


def test_test_mode_blocked_for_unmarked_html(framework, make_request):
    html = "<div class='g-recaptcha'></div><script src='https://www.google.com/recaptcha/api.js'></script>"
    assert framework.run(make_request(mode="test", html=html)).status == Status.POLICY_BLOCKED


def test_test_mode_allowed_on_localhost(framework, make_request):
    html = "<div class='g-recaptcha'></div><script src='https://www.google.com/recaptcha/api.js'></script>"
    r = framework.run(make_request(mode="test", html=html, url="http://localhost:8080/login"))
    assert r.status == Status.TEST_VERIFIED


def test_test_mode_config_file(framework, make_request, fixture_html, settings):
    import json
    settings.test_data_dir.mkdir(parents=True, exist_ok=True)
    (settings.test_data_dir / "test_responses.json").write_text(json.dumps({"default": "pass", "by_type": {"hcaptcha": "fail"}}))
    assert framework.run(make_request(mode="test", html=fixture_html("hcaptcha_mock.html"))).status == Status.TEST_FAILED
    assert framework.run(make_request(mode="test", html=fixture_html("turnstile_mock.html"))).status == Status.TEST_VERIFIED


def test_test_response_requires_test_mode(make_request):
    with pytest.raises(ValueError):
        make_request(html="<p>x</p>", test_response="pass")


# --------------------------------------------------------------- official API
def test_official_api_route(framework, make_request, fixture_html, write_sites, mock_http, monkeypatch):
    write_sites(API_SITE)
    monkeypatch.setenv("PORTAL_TOKEN", "secret-token-value")
    seen = {}

    def ok(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"ok": True})

    mock_http.add("GET", "https://api.example.org/v1/status", ok)
    r = framework.run(make_request(html=fixture_html("recaptcha_mock.html"), url=PORTAL))
    assert r.status == Status.OFFICIAL_API_ROUTE
    assert r.official_api["base_url"] == "https://api.example.org/v1"
    assert seen["auth"] == "Bearer secret-token-value"
    assert "secret-token-value" not in r.to_json()  # secrets never returned
    attempts = framework.db.query("SELECT * FROM api_attempts")
    assert attempts and attempts[0]["success"] == 1


def test_official_api_missing_credential_falls_back(framework, make_request, fixture_html, write_sites, monkeypatch):
    write_sites(API_SITE)
    monkeypatch.delenv("PORTAL_TOKEN", raising=False)
    r = framework.run(make_request(html=fixture_html("recaptcha_mock.html"), url=PORTAL))
    assert r.status == Status.HUMAN_VERIFICATION_REQUIRED


def test_official_api_unauthorized_falls_back_without_retry(framework, make_request, fixture_html, write_sites, mock_http, monkeypatch):
    write_sites(API_SITE)
    monkeypatch.setenv("PORTAL_TOKEN", "bad")
    mock_http.add("GET", "https://api.example.org", lambda r: httpx.Response(401))
    r = framework.run(make_request(html=fixture_html("recaptcha_mock.html"), url=PORTAL))
    assert r.status == Status.HUMAN_VERIFICATION_REQUIRED
    assert "authorization failed" in r.message
    assert len(mock_http.calls) == 1


def test_official_api_rate_limit_not_hammered(framework, make_request, fixture_html, write_sites, mock_http, monkeypatch):
    write_sites(API_SITE)
    monkeypatch.setenv("PORTAL_TOKEN", "t")
    mock_http.add("GET", "https://api.example.org", lambda r: httpx.Response(429))
    framework.run(make_request(html=fixture_html("recaptcha_mock.html"), url=PORTAL))
    assert len(mock_http.calls) == 1


def test_official_api_network_failure_retries_then_falls_back(framework, make_request, fixture_html, write_sites, mock_http, monkeypatch):
    write_sites(API_SITE)
    monkeypatch.setenv("PORTAL_TOKEN", "t")  # no route registered -> ConnectError
    r = framework.run(make_request(html=fixture_html("recaptcha_mock.html"), url=PORTAL))
    assert r.status == Status.HUMAN_VERIFICATION_REQUIRED
    assert "network error" in r.message
    assert len(mock_http.calls) == 2  # 1 + HTTP_RETRIES(1)
    assert framework.db.query("SELECT success FROM api_attempts")[0]["success"] == 0


def test_official_api_requires_https(framework, make_request, fixture_html, write_sites, mock_http, monkeypatch):
    write_sites({"portal.example.org": {"official_api": {"base_url": "http://api.example.org/v1"}}})
    r = framework.run(make_request(html=fixture_html("recaptcha_mock.html"), url=PORTAL))
    assert r.status == Status.HUMAN_VERIFICATION_REQUIRED and mock_http.calls == []


def test_site_key_override(framework, make_request, fixture_html, write_sites, mock_http, monkeypatch):
    write_sites({"my-portal": API_SITE["portal.example.org"]})
    monkeypatch.setenv("PORTAL_TOKEN", "t")
    mock_http.add("GET", "https://api.example.org", lambda r: httpx.Response(200))
    r = framework.run(make_request(html=fixture_html("hcaptcha_mock.html"), site="my-portal"))
    assert r.status == Status.OFFICIAL_API_ROUTE


def test_detect_only_reports_planned_method(framework, make_request, fixture_html):
    r = framework.run(make_request(action="detect", html=fixture_html("turnstile_mock.html")))
    assert r.status == Status.CAPTCHA_DETECTED and r.resolution_method == "human_in_the_loop"
    assert framework.hv.pending() == []  # detect never creates operator work


def test_no_captcha(framework, make_request, fixture_html):
    r = framework.run(make_request(html=fixture_html("no_captcha.html")))
    assert r.status == Status.NO_CAPTCHA and r.exit_code == 0


def test_already_completed_widget(framework, make_request, fixture_html):
    r = framework.run(make_request(html=fixture_html("recaptcha_completed_mock.html")))
    assert r.status == Status.VERIFIED and r.resolution_method == "already_completed"


# ------------------------------------------------------- provider verification
def test_provider_verify_owned_site(settings, monkeypatch):
    monkeypatch.setenv("OWN_SECRET", "s3cret")
    body = {}

    def handler(request):
        body["data"] = request.content.decode()
        return httpx.Response(200, json={"success": True, "hostname": "intranet.local"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = verify_provider_token("recaptcha", "user-token", {"owned": True, "provider_secret_env": "OWN_SECRET"},
                                settings, http_client=client)
    assert out["success"] is True
    assert "secret=s3cret" in body["data"] and "response=user-token" in body["data"]


def test_provider_verify_refused_for_unowned_site(settings):
    with pytest.raises(PermissionError):
        verify_provider_token("recaptcha", "t", {"owned": False}, settings)


def test_provider_verify_missing_secret(settings, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    with pytest.raises(RuntimeError):
        verify_provider_token("hcaptcha", "t", {"owned": True, "provider_secret_env": "NOPE"}, settings)


def test_custom_resolver_chain(settings, db, make_request, fixture_html):
    """Resolvers are pluggable without touching the rest of the framework."""
    from core.framework import CaptchaFramework
    from core.resolution_engine import ResolutionOutcome

    class AlwaysEscalate:
        name = "ticket_system"
        def can_handle(self, ctx): return True
        def resolve(self, ctx): return ResolutionOutcome(Status.HUMAN_VERIFICATION_REQUIRED, self.name, "ticket opened")

    fw = CaptchaFramework(settings=settings, db=db)
    fw.engine.resolvers = [AlwaysEscalate()]
    r = fw.run(make_request(html=fixture_html("recaptcha_mock.html")))
    assert r.resolution_method == "ticket_system"
