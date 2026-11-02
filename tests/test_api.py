"""REST API (FastAPI TestClient)."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from api.server import create_app


@pytest.fixture
def client(framework):
    return TestClient(create_app(framework))


TASK = {"task_id": "API_1", "mode": "test", "html_file": "test_data/fixtures/recaptcha_mock.html"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_run_task(client):
    r = client.post("/v1/tasks", json=TASK)
    assert r.status_code == 200 and r.json()["status"] == "TEST_VERIFIED"


def test_validation_error(client):
    r = client.post("/v1/tasks", json={"task_id": "bad id"})
    assert r.status_code == 422


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("CAPTCHA_FW_API_KEY", "k-123")
    assert client.post("/v1/tasks", json=TASK).status_code == 401
    assert client.post("/v1/tasks", json=TASK, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/v1/tasks", json=TASK, headers={"X-API-Key": "k-123"}).status_code == 200


def test_non_local_client_needs_key(framework):
    remote = TestClient(create_app(framework), client=("10.1.2.3", 5000))
    assert remote.get("/v1/verifications").status_code == 403


def test_operator_flow(client):
    r = client.post("/v1/tasks", json={"task_id": "API_2", "html_file": "test_data/fixtures/turnstile_mock.html"})
    rid = r.json()["request_id"]
    assert r.json()["status"] == "HUMAN_VERIFICATION_REQUIRED"
    pending = client.get("/v1/verifications").json()
    assert [p["request_id"] for p in pending] == [rid]
    assert client.post(f"/v1/verifications/{rid}/complete", json={"operator": ""}).status_code == 422
    done = client.post(f"/v1/verifications/{rid}/complete", json={"operator": "Jane"})
    assert done.status_code == 200 and done.json()["status"] == "COMPLETED"
    assert client.post(f"/v1/verifications/{rid}/reject", json={"operator": "Jane"}).status_code == 409
    st = client.post("/v1/tasks", json={"task_id": "API_2", "action": "check_status", "request_id": rid})
    assert st.json()["status"] == "VERIFIED"


def test_unknown_verification(client):
    assert client.get("/v1/verifications/HV_none").status_code == 404
    assert client.get("/v1/verifications/HV_none/screenshot").status_code == 404


def test_screenshot_outside_dir_refused(client, framework, tmp_path):
    outside = tmp_path / "evil.png"
    outside.write_bytes(b"x")
    vr = framework.hv.create_request("T", "text", None, str(outside), 60)
    assert client.get(f"/v1/verifications/{vr['request_id']}/screenshot").status_code == 403


def test_screenshot_served(client, framework, settings):
    p = settings.screenshot_dir / "s.png"
    p.write_bytes(b"\x89PNG\r\n")
    vr = framework.hv.create_request("T", "text", None, str(p), 60)
    r = client.get(f"/v1/verifications/{vr['request_id']}/screenshot")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_feedback_and_report(client):
    client.post("/v1/tasks", json=TASK)
    eid = client.get("/v1/executions").json()[0]["id"]
    assert client.post("/v1/feedback", json={"execution_id": eid, "actual_type": "recaptcha"}).status_code == 200
    assert client.post("/v1/feedback", json={"execution_id": 99999, "actual_type": "x"}).status_code == 404
    rep = client.get("/v1/reports/performance").json()
    assert rep["detection_accuracy"]["labelled_samples"] == 1


def test_models_endpoints(client, framework):
    models = client.get("/v1/models").json()
    assert models[0]["status"] == "DEPLOYED"
    framework.db.register_model("1.0.5", models[0]["file_path"], 1.0, 1.0,
                                json.dumps({"candidate": {"accuracy": 1.0}, "baseline": {"accuracy": 1.0}}), "t")
    assert client.post("/v1/models/1.0.5/deploy").status_code == 409  # not approved
    assert client.post("/v1/models/1.0.5/approve", json={"approver": "auto"}).status_code == 409
    assert client.post("/v1/models/1.0.5/approve", json={"approver": "Jane"}).status_code == 200
    assert client.post("/v1/models/1.0.5/deploy").json()["status"] == "DEPLOYED"


def test_provider_verify(client, write_sites, mock_http, monkeypatch):
    write_sites({"intranet.local": {"owned": True, "provider_secret_env": "OWN_SECRET"},
                 "other.example": {"owned": False}})
    monkeypatch.setenv("OWN_SECRET", "s")
    mock_http.add("POST", "https://www.google.com/recaptcha/api/siteverify",
                  lambda r: httpx.Response(200, json={"success": True, "hostname": "intranet.local"}))
    ok = client.post("/v1/provider/verify", json={"site": "intranet.local", "provider": "recaptcha", "token": "t"})
    assert ok.status_code == 200 and ok.json()["success"] is True
    no = client.post("/v1/provider/verify", json={"site": "other.example", "provider": "recaptcha", "token": "t"})
    assert no.status_code == 403
    assert client.post("/v1/provider/verify", json={"site": "nope", "provider": "recaptcha", "token": "t"}).status_code == 404


def test_operator_console_page(client):
    r = client.get("/operator")
    assert r.status_code == 200 and "Operator Console" in r.text
