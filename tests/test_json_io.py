"""JSON input/output contract and the command-line interface."""
import json
import os
import subprocess
import sys

import pytest

from core.response_handler import EXIT_CODES, Status, TaskResponse, parse_request
from core.security import redact
from tests.conftest import ROOT


def test_example_input_from_spec_is_valid():
    req = parse_request('{"task_id":"RPA_001","platform":"UiPath","browser":"Chrome",'
                        '"action":"detect_and_handle","timeout":120,"mode":"production","url":"https://x.example/login"}')
    assert req.task_id == "RPA_001" and req.timeout == 120


@pytest.mark.parametrize("raw,fragment", [
    ("not json", "not valid JSON"),
    ("[1,2]", "must be an object"),
    ('{"task_id":"A"}', "provide one of"),
    ('{"task_id":"A","html":"x","unexpected":1}', "unexpected"),
    ('{"task_id":"A","html":"x","timeout":0}', "timeout"),
    ('{"task_id":"A","html":"x","timeout":99999}', "timeout"),
    ('{"task_id":"A","html":"x","mode":"hack"}', "mode"),
    ('{"task_id":"A","action":"check_status"}', "request_id"),
    ('{"task_id":"A","url":"javascript:alert(1)"}', "scheme"),
    ('{"task_id":"A","url":"https://user:pw@site.example"}', "Credentials"),
    ('{"task_id":"../../etc","html":"x"}', "task_id"),
])
def test_invalid_inputs(raw, fragment):
    with pytest.raises(ValueError) as exc:
        parse_request(raw)
    assert fragment.lower() in str(exc.value).lower()


def test_bom_tolerated():
    assert parse_request('﻿{"task_id":"A","html":"x"}').task_id == "A"


def test_response_matches_spec_shape():
    r = TaskResponse(task_id="RPA_001", status=Status.HUMAN_VERIFICATION_REQUIRED, captcha_detected=True,
                     captcha_type="recaptcha", resolution_method="human_in_the_loop", message="Operator verification required")
    d = json.loads(r.to_json())
    for key in ("task_id", "status", "captcha_detected", "captcha_type", "resolution_method", "message", "exit_code"):
        assert key in d
    assert d["status"] == "HUMAN_VERIFICATION_REQUIRED" and d["exit_code"] == 11


def test_every_status_has_exit_code():
    assert set(EXIT_CODES) == set(Status)


def test_redaction():
    text = 'api_key=abc123 Authorization: Bearer eyJhbGciOi.payload.sig "g-recaptcha-response": "03AGdBq24xyz"'
    out = redact(text)
    assert "abc123" not in out and "eyJhbGciOi" not in out and "03AGdBq24xyz" not in out


# ------------------------------------------------------------------------ CLI
def _cli(args, settings_env, stdin=None, timeout=60):
    env = {**os.environ, **settings_env, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(ROOT / "main.py"), *args], input=stdin, capture_output=True,
                          text=True, cwd=str(ROOT), env=env, timeout=timeout, encoding="utf-8")


@pytest.fixture
def cli_env(settings):
    return {k: v for k, v in os.environ.items() if k.startswith("CAPTCHA_FW_")}


def test_cli_input_file_and_output_file(cli_env, tmp_path):
    out = tmp_path / "result.json"
    p = _cli(["run", "-i", "samples/sample_input_test.json", "-o", str(out)], cli_env)
    assert p.returncode == 0, p.stderr
    stdout = json.loads(p.stdout)
    assert stdout["status"] == "TEST_VERIFIED"
    assert json.loads(out.read_text(encoding="utf-8"))["task_id"] == "RPA_TEST_001"


def test_cli_stdin(cli_env):
    payload = json.dumps({"task_id": "STDIN_1", "mode": "test", "html_file": "test_data/fixtures/text_captcha_mock.html"})
    p = _cli([], cli_env, stdin=payload)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout)["captcha_type"] == "text"


def test_cli_inline_json_and_overrides(cli_env):
    payload = json.dumps({"task_id": "J1", "html_file": "test_data/fixtures/audio_mock.html", "test_response": "fail"})
    p = _cli(["run", "-j", payload, "--mode", "test", "--timeout", "15"], cli_env)
    assert p.returncode == EXIT_CODES[Status.TEST_FAILED]
    assert json.loads(p.stdout)["status"] == "TEST_FAILED"


def test_cli_invalid_json_exit_code(cli_env):
    p = _cli(["run", "-j", "{broken"], cli_env)
    assert p.returncode == 40 and json.loads(p.stdout)["status"] == "INVALID_INPUT"


def test_cli_missing_file(cli_env):
    p = _cli(["run", "-i", "nope.json"], cli_env)
    assert p.returncode == 40


def test_cli_human_required_and_operator_commands(cli_env):
    p = _cli(["run", "-i", "samples/sample_input_production_html.json"], cli_env)
    assert p.returncode == 11
    rid = json.loads(p.stdout)["request_id"]
    lst = _cli(["operator", "list", "--json"], cli_env)
    assert rid in lst.stdout
    done = _cli(["operator", "complete", rid, "--operator", "CLI Operator"], cli_env)
    assert json.loads(done.stdout)["status"] == "COMPLETED"
    status_req = json.dumps({"task_id": "RPA_001", "action": "check_status", "request_id": rid})
    chk = _cli(["run", "-j", status_req], cli_env)
    assert chk.returncode == 0 and json.loads(chk.stdout)["status"] == "VERIFIED"


def test_cli_blue_prism_format(cli_env):
    p = _cli(["run", "-i", "samples/sample_input_detect_only.json", "-p", "blue_prism"], cli_env)
    lines = dict(line.split("=", 1) for line in p.stdout.strip().splitlines())
    assert lines["status"] == "CAPTCHA_DETECTED" and lines["captcha_type"] == "hcaptcha"


def test_stdout_contains_only_json(cli_env):
    """Diagnostics must never pollute stdout, which RPA tools parse."""
    p = _cli(["run", "-i", "samples/sample_input_test.json"], cli_env)
    json.loads(p.stdout)
