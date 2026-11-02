"""RPA platform adapters."""
import json

import pytest

from core.response_handler import Status
from integrations import generic_rpa
from integrations.automation_anywhere import AutomationAnywhereAdapter, aa_handle, aa_status, parse_kv
from integrations.blue_prism import BluePrismAdapter, to_collection
from integrations.generic_rpa import get_adapter, handle_json
from integrations.power_automate import PowerAutomateAdapter, pad_handle
from integrations.uipath import UiPathAdapter, handle as uipath_handle


@pytest.fixture(autouse=True)
def inject(framework):
    generic_rpa.set_framework(framework)
    yield
    generic_rpa.set_framework(None)


TASK = {"task_id": "INT_1", "mode": "test", "html_file": "test_data/fixtures/recaptcha_mock.html"}


@pytest.mark.parametrize("name,cls", [
    ("uipath", UiPathAdapter), ("UiPath", UiPathAdapter), ("aa", AutomationAnywhereAdapter),
    ("a360", AutomationAnywhereAdapter), ("Blue Prism", BluePrismAdapter), ("blueprism", BluePrismAdapter),
    ("pad", PowerAutomateAdapter), ("power_automate", PowerAutomateAdapter), ("selenium", generic_rpa.GenericRPAAdapter),
    (None, generic_rpa.GenericRPAAdapter),
])
def test_adapter_lookup(name, cls):
    assert type(get_adapter(name)) is cls


def test_uipath_string_types_and_compact_output(framework):
    out = uipath_handle(json.dumps({**TASK, "timeout": "20", "screenshot": "False"}))
    assert "\n" not in out
    d = json.loads(out)
    assert d["status"] == "TEST_VERIFIED"
    assert framework.db.list_executions(1)[0]["platform"] == "UiPath"


def test_automation_anywhere_json_and_kv():
    d = json.loads(aa_handle(json.dumps(TASK)))
    assert d["status"] == "TEST_VERIFIED" and "signals" not in d
    kv = "task_id=AA_1;mode=test;html_file=test_data/fixtures/hcaptcha_mock.html;timeout=15;test_response=fail"
    assert aa_status(kv) == "TEST_FAILED"


def test_parse_kv():
    assert parse_kv("a=1;timeout=30;wait_for_operator=true;url=https://x.example/?q=a=b") == {
        "a": "1", "timeout": 30, "wait_for_operator": True, "url": "https://x.example/?q=a=b"}


def test_blue_prism_kv_and_collection():
    adapter = BluePrismAdapter()
    resp = adapter.handle({**TASK, "wait_for_operator": "False"})
    text = adapter.format_output(resp)
    flat = dict(line.split("=", 1) for line in text.splitlines())
    assert flat["status"] == "TEST_VERIFIED" and flat["captcha_detected"] == "true"
    assert to_collection(resp)[0]["captcha_type"] == "recaptcha"
    assert json.loads(BluePrismAdapter("json").format_output(resp))["exit_code"] == "0"


def test_power_automate():
    d = json.loads(pad_handle(json.dumps({**TASK, "timeout": "30"})))
    assert d["status"] == "TEST_VERIFIED" and isinstance(d["signals"], list)


def test_generic_handle_json_invalid():
    d = json.loads(handle_json("{oops"))
    assert d["status"] == Status.INVALID_INPUT.value


def test_generic_handle_file(tmp_path):
    inp = tmp_path / "in.json"
    out = tmp_path / "out" / "result.json"
    inp.write_text(json.dumps(TASK), encoding="utf-8-sig")  # RPA tools often write a BOM
    resp = get_adapter("generic").handle_file(inp, out)
    assert resp.status == Status.TEST_VERIFIED
    assert json.loads(out.read_text(encoding="utf-8"))["task_id"] == "INT_1"


def test_handle_file_missing(tmp_path):
    assert get_adapter("generic").handle_file(tmp_path / "missing.json").status == Status.INVALID_INPUT


def test_python_rpa_selenium_style_usage(framework):
    """Python-based RPA / Selenium / Playwright scripts pass page HTML directly."""
    page_source = "<div class='cf-turnstile' data-sitekey='0xabc'></div>"
    d = json.loads(handle_json(json.dumps({"task_id": "SEL_1", "platform": "selenium", "action": "detect",
                                           "html": page_source, "url": "https://shop.example.com/checkout"})))
    assert d["captcha_type"] == "turnstile" and d["status"] == "CAPTCHA_DETECTED"
