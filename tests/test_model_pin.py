"""CAPTCHA_FW_SIGNATURE_MODEL: deploy an already-approved signature model on hosts whose registry starts empty."""
from __future__ import annotations

import shutil

from tests.conftest import ROOT


def _framework(settings):
    from core.framework import CaptchaFramework
    return CaptchaFramework(settings=settings)


def _ship_model(settings, version="1.1.0"):
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "models" / f"classifier_v{version}.json", settings.models_dir)


def test_no_pin_keeps_baseline(settings):
    fw = _framework(settings)
    assert fw.apply_signature_model_pin() is None
    assert fw.detector.classifier.version == "1.0.0"


def test_pin_deploys_approved_version(settings):
    _ship_model(settings)
    settings.signature_model, settings.signature_model_approved_by = "1.1.0", "Bipin"
    fw = _framework(settings)
    assert fw.detector.classifier.version == "1.1.0"
    m = fw.db.get_model("1.1.0")
    assert m["status"] == "DEPLOYED" and m["approved_by"] == "Bipin" and "CAPTCHA_FW_SIGNATURE_MODEL" in m["notes"]
    assert fw.db.get_model("1.0.0")["status"] == "RETIRED"
    assert _framework(settings).apply_signature_model_pin() == "1.1.0"  # idempotent on restart


def test_pin_requires_named_approver_and_valid_model(settings):
    _ship_model(settings)
    for version, approver in (("1.1.0", None), ("1.1.0", "auto"), ("9.9.9", "Bipin"), ("../x", "Bipin")):
        settings.signature_model, settings.signature_model_approved_by = version, approver
        fw = _framework(settings)
        assert fw.apply_signature_model_pin() is None
        assert fw.detector.classifier.version == "1.0.0"


def test_pin_never_redeploys_rejected_model(settings):
    _ship_model(settings)
    fw = _framework(settings)
    fw.db.register_model("1.1.0", str(settings.models_dir / "classifier_v1.1.0.json"), None, None, None, "t")
    fw.db.set_model_status("1.1.0", "REJECTED")
    settings.signature_model, settings.signature_model_approved_by = "1.1.0", "Bipin"
    assert fw.apply_signature_model_pin() is None
    assert fw.detector.classifier.version == "1.0.0"
