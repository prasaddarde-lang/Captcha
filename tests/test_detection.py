"""CAPTCHA detection and classification."""
import pytest

from core.classifier import CaptchaClassifier, ModelError, PageFeatures, validate_model
from core.detector import CaptchaDetector, PageSnapshot, extract_features


@pytest.fixture(scope="module")
def detector():
    return CaptchaDetector(CaptchaClassifier())


@pytest.mark.parametrize("fixture,expected", [
    ("recaptcha_mock.html", "recaptcha"),
    ("hcaptcha_mock.html", "hcaptcha"),
    ("turnstile_mock.html", "turnstile"),
    ("text_captcha_mock.html", "text"),
    ("image_selection_mock.html", "image_selection"),
    ("audio_mock.html", "audio"),
])
def test_fixture_types_detected(detector, fixture_html, fixture, expected):
    res = detector.detect_html(fixture_html(fixture))
    assert res.detected
    assert res.classification.captcha_type == expected
    assert res.classification.confidence > 0.6


def test_no_captcha_page(detector, fixture_html):
    res = detector.detect_html(fixture_html("no_captcha.html"))
    assert not res.detected
    assert res.classification.captcha_type is None


@pytest.mark.parametrize("html", [
    "<html><body><h1>Hello</h1></body></html>",
    "<audio src='podcast.mp3' controls></audio><p>Our podcast</p>",
    "<label>Security question</label><input name='security_question'>",
    "",
])
def test_negatives(detector, html):
    assert not detector.detect_html(html).detected


def test_provider_from_frames_only(detector):
    # Live browsers report frame URLs even when the iframe is injected by script.
    res = detector.detect(PageSnapshot(html="<div id='x'></div>",
                                       frame_urls=["https://www.google.com/recaptcha/api2/anchor?k=abc"]))
    assert res.classification.captcha_type == "recaptcha"


def test_provider_beats_generic(detector):
    html = ("<div class='g-recaptcha' data-sitekey='k'></div><script src='https://www.google.com/recaptcha/api.js'></script>"
            "<label>Enter the characters you see</label><input name='captcha'><img alt='captcha'>")
    assert detector.detect_html(html).classification.captcha_type == "recaptcha"


@pytest.mark.parametrize("html,subtype", [
    ("<script src='https://www.google.com/recaptcha/api.js?render=6LcABCDEFGHIJKLMNOP'></script>", "v3_invisible"),
    ("<script src='https://www.google.com/recaptcha/enterprise.js'></script><div class='g-recaptcha'></div>", "enterprise"),
    ("<div class='g-recaptcha'></div><script src='https://www.google.com/recaptcha/api.js'></script>", "v2_checkbox"),
    ("<iframe src='https://www.google.com/recaptcha/api2/bframe?k=x'></iframe>", "image_challenge"),
    ("<div id='challenge-stage'></div><div id='challenge-form'></div><iframe src='https://challenges.cloudflare.com/x'></iframe>",
     "managed_challenge"),
])
def test_subtypes(detector, html, subtype):
    assert detector.detect_html(html).classification.subtype == subtype


def test_solved_widget_detected_without_reading_value(detector, fixture_html):
    res = detector.detect_html(fixture_html("recaptcha_completed_mock.html"))
    assert res.detected and res.is_solved
    # the response value itself must never be retained in extracted features
    assert "MOCK-COMPLETED-RESPONSE" not in res.features.text
    assert all("MOCK-COMPLETED-RESPONSE" not in a for a in res.features.attrs)


def test_unsolved_widget(detector, fixture_html):
    assert not detector.detect_html(fixture_html("recaptcha_mock.html")).is_solved


def test_malformed_html_does_not_crash(detector):
    res = detector.detect_html("<div class='h-captcha' <<<>>> <script src='https://js.hcaptcha.com/1/api.js'")
    assert res is not None


def test_scripts_and_styles_excluded_from_text():
    f = extract_features("<script>var s='select all images with buses';</script><p>Hello</p>")
    assert "select all" not in f.text and "hello" in f.text


def test_mock_marker_extracted(fixture_html):
    assert extract_features(fixture_html("hcaptcha_mock.html")).mock_marker == "hcaptcha"


def test_signals_reported(detector, fixture_html):
    sigs = {s["signal"] for s in detector.detect_html(fixture_html("hcaptcha_mock.html")).classification.signals}
    assert {"hcaptcha_script", "hcaptcha_class"} <= sigs


def test_detect_file(detector):
    from tests.conftest import FIXTURES
    assert detector.detect_file(FIXTURES / "turnstile_mock.html").classification.captcha_type == "turnstile"


# ------------------------------------------------------------------ classifier
def test_model_validation_rejects_bad_kind():
    with pytest.raises(ModelError):
        validate_model({"version": "x", "threshold": 1, "types": {"a": [{"id": "r", "kind": "nope", "pattern": "x", "weight": 1}]}})


def test_model_validation_rejects_bad_regex():
    with pytest.raises(ModelError):
        validate_model({"version": "x", "threshold": 1, "types": {"a": [{"id": "r", "kind": "class", "pattern": "(", "weight": 1}]}})


def test_threshold_respected():
    model = {"version": "t", "threshold": 10, "types": {"a": [{"id": "r", "kind": "class", "pattern": "^a$", "weight": 1}]}}
    clf = CaptchaClassifier(model)
    assert not clf.classify(PageFeatures(classes=["a"])).captcha_detected


def test_confidence_monotonic():
    clf = CaptchaClassifier()
    assert clf.confidence(1) < clf.confidence(3) < clf.confidence(8) < 1.0


def test_load_active_uses_deployed_model(settings, db, tmp_path):
    import json
    model = CaptchaClassifier().model
    model["version"] = "9.9.9"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(model))
    db.register_model("9.9.9", str(p), 1.0, 1.0, None, "t")
    db.set_model_status("9.9.9", "APPROVED", "Reviewer")
    db.set_model_status("9.9.9", "DEPLOYED")
    assert CaptchaClassifier.load_active(db).version == "9.9.9"


def test_load_active_falls_back_when_file_missing(settings, db):
    db.register_model("8.0.0", "C:/does/not/exist.json", None, None, None, "t")
    db.set_model_status("8.0.0", "APPROVED", "R")
    db.set_model_status("8.0.0", "DEPLOYED")
    assert CaptchaClassifier.load_active(db).version == "1.0.0"


# ------------------------------------------------- real page: BSE BEFS login (text CAPTCHA)
def _befs_html():
    from tests.conftest import FIXTURES
    return (FIXTURES / "bse_befs_login.html").read_text(encoding="utf-8")


def test_befs_login_missed_by_baseline_model():
    """Documents the real-world miss that led to model 1.1.0 (score 1.5 < threshold 2.0)."""
    c = CaptchaDetector(CaptchaClassifier()).detect_html(_befs_html()).classification
    assert not c.captcha_detected and c.scores["text"] == 1.5


def test_befs_login_detected_by_candidate_1_1_0():
    from tests.conftest import ROOT
    from core.classifier import load_model_file
    model = load_model_file(ROOT / "models" / "classifier_v1.1.0.json")
    c = CaptchaDetector(CaptchaClassifier(model)).detect_html(_befs_html()).classification
    assert c.captcha_detected and c.captcha_type == "text"
    assert {"text_captcha_image", "text_captcha_label"} <= {s["signal"] for s in c.signals}


def test_candidate_1_1_0_keeps_negatives_negative(fixture_html):
    from tests.conftest import ROOT
    from core.classifier import load_model_file
    model = load_model_file(ROOT / "models" / "classifier_v1.1.0.json")
    det = CaptchaDetector(CaptchaClassifier(model))
    for html in (fixture_html("no_captcha.html"), "<label>Enter your name</label><input name='name'>",
                 "<p>Our blog explains what a captcha is.</p>"):
        assert not det.detect_html(html).detected
