"""Learning: synthetic datasets, evaluation, candidate training, human approval, analytics, reports."""
import copy
import json

import pytest

from core.classifier import CaptchaClassifier
from core.response_handler import Status
from learning.analytics import CaptchaAnalytics
from learning.dataset_manager import LABELS, SyntheticDatasetManager, preprocess_image
from learning.model_evaluation import (ApprovalError, ModelRegistry, bump_version, evaluate_html,
                                       predict_from_signals, train_weights, to_signal_sample)
from learning.performance_report import build_report, render_html, write_report


@pytest.fixture
def dataset(settings):
    dm = SyntheticDatasetManager(settings)
    dm.generate_html_dataset(per_label=8, seed=3)
    return dm


def test_html_dataset_generation(dataset):
    samples = dataset.load()
    assert len(samples) == 8 * len(LABELS)
    assert {s.label for s in samples} == set(LABELS)
    html = samples[0].html(dataset.root)
    assert "data-mock-captcha" in html


def test_generation_is_deterministic(settings, tmp_path):
    a = SyntheticDatasetManager(settings, tmp_path / "a")
    b = SyntheticDatasetManager(settings, tmp_path / "b")
    a.generate_html_dataset(3, seed=11)
    b.generate_html_dataset(3, seed=11)
    assert a.load_labeled_html() == b.load_labeled_html()


def test_image_dataset_and_preprocessing(settings):
    dm = SyntheticDatasetManager(settings)
    items = dm.generate_image_dataset(count=3, seed=5)
    assert len(items) == 3 and all(i["label"] == "text" for i in items)
    binary = preprocess_image(dm.root / items[0]["file"])
    assert binary.shape == (60, 200) and set(binary.flatten().tolist()) <= {0, 255}


def test_mock_site(settings, tmp_path):
    site = SyntheticDatasetManager(settings).write_mock_site(tmp_path / "site")
    assert (site / "index.html").exists() and (site / "recaptcha.html").exists()


def test_split_keeps_every_label(dataset):
    train, test = SyntheticDatasetManager.split(dataset.load_labeled_html(), 0.25)
    assert {l for _, l in test} == set(LABELS) and {l for _, l in train} == set(LABELS)


def test_baseline_evaluation(dataset):
    m = evaluate_html(CaptchaClassifier(), dataset.load_labeled_html())
    assert m["accuracy"] >= 0.95 and m["macro_f1"] >= 0.95
    assert set(m["per_class"]) == set(LABELS)


def _weakened_model():
    """Baseline with hCaptcha signals crippled - simulates a model that misses a type."""
    model = copy.deepcopy(CaptchaClassifier().model)
    for r in model["types"]["hcaptcha"]:
        r["weight"] = 0.3
    model["version"] = "1.0.0"
    return model


def test_training_fixes_weak_type(dataset):
    weak = _weakened_model()
    clf = CaptchaClassifier(weak)
    samples = [to_signal_sample(clf, h, l) for h, l in dataset.load_labeled_html()]
    from learning.model_evaluation import evaluate_signal_samples
    before = evaluate_signal_samples(weak, samples)["accuracy"]
    trained = train_weights(weak, samples)
    after = evaluate_signal_samples(trained, samples)["accuracy"]
    assert before < 0.95 and after > before and after >= 0.95


def test_bump_version():
    assert bump_version("1.0.0") == "1.0.1" and bump_version("2.3.9") == "2.3.10"


def _deploy_weak_baseline(registry, settings, db):
    path = settings.models_dir / "classifier_weak.json"
    path.write_text(json.dumps(_weakened_model()))
    db.register_model("1.0.0", str(path), None, None, None, "release")
    db.set_model_status("1.0.0", "APPROVED", "release")
    db.set_model_status("1.0.0", "DEPLOYED")


def test_candidate_requires_named_human_approval(dataset, db, settings):
    reg = ModelRegistry(db, settings)
    _deploy_weak_baseline(reg, settings, db)
    cand = reg.train_candidate(dataset.load_labeled_html(), created_by="pytest")
    assert cand["status"] == "CANDIDATE"
    assert cand["candidate"]["accuracy"] >= cand["baseline"]["accuracy"]
    # not active until approved + deployed
    assert CaptchaClassifier.load_active(db).version == "1.0.0"
    with pytest.raises(ApprovalError):
        reg.deploy(cand["version"])
    for bad in ("", "  ", "auto", "system"):
        with pytest.raises(ApprovalError):
            reg.approve(cand["version"], bad)
    reg.approve(cand["version"], "Jane Reviewer")
    reg.deploy(cand["version"])
    assert CaptchaClassifier.load_active(db).version == cand["version"]
    assert db.get_model("1.0.0")["status"] == "RETIRED"
    reg.rollback("1.0.0")
    assert CaptchaClassifier.load_active(db).version == "1.0.0"


def test_regression_refused(db, settings):
    reg = ModelRegistry(db, settings)
    reg.ensure_baseline()
    metrics = {"candidate": {"accuracy": 0.5}, "baseline": {"accuracy": 0.9}}
    db.register_model("1.0.9", "x.json", 0.5, 0.5, json.dumps(metrics), "t")
    with pytest.raises(ApprovalError):
        reg.approve("1.0.9", "Jane")
    reg.approve("1.0.9", "Jane", allow_regression=True)  # explicit human override


def test_reject_candidate(db, settings):
    reg = ModelRegistry(db, settings)
    db.register_model("3.0.0", "x.json", 0.9, 0.9, "{}", "t")
    assert reg.reject("3.0.0", "Jane")["status"] == "REJECTED"
    with pytest.raises(ApprovalError):
        reg.approve("3.0.0", "Jane")


def test_feedback_samples_used_for_training(framework, make_request, fixture_html, db, settings):
    r = framework.run(make_request(action="detect", html=fixture_html("hcaptcha_mock.html")))
    eid = db.list_executions(1)[0]["id"]
    db.record_feedback("hcaptcha", eid, reviewer="ops")
    samples = ModelRegistry(db, settings).feedback_samples()
    assert samples and samples[0][1] == "hcaptcha" and "hcaptcha" in samples[0][0]
    assert predict_from_signals(CaptchaClassifier().model, samples[0][0]) == r.captcha_type


def test_framework_picks_up_deployed_model(framework, make_request, fixture_html, db, settings, dataset):
    reg = ModelRegistry(db, settings)
    _deploy_weak_baseline(reg, settings, db)
    framework.reload_model()
    r = framework.run(make_request(action="detect", html=fixture_html("hcaptcha_mock.html")))
    assert r.captcha_type != "hcaptcha"  # weak model misses it
    cand = reg.train_candidate(dataset.load_labeled_html())
    reg.approve(cand["version"], "Jane")
    reg.deploy(cand["version"])
    framework.reload_model()
    r2 = framework.run(make_request(action="detect", html=fixture_html("hcaptcha_mock.html")))
    assert r2.captcha_type == "hcaptcha" and r2.model_version == cand["version"]


# ------------------------------------------------------------------ analytics
def _populate(framework, make_request, fixture_html):
    framework.run(make_request(mode="test", html=fixture_html("recaptcha_mock.html")))
    framework.run(make_request(mode="test", html=fixture_html("audio_mock.html"), test_response="fail"))
    framework.run(make_request(html=fixture_html("no_captcha.html")))
    r = framework.run(make_request(html=fixture_html("hcaptcha_mock.html")))
    framework.hv.complete(r.request_id, "Op")
    for _ in range(3):  # recurring misclassification: a text CAPTCHA the model missed
        framework.run(make_request(action="detect", html="<input name='code'>"))
        framework.db.record_feedback("text", framework.db.list_executions(1)[0]["id"], reviewer="ops")


def test_analytics(framework, make_request, fixture_html):
    _populate(framework, make_request, fixture_html)
    a = CaptchaAnalytics(framework.db)
    s = a.summary()
    assert s["total_executions"] == 7
    assert s["by_status"]["TEST_VERIFIED"] == 1 and s["by_type"]["recaptcha"] == 1
    acc = a.detection_accuracy()
    assert acc["labelled_samples"] == 3 and acc["accuracy"] == 0.0
    v = a.verification_stats()
    assert v["human_by_status"]["COMPLETED"] == 1 and v["human_completion_rate"] == 1.0
    rf = a.recurring_failures(min_count=3)
    assert rf["misclassification_patterns"][0]["actual"] == "text"
    assert any("missed" in r for r in rf["recommendations"])
    assert a.daily_trend()[0]["total"] == 7


def test_report_files(framework, make_request, fixture_html, settings, dataset):
    _populate(framework, make_request, fixture_html)
    ev = evaluate_html(CaptchaClassifier(), dataset.load_labeled_html())
    paths = write_report(framework.db, settings.log_dir / "reports", ev)
    data = json.loads(open(paths["json"], encoding="utf-8").read())
    assert data["summary"]["total_executions"] == 7 and data["evaluation"]["accuracy"] >= 0.95
    html = open(paths["html"], encoding="utf-8").read()
    assert "Performance Report" in html and "recaptcha" in html


def test_report_escapes_html(db):
    db.record_execution(task_id="x", action="detect", mode="test", status="NO_CAPTCHA", platform="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in render_html(build_report(db))


def test_empty_report(db):
    assert "No data" in render_html(build_report(db)) or build_report(db)["summary"]["total_executions"] == 0
