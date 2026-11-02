"""Image-model pipeline: dataset generation/validation, training, checkpoints, evaluation,
registry (human approval, promotion, rollback), continuous trainer, framework/API/RPA integration."""
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tests.conftest import ROOT
from training.common import load_config

ENV_KEYS = ("CAPTCHA_FW_TRAINING_CONFIG", "CAPTCHA_FW_DATASETS_DIR", "CAPTCHA_FW_MODELS_DIR", "CAPTCHA_FW_REPORTS_DIR")


def _small_config() -> dict:
    cfg = copy.deepcopy(load_config(ROOT / "training" / "training_config.yaml"))
    cfg["dataset"]["total_samples"] = 400
    cfg["training"].update(epochs=6, early_stopping_patience=3, hidden_units=64, batch_size=64)
    cfg["augmentation"].update(per_iteration=12, max_total_extra=24)
    cfg["continuous"].update(max_iterations=2, max_run_minutes=10, patience=3)
    cfg["promotion"].update(initial_min_accuracy=0.5, initial_max_false_positive_rate=0.5)
    cfg["resources"].update(min_free_disk_mb=50)
    cfg["inference"]["min_confidence"] = 0.5
    return cfg


def _env_for(root: Path) -> dict:
    return {"CAPTCHA_FW_TRAINING_CONFIG": str(root / "config.yaml"), "CAPTCHA_FW_DATASETS_DIR": str(root / "datasets"),
            "CAPTCHA_FW_MODELS_DIR": str(root / "models"), "CAPTCHA_FW_REPORTS_DIR": str(root / "reports")}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Generate a 400-sample dataset and train once for the whole module."""
    from training.generate_dataset import generate
    from training.train_classifier import run_training

    root = tmp_path_factory.mktemp("imgtrain")
    cfg = _small_config()
    (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    os.environ.update(_env_for(root))
    try:
        generate(400, cfg, progress=False)
        summary = run_training(cfg, progress=False)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return SimpleNamespace(root=root, cfg=cfg, summary=summary)


@pytest.fixture
def tenv(built, tmp_path, monkeypatch, settings):
    """Isolated copy of the built dataset/models/reports, wired into framework settings."""
    from config import get_settings, reset_settings
    from training.common import get_paths

    dst = tmp_path / "t"
    shutil.copytree(built.root, dst)
    for k, v in _env_for(dst).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CAPTCHA_FW_IMAGE_MIN_CONFIDENCE", "0.5")  # the tiny test model is less confident
    reset_settings()
    cfg = load_config()
    return SimpleNamespace(root=dst, cfg=cfg, paths=get_paths(cfg), version=built.summary["version"],
                           settings=get_settings())


def _registry(tenv):
    from training.model_registry import ImageModelRegistry
    return ImageModelRegistry(tenv.cfg, tenv.paths)


def _promote(tenv):
    reg = _registry(tenv)
    reg.approve(tenv.version, "Test Reviewer")
    reg.promote(tenv.version)
    return reg


def _framework(tenv):
    from core.framework import CaptchaFramework
    from database.db_manager import DatabaseManager
    return CaptchaFramework(settings=tenv.settings, db=DatabaseManager(tenv.settings.db_path))


def _pick(tenv, category, min_conf=0.5):
    """A test-split image the model classifies correctly (the tiny test model is not perfect)."""
    from core.image_classifier import ImageCaptchaClassifier
    clf = ImageCaptchaClassifier(tenv.paths.models, 0.5)
    manifest = json.loads(tenv.paths.manifest.read_text(encoding="utf-8"))
    for s in manifest["samples"]:
        if s["split"] in ("test", "validation") and s["category"] == category:
            p = tenv.paths.datasets / s["file"]
            pred = clf.classify_file(p)
            if pred.category == category and pred.confidence >= min_conf:
                return p
    pytest.skip(f"tiny model classifies no {category} sample confidently")


# ------------------------------------------------------------------------- dataset
def test_dataset_counts_split_and_manifest(tenv):
    m = json.loads(tenv.paths.manifest.read_text(encoding="utf-8"))
    assert m["base_count"] == 400 and len(m["samples"]) == 400
    by = {}
    for s in m["samples"]:
        by.setdefault(s["split"], []).append(s)
        assert s["license"] and s["authorization_ref"] and s["source"].startswith("synthetic")
        assert s["content_label"] is None and s["label"] == s["category"]
        assert "text" not in s["generator"]["params"]  # rendered characters are never recorded
        assert (tenv.paths.datasets / s["file"]).exists() and s["file"].count("/") == 1
    assert {k: len(v) for k, v in by.items()} == {"train": 320, "validation": 40, "test": 40}
    assert len({s["sha256"] for s in m["samples"]}) == 400 and len({s["dhash"] for s in m["samples"]}) == 400


def test_generation_is_reproducible():
    from training.generate_dataset import RENDERERS, render_sample
    for i, cat in enumerate(RENDERERS):
        a = render_sample(cat, [1, i, 5, 0], 88)
        b = render_sample(cat, [1, i, 5, 0], 88)
        assert a[0] == b[0]


def test_generate_skips_when_present(tenv, capsys):
    from training.generate_dataset import generate
    generate(400, tenv.cfg)
    assert "already present" in capsys.readouterr().out


def test_validator_passes_then_catches_problems(tenv):
    from training.dataset_validator import validate
    assert validate(tenv.cfg)["ok"]
    m = json.loads(tenv.paths.manifest.read_text(encoding="utf-8"))
    s0, s1, s2 = m["samples"][:3]
    (tenv.paths.datasets / s0["file"]).write_bytes(b"not an image")
    (tenv.paths.datasets / s1["file"]).unlink()
    s2["content_label"] = "ABC12"
    m["samples"].append(dict(m["samples"][5]))
    tenv.paths.manifest.write_text(json.dumps(m), encoding="utf-8")
    r = validate(tenv.cfg)
    text = " ".join(r["errors"])
    assert not r["ok"]
    assert "sha256 mismatch" in text and "file missing" in text and "content_label" in text and "duplicate" in text


def test_features_are_cached(tenv, monkeypatch):
    from training.dataset_manager import TrainingDataset
    ds = TrainingDataset(tenv.cfg, tenv.paths)
    X, y, ids = ds.features("train")
    assert X.shape[0] == 320 and X.dtype == np.float32 and len(set(y.tolist())) == 10
    monkeypatch.setattr(ds.extractor.__class__, "__call__", lambda *a: (_ for _ in ()).throw(AssertionError("recomputed")))
    X2, _, _ = ds.features("train")
    assert np.allclose(X, X2)


# ------------------------------------------------------------------------ training
def test_architecture_comparison_and_candidate(tenv, built):
    s = built.summary
    assert set(s["comparison"]) == {"nearest_centroid", "softmax", "mlp"}
    m = _registry(tenv).get(tenv.version)
    assert m["status"] == "CANDIDATE" and m["recommended"]
    assert m["architecture"] == max(s["comparison"], key=lambda a: (s["comparison"][a]["val_macro_f1"], s["comparison"][a]["val_accuracy"]))
    t = m["test_metrics"]
    assert set(t["per_category"]) == set(tenv.cfg["dataset"]["categories"])
    assert t["ocr_metrics"]["character_accuracy"] == "not_applicable"
    for k in ("false_positive_rate", "false_negative_rate", "latency_ms", "resources", "at_runtime_threshold"):
        assert k in t
    run_dir = tenv.paths.checkpoints / tenv.version
    assert (run_dir / "model.npz").exists() and (run_dir / "train_log.jsonl").exists()
    assert not list(run_dir.glob("*_last.npz"))  # resumable state removed after completion
    assert m["device"]["used"] == "cpu"


def test_checkpoint_resume_after_time_limit(tenv):
    from training.train_classifier import TrainingStopped, run_training
    cfg = copy.deepcopy(tenv.cfg)
    cfg["resources"]["max_training_minutes"] = 0
    with pytest.raises(TrainingStopped):
        run_training(cfg, ["mlp"], "img-resume", progress=False)
    ck = tenv.paths.checkpoints / "img-resume" / "mlp_last.npz"
    assert ck.exists()
    cfg["resources"]["max_training_minutes"] = 10
    run_training(cfg, ["mlp"], "img-resume", progress=False)
    epochs = [json.loads(line)["epoch"] for line in (tenv.paths.checkpoints / "img-resume" / "train_log.jsonl").read_text().splitlines()]
    assert epochs[0] == 1 and epochs == list(range(1, len(epochs) + 1))  # continued, not restarted


def test_training_is_deterministic(tenv):
    from training.train_classifier import run_training
    from core.image_classifier import NumpyImageModel
    run_training(tenv.cfg, ["softmax"], "img-det-a", progress=False)
    run_training(tenv.cfg, ["softmax"], "img-det-b", progress=False)
    a = NumpyImageModel.load(tenv.paths.checkpoints / "img-det-a" / "model.npz")
    b = NumpyImageModel.load(tenv.paths.checkpoints / "img-det-b" / "model.npz")
    assert np.allclose(a.params["W"], b.params["W"], atol=1e-5)


def test_already_trained_guard(tenv):
    from training.dataset_manager import TrainingDataset
    from training.common import config_hash
    from training.train_classifier import already_trained
    ds = TrainingDataset(tenv.cfg, tenv.paths)
    archs = list(tenv.cfg["training"]["architectures"])
    assert already_trained(_registry(tenv), ds.dataset_hash(), config_hash(tenv.cfg), archs) == tenv.version


# ------------------------------------------------------------------------ registry
@pytest.mark.parametrize("name", ["", " ", "auto", "Claude", "system", "trainer"])
def test_approval_requires_named_human(tenv, name):
    from training.model_registry import ApprovalError
    with pytest.raises(ApprovalError):
        _registry(tenv).approve(tenv.version, name)


def test_promotion_requires_approval(tenv):
    from training.model_registry import ApprovalError
    with pytest.raises(ApprovalError):
        _registry(tenv).promote(tenv.version)


def test_approve_promote_and_archive(tenv):
    reg = _promote(tenv)
    assert reg.production_version() == tenv.version
    assert (tenv.paths.production / "image_classifier.npz").exists()
    assert (tenv.paths.archive / f"{tenv.version}.npz").exists()
    ok, reasons = reg.verify_production()
    assert ok, reasons


def _register_variant(tenv, version, acc_delta, fp_delta=0.0):
    reg = _registry(tenv)
    base = reg.get(tenv.version)
    src = tenv.paths.checkpoints / version / "model.npz"
    src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tenv.paths.checkpoints / tenv.version / "model.npz", src)
    t = copy.deepcopy(base["test_metrics"])
    t["accuracy"] = round(t["accuracy"] + acc_delta, 4)
    t["false_positive_rate"] = round(t["false_positive_rate"] + fp_delta, 4)
    return reg.register_candidate(version, src, {"architecture": base["architecture"], "test_metrics": t,
                                                 "validation_metrics": base["validation_metrics"], "recommended": True})


def test_regression_blocked_unless_overridden(tenv):
    from training.model_registry import ApprovalError
    _promote(tenv)
    _register_variant(tenv, "img-worse", -0.05)
    with pytest.raises(ApprovalError, match="criteria"):
        _registry(tenv).approve("img-worse", "Test Reviewer")
    _register_variant(tenv, "img-fp", 0.01, fp_delta=0.05)
    with pytest.raises(ApprovalError, match="false_positive_rate"):
        _registry(tenv).approve("img-fp", "Test Reviewer")
    assert _registry(tenv).approve("img-worse", "Test Reviewer", allow_regression=True)["status"] == "APPROVED"


def test_manual_rollback(tenv):
    reg = _promote(tenv)
    _register_variant(tenv, "img-next", 0.0)
    m = reg.get("img-next")
    m_data = json.loads(reg.file.read_text())
    m_data["models"]["img-next"]["status"] = "APPROVED"
    m_data["models"]["img-next"]["approved_by"] = "Test Reviewer"
    reg.file.write_text(json.dumps(m_data))
    reg.promote("img-next")
    assert reg.production_version() == "img-next" and reg.get(tenv.version)["status"] == "ARCHIVED"
    reg.rollback()
    assert reg.production_version() == tenv.version and reg.get("img-next")["status"] == "ARCHIVED"


def test_automatic_rollback_when_post_promotion_check_fails(tenv):
    from training.model_registry import ApprovalError
    reg = _promote(tenv)
    _register_variant(tenv, "img-inflated", 0.3)  # recorded metrics the real model cannot reproduce
    reg.approve("img-inflated", "Test Reviewer")
    with pytest.raises(ApprovalError, match="rolled back"):
        reg.promote("img-inflated")
    assert reg.production_version() == tenv.version
    assert reg.get("img-inflated")["status"] == "ROLLED_BACK"
    assert reg.verify_production()[0]


# -------------------------------------------------------------- continuous trainer
def test_continuous_run_targets_weakest_and_keeps_test_fixed(tenv):
    from training.continuous_trainer import ContinuousTrainer
    from training.dataset_manager import TrainingDataset
    before = TrainingDataset(tenv.cfg, tenv.paths).test_set_hash()
    ct = ContinuousTrainer(tenv.cfg, progress=False)
    weakest, _ = ct.weakest_category(tenv.version)
    results = ct.run(max_iterations=1)
    st = ct.state()
    assert len(results) == 1 and results[0]["weakest_category"] == weakest and results[0]["samples_added"] == 12
    assert st["extra_per_category"] == {weakest: 12} and st["in_progress"] is None
    assert TrainingDataset(tenv.cfg, tenv.paths).test_set_hash() == before
    assert not ct.lock_file.exists()
    assert (tenv.paths.history / "history.jsonl").read_text().count("\n") == 1
    assert _registry(tenv).production_version() is None  # never promoted automatically


def test_continuous_budget_and_patience_stop(tenv):
    from training.continuous_trainer import ContinuousTrainer
    ct = ContinuousTrainer(tenv.cfg, progress=False)
    st = ct.state()
    st["no_improve_streak"] = 3
    ct._save(st)
    assert ct.run() == [] and "patience" in ct.state()["stopped_reason"]
    st = ct.state()
    st.update(no_improve_streak=0, total_extra=24)
    ct._save(st)
    r = ct.run()
    assert r[0]["skipped"] and "budget" in ct.state()["stopped_reason"]


def test_continuous_lock(tenv):
    from training.common import write_json
    from training.continuous_trainer import ContinuousTrainer, LockError
    ct = ContinuousTrainer(tenv.cfg, progress=False)
    write_json(ct.lock_file, {"pid": os.getppid()})  # a live process that is not us
    with pytest.raises(LockError):
        ct.run(max_iterations=1)
    write_json(ct.lock_file, {"pid": 999999})  # stale lock is taken over
    ct.acquire()
    ct.release()


def test_continuous_resumes_interrupted_iteration(tenv):
    from training.continuous_trainer import ContinuousTrainer
    ct = ContinuousTrainer(tenv.cfg, progress=False)
    st = ct.state()
    st["in_progress"] = {"iteration": 0, "stage": "training", "version": "img-interrupted", "archs": ["softmax"],
                         "weakest": "math", "weakest_f1": 0.5, "added": 0, "reference": tenv.version, "note": "resume test"}
    ct._save(st)
    r = ct.iterate()
    assert r["version"] == "img-interrupted" and ct.state()["in_progress"] is None


def test_training_report(tenv):
    from training.training_report import write_training_report
    out = write_training_report(tenv.cfg, tenv.paths)
    md = Path(out["markdown"]).read_text(encoding="utf-8")
    assert "Architecture comparison" in md and "not applicable" in md and "Action needed" in md
    assert json.loads(Path(out["json"]).read_text(encoding="utf-8"))["reference"] == tenv.version


# ------------------------------------------------------------- framework integration
def test_image_only_requires_production_model(tenv, make_request):
    from core.response_handler import Status
    r = _framework(tenv).run(make_request(image_file=str(_any_image(tenv))))
    assert r.status == Status.ERROR and "No production image model" in r.message


def _any_image(tenv):
    return next((tenv.paths.datasets / "test").glob("grid_*.jpg"))


def test_image_only_detection_test_and_production_modes(tenv, make_request):
    from core.response_handler import Status
    _promote(tenv)
    fw = _framework(tenv)
    img = _pick(tenv, "grid")
    r = fw.run(make_request(image_file=str(img), mode="test"))
    assert r.status == Status.TEST_VERIFIED  # our own synthetic image -> test mode allowed
    assert r.detection_source == "image_model" and r.image_category == "grid" and r.captcha_type == "image_selection"
    assert r.image_model_version == tenv.version and r.captcha_subtype == "tile_grid"
    r2 = fw.run(make_request(image_file=str(img)))
    assert r2.status == Status.HUMAN_VERIFICATION_REQUIRED and r2.request_id
    assert fw.db.list_executions(1)[0]["image_model_version"] == tenv.version


def test_image_none_category_and_threshold(tenv, make_request):
    from core.response_handler import Status
    _promote(tenv)
    fw = _framework(tenv)
    r = fw.run(make_request(image_file=str(_pick(tenv, "none"))))
    assert r.status == Status.NO_CAPTCHA and r.image_category == "none"
    fw.image_classifier.min_confidence = 1.01  # nothing is confident enough
    r2 = fw.run(make_request(image_file=str(_pick(tenv, "math"))))
    assert r2.status == Status.NO_CAPTCHA and "below the" in r2.message


def test_metadata_stays_authoritative_with_image(tenv, make_request, fixture_html):
    _promote(tenv)
    r = _framework(tenv).run(make_request(action="detect", html=fixture_html("recaptcha_mock.html"),
                                          image_file=str(_pick(tenv, "grid"))))
    assert r.captcha_type == "recaptcha" and r.detection_source == "metadata+image_model" and r.image_category == "grid"


def test_image_input_validation(tenv, make_request, tmp_path):
    from core.response_handler import Status
    _promote(tenv)
    fw = _framework(tenv)
    bad = tenv.paths.datasets / "not_an_image.jpg"
    bad.write_text("hello")
    assert fw.run(make_request(image_file=str(bad))).status == Status.INVALID_INPUT
    assert fw.run(make_request(image_file=str(tenv.paths.datasets / "missing.jpg"))).status == Status.INVALID_INPUT
    outside = Path(os.environ.get("SystemRoot", "C:/Windows")) / "win.ini"
    assert fw.run(make_request(image_file=str(outside))).status == Status.INVALID_INPUT


def test_html_flow_unchanged_without_image(tenv, make_request, fixture_html):
    r = _framework(tenv).run(make_request(html=fixture_html("hcaptcha_mock.html"), action="detect"))
    assert r.detection_source == "metadata" and r.image_model_version is None and r.captcha_type == "hcaptcha"


def test_rpa_adapters_carry_image_fields(tenv):
    from integrations import generic_rpa
    from integrations.blue_prism import BluePrismAdapter
    from integrations.uipath import handle
    _promote(tenv)
    generic_rpa.set_framework(_framework(tenv))
    try:
        task = json.dumps({"task_id": "IMG_1", "mode": "test", "image_file": str(_pick(tenv, "audio"))})
        d = json.loads(handle(task))
        assert d["status"] == "TEST_VERIFIED" and d["image_category"] == "audio" and d["image_model_version"]
        kv = BluePrismAdapter().handle_json(task)
        assert "image_category=audio" in kv and "detection_source=image_model" in kv
    finally:
        generic_rpa.set_framework(None)


def test_api_image_model_endpoints(tenv):
    from fastapi.testclient import TestClient
    from api.server import create_app
    client = TestClient(create_app(_framework(tenv)))
    assert client.get("/health").json()["image_model_version"] is None
    assert client.post(f"/v1/models/image/{tenv.version}/approve", json={"approver": "auto"}).status_code == 409
    r = client.post(f"/v1/models/image/{tenv.version}/approve", json={"approver": "Test Reviewer"})
    assert r.status_code == 200 and r.json()["status"] == "PRODUCTION"
    assert client.get("/health").json()["image_model_version"] == tenv.version
    reg = client.get("/v1/models/image").json()
    assert reg["production"] == tenv.version and reg["loaded"] == tenv.version


# ---------------------------------------------------------------------- BAT scripts
def _bat(tenv, *args):
    env = {**os.environ, "CAPTCHA_FW_PYTHON": sys.executable}
    return subprocess.run(["cmd.exe", "/c", str(ROOT / "scripts" / args[0]), *args[1:]], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")


@pytest.mark.skipif(os.name != "nt", reason="Windows BAT scripts")
def test_bat_approve_evaluate_status(tenv):
    assert _bat(tenv, "approve_model.bat", tenv.version).returncode == 40          # usage
    assert _bat(tenv, "approve_model.bat", tenv.version, "auto").returncode == 41  # not a human
    p = _bat(tenv, "approve_model.bat", tenv.version, "Test Reviewer")
    assert p.returncode == 0 and "PRODUCTION" in p.stdout, p.stdout + p.stderr
    e = _bat(tenv, "evaluate_model.bat")
    assert e.returncode == 0 and "accuracy" in e.stdout, e.stdout + e.stderr
    s = _bat(tenv, "continuous_training.bat", "status")
    assert s.returncode == 0 and json.loads(s.stdout)["production"] == tenv.version
    assert _bat(tenv, "rollback_model.bat").returncode == 41  # nothing earlier to roll back to
