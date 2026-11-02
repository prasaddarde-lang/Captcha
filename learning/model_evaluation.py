"""Model evaluation, candidate training and the human-approval deployment workflow.

Lifecycle:   train_candidate -> CANDIDATE -> approve (named human) -> APPROVED -> deploy -> DEPLOYED
A candidate can never become active without an explicit approval by a named reviewer,
and approval is refused when the candidate is worse than the deployed model.

Training data is limited to (a) synthetic fixtures and (b) operator feedback about the
*type* of challenge seen, expressed as which metadata signals fired. No third-party
challenge content (images, audio, answers) is ever used.
"""
from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from config import Settings, get_settings
from core.classifier import CaptchaClassifier, load_model_file, validate_model
from core.detector import extract_features
from database.db_manager import DatabaseManager

log = logging.getLogger("captcha_fw.learning")

NONE = "none"
# A sample is represented by the set of signal ids that fired, grouped by type.
SignalSample = tuple[dict[str, list[str]], str]


class ApprovalError(RuntimeError):
    pass


def to_signal_sample(clf: CaptchaClassifier, html: str, label: str) -> SignalSample:
    _, signals = clf.match(extract_features(html))
    fired: dict[str, list[str]] = defaultdict(list)
    for s in signals:
        fired[s["captcha_type"]].append(s["signal"])
    return dict(fired), label


def predict_from_signals(model: dict, fired: dict[str, list[str]]) -> str:
    weights = {(t, r["id"]): float(r["weight"]) for t, rules in model["types"].items() for r in rules}
    scores = {t: sum(weights.get((t, sid), 0.0) for sid in ids) for t, ids in fired.items()}
    thr = float(model["threshold"])
    providers = set(model.get("provider_types", []))
    pool = {t: s for t, s in scores.items() if t in providers and s >= thr} or \
           {t: s for t, s in scores.items() if t not in providers and s >= thr}
    return max(pool.items(), key=lambda kv: kv[1])[0] if pool else NONE


def evaluate_signal_samples(model: dict, samples: Iterable[SignalSample]) -> dict:
    samples = list(samples)
    labels = sorted({lbl for _, lbl in samples} | set(model["types"]) | {NONE})
    confusion = {a: {p: 0 for p in labels} for a in labels}
    misclassified = []
    correct = 0
    for i, (fired, actual) in enumerate(samples):
        pred = predict_from_signals(model, fired)
        confusion.setdefault(actual, {p: 0 for p in labels})
        confusion[actual][pred] = confusion[actual].get(pred, 0) + 1
        if pred == actual:
            correct += 1
        else:
            misclassified.append({"index": i, "actual": actual, "predicted": pred})
    per_class = {}
    f1s = []
    present = sorted({lbl for _, lbl in samples})
    for c in present:
        tp = confusion[c][c]
        fp = sum(confusion[a].get(c, 0) for a in confusion if a != c)
        fn = sum(v for p, v in confusion[c].items() if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    # trim confusion to rows/cols that are non-empty for readability
    used = sorted({a for a in confusion if any(confusion[a].values())} |
                  {p for a in confusion for p, v in confusion[a].items() if v})
    return {
        "model_version": model["version"],
        "samples": len(samples),
        "accuracy": round(correct / len(samples), 4) if samples else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_class": per_class,
        "confusion": {a: {p: confusion[a].get(p, 0) for p in used} for a in used},
        "misclassified": misclassified[:50],
    }


def evaluate_html(classifier: CaptchaClassifier, labeled_html: Iterable[tuple[str, str]]) -> dict:
    samples = [to_signal_sample(classifier, html, label) for html, label in labeled_html]
    return evaluate_signal_samples(classifier.model, samples)


def bump_version(version: str) -> str:
    parts = [int(p) if p.isdigit() else 0 for p in version.split(".")] + [0, 0, 0]
    major, minor, patch = parts[:3]
    return f"{major}.{minor}.{patch + 1}"


def train_weights(model: dict, samples: list[SignalSample], epochs: int = 10, lr: float = 0.25,
                  min_weight: float = 0.1, max_weight: float = 5.0) -> dict:
    """Perceptron-style weight adjustment over signal samples.

    For each misclassified sample, signals of the true type are reinforced and
    signals of the wrongly predicted type are weakened. Rules are never added
    automatically - new signatures are a human code-review decision.
    """
    new = copy.deepcopy(model)
    index = {(t, r["id"]): r for t, rules in new["types"].items() for r in rules}
    for _ in range(epochs):
        errors = 0
        for fired, actual in samples:
            pred = predict_from_signals(new, fired)
            if pred == actual:
                continue
            errors += 1
            for sid in fired.get(actual, []):
                r = index.get((actual, sid))
                if r:
                    r["weight"] = round(min(max_weight, float(r["weight"]) + lr), 3)
            if pred != NONE:
                for sid in fired.get(pred, []):
                    r = index.get((pred, sid))
                    if r:
                        r["weight"] = round(max(min_weight, float(r["weight"]) - lr), 3)
        if errors == 0:
            break
    return validate_model(new)


class ModelRegistry:
    def __init__(self, db: DatabaseManager, settings: Settings | None = None):
        self.db = db
        self.settings = settings or get_settings()

    # ----------------------------------------------------------- bootstrap
    def ensure_baseline(self) -> dict:
        """Register the shipped baseline as the deployed model when the registry is empty."""
        if not self.db.list_models():
            clf = CaptchaClassifier()
            path = self.settings.models_dir / f"classifier_v{clf.version}.json"
            if not path.exists():
                path.write_text(json.dumps(clf.model, indent=2), encoding="utf-8")
            self.db.register_model(clf.version, str(path), None, None, None, "release",
                                   "baseline model shipped with the framework")
            self.db.set_model_status(clf.version, "APPROVED", "release")
            self.db.set_model_status(clf.version, "DEPLOYED")
        return self.db.deployed_model()  # type: ignore[return-value]

    def active_model(self) -> dict:
        dep = self.ensure_baseline()
        return load_model_file(dep["file_path"])

    # ------------------------------------------------------------ samples
    def feedback_samples(self) -> list[SignalSample]:
        """Operator feedback -> signal samples (signals recorded at detection time)."""
        rows = self.db.query(
            "SELECT f.execution_id, f.actual_type FROM detection_feedback f WHERE f.execution_id IS NOT NULL")
        samples = []
        for r in rows:
            sigs = self.db.query("SELECT captcha_type, signal FROM detection_signals WHERE execution_id = ?",
                                 (r["execution_id"],))
            fired: dict[str, list[str]] = defaultdict(list)
            for s in sigs:
                fired[s["captcha_type"]].append(s["signal"])
            samples.append((dict(fired), r["actual_type"]))
        return samples

    # ----------------------------------------------------------- training
    def train_candidate(self, labeled_html: list[tuple[str, str]], created_by: str = "learning",
                        include_feedback: bool = True, test_ratio: float = 0.3) -> dict:
        from learning.dataset_manager import SyntheticDatasetManager

        base = self.active_model()
        clf = CaptchaClassifier(base)
        train, test = SyntheticDatasetManager.split(labeled_html, test_ratio=test_ratio)
        train_s = [to_signal_sample(clf, h, l) for h, l in train]
        test_s = [to_signal_sample(clf, h, l) for h, l in test]
        if include_feedback:
            fb = self.feedback_samples()
            train_s += fb[: int(len(fb) * (1 - test_ratio))]
            test_s += fb[int(len(fb) * (1 - test_ratio)):]

        candidate = train_weights(base, train_s)
        existing = {m["version"] for m in self.db.list_models()}
        version = bump_version(base["version"])
        while version in existing:
            version = bump_version(version)
        candidate["version"] = version
        candidate["description"] = f"Candidate trained from {len(train_s)} samples (base {base['version']})"

        base_metrics = evaluate_signal_samples(base, test_s)
        cand_metrics = evaluate_signal_samples(candidate, test_s)
        path = self.settings.models_dir / f"classifier_v{version}.json"
        path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        metrics = {"candidate": cand_metrics, "baseline": base_metrics,
                   "train_samples": len(train_s), "test_samples": len(test_s)}
        self.db.register_model(version, str(path), cand_metrics["accuracy"], cand_metrics["macro_f1"],
                               json.dumps(metrics), created_by,
                               f"base={base['version']} base_acc={base_metrics['accuracy']}")
        log.info("candidate model %s registered (acc=%.3f vs base %.3f) - awaiting human approval",
                 version, cand_metrics["accuracy"], base_metrics["accuracy"])
        return {"version": version, "status": "CANDIDATE", "file": str(path), **metrics}

    def evaluate_version(self, version: str, labeled_html: list[tuple[str, str]]) -> dict:
        m = self.db.get_model(version)
        if not m:
            raise ApprovalError(f"Unknown model version {version}")
        return evaluate_html(CaptchaClassifier(load_model_file(m["file_path"])), labeled_html)

    # ------------------------------------------------------- human approval
    def approve(self, version: str, approver: str, allow_regression: bool = False) -> dict:
        if not approver or not approver.strip() or approver.strip().lower() in {"auto", "system", "learning"}:
            raise ApprovalError("Approval requires a named human reviewer")
        m = self.db.get_model(version)
        if not m:
            raise ApprovalError(f"Unknown model version {version}")
        if m["status"] != "CANDIDATE":
            raise ApprovalError(f"Model {version} is {m['status']}, only CANDIDATE models can be approved")
        metrics = json.loads(m["metrics_json"] or "{}")
        cand = metrics.get("candidate", {}).get("accuracy", m["accuracy"] or 0)
        base = metrics.get("baseline", {}).get("accuracy", 0)
        if cand < base and not allow_regression:
            raise ApprovalError(f"Candidate accuracy {cand} is below baseline {base}; refusing approval")
        self.db.set_model_status(version, "APPROVED", approver.strip())
        log.info("model %s approved by %s", version, approver)
        return self.db.get_model(version)  # type: ignore[return-value]

    def reject(self, version: str, reviewer: str) -> dict:
        m = self.db.get_model(version)
        if not m or m["status"] not in ("CANDIDATE", "APPROVED"):
            raise ApprovalError(f"Model {version} cannot be rejected")
        self.db.set_model_status(version, "REJECTED")
        log.info("model %s rejected by %s", version, reviewer)
        return self.db.get_model(version)  # type: ignore[return-value]

    def deploy(self, version: str) -> dict:
        m = self.db.get_model(version)
        if not m:
            raise ApprovalError(f"Unknown model version {version}")
        if m["status"] not in ("APPROVED", "RETIRED") or not m["approved_by"]:
            raise ApprovalError(f"Model {version} must be human-approved before deployment (status {m['status']})")
        if not Path(m["file_path"]).exists():
            raise ApprovalError(f"Model file missing: {m['file_path']}")
        load_model_file(m["file_path"])  # validate before activating
        self.db.set_model_status(version, "DEPLOYED")
        log.info("model %s deployed (approved by %s)", version, m["approved_by"])
        return self.db.get_model(version)  # type: ignore[return-value]

    def rollback(self, version: str) -> dict:
        """Re-deploy a previously approved (now retired) version."""
        return self.deploy(version)
