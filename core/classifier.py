"""Signature-based CAPTCHA classifier.

The "model" is a versioned JSON file of weighted signals (regexes over page
*metadata*: iframe/script URLs, class names, ids, field names, prompt text).
Models are produced by ``learning.model_evaluation`` and only become active
after a human approves and deploys them.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings

DEFAULT_MODEL_FILE = "classifier_v1.0.0.json"
FEATURE_KINDS = ("iframe_src", "script_src", "class", "id", "attr", "attr_name",
                 "input_name", "img", "text", "tag")


@dataclass
class PageFeatures:
    """Metadata extracted from a page; contains no challenge images, audio or answers."""

    iframe_src: list[str] = field(default_factory=list)
    script_src: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    attrs: list[str] = field(default_factory=list)        # "name=value" for data-* attributes
    attr_names: list[str] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    imgs: list[str] = field(default_factory=list)          # src/alt/id/class of <img>
    text: str = ""                                         # lower-cased visible text (truncated)
    tags: list[str] = field(default_factory=list)
    response_filled: dict[str, bool] = field(default_factory=dict)  # provider -> response field non-empty
    mock_marker: str | None = None                         # data-mock-captcha value (synthetic fixtures)

    def values(self, kind: str) -> list[str]:
        return {
            "iframe_src": self.iframe_src, "script_src": self.script_src, "class": self.classes,
            "id": self.ids, "attr": self.attrs, "attr_name": self.attr_names,
            "input_name": self.input_names, "img": self.imgs, "tag": self.tags,
            "text": [self.text] if self.text else [],
        }[kind]

    def merge_frames(self, frame_urls: list[str]) -> None:
        for u in frame_urls:
            if u and u not in self.iframe_src:
                self.iframe_src.append(u)


@dataclass
class Classification:
    captcha_detected: bool
    captcha_type: str | None
    subtype: str | None
    confidence: float
    scores: dict[str, float]
    signals: list[dict[str, Any]]
    model_version: str


class ModelError(ValueError):
    pass


def validate_model(model: dict) -> dict:
    for key in ("version", "threshold", "types"):
        if key not in model:
            raise ModelError(f"model missing '{key}'")
    for ctype, rules in model["types"].items():
        for r in rules:
            if r.get("kind") not in FEATURE_KINDS:
                raise ModelError(f"{ctype}/{r.get('id')}: unknown kind {r.get('kind')}")
            try:
                re.compile(r["pattern"])
            except (re.error, KeyError) as exc:
                raise ModelError(f"{ctype}/{r.get('id')}: bad pattern ({exc})") from exc
            float(r["weight"])
    return model


def load_model_file(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return validate_model(json.load(fh))


def default_model_path() -> Path:
    return get_settings().models_dir / DEFAULT_MODEL_FILE if (
        get_settings().models_dir / DEFAULT_MODEL_FILE).exists() else Path(__file__).resolve().parent.parent / "models" / DEFAULT_MODEL_FILE


class CaptchaClassifier:
    def __init__(self, model: dict | None = None):
        self.model = validate_model(copy.deepcopy(model)) if model else load_model_file(default_model_path())
        self.version: str = self.model["version"]
        self.threshold: float = float(self.model["threshold"])
        self.scale: float = float(self.model.get("confidence_scale", 2.0))
        self.provider_types: set[str] = set(self.model.get("provider_types", []))
        self._compiled = {
            ctype: [(r, re.compile(r["pattern"], re.IGNORECASE)) for r in rules]
            for ctype, rules in self.model["types"].items()
        }

    @classmethod
    def load_active(cls, db=None) -> "CaptchaClassifier":
        """Load the human-approved, deployed model if one exists; otherwise the baseline."""
        if db is not None:
            try:
                deployed = db.deployed_model()
                if deployed and Path(deployed["file_path"]).exists():
                    return cls(load_model_file(deployed["file_path"]))
            except Exception:  # fall back to baseline on any registry problem
                pass
        return cls()

    @property
    def types(self) -> list[str]:
        return list(self._compiled)

    def match(self, features: PageFeatures) -> tuple[dict[str, float], list[dict[str, Any]]]:
        scores: dict[str, float] = {}
        signals: list[dict[str, Any]] = []
        for ctype, rules in self._compiled.items():
            total = 0.0
            for rule, rx in rules:
                if any(rx.search(v) for v in features.values(rule["kind"])):
                    w = float(rule["weight"])
                    total += w
                    signals.append({"captcha_type": ctype, "signal": rule["id"], "weight": w})
            scores[ctype] = round(total, 3)
        return scores, signals

    def confidence(self, score: float) -> float:
        return round(1.0 - math.exp(-max(score, 0.0) / self.scale), 4)

    def classify(self, features: PageFeatures) -> Classification:
        scores, signals = self.match(features)
        providers = {t: s for t, s in scores.items() if t in self.provider_types and s >= self.threshold}
        generic = {t: s for t, s in scores.items() if t not in self.provider_types and s >= self.threshold}
        # A provider widget determines how the challenge must be handled, so it wins over generic hints.
        pool = providers or generic
        if not pool:
            best = max(scores.items(), key=lambda kv: kv[1], default=(None, 0.0))
            return Classification(False, None, None, self.confidence(best[1]) if best[1] else 0.0,
                                  scores, signals, self.version)
        ctype, score = max(pool.items(), key=lambda kv: kv[1])
        subtype = self._subtype(ctype, scores, features)
        return Classification(True, ctype, subtype, self.confidence(score), scores, signals, self.version)

    def _subtype(self, ctype: str, scores: dict[str, float], f: PageFeatures) -> str | None:
        if ctype not in self.provider_types:
            return None
        if scores.get("audio", 0) >= self.threshold:
            return "audio_challenge"
        if scores.get("image_selection", 0) >= self.threshold or any("bframe" in u for u in f.iframe_src):
            return "image_challenge"
        if ctype == "recaptcha":
            if any("enterprise" in u for u in f.script_src + f.iframe_src):
                return "enterprise"
            if any(re.search(r"render=(?!explicit)[A-Za-z0-9_\-]{10,}", u) for u in f.script_src) or "grecaptcha-badge" in f.classes:
                return "v3_invisible"
            return "v2_checkbox"
        if ctype == "turnstile" and any(i in f.ids for i in ("challenge-form", "cf-challenge-running", "challenge-stage")):
            return "managed_challenge"
        return "widget"
