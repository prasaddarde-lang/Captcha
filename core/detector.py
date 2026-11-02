"""CAPTCHA detection engine.

Extracts *metadata* from a page (DOM attributes, frame/script URLs, field names,
prompt text) and hands it to the classifier. It never reads, stores or attempts
to answer the challenge content itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from core.classifier import CaptchaClassifier, Classification, PageFeatures

log = logging.getLogger("captcha_fw.detector")

MAX_TEXT_CHARS = 200_000
RESPONSE_FIELDS = {
    "g-recaptcha-response": "recaptcha",
    "h-captcha-response": "hcaptcha",
    "cf-turnstile-response": "turnstile",
}
_SKIP_TEXT_TAGS = {"script", "style", "noscript", "template"}


class _FeatureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.f = PageFeatures()
        self._text: list[str] = []
        self._text_len = 0
        self._skip_depth = 0
        self._textarea_field: str | None = None
        self._textarea_has_content = False
        self._tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        self._tags.add(tag)
        if tag in _SKIP_TEXT_TAGS:
            self._skip_depth += 1
        if tag in ("iframe", "frame") and a.get("src"):
            self.f.iframe_src.append(a["src"])
        if tag == "script" and a.get("src"):
            self.f.script_src.append(a["src"])
        if a.get("class"):
            self.f.classes.extend(c for c in a["class"].split() if c)
        if a.get("id"):
            self.f.ids.append(a["id"])
        for k, v in a.items():
            if k.startswith("data-"):
                self.f.attr_names.append(k)
                self.f.attrs.append(f"{k}={v[:100]}")
                if k == "data-mock-captcha":
                    self.f.mock_marker = v or "generic"
        if tag in ("input", "textarea", "select"):
            for key in ("name", "id", "placeholder", "aria-label"):
                if a.get(key):
                    self.f.input_names.append(a[key])
            name = a.get("name") or a.get("id") or ""
            if name in RESPONSE_FIELDS:
                provider = RESPONSE_FIELDS[name]
                if tag == "textarea":
                    self._textarea_field = provider
                    self._textarea_has_content = False
                else:
                    self.f.response_filled[provider] = bool(a.get("value"))
        if tag == "img":
            desc = " ".join(a.get(k, "") for k in ("src", "alt", "id", "class", "title")).strip()
            if desc:
                self.f.imgs.append(desc[:300])

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TEXT_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "textarea" and self._textarea_field:
            self.f.response_filled[self._textarea_field] = self._textarea_has_content
            self._textarea_field = None

    def handle_data(self, data: str) -> None:
        if self._textarea_field is not None:
            # Only record whether a response exists - never its value.
            if data.strip():
                self._textarea_has_content = True
            return
        if self._skip_depth or self._text_len >= MAX_TEXT_CHARS:
            return
        chunk = " ".join(data.split())
        if chunk:
            self._text.append(chunk)
            self._text_len += len(chunk) + 1

    def result(self) -> PageFeatures:
        self.f.text = " ".join(self._text).lower()[:MAX_TEXT_CHARS]
        self.f.tags = sorted(self._tags)
        return self.f


def extract_features(html: str) -> PageFeatures:
    parser = _FeatureParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # malformed markup should never crash detection
        log.warning("HTML parse issue: %s", exc)
    return parser.result()


@dataclass
class PageSnapshot:
    html: str
    url: str | None = None
    frame_urls: list[str] = field(default_factory=list)
    response_filled: dict[str, bool] = field(default_factory=dict)
    screenshot_path: str | None = None
    source: str = "html"  # html | html_file | cdp | url


@dataclass
class DetectionResult:
    classification: Classification
    features: PageFeatures
    snapshot: PageSnapshot

    @property
    def detected(self) -> bool:
        return self.classification.captcha_detected

    @property
    def is_solved(self) -> bool:
        """True when the provider response field for the detected type has been filled
        (i.e. a human has completed the widget). The value itself is never read."""
        t = self.classification.captcha_type
        return bool(t and self.features.response_filled.get(t))


class CaptchaDetector:
    def __init__(self, classifier: CaptchaClassifier | None = None):
        self.classifier = classifier or CaptchaClassifier()

    def detect_html(self, html: str, frame_urls: list[str] | None = None,
                    url: str | None = None) -> DetectionResult:
        return self.detect(PageSnapshot(html=html, url=url, frame_urls=frame_urls or []))

    def detect_file(self, path: str | Path) -> DetectionResult:
        html = Path(path).read_text(encoding="utf-8", errors="replace")
        return self.detect(PageSnapshot(html=html, url=Path(path).resolve().as_uri(), source="html_file"))

    def detect(self, snapshot: PageSnapshot) -> DetectionResult:
        features = extract_features(snapshot.html or "")
        features.merge_frames(snapshot.frame_urls)
        for k, v in snapshot.response_filled.items():
            features.response_filled[k] = features.response_filled.get(k, False) or v
        cls = self.classifier.classify(features)
        log.info("detection: detected=%s type=%s subtype=%s conf=%.2f model=%s",
                 cls.captcha_detected, cls.captcha_type, cls.subtype, cls.confidence, cls.model_version)
        return DetectionResult(cls, features, snapshot)
