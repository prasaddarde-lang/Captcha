"""Synthetic dataset management.

Generates *synthetic* labelled fixtures for development and testing:
  * HTML pages that mimic how each CAPTCHA type appears in page metadata
    (provider script/iframe URLs, class names, field names, prompt text) plus
    tricky negatives. Every fixture carries ``data-mock-captcha`` so test mode
    can recognise it. Fixtures never load provider scripts when parsed.
  * Synthetic distorted-text images (OpenCV) used to exercise screenshot,
    preprocessing and operator-UI pipelines. They are generated here with known
    labels; no third-party CAPTCHA content is collected or used for training.
"""
from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

from config import Settings, get_settings

LABELS = ["recaptcha", "hcaptcha", "turnstile", "funcaptcha", "geetest", "aws_waf",
          "text", "image_selection", "audio", "none"]

_SITEKEY_CHARS = string.ascii_letters + string.digits + "-_"


def _key(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice(_SITEKEY_CHARS) for _ in range(n))


def _wrap(rng: random.Random, label: str, body: str, title: str | None = None) -> str:
    filler = rng.choice([
        "<p>Please sign in to continue.</p>",
        "<p>Complete the form below to submit your request.</p>",
        "<nav><a href='/'>Home</a> | <a href='/help'>Help</a></nav>",
        "<p>Order status lookup</p>",
        "",
    ])
    form_open = rng.choice(["<form method='post' action='/submit'>", "<form id='main-form'>", "<div class='form'>"])
    form_close = "</div>" if form_open.startswith("<div") else "</form>"
    fields = rng.choice([
        "<input name='username' placeholder='Username'><input type='password' name='password'>",
        "<input name='email' type='email'>",
        "<input name='reference' placeholder='Reference number'>",
        "",
    ])
    return (f"<!doctype html><html><head><title>{title or 'Mock page'}</title></head>"
            f"<body><div data-mock-captcha='{label}' class='mock-root'>{filler}{form_open}{fields}"
            f"{body}<button type='submit'>Submit</button>{form_close}</div></body></html>")


def _recaptcha(rng):
    variant = rng.choice(["checkbox", "invisible", "enterprise", "frame_only", "image"])
    k = _key(rng)
    if variant == "checkbox":
        body = (f"<script src='https://www.google.com/recaptcha/api.js' async defer></script>"
                f"<div class='g-recaptcha' data-sitekey='{k}'></div>"
                "<textarea name='g-recaptcha-response' style='display:none'></textarea>")
    elif variant == "invisible":
        body = (f"<script src='https://www.google.com/recaptcha/api.js?render={k}'></script>"
                "<div class='grecaptcha-badge'></div>")
    elif variant == "enterprise":
        body = (f"<script src='https://www.google.com/recaptcha/enterprise.js'></script>"
                f"<div class='g-recaptcha' data-sitekey='{k}'></div>")
    elif variant == "frame_only":
        body = f"<iframe title='reCAPTCHA' src='https://www.google.com/recaptcha/api2/anchor?k={k}&co=x'></iframe>"
    else:
        body = (f"<iframe src='https://www.google.com/recaptcha/api2/anchor?k={k}'></iframe>"
                f"<iframe src='https://www.google.com/recaptcha/api2/bframe?k={k}'></iframe>"
                "<div class='rc-imageselect'>Select all images with traffic lights</div>")
    return body


def _hcaptcha(rng):
    k = _key(rng, 36)
    return rng.choice([
        f"<script src='https://js.hcaptcha.com/1/api.js' async></script><div class='h-captcha' data-sitekey='{k}'></div>",
        f"<iframe src='https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html#frame=checkbox&sitekey={k}'></iframe>"
        "<textarea name='h-captcha-response'></textarea>",
        f"<div class='h-captcha' data-sitekey='{k}'></div><textarea name='h-captcha-response'></textarea>",
    ])


def _turnstile(rng):
    k = "0x" + _key(rng, 22)
    return rng.choice([
        f"<script src='https://challenges.cloudflare.com/turnstile/v0/api.js' defer></script>"
        f"<div class='cf-turnstile' data-sitekey='{k}'></div>",
        "<div id='challenge-stage'></div><div id='challenge-running'>Checking if the site connection is secure</div>"
        "<iframe src='https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/b/turnstile/if/ov2/av0/rcv0/0/abc'></iframe>",
        f"<div class='cf-turnstile' data-sitekey='{k}'></div><input type='hidden' name='cf-turnstile-response' value=''>",
    ])


def _funcaptcha(rng):
    return rng.choice([
        "<script src='https://client-api.arkoselabs.com/v2/ABCDEF/api.js'></script><div id='arkose-enforcement'></div>",
        "<iframe src='https://client-api.arkoselabs.com/fc/gc/?token=abc'></iframe><input id='fc-token' type='hidden'>",
        "<div id='FunCaptcha'></div><script src='https://api.funcaptcha.com/fc/api/'></script>",
    ])


def _geetest(rng):
    return rng.choice([
        "<script src='https://static.geetest.com/v4/gt4.js'></script><div class='geetest_captcha'></div>",
        "<div class='geetest_holder geetest_wind'></div><script src='/static/js/gt.js'></script>",
    ])


def _aws_waf(rng):
    return rng.choice([
        "<script src='https://abc123.edge.captcha-sdk.awswaf.com/abc123/jsapi.js'></script>"
        "<script src='https://abc123.token.awswaf.com/abc123/challenge.js'></script><div id='captcha-container'></div>",
        "<script src='https://abc.captcha.awswaf.com/abc/jsapi.js'></script><div id='aws-waf-captcha'></div>",
    ])


def _text(rng):
    name = rng.choice(["captcha", "captcha_code", "security_code", "verification_code", "txtCaptcha"])
    prompt = rng.choice(["Enter the characters you see", "Type the text shown in the image",
                         "Enter the code shown above", "Type the characters below"])
    img = rng.choice(["/captcha.php?r=123", "/images/captcha_image.jpg", "/api/captcha/img"])
    extra = rng.choice(["", "<a href='#' class='refresh-captcha'>Refresh</a>"])
    return (f"<label>{prompt}</label><img src='{img}' alt='CAPTCHA image' id='captchaImg'>"
            f"<input name='{name}' autocomplete='off'>{extra}")


def _image_selection(rng):
    noun = rng.choice(["bicycles", "crosswalks", "buses", "boats", "bridges"])
    tiles = "".join(f"<img class='captcha-tile' src='/mock/tile_{i}.png' alt=''>" for i in range(9))
    return rng.choice([
        f"<div class='image-grid' data-captcha-type='image_selection'><p>Select all images with {noun}</p>{tiles}</div>",
        f"<div class='img-select'><p>Click on each image containing {noun}</p>{tiles}</div>",
        f"<div class='captcha-grid'><strong>Select all squares with {noun}</strong>{tiles}</div>",
    ])


def _audio(rng):
    return rng.choice([
        "<div class='audio-challenge'><p>Press PLAY to listen</p><audio src='/mock/challenge.mp3' controls></audio>"
        "<input name='audio_response'></div>",
        "<button id='audio-button'>Get an audio challenge</button><audio controls></audio>"
        "<input name='audio-answer' placeholder='Type what you hear'>",
        "<div class='rc-audiochallenge'><audio controls></audio><p>Listen and type the numbers you hear</p>"
        "<input id='audio-response'></div>",
    ])


def _none(rng):
    return rng.choice([
        "<h1>Welcome back</h1><p>Your account dashboard.</p>",
        "<article><h2>What is a CAPTCHA?</h2><p>A blog article about how websites tell humans and "
        "computers apart.</p></article>",
        "<label>Security question: name of your first school</label><input name='security_question'>",
        "<p>Upload invoices</p><input type='file' name='invoice'><img src='/logo.png' alt='Company logo'>",
        "<div class='image-gallery'><img src='/a.png' alt='Product'><img src='/b.png' alt='Product'></div>",
        "<audio src='/podcast.mp3' controls></audio><p>Listen to our latest podcast episode.</p>",
    ])


GENERATORS = {
    "recaptcha": _recaptcha, "hcaptcha": _hcaptcha, "turnstile": _turnstile,
    "funcaptcha": _funcaptcha, "geetest": _geetest, "aws_waf": _aws_waf,
    "text": _text, "image_selection": _image_selection, "audio": _audio, "none": _none,
}


@dataclass
class Sample:
    file: str
    label: str

    def html(self, root: Path) -> str:
        return (root / self.file).read_text(encoding="utf-8")


class SyntheticDatasetManager:
    def __init__(self, settings: Settings | None = None, root: Path | None = None):
        self.settings = settings or get_settings()
        self.root = root or (self.settings.test_data_dir / "synthetic")

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # ------------------------------------------------------------- generation
    def make_html(self, label: str, rng: random.Random) -> str:
        return _wrap(rng, label, GENERATORS[label](rng), title=f"Mock {label} page")

    def generate_html_dataset(self, per_label: int = 20, seed: int = 42,
                              labels: list[str] | None = None) -> list[Sample]:
        rng = random.Random(seed)
        out_dir = self.root / "html"
        out_dir.mkdir(parents=True, exist_ok=True)
        samples: list[Sample] = []
        for label in labels or LABELS:
            for i in range(per_label):
                rel = f"html/{label}_{i:03d}.html"
                (self.root / rel).write_text(self.make_html(label, rng), encoding="utf-8")
                samples.append(Sample(rel, label))
        self._write_manifest(samples, kind="html", seed=seed)
        return samples

    def generate_image_dataset(self, count: int = 20, seed: int = 7) -> list[dict]:
        """Synthetic distorted-text images for pipeline / UI testing. Only the *category* is
        recorded - the rendered characters are deliberately not stored (no OCR training set)."""
        import cv2
        import numpy as np

        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        out_dir = self.root / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for i in range(count):
            text = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(rng.randint(4, 6)))
            img = np.full((60, 200, 3), 255, dtype=np.uint8)
            for _ in range(6):  # background noise lines
                p1 = (rng.randint(0, 199), rng.randint(0, 59))
                p2 = (rng.randint(0, 199), rng.randint(0, 59))
                cv2.line(img, p1, p2, tuple(int(c) for c in np_rng.integers(100, 220, 3)), 1)
            x = 10
            for ch in text:
                color = tuple(int(c) for c in np_rng.integers(0, 120, 3))
                cv2.putText(img, ch, (x, rng.randint(35, 50)), cv2.FONT_HERSHEY_SIMPLEX,
                            rng.uniform(0.9, 1.2), color, 2, cv2.LINE_AA)
                x += rng.randint(26, 32)
            m = cv2.getRotationMatrix2D((100, 30), rng.uniform(-6, 6), 1.0)
            img = cv2.warpAffine(img, m, (200, 60), borderValue=(255, 255, 255))
            noise = np_rng.integers(0, 40, img.shape, dtype=np.uint8)
            img = cv2.subtract(img, noise)
            rel = f"images/synthetic_{i:03d}.png"
            cv2.imwrite(str(self.root / rel), img)
            items.append({"file": rel, "label": "text", "kind": "synthetic_text_image"})
        (self.root / "images" / "labels.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        return items

    def write_mock_site(self, directory: Path | None = None, seed: int = 99) -> Path:
        """One page per label plus an index, for serving from a local mock web server."""
        rng = random.Random(seed)
        site = directory or (self.settings.test_data_dir / "mock_site")
        site.mkdir(parents=True, exist_ok=True)
        links = []
        for label in LABELS:
            (site / f"{label}.html").write_text(self.make_html(label, rng), encoding="utf-8")
            links.append(f"<li><a href='{label}.html'>{label}</a></li>")
        (site / "index.html").write_text(
            f"<!doctype html><html><head><title>Mock CAPTCHA site</title></head><body data-mock-captcha='none'>"
            f"<h1>Mock CAPTCHA test site</h1><ul>{''.join(links)}</ul></body></html>", encoding="utf-8")
        return site

    # --------------------------------------------------------------- loading
    def _write_manifest(self, samples: list[Sample], kind: str, seed: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps({
            "kind": kind, "seed": seed, "synthetic": True,
            "samples": [{"file": s.file, "label": s.label} for s in samples],
        }, indent=2), encoding="utf-8")

    def load(self) -> list[Sample]:
        if not self.manifest_path.exists():
            return self.generate_html_dataset()
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [Sample(s["file"], s["label"]) for s in data["samples"]]

    def load_labeled_html(self) -> list[tuple[str, str]]:
        return [(s.html(self.root), s.label) for s in self.load()]

    @staticmethod
    def split(items: list, test_ratio: float = 0.3, seed: int = 1) -> tuple[list, list]:
        """Stratified-ish split by shuffling within each label."""
        rng = random.Random(seed)
        by_label: dict[str, list] = {}
        for it in items:
            by_label.setdefault(it[1] if isinstance(it, tuple) else it.label, []).append(it)
        train, test = [], []
        for group in by_label.values():
            rng.shuffle(group)
            k = max(1, int(round(len(group) * test_ratio)))
            test.extend(group[:k])
            train.extend(group[k:])
        return train, test


def preprocess_image(path: str | Path):
    """OpenCV preprocessing used in synthetic-image pipeline tests
    (grayscale -> denoise -> adaptive threshold). Returns a binary ndarray."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
