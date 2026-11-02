"""Image-based CAPTCHA *type* classifier used at runtime.

Loads only the human-approved production model (``models/production/image_classifier.npz``)
and reports which kind of challenge a cropped screenshot shows. It never reads, segments
or answers the challenge. Reloads automatically when a new model is promoted.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.image_features import FeatureExtractor, load_image

log = logging.getLogger("captcha_fw.image")

# model category -> (framework captcha_type, subtype)
CATEGORY_MAP: dict[str, tuple[str | None, str | None]] = {
    "text": ("text", "plain_text"),
    "text_distorted": ("text", "distorted_text"),
    "math": ("text", "math_expression"),
    "audio": ("audio", "audio_widget"),
    "image_classification": ("image_selection", "image_classification"),
    "grid": ("image_selection", "tile_grid"),
    "recaptcha": ("recaptcha", "widget_image"),
    "hcaptcha": ("hcaptcha", "widget_image"),
    "turnstile": ("turnstile", "widget_image"),
    "none": (None, None),
}
ARCHITECTURES = ("nearest_centroid", "softmax", "mlp")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NumpyImageModel:
    """Small numpy model (nearest-centroid, softmax regression or 1-hidden-layer MLP)."""

    def __init__(self, arch: str, classes: list[str], params: dict[str, np.ndarray],
                 mean: np.ndarray, std: np.ndarray, meta: dict | None = None):
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch}")
        self.arch, self.classes, self.params = arch, list(classes), params
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.meta = meta or {}

    @property
    def version(self) -> str | None:
        return self.meta.get("version")

    @property
    def param_count(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def logits_std(self, Xs: np.ndarray) -> np.ndarray:
        p = self.params
        if self.arch == "nearest_centroid":
            d2 = ((Xs[:, None, :] - p["C"][None, :, :]) ** 2).mean(axis=2)
            return -d2 * float(p["temperature"][0])
        if self.arch == "softmax":
            return Xs @ p["W"] + p["b"]
        h = np.maximum(Xs @ p["W1"] + p["b1"], 0)
        return h @ p["W2"] + p["b2"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        out = []
        for i in range(0, len(X), 512):  # bounded memory for nearest-centroid
            out.append(softmax(self.logits_std(self.standardize(X[i:i + 512]))))
        return np.vstack(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, arch=np.array(self.arch), classes=np.array(self.classes), mean=self.mean, std=self.std,
                 meta_json=np.array(json.dumps(self.meta, default=str)),
                 **{f"param_{k}": v for k, v in self.params.items()})
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyImageModel":
        with np.load(path, allow_pickle=False) as z:
            params = {k[6:]: z[k].copy() for k in z.files if k.startswith("param_")}
            return cls(str(z["arch"]), [str(c) for c in z["classes"]], params, z["mean"].copy(), z["std"].copy(),
                       json.loads(str(z["meta_json"])))


@dataclass
class ImagePrediction:
    category: str
    captcha_type: str | None
    subtype: str | None
    confidence: float
    is_captcha: bool
    model_version: str | None
    probabilities: dict[str, float] = field(default_factory=dict)


class ImageCaptchaClassifier:
    def __init__(self, models_dir: str | Path, min_confidence: float = 0.80):
        self.model_file = Path(models_dir) / "production" / "image_classifier.npz"
        self.min_confidence = min_confidence
        self._model: NumpyImageModel | None = None
        self._mtime: float | None = None
        self._lock = threading.Lock()
        self._extractor: FeatureExtractor | None = None

    def _ensure(self) -> NumpyImageModel | None:
        with self._lock:
            if not self.model_file.exists():
                self._model = None
                return None
            mtime = self.model_file.stat().st_mtime
            if self._model is None or mtime != self._mtime:
                try:
                    self._model = NumpyImageModel.load(self.model_file)
                    spec = self._model.meta.get("feature_spec", {})
                    self._extractor = FeatureExtractor(spec.get("width", 96), spec.get("height", 48))
                    self._mtime = mtime
                    log.info("image model %s loaded", self._model.version)
                except Exception as exc:
                    log.error("could not load production image model: %s", exc)
                    self._model = None
            return self._model

    @property
    def available(self) -> bool:
        return self._ensure() is not None

    @property
    def version(self) -> str | None:
        m = self._ensure()
        return m.version if m else None

    def classify(self, img: np.ndarray) -> ImagePrediction:
        model = self._ensure()
        if model is None:
            raise RuntimeError("no production image model is deployed")
        probs = model.predict_proba(self._extractor(img)[None, :])[0]  # type: ignore[misc]
        i = int(probs.argmax())
        category = model.classes[i]
        ctype, subtype = CATEGORY_MAP.get(category, (None, None))
        conf = float(probs[i])
        return ImagePrediction(category, ctype, subtype, round(conf, 4),
                               bool(ctype) and conf >= self.min_confidence, model.version,
                               {c: round(float(p), 4) for c, p in zip(model.classes, probs)})

    def classify_file(self, path: str | Path) -> ImagePrediction:
        return self.classify(load_image(path))
