"""Dataset access for training: manifest queries, dataset hashing and a feature cache."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

from core.image_features import FeatureExtractor
from training.common import Paths, get_paths, load_config, read_json


class TrainingDataset:
    def __init__(self, cfg: dict | None = None, paths: Paths | None = None):
        self.cfg = cfg or load_config()
        self.paths = paths or get_paths(self.cfg)
        self.manifest = read_json(self.paths.manifest)
        if self.manifest is None:
            raise FileNotFoundError(f"dataset manifest not found: {self.paths.manifest}")
        self.categories: list[str] = list(self.manifest["categories"])
        self.extractor = FeatureExtractor(self.cfg["features"]["width"], self.cfg["features"]["height"])

    def samples(self, split: str | None = None, category: str | None = None) -> list[dict]:
        return [s for s in self.manifest["samples"]
                if (split is None or s["split"] == split) and (category is None or s["category"] == category)]

    def dataset_hash(self) -> str:
        h = hashlib.sha256()
        for s in sorted(self.manifest["samples"], key=lambda s: s["sample_id"]):
            h.update(f"{s['sample_id']}:{s['sha256']}:{s['split']}".encode())
        return h.hexdigest()[:16]

    def test_set_hash(self) -> str:
        h = hashlib.sha256()
        for s in sorted(self.samples("test"), key=lambda s: s["sample_id"]):
            h.update(f"{s['sample_id']}:{s['sha256']}".encode())
        return h.hexdigest()[:16]

    def image(self, sample: dict) -> np.ndarray:
        data = np.fromfile(str(self.paths.datasets / sample["file"]), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    # ---------------------------------------------------------------- features
    def _cache_file(self) -> Path:
        spec = self.extractor.spec
        return self.paths.cache / f"features_{spec['version']}_{spec['width']}x{spec['height']}.npz"

    def features(self, split: str, progress: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return (X float32, y int, sample_ids). Features are cached on disk (float16) keyed by
        sample_id + sha256, so unchanged samples are never recomputed."""
        samples = self.samples(split)
        cache = self._load_cache()
        missing = [s for s in samples if f"{s['sample_id']}:{s['sha256']}" not in cache]
        if missing:
            t0 = time.time()
            for i, s in enumerate(missing):
                cache[f"{s['sample_id']}:{s['sha256']}"] = self.extractor(self.image(s)).astype(np.float16)
                if progress and (i + 1) % 2000 == 0:
                    print(f"    features {i + 1}/{len(missing)} ({time.time() - t0:.0f}s)", flush=True)
            self._save_cache(cache)
        X = np.stack([cache[f"{s['sample_id']}:{s['sha256']}"] for s in samples]).astype(np.float32)
        y = np.array([self.categories.index(s["category"]) for s in samples], dtype=np.int64)
        return X, y, [s["sample_id"] for s in samples]

    def _load_cache(self) -> dict[str, np.ndarray]:
        f = self._cache_file()
        if not f.exists():
            return {}
        with np.load(f, allow_pickle=False) as z:
            return dict(zip(z["keys"].tolist(), z["X"]))

    def _save_cache(self, cache: dict[str, np.ndarray]) -> None:
        f = self._cache_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(cache)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, keys=np.array(keys), X=np.stack([cache[k] for k in keys]))
        tmp.replace(f)
