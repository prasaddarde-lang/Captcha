"""Shared training helpers: config loading, paths, thread limits, resources, hashing.

Import this module FIRST in every training entry point: ``apply_thread_limits()`` must
run before numpy/OpenCV start their thread pools.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_FILE = Path(__file__).with_name("training_config.yaml")
CAPTCHA_CATEGORIES = {"text", "text_distorted", "math", "audio", "image_classification", "grid",
                      "recaptcha", "hcaptcha", "turnstile"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or os.environ.get("CAPTCHA_FW_TRAINING_CONFIG", CONFIG_FILE), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_thread_limits(cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    n = str(int(cfg.get("resources", {}).get("max_cpu_threads", 4)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, n)
    try:
        import cv2
        cv2.setNumThreads(int(n))
    except Exception:
        pass
    return int(n)


@dataclass
class Paths:
    root: Path
    datasets: Path
    models: Path
    reports: Path

    @property
    def manifest(self) -> Path:
        return self.datasets / "manifest.json"

    @property
    def cache(self) -> Path:
        return self.datasets / "cache"

    @property
    def checkpoints(self) -> Path:
        return self.models / "checkpoints"

    @property
    def production(self) -> Path:
        return self.models / "production"

    @property
    def archive(self) -> Path:
        return self.models / "archive"

    @property
    def history(self) -> Path:
        return self.reports / "training_history"

    @property
    def performance(self) -> Path:
        return self.reports / "performance"

    def ensure(self) -> "Paths":
        for p in (self.datasets, self.checkpoints, self.production, self.archive, self.history,
                  self.performance, self.reports / "audit"):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict | None = None) -> Paths:
    cfg = cfg or load_config()
    ds = os.environ.get("CAPTCHA_FW_DATASETS_DIR") or str(ROOT / cfg["dataset"]["root"])
    models = os.environ.get("CAPTCHA_FW_MODELS_DIR") or str(ROOT / "models")
    reports = os.environ.get("CAPTCHA_FW_REPORTS_DIR") or str(ROOT / "reports")
    return Paths(ROOT, Path(ds), Path(models), Path(reports)).ensure()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_hash(cfg: dict, sections: tuple[str, ...] = ("features", "training")) -> str:
    return sha256_bytes(json.dumps({k: cfg.get(k) for k in sections}, sort_keys=True).encode())[:16]


def write_json(path: Path, data: Any) -> None:
    """Atomic JSON write (state files must survive interruption)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def free_disk_mb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1_048_576


def check_disk(path: Path, cfg: dict) -> None:
    need = float(cfg.get("resources", {}).get("min_free_disk_mb", 300))
    free = free_disk_mb(path)
    if free < need:
        raise ResourceLimitError(f"only {free:.0f} MB free on disk (minimum {need:.0f} MB)")


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except Exception:
        return 0.0


def detect_device(cfg: dict) -> dict:
    """Report GPU availability honestly. The numpy backend always computes on CPU."""
    info = {"requested": cfg.get("training", {}).get("device", "auto"), "gpu_available": False,
            "gpu_name": None, "used": "cpu"}
    if shutil.which("nvidia-smi"):
        import subprocess
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                info.update(gpu_available=True, gpu_name=out.splitlines()[0])
        except Exception:
            pass
    try:
        import cv2
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            info["gpu_available"] = True
    except Exception:
        pass
    info["note"] = ("GPU detected but the numpy/OpenCV backend trains on CPU" if info["gpu_available"]
                    else "no GPU detected; training on CPU")
    return info


class ResourceLimitError(RuntimeError):
    pass
