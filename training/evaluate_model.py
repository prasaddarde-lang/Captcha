"""Model evaluation on the fixed test set (or any split).

    python training/evaluate_model.py --production
    python training/evaluate_model.py --version img-1.0.0 [--split validation]

Metrics per category: precision, recall (= detection rate for CAPTCHA categories), F1, support;
overall accuracy, macro-F1, false-positive / false-negative detection rates (raw arg-max and at
the runtime confidence threshold), latency, model size, CPU time and memory.
OCR metrics (character / exact-string accuracy) are not applicable: no OCR model exists by design.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import apply_thread_limits, get_paths, load_config, now_iso, write_json  # noqa: E402

apply_thread_limits()

import numpy as np  # noqa: E402

from core.image_classifier import NumpyImageModel  # noqa: E402
from training.dataset_manager import TrainingDataset  # noqa: E402

NONE = "none"
OCR_NOTE = {"character_accuracy": "not_applicable", "exact_string_accuracy": "not_applicable",
            "reason": "No OCR / text-reading model is trained or shipped: reading challenge text would be "
                      "solving the CAPTCHA, which this framework does not do."}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    per = {}
    f1s = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": int(cm[i].sum())}
        if cm[i].sum():
            f1s.append(f1)
    out = {"accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else 0.0,
           "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
           "per_category": per,
           "confusion": {"labels": classes, "matrix": cm.tolist()}}
    out.update(detection_rates(y_true, y_pred, classes))
    return out


def detection_rates(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """Binary view: is there a CAPTCHA at all?"""
    if NONE not in classes:
        return {}
    ni = classes.index(NONE)
    true_none, pred_none = y_true == ni, y_pred == ni
    fp = int((true_none & ~pred_none).sum())
    fn = int((~true_none & pred_none).sum())
    return {"false_positive_rate": round(fp / max(1, int(true_none.sum())), 4),
            "false_negative_rate": round(fn / max(1, int((~true_none).sum())), 4),
            "detection_accuracy": round(float((true_none == pred_none).mean()), 4),
            "false_positives": fp, "false_negatives": fn}


def evaluate_model(model: NumpyImageModel, ds: TrainingDataset, split: str = "test",
                   min_confidence: float | None = None, latency_samples: int = 200) -> dict:
    cfg = ds.cfg
    min_confidence = float(cfg["inference"]["min_confidence"] if min_confidence is None else min_confidence)
    try:
        import psutil
        proc = psutil.Process()
        cpu0 = sum(proc.cpu_times()[:2])
    except Exception:
        proc = None
    X, y, ids = ds.features(split)
    # map the model's class order onto the dataset's
    order = [model.classes.index(c) for c in ds.categories]
    t0 = time.perf_counter()
    proba = model.predict_proba(X)[:, order]
    batch_s = time.perf_counter() - t0
    pred = proba.argmax(axis=1)
    m = classification_metrics(y, pred, ds.categories)

    # runtime operating point: below min_confidence the image model never declares a CAPTCHA
    conf = proba.max(axis=1)
    ni = ds.categories.index(NONE)
    gated = np.where((pred != ni) & (conf < min_confidence), ni, pred)
    g = detection_rates(y, gated, ds.categories)
    m["at_runtime_threshold"] = {"min_confidence": min_confidence, **g,
                                 "accuracy": round(float((gated == y).mean()), 4),
                                 "abstained": int(((pred != ni) & (conf < min_confidence)).sum())}
    m["confidence"] = {"mean_correct": round(float(conf[pred == y].mean()), 4) if (pred == y).any() else None,
                       "mean_incorrect": round(float(conf[pred != y].mean()), 4) if (pred != y).any() else None}

    # end-to-end single-image latency (decode + features + predict), as the framework does it
    samples = ds.samples(split)[:latency_samples]
    times = []
    for s in samples:
        t1 = time.perf_counter()
        model.predict_proba(ds.extractor(ds.image(s))[None, :])
        times.append((time.perf_counter() - t1) * 1000)
    m["latency_ms"] = {"mean": round(float(np.mean(times)), 3), "p50": round(float(np.percentile(times, 50)), 3),
                       "p95": round(float(np.percentile(times, 95)), 3), "samples": len(times),
                       "batch_ms_per_sample": round(batch_s * 1000 / max(1, len(X)), 4)}
    m["resources"] = {"param_count": model.param_count,
                      "model_size_bytes": int(sum(p.nbytes for p in model.params.values()) + model.mean.nbytes * 2)}
    if proc is not None:
        m["resources"].update(cpu_seconds=round(sum(proc.cpu_times()[:2]) - cpu0, 2),
                              rss_mb=round(proc.memory_info().rss / 1_048_576, 1))
    m["ocr_metrics"] = OCR_NOTE
    m.update(split=split, samples=int(len(y)), model_version=model.version, architecture=model.arch,
             test_set_hash=ds.test_set_hash() if split == "test" else None, evaluated_at=now_iso())
    return m


def print_summary(m: dict) -> None:
    print(f"model {m['model_version']} ({m['architecture']}) on {m['split']} ({m['samples']} samples)")
    print(f"  accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
          f"FP rate {m.get('false_positive_rate')}  FN rate {m.get('false_negative_rate')}")
    rt = m["at_runtime_threshold"]
    print(f"  at runtime threshold {rt['min_confidence']}: FP {rt['false_positive_rate']}  FN {rt['false_negative_rate']}  "
          f"abstained {rt['abstained']}")
    print(f"  latency mean {m['latency_ms']['mean']} ms  p95 {m['latency_ms']['p95']} ms  "
          f"size {m['resources']['model_size_bytes'] / 1024:.0f} KB")
    print(f"  {'category':<22}{'precision':>10}{'recall':>9}{'F1':>8}{'n':>6}")
    for c, v in m["per_category"].items():
        print(f"  {c:<22}{v['precision']:>10.4f}{v['recall']:>9.4f}{v['f1']:>8.4f}{v['support']:>6}")


def main(argv=None) -> int:
    from training.model_registry import ImageModelRegistry

    ap = argparse.ArgumentParser(description="Evaluate an image model")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--version")
    grp.add_argument("--production", action="store_true")
    grp.add_argument("--path")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = ap.parse_args(argv)
    cfg = load_config()
    paths = get_paths(cfg)
    reg = ImageModelRegistry(cfg, paths)
    if args.path:
        model_path = Path(args.path)
    elif args.version:
        model_path = reg.model_path(args.version)
    else:
        prod = reg.production_version()
        if not prod:
            pending = [v for v, m in reg.models().items() if m["status"] in ("CANDIDATE", "APPROVED")]
            print("no production model deployed yet" + (f"; candidates awaiting approval: {pending}" if pending else ""))
            return 2
        model_path = paths.production / "image_classifier.npz"
    m = evaluate_model(NumpyImageModel.load(model_path), TrainingDataset(cfg, paths), args.split)
    out = paths.performance / f"eval_{m['model_version']}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(out, m)
    print_summary(m)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
